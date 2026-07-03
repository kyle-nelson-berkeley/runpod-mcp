"""RunPod control-plane clients.

Two planes (verified June 2026, re-verified by the $0 live smoke):
  - REST v1  https://rest.runpod.io/v1   — authenticated, Bearer key; pods,
    network volumes, billing.
  - GraphQL  https://api.runpod.io/graphql — gpuTypes price/stock works
    UNAUTHENTICATED; used for availability checks only.

Every error string is passed through config.scrub() so an API key can never
leak into tool output or logs.
"""
import json

import httpx

from .config import scrub

REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"

# Substrings that classify a pod-start failure as "host has no free GPU"
# (stopped pods do NOT reserve their GPU — RUNBOOK/plan recovery recipe).
_NO_GPU_MARKERS = (
    "no longer any instances available",
    "no gpu available",
    "not enough free gpus",
    "does not have the resources",
    "insufficient gpu",
)


class ApiError(RuntimeError):
    """RunPod API failure; message is always scrubbed."""

    def __init__(self, message: str, status_code: int | None = None,
                 body: str | None = None):
        super().__init__(scrub(message))
        self.status_code = status_code
        self.body = scrub(body) if body is not None else None


def looks_like_no_gpu_error(text: str) -> bool:
    low = str(text).lower()
    return any(marker in low for marker in _NO_GPU_MARKERS)


class RunPodClient:
    """Thin REST v1 client. `transport` is injectable for offline tests."""

    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0):
        self._http = httpx.Client(
            base_url=REST_BASE,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    # ------------------------------------------------------------- plumbing

    def _request(self, method: str, path: str, payload: dict | None = None,
                 none_on: tuple[int, ...] = ()):
        try:
            resp = self._http.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise ApiError(f"RunPod API {method} {path} failed: "
                           f"{type(exc).__name__}: {exc}") from None
        if resp.status_code in none_on:
            return None
        if resp.status_code >= 400:
            raise ApiError(f"RunPod API {method} {path} -> HTTP "
                           f"{resp.status_code}: {resp.text[:500]}",
                           status_code=resp.status_code, body=resp.text[:500])
        if not resp.content:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            raise ApiError(f"RunPod API {method} {path} returned non-JSON: "
                           f"{resp.text[:200]}") from None

    # ----------------------------------------------------------------- pods

    def list_pods(self) -> list:
        data = self._request("GET", "/pods")
        if isinstance(data, dict) and "pods" in data:   # tolerate wrapped shape
            return data["pods"]
        return data or []

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def create_pod(self, payload: dict) -> dict:
        return self._request("POST", "/pods", payload)

    def start_pod(self, pod_id: str) -> dict | None:
        return self._request("POST", f"/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> dict | None:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    # -------------------------------------------------------network volumes

    def list_network_volumes(self) -> list:
        data = self._request("GET", "/networkvolumes")
        if isinstance(data, dict) and "networkVolumes" in data:
            return data["networkVolumes"]
        return data or []

    def create_network_volume(self, payload: dict) -> dict:
        return self._request("POST", "/networkvolumes", payload)

    # -------------------------------------------------------------- billing

    def billing_pods(self):
        return self._request("GET", "/billing/pods")

    def billing_network_volumes(self):
        """May not exist as an endpoint — spend_report labels itself honestly
        when this returns None (404/405)."""
        return self._request("GET", "/billing/networkvolumes",
                             none_on=(404, 405))


# ------------------------------------------------------------------ GraphQL

_GPU_QUERY = """
query GpuTypes($input: GpuTypeFilter, $lowestPriceInput: GpuLowestPriceInput) {
  gpuTypes(input: $input) {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: $lowestPriceInput) {
      stockStatus
      uninterruptablePrice
      minimumBidPrice
    }
  }
}
"""


def gpu_types(gpu_id: str = "NVIDIA GeForce RTX 4090",
              data_center_id: str | None = None,
              transport: httpx.BaseTransport | None = None) -> list:
    """Query 4090 price/stock via the public (unauthenticated) GraphQL API."""
    lowest_price_input: dict = {"gpuCount": 1}
    if data_center_id:
        lowest_price_input["dataCenterId"] = data_center_id
    body = {
        "query": _GPU_QUERY,
        "variables": {"input": {"id": gpu_id},
                      "lowestPriceInput": lowest_price_input},
    }
    try:
        with httpx.Client(timeout=30.0, transport=transport) as client:
            resp = client.post(GRAPHQL_URL, json=body)
    except httpx.HTTPError as exc:
        raise ApiError(f"GraphQL gpuTypes failed: {type(exc).__name__}: {exc}") \
            from None
    if resp.status_code >= 400:
        raise ApiError(f"GraphQL gpuTypes -> HTTP {resp.status_code}: "
                       f"{resp.text[:300]}", status_code=resp.status_code)
    data = resp.json()
    if data.get("errors"):
        raise ApiError(f"GraphQL gpuTypes errors: {data['errors']}")
    return data["data"]["gpuTypes"]
