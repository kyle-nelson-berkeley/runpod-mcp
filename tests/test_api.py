"""Unit 2: api.py — REST v1 client + unauthenticated GraphQL, on httpx.MockTransport."""
import json

import httpx
import pytest

from runpod_mcp import api

KEY = "rpa_TESTKEY456"


def make_client(handler):
    return api.RunPodClient(KEY, transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ REST v1

def test_list_pods_sends_bearer_and_parses_bare_list():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[{"id": "p1", "name": "lts-replication"}])

    pods = make_client(handler).list_pods()
    assert pods == [{"id": "p1", "name": "lts-replication"}]
    assert seen["method"] == "GET"
    assert seen["url"] == "https://rest.runpod.io/v1/pods"
    assert seen["auth"] == f"Bearer {KEY}"


def test_list_pods_parses_wrapped_shape():
    def handler(request):
        return httpx.Response(200, json={"pods": [{"id": "p2"}]})

    assert make_client(handler).list_pods() == [{"id": "p2"}]


def test_create_pod_posts_exact_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "new1"})

    payload = {"name": "lts-replication", "gpuTypeIds": ["NVIDIA GeForce RTX 4090"]}
    pod = make_client(handler).create_pod(payload)
    assert pod["id"] == "new1"
    assert seen["url"] == "https://rest.runpod.io/v1/pods"
    assert seen["body"] == payload


def test_lifecycle_endpoints_hit_expected_paths():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"id": "p1"})

    c = make_client(handler)
    c.get_pod("p1")
    c.start_pod("p1")
    c.stop_pod("p1")
    c.delete_pod("p1")
    assert calls == [
        ("GET", "/v1/pods/p1"),
        ("POST", "/v1/pods/p1/start"),
        ("POST", "/v1/pods/p1/stop"),
        ("DELETE", "/v1/pods/p1"),
    ]


def test_network_volume_endpoints():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            body = json.loads(request.content)
            assert body == {"name": "lts-replication", "size": 60,
                            "dataCenterId": "US-KS-2"}
            return httpx.Response(201, json={"id": "vol1"})
        return httpx.Response(200, json=[])

    c = make_client(handler)
    assert c.list_network_volumes() == []
    vol = c.create_network_volume(
        {"name": "lts-replication", "size": 60, "dataCenterId": "US-KS-2"})
    assert vol["id"] == "vol1"
    assert calls == [("GET", "/v1/networkvolumes"),
                     ("POST", "/v1/networkvolumes")]


def test_billing_endpoints_and_missing_volume_billing():
    def handler(request):
        if request.url.path == "/v1/billing/pods":
            return httpx.Response(200, json=[{"podId": "p1", "amount": 0.5}])
        return httpx.Response(404, json={"error": "not found"})

    c = make_client(handler)
    assert c.billing_pods() == [{"podId": "p1", "amount": 0.5}]
    # /billing/networkvolumes may not exist — must degrade to None, not raise
    assert c.billing_network_volumes() is None


# ---------------------------------------------------------------- api errors

def test_api_error_carries_status_and_scrubbed_body():
    def handler(request):
        return httpx.Response(401, text=f"bad key {KEY} rejected")

    with pytest.raises(api.ApiError) as exc:
        make_client(handler).list_pods()
    msg = str(exc.value)
    assert "401" in msg
    assert KEY not in msg            # scrubbed
    assert "rpa_[REDACTED]" in msg


def test_transport_error_is_scrubbed(monkeypatch):
    def handler(request):
        raise httpx.ConnectError(f"refused for Bearer {KEY}")

    with pytest.raises(api.ApiError) as exc:
        make_client(handler).list_pods()
    assert KEY not in str(exc.value)


def test_no_gpu_error_classifier():
    assert api.looks_like_no_gpu_error(
        "There are no longer any instances available with the requested specifications")
    assert api.looks_like_no_gpu_error("no GPU available on host machine")
    assert not api.looks_like_no_gpu_error("invalid api key")


# ------------------------------------------------------------------ GraphQL

def test_gpu_types_is_unauthenticated_and_parses():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = json.loads(request.content)
        seen["query"] = body["query"]
        return httpx.Response(200, json={"data": {"gpuTypes": [{
            "id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090",
            "memoryInGb": 24, "secureCloud": True, "communityCloud": True,
            "lowestPrice": {"stockStatus": "High",
                            "uninterruptablePrice": 0.69,
                            "minimumBidPrice": 0.34}}]}})

    out = api.gpu_types(transport=httpx.MockTransport(handler))
    assert seen["url"] == "https://api.runpod.io/graphql"
    assert seen["auth"] is None                       # public endpoint, no key
    assert "gpuTypes" in seen["query"]
    assert "stockStatus" in seen["query"]
    assert out[0]["lowestPrice"]["stockStatus"] == "High"


def test_gpu_types_datacenter_filter_lands_in_query():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"gpuTypes": []}})

    api.gpu_types(data_center_id="US-KS-2", transport=httpx.MockTransport(handler))
    blob = json.dumps(seen["body"])
    assert "US-KS-2" in blob


def test_gpu_types_graphql_errors_raise():
    def handler(request):
        return httpx.Response(200, json={"errors": [{"message": "boom"}]})

    with pytest.raises(api.ApiError, match="boom"):
        api.gpu_types(transport=httpx.MockTransport(handler))
