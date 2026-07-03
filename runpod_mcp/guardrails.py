"""Cost/safety guardrails — enforced in code, not prose.

Rules (from the consensus plan):
  - ONE pod max, named per pod_defaults.yaml; refuse if any other pod exists.
  - RTX 4090 only, exactly 1 GPU, SECURE cloud, interruptible FORCED false.
  - A network volume is required (data survives termination).
  - terminate_pod needs the verbatim confirm string "terminate <pod_name>".
  - One detached job at a time unless force=True (1 GPU; log-dir attribution
    assumes a single run).
"""

REQUIRED_GPU = "NVIDIA GeForce RTX 4090"
REQUIRED_CLOUD = "SECURE"


class GuardrailError(RuntimeError):
    """A guardrail refused the operation. Nothing was sent to the API."""


def assert_only_pod(pods: list, pod_name: str) -> None:
    """Refuse to operate while any pod other than ours exists on the account."""
    others = [p for p in pods if p.get("name") != pod_name]
    if others:
        names = ", ".join(f"{p.get('name')!r} (id={p.get('id')})" for p in others)
        raise GuardrailError(
            f"Account has pod(s) other than '{pod_name}': {names}. "
            "One-pod guardrail: resolve these in the RunPod console first.")


def enforce_pod_payload(payload: dict) -> dict:
    """Normalize + assert a POST /pods payload against the hard rules."""
    if payload.get("gpuTypeIds") != [REQUIRED_GPU]:
        raise GuardrailError(f"gpuTypeIds must be exactly ['{REQUIRED_GPU}'] "
                             f"(got {payload.get('gpuTypeIds')!r})")
    if payload.get("gpuCount", 1) != 1:
        raise GuardrailError(f"gpuCount must be 1 (got {payload.get('gpuCount')!r})")
    if payload.get("cloudType") != REQUIRED_CLOUD:
        raise GuardrailError(f"cloudType must be {REQUIRED_CLOUD} "
                             f"(got {payload.get('cloudType')!r})")
    if not payload.get("networkVolumeId"):
        raise GuardrailError("a network volume is required (networkVolumeId "
                             "missing/empty) — pod data must survive termination")
    payload["interruptible"] = False   # forced, never negotiable (spot = lost runs)
    return payload


def assert_terminate_confirm(confirm: str, pod_name: str) -> None:
    expected = f"terminate {pod_name}"
    if confirm != expected:
        raise GuardrailError(
            f"terminate_pod requires the verbatim confirm string '{expected}'. "
            "This deletes the pod (the network volume survives).")


def assert_no_live_jobs(live_jobs: list, force: bool = False) -> None:
    if live_jobs and not force:
        raise GuardrailError(
            f"Job(s) already running: {', '.join(live_jobs)}. One job at a "
            "time (1 GPU). Pass force=True only if you know what you're doing.")


def clamp_exec_timeout(requested: int, ceiling: int) -> int:
    """exec_on_pod is a bounded synchronous escape hatch — never unbounded."""
    return max(1, min(int(requested), int(ceiling)))
