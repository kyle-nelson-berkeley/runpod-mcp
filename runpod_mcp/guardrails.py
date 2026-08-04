"""Cost/safety guardrails — enforced in code, not prose.

Rules (from the consensus plan):
  - ONE pod PER DECLARED VEHICLE; any pod whose name is not declared in
    pod_defaults.yaml `vehicles:` is refused (bringing up one vehicle must
    tolerate the other vehicle's pod, and nothing else).
  - RTX 4090 only, exactly 1 GPU, SECURE cloud, interruptible FORCED false
    (per pod — unchanged by the two-vehicle split).
  - A network volume is required (data survives termination).
  - terminate_pod needs the verbatim confirm string "terminate <pod_name>".
  - One detached job at a time unless force=True (1 GPU; log-dir attribution
    assumes a single run).
"""

REQUIRED_GPU = "NVIDIA GeForce RTX 4090"
REQUIRED_CLOUD = "SECURE"


class GuardrailError(RuntimeError):
    """A guardrail refused the operation. Nothing was sent to the API."""


def assert_only_pod(pods: list, allowed_names) -> None:
    """Refuse to operate while any pod we did NOT declare exists on the account.

    `allowed_names` is the union of every vehicle's pod_name
    (config.declared_pod_names) — so ensure_pod for one vehicle tolerates the
    other vehicle's pod, and only genuinely-foreign pods are refused.
    """
    allowed = set(allowed_names)
    others = [p for p in pods if p.get("name") not in allowed]
    if others:
        names = ", ".join(f"{p.get('name')!r} (id={p.get('id')})" for p in others)
        allowed_txt = ", ".join(repr(n) for n in sorted(allowed))
        raise GuardrailError(
            f"Account has undeclared pod(s): {names}. Allowed (one per "
            f"declared vehicle): {allowed_txt}. Resolve these in the RunPod "
            "console first, or declare the vehicle in pod_defaults.yaml.")


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
