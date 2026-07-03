"""SSH/scp/rsync plumbing — thin subprocess wrappers, fully argv-injectable.

Hardening (consensus plan):
  - BatchMode=yes (never hang on a prompt), StrictHostKeyChecking=accept-new,
  - a DEDICATED known-hosts file (~/.ssh/known_hosts_runpod) truncated on
    every pod transition-to-running: the container disk wipes on stop, host
    keys regenerate each start, so stale entries only cause false MITM fails.
  - rsync never deletes remote or local content (--partial resume, no --delete).

The direct-SSH path (root@publicIp:portMappings["22"]) is used exclusively;
RunPod's proxy SSH has no scp support.
"""
import base64
import shlex
import subprocess
import time
from pathlib import Path

from .config import scrub


class SSHError(RuntimeError):
    """SSH/scp/rsync subprocess failure; message is scrubbed."""


class ConnCache:
    """60s-TTL cache of (publicIp, ssh_port) — the only local state we keep."""

    def __init__(self, ttl_sec: float = 60.0, clock=time.monotonic):
        self._ttl = ttl_sec
        self._clock = clock
        self._value: tuple[str, int] | None = None
        self._stamp = 0.0

    def get(self) -> tuple[str, int] | None:
        if self._value and (self._clock() - self._stamp) <= self._ttl:
            return self._value
        return None

    def put(self, host: str, port: int) -> None:
        self._value = (host, int(port))
        self._stamp = self._clock()

    def clear(self) -> None:
        self._value = None


class SSHClient:
    def __init__(self, identity, known_hosts, runner=subprocess.run):
        self.identity = Path(identity).expanduser()
        self.known_hosts = Path(known_hosts).expanduser()
        self._run = runner

    # ------------------------------------------------------------- options

    def _opts(self) -> list[str]:
        return ["-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "ConnectTimeout=15"]

    # ----------------------------------------------------------------- ssh

    def run(self, host: str, port: int, command: str, timeout: int = 60,
            check: bool = False) -> subprocess.CompletedProcess:
        argv = (["ssh", "-i", str(self.identity), "-p", str(port)]
                + self._opts() + [f"root@{host}", command])
        try:
            proc = self._run(argv, capture_output=True, text=True,
                             timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SSHError(f"ssh to {host}:{port} timed out after {timeout}s "
                           f"running: {scrub(command)[:200]}") from None
        if check and proc.returncode != 0:
            raise SSHError(f"ssh {host}:{port} rc={proc.returncode}: "
                           f"{scrub(proc.stderr).strip()[:500]} "
                           f"(cmd: {scrub(command)[:200]})")
        return proc

    def push_text(self, host: str, port: int, text: str, remote_path: str,
                  executable: bool = False) -> None:
        """Write text to a remote file via base64 — no quoting pitfalls, no
        temp files, and file content never appears in `ps` output readably."""
        b64 = base64.b64encode(text.encode()).decode()
        rp = shlex.quote(remote_path)
        rdir = shlex.quote(str(Path(remote_path).parent))
        cmd = f"mkdir -p {rdir} && echo {b64} | base64 -d > {rp}"
        if executable:
            cmd += f" && chmod +x {rp}"
        self.run(host, port, cmd, timeout=60, check=True)

    # ----------------------------------------------------------------- scp

    def push_file(self, host: str, port: int, local_path, remote_path: str) -> None:
        argv = (["scp", "-i", str(self.identity), "-P", str(port)]
                + self._opts()
                + [str(local_path), f"root@{host}:{remote_path}"])
        proc = self._run(argv, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise SSHError(f"scp to {host}:{port}:{remote_path} "
                           f"rc={proc.returncode}: {scrub(proc.stderr)[:500]}")

    # --------------------------------------------------------------- rsync

    def rsync_pull(self, host: str, port: int, remote_dir: str, local_dir,
                   timeout: int = 600) -> str:
        """Pull remote_dir into local_dir. NEVER deletes on either side."""
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rsh = (f"ssh -i {self.identity} -p {port} "
               + " ".join(f"{a} {b}" for a, b in
                          zip(self._opts()[::2], self._opts()[1::2])))
        argv = ["rsync", "-az", "--partial", "-e", rsh,
                f"root@{host}:{remote_dir}", str(local_dir)]
        proc = self._run(argv, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise SSHError(f"rsync from {host}:{port}:{remote_dir} "
                           f"rc={proc.returncode}: {scrub(proc.stderr)[:500]}")
        return proc.stdout

    # --------------------------------------------------------- known hosts

    def truncate_known_hosts(self) -> None:
        """Called on every pod transition-to-running (create AND resume)."""
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        self.known_hosts.write_text("")
