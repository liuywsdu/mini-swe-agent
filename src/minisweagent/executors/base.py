"""Shared process execution and result format for tool backends."""

import os
import signal
import subprocess
import time
from typing import Any

from minisweagent.utils.cgroup import diff_cgroup_stats


def run_process(
    command: str | list[str],
    *,
    timeout: float,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    shell: bool = False,
    capture_stderr: bool = False,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)


class CommandExecutor:
    config: Any
    cgroup_layer: str | None = None
    # Measurement layer label for cgroup readings (container_cgroup / guest_cgroup).
    network_observation_point: str | None = None
    # Where network counters are observed (guest_proc_net_dev / container_netns_init_pid).

    def cgroup_stats(self) -> dict[str, int | None] | None:
        # Cumulative cgroup counters for this executor's workload; None when unavailable.
        return None

    def net_dev_stats(self) -> dict[str, int] | None:
        # Cumulative {rx_bytes, tx_bytes} at the observation point; None when unavailable.
        return None

    def execute(
        self, command: str, cwd: str = "", *, env: dict[str, str] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        timeout = self.config.timeout if timeout is None else timeout
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        started_wall, started_monotonic = time.time(), time.monotonic()
        cgroup_before = self.cgroup_stats()
        try:
            result = self._execute_command(command, cwd, env or {}, timeout)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
            if getattr(self.config, "profile_wrapper", None):
                output["profile"] = result.stderr or ""
        except (OSError, subprocess.SubprocessError) as e:
            raw_output = getattr(e, "output", None) or ""
            output = {
                "output": raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        output["timing"] = {
            "started_wall": started_wall,
            "started_monotonic": started_monotonic,
            "wall_seconds": time.time() - started_wall,
            "monotonic_seconds": time.monotonic() - started_monotonic,
        }
        resources = diff_cgroup_stats(cgroup_before, self.cgroup_stats())
        if resources is not None:
            resources["layer"] = self.cgroup_layer
        output["resources"] = resources
        return output

    def _execute_command(
        self, command: str, cwd: str, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError
