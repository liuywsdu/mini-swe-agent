import os
import subprocess

from pydantic import BaseModel

from minisweagent.executors.base import CommandExecutor, run_process
from minisweagent.utils.cgroup import read_own_cgroup_stats
from minisweagent.utils.netdev import PROC_NET_DEV, read_net_dev


class LocalExecutorConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    profile_wrapper: list[str] = []
    # Optional syscall profiler prepended to the shell, e.g. ["strace", "-f", "-c"]. Profiling runs only.
    # The wrapper's stderr is captured separately and stored as output["profile"].


class LocalExecutor(CommandExecutor):
    cgroup_layer = "guest_cgroup"
    network_observation_point = "guest_proc_net_dev"

    def __init__(self, *, config_class: type = LocalExecutorConfig, **kwargs):
        self.config = config_class(**kwargs)

    def cgroup_stats(self) -> dict[str, int | None] | None:
        return read_own_cgroup_stats()

    def net_dev_stats(self) -> dict[str, int] | None:
        return read_net_dev(PROC_NET_DEV)

    def _execute_command(
        self, command: str, cwd: str, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if self.config.profile_wrapper:
            return run_process(
                [*self.config.profile_wrapper, "/bin/sh", "-c", f"( {command} ) 2>&1"],
                cwd=cwd or self.config.cwd or os.getcwd(),
                env=os.environ | self.config.env | env,
                timeout=timeout,
                capture_stderr=True,
            )
        return run_process(
            command,
            shell=True,
            cwd=cwd or self.config.cwd or os.getcwd(),
            env=os.environ | self.config.env | env,
            timeout=timeout,
        )

    def cleanup(self) -> None:
        pass
