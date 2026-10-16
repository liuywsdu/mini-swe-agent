import logging
import os
import shlex
import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel

from minisweagent.executors.base import CommandExecutor, run_process
from minisweagent.utils.cgroup import read_container_cgroup_stats
from minisweagent.utils.netdev import read_net_dev


class DockerExecutorConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    runtime: str = ""
    """Optional container runtime registered with the daemon, e.g. runsc."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    profile_wrapper: list[str] = []
    """Optional syscall profiler prepended to the interpreter inside the container, e.g. ["strace", "-f", "-c"].
    Profiling runs only. The wrapper's stderr is captured separately and stored as output["profile"]."""


class DockerExecutor(CommandExecutor):
    cgroup_layer = "container_cgroup"
    network_observation_point = "container_netns_init_pid"

    def __init__(
        self,
        *,
        config_class: type = DockerExecutorConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerExecutorConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex}"
        cmd = build_run_command(self.config, container_name)
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def _execute_command(
        self, command: str, cwd: str, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert self.container_id, "Container not started"
        return run_process(
            build_exec_command(self.config, self.container_id, command, cwd, env),
            timeout=timeout,
            capture_stderr=bool(self.config.profile_wrapper),
        )

    def cgroup_stats(self) -> dict[str, int | None] | None:
        return read_container_cgroup_stats(self.container_id) if self.container_id else None

    def net_dev_stats(self) -> dict[str, int] | None:
        # /proc/<container init pid>/net/dev reflects the container network namespace.
        # Observation failures must not break the agent run: record null instead.
        if not self.container_id:
            return None
        try:
            result = subprocess.run(
                [self.config.executable, "inspect", "-f", "{{.State.Pid}}", self.container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            pid = int(result.stdout.strip()) if result.returncode == 0 else 0
            return read_net_dev(Path(f"/proc/{pid}/net/dev")) if pid > 0 else None
        except (OSError, ValueError):
            return None

    def cleanup(self) -> None:
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            result = subprocess.run(
                [self.config.executable, "rm", "-f", self.container_id],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode and "No such container" not in result.stderr:
                result.check_returncode()
            self.container_id = None

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()


def build_run_command(config: DockerExecutorConfig, name: str) -> list[str]:
    return [
        config.executable,
        "run",
        "-d",
        "--name",
        name,
        "-w",
        config.cwd,
        *config.run_args,
        *([f"--runtime={config.runtime}"] if config.runtime else []),
        config.image,
        "sleep",
        config.container_timeout,
    ]


def build_exec_command(
    config: DockerExecutorConfig, container_id: str, command: str, cwd: str, env: dict[str, str]
) -> list[str]:
    environment = {name: os.environ[name] for name in config.forward_env if name in os.environ} | config.env | env
    if config.profile_wrapper:
        command = f"( {command} ) 2>&1"
    return [
        config.executable,
        "exec",
        "-w",
        cwd or config.cwd,
        *[arg for key, value in environment.items() for arg in ("-e", f"{key}={value}")],
        container_id,
        *config.profile_wrapper,
        *config.interpreter,
        command,
    ]
