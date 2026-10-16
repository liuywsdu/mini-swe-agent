"""Incus-managed containers or VMs. VM commands use incus-agent, never a guest LLM agent."""

import logging
import os
import shutil
import subprocess
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from minisweagent.executors.base import CommandExecutor, run_process


class IncusExecutorConfig(BaseModel):
    image: str
    instance_type: Literal["container", "vm"] = "container"
    executable: str = os.getenv("MSWEA_INCUS_EXECUTABLE", "incus")
    project: str = "default"
    profiles: list[str] = ["default"]
    storage: str = ""
    instance_config: dict[str, str] = {}
    cwd: str = "/root"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    interpreter: list[str] = Field(default=["bash", "-lc"], min_length=1)
    profile_wrapper: list[str] = []
    # Optional syscall profiler prepended to the interpreter inside the instance, e.g. ["strace", "-f", "-c"].
    # Profiling runs only. The wrapper's stderr is captured separately and stored as output["profile"].
    timeout: float = Field(default=30, gt=0)
    startup_timeout: float = Field(default=240, gt=0)
    cleanup_timeout: float = Field(default=60, gt=0)


def build_launch_command(config: IncusExecutorConfig, name: str) -> list[str]:
    return [
        config.executable,
        "launch",
        config.image,
        name,
        "--project",
        config.project,
        *(["--vm"] if config.instance_type == "vm" else []),
        *[arg for profile in (config.profiles or [""]) for arg in ("--profile", profile)],
        *(["--storage", config.storage] if config.storage else []),
        *[arg for key, value in config.instance_config.items() for arg in ("--config", f"{key}={value}")],
        "--config",
        f"user.minisweagent={name}",
    ]


def build_exec_command(
    config: IncusExecutorConfig, name: str, command: str, cwd: str, env: dict[str, str]
) -> list[str]:
    environment = {key: os.environ[key] for key in config.forward_env if key in os.environ} | config.env | env
    return [
        config.executable,
        "exec",
        name,
        "--project",
        config.project,
        "--mode=non-interactive",
        "--disable-stdin",
        "--cwd",
        cwd or config.cwd,
        *[arg for key, value in environment.items() for arg in ("--env", f"{key}={value}")],
        "--",
        *config.profile_wrapper,
        *config.interpreter,
        f"( {command} ) 2>&1" if config.profile_wrapper else command,
    ]


class IncusExecutor(CommandExecutor):
    def __init__(self, *, config_class: type = IncusExecutorConfig, logger: logging.Logger | None = None, **kwargs):
        self.config = config_class(**kwargs)
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.instance_name: str | None = None
        if shutil.which(self.config.executable) is None:
            raise FileNotFoundError(f"Executable not found: {self.config.executable}")
        self.instance_name = f"minisweagent-{uuid.uuid4().hex}"
        deadline = time.monotonic() + self.config.startup_timeout
        try:
            result = run_process(
                build_launch_command(self.config, self.instance_name), timeout=self.config.startup_timeout
            )
            if result.returncode:
                raise RuntimeError(f"Incus launch failed: {result.stdout}")
            while True:
                result = self.execute("true", cwd="/", timeout=min(5, max(0.1, deadline - time.monotonic())))
                if result["returncode"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Incus instance not ready (VMs require incus-agent): {result['output']}")
                time.sleep(0.2)
        except Exception:
            self.cleanup()
            raise

    def _execute_command(
        self, command: str, cwd: str, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess:
        assert self.instance_name, "Incus instance is closed"
        return run_process(
            build_exec_command(self.config, self.instance_name, command, cwd, env),
            timeout=timeout,
            capture_stderr=bool(self.config.profile_wrapper),
        )

    def cleanup(self) -> None:
        if getattr(self, "instance_name", None) is not None:
            result = run_process(
                [
                    self.config.executable,
                    "delete",
                    self.instance_name,
                    "--force",
                    "--project",
                    self.config.project,
                ],
                timeout=self.config.cleanup_timeout,
            )
            if (
                result.returncode
                and "not found" not in result.stdout.lower()
                and "does not exist" not in result.stdout.lower()
            ):
                result.check_returncode()
            self.instance_name = None

    def __del__(self):
        self.cleanup()
