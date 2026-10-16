"""Translate agent bash actions to a model-independent tool executor."""

import platform
from typing import Any

from minisweagent import ToolExecutor
from minisweagent.exceptions import Submitted
from minisweagent.executors import get_executor
from minisweagent.utils.serialize import recursive_merge


class ExecutorEnvironment:
    executor_class: type[ToolExecutor] | None = None

    def __init__(self, *, executor: ToolExecutor | dict | None = None, **kwargs):
        if self.executor_class is not None:
            if executor is not None:
                raise ValueError("Use ExecutorEnvironment to supply an executor instance or configuration")
            self.executor = self.executor_class(**kwargs)
        else:
            if executor is None:
                raise ValueError("ExecutorEnvironment requires an executor instance or an executor configuration")
            if isinstance(executor, dict):
                self.executor = get_executor(recursive_merge(kwargs, executor))
            else:
                if kwargs:
                    raise ValueError("Configure an injected executor before passing it to ExecutorEnvironment")
                self.executor = executor

    @property
    def config(self) -> Any:
        return self.executor.config

    def execute(self, action: dict, cwd: str = "", *, timeout: float | None = None) -> dict[str, Any]:
        output = self.executor.execute(action.get("command", ""), cwd=cwd, timeout=timeout)
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            error = Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
            error.execution_output = output
            raise error

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(platform.uname()._asdict(), self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        config = self.config.model_dump(mode="json")
        if self.executor_class is None:
            config = {
                "executor": {**config, "backend": f"{type(self.executor).__module__}.{type(self.executor).__name__}"}
            }
        return {
            "info": {
                "config": {
                    "environment": config,
                    "environment_type": f"{type(self).__module__}.{type(self).__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        self.executor.cleanup()
