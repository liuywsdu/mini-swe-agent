import copy
import json
import os

import pytest
from typer.testing import CliRunner

from minisweagent.agents import get_agent
from minisweagent.agents.journal import JournalAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.environments.executor import ExecutorEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.executors import get_executor
from minisweagent.executors.docker import DockerExecutorConfig, build_exec_command, build_run_command
from minisweagent.executors.firecracker import _build_remote_command
from minisweagent.executors.gvisor import GVisorExecutorConfig
from minisweagent.models.replay import RecordingModel
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.replay import app, export_trajectory, get_replay_environment_config


@pytest.mark.parametrize(("agent_class", "mode"), [("default", "yolo"), ("interactive", "yolo")])
def test_builtin_agent_templates_with_injected_executor(agent_class, mode, tmp_path):
    command = "printf artifact > result.txt; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; cat result.txt"
    response = make_toolcall_output(
        "Done",
        [
            {
                "id": "submit",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        [{"command": command, "tool_call_id": "submit"}],
    )
    model = DeterministicToolcallModel(outputs=[response])
    environment = get_environment(
        get_config_from_spec("mini.yaml")["environment"]
        | {
            "environment_class": "executor",
            "env": {"SHARED": "base", "OVERRIDE": "base"},
            "executor": {"backend": "local", "cwd": str(tmp_path), "env": {"OVERRIDE": "nested"}},
        }
    )
    config = get_config_from_spec("mini.yaml")["agent"] | {
        "agent_class": agent_class,
        "mode": mode,
        "confirm_exit": False,
        "output_path": None,
    }
    try:
        agent = get_agent(model, environment, config)
        assert agent.run("write a file")["submission"] == "artifact"
        assert (tmp_path / "result.txt").read_text() == "artifact"
        assert agent.model is model
        assert agent.n_calls == 1
        assert environment.config.env == {"SHARED": "base", "OVERRIDE": "nested"}
    finally:
        environment.cleanup()


def test_executor_is_independent_of_agent_submission_and_keeps_call_options_local(tmp_path):
    other = tmp_path / "directory with spaces"
    other.mkdir()
    config = {"backend": "local", "cwd": str(tmp_path), "env": {"EXECUTOR_VALUE": "configured"}}
    original = copy.deepcopy(config)
    executor = get_executor(config)
    try:
        result = executor.execute(
            'printf "%s\\n" "$EXECUTOR_VALUE" "$PWD"; printf "stderr\\n" >&2; exit 7',
            cwd=str(other),
            env={"EXECUTOR_VALUE": "a 'quote'; $(exit 42)"},
        )
        assert set(result) == {"output", "returncode", "exception_info", "timing", "resources"}
        assert {key: result[key] for key in ("output", "returncode", "exception_info")} == {
            "output": f"a 'quote'; $(exit 42)\n{other}\nstderr\n",
            "returncode": 7,
            "exception_info": "",
        }
        assert executor.execute('printf "%s\\n" "$EXECUTOR_VALUE" "$PWD"')["output"] == f"configured\n{tmp_path}\n"
        command = "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch\\n'"
        assert executor.execute(command)["output"].endswith("patch\n")
        environment = ExecutorEnvironment(executor=executor)
        with pytest.raises(Submitted) as submitted:
            environment.execute({"command": command, "tool_call_id": "ignored-by-executor"})
        assert submitted.value.messages[0]["extra"]["submission"] == "patch\n"
        assert environment.execute({"command": command + "; exit 2"})["returncode"] == 2
        assert config == original
    finally:
        executor.cleanup()


def test_executor_timeout_preserves_partial_output_and_missing_cwd_is_structured(tmp_path):
    executor = get_executor({"backend": "local", "cwd": str(tmp_path)})
    output = executor.execute("printf ready; sleep 5", timeout=0.2)
    assert output["returncode"] == -1
    assert output["output"] == "ready"
    assert output["extra"]["exception_type"] == "TimeoutExpired"
    missing = executor.execute("true", cwd=str(tmp_path / "missing"))
    assert missing["returncode"] == -1
    assert missing["extra"]["exception_type"] == "FileNotFoundError"
    assert executor.execute("printf usable")["output"] == "usable"


def test_backend_command_translation_preserves_payload_and_selects_runtime(tmp_path):
    command = 'printf "%s" "$VALUE"; printf "%s" "$PWD"'
    value = "a 'quote'; $(exit 42)"
    config = DockerExecutorConfig(image="test-image", cwd="/testbed", forward_env=["PATH"], env={"VALUE": "default"})
    original = config.model_dump()
    argv = build_exec_command(config, "container-123", command, "/a directory", {"VALUE": value})
    assert argv[-4:] == ["container-123", "bash", "-lc", command]
    assert argv[2:4] == ["-w", "/a directory"]
    assert argv.count("VALUE=" + value) == 1
    assert "VALUE=default" not in argv
    assert "PATH=" + os.environ["PATH"] in argv
    assert config.model_dump() == original
    assert not any(arg.startswith("--runtime") for arg in build_run_command(config, "ordinary"))
    gvisor = build_run_command(GVisorExecutorConfig(image="test-image"), "sandbox")
    assert gvisor.index("--runtime=runsc") < gvisor.index("test-image")
    remote = _build_remote_command(command, str(tmp_path), {"VALUE": value}, ["bash", "-c"])
    assert get_executor({"backend": "local"}).execute(remote)["output"] == value + str(tmp_path)


@pytest.mark.parametrize(
    ("backend", "binary_option", "options"),
    [
        ("docker", "executable", {"image": "unused"}),
        ("gvisor", "executable", {"image": "unused"}),
        ("firecracker", "firecracker_executable", {"kernel_image": "unused", "root_drive": "unused"}),
    ],
)
def test_unavailable_executor_does_not_fall_back_to_local(backend, binary_option, options, tmp_path):
    with pytest.raises(FileNotFoundError):
        get_executor({"backend": backend, binary_option: str(tmp_path / "absent-binary"), **options})


def test_environment_config_roundtrip_and_backend_override(tmp_path):
    environment = get_environment(
        {"environment_class": "executor", "executor": {"backend": "local", "cwd": str(tmp_path)}}
    )
    metadata = environment.serialize()["info"]["config"]
    restored = get_environment({**metadata["environment"], "environment_class": metadata["environment_type"]})
    assert restored.execute({"command": "pwd"})["output"].strip() == str(tmp_path)
    with pytest.raises(ValueError, match="fresh workspace"):
        get_replay_environment_config(metadata, "next")
    source = {
        "environment_type": "minisweagent.environments.docker.DockerEnvironment",
        "environment": {"image": "old-image", "run_args": ["--rm"], "cwd": "/testbed"},
    }
    original = copy.deepcopy(source)
    override = {"environment_class": "executor", "executor": {"backend": "local", "cwd": str(tmp_path)}}
    assert get_replay_environment_config(source, "next", override) == override
    same_backend = get_replay_environment_config(source, "next", {"environment_class": "docker", "timeout": 5})
    assert same_backend["image"] == "old-image"
    assert same_backend["run_args"][-3:] == ["--pull=never", "--label", "agentos.replay=next"]
    gvisor = get_replay_environment_config(
        source, "next", {"environment_class": "executor", "executor": {"backend": "gvisor", "image": "new-image"}}
    )
    assert gvisor["executor"]["run_args"][-3:] == ["--pull=never", "--label", "agentos.replay=next"]
    with pytest.raises(ValueError, match="mounted storage"):
        get_replay_environment_config(
            {
                "environment_type": "minisweagent.environments.executor.ExecutorEnvironment",
                "environment": {
                    "run_args": ["--rm", "--mount", "type=bind,src=/somewhere,dst=/testbed"],
                    "executor": {"backend": "gvisor", "image": "test-image"},
                },
            },
            "next",
        )
    assert source == original


def test_replay_cli_uses_injected_executor_and_records_new_tool_outputs(tmp_path, reset_global_stats):
    source_dir, replay_dir = tmp_path / "source", tmp_path / "replay"
    source_dir.mkdir()
    replay_dir.mkdir()
    commands = [
        'printf "%s" "$EXECUTOR_VALUE" > result.txt; cat result.txt',
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; cat result.txt",
    ]
    responses = [
        make_toolcall_output(
            "",
            [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }
            ],
            [{"command": command, "tool_call_id": f"call-{index}"}],
        )
        for index, command in enumerate(commands)
    ]
    model = RecordingModel(
        store_path=str(tmp_path / "tapes"),
        episode_id="source",
        backend={
            "model_class": "minisweagent.models.test_models.DeterministicToolcallModel",
            "model_name": "offline-executor-test",
            "outputs": responses,
        },
    )
    agent = JournalAgent(
        model,
        LocalEnvironment(cwd=str(source_dir), env={"EXECUTOR_VALUE": "original"}),
        system_template="system",
        instance_template="{{ task }}",
        step_limit=3,
    )
    assert agent.run("task")["submission"] == "original"
    agent.save(tmp_path / "source.index.json")
    override = tmp_path / "executor.json"
    override.write_text(
        json.dumps(
            {
                "environment": {
                    "environment_class": "executor",
                    "executor": {
                        "backend": "local",
                        "cwd": str(replay_dir),
                        "env": {"EXECUTOR_VALUE": "new-output"},
                    },
                }
            }
        )
    )
    output = tmp_path / "replay.index.json"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--store",
            str(tmp_path / "tapes"),
            "--source",
            "source",
            "--episode",
            "replayed",
            "--output",
            str(output),
            "--environment-config",
            str(override),
            "--timing",
            "instant",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (source_dir / "result.txt").read_text() == "original"
    assert (replay_dir / "result.txt").read_text() == "new-output"
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["actual_api_calls"] == 0
    assert summary["submission_matches"] is False
    assert summary["input_difference_rounds"] == 1
    replayed = export_trajectory(model.store, "replayed")
    assert replayed["info"]["submission"] == "new-output"
    assert "new-output" in replayed["messages"][3]["content"]
    assert replayed["info"]["config"]["environment_type"].endswith(".ExecutorEnvironment")
