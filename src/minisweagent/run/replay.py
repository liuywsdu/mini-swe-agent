"""Import, inspect and export model tapes; replay using the normal agent loop."""

from __future__ import annotations

import copy
import json
import time
from contextlib import closing
from pathlib import Path

import typer
import yaml

from minisweagent.agents.journal import JournalAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment, get_environment_class
from minisweagent.environments.executor import ExecutorEnvironment
from minisweagent.executors import get_executor_class
from minisweagent.executors.docker import DockerExecutor
from minisweagent.executors.local import LocalExecutor
from minisweagent.executors.remote import RemoteExecutor
from minisweagent.models.replay import ReplayModel
from minisweagent.utils.replay_store import ReplayStore, without_secrets
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(no_args_is_help=True)


def import_trajectory(
    store: ReplayStore, trajectory: Path, episode_id: str, events: Path | None = None, task_id: str = ""
) -> dict:
    data = json.loads(trajectory.read_text())
    if data.get("trajectory_format") == "mini-swe-agent-index-1":
        raise ValueError("An indexed trajectory already belongs to a replay store; use its episode_id directly")
    messages = data["messages"]
    selected = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" or message.get("extra", {}).get("interrupt_type") == "FormatError"
    ]
    if any(message.get("role") == "assistant" and "actions" not in message.get("extra", {}) for _, message in selected):
        raise ValueError("Import currently requires mini tool-call trajectories with parsed actions")
    timings = []
    if events is not None:
        for line in events.open():
            row = json.loads(line)
            if row.get("kind") == "model" and (row.get("task_id") or row.get("instance_id")) == task_id:
                timings.append(row["duration_s"])
        if len(timings) != len(selected):
            raise ValueError(f"Timing/model-round mismatch: {len(timings)} durations for {len(selected)} rounds")
    if data["info"]["model_stats"]["api_calls"] != len(selected):
        raise ValueError("Trajectory contains a missing model response; cannot silently renumber its rounds")
    model_config = data["info"]["config"]["model"]
    if data["info"]["config"].get("model_type", "").endswith(".RecordingModel"):
        model_config = model_config["backend"]
    model_config = data["info"].get("replay", {}).get("source_model_config", model_config)
    metadata = {
        "source_file": str(trajectory.resolve()),
        "task_id": task_id,
        "model_config": model_config,
        "trajectory_info": data["info"],
        "rounds": len(selected),
        "timing_source": str(events.resolve()) if events else None,
        "request_provenance": "reconstructed_from_trajectory",
        "sdk_attempts_recorded": False,
        "initial_messages": store.message_refs(messages[:2]),
    }
    store.create_episode(episode_id, "import", metadata)
    refs = store.message_refs(messages)
    for round, (index, message) in enumerate(selected, start=1):
        store.start_call(
            episode_id,
            round,
            0,
            {"level": "model.query", "messages": refs[:index], "kwargs": {}, "provenance": "legacy_reconstruction"},
        )
        error = None
        if message.get("extra", {}).get("interrupt_type") == "FormatError":
            error = {"type": "FormatError", "messages": [message]}
        store.finish_call(
            episode_id,
            round,
            0,
            response=None if error else message,
            error=error,
            duration=timings[round - 1] if timings else None,
        )
    for index, message in enumerate(messages):
        compact = dict(message)
        if "extra" in compact:
            compact["extra"] = dict(compact["extra"])
            if "raw_output" in compact["extra"]:
                compact["extra"]["raw_output_ref"] = store.put_text(compact["extra"].pop("raw_output"))
        store.event(episode_id, "message", {"index": index, "ref": store.put(compact)})
    return without_secrets(metadata)


def export_trajectory(store: ReplayStore, episode_id: str) -> dict:
    metadata = store.metadata(episode_id)
    messages = []
    for event in store.events(episode_id, "message"):
        message = store.get(event["ref"])
        extra = message.get("extra", {})
        if "raw_output_ref" in extra:
            extra["raw_output"] = store.text(extra.pop("raw_output_ref"))
        messages.append(message)
    if not messages:
        raise ValueError("No agent message journal: use JournalAgent when recording to support full trajectory export")
    info = metadata.get("trajectory", {}).get("info", metadata.get("trajectory_info", {}))
    return {"info": info, "messages": messages, "trajectory_format": "mini-swe-agent-1.1"}


@app.command("import")
def import_command(
    trajectory: Path,
    store: Path = typer.Option(...),
    episode: str = typer.Option(...),
    events: Path | None = typer.Option(None, help="Original profiler events.jsonl, for exact model-call durations"),
    task_id: str = typer.Option("", help="ID in the timing events; defaults to trajectory's parent directory name"),
    benchmark: str = typer.Option("", help="Benchmark collection this episode belongs to"),
    attempt: int = typer.Option(0, help="Attempt number within the (benchmark, task_id) matrix"),
    profile_digest: str = typer.Option("", help="Digest of the measurement profile this episode runs under"),
) -> None:
    tape = ReplayStore(store)
    result = import_trajectory(tape, trajectory, episode, events, task_id or trajectory.parent.name)
    metadata = tape.metadata(episode)
    metadata.update(benchmark=benchmark, attempt=attempt, profile_digest=profile_digest)
    tape.update_metadata(episode, metadata)
    typer.echo(
        json.dumps({"episode_id": episode, "rounds": result["rounds"], "timing_source": result["timing_source"]})
    )


@app.command("export")
def export_command(
    store: Path = typer.Option(...), episode: str = typer.Option(...), output: Path = typer.Option(...)
) -> None:
    if output.exists():
        raise ValueError("Refusing to overwrite an existing trajectory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export_trajectory(ReplayStore(store), episode), ensure_ascii=False, indent=2))
    typer.echo(str(output))


@app.command("export-events")
def export_events_command(
    store: Path = typer.Option(...),
    episode: str = typer.Option(...),
    output: Path = typer.Option(..., help="guest_events.jsonl destination; existing files are never overwritten"),
) -> None:
    """Export guest-layer events as JSONL: {kind, round, wall_ts, monotonic_ts, layer, payload} per line."""
    if output.exists():
        raise ValueError("Refusing to overwrite an existing events file")
    tape = ReplayStore(store)
    rows = tape.event_rows(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    typer.echo(
        json.dumps(
            {"episode_id": episode, "events": len(rows), "completed": tape.completed(episode)},
            ensure_ascii=False,
        )
    )


@app.command("inspect")
def inspect_command(store: Path = typer.Option(...), episode: str = typer.Option(...)) -> None:
    tape = ReplayStore(store)
    with closing(tape.connect()) as db:
        calls = [
            dict(row)
            for row in db.execute(
                "SELECT round, attempt, status, duration FROM calls WHERE episode=? ORDER BY round, attempt", (episode,)
            )
        ]
    typer.echo(
        json.dumps(
            {
                "episode_id": episode,
                "calls": calls,
                "replay_checks": tape.events(episode, "replay_match"),
                "replay_timings": tape.events(episode, "replay_timing"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def get_replay_environment_config(source_config: dict, episode: str, overrides: dict | None = None) -> dict:
    config = copy.deepcopy(source_config["environment"])
    config["environment_class"] = source_config.get("environment_type")
    if overrides is not None:
        overrides = copy.deepcopy(overrides.get("environment", overrides))
        if not config["environment_class"] or (
            "environment_class" in overrides
            and get_environment_class(overrides["environment_class"])
            is not get_environment_class(config["environment_class"])
        ):
            # Foreign or missing source environment: the override fully takes over
            config = {}
        config.update(overrides)
    if not config.get("environment_class"):
        raise ValueError(
            "Source episode has no usable environment_type; provide --environment-config with an environment_class"
        )
    environment_class = get_environment_class(config["environment_class"])
    executor_class = getattr(environment_class, "executor_class", None)
    execution_config = config
    if issubclass(environment_class, ExecutorEnvironment) and executor_class is None:
        execution_config = recursive_merge(
            {key: value for key, value in config.items() if key not in ("executor", "environment_class")},
            config["executor"],
        )
        config = {"environment_class": config["environment_class"], "executor": execution_config}
        executor_class = get_executor_class(execution_config.get("backend", "local"))
    fresh_remote_workspace = False
    if executor_class is not None and issubclass(executor_class, RemoteExecutor):
        remote_config = execution_config
        execution_config = remote_config["sandbox"]
        executor_class = get_executor_class(execution_config["backend"])
        fresh_remote_workspace = not execution_config.get("cwd") and not remote_config.get("cwd")
    if overrides is None:
        if executor_class is not None and issubclass(executor_class, LocalExecutor) and not fresh_remote_workspace:
            raise ValueError("Provide --environment-config with a fresh workspace before replaying a local environment")
        if any(
            arg in ("-v", "--volume", "--mount", "--volumes-from")
            or arg.startswith(("--volume=", "--mount=", "--volumes-from=", "-v/"))
            for arg in execution_config.get("run_args", [])
        ):
            raise ValueError("Source uses mounted storage; provide --environment-config pointing to a fresh workspace")
    if executor_class is not None and issubclass(executor_class, DockerExecutor):
        execution_config["run_args"] = [
            *execution_config.get("run_args", ["--rm"]),
            "--pull=never",
            "--label",
            f"agentos.replay={episode}",
        ]
    return config


@app.command("run")
def run_command(
    store: Path = typer.Option(...),
    source: str = typer.Option(..., help="Source episode ID"),
    output: Path = typer.Option(..., help="New indexed trajectory; existing files are never overwritten"),
    episode: str = typer.Option("", help="New replay episode ID; defaults to a UUID"),
    timing: str = typer.Option("recorded", help="recorded, instant or simulated"),
    time_scale: float = typer.Option(1.0, help="Multiplier for recorded or simulated response durations"),
    timing_manifest: Path | None = typer.Option(None, help="Frozen simulated timing file or manifest directory"),
    timing_replica_id: str = typer.Option("0", help="Replica ID bound into the simulated timing manifest"),
    divergence: str = typer.Option("record", help="record or error for changed input text"),
    environment_config: Path | None = typer.Option(None, help="Optional JSON/YAML environment override"),
    benchmark: str = typer.Option("", help="Benchmark collection this episode belongs to"),
    task_id: str = typer.Option("", help="Task ID within the benchmark; defaults to the source episode's"),
    attempt: int = typer.Option(0, help="Attempt number within the (benchmark, task_id) matrix"),
    profile_digest: str = typer.Option(
        "", help="Expected measurement profile digest; a mismatch with the source episode aborts the run"
    ),
) -> None:
    if output.exists():
        raise ValueError("Refusing to overwrite an existing trajectory")
    model = ReplayModel(
        store_path=str(store),
        source_episode_id=source,
        episode_id=episode,
        timing=timing,
        time_scale=time_scale,
        divergence=divergence,
        timing_manifest=str(timing_manifest) if timing_manifest is not None else None,
        timing_replica_id=timing_replica_id,
        benchmark=benchmark,
        task_id=task_id,
        attempt=attempt,
        profile_digest=profile_digest,
    )
    source_info = model.source.get("trajectory_info", model.source.get("trajectory", {}).get("info", {}))
    if not source_info:
        raise ValueError("Agent/environment metadata missing; record using JournalAgent or supply your own run script")
    config = source_info["config"]
    with closing(model.store.connect()) as db:
        unavailable = db.execute(
            "SELECT COUNT(*) FROM calls WHERE episode=? AND attempt=0 AND (status='pending' OR duration IS NULL)",
            (source,),
        ).fetchone()[0]
    if timing == "recorded" and unavailable:
        raise ValueError("Source has incomplete calls or missing durations; no environment was started")
    overrides = None
    if environment_config is not None:
        overrides = (
            yaml.safe_load(environment_config.read_text())
            if environment_config.is_file()
            else get_config_from_spec(environment_config)
        )
    env_config = get_replay_environment_config(config, model.episode_id, overrides)
    first = model.store.call(source, 1)["request"]["messages"]
    prompts = [model.store.get(ref) for ref in first]
    if len(prompts) != 2 or [m.get("role") for m in prompts] != ["system", "user"]:
        raise ValueError("CLI expects an initial system/user pair; use ReplayModel in a custom run script otherwise")
    env = get_environment(env_config)
    try:
        started = time.perf_counter()
        agent_config = dict(config["agent"])
        agent_config.update(system_template="{{ replay_system }}", instance_template="{{ task }}", output_path=output)
        agent = JournalAgent(model, env, **agent_config)
        result = agent.run(prompts[1]["content"], replay_system=prompts[0]["content"])
        expected = source_info.get("submission")
        summary = {
            "episode_id": model.episode_id,
            "source_episode_id": source,
            "actual_api_calls": 0,
            "actual_cost": 0.0,
            "model_rounds": agent.n_calls,
            "agent_wall_seconds": time.perf_counter() - started,
            "replay_model_seconds": sum(
                e["elapsed_seconds"] for e in model.store.events(model.episode_id, "replay_timing")
            ),
            "timing": timing,
            "time_scale": time_scale,
            "exit_status": result.get("exit_status"),
            "submission_matches": result.get("submission") == expected if expected is not None else None,
            "input_difference_rounds": sum(
                bool(e["differences"]) for e in model.store.events(model.episode_id, "replay_match")
            ),
        }
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        typer.echo(json.dumps(summary))
    finally:
        if hasattr(env, "cleanup"):
            env.cleanup()


if __name__ == "__main__":
    app()
