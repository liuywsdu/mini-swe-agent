"""Replace model requests by indexed responses while keeping the real agent loop."""

from __future__ import annotations

import copy
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Literal

from minisweagent.exceptions import FormatError
from minisweagent.models import get_model, get_model_class
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, format_toolcall_observation_messages
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.utils.replay_store import ReplayStore, without_secrets
from minisweagent.utils.replay_timing import FrozenTiming

OBSERVATION = "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"


@contextmanager
def _wait_boundary(store: ReplayStore, episode: str, round: int):
    """Bracket a model wait window with wait_boundary anchors (dual timestamps come from the store)."""
    store.event(episode, "wait_boundary", {"round": round, "phase": "start"}, round=round)
    try:
        yield
    finally:
        store.event(episode, "wait_boundary", {"round": round, "phase": "end"}, round=round)


@dataclass
class RecordingConfig:
    store_path: str
    episode_id: str
    backend: dict
    model_name: str = "recording"
    benchmark: str = ""
    task_id: str = ""
    attempt: int = 0
    profile_digest: str = ""


class _RecordingLitellm(LitellmModel):
    recorder: RecordingModel

    def _query(self, messages: list[dict], **kwargs):
        owner = self.recorder
        owner.attempt += 1
        request = {
            "level": "litellm.completion",
            "messages": owner.store.message_refs(messages),
            "model": self.config.model_name,
            "tools": [BASH_TOOL],
            "kwargs": without_secrets(self.config.model_kwargs | kwargs),
        }
        with owner.store.capture(owner.episode_id, owner.round, request, owner.attempt) as capture:
            response = super()._query(messages, **kwargs)
            capture.response = response.model_dump(mode="json")
            return response


class RecordingModel:
    def __init__(
        self,
        *,
        store_path: str,
        episode_id: str,
        backend: dict | None = None,
        model_name: str = "recording",
        benchmark: str = "",
        task_id: str = "",
        attempt: int = 0,
        profile_digest: str = "",
        **kwargs,
    ):
        self.config = RecordingConfig(
            store_path,
            episode_id,
            backend or {"model_name": model_name, **kwargs},
            model_name,
            benchmark,
            task_id,
            attempt,
            profile_digest,
        )
        self.store = ReplayStore(self.config.store_path)
        self.episode_id = self.config.episode_id
        self.round = self.attempt = 0
        backend = copy.deepcopy(self.config.backend)
        cls = get_model_class(backend.get("model_name", ""), backend.get("model_class", ""))
        if cls is LitellmModel:
            backend.pop("model_class", None)
            self.backend = _RecordingLitellm(**backend)
            self.backend.recorder = self
        else:
            if cls in (RecordingModel, ReplayModel):
                raise ValueError(
                    "A recorder backend must be a live or deterministic model, not another recorder/replayer"
                )
            self.backend = get_model(config=backend)
        self.store.create_episode(
            self.episode_id,
            "record",
            {
                "model_config": self.backend.config.model_dump(mode="json"),
                "sdk_attempts_recorded": cls is LitellmModel,
                "benchmark": self.config.benchmark,
                "task_id": self.config.task_id,
                "attempt": self.config.attempt,
                "profile_digest": self.config.profile_digest,
            },
        )

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.round += 1
        self.attempt = 0
        request = {
            "level": "model.query",
            "messages": self.store.message_refs(messages),
            "kwargs": without_secrets(kwargs),
        }
        with _wait_boundary(self.store, self.episode_id, self.round):
            with self.store.capture(self.episode_id, self.round, request) as capture:
                response = self.backend.query(messages, **kwargs)
                response.setdefault("extra", {})["trace"] = {"episode_id": self.episode_id, "round": self.round}
                capture.response = response
                return response

    def format_message(self, **kwargs) -> dict:
        return self.backend.format_message(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return self.backend.format_observation_messages(message, outputs, template_vars)

    def get_template_vars(self, **kwargs) -> dict:
        return self.backend.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": without_secrets(asdict(self.config)),
                    "model_type": f"{type(self).__module__}.{type(self).__name__}",
                }
            }
        }


@dataclass
class ReplayConfig:
    store_path: str
    source_episode_id: str
    episode_id: str = ""
    model_name: str = "replay"
    timing: Literal["recorded", "instant", "simulated"] = "recorded"
    time_scale: float = 1.0
    divergence: Literal["record", "error"] = "record"
    timing_manifest: str | None = None
    timing_replica_id: str = "0"
    benchmark: str = ""
    task_id: str = ""
    attempt: int = 0
    profile_digest: str = ""


class ReplayModel:
    def __init__(self, **kwargs):
        self.config = ReplayConfig(**kwargs)
        if (
            self.config.timing not in ("recorded", "instant", "simulated")
            or not math.isfinite(self.config.time_scale)
            or self.config.time_scale < 0
        ):
            raise ValueError("Use timing=recorded|instant|simulated and a finite nonnegative time_scale")
        if (self.config.timing == "simulated") != bool(self.config.timing_manifest):
            raise ValueError("timing=simulated requires timing_manifest; other timing modes must not supply it")
        if self.config.divergence not in ("record", "error"):
            raise ValueError("Use divergence=record|error")
        self.store = ReplayStore(self.config.store_path)
        self.source = self.store.metadata(self.config.source_episode_id)
        source_digest = self.source.get("profile_digest", "")
        if self.config.profile_digest and source_digest and self.config.profile_digest != source_digest:
            raise ValueError(
                f"profile_digest mismatch: {self.config.profile_digest} != source {source_digest}; refusing to start"
            )
        self.frozen_timing = (
            FrozenTiming(
                self.config.timing_manifest, self.store, self.config.source_episode_id, self.config.timing_replica_id
            )
            if self.config.timing == "simulated"
            else None
        )
        self.episode_id = self.config.episode_id or "replay-" + uuid.uuid4().hex
        self.config.episode_id = self.episode_id
        self.round = 0
        self.store.create_episode(
            self.episode_id,
            "replay",
            {
                "source_episode_id": self.config.source_episode_id,
                "model_config": self.source.get("model_config", {}),
                "timing": self.config.timing,
                "time_scale": self.config.time_scale,
                "simulated_timing": self.frozen_timing.reference if self.frozen_timing else None,
                "benchmark": self.config.benchmark or self.source.get("benchmark", ""),
                "task_id": self.config.task_id or self.source.get("task_id", ""),
                "attempt": self.config.attempt,
                "profile_digest": self.config.profile_digest or source_digest,
            },
        )

    def _audit(self, expected_refs: list[str], messages: list[dict]) -> dict:
        actual_refs = self.store.message_refs(messages)
        expected = [self.store.get(ref) for ref in expected_refs]

        def structure(rows: list[dict]) -> list[tuple]:
            return [
                (m.get("role"), m.get("tool_call_id"), [t.get("id") for t in m.get("tool_calls") or []]) for m in rows
            ]

        differences = [
            {
                "index": index,
                "role": messages[index].get("role"),
                "tool_call_id": messages[index].get("tool_call_id"),
                "expected": left,
                "actual": right,
            }
            for index, (left, right) in enumerate(zip(expected_refs, actual_refs))
            if left != right
        ]
        audit = {
            "round": self.round,
            "structure_matches": structure(expected) == structure(messages),
            "expected_message_count": len(expected_refs),
            "actual_message_count": len(messages),
            "differences": differences,
            "actual_messages": actual_refs,
        }
        self.store.event(self.episode_id, "replay_match", audit, round=self.round)
        if not audit["structure_matches"]:
            raise ValueError(
                f"Replay round {self.round}: message/tool-call structure changed; refusing to shift rounds"
            )
        if differences and self.config.divergence == "error":
            raise ValueError(f"Replay round {self.round}: input differs from recording; see replay_match events")
        return audit

    def query(self, messages: list[dict], **kwargs) -> dict:
        started = time.perf_counter()
        self.round += 1
        with _wait_boundary(self.store, self.episode_id, self.round):
            call = self.store.call(self.config.source_episode_id, self.round)
            if call["status"] == "pending":
                raise ValueError(f"Replay round {self.round} was interrupted before its response was saved")
            audit = self._audit(call["request"]["messages"], messages)
            wait_seconds = 0.0
            target_seconds = 0.0
            simulated = self.frozen_timing.row(self.round) if self.frozen_timing else None
            if self.config.timing == "recorded":
                if call["duration"] is None:
                    raise ValueError(
                        "Recorded model duration is unavailable; import timing events or explicitly use timing=instant"
                    )
                if not math.isfinite(call["duration"]) or call["duration"] < 0:
                    raise ValueError("Recorded model duration must be finite and nonnegative")
                target_seconds = call["duration"] * self.config.time_scale
            elif simulated is not None:
                target_seconds = simulated["duration_seconds"] * self.config.time_scale
            if not math.isfinite(target_seconds):
                raise ValueError("Scaled replay duration must be finite")
            before_sleep = time.perf_counter()
            if self.config.timing != "instant":
                wait_seconds = max(0, target_seconds - (before_sleep - started))
                time.sleep(wait_seconds)
            after_sleep = time.perf_counter()
            self.store.event(
                self.episode_id,
                "replay_timing",
                {
                    "round": self.round,
                    "timing": self.config.timing,
                    "time_scale": self.config.time_scale,
                    "source_duration_seconds": call["duration"],
                    "simulated_duration_seconds": simulated["duration_seconds"] if simulated else None,
                    "simulation": (self.frozen_timing.reference | {"donor_sample_id": simulated["donor_sample_id"]})
                    if simulated
                    else None,
                    "target_duration_seconds": target_seconds,
                    "requested_sleep_seconds": wait_seconds,
                    "actual_sleep_seconds": after_sleep - before_sleep,
                    "pre_wait_seconds": before_sleep - started,
                    "deadline_lateness_seconds": max(0, after_sleep - started - target_seconds)
                    if self.config.timing != "instant"
                    else None,
                    "elapsed_seconds": after_sleep - started,
                },
                round=self.round,
            )
            if call["error"] is not None:
                error = call["error"]
                if error["type"] == "FormatError":
                    messages = copy.deepcopy(error["messages"])
                    for message in messages:
                        extra = message.setdefault("extra", {})
                        extra["source_cost"] = extra.get("cost", 0.0)
                        extra["cost"] = 0.0
                    raise FormatError(*messages)
                raise RuntimeError(f"Recorded model failure in round {self.round}: {error['type']}; no API fallback")
            if not isinstance(call["response"], dict):
                raise ValueError(f"Replay round {self.round} has no model response")
            response = copy.deepcopy(call["response"])
            extra = response.setdefault("extra", {})
            extra["replay"] = {
                "source_episode_id": self.config.source_episode_id,
                "source_round": self.round,
                "episode_id": self.episode_id,
                "source_cost": extra.get("cost", 0.0),
                "source_timestamp": extra.get("timestamp"),
                "source_duration_seconds": call["duration"],
                "simulated_duration_seconds": simulated["duration_seconds"] if simulated else None,
                "timing_manifest_sha256": self.frozen_timing.sha256 if self.frozen_timing else None,
                "actual_api_calls": 0,
                "input_difference_count": len(audit["differences"]),
            }
            extra["cost"] = 0.0
            extra["timestamp"] = time.time()
            return response

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(
            kwargs, pattern=self.source.get("model_config", {}).get("multimodal_regex", "")
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        result = format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.source.get("model_config", {}).get("observation_template", OBSERVATION),
            template_vars=template_vars,
            multimodal_regex=self.source.get("model_config", {}).get("multimodal_regex", ""),
        )
        self.store.event(
            self.episode_id,
            "tool_results",
            {
                "round": self.round,
                "messages": self.store.message_refs(result),
                "returncodes": [r.get("returncode") for r in outputs],
                "durations": [(r.get("timing") or {}).get("monotonic_seconds") for r in outputs],
            },
            round=self.round,
        )
        return result

    def get_template_vars(self, **kwargs) -> dict:
        return self.source.get("model_config", {})

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": asdict(self.config),
                    "model_type": f"{type(self).__module__}.{type(self).__name__}",
                },
                "replay": {
                    "source_episode_id": self.config.source_episode_id,
                    "source_model_config": self.source.get("model_config", {}),
                    "episode_id": self.episode_id,
                    "model_rounds": self.round,
                    "actual_api_calls": 0,
                    "actual_cost": 0.0,
                },
            }
        }
