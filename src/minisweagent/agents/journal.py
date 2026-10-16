"""The default agent loop with incremental, artifact-backed trajectory storage."""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.utils.replay_store import without_secrets
from minisweagent.utils.netdev import diff_net_dev
from minisweagent.utils.heartbeat import Heartbeat


class JournalAgent(DefaultAgent):
    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, **kwargs)
        self.store, self.episode_id = model.store, model.episode_id
        self.message_refs: list[str] = []
        self.save(None)

    def add_messages(self, *messages: dict) -> list[dict]:
        compact = []
        for message in messages:
            item = dict(message)
            if "extra" in item:
                item["extra"] = dict(item["extra"])
                if "raw_output" in item["extra"]:
                    item["extra"]["raw_output_ref"] = self.store.put_text(item["extra"].pop("raw_output"))
            reference = self.store.put(item)
            self.store.event(self.episode_id, "message", {"index": len(self.message_refs), "ref": reference})
            self.message_refs.append(reference)
            compact.append(item)
        return super().add_messages(*compact)

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for index, action in enumerate(message.get("extra", {}).get("actions", [])):
            try:
                outputs.append(self.env.execute(action))
            except Submitted as e:
                self._record_action_timing(index, getattr(e, "execution_output", None))
                raise
            self._record_action_timing(index, outputs[-1])
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _record_action_timing(self, action_index: int, output: dict | None) -> None:
        resources = (output or {}).get("resources")
        self.store.event(
            self.episode_id,
            "action_timing",
            {
                "round": self.model.round,
                "action_index": action_index,
                "timing": (output or {}).get("timing"),
                "resources": resources,
                "workspace_bytes": self._workspace_bytes()
                if self.config.profile_workspace_per_action
                else None,
            },
            round=self.model.round,
            layer=(resources or {}).get("layer"),
        )
        profile = (output or {}).get("profile")
        if profile:
            self.store.event(
                self.episode_id,
                "syscall_profile",
                {"round": self.model.round, "action_index": action_index, "blob": self.store.put_text(profile)},
                round=self.model.round,
                layer=(resources or {}).get("layer"),
            )

    def _workspace_bytes(self) -> int | None:
        """Workspace occupancy via du inside the sandbox; measurement failures are recorded as null."""
        try:
            output = self.env.execute({"command": "du -sb ."})
            return int(output["output"].split()[0]) if output.get("returncode") == 0 else None
        except Exception:
            return None

    def _executor_net_stats(self) -> dict[str, int] | None:
        reader = getattr(getattr(self.env, "executor", None), "net_dev_stats", None)
        return reader() if reader is not None else None

    def _record_session_snapshot(self, final: bool) -> None:
        executor = getattr(self.env, "executor", None)
        network = self._executor_net_stats()
        if not final:
            self._net_start = network
        payload = {"final": final, "workspace_bytes": self._workspace_bytes(), "network": network}
        if final:
            payload["network_delta"] = diff_net_dev(getattr(self, "_net_start", None), network)
        self.store.event(
            self.episode_id,
            "session_snapshot",
            payload,
            layer=getattr(executor, "cgroup_layer", None),
        )

    def run(self, task: str = "", **kwargs) -> dict:
        if self.message_refs:
            raise ValueError("Create a fresh model and episode for each agent task")
        executor = getattr(self.env, "executor", None)
        metadata = self.store.metadata(self.episode_id)
        metadata["network_observation_point"] = getattr(executor, "network_observation_point", None)
        self.store.update_metadata(self.episode_id, metadata)
        with Heartbeat(self.store.path / "heartbeat.json", episode_id=self.episode_id):
            self._record_session_snapshot(final=False)
            try:
                return super().run(task, **kwargs)
            finally:
                self._record_session_snapshot(final=True)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        data = self.serialize(*extra_dicts)
        data.pop("messages")
        data = without_secrets(data)
        data.update(
            trajectory_format="mini-swe-agent-index-1",
            store_path=str(self.store.path),
            episode_id=self.episode_id,
            message_refs=list(self.message_refs),
        )
        metadata = self.store.metadata(self.episode_id)
        metadata["trajectory"] = data
        self.store.update_metadata(self.episode_id, metadata)
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            temporary.replace(path)
        return data
