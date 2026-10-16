"""Indexed model calls and deduplicated artifacts for recording and replay.

Event conventions (benchmark data collection plan, sections 2 and 6):

- Every event carries dual timestamps: ``wall_ts`` (``time.time``) and
  ``monotonic_ts`` (``time.monotonic``), auto-stamped by ``ReplayStore.event``.
  Rows recorded before these columns existed have NULL ``monotonic_ts`` and a
  ``wall_ts`` backfilled from ``created``.
- ``round`` is the agent loop round the event belongs to; NULL for
  session-level events.
- ``layer`` declares the measurement layer: ``container_cgroup`` or
  ``guest_cgroup`` (produced by mini); ``host_vmm`` is produced by the VM
  harness and never appears here. NULL when not applicable.
- Metric event kinds and their payload fields:
  - ``action_timing``: ``{round, action_index, timing, resources}``
  - ``wait_boundary``: ``{round, phase}`` with ``phase`` in ``start``/``end``
  - ``session_snapshot``: ``{final, workspace_bytes, ...}``; ``final: true``
    marks episode completeness
  - ``syscall_profile``: ``{round, action_index, blob}`` (blob is an artifact key)
- Unsupported measurements are recorded as null, never omitted silently.
- Export shape for ``guest_events.jsonl``: ``{kind, round, wall_ts,
  monotonic_ts, layer, payload}`` as returned by ``ReplayStore.event_rows``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if k.lower().replace("-", "_") in {"api_key", "authorization", "api_token", "access_token"}
            else without_secrets(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [without_secrets(v) for v in value]
    return value


class ReplayStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "blobs").mkdir(exist_ok=True)
        self.database = self.path / "index.sqlite3"
        with closing(self.connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY, mode TEXT NOT NULL, metadata TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    episode TEXT NOT NULL, round INTEGER NOT NULL, attempt INTEGER NOT NULL,
                    request TEXT NOT NULL, response TEXT, error TEXT, duration REAL,
                    status TEXT NOT NULL, started REAL NOT NULL,
                    PRIMARY KEY (episode, round, attempt));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, episode TEXT NOT NULL, kind TEXT NOT NULL,
                    round INTEGER, payload TEXT NOT NULL, created REAL NOT NULL,
                    wall_ts REAL, monotonic_ts REAL, layer TEXT);
            """)
            self._migrate_events(db)
            db.executescript("""
                CREATE INDEX IF NOT EXISTS episode_events ON events(episode, id);
                CREATE INDEX IF NOT EXISTS episode_round_events ON events(episode, round, id);
            """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def put_text(self, text: str) -> str:
        temporary = self.path / "blobs" / (uuid.uuid4().hex + ".tmp")
        digest = hashlib.sha256()
        with gzip.open(temporary, "wb") as stream:
            for start in range(0, len(text), 1024 * 1024):
                chunk = text[start : start + 1024 * 1024].encode("utf-8")
                digest.update(chunk)
                stream.write(chunk)
        key = digest.hexdigest()
        destination = self.path / "blobs" / (key + ".gz")
        if destination.exists():
            temporary.unlink()
        else:
            temporary.replace(destination)
        return key

    def put(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not (self.path / "blobs" / (key + ".gz")).exists():
            self.put_text(text)
        return key

    def text(self, key: str) -> str:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact ID")
        data = gzip.decompress((self.path / "blobs" / (key + ".gz")).read_bytes())
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError(f"Corrupt replay artifact: {key}")
        return data.decode("utf-8")

    def get(self, key: str) -> Any:
        return json.loads(self.text(key))

    def message_refs(self, messages: list[dict]) -> list[str]:
        return [self.put({k: v for k, v in message.items() if k != "extra"}) for message in messages]

    def create_episode(self, episode: str, mode: str, metadata: dict) -> None:
        with closing(self.connect()) as db:
            db.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?)",
                (episode, mode, self.put(without_secrets(metadata)), time.time()),
            )

    def metadata(self, episode: str) -> dict:
        with closing(self.connect()) as db:
            row = db.execute("SELECT metadata FROM episodes WHERE id=?", (episode,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown episode: {episode}")
        return self.get(row["metadata"])

    def update_metadata(self, episode: str, metadata: dict) -> None:
        with closing(self.connect()) as db:
            db.execute("UPDATE episodes SET metadata=? WHERE id=?", (self.put(without_secrets(metadata)), episode))

    @staticmethod
    def _migrate_events(db: sqlite3.Connection) -> None:
        existing = {row["name"] for row in db.execute("PRAGMA table_info(events)")}
        for name, ddl in {"round": "INTEGER", "wall_ts": "REAL", "monotonic_ts": "REAL", "layer": "TEXT"}.items():
            if name not in existing:
                db.execute(f"ALTER TABLE events ADD COLUMN {name} {ddl}")
        if "wall_ts" not in existing:
            db.execute("UPDATE events SET wall_ts=created WHERE wall_ts IS NULL")

    def event(self, episode: str, kind: str, value: Any, *, round: int | None = None, layer: str | None = None) -> None:
        reference = self.put(value)
        wall = time.time()
        with closing(self.connect()) as db:
            db.execute(
                "INSERT INTO events(episode, kind, round, payload, created, wall_ts, monotonic_ts, layer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (episode, kind, round, reference, wall, wall, time.monotonic(), layer),
            )

    def events(self, episode: str, kind: str, *, round: int | None = None) -> list[Any]:
        return [row["payload"] for row in self.event_rows(episode, kind, round=round)]

    def event_rows(self, episode: str, kind: str | None = None, *, round: int | None = None) -> list[dict]:
        """Events as {kind, round, wall_ts, monotonic_ts, layer, payload} (guest_events.jsonl shape)."""
        query = "SELECT kind, round, wall_ts, monotonic_ts, layer, payload FROM events WHERE episode=?"
        params: list = [episode]
        if kind is not None:
            query += " AND kind=?"
            params.append(kind)
        if round is not None:
            query += " AND round=?"
            params.append(round)
        with closing(self.connect()) as db:
            rows = db.execute(query + " ORDER BY id", params).fetchall()
        return [
            {
                "kind": row["kind"],
                "round": row["round"],
                "wall_ts": row["wall_ts"],
                "monotonic_ts": row["monotonic_ts"],
                "layer": row["layer"],
                "payload": self.get(row["payload"]),
            }
            for row in rows
        ]

    def completed(self, episode: str) -> bool:
        """Episode completeness gate for the harness: a session_snapshot with final: true exists."""
        return any(event.get("final") for event in self.events(episode, "session_snapshot"))

    def start_call(self, episode: str, round: int, attempt: int, request: dict) -> None:
        reference = self.put(request)
        with closing(self.connect()) as db:
            db.execute(
                "INSERT INTO calls(episode, round, attempt, request, status, started) VALUES (?, ?, ?, ?, 'pending', ?)",
                (episode, round, attempt, reference, time.time()),
            )

    def finish_call(
        self,
        episode: str,
        round: int,
        attempt: int,
        *,
        response: Any = None,
        error: dict | None = None,
        duration: float | None = None,
    ) -> None:
        response_ref = self.put(response) if response is not None else None
        error_ref = self.put(error) if error is not None else None
        with closing(self.connect()) as db:
            changed = db.execute(
                """UPDATE calls SET response=?, error=?, duration=?, status=?
                WHERE episode=? AND round=? AND attempt=? AND status='pending'""",
                (response_ref, error_ref, duration, "error" if error is not None else "ok", episode, round, attempt),
            )
        if changed.rowcount != 1:
            raise ValueError("Cannot overwrite a completed or missing call")

    def call(self, episode: str, round: int, attempt: int = 0) -> dict:
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT * FROM calls WHERE episode=? AND round=? AND attempt=?", (episode, round, attempt)
            ).fetchone()
        if row is None:
            raise KeyError(
                f"Missing replay record: episode={episode!r}, round={round}, attempt={attempt}; no API fallback"
            )
        result = dict(row)
        for field in ("request", "response", "error"):
            result[field] = self.get(result[field]) if result[field] is not None else None
        return result

    def capture(self, episode: str, round: int, request: dict, attempt: int = 0) -> CallCapture:
        return CallCapture(self, episode, round, attempt, request)


@dataclass
class CallCapture:
    store: ReplayStore
    episode: str
    round: int
    attempt: int
    request: dict
    response: Any = None
    started: float = 0.0

    def __enter__(self) -> CallCapture:
        self.store.start_call(self.episode, self.round, self.attempt, self.request)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        error = None
        if exc_type is not None:
            error = {
                "type": exc_type.__name__,
                "module": exc_type.__module__,
                "status_code": getattr(exc, "status_code", None),
                "messages": list(getattr(exc, "messages", [])),
            }
        self.store.finish_call(
            self.episode,
            self.round,
            self.attempt,
            response=self.response,
            error=error,
            duration=time.perf_counter() - self.started,
        )
        return False
