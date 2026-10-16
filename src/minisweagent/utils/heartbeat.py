"""Heartbeat file for host-guest clock calibration (plan section 5, clock probe).

While an episode runs, the guest periodically writes wall+monotonic timestamps
to a well-known file next to the store. When the store lives on the shared
mount (virtiofs/9p), the harness reads this file and stamps host time to derive
clock_offset; cross-layer alignment is always "guest wall - offset -> host wall".
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Heartbeat:
    """Periodically write guest wall+monotonic timestamps to path (atomically)."""

    def __init__(self, path: str | Path, interval: float = 1.0, episode_id: str = ""):
        self.path = Path(path)
        self.interval = interval
        self.episode_id = episode_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> dict:
        payload = {"episode_id": self.episode_id, "wall_ts": time.time(), "monotonic_ts": time.monotonic()}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(self.path)
        return payload

    def __enter__(self) -> Heartbeat:
        self.beat()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        self.beat()  # leave a fresh final timestamp
        return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()
