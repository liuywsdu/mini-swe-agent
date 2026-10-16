"""cgroup CPU/memory/IO counters for docker-layer resource attribution (plan sections 2/3).

v2 is primary; v1 is a fallback. Parsers are pure functions separated from file
IO for testing. Counters are cumulative since cgroup creation; use
``diff_cgroup_stats`` at action boundaries. Unsupported fields read as None; a
wholly unavailable cgroup (non-Linux, missing cgroupfs) reads as None.
"""

from __future__ import annotations

from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def parse_keyed(text: str) -> dict[str, int]:
    """'key value' pairs, e.g. cgroup v2 cpu.stat."""
    result = {}
    for line in text.splitlines():
        key, sep, value = line.partition(" ")
        if sep and value.isdigit():
            result[key] = int(value)
    return result


def parse_io_stat(text: str) -> dict[str, int]:
    """Sum rbytes/wbytes across devices (cgroup v2 io.stat)."""
    totals = {"rbytes": 0, "wbytes": 0}
    for line in text.splitlines():
        for field in line.split()[1:]:
            key, sep, value = field.partition("=")
            if sep and key in totals:
                totals[key] += int(value)
    return totals


def parse_io_service_bytes(text: str) -> dict[str, int]:
    """Sum Read/Write bytes across devices (cgroup v1 blkio *io_service_bytes)."""
    totals = {"rbytes": 0, "wbytes": 0}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("Read", "Write"):
            totals["rbytes" if parts[1] == "Read" else "wbytes"] += int(parts[2])
    return totals


def parse_proc_self_cgroup(text: str) -> tuple[str | None, str | None]:
    """(v2 relative path, v1 relative path) from /proc/self/cgroup contents."""
    v2_rel = v1_rel = None
    for line in text.splitlines():
        _, controllers, rel = line.split(":", 2)
        if controllers == "":
            v2_rel = rel
        elif v1_rel is None and "cpuacct" in controllers.split(","):
            v1_rel = rel
    return v2_rel, v1_rel


def _text(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _int(text: str | None) -> int | None:
    return int(text.strip()) if text is not None else None


def read_cgroup_v2_stats(path: Path) -> dict[str, int | None]:
    cpu_stat, io_stat = _text(path / "cpu.stat"), _text(path / "io.stat")
    io = parse_io_stat(io_stat) if io_stat is not None else None
    return {
        "cpu_usage_usec": parse_keyed(cpu_stat).get("usage_usec") if cpu_stat is not None else None,
        "memory_current_bytes": _int(_text(path / "memory.current")),
        "memory_peak_bytes": _int(_text(path / "memory.peak")),
        "io_read_bytes": io["rbytes"] if io is not None else None,
        "io_write_bytes": io["wbytes"] if io is not None else None,
    }


def read_cgroup_v1_stats(rel: str, root: Path = CGROUP_ROOT) -> dict[str, int | None]:
    usage_ns = _int(_text(root / "cpuacct" / rel / "cpuacct.usage"))
    io_text = _text(root / "blkio" / rel / "blkio.throttle.io_service_bytes")
    if io_text is None:
        io_text = _text(root / "blkio" / rel / "blkio.io_service_bytes")
    io = parse_io_service_bytes(io_text) if io_text is not None else None
    return {
        "cpu_usage_usec": usage_ns // 1000 if usage_ns is not None else None,
        "memory_current_bytes": _int(_text(root / "memory" / rel / "memory.usage_in_bytes")),
        "memory_peak_bytes": _int(_text(root / "memory" / rel / "memory.max_usage_in_bytes")),
        "io_read_bytes": io["rbytes"] if io is not None else None,
        "io_write_bytes": io["wbytes"] if io is not None else None,
    }


def read_own_cgroup_stats(root: Path = CGROUP_ROOT, proc: Path = PROC_SELF_CGROUP) -> dict[str, int | None] | None:
    """Counters for the cgroup this process belongs to (guest layer for the local executor)."""
    if not proc.is_file():
        return None
    v2_rel, v1_rel = parse_proc_self_cgroup(proc.read_text())
    if v2_rel is not None:
        path = root / v2_rel.lstrip("/")
        return read_cgroup_v2_stats(path) if path.is_dir() else None
    return read_cgroup_v1_stats(v1_rel, root) if v1_rel is not None else None


def read_container_cgroup_stats(container_id: str, root: Path = CGROUP_ROOT) -> dict[str, int | None] | None:
    """Counters for a docker container's cgroup, located from its container ID."""
    for candidate in (root / "system.slice" / f"docker-{container_id}.scope", root / "docker" / container_id):
        if (candidate / "cgroup.controllers").exists() or (candidate / "cpu.stat").exists():
            return read_cgroup_v2_stats(candidate)
    for rel in (f"docker/{container_id}", f"system.slice/docker-{container_id}.scope"):
        if (root / "cpuacct" / rel).is_dir() or (root / "memory" / rel).is_dir():
            return read_cgroup_v1_stats(rel, root)
    return None


def _delta(before: int | None, after: int | None) -> int | None:
    return after - before if before is not None and after is not None else None


def diff_cgroup_stats(before: dict | None, after: dict | None) -> dict[str, int | None] | None:
    """Action-boundary diff: deltas for cumulative counters, latest values for gauges."""
    if before is None or after is None:
        return None
    return {
        "cpu_usage_usec": _delta(before["cpu_usage_usec"], after["cpu_usage_usec"]),
        "memory_current_bytes": after["memory_current_bytes"],
        "memory_peak_bytes": after["memory_peak_bytes"],
        "io_read_bytes": _delta(before["io_read_bytes"], after["io_read_bytes"]),
        "io_write_bytes": _delta(before["io_write_bytes"], after["io_write_bytes"]),
    }
