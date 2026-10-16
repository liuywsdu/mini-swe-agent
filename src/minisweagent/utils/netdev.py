"""Network rx/tx byte counters from /proc/net/dev-style files (plan section 2).

Parsers are pure functions separated from file IO for testing. Counters are
cumulative; use ``diff_net_dev`` at session boundaries. Loopback is excluded:
the metric targets external traffic. Unavailable observation points read as
None.
"""

from __future__ import annotations

from pathlib import Path

PROC_NET_DEV = Path("/proc/net/dev")


def parse_net_dev(text: str) -> dict[str, int]:
    """Sum rx/tx bytes over non-loopback interfaces."""
    rx = tx = 0
    for line in text.splitlines():
        iface, sep, rest = line.partition(":")
        if not sep or iface.strip() == "lo":
            continue
        fields = rest.split()
        if len(fields) < 16:
            continue
        rx += int(fields[0])
        tx += int(fields[8])
    return {"rx_bytes": rx, "tx_bytes": tx}


def read_net_dev(path: Path) -> dict[str, int] | None:
    return parse_net_dev(path.read_text()) if path.is_file() else None


def diff_net_dev(before: dict | None, after: dict | None) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {"rx_bytes": after["rx_bytes"] - before["rx_bytes"], "tx_bytes": after["tx_bytes"] - before["tx_bytes"]}
