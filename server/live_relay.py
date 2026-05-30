#
# In-memory live transcript relay (Railway web app ↔ Pipecat Cloud bot).
#

from __future__ import annotations

import asyncio
import os
from typing import Any

_LIVE_CHANNELS: dict[str, set[asyncio.Queue]] = {}
_LIVE_HISTORY: dict[str, list[dict[str, Any]]] = {}
_LIVE_BACKLOG = int(os.getenv("LIVE_RELAY_BACKLOG", "300"))


def stress_channel(session_id: str) -> str:
    """Per-exam channel so a new stress test never replays an old call."""
    return f"stress-{session_id}"


def history_since_reset(channel: str) -> list[dict[str, Any]]:
    hist = _LIVE_HISTORY.get(channel, [])
    start = 0
    for i, ev in enumerate(hist):
        if ev.get("type") == "reset":
            start = i
    return hist[start:]


def prime_channel(channel: str) -> None:
    _LIVE_HISTORY[channel] = [{"type": "reset"}]


def push_event(channel: str, event: dict[str, Any]) -> int:
    if event.get("type") == "reset":
        _LIVE_HISTORY[channel] = [event]
    else:
        hist = _LIVE_HISTORY.setdefault(channel, [])
        hist.append(event)
        if len(hist) > _LIVE_BACKLOG:
            del hist[: len(hist) - _LIVE_BACKLOG]
    subscribers = 0
    for q in list(_LIVE_CHANNELS.get(channel, ())):
        try:
            q.put_nowait(event)
            subscribers += 1
        except asyncio.QueueFull:  # pragma: no cover
            pass
    return subscribers


def subscribe(channel: str) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
    q: asyncio.Queue = asyncio.Queue()
    _LIVE_CHANNELS.setdefault(channel, set()).add(q)
    return q, history_since_reset(channel)


def unsubscribe(channel: str, q: asyncio.Queue) -> None:
    _LIVE_CHANNELS.get(channel, set()).discard(q)
