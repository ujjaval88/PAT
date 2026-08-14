"""Timestamp-based multi-rate scheduling primitives.

schedule events using timestamps rather than
scattered modulus checks, since not every rate divides evenly into the
physics timestep (this matters most for the 60 Hz camera against a 50 us
physics step).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class PeriodicSchedule:
    def __init__(self, rate_hz: float, start_time_s: float = 0.0) -> None:
        self.period_s = 1.0 / rate_hz
        self.next_due_s = start_time_s

    def is_due(self, time_s: float) -> bool:
        return time_s + 1e-12 >= self.next_due_s

    def advance(self) -> None:
        self.next_due_s += self.period_s


@dataclass(order=True)
class _QueuedItem(Generic[T]):
    arrival_time_s: float
    seq: int
    payload: T = field(compare=False)


class ArrivalQueue(Generic[T]):
    """Delivers payloads in arrival-time order, even if they were pushed out
    of order (a higher-latency capture can be overtaken by a later, lower-
    latency one -- this must be handled correctly, not just FIFO)."""

    def __init__(self) -> None:
        self._heap: list[_QueuedItem[T]] = []
        self._seq = 0

    def push(self, arrival_time_s: float, payload: T) -> None:
        heapq.heappush(self._heap, _QueuedItem(arrival_time_s, self._seq, payload))
        self._seq += 1

    def pop_ready(self, current_time_s: float) -> list[T]:
        ready: list[T] = []
        while self._heap and self._heap[0].arrival_time_s <= current_time_s:
            ready.append(heapq.heappop(self._heap).payload)
        return ready
