"""Simulation clock. Time is recomputed from the step index rather than
accumulated by repeated addition, to avoid float drift over long runs."""

from __future__ import annotations


class SimClock:
    def __init__(self, dt_s: float) -> None:
        self.dt_s = dt_s
        self.step_index = 0

    @property
    def time_s(self) -> float:
        return self.step_index * self.dt_s

    def advance(self) -> float:
        self.step_index += 1
        return self.time_s
