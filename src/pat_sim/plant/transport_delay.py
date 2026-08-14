"""Discrete transport delay: output at step k is the input from N = delay/dt steps ago."""

from __future__ import annotations

from collections import deque


class TransportDelay:
    def __init__(self, delay_s: float, dt_s: float, initial_value: float = 0.0) -> None:
        self.n_samples = round(delay_s / dt_s)
        self._buffer: deque[float] = deque(
            [initial_value] * self.n_samples, maxlen=self.n_samples or None
        )

    def step(self, value: float) -> float:
        if self.n_samples == 0:
            return value
        out = self._buffer.popleft()
        self._buffer.append(value)
        return out
