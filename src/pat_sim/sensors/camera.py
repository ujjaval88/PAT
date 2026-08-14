"""Camera: observes LOS error at 60 Hz with a randomly delayed arrival.

Latency is modeled as max(0, Normal(mean, std)) -- clipped so it can never go
negative, matching the supplied mean/std.
"""

from __future__ import annotations

import numpy as np

from pat_sim.config import CameraConfig
from pat_sim.sensors.base import Measurement


class Camera:
    def __init__(self, config: CameraConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    @property
    def sample_interval_s(self) -> float:
        return 1.0 / self.config.rate_hz

    def draw_latency_s(self) -> float:
        raw = self.rng.normal(self.config.latency_mean_s, self.config.latency_std_s)
        return max(0.0, float(raw))

    def sample(self, true_los_rad: float, capture_time_s: float) -> Measurement:
        noise = self.rng.normal(0.0, self.config.noise_std_rad)
        latency_s = self.draw_latency_s()
        return Measurement(
            value=true_los_rad + noise,
            capture_time_s=capture_time_s,
            arrival_time_s=capture_time_s + latency_s,
            valid=True,
        )
