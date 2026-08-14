"""Rate gyro: observes inertial gimbal rate plus a slow random-walk bias and white noise."""

from __future__ import annotations

import math

import numpy as np

from pat_sim.config import GyroConfig
from pat_sim.sensors.base import Measurement


class Gyro:
    def __init__(self, config: GyroConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.bias_rad_s = 0.0
        self._dt_s = 1.0 / config.rate_hz

    def sample(self, true_rate_rad_s: float, capture_time_s: float) -> Measurement:
        walk_std = self.config.bias_random_walk_std_rad_s_per_sqrt_s
        self.bias_rad_s += walk_std * math.sqrt(self._dt_s) * self.rng.standard_normal()
        noise = self.rng.normal(0.0, self.config.noise_std_rad_s)
        return Measurement(
            value=true_rate_rad_s + self.bias_rad_s + noise,
            capture_time_s=capture_time_s,
            arrival_time_s=capture_time_s + self.config.latency_s,
            valid=True,
        )
