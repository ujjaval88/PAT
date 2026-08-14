"""QPD: observes fine LOS error, valid only within its linear range.
"""

from __future__ import annotations

import numpy as np

from pat_sim.config import QpdConfig
from pat_sim.sensors.base import Measurement


class QPD:
    def __init__(self, config: QpdConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    @property
    def sample_interval_s(self) -> float:
        return 1.0 / self.config.rate_hz

    def sample(self, true_fine_los_error_rad: float, capture_time_s: float) -> Measurement:
        noise = self.rng.normal(0.0, self.config.noise_std_rad)
        valid = abs(true_fine_los_error_rad) <= self.config.valid_range_rad
        return Measurement(
            value=true_fine_los_error_rad + noise,
            capture_time_s=capture_time_s,
            arrival_time_s=capture_time_s + self.config.latency_s,
            valid=valid,
        )
