"""Gimbal encoder: observes shaft angle relative to the base, quantized to its bit resolution.

Callers must pass the shaft angle (theta_g), never LOS (theta_g + theta_b). The
encoder is mounted between the shaft and the base, so base motion cancels out of
its reading and it cannot see platform vibration at all. Nothing in this class
enforces that -- it is the caller's responsibility to pass the right signal.
"""

from __future__ import annotations

from pat_sim.config import EncoderConfig
from pat_sim.sensors.base import Measurement


class Encoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config

    def sample(self, true_shaft_angle_rad: float, capture_time_s: float) -> Measurement:
        q = self.config.quantum_rad
        quantized = round(true_shaft_angle_rad / q) * q
        return Measurement(
            value=quantized,
            capture_time_s=capture_time_s,
            arrival_time_s=capture_time_s + self.config.latency_s,
            valid=True,
        )
