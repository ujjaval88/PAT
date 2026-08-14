"""Motor torque saturation, plus a generic rate limiter.
"""

from __future__ import annotations


def saturate_torque(torque_n_m: float, torque_max_n_m: float) -> float:
    return max(-torque_max_n_m, min(torque_max_n_m, torque_n_m))


class SlewRateLimiter:
    def __init__(self, slew_max_n_m_per_s: float, initial_value: float = 0.0) -> None:
        self.slew_max_n_m_per_s = slew_max_n_m_per_s
        self.value = initial_value

    def step(self, target_n_m: float, dt_s: float) -> float:
        max_delta = self.slew_max_n_m_per_s * dt_s
        delta = target_n_m - self.value
        delta = max(-max_delta, min(max_delta, delta))
        self.value += delta
        return self.value
