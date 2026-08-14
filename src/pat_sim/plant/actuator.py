"""BLDC current-loop actuator lag: tau * Tm_dot + Tm = Tcmd."""

from __future__ import annotations

import math


class Actuator:
    def __init__(self, tau_s: float, initial_torque_n_m: float = 0.0) -> None:
        self.tau_s = tau_s
        self.torque_n_m = initial_torque_n_m

    def step(self, commanded_torque_n_m: float, dt_s: float) -> float:
        """Exact exponential update for constant Tcmd held over dt_s."""
        decay = math.exp(-dt_s / self.tau_s)
        self.torque_n_m = commanded_torque_n_m + (self.torque_n_m - commanded_torque_n_m) * decay
        return self.torque_n_m

    def frequency_response(self, omega_rad_s: float) -> complex:
        return 1.0 / (1j * omega_rad_s * self.tau_s + 1.0)

    def bandwidth_hz(self) -> float:
        return 1.0 / (2.0 * math.pi * self.tau_s)

    def pole(self) -> complex:
        return complex(-1.0 / self.tau_s, 0.0)
