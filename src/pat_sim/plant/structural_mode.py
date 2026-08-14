"""A single lightly-damped structural mode: wn^2 / (s^2 + 2*zeta*wn*s + wn^2).

Time domain: y'' + 2*zeta*wn*y' + wn^2*y = wn^2*u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StructuralModeState:
    y_rad: float = 0.0
    y_dot_rad_s: float = 0.0


class StructuralMode:
    def __init__(
        self,
        freq_hz: float,
        zeta: float,
        enabled: bool = True,
        state: StructuralModeState | None = None,
    ) -> None:
        self.freq_hz = freq_hz
        self.zeta = zeta
        self.enabled = enabled
        self.state = state if state is not None else StructuralModeState()

    @property
    def omega_n_rad_s(self) -> float:
        return 2.0 * math.pi * self.freq_hz

    def frequency_response(self, omega_rad_s: float) -> complex:
        if not self.enabled:
            return complex(1.0, 0.0)
        wn = self.omega_n_rad_s
        s = 1j * omega_rad_s
        return wn**2 / (s**2 + 2 * self.zeta * wn * s + wn**2)

    def poles(self) -> tuple[complex, complex]:
        wn = self.omega_n_rad_s
        damped = wn * math.sqrt(1.0 - self.zeta**2)
        real = -self.zeta * wn
        return complex(real, damped), complex(real, -damped)

    def step(self, input_rad: float, dt_s: float) -> float:
        """RK4 step assuming input_rad is held constant (ZOH) over dt_s."""
        if not self.enabled:
            self.state = StructuralModeState(y_rad=input_rad, y_dot_rad_s=0.0)
            return input_rad

        wn = self.omega_n_rad_s
        zeta = self.zeta

        def deriv(s: np.ndarray) -> np.ndarray:
            y, y_dot = s
            y_ddot = wn**2 * (input_rad - y) - 2 * zeta * wn * y_dot
            return np.array([y_dot, y_ddot])

        s0 = np.array([self.state.y_rad, self.state.y_dot_rad_s])
        k1 = deriv(s0)
        k2 = deriv(s0 + dt_s / 2 * k1)
        k3 = deriv(s0 + dt_s / 2 * k2)
        k4 = deriv(s0 + dt_s * k3)
        s1 = s0 + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state = StructuralModeState(y_rad=s1[0], y_dot_rad_s=s1[1])
        return self.state.y_rad
