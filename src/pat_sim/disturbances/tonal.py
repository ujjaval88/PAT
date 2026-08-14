"""Wandering-frequency tonal disturbance.

Phase is carried as integrator state (phase[k+1] = phase[k] + 2*pi*f[k]*dt) rather
than evaluated as sin(2*pi*f(t)*t), which would not preserve phase correctly for a
time-varying frequency. The wander itself is a bounded Ornstein-Uhlenbeck process
so it moves slowly (time constant ~tens of seconds) and stays
within +/- wander_hz of nominal.
"""

from __future__ import annotations

import math

import numpy as np

from pat_sim.config import ToneConfig


class WanderingTone:
    def __init__(
        self,
        config: ToneConfig,
        dt_s: float,
        rng: np.random.Generator,
        wander_enabled: bool = True,
        wander_tau_s: float = 20.0,
    ) -> None:
        self.nominal_freq_hz = config.nominal_freq_hz
        self.amplitude_rad = config.amplitude_rad
        self.wander_hz = config.wander_hz
        self.dt_s = dt_s
        self.rng = rng
        self.wander_enabled = wander_enabled
        self.wander_tau_s = wander_tau_s
        self._sigma = self.wander_hz * math.sqrt(2.0 / wander_tau_s) if self.wander_hz > 0 else 0.0
        self._phase_rad = 0.0
        self._freq_offset_hz = 0.0

    def instantaneous_freq_hz(self) -> float:
        return self.nominal_freq_hz + (self._freq_offset_hz if self.wander_enabled else 0.0)

    def step(self) -> float:
        if self.wander_enabled and self.wander_hz > 0:
            drift = -self._freq_offset_hz / self.wander_tau_s * self.dt_s
            noise = self._sigma * math.sqrt(self.dt_s) * self.rng.standard_normal()
            self._freq_offset_hz = float(
                np.clip(self._freq_offset_hz + drift + noise, -self.wander_hz, self.wander_hz)
            )
        freq_hz = self.instantaneous_freq_hz()
        self._phase_rad += 2.0 * math.pi * freq_hz * self.dt_s
        return self.amplitude_rad * math.sin(self._phase_rad)

    def generate(self, n_samples: int) -> np.ndarray:
        return np.array([self.step() for _ in range(n_samples)])
