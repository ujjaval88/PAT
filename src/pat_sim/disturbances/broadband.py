"""Broadband platform vibration: white noise shaped by a 2nd-order low-pass,
scaled to a target RMS."""

from __future__ import annotations

import numpy as np
from scipy import signal


class BroadbandDisturbance:
    def __init__(
        self,
        cutoff_hz: float,
        target_rms_rad: float,
        dt_s: float,
        rng: np.random.Generator,
        order: int = 2,
        calibration_duration_s: float = 30.0,
        calibration_seed: int = 0,
    ) -> None:
        self.dt_s = dt_s
        self.b, self.a = signal.butter(order, cutoff_hz, fs=1.0 / dt_s)
        self.rng = rng
        # Gain is calibrated once against a fixed, independent RNG stream so the
        # target RMS is hit regardless of the seed used for the production stream.
        self.gain = self._calibrate_gain(target_rms_rad, calibration_duration_s, calibration_seed)
        self._zi = signal.lfilter_zi(self.b, self.a) * 0.0

    def _calibrate_gain(self, target_rms_rad: float, duration_s: float, seed: int) -> float:
        calib_rng = np.random.default_rng(seed)
        n = max(int(duration_s / self.dt_s), 1000)
        white = calib_rng.standard_normal(n)
        filtered = signal.lfilter(self.b, self.a, white)
        settle = n // 10
        raw_rms = float(np.sqrt(np.mean(filtered[settle:] ** 2)))
        return target_rms_rad / raw_rms

    def generate(self, n_samples: int) -> np.ndarray:
        white = self.rng.standard_normal(n_samples)
        filtered, self._zi = signal.lfilter(self.b, self.a, white, zi=self._zi)
        return self.gain * filtered

    def step(self) -> float:
        return float(self.generate(1)[0])
