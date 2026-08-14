"""RMS and power-spectral-density helpers for disturbance/sensor verification."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def welch_psd(
    x: np.ndarray, fs_hz: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = nperseg or min(len(x), 8192)
    f, pxx = welch(x, fs=fs_hz, nperseg=n)
    return f, pxx


def dominant_frequency_hz(
    x: np.ndarray, fs_hz: float, band_hz: tuple[float, float] | None = None
) -> float:
    f, pxx = welch_psd(x, fs_hz)
    if band_hz is not None:
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        f, pxx = f[mask], pxx[mask]
    return float(f[int(np.argmax(pxx))])
