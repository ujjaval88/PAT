"""Fundamental performance limits that no controller design can beat.
"""

from __future__ import annotations

import math

import numpy as np


def delay_limited_sensitivity(freq_hz: np.ndarray | float, delay_s: float) -> np.ndarray:
    """|S| = 2*|sin(pi*f*T)| -- the ideal-Smith-predictor sensitivity bound."""
    f = np.asarray(freq_hz, dtype=float)
    return 2.0 * np.abs(np.sin(math.pi * f * delay_s))


def delay_rejection_bandwidth_hz(delay_s: float) -> float:
    """Frequency where the delay bound crosses |S| = 1: below this the loop can
    help, above it the best possible loop amplifies. Equals 1/(6*T)."""
    return 1.0 / (6.0 * delay_s)


def broadband_psd(freq_hz: np.ndarray, cutoff_hz: float, target_rms_rad: float) -> np.ndarray:
    """PSD of white noise through a 2nd-order low-pass, normalized to the given
    total RMS. Flat below the cutoff, -40 dB/decade above."""
    f = np.asarray(freq_hz, dtype=float)
    shape = 1.0 / (1.0 + (f / cutoff_hz) ** 4)
    enbw_hz = cutoff_hz * math.pi / (2.0 * math.sqrt(2.0))
    p0 = target_rms_rad**2 / enbw_hz
    return p0 * shape


def variance_ratio(
    freq_hz: np.ndarray, psd: np.ndarray, sensitivity_mag: np.ndarray
) -> float:
    """Closed-loop output variance divided by open-loop variance, given |S(f)|."""
    closed = np.trapezoid(psd * sensitivity_mag**2, freq_hz)
    open_loop = np.trapezoid(psd, freq_hz)
    return float(closed / open_loop)


def delay_limited_variance_ratio(
    delay_s: float,
    cutoff_hz: float = 80.0,
    target_rms_rad: float = 200e-6,
    f_max_hz: float = 2000.0,
    n_points: int = 200_000,
) -> float:
    """Best achievable broadband variance ratio for a loop with delay T."""
    freq_hz = np.linspace(1e-3, f_max_hz, n_points)
    psd = broadband_psd(freq_hz, cutoff_hz, target_rms_rad)
    s_mag = delay_limited_sensitivity(freq_hz, delay_s)
    first_peak_hz = 1.0 / (2.0 * delay_s)
    s_mag = np.where(freq_hz > first_peak_hz, 1.0, s_mag)
    return variance_ratio(freq_hz, psd, s_mag)


def rate_limited_max_amplitude_rad(
    freq_hz: np.ndarray | float, rate_limit_rad_s: float
) -> np.ndarray:
    """Largest LOS amplitude the gimbal can cancel before the axis angular-rate
    limit binds: A_max = rate_limit / w. Falls only as 1/w, so it is generous
    across this disturbance's whole band."""
    f = np.asarray(freq_hz, dtype=float)
    return rate_limit_rad_s / (2.0 * math.pi * f)


def torque_limited_max_amplitude_rad(
    freq_hz: np.ndarray | float, inertia_kg_m2: float, torque_limit_n_m: float
) -> np.ndarray:
    """Largest LOS amplitude the gimbal can cancel before torque saturation
    binds: A_max = T_limit / (J * w^2)."""
    f = np.asarray(freq_hz, dtype=float)
    omega = 2.0 * math.pi * f
    return torque_limit_n_m / (inertia_kg_m2 * omega**2)


def required_rate_rad_s(freq_hz: float, amplitude_rad: float) -> float:
    """Peak axis rate needed to cancel a sinusoid of the given amplitude: A*w."""
    return amplitude_rad * 2.0 * math.pi * freq_hz


def required_torque_n_m(freq_hz: float, amplitude_rad: float, inertia_kg_m2: float) -> float:
    """Peak torque needed to cancel a sinusoid of the given amplitude: J*A*w^2."""
    omega = 2.0 * math.pi * freq_hz
    return inertia_kg_m2 * amplitude_rad * omega**2


def torque_limited_bandwidth_hz(
    inertia_kg_m2: float,
    torque_limit_n_m: float,
    cutoff_hz: float = 80.0,
    target_rms_rad: float = 200e-6,
    headroom_factor: float = 3.0,
) -> float:
    """Highest f_max for which cancelling all broadband content below f_max
    keeps RMS torque within torque_limit/headroom_factor."""
    enbw_hz = cutoff_hz * math.pi / (2.0 * math.sqrt(2.0))
    p0 = target_rms_rad**2 / enbw_hz
    budget = torque_limit_n_m / headroom_factor
    coeff = inertia_kg_m2**2 * p0 * (2.0 * math.pi) ** 4 / 5.0
    return float((budget**2 / coeff) ** (1.0 / 5.0))
