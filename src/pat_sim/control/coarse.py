"""Discrete-time coarse-pointing controller: a discretized lead-lag base
compensator plus an optional adaptive resonant (internal-model) term for a
drifting tonal disturbance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

from pat_sim.analysis.sensitivity import CameraLoopDesign


@dataclass(frozen=True)
class ResonantTermConfig:
    gain: float
    nominal_freq_hz: float
    search_band_hz: tuple[float, float]
    phase_compensation_deg: float = 0.0
    damping_ratio: float = 0.03
    retune_interval_s: float = 1.0
    buffer_duration_s: float = 2.0
    enabled: bool = True


@dataclass(frozen=True)
class ControllerConfig:
    lead_lag: CameraLoopDesign
    torque_limit_n_m: float
    resonant: tuple[ResonantTermConfig, ...] = ()


def _polynomial_power(coeffs: np.ndarray, n: int) -> np.ndarray:
    result = np.array([1.0])
    for _ in range(n):
        result = np.convolve(result, coeffs)
    return result


def _lead_lag_tf_coeffs(design: CameraLoopDesign) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (num, den) for the whole compensator.

    Kept for frequency-domain checks only. Do NOT discretize this directly for
    the real-time filter -- see _lead_lag_sos."""
    t = design.lead_time_constant_s
    alpha = design.lead_lag_alpha
    num = design.kp * _polynomial_power(np.array([t, 1.0]), design.n_stages)
    den = _polynomial_power(np.array([t / alpha, 1.0]), design.n_stages)
    if design.rolloff_freq_rad_s is not None:
        # extra poles only (no matching zeros) -- num unchanged, degree(den) grows
        rolloff_poly = np.array([1.0 / design.rolloff_freq_rad_s, 1.0])
        den = np.convolve(den, _polynomial_power(rolloff_poly, design.n_rolloff))
    if design.notch_freq_rad_s is not None:
        wn = design.notch_freq_rad_s
        num = np.convolve(num, np.array([1.0, 2.0 * design.notch_zeta_zero * wn, wn**2]))
        den = np.convolve(den, np.array([1.0, 2.0 * design.notch_zeta_pole * wn, wn**2]))
    return num, den


def _lead_lag_sos(design: CameraLoopDesign, dt_s: float) -> np.ndarray:
    """Discretize the compensator as a cascade of second-order sections.

    Discretizing the expanded high-order polynomial in one shot is numerically
    unusable here: with three lead stages plus a rolloff pole the coefficients
    span ~7 orders of magnitude (t^3 ~ 1e-7 against a constant term of 1), and
    scipy's tf2ss-based conversion reports rcond ~1e-17 -- the resulting filter
    bears no relation to the design and diverges in closed loop. Transforming
    each first/second-order factor on its own keeps every conversion
    well-conditioned, which is why SOS is the standard form for anything above
    about second order."""
    sections: list[np.ndarray] = []
    t = design.lead_time_constant_s
    alpha = design.lead_lag_alpha

    for i in range(design.n_stages):
        num = np.array([t, 1.0]) * (design.kp if i == 0 else 1.0)
        den = np.array([t / alpha, 1.0])
        b, a = signal.bilinear(num, den, fs=1.0 / dt_s)
        sections.append(np.array([b[0], b[1], 0.0, a[0], a[1], 0.0]))

    if design.rolloff_freq_rad_s is not None:
        for _ in range(design.n_rolloff):
            b, a = signal.bilinear(
                np.array([1.0]), np.array([1.0 / design.rolloff_freq_rad_s, 1.0]), fs=1.0 / dt_s
            )
            b = np.atleast_1d(b)
            sections.append(np.array([b[0], b[1] if b.size > 1 else 0.0, 0.0, a[0], a[1], 0.0]))

    if design.notch_freq_rad_s is not None:
        wn = design.notch_freq_rad_s
        num = np.array([1.0, 2.0 * design.notch_zeta_zero * wn, wn**2])
        den = np.array([1.0, 2.0 * design.notch_zeta_pole * wn, wn**2])
        b, a = signal.bilinear(num, den, fs=1.0 / dt_s)
        sections.append(np.array([b[0], b[1], b[2], a[0], a[1], a[2]]))

    return np.array(sections)


class _ResonantTerm:
    """Discretized the same way as the lead-lag (bilinear transform + lfilter)
    rather than RK4-integrated as its own ODE. This matters: a lightly-damped
    2nd-order bandpass is sensitive enough that mixing two different
    discretization schemes (RK4 here, bilinear there) inside one closed loop
    introduced enough phase error to destabilize it in testing, even though
    the continuous-time design was stable and phase-aligned. Bilinear
    transform provably preserves the continuous design's stability margins;
    RK4 of a stiff oscillator at a coarse 1 ms step does not carry that
    guarantee."""

    def __init__(self, config: ResonantTermConfig, dt_s: float) -> None:
        self.config = config
        self.dt_s = dt_s
        self.freq_hz = config.nominal_freq_hz
        self._num_d, self._den_d = self._discretize(self.freq_hz)
        self._zi = signal.lfilter_zi(self._num_d, self._den_d) * 0.0
        self._buffer_len: int = max(int(config.buffer_duration_s / dt_s), 1)
        self._buffer: deque[float] = deque(maxlen=self._buffer_len)
        self._time_since_retune_s = 0.0

    def _discretize(self, freq_hz: float) -> tuple[np.ndarray, np.ndarray]:
        w0 = 2.0 * math.pi * freq_hz
        kr = self.config.gain
        zeta_r = self.config.damping_ratio
        phi = math.radians(self.config.phase_compensation_deg)
        # Kr * (s*cos(phi) - w0*sin(phi)) -- gives R(j*w0) = Kr/(2*zeta_r*w0) * exp(j*phi)
        num = np.array([kr * math.cos(phi), -kr * w0 * math.sin(phi)])
        den = np.array([1.0, 2.0 * zeta_r * w0, w0**2])
        num_d, den_d, _ = signal.cont2discrete((num, den), self.dt_s, method="bilinear")
        return np.asarray(num_d).flatten(), np.asarray(den_d).flatten()

    def output_and_candidate(self, error_rad: float) -> tuple[float, np.ndarray]:
        """Returns (output_contribution, candidate_zi) without committing state."""
        out_arr, zi_candidate = signal.lfilter(self._num_d, self._den_d, [error_rad], zi=self._zi)
        return float(out_arr[0]), zi_candidate

    def commit(self, zi: np.ndarray) -> None:
        self._zi = zi

    def observe_and_maybe_retune(self, error_rad: float) -> None:
        self._buffer.append(error_rad)
        self._time_since_retune_s += self.dt_s
        if (
            self._time_since_retune_s < self.config.retune_interval_s
            or len(self._buffer) < self._buffer_len
        ):
            return
        self._time_since_retune_s = 0.0
        buf = np.array(self._buffer)
        windowed = buf * np.hanning(len(buf))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(buf), d=self.dt_s)
        lo, hi = self.config.search_band_hz
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            peak_idx = int(np.argmax(spectrum[mask]))
            new_freq_hz = float(freqs[mask][peak_idx])
            if new_freq_hz != self.freq_hz:
                self.freq_hz = new_freq_hz
                # same filter order regardless of frequency -- zi carries over
                # smoothly rather than resetting to zero on every retune.
                self._num_d, self._den_d = self._discretize(new_freq_hz)


class CoarseController:
    def __init__(self, config: ControllerConfig, dt_s: float) -> None:
        self.config = config
        self.dt_s = dt_s
        self._sos = _lead_lag_sos(config.lead_lag, dt_s)
        self._lead_lag_zi = np.zeros((self._sos.shape[0], 2))
        self._resonant_terms = [
            _ResonantTerm(term_config, dt_s)
            for term_config in config.resonant
            if term_config.enabled
        ]
        self.saturated_last_step = False

    def update(self, error_rad: float) -> float:
        lead_lag_out_arr, self._lead_lag_zi = signal.sosfilt(
            self._sos, [error_rad], zi=self._lead_lag_zi
        )
        u_unsat = float(lead_lag_out_arr[0])

        candidates: list[tuple[_ResonantTerm, np.ndarray]] = []
        for term in self._resonant_terms:
            term_out, zi_candidate = term.output_and_candidate(error_rad)
            u_unsat += term_out
            candidates.append((term, zi_candidate))

        limit = self.config.torque_limit_n_m
        u = max(-limit, min(limit, u_unsat))
        self.saturated_last_step = u != u_unsat

        for term, zi_candidate in candidates:
            # Anti-windup: only commit the tentative state update when the
            # output was not clipped. A lightly-damped resonator driven while
            # its command is being truncated would otherwise keep integrating
            # against an actuator that never delivered the requested torque.
            if not self.saturated_last_step:
                term.commit(zi_candidate)
            term.observe_and_maybe_retune(error_rad)

        return u

    @property
    def resonant_freqs_hz(self) -> tuple[float, ...]:
        return tuple(term.freq_hz for term in self._resonant_terms)
