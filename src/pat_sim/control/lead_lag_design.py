"""The delivered Part 2 controller: a two-stage lead compensator with rolloff.

Structure:

    C(s) = Kp * [(1 + sT) / (1 + sT/alpha)]^2 * 1/(1 + s/w_p)

Two lead stages to supply the phase a double integrator plus delay demands, one
real pole to gain-stabilise the structural modes. No integrator and no term
aimed at any particular frequency.
"""

from __future__ import annotations

import math

import numpy as np

from pat_sim.analysis.sensitivity import (
    CameraLoopDesign,
    camera_extra_delay_s,
    design_camera_only_loop,
    find_rejection_bandwidth_rad_s,
    find_sensitivity_peak,
    open_loop_response,
    sensitivity,
)
from pat_sim.config import DEFAULT_CONFIG, SystemConfig

# Less conservative than Part 1's 95th-percentile fundamental-limit
# demonstration: this is a real controller, not a worst-case bound.
LATENCY_PERCENTILE = 0.90
PHASE_MARGIN_DEG = 45.0
LEAD_ALPHA = 8.0
N_STAGES = 2
ROLLOFF_FREQ_HZ = 20.0  # keeps |L| down well before the 75 Hz mode
TORQUE_LIMIT_N_M = 0.5
CONTROLLER_RATE_HZ = 1000.0

# 13 Hz is where the 60 Hz camera folds the 47 Hz tone (|47 - 60|), so it is
# reported alongside the two physical tones.
MARKED_TONES_HZ = (22.0, 47.0, 13.0)


def build_design(config: SystemConfig = DEFAULT_CONFIG) -> CameraLoopDesign:
    """Crossover is placed at the delay-limited point, omega_c = PM / T_total.

    A pure delay T costs omega*T radians of phase, so demanding a 45 degree
    margin caps omega_c at 45 degrees' worth of delay phase no matter how the
    rest of the compensator is shaped. Everything else in the design -- the two
    lead stages and the rolloff pole -- then buys back the phase the plant
    itself costs. This is a Part 1 fundamental limit, not a tuning choice."""
    extra_delay_s = camera_extra_delay_s(config.camera, LATENCY_PERCENTILE)
    total_delay_s = extra_delay_s + config.plant.transport_delay_s
    omega_c_rad_s = math.radians(PHASE_MARGIN_DEG) / total_delay_s
    return design_camera_only_loop(
        config.plant,
        omega_c_rad_s,
        PHASE_MARGIN_DEG,
        extra_delay_s,
        alpha=LEAD_ALPHA,
        n_stages=N_STAGES,
        rolloff_freq_rad_s=2 * math.pi * ROLLOFF_FREQ_HZ,
    )


def design_parameters(design: CameraLoopDesign) -> dict[str, str]:
    """Formatted design point, for the report and the design script."""
    return {
        "structure": f"Kp * [(1+sT)/(1+sT/alpha)]^{N_STAGES} * 1/(1+s/w_p)",
        "latency design point": f"{LATENCY_PERCENTILE * 100:.0f}th percentile of camera latency",
        "modelled extra delay": f"{design.extra_delay_s * 1e3:.2f} ms (camera ZOH + latency)",
        "crossover f_c": f"{design.omega_c_rad_s / (2 * math.pi):.2f} Hz",
        "phase margin target": f"{PHASE_MARGIN_DEG:.1f} deg",
        "lead alpha / stages": f"{LEAD_ALPHA:g} / {N_STAGES}",
        "lead time constant T": f"{design.lead_time_constant_s * 1e3:.3f} ms",
        "gain Kp": f"{design.kp:.4g}",
        "rolloff pole w_p": f"{ROLLOFF_FREQ_HZ:.1f} Hz (1 real pole)",
        "integrator": "none (no pole at the origin, so no wind-up)",
        "torque limit": f"{TORQUE_LIMIT_N_M:g} N*m",
        "update rate": f"{CONTROLLER_RATE_HZ:.0f} Hz, bilinear-discretised as an SOS cascade",
    }


def sensitivity_magnitudes(
    design: CameraLoopDesign, config: SystemConfig, freqs_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(|S|, |L|) on the given frequency grid."""
    loop = open_loop_response(design, config.plant, 2.0 * np.pi * freqs_hz)
    return np.abs(sensitivity(loop)), np.abs(loop)


def sensitivity_metrics(
    design: CameraLoopDesign, config: SystemConfig, freqs_hz: np.ndarray
) -> dict[str, str]:
    """Formatted closed-loop sensitivity summary. The sensitivity script and
    the evaluation report both quote this, so they cannot disagree."""
    omega = 2.0 * np.pi * freqs_hz
    w_peak, ms = find_sensitivity_peak(design, config.plant, omega)
    w_rejection = find_rejection_bandwidth_rad_s(design, config.plant, omega)
    s_magnitude, _ = sensitivity_magnitudes(design, config, freqs_hz)
    metrics = {
        "crossover f_c": f"{design.omega_c_rad_s / (2 * math.pi):.2f} Hz",
        "rejection bandwidth (-3 dB)": (
            f"{w_rejection / (2 * math.pi):.2f} Hz"
            if w_rejection is not None
            else "not found on grid"
        ),
        "peak sensitivity Ms": (
            f"{ms:.2f} ({20 * math.log10(ms):+.1f} dB) at {w_peak / (2 * math.pi):.2f} Hz"
        ),
    }
    for tone_hz in MARKED_TONES_HZ:
        value = float(np.interp(tone_hz, freqs_hz, s_magnitude))
        label = f"|S({tone_hz:.0f} Hz)|" + (" -- 47 Hz alias" if tone_hz == 13.0 else "")
        metrics[label] = f"{value:.3f} ({20 * math.log10(value):+.2f} dB)"
    return metrics
