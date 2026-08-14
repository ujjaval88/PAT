"""Camera-only loop fundamental-limit design and closed-loop sensitivity.

The compensator is deliberately minimal: a gain times a two-stage lead-lag,
Kp*[(1+s*T)/(1+s*T/alpha)]^2. Its only job is to demonstrate the fundamental
limit imposed by delay -- it is not a candidate Part 2 controller. A pure PD
(lead with no pole) was tried first and rejected: its gain grows without bound
at high frequency, which re-crosses unity gain right at the 75 Hz structural
mode (Q~16.7) and produces a spurious near-instability that has nothing to do
with the camera-only delay limit being characterized here. Giving each lead
stage a matching high-frequency pole (ratio alpha, a fixed documented
assumption) bounds the gain, which is what any real compensator would need
anyway. T is solved so the two-stage phase exactly matches the required lead
at omega_c; Kp is then set so |L(j*omega_c)| = 1 exactly.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from pat_sim.analysis.frequency_response import plant_frequency_response
from pat_sim.config import CameraConfig, PlantConfig
from pat_sim.plant.structural_mode import StructuralMode

# Pole/zero ratio for each lead-lag stage. Assumption: 8 is a conventional
# choice giving a healthy ~51 deg of phase per stage while keeping high-
# frequency gain bounded (finite gain boost of alpha^2 per two stages, rather
# than unbounded growth from a pure differentiator).
LEAD_LAG_ALPHA = 8.0


def camera_zoh_delay_s(camera_config: CameraConfig) -> float:
    return 1.0 / (2.0 * camera_config.rate_hz)


def camera_latency_percentile_s(camera_config: CameraConfig, percentile: float) -> float:
    """Quantile of max(0, Normal(mean, std)). Valid for percentile above the
    clip probability Phi(-mean/std); Part 1 only ever queries the upper tail
    (75th-99th), which is safely above that (~31% for the supplied mean/std)."""
    z = stats.norm.ppf(percentile)
    raw = camera_config.latency_mean_s + camera_config.latency_std_s * z
    return max(0.0, float(raw))


def camera_latency_mean_s(camera_config: CameraConfig) -> float:
    """E[max(0, X)] for X ~ Normal(mean, std): mu*Phi(mu/sigma) + sigma*phi(mu/sigma)."""
    mu, sigma = camera_config.latency_mean_s, camera_config.latency_std_s
    z = mu / sigma
    return mu * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)


def camera_extra_delay_s(camera_config: CameraConfig, latency_percentile: float) -> float:
    return camera_zoh_delay_s(camera_config) + camera_latency_percentile_s(
        camera_config, latency_percentile
    )


@dataclass(frozen=True)
class CameraLoopDesign:
    omega_c_rad_s: float
    kp: float
    lead_time_constant_s: float
    extra_delay_s: float
    phase_margin_target_deg: float
    lead_lag_alpha: float = LEAD_LAG_ALPHA
    n_stages: int = 2
    rolloff_freq_rad_s: float | None = None
    n_rolloff: int = 1
    notch_freq_rad_s: float | None = None
    notch_zeta_zero: float = 0.005
    notch_zeta_pole: float = 0.5


def _unwrapped_plant_phase_deg(
    plant_config: PlantConfig, omega_rad_s: float, extra_delay_s: float
) -> float:
    """Sum each factor's phase analytically instead of taking angle() of the
    product. A double integrator plus multiple delays accumulates phase well
    past +/-180 deg, which np.angle()/cmath.phase() would silently wrap back
    into the principal (-180, 180] range for a single complex sample."""
    phase_deg = -180.0  # rigid body: double integrator, fixed -180 deg by convention
    for mode_cfg in plant_config.modes:
        mode = StructuralMode(mode_cfg.freq_hz, mode_cfg.zeta)
        # Each mode's own phase is bounded within (-180, 0] -- safe to read directly.
        phase_deg += math.degrees(cmath.phase(mode.frequency_response(omega_rad_s)))
    actuator_response = 1.0 / (1j * omega_rad_s * plant_config.actuator_tau_s + 1.0)
    phase_deg += math.degrees(cmath.phase(actuator_response))  # bounded (-90, 0]
    total_delay_s = plant_config.transport_delay_s + extra_delay_s
    phase_deg += -math.degrees(omega_rad_s * total_delay_s)  # unbounded, exact
    return phase_deg


def _lead_lag_stage(
    omega_rad_s: float | np.ndarray, lead_time_constant_s: float, alpha: float
) -> np.ndarray | complex:
    s_t = 1j * omega_rad_s * lead_time_constant_s
    return (1.0 + s_t) / (1.0 + s_t / alpha)


def _rolloff_stage(
    omega_rad_s: float | np.ndarray, rolloff_freq_rad_s: float | None, n_rolloff: int = 1
) -> np.ndarray | complex:
    """n identical real poles at rolloff_freq_rad_s.

    Multiple poles are the tool for GAIN-stabilising a lightly damped mode:
    they buy steep attenuation above crossover without depending on the mode's
    exact frequency, unlike a notch. The cost is -n*atan(w_c/w_p) of phase at
    crossover, which the lead network must make up."""
    if rolloff_freq_rad_s is None:
        return 1.0 + 0.0j
    return 1.0 / (1j * omega_rad_s / rolloff_freq_rad_s + 1.0) ** n_rolloff


def _notch_stage(
    omega_rad_s: float | np.ndarray,
    notch_freq_rad_s: float | None,
    zeta_zero: float,
    zeta_pole: float,
) -> np.ndarray | complex:
    """(s^2 + 2*zeta_zero*wn*s + wn^2) / (s^2 + 2*zeta_pole*wn*s + wn^2), zeta_zero
    << zeta_pole -- deep, narrow notch right at wn, comfortably below wc so it
    only lightly affects phase there, used to keep a structural mode's own
    resonance from ever being significantly excited by this loop."""
    if notch_freq_rad_s is None:
        return 1.0 + 0.0j
    s = 1j * omega_rad_s
    wn = notch_freq_rad_s
    return (s**2 + 2 * zeta_zero * wn * s + wn**2) / (s**2 + 2 * zeta_pole * wn * s + wn**2)


def design_camera_only_loop(
    plant_config: PlantConfig,
    omega_c_rad_s: float,
    phase_margin_deg: float,
    extra_delay_s: float,
    alpha: float = LEAD_LAG_ALPHA,
    n_stages: int = 2,
    rolloff_freq_rad_s: float | None = None,
    n_rolloff: int = 1,
    notch_freq_rad_s: float | None = None,
    notch_zeta_zero: float = 0.005,
    notch_zeta_pole: float = 0.5,
) -> CameraLoopDesign:
    """n_stages controls how much total phase is available (n * max-single-stage)
    without needing to raise alpha. Raising alpha (or n_stages) too far is a
    trap: more phase per stage also means more high-frequency gain boost
    (alpha^n), which risks re-crossing unity gain right at a structural mode
    (see module docstring). rolloff_freq_rad_s adds a single extra real pole,
    placed well above omega_c but comfortably below the first structural mode,
    to explicitly cap high-frequency gain independent of the lead design --
    the safer way to buy headroom near a mode than pushing alpha/n_stages
    further. notch_freq_rad_s adds a deep, narrow notch (see _notch_stage) --
    needed once omega_c is pushed close enough to a mode (as with the gyro's
    much smaller delay budget in Part 3) that rolloff alone can't buy enough
    gain margin there without also killing the achievable crossover. Both
    stages' phase contributions at omega_c are included in the lead solve
    below, so the achieved phase margin at omega_c is still exact."""
    p_wc = plant_frequency_response(np.array([omega_c_rad_s]), plant_config)[0]
    delay_extra = np.exp(-1j * omega_c_rad_s * extra_delay_s)
    rolloff_at_wc = _rolloff_stage(omega_c_rad_s, rolloff_freq_rad_s, n_rolloff)
    notch_at_wc = _notch_stage(omega_c_rad_s, notch_freq_rad_s, notch_zeta_zero, notch_zeta_pole)
    base = p_wc * delay_extra * rolloff_at_wc * notch_at_wc
    base_phase_deg = (
        _unwrapped_plant_phase_deg(plant_config, omega_c_rad_s, extra_delay_s)
        + math.degrees(cmath.phase(rolloff_at_wc))
        + math.degrees(cmath.phase(notch_at_wc))
    )
    target_total_phase_deg = -180.0 + phase_margin_deg
    needed_lead_deg = target_total_phase_deg - base_phase_deg

    max_single_stage_deg = math.degrees(math.asin((alpha - 1.0) / (alpha + 1.0)))
    max_total_stage_deg = n_stages * max_single_stage_deg
    if not 0.0 < needed_lead_deg < max_total_stage_deg:
        raise ValueError(
            f"Requested phase margin {phase_margin_deg} deg at omega_c="
            f"{omega_c_rad_s:.3g} rad/s needs {needed_lead_deg:.1f} deg of lead, "
            f"which {n_stages} lead-lag stage(s) (alpha={alpha}) cannot realize "
            f"(max {max_total_stage_deg:.1f} deg). Lower omega_c, or add more stages."
        )

    target_single_stage_deg = needed_lead_deg / n_stages

    def stage_phase_error_deg(u: float) -> float:
        return math.degrees(math.atan(u) - math.atan(u / alpha)) - target_single_stage_deg

    u_solution = brentq(stage_phase_error_deg, 1e-9, math.sqrt(alpha))
    lead_time_constant_s = u_solution / omega_c_rad_s

    stage_n = _lead_lag_stage(omega_c_rad_s, lead_time_constant_s, alpha) ** n_stages
    kp = 1.0 / (abs(base) * abs(stage_n))

    return CameraLoopDesign(
        omega_c_rad_s=omega_c_rad_s,
        kp=kp,
        lead_time_constant_s=lead_time_constant_s,
        extra_delay_s=extra_delay_s,
        phase_margin_target_deg=phase_margin_deg,
        lead_lag_alpha=alpha,
        n_stages=n_stages,
        rolloff_freq_rad_s=rolloff_freq_rad_s,
        n_rolloff=n_rolloff,
        notch_freq_rad_s=notch_freq_rad_s,
        notch_zeta_zero=notch_zeta_zero,
        notch_zeta_pole=notch_zeta_pole,
    )


def open_loop_response(
    design: CameraLoopDesign, plant_config: PlantConfig, omega_rad_s: np.ndarray
) -> np.ndarray:
    omega = np.asarray(omega_rad_s, dtype=float)
    p = plant_frequency_response(omega, plant_config)
    delay_extra = np.exp(-1j * omega * design.extra_delay_s)
    stage = _lead_lag_stage(omega, design.lead_time_constant_s, design.lead_lag_alpha)
    rolloff = _rolloff_stage(omega, design.rolloff_freq_rad_s, design.n_rolloff)
    notch = _notch_stage(
        omega, design.notch_freq_rad_s, design.notch_zeta_zero, design.notch_zeta_pole
    )
    c = design.kp * stage**design.n_stages * rolloff * notch
    return c * p * delay_extra


def plant_phase_deg_at(
    plant_config: PlantConfig, freq_hz: float, extra_delay_s: float
) -> float:
    """Principal-value phase of P(j*w)*exp(-j*w*T_extra) in degrees.

    Used to set the phase compensation of a resonant (internal-model) term:
    choosing phi = -this makes the resonant branch's contribution to the LOOP
    real and positive at w0, so loop gain runs off to +infinity along the
    positive real axis instead of toward -1. Principal value is the right
    thing here (unlike the crossover solve, which needs unwrapped phase),
    because the compensation only enters as exp(j*phi) and is 360-periodic."""
    p = plant_frequency_response(np.array([2.0 * math.pi * freq_hz]), plant_config)[0]
    p_total = p * np.exp(-1j * 2.0 * math.pi * freq_hz * extra_delay_s)
    return float(math.degrees(cmath.phase(p_total)))


def sensitivity(open_loop_L: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + open_loop_L)


def complementary_sensitivity(open_loop_L: np.ndarray) -> np.ndarray:
    return open_loop_L / (1.0 + open_loop_L)


def achieved_crossover_and_margin(
    design: CameraLoopDesign, plant_config: PlantConfig
) -> tuple[float, float]:
    l_wc = open_loop_response(design, plant_config, np.array([design.omega_c_rad_s]))[0]
    magnitude = abs(l_wc)
    phase_margin_deg = math.degrees(np.angle(l_wc)) + 180.0
    return magnitude, phase_margin_deg


def find_sensitivity_peak(
    design: CameraLoopDesign, plant_config: PlantConfig, omega_grid_rad_s: np.ndarray
) -> tuple[float, float]:
    open_loop = open_loop_response(design, plant_config, omega_grid_rad_s)
    s = sensitivity(open_loop)
    idx = int(np.argmax(np.abs(s)))
    return float(omega_grid_rad_s[idx]), float(np.abs(s[idx]))


def find_rejection_bandwidth_rad_s(
    design: CameraLoopDesign, plant_config: PlantConfig, omega_grid_rad_s: np.ndarray
) -> float | None:
    """First frequency (ascending, below omega_c) where |S| crosses up through
    -3 dB (1/sqrt(2)). Below this, the loop meaningfully attenuates disturbance;
    above it, rejection is no longer guaranteed. None if never reached on the grid."""
    grid = np.sort(omega_grid_rad_s[omega_grid_rad_s <= design.omega_c_rad_s])
    if grid.size == 0:
        return None
    s_mag = np.abs(sensitivity(open_loop_response(design, plant_config, grid)))
    threshold = 1.0 / math.sqrt(2.0)
    above = s_mag >= threshold
    if not above.any():
        return None
    return float(grid[int(np.argmax(above))])
