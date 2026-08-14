import math

import numpy as np
import pytest

from pat_sim.analysis.sensitivity import camera_extra_delay_s, design_camera_only_loop
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.coarse import CoarseController, ControllerConfig

CFG = DEFAULT_CONFIG


def _default_design():
    extra_delay = camera_extra_delay_s(CFG.camera, 0.90)
    total_delay = extra_delay + CFG.plant.transport_delay_s
    omega_c = math.radians(45.0) / total_delay
    return design_camera_only_loop(
        CFG.plant, omega_c, 45.0, extra_delay, alpha=8.0, n_stages=2,
        rolloff_freq_rad_s=2 * math.pi * 20.0,
    )


def test_output_saturates_at_configured_limit():
    design = _default_design()
    cc = ControllerConfig(lead_lag=design, torque_limit_n_m=0.1, resonant=())
    controller = CoarseController(cc, dt_s=1e-3)
    # a huge error should saturate the output, not exceed the configured limit
    u = controller.update(error_rad=1000.0)
    assert abs(u) <= 0.1 + 1e-12
    assert controller.saturated_last_step is True


def test_zero_error_gives_zero_steady_state_output():
    design = _default_design()
    cc = ControllerConfig(lead_lag=design, torque_limit_n_m=0.5, resonant=())
    controller = CoarseController(cc, dt_s=1e-3)
    u = 0.0
    for _ in range(500):
        u = controller.update(0.0)
    assert abs(u) < 1e-9
    assert controller.saturated_last_step is False


def test_small_error_stays_within_limit_and_unsaturated():
    design = _default_design()
    cc = ControllerConfig(lead_lag=design, torque_limit_n_m=0.5, resonant=())
    controller = CoarseController(cc, dt_s=1e-3)
    dt = 1e-3
    for k in range(2000):
        t = k * dt
        e = -50e-6 * math.sin(2 * math.pi * 5.0 * t)
        u = controller.update(e)
        assert abs(u) <= 0.5 + 1e-12
    assert controller.saturated_last_step is False


def test_resonant_frequency_tracker_locks_onto_pure_tone():
    from pat_sim.control.coarse import ResonantTermConfig

    design = _default_design()
    cc = ControllerConfig(
        lead_lag=design,
        torque_limit_n_m=0.5,
        resonant=(
            ResonantTermConfig(
                gain=1.0, nominal_freq_hz=20.0, search_band_hz=(18.0, 26.0),
                retune_interval_s=1.0, buffer_duration_s=2.0,
            ),
        ),
    )
    controller = CoarseController(cc, dt_s=1e-3)
    dt = 1e-3
    for k in range(2000):
        t = k * dt
        e = 150e-6 * math.sin(2 * math.pi * 22.0 * t)
        controller.update(e)
    assert controller.resonant_freqs_hz == (22.0,)


def test_two_resonant_terms_track_two_tones_independently():
    from pat_sim.control.coarse import ResonantTermConfig

    design = _default_design()
    cc = ControllerConfig(
        lead_lag=design,
        torque_limit_n_m=0.5,
        resonant=(
            ResonantTermConfig(gain=1.0, nominal_freq_hz=20.0, search_band_hz=(18.0, 26.0)),
            ResonantTermConfig(gain=1.0, nominal_freq_hz=45.0, search_band_hz=(43.0, 51.0)),
        ),
    )
    controller = CoarseController(cc, dt_s=1e-3)
    dt = 1e-3
    for k in range(3000):
        t = k * dt
        e = 150e-6 * math.sin(2 * math.pi * 22.0 * t) + 100e-6 * math.sin(2 * math.pi * 47.0 * t)
        controller.update(e)
    assert controller.resonant_freqs_hz == (22.0, 47.0)


def test_phase_compensation_sets_resonant_phase_at_center_frequency():
    """R(j*w0) must have magnitude Kr/(2*zeta*w0) and phase exactly phi --
    this is what makes the loop run to +infinity along the safe direction."""
    from scipy import signal

    from pat_sim.control.coarse import ResonantTermConfig, _ResonantTerm

    dt_s = 1e-4
    for phi_deg in (-120.0, -45.0, 0.0, 60.0):
        cfg = ResonantTermConfig(
            gain=1.0, nominal_freq_hz=22.0, search_band_hz=(18.0, 26.0),
            phase_compensation_deg=phi_deg, damping_ratio=0.05,
        )
        term = _ResonantTerm(cfg, dt_s)
        w0 = 2 * math.pi * 22.0
        _, h = signal.freqz(term._num_d, term._den_d, worN=np.array([w0 * dt_s]))
        assert math.degrees(np.angle(h[0])) == pytest.approx(phi_deg, abs=1.0)
        assert abs(h[0]) == pytest.approx(1.0 / (2 * 0.05 * w0), rel=0.02)


def test_bilinear_discretized_response_matches_continuous_design_near_crossover():
    from scipy import signal

    from pat_sim.control.coarse import _lead_lag_tf_coeffs

    design = _default_design()
    num, den = _lead_lag_tf_coeffs(design)
    dt_s = 1e-3
    w_test = np.array([design.omega_c_rad_s])
    _, h_cont = signal.freqs(num, den, worN=w_test)
    num_d, den_d, _ = signal.cont2discrete((num, den), dt_s, method="bilinear")
    _, h_disc = signal.freqz(num_d.flatten(), den_d.flatten(), worN=w_test * dt_s)
    assert abs(h_disc[0]) == pytest.approx(abs(h_cont[0]), rel=1e-3)
