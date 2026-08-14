import math

import numpy as np
import pytest

from pat_sim.analysis.fundamental_limits import (
    broadband_psd,
    delay_limited_sensitivity,
    delay_limited_variance_ratio,
    delay_rejection_bandwidth_hz,
    rate_limited_max_amplitude_rad,
    required_rate_rad_s,
    required_torque_n_m,
    torque_limited_bandwidth_hz,
    variance_ratio,
)
from pat_sim.config import DEFAULT_CONFIG

CFG = DEFAULT_CONFIG
J = CFG.plant.inertia_kg_m2
RATE_LIMIT = CFG.motor_limits.rate_max_rad_s
TORQUE_LIMIT = CFG.motor_limits.torque_max_n_m


def test_delay_bound_is_zero_at_dc_and_two_at_first_peak():
    delay_s = 3.2e-3
    assert delay_limited_sensitivity(0.0, delay_s) == pytest.approx(0.0, abs=1e-12)
    assert delay_limited_sensitivity(1.0 / (2 * delay_s), delay_s) == pytest.approx(2.0)


def test_delay_bound_crosses_unity_at_one_sixth_over_t():
    delay_s = 3.2e-3
    f_unity = delay_rejection_bandwidth_hz(delay_s)
    assert f_unity == pytest.approx(1.0 / (6 * delay_s))
    assert delay_limited_sensitivity(f_unity, delay_s) == pytest.approx(1.0)


def test_delay_bound_amplifies_above_unity_crossing():
    delay_s = 3.2e-3
    f_unity = delay_rejection_bandwidth_hz(delay_s)
    assert delay_limited_sensitivity(f_unity * 1.2, delay_s) > 1.0
    assert delay_limited_sensitivity(f_unity * 0.5, delay_s) < 1.0


def test_broadband_psd_integrates_to_target_rms():
    freq_hz = np.linspace(1e-3, 5000.0, 500_000)
    psd = broadband_psd(freq_hz, cutoff_hz=80.0, target_rms_rad=200e-6)
    rms = math.sqrt(float(np.trapezoid(psd, freq_hz)))
    assert rms == pytest.approx(200e-6, rel=0.01)


def test_current_gyro_path_delay_cannot_improve_broadband():
    """The headline Part 3 result: with ~3.2 ms of loop delay, even an ideal
    Smith predictor leaves broadband variance slightly WORSE than open loop,
    because the delay bound's amplification band overlaps real disturbance
    energy. No controller tuning escapes this."""
    ratio = delay_limited_variance_ratio(3.2e-3)
    assert ratio > 1.0


def test_shorter_delay_monotonically_improves_broadband_bound():
    ratios = [delay_limited_variance_ratio(t) for t in (3.2e-3, 2.45e-3, 1.5e-3, 1.0e-3)]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 0.25  # 1 ms delay would allow a large improvement


def test_variance_ratio_of_unity_sensitivity_is_one():
    freq_hz = np.linspace(1e-3, 500.0, 20_000)
    psd = broadband_psd(freq_hz, 80.0, 200e-6)
    assert variance_ratio(freq_hz, psd, np.ones_like(freq_hz)) == pytest.approx(1.0)


def test_rate_limited_amplitude_falls_as_inverse_frequency():
    a_20 = rate_limited_max_amplitude_rad(20.0, RATE_LIMIT)
    a_40 = rate_limited_max_amplitude_rad(40.0, RATE_LIMIT)
    assert a_20 / a_40 == pytest.approx(2.0, rel=1e-9)


def test_both_tones_are_comfortably_within_actuator_authority():
    """Regression test for a misread spec. The "+/-3 rad/s" actuator limit is
    the AXIS ANGULAR RATE, not a 3 N*m/s torque slew rate. Under the correct
    reading both tones are cancellable with two orders of magnitude of margin;
    under the old reading the 47 Hz tone was wrongly declared impossible."""
    for freq_hz, amplitude_rad in ((22.0, 150e-6), (47.0, 100e-6)):
        assert required_rate_rad_s(freq_hz, amplitude_rad) < 0.05 * RATE_LIMIT
        assert required_torque_n_m(freq_hz, amplitude_rad, J) < 0.1 * TORQUE_LIMIT
        assert rate_limited_max_amplitude_rad(freq_hz, RATE_LIMIT) > 50 * amplitude_rad


def test_torque_authority_allows_cancelling_the_full_disturbance_band():
    """Cancelling broadband content out to 80 Hz needs ~0.16 N*m at 3 sigma
    against a 0.5 N*m limit, so actuator authority does not set the achievable
    bandwidth -- loop delay does."""
    assert torque_limited_bandwidth_hz(J, TORQUE_LIMIT) > 80.0


def test_rate_limit_is_far_from_binding_for_this_disturbance():
    """Peak base-motion rate is ~0.42 rad/s against the 3 rad/s limit."""
    assert rate_limited_max_amplitude_rad(80.0, RATE_LIMIT) > 5e-3
