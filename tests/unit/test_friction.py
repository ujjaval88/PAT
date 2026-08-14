import pytest

from pat_sim.plant.friction import friction_torque

TC = 0.005
B = 0.001


def test_positive_velocity():
    assert friction_torque(1.0, TC, B) == pytest.approx(0.006)


def test_negative_velocity_symmetric():
    assert friction_torque(-1.0, TC, B) == pytest.approx(-0.006)


def test_zero_velocity_unsmoothed():
    assert friction_torque(0.0, TC, B) == pytest.approx(0.0)


def test_smoothed_sign_continuous_near_zero():
    eps = 0.01
    small_positive = friction_torque(1e-4, TC, B, smoothing_velocity_rad_s=eps)
    small_negative = friction_torque(-1e-4, TC, B, smoothing_velocity_rad_s=eps)
    assert small_positive > 0
    assert small_negative < 0
    assert small_positive == pytest.approx(-small_negative, abs=1e-9)
    assert abs(small_positive) < TC  # smoothed, hasn't reached full Coulomb level yet
