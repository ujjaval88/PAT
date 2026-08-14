import pytest

from pat_sim.plant.limits import SlewRateLimiter, saturate_torque


@pytest.mark.parametrize(
    ("commanded", "expected"),
    [(-1.0, -0.5), (-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 0.5)],
)
def test_saturation_symmetric(commanded, expected):
    assert saturate_torque(commanded, torque_max_n_m=0.5) == pytest.approx(expected)


def test_slew_rate_limit_caps_delta_per_step():
    dt_s = 50e-6
    slew_max = 3.0
    max_delta = slew_max * dt_s
    limiter = SlewRateLimiter(slew_max_n_m_per_s=slew_max)
    prev = limiter.value
    for _ in range(200):
        current = limiter.step(target_n_m=1.0, dt_s=dt_s)
        assert abs(current - prev) <= max_delta + 1e-12
        prev = current


def test_slew_rate_limit_symmetric_falling():
    dt_s = 50e-6
    slew_max = 3.0
    max_delta = slew_max * dt_s
    limiter = SlewRateLimiter(slew_max_n_m_per_s=slew_max, initial_value=1.0)
    prev = limiter.value
    for _ in range(200):
        current = limiter.step(target_n_m=-1.0, dt_s=dt_s)
        assert abs(current - prev) <= max_delta + 1e-12
        prev = current
