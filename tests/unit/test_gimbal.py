import math

import pytest

from pat_sim.config import DEFAULT_CONFIG, CoggingConfig, FrictionConfig
from pat_sim.plant.gimbal import Gimbal

CFG = DEFAULT_CONFIG


def test_open_loop_step_matches_rigid_body_after_mode_transients_settle():
    dt = 50e-6
    gimbal = Gimbal(
        CFG.plant, CFG.motor_limits, FrictionConfig(0.0, 0.0), CoggingConfig(0.0, 12, 0.0), dt
    )
    torque = 0.01
    duration = 0.5  # several mode-1 decay time constants (~71 ms)
    n = int(duration / dt)
    out = None
    for _ in range(n):
        out = gimbal.step(torque)
    theta_analytical = 0.5 * (torque / CFG.plant.inertia_kg_m2) * duration**2
    assert out.theta_g_rad == pytest.approx(theta_analytical, rel=0.02)


def test_torque_saturation_flag_set_when_commanding_beyond_limit():
    dt = 50e-6
    gimbal = Gimbal(CFG.plant, CFG.motor_limits, CFG.friction, CFG.cogging, dt)
    out = None
    for _ in range(50):
        out = gimbal.step(10.0)  # way beyond 0.5 N*m limit
    assert out.saturated is True
    assert abs(out.motor_torque_n_m) <= CFG.motor_limits.torque_max_n_m + 1e-9


def test_axis_rate_limit_engages_symmetrically_and_clamps_at_3_rad_s():
    """The +/-3 rad/s spec is the axis ANGULAR RATE. Sustained full torque must
    drive the rate up to the limit and no further, in both directions."""
    dt = 50e-6
    for sign in (+1.0, -1.0):
        gimbal = Gimbal(
            CFG.plant, CFG.motor_limits, FrictionConfig(0.0, 0.0), CoggingConfig(0.0, 12, 0.0), dt
        )
        out = None
        for _ in range(int(0.5 / dt)):
            out = gimbal.step(sign * 10.0)  # far beyond torque saturation
        rate = gimbal.rigid_body.state.theta_dot_rad_s
        assert abs(rate) == pytest.approx(CFG.motor_limits.rate_max_rad_s, rel=1e-6)
        assert math.copysign(1.0, rate) == sign
        assert out.rate_limited is True


def test_rate_limit_does_not_engage_at_modest_torque():
    dt = 50e-6
    gimbal = Gimbal(
        CFG.plant, CFG.motor_limits, FrictionConfig(0.0, 0.0), CoggingConfig(0.0, 12, 0.0), dt
    )
    out = None
    for _ in range(int(0.1 / dt)):
        out = gimbal.step(0.001)
    assert out.rate_limited is False
    assert abs(gimbal.rigid_body.state.theta_dot_rad_s) < CFG.motor_limits.rate_max_rad_s


def test_zero_torque_from_rest_stays_near_zero():
    dt = 50e-6
    gimbal = Gimbal(CFG.plant, CFG.motor_limits, CFG.friction, CFG.cogging, dt)
    out = None
    for _ in range(1000):
        out = gimbal.step(0.0)
    assert abs(out.theta_g_rad) < 1e-9
