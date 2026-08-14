import pytest

from pat_sim.plant.rigid_body import RigidBody

J = 2.5e-3


def test_acceleration_matches_torque_over_inertia():
    body = RigidBody(inertia_kg_m2=J)
    assert body.acceleration_rad_s2(0.01) == pytest.approx(4.0)


@pytest.mark.parametrize("dt_s", [1e-2, 1e-3, 1e-4, 1e-5])
def test_constant_torque_integration_matches_analytical(dt_s):
    body = RigidBody(inertia_kg_m2=J)
    steps = round(1.0 / dt_s)
    for _ in range(steps):
        body.step(torque_n_m=0.01, dt_s=dt_s)
    # theta(1s) = 0.5 * (T/J) * t^2 = 0.5*4*1 = 2 rad. RK4 is exact for the
    # constant-acceleration (quadratic) trajectory, so this should hold tightly
    # for every dt, not just improve in the limit.
    assert body.state.theta_rad == pytest.approx(2.0, abs=1e-6)
    assert body.state.theta_dot_rad_s == pytest.approx(4.0, abs=1e-6)
