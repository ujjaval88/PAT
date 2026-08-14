import math

import pytest

from pat_sim.plant.actuator import Actuator

TAU_S = 0.2e-3


def test_step_response_at_one_time_constant():
    actuator = Actuator(tau_s=TAU_S)
    torque = actuator.step(commanded_torque_n_m=1.0, dt_s=TAU_S)
    assert torque == pytest.approx(1.0 - math.exp(-1.0), abs=1e-6)


def test_bandwidth_hz():
    actuator = Actuator(tau_s=TAU_S)
    assert actuator.bandwidth_hz() == pytest.approx(795.8, abs=0.1)


def test_dc_frequency_response_is_unity():
    actuator = Actuator(tau_s=TAU_S)
    resp = actuator.frequency_response(0.0)
    assert resp == pytest.approx(1.0 + 0.0j)


def test_pole_location():
    actuator = Actuator(tau_s=TAU_S)
    assert actuator.pole() == pytest.approx(-5000.0 + 0.0j)
