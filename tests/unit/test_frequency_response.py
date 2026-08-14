import math

import numpy as np
import pytest

from pat_sim.analysis.frequency_response import magnitude_db, plant_frequency_response, plant_poles
from pat_sim.config import DEFAULT_CONFIG

PLANT = DEFAULT_CONFIG.plant


def test_low_frequency_rigid_body_slope_is_minus_40db_per_decade():
    omega = np.array([0.1, 1.0])  # rad/s, well below modes and actuator break
    resp = plant_frequency_response(omega, PLANT)
    mag_db = magnitude_db(resp)
    slope = mag_db[1] - mag_db[0]  # one decade
    assert slope == pytest.approx(-40.0, abs=0.5)


def test_structural_resonances_near_75_and_220_hz():
    freqs_hz = np.linspace(1.0, 400.0, 40_000)
    resp = plant_frequency_response(2 * np.pi * freqs_hz, PLANT)
    mag = np.abs(resp)

    near_75 = (freqs_hz > 60) & (freqs_hz < 90)
    near_220 = (freqs_hz > 190) & (freqs_hz < 250)
    peak_75 = freqs_hz[near_75][np.argmax(mag[near_75])]
    peak_220 = freqs_hz[near_220][np.argmax(mag[near_220])]

    assert peak_75 == pytest.approx(75.0, abs=1.0)
    # In the combined response mode 2's peak sits a few Hz below its own
    # resonance, because it rides on the rigid body's falling slope and mode
    # 1's rolloff. That is expected, hence the wider tolerance here.
    assert peak_220 == pytest.approx(220.0, abs=3.0)


def test_delay_contributes_phase_not_magnitude():
    omega = np.array([2 * math.pi * 10.0])
    with_delay = plant_frequency_response(omega, PLANT, include_delay=True)
    without_delay = plant_frequency_response(omega, PLANT, include_delay=False)

    assert abs(with_delay[0]) == pytest.approx(abs(without_delay[0]), rel=1e-9)

    expected_extra_phase_rad = -omega[0] * PLANT.transport_delay_s
    actual_extra_phase_rad = np.angle(with_delay[0]) - np.angle(without_delay[0])
    # both angles are principal-valued and small here (10 Hz, well inside +/-180 budget)
    assert actual_extra_phase_rad == pytest.approx(expected_extra_phase_rad, abs=1e-6)


def test_actuator_factor_matches_manual_product():
    omega = np.array([2 * math.pi * 500.0])
    resp = plant_frequency_response(omega, PLANT, include_delay=False)

    s = 1j * omega[0]
    manual = 1.0 / (PLANT.inertia_kg_m2 * s**2)
    for mode_cfg in PLANT.modes:
        wn = 2 * math.pi * mode_cfg.freq_hz
        manual *= wn**2 / (s**2 + 2 * mode_cfg.zeta * wn * s + wn**2)
    manual *= 1.0 / (1j * omega[0] * PLANT.actuator_tau_s + 1.0)

    assert resp[0] == pytest.approx(manual, rel=1e-9)


def test_pole_map_reference_values():
    poles = plant_poles(PLANT)
    assert poles[0] == 0j
    assert poles[1] == 0j
    assert poles[2] == pytest.approx(-5000.0 + 0j)
    assert poles[3] == pytest.approx(-14.14 + 471.0j, abs=0.5)
    assert poles[4] == pytest.approx(-14.14 - 471.0j, abs=0.5)
    assert poles[5] == pytest.approx(-55.29 + 1381.0j, abs=0.5)
    assert poles[6] == pytest.approx(-55.29 - 1381.0j, abs=0.5)
