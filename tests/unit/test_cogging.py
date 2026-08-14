import math

import numpy as np

from pat_sim.analysis.psd import dominant_frequency_hz
from pat_sim.plant.cogging import cogging_torque

AMPLITUDE = 0.02
NCOG = 12


def test_peak_amplitude():
    theta = np.linspace(0.0, 4 * math.pi, 200_000)
    torque = np.array([cogging_torque(t, AMPLITUDE, NCOG) for t in theta])
    assert abs(np.max(np.abs(torque)) - AMPLITUDE) < 1e-6


def test_twelve_cycles_per_revolution_periodicity():
    period = 2 * math.pi / NCOG
    theta0 = 0.37
    assert cogging_torque(theta0, AMPLITUDE, NCOG) == cogging_torque(
        theta0 + period, AMPLITUDE, NCOG
    )


def test_temporal_frequency_scales_with_speed_not_cycle_count():
    dt_s = 1e-4
    duration_s = 2.0
    n = int(duration_s / dt_s)

    for speed_rad_s in (2.0, 5.0):
        theta = speed_rad_s * np.arange(n) * dt_s
        torque = np.array([cogging_torque(t, AMPLITUDE, NCOG) for t in theta])
        expected_temporal_hz = NCOG * speed_rad_s / (2 * math.pi)
        measured_hz = dominant_frequency_hz(torque, fs_hz=1.0 / dt_s)
        assert abs(measured_hz - expected_temporal_hz) < 0.5
