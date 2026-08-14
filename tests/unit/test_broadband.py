import numpy as np

from pat_sim.analysis.psd import rms, welch_psd
from pat_sim.disturbances.broadband import BroadbandDisturbance

DT_S = 1e-3  # 1 kHz; comfortably above the 80 Hz cutoff for a light unit test


def test_rms_matches_target_within_tolerance():
    rng = np.random.default_rng(42)
    dist = BroadbandDisturbance(cutoff_hz=80.0, target_rms_rad=200e-6, dt_s=DT_S, rng=rng)
    samples = dist.generate(60_000)  # 60 s
    measured = rms(samples[5000:])  # drop filter settling transient
    assert abs(measured - 200e-6) / 200e-6 < 0.15


def test_deterministic_for_fixed_seed():
    a = BroadbandDisturbance(80.0, 200e-6, DT_S, np.random.default_rng(7))
    b = BroadbandDisturbance(80.0, 200e-6, DT_S, np.random.default_rng(7))
    assert np.allclose(a.generate(1000), b.generate(1000))


def test_spectrum_rolls_off_above_cutoff():
    rng = np.random.default_rng(1)
    dist = BroadbandDisturbance(cutoff_hz=80.0, target_rms_rad=200e-6, dt_s=DT_S, rng=rng)
    samples = dist.generate(60_000)[5000:]
    f, pxx = welch_psd(samples, fs_hz=1.0 / DT_S)
    power_below = np.mean(pxx[(f > 10) & (f < 30)])
    power_above = np.mean(pxx[(f > 200) & (f < 400)])
    assert power_above < power_below / 10.0
