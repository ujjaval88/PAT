import numpy as np

from pat_sim.analysis.psd import dominant_frequency_hz
from pat_sim.config import ToneConfig
from pat_sim.disturbances.tonal import WanderingTone

DT_S = 1e-3


def test_wander_disabled_amplitude_and_frequency():
    cfg = ToneConfig(nominal_freq_hz=22.0, amplitude_rad=150e-6, wander_hz=2.0)
    rng = np.random.default_rng(0)
    tone = WanderingTone(cfg, DT_S, rng, wander_enabled=False)
    samples = tone.generate(4000)  # 4 s

    assert np.max(np.abs(samples)) <= 150e-6 + 1e-9
    assert np.max(np.abs(samples)) > 0.99 * 150e-6
    measured_hz = dominant_frequency_hz(samples, fs_hz=1.0 / DT_S)
    assert abs(measured_hz - 22.0) < 0.5


def test_wander_disabled_frequency_47hz():
    cfg = ToneConfig(nominal_freq_hz=47.0, amplitude_rad=100e-6, wander_hz=3.0)
    rng = np.random.default_rng(1)
    tone = WanderingTone(cfg, DT_S, rng, wander_enabled=False)
    samples = tone.generate(4000)
    measured_hz = dominant_frequency_hz(samples, fs_hz=1.0 / DT_S)
    assert abs(measured_hz - 47.0) < 0.5


def test_wander_enabled_stays_within_bounds_22hz():
    cfg = ToneConfig(nominal_freq_hz=22.0, amplitude_rad=150e-6, wander_hz=2.0)
    rng = np.random.default_rng(2)
    tone = WanderingTone(cfg, DT_S, rng, wander_enabled=True)
    freqs = []
    for _ in range(50_000):
        tone.step()
        freqs.append(tone.instantaneous_freq_hz())
    assert min(freqs) >= 20.0 - 1e-9
    assert max(freqs) <= 24.0 + 1e-9


def test_wander_enabled_stays_within_bounds_47hz():
    cfg = ToneConfig(nominal_freq_hz=47.0, amplitude_rad=100e-6, wander_hz=3.0)
    rng = np.random.default_rng(3)
    tone = WanderingTone(cfg, DT_S, rng, wander_enabled=True)
    freqs = []
    for _ in range(50_000):
        tone.step()
        freqs.append(tone.instantaneous_freq_hz())
    assert min(freqs) >= 44.0 - 1e-9
    assert max(freqs) <= 50.0 + 1e-9
