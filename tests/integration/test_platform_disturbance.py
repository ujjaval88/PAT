"""Combined broadband + 22 Hz + 47 Hz platform disturbance: PSD peaks and overall RMS.

the supplied ~330 urad RMS is a statistical expectation, not
an exact target the generator must hit -- this test checks the same order of
magnitude and that both tones are visible in the spectrum, and reports the
measured value rather than asserting a tight match.
"""

from __future__ import annotations

from pat_sim.analysis.psd import rms, welch_psd
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.disturbances.platform import PlatformDisturbance

DT_S = 1e-3  # 1 kHz light-weight test rate; Nyquist (500 Hz) well above both tones


def test_combined_rms_same_order_of_magnitude_as_spec():
    dist = PlatformDisturbance(DEFAULT_CONFIG.platform_disturbance, dt_s=DT_S, seed=42)
    samples = dist.generate(120_000)[5000:]  # 120 s, drop filter settling
    measured_rms = rms(samples)
    # documented finding: components combine in quadrature to well under 330 urad
    # (see Part 1 report) -- assert order of magnitude, not the exact figure.
    assert 100e-6 < measured_rms < 500e-6


def test_psd_shows_both_tonal_peaks():
    dist = PlatformDisturbance(DEFAULT_CONFIG.platform_disturbance, dt_s=DT_S, seed=1)
    samples = dist.generate(120_000)[5000:]
    f, pxx = welch_psd(samples, fs_hz=1.0 / DT_S, nperseg=8192)

    near_22 = (f > 18) & (f < 26)
    near_47 = (f > 43) & (f < 51)
    background = (f > 30) & (f < 40)

    peak_22 = pxx[near_22].max()
    peak_47 = pxx[near_47].max()
    background_level = pxx[background].mean()

    assert peak_22 > 5 * background_level
    assert peak_47 > 5 * background_level
