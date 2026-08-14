import cmath
import math

import numpy as np
import pytest

from pat_sim.plant.structural_mode import StructuralMode


def test_mode1_poles():
    mode = StructuralMode(freq_hz=75.0, zeta=0.03)
    p1, p2 = mode.poles()
    assert p1 == pytest.approx(-14.14 + 471.0j, abs=0.5)
    assert p2 == pytest.approx(-14.14 - 471.0j, abs=0.5)


def test_mode2_poles():
    mode = StructuralMode(freq_hz=220.0, zeta=0.04)
    p1, p2 = mode.poles()
    assert p1 == pytest.approx(-55.29 + 1381.0j, abs=0.5)
    assert p2 == pytest.approx(-55.29 - 1381.0j, abs=0.5)


def test_resonance_peak_near_natural_frequency():
    mode = StructuralMode(freq_hz=75.0, zeta=0.03)
    freqs_hz = np.linspace(60.0, 90.0, 3000)
    mags = [abs(mode.frequency_response(2 * math.pi * f)) for f in freqs_hz]
    peak_freq = freqs_hz[int(np.argmax(mags))]
    assert peak_freq == pytest.approx(75.0, abs=1.0)


def test_disabled_mode_is_passthrough():
    mode = StructuralMode(freq_hz=75.0, zeta=0.03, enabled=False)
    assert mode.frequency_response(2 * math.pi * 75.0) == cmath.rect(1.0, 0.0)
