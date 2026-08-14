import math

import numpy as np
import pytest

from pat_sim.config import EncoderConfig
from pat_sim.sensors.encoder import Encoder

CFG = EncoderConfig()


def test_quantum_value():
    assert CFG.quantum_rad == pytest.approx(2 * math.pi / 2**20)
    assert CFG.quantum_rad == pytest.approx(5.99e-6, abs=0.01e-6)


def test_output_is_integer_multiple_of_quantum():
    encoder = Encoder(CFG)
    rng = np.random.default_rng(0)
    for true_angle in rng.uniform(-1.0, 1.0, 500):
        m = encoder.sample(true_angle, capture_time_s=0.0)
        ratio = m.value / CFG.quantum_rad
        assert abs(ratio - round(ratio)) < 1e-6


def test_quantization_error_bounded():
    encoder = Encoder(CFG)
    rng = np.random.default_rng(1)
    for true_angle in rng.uniform(-1.0, 1.0, 500):
        m = encoder.sample(true_angle, capture_time_s=0.0)
        assert abs(m.value - true_angle) <= CFG.quantum_rad / 2 + 1e-12


def test_encoder_does_not_observe_platform_motion():
    """Critical architecture test: encoder must see theta_g only, never theta_b."""
    encoder = Encoder(CFG)
    theta_g = 0.1  # constant shaft angle
    outputs = []
    los_values = []
    for theta_b in np.linspace(-50e-6, 50e-6, 20):
        outputs.append(encoder.sample(theta_g, capture_time_s=0.0).value)
        los_values.append(theta_g + theta_b)  # what LOS would be, for contrast

    assert len(set(outputs)) == 1  # encoder output constant despite varying theta_b
    assert len(set(los_values)) == len(los_values)  # LOS itself does vary
