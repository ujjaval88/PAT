import numpy as np
import pytest

from pat_sim.config import QpdConfig
from pat_sim.sensors.qpd import QPD

CFG_NOISELESS = QpdConfig(noise_std_rad=0.0)


def test_sample_interval():
    assert 1.0 / CFG_NOISELESS.rate_hz == pytest.approx(100e-6)


@pytest.mark.parametrize(
    ("true_value", "expected_valid"),
    [(-200e-6, True), (200e-6, True), (-201e-6, False), (201e-6, False)],
)
def test_validity_boundaries(true_value, expected_valid):
    qpd = QPD(CFG_NOISELESS, np.random.default_rng(0))
    m = qpd.sample(true_value, capture_time_s=0.0)
    assert m.valid is expected_valid


def test_noise_std_matches_config():
    cfg = QpdConfig()
    qpd = QPD(cfg, np.random.default_rng(1))
    values = [qpd.sample(0.0, capture_time_s=0.0).value for _ in range(20_000)]
    measured_std = float(np.std(values))
    assert abs(measured_std - cfg.noise_std_rad) / cfg.noise_std_rad < 0.1
