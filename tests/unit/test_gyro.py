import math

import numpy as np
import pytest

from pat_sim.config import GyroConfig
from pat_sim.sensors.gyro import Gyro

CFG = GyroConfig()


def test_noise_std_isolated_from_bias():
    gyro = Gyro(CFG, np.random.default_rng(0))
    residuals = []
    for _ in range(20_000):
        m = gyro.sample(true_rate_rad_s=0.0, capture_time_s=0.0)
        residuals.append(m.value - gyro.bias_rad_s)  # subtract bias *after* this step's update
    measured_std = float(np.std(residuals))
    assert abs(measured_std - CFG.noise_std_rad_s) / CFG.noise_std_rad_s < 0.1


def test_bias_random_walk_variance_grows_with_time():
    cfg = GyroConfig(noise_std_rad_s=0.0, bias_random_walk_std_rad_s_per_sqrt_s=1e-3)
    n_steps = 4000
    n_realizations = 200
    finals = []
    for seed in range(n_realizations):
        gyro = Gyro(cfg, np.random.default_rng(seed))
        for _ in range(n_steps):
            gyro.sample(0.0, 0.0)
        finals.append(gyro.bias_rad_s)
    measured_std = float(np.std(finals))
    duration_s = n_steps / cfg.rate_hz
    expected_std = cfg.bias_random_walk_std_rad_s_per_sqrt_s * math.sqrt(duration_s)
    assert abs(measured_std - expected_std) / expected_std < 0.25


def test_sample_rate_and_latency_config():
    assert CFG.rate_hz == 1000.0
    assert CFG.latency_s == pytest.approx(1e-3)
