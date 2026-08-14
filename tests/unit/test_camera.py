import numpy as np
import pytest

from pat_sim.config import CameraConfig
from pat_sim.sensors.camera import Camera

CFG = CameraConfig()


def test_sample_interval():
    camera = Camera(CFG, np.random.default_rng(0))
    assert camera.sample_interval_s == pytest.approx(1.0 / 60.0)


def test_noise_std_matches_config():
    camera = Camera(CFG, np.random.default_rng(0))
    values = [
        camera.sample(0.0, capture_time_s=k * camera.sample_interval_s).value
        for k in range(20_000)
    ]
    measured_std = float(np.std(values))
    assert abs(measured_std - CFG.noise_std_rad) / CFG.noise_std_rad < 0.1


def test_latency_never_negative():
    camera = Camera(CFG, np.random.default_rng(1))
    latencies = [camera.draw_latency_s() for _ in range(20_000)]
    assert min(latencies) >= 0.0


def test_latency_deterministic_for_fixed_seed():
    a = Camera(CFG, np.random.default_rng(5))
    b = Camera(CFG, np.random.default_rng(5))
    assert [a.draw_latency_s() for _ in range(100)] == [b.draw_latency_s() for _ in range(100)]


def test_arrival_time_not_before_capture_time():
    camera = Camera(CFG, np.random.default_rng(2))
    for k in range(5000):
        m = camera.sample(0.0, capture_time_s=k * 1e-3)
        assert m.arrival_time_s >= m.capture_time_s


def test_capture_time_unaffected_by_latency():
    camera = Camera(CFG, np.random.default_rng(3))
    m = camera.sample(0.0, capture_time_s=1.2345)
    assert m.capture_time_s == 1.2345


def test_latency_mean_matches_clipped_gaussian_moment():
    camera = Camera(CFG, np.random.default_rng(4))
    latencies = np.array([camera.draw_latency_s() for _ in range(200_000)])
    # E[max(0,X)] for X~N(10ms,20ms): mu*Phi(mu/sigma) + sigma*phi(mu/sigma) ~= 13.96ms
    assert abs(latencies.mean() - 13.96e-3) < 0.5e-3
