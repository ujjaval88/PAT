import math

import numpy as np
import pytest

from pat_sim.config import CameraConfig, GyroConfig
from pat_sim.estimation.estimator import KalmanFusionEstimator


def test_tracks_sinusoid_with_delayed_corrections():
    """Regression test for an off-by-one in the buffered replay: replaying
    from the found snapshot's own index (instead of the next one) double-
    counted one gyro sample per correction, which compounded badly over many
    corrections/sec."""
    gyro_cfg = GyroConfig(noise_std_rad_s=1e-6, bias_random_walk_std_rad_s_per_sqrt_s=1e-8)
    cam_cfg = CameraConfig(noise_std_rad=1e-8)
    est = KalmanFusionEstimator(
        gyro_cfg, cam_cfg, gyro_dt_s=1e-3, initial_theta_variance=1e-9, initial_bias_variance=1e-12
    )

    dt = 1e-3
    duration = 2.0
    n = int(duration / dt)
    latency_s = 0.030
    freq_hz = 22.0
    amplitude_rad = 150e-6

    rng = np.random.default_rng(0)
    pending: list[tuple[float, float, float]] = []
    errors = []
    for k in range(n):
        t = k * dt
        true_theta = amplitude_rad * math.sin(2 * math.pi * freq_hz * t)
        true_rate = amplitude_rad * 2 * math.pi * freq_hz * math.cos(2 * math.pi * freq_hz * t)
        gyro_meas = true_rate + rng.normal(0, 1e-6)
        est.predict_with_gyro(gyro_meas, dt)
        if k % round((1 / 60) / dt) == 0:
            pending.append((t + latency_s, true_theta + rng.normal(0, 1e-8), t))
        ready = [p for p in pending if p[0] <= t]
        pending = [p for p in pending if p[0] > t]
        for _arrival, value, capture in ready:
            est.correct_with_camera(value, capture)
        errors.append(est.estimate() - true_theta)

    errors_arr = np.array(errors)
    settle = int(0.5 / dt)
    rms_error = float(np.sqrt(np.mean(errors_arr[settle:] ** 2)))
    # amplitude is 150 urad; a correctly-replaying filter tracks to a small
    # fraction of that even with 30ms delayed corrections.
    assert rms_error < 20e-6


def test_bias_is_estimated_from_a_constant_offset():
    gyro_cfg = GyroConfig(noise_std_rad_s=1e-6, bias_random_walk_std_rad_s_per_sqrt_s=0.0)
    cam_cfg = CameraConfig(noise_std_rad=1e-7)
    est = KalmanFusionEstimator(gyro_cfg, cam_cfg, gyro_dt_s=1e-3)

    true_bias = 2e-4  # rad/s, a constant (unmodeled-drift) offset
    dt = 1e-3
    n = 5000
    theta_true = 0.0
    for k in range(n):
        t = k * dt
        gyro_meas = 0.0 + true_bias  # true rate is zero; gyro reports the bias
        est.predict_with_gyro(gyro_meas, dt)
        if k % 16 == 0:
            est.correct_with_camera(theta_true, t)

    assert est.bias_estimate_rad_s == pytest.approx(true_bias, rel=0.2)


def test_estimate_starts_at_zero():
    est = KalmanFusionEstimator(GyroConfig(), CameraConfig(), gyro_dt_s=1e-3)
    assert est.estimate() == 0.0
