"""47 Hz LOS content, sampled by the 60 Hz camera, must alias to |47-60| = 13 Hz.
This directly validates one of the Part 1 fundamental-limit conclusions."""

import math

import numpy as np

from pat_sim.analysis.psd import dominant_frequency_hz
from pat_sim.config import CameraConfig
from pat_sim.sensors.camera import Camera


def test_47hz_los_aliases_to_13hz_under_60hz_camera_sampling():
    camera_rate_hz = 60.0
    cfg = CameraConfig(rate_hz=camera_rate_hz, noise_std_rad=0.0)
    camera = Camera(cfg, np.random.default_rng(0))

    n_samples = 3000  # 50 s at 60 Hz -> ~0.02 Hz FFT bin resolution
    capture_times = np.arange(n_samples) / camera_rate_hz
    true_los = 150e-6 * np.sin(2 * math.pi * 47.0 * capture_times)

    measured = np.array(
        [camera.sample(true_los[k], capture_times[k]).value for k in range(n_samples)]
    )

    dominant_hz = dominant_frequency_hz(measured, fs_hz=camera_rate_hz)
    assert abs(dominant_hz - 13.0) < 0.3
