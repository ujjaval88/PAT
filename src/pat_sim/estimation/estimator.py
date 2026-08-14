"""Kalman filter fusing gyro-driven propagation with delayed camera corrections.

States: [theta_los, gyro_bias]. The gyro rate measurement drives prediction
(strapdown-style integration) rather than being treated as a measurement of a
separate rate state -- theta_dot_los is not tracked explicitly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pat_sim.config import CameraConfig, GyroConfig


@dataclass
class _Snapshot:
    time_s: float
    x: np.ndarray
    p: np.ndarray
    gyro_meas_rad_s: float


class KalmanFusionEstimator:
    def __init__(
        self,
        gyro_config: GyroConfig,
        camera_config: CameraConfig,
        gyro_dt_s: float,
        buffer_duration_s: float = 0.15,
        initial_theta_variance: float = 1e-6,
        initial_bias_variance: float = 1e-8,
    ) -> None:
        self.gyro_dt_s = gyro_dt_s
        self.x = np.array([0.0, 0.0])
        self.p = np.diag([initial_theta_variance, initial_bias_variance])
        self.q = np.diag(
            [
                (gyro_config.noise_std_rad_s * gyro_dt_s) ** 2,
                (gyro_config.bias_random_walk_std_rad_s_per_sqrt_s**2) * gyro_dt_s,
            ]
        )
        self.r_camera = camera_config.noise_std_rad**2
        self._buffer: deque[_Snapshot] = deque()
        self._buffer_duration_s = buffer_duration_s
        self._time_s = 0.0

    def predict_with_gyro(self, gyro_rate_rad_s: float, dt_s: float) -> None:
        f = np.array([[1.0, -dt_s], [0.0, 1.0]])
        b = np.array([dt_s, 0.0])
        self.x = f @ self.x + b * gyro_rate_rad_s
        self.p = f @ self.p @ f.T + self.q
        self._time_s += dt_s
        self._buffer.append(_Snapshot(self._time_s, self.x.copy(), self.p.copy(), gyro_rate_rad_s))
        while self._buffer and self._time_s - self._buffer[0].time_s > self._buffer_duration_s:
            self._buffer.popleft()

    def _apply_correction(
        self, x: np.ndarray, p: np.ndarray, camera_los_rad: float
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.array([1.0, 0.0])
        y = camera_los_rad - h @ x
        s = h @ p @ h.T + self.r_camera
        k = (p @ h.T) / s
        x_new = x + k * y
        p_new = (np.eye(2) - np.outer(k, h)) @ p
        return x_new, p_new

    def correct_with_camera(self, camera_los_rad: float, capture_time_s: float) -> None:
        if not self._buffer or capture_time_s < self._buffer[0].time_s:
            self.x, self.p = self._apply_correction(self.x, self.p, camera_los_rad)
            return

        idx = 0
        for i, snap in enumerate(self._buffer):
            if snap.time_s <= capture_time_s:
                idx = i
            else:
                break

        snap = self._buffer[idx]
        x, p = self._apply_correction(snap.x, snap.p, camera_los_rad)

        f = np.array([[1.0, -self.gyro_dt_s], [0.0, 1.0]])
        b = np.array([self.gyro_dt_s, 0.0])
        # _buffer[idx].gyro_meas_rad_s already produced _buffer[idx].x -- replay
        # must start from idx+1, or that gyro sample gets double-counted.
        for j in range(idx + 1, len(self._buffer)):
            u = self._buffer[j].gyro_meas_rad_s
            x = f @ x + b * u
            p = f @ p @ f.T + self.q

        self.x, self.p = x, p

    def estimate(self) -> float:
        return float(self.x[0])

    @property
    def bias_estimate_rad_s(self) -> float:
        return float(self.x[1])
