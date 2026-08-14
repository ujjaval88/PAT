"""Closed-loop time-domain simulator.

Update order on every physics tick:

    platform disturbance
      -> plant (actuator, delay, limits, friction, cogging, rigid body, modes)
      -> line of sight
      -> sensors (captured now, released when they actually arrive)
      -> estimator
      -> controller
      -> log

The controller runs at 1 kHz and uses the most recent information that has
arrived. It never looks ahead at a measurement that has not been delivered yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pat_sim.config import SystemConfig
from pat_sim.disturbances.platform import PlatformDisturbance
from pat_sim.plant.gimbal import Gimbal
from pat_sim.sensors.camera import Camera
from pat_sim.sensors.gyro import Gyro
from pat_sim.simulation.clock import SimClock
from pat_sim.simulation.scheduler import ArrivalQueue, PeriodicSchedule


class Controller(Protocol):
    def update(self, error_rad: float) -> float: ...


class Estimator(Protocol):
    def predict_with_gyro(self, gyro_rate_rad_s: float, dt_s: float) -> None: ...
    def correct_with_camera(self, camera_los_rad: float, capture_time_s: float) -> None: ...
    def estimate(self) -> float: ...


@dataclass
class SimulationLog:
    time_s: list[float] = field(default_factory=list)
    theta_los_rad: list[float] = field(default_factory=list)
    theta_g_rad: list[float] = field(default_factory=list)
    theta_b_rad: list[float] = field(default_factory=list)
    # The zero-order-held camera value. Named for the camera-only case, where
    # it is the controller's input. When an estimator is present the controller
    # is driven by estimate_rad instead and this stays the raw camera hold, so
    # the two can be compared directly.
    controller_input_rad: list[float] = field(default_factory=list)
    # What the controller was actually given: the estimator output when one is
    # fitted, otherwise the camera hold. Logged separately because measuring
    # estimator quality against controller_input_rad silently measures the
    # camera instead, and reports the same number no matter what the estimator
    # does.
    estimate_rad: list[float] = field(default_factory=list)
    motor_torque_n_m: list[float] = field(default_factory=list)
    # Truth: the mechanical torque disturbance (friction + cogging) and the
    # gyro reading that a disturbance observer sees. Logged so an observer
    # can be scored against truth without ever being given access to it.
    disturbance_torque_n_m: list[float] = field(default_factory=list)
    gyro_rate_rad_s: list[float] = field(default_factory=list)
    saturated: list[bool] = field(default_factory=list)
    rate_limited: list[bool] = field(default_factory=list)
    controller_saturated: list[bool] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time_s": np.array(self.time_s),
            "theta_los_rad": np.array(self.theta_los_rad),
            "theta_g_rad": np.array(self.theta_g_rad),
            "theta_b_rad": np.array(self.theta_b_rad),
            "controller_input_rad": np.array(self.controller_input_rad),
            "estimate_rad": np.array(self.estimate_rad),
            "motor_torque_n_m": np.array(self.motor_torque_n_m),
            "disturbance_torque_n_m": np.array(self.disturbance_torque_n_m),
            "gyro_rate_rad_s": np.array(self.gyro_rate_rad_s),
            "saturated": np.array(self.saturated),
            "rate_limited": np.array(self.rate_limited),
            "controller_saturated": np.array(self.controller_saturated),
        }


class Simulator:
    def __init__(
        self,
        config: SystemConfig,
        dt_s: float,
        seed: int,
        controller: Controller,
        controller_rate_hz: float = 1000.0,
        use_gyro: bool = False,
        estimator: Estimator | None = None,
        model_compute_delay: bool = True,
        external_torque_fn: Callable[[float], float] | None = None,
    ) -> None:
        self.config = config
        self.dt_s = dt_s
        self.clock = SimClock(dt_s)
        self.controller = controller
        self.estimator = estimator
        self.use_gyro = use_gyro
        self.model_compute_delay = model_compute_delay
        # Optional known torque injected at the MECHANICAL summing junction,
        # alongside friction and cogging. Exists so input-disturbance
        # rejection can be measured as a transfer function against a signal
        # whose amplitude and phase are known exactly, rather than inferred
        # from the difference between two runs.
        self.external_torque_fn = external_torque_fn

        self.gimbal = Gimbal(
            config.plant, config.motor_limits, config.friction, config.cogging, dt_s
        )
        self.platform = PlatformDisturbance(config.platform_disturbance, dt_s, seed)

        seed_seq = np.random.SeedSequence(seed)
        camera_seed, gyro_seed = seed_seq.spawn(2)
        self.camera = Camera(config.camera, np.random.default_rng(camera_seed))
        self.camera_schedule = PeriodicSchedule(config.camera.rate_hz)
        self.camera_queue: ArrivalQueue[tuple[float, float]] = ArrivalQueue()

        if use_gyro:
            self.gyro = Gyro(config.gyro, np.random.default_rng(gyro_seed))
            self.gyro_schedule = PeriodicSchedule(config.gyro.rate_hz)
            # The gyro's 1 ms latency is a real part of the loop delay and
            # must be queued the same way the camera's is. Consuming the sample
            # at capture time would quietly remove 1 ms from the loop, which is
            # a large slice of a 2.5 ms budget, and achievable bandwidth is very
            # sensitive to total delay.
            self.gyro_queue: ArrivalQueue[float] = ArrivalQueue()

        self.controller_schedule = PeriodicSchedule(controller_rate_hz)

        self._last_camera_value_rad = 0.0
        self._last_estimate_rad = 0.0
        self._last_camera_capture_time_s = float("-inf")
        self._last_torque_cmd_n_m = 0.0
        # Torque computed at controller tick k does not reach the actuator
        # until tick k+1. This is the one-sample compute delay the problem
        # statement requires; without it the loop delay is short by nearly a
        # full controller period (1 ms at 1 kHz), and achievable rejection is
        # very sensitive to total delay.
        self._pending_torque_cmd_n_m = 0.0
        self._theta_los_at_last_gyro_sample_rad = 0.0
        self._last_gyro_rate_rad_s = 0.0

        self.log = SimulationLog()

    def run(self, duration_s: float) -> SimulationLog:
        n_steps = int(duration_s / self.dt_s)
        for _ in range(n_steps):
            t = self.clock.time_s

            theta_b = self.platform.step()

            external_torque = (
                self.external_torque_fn(t) if self.external_torque_fn is not None else 0.0
            )
            gimbal_out = self.gimbal.step(self._last_torque_cmd_n_m, external_torque)
            theta_los = gimbal_out.theta_g_rad + theta_b

            if self.camera_schedule.is_due(t):
                meas = self.camera.sample(theta_los, t)
                self.camera_queue.push(meas.arrival_time_s, (meas.value, meas.capture_time_s))
                self.camera_schedule.advance()

            for value, capture_time_s in self.camera_queue.pop_ready(t):
                # Camera latency can exceed its own frame period, so frames can
                # arrive out of capture order. The held value must not be
                # rewound by a frame that is older than the one already in use.
                # The estimator still receives every frame, with its capture
                # time, and deals with the ordering itself.
                if capture_time_s > self._last_camera_capture_time_s:
                    self._last_camera_value_rad = value
                    self._last_camera_capture_time_s = capture_time_s
                if self.estimator is not None:
                    self.estimator.correct_with_camera(value, capture_time_s)

            if self.use_gyro and self.gyro_schedule.is_due(t):
                # Rate is averaged over the gyro's own 1 ms sample interval,
                # not differenced across the 50 us physics tick. Differencing
                # at 20 kHz produces noise well past 1 kHz, and sampling that
                # at the gyro rate would alias it straight into the measurement
                # band. A real gyro's sensing bandwidth prevents that; here it
                # is prevented by measuring displacement over the sensor's own
                # sample period.
                theta_dot_los = (
                    theta_los - self._theta_los_at_last_gyro_sample_rad
                ) / self.gyro_schedule.period_s
                self._theta_los_at_last_gyro_sample_rad = theta_los
                gyro_meas = self.gyro.sample(theta_dot_los, t)
                self._last_gyro_rate_rad_s = gyro_meas.value
                self.gyro_queue.push(gyro_meas.arrival_time_s, gyro_meas.value)
                self.gyro_schedule.advance()

            if self.use_gyro and self.estimator is not None:
                for gyro_value in self.gyro_queue.pop_ready(t):
                    self.estimator.predict_with_gyro(gyro_value, self.gyro_schedule.period_s)

            controller_saturated = False
            if self.controller_schedule.is_due(t):
                if self.model_compute_delay:
                    # Release the command computed one tick ago, then compute
                    # the next one -- the actuator never sees a torque derived
                    # from measurements taken during this same tick.
                    self._last_torque_cmd_n_m = self._pending_torque_cmd_n_m
                controller_input = (
                    self.estimator.estimate()
                    if self.estimator is not None
                    else self._last_camera_value_rad
                )
                self._last_estimate_rad = controller_input
                error = -controller_input
                new_cmd = self.controller.update(error)
                if self.model_compute_delay:
                    self._pending_torque_cmd_n_m = new_cmd
                else:
                    self._last_torque_cmd_n_m = new_cmd
                controller_saturated = getattr(self.controller, "saturated_last_step", False)
                self.controller_schedule.advance()

            self.log.time_s.append(t)
            self.log.theta_los_rad.append(theta_los)
            self.log.theta_g_rad.append(gimbal_out.theta_g_rad)
            self.log.theta_b_rad.append(theta_b)
            self.log.controller_input_rad.append(self._last_camera_value_rad)
            self.log.estimate_rad.append(self._last_estimate_rad)
            self.log.motor_torque_n_m.append(gimbal_out.motor_torque_n_m)
            self.log.disturbance_torque_n_m.append(gimbal_out.disturbance_torque_n_m)
            self.log.gyro_rate_rad_s.append(self._last_gyro_rate_rad_s)
            self.log.saturated.append(gimbal_out.saturated)
            self.log.rate_limited.append(gimbal_out.rate_limited)
            self.log.controller_saturated.append(controller_saturated)

            self.clock.advance()

        return self.log
