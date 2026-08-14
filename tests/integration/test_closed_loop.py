"""Closed-loop regression tests for the Part 2/3 simulator + controller +
estimator stack. Short durations relative to the deliverable scripts (which
run longer for good RMS statistics) -- these check basic sanity (no runaway,
limits respected, sane wiring), not the exact reported RMS numbers."""

import math

import numpy as np

from pat_sim.analysis.sensitivity import camera_extra_delay_s, design_camera_only_loop
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.coarse import CoarseController, ControllerConfig
from pat_sim.estimation.estimator import KalmanFusionEstimator
from pat_sim.simulation.simulator import Simulator

CFG = DEFAULT_CONFIG


def _camera_only_controller():
    extra_delay = camera_extra_delay_s(CFG.camera, 0.90)
    total_delay = extra_delay + CFG.plant.transport_delay_s
    omega_c = math.radians(45.0) / total_delay
    design = design_camera_only_loop(
        CFG.plant, omega_c, 45.0, extra_delay, alpha=8.0, n_stages=2,
        rolloff_freq_rad_s=2 * math.pi * 20.0,
    )
    return CoarseController(
        ControllerConfig(lead_lag=design, torque_limit_n_m=0.5, resonant=()), dt_s=1e-3
    )


def test_camera_only_closed_loop_does_not_saturate_or_diverge():
    controller = _camera_only_controller()
    sim = Simulator(CFG, dt_s=50e-6, seed=1, controller=controller, controller_rate_hz=1000.0)
    log = sim.run(duration_s=1.0)
    theta = np.array(log.theta_los_rad)
    assert np.all(np.isfinite(theta))
    assert np.max(np.abs(theta)) < 2000e-6  # generous sanity bound, not a blow-up
    assert all(abs(t) <= CFG.motor_limits.torque_max_n_m + 1e-9 for t in log.motor_torque_n_m)


def test_axis_rate_stays_within_limit_and_is_not_active_for_this_disturbance():
    """The +/-3 rad/s limit is on gimbal angular rate. Against this disturbance
    (peak base rate ~0.42 rad/s) it should never engage -- if it does, either
    the loop is misbehaving or the limit has been misinterpreted again."""
    controller = _camera_only_controller()
    sim = Simulator(CFG, dt_s=50e-6, seed=2, controller=controller, controller_rate_hz=1000.0)
    log = sim.run(duration_s=0.5)
    assert not any(log.rate_limited)


def test_fused_closed_loop_with_kalman_estimator_does_not_diverge():
    extra_delay_gyro = 1.5e-3
    omega_c = 2 * math.pi * 2.0
    design = design_camera_only_loop(
        CFG.plant, omega_c, 45.0, extra_delay_gyro, alpha=8.0, n_stages=2
    )
    controller = CoarseController(
        ControllerConfig(lead_lag=design, torque_limit_n_m=0.5, resonant=()), dt_s=1e-3
    )
    estimator = KalmanFusionEstimator(CFG.gyro, CFG.camera, gyro_dt_s=1e-3)
    sim = Simulator(
        CFG, dt_s=50e-6, seed=3, controller=controller, controller_rate_hz=1000.0,
        use_gyro=True, estimator=estimator,
    )
    log = sim.run(duration_s=1.0)
    theta = np.array(log.theta_los_rad)
    assert np.all(np.isfinite(theta))
    assert np.max(np.abs(theta)) < 2000e-6


def test_zero_disturbance_zero_torque_stays_at_rest():
    import dataclasses

    from pat_sim.config import BroadbandDisturbanceConfig, PlatformDisturbanceConfig, ToneConfig

    silent_cfg = dataclasses.replace(
        CFG,
        platform_disturbance=PlatformDisturbanceConfig(
            broadband=BroadbandDisturbanceConfig(target_rms_rad=0.0),
            tone_22hz=ToneConfig(nominal_freq_hz=22.0, amplitude_rad=0.0, wander_hz=0.0),
            tone_47hz=ToneConfig(nominal_freq_hz=47.0, amplitude_rad=0.0, wander_hz=0.0),
        ),
    )

    class ZeroController:
        saturated_last_step = False

        def update(self, error_rad: float) -> float:
            return 0.0

    sim = Simulator(
        silent_cfg, dt_s=50e-6, seed=4, controller=ZeroController(), controller_rate_hz=1000.0
    )
    log = sim.run(duration_s=0.1)
    assert all(abs(x) < 1e-12 for x in log.theta_los_rad)
