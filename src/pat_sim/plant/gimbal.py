"""Combined time-domain plant: actuator -> delay -> saturation -> rate limit
-> mechanical torque summing (friction, cogging) -> rigid body -> mode 1 ->
mode 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pat_sim.config import CoggingConfig, FrictionConfig, MotorLimitsConfig, PlantConfig
from pat_sim.plant.actuator import Actuator
from pat_sim.plant.cogging import cogging_torque
from pat_sim.plant.friction import friction_torque
from pat_sim.plant.limits import saturate_torque
from pat_sim.plant.rigid_body import RigidBody, RigidBodyState
from pat_sim.plant.structural_mode import StructuralMode
from pat_sim.plant.transport_delay import TransportDelay


@dataclass
class GimbalOutput:
    theta_g_rad: float
    theta_dot_g_rad_s: float
    motor_torque_n_m: float
    saturated: bool
    rate_limited: bool
    # Truth signal, for validating a disturbance observer. This is the
    # mechanical torque acting at the summing junction that is NOT the motor
    # torque -- friction plus cogging. Estimators must never read it; it exists
    # so an estimate of it can be scored against the real thing.
    disturbance_torque_n_m: float = 0.0
    # Rigid-body shaft angle, i.e. before the structural modes. theta_g_rad is
    # the mode-filtered angle that reaches the LOS; the torque acts here.
    theta_shaft_rad: float = 0.0


class Gimbal:
    def __init__(
        self,
        plant_config: PlantConfig,
        motor_limits: MotorLimitsConfig,
        friction_config: FrictionConfig,
        cogging_config: CoggingConfig,
        dt_s: float,
    ) -> None:
        self.dt_s = dt_s
        self.motor_limits = motor_limits
        self.friction_config = friction_config
        self.cogging_config = cogging_config

        self.actuator = Actuator(tau_s=plant_config.actuator_tau_s)
        self.transport_delay = TransportDelay(delay_s=plant_config.transport_delay_s, dt_s=dt_s)
        self.rigid_body = RigidBody(inertia_kg_m2=plant_config.inertia_kg_m2)
        self.modes = [StructuralMode(m.freq_hz, m.zeta) for m in plant_config.modes]

        self._theta_g_prev_rad = 0.0

    def step(self, commanded_torque_n_m: float, external_torque_n_m: float = 0.0) -> GimbalOutput:
        t_actuator = self.actuator.step(commanded_torque_n_m, self.dt_s)
        t_delayed = self.transport_delay.step(t_actuator)
        t_motor = saturate_torque(t_delayed, self.motor_limits.torque_max_n_m)
        saturated = bool(t_motor != t_delayed)

        theta = self.rigid_body.state.theta_rad
        theta_dot = self.rigid_body.state.theta_dot_rad_s
        t_friction = friction_torque(
            theta_dot, self.friction_config.coulomb_n_m, self.friction_config.viscous_n_m_s_per_rad
        )
        t_cogging = cogging_torque(
            theta,
            self.cogging_config.amplitude_n_m,
            self.cogging_config.cycles_per_revolution,
            self.cogging_config.phase_rad,
        )
        t_net = t_motor - t_friction - t_cogging + external_torque_n_m

        rb_state = self.rigid_body.step(t_net, self.dt_s)

        # Axis angular-rate limit (+/-3 rad/s). Physically this comes from
        # back-EMF/voltage saturation: past some speed the motor cannot be
        # driven faster regardless of commanded torque. Clamping the velocity
        # state is the standard representation. For this disturbance it is
        # essentially never active (peak demand ~0.42 rad/s, ~7x margin) --
        # it exists to catch slews and acquisition transients, not vibration
        # rejection.
        rate_max = self.motor_limits.rate_max_rad_s
        # bool() because the rigid-body state comes back through numpy: a bare
        # np.bool_ would silently fail `is True` identity checks downstream.
        rate_limited = bool(abs(rb_state.theta_dot_rad_s) > rate_max)
        if rate_limited:
            clamped = math.copysign(rate_max, rb_state.theta_dot_rad_s)
            self.rigid_body.state = RigidBodyState(
                theta_rad=rb_state.theta_rad, theta_dot_rad_s=clamped
            )
            rb_state = self.rigid_body.state

        y = rb_state.theta_rad
        for mode in self.modes:
            y = mode.step(y, self.dt_s)
        theta_g = y
        theta_dot_g = (theta_g - self._theta_g_prev_rad) / self.dt_s
        self._theta_g_prev_rad = theta_g

        return GimbalOutput(
            theta_g_rad=theta_g,
            theta_dot_g_rad_s=theta_dot_g,
            motor_torque_n_m=t_motor,
            saturated=saturated,
            rate_limited=rate_limited,
            disturbance_torque_n_m=t_friction + t_cogging,
            theta_shaft_rad=rb_state.theta_rad,
        )
