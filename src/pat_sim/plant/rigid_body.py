"""Rigid-body gimbal dynamics: J * theta_ddot = T_net."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigidBodyState:
    theta_rad: float = 0.0
    theta_dot_rad_s: float = 0.0


class RigidBody:
    def __init__(self, inertia_kg_m2: float, state: RigidBodyState | None = None) -> None:
        self.inertia_kg_m2 = inertia_kg_m2
        self.state = state if state is not None else RigidBodyState()

    def acceleration_rad_s2(self, torque_n_m: float) -> float:
        return torque_n_m / self.inertia_kg_m2

    def step(self, torque_n_m: float, dt_s: float) -> RigidBodyState:
        """RK4 step assuming torque is held constant (ZOH) over dt_s."""
        accel = torque_n_m / self.inertia_kg_m2

        def deriv(s: np.ndarray) -> np.ndarray:
            return np.array([s[1], accel])

        s0 = np.array([self.state.theta_rad, self.state.theta_dot_rad_s])
        k1 = deriv(s0)
        k2 = deriv(s0 + dt_s / 2 * k1)
        k3 = deriv(s0 + dt_s / 2 * k2)
        k4 = deriv(s0 + dt_s * k3)
        s1 = s0 + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state = RigidBodyState(theta_rad=s1[0], theta_dot_rad_s=s1[1])
        return self.state
