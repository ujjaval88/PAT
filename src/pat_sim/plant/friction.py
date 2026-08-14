"""Coulomb + viscous friction: Tf = Tc*sign(theta_dot) + b*theta_dot."""

from __future__ import annotations

import math


def friction_torque(
    theta_dot_rad_s: float,
    coulomb_n_m: float,
    viscous_n_m_s_per_rad: float,
    smoothing_velocity_rad_s: float | None = None,
) -> float:
    if smoothing_velocity_rad_s is not None and smoothing_velocity_rad_s > 0:
        sign = math.tanh(theta_dot_rad_s / smoothing_velocity_rad_s)
    else:
        sign = math.copysign(1.0, theta_dot_rad_s) if theta_dot_rad_s != 0 else 0.0
    return coulomb_n_m * sign + viscous_n_m_s_per_rad * theta_dot_rad_s
