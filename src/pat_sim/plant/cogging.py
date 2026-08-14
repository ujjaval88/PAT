"""Cogging torque: periodic in mechanical angle, not time. Tcog = Acog*sin(Ncog*theta + phi)."""

from __future__ import annotations

import math


def cogging_torque(
    theta_rad: float,
    amplitude_n_m: float,
    cycles_per_revolution: int,
    phase_rad: float = 0.0,
) -> float:
    return amplitude_n_m * math.sin(cycles_per_revolution * theta_rad + phase_rad)
