"""Unit conversion helpers.

Internal state is always SI (rad, rad/s, N*m, s). These helpers exist only
for the configuration/reporting boundary.
"""

from __future__ import annotations

MICRORAD_PER_RAD = 1e6
MRAD_PER_RAD = 1e3
MS_PER_S = 1e3
US_PER_S = 1e6


def urad_to_rad(value_urad: float) -> float:
    return value_urad / MICRORAD_PER_RAD


def rad_to_urad(value_rad: float) -> float:
    return value_rad * MICRORAD_PER_RAD


def mrad_to_rad(value_mrad: float) -> float:
    return value_mrad / MRAD_PER_RAD


def ms_to_s(value_ms: float) -> float:
    return value_ms / MS_PER_S


def s_to_ms(value_s: float) -> float:
    return value_s * MS_PER_S
