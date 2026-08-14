"""Common measurement contract for all sensors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Measurement:
    value: float
    capture_time_s: float
    arrival_time_s: float
    valid: bool = True
