"""Nominal open-loop plant frequency response and pole map.

P(s) = [1/(J*s^2)] * Prod_k[mode_k(s)] * [1/(tau*s+1)] * exp(-Td*s)

The transport delay is evaluated exactly at s=j*omega (magnitude 1, phase
-omega*Td) rather than approximated (e.g. Pade), since only the frequency
response is needed here -- no state-space/time-domain realization of the delay
is required for Part 1.
"""

from __future__ import annotations

import numpy as np

from pat_sim.config import PlantConfig
from pat_sim.plant.structural_mode import StructuralMode


def plant_frequency_response(
    omega_rad_s: np.ndarray, plant_config: PlantConfig, include_delay: bool = True
) -> np.ndarray:
    omega = np.asarray(omega_rad_s, dtype=float)
    s = 1j * omega
    response = 1.0 / (plant_config.inertia_kg_m2 * s**2)
    for mode_cfg in plant_config.modes:
        mode = StructuralMode(mode_cfg.freq_hz, mode_cfg.zeta)
        response = response * np.array([mode.frequency_response(w) for w in omega])
    response = response * (1.0 / (1j * omega * plant_config.actuator_tau_s + 1.0))
    if include_delay:
        response = response * np.exp(-1j * omega * plant_config.transport_delay_s)
    return response


def plant_poles(plant_config: PlantConfig) -> list[complex]:
    poles: list[complex] = [complex(0.0, 0.0), complex(0.0, 0.0)]
    poles.append(complex(-1.0 / plant_config.actuator_tau_s, 0.0))
    for mode_cfg in plant_config.modes:
        mode = StructuralMode(mode_cfg.freq_hz, mode_cfg.zeta)
        poles.extend(mode.poles())
    return poles


def magnitude_db(response: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(response))


def phase_deg(response: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.angle(response)))
