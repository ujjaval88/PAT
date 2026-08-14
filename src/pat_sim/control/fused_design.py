"""The delivered Part 3 controller: the same two-stage lead compensator with
rolloff, but driven by the fused gyro+camera estimate instead of the raw camera.

Same structure as Part 2:

    C(s) = Kp * [(1 + sT) / (1 + sT/alpha)]^2 * 1/(1 + s/w_p)
"""

from __future__ import annotations

import math

from pat_sim.analysis.sensitivity import CameraLoopDesign, design_camera_only_loop
from pat_sim.config import DEFAULT_CONFIG, SystemConfig

# Delay the loop sees on the gyro path. The plant's own 0.5 ms transport delay
# and its actuator lag are already inside P(s) and are not counted here.
GYRO_ZOH_S = 0.5e-3
GYRO_LATENCY_S = 1.0e-3
COMPUTE_DELAY_S = 1.0e-3
EXTRA_DELAY_S = GYRO_ZOH_S + GYRO_LATENCY_S + COMPUTE_DELAY_S

PHASE_MARGIN_DEG = 45.0
LEAD_ALPHA = 8.0
N_STAGES = 2

# Chosen by measurement, not from the delay budget. The 2.5 ms delay alone
# would allow a crossover near 50 Hz, but a sweep of the closed loop shows the
# error rising steadily above a few Hz: the disturbance sits above crossover
# either way, and past about 8 Hz the loop gain at the 75 Hz structural mode
# exceeds 1 and the mode stops being safely attenuated.
CROSSOVER_HZ = 2.0
ROLLOFF_FREQ_HZ = 10.0

TORQUE_LIMIT_N_M = 0.5
CONTROLLER_RATE_HZ = 1000.0
GYRO_DT_S = 1.0e-3


def build_design(
    config: SystemConfig = DEFAULT_CONFIG,
    crossover_hz: float = CROSSOVER_HZ,
    rolloff_freq_hz: float | None = ROLLOFF_FREQ_HZ,
) -> CameraLoopDesign:
    """Two-stage lead plus one rolloff pole, solved for the requested crossover
    and phase margin on the gyro delay budget."""
    return design_camera_only_loop(
        config.plant,
        2.0 * math.pi * crossover_hz,
        PHASE_MARGIN_DEG,
        EXTRA_DELAY_S,
        alpha=LEAD_ALPHA,
        n_stages=N_STAGES,
        rolloff_freq_rad_s=(
            2.0 * math.pi * rolloff_freq_hz if rolloff_freq_hz is not None else None
        ),
    )


def design_parameters(design: CameraLoopDesign) -> dict[str, str]:
    """Formatted design point, for the design script."""
    return {
        "structure": f"Kp * [(1+sT)/(1+sT/alpha)]^{N_STAGES} * 1/(1+s/w_p)",
        "driven by": "fused gyro + camera estimate",
        "modelled extra delay": (
            f"{design.extra_delay_s * 1e3:.2f} ms "
            "(gyro ZOH + gyro latency + compute delay)"
        ),
        "crossover f_c": f"{design.omega_c_rad_s / (2 * math.pi):.2f} Hz",
        "phase margin target": f"{PHASE_MARGIN_DEG:.1f} deg",
        "lead alpha / stages": f"{LEAD_ALPHA:g} / {N_STAGES}",
        "lead time constant T": f"{design.lead_time_constant_s * 1e3:.3f} ms",
        "gain Kp": f"{design.kp:.4g}",
        "rolloff pole w_p": f"{ROLLOFF_FREQ_HZ:.1f} Hz (1 real pole)",
        "integrator": "none (no pole at the origin, so no wind-up)",
        "torque limit": f"{TORQUE_LIMIT_N_M:g} N*m",
        "update rate": f"{CONTROLLER_RATE_HZ:.0f} Hz, bilinear-discretised as an SOS cascade",
    }
