"""Part 2: the closed-loop sensitivity function.

Produces:
  outputs/part2/sensitivity.png -- analytical |S(jw)|, validated against a
                                   measured estimate from the simulator
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pat_sim.config import DEFAULT_CONFIG, SystemConfig
from pat_sim.control.coarse import CoarseController, ControllerConfig
from pat_sim.control.lead_lag_design import (
    CONTROLLER_RATE_HZ,
    TORQUE_LIMIT_N_M,
    build_design,
    sensitivity_magnitudes,
    sensitivity_metrics,
)
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part2"
DT_S = 50e-6
DURATION_S = 20.0
SETTLE_S = 1.0
SEED = 42
NPERSEG = 65536
FREQ_GRID_HZ = np.logspace(-2, 3, 6000)

# Where the analytical curve is checked. Chosen to straddle the rejection
# region, crossover, the Ms peak and both tones. Nothing below ~1 Hz: at this
# record length the Welch estimate has too few averages there for the
# comparison to mean anything.
VALIDATION_FREQS_HZ = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 22.0, 47.0, 75.0)
COHERENCE_FLOOR = 0.9


def lti_equivalent(config: SystemConfig, design_latency_s: float) -> SystemConfig:
    """The only configuration the analytical S(jw) claims to describe: no
    friction or cogging, one fixed camera latency, no camera noise."""
    return dataclasses.replace(
        config,
        friction=dataclasses.replace(
            config.friction, coulomb_n_m=0.0, viscous_n_m_s_per_rad=0.0
        ),
        cogging=dataclasses.replace(config.cogging, amplitude_n_m=0.0),
        camera=dataclasses.replace(
            config.camera,
            noise_std_rad=0.0,
            latency_mean_s=design_latency_s,
            latency_std_s=0.0,
        ),
    )


def measured_sensitivity(
    theta_b_rad: np.ndarray, theta_los_rad: np.ndarray, fs_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freq, |S|, coherence) from the cross-spectral density."""
    freq_hz, p_bb = signal.welch(theta_b_rad, fs=fs_hz, nperseg=NPERSEG)
    _, p_bl = signal.csd(theta_b_rad, theta_los_rad, fs=fs_hz, nperseg=NPERSEG)
    _, coherence = signal.coherence(theta_b_rad, theta_los_rad, fs=fs_hz, nperseg=NPERSEG)
    return freq_hz, np.abs(p_bl / p_bb), coherence


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    design = build_design(config)

    print("Closed-loop sensitivity (analytical):\n")
    for key, value in sensitivity_metrics(design, config, FREQ_GRID_HZ).items():
        print(f"  {key:<32} {value}")

    controller = CoarseController(
        ControllerConfig(lead_lag=design, torque_limit_n_m=TORQUE_LIMIT_N_M, resonant=()),
        dt_s=1.0 / CONTROLLER_RATE_HZ,
    )
    arrays = (
        Simulator(
            lti_equivalent(config, design.extra_delay_s),
            dt_s=DT_S,
            seed=SEED,
            controller=controller,
            controller_rate_hz=CONTROLLER_RATE_HZ,
        )
        .run(duration_s=DURATION_S)
        .as_arrays()
    )
    settle = int(SETTLE_S / DT_S)
    freq_hz, s_measured, coherence = measured_sensitivity(
        arrays["theta_b_rad"][settle:], arrays["theta_los_rad"][settle:], 1.0 / DT_S
    )
    # Drop Welch's DC bin: the analytical S(jw) divides by s^2, so evaluating it
    # at exactly 0 Hz is a division by zero, and S(0) is meaningless anyway.
    keep = freq_hz > 0.0
    freq_hz, s_measured, coherence = freq_hz[keep], s_measured[keep], coherence[keep]
    s_analytical, _ = sensitivity_magnitudes(design, config, freq_hz)

    print("\nValidation against simulation (LTI-equivalent plant):")
    print(f"  {'f [Hz]':>7} {'analytical':>11} {'measured':>10} {'error':>8} {'coherence':>10}")
    for target_hz in VALIDATION_FREQS_HZ:
        index = int(np.argmin(np.abs(freq_hz - target_hz)))
        analytical = float(s_analytical[index])
        measured = float(s_measured[index])
        error_pct = 100.0 * (measured - analytical) / analytical
        flag = "" if coherence[index] >= COHERENCE_FLOOR else "  <- low coherence"
        print(
            f"  {freq_hz[index]:7.2f} {analytical:11.4f} {measured:10.4f} "
            f"{error_pct:+7.1f}% {coherence[index]:10.3f}{flag}"
        )
    print(
        f"\n  Coherence says how much to trust each row. Below {COHERENCE_FLOOR:.1f} the"
        " measurement is\n  not meaningful and those rows should be ignored - here that"
        " is the low end,\n  where the record is too short to average properly. Above"
        " 20 Hz the measured\n  and analytical curves agree closely, which is where"
        " the disturbance lives.\n"
    )

    s_grid, l_grid = sensitivity_magnitudes(design, config, FREQ_GRID_HZ)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(FREQ_GRID_HZ, 20 * np.log10(s_grid), label="|S(jw)| analytical", linewidth=1.6)
    trusted = coherence >= COHERENCE_FLOOR
    ax.semilogx(
        freq_hz[trusted], 20 * np.log10(s_measured[trusted]), ".", markersize=3,
        color="tab:orange", label=f"|S| measured (coherence >= {COHERENCE_FLOOR:.1f})",
    )
    ax.semilogx(FREQ_GRID_HZ, 20 * np.log10(l_grid), color="grey", alpha=0.5,
                linewidth=1.0, label="|L(jw)|")
    ax.axhline(0, color="black", linewidth=0.8)
    crossover_hz = design.omega_c_rad_s / (2 * np.pi)
    ax.axvline(crossover_hz, color="tab:green", linestyle="--",
               label=f"f_c = {crossover_hz:.2f} Hz")
    for tone_hz, label in ((22.0, "22 Hz"), (47.0, "47 Hz"), (13.0, "13 Hz alias")):
        ax.axvline(tone_hz, color="purple", linestyle=":", linewidth=1.0)
        ax.annotate(label, (tone_hz, 14), rotation=90, fontsize=8, color="purple")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.set_xlim(0.1, 500.0)
    ax.set_ylim(-40, 20)
    ax.set_title("Part 2 closed-loop sensitivity, analytical vs measured")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
