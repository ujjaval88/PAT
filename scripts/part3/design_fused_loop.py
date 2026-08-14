"""Part 3: design the lead compensator for the fused loop, and show why the
crossover ends up where it does.

Produces:
  outputs/part3/fused_loop_bode.png   -- |L| and phase for the delivered design
  outputs/part3/crossover_sweep.png   -- measured LOS error vs crossover, against
                                         the loop gain at the 75 Hz mode
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.frequency_response import magnitude_db, phase_deg
from pat_sim.analysis.psd import rms
from pat_sim.analysis.sensitivity import (
    achieved_crossover_and_margin,
    find_sensitivity_peak,
    open_loop_response,
)
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.coarse import CoarseController, ControllerConfig
from pat_sim.control.fused_design import (
    CONTROLLER_RATE_HZ,
    CROSSOVER_HZ,
    EXTRA_DELAY_S,
    GYRO_DT_S,
    PHASE_MARGIN_DEG,
    ROLLOFF_FREQ_HZ,
    TORQUE_LIMIT_N_M,
    build_design,
    design_parameters,
)
from pat_sim.estimation.estimator import KalmanFusionEstimator
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part3"
DT_S = 50e-6
DURATION_S = 10.0
SETTLE_S = 1.0
SEED = 42
MODE_FREQS_HZ = (75.0, 220.0)
TONE_FREQS_HZ = (22.0, 47.0)
FREQ_GRID_HZ = np.logspace(-2, 3, 6000)

# Crossover / rolloff pairs to try. The rolloff has to stay above crossover to
# be useful and below the 75 Hz mode to gain-stabilise it, so it is moved up
# with the crossover rather than held fixed.
SWEEP = (
    (1.0, 5.0),
    (2.0, 10.0),
    (3.0, 15.0),
    (5.0, 15.0),
    (5.0, 25.0),
    (8.0, 25.0),
    (10.0, 30.0),
    (15.0, 45.0),
    (20.0, 60.0),
)


class _ZeroController:
    saturated_last_step = False

    def update(self, error_rad: float) -> float:
        del error_rad
        return 0.0


def frictionless(config):
    return dataclasses.replace(
        config,
        friction=dataclasses.replace(
            config.friction, coulomb_n_m=0.0, viscous_n_m_s_per_rad=0.0
        ),
        cogging=dataclasses.replace(config.cogging, amplitude_n_m=0.0),
    )


def run_closed_loop(config, design) -> dict[str, np.ndarray]:
    controller = (
        _ZeroController()
        if design is None
        else CoarseController(
            ControllerConfig(lead_lag=design, torque_limit_n_m=TORQUE_LIMIT_N_M, resonant=()),
            dt_s=1.0 / CONTROLLER_RATE_HZ,
        )
    )
    estimator = KalmanFusionEstimator(config.gyro, config.camera, gyro_dt_s=GYRO_DT_S)
    arrays = (
        Simulator(
            config,
            dt_s=DT_S,
            seed=SEED,
            controller=controller,
            controller_rate_hz=CONTROLLER_RATE_HZ,
            use_gyro=True,
            estimator=estimator,
        )
        .run(duration_s=DURATION_S)
        .as_arrays()
    )
    settle = int(SETTLE_S / DT_S)
    return {key: value[settle:] for key, value in arrays.items()}


def loop_gain_at(design, config, freq_hz: float) -> float:
    return abs(open_loop_response(design, config.plant, np.array([2 * math.pi * freq_hz]))[0])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    design = build_design(config)

    print("Part 3 controller: two-stage lead + rolloff on the fused estimate\n")
    for key, value in design_parameters(design).items():
        print(f"  {key:<24} {value}")

    magnitude, phase_margin_deg = achieved_crossover_and_margin(design, config.plant)
    print("\nDesign verification at the design crossover:")
    print(f"  |L(j*w_c)|      {magnitude:.6f}   (target 1.000000)")
    print(f"  phase margin    {phase_margin_deg:.2f} deg   (target {PHASE_MARGIN_DEG:.2f} deg)")

    print("\nStructural modes (|L| below 1 means the mode cannot destabilise the loop):")
    for freq_hz in MODE_FREQS_HZ:
        loop = loop_gain_at(design, config, freq_hz)
        verdict = "safe" if loop < 1.0 else "NOT safe"
        print(f"  |L({freq_hz:5.0f} Hz)| = {loop:9.5f}  ->  {verdict}")

    delay_limit_hz = math.radians(PHASE_MARGIN_DEG) / EXTRA_DELAY_S / (2 * math.pi)
    print(f"\n  On delay alone this loop could cross over at about {delay_limit_hz:.0f} Hz.")
    print("  The sweep below shows what happens if you try.\n")

    # --- crossover sweep ---------------------------------------------------
    #
    # Both plants, because on the delivered plant a low-crossover loop never
    # commands past the friction level, so several crossovers come out with
    # exactly the open-loop error and the table cannot tell them apart. The
    # frictionless column is where the design can actually be seen acting.
    open_urad = rms(run_closed_loop(config, None)["theta_los_rad"]) * 1e6
    quiet_config = frictionless(config)
    print(f"Crossover sweep. Open loop is {open_urad:.1f} urad for reference.\n")
    print(
        f"  {'f_c [Hz]':>9} {'rolloff':>9} {'delivered':>10} {'frictionless':>13} "
        f"{'|L| at 75 Hz':>13} {'Ms':>7}"
    )
    sweep_crossovers: list[float] = []
    sweep_delivered: list[float] = []
    sweep_ideal: list[float] = []
    sweep_mode_gain: list[float] = []
    for crossover_hz, rolloff_hz in SWEEP:
        candidate = build_design(config, crossover_hz, rolloff_hz)
        delivered_urad = rms(run_closed_loop(config, candidate)["theta_los_rad"]) * 1e6
        ideal_urad = rms(run_closed_loop(quiet_config, candidate)["theta_los_rad"]) * 1e6
        mode_gain = loop_gain_at(candidate, config, MODE_FREQS_HZ[0])
        _, ms = find_sensitivity_peak(candidate, config.plant, 2 * np.pi * FREQ_GRID_HZ)
        sweep_crossovers.append(crossover_hz)
        sweep_delivered.append(delivered_urad)
        sweep_ideal.append(ideal_urad)
        sweep_mode_gain.append(mode_gain)
        print(
            f"  {crossover_hz:9.0f} {rolloff_hz:7.0f} Hz {delivered_urad:10.1f} "
            f"{ideal_urad:13.1f} {mode_gain:13.2f} {ms:7.2f}"
        )

    print(
        "\n  Read the frictionless column. Error grows with crossover everywhere, and\n"
        "  no setting beats open loop. The loop gain at the 75 Hz mode passes 1 near\n"
        "  8 Hz, and above that the error climbs sharply as the mode stops being safe.\n"
    )
    print(
        "  The delivered plant column is flat at the open-loop value for the low\n"
        "  crossovers. That is friction, not success: those loops never command past\n"
        "  the friction level, so the gimbal barely moves and closed loop matches\n"
        "  open loop. It is the reason the frictionless column is needed to choose.\n"
    )
    print(
        f"  So the delivered design uses {CROSSOVER_HZ:.0f} Hz, not the"
        f" {delay_limit_hz:.0f} Hz the delay would allow.\n"
        "  Fusion removed the delay limit, and the structural mode took its place as\n"
        "  the cap. Crossover is kept low because pushing it only makes things worse.\n"
    )

    # --- plots --------------------------------------------------------------
    loop_grid = open_loop_response(design, config.plant, 2 * np.pi * FREQ_GRID_HZ)
    crossover_hz = design.omega_c_rad_s / (2 * math.pi)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_mag.semilogx(FREQ_GRID_HZ, magnitude_db(loop_grid), label="|L(jw)|")
    ax_mag.axhline(0, color="black", linewidth=0.8)
    ax_mag.axvline(crossover_hz, color="tab:green", linestyle="--",
                   label=f"f_c = {crossover_hz:.2f} Hz")
    ax_mag.axvline(ROLLOFF_FREQ_HZ, color="tab:orange", linestyle="-.",
                   label=f"rolloff pole {ROLLOFF_FREQ_HZ:.0f} Hz")
    ax_mag.set_ylabel("|L| [dB]")
    ax_mag.set_ylim(-120, 60)
    ax_mag.set_title("Part 3 open-loop response, fused estimate")

    ax_phase.semilogx(FREQ_GRID_HZ, phase_deg(loop_grid))
    ax_phase.axhline(-180, color="black", linewidth=0.8, linestyle=":")
    ax_phase.axvline(crossover_hz, color="tab:green", linestyle="--")
    ax_phase.set_ylabel("phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")

    for axis in (ax_mag, ax_phase):
        for freq_hz in MODE_FREQS_HZ:
            axis.axvline(freq_hz, color="tab:red", linestyle=":", linewidth=1.0)
        for freq_hz in TONE_FREQS_HZ:
            axis.axvline(freq_hz, color="purple", linestyle=":", linewidth=1.0)
        axis.grid(True, which="both", alpha=0.3)
    ax_mag.annotate(f"PM = {phase_margin_deg:.1f} deg", xy=(crossover_hz, 0),
                    xytext=(crossover_hz * 2.5, 25),
                    arrowprops={"arrowstyle": "->", "color": "tab:green"}, color="tab:green")
    ax_mag.annotate("modes 75 / 220 Hz", xy=(75.0, -95), color="tab:red", fontsize=8)
    ax_mag.annotate("tones 22 / 47 Hz", xy=(22.0, -110), color="purple", fontsize=8)
    ax_mag.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    bode_path = OUTPUT_DIR / "fused_loop_bode.png"
    fig.savefig(bode_path, dpi=150)
    plt.close(fig)

    fig, ax_rms = plt.subplots(figsize=(9, 5))
    ax_rms.plot(sweep_crossovers, sweep_ideal, "o-", color="tab:blue",
                label="LOS error, frictionless")
    ax_rms.set_xlabel("crossover [Hz]")
    ax_rms.set_ylabel("LOS error [urad RMS]", color="tab:blue")
    ax_rms.tick_params(axis="y", labelcolor="tab:blue")
    ax_rms.axvline(CROSSOVER_HZ, color="tab:green", linestyle="--",
                   label=f"delivered {CROSSOVER_HZ:.0f} Hz")
    ax_rms.grid(True, alpha=0.3)

    ax_mode = ax_rms.twinx()
    ax_mode.plot(sweep_crossovers, sweep_mode_gain, "s--", color="tab:red",
                 label="|L| at the 75 Hz mode")
    ax_mode.axhline(1.0, color="tab:red", linestyle=":", linewidth=1.0)
    ax_mode.set_ylabel("|L| at 75 Hz", color="tab:red")
    ax_mode.set_yscale("log")
    ax_mode.tick_params(axis="y", labelcolor="tab:red")

    lines = ax_rms.get_lines() + ax_mode.get_lines()
    ax_rms.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper left")
    ax_rms.set_title("Pushing the crossover makes it worse, once the mode stops being safe")
    fig.tight_layout()
    sweep_path = OUTPUT_DIR / "crossover_sweep.png"
    fig.savefig(sweep_path, dpi=150)
    plt.close(fig)
    print(f"Plots written to {bode_path} and {sweep_path}")


if __name__ == "__main__":
    main()
