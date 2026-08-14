"""Part 3: closed-loop results, before and after fusion.

Produces:
  outputs/part3/psd_before_after.png -- LOS error PSD, open loop vs camera-only
                                        vs fused
  outputs/part3/time_domain.png      -- LOS error and commanded torque traces

Four runs:

1. open loop, delivered plant - the "before".
2. camera-only lead-lag, the Part 2 controller, for reference.
3. fused lead-lag, delivered plant - the "after".
4. fused lead-lag, frictionless - the same controller with the friction
   deadband removed, so the designed loop can be seen acting.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.psd import rms, welch_psd
from pat_sim.config import DEFAULT_CONFIG, SystemConfig
from pat_sim.control.coarse import CoarseController, ControllerConfig
from pat_sim.control.fused_design import (
    CONTROLLER_RATE_HZ,
    GYRO_DT_S,
    TORQUE_LIMIT_N_M,
)
from pat_sim.control.fused_design import build_design as build_fused_design
from pat_sim.control.lead_lag_design import build_design as build_camera_design
from pat_sim.estimation.estimator import KalmanFusionEstimator
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part3"
DT_S = 50e-6
DURATION_S = 20.0
SETTLE_S = 1.0
SEED = 42
NPERSEG = 65536
TARGET_RMS_URAD = 15.0
PLOT_WINDOW_S = 2.0

BANDS_HZ = (
    (0.0, 1.5),
    (1.5, 5.0),
    (5.0, 20.0),
    (20.0, 60.0),
    (60.0, 200.0),
    (200.0, 10000.0),
)


class _ZeroController:
    saturated_last_step = False

    def update(self, error_rad: float) -> float:
        del error_rad
        return 0.0


def frictionless(config: SystemConfig) -> SystemConfig:
    return dataclasses.replace(
        config,
        friction=dataclasses.replace(
            config.friction, coulomb_n_m=0.0, viscous_n_m_s_per_rad=0.0
        ),
        cogging=dataclasses.replace(config.cogging, amplitude_n_m=0.0),
    )


def make_controller(design) -> CoarseController:
    return CoarseController(
        ControllerConfig(lead_lag=design, torque_limit_n_m=TORQUE_LIMIT_N_M, resonant=()),
        dt_s=1.0 / CONTROLLER_RATE_HZ,
    )


def run(config: SystemConfig, controller, use_fusion: bool) -> dict[str, np.ndarray]:
    estimator = (
        KalmanFusionEstimator(config.gyro, config.camera, gyro_dt_s=GYRO_DT_S)
        if use_fusion
        else None
    )
    arrays = (
        Simulator(
            config,
            dt_s=DT_S,
            seed=SEED,
            controller=controller,
            controller_rate_hz=CONTROLLER_RATE_HZ,
            use_gyro=use_fusion,
            estimator=estimator,
        )
        .run(duration_s=DURATION_S)
        .as_arrays()
    )
    settle = int(SETTLE_S / DT_S)
    return {key: value[settle:] for key, value in arrays.items()}


def band_rms_urad(freq_hz: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freq_hz >= low_hz) & (freq_hz < high_hz)
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.trapezoid(psd[mask], freq_hz[mask]))) * 1e6


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    coulomb_n_m = config.friction.coulomb_n_m

    open_result = run(config, _ZeroController(), use_fusion=False)
    camera_result = run(config, make_controller(build_camera_design(config)), use_fusion=False)
    fused_result = run(config, make_controller(build_fused_design(config)), use_fusion=True)
    ideal_result = run(
        frictionless(config), make_controller(build_fused_design(config)), use_fusion=True
    )

    open_urad = rms(open_result["theta_los_rad"]) * 1e6
    camera_urad = rms(camera_result["theta_los_rad"]) * 1e6
    fused_urad = rms(fused_result["theta_los_rad"]) * 1e6
    ideal_urad = rms(ideal_result["theta_los_rad"]) * 1e6
    peak_torque = float(np.max(np.abs(fused_result["motor_torque_n_m"])))

    print("LOS error at the camera, before and after fusion\n")
    print(f"  {'configuration':<38} {'RMS [urad]':>11} {'vs open':>9}")
    for label, value in (
        ("open loop", open_urad),
        ("camera-only lead-lag (Part 2)", camera_urad),
        ("fused lead-lag (Part 3)", fused_urad),
        ("fused lead-lag, frictionless", ideal_urad),
    ):
        delta = "" if value == open_urad else f"{value - open_urad:+8.1f}"
        print(f"  {label:<38} {value:11.1f} {delta:>9}")
    print(f"\n  target {TARGET_RMS_URAD:.0f} urad")
    print(f"  shortfall: {fused_urad / TARGET_RMS_URAD:.0f}x over target")

    print("\nWhy the delivered and frictionless runs differ:")
    print(f"  peak commanded torque   {peak_torque * 1e3:8.3f} mN*m")
    print(f"  friction level          {coulomb_n_m * 1e3:8.3f} mN*m")
    print(f"  gimbal motion           {rms(fused_result['theta_g_rad']) * 1e6:8.2f} urad RMS")
    if peak_torque < coulomb_n_m:
        print("  -> Torque never gets past friction, so the gimbal barely moves.")
        print("     Closed loop matches open loop because of friction, not control.")
        print("     Judge the design by the frictionless run instead.")
    else:
        print("  -> Torque gets past friction, so both runs are real control results.")

    freq_hz, psd_open = welch_psd(open_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_camera = welch_psd(camera_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_fused = welch_psd(fused_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_ideal = welch_psd(ideal_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)

    print("\nError PSD by band (RMS contribution, urad)\n")
    print(f"  {'band [Hz]':>16} {'open':>9} {'fused':>9} {'frictionless':>13} {'ratio':>7}")
    for low_hz, high_hz in BANDS_HZ:
        before = band_rms_urad(freq_hz, psd_open, low_hz, high_hz)
        after = band_rms_urad(freq_hz, psd_fused, low_hz, high_hz)
        ideal = band_rms_urad(freq_hz, psd_ideal, low_hz, high_hz)
        ratio = ideal / before if before > 0 else float("nan")
        print(
            f"  {low_hz:7.1f} - {high_hz:6.1f} {before:9.1f} {after:9.1f} "
            f"{ideal:13.1f} {ratio:7.2f}"
        )
    print("\n  'ratio' is frictionless closed loop divided by open loop.")
    print("  Below 1 means the loop helps in that band, above 1 means it hurts.")

    print(
        "\nWhat fusion did and did not buy:\n"
        "  It fixed what the loop can SEE. The fused estimate tracks the true LOS\n"
        "  about twice as well as the camera alone, and both tones are now visible\n"
        "  at their real frequencies instead of folded.\n"
        "  It did not change what the loop can REJECT. The disturbance still sits\n"
        "  mostly above crossover, and crossover is now capped by the 75 Hz mode\n"
        "  rather than by camera delay. Better information does not by itself move\n"
        "  the bandwidth."
    )

    # --- plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(freq_hz, psd_open, linewidth=3.0, alpha=0.45, color="tab:blue",
                label=f"open loop ({open_urad:.0f} urad)")
    ax.semilogy(freq_hz, psd_camera, linewidth=1.0, color="tab:grey",
                label=f"camera-only lead-lag ({camera_urad:.0f} urad)")
    ax.semilogy(freq_hz, psd_fused, linewidth=1.0, color="tab:orange",
                label=f"fused lead-lag ({fused_urad:.0f} urad)")
    ax.semilogy(freq_hz, psd_ideal, linewidth=1.0, color="tab:green",
                label=f"fused, frictionless ({ideal_urad:.0f} urad)")
    for tone_hz in (22.0, 47.0):
        ax.axvline(tone_hz, color="purple", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 100)
    band = freq_hz <= 100.0
    peak = float(np.max(psd_ideal[band]))
    ax.set_ylim(peak / 100.0, peak * 3.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [rad^2/Hz]")
    ax.set_title("Part 3: LOS error PSD, before and after fusion")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    psd_path = OUTPUT_DIR / "psd_before_after.png"
    fig.savefig(psd_path, dpi=150)
    plt.close(fig)

    window = slice(0, int(PLOT_WINDOW_S / DT_S))
    fig, (ax_error, ax_torque) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_error.plot(open_result["time_s"][window], open_result["theta_los_rad"][window] * 1e6,
                  linewidth=0.8, label=f"open loop ({open_urad:.0f} urad)")
    ax_error.plot(fused_result["time_s"][window], fused_result["theta_los_rad"][window] * 1e6,
                  linewidth=0.8, label=f"fused lead-lag ({fused_urad:.0f} urad)")
    ax_error.axhline(TARGET_RMS_URAD, color="tab:red", linestyle="--", linewidth=0.8,
                     label=f"{TARGET_RMS_URAD:.0f} urad target")
    ax_error.axhline(-TARGET_RMS_URAD, color="tab:red", linestyle="--", linewidth=0.8)
    ax_error.set_ylabel("LOS error [urad]")
    ax_error.set_title("Part 3 time domain: LOS error and commanded torque")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(fontsize=8)

    ax_torque.plot(fused_result["time_s"][window],
                   fused_result["motor_torque_n_m"][window] * 1e3, linewidth=0.8,
                   label="commanded torque")
    ax_torque.axhline(coulomb_n_m * 1e3, color="tab:red", linestyle="--", linewidth=0.9,
                      label=f"friction level {coulomb_n_m * 1e3:.0f} mN*m")
    ax_torque.axhline(-coulomb_n_m * 1e3, color="tab:red", linestyle="--", linewidth=0.9)
    ax_torque.set_xlabel("Time [s]")
    ax_torque.set_ylabel("torque [mN*m]")
    ax_torque.grid(True, alpha=0.3)
    ax_torque.legend(fontsize=8)
    fig.tight_layout()
    time_path = OUTPUT_DIR / "time_domain.png"
    fig.savefig(time_path, dpi=150)
    plt.close(fig)
    print(f"\nPlots written to {psd_path} and {time_path}")


if __name__ == "__main__":
    main()
