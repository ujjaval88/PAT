"""Part 2: error PSD before and after, time-domain RMS, and the
actuator-saturation / wind-up behaviour.

Produces:
  outputs/part2/psd_before_after.png -- LOS error PSD, open loop vs closed loop
  outputs/part2/time_domain.png      -- LOS error and commanded torque traces
  outputs/part2/saturation.png       -- torque headroom, and what happens when
                                        the actuator is genuinely saturated

Four runs, because a single closed-loop number cannot be interpreted here:

1. *open loop* on the delivered plant -- the "before".
2. *closed loop* on the delivered plant -- the "after", as actually built.
3. *closed loop, frictionless* -- the same compensator with the Coulomb
   deadband removed.
4. *saturation stress* -- the disturbance scaled up until the torque limit is
   genuinely active, to exercise the saturation path and demonstrate the
   wind-up behaviour rather than asserting it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.psd import rms, welch_psd
from pat_sim.config import DEFAULT_CONFIG, SystemConfig
from pat_sim.control.coarse import CoarseController, ControllerConfig
from pat_sim.control.lead_lag_design import (
    CONTROLLER_RATE_HZ,
    TORQUE_LIMIT_N_M,
    build_design,
)
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part2"
DT_S = 50e-6
DURATION_S = 20.0
SETTLE_S = 1.0
SEED = 42
NPERSEG = 65536
TARGET_RMS_URAD = 50.0
OPEN_LOOP_REFERENCE_URAD = 330.0

# Bands chosen to separate where the loop helps from where it hurts. The first
# edge is near the design's -3 dB rejection bandwidth: lumping that into a
# single 0-5 Hz band would hide the only region where |S| < 1 underneath the
# Ms peak and report one averaged verdict for two opposite effects.
BANDS_HZ = (
    (0.0, 1.5),
    (1.5, 5.0),
    (5.0, 20.0),
    (20.0, 60.0),
    (60.0, 200.0),
    (200.0, 10000.0),
)

# How hard the disturbance has to be pushed before the 0.5 N*m limit is
# genuinely active. Not a physical scenario -- a stress test for the saturation
# path, which is otherwise never exercised. Commanded torque scales linearly
# with the disturbance, and nominal peak demand is ~0.9% of the limit, so it
# takes a factor of this order to reach it at all.
SATURATION_SCALE = 200.0


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


def scaled_disturbance(config: SystemConfig, scale: float) -> SystemConfig:
    platform = config.platform_disturbance
    return dataclasses.replace(
        config,
        platform_disturbance=dataclasses.replace(
            platform,
            broadband=dataclasses.replace(
                platform.broadband, target_rms_rad=platform.broadband.target_rms_rad * scale
            ),
            tone_22hz=dataclasses.replace(
                platform.tone_22hz, amplitude_rad=platform.tone_22hz.amplitude_rad * scale
            ),
            tone_47hz=dataclasses.replace(
                platform.tone_47hz, amplitude_rad=platform.tone_47hz.amplitude_rad * scale
            ),
        ),
    )


def make_controller(design) -> CoarseController:
    return CoarseController(
        ControllerConfig(lead_lag=design, torque_limit_n_m=TORQUE_LIMIT_N_M, resonant=()),
        dt_s=1.0 / CONTROLLER_RATE_HZ,
    )


def run(config: SystemConfig, controller) -> dict[str, np.ndarray]:
    arrays = (
        Simulator(
            config,
            dt_s=DT_S,
            seed=SEED,
            controller=controller,
            controller_rate_hz=CONTROLLER_RATE_HZ,
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
    design = build_design(config)
    coulomb_n_m = config.friction.coulomb_n_m

    open_result = run(config, _ZeroController())
    closed_result = run(config, make_controller(design))
    ideal_result = run(frictionless(config), make_controller(design))

    open_urad = rms(open_result["theta_los_rad"]) * 1e6
    closed_urad = rms(closed_result["theta_los_rad"]) * 1e6
    ideal_urad = rms(ideal_result["theta_los_rad"]) * 1e6
    peak_torque = float(np.max(np.abs(closed_result["motor_torque_n_m"])))

    print("Time-domain RMS (LOS error at the camera)\n")
    print(f"  {'configuration':<34} {'RMS [urad]':>11} {'vs open':>9}")
    for label, value in (
        ("open loop (delivered plant)", open_urad),
        ("closed loop (delivered plant)", closed_urad),
        ("closed loop (frictionless)", ideal_urad),
    ):
        delta = "" if value == open_urad else f"{value - open_urad:+8.1f}"
        print(f"  {label:<34} {value:11.1f} {delta:>9}")
    print(
        f"\n  target {TARGET_RMS_URAD:.0f} urad, from a nominal "
        f"~{OPEN_LOOP_REFERENCE_URAD:.0f} urad open loop"
    )
    print(f"  shortfall: {closed_urad / TARGET_RMS_URAD:.1f}x over target")

    print("\nWhy the delivered and frictionless runs disagree:")
    print(f"  peak commanded torque      {peak_torque * 1e3:8.3f} mN*m")
    print(f"  Coulomb breakaway          {coulomb_n_m * 1e3:8.3f} mN*m")
    print(f"  gimbal motion (theta_g)    {rms(closed_result['theta_g_rad']) * 1e6:8.2f} urad RMS")
    if peak_torque < coulomb_n_m:
        print("  -> Torque never gets past friction, so the gimbal barely moves.")
        print("     Closed loop matches open loop because of friction, not control.")
        print("     Judge the design by the frictionless run instead.")
    else:
        print("  -> Torque gets past friction, so both runs are real control results.")

    freq_open, psd_open = welch_psd(open_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)
    freq_closed, psd_closed = welch_psd(closed_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_ideal = welch_psd(ideal_result["theta_los_rad"], 1.0 / DT_S, nperseg=NPERSEG)

    print("\nError PSD by band, before and after (RMS contribution, urad)\n")
    print(f"  {'band [Hz]':>16} {'open':>9} {'closed':>9} {'frictionless':>13} {'ratio':>7}")
    for low_hz, high_hz in BANDS_HZ:
        before = band_rms_urad(freq_open, psd_open, low_hz, high_hz)
        after = band_rms_urad(freq_closed, psd_closed, low_hz, high_hz)
        ideal = band_rms_urad(freq_open, psd_ideal, low_hz, high_hz)
        ratio = ideal / before if before > 0 else float("nan")
        print(
            f"  {low_hz:7.1f} - {high_hz:6.1f} {before:9.1f} {after:9.1f} "
            f"{ideal:13.1f} {ratio:7.2f}"
        )
    print("\n  'ratio' is frictionless closed loop divided by open loop.")
    print("  Below 1 means the loop helps in that band, above 1 means it hurts.")

    # --- saturation and wind-up -------------------------------------------
    stress_config = scaled_disturbance(config, SATURATION_SCALE)
    stress_result = run(stress_config, make_controller(design))
    saturated_fraction = 100.0 * float(np.mean(stress_result["controller_saturated"]))
    nominal_saturated = 100.0 * float(np.mean(closed_result["controller_saturated"]))

    print("\nActuator saturation and integrator wind-up\n")
    print(f"  nominal disturbance: controller saturated {nominal_saturated:.3f}% of ticks,")
    print(
        f"    peak torque {peak_torque * 1e3:.3f} mN*m = "
        f"{100 * peak_torque / TORQUE_LIMIT_N_M:.2f}% of the {TORQUE_LIMIT_N_M} N*m limit"
    )
    print(
        f"  disturbance x{SATURATION_SCALE:.0f}: controller saturated "
        f"{saturated_fraction:.1f}% of ticks,"
    )
    print(
        f"    LOS RMS {rms(stress_result['theta_los_rad']) * 1e6:.0f} urad, "
        f"peak torque {float(np.max(np.abs(stress_result['motor_torque_n_m']))) * 1e3:.1f} mN*m"
    )
    # The no-wind-up claim is structural, so it is checked structurally rather
    # than inferred from a trace: wind-up requires a state that integrates the
    # error, i.e. a compensator pole at s = 0. This one has none.
    lead_pole_rad_s = design.lead_lag_alpha / design.lead_time_constant_s
    rolloff_pole_rad_s = design.rolloff_freq_rad_s or float("inf")
    poles_rad_s = [lead_pole_rad_s] * design.n_stages + [rolloff_pole_rad_s]
    print("\n  Compensator poles (continuous, rad/s):")
    for pole in poles_rad_s:
        print(f"    s = {-pole:+.3f}   ({pole / (2 * np.pi):.2f} Hz)")
    print("  None of these poles is at zero, so there is no integrator.")
    print("\n  No integrator means nothing can wind up: there is no state that keeps")
    print("  building while the torque is stuck at the limit. Once the demand drops")
    print("  back inside the limit the output is correct again on the next tick.")
    print("  Saturation is applied at the controller output.")
    if saturated_fraction <= 0.0:
        print(
            f"\n  WARNING: the x{SATURATION_SCALE:.0f} case never reached the limit, so"
            " saturation was\n  not actually tested. Raise SATURATION_SCALE."
        )

    # --- plots -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    # Open loop is drawn thick and dashed underneath: on the delivered plant it
    # is IDENTICAL to the closed loop, and a thin line would simply vanish under
    # it and read as a missing trace rather than as the result it is.
    ax.semilogy(freq_open, psd_open, linewidth=3.0, alpha=0.45, color="tab:blue",
                label=f"open loop ({open_urad:.0f} urad RMS)")
    ax.semilogy(freq_closed, psd_closed, linewidth=1.0, color="tab:orange",
                label=f"closed loop, delivered ({closed_urad:.0f} urad RMS)")
    ax.semilogy(freq_open, psd_ideal, linewidth=1.0, color="tab:green",
                label=f"closed loop, frictionless ({ideal_urad:.0f} urad RMS)")
    crossover_hz = design.omega_c_rad_s / (2 * np.pi)
    ax.axvline(crossover_hz, color="tab:green", linestyle="--", linewidth=1.0,
               label=f"f_c = {crossover_hz:.2f} Hz")
    for tone_hz in (13.0, 22.0, 47.0):
        ax.axvline(tone_hz, color="purple", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 100)
    # Two decades below the peak. Left to autoscale, the near-null DC bin drags
    # the axis to 1e-25 and the whole comparison collapses into the top eighth.
    in_band = freq_open <= 100.0
    peak = float(np.max(psd_ideal[in_band]))
    ax.set_ylim(peak / 100.0, peak * 3.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [rad^2/Hz]")
    ax.set_title("Part 2: LOS error PSD, before and after")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    psd_path = OUTPUT_DIR / "psd_before_after.png"
    fig.savefig(psd_path, dpi=150)
    plt.close(fig)

    window = slice(0, int(2.0 / DT_S))
    fig, (ax_error, ax_torque) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_error.plot(open_result["time_s"][window], open_result["theta_los_rad"][window] * 1e6,
                  linewidth=0.8, label=f"open loop ({open_urad:.0f} urad)")
    ax_error.plot(closed_result["time_s"][window], closed_result["theta_los_rad"][window] * 1e6,
                  linewidth=0.8, label=f"closed loop ({closed_urad:.0f} urad)")
    ax_error.axhline(TARGET_RMS_URAD, color="tab:red", linestyle="--", linewidth=0.8,
                     label=f"{TARGET_RMS_URAD:.0f} urad target")
    ax_error.axhline(-TARGET_RMS_URAD, color="tab:red", linestyle="--", linewidth=0.8)
    ax_error.set_ylabel("LOS error [urad]")
    ax_error.set_title("Part 2 time domain: LOS error and commanded torque")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(fontsize=8)

    ax_torque.plot(closed_result["time_s"][window],
                   closed_result["motor_torque_n_m"][window] * 1e3, linewidth=0.8,
                   label="commanded torque")
    ax_torque.axhline(coulomb_n_m * 1e3, color="tab:red", linestyle="--", linewidth=0.9,
                      label=f"Coulomb breakaway {coulomb_n_m * 1e3:.0f} mN*m")
    ax_torque.axhline(-coulomb_n_m * 1e3, color="tab:red", linestyle="--", linewidth=0.9)
    ax_torque.set_xlabel("Time [s]")
    ax_torque.set_ylabel("torque [mN*m]")
    ax_torque.grid(True, alpha=0.3)
    ax_torque.legend(fontsize=8)
    fig.tight_layout()
    time_path = OUTPUT_DIR / "time_domain.png"
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    stress_window = slice(0, int(1.0 / DT_S))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(stress_result["time_s"][stress_window],
            stress_result["motor_torque_n_m"][stress_window] * 1e3, linewidth=0.8,
            label=f"delivered torque, disturbance x{SATURATION_SCALE:.0f}")
    ax.axhline(TORQUE_LIMIT_N_M * 1e3, color="tab:red", linestyle="--",
               label=f"limit +/-{TORQUE_LIMIT_N_M * 1e3:.0f} mN*m")
    ax.axhline(-TORQUE_LIMIT_N_M * 1e3, color="tab:red", linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("torque [mN*m]")
    ax.set_title(
        f"Saturation stress test: limit active {saturated_fraction:.1f}% of ticks, "
        "no wind-up (no integrator)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saturation_path = OUTPUT_DIR / "saturation.png"
    fig.savefig(saturation_path, dpi=150)
    plt.close(fig)

    print(f"\nPlots written to {psd_path}, {time_path}, {saturation_path}")


if __name__ == "__main__":
    main()
