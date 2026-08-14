"""Part 2: the tonals -- how far they drift, and what the loop does
about them.

Produces:
  outputs/part2/tonal_drift.png    -- the instantaneous tone frequencies over
                                      two minutes, against the loop's |S| across
                                      the band each tone actually visits
  outputs/part2/tonal_aliasing.png -- the same LOS seen by the 60 Hz camera,
                                      showing where the 47 Hz tone ends up
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.psd import welch_psd
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.lead_lag_design import (
    build_design,
    sensitivity_magnitudes,
)
from pat_sim.disturbances.platform import PlatformDisturbance
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part2"
DT_S = 50e-6
DRIFT_DURATION_S = 120.0
ALIAS_DURATION_S = 20.0
SETTLE_S = 1.0
SEED = 42
NPERSEG = 65536
FREQ_GRID_HZ = np.logspace(-2, 3, 6000)

# The wander is an Ornstein-Uhlenbeck process with this correlation time, which
# is what makes "slow wander" misleading -- see the diffusion figure below.
WANDER_TAU_S = 20.0


class _ZeroController:
    saturated_last_step = False

    def update(self, error_rad: float) -> float:
        del error_rad
        return 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    platform = config.platform_disturbance
    design = build_design(config)
    crossover_hz = design.omega_c_rad_s / (2 * math.pi)

    # --- 1. how far, and how fast ------------------------------------------
    generator = PlatformDisturbance(platform, DT_S, SEED)
    stride = int(round(0.01 / DT_S))  # 100 Hz is plenty for a slow wander
    times_s: list[float] = []
    freq_22: list[float] = []
    freq_47: list[float] = []
    for step in range(int(DRIFT_DURATION_S / DT_S)):
        generator.broadband.step()
        generator.tone_22hz.step()
        generator.tone_47hz.step()
        if step % stride == 0:
            times_s.append(step * DT_S)
            freq_22.append(generator.tone_22hz.instantaneous_freq_hz())
            freq_47.append(generator.tone_47hz.instantaneous_freq_hz())
    measured = {"22 Hz": np.array(freq_22), "47 Hz": np.array(freq_47)}
    tones = {"22 Hz": platform.tone_22hz, "47 Hz": platform.tone_47hz}

    print(f"Tone drift, measured from the generator over {DRIFT_DURATION_S:.0f} s\n")
    print(f"  {'tone':>8} {'nominal':>9} {'spec':>9} {'measured range':>19} {'std':>8}")
    for label, values in measured.items():
        tone = tones[label]
        print(
            f"  {label:>8} {tone.nominal_freq_hz:8.1f}  +/-{tone.wander_hz:5.1f} "
            f"{values.min():8.2f} - {values.max():6.2f} Hz {values.std():7.2f}"
        )

    # A slow correlation time is not a slowly moving frequency. The diffusion
    # coefficient is what says how quickly the tone leaves any fixed frequency.
    print("\n  diffusion rate (how fast the frequency moves, not how far):")
    for label, tone in tones.items():
        diffusion = tone.wander_hz * math.sqrt(2.0 / WANDER_TAU_S)
        print(f"    {label:>8}  {diffusion:.2f} Hz per root-second")
    print(
        f"\n  The wander is called slow because it takes ~{WANDER_TAU_S:.0f} s to drift"
        " across its range,\n  but the frequency still moves about 1 Hz every second."
        " Anything aimed at a\n  tone frequency would have to keep following it.\n"
    )

    # --- 2. what the loop does across the band each tone visits ------------
    s_grid, _ = sensitivity_magnitudes(design, config, FREQ_GRID_HZ)
    print("What the delivered loop does at the tones, across their whole wander band:")
    print(f"  {'tone':>8} {'band':>18} {'|S| min':>9} {'|S| max':>9} {'|S| at nominal':>16}")
    for label, tone in tones.items():
        low = tone.nominal_freq_hz - tone.wander_hz
        high = tone.nominal_freq_hz + tone.wander_hz
        band = (FREQ_GRID_HZ >= low) & (FREQ_GRID_HZ <= high)
        at_nominal = float(np.interp(tone.nominal_freq_hz, FREQ_GRID_HZ, s_grid))
        print(
            f"  {label:>8} {low:7.1f} - {high:5.1f} Hz {s_grid[band].min():9.3f} "
            f"{s_grid[band].max():9.3f} {at_nominal:16.3f}"
        )
    print(
        f"\n  |S| stays near 1 everywhere in both bands, so it makes no difference"
        " where the\n  tone sits: the loop does not reject it anywhere. Crossover is"
        f" {crossover_hz:.2f} Hz and\n  the tones are at 22 and 47 Hz, far above it."
        " So the drift is not the problem\n  here - the low crossover is, and that"
        " comes from the camera delay.\n"
    )

    # --- 3. what the camera actually sees ----------------------------------
    log = Simulator(
        config,
        dt_s=DT_S,
        seed=SEED,
        controller=_ZeroController(),
        controller_rate_hz=1000.0,
    ).run(duration_s=ALIAS_DURATION_S)
    arrays = log.as_arrays()
    settle = int(SETTLE_S / DT_S)
    true_los = arrays["theta_los_rad"][settle:]
    camera_hold = arrays["controller_input_rad"][settle:]

    freq_true, psd_true = welch_psd(true_los, 1.0 / DT_S, nperseg=NPERSEG)
    freq_camera, psd_camera = welch_psd(camera_hold, 1.0 / DT_S, nperseg=NPERSEG)

    def peak_near(freq_hz: np.ndarray, psd: np.ndarray, centre_hz: float, half_width_hz: float):
        band = (freq_hz >= centre_hz - half_width_hz) & (freq_hz <= centre_hz + half_width_hz)
        index = int(np.argmax(psd[band]))
        return float(freq_hz[band][index]), float(psd[band][index])

    nyquist_hz = config.camera.rate_hz / 2.0
    alias_hz = abs(platform.tone_47hz.nominal_freq_hz - config.camera.rate_hz)
    tone_22_hz = platform.tone_22hz.nominal_freq_hz
    tone_47_hz = platform.tone_47hz.nominal_freq_hz
    print("What the 60 Hz camera reports (open loop, its zero-order-held output):")
    print(f"  camera Nyquist  {nyquist_hz:.1f} Hz\n")
    print(f"  {'band searched':>22} {'true LOS peak':>28} {'camera peak':>28}")
    for label, centre_hz in (
        (f"around {tone_22_hz:.0f} Hz", tone_22_hz),
        (f"around {tone_47_hz:.0f} Hz", tone_47_hz),
        (f"around {alias_hz:.0f} Hz (the alias)", alias_hz),
    ):
        true_peak = peak_near(freq_true, psd_true, centre_hz, 3.0)
        camera_peak = peak_near(freq_camera, psd_camera, centre_hz, 3.0)
        print(
            f"  {label:>22} {true_peak[0]:9.2f} Hz @ {true_peak[1]:.2e} "
            f"{camera_peak[0]:9.2f} Hz @ {camera_peak[1]:.2e}"
        )
    true_at_47 = peak_near(freq_true, psd_true, tone_47_hz, 3.0)
    camera_at_47 = peak_near(freq_camera, psd_camera, tone_47_hz, 3.0)
    camera_at_alias = peak_near(freq_camera, psd_camera, alias_hz, 3.0)
    true_at_alias = peak_near(freq_true, psd_true, alias_hz, 3.0)
    print(
        f"\n  The 22 Hz tone is fine: it is below Nyquist and shows up where it should.\n"
        f"  The 47 Hz tone is not. The camera shows"
        f" {10 * math.log10(camera_at_47[1] / true_at_47[1]):.0f} dB there, but"
        f" {10 * math.log10(camera_at_alias[1] / true_at_alias[1]):+.0f} dB at"
        f" {alias_hz:.0f} Hz,\n  where the real LOS has nothing but broadband."
        " The tone has moved, not shrunk.\n"
    )
    print(
        f"  The controller therefore sees the 47 Hz tone as a {alias_hz:.0f} Hz signal and"
        " cannot tell\n  it apart from a real one at that frequency. Acting on it would"
        " push torque at\n  the wrong frequency. No control law can fix this; only a"
        " faster sensor can.\n"
    )
    print(
        "Summary: this controller does not target the tones at all, so nothing has to\n"
        "track their drift. The cost is that both tones pass straight through, with\n"
        "|S| near 1. Rejecting them needs a higher crossover than the camera delay\n"
        "allows, and for 47 Hz a sensor fast enough not to fold it."
    )

    # --- plots -------------------------------------------------------------
    fig, (ax_drift, ax_sens) = plt.subplots(2, 1, figsize=(9, 7))
    for label, colour in (("22 Hz", "tab:blue"), ("47 Hz", "tab:orange")):
        tone = tones[label]
        ax_drift.plot(times_s, measured[label], linewidth=0.9, color=colour,
                      label=f"{label} tone")
        ax_drift.axhline(tone.nominal_freq_hz, color=colour, linestyle="--", linewidth=0.8)
        ax_drift.fill_between(
            [0, DRIFT_DURATION_S],
            tone.nominal_freq_hz - tone.wander_hz,
            tone.nominal_freq_hz + tone.wander_hz,
            color=colour, alpha=0.10,
        )
    ax_drift.set_xlabel("Time [s]")
    ax_drift.set_ylabel("instantaneous frequency [Hz]")
    ax_drift.set_title("The tones drift continuously; shaded = the specified wander band")
    ax_drift.grid(True, alpha=0.3)
    ax_drift.legend(fontsize=8)

    ax_sens.semilogx(FREQ_GRID_HZ, 20 * np.log10(s_grid), label="|S(jw)| delivered loop")
    ax_sens.axhline(0, color="black", linewidth=0.8)
    ax_sens.axvline(crossover_hz, color="tab:green", linestyle="--",
                    label=f"f_c = {crossover_hz:.2f} Hz")
    for label, colour in (("22 Hz", "tab:blue"), ("47 Hz", "tab:orange")):
        tone = tones[label]
        ax_sens.axvspan(
            tone.nominal_freq_hz - tone.wander_hz,
            tone.nominal_freq_hz + tone.wander_hz,
            color=colour, alpha=0.20,
            label=f"{label} tone, wander band",
        )
    ax_sens.set_xlim(0.1, 200)
    ax_sens.set_ylim(-40, 20)
    ax_sens.set_xlabel("Frequency [Hz]")
    ax_sens.set_ylabel("|S| [dB]")
    ax_sens.set_title("|S| is ~0 dB across both bands: drift is not the obstacle, crossover is")
    ax_sens.grid(True, which="both", alpha=0.3)
    ax_sens.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    drift_path = OUTPUT_DIR / "tonal_drift.png"
    fig.savefig(drift_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(freq_true, psd_true, linewidth=1.0, label="true LOS")
    ax.semilogy(freq_camera, psd_camera, linewidth=1.0, alpha=0.85,
                label="as reported by the 60 Hz camera")
    ax.axvline(nyquist_hz, color="black", linestyle="--", linewidth=1.0,
               label=f"camera Nyquist {nyquist_hz:.0f} Hz")
    ax.axvline(platform.tone_47hz.nominal_freq_hz, color="tab:orange", linestyle=":",
               linewidth=1.2)
    ax.axvline(alias_hz, color="tab:red", linestyle=":", linewidth=1.2)
    ax.annotate(
        f"47 Hz tone reappears here, at {alias_hz:.0f} Hz",
        xy=(alias_hz, peak_near(freq_camera, psd_camera, alias_hz, 3.0)[1]),
        xytext=(alias_hz + 6, peak_near(freq_camera, psd_camera, alias_hz, 3.0)[1] * 6),
        arrowprops={"arrowstyle": "->", "color": "tab:red"}, color="tab:red", fontsize=9,
    )
    band = freq_true <= 100.0
    peak = float(np.max(psd_true[band]))
    ax.set_xlim(0, 100)
    ax.set_ylim(peak / 1e4, peak * 5.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [rad^2/Hz]")
    ax.set_title("The camera does not attenuate the 47 Hz tone -- it relocates it")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    alias_path = OUTPUT_DIR / "tonal_aliasing.png"
    fig.savefig(alias_path, dpi=150)
    plt.close(fig)
    print(f"\nPlots written to {drift_path} and {alias_path}")


if __name__ == "__main__":
    main()
