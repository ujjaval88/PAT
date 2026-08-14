"""Part 1: the camera-only fundamental-limit derivation and conclusion.

Produces:
  outputs/part1/sensitivity.png -- S(jw) for the camera-only loop, with f_c,
                                    rejection bandwidth, Ms peak, and the
                                    22/47/13 Hz tones marked
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.sensitivity import (
    achieved_crossover_and_margin,
    camera_extra_delay_s,
    design_camera_only_loop,
    find_rejection_bandwidth_rad_s,
    find_sensitivity_peak,
    open_loop_response,
    sensitivity,
)
from pat_sim.config import DEFAULT_CONFIG

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part1"


def build_camera_only_design(config):
    extra_delay_s = camera_extra_delay_s(
        config.camera, config.part1_design.latency_design_percentile
    )
    total_delay_s = extra_delay_s + config.plant.transport_delay_s
    omega_c_guess = math.radians(config.part1_design.phase_margin_deg) / total_delay_s
    design = design_camera_only_loop(
        config.plant, omega_c_guess, config.part1_design.phase_margin_deg, extra_delay_s
    )
    return design, extra_delay_s


def plot_sensitivity(design, config, out_dir: Path) -> tuple[float, float, float | None]:
    freqs_hz = np.logspace(-1, 3, 6000)
    omega = 2 * np.pi * freqs_hz
    s = sensitivity(open_loop_response(design, config.plant, omega))
    s_db = 20 * np.log10(np.abs(s))

    w_peak, ms = find_sensitivity_peak(design, config.plant, omega)
    f_peak = w_peak / (2 * math.pi)
    f_c = design.omega_c_rad_s / (2 * math.pi)
    w_rej = find_rejection_bandwidth_rad_s(design, config.plant, omega)
    f_rej = w_rej / (2 * math.pi) if w_rej is not None else None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(freqs_hz, s_db, label="|S(jw)|")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-3, color="gray", linestyle=":", linewidth=0.8, label="-3 dB")
    ax.axvline(f_c, color="tab:green", linestyle="--", label=f"f_c = {f_c:.2f} Hz")
    if f_rej is not None:
        ax.axvline(
            f_rej, color="tab:blue", linestyle="--", label=f"rejection BW = {f_rej:.2f} Hz"
        )
    ax.axvline(
        f_peak, color="tab:red", linestyle="--", label=f"Ms peak = {ms:.2f} @ {f_peak:.2f} Hz"
    )
    for f_tone, label in [(22, "22 Hz"), (47, "47 Hz"), (13, "13 Hz alias")]:
        ax.axvline(f_tone, color="purple", linestyle=":", linewidth=1.0)
        ax.annotate(
            label, (f_tone, ax.get_ylim()[1] * 0.9), rotation=90, fontsize=8, color="purple"
        )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|S(jw)| [dB]")
    ax.set_title("Camera-only loop sensitivity (95th-pct latency design, PM=45deg)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=150)
    plt.close(fig)

    return f_peak, ms, f_rej


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG

    design, extra_delay_s = build_camera_only_design(config)
    magnitude, achieved_pm = achieved_crossover_and_margin(design, config.plant)
    f_c = design.omega_c_rad_s / (2 * math.pi)

    print("Camera-only loop design (95th-pct latency, PM target = 45 deg):")
    print(f"  extra delay (ZOH + latency)   = {extra_delay_s * 1e3:.2f} ms")
    print(f"  crossover f_c                 = {f_c:.2f} Hz")
    print(f"  achieved |L(j*wc)|            = {magnitude:.6f} (target 1.0)")
    print(f"  achieved phase margin         = {achieved_pm:.3f} deg (target 45.0)")

    f_peak, ms, f_rej = plot_sensitivity(design, config, OUTPUT_DIR)
    rej_line = f"{f_rej:.2f} Hz" if f_rej else "not found on grid"
    print(f"  rejection bandwidth (-3 dB)   = {rej_line}")
    ms_db = 20 * math.log10(ms)
    print(f"  sensitivity peak Ms           = {ms:.2f} ({ms_db:.1f} dB) @ {f_peak:.2f} Hz")

    print("\n|S(j*2*pi*f)| at the disturbance tones:")
    for f_tone in (22.0, 47.0, 13.0):
        omega = np.array([2 * math.pi * f_tone])
        s = sensitivity(open_loop_response(design, config.plant, omega))[0]
        note = (
            "  (13 Hz is where the real 60 Hz camera sees the 47 Hz tone's alias)"
            if f_tone == 13.0
            else ""
        )
        print(f"  f={f_tone:5.1f} Hz: |S|={abs(s):.3f}  ({20 * math.log10(abs(s)):+.2f} dB){note}")

    print(f"\nPlots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
