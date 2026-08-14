"""Part 1: plant characterization -- Bode and pole-zero of the nominal open-loop plant.

Produces:
  outputs/part1/bode.png      -- open-loop plant Bode (magnitude + phase)
  outputs/part1/pole_zero.png -- pole map of the rational part of the plant
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.frequency_response import (
    magnitude_db,
    phase_deg,
    plant_frequency_response,
    plant_poles,
)
from pat_sim.config import DEFAULT_CONFIG

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part1"


def plot_bode(plant_config, out_dir: Path) -> None:
    freqs_hz = np.logspace(-1, 3.3, 4000)
    omega = 2 * np.pi * freqs_hz
    resp = plant_frequency_response(omega, plant_config)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_mag.semilogx(freqs_hz, magnitude_db(resp))
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Open-loop plant P(s): rigid body x mode1 x mode2 x actuator x delay")
    ax_mag.grid(True, which="both", alpha=0.3)
    markers = [(75, "mode 1 (75 Hz)"), (220, "mode 2 (220 Hz)"), (795.8, "actuator (796 Hz)")]
    for f, label in markers:
        ax_mag.axvline(f, color="gray", linestyle="--", linewidth=0.8)
        ax_mag.annotate(label, (f, ax_mag.get_ylim()[1]), rotation=90, va="top", fontsize=8)

    ax_phase.semilogx(freqs_hz, phase_deg(resp))
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "bode.png", dpi=150)
    plt.close(fig)


def plot_pole_zero(plant_config, out_dir: Path) -> None:
    poles = plant_poles(plant_config)
    fig, ax = plt.subplots(figsize=(6, 6))
    re = [p.real for p in poles]
    im = [p.imag for p in poles]
    ax.scatter(re, im, marker="x", s=80, color="tab:red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Real [rad/s]")
    ax.set_ylabel("Imag [rad/s]")
    ax.set_title("Pole map: rational part of P(s)\n(delay excluded -- contributes phase only)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pole_zero.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG

    plot_bode(config.plant, OUTPUT_DIR)
    plot_pole_zero(config.plant, OUTPUT_DIR)

    print("Rigid body: double pole at s=0.")
    for p in plant_poles(config.plant)[2:]:
        print(f"  pole: {p:.2f}")
    print(f"\nPlots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
