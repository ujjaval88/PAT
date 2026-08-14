"""Part 2: design the two-stage lead compensator and verify the design point.

Produces:
  outputs/part2/loop_bode.png -- |L| and phase, with crossover, margins and
                                 both structural modes marked
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.frequency_response import magnitude_db, phase_deg
from pat_sim.analysis.sensitivity import (
    achieved_crossover_and_margin,
    open_loop_response,
)
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.lead_lag_design import (
    PHASE_MARGIN_DEG,
    ROLLOFF_FREQ_HZ,
    build_design,
    design_parameters,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part2"
MODE_FREQS_HZ = (75.0, 220.0)
TONE_FREQS_HZ = (22.0, 47.0)
FREQ_GRID_HZ = np.logspace(-2, 3, 6000)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    design = build_design(config)

    print("Part 2 controller: two-stage lead + rolloff\n")
    for key, value in design_parameters(design).items():
        print(f"  {key:<32} {value}")

    magnitude, phase_margin_deg = achieved_crossover_and_margin(design, config.plant)
    print("\nDesign verification at the design crossover:")
    print(f"  |L(j*w_c)|      {magnitude:.6f}   (target 1.000000)")
    print(f"  phase margin    {phase_margin_deg:.2f} deg   (target {PHASE_MARGIN_DEG:.2f} deg)")

    # Gain stabilisation is the entire reason the rolloff pole exists, so the
    # modal gain margins belong in the design record rather than being taken on
    # trust. |L| < 1 at a mode means stability does not depend on that mode
    # sitting at exactly its nominal frequency.
    print("\nStructural modes (|L| below 1 means the mode cannot destabilise the loop):")
    for freq_hz in MODE_FREQS_HZ:
        loop = abs(open_loop_response(design, config.plant, np.array([2 * math.pi * freq_hz]))[0])
        verdict = "gain-stabilised" if loop < 1.0 else "NOT gain-stabilised"
        print(
            f"  |L({freq_hz:5.0f} Hz)| = {loop:9.5f}  ->  "
            f"{-20 * math.log10(loop):+6.1f} dB   {verdict}"
        )

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
    ax_mag.set_title("Part 2 open-loop response: two-stage lead + rolloff")

    ax_phase.semilogx(FREQ_GRID_HZ, phase_deg(loop_grid), label="phase")
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
    ax_mag.annotate(
        f"PM = {phase_margin_deg:.1f} deg",
        xy=(crossover_hz, 0), xytext=(crossover_hz * 2.5, 25),
        arrowprops={"arrowstyle": "->", "color": "tab:green"}, color="tab:green",
    )
    ax_mag.annotate("modes 75 / 220 Hz", xy=(75.0, -95), color="tab:red", fontsize=8)
    ax_mag.annotate("tones 22 / 47 Hz", xy=(22.0, -110), color="purple", fontsize=8)
    ax_mag.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    path = OUTPUT_DIR / "loop_bode.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot written to {path}")


if __name__ == "__main__":
    main()
