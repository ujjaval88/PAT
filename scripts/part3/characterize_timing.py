"""Part 3: sensor timing characterisation.

Produces:
  outputs/part3/timing_budget.png -- delay and Nyquist for each sensor, and the
                                     crossover each delay budget permits
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.sensitivity import (
    camera_latency_percentile_s,
    camera_zoh_delay_s,
)
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.fused_design import EXTRA_DELAY_S, PHASE_MARGIN_DEG

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part3"
LATENCY_PERCENTILE = 0.90
TONE_FREQS_HZ = (22.0, 47.0)
MODE_FREQ_HZ = 75.0


def delay_limited_crossover_hz(total_delay_s: float) -> float:
    """A pure delay T costs w*T of phase, so demanding a phase margin caps the
    crossover at that many radians' worth of delay."""
    return math.radians(PHASE_MARGIN_DEG) / total_delay_s / (2.0 * math.pi)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG

    camera_zoh_s = camera_zoh_delay_s(config.camera)
    camera_latency_s = camera_latency_percentile_s(config.camera, LATENCY_PERCENTILE)
    camera_total_s = camera_zoh_s + camera_latency_s
    gyro_zoh_s = 0.5 / config.gyro.rate_hz
    gyro_total_s = EXTRA_DELAY_S

    print("Sensor timing\n")
    header = f"  {'sensor':>10} {'rate':>10} {'Nyquist':>10} {'delay':>12}"
    print(f"{header} {'crossover it allows':>22}")
    for name, rate_hz, nyquist_hz, delay_s in (
        ("camera", config.camera.rate_hz, config.camera.rate_hz / 2, camera_total_s),
        ("gyro", config.gyro.rate_hz, config.gyro.rate_hz / 2, gyro_total_s),
        ("encoder", config.encoder.rate_hz, config.encoder.rate_hz / 2, config.encoder.latency_s),
    ):
        crossover = delay_limited_crossover_hz(delay_s) if delay_s > 0 else float("inf")
        crossover_text = "no delay limit" if delay_s <= 0 else f"{crossover:.0f} Hz"
        print(
            f"  {name:>10} {rate_hz:8.0f} Hz {nyquist_hz:8.0f} Hz "
            f"{delay_s * 1e3:9.2f} ms {crossover_text:>22}"
        )

    print("\nCamera delay breakdown:")
    print(f"  zero-order hold (half a frame)  {camera_zoh_s * 1e3:6.2f} ms")
    print(
        f"  latency, {LATENCY_PERCENTILE * 100:.0f}th percentile      "
        f"{camera_latency_s * 1e3:6.2f} ms"
    )
    print(f"  total                           {camera_total_s * 1e3:6.2f} ms")
    print("\nGyro delay breakdown:")
    print(f"  zero-order hold (half a sample) {gyro_zoh_s * 1e3:6.2f} ms")
    print(f"  latency                         {config.gyro.latency_s * 1e3:6.2f} ms")
    print(f"  one controller tick             {1e3 / 1000.0:6.2f} ms")
    print(f"  total                           {gyro_total_s * 1e3:6.2f} ms")
    print(f"\n  The gyro path has {camera_total_s / gyro_total_s:.0f}x less delay than the camera.")

    print("\nWhat each sensor can see:")
    for tone_hz in TONE_FREQS_HZ:
        camera_ok = tone_hz < config.camera.rate_hz / 2
        alias_hz = abs(tone_hz - config.camera.rate_hz)
        camera_text = (
            f"visible at {tone_hz:.0f} Hz" if camera_ok else f"folds to {alias_hz:.0f} Hz"
        )
        print(f"  {tone_hz:.0f} Hz tone: camera {camera_text:<22} gyro visible at {tone_hz:.0f} Hz")
    print(
        "\n  The camera cannot see anything above 30 Hz in its true place. That is a\n"
        "  limit on information, not on control, so no controller can undo it.\n"
        "  The gyro samples at 1 kHz, so both tones are well inside its range."
    )

    print("\nWhat the smaller delay is worth:")
    camera_limit_hz = delay_limited_crossover_hz(camera_total_s)
    gyro_limit_hz = delay_limited_crossover_hz(gyro_total_s)
    print(f"  camera path allows about {camera_limit_hz:5.1f} Hz crossover")
    print(f"  gyro path allows about   {gyro_limit_hz:5.1f} Hz crossover")
    print(
        f"\n  The gyro number is not achievable. It sits far above the {MODE_FREQ_HZ:.0f} Hz\n"
        "  structural mode, which becomes the limit long before then. Fusion moves the\n"
        "  bottleneck from the sensor to the structure, and the delivered loop crosses\n"
        "  over at 2 Hz rather than 50."
    )

    # --- plot ---------------------------------------------------------------
    delays_ms = np.logspace(-1, 2, 400)
    crossovers_hz = [delay_limited_crossover_hz(d * 1e-3) for d in delays_ms]

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(12, 5))

    names = ["camera", "gyro"]
    totals_ms = [camera_total_s * 1e3, gyro_total_s * 1e3]
    ax_bar.bar(names, totals_ms, color=["tab:blue", "tab:orange"])
    for index, value in enumerate(totals_ms):
        ax_bar.annotate(f"{value:.2f} ms", (index, value), ha="center", va="bottom")
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("delay seen by the loop [ms]")
    ax_bar.set_title("Delay budget: the gyro path is ~18x shorter")
    ax_bar.grid(True, axis="y", which="both", alpha=0.3)

    ax_curve.loglog(delays_ms, crossovers_hz, label=f"limit at {PHASE_MARGIN_DEG:.0f} deg margin")
    for name, total_s, colour in (
        ("camera", camera_total_s, "tab:blue"),
        ("gyro", gyro_total_s, "tab:orange"),
    ):
        ax_curve.plot(
            [total_s * 1e3], [delay_limited_crossover_hz(total_s)], "o", color=colour, label=name
        )
    ax_curve.axhline(MODE_FREQ_HZ, color="tab:red", linestyle="--",
                     label=f"{MODE_FREQ_HZ:.0f} Hz structural mode")
    for tone_hz in TONE_FREQS_HZ:
        ax_curve.axhline(tone_hz, color="purple", linestyle=":", linewidth=1.0)
    ax_curve.set_xlabel("loop delay [ms]")
    ax_curve.set_ylabel("crossover the delay allows [Hz]")
    ax_curve.set_title("Delay stops being the limit once the gyro is used")
    ax_curve.grid(True, which="both", alpha=0.3)
    ax_curve.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "timing_budget.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot written to {path}")


if __name__ == "__main__":
    main()
