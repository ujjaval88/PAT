"""Part 3: how well the fused estimate tracks the true LOS, against the camera
alone.

Produces:
  outputs/part3/fusion_tracking.png -- true LOS, the camera-only held value and
                                       the fused estimate, plus the error PSD
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.psd import rms, welch_psd
from pat_sim.config import DEFAULT_CONFIG
from pat_sim.control.fused_design import GYRO_DT_S
from pat_sim.estimation.estimator import KalmanFusionEstimator
from pat_sim.simulation.simulator import Simulator

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part3"
DT_S = 50e-6
DURATION_S = 20.0
SETTLE_S = 1.0
SEED = 42
NPERSEG = 32768
PLOT_WINDOW_S = 0.5


class _ZeroController:
    saturated_last_step = False

    def update(self, error_rad: float) -> float:
        del error_rad
        return 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG

    estimator = KalmanFusionEstimator(config.gyro, config.camera, gyro_dt_s=GYRO_DT_S)
    arrays = (
        Simulator(
            config,
            dt_s=DT_S,
            seed=SEED,
            controller=_ZeroController(),
            controller_rate_hz=1000.0,
            use_gyro=True,
            estimator=estimator,
        )
        .run(duration_s=DURATION_S)
        .as_arrays()
    )
    settle = int(SETTLE_S / DT_S)
    time_s = arrays["time_s"][settle:]
    true_los = arrays["theta_los_rad"][settle:]
    camera_hold = arrays["controller_input_rad"][settle:]
    fused = arrays["estimate_rad"][settle:]

    camera_error = camera_hold - true_los
    fused_error = fused - true_los
    true_rms = rms(true_los) * 1e6
    camera_rms = rms(camera_error) * 1e6
    fused_rms = rms(fused_error) * 1e6

    print("How well can the disturbance be seen? (open loop, controller off)\n")
    print(f"  true LOS motion                 {true_rms:7.1f} urad RMS")
    print(f"  camera-only held value, error   {camera_rms:7.1f} urad RMS")
    print(f"  fused gyro+camera, error        {fused_rms:7.1f} urad RMS")
    print(f"\n  fusion is {camera_rms / fused_rms:.1f}x better at tracking the true LOS")
    if camera_rms > true_rms:
        print(
            f"\n  Note the camera-only error ({camera_rms:.0f} urad) is larger than the signal\n"
            f"  it is estimating ({true_rms:.0f} urad). Holding a 60 Hz sample for 17 ms while\n"
            "  the disturbance keeps moving is worse than useless as an estimate."
        )

    freq_hz, psd_true = welch_psd(true_los, 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_camera = welch_psd(camera_error, 1.0 / DT_S, nperseg=NPERSEG)
    _, psd_fused = welch_psd(fused_error, 1.0 / DT_S, nperseg=NPERSEG)

    print("\nWhere the estimate improves, by band (error RMS, urad):\n")
    print(f"  {'band [Hz]':>16} {'camera only':>12} {'fused':>9} {'ratio':>7}")
    for low_hz, high_hz in ((0.0, 5.0), (5.0, 20.0), (20.0, 60.0), (60.0, 200.0)):
        mask = (freq_hz >= low_hz) & (freq_hz < high_hz)
        camera_band = float(np.sqrt(np.trapezoid(psd_camera[mask], freq_hz[mask]))) * 1e6
        fused_band = float(np.sqrt(np.trapezoid(psd_fused[mask], freq_hz[mask]))) * 1e6
        ratio = fused_band / camera_band if camera_band > 0 else float("nan")
        print(
            f"  {low_hz:7.1f} - {high_hz:6.1f} {camera_band:12.1f} {fused_band:9.1f} {ratio:7.2f}"
        )
    print("\n  'ratio' is fused divided by camera only. Below 1 means fusion helps.")
    print(
        "\n  The gyro supplies the fast motion the camera cannot sample, and the camera\n"
        "  supplies the absolute reference the gyro cannot provide on its own. Neither\n"
        "  sensor is sufficient alone."
    )

    # --- plots --------------------------------------------------------------
    window = slice(0, int(PLOT_WINDOW_S / DT_S))
    fig, (ax_trace, ax_psd) = plt.subplots(2, 1, figsize=(9, 8))
    ax_trace.plot(time_s[window], true_los[window] * 1e6, linewidth=1.4, label="true LOS")
    ax_trace.plot(time_s[window], camera_hold[window] * 1e6, linewidth=0.9,
                  label=f"camera only ({camera_rms:.0f} urad error)")
    ax_trace.plot(time_s[window], fused[window] * 1e6, linewidth=0.9,
                  label=f"fused estimate ({fused_rms:.0f} urad error)")
    ax_trace.set_xlabel("Time [s]")
    ax_trace.set_ylabel("LOS [urad]")
    ax_trace.set_title("Tracking the true line of sight, open loop")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(fontsize=8)

    ax_psd.semilogy(freq_hz, psd_true, linewidth=1.0, label="true LOS")
    ax_psd.semilogy(freq_hz, psd_camera, linewidth=1.0, label="camera-only error")
    ax_psd.semilogy(freq_hz, psd_fused, linewidth=1.0, label="fused error")
    ax_psd.set_xlim(0, 100)
    band = freq_hz <= 100.0
    peak = float(np.max(psd_camera[band]))
    ax_psd.set_ylim(peak / 1e5, peak * 5.0)
    ax_psd.set_xlabel("Frequency [Hz]")
    ax_psd.set_ylabel("PSD [rad^2/Hz]")
    ax_psd.set_title("Estimate error spectrum: fusion wins across the disturbance band")
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "fusion_tracking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot written to {path}")


if __name__ == "__main__":
    main()
