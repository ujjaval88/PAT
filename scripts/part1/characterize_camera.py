"""Part 1: camera characterization -- latency design budget, aliasing, and what
the 60 Hz camera does (and doesn't) see of the platform disturbance.

Produces:
  outputs/part1/platform_psd.png -- PSD of the combined platform disturbance
  outputs/part1/camera_alias.png -- PSD of a 47 Hz LOS tone as seen by the 60 Hz camera
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.analysis.psd import dominant_frequency_hz, rms, welch_psd
from pat_sim.analysis.sensitivity import (
    camera_latency_mean_s,
    camera_latency_percentile_s,
    camera_zoh_delay_s,
)
from pat_sim.config import DEFAULT_CONFIG, CameraConfig
from pat_sim.disturbances.platform import PlatformDisturbance
from pat_sim.sensors.camera import Camera

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part1"
DT_S = 1e-3  # 1 kHz: comfortably above 80 Hz cutoff and the 47 Hz tone's Nyquist need


def print_latency_design_table(config) -> None:
    print("Camera latency design table (Y = max(0, N(10ms, 20ms))):")
    print(f"{'design point':<14}{'latency [ms]':>14}{'T_total [ms]':>16}{'f_c [Hz]':>12}")
    zoh_s = camera_zoh_delay_s(config.camera)
    points: list[tuple[str, float]] = [
        ("median", camera_latency_percentile_s(config.camera, 0.50)),
        ("mean", camera_latency_mean_s(config.camera)),
        ("75th pct", camera_latency_percentile_s(config.camera, 0.75)),
        ("90th pct", camera_latency_percentile_s(config.camera, 0.90)),
        ("95th pct", camera_latency_percentile_s(config.camera, 0.95)),
        ("99th pct", camera_latency_percentile_s(config.camera, 0.99)),
    ]
    for label, latency_s in points:
        total_s = zoh_s + latency_s + config.plant.transport_delay_s
        omega_c = math.radians(config.part1_design.phase_margin_deg) / total_s
        marker = " <- design point" if label == "95th pct" else ""
        print(
            f"{label:<14}{latency_s * 1e3:>14.2f}{total_s * 1e3:>16.2f}"
            f"{omega_c / (2 * math.pi):>12.2f}{marker}"
        )


def platform_disturbance_report() -> None:
    config = DEFAULT_CONFIG.platform_disturbance
    dist = PlatformDisturbance(config, dt_s=DT_S, seed=42)
    n_samples = 120_000  # 120 s
    samples = dist.generate(n_samples)[5000:]  # drop filter settling transient

    measured_rms_urad = rms(samples) * 1e6
    print("\nPlatform disturbance (combined broadband + 22 Hz + 47 Hz):")
    print(f"  measured total RMS = {measured_rms_urad:.1f} urad  (the spec expects ~330 urad)")

    # broadband RMS/tone RMS are recomputed independently to report a like-for-like
    # component breakdown against the combined measurement above.
    fresh = PlatformDisturbance(config, dt_s=DT_S, seed=42)
    broadband_samples = fresh.broadband.generate(n_samples)[5000:]
    broadband_rms_urad = rms(broadband_samples) * 1e6
    tone22_rms_urad = config.tone_22hz.amplitude_rad / math.sqrt(2) * 1e6
    tone47_rms_urad = config.tone_47hz.amplitude_rad / math.sqrt(2) * 1e6
    quadrature_urad = math.sqrt(broadband_rms_urad**2 + tone22_rms_urad**2 + tone47_rms_urad**2)
    print(
        f"  component quadrature sum ~= {quadrature_urad:.1f} urad "
        f"(broadband {broadband_rms_urad:.1f} + 22Hz {tone22_rms_urad:.1f} rms "
        f"+ 47Hz {tone47_rms_urad:.1f} rms)"
    )
    print(
        "  This is below the ~330 urad the spec quotes. That figure is a statistical"
    )
    print(
        "  expectation, not a target the components have to sum to, so the difference"
    )
    print("  is reported rather than tuned away.")

    f, pxx = welch_psd(samples, fs_hz=1.0 / DT_S, nperseg=16384)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f, pxx)
    ax.set_xlim(0, 100)
    for f_tone, label in [(22, "22 Hz"), (47, "47 Hz")]:
        ax.axvline(f_tone, color="purple", linestyle=":", label=label)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [rad^2/Hz]")
    ax.set_title("Combined platform disturbance PSD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "platform_psd.png", dpi=150)
    plt.close(fig)


def camera_alias_report() -> None:
    camera_rate_hz = 60.0
    cfg = CameraConfig(rate_hz=camera_rate_hz, noise_std_rad=0.0)
    camera = Camera(cfg, np.random.default_rng(0))

    n_samples = 3000
    capture_times = np.arange(n_samples) / camera_rate_hz
    true_los = 150e-6 * np.sin(2 * math.pi * 47.0 * capture_times)
    measured = np.array(
        [camera.sample(true_los[k], capture_times[k]).value for k in range(n_samples)]
    )

    dominant_hz = dominant_frequency_hz(measured, fs_hz=camera_rate_hz)
    print("\nCamera aliasing demo: true 47 Hz LOS tone sampled at 60 Hz")
    print("  expected alias = |47 - 60| = 13 Hz")
    print(f"  measured dominant frequency in sampled sequence = {dominant_hz:.2f} Hz")

    f, pxx = welch_psd(measured, fs_hz=camera_rate_hz, nperseg=1024)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f, pxx)
    ax.axvline(13.0, color="tab:red", linestyle="--", label="expected alias (13 Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [rad^2/Hz]")
    ax.set_title("47 Hz LOS tone as seen by the 60 Hz camera")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "camera_alias.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_latency_design_table(DEFAULT_CONFIG)
    platform_disturbance_report()
    camera_alias_report()
    print(f"\nPlots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
