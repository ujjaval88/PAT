"""Part 3: which sensor sees the disturbance, and which does not.

Produces:
  outputs/part3/sensor_visibility.png -- what each sensor reports while the
                                         platform moves and the gimbal is still
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_sim.config import DEFAULT_CONFIG
from pat_sim.sensors.encoder import Encoder
from pat_sim.sensors.gyro import Gyro

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "part3"
SEED = 42
N_SAMPLES = 400
PLATFORM_FREQ_HZ = 22.0
PLATFORM_AMPLITUDE_RAD = 150e-6
DT_S = 1e-3
SHAFT_ANGLE_RAD = 0.1


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    rng = np.random.default_rng(SEED)

    # The gimbal is held still and only the platform moves. This is the case
    # that separates the sensors: LOS is changing, but the shaft is not.
    time_s = np.arange(N_SAMPLES) * DT_S
    omega = 2.0 * np.pi * PLATFORM_FREQ_HZ
    theta_b = PLATFORM_AMPLITUDE_RAD * np.sin(omega * time_s)
    theta_b_rate = PLATFORM_AMPLITUDE_RAD * omega * np.cos(omega * time_s)
    theta_los = SHAFT_ANGLE_RAD + theta_b

    encoder = Encoder(config.encoder)
    gyro = Gyro(config.gyro, rng)
    encoder_out = np.array(
        [encoder.sample(SHAFT_ANGLE_RAD, t).value for t in time_s]
    )
    gyro_out = np.array(
        [gyro.sample(float(rate), t).value for rate, t in zip(theta_b_rate, time_s, strict=True)]
    )

    print("Test: the platform moves, the gimbal shaft is held still.\n")
    print(
        f"  platform motion   {PLATFORM_AMPLITUDE_RAD * 1e6:.0f} urad"
        f" at {PLATFORM_FREQ_HZ:.0f} Hz"
    )
    print(f"  shaft angle       {SHAFT_ANGLE_RAD:.3f} rad, constant")
    print(f"  LOS               moves by {theta_los.std() * 1e6:.1f} urad RMS\n")

    print(f"  {'sensor':>10} {'measures':>34} {'reading changes?':>18}")
    encoder_span = float(encoder_out.max() - encoder_out.min())
    gyro_span = float(gyro_out.max() - gyro_out.min())
    print(
        f"  {'encoder':>10} {'shaft angle relative to the base':>34} "
        f"{'no' if encoder_span == 0 else 'yes':>18}"
    )
    print(
        f"  {'gyro':>10} {'inertial rate of the line of sight':>34} "
        f"{'no' if gyro_span == 0 else 'yes':>18}"
    )
    print(f"\n  encoder output: {len(set(encoder_out))} distinct value(s) over the whole run")
    print(f"  gyro output:    swings {gyro_span:.4f} rad/s, tracking the platform")

    # Correlate each reading against the platform motion it is supposed to see.
    gyro_match = float(np.corrcoef(gyro_out, theta_b_rate)[0, 1])
    print(f"\n  gyro vs true platform rate: correlation {gyro_match:.4f}")
    print(
        "\n  The encoder is blind to platform motion by construction: it measures the\n"
        "  shaft against the base, and the base is what is moving, so the motion\n"
        "  cancels out of its reading. Its timing is excellent - 1 kHz, no latency,\n"
        "  better than the gyro - and that does not help, because it cannot see the\n"
        "  disturbance at all.\n"
        "\n  The gyro is inertial, so it measures the LOS rate whether that rate comes\n"
        "  from the gimbal or from the platform. That indifference is exactly what is\n"
        "  needed here, since the platform is the disturbance.\n"
        "\n  Decision: fuse gyro and camera. The gyro carries the disturbance\n"
        "  information; the camera is the only absolute angle reference and is what\n"
        "  keeps the gyro bias observable. The encoder is not used. The QPD is not\n"
        "  used either: its +/-200 urad range is smaller than the open-loop jitter,\n"
        "  so it cannot acquire until a coarse loop has already reduced the error."
    )

    # --- plot ---------------------------------------------------------------
    fig, (ax_angle, ax_rate) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_angle.plot(time_s, (theta_los - SHAFT_ANGLE_RAD) * 1e6, label="true LOS motion")
    ax_angle.plot(time_s, (encoder_out - SHAFT_ANGLE_RAD) * 1e6, label="encoder reading")
    ax_angle.set_ylabel("angle [urad]")
    ax_angle.set_title("The platform moves the LOS; the encoder does not notice")
    ax_angle.grid(True, alpha=0.3)
    ax_angle.legend(fontsize=8)

    ax_rate.plot(time_s, theta_b_rate, label="true LOS rate")
    ax_rate.plot(time_s, gyro_out, linewidth=0.9, label="gyro reading")
    ax_rate.set_xlabel("Time [s]")
    ax_rate.set_ylabel("rate [rad/s]")
    ax_rate.set_title("The gyro is inertial, so it sees the same motion")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "sensor_visibility.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot written to {path}")


if __name__ == "__main__":
    main()
