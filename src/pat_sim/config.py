"""Central parameter configuration.

Every value is tagged in its comment as one of:

- supplied      -- given directly in the problem statement
- derived       -- computed from supplied values
- assumption    -- not specified; an engineering choice made explicitly here
- experiment    -- a knob for sweeps/studies, not a system property
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructuralModeConfig:
    freq_hz: float  # supplied
    zeta: float  # supplied

    @property
    def omega_n(self) -> float:
        return 2.0 * math.pi * self.freq_hz


@dataclass(frozen=True)
class PlantConfig:
    """Rigid body + structural modes + actuator + transport delay. supplied."""

    inertia_kg_m2: float = 2.5e-3
    modes: tuple[StructuralModeConfig, ...] = (
        StructuralModeConfig(freq_hz=75.0, zeta=0.03),
        StructuralModeConfig(freq_hz=220.0, zeta=0.04),
    )
    actuator_tau_s: float = 0.2e-3
    transport_delay_s: float = 0.5e-3


@dataclass(frozen=True)
class MotorLimitsConfig:
    """supplied: "torque saturates at +/-0.5 N*m and rate at +/-3 rad/s".
    """

    torque_max_n_m: float = 0.5
    rate_max_rad_s: float = 3.0


@dataclass(frozen=True)
class FrictionConfig:
    """supplied (approximate values, "Tc ~= 0.005", "b ~= 1e-3")."""

    coulomb_n_m: float = 0.005
    viscous_n_m_s_per_rad: float = 1e-3


@dataclass(frozen=True)
class CoggingConfig:
    """Amplitude/cycle-count supplied; phase is explicitly unspecified (assumption)."""

    amplitude_n_m: float = 0.02
    cycles_per_revolution: int = 12
    phase_rad: float = 0.0  # assumption: not specified, so kept configurable


@dataclass(frozen=True)
class SimulationConfig:
    """supplied (baseline physics rate 20 kHz -> dt = 50 us)."""

    physics_rate_hz: float = 20_000.0

    @property
    def dt_s(self) -> float:
        return 1.0 / self.physics_rate_hz


@dataclass(frozen=True)
class BroadbandDisturbanceConfig:
    """supplied."""

    lowpass_cutoff_hz: float = 80.0
    target_rms_rad: float = 200e-6


@dataclass(frozen=True)
class ToneConfig:
    """supplied."""

    nominal_freq_hz: float
    amplitude_rad: float
    wander_hz: float


@dataclass(frozen=True)
class PlatformDisturbanceConfig:
    """supplied. Total RMS (~330 urad) is a statistical expectation, not an exact target."""

    broadband: BroadbandDisturbanceConfig = field(default_factory=BroadbandDisturbanceConfig)
    tone_22hz: ToneConfig = field(
        default_factory=lambda: ToneConfig(
            nominal_freq_hz=22.0, amplitude_rad=150e-6, wander_hz=2.0
        )
    )
    tone_47hz: ToneConfig = field(
        default_factory=lambda: ToneConfig(
            nominal_freq_hz=47.0, amplitude_rad=100e-6, wander_hz=3.0
        )
    )
    expected_total_rms_rad: float = 330e-6


@dataclass(frozen=True)
class CameraConfig:
    """supplied. Latency distribution shape (max(0, Normal)) is an assumption: 
    mean/std are supplied, the clipped-Gaussian shape is the unspecified modeling 
    decision."""

    rate_hz: float = 60.0
    latency_mean_s: float = 10e-3
    latency_std_s: float = 20e-3
    noise_std_rad: float = 15e-6


@dataclass(frozen=True)
class GyroConfig:
    """Rate/latency/noise supplied; bias random-walk magnitude is an explicit
    assumption (not specified)."""

    rate_hz: float = 1000.0
    latency_s: float = 1e-3
    noise_std_rad_s: float = 8e-4
    bias_random_walk_std_rad_s_per_sqrt_s: float = 1e-5  # assumption, configurable


@dataclass(frozen=True)
class EncoderConfig:
    """supplied."""

    rate_hz: float = 1000.0
    latency_s: float = 0.0
    resolution_bits: int = 20

    @property
    def quantum_rad(self) -> float:
        return 2.0 * math.pi / (2**self.resolution_bits)


@dataclass(frozen=True)
class QpdConfig:
    """supplied."""

    rate_hz: float = 10_000.0
    latency_s: float = 50e-6
    noise_std_rad: float = 0.5e-6
    valid_range_rad: float = 200e-6


@dataclass(frozen=True)
class Part1DesignConfig:
    """These are not system properties -- these are choices about how conservatively
    to design the camera-only loop against latency variability."""

    phase_margin_deg: float = 45.0
    latency_design_percentile: float = 0.95


@dataclass(frozen=True)
class SystemConfig:
    plant: PlantConfig = field(default_factory=PlantConfig)
    motor_limits: MotorLimitsConfig = field(default_factory=MotorLimitsConfig)
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    cogging: CoggingConfig = field(default_factory=CoggingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    platform_disturbance: PlatformDisturbanceConfig = field(
        default_factory=PlatformDisturbanceConfig
    )
    camera: CameraConfig = field(default_factory=CameraConfig)
    gyro: GyroConfig = field(default_factory=GyroConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    qpd: QpdConfig = field(default_factory=QpdConfig)
    part1_design: Part1DesignConfig = field(default_factory=Part1DesignConfig)


DEFAULT_CONFIG = SystemConfig()
