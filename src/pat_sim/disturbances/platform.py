"""Platform-induced LOS vibration: broadband + 22 Hz tone + 47 Hz tone. Enters as base
angular motion, adding directly to LOS (not a mechanical torque disturbance)."""

from __future__ import annotations

import numpy as np

from pat_sim.config import PlatformDisturbanceConfig
from pat_sim.disturbances.broadband import BroadbandDisturbance
from pat_sim.disturbances.tonal import WanderingTone


class PlatformDisturbance:
    def __init__(
        self,
        config: PlatformDisturbanceConfig,
        dt_s: float,
        seed: int,
        wander_enabled: bool = True,
    ) -> None:
        seed_seq = np.random.SeedSequence(seed)
        broadband_seed, tone22_seed, tone47_seed = seed_seq.spawn(3)
        self.broadband = BroadbandDisturbance(
            cutoff_hz=config.broadband.lowpass_cutoff_hz,
            target_rms_rad=config.broadband.target_rms_rad,
            dt_s=dt_s,
            rng=np.random.default_rng(broadband_seed),
        )
        self.tone_22hz = WanderingTone(
            config.tone_22hz, dt_s, np.random.default_rng(tone22_seed), wander_enabled
        )
        self.tone_47hz = WanderingTone(
            config.tone_47hz, dt_s, np.random.default_rng(tone47_seed), wander_enabled
        )

    def step(self) -> float:
        return self.broadband.step() + self.tone_22hz.step() + self.tone_47hz.step()

    def generate(self, n_samples: int) -> np.ndarray:
        return (
            self.broadband.generate(n_samples)
            + self.tone_22hz.generate(n_samples)
            + self.tone_47hz.generate(n_samples)
        )
