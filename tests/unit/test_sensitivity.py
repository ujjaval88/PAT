import math

import numpy as np
import pytest

from pat_sim.analysis.sensitivity import (
    achieved_crossover_and_margin,
    camera_extra_delay_s,
    complementary_sensitivity,
    design_camera_only_loop,
    open_loop_response,
    sensitivity,
)
from pat_sim.config import DEFAULT_CONFIG

CFG = DEFAULT_CONFIG


def _design_default():
    extra_delay = camera_extra_delay_s(CFG.camera, CFG.part1_design.latency_design_percentile)
    total_delay = extra_delay + CFG.plant.transport_delay_s
    omega_c = math.radians(CFG.part1_design.phase_margin_deg) / total_delay
    return design_camera_only_loop(
        CFG.plant, omega_c, CFG.part1_design.phase_margin_deg, extra_delay
    )


def test_design_hits_exact_crossover_and_margin():
    design = _design_default()
    magnitude, phase_margin_deg = achieved_crossover_and_margin(design, CFG.plant)
    assert magnitude == pytest.approx(1.0, rel=1e-9)
    assert phase_margin_deg == pytest.approx(CFG.part1_design.phase_margin_deg, abs=1e-6)


def test_sensitivity_plus_complementary_equals_one():
    design = _design_default()
    omega_grid = np.logspace(-1, 3, 2000)
    open_loop = open_loop_response(design, CFG.plant, omega_grid)
    s = sensitivity(open_loop)
    t = complementary_sensitivity(open_loop)
    np.testing.assert_allclose(s + t, np.ones_like(s), atol=1e-9)


def test_reports_finite_sensitivity_at_tonal_frequencies():
    design = _design_default()
    for f_hz in (22.0, 47.0, 13.0):
        omega = np.array([2 * math.pi * f_hz])
        s = sensitivity(open_loop_response(design, CFG.plant, omega))[0]
        assert np.isfinite(s.real) and np.isfinite(s.imag)
        assert 0.0 < abs(s) < 10.0  # sanity bound: no blow-up / no zero-division artifact
