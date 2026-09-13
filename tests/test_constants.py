import numpy as np
import pytest
from dark_octet.constants import (
    PhysicalConstants,
    BenchmarkParameters,
    PHYS,
    BENCH,
)


def test_planck_length_scale():
    lP = np.sqrt(PHYS.hbar_gev_s * PHYS.g_newton_si / PHYS.c_m_s**3)
    assert 1.5e-35 < lP < 1.7e-35


def test_alpha_dark():
    alpha = BENCH.alpha_dark_computed
    assert abs(alpha - BENCH.alpha_dark) < 0.001


def test_v_dark():
    v = BENCH.v_dark_gev
    assert 5.0 < v < 30.0


def test_instanton_action():
    S = 8.0 * np.pi**2 / BENCH.alpha_dark
    assert abs(S - BENCH.instanton_action) < 1.0


def test_benchmark_g_dark():
    assert BENCH.g_dark == pytest.approx(0.5)


def test_benchmark_g_f():
    assert BENCH.g_f == pytest.approx(1e-3)


def test_benchmark_m_zprime():
    assert BENCH.m_zprime_gev == pytest.approx(9.7)


def test_planck_mass_gev():
    assert 1.2e19 < PHYS.m_planck_gev < 1.25e19


def test_cm2_per_gev2():
    assert abs(PHYS.cm2_per_gev2 - 3.8937966e-28) < 1e-31


def test_reduced_planck():
    mpl_red = PHYS.m_planck_reduced_gev
    assert mpl_red < PHYS.m_planck_gev
    assert mpl_red > 1e18