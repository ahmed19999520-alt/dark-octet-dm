import pytest
import numpy as np
from dark_octet.boltzmann.solver import BoltzmannSolver
from dark_octet.boltzmann.decay_rates import (
    decay_rate_two_body,
    lifetime_seconds,
    REFERENCE_DECAY_RATES,
)
from dark_octet.boltzmann.cross_sections import (
    sigma_v_chi0_pair,
    drell_yan_xs_ee_to_charged_dark,
    drell_yan_xs_pp_chi0chi0,
)
from dark_octet.constants import BENCH, PHYS


def test_sigma_v_chi0_order_of_magnitude():
    sv = sigma_v_chi0_pair(T_gev=165.67 / 25.0)
    assert 1e-27 < sv < 1e-24


def test_decay_rate_sigma0():
    gamma = decay_rate_two_body(
        m_parent_gev=134.00,
        m_dm_gev=165.67,
        m_phi_gev=0.01,
        g_dark=BENCH.g_dark,
    )
    assert gamma == pytest.approx(0.0, abs=1e-30)


def test_decay_rate_chipp():
    gamma = decay_rate_two_body(
        m_parent_gev=172.67,
        m_dm_gev=165.67,
        m_phi_gev=0.01,
        g_dark=BENCH.g_dark,
    )
    assert gamma > 0.0


def test_lifetime_chi0_infinite():
    tau = lifetime_seconds(decay_rate_gev=0.0)
    assert tau == np.inf


def test_lifetime_sigma0_shorter_than_bbn():
    gamma = decay_rate_two_body(
        m_parent_gev=134.00,
        m_dm_gev=165.67,
        m_phi_gev=0.01,
        g_dark=BENCH.g_dark,
    )
    tau = lifetime_seconds(gamma)
    assert tau == np.inf


def test_drell_yan_ee_threshold_zero():
    sigma = drell_yan_xs_ee_to_charged_dark(
        sqrt_s_gev=300.0,
        m_chi_p=165.67,
    )
    assert sigma == 0.0


def test_drell_yan_ee_above_threshold():
    sigma = drell_yan_xs_ee_to_charged_dark(
        sqrt_s_gev=365.0,
        m_chi_p=165.67,
    )
    assert sigma > 0.0


def test_drell_yan_pp_100tev():
    sigma = drell_yan_xs_pp_chi0chi0(sqrt_s_tev=100.0)
    assert 0.1 < sigma < 5.0


def test_solver_freeze_out_range():
    solver = BoltzmannSolver()
    sol = solver.solve()
    assert 15.0 < sol.freeze_out_x < 40.0


def test_solver_omega_range():
    solver = BoltzmannSolver()
    sol = solver.solve()
    assert 0.05 < sol.omega_dm_h2 < 0.30


def test_solver_y_inf_positive():
    solver = BoltzmannSolver()
    sol = solver.solve()
    assert sol.y_inf_chi0 > 0.0


def test_solver_species_count():
    solver = BoltzmannSolver()
    sol = solver.solve()
    assert sol.yields.shape[0] == 8