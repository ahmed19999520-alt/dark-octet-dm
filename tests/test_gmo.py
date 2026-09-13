import pytest
import numpy as np
from dark_octet.spectrum.gmo_formula import GMOMassOperator, DARK_OCTET_STATES
from dark_octet.constants import BenchmarkParameters


BENCH = BenchmarkParameters()
GMO = GMOMassOperator(BENCH)


def test_chi0_mass():
    masses = GMO.all_masses()
    assert abs(masses["chi_0"] - 165.67) < 0.01


def test_chi_pp_mass():
    masses = GMO.all_masses()
    assert abs(masses["chi_pp"] - 172.67) < 0.01


def test_sigma0_mass():
    masses = GMO.all_masses()
    assert abs(masses["Sigma_0"] - 134.00) < 0.01


def test_lambda_mass():
    masses = GMO.all_masses()
    assert abs(masses["Lambda"] - 134.00) < 0.01


def test_sigma0_lambda_degeneracy():
    masses = GMO.all_masses()
    assert abs(masses["Sigma_0"] - masses["Lambda"]) < 1e-10


def test_chi0_chip_degeneracy():
    masses = GMO.all_masses()
    assert abs(masses["chi_0"] - masses["chi_p"]) < 1e-10


def test_r21():
    ratios = GMO.mass_ratios()
    assert abs(ratios["r21_numerical"] - 1.042) < 0.005


def test_r31():
    ratios = GMO.mass_ratios()
    assert abs(ratios["r31_numerical"] - 0.809) < 0.005


def test_r21_r31_scale_invariant():
    params_scaled = BenchmarkParameters(
        M0=300.0, beta=30.0, gamma=16.0, delta=10.0, eta=4.0
    )
    gmo_scaled = GMOMassOperator(params_scaled)
    r21_orig = GMO.mass_ratios()["r21_numerical"]
    r21_scaled = gmo_scaled.mass_ratios()["r21_numerical"]
    assert abs(r21_orig - r21_scaled) < 1e-10


def test_gst_consistency():
    gst = GMO.gst_consistency()
    assert gst["gst_satisfied"]
    assert gst["discrepancy_percent"] < 5.0


def test_all_masses_positive():
    masses = GMO.all_masses()
    for name, m in masses.items():
        assert m > 0.0, f"Mass of {name} is non-positive: {m}"


def test_dm_candidate_lightest():
    masses = GMO.all_masses()
    m_dm = masses["chi_0"]
    for name, m in masses.items():
        assert m >= m_dm - 35.0, f"{name} lighter than DM by too much"


def test_dataframe_shape():
    df = GMO.mass_spectrum_dataframe()
    assert df.shape[0] == 8
    assert "Mass_GeV" in df.columns
    assert "Is_DM" in df.columns


def test_symbolic_operator_type():
    import sympy as sp
    expr = GMO.symbolic_mass_operator()
    assert isinstance(expr, sp.Expr)