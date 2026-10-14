import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.spectrum.gmo_formula import (
    gmo_mass, octet_spectrum, mass_splittings, mass_ratios,
    ratio_r21_analytic, ratio_r31_analytic, ratio_r41_analytic,
    validate_spectrum, benchmark_ratios, gst_ratio,
)
from dark_octet.constants import (
    M0_GEV, BETA_GEV, GAMMA_GEV, DELTA_GEV, ETA_GEV,
    YCHI_VDELTA, MASS_PAPER, OCTET_ORDER,
)


class TestGMOMassFormula:

    def test_chi0_mass(self):
        m = gmo_mass(1.5, +1, 0, False)
        assert abs(m - 165.67) < 0.005

    def test_chi_p_mass(self):
        m = gmo_mass(1.5, +1, 0, True)
        assert abs(m - 167.37) < 0.005

    def test_chi_m_mass(self):
        m = gmo_mass(1.5, +1, -1, False)
        assert abs(m - 168.67) < 0.005

    def test_chi_pp_mass(self):
        m = gmo_mass(1.5, +1, +1, True)
        assert abs(m - 180.37) < 0.005

    def test_sigma0_mass(self):
        m = gmo_mass(1.0, 0, 0, False)
        assert abs(m - 192.67) < 0.005

    def test_sigma_m_mass(self):
        m = gmo_mass(1.0, 0, -1, False)
        assert abs(m - 195.67) < 0.005

    def test_sigma_p_mass(self):
        m = gmo_mass(1.0, 0, +1, True)
        assert abs(m - 207.37) < 0.005

    def test_lambda_mass(self):
        m = gmo_mass(0.0, 0, 0, False)
        assert abs(m - 208.67) < 0.005

    def test_lambda_equals_m0(self):
        m = gmo_mass(0.0, 0, 0, False)
        assert abs(m - M0_GEV) < 1e-10

    def test_triplet_contribution(self):
        m_no  = gmo_mass(1.5, +1, 0, False)
        m_yes = gmo_mass(1.5, +1, 0, True)
        assert abs(m_yes - m_no - YCHI_VDELTA) < 1e-10

    def test_darkicity_linear_term(self):
        m0  = gmo_mass(1.0, 0, 0, False)
        mp1 = gmo_mass(1.0, 0, +1, False)
        mm1 = gmo_mass(1.0, 0, -1, False)
        assert abs(mp1 - m0 - DELTA_GEV) < 1e-8
        assert abs(m0 - mm1 - DELTA_GEV) < 1e-8

    def test_darkicity_quadratic_term(self):
        m0  = gmo_mass(1.0, 0, 0, False)
        mp1 = gmo_mass(1.0, 0, +1, False)
        mm1 = gmo_mass(1.0, 0, -1, False)
        avg = 0.5 * (mp1 + mm1)
        assert abs(avg - m0 - ETA_GEV) < 1e-8


class TestOctetSpectrum:

    def test_full_spectrum_keys(self):
        spec = octet_spectrum()
        for name in OCTET_ORDER:
            assert name in spec

    def test_spectrum_ordering(self):
        spec = octet_spectrum()
        masses = [spec[n] for n in OCTET_ORDER]
        assert masses[0] < masses[-1]

    def test_validate_spectrum_all_pass(self):
        result = validate_spectrum(tol=0.005)
        for name, data in result.items():
            assert data["pass"], f"{name}: code={data['code']}, paper={data['paper']}"


class TestMassRatios:

    def setup_method(self):
        self.masses = octet_spectrum()
        self.ratios = mass_ratios(self.masses)

    def test_r21_value(self):
        assert abs(self.ratios["r21"] - 1.089) < 0.005

    def test_r31_value(self):
        assert abs(self.ratios["r31"] - 1.163) < 0.006

    def test_r41_value(self):
        assert abs(self.ratios["r41"] - 1.260) < 0.008

    def test_r21_analytic(self):
        r = ratio_r21_analytic()
        assert abs(r - 1.089) < 0.005

    def test_r31_analytic(self):
        r = ratio_r31_analytic()
        assert abs(r - 1.163) < 0.006

    def test_r41_analytic(self):
        r = ratio_r41_analytic()
        assert abs(r - 1.260) < 0.008

    def test_ratios_m0_independent(self):
        m1 = octet_spectrum(M0=208.67)
        m2 = octet_spectrum(M0=250.00)
        r1 = mass_ratios(m1)
        r2 = mass_ratios(m2)
        assert abs(r1["r21"] - r2["r21"]) > 1e-4

    def test_delta_m_chi_p(self):
        splits = mass_splittings(self.masses)
        assert abs(splits["chi_p"] - 1.70) < 0.005

    def test_delta_m_chi_pp(self):
        splits = mass_splittings(self.masses)
        assert abs(splits["chi_pp"] - 14.70) < 0.005

    def test_delta_m_sigma0(self):
        splits = mass_splittings(self.masses)
        assert abs(splits["Sigma0"] - 27.00) < 0.005

    def test_delta_m_lambda(self):
        splits = mass_splittings(self.masses)
        assert abs(splits["Lambda"] - 43.00) < 0.005


class TestGSTRelation:

    def test_gst_ratio_value(self):
        ratio = gst_ratio()
        assert abs(ratio - 1.875) < 0.001

    def test_gst_epsilon_estimate(self):
        ratio = gst_ratio()
        eps   = 0.12
        kappa = 1.56
        residual = abs(ratio - eps * kappa * 10.0)
        assert residual < 0.5
