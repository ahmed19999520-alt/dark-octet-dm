import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.constants import (
    G_DARK, ALPHA_DARK, G_F, M_ZP_GEV, M_ZP2, M_ZP4,
    M0_GEV, BETA_GEV, GAMMA_GEV, DELTA_GEV, ETA_GEV,
    V_DELTA_GEV, Y_CHI, YCHI_VDELTA,
    HBAR_C_GEV_CM, GEV_TO_S, GEV_PER_G, GEV2_TO_CM2,
    M_PL_GEV, M_PL_REDUCED_GEV,
    ALPHA_EM, C_LIGHT_KM_S,
    OCTET_ORDER, OCTET_QN, OCTET_GDOF, MASS_PAPER,
)


class TestPhysicalConstants:

    def test_alpha_dark_value(self):
        expected = G_DARK ** 2 / (4.0 * np.pi)
        assert abs(ALPHA_DARK - expected) < 1e-10

    def test_alpha_dark_benchmark(self):
        assert abs(ALPHA_DARK - 0.020) < 5e-4

    def test_mzp2_consistency(self):
        assert abs(M_ZP2 - M_ZP_GEV ** 2) < 1e-10

    def test_mzp4_consistency(self):
        assert abs(M_ZP4 - M_ZP_GEV ** 4) < 1e-6

    def test_ychi_vdelta(self):
        assert abs(YCHI_VDELTA - Y_CHI * V_DELTA_GEV) < 1e-10

    def test_ychi_vdelta_value(self):
        assert abs(YCHI_VDELTA - 1.70) < 1e-10

    def test_hbar_c_value(self):
        assert abs(HBAR_C_GEV_CM - 1.97327e-14) < 1e-18

    def test_gev_to_s(self):
        assert abs(GEV_TO_S - 6.582e-25) < 1e-28

    def test_gev_per_g(self):
        assert abs(1.0 / GEV_PER_G - 1.78266e-24) < 1e-27

    def test_gev2_to_cm2(self):
        assert abs(GEV2_TO_CM2 - (1.97327e-14) ** 2) < 1e-30

    def test_mpl_reduced_value(self):
        assert abs(M_PL_REDUCED_GEV / 2.435e18 - 1.0) < 0.01

    def test_alpha_em_value(self):
        assert abs(ALPHA_EM - 1.0 / 137.036) < 1e-6

    def test_c_light_kms(self):
        assert abs(C_LIGHT_KM_S - 299792.458) < 0.001

    def test_gf_kinetic_mixing(self):
        assert abs(G_F - 1e-3) < 1e-20


class TestBenchmarkParameters:

    def test_m0_value(self):
        assert abs(M0_GEV - 208.67) < 1e-10

    def test_beta_value(self):
        assert abs(BETA_GEV - (-15.0)) < 1e-10

    def test_gamma_value(self):
        assert abs(GAMMA_GEV - 8.0) < 1e-10

    def test_delta_value(self):
        assert abs(DELTA_GEV - 5.0) < 1e-10

    def test_eta_value(self):
        assert abs(ETA_GEV - 8.0) < 1e-10


class TestOctetMetadata:

    def test_octet_order_length(self):
        assert len(OCTET_ORDER) == 8

    def test_octet_qn_keys(self):
        for name in OCTET_ORDER:
            assert name in OCTET_QN
            qn = OCTET_QN[name]
            assert "I" in qn and "Y" in qn and "T" in qn and "Q_pos" in qn

    def test_octet_gdof_sum(self):
        total = sum(OCTET_GDOF.values())
        assert total == 18

    def test_mass_paper_values(self):
        assert abs(MASS_PAPER["chi0"]   - 165.67) < 1e-10
        assert abs(MASS_PAPER["Lambda"] - 208.67) < 1e-10
