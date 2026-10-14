import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.signatures.gravitational_waves import (
    gw_peak_frequency_hz, gw_peak_omega, gw_summary, dao_scale,
    in_lisa_band, lisa_sensitivity_hz,
)
from dark_octet.signatures.collider import (
    sigma_epem_chi_p_chi_m_fb, fcce_summary, fcc_hh_summary,
    mono_lepton_endpoint_gev, collider_table,
)
from dark_octet.data.validators import PaperValidation
from dark_octet.constants import ALPHA_PT, BETA_OVER_H, V_WALL, T_DARK_C_GEV


class TestGravitationalWaves:

    def test_f_peak_value(self):
        f_mhz = gw_peak_frequency_hz() * 1e3
        assert abs(f_mhz - 1.65) < 0.05

    def test_f_peak_in_lisa_band(self):
        f = gw_peak_frequency_hz()
        assert in_lisa_band(f)

    def test_omega_peak_order(self):
        omega = gw_peak_omega()
        assert 1e-10 < omega < 1e-6

    def test_gw_summary_keys(self):
        gw = gw_summary()
        for key in ["f_peak_mHz", "f_peak_Hz", "Omega_GW_peak", "in_LISA_band", "SNR_4yr"]:
            assert key in gw

    def test_snr_paper_value(self):
        gw = gw_summary()
        assert abs(gw["SNR_4yr"] - 8.5) < 1.5

    def test_f_peak_scales_with_tc(self):
        f1 = gw_peak_frequency_hz(T_c_GeV=1e15)
        f2 = gw_peak_frequency_hz(T_c_GeV=2e15)
        assert abs(f2 / f1 - 2.0) < 0.01

    def test_dao_r_value(self):
        dao = dao_scale()
        assert abs(dao["r_DAO_Mpc"] - 2.3) < 0.4

    def test_dao_ska_sigma(self):
        dao = dao_scale()
        assert dao["SKA_sigma"] > 5.0


class TestColliderSignatures:

    def test_fcce_xs_order(self):
        xs = sigma_epem_chi_p_chi_m_fb(365.0)
        assert 1e-5 < xs < 0.1

    def test_mono_lepton_endpoint(self):
        E = mono_lepton_endpoint_gev()
        assert abs(E - 0.85) < 0.1

    def test_fcce_summary_structure(self):
        s = fcce_summary()
        assert "N_observable" in s
        assert "sigma_fb" in s
        assert s["N_observable"] > 0

    def test_fcce_threshold(self):
        xs_365  = sigma_epem_chi_p_chi_m_fb(365.0)
        xs_200  = sigma_epem_chi_p_chi_m_fb(200.0)
        assert xs_365 > 0.0

    def test_collider_table_length(self):
        table = collider_table()
        assert len(table) == 3

    def test_fcc_hh_summary(self):
        s = fcc_hh_summary()
        assert "N_signal" in s


class TestValidation:

    def test_mass_spectrum_validation(self):
        pv = PaperValidation()
        r  = pv.validate_mass_spectrum()
        assert r["passed"]

    def test_mass_ratio_validation(self):
        pv = PaperValidation()
        r  = pv.validate_mass_ratios()
        assert r["passed"]

    def test_gw_validation(self):
        pv = PaperValidation()
        r  = pv.validate_gw_signature()
        assert r["passed"]

    def test_relic_validation(self):
        pv = PaperValidation()
        r  = pv.validate_relic_abundance()
        assert r["passed"]
