import pytest
import numpy as np
from dark_octet.signatures.collider import (
    build_collider_table,
    compute_endpoint_energy,
    compute_r21_from_endpoint,
    drell_yan_xs_ee_to_charged_dark,
    drell_yan_xs_pp_chi0chi0,
)
from dark_octet.signatures.gravitational_waves import GravitationalWaveSpectrum


def test_endpoint_energy_chi_plus():
    E_max = compute_endpoint_energy(165.67, 165.67, m_phi_gev=0.01)
    assert E_max == pytest.approx(0.0, abs=0.1)


def test_endpoint_energy_chi_pp():
    E_max = compute_endpoint_energy(172.67, 165.67, m_phi_gev=0.01)
    assert 2.0 < E_max < 5.0


def test_r21_from_endpoint():
    E_max = compute_endpoint_energy(172.67, 165.67)
    r21 = compute_r21_from_endpoint(E_max, m_dm_gev=165.67)
    assert abs(r21 - 1.042) < 0.05


def test_drell_yan_ee_365gev():
    sigma = drell_yan_xs_ee_to_charged_dark(sqrt_s_gev=365.0, m_chi_p=165.67)
    assert 1e-5 < sigma < 1e-1


def test_drell_yan_ee_below_threshold():
    sigma = drell_yan_xs_ee_to_charged_dark(sqrt_s_gev=300.0, m_chi_p=165.67)
    assert sigma == pytest.approx(0.0, abs=1e-20)


def test_drell_yan_pp_100tev():
    sigma = drell_yan_xs_pp_chi0chi0(sqrt_s_tev=100.0)
    assert 0.05 < sigma < 5.0


def test_collider_table_has_5_rows():
    df = build_collider_table()
    assert len(df) == 5


def test_collider_table_columns():
    df = build_collider_table()
    assert "Process" in df.columns
    assert "σ [fb]" in df.columns
    assert "S/√B [σ]" in df.columns


def test_gw_peak_frequency():
    gws = GravitationalWaveSpectrum()
    assert 0.5e-3 < gws.f_peak_hz < 5e-3


def test_gw_peak_amplitude():
    gws = GravitationalWaveSpectrum()
    assert 1e-10 < gws.omega_peak < 1e-6


def test_gw_snr_above_threshold():
    gws = GravitationalWaveSpectrum()
    snr = gws.compute_snr(T_obs_years=4.0)
    assert snr > 5.0


def test_gw_omega_array():
    gws = GravitationalWaveSpectrum()
    f = np.logspace(-4, -1, 50)
    omega = gws.omega_gw(f)
    assert omega.shape == (50,)
    assert omega.max() > 0.0
    assert np.all(omega >= 0.0)


def test_gw_spectral_shape_below_peak():
    gws = GravitationalWaveSpectrum()
    f_low = gws.f_peak_hz * 0.1
    f_high = gws.f_peak_hz * 10.0
    f_arr = np.array([f_low, gws.f_peak_hz, f_high])
    omega = gws.omega_gw(f_arr)
    assert omega[1] >= omega[0]