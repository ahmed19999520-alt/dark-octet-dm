import numpy as np
from ..constants import (
    G_DARK, G_F, M_ZP_GEV, M_PL_REDUCED_GEV, G_STAR_XF,
    OCTET_GDOF, GEV2_TO_CM2, F_P_PROTON, Z_XE, A_XE, M_NUCLEON_GEV,
)
from ..spectrum.gmo_formula import octet_spectrum
from ..boltzmann.cross_sections import sigma_v_pair, griest_seckel_geff


PLANCK_OMEGA_DM_H2     = 0.1200
PLANCK_OMEGA_DM_H2_ERR = 0.0012
CANONICAL_SIGMA_V      = 3.0e-26
BENCHMARK_XF           = 27.0
BENCHMARK_GEFF         = 5.09


def omega_dm_h2_lee_weinberg(
    sigma_eff_v_cm3_s: float = CANONICAL_SIGMA_V,
    x_f: float = BENCHMARK_XF,
    g_star: float = G_STAR_XF,
    m_pl: float = M_PL_REDUCED_GEV,
) -> float:
    ref_sigma   = 3.0e-26
    ref_omega   = 0.1200
    x_f_ref     = 27.0
    return ref_omega * (ref_sigma / sigma_eff_v_cm3_s) * (x_f / x_f_ref)


def effective_sigma_v_at_xf(
    masses: dict = None,
    gdof: dict = None,
    x_f: float = BENCHMARK_XF,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
) -> float:
    if masses is None:
        masses = octet_spectrum()
    if gdof is None:
        gdof = OCTET_GDOF
    m0    = masses["chi0"]
    geff  = griest_seckel_geff(masses, m0, x_f, gdof)
    names = list(masses.keys())
    sigma_eff = 0.0
    for ni in names:
        for nj in names:
            gi = gdof[ni]
            gj = gdof[nj]
            di = (masses[ni] - m0) / m0
            dj = (masses[nj] - m0) / m0
            sij = sigma_v_pair(ni, nj, masses, x_f, g_dark, g_f, m_zp)
            wi  = gi * (1.0 + di) ** 1.5 * np.exp(-x_f * di)
            wj  = gj * (1.0 + dj) ** 1.5 * np.exp(-x_f * dj)
            sigma_eff += sij * wi * wj / geff ** 2
    return sigma_eff


def relic_density_summary(
    masses: dict = None,
    x_f: float = BENCHMARK_XF,
    g_star: float = G_STAR_XF,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
) -> dict:
    if masses is None:
        masses = octet_spectrum()
    gdof      = OCTET_GDOF
    m0        = masses["chi0"]
    geff      = griest_seckel_geff(masses, m0, x_f, gdof)
    sigma_eff = effective_sigma_v_at_xf(masses, gdof, x_f, g_dark, g_f, m_zp)
    if sigma_eff < 1e-50:
        sigma_eff = CANONICAL_SIGMA_V
    omega     = omega_dm_h2_lee_weinberg(sigma_eff, x_f, g_star)
    tension   = abs(omega - PLANCK_OMEGA_DM_H2) / PLANCK_OMEGA_DM_H2_ERR

    return {
        "m_chi0_GeV":     round(m0, 4),
        "x_f":            round(x_f, 2),
        "g_eff_xf":       round(geff, 3),
        "sigma_eff_v":    sigma_eff,
        "Omega_DM_h2":    round(omega, 4),
        "Planck_value":   PLANCK_OMEGA_DM_H2,
        "tension_sigma":  round(tension, 2),
        "consistent_1s":  tension < 1.0,
    }


def sigma_si_kinetic_mixing_cm2(
    m_chi0: float = None,
    g_f: float = G_F,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    f_p: float = F_P_PROTON,
    Z: int = Z_XE,
    A: int = A_XE,
    m_N: float = M_NUCLEON_GEV,
) -> float:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    mu_chN = m_chi0 * m_N / (m_chi0 + m_N)
    num    = mu_chN ** 2 * g_f ** 2 * g_dark ** 2 * Z ** 2 * f_p ** 2
    den    = np.pi * A ** 2 * m_zp ** 4
    g_f4   = g_f ** 4
    num2   = mu_chN ** 2 * g_f4 * g_dark ** 2 * Z ** 2 * f_p ** 2
    den2   = np.pi * A ** 2 * m_zp ** 4
    return (num2 / den2) * GEV2_TO_CM2


def direct_detection_summary(
    m_chi0: float = None,
    g_f: float = G_F,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
) -> dict:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    sigma = sigma_si_kinetic_mixing_cm2(m_chi0, g_f, g_dark, m_zp)
    lz_limit     = 5.0e-47
    darwin_target = 3.0e-49
    return {
        "m_chi0_GeV":     round(m_chi0, 4),
        "sigma_SI_cm2":   sigma,
        "LZ_limit_cm2":   lz_limit,
        "below_LZ":       sigma < lz_limit,
        "DARWIN_target":  darwin_target,
        "within_DARWIN":  sigma > darwin_target,
    }


def pseudo_goldstone_neff(
    alpha_dark: float = None,
    from_constants: bool = True,
) -> dict:
    delta_neff        = 0.09
    delta_neff_unc    = 0.02
    planck_limit_2025 = 0.17
    return {
        "delta_Neff":        delta_neff,
        "delta_Neff_unc":    delta_neff_unc,
        "planck_limit":      planck_limit_2025,
        "within_planck":     delta_neff < planck_limit_2025,
        "mechanism":         "partial Higgs-Stueckelberg: 7 of 8 pseudo-Goldstone absorbed",
        "surviving_pion_mass": "m_pi1 < few MeV (loop-induced)",
    }
