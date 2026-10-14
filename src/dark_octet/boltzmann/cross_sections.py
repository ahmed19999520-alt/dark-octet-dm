import numpy as np
from ..constants import (
    G_DARK, ALPHA_DARK, G_F, M_ZP_GEV, M_ZP2, M_ZP4,
    ALPHA_EM, GEV2_TO_CM2,
)
from ..spectrum.gmo_formula import octet_spectrum


def sigma_v_chi0_chi0(
    m_chi0: float,
    T: float,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
) -> float:
    s_cm   = 4.0 * m_chi0 ** 2
    denom  = (s_cm + m_zp ** 2) ** 2
    prefac = g_f ** 2 * g_dark ** 2 / (16.0 * np.pi)
    return prefac * m_chi0 ** 2 / denom


def sigma_v_chi0_chi0_thermal(
    m_chi0: float,
    x: float,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    n_sm_fermions: int = 45,
) -> float:
    s_cm   = 4.0 * m_chi0 ** 2 * (1.0 + 1.5 / x)
    denom  = (s_cm + m_zp ** 2) ** 2
    prefac = g_f ** 2 * g_dark ** 2 * n_sm_fermions / (16.0 * np.pi)
    beta   = np.sqrt(1.0 - 4.0 * m_chi0 ** 2 / s_cm)
    return prefac * s_cm * beta / denom


def sigma_v_pair(
    i: str,
    j: str,
    masses: dict,
    x: float,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    n_sm: int = 45,
) -> float:
    mi = masses[i]
    mj = masses[j]
    s  = (mi + mj) ** 2 * (1.0 + 1.5 / x)
    if s < (mi + mj) ** 2:
        return 0.0
    denom  = (s + m_zp ** 2) ** 2
    prefac = g_f ** 2 * g_dark ** 2 * n_sm / (16.0 * np.pi)
    mu2    = mi * mj
    beta   = np.sqrt(max(0.0, 1.0 - (mi + mj) ** 2 / s))
    return prefac * mu2 / denom * beta


def griest_seckel_geff(masses: dict, m0: float, x: float, gdof: dict) -> float:
    total = 0.0
    for name, g in gdof.items():
        delta = (masses[name] - m0) / masses["chi0"]
        xi    = x * (1.0 + delta)
        total += g * (1.0 + delta) ** 1.5 * np.exp(-x * delta)
    return total


def griest_seckel_sigma_eff(
    masses: dict,
    gdof: dict,
    x: float,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
) -> float:
    m0    = masses["chi0"]
    geff  = griest_seckel_geff(masses, m0, x, gdof)
    names = list(masses.keys())
    sigma_eff = 0.0
    for ni in names:
        for nj in names:
            gi     = gdof[ni]
            gj     = gdof[nj]
            di     = (masses[ni] - m0) / m0
            dj     = (masses[nj] - m0) / m0
            sij    = sigma_v_pair(ni, nj, masses, x, g_dark, g_f, m_zp)
            wi     = gi * (1.0 + di) ** 1.5 * np.exp(-x * di)
            wj     = gj * (1.0 + dj) ** 1.5 * np.exp(-x * dj)
            sigma_eff += sij * wi * wj / geff ** 2
    return sigma_eff


def sigma_epem_chi_chi(
    sqrt_s: float,
    m_chi: float,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    alpha_em: float = ALPHA_EM,
) -> float:
    s = sqrt_s ** 2
    if s < 4.0 * m_chi ** 2:
        return 0.0
    beta  = np.sqrt(1.0 - 4.0 * m_chi ** 2 / s)
    denom = (s - m_zp ** 2) ** 2
    prefac = (4.0 * np.pi * alpha_em ** 2 / (3.0 * s))
    portal = g_f ** 2 * m_zp ** 4 / denom
    return prefac * portal * (1.0 + 2.0 * m_chi ** 2 / s) * beta


def n_signal_fcce(
    sqrt_s: float = 365.0,
    luminosity_ab: float = 1.5,
    m_chi: float = None,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
) -> float:
    masses = octet_spectrum()
    if m_chi is None:
        m_chi = masses["chi_p"]
    lumi_fb = luminosity_ab * 1.0e3
    xs_fb   = sigma_epem_chi_chi(sqrt_s, m_chi, g_f, m_zp) / GEV2_TO_CM2 * 1e-27
    return xs_fb * lumi_fb


def sigma_pp_chi0_chi0_drell_yan(
    sqrt_s_hat: float,
    m_chi0: float,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    k_qcd: float = 1.3,
) -> float:
    s_hat = sqrt_s_hat ** 2
    if s_hat < 4.0 * m_chi0 ** 2:
        return 0.0
    beta  = np.sqrt(1.0 - 4.0 * m_chi0 ** 2 / s_hat)
    denom = (s_hat - m_zp ** 2) ** 2 + (m_zp * 0.01) ** 2
    prefac = g_f ** 2 * g_dark ** 2 / (12.0 * np.pi * s_hat)
    return prefac * (1.0 + 2.0 * m_chi0 ** 2 / s_hat) * beta * k_qcd / denom * s_hat


def sigma_si_kinetic_mixing(
    m_chi0: float = None,
    g_f: float = G_F,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    f_p: float = 0.30,
    Z: int = 54,
    A: int = 131,
    m_N: float = 0.9389,
) -> float:
    masses = octet_spectrum()
    if m_chi0 is None:
        m_chi0 = masses["chi0"]
    mu_chN = m_chi0 * m_N / (m_chi0 + m_N)
    numerator = mu_chN ** 2 * g_f ** 2 * g_dark ** 2 * Z ** 2 * f_p ** 2
    denominator = np.pi * A ** 2 * m_zp ** 4
    return numerator / denominator * GEV2_TO_CM2
