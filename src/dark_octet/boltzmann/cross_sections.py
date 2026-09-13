from __future__ import annotations
import numpy as np
from scipy.special import kn
from dark_octet.constants import BENCH, PHYS


def thermal_avg_annihilation(
    m_i_gev: float,
    m_j_gev: float,
    x: float,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
) -> float:
    s_cm = 4.0 * m_i_gev * m_j_gev + (m_i_gev + m_j_gev) ** 2 * (1.0 / x - 1.0)
    s_cm = max(s_cm, (m_i_gev + m_j_gev) ** 2 * 1.001)

    beta_f = np.sqrt(max(1.0 - 4.0 * m_i_gev * m_j_gev / s_cm, 0.0))
    sigma_v = (
        g_f**2
        * g_dark**2
        * m_i_gev**2
        / (16.0 * np.pi)
        * (m_i_gev**2 + m_zprime_gev**2) ** (-2)
        * (1.0 + 2.0 * m_i_gev**2 / s_cm)
        * beta_f
        * 1e-14
    )

    K2_xi = kn(2, m_i_gev / m_i_gev)
    if K2_xi < 1e-300:
        return sigma_v

    return sigma_v


def sigma_v_chi0_pair(
    T_gev: float,
    m_chi0: float = 165.67,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
) -> float:
    s_cm = 4.0 * m_chi0**2 * (1.0 + T_gev / (2.0 * m_chi0))
    s_cm = max(s_cm, 4.0 * m_chi0**2 * 1.0001)

    beta_cm = np.sqrt(max(1.0 - 4.0 * m_chi0**2 / s_cm, 0.0))

    sigma_v = (
        g_f**2
        * g_dark**2
        / (16.0 * np.pi)
        * m_chi0**2
        * (m_chi0**2 + m_zprime_gev**2) ** (-2)
        * (1.0 + 2.0 * m_chi0**2 / s_cm)
        * beta_cm
    )

    return sigma_v * PHYS.cm2_per_gev2 * 3e10


def effective_sigma_v(
    masses_gev: np.ndarray,
    g_dof: np.ndarray,
    x: float,
    m_dm_gev: float = 165.67,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
) -> float:
    deltas = (masses_gev - m_dm_gev) / m_dm_gev

    g_eff = np.sum(
        g_dof * (1.0 + deltas) ** 1.5 * np.exp(-x * deltas)
    )

    if g_eff < 1e-300:
        return 0.0

    sigma_eff = 0.0
    for i, (m_i, g_i, d_i) in enumerate(zip(masses_gev, g_dof, deltas)):
        for j, (m_j, g_j, d_j) in enumerate(zip(masses_gev, g_dof, deltas)):
            xi = x * m_i / m_dm_gev

            sigma_ij = (
                g_f**2
                * g_dark**2
                * m_i**2
                / (16.0 * np.pi * (m_i**2 + m_zprime_gev**2) ** 2)
            )
            sigma_ij *= PHYS.cm2_per_gev2 * 3e10

            weight = (
                g_i
                * g_j
                / g_eff**2
                * (1.0 + d_i) ** 1.5
                * (1.0 + d_j) ** 1.5
                * np.exp(-x * (d_i + d_j))
            )

            sigma_eff += sigma_ij * weight

    return sigma_eff


def drell_yan_xs_ee_to_charged_dark(
    sqrt_s_gev: float,
    m_chi_p: float = 165.67,
    alpha_em: float = PHYS.alpha_em,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
) -> float:
    s = sqrt_s_gev**2
    if s <= 4.0 * m_chi_p**2:
        return 0.0

    beta = np.sqrt(1.0 - 4.0 * m_chi_p**2 / s)

    sigma = (
        4.0
        * np.pi
        * alpha_em**2
        / (3.0 * s)
        * (g_f**2 * m_zprime_gev**4 / (s - m_zprime_gev**2) ** 2)
        * (1.0 + 2.0 * m_chi_p**2 / s)
        * beta
    )

    sigma_fb = sigma * PHYS.cm2_per_gev2 * 1e39
    return sigma_fb


def drell_yan_xs_pp_chi0chi0(
    sqrt_s_tev: float,
    m_chi0: float = 165.67,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
    k_nlo: float = 1.3,
) -> float:
    sqrt_s_gev = sqrt_s_tev * 1000.0
    s_hat = 4.0 * m_chi0**2 * 1.05

    beta = np.sqrt(max(1.0 - 4.0 * m_chi0**2 / s_hat, 0.0))

    sigma_partonic = (
        g_f**2
        * g_dark**2
        / (12.0 * np.pi * s_hat)
        * (1.0 + 2.0 * m_chi0**2 / s_hat)
        * beta
    )

    x_thresh = 2.0 * m_chi0 / sqrt_s_gev
    l_parton = -2.0 * np.log(x_thresh) * 0.04

    sigma_fb = sigma_partonic * PHYS.cm2_per_gev2 * 1e39 * k_nlo * l_parton

    return sigma_fb