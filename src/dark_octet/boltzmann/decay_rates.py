from __future__ import annotations
import numpy as np
from dark_octet.constants import BENCH, PHYS


def decay_rate_two_body(
    m_parent_gev: float,
    m_dm_gev: float,
    m_phi_gev: float = 0.01,
    g_dark: float = BENCH.g_dark,
) -> float:
    delta_m = m_parent_gev - m_dm_gev

    if delta_m <= m_phi_gev:
        return 0.0

    kin_factor = max(
        1.0 - (m_dm_gev + m_phi_gev) ** 2 / m_parent_gev**2,
        0.0,
    ) ** 1.5

    gamma = g_dark**2 * delta_m**2 / (8.0 * np.pi * m_parent_gev**3) * kin_factor

    return gamma


def lifetime_seconds(decay_rate_gev: float) -> float:
    if decay_rate_gev <= 0.0:
        return np.inf
    return PHYS.hbar_gev_s / decay_rate_gev


def compute_all_decay_rates(
    masses_gev: dict[str, float],
    m_dm_gev: float = 165.67,
    m_phi_gev: float = 0.01,
    g_dark: float = BENCH.g_dark,
) -> dict[str, dict]:
    results = {}
    for name, mass in masses_gev.items():
        if name == "chi_0":
            results[name] = {
                "mass_gev": mass,
                "delta_m_gev": 0.0,
                "decay_rate_gev": 0.0,
                "lifetime_s": np.inf,
                "stable": True,
            }
            continue

        rate = decay_rate_two_body(
            m_parent_gev=mass,
            m_dm_gev=m_dm_gev,
            m_phi_gev=m_phi_gev,
            g_dark=g_dark,
        )
        tau_s = lifetime_seconds(rate)

        results[name] = {
            "mass_gev": mass,
            "delta_m_gev": mass - m_dm_gev,
            "decay_rate_gev": rate,
            "lifetime_s": tau_s,
            "stable": False,
        }

    return results


REFERENCE_DECAY_RATES = {
    "chi_pp":  {"expected_gev": 3.46e-7, "tau_s_expected": 1.90e-18},
    "chi_p":   {"expected_gev": 0.0,     "tau_s_expected": np.inf},
    "chi_0":   {"expected_gev": 0.0,     "tau_s_expected": np.inf},
    "chi_m":   {"expected_gev": 1.08e-8, "tau_s_expected": 6.12e-17},
    "Sigma_p": {"expected_gev": 3.14e-4, "tau_s_expected": 2.09e-21},
    "Sigma_0": {"expected_gev": 8.42e-4, "tau_s_expected": 7.82e-22},
    "Sigma_m": {"expected_gev": 3.98e-4, "tau_s_expected": 1.66e-21},
    "Lambda":  {"expected_gev": 8.42e-4, "tau_s_expected": 7.82e-22},
}