import numpy as np
from ..constants import G_DARK, GEV_TO_S, OCTET_ORDER
from ..spectrum.gmo_formula import octet_spectrum, mass_splittings


def decay_rate_two_body(
    m_parent: float,
    m_daughter: float,
    delta_m: float,
    g_dark: float = G_DARK,
    m_phi: float = 1e-4,
) -> float:
    if delta_m <= m_phi:
        return 0.0
    if m_parent <= m_daughter + m_phi:
        return 0.0
    dm_ratio = delta_m / m_parent
    norm = 4.0 * np.pi if dm_ratio < 0.04 else 8.0 * np.pi
    return g_dark ** 2 * delta_m ** 2 / (norm * m_parent ** 3)


def all_decay_rates(
    masses: dict = None,
    g_dark: float = G_DARK,
    m_phi: float = 1e-4,
) -> dict:
    if masses is None:
        masses = octet_spectrum()
    m0      = masses["chi0"]
    splits  = mass_splittings(masses)
    rates   = {}
    for name in OCTET_ORDER:
        if name == "chi0":
            rates[name] = 0.0
            continue
        dm = splits[name]
        rates[name] = decay_rate_two_body(
            masses[name], m0, dm, g_dark, m_phi
        )
    return rates


def all_lifetimes_s(masses: dict = None, g_dark: float = G_DARK) -> dict:
    rates = all_decay_rates(masses, g_dark)
    lifetimes = {}
    for name, gamma in rates.items():
        if gamma == 0.0:
            lifetimes[name] = float("inf")
        else:
            lifetimes[name] = GEV_TO_S / gamma
    return lifetimes


def print_lifetime_table(masses: dict = None) -> None:
    if masses is None:
        masses = octet_spectrum()
    rates     = all_decay_rates(masses)
    lifetimes = all_lifetimes_s(masses)
    print(f"{'State':<12} {'Gamma [GeV]':>14} {'tau [s]':>14}")
    print("-" * 44)
    for name in OCTET_ORDER:
        g = rates[name]
        t = lifetimes[name]
        if name == "chi0":
            print(f"{name:<12} {'stable':>14} {'inf':>14}")
        else:
            print(f"{name:<12} {g:>14.4e} {t:>14.4e}")


def bbn_safety_check(masses: dict = None) -> dict:
    lifetimes = all_lifetimes_s(masses)
    bbn_scale = 1e-3
    results   = {}
    for name, tau in lifetimes.items():
        if name == "chi0":
            results[name] = {"tau_s": float("inf"), "before_bbn": True}
        else:
            results[name] = {
                "tau_s":      tau,
                "before_bbn": tau < bbn_scale,
            }
    return results
