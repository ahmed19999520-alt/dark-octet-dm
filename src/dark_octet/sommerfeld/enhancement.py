import numpy as np
from ..constants import (
    G_DARK, ALPHA_DARK, M_ZP_GEV, M_ZP2, C_LIGHT_KM_S,
    GEV2_TO_CM2, GEV_PER_G,
)
from ..spectrum.gmo_formula import octet_spectrum
from .schrodinger import sommerfeld_yukawa, sommerfeld_coulomb_analytic


def xZp(v_rel_kms: float, mu_GeV: float, m_zp: float = M_ZP_GEV) -> float:
    v_frac = v_rel_kms / C_LIGHT_KM_S
    return m_zp ** 2 / (4.0 * mu_GeV ** 2 * v_frac ** 2)


def F_yukawa(x: float) -> float:
    if x < 1e-8:
        return 0.0
    return x * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))


def sigma_T_born_gevm2(
    v_rel_kms: float,
    m_chi0: float = None,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
) -> float:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    mu     = m_chi0 / 2.0
    v_frac = v_rel_kms / C_LIGHT_KM_S
    x      = xZp(v_rel_kms, mu, m_zp)
    F      = F_yukawa(x)
    alpha  = g_dark ** 2 / (4.0 * np.pi)
    return (4.0 * np.pi * alpha ** 2 * mu ** 2 / m_zp ** 2) * F / x


def sigma_T_sommerfeld_gevm2(
    v_rel_kms: float,
    m_chi0: float = None,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    alpha_dark: float = ALPHA_DARK,
    yukawa_suppression: float = 0.52,
) -> float:
    sigma_b = sigma_T_born_gevm2(v_rel_kms, m_chi0, g_dark, m_zp)
    S       = sommerfeld_yukawa(v_rel_kms, alpha_dark, m_zp, yukawa_suppression)
    return sigma_b * S


def sigma_T_per_mass(
    v_rel_kms: float,
    m_chi0: float = None,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    alpha_dark: float = ALPHA_DARK,
    yukawa_suppression: float = 0.52,
) -> float:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    sigma  = sigma_T_sommerfeld_gevm2(
        v_rel_kms, m_chi0, g_dark, m_zp, alpha_dark, yukawa_suppression
    )
    sigma_cm2   = sigma * GEV2_TO_CM2
    m_chi0_g    = m_chi0 / GEV_PER_G
    return sigma_cm2 / m_chi0_g


def v_critical_kms(
    m_chi0: float = None,
    m_zp: float = M_ZP_GEV,
) -> float:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    return (m_zp / m_chi0) * C_LIGHT_KM_S


OBSERVATIONAL_LIMITS = {
    "ultra_faint_dwarfs": {"v_kms": 5.0,    "limit_cm2_g": 1.0},
    "dwarf_spheroidals":  {"v_kms": 10.0,   "limit_cm2_g": 1.0},
    "milky_way_sats":     {"v_kms": 50.0,   "limit_cm2_g": 0.5},
    "galaxy_groups":      {"v_kms": 300.0,  "limit_cm2_g": 0.1},
    "galaxy_clusters":    {"v_kms": 1000.0, "limit_cm2_g": 0.47},
    "bullet_cluster":     {"v_kms": 1000.0, "limit_cm2_g": 0.47},
}

PAPER_VALUES = {
    5.0:    0.62,
    10.0:   0.47,
    300.0:  0.026,
    1000.0: 0.0044,
}


def sidm_table(
    m_chi0: float = None,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    alpha_dark: float = ALPHA_DARK,
) -> list:
    velocities = sorted(OBSERVATIONAL_LIMITS.values(), key=lambda d: d["v_kms"])
    rows = []
    for system, data in OBSERVATIONAL_LIMITS.items():
        v     = data["v_kms"]
        limit = data["limit_cm2_g"]
        sigma = sigma_T_per_mass(v, m_chi0, g_dark, m_zp, alpha_dark)
        S     = sommerfeld_yukawa(v, alpha_dark, m_zp)
        rows.append({
            "system":          system,
            "v_kms":           v,
            "sigma_T_m_code":  round(sigma, 5),
            "limit_cm2_g":     limit,
            "satisfies":       sigma < limit,
            "S_yukawa":        round(S, 1),
        })
    return rows


def all_bounds_satisfied(
    m_chi0: float = None,
    g_dark: float = G_DARK,
    m_zp: float = M_ZP_GEV,
    alpha_dark: float = ALPHA_DARK,
) -> bool:
    rows = sidm_table(m_chi0, g_dark, m_zp, alpha_dark)
    return all(r["satisfies"] for r in rows)


def plot_sigma_T(save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_arr   = np.logspace(0.4, 3.2, 300)
    sigma_v = np.array([sigma_T_per_mass(v) for v in v_arr])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(v_arr, sigma_v, "b-", lw=2.2, label="Model (this work)")

    ax.axhline(1.0,  color="orange", ls=":",  lw=1.5,
               label=r"Dwarf spheroidal limit ($<1.0$ cm$^2$g$^{-1}$)")
    ax.axhline(0.47, color="red",    ls="--", lw=1.5,
               label=r"Bullet Cluster limit ($<0.47$ cm$^2$g$^{-1}$)")
    ax.axhline(0.1,  color="green",  ls="-.", lw=1.2,
               label=r"Galaxy groups limit ($<0.1$ cm$^2$g$^{-1}$)")

    for v, val in PAPER_VALUES.items():
        ax.scatter(v, val, s=80, color="black", zorder=6)

    ax.set_xlabel(r"$v_{\rm rel}$ [km s$^{-1}$]", fontsize=13)
    ax.set_ylabel(r"$\sigma_T/m_{\chi^0}$ [cm$^2$ g$^{-1}$]", fontsize=12)
    ax.set_title(
        r"Self-interaction cross section with Sommerfeld enhancement",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, which="both", ls="--", alpha=0.35)
    ax.xaxis.grid(True, which="both", ls="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
