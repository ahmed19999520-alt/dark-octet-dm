import numpy as np
from scipy.integrate import solve_ivp
from ..constants import ALPHA_DARK, M_ZP_GEV, C_LIGHT_KM_S
from ..spectrum.gmo_formula import octet_spectrum


def dimensionless_params(
    v_rel_kms: float,
    alpha_dark: float = ALPHA_DARK,
    m_zp: float = M_ZP_GEV,
    m_chi0: float = None,
) -> dict:
    if m_chi0 is None:
        m_chi0 = octet_spectrum()["chi0"]
    v_frac = v_rel_kms / C_LIGHT_KM_S
    mu     = m_chi0 / 2.0
    eps_v  = v_frac / (2.0 * alpha_dark)
    eps_phi = m_zp / (alpha_dark * mu)
    k       = v_frac * mu / m_zp
    return {
        "v_frac":   v_frac,
        "mu_GeV":   mu,
        "eps_v":    eps_v,
        "eps_phi":  eps_phi,
        "k":        k,
        "alpha_dark": alpha_dark,
        "m_zp_GeV": m_zp,
    }


def schrodinger_rhs(xi: float, u: np.ndarray, eps_v: float, eps_phi: float) -> list:
    u_re, du_re, u_im, du_im = u
    xi_s    = max(xi, 1e-12)
    pot     = np.exp(-xi_s) / (eps_phi * xi_s)
    k2      = (eps_v / eps_phi) ** 2
    d2u_re  = -(k2 + pot) * u_re
    d2u_im  = -(k2 + pot) * u_im
    return [du_re, d2u_re, du_im, d2u_im]


def sommerfeld_enhancement_numerical(
    v_rel_kms: float,
    alpha_dark: float = ALPHA_DARK,
    m_zp: float = M_ZP_GEV,
    m_chi0: float = None,
    xi_max: float = 200.0,
    n_points: int = 4000,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> float:
    p  = dimensionless_params(v_rel_kms, alpha_dark, m_zp, m_chi0)
    eps_v   = p["eps_v"]
    eps_phi = p["eps_phi"]
    k       = p["k"]

    xi_span = (xi_max, 1e-6)
    xi_eval = np.linspace(xi_max, 1e-6, n_points)

    phi0  = k * xi_max
    u0_re = np.cos(phi0)
    u0_im = np.sin(phi0)
    d0_re = -k * np.sin(phi0)
    d0_im =  k * np.cos(phi0)

    sol = solve_ivp(
        schrodinger_rhs,
        xi_span,
        [u0_re, d0_re, u0_im, d0_im],
        args=(eps_v, eps_phi),
        method="DOP853",
        t_eval=xi_eval,
        rtol=rtol,
        atol=atol,
    )

    u_re_0 = sol.y[0, -1]
    u_im_0 = sol.y[2, -1]
    u_sq_0 = u_re_0 ** 2 + u_im_0 ** 2

    free_sq = (k * 1e-6) ** 2 + 1.0
    if free_sq < 1e-30:
        return 1.0

    S = u_sq_0 / (k + 1e-40) ** 2 * k ** 2
    return max(S, 1.0)


def sommerfeld_coulomb_analytic(
    v_rel_kms: float,
    alpha_dark: float = ALPHA_DARK,
) -> float:
    v_frac = v_rel_kms / C_LIGHT_KM_S
    arg    = 2.0 * np.pi * alpha_dark / v_frac
    if arg > 700.0:
        return arg
    return arg / (1.0 - np.exp(-arg))


def yukawa_suppression_factor(
    v_rel_kms: float,
    alpha_dark: float = ALPHA_DARK,
    m_zp: float = M_ZP_GEV,
    m_chi0: float = None,
    v_ref_kms: float = 10.0,
    s_ref: float = 0.52,
) -> float:
    s_c_v   = sommerfeld_coulomb_analytic(v_rel_kms, alpha_dark)
    s_c_ref = sommerfeld_coulomb_analytic(v_ref_kms, alpha_dark)
    return s_ref * s_c_v / s_c_ref if s_c_ref > 0 else s_ref


def sommerfeld_yukawa(
    v_rel_kms: float,
    alpha_dark: float = ALPHA_DARK,
    m_zp: float = M_ZP_GEV,
    suppression: float = 0.52,
) -> float:
    s_c = sommerfeld_coulomb_analytic(v_rel_kms, alpha_dark)
    return suppression * s_c


def sommerfeld_table(
    velocities_kms: list = None,
    alpha_dark: float = ALPHA_DARK,
    m_zp: float = M_ZP_GEV,
) -> list:
    if velocities_kms is None:
        velocities_kms = [5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0]
    rows = []
    for v in velocities_kms:
        s_c   = sommerfeld_coulomb_analytic(v, alpha_dark)
        s_yuk = sommerfeld_yukawa(v, alpha_dark, m_zp)
        rows.append({
            "v_kms":         v,
            "S_coulomb":     round(s_c, 2),
            "S_yukawa_0.52": round(s_yuk, 2),
        })
    return rows
