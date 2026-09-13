from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from dark_octet.constants import BENCH


@dataclass
class SchrodingerResult:
    xi_array: np.ndarray
    u_real: np.ndarray
    u_imag: np.ndarray
    u_prime_real: np.ndarray
    u_prime_imag: np.ndarray
    psi_at_origin: complex
    enhancement_s_wave: float
    epsilon_v: float
    epsilon_phi: float


def solve_schrodinger_bvp(
    v_rel: float,
    alpha_dark: float = BENCH.alpha_dark,
    m_dm_gev: float = 165.67,
    m_zprime_gev: float = BENCH.m_zprime_gev,
    xi_max: float = 100.0,
    n_grid: int = 10000,
    ell: int = 0,
) -> SchrodingerResult:
    mu = m_dm_gev / 2.0
    epsilon_v = v_rel / (2.0 * alpha_dark)
    epsilon_phi = m_zprime_gev / (alpha_dark * mu)

    k_sq = (epsilon_v / epsilon_phi) ** 2

    def rhs_real(xi: float, state: np.ndarray) -> np.ndarray:
        u_r, du_r, u_i, du_i = state

        if xi < 1e-10:
            return [du_r, 0.0, du_i, 0.0]

        yukawa = np.exp(-xi) / (epsilon_phi * xi)
        centrifugal = ell * (ell + 1) / xi**2 if xi > 0 else 0.0

        potential_term = k_sq + yukawa - centrifugal

        d2u_r = -potential_term * u_r
        d2u_i = -potential_term * u_i

        return [du_r, d2u_r, du_i, d2u_i]

    k_mag = np.sqrt(abs(k_sq)) if k_sq > 0 else 1e-10

    xi_init = xi_max
    u_r_init = np.cos(k_mag * xi_max)
    u_i_init = np.sin(k_mag * xi_max)
    du_r_init = -k_mag * np.sin(k_mag * xi_max)
    du_i_init = k_mag * np.cos(k_mag * xi_max)

    state0 = [u_r_init, du_r_init, u_i_init, du_i_init]

    xi_span = (xi_max, 1e-4)
    xi_eval = np.linspace(xi_max, 1e-4, n_grid)

    sol = solve_ivp(
        rhs_real,
        xi_span,
        state0,
        t_eval=xi_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-13,
        max_step=0.01,
    )

    xi_sorted = sol.t[::-1]
    u_r_sorted = sol.y[0][::-1]
    u_i_sorted = sol.y[2][::-1]
    du_r_sorted = sol.y[1][::-1]
    du_i_sorted = sol.y[3][::-1]

    psi_origin = complex(u_r_sorted[0], u_i_sorted[0])
    psi_free = complex(np.cos(0.0), np.sin(0.0))

    amp_ratio = abs(psi_origin) / (abs(psi_free) + 1e-30)
    S = amp_ratio**2 / (v_rel + 1e-30)

    return SchrodingerResult(
        xi_array=xi_sorted,
        u_real=u_r_sorted,
        u_imag=u_i_sorted,
        u_prime_real=du_r_sorted,
        u_prime_imag=du_i_sorted,
        psi_at_origin=psi_origin,
        enhancement_s_wave=S,
        epsilon_v=epsilon_v,
        epsilon_phi=epsilon_phi,
    )


def coulomb_enhancement(
    v_rel: float,
    alpha_dark: float = BENCH.alpha_dark,
) -> float:
    x_c = 2.0 * np.pi * alpha_dark / v_rel
    if x_c > 700.0:
        return x_c
    return x_c / (1.0 - np.exp(-x_c))


def yukawa_form_factor(
    v_rel: float,
    m_zprime_gev: float = BENCH.m_zprime_gev,
    m_dm_gev: float = 165.67,
    alpha_dark: float = BENCH.alpha_dark,
) -> float:
    mu = m_dm_gev / 2.0
    x_zp = m_zprime_gev**2 / (4.0 * mu**2 * v_rel**2)

    if x_zp < 1e-10:
        return 0.5

    F = x_zp * (np.log(1.0 + 1.0 / x_zp) - 1.0 / (1.0 + x_zp))
    return F