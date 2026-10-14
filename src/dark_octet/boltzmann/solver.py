import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kn
from ..constants import (
    G_DARK, G_F, M_ZP_GEV, M_PL_REDUCED_GEV,
    G_STAR_XF, G_STAR_S_XF, OCTET_GDOF, OCTET_ORDER,
)
from ..spectrum.gmo_formula import octet_spectrum, mass_splittings
from .cross_sections import sigma_v_pair, griest_seckel_geff
from .decay_rates import all_decay_rates


def entropy_density(T_GeV: float, g_star_s: float = G_STAR_S_XF) -> float:
    return (2.0 * np.pi ** 2 / 45.0) * g_star_s * T_GeV ** 3


def hubble_rate(T_GeV: float, g_star: float = G_STAR_XF) -> float:
    return np.sqrt(np.pi ** 2 * g_star / 90.0) * T_GeV ** 2 / M_PL_REDUCED_GEV


def yield_equilibrium(
    name: str,
    x: float,
    masses: dict,
    gdof: dict = OCTET_GDOF,
    g_star_s: float = G_STAR_S_XF,
) -> float:
    m  = masses[name]
    m0 = masses["chi0"]
    xi = x * m / m0
    g  = gdof[name]
    bessel = kn(2, xi)
    if bessel < 1e-300:
        return 0.0
    return (45.0 / (4.0 * np.pi ** 4)) * (g / g_star_s) * xi ** 2 * bessel


def boltzmann_rhs(
    x: float,
    Y: np.ndarray,
    masses: dict,
    gdof: dict,
    rates: dict,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    g_star: float = G_STAR_XF,
    g_star_s: float = G_STAR_S_XF,
) -> np.ndarray:
    names  = OCTET_ORDER
    m0     = masses["chi0"]
    T_GeV  = m0 / x
    H      = hubble_rate(T_GeV, g_star)
    s      = entropy_density(T_GeV, g_star_s)
    factor = s / (H * x)

    Y_eq = np.array([
        yield_equilibrium(n, x, masses, gdof, g_star_s) for n in names
    ])

    dY = np.zeros(len(names))
    for i, ni in enumerate(names):
        annihilation_sum = 0.0
        for j, nj in enumerate(names):
            sij = sigma_v_pair(ni, nj, masses, x, g_dark, g_f, m_zp)
            Yeq_i = Y_eq[i]
            Yeq_j = Y_eq[j]
            if sij > 0.0 and Yeq_i > 0.0 and Yeq_j > 0.0:
                annihilation_sum += sij * (
                    Y[i] * Y[j] - Yeq_i * Yeq_j
                )
        dY[i] -= factor * annihilation_sum

        for j, nj in enumerate(names):
            if j == i:
                continue
            gij = rates.get(nj, 0.0)
            gji = rates.get(ni, 0.0)
            Yeq_j = Y_eq[j]
            Yeq_i = Y_eq[i]
            if Yeq_i > 1e-300:
                dY[i] += (gij * Y[j] - gji * Y[i]) / (H * x)

    return dY


def solve_boltzmann(
    x_init: float = 1.0,
    x_fin: float = 1000.0,
    n_eval: int = 2000,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict:
    masses = octet_spectrum()
    gdof   = OCTET_GDOF
    rates  = all_decay_rates(masses, g_dark)
    names  = OCTET_ORDER

    Y0 = np.array([
        yield_equilibrium(n, x_init, masses, gdof) for n in names
    ])

    x_span = (x_init, x_fin)
    x_eval = np.logspace(np.log10(x_init), np.log10(x_fin), n_eval)

    sol = solve_ivp(
        boltzmann_rhs,
        x_span,
        Y0,
        args=(masses, gdof, rates, g_dark, g_f, m_zp),
        method="BDF",
        t_eval=x_eval,
        rtol=rtol,
        atol=atol,
        jac=None,
        dense_output=False,
    )

    x_arr  = sol.t
    Y_arr  = sol.y
    Y_chi0 = Y_arr[0]
    Y_inf  = float(Y_chi0[-1])

    return {
        "x":       x_arr,
        "Y":       Y_arr,
        "Y_chi0":  Y_chi0,
        "Y_inf":   Y_inf,
        "names":   names,
        "masses":  masses,
        "success": sol.success,
    }


def relic_abundance_lee_weinberg(
    sigma_eff_v: float,
    x_f: float,
    g_star: float = G_STAR_XF,
    m_pl: float = M_PL_REDUCED_GEV,
) -> float:
    C = 2.874e8
    return C * x_f / (m_pl * np.sqrt(g_star) * sigma_eff_v)


def find_freeze_out(
    masses: dict,
    gdof: dict = OCTET_GDOF,
    g_dark: float = G_DARK,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    x_min: float = 10.0,
    x_max: float = 60.0,
    n_x: int = 200,
) -> dict:
    m0    = masses["chi0"]
    names = OCTET_ORDER
    x_arr = np.linspace(x_min, x_max, n_x)
    xf    = x_max

    for x in x_arr:
        T      = m0 / x
        H      = hubble_rate(T)
        s      = entropy_density(T)
        geff   = griest_seckel_geff(masses, m0, x, gdof)
        sigma  = sigma_v_pair("chi0", "chi0", masses, x, g_dark, g_f, m_zp)
        n_eq   = yield_equilibrium("chi0", x, masses, gdof) * s
        rate   = sigma * n_eq
        if rate < H:
            xf = x
            break

    sigma_eff = sigma_v_pair("chi0", "chi0", masses, xf, g_dark, g_f, m_zp)
    geff_xf   = griest_seckel_geff(masses, m0, xf, gdof)

    return {
        "x_f":       xf,
        "T_f_GeV":   m0 / xf,
        "g_eff_xf":  geff_xf,
        "sigma_eff": sigma_eff,
    }


def omega_dm_h2_from_yield(Y_inf: float, m_chi0: float) -> float:
    s0  = 2891.2
    rho_c_h2 = 8.099e-47
    return m_chi0 * Y_inf * s0 / rho_c_h2 * 1.78266e-24 / (3e10) ** 2 * (1.97327e-14 * 3e10) ** 3 * (3e10) ** 2


def omega_dm_h2_analytic(
    sigma_eff_v_cm3_s: float = 3.0e-26,
    x_f: float = 27.0,
    g_star: float = G_STAR_XF,
) -> float:
    ref_sigma = 3.0e-26
    ref_omega = 0.1200
    x_f_ref   = 27.0
    return ref_omega * (ref_sigma / sigma_eff_v_cm3_s) * (x_f / x_f_ref)


def plot_yields(result: dict, save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x      = result["x"]
    Y      = result["Y"]
    names  = result["names"]
    masses = result["masses"]
    m0     = masses["chi0"]

    x_eq   = np.logspace(0, 3, 500)
    Y_eq   = np.array([
        yield_equilibrium("chi0", xi, masses) for xi in x_eq
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x_eq, Y_eq, "k:", lw=1.5, label=r"$Y^{\rm eq}_{\chi^0}$")
    ax.semilogy(x, Y[0], "b-", lw=2.0, label=r"$\chi^0$ (stable DM)")

    for i, name in enumerate(names[1:4], start=1):
        ax.semilogy(x, Y[i], lw=1.2, ls="--",
                    label=rf"${name}$ (decays)")

    ax.axvline(27.0, color="gray", ls=":", lw=1.0)
    ax.text(29, 5e-10, r"$x_f\approx27$", fontsize=10)
    ax.set_xlabel(r"$x = M_0/T$", fontsize=13)
    ax.set_ylabel(r"Comoving yield $Y_i = n_i/s$", fontsize=12)
    ax.set_title("8-species Boltzmann — comoving yields", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-15, 1e-7)
    ax.yaxis.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
