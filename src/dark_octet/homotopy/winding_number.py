import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from ..constants import G_DARK, ALPHA_DARK, T_DARK_C_GEV


VDARK_GEV = T_DARK_C_GEV / (4.0 * np.pi)


def bps_analytic(xi: np.ndarray) -> tuple:
    xi = np.asarray(xi, dtype=float)
    safe = np.where(xi < 1e-10, 1e-10, xi)
    f = 1.0 / np.tanh(safe) - 1.0 / safe
    g = safe / np.sinh(safe)
    f = np.where(xi < 1e-10, xi / 3.0, f)
    g = np.where(xi < 1e-10, 1.0 - xi ** 2 / 6.0, g)
    return f, g


def bps_ode_rhs(xi: float, y: np.ndarray, lambda_ratio: float) -> list:
    f, df, g, dg = y
    xi_safe = max(xi, 1e-12)
    d2f = (2.0 * g ** 2 * f / xi_safe ** 2
           - 2.0 * df / xi_safe
           + lambda_ratio ** 2 * f * (f ** 2 - 1.0))
    d2g = (g * (g ** 2 - 1.0) / xi_safe ** 2
           + f ** 2 * g)
    return [df, d2f, dg, d2g]


def bps_numerical(
    xi_max: float = 20.0,
    n_points: int = 2000,
    lambda_ratio: float = 0.0,
    g_dark: float = G_DARK,
) -> dict:
    xi_span  = (1e-4, xi_max)
    xi_eval  = np.linspace(1e-4, xi_max, n_points)

    f0_analytic, g0_analytic = bps_analytic(np.array([xi_span[0]]))[0][0], \
                                bps_analytic(np.array([xi_span[0]]))[1][0]

    eps = 1e-4
    df0 = (bps_analytic(np.array([eps + 1e-6]))[0][0]
           - bps_analytic(np.array([eps]))[0][0]) / 1e-6
    dg0 = (bps_analytic(np.array([eps + 1e-6]))[1][0]
           - bps_analytic(np.array([eps]))[1][0]) / 1e-6

    y0 = [f0_analytic, df0, g0_analytic, dg0]
    sol = solve_ivp(
        bps_ode_rhs,
        xi_span,
        y0,
        args=(lambda_ratio,),
        t_eval=xi_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
    )

    xi      = sol.t
    f       = sol.y[0]
    g       = sol.y[2]
    f       = np.clip(f, 0.0, 1.0)
    g       = np.clip(g, 0.0, 1.0)
    i_half  = np.argmin(np.abs(f - 0.5))
    xi_core = xi[i_half]

    return {
        "xi":      xi,
        "f":       f,
        "g":       g,
        "xi_core": xi_core,
        "f_inf":   float(f[-1]),
        "g_inf":   float(g[-1]),
    }


def topological_charge_integral(
    xi: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    g_dark: float = G_DARK,
) -> float:
    df = np.gradient(f, xi)
    dg = np.gradient(g, xi)

    integrand = (
        4.0 * np.pi * xi ** 2
        * (2.0 * f ** 2 * (1.0 - g) ** 2 / xi ** 2 + (df) ** 2 + (dg / xi) ** 2)
    )
    from scipy.integrate import trapezoid
    norm = g_dark ** 2 / (32.0 * np.pi ** 2)
    integral_val = trapezoid(integrand, xi) / (8.0 * np.pi ** 2 / g_dark ** 2)
    return norm * integral_val * (8.0 * np.pi ** 2 / g_dark ** 2)


def topological_charge_analytic(g_dark: float = G_DARK) -> float:
    return g_dark ** 2 / (32.0 * np.pi ** 2) * (8.0 * np.pi ** 2 / g_dark ** 2)


def bps_mass_bound(v_dark: float = VDARK_GEV, g_dark: float = G_DARK) -> float:
    return 4.0 * np.pi * v_dark / g_dark


def instanton_suppression(alpha_dark: float = ALPHA_DARK) -> float:
    return np.exp(-8.0 * np.pi ** 2 / alpha_dark)


def instanton_suppression_exponent(alpha_dark: float = ALPHA_DARK) -> float:
    return -8.0 * np.pi ** 2 / alpha_dark


def core_radius_analytical(g_dark: float = G_DARK, v_dark: float = VDARK_GEV) -> float:
    return 1.2 / (g_dark * v_dark)


def plot_bps_profiles(save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xi_arr = np.linspace(1e-3, 10.0, 1000)
    f_bps, g_bps = bps_analytic(xi_arr)

    xi_core = xi_arr[np.argmin(np.abs(f_bps - 0.5))]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xi_arr, f_bps, "b-",  lw=2.0, label=r"$f(\xi)$ (scalar)")
    ax.plot(xi_arr, g_bps, "r--", lw=2.0, label=r"$g(\xi)$ (gauge)")
    ax.axvline(xi_core, color="gray", ls=":", lw=1.2,
               label=rf"$\xi_{{\rm core}}\approx{xi_core:.2f}$")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\xi\equiv r\,g_{\rm dark}\,v_{\rm dark}$", fontsize=13)
    ax.set_ylabel("Profile functions $f,g$", fontsize=13)
    ax.set_title(
        r"BPS profile functions — $g_{\rm dark}=0.5$, $\alpha_{\rm dark}=0.020$",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
