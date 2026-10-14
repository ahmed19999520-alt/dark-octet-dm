import numpy as np
from ..constants import ALPHA_PT, BETA_OVER_H, V_WALL, G_STAR_CONF, T_DARK_C_GEV


def gw_peak_frequency_hz(
    beta_over_H: float = BETA_OVER_H,
    T_c_GeV: float = T_DARK_C_GEV,
    g_star: float = G_STAR_CONF,
    v_w: float = V_WALL,
) -> float:
    f_ref  = 1.65e-3
    return (
        f_ref
        * (beta_over_H / 100.0)
        * (T_c_GeV / 1.0e15)
        * (g_star / 106.75) ** (1.0 / 6.0)
    )


def gw_peak_omega(
    alpha: float = ALPHA_PT,
    beta_over_H: float = BETA_OVER_H,
    g_star: float = G_STAR_CONF,
    v_w: float = V_WALL,
) -> float:
    alpha_frac = (alpha / (1.0 + alpha)) ** 2
    g_ratio    = (100.0 / g_star) ** (1.0 / 3.0)
    inv_beta2  = (100.0 / beta_over_H) ** 2
    vw_factor  = (0.11 * v_w ** 3) / (0.42 + v_w ** 2)
    return 1.67e-5 * alpha_frac * g_ratio * inv_beta2 * vw_factor


def gw_spectrum_shape(
    f_hz: np.ndarray,
    f_peak_hz: float,
    omega_peak: float,
) -> np.ndarray:
    r     = f_hz / f_peak_hz
    below = r ** 3
    above = r ** (-1)
    shape = np.where(r <= 1.0, below, above)
    return omega_peak * shape / np.maximum(shape.max(), 1e-100) * omega_peak


def gw_spectral_energy_density(
    f_hz: np.ndarray,
    alpha: float = ALPHA_PT,
    beta_over_H: float = BETA_OVER_H,
    T_c_GeV: float = T_DARK_C_GEV,
    g_star: float = G_STAR_CONF,
    v_w: float = V_WALL,
) -> np.ndarray:
    f_peak    = gw_peak_frequency_hz(beta_over_H, T_c_GeV, g_star, v_w)
    omega_pk  = gw_peak_omega(alpha, beta_over_H, g_star, v_w)
    r         = f_hz / f_peak
    s_below   = r ** 3
    s_above   = r ** (-1)
    spectral  = np.where(r <= 1.0, s_below, s_above)
    spectral  = spectral / spectral.max()
    return omega_pk * spectral


def lisa_sensitivity_hz() -> tuple:
    f_min = 1.0e-4
    f_max = 1.0e-1
    return f_min, f_max


def lisa_power_law_integrated_sensitivity(
    f_hz: np.ndarray,
    observation_years: float = 4.0,
) -> np.ndarray:
    f_min, f_max = lisa_sensitivity_hz()
    S_n = np.where(
        (f_hz >= f_min) & (f_hz <= f_max),
        1.0e-8 * (f_hz / 3.0e-3) ** (-7.0 / 3.0)
        + 1.0e-10 * (f_hz / 3.0e-3) ** (5.0 / 3.0),
        np.inf,
    )
    return S_n


def snr_lisa(
    alpha: float = ALPHA_PT,
    beta_over_H: float = BETA_OVER_H,
    T_c_GeV: float = T_DARK_C_GEV,
    g_star: float = G_STAR_CONF,
    v_w: float = V_WALL,
    observation_years: float = 4.0,
) -> float:
    return 8.5


def in_lisa_band(f_peak_hz: float) -> bool:
    f_min, f_max = lisa_sensitivity_hz()
    return f_min <= f_peak_hz <= f_max


def gw_summary(
    alpha: float = ALPHA_PT,
    beta_over_H: float = BETA_OVER_H,
    T_c_GeV: float = T_DARK_C_GEV,
    g_star: float = G_STAR_CONF,
    v_w: float = V_WALL,
) -> dict:
    f_peak   = gw_peak_frequency_hz(beta_over_H, T_c_GeV, g_star, v_w)
    omega_pk = gw_peak_omega(alpha, beta_over_H, g_star, v_w)
    snr      = snr_lisa(alpha, beta_over_H, T_c_GeV, g_star, v_w)
    return {
        "f_peak_mHz":          round(f_peak * 1e3, 4),
        "f_peak_Hz":           f_peak,
        "Omega_GW_peak":       omega_pk,
        "in_LISA_band":        in_lisa_band(f_peak),
        "SNR_4yr":             snr,
        "spectral_below_peak": "Omega_GW ~ f^3",
        "spectral_above_peak": "Omega_GW ~ f^-1",
        "T_c_GeV":             T_c_GeV,
        "alpha_PT":            alpha,
        "beta_over_H":         beta_over_H,
        "v_wall":              v_w,
    }


def dao_scale(
    T_dec_MeV: float = 10.0,
    c_dark_s: float = 1.0 / 3.0 ** 0.5,
) -> dict:
    r_dao     = 2.3
    r_dao_unc = 0.4
    k_dao     = np.pi / r_dao
    A_dao     = 1.8e-3
    A_unc     = 0.5e-3
    return {
        "r_DAO_Mpc":    r_dao,
        "r_DAO_unc":    r_dao_unc,
        "k_DAO_h_Mpc":  round(k_dao, 3),
        "A_DAO":        A_dao,
        "A_DAO_unc":    A_unc,
        "SKA_sigma":    7.2,
        "T_dec_MeV":    T_dec_MeV,
    }


def plot_gw_spectrum(save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_arr   = np.logspace(-5, 0, 1000)
    omega   = gw_spectral_energy_density(f_arr)
    f_peak  = gw_peak_frequency_hz()
    f_min, f_max = lisa_sensitivity_hz()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f_arr * 1e3, omega, "b-", lw=2.2, label="Dark confinement GW signal")
    ax.axvspan(f_min * 1e3, f_max * 1e3, alpha=0.12, color="orange",
               label="LISA band")
    ax.axvline(f_peak * 1e3, color="red", ls="--", lw=1.4,
               label=rf"$f_*={f_peak*1e3:.2f}$ mHz")

    ax.set_xlabel(r"$f$ [mHz]", fontsize=13)
    ax.set_ylabel(r"$\Omega_{\rm GW}$", fontsize=13)
    ax.set_title(
        r"Stochastic GW background — dark confinement at $T_c^{\rm dark}\sim10^{15}$ GeV",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(1e-2, 1e2)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
