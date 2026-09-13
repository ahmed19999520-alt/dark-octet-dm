from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dark_octet.constants import BENCH


@dataclass
class GravitationalWaveSpectrum:
    T_c_gev: float = field(default_factory=lambda: BENCH.t_c_dark_gev)
    alpha_pt: float = field(default_factory=lambda: BENCH.alpha_pt)
    beta_over_h: float = field(default_factory=lambda: BENCH.beta_over_h)
    v_wall: float = field(default_factory=lambda: BENCH.v_wall)
    g_star: float = 106.75

    def __init__(
        self,
        T_c_gev: float = None,
        alpha_pt: float = None,
        beta_over_h: float = None,
        v_wall: float = None,
        g_star: float = 106.75,
    ):
        self.T_c_gev = T_c_gev or BENCH.t_c_dark_gev
        self.alpha_pt = alpha_pt or BENCH.alpha_pt
        self.beta_over_h = beta_over_h or BENCH.beta_over_h
        self.v_wall = v_wall or BENCH.v_wall
        self.g_star = g_star

    @property
    def f_peak_hz(self) -> float:
        return (
            1.65e-3
            * (self.beta_over_h / 100.0)
            * (self.T_c_gev / 1e15)
            * (self.g_star / 106.75) ** (1.0 / 6.0)
        )

    @property
    def omega_peak(self) -> float:
        alpha_ratio = self.alpha_pt / (1.0 + self.alpha_pt)
        eff_factor = 0.11 * self.v_wall**3 / (0.42 + self.v_wall**2)
        return (
            1.67e-5
            * alpha_ratio**2
            * (100.0 / self.g_star) ** (1.0 / 3.0)
            * (100.0 / self.beta_over_h) ** 2
            * eff_factor
        )

    def spectral_shape_bubble(self, f: np.ndarray) -> np.ndarray:
        fp = self.f_peak_hz
        ratio = f / fp
        numerator = ratio**3 * (8.0 + ratio**2)
        denominator = (fp**2 + f**2 + f**4 / fp**2) ** 1.5 * fp**(-3)
        return numerator / (denominator * fp**3 + 1e-300)

    def spectral_shape_sound(self, f: np.ndarray) -> np.ndarray:
        f_sw = 1.9e-3 * (100.0 / self.beta_over_h) * (self.T_c_gev / 1e15) * (self.g_star / 100.0) ** 0.5
        ratio = f / f_sw
        return ratio**3 / (1.0 + 3.0 * ratio**2) ** 3.5

    def omega_gw(self, f_array: np.ndarray) -> np.ndarray:
        A_bub = self.omega_peak
        A_sw = (
            2.65e-6
            * (self.alpha_pt / (1.0 + self.alpha_pt)) ** 2
            * (100.0 / self.g_star) ** (1.0 / 3.0)
            * (100.0 / self.beta_over_h)
            * self.v_wall
        )
        A_tu = (
            3.35e-4
            * (self.alpha_pt / (1.0 + self.alpha_pt)) ** 1.5
            * (100.0 / self.g_star) ** (1.0 / 3.0)
            * (100.0 / self.beta_over_h)
            * self.v_wall
        )

        bubble = A_bub * self.spectral_shape_bubble(f_array)
        sound = A_sw * self.spectral_shape_sound(f_array)

        return bubble + sound

    def lisa_sensitivity(self, f_array: np.ndarray) -> np.ndarray:
        return 1.5e-41 * (f_array / 1e-3) ** (-4) + 1e-49 * (f_array / 1e-3) ** 2

    def compute_snr(self, T_obs_years: float = 4.0) -> float:
        f_array = np.logspace(-4, -1, 500)
        omega = self.omega_gw(f_array)
        sens = self.lisa_sensitivity(f_array)

        integrand = (omega / sens) ** 2
        T_obs_s = T_obs_years * 365.25 * 24.0 * 3600.0
        snr_sq = np.trapz(integrand, f_array) * T_obs_s
        return np.sqrt(snr_sq)

    def plot_spectrum(self, save_path: str = "gw_spectrum.pdf") -> None:
        f_array = np.logspace(-4, -1, 500)
        omega = self.omega_gw(f_array)
        sens = self.lisa_sensitivity(f_array)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.loglog(
            f_array,
            omega,
            color="#003087",
            linewidth=2.5,
            label=rf"EQST-GP: $T_c^{{\rm dark}} = 10^{{15}}$ GeV",
        )
        ax.loglog(
            f_array,
            sens,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label="LISA sensitivity (approx.)",
        )

        ax.axvline(
            x=self.f_peak_hz,
            color="#8B0000",
            linewidth=1.2,
            linestyle=":",
        )
        ax.text(
            self.f_peak_hz * 1.15,
            self.omega_peak * 5.0,
            rf"$f_* = {self.f_peak_hz*1000:.2f}$ mHz",
            fontsize=10,
            color="#8B0000",
        )

        f_below = f_array[f_array < self.f_peak_hz]
        if len(f_below) > 0:
            ax.loglog(
                f_below,
                self.omega_peak * (f_below / self.f_peak_hz) ** 3,
                color="orange",
                linewidth=1.2,
                linestyle="-.",
                alpha=0.7,
                label=r"$\Omega_{\rm GW} \propto f^3$ (below peak)",
            )

        snr = self.compute_snr()
        ax.text(
            0.05,
            0.90,
            rf"$\Omega_{{GW}}^{{\rm peak}} \approx {self.omega_peak:.2e}$"
            + "\n"
            + rf"SNR$_{{4\rm yr}} \approx {snr:.1f}$",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(r"Frequency $f$ [Hz]", fontsize=13)
        ax.set_ylabel(r"$\Omega_{\rm GW}(f)$", fontsize=13)
        ax.set_xlim(1e-4, 0.1)
        ax.set_ylim(1e-12, 1e-6)
        ax.legend(fontsize=11)
        ax.set_title(
            "Stochastic Gravitational Wave Background\n"
            r"from $\mathrm{SU}(3)_{\rm dark}$ Confinement Transition",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


from dataclasses import field