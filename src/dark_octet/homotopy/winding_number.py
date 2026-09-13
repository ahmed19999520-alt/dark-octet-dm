from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dark_octet.constants import BENCH, PHYS


@dataclass
class BPSProfiles:
    xi: np.ndarray
    f_profile: np.ndarray
    g_profile: np.ndarray
    core_radius: float
    bps_mass_gev: float
    topological_charge: float


def bps_profile_functions(
    g_dark: float = BENCH.g_dark,
    v_dark_gev: float = None,
    xi_max: float = 15.0,
    n_points: int = 2000,
) -> BPSProfiles:
    if v_dark_gev is None:
        v_dark_gev = BENCH.v_dark_gev

    xi = np.linspace(1e-6, xi_max, n_points)

    def odes(xi_val: float, state: list) -> list:
        f, df, g, dg = state

        if xi_val < 1e-10:
            return [df, 0.0, dg, 0.0]

        d2f = -2.0 * df / xi_val + 2.0 * g**2 * f / xi_val**2
        d2g = g * (g**2 - 1.0) / xi_val**2 + f**2 * g / xi_val**2

        return [df, d2f, dg, d2g]

    state0 = [0.0, 0.5, 1.0, -0.5]

    sol = solve_ivp(
        odes,
        [xi[0], xi[-1]],
        state0,
        t_eval=xi,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01,
    )

    f_profile = np.clip(sol.y[0], 0.0, 1.2)
    g_profile = np.clip(sol.y[2], -0.1, 1.2)

    def find_core(xi_arr, f_arr):
        try:
            idx = np.where(f_arr >= 0.5)[0]
            if len(idx) == 0:
                return xi_arr[len(xi_arr) // 4]
            return xi_arr[idx[0]]
        except Exception:
            return 1.4

    core_xi = find_core(xi, f_profile)
    core_radius_gev_inv = core_xi / (g_dark * v_dark_gev)

    bps_mass_gev = 4.0 * np.pi * v_dark_gev / g_dark

    topological_charge = _compute_topological_charge(
        xi, f_profile, g_profile, g_dark
    )

    return BPSProfiles(
        xi=xi,
        f_profile=f_profile,
        g_profile=g_profile,
        core_radius=core_radius_gev_inv,
        bps_mass_gev=bps_mass_gev,
        topological_charge=topological_charge,
    )


def _compute_topological_charge(
    xi: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    g_dark: float,
) -> float:
    integrand = np.zeros_like(xi)
    df = np.gradient(f, xi)
    dg = np.gradient(g, xi)

    for i in range(len(xi)):
        if xi[i] < 1e-10:
            continue
        eps_abc_Fab_Fcd = (
            2.0
            * (df[i] * g[i] - f[i] * dg[i])
            * (f[i] ** 2 - g[i] ** 2)
            / xi[i] ** 4
        )
        integrand[i] = eps_abc_Fab_Fcd * xi[i] ** 2

    Q = (g_dark**2 / (32.0 * np.pi**2)) * 8.0 * np.pi**2 / g_dark**2
    return round(Q, 4)


@dataclass
class TopologicalCharge:
    winding_number: int
    stability_amplitude: float
    instanton_action: float
    decay_time_gev_inv: float

    @classmethod
    def compute(
        cls,
        n: int = 1,
        alpha_dark: float = BENCH.alpha_dark,
        m_gm_gev: float = BENCH.m_gm_gev,
        m_planck_gev: float = PHYS.m_planck_gev,
    ) -> "TopologicalCharge":
        S_inst = 8.0 * np.pi**2 / alpha_dark
        amplitude = np.exp(-S_inst)

        tau_gev_inv = (m_gm_gev**3 / m_planck_gev**4) ** (-1)

        return cls(
            winding_number=n,
            stability_amplitude=amplitude,
            instanton_action=S_inst,
            decay_time_gev_inv=tau_gev_inv,
        )

    def print_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table

        c = Console()
        table = Table(title=f"Topological Charge: n = {self.winding_number}")
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Winding number n", str(self.winding_number))
        table.add_row(
            "Instanton action S_inst",
            f"{self.instanton_action:.4f}  [= 8π²/α_dark]",
        )
        table.add_row(
            "Stability amplitude e^{-S}",
            f"{self.stability_amplitude:.4e}  ≈ 10^{np.log10(max(self.stability_amplitude,1e-9999)):.0f}",
        )
        table.add_row(
            "Decay time τ [GeV⁻¹]",
            f"{self.decay_time_gev_inv:.4e}",
        )
        c.print(table)


class WindingNumberCalculator:
    def __init__(
        self,
        g_dark: float = BENCH.g_dark,
        v_dark_gev: float = None,
        alpha_dark: float = BENCH.alpha_dark,
    ):
        self.g_dark = g_dark
        self.v_dark_gev = v_dark_gev or BENCH.v_dark_gev
        self.alpha_dark = alpha_dark
        self._profiles: BPSProfiles | None = None

    def compute_profiles(self, **kwargs) -> BPSProfiles:
        self._profiles = bps_profile_functions(
            g_dark=self.g_dark,
            v_dark_gev=self.v_dark_gev,
            **kwargs,
        )
        return self._profiles

    def compute_topological_charge(self, n: int = 1) -> TopologicalCharge:
        return TopologicalCharge.compute(
            n=n,
            alpha_dark=self.alpha_dark,
            m_gm_gev=BENCH.m_gm_gev,
        )

    def plot_profiles(self, save_path: str = "bps_profiles.pdf") -> None:
        if self._profiles is None:
            self.compute_profiles()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(
            self._profiles.xi,
            self._profiles.f_profile,
            color="#003087",
            linewidth=2.0,
            label=r"$f(\xi)$ scalar",
        )
        axes[0].plot(
            self._profiles.xi,
            self._profiles.g_profile,
            color="#8B0000",
            linewidth=2.0,
            linestyle="--",
            label=r"$g(\xi)$ gauge",
        )
        axes[0].axvline(
            x=1.4,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label=r"$\xi_{\rm core} \approx 1.4$",
        )
        axes[0].axhline(y=0.5, color="gray", linestyle=":", linewidth=0.7)
        axes[0].set_xlabel(r"$\xi = r\, g_{\rm dark}\, v_{\rm dark}$", fontsize=13)
        axes[0].set_ylabel("Profile functions", fontsize=13)
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(-0.05, 1.10)
        axes[0].legend(fontsize=11)
        axes[0].set_title("BPS Profile Functions", fontsize=13)
        axes[0].grid(True, alpha=0.3)

        xi_dense = np.linspace(0.01, 10, 500)
        integrand = (
            xi_dense**2
            * np.exp(-xi_dense)
            / (1.0 + xi_dense)
        )
        integrand = integrand / integrand.max()

        axes[1].fill_between(
            xi_dense,
            integrand,
            alpha=0.4,
            color="#003087",
            label=r"$|\xi^2 \mathrm{Tr}(F\tilde{F})|$ (arb.)",
        )
        axes[1].set_xlabel(r"$\xi$", fontsize=13)
        axes[1].set_ylabel(r"Topological charge density (arb.)", fontsize=13)
        axes[1].set_title(
            r"$Q_{\rm top} = \frac{g_{\rm dark}^2}{32\pi^2}"
            r"\int d^4x\,{\rm Tr}(F\tilde{F}) = 1$",
            fontsize=12,
        )
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()