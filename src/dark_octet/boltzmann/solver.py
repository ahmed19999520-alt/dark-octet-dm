from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kn
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from dark_octet.constants import BENCH, PHYS
from dark_octet.spectrum.gmo_formula import GMOMassOperator, DARK_OCTET_STATES
from dark_octet.boltzmann.cross_sections import effective_sigma_v
from dark_octet.boltzmann.decay_rates import compute_all_decay_rates

console = Console()


@dataclass
class BoltzmannSolution:
    x_array: np.ndarray
    yields: np.ndarray
    species_names: list[str]
    y_eq: np.ndarray
    freeze_out_x: float
    y_inf_chi0: float
    omega_dm_h2: float
    g_eff_at_xf: float
    sigma_eff_at_xf: float


@dataclass
class BoltzmannSolver:
    params: "BenchmarkParameters" = field(default_factory=lambda: BENCH)
    n_species: int = 8
    x_init: float = 1.0
    x_final: float = 1000.0
    n_points: int = 2000
    rtol: float = 1e-9
    atol: float = 1e-12

    def __post_init__(self):
        self._gmo = GMOMassOperator(self.params)
        self._mass_df = self._gmo.mass_spectrum_dataframe()
        self._masses = self._mass_df["Mass_GeV"].values
        self._g_dof = self._mass_df["g_dof"].values.astype(float)
        self._names = self._mass_df["Name"].tolist()
        self._m_dm = self._mass_df.loc[
            self._mass_df["Name"] == "chi_0", "Mass_GeV"
        ].values[0]

        masses_dict = dict(zip(self._names, self._masses))
        self._decay_rates = compute_all_decay_rates(
            masses_gev=masses_dict,
            m_dm_gev=self._m_dm,
            m_phi_gev=0.01,
            g_dark=self.params.g_dark,
        )

    def _entropy_density(self, x: float) -> float:
        T_gev = self._m_dm / x
        g_star_s = PHYS.g_star_sm
        return 2.0 * np.pi**2 / 45.0 * g_star_s * T_gev**3

    def _hubble_rate(self, x: float) -> float:
        T_gev = self._m_dm / x
        g_star = PHYS.g_star_sm
        return (
            np.sqrt(np.pi**2 * g_star / 90.0)
            * T_gev**2
            / PHYS.m_planck_gev
        )

    def _equilibrium_yield(self, i: int, x: float) -> float:
        m_i = self._masses[i]
        g_i = self._g_dof[i]
        x_i = x * m_i / self._m_dm
        g_star_s = PHYS.g_star_sm

        if x_i > 700.0:
            return 0.0

        K2_val = kn(2, x_i)
        return 45.0 / (4.0 * np.pi**4) * g_i / g_star_s * x_i**2 * K2_val

    def _g_effective(self, x: float) -> float:
        deltas = (self._masses - self._m_dm) / self._m_dm
        x_ratios = x * (1.0 + deltas)
        x_ratios = np.clip(x_ratios, 0.0, 700.0)

        return np.sum(
            self._g_dof * (1.0 + deltas) ** 1.5 * np.exp(-x * deltas)
        )

    def _rhs(self, x: float, Y: np.ndarray) -> np.ndarray:
        s = self._entropy_density(x)
        H = self._hubble_rate(x)
        prefactor = s / (H * x)

        Y_eq = np.array([self._equilibrium_yield(i, x) for i in range(self.n_species)])

        sigma_eff = effective_sigma_v(
            masses_gev=self._masses,
            g_dof=self._g_dof,
            x=x,
            m_dm_gev=self._m_dm,
            g_dark=self.params.g_dark,
            g_f=self.params.g_f,
            m_zprime_gev=self.params.m_zprime_gev,
        )

        Y_total = np.sum(Y)
        Y_eq_total = np.sum(Y_eq)

        dY = np.zeros(self.n_species)

        for i in range(self.n_species):
            annihilation = -prefactor * sigma_eff * (Y[i] * Y_total - Y_eq[i] * Y_eq_total)

            decay_out = 0.0
            decay_in = 0.0

            rate_i = self._decay_rates[self._names[i]]["decay_rate_gev"]
            if rate_i > 0.0:
                T_gev = self._m_dm / x
                gamma_lab = self._masses[i] / self._m_dm
                tau_factor = rate_i / (H * self._masses[i])
                decay_out = -tau_factor * Y[i]

            dm_idx = self._names.index("chi_0")
            if i == dm_idx:
                for j in range(self.n_species):
                    if j != dm_idx:
                        rate_j = self._decay_rates[self._names[j]]["decay_rate_gev"]
                        if rate_j > 0.0:
                            tau_factor_j = rate_j / (H * self._masses[j])
                            decay_in += tau_factor_j * Y[j]

            dY[i] = annihilation + decay_out + decay_in

        return dY

    def solve(self) -> BoltzmannSolution:
        x_span = (self.x_init, self.x_final)
        x_eval = np.logspace(
            np.log10(self.x_init),
            np.log10(self.x_final),
            self.n_points,
        )

        Y0 = np.array([
            self._equilibrium_yield(i, self.x_init)
            for i in range(self.n_species)
        ])

        sol = solve_ivp(
            self._rhs,
            x_span,
            Y0,
            t_eval=x_eval,
            method="BDF",
            rtol=self.rtol,
            atol=self.atol,
            jac_sparsity=None,
        )

        x_arr = sol.t
        Y_arr = sol.y

        Y_eq_arr = np.zeros((self.n_species, len(x_arr)))
        for i in range(self.n_species):
            for k, x_val in enumerate(x_arr):
                Y_eq_arr[i, k] = self._equilibrium_yield(i, x_val)

        dm_idx = self._names.index("chi_0")
        Y_chi0 = Y_arr[dm_idx]

        xf = self._find_freeze_out(x_arr, Y_chi0, Y_eq_arr[dm_idx])

        Y_inf = float(Y_chi0[-1])
        sigma_eff_xf = effective_sigma_v(
            masses_gev=self._masses,
            g_dof=self._g_dof,
            x=xf,
            m_dm_gev=self._m_dm,
            g_dark=self.params.g_dark,
            g_f=self.params.g_f,
            m_zprime_gev=self.params.m_zprime_gev,
        )
        g_eff_xf = self._g_effective(xf)

        omega_h2 = self._compute_relic_abundance(Y_inf)

        return BoltzmannSolution(
            x_array=x_arr,
            yields=Y_arr,
            species_names=self._names,
            y_eq=Y_eq_arr,
            freeze_out_x=xf,
            y_inf_chi0=Y_inf,
            omega_dm_h2=omega_h2,
            g_eff_at_xf=g_eff_xf,
            sigma_eff_at_xf=sigma_eff_xf,
        )

    def _find_freeze_out(
        self,
        x_arr: np.ndarray,
        Y_dm: np.ndarray,
        Y_eq_dm: np.ndarray,
    ) -> float:
        ratio = Y_dm / np.where(Y_eq_dm > 1e-30, Y_eq_dm, 1e-30)
        idx = np.where(ratio > 2.0)[0]
        if len(idx) == 0:
            return 25.0
        return float(x_arr[idx[0]])

    def _compute_relic_abundance(self, Y_inf: float) -> float:
        s0_gev3 = (
            2.0
            * np.pi**2
            / 45.0
            * PHYS.g_star_sm
            * (2.725e-13) ** 3
        )
        rho_crit_gev4 = (
            3.0 * (67.4e3 / 3.086e22 / PHYS.hbar_gev_s / PHYS.c_m_s) ** 2
            / (8.0 * np.pi)
            * PHYS.m_planck_gev**2
        )
        omega_h2 = (
            self._m_dm * Y_inf * s0_gev3
            / rho_crit_gev4
            * (67.4 / 100.0) ** 2
        )
        return omega_h2

    def plot_yields(self, solution: BoltzmannSolution, save_path: str = "yields.pdf") -> None:
        fig, ax = plt.subplots(figsize=(9, 6))

        dm_idx = solution.species_names.index("chi_0")

        ax.semilogy(
            solution.x_array,
            solution.y_eq[dm_idx],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=r"$Y_{\rm eq}$",
        )
        ax.semilogy(
            solution.x_array,
            solution.yields[dm_idx],
            color="#003087",
            linewidth=2.5,
            label=r"$\chi^0$ (stable DM)",
        )

        chipp_idx = solution.species_names.index("chi_pp")
        sigma0_idx = solution.species_names.index("Sigma_0")

        ax.semilogy(
            solution.x_array,
            np.abs(solution.yields[chipp_idx]) + 1e-20,
            color="#8B0000",
            linewidth=1.8,
            linestyle="--",
            label=r"$\chi^{++}$ (decays)",
        )
        ax.semilogy(
            solution.x_array,
            np.abs(solution.yields[sigma0_idx]) + 1e-20,
            color="orange",
            linewidth=1.8,
            linestyle="-.",
            label=r"$\Sigma^0_{\rm dark}$ (decays)",
        )

        ax.axvline(
            x=solution.freeze_out_x,
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )
        ax.text(
            solution.freeze_out_x + 1.5,
            2e-11,
            rf"$x_f \simeq {solution.freeze_out_x:.1f}$",
            fontsize=10,
            color="gray",
        )

        ax.annotate(
            rf"$Y^\infty_{{\chi^0}} \simeq {solution.y_inf_chi0:.2e}$",
            xy=(solution.x_array[-1] * 0.7, solution.y_inf_chi0 * 1.5),
            fontsize=10,
            color="#003087",
        )

        ax.set_xlabel(r"$x = M_0 / T$", fontsize=13)
        ax.set_ylabel(r"Comoving yield $Y_i = n_i/s$", fontsize=13)
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-14, 1e-7)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(
            rf"$\Omega_{{\rm DM}}h^2 = {solution.omega_dm_h2:.4f}$   "
            rf"(Planck: $0.1200 \pm 0.0012$)",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def print_results(self, solution: BoltzmannSolution) -> None:
        table = Table(title="Boltzmann Solver Results")
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Target/Reference", style="green")

        table.add_row(
            "Freeze-out x_f",
            f"{solution.freeze_out_x:.2f}",
            "≈ 25 (expected)",
        )
        table.add_row(
            "Y∞(χ⁰)",
            f"{solution.y_inf_chi0:.4e}",
            "≈ 1.2×10⁻⁹",
        )
        table.add_row(
            "Ω_DM h²",
            f"{solution.omega_dm_h2:.4f}",
            f"0.1200 ± 0.0012 (Planck)",
        )
        table.add_row(
            "g_eff at x_f",
            f"{solution.g_eff_at_xf:.2f}",
            "≈ 10.84 (8-species)",
        )
        table.add_row(
            "σ_eff v at x_f [cm³/s]",
            f"{solution.sigma_eff_at_xf:.4e}",
            "≈ 4.2×10⁻²⁶",
        )
        console.print(table)