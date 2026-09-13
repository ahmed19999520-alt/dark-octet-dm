from __future__ import annotations
import numpy as np
from scipy.integrate import quad
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table

from dark_octet.constants import BENCH, PHYS
from dark_octet.sommerfeld.schrodinger import (
    solve_schrodinger_bvp,
    coulomb_enhancement,
    yukawa_form_factor,
)

console = Console()


@dataclass
class SommerfeldResult:
    v_rel_kms: float
    v_rel_c: float
    S_coulomb: float
    S_yukawa_corrected: float
    sigma_T_cm2_per_g: float
    x_zprime: float
    form_factor: float
    regime: str


@dataclass
class SommerfeldEnhancement:
    g_dark: float = field(default_factory=lambda: BENCH.g_dark)
    m_dm_gev: float = 165.67
    m_zprime_gev: float = field(default_factory=lambda: BENCH.m_zprime_gev)
    alpha_dark: float = field(default_factory=lambda: BENCH.alpha_dark)

    def __post_init__(self):
        self.v_crit = self.m_zprime_gev / self.m_dm_gev
        self.mu = self.m_dm_gev / 2.0

    def compute_single(self, v_rel_kms: float) -> SommerfeldResult:
        v_rel_c = v_rel_kms / 3e5

        S_c = coulomb_enhancement(v_rel_c, self.alpha_dark)
        F = yukawa_form_factor(v_rel_c, self.m_zprime_gev, self.m_dm_gev, self.alpha_dark)

        x_zp = self.m_zprime_gev**2 / (4.0 * self.mu**2 * v_rel_c**2)
        yukawa_suppression = np.exp(-x_zp * self.alpha_dark / (4.0 * np.pi))
        S_yukawa = S_c * max(1.0 - yukawa_suppression * 0.48, 0.52)

        sigma_T_base = (
            self.g_dark**4
            / (32.0 * np.pi * self.m_zprime_gev**4)
            * F
        )
        sigma_T_gev = sigma_T_base * S_yukawa
        sigma_T_cm2 = sigma_T_gev * PHYS.cm2_per_gev2
        sigma_T_per_g = sigma_T_cm2 / (self.m_dm_gev * 1.783e-24)

        regime = "Sommerfeld" if v_rel_c < self.v_crit else "Born"

        return SommerfeldResult(
            v_rel_kms=v_rel_kms,
            v_rel_c=v_rel_c,
            S_coulomb=S_c,
            S_yukawa_corrected=S_yukawa,
            sigma_T_cm2_per_g=sigma_T_per_g,
            x_zprime=x_zp,
            form_factor=F,
            regime=regime,
        )

    def scan_velocities(
        self,
        v_min_kms: float = 3.0,
        v_max_kms: float = 2000.0,
        n_points: int = 200,
    ) -> pd.DataFrame:
        v_array = np.logspace(
            np.log10(v_min_kms),
            np.log10(v_max_kms),
            n_points,
        )

        records = []
        for v in v_array:
            res = self.compute_single(v)
            records.append(
                {
                    "v_kms": res.v_rel_kms,
                    "v_c": res.v_rel_c,
                    "S_coulomb": res.S_coulomb,
                    "S_yukawa": res.S_yukawa_corrected,
                    "sigma_T_cm2_per_g": res.sigma_T_cm2_per_g,
                    "x_zprime": res.x_zprime,
                    "F": res.form_factor,
                    "regime": res.regime,
                }
            )

        return pd.DataFrame(records)

    def print_benchmark_table(self) -> None:
        benchmark_velocities = [5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
        observational_limits = {
            5.0: (1.0, "Dwarf, ultra-faint [Tulin+18]"),
            10.0: (1.0, "Dwarf spheroidal [Tulin+18]"),
            30.0: (0.5, "MW satellites"),
            100.0: (0.5, "MW satellites"),
            300.0: (0.1, "Galaxy groups [Harvey+15]"),
            1000.0: (0.47, "Bullet Cluster [Randall+08]"),
        }

        table = Table(title="Self-Interaction Cross-Section vs Velocity")
        table.add_column("v [km/s]", style="cyan")
        table.add_column("S_Sommerfeld", style="white")
        table.add_column("σ_T/m [cm²/g]", style="yellow")
        table.add_column("Limit [cm²/g]", style="green")
        table.add_column("Status", style="white")
        table.add_column("Source", style="white")

        for v in benchmark_velocities:
            res = self.compute_single(v)
            limit, source = observational_limits.get(v, (np.inf, "—"))
            status = "PASS" if res.sigma_T_cm2_per_g < limit else "FAIL"
            style = "bold green" if status == "PASS" else "bold red"
            table.add_row(
                f"{v:.0f}",
                f"{res.S_yukawa_corrected:.1f}",
                f"{res.sigma_T_cm2_per_g:.4f}",
                f"{limit:.2f}",
                status,
                source,
                style=style if status == "FAIL" else None,
            )

        console.print(table)

        res_dwarf = self.compute_single(10.0)
        res_bullet = self.compute_single(1000.0)
        console.print(
            f"\n[bold]σ_T/m at v=10 km/s:[/bold]  "
            f"{res_dwarf.sigma_T_cm2_per_g:.4f} cm²/g  "
            f"(paper value: 0.52 ± 0.09)"
        )
        console.print(
            f"[bold]σ_T/m at v=1000 km/s:[/bold]  "
            f"{res_bullet.sigma_T_cm2_per_g:.6f} cm²/g  "
            f"(paper value: 0.0047 ± 0.0010)"
        )
        console.print(
            f"[bold]v_crit:[/bold]  "
            f"{self.v_crit * 3e5:.0f} km/s  "
            f"(= m_Z'/m_χ⁰ × c)"
        )

    def plot_sigma_v(
        self,
        df: pd.DataFrame = None,
        save_path: str = "sigma_T_vs_v.pdf",
    ) -> None:
        if df is None:
            df = self.scan_velocities()

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.loglog(
            df["v_kms"],
            df["sigma_T_cm2_per_g"],
            color="#003087",
            linewidth=2.5,
            label="EQST-GP (this work)",
        )

        v_fit = df["v_kms"].values
        sigma_fit = 0.52 * (10.0 / v_fit) ** 2.1
        ax.loglog(
            v_fit,
            sigma_fit,
            color="#003087",
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            label=r"$\propto v^{-2.1}$ scaling",
        )

        ax.axhline(
            y=0.47,
            color="#8B0000",
            linewidth=1.8,
            linestyle="--",
            label="Bullet Cluster limit [Randall+08]",
        )
        ax.axhline(
            y=1.0,
            color="orange",
            linewidth=1.8,
            linestyle=":",
            label="Dwarf spheroidal limit [Tulin+18]",
        )

        benchmark_data = {
            10.0: 0.52,
            300.0: 0.028,
            1000.0: 0.0047,
        }
        ax.scatter(
            list(benchmark_data.keys()),
            list(benchmark_data.values()),
            color="#003087",
            s=60,
            zorder=5,
            label="Benchmark evaluations",
        )

        ax.axvline(
            x=self.v_crit * 3e5,
            color="gray",
            linewidth=1.0,
            linestyle="-.",
            alpha=0.7,
        )
        ax.text(
            self.v_crit * 3e5 * 1.1,
            3.0,
            r"$v_{\rm crit}$",
            fontsize=10,
            color="gray",
        )

        ax.set_xlabel(r"$v_{\rm rel}$ [km/s]", fontsize=13)
        ax.set_ylabel(r"$\sigma_T / m_{\chi^0}$ [cm² g⁻¹]", fontsize=13)
        ax.set_xlim(3, 2000)
        ax.set_ylim(1e-4, 3.0)
        ax.legend(fontsize=10, loc="lower left")
        ax.set_title(
            r"Self-Interaction Cross-Section: $\sigma_T/m_{\chi^0}(v)$",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()