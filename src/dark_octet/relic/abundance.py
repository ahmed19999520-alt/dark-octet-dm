from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from dark_octet.constants import BENCH, PHYS


@dataclass
class RelicAbundance:
    m_dm_gev: float = 165.67
    sigma_eff_v_cm3_s: float = 4.2e-26
    g_star_xf: float = PHYS.g_star_sm
    g_eff_xf: float = 10.84

    def freeze_out_x(self) -> float:
        xf = 25.0
        for _ in range(20):
            sigma_v_gev2 = self.sigma_eff_v_cm3_s / (PHYS.cm2_per_gev2 * 3e10)
            log_arg = max(
                0.038
                * self.g_eff_xf
                * PHYS.m_planck_gev
                * self.m_dm_gev
                * sigma_v_gev2
                / (np.sqrt(self.g_star_xf) * xf),
                1e-300,
            )
            xf_new = np.log(log_arg)
            if abs(xf_new - xf) < 0.001:
                break
            xf = xf_new if xf_new > 1.0 else 25.0
        return xf

    def omega_dm_h2_lee_weinberg(self) -> float:
        xf = self.freeze_out_x()
        sigma_v_gev2 = self.sigma_eff_v_cm3_s / (PHYS.cm2_per_gev2 * 3e10)
        numerator = 1.07e9 / PHYS.m_planck_gev
        denominator = np.sqrt(self.g_star_xf) * xf * sigma_v_gev2
        return numerator / denominator

    def co_annihilation_enhancement_factor(
        self,
        masses_gev: np.ndarray,
        g_dof: np.ndarray,
        xf: float = 25.0,
    ) -> float:
        m_dm = self.m_dm_gev
        deltas = (masses_gev - m_dm) / m_dm
        g_eff = np.sum(g_dof * (1.0 + deltas) ** 1.5 * np.exp(-xf * deltas))
        g_single = g_dof[np.argmin(np.abs(masses_gev - m_dm))]
        return g_eff / g_single

    def sigma_si_kinetic_mixing(
        self,
        m_target_gev: float = 0.938272,
        Z_target: float = 54.0,
        A_target: float = 131.0,
        g_dark: float = BENCH.g_dark,
        g_f: float = BENCH.g_f,
        m_zprime_gev: float = BENCH.m_zprime_gev,
        f_p: float = 0.30,
    ) -> float:
        mu_chi_N = self.m_dm_gev * m_target_gev / (self.m_dm_gev + m_target_gev)

        sigma = (
            mu_chi_N**2
            * g_f**2
            * g_dark**2
            * Z_target**2
            * f_p**2
            / (np.pi * A_target**2 * m_zprime_gev**4)
        )
        return sigma * PHYS.cm2_per_gev2

    def print_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table

        c = Console()
        xf = self.freeze_out_x()
        omega = self.omega_dm_h2_lee_weinberg()
        sigma_si = self.sigma_si_kinetic_mixing()

        table = Table(title="Relic Abundance Summary")
        table.add_column("Quantity", style="cyan")
        table.add_column("Computed", style="yellow")
        table.add_column("Reference", style="green")

        table.add_row("x_f = M₀/T_f", f"{xf:.2f}", "≈ 25")
        table.add_row("Ω_DM h²", f"{omega:.4f}", "0.1200 ± 0.0012 [Planck]")
        table.add_row(
            "σ_SI^KM [cm²]",
            f"{sigma_si:.4e}",
            "< 5×10⁻⁴⁷ [LZ 2024]; proj. 3×10⁻⁴⁹ [DARWIN]",
        )
        c.print(table)