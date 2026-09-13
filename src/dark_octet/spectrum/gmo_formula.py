from __future__ import annotations
import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import NamedTuple
import pandas as pd
from rich.console import Console
from rich.table import Table
from dark_octet.constants import BenchmarkParameters, BENCH

console = Console()


class QuantumNumbers(NamedTuple):
    I_dark: float
    Y_dark: float
    T_dark: float
    g_dof: int
    name: str
    latex_name: str


DARK_OCTET_STATES: list[QuantumNumbers] = [
    QuantumNumbers(1.5,  1.0,  1.0, 2, "chi_pp",    r"\chi^{++}"),
    QuantumNumbers(1.5,  1.0,  0.0, 2, "chi_p",     r"\chi^{+}"),
    QuantumNumbers(1.5,  1.0,  0.0, 2, "chi_0",     r"\chi^{0}"),
    QuantumNumbers(1.5,  1.0, -1.0, 2, "chi_m",     r"\chi^{-}"),
    QuantumNumbers(1.0,  0.0,  1.0, 2, "Sigma_p",   r"\Sigma^{+}_{\rm dark}"),
    QuantumNumbers(1.0,  0.0,  0.0, 2, "Sigma_0",   r"\Sigma^{0}_{\rm dark}"),
    QuantumNumbers(1.0,  0.0, -1.0, 2, "Sigma_m",   r"\Sigma^{-}_{\rm dark}"),
    QuantumNumbers(0.0,  0.0,  0.0, 1, "Lambda",    r"\Lambda_{\rm dark}"),
]


@dataclass
class GMOMassOperator:
    params: BenchmarkParameters = field(default_factory=BenchmarkParameters)

    def mass_eigenvalue(self, qn: QuantumNumbers) -> float:
        M0 = self.params.M0
        beta = self.params.beta
        gamma = self.params.gamma
        delta = self.params.delta
        eta = self.params.eta
        zeta = self.params.zeta

        Y = qn.Y_dark
        I = qn.I_dark
        T = qn.T_dark

        return (
            M0
            + beta * Y
            + gamma * (Y**2 / 4.0 - I * (I + 1.0))
            + delta * T
            + eta * T**2
            + zeta * Y * T
        )

    def all_masses(self) -> dict[str, float]:
        return {qn.name: self.mass_eigenvalue(qn) for qn in DARK_OCTET_STATES}

    def mass_spectrum_dataframe(self) -> pd.DataFrame:
        records = []
        masses = self.all_masses()
        m_chi0 = masses["chi_0"]

        for qn in DARK_OCTET_STATES:
            M = masses[qn.name]
            delta_M = M - m_chi0
            records.append(
                {
                    "State": qn.latex_name,
                    "Name": qn.name,
                    "I_dark": qn.I_dark,
                    "Y_dark": qn.Y_dark,
                    "T_dark": qn.T_dark,
                    "g_dof": qn.g_dof,
                    "Mass_GeV": M,
                    "DeltaM_GeV": delta_M,
                    "Is_DM": qn.name == "chi_0",
                }
            )
        return pd.DataFrame(records)

    def mass_ratios(self) -> dict[str, float]:
        masses = self.all_masses()
        m_chi0 = masses["chi_0"]
        m_chipp = masses["chi_pp"]
        m_sigma0 = masses["Sigma_0"]

        r21 = m_chipp / m_chi0
        r31 = m_sigma0 / m_chi0

        delta_coeff = self.params.delta + self.params.eta
        denom = self.params.M0 + self.params.beta - 7.0 * self.params.gamma / 2.0

        r21_analytic = 1.0 + delta_coeff / denom
        r31_analytic = 1.0 - (
            self.params.beta + 25.0 * self.params.gamma / 12.0
        ) / denom

        return {
            "r21_numerical": r21,
            "r21_analytic": r21_analytic,
            "r31_numerical": r31,
            "r31_analytic": r31_analytic,
            "m_chi0_GeV": m_chi0,
            "m_chipp_GeV": m_chipp,
            "m_sigma0_GeV": m_sigma0,
        }

    def gst_consistency(self) -> dict[str, float]:
        beta_over_gamma = self.params.beta / self.params.gamma
        epsilon_dark_estimate = 0.12
        kappa_ratio = 1.56
        gst_product = epsilon_dark_estimate * kappa_ratio
        discrepancy_percent = abs(beta_over_gamma - gst_product) / beta_over_gamma * 100.0

        return {
            "beta_over_gamma": beta_over_gamma,
            "epsilon_dark_times_kappa": gst_product,
            "discrepancy_percent": discrepancy_percent,
            "gst_satisfied": discrepancy_percent < 5.0,
        }

    def symbolic_mass_operator(self) -> sp.Expr:
        M0, beta, gamma, delta, eta, zeta = sp.symbols(
            r"M_0 \beta \gamma \delta \eta \zeta", real=True
        )
        Y, I, T = sp.symbols(r"Y_{\rm dark} I_{\rm dark} T_{\rm dark}", real=True)

        M_op = (
            M0
            + beta * Y
            + gamma * (Y**2 / 4 - I * (I + 1))
            + delta * T
            + eta * T**2
            + zeta * Y * T
        )
        return M_op

    def print_spectrum(self) -> None:
        df = self.mass_spectrum_dataframe()
        ratios = self.mass_ratios()
        gst = self.gst_consistency()

        table = Table(
            title=f"Dark Octet Mass Spectrum  "
                  f"[M₀={self.params.M0}, β={self.params.beta}, "
                  f"γ={self.params.gamma}, δ={self.params.delta}, "
                  f"η={self.params.eta}] GeV"
        )

        for col in ["State", "I_dark", "Y_dark", "T_dark", "Mass_GeV", "DeltaM_GeV"]:
            table.add_column(col, style="cyan" if col == "State" else "white")

        for _, row in df.iterrows():
            style = "bold yellow" if row["Is_DM"] else "white"
            table.add_row(
                row["State"],
                str(row["I_dark"]),
                str(row["Y_dark"]),
                str(row["T_dark"]),
                f"{row['Mass_GeV']:.4f}",
                f"{row['DeltaM_GeV']:+.4f}",
                style=style,
            )

        console.print(table)
        console.print(
            f"\n[bold green]Mass ratios:[/bold green]  "
            f"r₂₁ = {ratios['r21_numerical']:.4f}  "
            f"(analytic: {ratios['r21_analytic']:.4f})   "
            f"r₃₁ = {ratios['r31_numerical']:.4f}  "
            f"(analytic: {ratios['r31_analytic']:.4f})"
        )
        console.print(
            f"[bold green]GST check:[/bold green]  "
            f"β/γ = {gst['beta_over_gamma']:.4f}   "
            f"ε_dark × κ = {gst['epsilon_dark_times_kappa']:.4f}   "
            f"discrepancy = {gst['discrepancy_percent']:.2f}%   "
            f"{'PASS' if gst['gst_satisfied'] else 'FAIL'}"
        )