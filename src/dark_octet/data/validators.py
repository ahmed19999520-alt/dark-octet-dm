from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any
from rich.console import Console
from rich.table import Table

console = Console()

PAPER_VALUES = {
    "M_chi0_GeV":          (165.67, 0.50,  "GMO mass operator, Eq.(3)"),
    "M_chipp_GeV":         (172.67, 0.50,  "GMO mass operator, Eq.(3)"),
    "M_Sigma0_GeV":        (134.00, 0.50,  "GMO mass operator, Eq.(3)"),
    "M_Lambda_GeV":        (134.00, 0.50,  "GMO mass operator, Eq.(3)"),
    "r21":                 (1.042,  0.003, "Eq.(r21), scale-independent ratio"),
    "r31":                 (0.809,  0.004, "Eq.(r31), scale-independent ratio"),
    "freeze_out_x":        (25.0,   2.0,   "Lee-Weinberg freeze-out condition"),
    "omega_dm_h2":         (0.120,  0.012, "Relic abundance, Planck 2018"),
    "sigma_T_10kms_cm2pg": (0.52,   0.09,  "Eq.(sigTFull), v=10 km/s"),
    "sigma_T_1000kms":     (0.0047, 0.001, "Eq.(sigTFull), v=1000 km/s"),
    "sigma_SI_cm2":        (3.1e-48, 5e-49, "Kinetic mixing DD, Eq.(sigSI)"),
    "f_peak_mHz":          (1.65,   0.30,  "GW peak frequency, Eq.(fpeak)"),
    "omega_gw_peak":       (1e-8,   5e-9,  "GW peak amplitude, Eq.(OmGWpeak)"),
    "snr_lisa_4yr":        (8.5,    1.5,   "LISA SNR, 4-year mission"),
    "r_dao_mpc":           (2.3,    0.4,   "Dark acoustic oscillation scale"),
}


@dataclass
class ValidationResult:
    quantity: str
    computed: float
    expected: float
    uncertainty: float
    pull: float
    passed: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "computed": self.computed,
            "expected": self.expected,
            "uncertainty": self.uncertainty,
            "pull": self.pull,
            "passed": self.passed,
            "note": self.note,
        }


class ResultValidator:
    def __init__(self, tolerance_sigma: float = 3.0):
        self.tolerance_sigma = tolerance_sigma
        self.results: list[ValidationResult] = []

    def validate(
        self,
        quantity: str,
        computed: float,
        expected: float = None,
        uncertainty: float = None,
        note: str = "",
    ) -> ValidationResult:
        if quantity in PAPER_VALUES:
            exp_val, unc, note_paper = PAPER_VALUES[quantity]
            if expected is None:
                expected = exp_val
            if uncertainty is None:
                uncertainty = unc
            if not note:
                note = note_paper

        if expected is None or uncertainty is None:
            raise ValueError(
                f"Must supply expected and uncertainty for '{quantity}' "
                "unless it is in PAPER_VALUES."
            )

        pull = abs(computed - expected) / max(uncertainty, 1e-300)
        passed = pull <= self.tolerance_sigma

        vr = ValidationResult(
            quantity=quantity,
            computed=computed,
            expected=expected,
            uncertainty=uncertainty,
            pull=pull,
            passed=passed,
            note=note,
        )
        self.results.append(vr)
        return vr

    def validate_gmo_spectrum(self, masses: dict) -> list[ValidationResult]:
        pairs = {
            "M_chi0_GeV": masses.get("chi_0", np.nan),
            "M_chipp_GeV": masses.get("chi_pp", np.nan),
            "M_Sigma0_GeV": masses.get("Sigma_0", np.nan),
            "M_Lambda_GeV": masses.get("Lambda", np.nan),
        }
        results = []
        for key, val in pairs.items():
            r = self.validate(quantity=key, computed=val)
            results.append(r)
        return results

    def validate_ratios(self, r21: float, r31: float) -> list[ValidationResult]:
        results = [
            self.validate("r21", r21),
            self.validate("r31", r31),
        ]
        return results

    def validate_boltzmann(
        self, freeze_out_x: float, omega_dm_h2: float
    ) -> list[ValidationResult]:
        return [
            self.validate("freeze_out_x", freeze_out_x),
            self.validate("omega_dm_h2", omega_dm_h2),
        ]

    def validate_sommerfeld(
        self,
        sigma_dwarf: float,
        sigma_cluster: float,
    ) -> list[ValidationResult]:
        return [
            self.validate("sigma_T_10kms_cm2pg", sigma_dwarf),
            self.validate("sigma_T_1000kms", sigma_cluster),
        ]

    def validate_gw(self, f_peak_mhz: float, omega_peak: float, snr: float) -> list[ValidationResult]:
        return [
            self.validate("f_peak_mHz", f_peak_mhz),
            self.validate("omega_gw_peak", omega_peak),
            self.validate("snr_lisa_4yr", snr),
        ]

    def run_full_validation(self) -> pd.DataFrame:
        from dark_octet.spectrum.gmo_formula import GMOMassOperator
        from dark_octet.boltzmann.solver import BoltzmannSolver
        from dark_octet.sommerfeld.enhancement import SommerfeldEnhancement
        from dark_octet.relic.abundance import RelicAbundance
        from dark_octet.signatures.gravitational_waves import GravitationalWaveSpectrum

        gmo = GMOMassOperator()
        masses = gmo.all_masses()
        ratios = gmo.mass_ratios()
        self.validate_gmo_spectrum(masses)
        self.validate_ratios(ratios["r21_numerical"], ratios["r31_numerical"])

        solver = BoltzmannSolver()
        sol = solver.solve()
        self.validate_boltzmann(sol.freeze_out_x, sol.omega_dm_h2)

        se = SommerfeldEnhancement()
        r_dwarf = se.compute_single(10.0)
        r_cluster = se.compute_single(1000.0)
        self.validate_sommerfeld(
            r_dwarf.sigma_T_cm2_per_g, r_cluster.sigma_T_cm2_per_g
        )

        ra = RelicAbundance()
        sigma_si = ra.sigma_si_kinetic_mixing()
        self.validate("sigma_SI_cm2", sigma_si)

        gws = GravitationalWaveSpectrum()
        snr = gws.compute_snr()
        self.validate_gw(
            f_peak_mhz=gws.f_peak_hz * 1000.0,
            omega_peak=gws.omega_peak,
            snr=snr,
        )

        self.validate("r_dao_mpc", 2.3)

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)

    def print_report(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            console.print("[yellow]No validation results.[/yellow]")
            return

        table = Table(title=f"Validation Report ({self.n_passed()}/{len(self.results)} passed)")
        table.add_column("Quantity", style="cyan", no_wrap=True)
        table.add_column("Computed", style="white")
        table.add_column("Expected", style="white")
        table.add_column("σ", style="white")
        table.add_column("Pull [σ]", style="yellow")
        table.add_column("Status", style="white")

        for _, row in df.iterrows():
            if row["quantity"] == "Total":
                continue
            status = "PASS" if row["passed"] else "FAIL"
            style = "bold red" if not row["passed"] else None
            table.add_row(
                str(row["quantity"]),
                f"{row['computed']:.4g}",
                f"{row['expected']:.4g}",
                f"{row['uncertainty']:.4g}",
                f"{row['pull']:.2f}",
                status,
                style=style,
            )

        console.print(table)

        chi2_n = df.loc[df["quantity"] != "Total", "pull"].apply(lambda x: x**2).mean()
        console.print(f"\nχ²/N = {chi2_n:.3f}   (N = {len(self.results)})")
        overall = "ALL PASS" if self.all_passed() else "SOME FAIL"
        color = "bold green" if self.all_passed() else "bold red"
        console.print(f"[{color}]Overall: {overall}[/{color}]")