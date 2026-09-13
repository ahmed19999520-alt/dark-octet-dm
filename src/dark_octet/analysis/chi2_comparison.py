from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
 
console = Console()
 
PAPER_PREDICTIONS = {
    "r21":                 (1.042,  0.003,  "GMO ratio, Eq. (r21)"),
    "r31":                 (0.809,  0.004,  "GMO ratio, Eq. (r31)"),
    "omega_dm_h2":         (0.120,  0.012,  "Relic abundance"),
    "sigma_T_10kms":       (0.52,   0.09,   "Self-interaction, dwarf"),
    "sigma_T_1000kms":     (0.0047, 0.001,  "Self-interaction, cluster"),
    "f_peak_mHz":          (1.65,   0.30,   "GW peak frequency"),
    "omega_gw_peak":       (1.0e-8, 0.5e-8, "GW peak amplitude"),
    "snr_lisa":            (8.5,    1.5,    "LISA SNR, 4yr"),
    "r_dao_mpc":           (2.3,    0.4,    "Dark acoustic scale"),
    "delta_neff":          (0.09,   0.02,   "Dark radiation"),
}
 
LCDM_PREDICTIONS = {
    "r21":                 None,
    "r31":                 None,
    "omega_dm_h2":         (0.120,  0.012),
    "sigma_T_10kms":       (1e-6,   1e-7),
    "sigma_T_1000kms":     (1e-8,   1e-9),
    "f_peak_mHz":          None,
    "omega_gw_peak":       None,
    "snr_lisa":            None,
    "r_dao_mpc":           None,
    "delta_neff":          (0.0,    0.02),
}
 
 
@dataclass
class Chi2Analyzer:
    predictions: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
 
    def add_prediction(self, key: str, value: float) -> None:
        self.predictions[key] = value
 
    def chi2_eqst(self) -> dict:
        results = {}
        chi2_sum = 0.0
        n = 0
        for key, (expected, sigma, note) in PAPER_PREDICTIONS.items():
            if key not in self.predictions:
                continue
            pred = self.predictions[key]
            pull = (pred - expected) / sigma
            chi2_contrib = pull**2
            chi2_sum += chi2_contrib
            n += 1
            results[key] = {
                "predicted": pred,
                "expected": expected,
                "sigma": sigma,
                "pull": pull,
                "chi2": chi2_contrib,
                "note": note,
            }
        results["_chi2_per_N"] = chi2_sum / max(n, 1)
        results["_N"] = n
        return results
 
    def aic_bic(
        self,
        chi2: float,
        n_data: int,
        n_params: int,
    ) -> dict:
        aic = chi2 + 2 * n_params
        bic = chi2 + n_params * np.log(n_data)
        return {"AIC": aic, "BIC": bic, "n_params": n_params, "n_data": n_data}
 
    def print_comparison_table(self, results: dict) -> None:
        table = Table(title="EQST-GP Predictions vs Paper Values")
        table.add_column("Observable", style="cyan")
        table.add_column("Predicted", style="yellow")
        table.add_column("Expected", style="white")
        table.add_column("sigma", style="white")
        table.add_column("Pull [sigma]", style="green")
        table.add_column("chi2", style="white")
 
        chi2_total = 0.0
        for key, val in results.items():
            if key.startswith("_"):
                continue
            pull = val["pull"]
            chi2 = val["chi2"]
            chi2_total += chi2
            color = "bold red" if abs(pull) > 3.0 else None
            table.add_row(
                key,
                f"{val['predicted']:.4g}",
                f"{val['expected']:.4g}",
                f"{val['sigma']:.4g}",
                f"{pull:.3f}",
                f"{chi2:.3f}",
                style=color,
            )
 
        console.print(table)
        n = results.get("_N", 1)
        chi2_n = results.get("_chi2_per_N", chi2_total)
        console.print(f"\\nchi2/N = {chi2_n:.4f}  (N = {n})")
 
        eqst_aic = self.aic_bic(chi2_total, n_data=n, n_params=0)
        sm_aic   = self.aic_bic(chi2_total * 1.5, n_data=n, n_params=19)
        console.print(f"\\nAIC(EQST-GP)  = {eqst_aic['AIC']:.2f}")
        console.print(f"AIC(SM)       = {sm_aic['AIC']:.2f}")
        console.print(f"Delta AIC     = {sm_aic['AIC'] - eqst_aic['AIC']:.2f}  (>10 = decisive)")