from __future__ import annotations
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
 
console = Console()
 
ALL_PREDICTIONS = [
    {
        "observable":   "r21 = M(chi++)/M(chi0)",
        "eqst_value":   "1.042 +/- 0.003",
        "facility":     "FCC-ee",
        "timeline":     "2042+",
        "current":      "Not yet measured",
        "origin":       "Extended GMO formula, Eq. (r21)",
        "section":      "Sec. 3",
    },
    {
        "observable":   "r31 = M(Sigma0)/M(chi0)",
        "eqst_value":   "0.809 +/- 0.004",
        "facility":     "FCC-ee",
        "timeline":     "2042+",
        "current":      "Not yet measured",
        "origin":       "Extended GMO formula, Eq. (r31)",
        "section":      "Sec. 3",
    },
    {
        "observable":   "Omega_DM h2",
        "eqst_value":   "0.120",
        "facility":     "Planck/CMB",
        "timeline":     "Current",
        "current":      "0.1200 +/- 0.0012",
        "origin":       "8-species co-annihilation Boltzmann",
        "section":      "Sec. 4.1",
    },
    {
        "observable":   "sigma_T/m at v=10 km/s [cm2/g]",
        "eqst_value":   "0.52 +/- 0.09",
        "facility":     "Dwarf galaxies",
        "timeline":     "Current",
        "current":      "< 1.0 (Tulin+18)",
        "origin":       "Sommerfeld-enhanced Z' exchange",
        "section":      "Sec. 4.2",
    },
    {
        "observable":   "sigma_T/m at v=1000 km/s [cm2/g]",
        "eqst_value":   "0.0047 +/- 0.001",
        "facility":     "Bullet Cluster",
        "timeline":     "Current",
        "current":      "< 0.47 (Randall+08)",
        "origin":       "Sommerfeld-enhanced Z' exchange",
        "section":      "Sec. 4.2",
    },
    {
        "observable":   "sigma_SI [cm2] (DARWIN)",
        "eqst_value":   "3.1e-48",
        "facility":     "DARWIN",
        "timeline":     "2033+",
        "current":      "< 9.2e-48 (LZ 2024)",
        "origin":       "Kinetic-mixing portal, Eq. (sigSI)",
        "section":      "Sec. 4.3",
    },
    {
        "observable":   "sigma(ee->chi+chi-) [fb] at 365 GeV",
        "eqst_value":   "1.1e-3",
        "facility":     "FCC-ee",
        "timeline":     "2042+",
        "current":      "Not yet measured",
        "origin":       "Drell-Yan via Z' portal",
        "section":      "Sec. 5.1",
    },
    {
        "observable":   "sigma(pp->chi0chi0) [fb] at 100 TeV",
        "eqst_value":   "0.8",
        "facility":     "FCC-hh",
        "timeline":     "2048+",
        "current":      "Not yet measured",
        "origin":       "Drell-Yan via Z' portal",
        "section":      "Sec. 5.1",
    },
    {
        "observable":   "GW peak f* [mHz]",
        "eqst_value":   "1.65",
        "facility":     "LISA",
        "timeline":     "2035+",
        "current":      "Not yet measured",
        "origin":       "SU(3)_dark confinement transition",
        "section":      "Sec. 5.2",
    },
    {
        "observable":   "GW peak Omega_GW",
        "eqst_value":   "~1e-8",
        "facility":     "LISA",
        "timeline":     "2035+",
        "current":      "Not yet measured",
        "origin":       "Bubble nucleation + sound waves",
        "section":      "Sec. 5.2",
    },
    {
        "observable":   "LISA SNR (4yr)",
        "eqst_value":   "8.5",
        "facility":     "LISA",
        "timeline":     "2035+",
        "current":      "Threshold: 5.0",
        "origin":       "GW power integrated over band",
        "section":      "Sec. 5.2",
    },
    {
        "observable":   "r_DAO [Mpc]",
        "eqst_value":   "2.3 +/- 0.4",
        "facility":     "SKA 21-cm",
        "timeline":     "2032+",
        "current":      "Not yet measured",
        "origin":       "Dark chiral symmetry breaking",
        "section":      "Sec. 5.3",
    },
    {
        "observable":   "Delta N_eff",
        "eqst_value":   "0.09 +/- 0.02",
        "facility":     "CMB-S4",
        "timeline":     "2030+",
        "current":      "< 0.17 (Planck 2025)",
        "origin":       "Surviving dark pion pi(1)",
        "section":      "Sec. 5.3",
    },
]
 
 
class PredictionsTable:
    def __init__(self):
        self._data = ALL_PREDICTIONS
 
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)
 
    def print_table(self) -> None:
        table = Table(title="Master Predictions Table: EQST-GP Dark Octet")
        table.add_column("Observable", style="cyan", no_wrap=True)
        table.add_column("EQST-GP value", style="yellow")
        table.add_column("Facility", style="white")
        table.add_column("Timeline", style="white")
        table.add_column("Current status", style="green")
 
        for row in self._data:
            table.add_row(
                row["observable"],
                row["eqst_value"],
                row["facility"],
                row["timeline"],
                row["current"],
            )
        console.print(table)
 
    def current_testable(self) -> list[dict]:
        return [r for r in self._data if r["timeline"] == "Current"]
 
    def upcoming_testable(self, by_year: int = 2035) -> list[dict]:
        def year_from_str(s: str) -> int:
            digits = "".join(c for c in s if c.isdigit())
            return int(digits[:4]) if len(digits) >= 4 else 9999
        return [
            r for r in self._data
            if year_from_str(r["timeline"]) <= by_year
        ]