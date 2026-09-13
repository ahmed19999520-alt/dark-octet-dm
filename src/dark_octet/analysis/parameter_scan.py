from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
 
from dark_octet.constants import BenchmarkParameters
from dark_octet.spectrum.gmo_formula import GMOMassOperator
 
console = Console()
 
 
@dataclass
class ParameterScanner:
    base_params: BenchmarkParameters = field(default_factory=BenchmarkParameters)
 
    def scan_1d(
        self,
        param_name: str,
        values: np.ndarray,
        observable_fn: Callable,
        observable_name: str = "observable",
    ) -> pd.DataFrame:
        records = []
        for val in track(values, description=f"Scanning {param_name}..."):
            kwargs = {
                "M0":    self.base_params.M0,
                "beta":  self.base_params.beta,
                "gamma": self.base_params.gamma,
                "delta": self.base_params.delta,
                "eta":   self.base_params.eta,
                "zeta":  self.base_params.zeta,
            }
            kwargs[param_name] = val
            params = BenchmarkParameters(**{
                k: v for k, v in kwargs.items()
                if k in BenchmarkParameters.__dataclass_fields__
            })
            gmo = GMOMassOperator(params)
            obs = observable_fn(gmo)
            records.append({param_name: val, observable_name: obs})
        return pd.DataFrame(records)
 
    def scan_r21_vs_delta(
        self,
        delta_range: np.ndarray = None,
    ) -> pd.DataFrame:
        if delta_range is None:
            delta_range = np.linspace(0, 15, 50)
 
        def obs(gmo):
            return gmo.mass_ratios()["r21_numerical"]
 
        df = self.scan_1d("delta", delta_range, obs, "r21")
        return df
 
    def scan_r31_vs_gamma(
        self,
        gamma_range: np.ndarray = None,
    ) -> pd.DataFrame:
        if gamma_range is None:
            gamma_range = np.linspace(2, 20, 50)
 
        def obs(gmo):
            return gmo.mass_ratios()["r31_numerical"]
 
        df = self.scan_1d("gamma", gamma_range, obs, "r31")
        return df
 
    def plot_sensitivity(
        self,
        df_r21: pd.DataFrame,
        df_r31: pd.DataFrame,
        save_path: str = "parameter_sensitivity.pdf",
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
        axes[0].plot(
            df_r21["delta"], df_r21["r21"],
            color="#003087", linewidth=2.0, label=r"$r_{21}(\delta)$"
        )
        axes[0].axhline(y=1.042, color="#8B0000", linestyle="--", linewidth=1.5,
                        label=r"Paper: $r_{21}=1.042$")
        axes[0].fill_between(
            df_r21["delta"],
            1.042 - 0.003, 1.042 + 0.003,
            alpha=0.2, color="#8B0000", label=r"$\\pm 1\\sigma$"
        )
        axes[0].set_xlabel(r"$\\delta$ [GeV]", fontsize=13)
        axes[0].set_ylabel(r"$r_{21} = M(\\chi^{++})/M(\\chi^0)$", fontsize=13)
        axes[0].set_title("Mass Ratio Sensitivity to Darkicity Coefficient", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
 
        axes[1].plot(
            df_r31["gamma"], df_r31["r31"],
            color="#003087", linewidth=2.0, label=r"$r_{31}(\\gamma)$"
        )
        axes[1].axhline(y=0.809, color="#8B0000", linestyle="--", linewidth=1.5,
                        label=r"Paper: $r_{31}=0.809$")
        axes[1].fill_between(
            df_r31["gamma"],
            0.809 - 0.004, 0.809 + 0.004,
            alpha=0.2, color="#8B0000", label=r"$\\pm 1\\sigma$"
        )
        axes[1].set_xlabel(r"$\\gamma$ [GeV]", fontsize=13)
        axes[1].set_ylabel(r"$r_{31} = M(\\Sigma^0)/M(\\chi^0)$", fontsize=13)
        axes[1].set_title("Mass Ratio Sensitivity to Quadratic Hypercharge", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()