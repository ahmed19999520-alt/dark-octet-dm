from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ConstraintLoader:
    json_path: str = "data/observational_constraints.json"

    def load(self) -> dict:
        with open(self.json_path, "r") as f:
            return json.load(f)

    def relic_abundance(self) -> dict:
        d = self.load()
        return d["relic_abundance"]

    def self_interaction_array(self) -> pd.DataFrame:
        d = self.load()
        return pd.DataFrame(d["self_interaction_limits"])

    def direct_detection(self) -> dict:
        d = self.load()
        return d["direct_detection"]

    def collider_params(self) -> dict:
        d = self.load()
        return {
            "fcc_ee": d["collider"]["fcc_ee"],
            "fcc_hh": d["collider"]["fcc_hh"],
        }

    def gw_params(self) -> dict:
        d = self.load()
        return d["gravitational_waves"]["lisa"]

    def dao_params(self) -> dict:
        d = self.load()
        return d["dark_acoustic_oscillations"]["ska"]

    def validate_sigma_T_prediction(
        self,
        v_kms: float,
        sigma_pred: float,
    ) -> dict:
        df = self.self_interaction_array()
        df["v_diff"] = abs(df["v_halo_kms"] - v_kms)
        closest = df.nsmallest(1, "v_diff").iloc[0]

        limit = closest["sigma_T_per_m_upper"]
        passed = sigma_pred < limit

        return {
            "v_kms": v_kms,
            "sigma_pred": sigma_pred,
            "constraint_source": closest["reference"],
            "upper_limit": limit,
            "margin": limit - sigma_pred,
            "margin_fraction": (limit - sigma_pred) / limit,
            "passed": passed,
        }

    def build_chi2_table(
        self,
        predictions: dict,
        uncertainties: dict,
    ) -> pd.DataFrame:
        d = self.load()

        targets = {
            "omega_dm_h2": (
                d["relic_abundance"]["omega_dm_h2"],
                0.0012,
            ),
            "sigma_SI_cm2": (
                d["direct_detection"]["lz_2024"]["sigma_SI_upper_cm2"],
                None,
            ),
        }

        records = []
        chi2_total = 0.0
        n_dof = 0

        for key, pred_val in predictions.items():
            if key not in targets:
                continue
            target, sigma_target = targets[key]
            sigma_theory = uncertainties.get(key, 0.0)

            if sigma_target is None:
                is_upper_limit = True
                pull = max(0.0, pred_val - target) / (sigma_theory + 1e-30)
                chi2_contrib = pull**2
            else:
                is_upper_limit = False
                sigma_total = np.sqrt(sigma_theory**2 + sigma_target**2)
                pull = (pred_val - target) / sigma_total
                chi2_contrib = pull**2
                n_dof += 1

            chi2_total += chi2_contrib

            records.append(
                {
                    "Observable": key,
                    "Predicted": pred_val,
                    "Target": target,
                    "σ_exp": sigma_target,
                    "σ_theory": sigma_theory,
                    "Pull (σ)": pull,
                    "χ² contrib": chi2_contrib,
                    "Upper limit": is_upper_limit,
                }
            )

        df = pd.DataFrame(records)
        df.loc["Total"] = {
            "Observable": "χ²/N",
            "χ² contrib": chi2_total / max(n_dof, 1),
        }
        return df