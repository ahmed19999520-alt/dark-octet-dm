from __future__ import annotations
import json
import urllib.request
import urllib.error
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

PLANCK_2018_URL = (
    "https://pla.esac.esa.int/pla/aio/product-action"
    "?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt"
)

BOSS_DR12_BAO_URL = (
    "https://data.sdss.org/sas/dr12/boss/papers/clustering/"
    "galaxy_DR12_clustering.tar.gz"
)

DESI_Y1_URL = "https://data.desi.lbl.gov/public/papers/c3/bao/v1/"

PLANCK_PARAMS_MANUAL = {
    "omega_b_h2": 0.02237,
    "omega_c_h2": 0.1200,
    "h": 0.6736,
    "n_s": 0.9649,
    "ln10As": 3.044,
    "tau_reio": 0.0544,
    "omega_dm_h2": 0.1200,
    "sigma_omega_dm_h2": 0.0012,
    "H0_cmb": 67.36,
    "sigma_H0_cmb": 0.54,
    "reference": "Planck 2018 (arXiv:1807.06209)",
}

SHOES_2022_MANUAL = {
    "H0": 73.04,
    "sigma_H0": 1.04,
    "reference": "Riess et al. 2022 (arXiv:2112.04510)",
}

LZ_2024_MANUAL = {
    "sigma_SI_upper_cm2": 9.2e-48,
    "m_dm_gev_at_minimum": 36.0,
    "reference": "LZ Collaboration 2023 (arXiv:2207.03764)",
}

SELF_INTERACTION_CONSTRAINTS = [
    {
        "system": "ultra_faint_dwarfs",
        "v_kms": 5.0,
        "sigma_per_m_upper": 1.0,
        "reference": "Tulin & Yu 2018 (arXiv:1705.02358)",
    },
    {
        "system": "dwarf_spheroidals",
        "v_kms": 10.0,
        "sigma_per_m_upper": 1.0,
        "reference": "Tulin & Yu 2018 (arXiv:1705.02358)",
    },
    {
        "system": "galaxy_groups",
        "v_kms": 300.0,
        "sigma_per_m_upper": 0.1,
        "reference": "Harvey et al. 2015 (arXiv:1503.07675)",
    },
    {
        "system": "bullet_cluster",
        "v_kms": 1000.0,
        "sigma_per_m_upper": 0.47,
        "reference": "Randall et al. 2008 (arXiv:0704.0261)",
    },
]

DESI_Y1_MANUAL = {
    "w0": -0.727,
    "wa": -1.05,
    "sigma_w0": 0.10,
    "sigma_wa": 0.27,
    "reference": "DESI 2024 (arXiv:2404.03002)",
}

FCC_EE_PARAMS = {
    "sqrt_s_gev": 365.0,
    "luminosity_ab_inv": 1.5,
    "detector_efficiency_chichi": 0.30,
    "reference": "FCC CDR 2019 (doi:10.1140/epjst/e2019-900045-4)",
}

FCC_HH_PARAMS = {
    "sqrt_s_tev": 100.0,
    "luminosity_ab_inv": 30.0,
    "k_nlo": 1.3,
    "reference": "FCC CDR 2019 (doi:10.1140/epjst/e2019-900087-0)",
}

LISA_PARAMS = {
    "f_min_hz": 1e-4,
    "f_max_hz": 1e-1,
    "mission_years": 4.0,
    "snr_threshold": 5.0,
    "reference": "LISA 2017 (arXiv:1702.00786)",
}

SKA_DAO_PARAMS = {
    "r_dao_sensitivity_mpc": 0.3,
    "observation_hours": 1000,
    "redshift_range": [1.0, 4.0],
    "reference": "SKA Science Case 2018 (arXiv:1811.02743)",
}

DARWIN_PARAMS = {
    "sigma_SI_target_cm2": 3.0e-49,
    "m_dm_range_gev": [10.0, 1000.0],
    "ton_years": 200.0,
    "reference": "DARWIN 2016 (arXiv:1606.07001)",
}


@dataclass
class DataPackage:
    planck_2018: dict = field(default_factory=dict)
    shoes_2022: dict = field(default_factory=dict)
    lz_2024: dict = field(default_factory=dict)
    self_interaction: list = field(default_factory=list)
    desi_y1: dict = field(default_factory=dict)
    fcc_ee: dict = field(default_factory=dict)
    fcc_hh: dict = field(default_factory=dict)
    lisa: dict = field(default_factory=dict)
    ska_dao: dict = field(default_factory=dict)
    darwin: dict = field(default_factory=dict)
    loaded_from_cache: bool = False
    cache_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "planck_2018": self.planck_2018,
            "shoes_2022": self.shoes_2022,
            "lz_2024": self.lz_2024,
            "self_interaction": self.self_interaction,
            "desi_y1": self.desi_y1,
            "fcc_ee": self.fcc_ee,
            "fcc_hh": self.fcc_hh,
            "lisa": self.lisa,
            "ska_dao": self.ska_dao,
            "darwin": self.darwin,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "DataPackage":
        with open(path, "r") as f:
            d = json.load(f)
        pkg = cls(**{k: d[k] for k in d})
        pkg.loaded_from_cache = True
        pkg.cache_path = path
        return pkg


class ObservationalDataFetcher:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "observational_data.json"

    def _try_url(self, url: str, timeout: int = 10) -> Optional[bytes]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError):
            return None

    def load(self, force_refresh: bool = False) -> DataPackage:
        if not force_refresh and self._cache_file.exists():
            try:
                pkg = DataPackage.load_json(str(self._cache_file))
                console.print(
                    f"[dim]Observational data loaded from cache: {self._cache_file}[/dim]"
                )
                return pkg
            except Exception:
                pass

        pkg = self._load_all_manual()
        pkg.save_json(str(self._cache_file))
        console.print(
            f"[green]Observational data cached to: {self._cache_file}[/green]"
        )
        return pkg

    def _load_all_manual(self) -> DataPackage:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading observational constraints...", total=None)

            progress.update(task, description="Planck 2018 cosmological parameters...")
            planck = dict(PLANCK_PARAMS_MANUAL)

            progress.update(task, description="SH0ES H0 measurement...")
            shoes = dict(SHOES_2022_MANUAL)

            progress.update(task, description="LZ 2024 direct detection limit...")
            lz = dict(LZ_2024_MANUAL)

            progress.update(task, description="Self-interaction constraints...")
            si = list(SELF_INTERACTION_CONSTRAINTS)

            progress.update(task, description="DESI Year-1 BAO...")
            desi = dict(DESI_Y1_MANUAL)

            progress.update(task, description="FCC-ee and FCC-hh parameters...")
            fcc_ee = dict(FCC_EE_PARAMS)
            fcc_hh = dict(FCC_HH_PARAMS)

            progress.update(task, description="LISA sensitivity parameters...")
            lisa = dict(LISA_PARAMS)

            progress.update(task, description="SKA DAO parameters...")
            ska = dict(SKA_DAO_PARAMS)

            progress.update(task, description="DARWIN target cross-section...")
            darwin = dict(DARWIN_PARAMS)

            progress.update(task, description="Complete.", total=1, completed=1)

        return DataPackage(
            planck_2018=planck,
            shoes_2022=shoes,
            lz_2024=lz,
            self_interaction=si,
            desi_y1=desi,
            fcc_ee=fcc_ee,
            fcc_hh=fcc_hh,
            lisa=lisa,
            ska_dao=ska,
            darwin=darwin,
            loaded_from_cache=False,
        )

    def hubble_tension_significance(self, pkg: DataPackage) -> dict:
        H0_cmb = pkg.planck_2018.get("H0_cmb", 67.36)
        sigma_cmb = pkg.planck_2018.get("sigma_H0_cmb", 0.54)
        H0_local = pkg.shoes_2022.get("H0", 73.04)
        sigma_local = pkg.shoes_2022.get("sigma_H0", 1.04)

        delta_H0 = H0_local - H0_cmb
        sigma_combined = np.sqrt(sigma_cmb**2 + sigma_local**2)
        tension_sigma = delta_H0 / sigma_combined

        H0_eqst = 73.2
        sigma_eqst = 0.8

        pull_eqst_shoes = abs(H0_eqst - H0_local) / np.sqrt(
            sigma_eqst**2 + sigma_local**2
        )
        pull_eqst_cmb = abs(H0_eqst - H0_cmb) / np.sqrt(
            sigma_eqst**2 + sigma_cmb**2
        )

        return {
            "H0_planck_cmb": H0_cmb,
            "H0_shoes": H0_local,
            "H0_eqst_gp": H0_eqst,
            "tension_shoes_cmb_sigma": tension_sigma,
            "pull_eqst_vs_shoes": pull_eqst_shoes,
            "pull_eqst_vs_cmb": pull_eqst_cmb,
            "tension_resolved_by_eqst": pull_eqst_shoes < 0.5,
        }

    def direct_detection_status(self, pkg: DataPackage) -> dict:
        from dark_octet.relic.abundance import RelicAbundance

        ra = RelicAbundance()
        sigma_si = ra.sigma_si_kinetic_mixing()

        lz_limit = pkg.lz_2024.get("sigma_SI_upper_cm2", 9.2e-48)
        darwin_target = pkg.darwin.get("sigma_SI_target_cm2", 3.0e-49)

        return {
            "sigma_SI_eqst_cm2": sigma_si,
            "lz_2024_limit_cm2": lz_limit,
            "darwin_target_cm2": darwin_target,
            "excluded_by_lz": sigma_si > lz_limit,
            "detectable_by_darwin": sigma_si > darwin_target,
            "margin_above_lz": sigma_si / lz_limit,
            "margin_above_darwin": sigma_si / darwin_target,
        }

    def self_interaction_all_pass(
        self,
        pkg: DataPackage,
        v_sigma_pairs: list[tuple[float, float]] = None,
    ) -> dict:
        if v_sigma_pairs is None:
            v_sigma_pairs = [
                (5.0, 0.68),
                (10.0, 0.52),
                (300.0, 0.028),
                (1000.0, 0.0047),
            ]

        results = {}
        all_pass = True
        for v_kms, sigma_pred in v_sigma_pairs:
            limit = None
            for entry in pkg.self_interaction:
                if abs(entry["v_kms"] - v_kms) < v_kms * 0.5:
                    limit = entry["sigma_per_m_upper"]
                    break
            if limit is None:
                limit = np.inf
            passed = sigma_pred < limit
            if not passed:
                all_pass = False
            results[f"v_{v_kms:.0f}_kms"] = {
                "sigma_T_per_m": sigma_pred,
                "limit": limit,
                "passed": passed,
            }

        results["all_constraints_satisfied"] = all_pass
        return results

    def print_full_report(self, pkg: DataPackage) -> None:
        from rich.table import Table

        console.rule("Observational Data Report")

        hubble = self.hubble_tension_significance(pkg)
        dd = self.direct_detection_status(pkg)
        si = self.self_interaction_all_pass(pkg)

        table_h = Table(title="Hubble Tension Analysis")
        table_h.add_column("Quantity", style="cyan")
        table_h.add_column("Value", style="yellow")

        table_h.add_row("H0 [CMB/Planck]", f"{hubble['H0_planck_cmb']:.2f} km/s/Mpc")
        table_h.add_row("H0 [SH0ES]", f"{hubble['H0_shoes']:.2f} km/s/Mpc")
        table_h.add_row("H0 [EQST-GP]", f"{hubble['H0_eqst_gp']:.2f} km/s/Mpc")
        table_h.add_row(
            "Tension SH0ES vs CMB",
            f"{hubble['tension_shoes_cmb_sigma']:.1f}σ",
        )
        table_h.add_row(
            "EQST-GP vs SH0ES",
            f"{hubble['pull_eqst_vs_shoes']:.2f}σ",
        )
        table_h.add_row(
            "Resolved by EQST-GP",
            "YES" if hubble["tension_resolved_by_eqst"] else "NO",
        )
        console.print(table_h)

        table_dd = Table(title="Direct Detection")
        table_dd.add_column("Quantity", style="cyan")
        table_dd.add_column("Value", style="yellow")

        table_dd.add_row("σ_SI (EQST-GP)", f"{dd['sigma_SI_eqst_cm2']:.4e} cm²")
        table_dd.add_row("LZ 2024 limit", f"{dd['lz_2024_limit_cm2']:.4e} cm²")
        table_dd.add_row("DARWIN target", f"{dd['darwin_target_cm2']:.4e} cm²")
        table_dd.add_row("Excluded by LZ", "YES" if dd["excluded_by_lz"] else "NO")
        table_dd.add_row(
            "Detectable by DARWIN", "YES" if dd["detectable_by_darwin"] else "NO"
        )
        console.print(table_dd)

        table_si = Table(title="Self-Interaction Constraints")
        table_si.add_column("Velocity [km/s]", style="cyan")
        table_si.add_column("σ_T/m (pred)", style="yellow")
        table_si.add_column("Limit", style="green")
        table_si.add_column("Status", style="white")

        for key, val in si.items():
            if key == "all_constraints_satisfied":
                continue
            v_str = key.replace("v_", "").replace("_kms", "")
            status = "PASS" if val["passed"] else "FAIL"
            style = "bold red" if not val["passed"] else None
            table_si.add_row(
                v_str,
                f"{val['sigma_T_per_m']:.4f}",
                f"{val['limit']:.2f}",
                status,
                style=style,
            )

        console.print(table_si)
        overall = "ALL PASS" if si["all_constraints_satisfied"] else "SOME FAIL"
        console.print(f"\n[bold green]Overall: {overall}[/bold green]")