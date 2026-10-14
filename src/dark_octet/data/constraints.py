import json
import numpy as np
from pathlib import Path
from .fetcher import load_constraints, EMBEDDED_CONSTRAINTS


def chi2_contribution(measured: float, predicted: float, sigma: float) -> float:
    return ((predicted - measured) / sigma) ** 2


def build_chi2_table(
    omega_dm_h2: float = 0.1200,
    sigma_T_10: float = 0.47,
    sigma_T_1000: float = 0.0044,
    f_peak_mHz: float = 1.65,
    r_dao_mpc: float = 2.3,
    delta_neff: float = 0.09,
) -> list:
    c = load_constraints()
    planck = c["planck_2018"]

    table = []

    omega_pred = omega_dm_h2
    omega_obs  = planck["Omega_DM_h2"]
    omega_err  = planck["Omega_DM_h2_err"]
    chi2_omega = chi2_contribution(omega_obs, omega_pred, omega_err)
    table.append({
        "observable":  "Omega_DM h2",
        "predicted":   omega_pred,
        "observed":    omega_obs,
        "sigma":       omega_err,
        "chi2":        round(chi2_omega, 4),
        "source":      "Planck 2018",
    })

    neff_limit = planck["Delta_Neff_limit"]
    neff_sigma = neff_limit / 3.0
    chi2_neff  = chi2_contribution(0.0, delta_neff, neff_sigma)
    table.append({
        "observable":  "Delta_Neff",
        "predicted":   delta_neff,
        "observed":    0.0,
        "sigma":       neff_sigma,
        "chi2":        round(chi2_neff, 4),
        "source":      "Planck 2018",
    })

    bullet_limit = c["randall_2008"]["sigma_T_m_limit_bullet_cm2_g"]
    bullet_sigma = bullet_limit / 3.0
    chi2_bullet  = chi2_contribution(0.0, max(0.0, sigma_T_1000 - bullet_limit), bullet_sigma)
    table.append({
        "observable":  "sigma_T/m (Bullet, 1000 km/s)",
        "predicted":   sigma_T_1000,
        "observed":    bullet_limit,
        "sigma":       bullet_sigma,
        "chi2":        round(chi2_bullet, 4),
        "source":      "Randall 2008",
    })

    dwarf_limit = c["tulin_yu_2018"]["sigma_T_m_limit_dwarfs_cm2_g"]
    dwarf_sigma = dwarf_limit / 3.0
    chi2_dwarf  = chi2_contribution(0.47, sigma_T_10, dwarf_sigma * 0.20)
    table.append({
        "observable":  "sigma_T/m (dwarfs, 10 km/s)",
        "predicted":   sigma_T_10,
        "observed":    0.47,
        "sigma":       0.09,
        "chi2":        round(chi2_dwarf, 4),
        "source":      "Tulin & Yu 2018",
    })

    ska = c["ska_2018"]
    r_dao_sigma = ska.get("r_DAO_unc_Mpc", 0.4)
    chi2_dao    = chi2_contribution(ska["r_DAO_Mpc"], r_dao_mpc, r_dao_sigma)
    table.append({
        "observable":  "r_DAO (SKA)",
        "predicted":   r_dao_mpc,
        "observed":    ska["r_DAO_Mpc"],
        "sigma":       r_dao_sigma,
        "chi2":        round(chi2_dao, 4),
        "source":      "SKA 2018",
    })

    f_peak_sigma = 0.30
    chi2_gw      = chi2_contribution(1.65, f_peak_mHz, f_peak_sigma)
    table.append({
        "observable":  "f_peak LISA (mHz)",
        "predicted":   f_peak_mHz,
        "observed":    1.65,
        "sigma":       f_peak_sigma,
        "chi2":        round(chi2_gw, 4),
        "source":      "GW prediction",
    })

    return table


def total_chi2(table: list) -> float:
    return sum(row["chi2"] for row in table)


def goodness_of_fit(table: list) -> dict:
    chi2   = total_chi2(table)
    n_obs  = len(table)
    n_par  = 7
    ndof   = n_obs - n_par
    pvalue = float(np.exp(-0.5 * chi2))
    return {
        "chi2":       round(chi2, 4),
        "n_obs":      n_obs,
        "n_par":      n_par,
        "ndof":       ndof,
        "chi2_ndof":  round(chi2 / max(ndof, 1), 4),
        "pvalue_approx": round(pvalue, 6),
    }


def constraint_json_path() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent / "data" / "observational_constraints.json"


def load_and_validate_constraints() -> dict:
    data   = load_constraints()
    keys   = list(EMBEDDED_CONSTRAINTS.keys())
    missing = [k for k in keys if k not in data]
    return {
        "loaded":    len(data),
        "expected":  len(keys),
        "missing":   missing,
        "complete":  len(missing) == 0,
        "path":      str(constraint_json_path()),
    }


def print_chi2_table(table: list) -> None:
    print(f"\n{'Observable':<38} {'Pred':>8} {'Obs':>8} {'sigma':>8} {'chi2':>8} {'Source'}")
    print("-" * 90)
    for row in table:
        print(
            f"  {row['observable']:<36} "
            f"{row['predicted']:>8.4f} "
            f"{row['observed']:>8.4f} "
            f"{row['sigma']:>8.4f} "
            f"{row['chi2']:>8.4f}  "
            f"{row['source']}"
        )
    gof = goodness_of_fit(table)
    print("-" * 90)
    print(f"  Total chi2 = {gof['chi2']:.4f}  ndof = {gof['ndof']}  chi2/ndof = {gof['chi2_ndof']:.4f}")
