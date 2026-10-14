import json
import os
from pathlib import Path


CONSTRAINTS_FILE = Path(__file__).parent.parent.parent.parent.parent / "data" / "observational_constraints.json"


EMBEDDED_CONSTRAINTS = {
    "planck_2018": {
        "Omega_DM_h2":      0.1200,
        "Omega_DM_h2_err":  0.0012,
        "H0_CMB_km_s_Mpc":  67.27,
        "H0_CMB_err":       0.60,
        "n_s":              0.9649,
        "Delta_Neff_limit": 0.17,
        "reference":        "arXiv:1807.06209",
    },
    "shoes_2022": {
        "H0_local_km_s_Mpc": 73.04,
        "H0_local_err":       1.04,
        "reference":          "arXiv:2112.04510",
        "tension_sigma":      4.9,
    },
    "lz_2024": {
        "sigma_SI_limit_cm2":  5.0e-47,
        "m_DM_GeV_reference":  36.0,
        "reference":           "arXiv:2207.03764",
    },
    "tulin_yu_2018": {
        "sigma_T_m_limit_dwarfs_cm2_g": 1.0,
        "sigma_T_m_limit_groups_cm2_g": 0.1,
        "reference": "arXiv:1705.02358",
    },
    "harvey_2015": {
        "sigma_T_m_limit_groups_cm2_g": 0.1,
        "reference": "arXiv:1503.07675",
    },
    "randall_2008": {
        "sigma_T_m_limit_bullet_cm2_g": 0.47,
        "reference": "arXiv:0704.0261",
    },
    "desi_2024": {
        "w0": -0.727,
        "w0_err": 0.10,
        "reference": "arXiv:2404.03002",
    },
    "lisa_2017": {
        "f_min_Hz": 1.0e-4,
        "f_max_Hz": 1.0e-1,
        "reference": "arXiv:1702.00786",
    },
    "ska_2018": {
        "r_DAO_Mpc": 2.3,
        "k_DAO_h_Mpc": 1.4,
        "sigma_detection": 7.2,
        "reference": "arXiv:1811.02743",
    },
    "darwin_2016": {
        "sigma_SI_target_cm2": 3.0e-49,
        "m_DM_GeV_reference":  100.0,
        "reference":           "arXiv:1606.07001",
        "expected_year":       2033,
    },
    "fcc_cdr_2019": {
        "sqrt_s_ee_GeV":   365.0,
        "lumi_ee_ab":      1.5,
        "sqrt_s_hh_TeV":   100.0,
        "lumi_hh_ab":      30.0,
        "reference":       "doi:10.1140/epjst/e2019-900045-4",
    },
}


def load_constraints(path: str = None) -> dict:
    if path is not None:
        with open(path, "r") as f:
            return json.load(f)
    if CONSTRAINTS_FILE.exists():
        with open(CONSTRAINTS_FILE, "r") as f:
            return json.load(f)
    return EMBEDDED_CONSTRAINTS


def get_planck() -> dict:
    return load_constraints().get("planck_2018", EMBEDDED_CONSTRAINTS["planck_2018"])


def get_lz_limit() -> float:
    data = load_constraints().get("lz_2024", EMBEDDED_CONSTRAINTS["lz_2024"])
    return data["sigma_SI_limit_cm2"]


def get_bullet_cluster_limit() -> float:
    data = load_constraints().get("randall_2008", EMBEDDED_CONSTRAINTS["randall_2008"])
    return data["sigma_T_m_limit_bullet_cm2_g"]


def get_dwarf_limit() -> float:
    data = load_constraints().get("tulin_yu_2018", EMBEDDED_CONSTRAINTS["tulin_yu_2018"])
    return data["sigma_T_m_limit_dwarfs_cm2_g"]


def get_lisa_band() -> tuple:
    data = load_constraints().get("lisa_2017", EMBEDDED_CONSTRAINTS["lisa_2017"])
    return data["f_min_Hz"], data["f_max_Hz"]


def get_darwin_target() -> float:
    data = load_constraints().get("darwin_2016", EMBEDDED_CONSTRAINTS["darwin_2016"])
    return data["sigma_SI_target_cm2"]


def get_ska_dao() -> dict:
    return load_constraints().get("ska_2018", EMBEDDED_CONSTRAINTS["ska_2018"])


def print_constraint_table() -> None:
    c = load_constraints()
    print(f"{'Dataset':<22} {'Key quantity':<38} {'Value'}")
    print("-" * 80)
    print(f"{'Planck 2018':<22} {'Omega_DM h^2':<38} {c['planck_2018']['Omega_DM_h2']} +/- {c['planck_2018']['Omega_DM_h2_err']}")
    print(f"{'LZ 2024':<22} {'sigma_SI limit [cm^2]':<38} {c['lz_2024']['sigma_SI_limit_cm2']:.2e}")
    print(f"{'Randall 2008':<22} {'sigma_T/m Bullet [cm^2/g]':<38} {c['randall_2008']['sigma_T_m_limit_bullet_cm2_g']}")
    print(f"{'LISA 2017':<22} {'Band [Hz]':<38} [{c['lisa_2017']['f_min_Hz']:.0e}, {c['lisa_2017']['f_max_Hz']:.0e}]")
    print(f"{'SKA 2018':<22} {'r_DAO [Mpc]':<38} {c['ska_2018']['r_DAO_Mpc']} +/- 0.4")
    print(f"{'DARWIN 2016':<22} {'sigma_SI target [cm^2]':<38} {c['darwin_2016']['sigma_SI_target_cm2']:.2e}")
