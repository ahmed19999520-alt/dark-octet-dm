import numpy as np
from ..constants import (
    M0_GEV, BETA_GEV, GAMMA_GEV, DELTA_GEV, ETA_GEV,
    YCHI_VDELTA, OCTET_QN, OCTET_ORDER, MASS_PAPER,
)


def gmo_mass(
    I: float,
    Y: float,
    T: float,
    Q_pos: bool,
    M0: float = M0_GEV,
    beta: float = BETA_GEV,
    gamma: float = GAMMA_GEV,
    delta: float = DELTA_GEV,
    eta: float = ETA_GEV,
    ychi_vdelta: float = YCHI_VDELTA,
) -> float:
    casimir = I * (I + 1.0)
    triplet = ychi_vdelta if Q_pos else 0.0
    return (
        M0
        + beta * Y
        + gamma * (Y ** 2 / 4.0 - casimir)
        + delta * T
        + eta * T ** 2
        + triplet
    )


def octet_spectrum(
    M0: float = M0_GEV,
    beta: float = BETA_GEV,
    gamma: float = GAMMA_GEV,
    delta: float = DELTA_GEV,
    eta: float = ETA_GEV,
    ychi_vdelta: float = YCHI_VDELTA,
) -> dict:
    masses = {}
    for name in OCTET_ORDER:
        qn = OCTET_QN[name]
        masses[name] = gmo_mass(
            qn["I"], qn["Y"], qn["T"], qn["Q_pos"],
            M0, beta, gamma, delta, eta, ychi_vdelta,
        )
    return masses


def mass_splittings(masses: dict) -> dict:
    m0 = masses["chi0"]
    return {k: v - m0 for k, v in masses.items()}


def mass_ratios(masses: dict) -> dict:
    m0 = masses["chi0"]
    return {
        "r21": masses["chi_pp"] / m0,
        "r31": masses["Sigma0"] / m0,
        "r41": masses["Lambda"] / m0,
    }


def ratio_r21_analytic(
    M0: float = M0_GEV,
    beta: float = BETA_GEV,
    gamma: float = GAMMA_GEV,
    delta: float = DELTA_GEV,
    eta: float = ETA_GEV,
    ychi_vdelta: float = YCHI_VDELTA,
) -> float:
    m0 = gmo_mass(1.5, +1, 0, False, M0, beta, gamma, delta, eta, ychi_vdelta)
    return 1.0 + (delta + eta + ychi_vdelta) / m0


def ratio_r31_analytic(
    M0: float = M0_GEV,
    beta: float = BETA_GEV,
    gamma: float = GAMMA_GEV,
    delta: float = DELTA_GEV,
    eta: float = ETA_GEV,
    ychi_vdelta: float = YCHI_VDELTA,
) -> float:
    m0 = gmo_mass(1.5, +1, 0, False, M0, beta, gamma, delta, eta, ychi_vdelta)
    return 1.0 + (abs(beta) + 1.5 * gamma) / m0


def ratio_r41_analytic(
    M0: float = M0_GEV,
    beta: float = BETA_GEV,
    gamma: float = GAMMA_GEV,
    delta: float = DELTA_GEV,
    eta: float = ETA_GEV,
    ychi_vdelta: float = YCHI_VDELTA,
) -> float:
    m0 = gmo_mass(1.5, +1, 0, False, M0, beta, gamma, delta, eta, ychi_vdelta)
    return 1.0 + (abs(beta) + 3.5 * gamma) / m0


def gst_ratio(beta: float = BETA_GEV, gamma: float = GAMMA_GEV) -> float:
    return abs(beta) / gamma


def validate_spectrum(tol: float = 0.005) -> dict:
    masses = octet_spectrum()
    results = {}
    for name, m_paper in MASS_PAPER.items():
        m_code = masses[name]
        results[name] = {
            "code":  round(m_code, 4),
            "paper": m_paper,
            "diff":  round(abs(m_code - m_paper), 6),
            "pass":  abs(m_code - m_paper) < tol,
        }
    return results


def benchmark_ratios() -> dict:
    masses = octet_spectrum()
    r = mass_ratios(masses)
    return {
        "r21": {"value": round(r["r21"], 4), "paper": 1.089, "unc": 0.005},
        "r31": {"value": round(r["r31"], 4), "paper": 1.163, "unc": 0.006},
        "r41": {"value": round(r["r41"], 4), "paper": 1.260, "unc": 0.008},
    }


def print_spectrum_table() -> None:
    masses  = octet_spectrum()
    splits  = mass_splittings(masses)
    header  = f"{'State':<10} {'I':>5} {'Y':>5} {'T':>5} {'M [GeV]':>10} {'DeltaM [GeV]':>14}"
    print(header)
    print("-" * len(header))
    for name in OCTET_ORDER:
        qn = OCTET_QN[name]
        m  = masses[name]
        dm = splits[name]
        dm_str = "---" if name == "chi0" else f"{dm:+.2f}"
        print(f"{name:<10} {qn['I']:>5.1f} {qn['Y']:>5d} {qn['T']:>5d} {m:>10.2f} {dm_str:>14}")
