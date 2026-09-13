from dataclasses import dataclass, field
from typing import Final
import numpy as np

HBAR_GEV_S: Final[float] = 6.582119569e-25
C_M_S: Final[float] = 2.99792458e8
G_N_SI: Final[float] = 6.67430e-11
K_B_SI: Final[float] = 1.380649e-23
M_PLANCK_GEV: Final[float] = 1.22089e19
M_PROTON_GEV: Final[float] = 0.938272088
ALPHA_EM: Final[float] = 7.2973525693e-3
ALPHA_S_MZ: Final[float] = 0.1180
M_Z_GEV: Final[float] = 91.1876
G_STAR_SM: Final[float] = 106.75
OMEGA_DM_H2_PLANCK: Final[float] = 0.1200
SIGMA_OMEGA_DM_H2: Final[float] = 0.0012
H0_SHOES_KMS_MPC: Final[float] = 73.04
ETA_B_PLANCK: Final[float] = 6.12e-10
CM2_PER_GEV2: Final[float] = 3.8937966e-28
GEV_PER_CM: Final[float] = 5.0677e13
S_PER_GEV_INV: Final[float] = 6.582e-25


@dataclass(frozen=True)
class PhysicalConstants:
    hbar_gev_s: float = HBAR_GEV_S
    c_m_s: float = C_M_S
    g_newton_si: float = G_N_SI
    k_boltzmann_si: float = K_B_SI
    m_planck_gev: float = M_PLANCK_GEV
    m_proton_gev: float = M_PROTON_GEV
    alpha_em: float = ALPHA_EM
    alpha_s_mz: float = ALPHA_S_MZ
    m_z_gev: float = M_Z_GEV
    g_star_sm: float = G_STAR_SM
    omega_dm_h2: float = OMEGA_DM_H2_PLANCK
    sigma_omega_dm_h2: float = SIGMA_OMEGA_DM_H2
    eta_b: float = ETA_B_PLANCK
    cm2_per_gev2: float = CM2_PER_GEV2

    @property
    def m_planck_reduced_gev(self) -> float:
        return self.m_planck_gev / np.sqrt(8 * np.pi)


@dataclass(frozen=True)
class BenchmarkParameters:
    M0: float = 150.0
    beta: float = 15.0
    gamma: float = 8.0
    delta: float = 5.0
    eta: float = 2.0
    zeta: float = 0.0
    g_dark: float = 0.5
    m_zprime_gev: float = 9.7
    g_f: float = 1.0e-3
    alpha_dark: float = 0.020
    lambda_dark_gev: float = 1.0e15
    m_gm_gev: float = 1.0e16
    t_c_dark_gev: float = 1.0e15
    alpha_pt: float = 0.10
    beta_over_h: float = 100.0
    v_wall: float = 0.80
    m_pi_dark_gev: float = 5.0e9
    delta_neff: float = 0.09
    r_dao_mpc: float = 2.3
    instanton_action: float = 8.0 * np.pi**2 / 0.020

    @property
    def alpha_dark_computed(self) -> float:
        return self.g_dark**2 / (4 * np.pi)

    @property
    def v_dark_gev(self) -> float:
        return self.m_zprime_gev / (
            self.g_dark * np.sqrt(2.0)
        )

    @property
    def core_radius_inv_gev(self) -> float:
        return 1.2 / (self.g_dark * self.v_dark_gev)


PHYS = PhysicalConstants()
BENCH = BenchmarkParameters()