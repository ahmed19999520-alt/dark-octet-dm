import numpy as np

HBAR_C_GEV_FM    = 0.197326980
HBAR_C_GEV_CM    = HBAR_C_GEV_FM * 1e-13
GEV_TO_CM        = HBAR_C_GEV_CM
GEV2_TO_CM2      = GEV_TO_CM ** 2
GEV_TO_S         = 6.582119569e-25
C_LIGHT_KM_S     = 2.99792458e5
M_PL_GEV         = 1.22091932e19
M_PL_REDUCED_GEV = 2.43548e18
ALPHA_EM         = 1.0 / 137.03599908
GEV_PER_G        = 1.0 / 1.78266192e-24
CM2_PER_GEV2     = GEV2_TO_CM2
RHO_CRIT_H2_CGS  = 1.87882e-29

G_DARK           = 0.5
ALPHA_DARK       = G_DARK ** 2 / (4.0 * np.pi)
G_F              = 1.0e-3
M_ZP_GEV         = 9.7
M_ZP2            = M_ZP_GEV ** 2
M_ZP4            = M_ZP_GEV ** 4

M0_GEV           = 208.67
BETA_GEV         = -15.0
GAMMA_GEV        = 8.0
DELTA_GEV        = 5.0
ETA_GEV          = 8.0
V_DELTA_GEV      = 17.0
Y_CHI            = 0.10
YCHI_VDELTA      = Y_CHI * V_DELTA_GEV

T_DARK_C_GEV     = 1.0e15
B0_DARK          = 7.0
NC_DARK          = 3
NF_DARK          = 3

ALPHA_PT         = 0.10
BETA_OVER_H      = 100.0
V_WALL           = 0.80
G_STAR_CONF      = 106.75
G_STAR_XF        = 106.75
G_STAR_S_XF      = 106.75

F_P_PROTON       = 0.30
Z_XE             = 54
A_XE             = 131
M_NUCLEON_GEV    = 0.9389

VDARK_GEV        = T_DARK_C_GEV / (4.0 * np.pi)
M_GM_GEV         = 4.0 * np.pi * VDARK_GEV / G_DARK

OCTET_QN = {
    "chi0":    {"I": 1.5, "Y": +1, "T":  0, "Q_pos": False},
    "chi_p":   {"I": 1.5, "Y": +1, "T":  0, "Q_pos": True},
    "chi_m":   {"I": 1.5, "Y": +1, "T": -1, "Q_pos": False},
    "chi_pp":  {"I": 1.5, "Y": +1, "T": +1, "Q_pos": True},
    "Sigma0":  {"I": 1.0, "Y":  0, "T":  0, "Q_pos": False},
    "Sigma_m": {"I": 1.0, "Y":  0, "T": -1, "Q_pos": False},
    "Sigma_p": {"I": 1.0, "Y":  0, "T": +1, "Q_pos": True},
    "Lambda":  {"I": 0.0, "Y":  0, "T":  0, "Q_pos": False},
}

OCTET_GDOF = {
    "chi0": 2, "chi_p": 2, "chi_m": 2, "chi_pp": 2,
    "Sigma0": 3, "Sigma_m": 3, "Sigma_p": 3, "Lambda": 1,
}

MASS_PAPER = {
    "chi0":    165.67,
    "chi_p":   167.37,
    "chi_m":   168.67,
    "chi_pp":  180.37,
    "Sigma0":  192.67,
    "Sigma_m": 195.67,
    "Sigma_p": 207.37,
    "Lambda":  208.67,
}

DELTA_M_PAPER = {k: v - MASS_PAPER["chi0"] for k, v in MASS_PAPER.items()}

OCTET_ORDER = ["chi0", "chi_p", "chi_m", "chi_pp", "Sigma0", "Sigma_m", "Sigma_p", "Lambda"]
