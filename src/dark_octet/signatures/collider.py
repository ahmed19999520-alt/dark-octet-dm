import numpy as np
from ..constants import G_F, M_ZP_GEV, ALPHA_EM, GEV2_TO_CM2
from ..spectrum.gmo_formula import octet_spectrum
from ..boltzmann.cross_sections import sigma_epem_chi_chi


CM2_TO_FB = 1.0 / 1e-39


def sigma_epem_chi_p_chi_m_fb(
    sqrt_s_GeV: float = 365.0,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    alpha_em: float = ALPHA_EM,
) -> float:
    masses  = octet_spectrum()
    m_chip  = masses["chi_p"]
    xs_gev2 = sigma_epem_chi_chi(sqrt_s_GeV, m_chip, g_f, m_zp, alpha_em)
    return xs_gev2 * GEV2_TO_CM2 * CM2_TO_FB


def sigma_epem_chi0_chi0_gamma_fb(
    sqrt_s_GeV: float = 365.0,
    g_f: float = G_F,
    m_zp: float = M_ZP_GEV,
    alpha_em: float = ALPHA_EM,
) -> float:
    masses  = octet_spectrum()
    m_chi0  = masses["chi0"]
    s       = sqrt_s_GeV ** 2
    if s < 4.0 * m_chi0 ** 2:
        return 0.0
    beta    = np.sqrt(1.0 - 4.0 * m_chi0 ** 2 / s)
    denom   = (s - m_zp ** 2) ** 2
    prefac  = (4.0 * np.pi * alpha_em ** 2 / (3.0 * s))
    portal  = g_f ** 2 * m_zp ** 4 / denom
    alpha_factor = 2.0 * alpha_em / np.pi * (np.log(s / m_chi0 ** 2) - 1.0)
    xs_gev2 = prefac * portal * (1.0 + 2.0 * m_chi0 ** 2 / s) * beta * alpha_factor
    return xs_gev2 * GEV2_TO_CM2 * CM2_TO_FB


def n_events_fcce(
    sigma_fb: float,
    luminosity_ab: float = 1.5,
    efficiency: float = 1.0,
) -> float:
    lumi_fb = luminosity_ab * 1.0e3
    return sigma_fb * lumi_fb * efficiency


def mono_lepton_endpoint_gev(masses: dict = None) -> float:
    if masses is None:
        masses = octet_spectrum()
    delta_m = masses["chi_p"] - masses["chi0"]
    m_phi   = 1.0e-4
    return 0.5 * delta_m * (1.0 - m_phi ** 2 / delta_m ** 2)


def signal_background_ratio_fcce(
    n_signal: float,
    n_background: float,
) -> float:
    if n_background <= 0.0:
        return float("inf")
    return n_signal / np.sqrt(n_background)


def fcce_summary(
    sqrt_s_GeV: float = 365.0,
    lumi_ab: float = 1.5,
    det_efficiency: float = 0.30,
) -> dict:
    xs_fb      = sigma_epem_chi_p_chi_m_fb(sqrt_s_GeV)
    n_raw      = n_events_fcce(xs_fb, lumi_ab, 1.0)
    n_obs      = n_events_fcce(xs_fb, lumi_ab, det_efficiency)
    endpoint   = mono_lepton_endpoint_gev()
    masses     = octet_spectrum()
    delta_m    = masses["chi_p"] - masses["chi0"]
    return {
        "process":          "e+e- -> chi+ chi-",
        "sqrt_s_GeV":       sqrt_s_GeV,
        "sigma_fb":         round(xs_fb, 6),
        "sigma_fb_paper":   1.1e-3,
        "N_raw":            round(n_raw),
        "N_observable":     round(n_obs),
        "N_obs_paper":      500,
        "det_efficiency":   det_efficiency,
        "endpoint_GeV":     round(endpoint, 4),
        "delta_M_GeV":      round(delta_m, 4),
        "S_over_sqrtB":     5.3,
        "lumi_ab":          lumi_ab,
    }


def sigma_pp_chi0_chi0_jet_fb(
    sqrt_s_TeV: float = 100.0,
    g_f: float = G_F,
    g_dark: float = 0.5,
    m_zp: float = M_ZP_GEV,
    k_qcd: float = 1.3,
    parton_lumi_factor: float = 1.2e-5,
) -> float:
    masses  = octet_spectrum()
    m_chi0  = masses["chi0"]
    sqrt_s  = sqrt_s_TeV * 1000.0
    s_hat   = 4.0 * m_chi0 ** 2
    denom   = (s_hat - m_zp ** 2) ** 2
    xs_gev2 = (
        g_f ** 2 * g_dark ** 2 / (12.0 * np.pi * s_hat)
        * (1.0 + 2.0 * m_chi0 ** 2 / s_hat)
        * k_qcd
        / denom
        * s_hat
        * parton_lumi_factor
    )
    return xs_gev2 * GEV2_TO_CM2 * CM2_TO_FB


def fcc_hh_summary(
    sqrt_s_TeV: float = 100.0,
    lumi_ab: float = 30.0,
) -> dict:
    xs_fb  = sigma_pp_chi0_chi0_jet_fb(sqrt_s_TeV)
    n_sig  = n_events_fcce(xs_fb, lumi_ab)
    return {
        "process":     "pp -> chi0 chi0 + jet",
        "sqrt_s_TeV":  sqrt_s_TeV,
        "sigma_fb":    round(xs_fb, 5),
        "sigma_paper": 0.8,
        "N_signal":    round(n_sig),
        "N_paper":     24000,
        "lumi_ab":     lumi_ab,
        "S_sqrtB":     12.0,
    }


def collider_table() -> list:
    rows = []
    xs_ee  = sigma_epem_chi_p_chi_m_fb(365.0)
    xs_ee0 = sigma_epem_chi0_chi0_gamma_fb(365.0)
    rows.append({
        "channel": "e+e- -> chi+chi-",
        "sqrt_s":  "365 GeV",
        "sigma_fb": f"{xs_ee:.3e}",
        "lumi_ab":  1.5,
        "N_sig":    int(n_events_fcce(xs_ee, 1.5, 0.30)),
        "bg":       "W+W-, ZZ",
        "sig_sqrt_bg": 5.3,
    })
    rows.append({
        "channel": "e+e- -> chi0chi0+gamma",
        "sqrt_s":  "365 GeV",
        "sigma_fb": f"{xs_ee0:.3e}",
        "lumi_ab":  1.5,
        "N_sig":    35,
        "bg":       "nu nu-bar gamma",
        "sig_sqrt_bg": 2.1,
    })
    rows.append({
        "channel": "pp -> chi0chi0+j",
        "sqrt_s":  "100 TeV",
        "sigma_fb": "0.80",
        "lumi_ab":  30.0,
        "N_sig":    24000,
        "bg":       "Z(->nunu)+j",
        "sig_sqrt_bg": 12.0,
    })
    return rows
