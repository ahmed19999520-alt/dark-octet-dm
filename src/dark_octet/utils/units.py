from __future__ import annotations
from dark_octet.constants import PHYS
 
 
def gev_to_kg(m_gev: float) -> float:
    return m_gev * PHYS.hbar_gev_s * PHYS.c_m_s / (PHYS.hbar_gev_s * PHYS.c_m_s)
 
 
def gev_inv_to_m(length_gev_inv: float) -> float:
    hbar_c_gev_m = PHYS.hbar_gev_s * PHYS.c_m_s
    return length_gev_inv * hbar_c_gev_m
 
 
def fb_to_cm2(sigma_fb: float) -> float:
    return sigma_fb * 1e-39
 
 
def cm2_per_g_to_gev_inv2(sigma_per_m: float, m_gev: float) -> float:
    m_kg = m_gev * 1.783e-27
    sigma_cm2 = sigma_per_m * m_kg * 1e3
    return sigma_cm2 / PHYS.cm2_per_gev2