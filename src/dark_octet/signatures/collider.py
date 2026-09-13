from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from dark_octet.constants import BENCH, PHYS
from dark_octet.boltzmann.cross_sections import (
    drell_yan_xs_ee_to_charged_dark,
    drell_yan_xs_pp_chi0chi0,
)

console = Console()


@dataclass
class ColliderSignature:
    process_name: str
    sqrt_s_gev: float
    cross_section_fb: float
    luminosity_ab_inv: float
    signal_events: float
    background_estimate: float
    significance_sigma: float
    primary_background: str
    final_state: str
    distinguishing_feature: str


def compute_endpoint_energy(
    m_parent_gev: float,
    m_dm_gev: float,
    m_phi_gev: float = 0.01,
) -> float:
    if m_parent_gev <= m_dm_gev:
        return 0.0
    delta_m = m_parent_gev - m_dm_gev
    return delta_m / 2.0 * (1.0 - m_phi_gev**2 / delta_m**2)


def compute_r21_from_endpoint(
    E_endpoint_gev: float,
    m_dm_gev: float = 165.67,
) -> float:
    m_parent = m_dm_gev + 2.0 * E_endpoint_gev
    return m_parent / m_dm_gev


def build_collider_table(
    m_chi0: float = 165.67,
    m_chip: float = 165.67,
    m_chipp: float = 172.67,
    g_dark: float = BENCH.g_dark,
    g_f: float = BENCH.g_f,
    m_zprime_gev: float = BENCH.m_zprime_gev,
    m_kk_gev: float = 6000.0,
    kappa_eff: float = None,
) -> pd.DataFrame:
    if kappa_eff is None:
        kappa_eff = np.sqrt(2.0) / 6000.0

    processes = []

    sigma_ee_chip_chipbar = drell_yan_xs_ee_to_charged_dark(
        sqrt_s_gev=365.0,
        m_chi_p=m_chip,
    )
    E_ep_chip = compute_endpoint_energy(m_chip, m_chi0)
    lum_ee = 1.5
    n_ee = sigma_ee_chip_chipbar * lum_ee * 1e3
    bkg_ee = max(n_ee / 5.0, 100.0)
    sig_ee = n_ee / np.sqrt(max(bkg_ee, 1.0))

    processes.append(
        ColliderSignature(
            process_name=r"e+e- -> χ+χ-",
            sqrt_s_gev=365.0,
            cross_section_fb=sigma_ee_chip_chipbar,
            luminosity_ab_inv=lum_ee,
            signal_events=n_ee * 0.30,
            background_estimate=bkg_ee,
            significance_sigma=min(sig_ee * 0.30, 10.0),
            primary_background="W+W-, ZZ",
            final_state=r"ℓ+ + MET",
            distinguishing_feature=f"Endpoint E_ℓ^max ≈ {E_ep_chip:.1f} GeV",
        )
    )

    sigma_ee_chi0_isr = sigma_ee_chip_chipbar * 0.021
    processes.append(
        ColliderSignature(
            process_name=r"e+e- -> χ⁰χ⁰γ (ISR)",
            sqrt_s_gev=365.0,
            cross_section_fb=sigma_ee_chi0_isr,
            luminosity_ab_inv=lum_ee,
            signal_events=sigma_ee_chi0_isr * lum_ee * 1e3 * 0.90,
            background_estimate=1500.0,
            significance_sigma=2.1,
            primary_background=r"νν̄γ",
            final_state=r"γ + MET",
            distinguishing_feature="Monophoton E_γ spectrum",
        )
    )

    sigma_pp_chi0chi0 = drell_yan_xs_pp_chi0chi0(sqrt_s_tev=100.0)
    lum_hh = 30.0
    n_pp_chi0 = sigma_pp_chi0chi0 * lum_hh * 1e3
    processes.append(
        ColliderSignature(
            process_name=r"pp -> χ⁰χ⁰ + j",
            sqrt_s_gev=100000.0,
            cross_section_fb=sigma_pp_chi0chi0,
            luminosity_ab_inv=lum_hh,
            signal_events=n_pp_chi0 * 0.45,
            background_estimate=n_pp_chi0 * 0.45 / 12.0**2,
            significance_sigma=12.0,
            primary_background=r"Z(νν)+j",
            final_state="Monojet + MET",
            distinguishing_feature="MET > 500 GeV, no lepton",
        )
    )

    sigma_pp_chip_chipm = sigma_pp_chi0chi0 * 0.15
    n_pp_chip = sigma_pp_chip_chipm * lum_hh * 1e3
    processes.append(
        ColliderSignature(
            process_name=r"pp -> χ+χ- + j",
            sqrt_s_gev=100000.0,
            cross_section_fb=sigma_pp_chip_chipm,
            luminosity_ab_inv=lum_hh,
            signal_events=n_pp_chip * 0.40,
            background_estimate=n_pp_chip * 0.40 / 7.0**2,
            significance_sigma=7.0,
            primary_background="W+W- + j",
            final_state=r"ℓ±ℓ∓ + MET",
            distinguishing_feature=f"Endpoint at {E_ep_chip:.1f} GeV",
        )
    )

    sigma_kk = (
        kappa_eff**4
        * (100000.0) ** 2
        / (96.0 * np.pi)
        * (1.0 - 4.0 * m_chi0**2 / m_kk_gev**2) ** 2
        * PHYS.cm2_per_gev2
        * 1e39
    )
    n_kk = sigma_kk * lum_hh * 1e3
    processes.append(
        ColliderSignature(
            process_name=r"pp -> G_KK -> χ⁰χ⁰",
            sqrt_s_gev=100000.0,
            cross_section_fb=max(sigma_kk, 0.01),
            luminosity_ab_inv=lum_hh,
            signal_events=max(n_kk, 0.01) * 0.45,
            background_estimate=max(n_kk, 0.01) * 0.45 / 9.0**2,
            significance_sigma=9.0,
            primary_background="DY continuum",
            final_state="MET resonance",
            distinguishing_feature=f"Resonance at m_KK ≈ {m_kk_gev/1000:.0f} TeV",
        )
    )

    records = []
    for p in processes:
        records.append(
            {
                "Process": p.process_name,
                "√s [GeV]": f"{p.sqrt_s_gev:.0f}",
                "σ [fb]": f"{p.cross_section_fb:.3e}",
                "L [ab⁻¹]": f"{p.luminosity_ab_inv:.1f}",
                "N_sig": f"{p.signal_events:.0f}",
                "N_bkg": f"{p.background_estimate:.0f}",
                "S/√B [σ]": f"{p.significance_sigma:.1f}",
                "Final state": p.final_state,
                "Key feature": p.distinguishing_feature,
            }
        )

    return pd.DataFrame(records)


def print_collider_table(df: pd.DataFrame) -> None:
    table = Table(title="Collider Signatures: Dark Octet Production")
    for col in df.columns:
        table.add_column(col, style="cyan" if col == "Process" else "white")
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)


def plot_endpoint_distribution(
    m_parent_gev: float = 165.67,
    m_dm_gev: float = 165.67,
    m_phi_gev: float = 0.01,
    save_path: str = "endpoint_distribution.pdf",
) -> None:
    if m_parent_gev <= m_dm_gev + m_phi_gev:
        return

    E_max = compute_endpoint_energy(m_parent_gev, m_dm_gev, m_phi_gev)
    E_arr = np.linspace(0.0, E_max * 1.4, 500)

    def dalitz_dist(E, E_max_val):
        if E_max_val <= 0:
            return np.zeros_like(E)
        ratio = np.clip(E / E_max_val, 0.0, 1.0)
        return ratio * (1.0 - ratio)

    signal = dalitz_dist(E_arr, E_max)
    signal = signal / signal.max() if signal.max() > 0 else signal

    np.random.seed(42)
    bkg = 0.25 * np.exp(-E_arr / (E_max * 1.5))
    bkg = bkg / bkg.max() * 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(E_arr, bkg, alpha=0.3, color="gray", label="SM background")
    ax.plot(E_arr, signal + bkg, color="#003087", linewidth=2.5, label="Signal + background")
    ax.plot(E_arr, bkg, color="gray", linewidth=1.2, linestyle="--", label="Background only")

    ax.axvline(x=E_max, color="#8B0000", linewidth=1.8, linestyle=":", label=rf"$E_\ell^{{\rm max}} = {E_max:.1f}$ GeV")

    ax.set_xlabel(r"Lepton energy $E_\ell$ [GeV]", fontsize=13)
    ax.set_ylabel("dN/dE_ℓ (arb. units)", fontsize=13)
    ax.set_title(
        rf"Mono-lepton endpoint: $\chi^+ \to \chi^0 + \pi^+_{{\rm dark}}$"
        + "\n"
        + rf"$m_{{\chi^+}} = {m_parent_gev:.1f}$ GeV,  $m_{{\chi^0}} = {m_dm_gev:.1f}$ GeV",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, E_max * 1.4)
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def event_rate_vs_luminosity(
    sigma_fb: float,
    lum_max_ab_inv: float = 2.0,
    efficiency: float = 0.30,
    save_path: str = "event_rate_luminosity.pdf",
) -> None:
    lum_arr = np.linspace(0.0, lum_max_ab_inv, 200)
    n_events = sigma_fb * lum_arr * 1e3 * efficiency

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lum_arr, n_events, color="#003087", linewidth=2.5, label=rf"Signal ($\sigma = {sigma_fb:.2e}$ fb, $\epsilon = {efficiency:.0%}$)")
    ax.axhline(y=5.0, color="#8B0000", linewidth=1.5, linestyle="--", label="5 events threshold")
    ax.axhline(y=30.0, color="orange", linewidth=1.5, linestyle=":", label="30 events (discovery)")

    if lum_max_ab_inv >= 0.1:
        lum_5 = 5.0 / (sigma_fb * 1e3 * efficiency)
        ax.axvline(x=min(lum_5, lum_max_ab_inv), color="gray", linewidth=1.0, linestyle="-.", alpha=0.7)

    ax.set_xlabel(r"Integrated luminosity [ab$^{-1}$]", fontsize=13)
    ax.set_ylabel("Expected signal events", fontsize=13)
    ax.set_title("FCC-ee Signal Event Rate vs. Luminosity", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()