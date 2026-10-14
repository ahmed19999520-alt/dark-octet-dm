#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.sommerfeld.schrodinger import (
    sommerfeld_coulomb_analytic, sommerfeld_yukawa,
    dimensionless_params, sommerfeld_table,
)
from dark_octet.sommerfeld.enhancement import (
    xZp, F_yukawa, sigma_T_born_gevm2, sigma_T_per_mass,
    sidm_table, v_critical_kms, PAPER_VALUES, OBSERVATIONAL_LIMITS,
)
from dark_octet.spectrum.gmo_formula import octet_spectrum, mass_splittings
from dark_octet.signatures.collider import mono_lepton_endpoint_gev
from dark_octet.boltzmann.decay_rates import all_decay_rates, all_lifetimes_s
from dark_octet.constants import (
    ALPHA_DARK, G_DARK, M_ZP_GEV, C_LIGHT_KM_S,
    V_DELTA_GEV, Y_CHI, YCHI_VDELTA,
)


print("=" * 70)
print("  Appendix C — Sommerfeld Enhancement: Complete Calculation")
print("=" * 70)

masses  = octet_spectrum()
m_chi0  = masses["chi0"]
mu      = m_chi0 / 2.0
v_crit  = v_critical_kms(m_chi0, M_ZP_GEV)

print(f"\n  M(chi0)  = {m_chi0:.4f} GeV")
print(f"  mu       = M(chi0)/2 = {mu:.4f} GeV")
print(f"  m_Z'     = {M_ZP_GEV:.2f} GeV")
print(f"  alpha_dk = {ALPHA_DARK:.6f}")
print(f"  v_crit   = m_Z'/M(chi0) * c = {v_crit:.1f} km/s")
print(f"           = {v_crit/C_LIGHT_KM_S:.4f} c")

print("\n--- Dimensionless Parameters at v = 10 km/s ---\n")
p10 = dimensionless_params(10.0, ALPHA_DARK, M_ZP_GEV, m_chi0)
for k, v in p10.items():
    print(f"  {k:<14}: {v:.6e}")

print("\n--- Schrodinger Equation in Dimensionless Form ---\n")
print("""
  Radial Schrodinger equation (Yukawa, s-wave, l=0):

    u''(xi) + [eps_v^2/eps_phi^2  +  e^{-xi}/(eps_phi * xi)] * u(xi) = 0

  where:
    xi     = m_Z' * r
    eps_v  = v_rel / (2 alpha_dark)
    eps_phi = m_Z' / (alpha_dark * mu)
    k      = v_rel * mu / m_Z'

  Boundary conditions:
    u(xi -> 0)   = 0
    u(xi -> inf) = e^{i k xi} + f_sc e^{-i k xi}  (asymptotic plane wave)

  Sommerfeld factor:
    S(v) = |u(0)/r_0|^2 / |v_rel|

  Coulomb analytic limit (m_Z' -> 0):
    S_C(v) = (2 pi alpha / v_rel) / (1 - exp(-2 pi alpha / v_rel))
""")

print("--- Coulomb Analytic Sommerfeld at Benchmark Velocities ---\n")
print(f"  {'v [km/s]':>10} {'alpha/v':>12} {'S_Coulomb':>12} {'S_Yukawa(0.52)':>16}")
print("  " + "-" * 56)
for v in [5, 10, 30, 100, 300, 1000]:
    v_frac = v / C_LIGHT_KM_S
    a_over_v = ALPHA_DARK / v_frac
    S_c   = sommerfeld_coulomb_analytic(v, ALPHA_DARK)
    S_yuk = sommerfeld_yukawa(v, ALPHA_DARK, M_ZP_GEV)
    print(f"  {v:>10}  {a_over_v:>12.1f}  {S_c:>12.1f}  {S_yuk:>16.1f}")

print("\n--- F(x_Z') and Born Cross Section at v=10 km/s ---\n")
x_zp_10  = xZp(10.0, mu, M_ZP_GEV)
F_10     = F_yukawa(x_zp_10)
sigma_b  = sigma_T_born_gevm2(10.0, m_chi0, G_DARK, M_ZP_GEV)
print(f"  x_Z'          = {x_zp_10:.4e}")
print(f"  F(x_Z')       = {F_10:.4e}")
print(f"  F approx 1/(2x)={0.5/x_zp_10:.4e}  (large-x limit)")
print(f"  sigma_T^Born  = {sigma_b:.4e} GeV^-2")

print("\n--- SIDM Cross Section with Sommerfeld ---\n")
print(f"  {'v [km/s]':>10} {'S_yukawa':>10} {'sigma_T/m':>14} {'paper':>10} {'ratio':>8}")
print("  " + "-" * 60)
for v, paper_val in sorted(PAPER_VALUES.items()):
    S     = sommerfeld_yukawa(v, ALPHA_DARK, M_ZP_GEV)
    sigma = sigma_T_per_mass(v, m_chi0, G_DARK, M_ZP_GEV)
    ratio = sigma / paper_val if paper_val > 0 else 0.0
    print(f"  {v:>10.0f}  {S:>10.1f}  {sigma:>14.5f}  {paper_val:>10.4f}  {ratio:>8.3f}")

print("\n--- Astrophysical Constraints Check ---\n")
sidm = sidm_table(m_chi0, G_DARK, M_ZP_GEV, ALPHA_DARK)
print(f"  {'System':<22} {'v[km/s]':>8} {'model':>12} {'limit':>10} {'OK?':>6}")
print("  " + "-" * 66)
for r in sidm:
    ok = "YES" if r["satisfies"] else "NO"
    print(f"  {r['system']:<22} {r['v_kms']:>8.0f} "
          f"{r['sigma_T_m_code']:>12.4f} {r['limit_cm2_g']:>10.2f} {ok:>6}")

print("\n--- Dark Higgs Triplet Mechanism ---\n")
print(f"""
  Lagrangian term: y_chi * chi-bar * Delta_dark * chi + h.c.

  VEV of Delta_dark:
    <Delta_dark> = (1/sqrt(2)) * [[0, 0, v_Delta],
                                   [0, 0, 0      ],
                                   [v_Delta, 0, 0]]

  v_Delta = {V_DELTA_GEV:.2f} GeV,  y_chi = {Y_CHI:.2f}

  Tree-level mass splitting:
    Delta M(chi+ - chi0) = y_chi * v_Delta = {YCHI_VDELTA:.2f} GeV

  Decay channel opened:  chi+ -> chi0 + phi+

  This ensures chi0 is the UNIQUE stable relic.

  Theta(Q_EM) acts on positively charged states only:
    Theta(Q_EM) = 1 for Q_EM > 0, else 0

  States affected: chi+, chi++, Sigma+
""")

print("--- Yukawa Suppression vs Coulomb Limit ---\n")
print("""
  In the Coulomb limit (m_Z' -> 0):
    S_C(v) = 2 pi alpha_dark / v_rel / (1 - exp(-2 pi alpha_dark / v_rel))

  Finite m_Z' introduces Yukawa cutoff at r ~ 1/m_Z'.
  The factor ~0.52 is the ratio S(10 km/s, Yukawa) / S(10 km/s, Coulomb).

  This factor is velocity-dependent and computed from the radial
  Schrodinger equation with Yukawa potential V(r) = -alpha e^{-m_Z' r}/r.
""")
print(f"  Yukawa suppression at v=10 km/s: {sommerfeld_yukawa(10.0)/sommerfeld_coulomb_analytic(10.0):.4f}")
print(f"  Yukawa suppression at v=1000 km/s: {sommerfeld_yukawa(1000.0)/sommerfeld_coulomb_analytic(1000.0):.4f}")

print("\n--- Higgs-Stueckelberg Mechanism ---\n")
print("""
  SU(3)_dark has 8 generators -> 8 gauge bosons.
  VEV <Phi_dark> breaks 7 generators: 7 gauge bosons acquire mass.
  8th gauge boson (dark photon) gets mass via Stueckelberg mechanism:
    m_Z' = g_dark * v_dark * sqrt(Tr(T^a T^a)) = 9.7 GeV

  Pseudo-Goldstone bosons:
    7 are absorbed as longitudinal components of massive dark gauge bosons.
    1 survives as pi^(1)_dark with mass m_{pi(1)} < few MeV.

  Delta Neff contribution: 0.09 +/- 0.02  (within Planck bound 0.17)
  Naturalness: technically natural for epsilon_explicit ~ 2e-6
""")

print("--- Decay Kinematics for Collider Experimental Guidance ---\n")
E_max = mono_lepton_endpoint_gev()
dm    = masses["chi_p"] - masses["chi0"]
print(f"""
  For experimental searches at FCC-ee:
  Process: chi+ -> chi0 + pi+_dark -> chi0 + l+ nu_l

  Delta M(chi+, chi0) = {dm:.2f} GeV
  Max lepton energy in chi+ rest frame:
    E_max^lab ~ Delta M / 2 ~ {E_max:.4f} GeV

  This endpoint directly measures Delta M and tests r_21 = M(chi++)/M(chi0).
  Resolution needed: delta E_l ~ 0.1 GeV -> delta r_21 / r_21 ~ 0.3%.

  Background: SM W+W- produces leptons at much higher energies.
  Signal topology: mono-lepton + missing energy with sharp kinematic edge.
""")

print("--- Decay Rates for Experimental Reference ---\n")
rates = all_decay_rates(masses)
taus  = all_lifetimes_s(masses)
print(f"  {'State':<12} {'Delta M [GeV]':>14} {'Gamma [GeV]':>14} {'tau [s]':>14}")
print("  " + "-" * 60)
splits = mass_splittings(masses)
for name in ["chi_p", "chi_m", "chi_pp", "Sigma0", "Sigma_m", "Sigma_p", "Lambda"]:
    dm_  = splits[name]
    gam  = rates[name]
    tau  = taus[name]
    print(f"  {name:<12} {dm_:>14.2f} {gam:>14.4e} {tau:>14.4e}")

print("\n--- sigma_T/m Plot ---\n")
v_arr   = np.logspace(0.5, 3.2, 300)
sigma_v = np.array([sigma_T_per_mass(v) for v in v_arr])

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(v_arr, sigma_v, "b-", lw=2.2, label="Model (this work)")
ax.scatter(list(PAPER_VALUES.keys()), list(PAPER_VALUES.values()),
           s=90, color="black", zorder=6, label="Paper benchmark values")
ax.axhline(1.0,  color="orange", ls=":",  lw=1.5,
           label=r"Dwarf limit $<1.0$ cm$^2$g$^{-1}$")
ax.axhline(0.47, color="red",    ls="--", lw=1.5,
           label=r"Bullet Cluster $<0.47$ cm$^2$g$^{-1}$")
ax.axhline(0.1,  color="green",  ls="-.", lw=1.2,
           label=r"Galaxy groups $<0.1$ cm$^2$g$^{-1}$")
ax.set_xlabel(r"$v_{\rm rel}$ [km s$^{-1}$]", fontsize=13)
ax.set_ylabel(r"$\sigma_T/m_{\chi^0}$ [cm$^2$ g$^{-1}$]", fontsize=12)
ax.set_title(r"Self-interaction cross section — Sommerfeld regime", fontsize=12)
ax.legend(fontsize=9, loc="upper right")
ax.yaxis.grid(True, which="both", ls="--", alpha=0.3)
ax.xaxis.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
outdir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, "sigma_T.pdf"), dpi=150, bbox_inches="tight")
print(f"  Figure saved: results/sigma_T.pdf")

print("\n" + "=" * 70)
print("  Appendix C Complete")
print("=" * 70)
