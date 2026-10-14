#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.homotopy.long_exact_sequence import (
    HomotopyProof, verify_homotopy_groups, KNOWN_GROUPS,
)
from dark_octet.homotopy.winding_number import (
    bps_analytic, bps_numerical, bps_mass_bound,
    topological_charge_analytic, instanton_suppression_exponent,
    core_radius_analytical, VDARK_GEV,
)
from dark_octet.constants import G_DARK, ALPHA_DARK, T_DARK_C_GEV


print("=" * 70)
print("  Appendix A — Homotopy Classification and Topological Stability")
print("=" * 70)

print("\n--- Step 1: Covering Space Argument ---\n")
proof = HomotopyProof()
s1    = proof.step1_covering_space()
print(f"  {s1['statement']}")
print(f"  Covering sheets : {s1['covering_sheets']}")
print(f"  Discrete fiber  : {s1['discrete_fiber']}")
print(f"  iso k>=2        : {s1['isomorphism_k_ge_2']}")

print("\n--- Step 2: Long Exact Homotopy Sequence ---\n")
s2 = proof.step2_long_exact_sequence()
for line in s2["sequence"]:
    print(f"  {line}")
print(f"\n  pi2(SU(3))     = {s2['pi2_SU3']}")
print(f"  pi1(SO(3))     = {s2['pi1_SO3']}")
print(f"  pi1(SU(3))     = {s2['pi1_SU3']}")
print(f"  Naive result   : {s2['naive_result']}")
print(f"  Note: {s2['note']}")

print("\n--- Step 3: Hurewicz Extension by pi3(SU(3)) = Z ---\n")
s3 = proof.step3_hurewicz_extension()
print(f"  Short exact seq : {s3['ses']}")
print(f"  pi3(SU(3))      = {s3['pi3_SU3']}")
print(f"  pi3(SU(3)/SO(3))= {s3['pi3_SU3_SO3']}")
print(f"  {s3['hurewicz']}")
print(f"  {s3['uct']}")
print(f"  FINAL RESULT    : pi2(SU(3)/[SO(3)xZ2]) = {s3['final_result']}")

print("\n--- Step 4: Winding Number and Topological Charge ---\n")
s4 = proof.step4_winding_number_identification()
print(f"  Generator     : {s4['generator']}")
print(f"  Q_top formula : {s4['topological_charge']}")
print(f"  Instanton amp : {s4['instanton_suppression']}")
print(f"  Stability     : {s4['stability']}")

print("\n--- Numerical Verification ---\n")
result = verify_homotopy_groups()
print(f"  pi2 = {result['result']}")
print(f"  Is Z : {result['is_Z']}")

print("\n--- BPS Mass Bound ---\n")
M_GM = bps_mass_bound(VDARK_GEV, G_DARK)
print(f"  v_dark = T_c / (4 pi) = {VDARK_GEV:.4e} GeV")
print(f"  M_GM >= 4 pi v_dark / g_dark = {M_GM:.4e} GeV")
print(f"  Instanton exponent = {instanton_suppression_exponent():.1f}")

Q = topological_charge_analytic(G_DARK)
print(f"  Q_top (analytic) = {Q:.6f}")

r_core = core_radius_analytical(G_DARK, VDARK_GEV)
print(f"  Core radius r_core ~ {r_core:.4e} GeV^-1")

print("\n--- BPS Profile Functions (Analytic) ---\n")
xi_arr     = np.linspace(1e-3, 10.0, 2000)
f_bps, g_bps = bps_analytic(xi_arr)
xi_core    = xi_arr[np.argmin(np.abs(f_bps - 0.5))]
print(f"  xi_core (f=0.5) = {xi_core:.3f}")
print(f"  f(xi->inf) = {f_bps[-1]:.6f}  (expected: 1.0)")
print(f"  g(xi->inf) = {g_bps[-1]:.6f}  (expected: 0.0)")
print(f"  f(xi->0)   = {f_bps[0]:.6f}   (expected: ~0)")
print(f"  g(xi->0)   = {g_bps[0]:.6f}  (expected: ~1)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(xi_arr, f_bps, "b-",  lw=2.0, label=r"$f(\xi)$ (scalar)")
ax.plot(xi_arr, g_bps, "r--", lw=2.0, label=r"$g(\xi)$ (gauge)")
ax.axvline(xi_core, color="gray", ls=":", lw=1.2,
           label=rf"$\xi_{{\rm core}}\approx{xi_core:.2f}$")
ax.axhline(0.5,  color="gray", ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel(r"$\xi\equiv r\,g_{\rm dark}\,v_{\rm dark}$", fontsize=13)
ax.set_ylabel("Profile functions", fontsize=13)
ax.set_title(
    r"BPS profile $f(\xi)$, $g(\xi)$ — $g_{\rm dark}=0.5$, $\alpha_{\rm dark}=0.020$",
    fontsize=11,
)
ax.legend(fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(-0.05, 1.15)
ax.yaxis.grid(True, ls="--", alpha=0.35)
plt.tight_layout()
outdir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, "bps_profiles.pdf"), dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: results/bps_profiles.pdf")

print("\n--- Topological Charge Integral Verification ---\n")
from scipy.integrate import trapezoid
df = np.gradient(f_bps, xi_arr)
dg = np.gradient(g_bps, xi_arr)
integrand = (
    2.0 * f_bps ** 2 * (1.0 - g_bps) ** 2 / xi_arr ** 2
    + df ** 2
    + (dg / xi_arr) ** 2
)
integral = trapezoid(4.0 * np.pi * xi_arr ** 2 * integrand, xi_arr)
Q_numerical = G_DARK ** 2 / (32.0 * np.pi ** 2) * integral / (G_DARK ** 2 / (4.0 * np.pi))
print(f"  Q_top (numerical, approximate) ~ {Q_numerical:.3f}")
print(f"  Q_top (analytic, BPS)          = {Q:.6f}")

print("\n" + "=" * 70)
print("  Appendix A Complete")
print("=" * 70)
