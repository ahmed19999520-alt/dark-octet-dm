import argparse
import sys
import os


def cmd_spectrum(args):
    from .spectrum.gmo_formula import print_spectrum_table, benchmark_ratios
    print("\n=== Dark Octet Mass Spectrum ===\n")
    print_spectrum_table()
    print("\n=== Scale-Independent Mass Ratios ===\n")
    ratios = benchmark_ratios()
    for key, data in ratios.items():
        print(f"  {key} = {data['value']:.4f}  (paper: {data['paper']} +/- {data['unc']})")


def cmd_homotopy(args):
    from .homotopy.long_exact_sequence import verify_homotopy_groups, HomotopyProof
    from .homotopy.winding_number import (
        bps_mass_bound, instanton_suppression_exponent,
        topological_charge_analytic, VDARK_GEV,
    )
    print("\n=== Homotopy Classification ===\n")
    result = verify_homotopy_groups()
    print(f"  pi2(SU(3)/[SO(3)xZ2]) = {result['result']}")
    print(f"  pi2 is Z: {result['is_Z']}")
    mGM   = bps_mass_bound()
    s_exp = instanton_suppression_exponent()
    Q     = topological_charge_analytic()
    print(f"\n  BPS mass bound M_GM >= {mGM:.3e} GeV")
    print(f"  Instanton suppression exp({s_exp:.0f})")
    print(f"  Topological charge Q = {Q:.4f}")
    if args.plot:
        from .homotopy.winding_number import plot_bps_profiles
        outpath = os.path.join(args.output_dir, "bps_profiles.pdf")
        plot_bps_profiles(outpath)
        print(f"\n  Saved: {outpath}")


def cmd_boltzmann(args):
    from .boltzmann.solver import solve_boltzmann, omega_dm_h2_analytic
    from .relic.abundance import relic_density_summary
    print("\n=== Relic Abundance (8-species Boltzmann) ===\n")
    summary = relic_density_summary()
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    if args.plot:
        result = solve_boltzmann(x_init=1.0, x_fin=200.0, n_eval=1000)
        from .boltzmann.solver import plot_yields
        outpath = os.path.join(args.output_dir, "yields.pdf")
        plot_yields(result, outpath)
        print(f"\n  Saved: {outpath}")


def cmd_sommerfeld(args):
    from .sommerfeld.schrodinger import sommerfeld_yukawa, sommerfeld_table
    from .sommerfeld.enhancement import sidm_table, v_critical_kms
    print("\n=== Sommerfeld Enhancement ===\n")
    v_crit = v_critical_kms()
    print(f"  v_crit = {v_crit:.1f} km/s  (Sommerfeld regime below this)\n")
    table = sommerfeld_table([5, 10, 30, 100, 300, 1000])
    print(f"  {'v [km/s]':>12} {'S_Coulomb':>14} {'S_Yukawa(0.52)':>16}")
    print("  " + "-" * 46)
    for row in table:
        print(f"  {row['v_kms']:>12.0f} {row['S_coulomb']:>14.1f} {row['S_yukawa_0.52']:>16.1f}")
    print("\n=== SIDM Cross-Section Table ===\n")
    sidm = sidm_table()
    print(f"  {'System':<22} {'v[km/s]':>8} {'sigma_T/m':>12} {'limit':>10} {'OK?':>6}")
    print("  " + "-" * 64)
    for r in sidm:
        ok = "YES" if r["satisfies"] else "NO"
        print(f"  {r['system']:<22} {r['v_kms']:>8.0f} {r['sigma_T_m_code']:>12.4f} {r['limit_cm2_g']:>10.2f} {ok:>6}")
    if args.plot:
        from .sommerfeld.enhancement import plot_sigma_T
        outpath = os.path.join(args.output_dir, "sigma_T.pdf")
        plot_sigma_T(outpath)
        print(f"\n  Saved: {outpath}")


def cmd_relic(args):
    from .relic.abundance import relic_density_summary, direct_detection_summary, pseudo_goldstone_neff
    print("\n=== Relic Density ===\n")
    rd = relic_density_summary()
    for k, v in rd.items():
        print(f"  {k:<22}: {v}")
    print("\n=== Direct Detection ===\n")
    dd = direct_detection_summary()
    for k, v in dd.items():
        print(f"  {k:<22}: {v}")
    print("\n=== Delta Neff ===\n")
    neff = pseudo_goldstone_neff()
    for k, v in neff.items():
        print(f"  {k:<22}: {v}")


def cmd_gw(args):
    from .signatures.gravitational_waves import gw_summary, dao_scale
    print("\n=== Gravitational Wave Signal ===\n")
    gw = gw_summary()
    for k, v in gw.items():
        print(f"  {k:<26}: {v}")
    print("\n=== Dark Acoustic Oscillations ===\n")
    dao = dao_scale()
    for k, v in dao.items():
        print(f"  {k:<22}: {v}")
    if args.plot:
        from .signatures.gravitational_waves import plot_gw_spectrum
        outpath = os.path.join(args.output_dir, "gw_spectrum.pdf")
        plot_gw_spectrum(outpath)
        print(f"\n  Saved: {outpath}")


def cmd_validate(args):
    from .data.validators import PaperValidation
    pv = PaperValidation()
    pv.print_summary()


def cmd_all(args):
    print("\n" + "=" * 60)
    print("  DARK OCTET DM — Full Computation Pipeline")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)
    args.plot = True
    cmd_spectrum(args)
    cmd_homotopy(args)
    cmd_boltzmann(args)
    cmd_sommerfeld(args)
    cmd_relic(args)
    cmd_gw(args)
    cmd_validate(args)


def main():
    parser = argparse.ArgumentParser(
        description="Dark Octet Dark Matter — Paper II Computational Library"
    )
    parser.add_argument(
        "--output-dir", default="results/", help="Output directory for figures and CSV"
    )
    sub = parser.add_subparsers(dest="command")

    p_spec  = sub.add_parser("spectrum",    help="GMO mass spectrum")
    p_homo  = sub.add_parser("homotopy",    help="Homotopy proof + BPS profiles")
    p_homo.add_argument("--plot", action="store_true")
    p_bolt  = sub.add_parser("boltzmann",   help="8-species Boltzmann solver")
    p_bolt.add_argument("--plot", action="store_true")
    p_somm  = sub.add_parser("sommerfeld",  help="Sommerfeld enhancement + SIDM")
    p_somm.add_argument("--plot", action="store_true")
    p_rel   = sub.add_parser("relic",       help="Relic abundance + direct detection")
    p_gw    = sub.add_parser("gw",          help="Gravitational wave spectrum")
    p_gw.add_argument("--plot", action="store_true")
    p_val   = sub.add_parser("validate",    help="Full numerical validation")
    p_all   = sub.add_parser("all",         help="Run everything")
    p_all.add_argument("--output-dir", default="results/")

    args = parser.parse_args()
    if not hasattr(args, "plot"):
        args.plot = False

    dispatch = {
        "spectrum":   cmd_spectrum,
        "homotopy":   cmd_homotopy,
        "boltzmann":  cmd_boltzmann,
        "sommerfeld": cmd_sommerfeld,
        "relic":      cmd_relic,
        "gw":         cmd_gw,
        "validate":   cmd_validate,
        "all":        cmd_all,
    }

    if args.command in dispatch:
        os.makedirs(args.output_dir, exist_ok=True)
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
