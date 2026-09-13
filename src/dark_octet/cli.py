import click
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="dark-octet")
def main():
    pass


@main.command("spectrum")
@click.option("--M0", default=150.0, type=float)
@click.option("--beta", default=15.0, type=float)
@click.option("--gamma", default=8.0, type=float)
@click.option("--delta", default=5.0, type=float)
@click.option("--eta", default=2.0, type=float)
@click.option("--save-csv", default=None, type=str)
def cmd_spectrum(m0, beta, gamma, delta, eta, save_csv):
    from dark_octet.spectrum.gmo_formula import GMOMassOperator
    from dark_octet.constants import BenchmarkParameters

    params = BenchmarkParameters(
        M0=m0, beta=beta, gamma=gamma, delta=delta, eta=eta
    )
    gmo = GMOMassOperator(params)
    gmo.print_spectrum()

    if save_csv:
        df = gmo.mass_spectrum_dataframe()
        df.to_csv(save_csv, index=False)
        console.print(f"[green]Saved spectrum to {save_csv}[/green]")


@main.command("homotopy")
@click.option("--plot", is_flag=True, default=False)
@click.option("--output", default="bps_profiles.pdf", type=str)
def cmd_homotopy(plot, output):
    from dark_octet.homotopy import compute_pi2_coset
    from dark_octet.homotopy.winding_number import (
        WindingNumberCalculator,
        TopologicalCharge,
    )

    result = compute_pi2_coset(verbose=True)
    console.print(
        Panel(
            f"π₂(SU(3)/[SO(3)×Z₂]) = ℤ\n"
            f"Stability amplitude: {result['stability_amplitude']:.4e}\n"
            f"Exactness verified: {result['exactness_verified']}",
            title="Homotopy Result",
        )
    )

    tc = TopologicalCharge.compute(n=1)
    tc.print_summary()

    if plot:
        calc = WindingNumberCalculator()
        calc.compute_profiles()
        calc.plot_profiles(save_path=output)
        console.print(f"[green]BPS profiles saved to {output}[/green]")


@main.command("boltzmann")
@click.option("--x-final", default=1000.0, type=float)
@click.option("--plot", is_flag=True, default=False)
@click.option("--output", default="yields.pdf", type=str)
def cmd_boltzmann(x_final, plot, output):
    from dark_octet.boltzmann.solver import BoltzmannSolver

    solver = BoltzmannSolver(x_final=x_final)
    sol = solver.solve()
    solver.print_results(sol)

    if plot:
        solver.plot_yields(sol, save_path=output)
        console.print(f"[green]Yield plot saved to {output}[/green]")


@main.command("sommerfeld")
@click.option("--v-min", default=3.0, type=float)
@click.option("--v-max", default=2000.0, type=float)
@click.option("--n-points", default=200, type=int)
@click.option("--plot", is_flag=True, default=False)
@click.option("--output", default="sigma_T.pdf", type=str)
def cmd_sommerfeld(v_min, v_max, n_points, plot, output):
    from dark_octet.sommerfeld.enhancement import SommerfeldEnhancement

    se = SommerfeldEnhancement()
    se.print_benchmark_table()

    if plot:
        df = se.scan_velocities(v_min_kms=v_min, v_max_kms=v_max, n_points=n_points)
        se.plot_sigma_v(df=df, save_path=output)
        console.print(f"[green]σ_T plot saved to {output}[/green]")


@main.command("relic")
def cmd_relic():
    from dark_octet.relic.abundance import RelicAbundance
    import numpy as np

    from dark_octet.spectrum.gmo_formula import GMOMassOperator
    from dark_octet.constants import BenchmarkParameters

    gmo = GMOMassOperator(BenchmarkParameters())
    df = gmo.mass_spectrum_dataframe()
    masses = df["Mass_GeV"].values
    g_dof = df["g_dof"].values.astype(float)

    ra = RelicAbundance()
    ra.print_summary()

    enh = ra.co_annihilation_enhancement_factor(
        masses_gev=masses, g_dof=g_dof, xf=25.0
    )
    console.print(f"\nCo-annihilation enhancement factor g_eff/g_single = {enh:.3f}")


@main.command("gw")
@click.option("--plot", is_flag=True, default=False)
@click.option("--output", default="gw_spectrum.pdf", type=str)
def cmd_gw(plot, output):
    from dark_octet.signatures.gravitational_waves import GravitationalWaveSpectrum

    gws = GravitationalWaveSpectrum()
    snr = gws.compute_snr(T_obs_years=4.0)

    console.print(
        Panel(
            f"Peak frequency:   {gws.f_peak_hz*1000:.3f} mHz\n"
            f"Peak amplitude:   Ω_GW ≈ {gws.omega_peak:.3e}\n"
            f"LISA SNR (4 yr):  {snr:.2f}",
            title="Gravitational Wave Spectrum",
        )
    )

    if plot:
        gws.plot_spectrum(save_path=output)
        console.print(f"[green]GW spectrum saved to {output}[/green]")


@main.command("all")
@click.option("--output-dir", default="output", type=str)
def cmd_all(output_dir):
    from dark_octet.homotopy import compute_pi2_coset
    from dark_octet.homotopy.winding_number import WindingNumberCalculator, TopologicalCharge
    from dark_octet.spectrum.gmo_formula import GMOMassOperator
    from dark_octet.boltzmann.solver import BoltzmannSolver
    from dark_octet.sommerfeld.enhancement import SommerfeldEnhancement
    from dark_octet.relic.abundance import RelicAbundance
    from dark_octet.signatures.gravitational_waves import GravitationalWaveSpectrum
    from dark_octet.constants import BenchmarkParameters
    import os

    os.makedirs(output_dir, exist_ok=True)

    console.print(Panel("Running full EQST-GP dark octet computation", style="bold blue"))

    console.rule("Appendix A: Homotopy")
    compute_pi2_coset(verbose=True)
    tc = TopologicalCharge.compute(n=1)
    tc.print_summary()
    calc = WindingNumberCalculator()
    calc.compute_profiles()
    calc.plot_profiles(save_path=f"{output_dir}/bps_profiles.pdf")

    console.rule("Section 3: Mass Spectrum")
    gmo = GMOMassOperator(BenchmarkParameters())
    gmo.print_spectrum()
    df_spec = gmo.mass_spectrum_dataframe()
    df_spec.to_csv(f"{output_dir}/dark_octet_spectrum.csv", index=False)

    console.rule("Appendix B: Boltzmann")
    solver = BoltzmannSolver()
    sol = solver.solve()
    solver.print_results(sol)
    solver.plot_yields(sol, save_path=f"{output_dir}/yields.pdf")

    console.rule("Appendix C: Sommerfeld")
    se = SommerfeldEnhancement()
    se.print_benchmark_table()
    df_somm = se.scan_velocities()
    df_somm.to_csv(f"{output_dir}/sommerfeld_scan.csv", index=False)
    se.plot_sigma_v(df=df_somm, save_path=f"{output_dir}/sigma_T.pdf")

    console.rule("Section 4.1: Relic Abundance")
    ra = RelicAbundance()
    ra.print_summary()

    console.rule("Section 5.2: Gravitational Waves")
    gws = GravitationalWaveSpectrum()
    snr = gws.compute_snr()
    gws.plot_spectrum(save_path=f"{output_dir}/gw_spectrum.pdf")
    console.print(f"LISA SNR (4 yr): {snr:.2f}")

    console.print(Panel(f"All outputs written to ./{output_dir}/", style="bold green"))