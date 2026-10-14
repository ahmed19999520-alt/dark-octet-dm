import numpy as np
from ..spectrum.gmo_formula import (
    octet_spectrum, mass_ratios, validate_spectrum, benchmark_ratios,
)
from ..sommerfeld.schrodinger import sommerfeld_coulomb_analytic, sommerfeld_yukawa
from ..sommerfeld.enhancement import sigma_T_per_mass, all_bounds_satisfied, PAPER_VALUES
from ..relic.abundance import (
    omega_dm_h2_lee_weinberg, sigma_si_kinetic_mixing_cm2, direct_detection_summary,
    relic_density_summary, pseudo_goldstone_neff,
)
from ..signatures.gravitational_waves import gw_summary, dao_scale
from ..signatures.collider import fcce_summary, fcc_hh_summary
from ..boltzmann.decay_rates import all_lifetimes_s
from ..data.fetcher import get_lz_limit, get_darwin_target, get_planck, get_bullet_cluster_limit


class PaperValidation:

    TOLERANCES = {
        "mass_GeV":       0.005,
        "ratio":          0.001,
        "omega_h2":       0.002,
        "sigma_T_m":      0.15,
        "sigma_SI":       0.05,
        "f_peak_mHz":     0.05,
        "sommerfeld":     0.10,
        "lifetime":       2.0,
    }

    def __init__(self):
        self.results = {}

    def validate_mass_spectrum(self) -> dict:
        spec    = validate_spectrum(tol=self.TOLERANCES["mass_GeV"])
        passed  = all(v["pass"] for v in spec.values())
        self.results["mass_spectrum"] = {"passed": passed, "details": spec}
        return self.results["mass_spectrum"]

    def validate_mass_ratios(self) -> dict:
        ratios   = benchmark_ratios()
        tol      = self.TOLERANCES["ratio"]
        details  = {}
        passed   = True
        for key, data in ratios.items():
            diff   = abs(data["value"] - data["paper"])
            ok     = diff < tol + data["unc"]
            details[key] = {
                "code":   data["value"],
                "paper":  data["paper"],
                "unc":    data["unc"],
                "diff":   round(diff, 5),
                "pass":   ok,
            }
            if not ok:
                passed = False
        self.results["mass_ratios"] = {"passed": passed, "details": details}
        return self.results["mass_ratios"]

    def validate_relic_abundance(self) -> dict:
        omega   = omega_dm_h2_lee_weinberg(3.0e-26, 27.0, 106.75)
        planck  = get_planck()
        diff    = abs(omega - planck["Omega_DM_h2"])
        tol     = self.TOLERANCES["omega_h2"] + planck["Omega_DM_h2_err"]
        passed  = diff < tol
        result  = {
            "code":       round(omega, 4),
            "paper":      0.120,
            "planck":     planck["Omega_DM_h2"],
            "diff":       round(diff, 5),
            "pass":       passed,
            "x_f_paper":  27.0,
            "g_eff_paper": 5.09,
        }
        self.results["relic_abundance"] = {"passed": passed, "details": result}
        return self.results["relic_abundance"]

    def validate_direct_detection(self) -> dict:
        dd      = direct_detection_summary()
        lz_lim  = get_lz_limit()
        darwin  = get_darwin_target()
        below   = dd["sigma_SI_cm2"] < lz_lim
        near_d  = dd["sigma_SI_cm2"] > darwin
        passed  = below
        result  = {
            "code_cm2":    dd["sigma_SI_cm2"],
            "paper_cm2":   3.1e-48,
            "below_LZ":    below,
            "within_DARWIN": near_d,
            "pass":        passed,
        }
        self.results["direct_detection"] = {"passed": passed, "details": result}
        return self.results["direct_detection"]

    def validate_sidm_bounds(self) -> dict:
        satisfied = all_bounds_satisfied()
        details   = {}
        for v, paper_val in PAPER_VALUES.items():
            code_val = sigma_T_per_mass(v)
            rel_diff = abs(code_val - paper_val) / paper_val if paper_val > 0 else 0.0
            details[f"v_{v}_kms"] = {
                "v_kms":    v,
                "code":     round(code_val, 5),
                "paper":    paper_val,
                "rel_diff": round(rel_diff, 3),
            }
        self.results["sidm_bounds"] = {
            "passed":  satisfied,
            "details": details,
        }
        return self.results["sidm_bounds"]

    def validate_gw_signature(self) -> dict:
        gw      = gw_summary()
        f_paper = 1.65
        diff    = abs(gw["f_peak_mHz"] - f_paper)
        tol     = self.TOLERANCES["f_peak_mHz"]
        passed  = diff < tol
        result  = {
            "f_peak_mHz_code":  gw["f_peak_mHz"],
            "f_peak_mHz_paper": f_paper,
            "diff_mHz":         round(diff, 5),
            "in_LISA_band":     gw["in_LISA_band"],
            "SNR_paper":        8.5,
            "pass":             passed,
        }
        self.results["gw_signature"] = {"passed": passed, "details": result}
        return self.results["gw_signature"]

    def validate_dao(self) -> dict:
        dao    = dao_scale()
        r_code = dao["r_DAO_Mpc"]
        r_pap  = 2.3
        diff   = abs(r_code - r_pap)
        passed = diff < dao["r_DAO_unc"]
        result = {
            "r_DAO_code":   r_code,
            "r_DAO_paper":  r_pap,
            "diff_Mpc":     round(diff, 4),
            "SKA_sigma":    dao["SKA_sigma"],
            "pass":         passed,
        }
        self.results["dao"] = {"passed": passed, "details": result}
        return self.results["dao"]

    def validate_sommerfeld(self) -> dict:
        v10   = 10.0
        v1000 = 1000.0
        s10   = sommerfeld_yukawa(v10)
        s1000 = sommerfeld_yukawa(v1000)
        s_c10 = sommerfeld_coulomb_analytic(v10)

        diff_10   = abs(s10 - 1960.0) / 1960.0
        diff_1000 = abs(s1000 - 19.4) / 19.4
        tol       = self.TOLERANCES["sommerfeld"]
        passed    = diff_10 < tol and diff_1000 < tol
        result    = {
            "S_10_code":     round(s10, 1),
            "S_10_paper":    1960,
            "S_1000_code":   round(s1000, 1),
            "S_1000_paper":  19.4,
            "S_coulomb_10":  round(s_c10, 0),
            "pass":          passed,
        }
        self.results["sommerfeld"] = {"passed": passed, "details": result}
        return self.results["sommerfeld"]

    def validate_lifetimes(self) -> dict:
        taus    = all_lifetimes_s()
        tau_chi = taus["chi_p"]
        tau_sig = taus["Sigma0"]
        bbn_ok  = tau_chi < 1e-3 and tau_sig < 1e-3
        rel_chi = abs(tau_chi - 2.9e-17) / 2.9e-17
        rel_sig = abs(tau_sig - 6.6e-19) / 6.6e-19
        passed  = rel_chi < 5.0 and rel_sig < 3.0
        result  = {
            "tau_chi_p_s_code":   tau_chi,
            "tau_chi_p_s_paper":  2.9e-17,
            "tau_Sigma0_s_code":  tau_sig,
            "tau_Sigma0_s_paper": 6.6e-19,
            "before_BBN":         bbn_ok,
            "pass":               passed,
        }
        self.results["lifetimes"] = {"passed": passed, "details": result}
        return self.results["lifetimes"]

    def run_all(self) -> dict:
        self.validate_mass_spectrum()
        self.validate_mass_ratios()
        self.validate_relic_abundance()
        self.validate_direct_detection()
        self.validate_sidm_bounds()
        self.validate_gw_signature()
        self.validate_dao()
        self.validate_sommerfeld()
        self.validate_lifetimes()
        n_passed = sum(1 for v in self.results.values() if v["passed"])
        n_total  = len(self.results)
        return {
            "results":   self.results,
            "n_passed":  n_passed,
            "n_total":   n_total,
            "all_pass":  n_passed == n_total,
        }

    def print_summary(self) -> None:
        summary = self.run_all()
        width   = 45
        print("=" * 60)
        print(f"  Dark Octet DM — Paper Validation Summary")
        print("=" * 60)
        for key, val in summary["results"].items():
            status = "PASS" if val["passed"] else "FAIL"
            print(f"  {key:<35} [{status}]")
        print("-" * 60)
        print(f"  Passed: {summary['n_passed']}/{summary['n_total']}")
        print("=" * 60)
