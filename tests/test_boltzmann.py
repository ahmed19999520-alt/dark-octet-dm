import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.boltzmann.cross_sections import (
    sigma_v_chi0_chi0, griest_seckel_geff, griest_seckel_sigma_eff,
    sigma_si_kinetic_mixing,
)
from dark_octet.boltzmann.decay_rates import (
    decay_rate_two_body, all_decay_rates, all_lifetimes_s, bbn_safety_check,
)
from dark_octet.boltzmann.solver import (
    entropy_density, hubble_rate, yield_equilibrium,
    omega_dm_h2_analytic, relic_abundance_lee_weinberg,
)
from dark_octet.relic.abundance import (
    omega_dm_h2_lee_weinberg, sigma_si_kinetic_mixing_cm2,
    direct_detection_summary, relic_density_summary,
)
from dark_octet.spectrum.gmo_formula import octet_spectrum
from dark_octet.constants import (
    G_DARK, G_F, M_ZP_GEV, G_STAR_XF, M_PL_REDUCED_GEV, OCTET_GDOF,
)


class TestCrossSections:

    def setup_method(self):
        self.masses = octet_spectrum()

    def test_sigma_v_positive(self):
        sv = sigma_v_chi0_chi0(165.67, 6.0)
        assert sv > 0.0

    def test_geff_benchmark(self):
        g = griest_seckel_geff(self.masses, self.masses["chi0"], 27.0, OCTET_GDOF)
        assert abs(g - 5.09) < 0.5

    def test_geff_single_species_limit(self):
        single = {"chi0": self.masses["chi0"]}
        gdof_s = {"chi0": 2}
        g = griest_seckel_geff(single, self.masses["chi0"], 27.0, gdof_s)
        assert abs(g - 2.0) < 0.01


class TestDecayRates:

    def setup_method(self):
        self.masses = octet_spectrum()

    def test_chi0_stable(self):
        rates = all_decay_rates(self.masses)
        assert rates["chi0"] == 0.0

    def test_chi_p_lifetime(self):
        taus = all_lifetimes_s(self.masses)
        assert 1e-18 < taus["chi_p"] < 1e-14

    def test_sigma0_lifetime(self):
        taus = all_lifetimes_s(self.masses)
        assert abs(taus["Sigma0"] - 6.6e-19) / 6.6e-19 < 2.0

    def test_all_decay_before_bbn(self):
        bbn = bbn_safety_check(self.masses)
        for name, data in bbn.items():
            if name != "chi0":
                assert data["before_bbn"], f"{name} decays after BBN"

    def test_heavier_decays_faster(self):
        taus = all_lifetimes_s(self.masses)
        assert taus["Lambda"] < taus["chi_p"]

    def test_decay_rate_formula_chi_p(self):
        dm    = 1.70
        mp    = 167.37
        md    = 165.67
        gamma = decay_rate_two_body(mp, md, dm, G_DARK)
        assert gamma > 0.0
        tau   = 6.582119569e-25 / gamma
        assert abs(tau - 2.9e-17) / 2.9e-17 < 2.0


class TestBoltzmannKinematics:

    def setup_method(self):
        self.masses = octet_spectrum()

    def test_entropy_density_positive(self):
        s = entropy_density(1.0)
        assert s > 0.0

    def test_hubble_rate_positive(self):
        H = hubble_rate(1.0)
        assert H > 0.0

    def test_yield_equilibrium_positive(self):
        Y = yield_equilibrium("chi0", 20.0, self.masses)
        assert Y > 0.0

    def test_yield_eq_decreases_with_x(self):
        Y1 = yield_equilibrium("chi0", 5.0,  self.masses)
        Y2 = yield_equilibrium("chi0", 50.0, self.masses)
        assert Y1 > Y2


class TestRelicAbundance:

    def test_omega_h2_benchmark(self):
        omega = omega_dm_h2_analytic(2.568e-9, 27.0, 106.75)
        assert abs(omega - 0.120) < 0.01

    def test_omega_h2_planck_consistent(self):
        omega = omega_dm_h2_lee_weinberg(2.568e-9, 27.0)
        assert abs(omega - 0.1200) / 0.1200 < 0.10

    def test_sigma_si_below_lz(self):
        sigma = sigma_si_kinetic_mixing_cm2()
        assert sigma < 5.0e-47

    def test_sigma_si_above_darwin(self):
        sigma = sigma_si_kinetic_mixing_cm2()
        assert sigma > 3.0e-49

    def test_sigma_si_below_lz(self):
        sigma = sigma_si_kinetic_mixing_cm2()
        assert sigma < 5.0e-47

    def test_relic_summary_keys(self):
        summary = relic_density_summary()
        assert "Omega_DM_h2" in summary
        assert "x_f" in summary
        assert "g_eff_xf" in summary
