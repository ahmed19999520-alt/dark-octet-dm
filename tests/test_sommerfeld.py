import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.sommerfeld.schrodinger import (
    sommerfeld_coulomb_analytic, sommerfeld_yukawa, sommerfeld_table,
    dimensionless_params,
)
from dark_octet.sommerfeld.enhancement import (
    xZp, F_yukawa, sigma_T_born_gevm2, sigma_T_per_mass,
    all_bounds_satisfied, v_critical_kms, PAPER_VALUES,
    OBSERVATIONAL_LIMITS,
)
from dark_octet.constants import ALPHA_DARK, M_ZP_GEV, C_LIGHT_KM_S


class TestSommerfeldCoulomb:

    def test_coulomb_at_10kms(self):
        S = sommerfeld_coulomb_analytic(10.0, ALPHA_DARK)
        assert S > 3000
        assert S < 5000

    def test_coulomb_at_1000kms(self):
        S = sommerfeld_coulomb_analytic(1000.0, ALPHA_DARK)
        assert S > 10.0
        assert S < 50.0

    def test_coulomb_decreases_with_velocity(self):
        S_slow = sommerfeld_coulomb_analytic(10.0)
        S_fast = sommerfeld_coulomb_analytic(1000.0)
        assert S_slow > S_fast

    def test_coulomb_large_coupling_limit(self):
        S = sommerfeld_coulomb_analytic(10.0, 0.020)
        arg = 2.0 * np.pi * 0.020 / (10.0 / C_LIGHT_KM_S)
        assert abs(S - arg) / arg < 0.01

    def test_yukawa_suppression_factor(self):
        S_c   = sommerfeld_coulomb_analytic(10.0)
        S_yuk = sommerfeld_yukawa(10.0)
        ratio = S_yuk / S_c
        assert abs(ratio - 0.52) < 0.02


class TestYukawaFormula:

    def test_F_large_x(self):
        x    = 1.0e6
        F    = F_yukawa(x)
        F_approx = 1.0 / (2.0 * x)
        assert abs(F - F_approx) / F_approx < 0.01

    def test_F_positive(self):
        for x in [0.1, 1.0, 10.0, 1000.0]:
            assert F_yukawa(x) > 0.0

    def test_F_monotone(self):
        xs = [0.1, 1.0, 10.0, 100.0]
        Fs = [F_yukawa(x) for x in xs]
        for i in range(len(Fs) - 1):
            assert Fs[i] > Fs[i + 1]

    def test_xZp_increases_with_lower_v(self):
        x1 = xZp(10.0, 82.835)
        x2 = xZp(1000.0, 82.835)
        assert x1 > x2

    def test_v_critical(self):
        v_crit = v_critical_kms()
        expected = (M_ZP_GEV / 165.67) * C_LIGHT_KM_S
        assert abs(v_crit - expected) < 1.0


class TestSIDMBounds:

    def test_sigma_T_dwarf_order(self):
        sigma = sigma_T_per_mass(10.0)
        assert 1e-5 < sigma < 10.0

    def test_all_bounds_check_structure(self):
        for system, data in OBSERVATIONAL_LIMITS.items():
            assert "v_kms" in data
            assert "limit_cm2_g" in data

    def test_velocity_scaling(self):
        s10   = sigma_T_per_mass(10.0)
        s1000 = sigma_T_per_mass(1000.0)
        assert s10 > s1000

    def test_bullet_cluster_ratio(self):
        s10   = sigma_T_per_mass(10.0)
        s1000 = sigma_T_per_mass(1000.0)
        ratio = s10 / s1000
        assert ratio > 10.0
