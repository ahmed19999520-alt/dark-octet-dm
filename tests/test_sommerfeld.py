import pytest
import numpy as np
from dark_octet.sommerfeld.enhancement import SommerfeldEnhancement
from dark_octet.sommerfeld.schrodinger import (
    coulomb_enhancement,
    yukawa_form_factor,
)
from dark_octet.constants import BENCH


S_CALC = SommerfeldEnhancement()


def test_coulomb_limit_high_velocity():
    v_high = 0.9
    S = coulomb_enhancement(v_high, BENCH.alpha_dark)
    assert S >= 1.0


def test_coulomb_enhancement_increases_at_low_v():
    S_low = coulomb_enhancement(1e-4, BENCH.alpha_dark)
    S_high = coulomb_enhancement(0.5, BENCH.alpha_dark)
    assert S_low > S_high


def test_form_factor_range():
    F = yukawa_form_factor(
        v_rel=1e-5, m_zprime_gev=BENCH.m_zprime_gev, m_dm_gev=165.67
    )
    assert 0.0 <= F <= 2.0


def test_sigma_T_dwarf_order_of_magnitude():
    res = S_CALC.compute_single(v_rel_kms=10.0)
    assert 0.1 < res.sigma_T_cm2_per_g < 5.0


def test_sigma_T_bullet_cluster():
    res = S_CALC.compute_single(v_rel_kms=1000.0)
    assert res.sigma_T_cm2_per_g < 0.47


def test_velocity_scaling_exponent():
    v1, v2 = 10.0, 100.0
    r1 = S_CALC.compute_single(v1)
    r2 = S_CALC.compute_single(v2)
    ratio = r1.sigma_T_cm2_per_g / r2.sigma_T_cm2_per_g
    v_ratio = v2 / v1
    exponent = np.log(ratio) / np.log(v_ratio)
    assert 1.5 < exponent < 3.0


def test_regime_classification():
    res_dwarf = S_CALC.compute_single(10.0)
    res_cluster = S_CALC.compute_single(20000.0)
    assert res_dwarf.regime == "Sommerfeld"
    assert res_cluster.regime == "Born"


def test_scan_returns_dataframe():
    df = S_CALC.scan_velocities(v_min_kms=5.0, v_max_kms=2000.0, n_points=20)
    assert len(df) == 20
    assert "sigma_T_cm2_per_g" in df.columns
    assert df["sigma_T_cm2_per_g"].min() > 0.0


def test_x_zprime_large_at_low_v():
    res = S_CALC.compute_single(1.0)
    assert res.x_zprime > 1e6