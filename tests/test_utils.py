import pytest
import numpy as np
from dark_octet.utils.numerics import safe_log, safe_exp, trapz_log, relative_error
from dark_octet.utils.units import gev_to_kg, gev_inv_to_m, fb_to_cm2
 
 
def test_safe_log_zero():
    result = safe_log(0.0)
    assert np.isfinite(result)
 
 
def test_safe_log_positive():
    assert abs(safe_log(1.0)) < 1e-15
 
 
def test_safe_exp_large():
    result = safe_exp(1000.0)
    assert np.isfinite(result)
 
 
def test_trapz_log():
    x = np.logspace(-1, 1, 100)
    y = np.ones_like(x)
    integral = trapz_log(y, x)
    expected = np.log(10.0) - np.log(0.1)
    assert abs(integral - expected) / expected < 0.02
 
 
def test_relative_error_exact():
    err = relative_error(1.042, 1.042)
    assert err == pytest.approx(0.0)
 
 
def test_relative_error_nonzero():
    err = relative_error(1.0, 2.0)
    assert abs(err - 0.5) < 1e-10
 
 
def test_relative_error_zero_ref():
    err = relative_error(1.0, 0.0)
    assert err == float("inf")
 
 
def test_fb_to_cm2():
    sigma_cm2 = fb_to_cm2(1.0)
    assert abs(sigma_cm2 - 1e-39) < 1e-50