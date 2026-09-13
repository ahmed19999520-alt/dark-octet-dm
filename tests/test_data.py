import pytest
import json
from pathlib import Path
from dark_octet.data.constraints import ConstraintLoader
from dark_octet.data.validators import ResultValidator, PAPER_VALUES
from dark_octet.data.fetcher import ObservationalDataFetcher, DataPackage


def test_constraint_loader_relic():
    loader = ConstraintLoader()
    ra = loader.relic_abundance()
    assert "omega_dm_h2" in ra
    assert abs(ra["omega_dm_h2"] - 0.120) < 0.001


def test_constraint_loader_si_array():
    loader = ConstraintLoader()
    df = loader.self_interaction_array()
    assert len(df) >= 3
    assert "sigma_T_per_m_upper" in df.columns


def test_constraint_loader_validate_sigma():
    loader = ConstraintLoader()
    result = loader.validate_sigma_T_prediction(v_kms=10.0, sigma_pred=0.52)
    assert result["passed"]
    assert result["v_kms"] == 10.0


def test_constraint_loader_fail_sigma():
    loader = ConstraintLoader()
    result = loader.validate_sigma_T_prediction(v_kms=1000.0, sigma_pred=2.0)
    assert not result["passed"]


def test_paper_values_completeness():
    required_keys = [
        "r21", "r31", "omega_dm_h2", "freeze_out_x",
        "sigma_T_10kms_cm2pg", "f_peak_mHz", "snr_lisa_4yr"
    ]
    for key in required_keys:
        assert key in PAPER_VALUES, f"Missing paper value: {key}"


def test_validator_r21():
    val = ResultValidator()
    r = val.validate("r21", computed=1.042)
    assert r.passed
    assert r.pull < 1.0


def test_validator_r31():
    val = ResultValidator()
    r = val.validate("r31", computed=0.809)
    assert r.passed


def test_validator_fail_case():
    val = ResultValidator(tolerance_sigma=2.0)
    r = val.validate("r21", computed=1.20)
    assert not r.passed
    assert r.pull > 2.0


def test_fetcher_load_returns_datapackage():
    fetcher = ObservationalDataFetcher(cache_dir="/tmp/dark_octet_test_cache")
    pkg = fetcher.load()
    assert isinstance(pkg, DataPackage)
    assert "omega_dm_h2" in pkg.planck_2018
    assert "H0" in pkg.shoes_2022


def test_fetcher_hubble_tension():
    fetcher = ObservationalDataFetcher(cache_dir="/tmp/dark_octet_test_cache")
    pkg = fetcher.load()
    tension = fetcher.hubble_tension_significance(pkg)
    assert tension["tension_shoes_cmb_sigma"] > 3.0
    assert tension["tension_resolved_by_eqst"]


def test_fetcher_direct_detection():
    fetcher = ObservationalDataFetcher(cache_dir="/tmp/dark_octet_test_cache")
    pkg = fetcher.load()
    dd = fetcher.direct_detection_status(pkg)
    assert not dd["excluded_by_lz"]
    assert dd["detectable_by_darwin"]


def test_fetcher_si_constraints():
    fetcher = ObservationalDataFetcher(cache_dir="/tmp/dark_octet_test_cache")
    pkg = fetcher.load()
    si = fetcher.self_interaction_all_pass(pkg)
    assert si["all_constraints_satisfied"]


def test_datapackage_save_load(tmp_path):
    fetcher = ObservationalDataFetcher(cache_dir=str(tmp_path))
    pkg = fetcher.load()
    out_path = str(tmp_path / "test_pkg.json")
    pkg.save_json(out_path)
    pkg2 = DataPackage.load_json(out_path)
    assert pkg2.planck_2018["omega_dm_h2"] == pytest.approx(0.1200)
    assert pkg2.loaded_from_cache


def test_full_validation_pipeline():
    val = ResultValidator(tolerance_sigma=4.0)
    df = val.run_full_validation()
    assert not df.empty
    n_pass = val.n_passed()
    n_total = len(val.results)
    assert n_pass >= int(0.80 * n_total), f"Only {n_pass}/{n_total} passed"