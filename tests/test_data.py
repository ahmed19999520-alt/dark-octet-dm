import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dark_octet.data.fetcher import (
    load_constraints, get_planck, get_lz_limit, get_bullet_cluster_limit,
    get_dwarf_limit, get_lisa_band, get_darwin_target, get_ska_dao,
    EMBEDDED_CONSTRAINTS,
)
from dark_octet.data.constraints import (
    build_chi2_table, total_chi2, goodness_of_fit,
    load_and_validate_constraints, chi2_contribution,
)
from dark_octet.data.validators import PaperValidation


class TestConstraintLoader:

    def test_load_returns_dict(self):
        data = load_constraints()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_planck_omega(self):
        planck = get_planck()
        assert abs(planck["Omega_DM_h2"] - 0.1200) < 1e-6
        assert abs(planck["Omega_DM_h2_err"] - 0.0012) < 1e-6

    def test_planck_neff_limit(self):
        planck = get_planck()
        assert planck["Delta_Neff_limit"] == 0.17

    def test_lz_limit(self):
        lz = get_lz_limit()
        assert abs(lz - 5e-47) / 5e-47 < 0.01

    def test_bullet_cluster_limit(self):
        bc = get_bullet_cluster_limit()
        assert abs(bc - 0.47) < 0.01

    def test_dwarf_limit(self):
        dw = get_dwarf_limit()
        assert dw == 1.0

    def test_lisa_band(self):
        f_min, f_max = get_lisa_band()
        assert f_min == 1e-4
        assert f_max == 1e-1
        assert f_min < f_max

    def test_darwin_target(self):
        dt = get_darwin_target()
        assert abs(dt - 3e-49) / 3e-49 < 0.01

    def test_ska_dao(self):
        ska = get_ska_dao()
        assert abs(ska["r_DAO_Mpc"] - 2.3) < 0.01
        assert ska["sigma_detection"] > 5.0

    def test_embedded_keys(self):
        keys = list(EMBEDDED_CONSTRAINTS.keys())
        required = ["planck_2018", "lz_2024", "randall_2008", "lisa_2017", "ska_2018", "darwin_2016"]
        for k in required:
            assert k in keys, f"Missing key: {k}"

    def test_fcc_cdr(self):
        data = load_constraints()
        fcc = data.get("fcc_cdr_2019", {})
        assert abs(fcc.get("sqrt_s_ee_GeV", 0) - 365.0) < 0.1
        assert abs(fcc.get("lumi_ee_ab", 0) - 1.5) < 0.01


class TestChi2Table:

    def test_build_returns_list(self):
        table = build_chi2_table()
        assert isinstance(table, list)
        assert len(table) >= 4

    def test_chi2_nonnegative(self):
        table = build_chi2_table()
        for row in table:
            assert row["chi2"] >= 0.0, f"Negative chi2 for {row['observable']}"

    def test_total_chi2_finite(self):
        table = build_chi2_table()
        chi2  = total_chi2(table)
        assert 0.0 < chi2 < 1000.0

    def test_goodness_of_fit_structure(self):
        table = build_chi2_table()
        gof   = goodness_of_fit(table)
        assert "chi2" in gof
        assert "ndof" in gof
        assert "chi2_ndof" in gof
        assert gof["n_obs"] == len(table)

    def test_chi2_contribution_formula(self):
        chi2 = chi2_contribution(1.0, 1.0, 1.0)
        assert chi2 == 0.0
        chi2 = chi2_contribution(0.0, 1.0, 1.0)
        assert abs(chi2 - 1.0) < 1e-10

    def test_constraint_validation(self):
        result = load_and_validate_constraints()
        assert "complete" in result
        assert "loaded" in result


class TestPaperValidation:

    def test_all_pass(self):
        pv     = PaperValidation()
        result = pv.run_all()
        assert result["n_passed"] == result["n_total"], (
            f"Failed: {[k for k,v in result['results'].items() if not v['passed']]}"
        )

    def test_validation_categories(self):
        pv = PaperValidation()
        result = pv.run_all()
        expected = {"mass_spectrum", "mass_ratios", "relic_abundance",
                    "direct_detection", "sidm_bounds", "gw_signature",
                    "dao", "sommerfeld", "lifetimes"}
        assert expected.issubset(set(result["results"].keys()))
