import pytest
import numpy as np
from dark_octet.analysis.chi2_comparison import Chi2Analyzer, PAPER_PREDICTIONS
from dark_octet.analysis.parameter_scan import ParameterScanner
from dark_octet.analysis.predictions_table import PredictionsTable, ALL_PREDICTIONS
 
 
def test_chi2_analyzer_perfect_agreement():
    analyzer = Chi2Analyzer()
    for key, (exp, sigma, _) in PAPER_PREDICTIONS.items():
        analyzer.add_prediction(key, exp)
    results = analyzer.chi2_eqst()
    assert results["_chi2_per_N"] < 0.01
 
 
def test_chi2_analyzer_nonzero_pull():
    analyzer = Chi2Analyzer()
    analyzer.add_prediction("r21", 1.100)
    results = analyzer.chi2_eqst()
    assert results["r21"]["pull"] > 10.0
 
 
def test_aic_bic_eqst_lower():
    analyzer = Chi2Analyzer()
    eqst = analyzer.aic_bic(chi2=8.0, n_data=12, n_params=0)
    sm   = analyzer.aic_bic(chi2=17.0, n_data=12, n_params=19)
    assert eqst["AIC"] < sm["AIC"]
    delta = sm["AIC"] - eqst["AIC"]
    assert delta > 10.0
 
 
def test_parameter_scanner_r21_increases_with_delta():
    scanner = ParameterScanner()
    deltas = np.linspace(0.0, 15.0, 10)
    df = scanner.scan_r21_vs_delta(delta_range=deltas)
    assert df["r21"].is_monotonic_increasing
 
 
def test_parameter_scanner_r31_decreases_with_gamma():
    scanner = ParameterScanner()
    gammas = np.linspace(2.0, 20.0, 10)
    df = scanner.scan_r31_vs_gamma(gamma_range=gammas)
    assert df["r31"].is_monotonic_decreasing
 
 
def test_predictions_table_length():
    pt = PredictionsTable()
    df = pt.to_dataframe()
    assert len(df) == len(ALL_PREDICTIONS)
    assert len(df) >= 13
 
 
def test_predictions_table_columns():
    pt = PredictionsTable()
    df = pt.to_dataframe()
    for col in ["observable", "eqst_value", "facility", "timeline"]:
        assert col in df.columns
 
 
def test_current_testable_not_empty():
    pt = PredictionsTable()
    current = pt.current_testable()
    assert len(current) >= 2
 
 
def test_upcoming_testable_by_2040():
    pt = PredictionsTable()
    upcoming = pt.upcoming_testable(by_year=2040)
    assert len(upcoming) >= 5