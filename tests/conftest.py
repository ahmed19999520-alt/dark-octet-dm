import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def benchmark_masses():
    from dark_octet.spectrum.gmo_formula import octet_spectrum
    return octet_spectrum()


@pytest.fixture(scope="session")
def benchmark_splits(benchmark_masses):
    from dark_octet.spectrum.gmo_formula import mass_splittings
    return mass_splittings(benchmark_masses)


@pytest.fixture(scope="session")
def benchmark_ratios(benchmark_masses):
    from dark_octet.spectrum.gmo_formula import mass_ratios
    return mass_ratios(benchmark_masses)


@pytest.fixture(scope="session")
def decay_rates(benchmark_masses):
    from dark_octet.boltzmann.decay_rates import all_decay_rates
    return all_decay_rates(benchmark_masses)


@pytest.fixture(scope="session")
def lifetimes(benchmark_masses):
    from dark_octet.boltzmann.decay_rates import all_lifetimes_s
    return all_lifetimes_s(benchmark_masses)


@pytest.fixture(scope="session")
def gw_data():
    from dark_octet.signatures.gravitational_waves import gw_summary
    return gw_summary()


@pytest.fixture(scope="session")
def observational_constraints():
    from dark_octet.data.fetcher import load_constraints
    return load_constraints()


@pytest.fixture(scope="session")
def chi2_table():
    from dark_octet.data.constraints import build_chi2_table
    return build_chi2_table()


@pytest.fixture(scope="session")
def paper_validation():
    from dark_octet.data.validators import PaperValidation
    pv = PaperValidation()
    pv.run_all()
    return pv


PAPER_MASSES = {
    "chi0":    165.67,
    "chi_p":   167.37,
    "chi_m":   168.67,
    "chi_pp":  180.37,
    "Sigma0":  192.67,
    "Sigma_m": 195.67,
    "Sigma_p": 207.37,
    "Lambda":  208.67,
}


def approx_rel(val: float, ref: float, tol: float = 0.005) -> bool:
    if ref == 0.0:
        return abs(val) < tol
    return abs(val - ref) / abs(ref) < tol


def approx_abs(val: float, ref: float, tol: float = 0.005) -> bool:
    return abs(val - ref) < tol
