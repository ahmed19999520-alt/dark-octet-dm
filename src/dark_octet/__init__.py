from .constants import (
    G_DARK, ALPHA_DARK, G_F, M_ZP_GEV,
    M0_GEV, BETA_GEV, GAMMA_GEV, DELTA_GEV, ETA_GEV,
    V_DELTA_GEV, Y_CHI, YCHI_VDELTA,
    MASS_PAPER, OCTET_ORDER, OCTET_QN, OCTET_GDOF,
)
from .spectrum.gmo_formula import (
    octet_spectrum, mass_ratios, mass_splittings,
    validate_spectrum, benchmark_ratios,
)
from .homotopy.long_exact_sequence import verify_homotopy_groups
from .homotopy.winding_number import bps_analytic, bps_mass_bound
from .boltzmann.cross_sections import sigma_v_chi0_chi0, sigma_si_kinetic_mixing
from .boltzmann.decay_rates import all_decay_rates, all_lifetimes_s
from .boltzmann.solver import omega_dm_h2_analytic, relic_abundance_lee_weinberg
from .sommerfeld.schrodinger import sommerfeld_coulomb_analytic, sommerfeld_yukawa
from .sommerfeld.enhancement import sigma_T_per_mass, sidm_table
from .relic.abundance import (
    omega_dm_h2_lee_weinberg, sigma_si_kinetic_mixing_cm2,
    direct_detection_summary, relic_density_summary,
)
from .signatures.gravitational_waves import gw_summary, dao_scale
from .signatures.collider import fcce_summary, collider_table
from .data.validators import PaperValidation
from .data.fetcher import load_constraints

__version__ = "2.0.0"
__author__  = "Ahmed Ali"
__email__   = "ahmed.ali.mpi@hotmail.com"
__orcid__   = "0009-0007-8516-2621"
