# Dark Octet Dark Matter 

Computational library supporting:

**"Topological Integer-Winding Dark Matter: Stability from π 2 (SU(3)/SO(3)) = Z"**
Ahmed Ali, 2026

---

## Repository Structure

```
dark-octet-dm/
├── src/dark_octet/          # Core physics library
│   ├── constants.py         # Physical and benchmark constants
│   ├── homotopy/            # Appendix A: π₂(SU(3)/[SO(3)×Z₂]) = Z
│   ├── spectrum/            # Section 3: Extended GMO mass formula
│   ├── boltzmann/           # Appendix B: 8-species Boltzmann hierarchy
│   ├── sommerfeld/          # Appendix C: Sommerfeld enhancement
│   ├── relic/               # Section 4.1: Ω_DM h² computation
│   └── signatures/          # Section 5: GW spectrum, collider rates
├── notebooks/               # Jupyter notebooks for all appendices
├── tests/                   # pytest test suite (35+ tests)
├── data/                    # Observational constraints JSON
└── docker/                  # Reproducible Docker environment
```

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/ahmed19999520-alt/dark-octet-dm
cd dark-octet-dm
docker compose -f docker/docker-compose.yml up
```
https://github.com/ahmed19999520-alt/dark-octet-dm

Open `http://localhost:8888` in your browser.

### Local install

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Run all computations

```bash
python -c "
from dark_octet.spectrum.gmo_formula import GMOMassOperator
from dark_octet.boltzmann.solver import BoltzmannSolver
from dark_octet.sommerfeld.enhancement import SommerfeldEnhancement
from dark_octet.relic.abundance import RelicAbundance

gmo = GMOMassOperator()
gmo.print_spectrum()

solver = BoltzmannSolver()
sol = solver.solve()
solver.print_results(sol)
solver.plot_yields(sol, save_path='yields.pdf')

se = SommerfeldEnhancement()
se.print_benchmark_table()
se.plot_sigma_v(save_path='sigma_T.pdf')

ra = RelicAbundance()
ra.print_summary()
"
```

## Key Numerical Results

| Quantity | Computed | Paper value |
|----------|----------|-------------|
| M(χ⁰) | 165.67 GeV | 165.67 GeV |
| r₂₁ = M(χ⁺⁺)/M(χ⁰) | 1.042 | 1.042 ± 0.003 |
| r₃₁ = M(Σ⁰)/M(χ⁰) | 0.809 | 0.809 ± 0.004 |
| x_f | ≈ 25 | ≈ 25 |
| Ω_DM h² | 0.120 | 0.1200 ± 0.0012 |
| σ_T/m at 10 km/s | 0.52 cm²/g | 0.52 ± 0.09 cm²/g |
| f_peak GW | 1.65 mHz | 1.65 mHz |
| SNR (LISA 4yr) | 8.5 | 8.5 |

## Notebooks

| Notebook | Content |
|----------|---------|
| `AppendixA_Homotopy.ipynb` | π₂ classification, BPS profiles, winding number |
| `AppendixB_Boltzmann.ipynb` | 8-species coupled ODEs, freeze-out, relic density |
| `AppendixC_Sommerfeld.ipynb` | Schrödinger equation, enhancement factor, σ_T(v) |

## Citation


```bibtex
@article{Ali2026TopologicalDMZ2,
  author  = {Ali, Ahmed},
  title   = {Topological Integer-Winding Dark Matter: Stability from π 2 (SU(3)/SO(3)) = Z},
  year    = {2026},
  note    = {Arxiv_hep-ph},
  url     = {https://github.com/ahmed19999520-alt/dark-octet-dm}
}
```


## License

MIT © Ahmed Ali 2026