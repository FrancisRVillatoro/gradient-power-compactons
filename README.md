# Gradient-power compactons

Reproducibility software and numerical data associated with

> **F. R. Villatoro**, *Edge-cusped compactons from gradient-power nonlinear dispersion: exact transitions and singular spectral structure* (2026).

**The manuscript itself is not distributed in this repository.**
This repository contains only source code, numerical data, and generated research outputs used for reproducibility.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/check_fast.py` | Exact symbolic checks of closed-form formulas |
| `scripts/numerics_static.py` | Static profile and independent spectral checks |
| `scripts/time_simulations_padded.py` | Padded Fourier-Galerkin time integrator |
| `scripts/generator_galerkin.py` | Structure-adapted Jacobi-Galerkin generator diagnostic |
| `scripts/generate_figures.py` | Regenerates numerical figures from formulas/data |
| `data/` | CSV datasets for numerical diagnostics and convergence tests |
| `figures/` | Generated vector figures |
| `requirements.txt` | Python dependencies |

## Environment

Tested with Python 3.12 and recent NumPy, SciPy, Matplotlib, Pandas and SymPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick verification

```bash
python scripts/check_fast.py
python scripts/generator_galerkin.py
python scripts/generate_figures.py
```

## Time-dependent simulations

Example:

```bash
python scripts/time_simulations_padded.py --r 3 --N 384 --T 0.1 --pad 3 --nout 1 --out scratch/r3_N384_T01.csv
python scripts/time_simulations_padded.py --r 5 --N 384 --T 0.1 --pad 3 --nout 1 --out scratch/r5_N384_T01.csv
```

## Release

Release `v1.0.0` is archived in Zenodo under DOI `10.5281/zenodo.22671976`.

Development repository:
`https://github.com/FrancisRVillatoro/gradient-power-compactons`

## License

Code and numerical data are released under the MIT License.
