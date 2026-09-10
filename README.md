# Gradient-power compactons — reproducibility code, data, and figures

Public reproducibility repository containing source code, numerical data, and generated figures for computations on gradient-power compactons.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/check_fast.py` | Exact symbolic checks of closed-form formulas |
| `scripts/numerics_static.py` | Static-profile and independent spectral checks |
| `scripts/time_simulations_padded.py` | Padded Fourier-Galerkin time integrator |
| `scripts/generator_galerkin.py` | Structure-adapted Jacobi-Galerkin generator diagnostic |
| `scripts/generate_figures.py` | Regenerates the numerical figures |
| `data/` | CSV datasets for numerical diagnostics and convergence tests |
| `figures/` | Generated reproducibility figures |
| `requirements.txt` | Python dependencies |

## Reproducibility environment

The required Python packages are listed in `requirements.txt`. No software environment is bundled with this repository.

## Archived release

Version `1.0.0`: DOI `10.5281/zenodo.22671976`

Development repository:
`https://github.com/FrancisRVillatoro/gradient-power-compactons`

## License

Code and numerical data are released under the MIT License.
