# Polarization of Gravitational Waves

This repository now separates the reusable numerical code from exploratory notebooks and paper-writing files.

## Layout

- `src/gw_turbulence/core.py`: numerical kernels and integral evaluators
- `src/gw_turbulence/plotting.py`: scan and plotting helpers
- `src/gw_turbulence/cli.py`: command-line workflow for standard study outputs
- `src/compute_H.py`: compatibility wrapper for older notebook/script imports
- `tests/test_derivations.py`: derivation-identity regression tests
- `tests/test_core.py`: numerical-kernel and grid-behavior tests
- `tests/test_plotting_cli.py`: plotting and CLI workflow tests
- `src/compute.ipynb`: minimal workflow notebook
- `src/numerical_benchmarks.ipynb`: timing and scaling checks
- `src/model_comparison.ipynb`: stationary vs decaying scan notebook
- `src/verify_transformations.ipynb`: symbolic and numeric derivation verification
- `src/legacy_symbolic_notes.md`: cleaned reference note from older symbolic scratch work
- `main.tex`, `derivation.tex`: manuscript and derivation notes

## Common usage

Run the regression tests from the repository root:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests
```

Generate the standard outputs used in the study:

```bash
python -m src.gw_turbulence.cli
```

Skip expensive 2D scans when you only want the 1D spectra:

```bash
python -m src.gw_turbulence.cli --skip-scans
```

Older notebooks that use `from compute_H import *` continue to work through `src/compute_H.py`.

## Next-step workflow

For future study steps, add new physics models in `core.py`, keep analysis-specific plotting in `plotting.py`, and keep notebooks focused on interpretation rather than defining reusable functions inline.
