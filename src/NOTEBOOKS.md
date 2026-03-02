# Notebook Breakdown

The notebooks now have distinct roles. Keep reusable logic in `src/gw_turbulence/` and use notebooks only for inspection, derivation checks, or one-off exploration.

## `compute.ipynb`

Minimal workflow notebook.

- Uses the packaged functions from `compute_H.py`
- Best for quick spectra generation and loading saved scan outputs
- Should stay lightweight and avoid long derivations

## `numerical_benchmarks.ipynb`

Low-level numerical diagnostics notebook.

- Benchmarks representative point evaluations
- Explores scaling relations and the decaying-kernel shape
- Best for deciding tolerances and runtime expectations

## `model_comparison.ipynb`

Model-comparison notebook.

- Builds coarse stationary and decaying grids
- Compares the two models side by side
- Includes a small-\(p\) consistency check before larger scans

## `verify_transformations.ipynb`

Derivation verification notebook.

- Symbolic and numeric checks tied to `derivation.tex`
- Reproducible checks should also exist in `tests/test_derivations.py`
- Use this notebook when a result is easier to inspect interactively than in a unit test

## Legacy symbolic work

The previous `Integralcalcualtaion.ipynb` scratch notebook has been retired.

- Its durable parts are summarized in `src/legacy_symbolic_notes.md`
- Anything reproducible should live in tests or package code, not in a long symbolic scratch notebook

## Reproducibility Rules

- Do not commit notebook outputs
- Do not commit generated figures or `.npy` grids
- Regenerate outputs with the CLI:

```bash
python -m src.gw_turbulence.cli
```

- Run regression checks with:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests
```
