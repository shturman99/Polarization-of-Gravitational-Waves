# Polarization of Gravitational Waves from Turbulence

Numerical code and manuscript for the study *"Polarization of Gravitational Waves
from Turbulence: Tension Between Simulation and Analytic Approaches."* The code
computes the gravitational-wave (GW) spectrum sourced by turbulent
(Kolmogorov / decaying-MHD) stresses, for several spatial spectra and temporal
decorrelation models, and reproduces every figure and quantitative claim in the
paper (`derivation.tex`).

## Requirements

- Python **3.10+**
- `numpy`, `scipy`, `mpmath`, `matplotlib`, `sympy`, `pillow` (installed
  automatically below)
- Optional: `mpi4py` for distributing the 2-D parameter scans across MPI ranks
- Optional (to rebuild the PDF): a LaTeX distribution with `latexmk`
  (e.g. TeX Live)

## Quickstart

```bash
git clone https://github.com/shturman99/Polarization-of-Gravitational-Waves.git
cd Polarization-of-Gravitational-Waves

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Option A — exact, tested versions (reproduces the paper bit-for-bit):
pip install -r requirements.txt
pip install -e .

# Option B — looser ranges + test tooling:
pip install -e ".[dev]"

# run the test suite
pytest
```

`pip install -e .` installs the reusable library as the importable package
`gw_turbulence`, so notebooks and scripts work without any `sys.path`
manipulation:

```python
from gw_turbulence import H_pq, H_pq_decaying, apply_paper_style, save_figure
```

## Repository layout

```
src/gw_turbulence/      installable library
  core.py               GW kernels and integral evaluators (stationary + decaying)
  plotting.py           scan/plot helpers built on the kernels
  plot_style.py         shared Matplotlib paper style (palette, sizes, save_figure)
  cli.py                command-line workflow for the standard study outputs
  mpi.py                optional MPI helpers (no-op unless mpi4py is installed)
Notebooks/              standalone analysis scripts, one per figure (see below)
                        plus exploratory .ipynb notebooks
tests/                  pytest suite (see "Tests")
images/                 generated figures (vector PDF) used by the manuscript
derivation.tex          the manuscript / full derivation
requirements.txt        pinned, tested dependency versions
```

## Reproducing the figures

Each script in `Notebooks/` is self-contained and regenerates one or more
figures into `images/`. Run them from the repository root, e.g.:

```bash
python Notebooks/fullspatial_selfsimilar.py     # IR-slope resolution figure
python Notebooks/fullspatial_decay.py           # source-scale GW peak
python Notebooks/roperpol_comparison.py         # digitized-data comparison
python Notebooks/stationary_mach_peak.py        # stationary Mach-peak scaling
```

The scripts add the needed paths themselves, so no install of the `Notebooks/`
folder is required — only the `gw_turbulence` package (`pip install -e .`) and
the dependencies above. Most scripts print a short validation table and write a
`*.pdf` into `images/`.

`Notebooks/validate_ir_resolution.py` independently re-derives the paper's
infrared-slope claims from scratch (a brute-force double-time integral) and
prints a PASS/FAIL summary.

## Command-line workflow

```bash
gw-turbulence              # standard study outputs (installed entry point)
gw-turbulence --skip-scans # 1-D spectra only; skip the expensive 2-D scans
# equivalently: python -m gw_turbulence.cli [--skip-scans]
```

## Tests

```bash
pytest                              # full suite
pytest tests/test_paper_claims.py   # recomputes every quantitative claim in the paper
```

`tests/test_paper_claims.py` recomputes each number stated in `derivation.tex`
from the kernels and the digitized reference data (no hard-coded values);
`test_core.py`, `test_derivations.py`, and `test_plotting_cli.py` cover the
kernels, derivation identities, and plotting/CLI workflow.

## Optional: MPI for large scans

```bash
pip install -e ".[mpi]"
mpirun -n 4 python -m gw_turbulence.cli   # distributes grid rows across ranks
```

Without `mpi4py` installed everything runs serially; MPI is never required.

## Building the manuscript

```bash
latexmk -pdf derivation.tex
```

Requires a LaTeX distribution (TeX Live or equivalent). The committed
`derivation.pdf` is kept in sync; figures are read from `images/`.

## License

MIT — see [LICENSE](LICENSE).
