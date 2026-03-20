# Notebook Change Log

## Overview of fixes and additions across all `.ipynb` files

---

## `decaying_turbulence_sweep.ipynb` *(new notebook)*

Dedicated sweep for decaying turbulence over `R = [10², 3×10², 10³, 3×10³, 10⁴]`
and `M = [0.1, 1.0]`.

### Cell `a0000009` — Time-evolution plots
- **Removed** `axvline(6e-9)` IR-break reference line (confirmed numerical artefact, not physical).
- **Added** two-panel layout: left = absolute spectra (log-log), right = relative difference
  `(h_c − h_c^ref) / h_c^ref` from the `R = 10³` reference curve (semilogx), making the
  otherwise invisible 8–20% R-dependence visible on a linear scale.

### Cell `a0000011` — Power-law fits
- **Removed** `axvline(6e-9)` IR-break annotation.
- **Fixed** inertial fit window: previously hard-coded `(1e-4, 2e-3)` Hz straddled the spectral
  peak, giving wrong negative/near-zero slopes. Now detects `f_peak` via `nanargmax` and fits
  on `[f_peak/100, f_peak/5]`, guaranteed to land on the rising inertial slope below the peak.

### Cell `a0000013` — Inertial-range zoom
- **Fixed** fit window: same peak-detection approach as above. Fit range is
  `[max(freq_inert[0], f_peak/20), min(f_peak*0.6, freq_inert[-1])]`,
  clamped to the available high-resolution grid so it always stays in-range.

---

## `decaying_tau1_study.ipynb` *(new notebook)*

Study of the temporal kernel discrepancy between `core.py` and `derivation.tex`,
and `τ₁` regime scan.

### Cell `s0000007` — Convolution functions
- **Increased** `n_pts` from 300 → 600 in both `convolution_paper` and `convolution_code`.
  The paper kernel `g_paper` oscillates rapidly; 300 points produced noisy curves.

### Cell `s0000012` — τ₁ regime spectra and ratio panel
- **Fixed** right panel being completely empty: the old guard `(C_p > 0) & (C_code_ref > 0)`
  silenced all values because `Re[g_paper · g_paper]` is often negative (oscillating kernel).
  Changed to `abs(C_p) / abs(C_code_ref)` with guard `abs(C_code_ref) > 0`.
- Updated y-axis label to `|C_paper(τ₁)| / |C_code|` and plot title to "Scaling ratio (|paper| / |code|)".

---

## `decaying_turbulence_low_frequency.ipynb`

### Cell `7613613c` — Normalized strain definition
- **Fixed** `normalized_strain`: replaced `np.clip(val, 0, None)` with
  `np.where(val > 0, val, np.nan)`. The old version produced `sqrt(0) = 0` at
  spectral cutoffs, which matplotlib renders as a vertical line drop on log plots.
  `np.nan` causes matplotlib to skip those points cleanly.

---

## `workflow_demo.ipynb`

### Cell `4zdawltvfhn` — Gogoberidze 2007 Fig. 1 reproduction
- **Fixed** `np.clip` → `np.where(val > 0, val, np.nan)` for both `scaled_exact` and
  `scaled_aero` strain arrays.

### Cell `0c20e8f8` — Stationary vs decaying comparison (full band)
- **Fixed** inline strain calculations for `sc_stat` and `sc_dec`:
  `np.clip` → `np.where(val > 0, val, np.nan)`.

### Cell `3f100a9a` — Stationary vs decaying comparison (smaller window)
- **Fixed** same `np.clip` → `np.where` replacement.
- Cleaned up guide-line f-string label syntax (was `f~to~{pow}`, now `f^{{{pow}}}`).

---

## Files with no changes required

| File | Reason |
|---|---|
| `compute.ipynb` | No strain normalization; grid/plotting only. |
| `numerical_benchmarks.ipynb` | No strain normalization; timing and kernel shape only. |
| `verify_transformations.ipynb` | Symbolic/numerical verification; no GW strain output. |

---

## Root cause of vertical-line artefacts

All strain normalizations previously used:
```python
np.sqrt(np.clip(q * H, 0, None))
```
At spectral cutoffs where `q * H` passes through zero (or becomes slightly negative from
numerical noise), `clip` returns `0`, and `sqrt(0) = 0`. On a log-log plot matplotlib
draws a vertical line from the last valid value down to zero.

The fix throughout all notebooks is:
```python
val = q * H
np.sqrt(np.where(val > 0, val, np.nan))
```
`np.nan` causes matplotlib to break the line, leaving no artefact.

---

## Root cause of wrong inertial power-law slopes

The old fit windows in `decaying_turbulence_sweep.ipynb` were fixed at `(1e-4, 2e-3)` Hz
(cells `a0000011` and `a0000013`). For `M = 0.1` the spectral peak is at `~1.5×10⁻⁴` Hz
and for `M = 1.0` at `~5×10⁻⁴` Hz. Both windows therefore began at or above the peak,
fitting the falling post-peak region and returning slopes of `≈ −0.18` to `≈ +0.01`
instead of the true inertial rising slope.

Fix: detect `f_peak = freq[nanargmax(sc)]` and fit in `[f_peak/100, f_peak/5]`.
