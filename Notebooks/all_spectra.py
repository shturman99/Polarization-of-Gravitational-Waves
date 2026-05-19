"""Unified survey of every dimensionless GW spectrum derived in derivation.tex.

Convention
----------
We plot the GW *energy-density* spectrum, related to the characteristic strain
by  Omega_GW(f) ~ f^2 * h_c^2(f).  In our dimensionless variables, with
h_c^2(p) = p * H(p, p), that means

    Omega_GW(p) ~ p^3 * H(p, q=p)

evaluated on the sound-cone diagonal q = p (or Omega = p for the Kraichnan
models).  Causality + a finite white-noise source give Omega_GW ~ p^3 in the
IR — provided H(p->0) is finite.  Models with a 1/p pole at the origin
(e.g. the monochromatic delta(k-k_0) source, K_0(p)/p) cannot satisfy this
because the underlying fluid has no IR power.  The script:

  1. computes S(p) for every derived model on a common log p grid,
  2. plots each model on its own figure with NO parameter annotation on the
     plot itself (axes show only p and S(p));  parameters live in the caption
     of the corresponding figure in derivation.tex,
  3. fits the log-log slope on the IR end of every curve and flags any case
     where it deviates from 3 by more than ``SLOPE_TOLERANCE``.

Run
---
    PYTHONPATH=src python3 Notebooks/all_spectra.py

Outputs go to ``images/spectrum_*.pdf``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.core import (  # noqa: E402
    H_delta_k_decay,
    H_delta_k_kraichnan,
    H_k0_analytic,
    H_pq,
    H_white_decay,
    H_white_kraichnan,
)
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

MACH_LIST = (0.03, 0.1, 0.3, 1.0, 3.0)
R_LIST = (1.0e3, 1.0e4, 1.0e5)

PS_FULL  = np.logspace(-3, 1, 60)         # full range, no triangle constraint
PS_DELTA = np.logspace(-3, np.log10(1.9), 60)  # delta(k-k0) needs p < 2

EXPECTED_IR_SLOPE = 3.0
SLOPE_TOLERANCE   = 0.5
IR_FRACTION       = 0.25   # use leftmost 25% of valid points for the slope fit


# ---------------------------------------------------------------------------
# Spectrum convention
# ---------------------------------------------------------------------------

def _diagonal(H_fn: Callable, ps: np.ndarray, **kw) -> np.ndarray:
    """Omega_GW(p) ~ p^3 * H(p, q=p) on the sound-cone diagonal."""
    out = np.empty_like(ps, dtype=float)
    for i, p in enumerate(ps):
        try:
            out[i] = p ** 3 * H_fn(p, p, **kw)
        except Exception as exc:
            print(f"    warn: H_fn failed at p={p:.3e} ({kw}): {exc}")
            out[i] = np.nan
    return out


def _h_k0(qs: np.ndarray, M: float, **kw) -> np.ndarray:
    """p->0 aeroacoustic limit:  Omega_GW(q) ~ q^3 * H_k0_analytic(q, M, ...)."""
    return qs ** 3 * H_k0_analytic(qs, M=M, **kw)


def _ir_slope(ps: np.ndarray, spec: np.ndarray) -> float:
    """log-log slope of `spec` on the IR end of the grid."""
    mask = (ps > 0) & (spec > 0) & np.isfinite(spec)
    if mask.sum() < 4:
        return float("nan")
    p_use = ps[mask]
    s_use = spec[mask]
    n_ir = max(4, int(IR_FRACTION * len(p_use)))
    slope, _ = np.polyfit(np.log(p_use[:n_ir]), np.log(s_use[:n_ir]), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _plot_model(name: str, ps: np.ndarray, curves: list[np.ndarray]):
    """One figure per model.  No parameter text on the plot — captions only."""
    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    for c, spec in enumerate(curves):
        mask = (ps > 0) & (spec > 0) & np.isfinite(spec)
        ax.loglog(ps[mask], spec[mask], color=PALETTE[c % len(PALETTE)])

    # p^3 reference (dotted grey), anchored to first-curve IR point
    spec0 = curves[0]
    mask0 = (ps > 0) & (spec0 > 0) & np.isfinite(spec0)
    if mask0.any():
        p_ref = ps[mask0][0]
        y_ref = spec0[mask0][0]
        p_line = ps[ps < 0.5]
        if len(p_line) > 0:
            ax.loglog(p_line, y_ref * (p_line / p_ref) ** 3,
                      color="0.4", ls=":")

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p) \propto p^{3}\,H(p,\,p)$")
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-model runners
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    out_pdf: Path
    slopes: list[tuple[dict, float]]   # (params, slope) per curve


def _record_flags(res: ModelResult, bucket: list):
    for params, slope in res.slopes:
        if not np.isfinite(slope) or abs(slope - EXPECTED_IR_SLOPE) > SLOPE_TOLERANCE:
            bucket.append((res.name, params, slope))


def model_kraichnan_kolmogorov(R: float = 1.0e4) -> ModelResult:
    """H_pq(p,p,M,R): Kraichnan decorrelation + Kolmogorov spectrum."""
    name = "spectrum_kraichnan_kolmogorov"
    curves, slopes = [], []
    for M in MACH_LIST:
        spec = _diagonal(H_pq, PS_FULL, M=M, R=R)
        curves.append(spec)
        slopes.append(({"M": M, "R": R}, _ir_slope(PS_FULL, spec)))
    out = _plot_model(name, PS_FULL, curves)
    print(f"saved {out}")
    return ModelResult(name, out, slopes)


def model_aeroacoustic_limit(R: float = 1.0e4) -> ModelResult:
    """H_k0_analytic(q,M,R): p->0 aeroacoustic limit (Kahniashvili et al. 2008)."""
    name = "spectrum_aeroacoustic_limit"
    qs = PS_FULL
    curves, slopes = [], []
    for M in MACH_LIST:
        spec = _h_k0(qs, M=M, R=R)
        curves.append(spec)
        slopes.append(({"M": M, "R": R}, _ir_slope(qs, spec)))
    out = _plot_model(name, qs, curves)
    print(f"saved {out}")
    return ModelResult(name, out, slopes)


def model_delta_k_kraichnan() -> ModelResult:
    """H_delta_k_kraichnan(p,Omega=p): closed form, no Mach dependence."""
    name = "spectrum_delta_k_kraichnan"
    spec = _diagonal(H_delta_k_kraichnan, PS_DELTA)
    out = _plot_model(name, PS_DELTA, [spec])
    print(f"saved {out}")
    return ModelResult(name, out, [({}, _ir_slope(PS_DELTA, spec))])


def model_delta_k_decay() -> ModelResult:
    """H_delta_k_decay(p,q=p): mono spectrum + decay temporal convolution."""
    name = "spectrum_delta_k_decay"
    spec = _diagonal(H_delta_k_decay, PS_DELTA, n_points=120)
    out = _plot_model(name, PS_DELTA, [spec])
    print(f"saved {out}")
    return ModelResult(name, out, [({}, _ir_slope(PS_DELTA, spec))])


def model_white_kraichnan() -> ModelResult:
    """H_white_kraichnan(p,Omega=p,R): real-space delta + Kraichnan."""
    name = "spectrum_white_kraichnan"
    curves, slopes = [], []
    for R in R_LIST:
        spec = _diagonal(H_white_kraichnan, PS_FULL, R=R)
        curves.append(spec)
        slopes.append(({"R": R}, _ir_slope(PS_FULL, spec)))
    out = _plot_model(name, PS_FULL, curves)
    print(f"saved {out}")
    return ModelResult(name, out, slopes)


def model_white_decay() -> ModelResult:
    """H_white_decay(p,q=p,R): real-space delta + decay."""
    name = "spectrum_white_decay"
    curves, slopes = [], []
    for R in R_LIST:
        spec = _diagonal(H_white_decay, PS_FULL, R=R, n_points=120)
        curves.append(spec)
        slopes.append(({"R": R}, _ir_slope(PS_FULL, spec)))
    out = _plot_model(name, PS_FULL, curves)
    print(f"saved {out}")
    return ModelResult(name, out, slopes)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_MODELS = (
    model_kraichnan_kolmogorov,
    model_aeroacoustic_limit,
    model_delta_k_kraichnan,
    model_delta_k_decay,
    model_white_kraichnan,
    model_white_decay,
)


def main():
    apply_paper_style()
    flagged: list[tuple[str, dict, float]] = []
    for runner in ALL_MODELS:
        try:
            res = runner()
            _record_flags(res, flagged)
        except Exception as exc:
            print(f"  !!! {runner.__name__} crashed: {exc}")
            flagged.append((runner.__name__, {"_error": str(exc)}, float("nan")))

    print()
    print("=" * 78)
    print(f"FLAGGED CASES (IR slope deviates from {EXPECTED_IR_SLOPE} by > {SLOPE_TOLERANCE})")
    print("=" * 78)
    if not flagged:
        print("  none.")
    else:
        for name, params, slope in flagged:
            print(f"  {name:36s}  slope = {slope:+.2f}   params = {params}")


if __name__ == "__main__":
    main()
