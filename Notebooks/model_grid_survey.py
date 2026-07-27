#!/usr/bin/env python3
r"""Systematic survey of every (spatial source) x (temporal UETC) kernel.

Every kernel in derivation.tex is a choice of a SPATIAL source model S and a
TEMPORAL unequal-time correlator T.  This script measures, on one common
convention, the properties that actually distinguish them:

    IR slope   d ln Omega / d ln p   well below the peak
    peak       argmax_p Omega_GW(p)
    UV slope   d ln Omega / d ln p   well above the peak

with Omega_GW(p) = p^3 H(p, p) on the sound-cone diagonal q = p.

The numbers printed here are the ones quoted in the comparison table of
derivation.tex, so the table and the code cannot drift apart.  Nothing is
hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gw_turbulence.core import (  # noqa: E402
    H_delta_k_decay,
    H_delta_k_kraichnan,
    H_pq,
    H_white_decay,
    H_white_kraichnan,
    K0_p,
)
from fullspatial_decay import H_decay_fast  # noqa: E402

M_REF = 1.0
R_REF = 1e4


def _slope(ps, spec, lo, hi):
    m = (ps >= lo) & (ps <= hi) & (spec > 0) & np.isfinite(spec)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(ps[m]), np.log(spec[m]), 1)[0])


def _diag(fn, ps):
    out = np.empty(len(ps))
    for i, p in enumerate(ps):
        try:
            out[i] = p**3 * fn(p)
        except Exception:
            out[i] = np.nan
    return out


def _peak(ps, spec):
    m = np.isfinite(spec) & (spec > 0)
    return float(ps[m][np.argmax(spec[m])]) if m.any() else float("nan")


def survey():
    # (label, callable H(p) on the diagonal, p-grid, IR band, UV band)
    truncated = np.geomspace(1e-4, 1.99, 90)      # models with Theta(2-p)
    full = np.geomspace(1e-4, 30.0, 90)

    models = [
        ("K41      x sweeping",
         lambda p: H_pq(p, p, M=M_REF, R=R_REF), full, (1e-4, 1e-3), (5, 25)),
        ("K41      x decay",
         lambda p: H_decay_fast(p, p, M=M_REF, R=R_REF), full, (1e-4, 1e-3), (5, 25)),
        ("delta(k) x sweeping",
         lambda p: H_delta_k_kraichnan(p, p), truncated, (1e-4, 1e-3), None),
        ("delta(k) x decay",
         lambda p: H_delta_k_decay(p, p), truncated, (1e-4, 1e-3), None),
        ("delta(r) x sweeping",
         lambda p: H_white_kraichnan(p, p, R=R_REF), full, (1e-4, 1e-3), (5, 25)),
        ("delta(r) x decay",
         lambda p: H_white_decay(p, p, R=R_REF), full, (1e-4, 1e-3), (5, 25)),
        ("delta(k) x impulsive",
         lambda p: K0_p(p) / p if 0 < p <= 2 else 0.0, truncated, (1e-4, 1e-3), None),
    ]

    print("=" * 78)
    print(f"MODEL GRID SURVEY   (M = {M_REF}, R = {R_REF:.0e}, diagonal q = p)")
    print("=" * 78)
    print(f"  {'model':<22}{'IR slope':>10}{'peak p':>10}{'UV slope':>11}   note")
    rows = []
    for label, fn, ps, ir, uv in models:
        spec = _diag(fn, ps)
        ir_s = _slope(ps, spec, *ir)
        pk = _peak(ps, spec)
        uv_s = _slope(ps, spec, *uv) if uv else float("nan")
        note = "Theta(2-p): no UV tail" if uv is None else ""
        rows.append((label, ir_s, pk, uv_s))
        uv_txt = "     --" if uv is None else f"{uv_s:11.2f}"
        print(f"  {label:<22}{ir_s:10.2f}{pk:10.2f}{uv_txt}   {note}")
    print("\n  IR band p in [1e-4,1e-3];  UV band p in [5,25] where a tail exists.")
    print("  CAVEATS, so the numbers are not over-read:")
    print("   * Theta(2-p) models are cut by the triangle inequality: their 'peak'")
    print("     is the cutoff edge and they have no UV tail at all.")
    print("   * delta(r) (white noise) has E ~ k^2, i.e. it is UV-DOMINATED; its")
    print("     spectrum rises to the dissipation cutoff, so its 'peak' is R, not")
    print("     a dynamical scale.")
    print("   * a sweeping (Gaussian) UETC gives a super-exponential cutoff, not a")
    print("     power law -- a fitted 'UV slope' there is meaningless and merely")
    print("     reports how steep the fit window happens to be.")
    print("   * the decay UV slope is M-dependent (the tail sets in at q ~ M).")
    return rows


def uv_is_power_law(label):
    """Only the power-law (decay) UETC has a genuine UV power law."""
    return "decay" in label and "delta(k)" not in label


def peak_mach_exponent(machs=(0.1, 0.2, 0.4, 0.8, 1.6)):
    """Mach exponent d ln p_peak / d ln M, at FIXED k0, for each temporal model.

    This is the regime diagnostic of Sec.(mach-exponent) in derivation.tex: it
    separates sources whose temporal factor imposes a frequency cutoff at q ~ M
    (the peak then tracks the cutoff, exponent 1) from those whose does not (the
    peak is then pinned by the spatial structure, exponent 0).
    """
    machs = np.asarray(machs, dtype=float)
    models = [
        ("stationary (Kraichnan sweeping)", lambda p, M: H_pq(p, p, M=M, R=R_REF)),
        ("decaying   (BK2016 power law)", lambda p, M: H_decay_fast(p, p, M=M, R=R_REF)),
        ("impulsive  (delta-in-time)", lambda p, M: K0_p(p) / p if 0 < p <= 2 else 0.0),
    ]
    print("\n" + "=" * 78)
    print("MACH EXPONENT OF THE PEAK  (fixed k0)")
    print("=" * 78)
    print(f"  {'model':<34}{'d ln p_peak / d ln M':>22}   p_peak(M)")
    out = {}
    for label, fn in models:
        ps = np.geomspace(1e-3, 60.0, 700)
        peaks = []
        for M in machs:
            spec = np.array([p**3 * fn(p, M) for p in ps])
            peaks.append(ps[np.argmax(spec)])
        peaks = np.array(peaks)
        exponent = float(np.polyfit(np.log(machs), np.log(peaks), 1)[0])
        out[label] = exponent
        print(f"  {label:<34}{exponent:>+22.3f}   "
              + " ".join(f"{v:.2f}" for v in peaks))
    print("\n  Gogoberidze et al. (2007) state f_peak ~ M k0 -- the stationary row.")
    print("  Auclair et al. (2022) find the peak pinned to the integral scale,")
    print("  independent of velocity -- the decaying AND impulsive rows.")
    print("  (M up to 1.6 is used to pin the exponent; M>1 is formally superluminal,")
    print("   u0>c, since M=u0/c in the Gogoberidze convention.)")
    print("  NOTE the impulsive kernel contains no M at all, so its exponent is")
    print("  identically 0: velocity-independence is NOT specific to the power-law")
    print("  UETC, and the diagnostic reads 'is there a q ~ M cutoff', not 'which UETC'.")
    return out


if __name__ == "__main__":
    survey()
    peak_mach_exponent()
