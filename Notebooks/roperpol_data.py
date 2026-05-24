#!/usr/bin/env python3
r"""Single source of truth for quantities DERIVED from the digitized Roper Pol Fig. 1.

Every number the paper quotes about the simulation (k0, the GW peak ratio, the energy
fraction Omega_M, the effective Mach range, and the measured slopes) is COMPUTED here
from Notebooks/roperpol_fig1_{fluid,gw}.csv -- nothing is hardcoded.  The figure scripts
(roperpol_comparison.py, gw_peak_vs_mach.py) and the test suite all import from here, so
the figures, the paper text, and the tests cannot drift apart.

Conventions: the CSV columns are (k, Omega/k).  "Omega_GW convention" means
Omega_GW(k) = k * (Omega_GW/k) = drho_GW/dln k; its peak is the "~2 k0" peak.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, Omega/k) for name in {'fluid','gw'}, sorted by k."""
    d = np.loadtxt(_DIR / f"roperpol_fig1_{name}.csv", delimiter=",", skiprows=1)
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1]


def _logpeak(k: np.ndarray, y: np.ndarray, w: int = 9) -> float:
    """Robust spectral-peak location: log-smoothed argmax + local log-parabola vertex."""
    ly = np.log(y)
    sm = np.convolve(ly, np.ones(w) / w, "same")
    i = int(np.argmax(sm[w:-w])) + w
    il, ir = max(i - 3, 0), min(i + 4, len(k))
    c = np.polyfit(np.log(k[il:ir]), np.log(y[il:ir]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


def slope(k: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Least-squares log-log slope of y(k) over [lo, hi]."""
    m = (k >= lo) & (k <= hi)
    return float(np.polyfit(np.log(k[m]), np.log(y[m]), 1)[0])


@lru_cache(maxsize=1)
def k0() -> float:
    """Fluid spectral-peak wavenumber k0 (source scale), robust log-parabola fit."""
    kf, of = load("fluid")
    return _logpeak(kf, of)


@lru_cache(maxsize=1)
def gw_peak_ratio() -> float:
    """GW peak in the Omega_GW = drho/dln k convention, divided by k0 (the '~2 k0')."""
    kg, og = load("gw")
    return _logpeak(kg, kg * og) / k0()


@lru_cache(maxsize=1)
def energy_fraction() -> float:
    """Turbulent energy fraction Omega_M = int (Omega_M/k) dk from the fluid curve."""
    kf, of = load("fluid")
    return float(np.trapz(of, kf))


def effective_mach() -> tuple[float, float]:
    """Effective Mach range (sqrt(Omega_M), sqrt(2 Omega_M)) -- deeply subsonic."""
    om = energy_fraction()
    return float(np.sqrt(om)), float(np.sqrt(2.0 * om))


# --- measured slopes used in the paper (computed, not asserted) -----------------
def fluid_ir_slope(lo: float = 120, hi: float = 400) -> float:
    kf, of = load("fluid")
    return slope(kf, of, lo, hi)            # Omega_M/k = E(k) ~ k^4 (Batchelor)


def fluid_uv_slope(lo: float = 2e3, hi: float = 3e4) -> float:
    kf, of = load("fluid")
    return slope(kf, of, lo, hi)            # ~ k^-5/3 (Kolmogorov)


def gw_ir_slope_omega_gw(lo: float = 130, hi: float = 600) -> float:
    kg, og = load("gw")
    return slope(kg, kg * og, lo, hi)       # Omega_GW ~ k^1 (flat Omega/k) -- the discrepancy


def gw_uv_slope_over_k(lo: float = 3e3, hi: float = 3e4) -> float:
    kg, og = load("gw")
    return slope(kg, og, lo, hi)            # Omega_GW/k ~ k^-11/3


if __name__ == "__main__":
    print(f"k0                 = {k0():.1f}")
    print(f"GW peak / k0       = {gw_peak_ratio():.3f}   (Omega_GW convention; ~2 k0)")
    print(f"Omega_M            = {energy_fraction():.3e}")
    print(f"effective Mach     = {effective_mach()[0]:.3f}--{effective_mach()[1]:.3f}")
    print(f"fluid IR / UV      = k^{fluid_ir_slope():+.2f} / k^{fluid_uv_slope():+.2f}")
    print(f"GW IR (Omega_GW)   = k^{gw_ir_slope_omega_gw():+.2f}   (causal would be +3)")
    print(f"GW UV (Omega/k)    = k^{gw_uv_slope_over_k():+.2f}   (k^-11/3 = -3.67)")
