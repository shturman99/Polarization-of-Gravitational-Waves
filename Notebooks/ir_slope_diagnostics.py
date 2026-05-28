#!/usr/bin/env python3
# =============================================================================
# DEPRECATED for the Sec. IV IR-slope conclusion (2026-05-25 audit).
# -----------------------------------------------------------------------------
# This tool fits the IR slope of the AEROACOUSTIC self-similar source
# (Omega ~ p^3 H_exact(p, T_em)).  Per the Sec. IV reframe, the aeroacoustic
# limit only resolves the deep-IR (p<<1) and uses a finite-window / coherence
# parametrization that the reframe found does NOT cleanly map to a single
# physical control (T_em and the decay rate eps0 BOTH flatten the slope).  The
# AUTHORITATIVE Sec. IV kernel for the IR-slope conclusion is now the
# first-principles full-spatial kernel in
#     Notebooks/fullspatial_selfsimilar_exact.py.
# Keep this file for provenance/diagnostics only; do NOT cite its slopes as the
# Sec. IV result.  See MEMORY: project_fullspatial_selfsimilar_exact.md.
# =============================================================================
"""Diagnostic: aeroacoustic self-similar GW IR slope vs source duration and decay rate.

Provenance for the numbers quoted in derivation.tex Sec.~principal-discrepancy
("Resolution"): Omega_GW^aeroac(p) = p^3 H_exact(omega=p, T_em).

scan_decay() -- IR-band slope (fit over 1/T_em..0.5) vs the decay rate eps0 at fixed
T_em: slope rises from ~1.9 (slow decay, eps0->0, coherent) to ~2.5 (fast decay,
eps0=4).  This is the "k^1.9 (slow) to k^2.5 (fast)" interpolation in the paper.

constant_amplitude_check() -- the coherent control: a frozen-amplitude source over a
finite window is the textbook top hat |int e^{iwt}|^2 ~ 1/w^2 -> slope 1 (measured
~1.1).  This is the paper's "coherent over its lifetime -> slope ~1.1 = the DNS law".

(An earlier duration scan, now folded into the figure tool fullspatial_selfsimilar.py,
established that the flattening onset sits at p~1/T_em.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Notebooks"))

from selfsimilar_hybrid import H_exact, eps0_of  # noqa: E402

LOITS = dict(u0=1.0, l0=1.0, p=10 / 7, q=2 / 7, s=4)


def band_slope(T_em, pars, p_lo, p_hi, n=9, n_T=200, n_k=90):
    ps = np.geomspace(p_lo, p_hi, n)
    om = np.array([p**3 * H_exact(p, T_em, n_T=n_T, n_k=n_k, **pars) for p in ps])
    good = om > 0
    c = np.polyfit(np.log(ps[good]), np.log(om[good]), 1)
    return c[0]


def scan_decay():
    T_em = 60.0
    print(f"IR-band slope (fit over 1/T_em..0.5) vs decay rate eps0,  T_em={T_em}")
    print("slow decay -> coherent top-hat -> slope 1 ; fast decay -> slope 3\n")
    print(f"  {'tau_st':>8}{'eps0':>9}{'IR slope':>10}")
    for tau_st in (1e3, 30.0, 8.0, 3.0, 1.0, 0.5, 0.25):
        pars = {**LOITS, "tau_st": tau_st}
        e0 = eps0_of(**pars)
        # IR band: above the duration-transition (1/T_em), below the eddy freq
        sl = band_slope(T_em, pars, p_lo=3.0 / T_em, p_hi=0.5)
        print(f"  {tau_st:8.3g}{e0:9.3g}{sl:10.2f}")


def constant_amplitude_check():
    """Pure top-hat: no decay at all (freeze amplitude), finite window -> expect slope ~1."""
    print("\nControl: near-frozen amplitude (tau_st=1e6), long window, vary T_em:")
    print(f"  {'T_em':>8}{'IR slope (1/T_em..0.5)':>26}")
    pars = {**LOITS, "tau_st": 1e6}
    for T_em in (20.0, 60.0, 200.0):
        n_T = int(min(500, max(120, 8 * T_em)))
        sl = band_slope(T_em, pars, p_lo=3.0 / T_em, p_hi=0.5, n_T=n_T)
        print(f"  {T_em:8.1f}{sl:26.2f}")


if __name__ == "__main__":
    scan_decay()
    constant_amplitude_check()
