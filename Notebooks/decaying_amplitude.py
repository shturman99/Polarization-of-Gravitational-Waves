#!/usr/bin/env python3
r"""Full-spatial decaying GW *amplitude* vs Mach number (companion to fullspatial_decay.py).

fullspatial_decay.py established the decaying-source peak *location* (p~2.4, M-indep).
This script computes the open other half: the amplitude exponent, i.e. how the GW
energy scales with M for the full-spatial decaying kernel.  This either confirms or
replaces the "M^4 -> M^{5-6}" claim in derivation.tex SIIE (currently cited from
Caprini, not derived here).

EXACT M-FACTORISATION.  In H_decay_fast (Notebooks/fullspatial_decay.py) the Mach
number enters in exactly two places: the prefactor _h_prefactor ~ M^3, and the
temporal factor _conv_ftprod(q, sqrt(x)/M, sqrt(y)/M).  Substituting u = t M in the
cosine integral gives

    _conv_ftprod(q, sqrt(x)/M, sqrt(y)/M) = M * F(q/M; x, y),

    F(w; x, y) = (2 pi / sqrt(x y)) int_0^inf cos(w u)
                 (1 + u/sqrt(x))^{-2/3} (1 + u/sqrt(y))^{-2/3} du,

with F independent of M.  The geometric weight, kernel_bracket and integration
bounds are all M-independent, so on the GW cone q = p,

    Omega_GW(p; M) = p^3 H(p,p;M,R) = M^4 * p^3 * Psi(p, p/M, R).        (*)

Consequences:
  - DEEP IR (p/M -> 0): Psi -> Psi(p,0) is M-independent -> IR amplitude ~ M^4
    exactly (one power steeper than the stationary M^3 prefactor).
  - PEAK (fixed p = p_peak ~ 2.4, so p/M falls as M grows): Psi(2.4, 2.4/M) rises
    with M, so the peak amplitude is steeper than M^4.

This script verifies (*) numerically and measures the IR, peak and integrated
amplitude exponents, decaying vs stationary.

Run: python Notebooks/decaying_amplitude.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

# reuse the validated full-spatial kernels / spectra
from fullspatial_decay import (  # noqa: E402
    H_decay_fast,
    _peak,
    omega_gw_decay,
    omega_gw_stat,
)


def _powerlaw_fit(M, y):
    """Return (exponent, prefactor) of a log-log least-squares fit y = A M^n."""
    M, y = np.asarray(M, float), np.asarray(y, float)
    n, lnA = np.polyfit(np.log(M), np.log(y), 1)
    return float(n), float(np.exp(lnA))


def _local_exponent(M, y):
    """d ln y / d ln M by centred finite differences (M log-spaced)."""
    lM, ly = np.log(np.asarray(M, float)), np.log(np.asarray(y, float))
    return np.gradient(ly, lM)


def ir_amplitude(spec_fn, M, p_ref_over_M=0.1):
    """IR amplitude A = Omega_GW(p_ref)/p_ref^3 at p_ref = p_ref_over_M * M.

    p_ref is tied to M so that p_ref/M is fixed and small -> samples the same point
    on the universal IR plateau for every M (so the measured exponent is the clean
    prefactor exponent, not contaminated by the p/M dependence inside Psi).
    """
    p = p_ref_over_M * M
    return spec_fn(p, M) / p**3


def _validate():
    print("=" * 72)
    print("VALIDATION: exact M^4 factorisation  H_decay_fast(p, wM, M)/M^4 = Psi(p,w)")
    print("=" * 72)
    print("  (hold p and w=q/M fixed, vary M; H/M^4 must be M-independent)")
    for p, w in ((0.5, 1.0), (2.4, 0.5)):
        print(f"\n  p={p}, w=q/M={w}:")
        ref = None
        for M in (0.4, 0.8, 1.6):
            h = H_decay_fast(p, w * M, M=M, R=1e4)
            scaled = h / M**4
            tag = "" if ref is None else f"   (ratio to first: {scaled/ref:.4f})"
            ref = scaled if ref is None else ref
            print(f"    M={M:4.1f}:  H/M^4 = {scaled:.6e}{tag}")
    print("\n  -> ratios ~1.000 confirm the analytic M^4 prefactor (Eq. *).")


def _measure(Ms):
    print("\n" + "=" * 72)
    print("AMPLITUDE EXPONENTS (decaying, R=1e4)")
    print("=" * 72)

    # IR amplitude (deep-IR plateau, p_ref = 0.1 M).
    A_ir = np.array([ir_amplitude(omega_gw_decay, M) for M in Ms])
    n_ir, _ = _powerlaw_fit(Ms, A_ir)
    print(f"\n  IR amplitude (p_ref=0.1 M):   Omega_GW/p^3 ~ M^{n_ir:.3f}   (expect 4.0)")

    # Peak amplitude (decaying peak is ~M-independent at p~2.4).
    p_dec = np.array([_peak(lambda p, m: omega_gw_decay(p, m), M) for M in Ms])
    A_pk = np.array([omega_gw_decay(pp, M) for pp, M in zip(p_dec, Ms)])
    n_pk, _ = _powerlaw_fit(Ms, A_pk)
    loc = _local_exponent(Ms, A_pk)
    print(f"  Peak location p_peak:         {np.array2string(p_dec, precision=2)}")
    print(f"  Peak amplitude Omega(p_peak): ~ M^{n_pk:.3f}   (expect >4, finite-duration steepening)")
    print(f"  Peak local exponent d ln/d ln M over M: "
          f"{np.array2string(loc, precision=2)}")

    # Integrated energy int Omega dln p.
    pg = np.geomspace(0.1, 12.0, 60)
    E = np.array([np.trapezoid([omega_gw_decay(p, M) for p in pg], np.log(pg)) for M in Ms])
    n_E, _ = _powerlaw_fit(Ms, E)
    print(f"  Integrated energy int Omega dln p: ~ M^{n_E:.3f}")

    # Stationary comparison.
    print("\n" + "-" * 72)
    print("STATIONARY comparison")
    A_ir_s = np.array([ir_amplitude(omega_gw_stat, M) for M in Ms])
    n_ir_s, _ = _powerlaw_fit(Ms, A_ir_s)
    p_st = np.array([_peak(lambda p, m: omega_gw_stat(p, m), M) for M in Ms])
    A_pk_s = np.array([omega_gw_stat(pp, M) for pp, M in zip(p_st, Ms)])
    n_pk_s, _ = _powerlaw_fit(Ms, A_pk_s)
    print(f"  IR amplitude:   ~ M^{n_ir_s:.3f}   (expect 3.0, the M^3 prefactor)")
    print(f"  Peak amplitude: ~ M^{n_pk_s:.3f}")
    return dict(A_ir=A_ir, n_ir=n_ir, p_dec=p_dec, A_pk=A_pk, n_pk=n_pk, loc=loc,
                E=E, n_E=n_E, A_ir_s=A_ir_s, n_ir_s=n_ir_s, A_pk_s=A_pk_s, n_pk_s=n_pk_s)


def _figure(Ms, res, name="decaying_amplitude"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)

    # Panel (a): M^4-rescaled spectra -> IR collapse, peak spreads upward.
    ps = np.geomspace(0.03, 10.0, 48)
    for M, ls in ((0.3, ":"), (0.5, "--"), (1.0, "-"), (2.0, "-.")):
        s = np.array([omega_gw_decay(p, M) for p in ps]) / M**4
        ax0.plot(ps, s, ls, color=PALETTE[1], lw=1.6, label=rf"$M={M}$")
    ax0.axvspan(2.0, 2.6, color=PALETTE[2], alpha=0.15, lw=0)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$p=k/k_0$")
    ax0.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,M^4$")
    ax0.set_title(r"$M^4$-rescaled spectra: IR collapse, peak spreads", fontsize=9.5)
    ax0.legend(fontsize=8, loc="lower center", ncol=2)

    # Panel (b): amplitude vs M, normalised to M=1, with fitted slopes.
    def _norm(a):
        i1 = int(np.argmin(np.abs(Ms - 1.0)))
        return np.asarray(a) / a[i1]

    ax1.plot(Ms, _norm(res["A_pk"]), "o-", color=PALETTE[1], lw=1.7, ms=4,
             label=rf"decaying peak $\sim M^{{{res['n_pk']:.1f}}}$")
    ax1.plot(Ms, _norm(res["A_ir"]), "s--", color=PALETTE[1], lw=1.4, ms=3.5,
             label=rf"decaying IR $\sim M^{{{res['n_ir']:.1f}}}$")
    ax1.plot(Ms, _norm(res["A_pk_s"]), "^-", color=PALETTE[0], lw=1.7, ms=4,
             label=rf"stationary peak $\sim M^{{{res['n_pk_s']:.1f}}}$")
    ax1.plot(Ms, _norm(res["A_ir_s"]), "v--", color=PALETTE[0], lw=1.4, ms=3.5,
             label=rf"stationary IR $\sim M^{{{res['n_ir_s']:.1f}}}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Mach number $M$")
    ax1.set_ylabel(r"amplitude (normalised to $M=1$)")
    ax1.set_title("amplitude scaling: decaying vs stationary", fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="upper left")
    for ax in (ax0, ax1):
        apply_max_ticks(ax)

    out = save_figure(fig, name)
    print(f"\nwrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    _validate()
    Ms = np.array([0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0])
    res = _measure(Ms)
    _figure(Ms, res)
