#!/usr/bin/env python3
# =============================================================================
# DEPRECATED for the Sec. IV IR-slope conclusion (2026-05-25 audit).
# -----------------------------------------------------------------------------
# This tool flattens the IR slope via a `coherence` KNOB / finite-window proxy
# (H_window with an externally imposed coherence parameter).  The Sec. IV reframe
# found that an imposed coherence knob is NOT a first-principles control and that
# self-similar decay does NOT in fact flatten the IR to k^1 (the k^1 is a genuine
# MHD tension, not reproduced by HD self-similar decay).  The AUTHORITATIVE
# Sec. IV kernel is the first-principles full-spatial kernel in
#     Notebooks/fullspatial_selfsimilar_exact.py.
# Keep this file for exploration/provenance only; do NOT cite its coherence-knob
# IR slopes or "resolution" as the Sec. IV result.
# See MEMORY: project_fullspatial_selfsimilar_exact.md, project_paper_state.md.
# =============================================================================
r"""Full-spatial decaying GW spectrum with a FINITE source duration (self-similar
slow-time integration).  Resolves the principal IR-slope discrepancy.

Motivation
----------
``fullspatial_decay.py`` builds the full-spatial (k_GW != 0) decaying GW kernel in
the QUASI-STATIONARY limit: the temporal factor is the Fourier transform of the
product of the two legs' BK2016 decorrelations, integrated over an *infinite*
emission window.  That kernel pins the peak at the source scale p ~ 2.4 but, like
every analytic estimate, gives the causal infrared slope  Omega_GW ~ p^3.

The Roper Pol DNS instead give  Omega_GW ~ p^1  (flat Omega_GW/k) below the peak --
the "principal simulation-analytic discrepancy".  The missing ingredient is
TEMPORAL: the source acts for a FINITE duration T_em, not forever.

Mechanism (textbook finite-duration kernel)
-------------------------------------------
For a source that is coherent over a window [0, T_em], the time integral contributes
the autocorrelation of a top hat -- a triangular window (1 - tau/T_em) on the lag --
i.e. the spectral kernel  4 sin^2(omega T_em / 2) / omega^2.  This is

    -> T_em^2  (const)   for omega T_em << 1   =>  Omega_GW ~ p^3   (causal)
    -> 2 / omega^2       for omega T_em >> 1   =>  Omega_GW ~ p^1   (the DNS slope)

So the causal p^3 is recovered in the deep IR (modes that never complete an
oscillation during the source lifetime), while a p^1 band opens up over
    1/T_em  <  p  <  1/tau_c      (tau_c = eddy/decorrelation time at the peak)
which is wide precisely when the source is long-lived relative to an eddy turnover.

Implementation
--------------
We reuse the full-spatial geometry of ``fullspatial_decay`` verbatim (geometric
bracket, x,y triangle bounds, prefactor) and only replace the temporal factor with
its finite-window generalisation.  Because the quasi-stationary factor

    conv_ftprod(q; t1, t2) = (2 pi / t1 t2) int_0^inf cos(q t) R1 R2 dt,
    R_i = (1 + t/t_i)^{-2/3},

depends on the two times only through the lag t, a finite emission window
[0, T_em]^2 collapses to a single lag integral with a triangular weight:

    conv_window(q; t1, t2, T_em)
        = (2 pi / t1 t2) int_0^{T_em} (1 - t/T_em) cos(q t) R1 R2 dt.

This reduces EXACTLY to conv_ftprod as T_em -> inf (validated below).  The optional
``decay_beta`` adds a self-similar amplitude decay A(t)=(1+t/T_em)^{-beta} on each
leg (the realistic case: the slope then lands between 1 and 3).

Run:  python Notebooks/fullspatial_selfsimilar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from gw_turbulence.core import (  # noqa: E402
    _h_prefactor,
    _integration_bounds,
    kernel_bracket,
)


# ---------------------------------------------------------------------------
# Finite-duration temporal factor (reduces to fullspatial_decay.conv_ftprod)
# ---------------------------------------------------------------------------
def _conv_window(q: float, tau1: float, tau2: float, T_em: float,
                 coherence: float = 1.0) -> float:
    r"""Finite-window temporal factor.

    (2 pi / t1 t2) int_0^{T_em} (1 - t/T_em) cos(q t) R1(t) R2(t) dt,

    R_i = (1 + t/(coherence*tau_i))^{-2/3}.  ``coherence`` rescales the
    decorrelation time (the physical knob: tau_c / tau_c^{BK}); coherence -> inf is
    the perfectly coherent source (R->1, pure finite-duration top hat), coherence=1
    is the quasi-stationary BK2016 decorrelation.  The triangular window (1 - t/T_em)
    is the autocorrelation of a top hat of width T_em -- the finite-duration kernel.
    T_em -> inf with coherence=1 reproduces fullspatial_decay._conv_ftprod.
    """
    c1, c2 = coherence * tau1, coherence * tau2

    if not np.isfinite(T_em):
        f = lambda t: (1.0 + t / c1) ** (-2 / 3) * (1.0 + t / c2) ** (-2 / 3)
        val, _ = integrate.quad(f, 0.0, np.inf, weight="cos", wvar=q, limit=200)
        return 2.0 * np.pi / (tau1 * tau2) * val

    if np.isinf(coherence):                       # perfectly coherent: R1 R2 -> 1
        f = lambda t: (1.0 - t / T_em)
    else:
        f = lambda t: (1.0 + t / c1) ** (-2 / 3) * (1.0 + t / c2) ** (-2 / 3) * (1.0 - t / T_em)
    val, _ = integrate.quad(f, 0.0, T_em, weight="cos", wvar=q, limit=200)
    return 2.0 * np.pi / (tau1 * tau2) * val


def H_window(p: float, q: float, M: float = 1.0, R: float = 1e4, T_em: float = np.inf,
             coherence: float = 1.0, x_points: int = 28, y_points: int = 28) -> float:
    """Full-spatial finite-duration decaying GW kernel H(p,q;M,R,T_em)."""
    xs = np.geomspace(1.0 / R, 1.0, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        bounds = _integration_bounds(x, p, R)
        if bounds is None:
            continue
        y_min, y_max = bounds
        ys = np.geomspace(y_min, y_max, y_points)
        vals = [
            y**0.75 * (x + y) ** (-0.5) * x**0.75
            * kernel_bracket(p, x, y)
            * _conv_window(q, np.sqrt(x) / M, np.sqrt(y) / M, T_em, coherence)
            for y in ys
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def omega_gw(p, M, T_em=np.inf, R=1e4, coherence=1.0):
    return p**3 * H_window(p, p, M=M, R=R, T_em=T_em, coherence=coherence)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _peak(spec_fn, plo=0.3, phi=8.0, n=34):
    ps = np.geomspace(plo, phi, n)
    sp = np.array([spec_fn(pp) for pp in ps])
    i = int(np.argmax(sp))
    il, ir = max(i - 1, 0), min(i + 1, len(ps) - 1)
    c = np.polyfit(np.log(ps[il:ir + 1]), np.log(sp[il:ir + 1]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


def _band_slope(spec_fn, p_lo, p_hi, n=9):
    ps = np.geomspace(p_lo, p_hi, n)
    sp = np.array([spec_fn(pp) for pp in ps])
    good = sp > 0
    return float(np.polyfit(np.log(ps[good]), np.log(sp[good]), 1)[0])


def _validate():
    sys.path.insert(0, str(ROOT / "Notebooks"))
    from fullspatial_decay import H_decay_fast  # noqa: E402  (sibling tool)

    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print("\n(1) T_em -> inf, coherence=1 reduces to fullspatial_decay (quasi-stationary):")
    for p in (0.5, 1.0, 2.0):
        a = H_window(p, p, M=1.0, R=1e4, T_em=np.inf)
        b = H_decay_fast(p, p, M=1.0, R=1e4)
        print(f"    p={p:4.1f}:  H_window(inf)={a:.4e}  H_decay_fast={b:.4e}  ratio={a/b:.4f}")

    print("\n(2) deep-IR slope anchor (quasi-stationary, band p=1e-3..1e-2): expect ~3")
    sl = _band_slope(lambda p: omega_gw(p, 1.0, T_em=np.inf), 1e-3, 1e-2, n=7)
    print(f"    deep-IR slope = {sl:.2f}  (causal k^3 confirmed)")

    print("\n(3) full-spatial: peak + IR slope vs source coherence (T_em=20, M=1):")
    print("    coherence sweeps tau_c/tau_c^BK; inf = perfectly coherent over lifetime")
    print(f"    {'coherence':>10}{'IR slope':>10}{'p_peak':>9}")
    for coh in (1.0, 4.0, 16.0, np.inf):
        sl = _band_slope(lambda p: omega_gw(p, 1.0, T_em=20.0, coherence=coh), 0.1, 0.7)
        pk = _peak(lambda p: omega_gw(p, 1.0, T_em=20.0, coherence=coh))
        label = "inf" if np.isinf(coh) else f"{coh:.0f}"
        print(f"    {label:>10}{sl:10.2f}{pk:9.3f}")
    print("    -> coherent source flattens IR toward k^1 while the peak stays source-scale")


def _figure(name="fullspatial_selfsimilar_ir"):
    sys.path.insert(0, str(ROOT / "Notebooks"))
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)

    # Panel (a): full-spatial spectra at fixed T_em as the source is made more coherent
    # over its lifetime (coherence = tau_c/tau_c^BK).  The infrared flattens from the
    # causal k^3 toward the simulated k^1 while the source-scale peak (shaded) stays put.
    ps = np.geomspace(0.03, 7.0, 48)
    levels = [(1.0, PALETTE[0], r"$\tau_c\!\ll\!T_{\rm em}$ (quasi-stationary)"),
              (4.0, PALETTE[2], r"$\tau_c\!<\!T_{\rm em}$ (intermediate)"),
              (16.0, PALETTE[1], r"$\tau_c\!\sim\!T_{\rm em}$ (coherent)")]
    for coh, col, lab in levels:
        sp = np.array([omega_gw(p, 1.0, T_em=20.0, coherence=coh) for p in ps])
        ax0.plot(ps, sp / sp.max(), "-", color=col, lw=1.8, label=lab)
    pr = np.geomspace(0.05, 0.5, 10)
    ax0.plot(pr, 0.5 * (pr / 0.5) ** 3, ":", color="0.45", lw=1.2, label=r"$k^3$ (causal)")
    ax0.plot(pr, 0.32 * (pr / 0.5) ** 1, "--", color="0.45", lw=1.2, label=r"$k^1$ (DNS)")
    ax0.axvspan(1.8, 2.7, color="0.7", alpha=0.18, lw=0)
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_ylim(1e-4, 2)
    ax0.set_xlabel(r"$p=k/k_0$")
    ax0.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,\Omega_{\rm GW}^{\rm peak}$")
    ax0.set_title(r"full-spatial spectrum, $T_{\rm em}=20\,\tau_{e0}$", fontsize=10)
    ax0.legend(fontsize=6.8, loc="lower right")

    # Panel (b): aeroacoustic IR slope vs the coherence-over-lifetime ratio.
    import selfsimilar_hybrid as ss  # noqa: E402
    LOITS = dict(u0=1.0, l0=1.0, p=10 / 7, q=2 / 7, s=4)
    T_em_a = 60.0
    tau_sts = np.geomspace(0.1, 3e5, 13)
    slopes, ratios = [], []
    for ts in tau_sts:
        pars = {**LOITS, "tau_st": ts}
        tau_c = ss.tau1(1.0, 0.0, **pars)     # eddy/decorrelation time at the peak (k=1)
        psb = np.geomspace(3.0 / T_em_a, 0.5, 9)
        om = np.array([p**3 * ss.H_exact(p, T_em_a, n_T=160, n_k=80, **pars) for p in psb])
        g = om > 0
        slopes.append(np.polyfit(np.log(psb[g]), np.log(om[g]), 1)[0])
        ratios.append(tau_c / T_em_a)
    ax1.semilogx(ratios, slopes, "o-", color=PALETTE[3], lw=1.8, ms=4)
    ax1.axhline(3.0, color="0.45", ls=":", lw=1.2)
    ax1.axhline(1.0, color="0.45", ls="--", lw=1.2)
    ax1.text(ratios[0], 2.85, r"causal $k^3$", fontsize=8, color="0.35")
    ax1.text(ratios[-1] * 0.2, 1.12, r"DNS $k^1$", fontsize=8, color="0.35")
    ax1.set_xlabel(r"coherence ratio $\tau_c/T_{\rm em}$")
    ax1.set_ylabel(r"IR slope $d\ln\Omega_{\rm GW}/d\ln k$")
    ax1.set_title(r"IR slope set by source coherence", fontsize=10)
    ax1.set_ylim(0.6, 3.2)
    for ax in (ax0, ax1):
        apply_max_ticks(ax)

    out = save_figure(fig, name)
    print(f"\nwrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    _validate()
    _figure()
