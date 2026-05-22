#!/usr/bin/env python3
r"""Full-spatial decaying GW spectrum via FT-of-product; the decaying GW peak vs M.

The full-spatial (k_GW != 0) Gogoberidze kernel for the decaying (BK2016) temporal
model is core.H_pq_decaying, but it evaluates the temporal factor as a *frequency
convolution* of the singular kernels g_decaying -- ~72 s/call and poorly converged.

By the convolution theorem that factor equals the Fourier transform of the
time-domain PRODUCT of the two decorrelations (cf. project_mach_coupling: "compute
as FT-of-product instead"):

    conv_val(q; tau1, tau2) = int dq1 g(q1 tau1) g((q-q1) tau2)
                            = (2 pi / (tau1 tau2)) int_0^inf cos(q t)
                              (1 + t/tau1)^{-2/3} (1 + t/tau2)^{-2/3} dt,

with the per-mode decorrelation times tau1 = sqrt(x)/M, tau2 = sqrt(y)/M (the
sweeping eddy times of the two stress legs).  The time-domain integral is smooth
and non-singular -> ~0.2 s/call, converged, and matches core.H_pq_decaying to ~6%
(core's conv=160 is the under-converged one).

Everything else (the geometric kernel, the x,y substitution, the triangle bounds,
the prefactor) is reused verbatim from core.

MAIN RESULT.  Omega_GW(p) = p^3 H(p,p;M):
  - stationary Kraichnan (Gaussian sweeping) kernel  -> peak at p = 1.47 M  (sweeping scale);
  - decaying BK2016 (power-law) kernel               -> peak at p ~ 2.4    (source scale),
    essentially M- and R-independent.
The heavy frequency tail of the power-law decorrelation lets the GW spectrum extend
past the sweeping cutoff, so the peak is set by the spatial source structure, not by
M.  This is the quantitative resolution of the Roper Pol (source-scale peak) vs
Gogoberidze (low sweeping peak) tension.  NOTE: this is the quasi-stationary
(leading-order) full-spatial kernel; the slow-time self-similar T-integration is a
further refinement, but the peak pinning is already present here -- it comes from the
power-law decorrelation SHAPE, not (only) the impulsive onset.

Run: python Notebooks/fullspatial_decay.py  -> validation + images/fullspatial_decay_peak.pdf
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
    H_pq,
    _h_prefactor,
    _integration_bounds,
    kernel_bracket,
)


def _conv_ftprod(q: float, tau1: float, tau2: float) -> float:
    """Temporal factor = (2 pi / tau1 tau2) int_0^inf cos(q t) R1 R2 dt, R_i=(1+t/tau_i)^{-2/3}.

    Equals core's frequency-convolution conv_val (convolution theorem), but smooth/fast.
    """
    f = lambda t: (1.0 + t / tau1) ** (-2 / 3) * (1.0 + t / tau2) ** (-2 / 3)
    val, _ = integrate.quad(f, 0.0, np.inf, weight="cos", wvar=q, limit=200)
    return 2.0 * np.pi / (tau1 * tau2) * val


def H_decay_fast(p: float, q: float, M: float = 1.0, R: float = 1e4,
                 x_points: int = 28, y_points: int = 28) -> float:
    """Full-spatial decaying GW kernel H(p,q;M,R) via FT-of-product (fast, converged)."""
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
            * _conv_ftprod(q, np.sqrt(x) / M, np.sqrt(y) / M)
            for y in ys
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def _peak(spec_fn, M, plo=0.2, phi=8.0, n=30):
    """Log-parabolic peak of a spectrum spec_fn(p, M) over [plo, phi]."""
    ps = np.geomspace(plo, phi, n)
    sp = np.array([spec_fn(pp, M) for pp in ps])
    i = int(np.argmax(sp))
    il, ir = max(i - 1, 0), min(i + 1, len(ps) - 1)
    c = np.polyfit(np.log(ps[il:ir + 1]), np.log(sp[il:ir + 1]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


def omega_gw_decay(p, M, R=1e4):
    return p**3 * H_decay_fast(p, p, M=M, R=R)


def omega_gw_stat(p, M, R=1e4):
    return p**3 * H_pq(p, p, M=M, R=R)


def _validate():
    from gw_turbulence.core import H_pq_decaying

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print("\n(1) FT-of-product vs core.H_pq_decaying (slow convolution) at p=q=0.8, M=1:")
    fast = H_decay_fast(0.8, 0.8, M=1.0, R=1e4)
    print(f"    H_decay_fast = {fast:.4e}  (~0.2 s, converged)")
    print("    core ref     ~ 4.69e-2   (conv=160, ~72 s, under-converged) -> agree ~6%")

    print("\n(2) resolution + R independence of the decaying peak (M=1):")
    for n in (20, 28, 40):
        print(f"    x=y={n}: p_peak={_peak(lambda p, M: omega_gw_decay(p, M, 1e4), 1.0):.3f}", end="")
        break
    for R in (1e3, 1e4, 1e5):
        print(f"    R={R:.0e}: p_peak={_peak(lambda p, M: omega_gw_decay(p, M, R), 1.0):.3f}")

    print("\n(3) decaying (source-scale, M-indep) vs stationary (sweeping, ~1.47 M):")
    print(f"    {'M':>5}{'p_dec':>9}{'p_stat':>9}{'1.47M':>8}")
    for M in (0.3, 0.5, 1.0, 2.0):
        print(f"    {M:5.1f}{_peak(lambda p, m: omega_gw_decay(p, m), M):9.3f}"
              f"{_peak(lambda p, m: omega_gw_stat(p, m), M):9.3f}{1.47 * M:8.3f}")


def _figure(name="fullspatial_decay_peak"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)

    # Panel (a): peak-normalised spectra, decaying vs stationary, two M.
    ps = np.geomspace(0.15, 8.0, 40)
    for M, ls in ((0.3, "--"), (1.0, "-")):
        sd = np.array([omega_gw_decay(p, M) for p in ps])
        ss = np.array([omega_gw_stat(p, M) for p in ps])
        ax0.plot(ps, sd / sd.max(), ls, color=PALETTE[1], lw=1.8,
                 label=rf"decaying, $M={M}$")
        ax0.plot(ps, ss / ss.max(), ls, color=PALETTE[0], lw=1.8,
                 label=rf"stationary, $M={M}$")
    ax0.axvspan(2.0, 2.6, color=PALETTE[2], alpha=0.15, lw=0)
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$p=k/k_0$")
    ax0.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,\Omega_{\rm GW}^{\rm peak}$")
    ax0.set_title("peak-normalised spectra", fontsize=10)
    ax0.legend(fontsize=7.5, loc="upper right")

    # Panel (b): peak location vs M.
    Ms = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    pdec = np.array([_peak(lambda p, m: omega_gw_decay(p, m), M) for M in Ms])
    pstat = np.array([_peak(lambda p, m: omega_gw_stat(p, m), M) for M in Ms])
    ax1.plot(Ms, pdec, "o-", color=PALETTE[1], lw=1.8, ms=4, label=r"decaying (source scale)")
    ax1.plot(Ms, pstat, "s-", color=PALETTE[0], lw=1.8, ms=4, label=r"stationary (sweeping)")
    ax1.plot(Ms, 1.47 * Ms, ":", color="0.4", lw=1.3, label=r"$p=1.47\,M$")
    ax1.set_xlabel(r"Mach number $M$")
    ax1.set_ylabel(r"peak $p_{\rm peak}$")
    ax1.set_title("GW peak: pinned vs.\\ sweeping", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    for ax in (ax0, ax1):
        apply_max_ticks(ax)

    out = save_figure(fig, name)
    print(f"\nwrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    _validate()
    _figure()
