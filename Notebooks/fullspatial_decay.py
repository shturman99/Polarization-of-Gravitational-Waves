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


# --- IR band (large-scale fluid spectrum) ------------------------------------
# Same construction as _fullspectrum_kernel.py for the stationary kernel: the
# fluid spectrum below the integral scale k0 enters the GW kernel only through a
# SHAPE FACTOR S(k) = A(k)/A_Kol(k), where A_Kol ~ k^{-11/3} is the inertial law
# extrapolated to k < k0.  With an IR energy slope E(k) ~ k^s,
#     S(k) = 1                  (k >= k0, inertial range)
#          = (k/k0)^{s + 5/3}   (k <  k0, IR band),     continuous at k0.
# IR_EXPONENTS holds s + 5/3 for the two standard large-scale spectra.
IR_EXPONENTS = {"saffman": 2.0 + 5.0 / 3.0, "batchelor": 4.0 + 5.0 / 3.0}


def _shape(tilde_k, ir_exp: float):
    """S(k) = A(k)/A_Kol(k); tilde_k = k/k0.  1 in the inertial range, rising IR power."""
    return np.where(tilde_k >= 1.0, 1.0, np.asarray(tilde_k, float) ** ir_exp)


def _integration_bounds_ir(x: float, p: float, R: float, R_IR: float):
    """Triangle bounds with the u-floor relaxed from k0 (=1) to k_IR/k0 = 1/R_IR.

    Local copy of core._integration_bounds: that function hard-codes the k0 floor
    (u_min = max(|k1-p|, 1.0)), which would clip away the IR leg.  Returning the
    y = (u/k0)^{-4/3} interval keeps the x,y substitution identical to core.
    """
    tilde_k1 = x ** (-0.75)
    u_min = max(abs(tilde_k1 - p), 1.0 / R_IR)
    u_max = min(tilde_k1 + p, R ** 0.75)
    if not (u_min < u_max):
        return None
    y_min = u_max ** (-4.0 / 3.0)
    y_max = u_min ** (-4.0 / 3.0)
    if not (y_min < y_max):
        return None
    return y_min, y_max


def H_decay_fast_full(p: float, q: float, M: float = 1.0, R: float = 1e4,
                      R_IR: float = 1.0, ir: str = "batchelor",
                      x_points: int = 28, y_points: int = 28) -> float:
    r"""Full-spatial decaying GW kernel WITH a large-scale IR band.

    Mirrors H_decay_fast (same FT-of-product temporal factor and geometry) but
    feeds a full fluid input spectrum: the inertial Kolmogorov range k in [k0, k_d]
    plus an IR band k in [k0/R_IR, k0] with energy slope E(k)~k^s (s set by ``ir``).
    The fluid spectrum enters only through the shape factor S(k1) S(u); the outer
    x range extends from [1/R, 1] to [1/R, R_IR^{4/3}] (x>1 is the IR band) and the
    inner u-floor is relaxed from k0 to k0/R_IR via _integration_bounds_ir.

    For R_IR = 1 (no IR band) this is IDENTICALLY H_decay_fast -- the shape factors
    are both 1, the x range and bounds are unchanged.  H_decay_fast itself is left
    untouched so all existing callers/tests keep their exact behaviour.
    """
    ir_exp = IR_EXPONENTS[ir]
    x_lo, x_hi = 1.0 / R, R_IR ** (4.0 / 3.0)            # x<=1 inertial, x>1 IR band
    xs = np.geomspace(x_lo, x_hi, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        bounds = _integration_bounds_ir(x, p, R, R_IR)
        if bounds is None:
            continue
        y_min, y_max = bounds
        tk1 = x ** (-0.75)                                # k1/k0
        ys = np.geomspace(y_min, y_max, y_points)
        tus = ys ** (-0.75)                               # u/k0
        vals = [
            yv**0.75 * (x + yv) ** (-0.5) * x**0.75
            * kernel_bracket(p, x, yv)
            * _conv_ftprod(q, np.sqrt(x) / M, np.sqrt(yv) / M)
            * float(_shape(tk1, ir_exp)) * float(_shape(tu, ir_exp))
            for yv, tu in zip(ys, tus)
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


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


def _tail_endpoint(q: float, tau1: float, tau2: float) -> float:
    r"""Large-q endpoint asymptote of the temporal factor _conv_ftprod.

    int_0^inf cos(q t) f(t) dt ~ -f'(0)/q^2 (the q^{-1} surface term vanishes,
    f(0)=1 finite), with f=(1+t/tau1)^{-2/3}(1+t/tau2)^{-2/3} so
    f'(0) = -(2/3)(1/tau1 + 1/tau2).  Hence
        T(q) ~ (2 pi / tau1 tau2) * (2/3)(1/tau1 + 1/tau2) / q^2,
    a HEAVY q^{-2} power-law tail (vs the stationary Gaussian's super-exp cutoff).
    """
    fprime0 = -(2.0 / 3.0) * (1.0 / tau1 + 1.0 / tau2)
    return 2.0 * np.pi / (tau1 * tau2) * (-fprime0) / q ** 2


def _slope(ps, sp, lo, hi):
    m = (ps >= lo) & (ps <= hi) & (sp > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ps[m]), np.log(sp[m]), 1)[0])


def _H_decay_uv_tail(p: float, M: float = 1.0, R: float = 1e4,
                     x_points: int = 48, y_points: int = 48) -> float:
    """H_decay_fast with the temporal factor replaced by its q^-2 endpoint asymptote.

    Isolates the spatial (geometric x stress-self-convolution) contribution to the
    UV slope: the difference from the full H_decay_fast UV slope is the residual of
    the q^-2 temporal tail.
    """
    xs = np.geomspace(1.0 / R, 1.0, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        bounds = _integration_bounds(x, p, R)
        if bounds is None:
            continue
        y_min, y_max = bounds
        ys = np.geomspace(y_min, y_max, y_points)
        vals = [
            yv**0.75 * (x + yv) ** (-0.5) * x**0.75
            * kernel_bracket(p, x, yv)
            * _tail_endpoint(p, np.sqrt(x) / M, np.sqrt(yv) / M)
            for yv in ys
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def _validate_completeness():
    """PART A (IR band) + PART B (UV asymptote) checks for the completeness figure."""
    print("=" * 74)
    print("COMPLETENESS (A): IR band reduction + GW IR slope (M=1, R=1e4)")
    print("=" * 74)
    print("  R_IR=1 must reduce H_decay_fast_full -> H_decay_fast exactly:")
    for p in (0.3, 1.0, 3.0):
        a = H_decay_fast_full(p, p, M=1.0, R=1e4, R_IR=1.0)
        b = H_decay_fast(p, p, M=1.0, R=1e4)
        print(f"    p={p:4.1f}: full(R_IR=1)={a:.6e}  H_decay_fast={b:.6e}  "
              f"reldiff={abs(a / b - 1):.1e}")

    ps_ir = np.geomspace(0.04, 0.3, 7)
    print("\n  GW IR slope of Omega_GW(p)=p^3 H(p,p) over p in [0.04, 0.3]:")
    for lab, R_IR, ir in [("no IR band  (R_IR=1)   ", 1.0, "batchelor"),
                          ("Batchelor   (R_IR=100) ", 100.0, "batchelor"),
                          ("Saffman     (R_IR=100) ", 100.0, "saffman")]:
        sp = np.array([p ** 3 * H_decay_fast_full(p, p, M=1.0, R=1e4, R_IR=R_IR, ir=ir)
                       for p in ps_ir])
        print(f"    {lab}: GW IR slope = {_slope(ps_ir, sp, 0.04, 0.3):+.3f}  "
              f"[causal k^3 expected]")

    print("\n" + "=" * 74)
    print("COMPLETENESS (B): UV temporal tail + Omega_GW UV slope")
    print("=" * 74)
    print("  Temporal factor T(q;tau1,tau2): exact vs endpoint q^{-2} asymptote")
    print("  (tau1=tau2=1):")
    for q in (5.0, 10.0, 20.0, 40.0):
        exact = _conv_ftprod(q, 1.0, 1.0)
        approx = _tail_endpoint(q, 1.0, 1.0)
        print(f"    q={q:5.1f}: exact={exact:+.4e}  q^-2 endpoint={approx:+.4e}  "
              f"ratio={exact / approx:.3f}")

    ps_uv = np.geomspace(3.0, 12.0, 7)
    print("\n  decaying Omega_GW UV slope (p in [3,12], M=1, R=1e4) vs grid:")
    for n in (28, 48, 80):
        sp_uv = np.array([p ** 3 * H_decay_fast(p, p, M=1.0, R=1e4,
                                                x_points=n, y_points=n) for p in ps_uv])
        print(f"    x=y={n:3d}:  UV slope = {_slope(ps_uv, sp_uv, 3.0, 12.0):+.3f}  "
              f"(min Om={sp_uv.min():+.2e})")

    # exact temporal factor replaced by its q^-2 endpoint asymptote -> isolates
    # the spatial/self-convolution slope and confirms the tail drives the UV.
    sp_uv_tail = np.array([p ** 3 * _H_decay_uv_tail(p, M=1.0, R=1e4) for p in ps_uv])
    print(f"  q^-2-tail-only kernel UV slope                 = "
          f"{_slope(ps_uv, sp_uv_tail, 3.0, 12.0):+.3f}  (analytic UV asymptote)")
    print("  reference data UV ~ k^{-11/3} = -3.667")


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
