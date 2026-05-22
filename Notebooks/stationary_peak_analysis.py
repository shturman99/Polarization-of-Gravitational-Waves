"""Why does the stationary GW spectrum peak at p ~ M and not at p ~ 2?

Three analytically-traceable demonstrations, all built from the *actual* kernel
Omega_GW(p) = p^3 H_pq(p, q=p; M, R) (core.H_pq), that the spectral peak is set
by the sweeping decorrelation at p ~ M, NOT by the source spatial scale (p ~ 2).

The decomposition being tested is

    Omega_GW(p; M)  =  A * M^3 * p^3 * G(p/M),

i.e. a universal causal rise (p^3, amplitude ~M^3) times a cutoff G that is a
function of the single combination xi = p/M.  The only place M enters H_pq is the
sweeping Gaussian  exp(-2 xy/(x+y) * q^2/M^2)  in core.integrand_y, so the cutoff
-- and hence the peak -- must scale with M.

(a) MECHANISM.  Each spectrum (solid) follows its causal IR asymptote A M^3 p^3
    (dashed) and rolls over where the sweeping cutoff switches on; the raw
    measured peak (dot) sits at the crossover, p_peak ~ 1.4 M.

(b) UNIVERSAL COLLAPSE.  Omega_GW/(M^3 p^3) plotted versus xi = p/M collapses
    every Mach number onto one curve A*G(xi).  This is the proof: the cutoff is a
    pure function of p/M.  The peak of xi^3 G(xi) sits at a universal xi_* (read
    off the collapsed data, no functional-form assumption), so p_peak = xi_* M.

(c) RAW PEAK SCALING.  The peak measured directly from each computed spectrum
    (parabolic refinement of the argmax) versus M, with a power-law fit
    p_peak = a M^b.  Subsonic-to-transonic gives b ~ 1; it saturates near the
    outer scale at large M.

All spectra use a dense p-grid for smoothness; the peak numbers are raw (read
from the computed spectra), not analytic estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.core import H_pq  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

R = 1.0e4

# Mach numbers shown as spectra (panels a, b)
MACH_SPEC = (0.03, 0.1, 0.3, 1.0, 3.0)
# dimensionless xi = p/M grid; identical for every M so the collapse is exact
XI = np.logspace(-1.5, 1.05, 170)

# finer Mach sweep used only to measure the peak location (panel c)
MACH_SCAN = np.logspace(np.log10(0.02), np.log10(5.0), 15)
XI_SCAN = np.logspace(-0.6, 0.85, 85)


def omega_gw(ps: np.ndarray, M: float) -> np.ndarray:
    """Omega_GW(p) = p^3 H_pq(p, p; M, R) on the sound-cone diagonal."""
    return np.array([p ** 3 * H_pq(p, p, M=M, R=R) for p in ps])


def refined_peak(ps: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    """Sub-grid peak (p_peak, Omega_peak) by a log-log parabola through the
    argmax and its two neighbours -- a raw read of the computed spectrum, not a
    model fit."""
    spec = np.asarray(spec)
    i = int(np.argmax(spec))
    if i == 0 or i == len(spec) - 1:
        return float(ps[i]), float(spec[i])
    lx = np.log(ps[i - 1:i + 2])
    ly = np.log(spec[i - 1:i + 2])
    denom = (lx[0] - lx[1]) * (lx[0] - lx[2]) * (lx[1] - lx[2])
    a = (lx[2] * (ly[1] - ly[0]) + lx[1] * (ly[0] - ly[2]) + lx[0] * (ly[2] - ly[1])) / denom
    b = (lx[2] ** 2 * (ly[0] - ly[1]) + lx[1] ** 2 * (ly[2] - ly[0])
         + lx[0] ** 2 * (ly[1] - ly[2])) / denom
    lx_v = -b / (2.0 * a)
    return float(np.exp(lx_v)), float(spec[i])


def main(name: str = "stationary_peak_analysis"):
    apply_paper_style()

    # ---- spectra on shared xi-grid (panels a, b) --------------------------
    spectra = {}      # M -> (ps, omega)
    collapse = {}     # M -> (xi, omega/(M^3 p^3))
    ir_const = {}     # M -> A = omega/(M^3 p^3) deep in the IR
    for M in MACH_SPEC:
        ps = M * XI
        om = omega_gw(ps, M)
        spectra[M] = (ps, om)
        reduced = om / (M ** 3 * ps ** 3)          # = A * G(xi)
        collapse[M] = (XI, reduced)
        ir_const[M] = float(np.median(reduced[:5]))  # A = G(0) plateau

    A = float(np.mean(list(ir_const.values())))
    A_spread = max(ir_const.values()) / min(ir_const.values())

    # universal xi_* : argmax of xi^3 * (A G) read off the collapsed curve
    xi_ref, red_ref = collapse[0.1]               # cleanest deeply-subsonic case
    shape = XI ** 3 * red_ref
    xi_star, _ = refined_peak(xi_ref, shape)

    # ---- raw peak scaling (panel c) ---------------------------------------
    scan_M, scan_pk = [], []
    print(f"{'M':>8} {'p_peak(raw)':>12} {'p_peak/M':>10}")
    for M in MACH_SCAN:
        ps = M * XI_SCAN
        om = omega_gw(ps, M)
        pk, _ = refined_peak(ps, om)
        scan_M.append(M)
        scan_pk.append(pk)
        print(f"{M:8.3f} {pk:12.4f} {pk / M:10.3f}")
    scan_M = np.array(scan_M)
    scan_pk = np.array(scan_pk)

    # power-law fit over the clean (subsonic--transonic) regime M <= 1
    fit_mask = scan_M <= 1.0
    b, lna = np.polyfit(np.log(scan_M[fit_mask]), np.log(scan_pk[fit_mask]), 1)
    a = np.exp(lna)
    print(f"\nIR collapse plateau  A = {A:.4e}  (spread max/min = {A_spread:.4f})")
    print(f"universal peak       xi_* = {xi_star:.3f}")
    print(f"peak-scaling fit     p_peak = {a:.3f} * M^{b:.3f}  (M <= 1)")

    # ---- figure -----------------------------------------------------------
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(11.0, 3.5), constrained_layout=True)

    # (a) mechanism: rise x cutoff = peak
    for c, M in enumerate(MACH_SPEC):
        col = PALETTE[(c + 1) % len(PALETTE)]
        ps, om = spectra[M]
        axA.loglog(ps, om, color=col, lw=1.6, label=rf"$M={M:g}$")
        axA.loglog(ps, A * M ** 3 * ps ** 3, color=col, ls=":", lw=0.9)  # IR asymptote
        pk, ompk = refined_peak(ps, om)
        axA.loglog(pk, ompk, "o", color=col, ms=4)
    axA.set_xlabel(r"$p=k/k_0$")
    axA.set_ylabel(r"$\Omega_{\rm GW}(p)=p^3H(p,p)$")
    axA.set_title(r"(a) causal rise $\times$ sweeping cutoff")
    axA.set_ylim(1e-13, 1e2)
    axA.legend(loc="upper left", fontsize=8, handlelength=1.3)
    apply_max_ticks(axA)

    # (b) universal collapse vs xi = p/M
    for c, M in enumerate(MACH_SPEC):
        col = PALETTE[(c + 1) % len(PALETTE)]
        xi, red = collapse[M]
        axB.loglog(xi, red, color=col, lw=1.4, label=rf"$M={M:g}$")
    axB.axvline(xi_star, color="0.4", ls="--", lw=1.0)
    axB.text(xi_star * 1.1, axB.get_ylim()[0] * 1e3,
             rf"$\xi_\ast={xi_star:.2f}$", fontsize=9, color="0.3")
    axB.set_xlabel(r"$\xi=p/M$")
    axB.set_ylabel(r"$\Omega_{\rm GW}/(M^3p^3)=A\,G(\xi)$")
    axB.set_title(r"(b) cutoff depends only on $p/M$")
    apply_max_ticks(axB)

    # (c) raw peak scaling
    axC.loglog(scan_M, scan_pk, "o", color=PALETTE[6], ms=5,
               label=r"raw $p_{\rm peak}$ (computed)")
    mline = np.logspace(np.log10(scan_M.min()), 0.0, 50)
    axC.loglog(mline, a * mline ** b, color=PALETTE[5], lw=1.4,
               label=rf"fit $p_{{\rm peak}}={a:.2f}\,M^{{{b:.2f}}}$")
    axC.loglog(mline, xi_star * mline, color="0.4", ls="--", lw=1.0,
               label=rf"$\xi_\ast M$ ($\xi_\ast={xi_star:.2f}$)")
    axC.axhline(2.0, color="0.7", ls=":", lw=1.0)
    axC.text(scan_M.min() * 1.1, 2.2, r"$p=2$ (source scale)", fontsize=8, color="0.5")
    axC.set_xlabel(r"$M$")
    axC.set_ylabel(r"$p_{\rm peak}$")
    axC.set_title(r"(c) raw peak $\propto M$")
    axC.legend(loc="lower right", fontsize=8, handlelength=1.4)
    apply_max_ticks(axC)

    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
