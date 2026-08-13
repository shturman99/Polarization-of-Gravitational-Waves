#!/usr/bin/env python3
r"""GW spectra from a source with a HARD finite coherence time.

Two candidate decorrelations, both of which "just drop to zero after some time":

    top-hat    f(tau) = Theta(tau_c - |tau|)
    triangle   f(tau) = (1 - |tau|/tau_c) Theta(tau_c - |tau|)

The top-hat is NOT a legitimate correlation function.  Its transform is a sinc,
which changes sign, so it violates Bochner's theorem (a valid autocorrelation
must have a non-negative transform) and produces NEGATIVE GW energy over roughly
half the spectrum, plus exact nulls at omega tau_c = n pi.  The triangle is the
autocorrelation of a source switched on for a window tau_c -- the box convolved
with itself -- and is positive by construction.

The triangle carries a cusp at zero lag, f'(0+) = -1/tau_c, so by the cusp
theorem of derivation.tex Eq.(eq:cusp-tail) its transform falls as omega^-2:
the same heavy tail as the BK2016 decaying model, and hence the same
source-scale peak pinning, without needing the BK2016 power law at all.

Following Eq.(eq:ft-of-product), the two-leg temporal factor is the transform of
the time-domain PRODUCT of the two legs' correlations,

    T(q; tau_1, tau_2) = (1/(tau_1 tau_2)) * 2 int_0^inf dtau cos(q tau) f_1 f_2,

with tau_1 = sqrt(x)/M, tau_2 = sqrt(y)/M the same per-leg times the decaying
kernel uses.  The control parameter is that = tau_c k_0 = sqrt(2 pi)/M, matching
kernel_comparison_grid.py.

Produces images/finite_coherence_gw.pdf.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gw_turbulence.core import kernel_bracket  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)
import band_split_gw as B  # noqa: E402

warnings.filterwarnings("ignore")

SQRT2PI = np.sqrt(2.0 * np.pi)
R_FID, R_IR_FID = 1e4, 1e2
THATS = (0.01, 0.1, 1.0, 10.0, 100.0)
P_GRID = np.geomspace(1e-3, 30.0, 44)
_TCOLOR = dict(zip(THATS, (PALETTE[5], PALETTE[2], PALETTE[0], PALETTE[1], PALETTE[6])))


# ------------------------------------------------------------ temporal factor --
def _moments(w, a):
    """int_0^a cos(w t) t^n dt for n = 0,1,2, safe as w -> 0."""
    w = np.asarray(w, float)
    a = np.asarray(a, float)
    small = np.abs(w * a) < 1e-4
    ws = np.where(small, 1.0, w)                      # dummy to avoid 0/0
    sa, ca = np.sin(ws * a), np.cos(ws * a)
    m0 = np.where(small, a, sa / ws)
    m1 = np.where(small, a**2 / 2.0, (ca - 1.0) / ws**2 + a * sa / ws)
    m2 = np.where(small, a**3 / 3.0,
                  a**2 * sa / ws + 2.0 * a * ca / ws**2 - 2.0 * sa / ws**3)
    return m0, m1, m2


def temporal_factor(q, tau1, tau2, kind: str):
    """T(q; tau1, tau2) for the two hard-cutoff models (arbitrary common norm)."""
    a = np.minimum(tau1, tau2)                        # product vanishes beyond this
    m0, m1, m2 = _moments(q, a)
    if kind == "tophat":                              # Theta * Theta = Theta
        integral = m0
    elif kind == "triangle":                          # (1-t/t1)(1-t/t2)
        integral = m0 - (1.0 / tau1 + 1.0 / tau2) * m1 + m2 / (tau1 * tau2)
    else:
        raise ValueError(kind)
    return 2.0 * integral / (tau1 * tau2)


# ------------------------------------------------------------------- kernel ---
def H_finite(p: float, q: float, that: float, kind: str, band: str = "both",
             R: float = R_FID, R_IR: float = R_IR_FID, ir: str = "batchelor",
             x_points: int = 260, y_points: int = 200) -> float:
    """Band kernel with the hard-cutoff temporal factor in place of sweeping."""
    M = SQRT2PI / that
    ir_exp = B.IR_EXPONENTS[ir]
    p = max(float(p), 1e-10)
    uf, uc = 1.0 / R_IR, R ** 0.75
    ks = B._outer_k_grid(uf, uc, x_points, p, uf, uc)
    xs = (ks ** (-4.0 / 3.0))[::-1]
    s1 = B.shape_band(xs ** (-0.75), ir_exp, band)
    acc = np.zeros_like(xs)
    for i, x in enumerate(xs):
        if s1[i] == 0.0:
            continue
        tk1 = x ** (-0.75)
        umin, umax = max(abs(tk1 - p), uf), min(tk1 + p, uc)
        if not umin < umax:
            continue
        ys = B._split_grid(umax ** (-4.0 / 3.0), umin ** (-4.0 / 3.0), y_points)
        s2 = B.shape_band(ys ** (-0.75), ir_exp, band)
        geom = ys ** 0.75 * x ** 0.75 * kernel_bracket(p, x, ys)
        tfac = temporal_factor(q, np.sqrt(x) / M, np.sqrt(ys) / M, kind)
        acc[i] = np.trapezoid(geom * tfac * s1[i] * s2, ys)
    return float(np.trapezoid(acc, xs)) / p


def omega_gw(ps, that, kind, **kw):
    return np.array([p ** 3 * H_finite(p, p, that, kind, **kw) for p in ps])


# ------------------------------------------------------------------- figure ---
def main() -> None:
    apply_paper_style(grid=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), constrained_layout=True)

    # (a) the two temporal factors, single scale
    ax = axes[0, 0]
    w = np.linspace(1e-3, 30.0, 3000)
    for kind, col, ls in (("tophat", PALETTE[6], "-"), ("triangle", PALETTE[5], "-")):
        T = temporal_factor(w, 1.0, 1.0, kind)
        ax.plot(w, T / T[0], ls, color=col, lw=1.3, label=kind)
    ax.axhline(0.0, color="0.6", lw=0.8)
    for n in range(1, 10):
        ax.axvline(n * np.pi, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"$\omega\tau_c$")
    ax.set_ylabel(r"$T(\omega)/T(0)$")
    ax.set_title(r"(a) temporal factor; grid $=n\pi$", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # (b),(c) the sourced spectra
    peaks, allvals = {}, []
    for ax, kind, tag in ((axes[0, 1], "tophat", "b"), (axes[1, 0], "triangle", "c")):
        peaks[kind] = []
        for that in THATS:
            y = omega_gw(P_GRID, that, kind)
            allvals.append(np.abs(y))
            pos, neg = y.copy(), y.copy()
            pos[y <= 0] = np.nan
            neg[y >= 0] = np.nan
            ax.loglog(P_GRID, pos, "-", color=_TCOLOR[that], lw=1.2,
                      label=rf"$\hat\tau={that:g}$")
            ax.loglog(P_GRID, -neg, ":", color=_TCOLOR[that], lw=1.2)
            frac = np.mean(y < 0)
            peaks[kind].append(P_GRID[np.argmax(np.nan_to_num(pos))])
            print(f"  {kind:9s} that={that:<6g} peak p={peaks[kind][-1]:6.3f}"
                  f"   negative on {frac:5.1%} of the grid", flush=True)
        ax.set_xlabel(r"$p=k/k_0$")
        ax.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^3H(p,p)$")
        ttl = "top-hat (dotted $=|\\Omega|$ where $\\Omega<0$)" if kind == "tophat" \
              else "triangle (positive everywhere)"
        ax.set_title(rf"({tag}) {ttl}", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        apply_max_ticks(ax, n=6)

    _a = np.concatenate(allvals)
    _hi = np.nanmax(_a[_a > 0])
    _ylim = (10.0 ** np.floor(np.log10(_hi) - 14.0), _hi * 5.0)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_ylim(*_ylim)

    # (d) peak position vs coherence time
    ax = axes[1, 1]
    ax.loglog(THATS, peaks["triangle"], "o-", ms=4, color=PALETTE[5],
              label="triangle")
    ax.loglog(THATS, peaks["tophat"], "s--", ms=4, color=PALETTE[6],
              label="top-hat (unphysical)")
    ax.axhline(2.4, color="0.45", ls=":", lw=1.0)
    ax.text(0.02, 2.7, r"decaying kernel, $p\simeq2.4$", fontsize=7.5, color="0.35")
    ax.loglog(THATS, 1.47 * SQRT2PI / np.array(THATS), ls="-.", color="0.6", lw=0.9)
    ax.text(0.02, 1.47 * SQRT2PI / 0.02 * 0.25, r"sweeping, $1.47M$",
            fontsize=7.5, color="0.4")
    ax.set_xlabel(r"$\hat\tau=\tau_ck_0$")
    ax.set_ylabel(r"$p_{\rm peak}$")
    ax.set_title(r"(d) peak position vs coherence time", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    apply_max_ticks(ax, n=6)

    out = save_figure(fig, "finite_coherence_gw")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
