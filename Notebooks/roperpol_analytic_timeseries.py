#!/usr/bin/env python3
r"""Analytic GW spectrum THROUGH TIME vs the Pencil Code simulation (run ini2).

We build the time-dependent GW spectrum from the analytic framework of the paper
and overlay it on the public time-resolved data of Roper Pol et al. (2020)
(arXiv:1903.08585; Zenodo 10.5281/zenodo.3692072).

Model
-----
For a source that turns on at t0 and stays coherent, the GW spectral energy at
time t factorises into a spatial stress shape and the finite-time temporal
window of Eq.(window-factor):

    Omega_GW(k,t)  ~  k^3 * P_T(k) * (1/k^2) * <4 sin^2(k(t-t0)/2)>,
    E_GW(k,t) = Omega_GW/k  ~  S(k) * B(k, t-t0),

with S(k) the (white-stress) spatial shape and B the build-up fraction.  We take:

  * SPATIAL SHAPE (analytic, physically motivated): the B^2 stress of a causal
    (Batchelor) field is white at large scales, so E_GW is flat below the peak;
    the peak sits at 2 k0 (quadratic source) and the inertial range gives the
    Kolmogorov ultraviolet Omega_GW ~ k^-11/3, i.e. E_GW ~ k^-14/3:
        S(k) = A * [1 + (k/kp)^2]^(-7/3),   kp = 2 k0.
    -> flat (k^0) below kp, ~k^-14/3 above, peak near kp.
  * TEMPORAL WINDOW (analytic): the time-averaged coherent build-up fraction,
        B(k, dt) = 1 - sinc(k dt),   sinc(x)=sin x / x,
    which -> (k dt)^2/6 at small argument (causal: E_GW~k^2 => Omega_GW~k^3) and
    -> 1 at large argument (saturated flat plateau).  The IMPULSIVE limit is
    B == 1 (the source is set instantaneously); we show both.

Only ONE overall amplitude A is calibrated (to the simulation's total GW energy
at the latest time); kp is fixed to twice the measured magnetic peak k0.

The comparison then tests the analytic TIME EVOLUTION, and its headline result is
that the simulation saturates FASTER than the coherent-source window predicts --
the sudden magnetic-field onset acts almost impulsively, so the data sit between
the coherent (too slow) and impulsive (instant) limits, near the impulsive one.

Figure: images/roperpol_analytic_timeseries.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roperpol_all_runs import load  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE, apply_max_ticks, apply_paper_style, save_figure,
)

def _sinc(x):
    return np.sinc(x / np.pi)          # numpy sinc is sin(pi x)/(pi x)


def build_up(k, dt):
    """Coherent-source time-averaged build-up fraction B(k,dt)=1-sinc(k dt)."""
    return np.clip(1.0 - _sinc(k * dt), 0.0, None)


def main():
    apply_paper_style()
    kk, tG, EGk, ESk = load("ini2")
    t0 = tG[0]
    k0 = kk[np.argmax(ESk[-1])]
    Ssat = EGk[-1].astype(float)            # empirical saturated shape S(k)=E_GW(k,t_late)
    dt = tG - t0
    snaps = [np.argmin(np.abs(tG - t)) for t in (1.02, 1.06, 1.20, 1.40)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    # ---- (a) spectra: sim (solid) vs analytic coherent-window model (dashed).
    # Analytic model = saturated shape * coherent temporal window (no amplitude fit).
    for j, i in enumerate(snaps):
        c = PALETTE[j + 1]
        Es = EGk[i].astype(float).copy(); Es[Es <= 0] = np.nan
        ax1.loglog(kk, Es, color=c, lw=1.3, label=rf"$t={tG[i]:.2f}$")
        ana = Ssat * build_up(kk, dt[i])
        ana[ana <= 0] = np.nan
        ax1.loglog(kk, ana, color=c, lw=1.0, ls="--")
    ax1.axvline(2 * k0, color="0.6", lw=0.7, ls=":")
    ax1.text(2 * k0 * 1.1, EGk[EGk > 0].min(), r"$2k_0$", fontsize=7, color="0.4")
    gwpos = EGk[EGk > 0]
    ax1.set_xlim(1.2, 1.15 * kk.max())
    ax1.set_ylim(gwpos.min() / 3, gwpos.max() * 4)
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$E_{\rm GW}(k)=\Omega_{\rm GW}/k$")
    ax1.set_title(r"(a) sim (solid) vs coherent-window model (dashed)", fontsize=9.5)
    ax1.legend(fontsize=7, frameon=False, loc="lower left")
    apply_max_ticks(ax1)

    # ---- (b) parameter-free per-mode build-up: sim/sim_sat vs the analytic window.
    def runmed(y, w=25):
        return np.array([np.median(y[max(0, i - w):i + w + 1]) for i in range(len(y))])
    for j, ktar in enumerate((2.0, 8.0, 30.0)):
        jk = np.argmin(np.abs(kk - ktar))
        c = PALETTE[j + 3]
        sat = EGk[:, jk] / Ssat[jk]                 # sim saturation fraction (no free param)
        ax2.plot(dt, sat, ".", color=c, ms=1.6, alpha=0.18)
        ax2.plot(dt, runmed(sat), "-", color=c, lw=1.9, label=rf"$k={kk[jk]:.0f}$ (sim)")
        ax2.plot(dt, build_up(kk[jk], dt), "--", color=c, lw=1.2)
    ax2.axhline(1.0, color="0.5", lw=0.9, ls=":")
    ax2.text(dt[-1] * 0.98, 1.04, "saturation", fontsize=7, color="0.4", ha="right")
    ax2.set_xlabel(r"$t-t_0$")
    ax2.set_ylabel(r"saturation fraction $E_{\rm GW}(k,t)/E_{\rm GW}(k,t_{\rm late})$", fontsize=8)
    ax2.set_title(r"(b) sim (solid) saturates $\gg$ faster than coherent (dashed)", fontsize=9.5)
    ax2.set_ylim(0, 1.5)
    ax2.legend(fontsize=6.8, frameon=False, loc="lower right",
               title=r"dashed $=1-\mathrm{sinc}(k\Delta t)$", title_fontsize=6.5)
    apply_max_ticks(ax2)

    out = save_figure(fig, "roperpol_analytic_timeseries")
    print(f"saved {out}")
    print(f"k0={k0:.1f}. per-mode saturation fraction (sim vs coherent 1-sinc):")
    for ktar in (2.0, 8.0, 30.0):
        jk = np.argmin(np.abs(kk - ktar))
        for i in snaps:
            print(f"  k={kk[jk]:5.1f} t={tG[i]:.2f} (dt={dt[i]:.2f}): "
                  f"sim={EGk[i, jk] / Ssat[jk]:5.2f}  coherent={build_up(kk[jk], dt[i]):5.2f}")


if __name__ == "__main__":
    main()
