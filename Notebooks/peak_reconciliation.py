"""Reconciliation: stationary (sweeping) peak vs impulsive (source-scale) peak.

Resolves the apparent contradiction between Gogoberidze (2007) / this work --
stationary turbulence, GW peak at p ~ 1.48 M, well below the outer scale for
subsonic flow -- and the Roper Pol et al. (2020) simulations, whose GW spectrum
peaks near the source scale (p ~ 1-2), nearly independent of the flow speed.

The stationary/sweeping curve is COMPUTED here from
Omega_GW(p) = p^3 H_pq(p, p; M, R)  (core.H_pq, R=1e4): the peak is read directly
from each computed spectrum by a sub-grid (log-log parabola) argmax.

The "source-scale" band p in [1, 2] is NOT a computation and NOT data: it is a
schematic of the competing spatial-scale picture (the GW peak pinned near the
quadratic-stress scale ~2 k0, independent of M) that an impulsively-sourced,
decaying source -- e.g. the Roper Pol simulations -- follows. It is drawn as a
labelled band so it cannot be mistaken for either our analytic curve or real data.

The two pictures coincide in the transonic band (M ~ 1), where every current
simulation sits, and diverge only in the subsonic corner -- which no simulation
has tested.
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
MACH = np.logspace(np.log10(0.02), np.log10(6.0), 13)
XI = np.logspace(-0.6, 0.85, 70)            # p/M grid bracketing the peak

# competing spatial-scale (impulsive) picture: peak pinned near the stress scale
SOURCE_BAND = (1.0, 2.0)


def refined_peak(ps: np.ndarray, spec: np.ndarray) -> float:
    """Sub-grid peak p_peak by a log-log parabola through the argmax neighbours."""
    spec = np.asarray(spec)
    i = int(np.argmax(spec))
    if i == 0 or i == len(spec) - 1:
        return float(ps[i])
    lx = np.log(ps[i - 1:i + 2])
    ly = np.log(spec[i - 1:i + 2])
    denom = (lx[0] - lx[1]) * (lx[0] - lx[2]) * (lx[1] - lx[2])
    a = (lx[2] * (ly[1] - ly[0]) + lx[1] * (ly[0] - ly[2]) + lx[0] * (ly[2] - ly[1])) / denom
    b = (lx[2] ** 2 * (ly[0] - ly[1]) + lx[1] ** 2 * (ly[2] - ly[0])
         + lx[0] ** 2 * (ly[1] - ly[2])) / denom
    return float(np.exp(-b / (2.0 * a)))


def main(name: str = "peak_reconciliation"):
    apply_paper_style()

    peaks = np.array([
        refined_peak(M * XI, np.array([(M * xi) ** 3 * H_pq(M * xi, M * xi, M=M, R=R)
                                       for xi in XI]))
        for M in MACH
    ])
    for M, pk in zip(MACH, peaks):
        print(f"M={M:7.3f}  p_peak={pk:8.4f}  p_peak/M={pk / M:6.3f}")

    # linear-law fit over the subsonic--transonic range
    mask = MACH <= 1.0
    b, lna = np.polyfit(np.log(MACH[mask]), np.log(peaks[mask]), 1)
    a = np.exp(lna)
    # where the stationary curve enters the source-scale band
    m_lo = SOURCE_BAND[0] / a
    m_hi = SOURCE_BAND[1] / a
    print(f"\nfit p_peak = {a:.2f} M^{b:.2f}  (M<=1)")
    print(f"enters source-scale band [{SOURCE_BAND[0]},{SOURCE_BAND[1]}] at M = {m_lo:.2f}-{m_hi:.2f}")

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)

    # competing spatial-scale picture: horizontal band (schematic, not data)
    ax.axhspan(*SOURCE_BAND, color=PALETTE[2], alpha=0.18, lw=0)
    ax.text(0.022, 1.42, "source-scale peak\n(impulsive: Roper Pol, Caprini)",
            fontsize=8, color=PALETTE[5], va="center")

    # transonic band where the two pictures coincide and simulations sit
    ax.axvspan(m_lo, m_hi, color="0.5", alpha=0.12, lw=0)
    ax.text(np.sqrt(m_lo * m_hi), 6.5, "transonic\n(simulations)", fontsize=8,
            color="0.35", ha="center", va="top")

    # stationary / sweeping prediction: computed peaks + linear law
    mline = np.logspace(np.log10(MACH.min()), np.log10(MACH.max()), 60)
    ax.loglog(mline, a * mline ** b, color=PALETTE[6], lw=1.3, ls="--",
              label=rf"stationary law $p_{{\rm peak}}={a:.2f}\,M$")
    ax.loglog(MACH, peaks, "o", color=PALETTE[6], ms=5,
              label=r"stationary kernel (this work)")

    # annotate the divergence in the subsonic corner
    ax.annotate("predictions diverge\n(subsonic, untested)",
                xy=(0.1, 0.15), xytext=(0.12, 0.9),
                fontsize=8, color="0.2", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))

    ax.set_xlabel(r"$M$ (turbulent Mach number)")
    ax.set_ylabel(r"$p_{\rm peak}=k_{\rm peak}/k_0$")
    ax.set_xlim(0.02, 6.0)
    ax.set_ylim(0.02, 10.0)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.6)
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
