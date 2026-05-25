#!/usr/bin/env python3
r"""EXPLORATORY (not a paper figure): wide-p view of the full-spatial self-similar
GW spectrum with denser sampling around the peak.

Extends the spectra panel of fullspatial_selfsimilar_exact to p in [0.04, 8] to show
the UV falloff above the source-scale peak, with extra p-density in the peak region
(1.2-3.5).  Uses the validated 60x60/n_T=90 grid; non-positive points are MASKED
(the fast-decay eps0=4 curve goes numerically negative in the deep UV where its
physical signal is tiny -- see the diagnostics, negative even at 80x80).

Writes images/fullspatial_selfsimilar_widep.pdf.  Does NOT touch the committed
paper figure (fullspatial_selfsimilar_exact.pdf) or its script.

Run: python Notebooks/explore_selfsimilar_widep.py   (~30-40 min at 60x60/n_T=90)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "Notebooks"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from fullspatial_selfsimilar_exact import omega_gw  # noqa: E402

XP = YP = 60
NT = 90

# p-grid: log-spaced, with extra density across the peak (1.2-3.5).
P_IR = np.geomspace(0.04, 1.2, 8)
P_PEAK = np.geomspace(1.2, 3.5, 13)
P_UV = np.geomspace(3.5, 8.0, 6)
PS = np.unique(np.concatenate([P_IR, P_PEAK, P_UV]))

CASES = [
    (1e12, "no decay ($\\varepsilon_0\\!\\to\\!0$)"),
    (1.0, "$\\varepsilon_0=1$"),
    (0.25, "$\\varepsilon_0=4$ (fast)"),
]


def _slope(ps, sp, p_lo, p_hi):
    m = (ps >= p_lo) & (ps <= p_hi) & (sp > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ps[m]), np.log(sp[m]), 1)[0])


def compute():
    print(f"wide-p self-similar spectra: {len(PS)} p-points x {len(CASES)} curves "
          f"@ {XP}x{YP}/n_T={NT}", flush=True)
    out = {}
    for tau_st, lab in CASES:
        sp = np.empty(len(PS))
        for i, p in enumerate(PS):
            sp[i] = omega_gw(p, 1.0, tau_st=tau_st, T_em=40.0,
                             x_points=XP, y_points=YP, n_T=NT)
            print(f"  tau_st={tau_st:>7.3g}  p={p:6.3f}  Om={sp[i]:+.3e}", flush=True)
        ir = _slope(PS, sp, 0.08, 0.5)
        uv = _slope(PS, sp, 3.5, 8.0)
        print(f"  -> {lab}: IR slope(0.08-0.5)={ir:.2f}  UV slope(3.5-8)={uv:.2f}  "
              f"npos={(sp > 0).sum()}/{len(sp)}", flush=True)
        out[tau_st] = sp
    return out


def figure(spectra, name="fullspatial_selfsimilar_widep"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    colors = [PALETTE[0], PALETTE[2], PALETTE[1]]

    for (tau_st, lab), col in zip(CASES, colors):
        sp = spectra[tau_st]
        m = sp > 0                       # mask non-positive (numerical) points
        norm = sp[m].max()
        ax.plot(PS[m], sp[m] / norm, "-", color=col, lw=1.8, label=lab)

    # IR guides (anchored in the inertial sub-peak band)
    pir = np.geomspace(0.06, 0.5, 8)
    ax.plot(pir, 0.55 * (pir / 0.5) ** 3, ":", color="0.45", lw=1.1, label=r"$k^3$ (causal)")
    ax.plot(pir, 0.55 * (pir / 0.5) ** 1, "--", color="0.45", lw=1.1, label=r"$k^1$ (DNS)")
    # UV guides (anchored at the peak, p~2.3)
    puv = np.geomspace(2.5, 8.0, 6)
    ax.plot(puv, 1.0 * (puv / 2.5) ** (-8.0 / 3.0), "-.", color="0.55", lw=1.1,
            label=r"$k^{-8/3}$")
    ax.plot(puv, 1.0 * (puv / 2.5) ** (-5.0), linestyle=(0, (1, 1)), color="0.7", lw=1.1,
            label=r"$k^{-5}$")

    ax.axvspan(1.8, 2.7, color="0.7", alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2)
    ax.set_xlim(0.035, 9)
    ax.set_xlabel(r"$p=k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,\Omega_{\rm GW}^{\rm peak}$")
    ax.set_title(r"full-spatial self-similar spectra (wide $p$)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"\nwrote {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    figure(compute())
