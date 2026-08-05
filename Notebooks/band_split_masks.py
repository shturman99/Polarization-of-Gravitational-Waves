#!/usr/bin/env python3
r"""Appendix figure for the band-split chain: the masks, and the invariance of the
white-noise floor.

Two panels, keyed directly to the equation chain of App.~\ref{sec:chain-bandsplit}:

  (a) the shape factor and its two band masks, Eqs. (A:bs-shape)-(A:bs-masks),
          S(k)     = (k/k0)^{s+5/3} Theta(k0-k) + Theta(k-k0)
          S_IR(k)  = S(k) Theta(k0-k)
          S_in(k)  = S(k) Theta(k-k0)

  (b) the content of Eq. (A:bs-floor): P_T(0) = int d^3q P_M(q)^2 depends on
      neither the infrared slope s nor the infrared bandwidth R_IR.  Plotted as
      Omega_GW(p) for the IR band ALONE at four bandwidths and two infrared
      slopes; in the infrared they collapse.

Produces images/band_split_masks.pdf.  Companion to band_split_gw.py, which
produces the main-text figure; this one shows what that figure cannot -- the
masks themselves and the robustness sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)
import band_split_gw as bs  # noqa: E402

M_FID, R_FID = 0.5, 1e4
R_IR_SWEEP = (10.0, 30.0, 100.0, 300.0)
P_GRID = np.geomspace(1e-3, 1.0, 26)
IR_PROBE = 5e-3          # where the infrared amplitude is quoted


def shape(tk: np.ndarray, s: float) -> np.ndarray:
    """S(k) = (k/k0)^{s+5/3} for k<k0, 1 for k>=k0."""
    return np.where(tk >= 1.0, 1.0, tk ** (s + 5.0 / 3.0))


def panel_masks(ax) -> None:
    tk = np.geomspace(1e-2, 1e3, 600)
    S = shape(tk, 4.0)
    ax.loglog(tk, S, color="0.55", lw=2.6, alpha=0.8, label=r"$\mathcal{S}$")
    ax.loglog(tk[tk < 1], S[tk < 1], color=PALETTE[5], lw=1.4, ls="--",
              label=r"$\mathcal{S}_{\rm IR}$")
    ax.loglog(tk[tk >= 1], S[tk >= 1], color=PALETTE[6], lw=1.4, ls="-.",
              label=r"$\mathcal{S}_{\rm in}$")
    ax.axvline(1.0, color="0.7", lw=0.7, ls=":")
    ax.set_xlabel(r"$k/k_0$")
    ax.set_ylabel(r"$\mathcal{S}(k)=A(k)/A_{\rm Kol}(k)$")
    ax.set_title(r"(a) shape factor and masks", fontsize=10)
    ax.set_ylim(1e-12, 5.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    apply_max_ticks(ax)


def panel_floor(ax) -> dict:
    out = {}
    for i, rir in enumerate(R_IR_SWEEP):
        y = np.array([bs.omega_gw(p, M=M_FID, R=R_FID, R_IR=rir, band="ir")
                      for p in P_GRID])
        out[("batchelor", rir)] = y
        ax.loglog(P_GRID, y, color=PALETTE[5], lw=1.1, alpha=0.35 + 0.2 * i,
                  label=rf"$R_{{\rm IR}}={rir:g}$")
    y2 = np.array([bs.omega_gw(p, M=M_FID, R=R_FID, R_IR=100.0, band="ir",
                               ir="saffman") for p in P_GRID])
    out[("saffman", 100.0)] = y2
    ax.loglog(P_GRID, y2, color=PALETTE[1], lw=1.4, ls="--",
              label=r"$s=2$, $R_{\rm IR}=100$")
    ref = out[("batchelor", 100.0)]
    i0 = 3
    ax.loglog(P_GRID, ref[i0] * (P_GRID / P_GRID[i0]) ** 3, ls=":",
              color="0.55", lw=0.8)
    ax.text(0.06, 0.9, r"$p^3$", transform=ax.transAxes, fontsize=8, color="0.4")
    ax.set_xlabel(r"$p=k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^3H[\mathcal{S}_{\rm IR}](p,p)$")
    ax.set_title(r"(b) the floor is independent of the source infrared", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    apply_max_ticks(ax)
    return out


def main() -> None:
    apply_paper_style(grid=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), constrained_layout=True)
    panel_masks(axes[0])
    data = panel_floor(axes[1])

    j = int(np.argmin(np.abs(P_GRID - IR_PROBE)))
    amps = [data[("batchelor", r)][j] for r in R_IR_SWEEP]
    spread = max(amps) / min(amps) - 1.0
    ratio_s = data[("saffman", 100.0)][j] / data[("batchelor", 100.0)][j]
    print(f"  at p={P_GRID[j]:.3g}: R_IR spread over {R_IR_SWEEP} = {spread*100:.2f}%")
    print(f"  s=2 vs s=4 amplitude ratio = {ratio_s:.2f}")
    for key in (("batchelor", 100.0), ("saffman", 100.0)):
        m = P_GRID <= 2e-2
        sl = np.polyfit(np.log(P_GRID[m]), np.log(data[key][m]), 1)[0]
        print(f"  IR slope [{key[0]}] = {sl:+.3f}")
    out = save_figure(fig, "band_split_masks")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
