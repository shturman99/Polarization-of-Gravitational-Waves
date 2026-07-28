#!/usr/bin/env python3
r"""Why the magnetic->GW transfer is fast and the GW peak does not follow the
inverse cascade (Roper Pol run ini3).

Both questions have one answer: GW production is IMPULSIVE. The sudden magnetic
field imposed at t0 rings each GW mode once (a step-function source), depositing
energy within ~one light-crossing time; thereafter the source is quasi-static
and cannot coherently pump the oscillating GW field, so production stops. Because
that burst happens at t~t0, when the field still has its INITIAL spectrum, the GW
spectrum is set at the initial scale 2 k0(t0) and -- GW being non-interacting --
FREEZES there. The subsequent helical inverse cascade drags the magnetic peak k0
to smaller k, but GW emission has already ceased, so the frozen GW peak cannot
follow it.

(a) cumulative GW energy vs the (decaying) magnetic energy, and the GW production
    rate |dE_GW/dt|: the GW energy is in place almost immediately and the rate
    collapses to ~0 while the magnetic energy is still decaying.
(b) magnetic peak k0(t) (inverse cascade, moving down) vs the GW peak (frozen):
    the GW peak barely moves while k0 falls, so the ratio k_GW/k0 actually rises.

Data: public Pencil Code spectra (Zenodo 10.5281/zenodo.3692072).
Figure: images/roperpol_emission_timing.pdf
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

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _rm(y, w=40):
    return np.array([np.median(y[max(0, i - w):i + w + 1]) for i in range(len(y))])


def main():
    apply_paper_style()
    kk, tG, EGk, ESk = load("ini3")
    EM = _trapz(ESk, kk, axis=1)
    EG = _trapz(EGk, kk, axis=1)
    k0 = _rm(kk[np.argmax(ESk, axis=1)])                 # magnetic peak
    kGW = _rm(kk[np.argmax(kk * EGk, axis=1)])           # GW peak (Omega_GW=k E_GW)
    rate = np.abs(np.gradient(EG, tG))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)

    # (a) impulsive production: cumulative GW energy + rate, vs decaying magnetic
    ax1.semilogy(tG, EG / EG[-1], color=PALETTE[5], lw=1.6, label=r"$\mathcal{E}_{\rm GW}(t)/\mathcal{E}_{\rm GW}^{\rm final}$")
    ax1.semilogy(tG, EM / EM[0], color=PALETTE[6], lw=1.6, ls="--", label=r"$\mathcal{E}_{\rm M}(t)/\mathcal{E}_{\rm M}^{0}$")
    ax1.semilogy(tG, rate / rate.max(), color=PALETTE[1], lw=1.0, alpha=0.8,
                 label=r"$|d\mathcal{E}_{\rm GW}/dt|$ (norm.)")
    ax1.axvspan(tG[0], 1.1, color="0.5", alpha=0.12, lw=0)
    ax1.text(1.11, 3e-3, "GW emitted here,\nthen production stops", fontsize=7, color="0.3")
    ax1.set_xlabel(r"time $t$")
    ax1.set_ylabel(r"normalised energy / rate")
    ax1.set_ylim(1e-4, 3)
    ax1.set_title(r"(a) GW production is impulsive", fontsize=9.5)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    apply_max_ticks(ax1)

    # (b) magnetic peak cascades down; GW peak frozen
    ax2.plot(tG, k0, color=PALETTE[6], lw=1.8, label=r"magnetic peak $k_0(t)$ (inverse cascade)")
    ax2.plot(tG, kGW, color=PALETTE[5], lw=1.8, label=r"GW peak $k_{\rm peak}^{\rm GW}(t)$ (frozen)")
    ax2.plot(tG, 2 * k0, color=PALETTE[6], lw=0.9, ls=":", label=r"$2k_0(t)$")
    ax2.set_xlabel(r"time $t$")
    ax2.set_ylabel(r"wavenumber $k$")
    ax2.set_ylim(0, 16)
    ax2.set_title(r"(b) $k_0$ cascades down, GW peak stays", fontsize=9.5)
    ax2.legend(fontsize=6.8, frameon=False, loc="upper right")
    apply_max_ticks(ax2)

    out = save_figure(fig, "roperpol_emission_timing")
    print(f"saved {out}")
    print(f"GW energy fraction in place by t=1.1: {np.interp(1.1, tG, EG) / EG[-1]:.0%}")
    print(f"magnetic peak k0: {k0[0]:.1f} -> {k0[-1]:.1f} (x{k0[0]/k0[-1]:.2f});  "
          f"GW peak: {kGW[0]:.1f} -> {kGW[-1]:.1f} (x{kGW[0]/kGW[-1]:.2f})")
    print(f"magnetic energy remaining at end: {EM[-1]/EM[0]:.0%}")


if __name__ == "__main__":
    main()
