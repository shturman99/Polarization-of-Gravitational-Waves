#!/usr/bin/env python3
r"""GW spectral peak frequency vs Mach number: Roper Pol (source-scale, ~2 k0) vs
Gogoberidze (sweeping, ~1.47 M).

Omega_GW(p) = p^3 H(p,p) = drho_GW/dln k.  We locate its peak p_peak=k_peak/k0 for
  - decaying (BK2016 power law, fullspatial_decay.H_decay_fast)  -> Roper-Pol picture
  - stationary Kraichnan sweeping (core.H_pq)                    -> Gogoberidze picture
as a function of M.

RESULT (verified):
  decaying  : p_peak ~ 2.4  (~2x the source scale k0), essentially M-INDEPENDENT
              -> matches the Roper Pol "GW peak at twice the source scale".
  stationary: p_peak ~ 1.47 M -- RISES with M, staying BELOW the 2k0 (Roper Pol) line
              for all subsonic M and below the source scale k0 itself for M<0.7
              (the sweeping cutoff p~M suppresses the peak as M drops).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

import roperpol_data  # noqa: E402  (single source of truth for digitized-data numbers)
from fullspatial_decay import H_decay_fast, H_pq  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)


def _peak(fn, plo=0.08, phi=10.0, n=44):
    ps = np.geomspace(plo, phi, n)
    sp = np.array([fn(p) for p in ps])
    i = int(np.argmax(sp))
    il, ir = max(i - 1, 0), min(i + 1, n - 1)
    c = np.polyfit(np.log(ps[il:ir + 1]), np.log(sp[il:ir + 1]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


# Digitized Roper Pol Fig.1 numbers -- COMPUTED, not hardcoded (see roperpol_data.py).
DATA_PEAK = roperpol_data.gw_peak_ratio()   # k_peak^GW / k0 (Omega_GW convention) ~ 1.84
DATA_M = roperpol_data.effective_mach()      # (sqrt(Om_M), sqrt(2 Om_M)) ~ (0.043, 0.060)


def main(name="gw_peak_vs_mach"):
    apply_paper_style()
    Ms = np.array([0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
    p_stat = np.array([_peak(lambda p: p ** 3 * H_pq(p, p, M=M, R=1e4), plo=0.02) for M in Ms])
    p_dec = np.array([_peak(lambda p: p ** 3 * H_decay_fast(p, p, M=M, R=1e4)) for M in Ms])

    # Time-resolved simulation tracks: the GW peak k_peak(t)/k0(t) and M(t)=u_rms(t)
    # for the two decaying runs, from the public Pencil Code data.
    from roperpol_all_runs import gw_peak_track  # noqa: E402
    tracks = {r: gw_peak_track(r) for r in ("ini2", "ini3")}

    import matplotlib.pyplot as plt
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    # ---- panel (a): the peak moves through time (absolute) but its RATIO to k0 is pinned
    t3, M3, r3 = tracks["ini3"]
    axa.plot(t3, r3, "o", color=PALETTE[1], ms=2.0, alpha=0.18)
    # running median to guide the eye through the resolution (discrete-shell) scatter
    win = 41
    med = np.array([np.median(r3[max(0, i - win): i + win + 1]) for i in range(len(r3))])
    axa.plot(t3, med, "-", color=PALETTE[1], lw=2.2, label=r"$k_{\rm peak}^{\rm GW}(t)/k_0(t)$ (ini3, median)")
    axa.axhline(2.4, color=PALETTE[2], lw=1.1, ls="--", label=r"decaying kernel $\simeq2.4$")
    axa.set_xlabel(r"time $t$")
    axa.set_ylabel(r"GW peak $/\,k_0$")
    axa.set_ylim(0, 3.6)
    axa.set_title(r"(a) peak stays source-pinned in time", fontsize=9.5)
    axr = axa.twinx()
    axr.plot(t3, M3, color="0.45", lw=1.2, ls=":")
    axr.set_ylabel(r"$M(t)=u_{\rm rms}/c$", color="0.45", fontsize=9)
    axr.tick_params(axis="y", labelcolor="0.45")
    axr.set_ylim(0, 0.05)
    axr.text(t3[len(t3) // 2], M3[len(M3) // 2] * 1.25, r"$M(t)\!\downarrow$",
             color="0.45", fontsize=8)
    axa.legend(loc="lower right", fontsize=7)
    apply_max_ticks(axa)

    # ---- panel (b): peak vs Mach, with the simulation TIME-TRACK coloured by time
    axb.axhline(2.0, color=PALETTE[2], lw=1.1, ls="--", label=r"$2k_0$")
    axb.axhline(1.0, color="0.6", lw=1.0, ls=":", label=r"source scale $k_0$")
    axb.plot(Ms, p_dec, "o-", color=PALETTE[1], lw=1.8, ms=4,
             label=r"decaying kernel ($\simeq2.4$)")
    axb.plot(Ms, p_stat, "s-", color=PALETTE[0], lw=1.8, ms=4,
             label=r"stationary $\simeq1.47\,M$")
    sc = None
    for r in ("ini2", "ini3"):
        t, M, ratio = tracks[r]
        good = (M > 5e-3)                       # drop the switch-on tail where u_rms~0
        tn = (t[good] - t[good].min()) / (t[good].max() - t[good].min())
        sc = axb.scatter(M[good], ratio[good], c=tn, cmap="viridis", s=9,
                         alpha=0.8, zorder=5, linewidths=0)
    cb = fig.colorbar(sc, ax=axb, pad=0.02, fraction=0.05)
    cb.set_label(r"time (norm.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axb.annotate(r"source decays: $M(t)\!\downarrow$, peak fixed",
                 xy=(0.03, 2.55), fontsize=7, color="0.25")
    axb.set_xscale("log")
    axb.set_xlabel(r"Mach number $M=u_0/c$")
    axb.set_ylabel(r"GW peak $p_{\rm peak}=k_{\rm peak}/k_0$")
    axb.set_xlim(0.015, 3.3)
    axb.set_ylim(0, 3.2)
    axb.set_title(r"(b) $M$-dependence: pinned, not sweeping", fontsize=9.5)
    axb.legend(loc="upper left", fontsize=6.8, framealpha=0.92)
    apply_max_ticks(axb)

    out = save_figure(fig, name)
    plt.close(fig)
    print(f"saved {out}")
    for r in ("ini2", "ini3"):
        t, M, ratio = tracks[r]
        g = M > 5e-3
        print(f"  {r}: M(t) {M[g].min():.3f}-{M[g].max():.3f}, peak/k0 "
              f"median {np.median(ratio[g]):.2f} (decaying 2.4; sweeping 1.47M="
              f"{1.47*np.median(M[g]):.2f})")
    return out


if __name__ == "__main__":
    main()
