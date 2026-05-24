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
    Ms = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
    p_stat = np.array([_peak(lambda p: p ** 3 * H_pq(p, p, M=M, R=1e4), plo=0.02) for M in Ms])
    p_dec = np.array([_peak(lambda p: p ** 3 * H_decay_fast(p, p, M=M, R=1e4)) for M in Ms])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)

    ax.axhline(2.0, color=PALETTE[2], lw=1.2, ls="--",
               label=r"$2k_0$ (Roper Pol: $\sim2\times$ source)")
    ax.axhline(1.0, color="0.6", lw=1.0, ls=":", label=r"source scale $k_0$")
    ax.plot(Ms, p_dec, "o-", color=PALETTE[1], lw=1.9, ms=4.5,
            label=r"decaying (Roper Pol kernel): $\simeq2.4$, $M$-indep.")
    ax.plot(Ms, p_stat, "s-", color=PALETTE[0], lw=1.9, ms=4.5,
            label=r"stationary (Gogoberidze): $\simeq1.47\,M$")

    # digitized Roper Pol simulation peak at its (subsonic) effective Mach
    Mc = np.sqrt(DATA_M[0] * DATA_M[1])
    ax.errorbar([Mc], [DATA_PEAK], xerr=[[Mc - DATA_M[0]], [DATA_M[1] - Mc]],
                fmt="*", color="crimson", ms=14, capsize=3, mec="0.2", mew=0.6, zorder=5,
                label=r"Roper Pol \emph{simulation} (digitized)")

    ax.set_xscale("log")
    ax.set_xlabel(r"effective Mach number $M$")
    ax.set_ylabel(r"GW peak $p_{\rm peak}=k_{\rm peak}/k_0$")
    ax.set_xlim(0.035, 3.3)
    ax.set_ylim(0, 3.0)
    ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92)
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    plt.close(fig)
    print(f"saved {out}")
    print(f"  data peak {DATA_PEAK} k0 at M~{Mc:.3f};  "
          f"stationary there ~{1.47*Mc:.2f} k0 (below k0), decaying ~2.3 k0")
    print(f"  {'M':>6}{'stationary':>12}{'decaying':>10}")
    for M, ps, pd in zip(Ms, p_stat, p_dec):
        print(f"  {M:6.2f}{ps:12.2f}{pd:10.2f}")
    return out


if __name__ == "__main__":
    main()
