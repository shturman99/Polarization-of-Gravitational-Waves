"""stationary_mach_peak.py + reference wavenumbers M^3 k0 (=eps) and M k0.

Same two panels as stationary_mach_peak.py -- (a) absolute Omega_GW(p)=p^3 H(p,p),
(b) M^3-compensated -- but with two sets of per-M vertical reference lines drawn
on the p = k/k0 axis:

  - DASHED at p = M^3 k0  (= eps, the cascade-rate scale the user asked about),
  - DOTTED at p = M   k0  (the sweeping scale).

It prints, for each M, the peak p_peak and the two ratios
  p_peak / (M^3 k0)   and   p_peak / (M k0),
so you can read directly which normalization makes the peak M-independent.
Spoiler from p_peak ~ 1.47 M: /(M^3 k0) = 1.47/M^2 (NOT constant); /(M k0) = 1.47.
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

MACH_LIST = (0.03, 0.1, 0.3, 1.0, 3.0)
R = 1.0e4
K0 = 1.0
PS = np.logspace(-3.5, 1.0, 130)


def spectrum(M: float) -> np.ndarray:
    return np.array([p ** 3 * H_pq(p, p, M=M, R=R) for p in PS])


def refined_peak(ps: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    spec = np.asarray(spec)
    i = int(np.argmax(spec))
    if i == 0 or i == len(spec) - 1:
        return float(ps[i]), float(spec[i])
    lx, ly = np.log(ps[i - 1:i + 2]), np.log(spec[i - 1:i + 2])
    denom = (lx[0] - lx[1]) * (lx[0] - lx[2]) * (lx[1] - lx[2])
    a = (lx[2] * (ly[1] - ly[0]) + lx[1] * (ly[0] - ly[2]) + lx[0] * (ly[2] - ly[1])) / denom
    b = (lx[2] ** 2 * (ly[0] - ly[1]) + lx[1] ** 2 * (ly[2] - ly[0])
         + lx[0] ** 2 * (ly[1] - ly[2])) / denom
    return float(np.exp(-b / (2.0 * a))), float(spec[i])


def main(name: str = "stationary_mach_peak_klines"):
    apply_paper_style()
    specs, peaks = {}, {}
    print(f"{'M':>7} {'p_peak':>10} {'M^3 k0':>10} {'M k0':>8} "
          f"{'pk/(M^3 k0)':>12} {'pk/(M k0)':>10}")
    for M in MACH_LIST:
        s = spectrum(M)
        ppk, ompk = refined_peak(PS, s)
        specs[M] = (s, ppk, ompk)
        peaks[M] = ppk
        print(f"{M:7.3f} {ppk:10.4f} {M**3*K0:10.4g} {M*K0:8.3f} "
              f"{ppk/(M**3*K0):12.4f} {ppk/(M*K0):10.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True)
    for c, M in enumerate(MACH_LIST):
        col = PALETTE[(c + 1) % len(PALETTE)]
        s, ppk, ompk = specs[M]
        m = (s > 0) & np.isfinite(s)
        axA.loglog(PS[m], s[m], color=col, label=rf"$M={M:g}$")
        axA.loglog(ppk, ompk, "o", color=col, ms=4)
        axB.loglog(PS[m], s[m] / M ** 3, color=col)
        axB.loglog(ppk, ompk / M ** 3, "o", color=col, ms=4)
        for ax in (axA, axB):
            ax.axvline(M ** 3 * K0, color=col, ls="--", lw=0.9, alpha=0.7)
            ax.axvline(M * K0, color=col, ls=":", lw=0.9, alpha=0.7)

    # legend proxies for the two line families
    axA.plot([], [], color="0.3", ls="--", lw=0.9, label=r"$p=M^{3}k_0\,(=\varepsilon)$")
    axA.plot([], [], color="0.3", ls=":", lw=0.9, label=r"$p=M k_0$")

    for ax in (axA, axB):
        ax.set_xlabel(r"$p = k/k_0$")
        ax.set_xlim(PS.min(), 1e2)
        apply_max_ticks(ax)
    axA.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^{3}H(p,p)$")
    axB.set_ylabel(r"$\Omega_{\rm GW}(p)/M^{3}$")
    axA.set_ylim(1e-13, 5.0 * max(specs[M][0].max() for M in MACH_LIST))
    axB.set_ylim(1e-14, 5.0 * max((specs[M][0] / M ** 3).max() for M in MACH_LIST))
    axA.set_title("(a) absolute")
    axB.set_title(r"(b) $M^{3}$-compensated")
    axA.legend(loc="upper left", fontsize=7.2, handlelength=1.4)
    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
