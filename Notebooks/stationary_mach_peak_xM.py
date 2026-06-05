"""stationary_mach_peak.py with the x-axis normalized by M (first power).

Same two panels -- (a) absolute Omega_GW(p)=p^3 H(p,p), (b) M^3-compensated --
but plotted against xi = p/M = k/(M k0).  Because p_peak ~ 1.47 M, every curve's
peak lands at the same xi_* ~ 1.47 (vertical line); the M^3-compensated panel
then collapses the whole spectrum onto one universal curve A G(xi).
A single dot per curve marks the measured peak; they stack on xi_* ~ 1.47.
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


def main(name: str = "stationary_mach_peak_xM"):
    apply_paper_style()
    specs = {}
    print(f"{'M':>7} {'p_peak':>10} {'xi_peak=p_peak/M':>18}")
    for M in MACH_LIST:
        s = spectrum(M)
        ppk, ompk = refined_peak(PS, s)
        specs[M] = (s, ppk, ompk)
        print(f"{M:7.3f} {ppk:10.4f} {ppk / M:18.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True)
    for c, M in enumerate(MACH_LIST):
        col = PALETTE[(c + 1) % len(PALETTE)]
        s, ppk, ompk = specs[M]
        xi = PS / M                                  # k/(M k0)
        m = (s > 0) & np.isfinite(s)
        axA.loglog(xi[m], s[m], color=col, label=rf"$M={M:g}$")
        axA.loglog(ppk / M, ompk, "o", color=col, ms=4)
        # on the xi=p/M axis the p^3 rise carries an extra M^3, so the IR
        # collapse needs M^6 (not M^3): Omega/M^6 = A xi^3 G(xi), pure xi.
        axB.loglog(xi[m], s[m] / M ** 6, color=col)
        axB.loglog(ppk / M, ompk / M ** 6, "o", color=col, ms=4)

    for ax in (axA, axB):
        ax.axvline(1.47, color="0.4", ls="--", lw=1.0)
        ax.set_xlabel(r"$\xi = p/M = k/(M k_0)$")
        apply_max_ticks(ax)
    axA.text(1.47 * 1.15, 1e-12, r"$\xi_\ast\simeq1.47$", fontsize=8, color="0.3")
    axA.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^{3}H(p,p)$")
    axB.set_ylabel(r"$\Omega_{\rm GW}(p)/M^{6}$")
    axA.set_ylim(1e-13, 5.0 * max(specs[M][0].max() for M in MACH_LIST))
    b_peak = max((specs[M][0] / M ** 6).max() for M in MACH_LIST)
    axB.set_ylim(b_peak * 1e-11, 5.0 * b_peak)
    axA.set_title("(a) absolute")
    axB.set_title(r"(b) $M^{6}$-compensated (full collapse)")
    axA.legend(loc="upper left", fontsize=7.6, handlelength=1.4)
    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
