r"""Stationary GW spectrum with the Kolmogorov constraint M=(eps/k0)^{1/3}.

The standard panel (stationary_mach_peak.py) varies M at FIXED k0 -- which
silently varies the cascade rate eps = u_rms^3 k0 = M^3 k0.  But M and k0 are
not independent: Kolmogorov ties them through

    M = (eps / k0)^{1/3}   <=>   k0 = eps * M^{-3}.

Holding the physical cascade rate eps fixed (what that relation implies) and
varying M therefore CHANGES k0: larger M => smaller k0 (bigger outer eddies).

Re-expressed in an M-independent physical wavenumber kappa = k/eps,

    kappa = p k0 / eps = p / M^3            (set eps = 1 as the wavenumber unit)

the kernel peak p_peak = k_peak/k0 = 1.47 M maps to

    kappa_peak = 1.47 M / M^3 = 1.47 / M^2,

i.e. the GW peak moves to LOWER physical frequency as M grows -- the opposite
direction to the fixed-k0 plot, where it scales as 1.47 M.

Panels:
  (a) peak-normalized spectra Omega_GW/Omega_peak vs kappa = k/eps = p/M^3;
      the peaks march to lower kappa (~1.47/M^2) as M increases.
  (b) the measured peak location vs M in both frames: fixed-eps (1.47/M^2,
      down) and the old fixed-k0 law (1.47 M, up).
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
# Mach numbers shown as spectra (panel a)
MACH_SPEC = (0.03, 0.1, 0.3, 1.0, 3.0)
# p = k/k0 grid (kernel-native); kappa = p/M^3 is built per curve
PS = np.logspace(-3.5, 1.0, 130)
# finer Mach sweep for the peak-location law (panel b)
MACH_SCAN = np.logspace(np.log10(0.02), np.log10(5.0), 15)
PS_SCAN = np.logspace(-3.5, 1.2, 120)


def omega_gw(ps: np.ndarray, M: float) -> np.ndarray:
    """Omega_GW(p) = p^3 H_pq(p, p; M, R) on the sound-cone diagonal."""
    return np.array([p ** 3 * H_pq(p, p, M=M, R=R) for p in ps])


def refined_peak(ps: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    """Sub-grid peak (p_peak, Omega_peak): log-log parabola through the argmax."""
    spec = np.asarray(spec)
    i = int(np.argmax(spec))
    if i == 0 or i == len(spec) - 1:
        return float(ps[i]), float(spec[i])
    lx = np.log(ps[i - 1:i + 2])
    ly = np.log(spec[i - 1:i + 2])
    denom = (lx[0] - lx[1]) * (lx[0] - lx[2]) * (lx[1] - lx[2])
    a = (lx[2] * (ly[1] - ly[0]) + lx[1] * (ly[0] - ly[2]) + lx[0] * (ly[2] - ly[1])) / denom
    b = (lx[2] ** 2 * (ly[0] - ly[1]) + lx[1] ** 2 * (ly[2] - ly[0])
         + lx[0] ** 2 * (ly[1] - ly[2])) / denom
    return float(np.exp(-b / (2.0 * a))), float(spec[i])


def main(name: str = "stationary_fixed_epsilon"):
    apply_paper_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.7), constrained_layout=True)

    # ---- (a) peak-normalized spectra on the physical axis kappa = k/eps ------
    print(f"{'M':>8} {'p_peak':>10} {'kappa_peak':>12} {'1.47/M^2':>10}")
    for c, M in enumerate(MACH_SPEC):
        col = PALETTE[(c + 1) % len(PALETTE)]
        om = omega_gw(PS, M)
        kappa = PS / M ** 3                       # k/eps = p/M^3 (eps = 1)
        ppk, ompk = refined_peak(PS, om)
        axA.loglog(kappa, om / ompk, color=col, lw=1.6, label=rf"$M={M:g}$")
        axA.loglog(ppk / M ** 3, 1.0, "o", color=col, ms=4)
        print(f"{M:8.3f} {ppk:10.4f} {ppk / M ** 3:12.4f} {1.47 / M ** 2:10.4f}")

    axA.set_xlabel(r"$\kappa = k/\varepsilon = p/M^{3}$")
    axA.set_ylabel(r"$\Omega_{\rm GW}/\Omega_{\rm peak}$")
    axA.set_title(r"(a) fixed $\varepsilon$: peak $\to$ lower $\kappa$ as $M\uparrow$")
    axA.set_ylim(1e-9, 3.0)
    axA.legend(loc="lower right", fontsize=8, handlelength=1.3)
    apply_max_ticks(axA)

    # ---- (b) peak-location law: fixed-eps (down) vs fixed-k0 (up) ------------
    scan_p, scan_M = [], []
    for M in MACH_SCAN:
        om = omega_gw(PS_SCAN, M)
        ppk, _ = refined_peak(PS_SCAN, om)
        scan_p.append(ppk)
        scan_M.append(M)
    scan_p = np.array(scan_p)
    scan_M = np.array(scan_M)
    kappa_peak = scan_p / scan_M ** 3             # fixed eps
    mline = np.logspace(np.log10(MACH_SCAN.min()), np.log10(MACH_SCAN.max()), 60)

    axB.loglog(scan_M, kappa_peak, "o", color=PALETTE[6], ms=5,
               label=r"fixed $\varepsilon$: $\kappa_{\rm peak}=k_{\rm peak}/\varepsilon$")
    axB.loglog(mline, 1.47 / mline ** 2, color=PALETTE[6], ls="--", lw=1.2,
               label=r"$1.47\,M^{-2}$")
    axB.loglog(scan_M, scan_p, "s", color=PALETTE[2], ms=4.5,
               label=r"fixed $k_0$: $k_{\rm peak}/k_0$")
    axB.loglog(mline, 1.47 * mline, color=PALETTE[2], ls=":", lw=1.2,
               label=r"$1.47\,M$")
    axB.set_xlabel(r"$M$")
    axB.set_ylabel(r"peak wavenumber")
    axB.set_title(r"(b) opposite trends in the two frames")
    axB.legend(loc="lower left", fontsize=7.6, handlelength=1.5)
    apply_max_ticks(axB)

    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
