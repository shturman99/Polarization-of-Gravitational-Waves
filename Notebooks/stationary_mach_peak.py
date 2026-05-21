"""Stationary Kraichnan--K41 turbulence (Gogoberidze 2007): GW spectrum vs Mach.

Back-to-basics check: for *stationary* turbulence the dimensionless GW spectrum
on the sound-cone diagonal is

    Omega_GW(p) propto p^3 * H_pq(p, q=p; M, R),

with H_pq the Appendix-A kernel of derivation.tex (Eq. Hijij-AppA-dimless).
The Mach number M enters only through the sweeping rate eta_k ~ M k^{2/3}, i.e.
through the Gaussian factor exp(-2 x y/(x+y) q^2/M^2) and the erfc.  This sets an
effective cutoff at p ~ M, so the spectral peak moves to higher frequency as M
grows (p_peak ~ M until the inertial-range/triangle cutoff bends it).

In the deep IR (p << M) the sweeping factors -> 1, so H(p->0) is set by the M^3
prefactor alone: Omega_GW = p^3 H ~ M^3 p^3, i.e. the characteristic strain
h_c = sqrt(p H) ~ M^{3/2} sqrt(p).  Compensating the energy spectrum by M^{-3}
(equivalently the strain by M^{-3/2}) therefore collapses all curves onto a
single universal p^3 line in the IR, leaving only the peak/UV M-dependence.

This script plots, for several M at fixed R:
  (a) the absolute spectrum  (amplitude grows ~ M^3, peak shifts right),
  (b) the M^3-compensated spectrum Omega_GW/M^3  (IR collapses onto p^3).
Peak locations and the IR-collapse spread are printed.
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
PS = np.logspace(-3.5, 1.0, 100)


def spectrum(M: float, R: float = R, ps: np.ndarray = PS) -> np.ndarray:
    """Omega_GW(p) ~ p^3 H_pq(p, p; M, R) on the sound-cone diagonal."""
    return np.array([p ** 3 * H_pq(p, p, M=M, R=R) for p in ps])


def main(name: str = "stationary_mach_peak"):
    apply_paper_style()
    specs, peaks = {}, {}
    for M in MACH_LIST:
        s = spectrum(M)
        specs[M] = s
        peaks[M] = PS[np.argmax(s)]
        print(f"M={M:<5}  p_peak={peaks[M]:.3f}  Omega_peak={s.max():.3e}")

    # IR-collapse test: Omega_GW/M^3 at the smallest p should be M-independent.
    ir_vals = [specs[M][0] / M ** 3 for M in MACH_LIST]
    print(f"\nIR collapse test at p={PS[0]:.1e}:  Omega_GW/M^3 spread "
          f"max/min = {max(ir_vals) / min(ir_vals):.4f}  (-> 1 means coincident)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)
    for c, M in enumerate(MACH_LIST):
        col = PALETTE[(c + 1) % len(PALETTE)]
        s = specs[M]
        m = (s > 0) & np.isfinite(s)
        ppk, ipk = peaks[M], np.argmax(s)
        # (a) absolute
        axA.loglog(PS[m], s[m], color=col, label=rf"$M={M:g}$")
        axA.loglog(ppk, s[ipk], "o", color=col, ms=4)
        # (b) M^3-compensated: IR collapses onto a universal p^3 line
        axB.loglog(PS[m], s[m] / M ** 3, color=col)
        axB.loglog(ppk, s[ipk] / M ** 3, "o", color=col, ms=4)

    # universal p^3 guide for the compensated panel: coefficient = H/M^3 in IR
    ir_coeff = float(np.mean(ir_vals)) / PS[0] ** 3
    p_ir = PS[PS < 0.5]
    axB.loglog(p_ir, ir_coeff * p_ir ** 3, color="0.4", ls=":", lw=1.0,
               label=r"$\propto p^{3}$")

    for ax in (axA, axB):
        ax.set_xlabel(r"$p = k/k_0$")
        apply_max_ticks(ax)
    axA.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^{3}H(p,p)$")
    axB.set_ylabel(r"$\Omega_{\rm GW}(p)/M^{3}$")
    peak_max = max(specs[M].max() for M in MACH_LIST)
    axA.set_ylim(1e-13, 5.0 * peak_max)
    comp_peak = max((specs[M] / M ** 3).max() for M in MACH_LIST)
    axB.set_ylim(1e-14, 5.0 * comp_peak)
    # M values are described in the caption (no on-plot legend); colours run in
    # palette order, increasing with curve amplitude and peak frequency.
    axA.set_title("(a) absolute")
    axB.set_title(r"(b) $M^{3}$-compensated (IR collapse)")
    axB.legend(loc="lower right")
    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
