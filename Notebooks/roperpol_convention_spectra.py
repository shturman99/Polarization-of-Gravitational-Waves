"""Analytic GW + fluid spectra in the Roper Pol et al. (2020) Omega(k)/k convention.

Roper Pol, Mandal, Brandenburg, Kahniashvili & Kosowsky, PRD 102, 083512 (2020)
[arXiv:1903.08585], Fig. 1 (run ini2) plot the spectra as Omega(k)/k versus k,
where Omega(k)=drho/dln k (energy per logarithmic wavenumber), so Omega(k)/k is
the per-dk spectrum E(k).

In our variables Omega_GW(p) ~ p^3 H(p,p) = drho_GW/dln k, hence

    Omega_GW(k)/k  ~  p^2 H(p,p),    p = k/k0.

FULL INPUT SPECTRUM.  Unlike the inertial-range-only kernel core.H_pq, here the GW
spectrum is computed from a *full* fluid input spectrum with both parts:

    E(k) ~ k^{s}       (k <  k0, infrared band; Batchelor s=4 / Saffman s=2)
    E(k) ~ k^{-5/3}    (k >= k0, Kolmogorov inertial range),

fed into the same Gogoberidze Appendix-A kernel via the shape factor
S(k)=A(k)/A_Kol(k) (Notebooks/_fullspectrum_kernel.py).  The IR band spans
k in [k0/R_IR, k0] with R_IR = k0/k_IR the infrared dynamic range.  Setting
R_IR=1 recovers core.H_pq exactly (verified in that module).

WHAT THE CODE ACTUALLY SHOWS (honest, no fabricated tails):
  * Feeding the fluid IR band raises the GW amplitude near and below the peak
    (~2-3x) but does NOT change the GW infrared slope, which stays at the *causal*
    Omega_GW ~ k^3 (Omega_GW/k ~ k^2) set by analyticity of the stress -- the GW IR
    is causal, not inherited from the fluid.
  * The ultraviolet (k>k0) is set by the inertial range and is essentially
    unchanged by the IR band; at M=1 it still steepens past both the k^{-11/3}
    (Roper Pol) and k^{-9/2} (Gogoberidze) reference slopes.  The
    simulation-analytic tension is therefore robust to the IR treatment.

NORMALIZATION.  Both curves are DIMENSIONLESS; the GW curve is the bare kernel with
the source stress set to O(1).  The physical relic Omega_GW is suppressed by the
source amplitude squared (~Omega_M^2), the horizon/efficiency factor (~H_*/k_*) and
gravitational couplings (together ~1e-9..-11), which is why the simulated
Omega_GW/k sits ~1e-13 while the bare kernel is ~1e-2.  The GW-vs-fluid VERTICAL
OFFSET here is therefore NOT physical; only shapes (slopes, peak location) compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _fullspectrum_kernel import IR_EXPONENTS, omega_gw_over_k  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

M = 1.0
R = 1.0e4          # k_d/k0 = R^{3/4} = 1000 (inertial range)
R_IR = 100.0       # k0/k_IR = 100 (two decades of infrared band)
IR = "batchelor"   # fluid IR energy slope E~k^4 fed into the kernel


def _fit_slope(p, y, lo, hi):
    """Least-squares log-log slope of y(p) over the window [lo, hi]."""
    m = (p >= lo) & (p <= hi)
    return float(np.polyfit(np.log(p[m]), np.log(y[m]), 1)[0])


def main(name: str = "roperpol_convention_spectra"):
    apply_paper_style()

    p = np.geomspace(1.0 / R_IR * 1.2, 60.0, 80)
    gw = np.array([omega_gw_over_k(pp, M=M, R=R, R_IR=R_IR, ir=IR) for pp in p])
    gw_kol = np.array([omega_gw_over_k(pp, M=M, R=R, R_IR=1.0) for pp in p])  # inertial only
    p_peak = p[np.argmax(gw)]

    ir_slope = _fit_slope(p, gw, 0.02, 0.3)      # causal GW infrared
    uv_slope = _fit_slope(p, gw, 3.0, 8.0)       # analytic GW ultraviolet

    # FULL fluid input spectrum Omega_M/k = E(k): IR band k^4 joined to Kolmogorov k^-5/3.
    # The GW-fluid vertical offset is unphysical (see header), so the fluid curve is
    # rescaled to share the GW peak -- only the SHAPES (slopes, peak) are comparable.
    s_ir = IR_EXPONENTS[IR] - 5.0 / 3.0          # E(k)~k^{s} below k0 (s=4 Batchelor)
    fluid = np.where(p >= 1.0, p ** (-5.0 / 3.0), p ** s_ir)
    fluid *= gw.max() / fluid.max()

    c_gw, c_fluid = PALETTE[6], PALETTE[0]
    fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)

    ax.loglog(p, fluid, "-", lw=2.0, color=c_fluid,
              label=rf"model $\Omega_{{\rm M}}/k=E(k)$: $k^{{{int(s_ir)}}}\!\to\!k^{{-5/3}}$ (rescaled)")
    ax.loglog(p, gw, "-", lw=2.2, color=c_gw,
              label=rf"$\Omega_{{\rm GW}}/k$, full input ($M={M:.0f}$, $R_{{\rm IR}}={R_IR:.0f}$)")
    ax.loglog(p, gw_kol, "--", lw=1.4, color=c_gw, alpha=0.65,
              label=r"$\Omega_{\rm GW}/k$, inertial only ($k\geq k_0$)")
    ax.axvline(1.0, color="0.6", lw=0.9, ls=":")  # k = k0

    # causal IR guide k^2 (=> Omega_GW ~ k^3), anchored to the computed GW curve.
    x0, x1 = 0.02, 0.25
    y0 = 1.7 * np.exp(np.interp(np.log(x0), np.log(p), np.log(gw)))
    xs = np.geomspace(x0, x1, 2)
    ax.loglog(xs, y0 * (xs / x0) ** 2.0, color="0.35", ls="--", lw=1.1,
              label=r"$k^{2}$ (causal IR, $\Omega_{\rm GW}\!\sim k^3$)")

    # UV reference slopes (guides, not data), anchored to the GW curve.
    xu0, xu1 = 2.5, 30.0
    yu0 = 1.5 * np.exp(np.interp(np.log(xu0), np.log(p), np.log(gw)))
    xu = np.geomspace(xu0, xu1, 2)
    ax.loglog(xu, yu0 * (xu / xu0) ** (-11.0 / 3.0), color="0.30", ls="-.", lw=1.1,
              label=r"$k^{-11/3}$ (Roper Pol; reference)")
    ax.loglog(xu, yu0 * (xu / xu0) ** (-9.0 / 2.0), color="0.55", ls=":", lw=1.3,
              label=r"$k^{-9/2}$ (Gogoberidze; reference)")

    ax.set_xlabel(r"$k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(k)/k$ and $\Omega_{\rm M}(k)/k$  (dimensionless)")
    ax.set_xlim(p[0], p[-1])
    ax.set_ylim(gw.max() * 1e-8, gw.max() * 5)
    ax.legend(loc="lower left", fontsize=7.5)
    apply_max_ticks(ax)
    out = save_figure(fig, name)

    # honest sensitivity print: Saffman vs Batchelor IR.
    gw_saff = np.array([omega_gw_over_k(pp, M=M, R=R, R_IR=R_IR, ir="saffman") for pp in p])
    print(f"saved {out}")
    print(f"  full-input GW peak at p={p_peak:.2f}")
    print(f"  GW IR slope k^{ir_slope:+.2f} (causal, expect +2);  "
          f"UV slope k^{uv_slope:+.2f} (vs ref -3.67, -4.50)")
    print(f"  IR-band uplift at peak: x{gw[np.argmax(gw)]/gw_kol[np.argmax(gw)]:.2f} (Batchelor)")
    print(f"  Saffman vs Batchelor at peak: x{gw_saff[np.argmax(gw)]/gw[np.argmax(gw)]:.2f} "
          f"(IR-slope sensitivity; same power laws)")
    return out


if __name__ == "__main__":
    main()
