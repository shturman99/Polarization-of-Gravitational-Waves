"""Analytic GW + fluid spectra in the Roper Pol et al. (2020) Omega(k)/k convention.

Roper Pol, Mandal, Brandenburg, Kahniashvili & Kosowsky, PRD 102, 083512 (2020)
[arXiv:1903.08585], Fig. 1 (run ini2) plot the spectra as Omega(k)/k versus k,
where Omega(k)=drho/dln k (energy per logarithmic wavenumber), so Omega(k)/k is
the per-dk spectrum E(k).

In our variables Omega_GW(p) ~ p^3 H(p,p) = drho_GW/dln k, hence

    Omega_GW(k)/k  ~  p^2 H(p,p),    p = k/k0,

from the stationary Kraichnan--K41 (Gogoberidze 2007) kernel.  We also show the
model fluid spectrum Omega_M(k)/k = E(k) ~ k^{-5/3}, plotted ONLY for k >= k0
(the Kolmogorov inertial range the kernel assumes -- no fabricated IR).

IMPORTANT (normalization).  Both curves are DIMENSIONLESS.  Omega_GW/k here is the
bare GW kernel with the source stress set to O(1); the physical relic Omega_GW is
suppressed by the source amplitude squared (~Omega_M^2), the horizon/efficiency
factor (~H_*/k_*), and gravitational couplings -- together ~1e-9..-11, which is
why Roper Pol's Omega_GW/k sits ~1e-13 while the bare kernel is ~1e-2.  The
GW-vs-fluid VERTICAL OFFSET in this figure is therefore NOT physical; only the
shapes (slopes, peak location) are comparable.  A calibrated absolute comparison
needs the run's k_*/H_* and Omega_M and is deferred.
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

M = 1.0
R = 1.0e4


def gw_over_k(p: np.ndarray) -> np.ndarray:
    """Omega_GW(k)/k = p^2 H(p,p) from the Kolmogorov (Gogoberidze) kernel."""
    return np.array([pp ** 2 * H_pq(pp, pp, M=M, R=R) for pp in p])


def _fit_slope(p, y, lo, hi):
    """Least-squares log-log slope of y(p) over the window [lo, hi]."""
    m = (p >= lo) & (p <= hi)
    return float(np.polyfit(np.log(p[m]), np.log(y[m]), 1)[0])


def main(name: str = "roperpol_convention_spectra"):
    apply_paper_style()

    p = np.logspace(-1, 3, 70)
    gw = gw_over_k(p)
    p_peak = p[np.argmax(gw)]

    # fitted IR slope of the GW spectrum (causal tail, k < k0)
    ir_slope = _fit_slope(p, gw, 0.1, 0.4)

    # model fluid spectrum Omega_M/k = E(k) ~ k^{-5/3}, Kolmogorov, k >= k0 only.
    pk = p[p >= 1.0]
    fluid = pk ** (-5.0 / 3.0)  # normalized to E(k0)=1; absolute level arbitrary

    c_gw, c_fluid = PALETTE[6], PALETTE[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)

    ax.loglog(pk, fluid, "-", lw=2.0, color=c_fluid,
              label=r"model $\Omega_{\rm M}/k = E(k)\sim k^{-5/3}$ ($k\geq k_0$)")
    ax.loglog(p, gw, "-", lw=2.0, color=c_gw,
              label=rf"analytic $\Omega_{{\rm GW}}/k$ (Kolmogorov, $M={M:.0f}$)")
    ax.axvline(p_peak, color="0.6", lw=0.9, ls=":")

    # fitted GW IR slope guide
    x0, x1 = 0.11, 0.45
    y0 = np.exp(np.interp(np.log(x0), np.log(p), np.log(gw)))
    xs = np.geomspace(x0, x1, 2)
    ax.loglog(xs, 1.6 * y0 * (xs / x0) ** ir_slope, color="0.35", ls="--", lw=1.2,
              label=rf"fitted IR slope $k^{{{ir_slope:+.2f}}}$ (causal)")

    # UV reference slopes (guides, not data), both anchored to the GW curve at the
    # same point so the eye can read how much steeper the analytic UV actually is.
    xu0, xu1 = 3.0, 30.0
    yu0 = 1.4 * np.exp(np.interp(np.log(xu0), np.log(p), np.log(gw)))
    xu = np.geomspace(xu0, xu1, 2)
    ax.loglog(xu, yu0 * (xu / xu0) ** (-11.0 / 3.0), color="0.30", ls="-.", lw=1.2,
              label=r"$k^{-11/3}$ (Roper Pol reported; reference)")
    ax.loglog(xu, yu0 * (xu / xu0) ** (-9.0 / 2.0), color="0.55", ls=":", lw=1.4,
              label=r"$k^{-9/2}$ (Gogoberidze reported; reference)")
    # measured post-peak slope, for the caption/print (the actual analytic UV)
    uv_slope = _fit_slope(p, gw, 3.0, 8.0)

    ax.set_xlabel(r"$k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(k)/k$ and $\Omega_{\rm M}(k)/k$  (dimensionless)")
    ax.set_xlim(p[0], p[-1])
    ax.set_ylim(gw.max() * 1e-6, fluid.max() * 5)
    ax.legend(loc="lower left", fontsize=8)
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"saved {out}")
    print(f"  GW peak at p={p_peak:.2f};  IR slope k^{ir_slope:+.2f} (causal);  "
          f"post-peak UV slope k^{uv_slope:+.2f}  (vs ref -3.67, -4.50)")
    return out


if __name__ == "__main__":
    main()
