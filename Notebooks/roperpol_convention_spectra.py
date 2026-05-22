"""Analytic GW spectrum in the Roper Pol et al. (2020) Omega(k)/k convention.

Roper Pol, Mandal, Brandenburg, Kahniashvili & Kosowsky, PRD 102, 083512 (2020)
[arXiv:1903.08585], Fig. 1 (run ini2) plot the spectra as Omega(k)/k versus k,
where Omega(k)=drho/dln k (energy per logarithmic wavenumber), so Omega(k)/k is
the per-dk spectrum E(k).

We plot ONLY what our derivation contains: the analytic GW spectrum from the
stationary Kraichnan--K41 (Gogoberidze 2007) kernel, which assumes a Kolmogorov
inertial range and is therefore shown only for k >= k0 (above the peak). In our
variables Omega_GW(p) ~ p^3 H(p,p) = drho_GW/dln k, hence

    Omega_GW(k)/k  ~  p^2 H(p,p),    p = k/k0.

For reference we draw the slope the simulation REPORTS, k^{-11/3}, as a dash-dot
guide. No simulation data points are plotted here: a quantitative comparison
against the actual simulated spectra (real digitized or run data) is deferred.
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

M = 3.0
R = 1.0e4


def gw_over_k(p: np.ndarray) -> np.ndarray:
    """Omega_GW(k)/k = p^2 H(p,p) from the Kolmogorov (Gogoberidze) kernel."""
    return np.array([pp ** 2 * H_pq(pp, pp, M=M, R=R) for pp in p])


def main(name: str = "roperpol_convention_spectra"):
    apply_paper_style()

    # analytic GW only over the inertial range k >= k0 (p >= 1)
    p = np.logspace(0.0, 2.3, 60)
    gw = gw_over_k(p)

    vermilion = PALETTE[6]
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    ax.loglog(p, gw, "-", lw=2.0, color=vermilion,
              label=r"analytic $\Omega_{\rm GW}/k$ (Kolmogorov, $M=3$)")

    # reference slope ONLY: the exponent k^{-11/3} reported by the Roper Pol et
    # al. simulation. This is a guide line, NOT their data -- no simulated
    # spectrum is plotted here. Anchored to the analytic curve in the window
    # where the two are compared.
    x0, x1 = 3.0, 40.0
    y0 = np.exp(np.interp(np.log(x0), np.log(p), np.log(gw)))
    xs = np.geomspace(x0, x1, 2)
    ax.loglog(xs, y0 * (xs / x0) ** (-11.0 / 3.0), color="0.35", ls="-.", lw=1.2,
              label=r"$k^{-11/3}$ (Roper Pol et al.\ reported slope; reference)")

    ax.set_xlabel(r"$k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(k)/k$")
    ax.set_xlim(p[0], p[-1])
    ax.set_ylim(gw.max() * 1e-5, gw.max() * 3)
    ax.legend(loc="lower left", fontsize=9)
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
