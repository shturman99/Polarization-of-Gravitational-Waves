#!/usr/bin/env python3
r"""Compare our analytic GW kernels to the DIGITIZED Roper Pol et al. (2020) Fig. 1.

Data: Notebooks/roperpol_fig1_{fluid,gw}.csv -- real pixel digitization of their
Omega_M(k)/k and Omega_GW(k)/k (see digitize_roperpol.py).  All data-derived numbers are
COMPUTED in roperpol_data.py (not hardcoded):
  fluid peak k0 ~ 673; fluid k^4 -> k^-5/3; GW IR slope ~ +1.3 (Omega_GW) i.e. flat
  Omega_GW/k; GW UV ~ k^-3.5 (~k^-11/3); energy fraction Omega_M = int(Omega_M/k)dk ~ 1.8e-3
  (subsonic, effective M ~ 0.05); GW peak (Omega_GW) ~ 1.84 k0.

We overlay our analytic Omega_GW(k)/k = p^2 H(p,p), p=k/k0, for BOTH temporal models
at a subsonic Mach number:
  - stationary  (full input spectrum, _fullspectrum_kernel.H_full, IR band + Kolmogorov)
  - decaying    (BK2016 power law, fullspatial_decay.H_decay_fast)
Each analytic curve is normalised by the ENERGY FRACTION: scaled so its total GW energy
equals the simulated Omega_GW = (H_*/k0) Omega_M^2, i.e. the absolute level is set by the
measured Omega_M, not a free fit.  Only the SHAPE (slopes, peak) is then a prediction.

Honest result: the data GW UV (~k^-11/3) lies BETWEEN the too-steep stationary
(Gaussian sweeping) and the too-shallow decaying (heavy power-law tail) kernels at
subsonic M; and the data GW IR is flat (impulsive/finite-duration source) whereas both
quasi-stationary kernels give the rising causal k^3 -- the finite-duration (self-similar)
treatment is what would flatten it.
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
from _fullspectrum_kernel import omega_gw_over_k as stat_over_k  # noqa: E402
from fullspatial_decay import H_decay_fast  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

K0 = roperpol_data.k0()   # fluid spectral peak (computed), sets k = K0 * p
M = 0.5             # subsonic Mach for the analytic shapes
R = 1.0e4
R_IR = 100.0


def _load(name):
    d = np.loadtxt(ROOT / "Notebooks" / f"roperpol_fig1_{name}.csv",
                   delimiter=",", skiprows=1)
    return d[np.argsort(d[:, 0])].T  # k, Omega/k


def _fit(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def dec_over_k(p, M=M, R=R):
    return p ** 2 * H_decay_fast(p, p, M=M, R=R)


def _energy_normalise(k, ok, target_energy):
    """Scale Omega/k so that int (Omega/k) dk = target_energy (the measured Omega_GW)."""
    return ok * target_energy / np.trapezoid(ok, k)


def main(name="roperpol_comparison"):
    apply_paper_style()
    kf, of = _load("fluid")
    kg, og = _load("gw")
    OmM = np.trapezoid(of, kf)          # turbulent energy fraction
    OmGW = np.trapezoid(og, kg)         # simulated GW energy = (H_*/k0) OmM^2

    # analytic shapes over the data k-range, p = k/k0
    p = np.geomspace(kg.min() / K0, kg.max() / K0, 55)
    k = K0 * p
    stat = np.array([stat_over_k(pp, M=M, R=R, R_IR=R_IR, ir="batchelor") for pp in p])
    dec = np.array([dec_over_k(pp) for pp in p])
    stat = _energy_normalise(k, stat, OmGW)   # absolute level from Omega_M (energy fraction)
    dec = _energy_normalise(k, dec, OmGW)

    fig, ax = plt = None, None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)

    ax.loglog(kf, of, "-", color="0.0", lw=2.0, label=r"$\Omega_{\rm M}/k$ (Roper Pol, digitized)")
    ax.loglog(kg, og, "-", color="0.45", lw=2.0, label=r"$\Omega_{\rm GW}/k$ (Roper Pol, digitized)")
    ax.loglog(k, stat, "-", color=PALETTE[0], lw=1.9,
              label=rf"stationary, $M={M}$ (UV $k^{{{_fit(k, stat, 3e3, 3e4):.1f}}}$)")
    ax.loglog(k, dec, "-", color=PALETTE[1], lw=1.9,
              label=rf"decaying, $M={M}$ (UV $k^{{{_fit(k, dec, 3e3, 3e4):.1f}}}$)")

    # reference slope guides anchored on the GW data (as in their figure)
    xu = np.geomspace(3e3, 3e4, 2)
    yu = 1.8 * np.exp(np.interp(np.log(xu[0]), np.log(kg), np.log(og)))
    ax.loglog(xu, yu * (xu / xu[0]) ** (-11.0 / 3.0), ":", color="0.3", lw=1.3,
              label=r"$k^{-11/3}$ (reported)")
    ax.axvline(K0, color="0.55", lw=0.9, ls=":")          # source scale k0
    ax.text(K0 * 1.05, 3e-6, r"$k_0$", color="0.4", fontsize=8)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(k)/k$ and $\Omega_{\rm M}(k)/k$")
    ax.set_ylim(1e-19, 1e-5)
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.62), fontsize=7.0,
              framealpha=0.92)  # empty mid-band between fluid and GW curves
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    plt.close(fig)

    print(f"saved {out}")
    print(f"  data: Omega_M={OmM:.2e}, Omega_GW={OmGW:.2e}, k0={K0:.0f}")
    print(f"  data GW: IR slope {_fit(kg, og, 150, 600):+.2f} (flat), "
          f"UV slope {_fit(kg, og, 3e3, 3e4):+.2f}")
    print(f"  stationary M={M}: IR {_fit(k, stat, 150, 600):+.2f}, UV {_fit(k, stat, 3e3, 3e4):+.2f}")
    print(f"  decaying   M={M}: IR {_fit(k, dec, 150, 600):+.2f}, UV {_fit(k, dec, 3e3, 3e4):+.2f}")
    return out


if __name__ == "__main__":
    main()
