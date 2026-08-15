#!/usr/bin/env python3
r"""Mach dependence of the band-split GW spectrum.

Repeats the band-split experiment of band_split_gw.py at a range of Mach numbers
and asks which features of each band's spectrum respond to M.

Four panels:
  (a) sub-inertial (Batchelor k^4) band alone, Omega_GW(p) at several M
  (b) inertial (Kolmogorov k^-5/3) band alone, same
  (c) peak position p_peak(M) for both bands, against the M^1 sweeping law
  (d) infrared amplitude Omega_GW(p=5e-3) for both bands, against the bare
      M^3 prefactor of Eq.(A:AppAdimless)

The temporal model is Kraichnan sweeping throughout, evaluated on the sound-cone
diagonal q = p.  With the temporal factor removed (q = 0) the kernel has NO M
dependence beyond the M^3 prefactor -- the exponential and the erfc both go to
unity -- so the no-correlation spectra of band_split_gw.py are M^3 times a single
universal shape and are not re-plotted here.

Produces images/band_split_mach.pdf.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)
from band_split_gw import H_band  # noqa: E402

warnings.filterwarnings("ignore")

R_FID, R_IR_FID = 1e4, 1e2
MS_CURVES = (0.03, 0.1, 0.3, 1.0, 3.0)          # curves drawn in (a),(b)
MS_SCALING = (0.03, 0.06, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0, 3.0)
P_GRID = np.geomspace(1e-3, 30.0, 44)
IR_PROBE = 5e-3
IR_FIT = (2e-3, 2e-2)

BANDS = ("ir", "inertial")
BAND_LABEL = {"ir": r"$k^{4}$ band", "inertial": r"$k^{-5/3}$ band"}
BAND_COLOR = {"ir": PALETTE[5], "inertial": PALETTE[6]}
_MSHADE = {m: a for m, a in zip(MS_CURVES, np.linspace(0.30, 1.0, len(MS_CURVES)))}


def omega(band: str, M: float) -> np.ndarray:
    return np.array([p ** 3 * H_band(p, p, M=M, R=R_FID, R_IR=R_IR_FID, band=band)
                     for p in P_GRID])


def ir_slope(y: np.ndarray) -> float:
    m = (P_GRID >= IR_FIT[0]) & (P_GRID <= IR_FIT[1]) & (y > 0) & np.isfinite(y)
    return float(np.polyfit(np.log(P_GRID[m]), np.log(y[m]), 1)[0])


def build() -> dict:
    data = {}
    for band in BANDS:
        for M in sorted(set(MS_CURVES) | set(MS_SCALING)):
            data[(band, M)] = omega(band, M)
            print(f"  {band:9s} M={M:<5g} slope={ir_slope(data[(band, M)]):+.3f}",
                  flush=True)
    return data


def main() -> None:
    data = build()
    apply_paper_style(grid=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), constrained_layout=True)
    # common y-range for (a),(b): every curve of both bands on scale
    _all = np.concatenate([data[(b, M)] for b in BANDS for M in MS_CURVES])
    _hi = np.nanmax(_all[_all > 0])
    _ylim = (10.0 ** np.floor(np.log10(_hi) - 16.0), _hi * 4.0)

    # (a),(b) spectra at several M, one panel per band
    for ax, band, tag in ((axes[0, 0], "ir", "a"), (axes[0, 1], "inertial", "b")):
        for M in MS_CURVES:
            y = data[(band, M)].copy()
            y[~(y > 0)] = np.nan
            ax.loglog(P_GRID, y, color=BAND_COLOR[band], lw=1.3,
                      alpha=_MSHADE[M], label=rf"$M={M:g}$")
        i0 = 4
        ref = data[(band, 1.0)]
        ax.loglog(P_GRID, ref[i0] * (P_GRID / P_GRID[i0]) ** 3, ls=":",
                  color="0.55", lw=0.8)
        ax.set_title(rf"({tag}) {BAND_LABEL[band]} alone", fontsize=10)
        ax.set_xlabel(r"$p=k/k_0$")
        ax.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^3H(p,p)$")
        ax.set_ylim(*_ylim)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        apply_max_ticks(ax, n=6)

    ms = np.array(MS_SCALING)
    # (c) peak position
    ax = axes[1, 0]
    for band in BANDS:
        pk = np.array([P_GRID[np.argmax(np.nan_to_num(data[(band, M)]))] for M in ms])
        ax.loglog(ms, pk, "o-", ms=3.5, lw=1.2, color=BAND_COLOR[band],
                  label=BAND_LABEL[band])
    ax.loglog(ms, 1.49 * ms, ls=":", color="0.5", lw=1.0)
    ax.text(1.6, 1.49 * 1.6 * 1.35, r"$1.49\,M$", fontsize=8, color="0.35",
            ha="center")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$p_{\rm peak}$")
    ax.set_title(r"(c) peak position", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    apply_max_ticks(ax, n=6)

    # (d) infrared amplitude
    ax = axes[1, 1]
    for band in BANDS:
        amp = np.array([np.interp(IR_PROBE, P_GRID, data[(band, M)]) for M in ms])
        ax.loglog(ms, amp, "o-", ms=3.5, lw=1.2, color=BAND_COLOR[band],
                  label=BAND_LABEL[band])
        print(f"  {band:9s}  A/M^3 spread over M: "
              f"{(amp / ms**3).max() / (amp / ms**3).min() - 1:.1%}")
    a0 = np.interp(IR_PROBE, P_GRID, data[("inertial", 1.0)])
    ax.loglog(ms, a0 * ms ** 3, ls=":", color="0.5", lw=1.0)
    ax.text(0.05, a0 * 0.05 ** 3 * 3, r"$M^{3}$", fontsize=8, color="0.35")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(rf"$\Omega_{{\rm GW}}(p={IR_PROBE:g})$")
    ax.set_title(r"(d) infrared amplitude", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    apply_max_ticks(ax, n=6)

    out = save_figure(fig, "band_split_mach")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
