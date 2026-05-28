#!/usr/bin/env python3
r"""Driving-dependence of the GW spectral peak in Roper Pol et al. (2020).
RP2020 Fig. 1 shows ONLY run ini2 (vortical) -- already digitized. Acoustic runs
(ac1-3) appear only in Figs 5-7 as Omega(t) / frequency-space strain (not k_*-norm,
LISA overlaid) -> NOT cleanly digitizable. So this is a THEORY/REFERENCE figure:
project's own kernels (computed) + the one real ini2 curve + the two REPORTED peak
positions as labelled guides (tagged 'reported, not data'). No data fabricated.
Reported (quoted, arXiv:1903.08585): vortical/magnetic peak ~2 k_*; acoustic ~1 k_*."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT/"src"))
if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz
import roperpol_data
from _fullspectrum_kernel import omega_gw_over_k as stat_over_k
from fullspatial_decay import H_decay_fast
from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

REPORTED_PEAK_VORTICAL = 2.0   # ini2 (magnetic/vortical): k_GW ~ 2 k_*  (reported)
REPORTED_PEAK_ACOUSTIC = 1.0   # ac1-3 (acoustic): "near k_*"           (reported)
M, R, R_IR = 0.5, 1.0e4, 100.0

def _load(name):
    d = np.loadtxt(ROOT/"Notebooks"/f"roperpol_fig1_{name}.csv", delimiter=",", skiprows=1)
    return d[np.argsort(d[:, 0])].T

def _peak(p, y):
    ly = np.log(p*y); i = int(np.argmax(ly)); il, ir = max(i-2,0), min(i+3,len(p))
    c = np.polyfit(np.log(p[il:ir]), ly[il:ir], 2); return float(np.exp(-c[1]/(2*c[0])))

def main(name="roperpol_multirun"):
    apply_paper_style(); import matplotlib.pyplot as plt
    k0 = roperpol_data.k0(); kg, og = _load("gw"); pg = kg/k0
    p = np.geomspace(0.05, 14.0, 90)
    stat = np.array([stat_over_k(pp, M=M, R=R, R_IR=R_IR, ir="batchelor") for pp in p])
    dec  = np.array([pp**2*H_decay_fast(pp, pp, M=M, R=R) for pp in p])
    stat_n, dec_n, data_n = stat/(p*stat).max(), dec/(p*dec).max(), og/(pg*og).max()
    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)
    ax.axvline(REPORTED_PEAK_ACOUSTIC, color=PALETTE[6], lw=1.6, ls=(0,(5,2)),
               label=r"acoustic peak $\simeq k_*$ (reported, not data)")
    ax.axvline(REPORTED_PEAK_VORTICAL, color=PALETTE[5], lw=1.6, ls=(0,(5,2)),
               label=r"vortical peak $\simeq 2k_*$ (reported, not data)")
    ax.loglog(pg, data_n, "-", color="0.0", lw=2.1,
              label=r"$\Omega_{\rm GW}/k$, run ini2 (digitized data, vortical)")
    ax.loglog(p, stat_n, "-", color=PALETTE[2], lw=1.8,
              label=rf"stationary kernel, $M={M}$ (peak ${_peak(p,stat):.2f}\,k_0$)")
    ax.loglog(p, dec_n, "-", color=PALETTE[1], lw=1.8,
              label=rf"decaying kernel, $M={M}$ (peak ${_peak(p,dec):.2f}\,k_0$)")
    ax.set_xlabel(r"$k/k_*$"); ax.set_ylabel(r"$\Omega_{\rm GW}(k)/k$  (peak-normalised)")
    ax.set_xlim(0.06, 14.0); ax.set_ylim(3e-5, 3.0)
    ax.legend(loc="lower center", fontsize=6.6, framealpha=0.93); apply_max_ticks(ax)
    out = save_figure(fig, name); plt.close(fig); print(f"saved {out}"); return out

if __name__ == "__main__":
    main()
