#!/usr/bin/env python3
r"""Why GW production stops even though the magnetic source stays active (ini3).

The source is NOT off: at large scales the helical inverse cascade actually
*grows* the magnetic energy.  Yet no extra GW appear.  The reason is a frequency
mismatch, not an amplitude one.

GW at comoving wavenumber k oscillate at omega_GW = k (they live on the light
cone).  A GW mode gains net energy only from the source's Fourier component at
that SAME frequency:  E_k(infty) = |int e^{i k t'} S(t') dt'|^2.  But the
turbulent source decorrelates at the sweeping/eddy rate
    omega_source ~ eta_k = (M/sqrt(2pi)) k0^{1/3} k^{2/3},
which for subsonic turbulence is ~ M k << k.  The source is therefore SLOW
compared with the GW oscillation (here by a factor ~1/M ~ 30): over each GW
period it does positive work for half a cycle and negative for the other half,
dE_k/dt = 2 S(t) hdot_k(t) averaging to ~0.  Only the sudden turn-on -- broadband
in time, i.e. containing omega = k -- radiates.  This is the aeroacoustic
(Lighthill) suppression of slow sources, and the origin of the M-dependence.

(a) The smoking gun: at k ~ 4 (inverse-cascade fed) the magnetic energy climbs
    several-fold while the GW energy there is flat -- an active, growing source
    that produces no GW.
(b) The frequency gap: omega_GW = k versus omega_source = eta_k; the shaded gap
    (factor ~ M) is why the source cannot resonantly drive the waves.

Figure: images/roperpol_adiabatic_source.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roperpol_all_runs import load, mach_series  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE, apply_max_ticks, apply_paper_style, save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz


def main():
    apply_paper_style()
    kk, tG, EGk, ESk = load("ini3")
    tt, ur = mach_series("ini3")
    M = np.interp(tG, tt, ur)
    Mbar = float(np.median(M[tG > 1.2]))
    k0 = np.median(kk[np.argmax(ESk[tG > 1.2], axis=1)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)

    # (a) an active/growing source that produces no GW: the k~4 (inverse-cascade) mode
    j = np.argmin(np.abs(kk - 4.0))
    k = kk[j]
    em = ESk[:, j] / ESk[0, j]
    eg = EGk[:, j] / np.median(EGk[tG > 1.2, j])
    ax1.plot(tG, em, color=PALETTE[6], lw=1.6, label=r"magnetic source $E_{\rm M}(k,t)$")
    ax1.plot(tG, eg, color=PALETTE[5], lw=1.0, alpha=0.7, label=r"GW $E_{\rm GW}(k,t)$")
    # running median of the (ringing) GW to show it is flat
    win = 40
    egm = np.array([np.median(eg[max(0, i - win):i + win + 1]) for i in range(len(eg))])
    ax1.plot(tG, egm, color=PALETTE[5], lw=2.0)
    ax1.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax1.set_xlabel(r"time $t$")
    ax1.set_ylabel(r"normalised energy at $k\simeq4$")
    ax1.set_title(r"(a) source grows, GW indifferent", fontsize=9.5)
    ax1.set_ylim(0, max(em.max(), 2.2) * 1.05)
    ax1.text(tG[len(tG) // 2], em[len(tG) // 2] * 1.03,
             "inverse cascade\nfeeds the source", fontsize=7, color=PALETTE[6])
    ax1.legend(fontsize=7, frameon=False, loc="center right")
    apply_max_ticks(ax1)

    # (b) the frequency gap: omega_GW = k vs omega_source = eta_k ~ M k0^{1/3} k^{2/3}
    kx = np.geomspace(2, 200, 50)
    wgw = kx
    wsrc = (Mbar / np.sqrt(2 * np.pi)) * k0 ** (1 / 3) * kx ** (2 / 3)
    ax2.loglog(kx, wgw, color=PALETTE[5], lw=1.8, label=r"$\omega_{\rm GW}=k$ (light cone)")
    ax2.loglog(kx, wsrc, color=PALETTE[6], lw=1.8,
               label=r"$\omega_{\rm source}=\eta_k\sim M k^{2/3}k_0^{1/3}$")
    ax2.fill_between(kx, wsrc, wgw, color="0.7", alpha=0.35)
    ax2.text(20, 20 * 0.28, r"resonance gap $\sim M$", fontsize=7.5, color="0.3",
             rotation=32, ha="center")
    ax2.text(20, wsrc[np.argmin(np.abs(kx - 20))] * 0.3,
             "no source power\nat $\\omega=k$", fontsize=6.8, color=PALETTE[6], ha="center")
    ax2.set_xlabel(r"wavenumber $k$")
    ax2.set_ylabel(r"frequency $\omega$")
    ax2.set_title(rf"(b) source is $\sim1/M\simeq{1/Mbar:.0f}\times$ too slow", fontsize=9.5)
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    apply_max_ticks(ax2)

    out = save_figure(fig, "roperpol_adiabatic_source")
    print(f"saved {out}")
    print(f"subsonic Mach M~{Mbar:.3f}, k0~{k0:.0f}: omega_source/omega_GW ~ M ~ {Mbar:.3f} "
          f"(source ~{1/Mbar:.0f}x slower than the GW oscillation)")
    m = tG > 1.2
    print(f"k=4 mode: source E_M x{ESk[m, j][-1] / ESk[m, j][0]:.1f} over t>1.2, "
          f"GW E_GW net {100 * (EGk[m, j][-1] - EGk[m, j][0]) / EGk[m, j].mean():+.0f}%")


if __name__ == "__main__":
    main()
