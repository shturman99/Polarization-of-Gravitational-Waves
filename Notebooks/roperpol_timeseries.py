#!/usr/bin/env python3
r"""Time-series analysis of the Roper Pol et al. (2020) run ini2 GW/magnetic spectra.

Data provenance
---------------
Roper Pol, Mandal, Brandenburg, Kahniashvili & Kosowsky, "Numerical simulations
of gravitational waves from early-universe turbulence", Phys. Rev. D 102, 083512
(2020), arXiv:1903.08585.  Public data: Zenodo DOI 10.5281/zenodo.3692072
(GW.tar, 184.7 MB) and the analysis repository
https://github.com/AlbertoRoper/GW_turbulence (dir PRD_1903_08585/M1152e_exp6k4/).

Run ini2 = M1152e_exp6k4: an initially imposed (then freely decaying) helical
magnetic field, the run shown in the paper's Fig. 1.  The Pencil Code
power-spectrum files are stacks of the spectrum saved at many times:
  power_krms.dat  -- the k-grid (shell wavenumbers), 576 shells
  power_mag.dat   -- magnetic energy spectrum E_M(k, t)
  power_GWs.dat   -- GW energy spectrum E_GW(k, t) = Omega_GW(k,t)/k
Each file: [time][576 values][time][576 values]...  (ASCII, %.2E, 8/line).

This script downloads those three files (a few MB each, cached locally), and
shows the TIME EVOLUTION of the spectra -- not only the late-time state that
Fig. 1 of the paper displays.  Three physics points come straight out of it:

1. The causal k^3 is a TRANSIENT.  At source switch-on (t~1.0) the GW infrared
   is Omega_GW ~ k^3 (E_GW ~ k^2); within dt~0.02 it saturates to the flat
   plateau Omega_GW ~ k^1 (E_GW ~ k^0) shown in Fig. 1.  So the analytic k^3 and
   the simulated k^1 are the same spectrum at different times, not a
   contradiction (cf. Roper Pol's own statement that the k^2->flat transition is
   "much shorter than the time it takes for the GW spectrum to become
   stationary").
2. STEADY OSCILLATORY STATE.  E_GW,tot(t) rises while the source acts, then
   fluctuates about a constant mean (the freely ringing GW field), even as the
   magnetic energy keeps decaying -- exactly the "fluctuate around a steady
   state" of the Fig. 1 caption.
3. INVERSE TRANSFER.  The magnetic peak k0(t) migrates to smaller k with time
   (helical inverse cascade), sustaining a coherent large-scale field.

Figure: images/roperpol_timeseries.pdf
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz

_BASE = ("https://raw.githubusercontent.com/AlbertoRoper/GW_turbulence/"
         "master/PRD_1903_08585/M1152e_exp6k4/data")
_CACHE = Path(__file__).resolve().parent / "roperpol_ini2_data"
_FILES = ("power_krms.dat", "power_mag.dat", "power_GWs.dat")


def _fetch():
    _CACHE.mkdir(exist_ok=True)
    for f in _FILES:
        dest = _CACHE / f
        if not dest.exists():
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(f"{_BASE}/{f}", dest)
    return _CACHE


def _toks(path):
    return np.array(Path(path).read_text().split(), dtype=float)


def load():
    d = _fetch()
    k = _toks(d / "power_krms.dat")
    nk = len(k)

    def spec(name):
        t = _toks(d / name)
        block = nk + 1
        nt = len(t) // block
        a = t[:nt * block].reshape(nt, block)
        return a[:, 0], a[:, 1:]

    tM, EM = spec("power_mag.dat")
    tG, EG = spec("power_GWs.dat")
    return k, tM, EM, tG, EG


def _ir_slope(kk, E, k0):
    m = (kk >= 1.2) & (kk <= 0.7 * k0) & (E > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(kk[m]), np.log(E[m]), 1)[0])


def main():
    apply_paper_style()
    k, tM, EM, tG, EG = load()
    kp = k > 0
    kk = k[kp]
    EMk, EGk = EM[:, kp], EG[:, kp]

    EMtot = _trapz(EMk, kk, axis=1)
    EGtot = _trapz(EGk, kk, axis=1)
    k0t = kk[np.argmax(EMk, axis=1)]
    gw_ir = np.array([_ir_slope(kk, EGk[i], k0t[i]) for i in range(len(tG))])

    late = tG > 1.15
    print(f"steady state (t>1.15): E_GW,tot = {EGtot[late].mean():.3e} "
          f"+/- {EGtot[late].std() / EGtot[late].mean():.1%}")
    print(f"E_M,tot decays {EMtot[late][0]:.2e} -> {EMtot[late][-1]:.2e}")
    print(f"efficiency E_GW/E_M^2 (late) = {(EGtot / EMtot**2)[late].mean():.2e}")
    print(f"k0 inverse transfer: {k0t[0]:.1f} -> {k0t[-1]:.1f}")

    fig, (ax1, ax2) = plt_subplots()

    # (a) GW spectra at several times: causal transient -> flat plateau.
    # Axes span the FULL data: every k shell (to the Nyquist k~575) and the whole
    # dynamic range from the peak down to the ultraviolet tail / noise floor.
    snaps = [(np.argmin(np.abs(tG - t)), t) for t in (1.00, 1.02, 1.06, 1.20, 1.40)]
    for j, (i, t) in enumerate(snaps):
        E = EGk[i].astype(float).copy()
        E[E <= 0] = np.nan
        ax1.loglog(kk, E, color=PALETTE[j + 1], lw=1.1, label=rf"$t={tG[i]:.2f}$")
    pos = EGk[EGk > 0]
    kmax = kk.max()
    ax1.set_xlim(1.2, 1.15 * kmax)
    ax1.set_ylim(pos.min() / 3, pos.max() * 3)
    # IR guide lines (below the peak, k ~ 1.3-8): causal k^2 vs flat
    kref = np.array([1.3, 8.0])
    ax1.loglog(kref, 8e-16 * (kref / kref[0])**2, color=PALETTE[0], ls=":", lw=1.1)
    ax1.text(2.7, 2.0e-14, r"$E_{\rm GW}\!\sim\!k^2$ ($\Omega_{\rm GW}\!\sim\!k^3$)",
             fontsize=6.5, color=PALETTE[0], rotation=32)
    ax1.loglog(kref, [1.9e-13, 1.9e-13], color=PALETTE[0], ls="--", lw=1.1)
    ax1.text(1.4, 2.6e-13, r"flat ($\Omega_{\rm GW}\!\sim\!k^1$)", fontsize=6.5, color=PALETTE[0])
    # UV inertial-range guide: Omega_GW ~ k^-11/3 => E_GW ~ k^-14/3
    kuv = np.array([20.0, kmax])
    ax1.loglog(kuv, 6e-14 * (kuv / kuv[0])**(-14.0 / 3.0), color=PALETTE[0], ls="-.", lw=1.0)
    ax1.text(45, 3e-16, r"$\Omega_{\rm GW}\!\sim\!k^{-11/3}$", fontsize=6.5, color=PALETTE[0])
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$E_{\rm GW}(k)=\Omega_{\rm GW}(k)/k$")
    ax1.set_title(r"(a) GW spectrum builds up", fontsize=10)
    ax1.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2, columnspacing=1.0)
    apply_max_ticks(ax1)

    # (b) energies + IR slope vs time (steady oscillatory state)
    ax2.semilogy(tG, EGtot, color=PALETTE[5], lw=1.4, label=r"$E_{\rm GW,tot}$")
    ax2.semilogy(tM, EMtot, color=PALETTE[6], lw=1.4, ls="--", label=r"$E_{\rm M,tot}$")
    ax2.axvspan(1.15, tG[-1], color="0.5", alpha=0.10, lw=0)
    ax2.text(1.27, 3e-3, "steady\noscillatory", fontsize=7, color="0.35", ha="center")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"total energy")
    ax2.set_title(r"(b) steady oscillatory state", fontsize=10)
    ax2.legend(fontsize=7, frameon=False, loc="center right")
    apply_max_ticks(ax2)

    axr = ax2.twinx()
    axr.plot(tG, np.where(np.isfinite(gw_ir), gw_ir + 1.0, np.nan),
             color=PALETTE[3], lw=1.0, alpha=0.8)
    axr.axhline(3.0, color=PALETTE[3], ls=":", lw=0.8)
    axr.axhline(1.0, color=PALETTE[3], ls=":", lw=0.8)
    axr.set_ylabel(r"GW IR slope $\;\Omega_{\rm GW}\!\sim\!k^{s}$", color=PALETTE[3], fontsize=9)
    axr.set_ylim(-0.5, 4.0)
    axr.tick_params(axis="y", labelcolor=PALETTE[3])
    axr.text(1.005, 3.15, r"causal $k^3$", fontsize=6.5, color=PALETTE[3])
    axr.text(1.25, 1.15, r"flat $k^1$", fontsize=6.5, color=PALETTE[3])

    out = save_figure(fig, "roperpol_timeseries")
    print(f"saved {out}")


def plt_subplots():
    import matplotlib.pyplot as plt
    return plt.subplots(1, 2, figsize=(FIGSIZES["large"][0], FIGSIZES["small"][1]),
                        constrained_layout=True)


if __name__ == "__main__":
    main()
