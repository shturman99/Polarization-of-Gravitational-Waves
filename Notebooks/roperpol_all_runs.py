#!/usr/bin/env python3
r"""Time-series of GW and source spectra for ALL 12 runs of Roper Pol et al. (2020).

Companion to roperpol_timeseries.py (which does the single detailed ini2 figure).
This module downloads the public Pencil Code power spectra for every run in
Table 1 of Roper Pol, Mandal, Brandenburg, Kahniashvili & Kosowsky,
Phys. Rev. D 102, 083512 (2020), arXiv:1903.08585 (Zenodo 10.5281/zenodo.3692072;
GitHub AlbertoRoper/GW_turbulence/PRD_1903_08585), and produces:

  * roperpol_spectra_gallery : a 4x3 grid, one panel per run, showing the GW
    energy spectrum E_GW(k)=Omega_GW/k at several times -- the spectral shape
    building up and (for the decaying runs) saturating.
  * roperpol_energy_evolution: total GW energy and total source energy versus
    time for all 12 runs, coloured by class -- decaying magnetic (ini) rise to a
    steady oscillatory state while the source decays; forced runs (helical,
    nonhelical, acoustic) keep growing while continuously driven.

The four run classes:
  ini1-3  imposed, then freely DECAYING magnetic field (source = magnetic)
  hel1-4  FORCED helical turbulence           (source = magnetic)
  noh1-2  FORCED nonhelical magnetic          (source = magnetic)
  ac1-3   FORCED acoustic / compressive       (source = kinetic; no B)

Data (a few tens of MB total) is cached under Notebooks/roperpol_runs_data/ and
gitignored; the script re-fetches anything missing.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz

_BASE = ("https://raw.githubusercontent.com/AlbertoRoper/GW_turbulence/"
         "master/PRD_1903_08585")
_CACHE = Path(__file__).resolve().parent / "roperpol_runs_data"

# label -> (directory, class, source-spectrum file)
RUNS: dict[str, tuple[str, str, str]] = {
    "ini1": ("M1152e_exp6k4_M4b",           "decaying", "power_mag.dat"),
    "ini2": ("M1152e_exp6k4",               "decaying", "power_mag.dat"),
    "ini3": ("M1152e_exp6k4_k60b",          "decaying", "power_mag.dat"),
    "hel1": ("F1152d2_sig1_t11_M2c_double", "helical",  "power_mag.dat"),
    "hel2": ("F1152a_sig1_t11d_double",     "helical",  "power_mag.dat"),
    "hel3": ("F1152a_sig1",                 "helical",  "power_mag.dat"),
    "hel4": ("F1152a_k10_sig1",             "helical",  "power_mag.dat"),
    "noh1": ("F1152b_sig0_t11_M4",          "nonhelical", "power_mag.dat"),
    "noh2": ("F1152a_sig0_t11b",            "nonhelical", "power_mag.dat"),
    "ac1":  ("E1152e_t11_M4d_double",       "acoustic", "power_kin.dat"),
    "ac2":  ("E1152e_t11_M4a_double",       "acoustic", "power_kin.dat"),
    "ac3":  ("E1152e_t11_M4e_double",       "acoustic", "power_kin.dat"),
}

_CLASS_COLOR = {
    "decaying":   PALETTE[6],   # vermilion
    "helical":    PALETTE[5],   # blue
    "nonhelical": PALETTE[3],   # bluish green
    "acoustic":   PALETTE[7],   # reddish purple (distinct from vermilion)
}


def _fetch(run: str) -> Path:
    d, _, src = RUNS[run]
    out = _CACHE / run
    out.mkdir(parents=True, exist_ok=True)
    for f in ("power_krms.dat", "power_GWs.dat", src):
        dest = out / f
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {run}/{f} ...")
        urllib.request.urlretrieve(f"{_BASE}/{d}/data/{f}", dest)
    return out


def _toks(p: Path) -> np.ndarray:
    return np.array(p.read_text().split(), dtype=float)


def _parse(path: Path, nk: int):
    t = _toks(path)
    block = nk + 1
    nt = len(t) // block
    a = t[:nt * block].reshape(nt, block)
    return a[:, 0], a[:, 1:]


#: power_GWs.dat stores the RAW shell sums of |gT|^2+|gX|^2 (Pencil's powerGWs()
#: runs with lhalf_factor_in_GW=F, so sum_k = gg2m exactly).  The EEGW diagnostic
#: instead applies EGWpref, which for cstress_prefactor='6' is 1/6.  Verified: the
#: published time series satisfy sum_k(power_GWs) = gg2m to 0.7% in all 12 runs.
EGW_PREFACTOR = 1.0 / 6.0


def load(run: str):
    """Return k (>0), times, E_GW[nt,nk], E_src[nt,nk] with all-zero snapshots dropped.

    NORMALIZATION (verified against the published time series, see EGW_PREFACTOR):
    power_mag.dat / power_kin.dat are true energy spectra -- sum_k = EEM (or EEK)
    to 0.3%.  power_GWs.dat is NOT: it carries neither the 1/2 nor the EGWpref=1/6
    that the EEGW diagnostic applies, so multiply by EGW_PREFACTOR to get Omega_GW.
    The factor is a constant and identical for all 12 runs, so spectral SHAPES
    (slopes, peak positions) and cross-run comparisons are unaffected; only
    absolute GW amplitudes and efficiencies need it.

    Note this is a *different* issue from the ini1-3 time-series bug: those three
    runs were integrated with the older EGWpref=1/(32 pi), so their published EEGW
    column is low by 32 pi/6 = 16.755 (Roper Pol's initialize_PRD_2020.py corrects
    it).  The *spectra* are raw gg^2 and are unaffected.
    """
    d = _fetch(run)
    k = _toks(d / "power_krms.dat")
    nk = len(k)
    tG, EG = _parse(d / "power_GWs.dat", nk)
    tS, ES = _parse(d / RUNS[run][2], nk)
    n = min(len(tG), len(tS))
    tG, EG, ES = tG[:n], EG[:n], ES[:n]
    kp = k > 0
    kk, EGk, ESk = k[kp], EG[:, kp], ES[:, kp]
    # Drop corrupt/near-zero final blocks that Pencil sometimes writes: a snapshot
    # is kept only if both its GW and source totals are a nonneglible fraction of
    # the run median (a hard >0 test misses blocks that are tiny but not exactly 0).
    eg, es = EGk.sum(1), ESk.sum(1)
    good = (eg > 1e-4 * np.median(eg[eg > 0])) & (es > 1e-4 * np.median(es[es > 0]))
    return kk, tG[good], EGk[good], ESk[good]


def _ts(run: str):
    """Return (array, column-name -> index) for the run's time_series.dat.

    The column layout is NOT the same in every run, so it must be read from
    legend.dat rather than hard-coded: the acoustic runs are purely hydrodynamic
    (lmagnetic=F) and therefore have no EEM/brms/bmax columns, which shifts every
    later column left by three.  Concretely
      ini*/hel*/noh*: it t dt EEK EEM EEGW hrms urms brms ...  -> urms at index 7
      ac1-3:          it t dt EEK     EEGW hrms urms umax ...  -> urms at index 6
    (index 7 in an ac run is umax, not urms).
    """
    d = RUNS[run][0]
    out = _CACHE / run
    out.mkdir(parents=True, exist_ok=True)
    for f in ("time_series.dat", "legend.dat"):
        dest = out / f
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"downloading {run}/{f} ...")
            urllib.request.urlretrieve(f"{_BASE}/{d}/data/{f}", dest)
    cols = (out / "legend.dat").read_text().replace("-", " ").split()
    return np.loadtxt(out / "time_series.dat", comments="#"), {c: i for i, c in enumerate(cols)}


def mach_series(run: str):
    """Return (t, urms) from the run's time_series.dat.

    In the simulation's units c=1, so urms is the Mach number M=u_0/c directly
    (the acoustic Mach is sqrt(3) larger, c_s=c/sqrt(3)).
    """
    a, idx = _ts(run)
    return a[:, idx["t"]], a[:, idx["urms"]]


def gw_peak_track(run: str):
    """(t, M(t), k_GWpeak/k0) for a run: the GW peak (Omega_GW=k E_GW convention)
    in units of the instantaneous magnetic peak k0(t), and M(t)=u_rms(t)."""
    kk, tG, EGk, ESk = load(run)
    k0 = kk[np.argmax(ESk, axis=1)]
    OmGW = kk * EGk                      # Omega_GW = k E_GW
    kpk = kk[np.argmax(OmGW, axis=1)]
    tt, ur = mach_series(run)
    return tG, np.interp(tG, tt, ur), kpk / k0


def fig_gallery():
    apply_paper_style(grid=False)
    fig, axes = plt.subplots(4, 3, figsize=(8.0, 9.0), constrained_layout=True)
    for ax, run in zip(axes.flat, RUNS):
        kk, tG, EGk, ESk = load(run)
        cls = RUNS[run][1]
        idx = np.linspace(0, len(tG) - 1, 4).astype(int)
        shades = np.linspace(0.35, 0.9, len(idx))
        base = _CLASS_COLOR[cls]
        for j, i in enumerate(idx):
            E = EGk[i].copy()
            E[E <= 0] = np.nan
            ax.loglog(kk, E, color=base, alpha=shades[j], lw=1.1,
                      label=rf"$t={tG[i]:.2f}$")
        k0 = kk[np.argmax(ESk[-1])]
        ax.axvline(k0, color="0.6", lw=0.7, ls=":")
        ax.set_xlim(1.2, 200)
        peak = np.nanmax(EGk[EGk > 0])
        ax.set_ylim(peak / 3e8, peak * 4)      # focus on the physical range, not the noise floor
        ax.set_title(rf"\texttt{{{run}}} ({cls})", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False, loc="lower left", handlelength=1.0)
        ax.tick_params(labelsize=7)
        apply_max_ticks(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$k$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E_{\rm GW}(k)=\Omega_{\rm GW}/k$", fontsize=8)
    out = save_figure(fig, "roperpol_spectra_gallery")
    print(f"saved {out}")


# Common target times: all 12 runs overlap only on t in [1.0, 1.17] (the shortest
# runs, hel1/ac2, end at 1.17), so equal-time snapshots must be taken there.
_COMMON_TIMES = (1.02, 1.06, 1.10, 1.15)


def fig_gallery_fixed(which="both", times=_COMMON_TIMES):
    """4x3 gallery with SHARED (identical) x and y axes across all panels AND the
    spectra taken at the SAME absolute times in every run (closest available
    snapshot to each target in `times`), so panels are comparable both in scale
    and in evolutionary stage.

    which="gw"     -> GW spectrum E_GW(k)=Omega_GW/k only
    which="source" -> source spectrum only (magnetic for M/F runs, kinetic for ac)
    which="both"   -> source (grey) AND GW (class colour) in every panel, on ONE
                      shared y-range covering both bands for all 12 runs.

    The "both" y-range (1e-16, 1e-1), 15 decades, is set empirically from the
    plotted snapshots: the largest source value over all runs/times is
    1.15e-2 (ini1) and the *smallest* GW peak is 4.1e-12 (hel4), so the window
    clears every source peak by a decade and still leaves every GW peak >=4.6
    decades above the floor.  Going deeper only exposes the k^-large numerical
    noise floor of the forced runs (which reaches 1e-43) and would squash both
    physical bands; Roper Pol et al. (2020) Fig. 1 spans a comparable ~14 decades.
    """
    apply_paper_style(grid=False)
    both = which == "both"
    if which == "gw":
        ylab = r"$E_{\rm GW}(k)=\Omega_{\rm GW}/k$"
        ylim = (1e-16, 1e-7)
        name, ttl = "roperpol_gw_gallery_fixed", "GW spectra"
    elif which == "source":
        ylab = r"$\Omega_{\rm src}/k$  (magnetic / kinetic)"
        ylim = (1e-9, 1e-1)
        name, ttl = "roperpol_source_gallery_fixed", "source (magnetic / kinetic) spectra"
    else:
        ylab = r"$\Omega/k$  (source, GW)"
        ylim = (1e-16, 1e-1)
        name = "roperpol_both_gallery_fixed"
        ttl = "source (grey) and GW (coloured) spectra"

    fig, axes = plt.subplots(4, 3, figsize=(8.0, 9.6), sharex=True, sharey=True,
                             constrained_layout=True)
    shades = np.linspace(0.30, 0.95, len(times))
    lo, hi = np.log10(ylim)
    kmax = 575.0
    max_dt = 0.0
    for ax, run in zip(axes.flat, RUNS):
        kk, tG, EGk, ESk = load(run)
        cls = RUNS[run][1]
        base = _CLASS_COLOR[cls]
        # in the combined panel the source is grey so the GW band keeps the class
        # colour; in the source-only gallery the source itself carries the colour.
        src_c = "0.45" if both else base
        gwpeak = 0.0
        for j, ttarget in enumerate(times):
            i = int(np.argmin(np.abs(tG - ttarget)))       # closest available snapshot
            max_dt = max(max_dt, abs(tG[i] - ttarget))
            if which != "gw":
                ys = ESk[i].astype(float).copy()
                ys[ys <= 0] = np.nan
                ax.loglog(kk, ys, color=src_c, alpha=shades[j], lw=1.0,
                          label=None if both else rf"$t={ttarget:.2f}$")
            if which != "source":
                yg = EGk[i].astype(float).copy()
                yg[yg <= 0] = np.nan
                gwpeak = max(gwpeak, float(np.nanmax(yg)))
                ax.loglog(kk, yg, color=base, alpha=shades[j], lw=1.0,
                          label=rf"$t={ttarget:.2f}$")
        k0 = kk[np.argmax(ESk[-1])]
        ax.axvline(k0, color="0.7", lw=0.6, ls=":")
        if both:
            # in-panel band labels: source sits at the top, GW just under its peak
            src = "mag" if RUNS[run][2] == "power_mag.dat" else "kin"
            bbox = dict(fc="white", ec="none", alpha=0.75, pad=1.0)
            ax.text(0.96, 0.97, rf"$\Omega_{{\rm {src}}}/k$", transform=ax.transAxes,
                    fontsize=6, color="0.35", ha="right", va="top", bbox=bbox)
            frac = (np.log10(gwpeak) - lo) / (hi - lo)
            ax.text(0.96, min(max(frac + 0.04, 0.06), 0.88),
                    r"$\Omega_{\rm GW}/k$", transform=ax.transAxes,
                    fontsize=6, color=base, ha="right", va="bottom", bbox=bbox)
            # 15 decades: let the locator thin the y ticks, else they are unreadable
            apply_max_ticks(ax, n=6, axes=("y",))
        ax.set_title(rf"\texttt{{{run}}} ({cls})", fontsize=8)
        ax.tick_params(labelsize=6.5)
        kmax = kk.max()
    axes[0, 0].set_xlim(1.2, 1.15 * kmax)
    axes[0, 0].set_ylim(*ylim)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="lower left", handlelength=1.0,
                      labelspacing=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$k$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle(ttl + r"  ---  fixed axes, same times $t=1.02,1.06,1.10,1.15$ "
                 r"(shade $=$ time); dotted $=k_0$", fontsize=9)
    out = save_figure(fig, name)
    print(f"saved {out}  (max |t_snapshot - t_target| across runs = {max_dt:.3f})")


def fig_source_gw_gallery():
    """4x3 grid, one panel per run, showing BOTH spectra together (Roper Pol Fig.1
    style): the source (magnetic or kinetic, grey) and the GW spectrum (class
    colour), each at several times, on one axis spanning the full dynamic range."""
    apply_paper_style(grid=False)
    fig, axes = plt.subplots(4, 3, figsize=(8.0, 9.6), constrained_layout=True)
    for ax, run in zip(axes.flat, RUNS):
        kk, tG, EGk, ESk = load(run)
        cls = RUNS[run][1]
        base = _CLASS_COLOR[cls]
        src = "mag" if RUNS[run][2] == "power_mag.dat" else "kin"
        idx = np.linspace(0, len(tG) - 1, 4).astype(int)
        shades = np.linspace(0.30, 0.95, len(idx))
        for j, i in enumerate(idx):
            Es = ESk[i].astype(float).copy(); Es[Es <= 0] = np.nan
            Eg = EGk[i].astype(float).copy(); Eg[Eg <= 0] = np.nan
            ax.loglog(kk, Es, color="0.45", alpha=shades[j], lw=1.0)   # source: grey
            ax.loglog(kk, Eg, color=base, alpha=shades[j], lw=1.0)     # GW: class colour
        k0 = kk[np.argmax(ESk[-1])]
        ax.axvline(k0, color="0.7", lw=0.6, ls=":")
        # y-range: from ~7 decades below the GW peak up to just above the source
        # peak, a fixed physical window (Roper Pol Fig.1 spans ~14 decades) that
        # avoids the deep GW noise floor of the forced runs.
        gwpeak, srcpeak = EGk[EGk > 0].max(), ESk[ESk > 0].max()
        ax.set_ylim(gwpeak * 1e-7, srcpeak * 5)
        ax.set_xlim(1.2, 1.15 * kk.max())
        # label the two bands inside the panel
        ax.text(0.94, 0.94, rf"$\Omega_{{\rm {src}}}/k$", transform=ax.transAxes,
                fontsize=6, color="0.35", ha="right", va="top")
        ax.text(0.94, 0.44, r"$\Omega_{\rm GW}/k$", transform=ax.transAxes,
                fontsize=6, color=base, ha="right", va="top")
        ax.set_title(rf"\texttt{{{run}}} ({cls})", fontsize=8)
        ax.tick_params(labelsize=6.5)
        apply_max_ticks(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$k$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Omega/k$  (source, GW)", fontsize=8)
    fig.suptitle(r"source (grey) and GW (coloured) spectra vs time; "
                 r"shade $=$ time, dotted $=k_0$", fontsize=9)
    out = save_figure(fig, "roperpol_source_gw_gallery")
    print(f"saved {out}")


def fig_energy():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    seen = set()
    for run in RUNS:
        kk, tG, EGk, _ = load(run)
        cls = RUNS[run][1]
        c = _CLASS_COLOR[cls]
        EG = _trapz(EGk, kk, axis=1)
        lbl = cls if cls not in seen else None
        seen.add(cls)
        ax.semilogy(tG, EG, color=c, lw=1.2, alpha=0.85, label=lbl)
        ax.text(tG[-1], EG[-1], f" {run}", fontsize=5.5, color=c, va="center")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"total GW energy $\;\mathcal{E}_{\rm GW}(t)=\int E_{\rm GW}(k)\,dk$")
    ax.set_title(r"GW energy vs time, all 12 runs", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right", title="source class",
              title_fontsize=8)
    ax.text(0.03, 0.06,
            "GW energy rises while the source acts,\nthen holds a steady oscillatory plateau",
            transform=ax.transAxes, fontsize=7, va="bottom", color="0.3")
    apply_max_ticks(ax)
    out = save_figure(fig, "roperpol_energy_evolution")
    print(f"saved {out}")


def main():
    fig_gallery()
    fig_source_gw_gallery()
    fig_gallery_fixed("gw")
    fig_gallery_fixed("source")
    fig_gallery_fixed("both")
    fig_energy()


if __name__ == "__main__":
    main()
