#!/usr/bin/env python3
r"""Task 2.5: run OUR kernel at Roper Pol et al. (2020)'s OWN parameters.

All three referees of ``REVIEW_2026-08-14.md`` (Sec. 5) converged on the same
complaint:

    "The manuscript EXPLAINS THE SIMULATIONS AWAY rather than explaining them:
     it never runs its own kernel at RP20's T_em, tau_c and k-range."

This script does exactly that, and only that.  It takes every number from the
repository's existing reductions of Ref. [RoperPol:2019wvy] run ``ini2`` -- the
pixel-digitized Fig. 1 (``roperpol_data``) and, when reachable, the raw public
Pencil Code spectra (``roperpol_timeseries``, Zenodo 10.5281/zenodo.3692072) --
then evaluates the finite-lifetime kernel of ``ir_branch_diagram`` at those
parameters over the band the simulation box actually contains, and measures the
local logarithmic slope d ln Omega_GW / d ln p there.

Nothing is fitted.  The lifetime is not tuned to the answer: three independent
closures for tau_c are carried through side by side and the spread is reported.

WHAT IS IMPORTED (never edited)
-------------------------------
``ir_branch_diagram``  H_lifetime / omega_global / omega_eddy / local_slope /
                       fit_slope -- the triangle correlator
                       f(tau) = (1-|tau|/tau_c) Theta(tau_c-|tau|) on top of
                       ``band_split_gw``'s spatial integral.
``roperpol_data``      k0, Omega_M, the effective Mach range and the measured
                       infrared slope, all computed from the digitized CSVs.
``roperpol_timeseries``(optional, network) the raw shell grid and the time
                       series, for the box fundamental, the Alfven speed and the
                       magnetic decay time.

Usage
-----
    python Notebooks/kernel_at_rp20.py            # figure + tables (offline OK)
    python Notebooks/kernel_at_rp20.py --raw      # also refresh raw-data numbers
    python Notebooks/kernel_at_rp20.py --scan     # band slope vs tau_c k0

Produces images/kernel_at_rp20.pdf.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_split_gw as B          # noqa: E402  (also installs the trapezoid shim)
import ir_branch_diagram as IB     # noqa: E402
import roperpol_data as RP         # noqa: E402

warnings.filterwarnings("ignore")

SQRT2PI = np.sqrt(2.0 * np.pi)

# --------------------------------------------------------------------------- #
#  RP20 run ini2: the parameters, and where each one comes from
# --------------------------------------------------------------------------- #
#
# The Pencil Code shell grid of ``power_krms.dat`` is in units of the box
# fundamental: shells k_rms = 1.29, 2.24, ... 575.0 on a 1152^3 grid.  The
# digitized Fig. 1 starts at k = 129.02, i.e. the SAME first shell, so the
# figure's abscissa is the shell number times
#
#     k_1 = 2 pi / L = 100  (in units of the Hubble rate at generation),
#
# which is the box fundamental quoted as k_box ~ 1e2 in derivation.tex
# Sec. sec:cps-reproduction.  The last digitized point, k = 5.84e4, is shell
# 575 x 101.6 -- the two ends agree on the calibration to 1.6%.
SHELL_TO_HUBBLE = 100.0
K_BOX = 129.0          # first resolved shell, k_rms = 1.29, in Hubble units
K_NYQ = 57500.0        # last shell, k_rms = 575, in Hubble units
KD_OVER_K0 = 44.6      # end of the digitized k^-5/3 range (k ~ 3e4) over k0

# The band over which ``roperpol_data.gw_ir_slope_omega_gw`` fits the measured
# infrared slope: k = 130 to 600, i.e. everything the box holds below the peak.
FIT_LO_K, FIT_HI_K = 130.0, 600.0

# Raw-data numbers (recomputed by ``--raw``; see _raw_parameters).
VA_INITIAL = 0.1245    # sqrt(2 E_M) at switch-on, run ini2
VA_FINAL = 0.0781      # sqrt(2 E_M) at the end of the run
TAU_DECAY = 0.610      # e-folding time of E_M,tot over the run (Hubble units)
IR_SLOPE_RAW = 1.295   # Omega_GW slope, shells 1-6, late-time average

P_LO, P_HI, PER_DEC = 0.02, 100.0, 48


def rp20_parameters(raw: bool = False) -> dict:
    """Every RP20 number this script uses, with its provenance recorded."""
    k0 = RP.k0()
    m_lo, m_hi = RP.effective_mach()
    par = dict(
        k0=k0,
        omega_m=RP.energy_fraction(),
        mach_lo=m_lo, mach_hi=m_hi,
        ir_slope_digitized=RP.gw_ir_slope_omega_gw(FIT_LO_K, FIT_HI_K),
        gw_peak=RP.gw_peak_ratio(),
        k_box=K_BOX, k_nyq=K_NYQ,
        va0=VA_INITIAL, va1=VA_FINAL, tau_decay=TAU_DECAY,
        ir_slope_raw=IR_SLOPE_RAW,
        raw="derivation.tex / digitized Fig. 1 (cached constants)",
    )
    if raw:
        par.update(_raw_parameters())
    par["p_box"] = par["k_box"] / par["k0"]
    par["p_nyq"] = par["k_nyq"] / par["k0"]
    par["p_fit_lo"] = FIT_LO_K / par["k0"]
    par["p_fit_hi"] = FIT_HI_K / par["k0"]
    # Three independent closures for the source lifetime, in units 1/k0.
    par["thats"] = {
        "eddy, v_A(t_0)=%.3f" % par["va0"]: 1.0 / par["va0"],
        "eddy, M=sqrt(2 Omega_M)=%.3f" % par["mach_hi"]: 1.0 / par["mach_hi"],
        "eddy, M=sqrt(Omega_M)=%.3f" % par["mach_lo"]: 1.0 / par["mach_lo"],
        "E_M e-folding, %.2f/H_*" % par["tau_decay"]: par["tau_decay"] * par["k0"],
    }
    return par


def _raw_parameters() -> dict:
    """Recompute the raw-data numbers from the public ini2 spectra (network)."""
    try:
        import roperpol_timeseries as TS
        k, tM, EM, tG, EG = TS.load()
    except Exception as exc:                       # offline, or Zenodo mirror down
        print(f"  [--raw] public ini2 data unreachable ({type(exc).__name__}: {exc});"
              f"\n          falling back on the cached constants above, which came"
              f"\n          from the same files, and on derivation.tex.")
        return {}
    kp = k > 0
    kk, EMk, EGk = k[kp], EM[:, kp], EG[:, kp]
    emt = np.trapezoid(EMk, kk, axis=1)
    late = tG > 1.2
    eavg = EGk[late].mean(0)
    m = (kk >= 1.0) & (kk <= 6.5) & (eavg > 0)
    ir = float(np.polyfit(np.log(kk[m]), np.log(kk[m] * eavg[m]), 1)[0])

    # Which branch of Eq.(eq:window-factor) is each box mode in?  A mode in the
    # k*Delta_eta << 1 branch saturates only when the SOURCE dies (a k-independent
    # time); a mode in the k*Delta_eta >> 1 branch saturates after ~one of its own
    # oscillations, i.e. at t - t0 ~ 1/k, while the source is still alive.  The
    # measured saturation times below go as 1/k across the whole box, so every
    # box mode is in the second branch -- which is the k^1 branch.
    t0 = tG[0]
    print("\n  [--raw] mode-by-mode GW saturation in run ini2:")
    print(f"    {'shell':>6}{'k (Hubble)':>12}{'t_sat(90%)':>12}{'1/k':>10}"
          f"{'t_sat*k':>10}")
    for i in range(6):
        y = EGk[:, i]
        ts = float(tG[int(np.argmax(y >= 0.9 * np.median(y[late])))] - t0)
        kh = float(kk[i]) * SHELL_TO_HUBBLE
        print(f"    {i + 1:6d}{kh:12.0f}{ts:12.4f}{1.0 / kh:10.4f}{ts * kh:10.2f}")
    print("    -> t_sat ~ 1/k, not a common k-independent time: every mode in the")
    print("       box completes its oscillations while the source still lives.")
    # The infrared slope in time: causal at switch-on, k^1 within ~0.02/H_*.
    print(f"    {'t - t_0':>10}{'IR slope':>10}")
    for tt in (0.000, 0.002, 0.005, 0.020, 0.100, 0.397):
        j = int(np.argmin(np.abs(tG - t0 - tt)))
        mm = (kk >= 1.0) & (kk <= 6.5) & (EGk[j] > 0)
        s = float(np.polyfit(np.log(kk[mm]), np.log(kk[mm] * EGk[j][mm]), 1)[0])
        print(f"    {tG[j] - t0:10.4f}{s:+10.3f}")

    return dict(
        k_box=float(kk[0]) * SHELL_TO_HUBBLE,
        k_nyq=float(kk[-1]) * SHELL_TO_HUBBLE,
        va0=float(np.sqrt(2.0 * emt[0])),
        va1=float(np.sqrt(2.0 * emt[-1])),
        tau_decay=float(-1.0 / np.polyfit(tM, np.log(emt), 1)[0]),
        ir_slope_raw=ir,
        raw="raw Pencil Code spectra, Zenodo 10.5281/zenodo.3692072",
    )


# --------------------------------------------------------------------------- #
#  the kernel at those parameters
# --------------------------------------------------------------------------- #
def kernel_grid(par: dict, per_decade: int = PER_DEC):
    """p grid and the spatial factor G(p) at RP20's own spectral extent."""
    n = int(round(per_decade * np.log10(P_HI / P_LO))) + 1
    ps = np.geomspace(P_LO, P_HI, n)
    r = KD_OVER_K0 ** (4.0 / 3.0)          # R = (k_d/k0)^{4/3}
    r_ir = par["k0"] / par["k_box"]        # R_IR = k0/k_IR, the box truncation
    return ps, IB.spatial_floor(ps, R=r, R_IR=r_ir), r, r_ir


def curves(par: dict, ps, G, r, r_ir, thats):
    """Omega_GW(p) for the hard-global and eddy lifetimes, plus the control."""
    out = {"coherent": ps ** 3 * G}        # no window at all: the k^3 control
    for t in thats:
        out[("global", t)] = IB.omega_global(ps, t, G)
        out[("eddy", t)] = IB.omega_eddy(ps, t * SQRT2PI, R=r, R_IR=r_ir)
    return out


def report(par: dict, ps, om, thats) -> None:
    lo, hi = par["p_fit_lo"], par["p_fit_hi"]
    probes = [par["p_box"], 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, hi, 1.0,
              par["gw_peak"], 5.0, 20.0, par["p_nyq"]]

    print("\n" + "=" * 92)
    print("RP20 run ini2 -- parameters used  (source: %s)" % par["raw"])
    print("=" * 92)
    print(f"  k0 (source peak, Hubble units)      {par['k0']:.1f}"
          f"      = {par['k0'] / SHELL_TO_HUBBLE:.2f} shells")
    print(f"  k_box (box fundamental, 1st shell)  {par['k_box']:.1f}"
          f"      -> p_box = {par['p_box']:.4f}")
    print(f"  k_Nyquist (last shell)              {par['k_nyq']:.0f}"
          f"    -> p_Nyq = {par['p_nyq']:.1f}")
    print(f"  k_d/k0 (end of the k^-5/3 range)    {KD_OVER_K0:.1f}"
          f"       -> R = (k_d/k0)^(4/3) = {KD_OVER_K0 ** (4 / 3):.0f}")
    print(f"  Omega_M                             {par['omega_m']:.3e}"
          f"  -> M = {par['mach_lo']:.3f}--{par['mach_hi']:.3f}")
    print(f"  v_A(t_0) .. v_A(t_end)              {par['va0']:.4f} .. {par['va1']:.4f}")
    print(f"  E_M e-folding time                  {par['tau_decay']:.3f}/H_*")
    print(f"  GW peak / k0                        {par['gw_peak']:.3f}")
    print(f"  MEASURED IR slope, k=130-600        {par['ir_slope_digitized']:+.3f}"
          f"  (digitized)   {par['ir_slope_raw']:+.3f}  (raw shells 1-6)")
    print(f"  fit band                            p = {lo:.3f} -- {hi:.3f}"
          f"   ({np.log10(hi / lo):.2f} decades)")

    print("\n" + "=" * 92)
    print("LOCAL SLOPE  d ln Omega_GW / d ln p  OF OUR KERNEL, ACROSS RP20's OWN BAND")
    print("=" * 92)
    head = f"  {'p = k/k0':>10}{'k (Hubble)':>12}"
    keys = [("coherent", None)] + [(m, t) for m in ("global", "eddy") for t in thats]
    for m, t in keys:
        head += f"{(m if t is None else f'{m[:3]} {t:.0f}'):>11}"
    print(head)
    slopes = {}
    for m, t in keys:
        key = "coherent" if t is None else (m, t)
        slopes[key] = IB.local_slope(ps, om[key])
    for p in probes:
        i = int(np.argmin(np.abs(ps - p)))
        row = f"  {ps[i]:10.4f}{ps[i] * par['k0']:12.0f}"
        for m, t in keys:
            key = "coherent" if t is None else (m, t)
            row += f"{slopes[key][i]:+11.2f}"
        print(row)

    print("\n" + "=" * 92)
    print(f"BAND-AVERAGED SLOPE over p = {lo:.3f}-{hi:.3f} (k = {FIT_LO_K:.0f}-{FIT_HI_K:.0f}),"
          f"  measured {par['ir_slope_digitized']:+.2f}")
    print("=" * 92)
    print(f"  {'lifetime model':<34}{'tau_c k0':>10}{'p_break':>10}"
          f"{'k_break':>10}{'band slope':>12}{'residual':>10}{'peak/k0':>9}")
    meas = par["ir_slope_digitized"]
    rows = []
    s = IB.fit_slope(ps, om["coherent"], lo, hi)
    print(f"  {'infinitely coherent (control)':<34}{'inf':>10}{'--':>10}{'--':>10}"
          f"{s:+12.3f}{s - meas:+10.3f}{B.peak_position(ps, om['coherent']):9.3f}")
    for m in ("global", "eddy"):
        for t in thats:
            y = om[(m, t)]
            s = IB.fit_slope(ps, y, lo, hi)
            rows.append((m, t, s))
            print(f"  {m + ' lifetime':<34}{t:10.1f}{np.pi / t:10.4f}"
                  f"{np.pi / t * par['k0']:10.1f}{s:+12.3f}{s - meas:+10.3f}"
                  f"{B.peak_position(ps, y):9.3f}")
    lo_s = min(r[2] for r in rows)
    hi_s = max(r[2] for r in rows)
    print(f"\n  our kernel spans {lo_s:+.2f} to {hi_s:+.2f} across the admissible "
          f"lifetimes;\n  the measurement {meas:+.2f} "
          f"{'IS' if lo_s <= meas <= hi_s else 'is NOT'} inside that span, and the "
          f"causal\n  asymptote +3 is nowhere in the box "
          f"(the coherent control already reads "
          f"{IB.fit_slope(ps, om['coherent'], lo, hi):+.2f} here).")
    return slopes


# --------------------------------------------------------------------------- #
#  figure
# --------------------------------------------------------------------------- #
def _data_local_slope(kg, og, win=61):
    """Local slope of the digitized Omega_GW = k (Omega/k), heavily smoothed."""
    lk, ly = np.log(kg), np.log(kg * og)
    out = np.full(lk.shape, np.nan)
    h = win // 2
    for i in range(lk.size):
        a, b = max(0, i - h), min(lk.size, i + h + 1)
        if b - a >= 8:
            out[i] = np.polyfit(lk[a:b], ly[a:b], 1)[0]
    return out


def figure(par: dict, ps, om, slopes, thats, name="kernel_at_rp20"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import (
        PALETTE, apply_max_ticks, apply_paper_style, save_figure,
    )

    apply_paper_style(grid=False)
    t_lo, t_hi = min(thats), max(thats)
    col = {t: PALETTE[i] for t, i in
           zip(thats, (6, 1, 3, 5, 2, 7))}
    meas = par["ir_slope_digitized"]
    pb, pn = par["p_box"], par["p_nyq"]
    lo, hi = par["p_fit_lo"], par["p_fit_hi"]

    kg, og = RP.load("gw")
    pg = kg / par["k0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)

    # ---- (a) the spectra -------------------------------------------------- #
    ax1.axvspan(pb, pn, color="0.88", zorder=0, lw=0)
    ax1.text(np.sqrt(pb * pn), 3.0, "band the ini2 box contains", fontsize=7.5,
             color="0.35", ha="center")
    yd = kg * og
    ax1.loglog(pg, yd / yd.max(), "-", color="0.0", lw=2.0,
               label=r"$\Omega_{\rm GW}$, run \texttt{ini2} (digitized)")
    y = om["coherent"]
    ax1.loglog(ps, y / y.max(), ":", color=PALETTE[1], lw=1.6,
               label=r"our kernel, no window (control)")
    for t in (t_lo, t_hi):
        y = om[("eddy", t)]
        ax1.loglog(ps, y / y.max(), "-", color=col[t], lw=1.5,
                   label=rf"our kernel, $\tau_ck_0={t:.0f}$")
    yc = om["coherent"] / om["coherent"].max()
    for power, span, anchor, tpos in ((3, (0.03, 0.15), 2.0 * np.interp(0.03, ps, yc), 3.0),
                                      (1, (0.25, 0.85), 0.30, 0.45)):
        pp = np.array(span)
        ax1.loglog(pp, anchor * (pp / pp[0]) ** power, "--", color="0.45", lw=1.0)
        ax1.text(np.sqrt(pp[0] * pp[1]),
                 anchor * (np.sqrt(pp[1] / pp[0])) ** power * tpos,
                 rf"$p^{{{power}}}$", color="0.35", fontsize=11, ha="center")
    ax1.set_xlim(P_LO, 1.2 * pn)
    ax1.set_ylim(1e-8, 8.0)
    ax1.set_xlabel(r"$p=k/k_0$")
    ax1.set_ylabel(r"$\Omega_{\rm GW}(p)/\Omega_{\rm GW}^{\rm peak}$")
    ax1.set_title(r"(a) spectra at RP20's parameters", fontsize=11)
    ax1.legend(frameon=False, fontsize=7.2, loc="lower left")
    apply_max_ticks(ax1, n=6)

    # ---- (b) the local slope, the panel the referees asked for ------------ #
    ax2.axvspan(pb, pn, color="0.88", zorder=0, lw=0)
    ax2.axvspan(lo, hi, color="0.74", zorder=0, lw=0)
    ax2.axhline(3.0, color="0.55", lw=0.9, ls=":")
    ax2.axhline(1.0, color="0.55", lw=0.9, ls=":")
    ax2.axhline(meas, color=PALETTE[7], lw=1.6,
                label=rf"measured $k^{{{meas:+.2f}}}$ (RP20 \texttt{{ini2}})")
    sd = _data_local_slope(kg, og)
    ax2.semilogx(pg, sd, "-", color="0.0", lw=1.1, alpha=0.6,
                 label=r"local slope of the data")
    ax2.semilogx(ps, slopes["coherent"], ":", color="0.35", lw=1.7,
                 label=r"no window (control) $\to p^{3}$")
    for t in thats:
        ax2.semilogx(ps, slopes[("eddy", t)], "-", color=col[t], lw=1.4,
                     label=rf"eddy $\tau$, $\tau_ck_0={t:.0f}$")
        ax2.axvline(np.pi / t, color=col[t], lw=0.9, ls="--", alpha=0.7)
    ax2.semilogx(ps, slopes[("global", t_lo)], "-.", color=col[t_lo], lw=1.0,
                 alpha=0.75, label=rf"hard global $\tau$, $\tau_ck_0={t_lo:.0f}$")
    ax2.text(np.pi / t_hi * 1.10, -1.45, r"$p_{\rm break}=\pi/\tau_c$", fontsize=7.5,
             color="0.3", rotation=90, va="bottom")
    ax2.set_xlim(P_LO, 20.0)
    ax2.set_ylim(-1.6, 5.3)
    ax2.set_xlabel(r"$p=k/k_0$")
    ax2.set_ylabel(r"$d\ln\Omega_{\rm GW}/d\ln p$")
    ax2.set_title(r"(b) local slope; shaded = the box, dark = the fit band",
                  fontsize=11)
    ax2.legend(frameon=False, fontsize=7.0, loc="upper left", ncol=2,
               columnspacing=1.0, handlelength=1.9)
    ax2.set_yticks([-1.5, 0.0, 1.0, 2.0, 3.0])

    out = save_figure(fig, name)
    plt.close(fig)
    print(f"\n[kernel_at_rp20] wrote {out}")
    return out


# --------------------------------------------------------------------------- #
def _scan(par, ps, G, r, r_ir) -> None:
    lo, hi = par["p_fit_lo"], par["p_fit_hi"]
    print("\n" + "=" * 78)
    print(f"BAND SLOPE over p = {lo:.3f}-{hi:.3f} versus the source lifetime")
    print("=" * 78)
    print(f"  {'tau_c k0':>10}{'p_break':>10}{'global':>10}{'eddy':>10}{'peak(eddy)':>12}")
    for t in (4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 60, 100, 200, 411, 1000):
        og_ = IB.omega_global(ps, float(t), G)
        oe_ = IB.omega_eddy(ps, float(t) * SQRT2PI, R=r, R_IR=r_ir)
        print(f"  {t:10.0f}{np.pi / t:10.4f}"
              f"{IB.fit_slope(ps, og_, lo, hi):+10.3f}"
              f"{IB.fit_slope(ps, oe_, lo, hi):+10.3f}"
              f"{B.peak_position(ps, oe_):12.3f}", flush=True)


def main() -> None:
    par = rp20_parameters(raw="--raw" in sys.argv)
    ps, G, r, r_ir = kernel_grid(par)
    thats = sorted(round(v, 1) for v in par["thats"].values())
    print("\nlifetime closures (tau_c k0):")
    for label, v in par["thats"].items():
        print(f"    {label:<40} tau_c k0 = {v:8.1f}   p_break = {np.pi / v:.4f}")
    om = curves(par, ps, G, r, r_ir, thats)
    slopes = report(par, ps, om, thats)
    if "--scan" in sys.argv:
        _scan(par, ps, G, r, r_ir)
    figure(par, ps, om, slopes, thats)


if __name__ == "__main__":
    main()
