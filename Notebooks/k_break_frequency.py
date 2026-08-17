r"""The $k^3\to k^1$ duration break in observed frequency, against LISA and PTA.

Task 2.3 of ``ACTION_PLAN.md``.  Nothing here is a new physical result: the break
itself is Ref. [RoperPol:2022iel] ($k_{\rm br}=1/\delta t_{\rm fin}$, derived and
confirmed in simulations), and the causal $k^3$ below it is the Cai-Pi-Sasaki
theorem [Cai:2019cdl].  What this script does is *arithmetic*: push the break
through the cosmological frequency map of Sec. ``sec:peak-units`` and ask where it
lands relative to the two bands that exist.

FORMULA CHAIN
-------------
1.  Break in source units (Sec. ``sec:ir-branch``, Eq. ``eq:k-break-pi``):

        k_break = pi / tau_c .

    The factor pi (not the naive 1/tau_c of Eq. ``eq:k-break``) is because the
    stress is *quadratic*: the two-leg temporal factor is the transform of the
    SQUARED tent correlator, whose one-sided slope at zero lag is f'(0+) = -2/tau_c.
    The cusp theorem then gives T -> 4/(tau_c omega^2) against T(0) = 2 tau_c/3, the
    asymptotes cross at omega tau_c = sqrt(6), and the local slope passes its
    midpoint where 1 + cos u = 2 sin(u)/u, i.e. at u = pi exactly.  Measured:
    3.07 (global lifetime) / 3.17 (eddy lifetime) vs pi = 3.1416.

2.  Hubble units.  With khat = k/H_* and tauhat_c = tau_c H_*,

        khat_break = pi / tauhat_c .

3.  Redshift (Eq. ``eq:fH``, Eq. ``eq:f0``).  A comoving mode is observed at
    f = (khat/2 pi) f_H with

        f_H = 1.6e-5 Hz (g_*/100)^(1/6) (T_*/100 GeV) ,

    hence the master relation

        f_break = f_H / (2 tauhat_c) .

    Note it does NOT contain gamma = l_0 H_* directly -- the break is set by the
    source *lifetime*, not the eddy size.  gamma enters only through tauhat_c when
    the lifetime is taken to be an eddy turnover time.

4.  Two closures for tauhat_c.
    (a) Global lifetime, a fraction eps of a Hubble time: tauhat_c = eps, so

            f_break = f_H / (2 eps) .

    (b) Outer-scale eddy turnover, tau_c = 1/(k_0 u_0), with khat_0 = 2 pi/gamma:

            tauhat_c = gamma / (2 pi u_0)   =>   f_break = pi u_0 f_H / gamma
                                                        = pi u_0 f_0 ,

        i.e. in kernel units simply p_break = k_break/k_0 = pi u_0.

5.  Resolution caveat.  Table ``tab:ir-branch`` resolves a clean k^1 band only for
    tauhat = tau_c k_0 >~ 1e3.  For the eddy closure tau_c k_0 = 1/u_0, so any
    u_0 >~ 1e-3 leaves the break sitting within a decade or so of the source-scale
    peak (p_peak ~ 1-2) and the k^1 band is narrow.  This is reported per corner.

LISA
----
Omega_sens(f) = (4 pi^2 / 3 H_0^2) f^3 S_n(f), with the standard analytic
sky-and-polarization-averaged LISA strain sensitivity (Robson, Cornish & Liu 2019,
arXiv:1803.01944, the LISA SciRD noise model; same numbers as Babak, Petiteau &
Sesana 2021):

    P_OMS(f) = (1.5e-11 m)^2 [1 + (2 mHz / f)^4]
    P_acc(f) = (3e-15 m/s^2)^2 [1 + (0.4 mHz / f)^2] [1 + (f / 8 mHz)^4]
    f_*      = c / (2 pi L),  L = 2.5e9 m
    S_n(f)   = (10 / 3 L^2) { P_OMS + 2 [1 + cos^2(f/f_*)] P_acc / (2 pi f)^4 }
               x [1 + (6/10) (f / f_*)^2]

The often-quoted "3-10 mHz" is the minimum of the STRAIN sensitivity; the extra
f^3 in Omega_sens pushes the energy-density minimum down.  Both are printed.

PTA
---
Nanohertz band: 1/(20 yr) at the low end (the longest baselines) up to the Nyquist
frequency of a few-week cadence, 1/(2 x 2 weeks).  Both are printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_paper_style,
    save_figure,
)

# ----------------------------------------------------------------- constants
C_LIGHT = 2.99792458e8           # m/s
MPC = 3.0856775814913673e22      # m
H0_LITTLE_H = 0.67
H0 = H0_LITTLE_H * 1.0e5 / MPC   # s^-1
YEAR = 3.155815e7                # s (Julian-ish; only used for PTA baselines)

# epochs: (label, T_* in GeV, g_*)
EPOCHS = (
    ("EW", 100.0, 100.0),
    ("QCD", 0.15, 15.0),
)

# free-parameter ranges (task specification)
GAMMA_RANGE = (1.0e-3, 1.0e-1)   # gamma = l_0 H_*, largest eddy in Hubble radii
U0_RANGE = (1.0e-2, 3.0e-1)      # outer-scale rms velocity
EPS_RANGE = (1.0e-2, 1.0e0)      # global lifetime as a fraction of a Hubble time

GAMMA_GRID = (1.0e-3, 1.0e-2, 1.0e-1)
U0_GRID = (1.0e-2, 1.0e-1, 3.0e-1)
EPS_GRID = (1.0e-2, 1.0e-1, 1.0e0)

# PTA band
PTA_LO = 1.0 / (20.0 * YEAR)             # longest baseline, 20 yr
PTA_HI = 1.0 / (2.0 * 14.0 * 86400.0)    # Nyquist of a fortnightly cadence


# ------------------------------------------------------------------ physics
def f_hubble(T_star_GeV: float, g_star: float) -> float:
    """Hubble frequency today, Eq. ``eq:fH``  [Hz]."""
    return 1.6e-5 * (g_star / 100.0) ** (1.0 / 6.0) * (T_star_GeV / 100.0)


# Break coefficient u_* solving d ln T/d ln u = -1.  WHICH ONE APPLIES depends on
# whether the finite lifetime is a WINDOW or a stationary triangular lag memory:
#
#   window (hard lifetime): the field correlator is R = 1 inside the window, so the
#       stress carries R^2 = R -- squaring does nothing -- and the two-leg factor is
#       the UN-squared tent 4 sin^2(uW/2)/u^2, giving u_* = 2.3311.
#   triangular lag memory:  the stress carries the SQUARED tent, u_* = pi exactly.
#
# Every closure in this script imposes a source LIFETIME, so the window value is the
# correct one.  Corrected 2026-08-17; the script previously used pi throughout, which
# made every f_break high by pi/2.3311 = 1.348.
BREAK_COEFF_WINDOW = 2.3311223
BREAK_COEFF_TRIANGLE = np.pi
BREAK_COEFF = BREAK_COEFF_WINDOW


def f_break_from_tauhat(tauhat_c: float, T_star_GeV: float, g_star: float) -> float:
    """f_break = BREAK_COEFF f_H / (2 pi tauhat_c), from k_break = BREAK_COEFF/tau_c
    and f = khat f_H/(2 pi)."""
    return BREAK_COEFF * f_hubble(T_star_GeV, g_star) / (2.0 * np.pi * tauhat_c)


def tauhat_eddy(gamma: float, u0: float) -> float:
    """Outer-scale eddy turnover time in Hubble times: tau_c H_* = gamma/(2 pi u0)."""
    return gamma / (2.0 * np.pi * u0)


# --------------------------------------------------------------------- LISA
def lisa_Sn(f: np.ndarray, L: float = 2.5e9) -> np.ndarray:
    """Sky- and polarization-averaged LISA strain sensitivity S_n(f) [1/Hz].

    Robson, Cornish & Liu (2019), arXiv:1803.01944, Eqs. (1)-(4), with the
    LISA SciRD noise levels and arm length L = 2.5e9 m.
    """
    f = np.asarray(f, dtype=float)
    f_star = C_LIGHT / (2.0 * np.pi * L)
    P_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f) ** 4)                 # m^2/Hz
    P_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / f) ** 2) \
        * (1.0 + (f / 8.0e-3) ** 4)                                    # (m/s^2)^2/Hz
    bracket = P_oms + 2.0 * (1.0 + np.cos(f / f_star) ** 2) \
        * P_acc / (2.0 * np.pi * f) ** 4
    return (10.0 / (3.0 * L ** 2)) * bracket * (1.0 + 0.6 * (f / f_star) ** 2)


def lisa_Omega_sens(f: np.ndarray) -> np.ndarray:
    """Omega_sens(f) = (4 pi^2 / 3 H_0^2) f^3 S_n(f)."""
    return (4.0 * np.pi ** 2 / (3.0 * H0 ** 2)) * np.asarray(f, float) ** 3 * lisa_Sn(f)


def _min_of(func, lo: float, hi: float) -> float:
    """Frequency of the minimum of ``func`` on [lo, hi] (log-space golden section)."""
    res = minimize_scalar(lambda lg: np.log(func(10.0 ** lg)),
                          bounds=(np.log10(lo), np.log10(hi)), method="bounded",
                          options={"xatol": 1e-10})
    return float(10.0 ** res.x)


def _factor_band(func, f_min: float, factor: float = 2.0) -> tuple[float, float]:
    """Frequencies bracketing f_min where ``func`` = factor x its minimum."""
    from scipy.optimize import brentq
    y0 = func(f_min)
    g = lambda f: func(f) - factor * y0  # noqa: E731
    lo = brentq(g, 1.0e-5, f_min, xtol=1e-16, rtol=1e-14)
    hi = brentq(g, f_min, 1.0, xtol=1e-16, rtol=1e-14)
    return lo, hi


# ------------------------------------------------------------------- report
def scan() -> dict:
    """Print the parameter-corner tables; return everything the figure needs."""
    out: dict = {}

    print("=" * 78)
    print("f_H per epoch   [Eq. eq:fH]")
    print("=" * 78)
    for label, T, g in EPOCHS:
        print(f"  {label:4s}  T_*={T:g} GeV, g_*={g:g}   f_H = {f_hubble(T, g):.4g} Hz")

    print()
    print("=" * 78)
    print("(a) GLOBAL lifetime  tau_c = eps / H_*   ->   f_break = f_H / (2 eps)")
    print("     f_break is INDEPENDENT of gamma; p_break = gamma/(2 eps) is not.")
    print("     tauhat = tau_c k_0 = 2 pi eps / gamma  (tab:ir-branch wants >~ 1e3)")
    print("=" * 78)
    print(f"{'epoch':6s}{'eps':>8s}{'f_break/Hz':>13s}{'gamma':>9s}"
          f"{'p_break':>10s}{'tau_c k_0':>11s}{'k^1 band?':>11s}")
    for label, T, g in EPOCHS:
        for eps in EPS_GRID:
            fb = f_break_from_tauhat(eps, T, g)
            for gam in GAMMA_GRID:
                # p_break = k_break/k_0 = (pi/eps)/(2 pi/gamma) = gamma/(2 eps)
                p_br = gam / (2.0 * eps)
                tk0 = 2.0 * np.pi * eps / gam
                resolved = "yes" if tk0 >= 1.0e3 else "no"
                print(f"{label:6s}{eps:8.3g}{fb:13.4g}{gam:9.3g}"
                      f"{p_br:10.4g}{tk0:11.4g}{resolved:>11s}")
    print("  [a clean k^1 band needs eps/gamma >~ 160, i.e. a long-lived source "
          "with small eddies;")
    print("   at the other corners the break has not separated from the "
          "source-scale roll-over.]")

    print()
    print("=" * 78)
    print("(b) EDDY lifetime  tau_c = 1/(k_0 u_0)  ->  f_break = pi u_0 f_H / gamma")
    print("     tauhat_c = gamma/(2 pi u_0);  p_break = pi u_0;  tauhat = 1/u_0")
    print("=" * 78)
    print(f"{'epoch':6s}{'gamma':>9s}{'u_0':>8s}{'tauhat_c':>11s}"
          f"{'f_break/Hz':>13s}{'p_break':>10s}{'tau_c k_0':>11s}{'k^1 band?':>11s}")
    for label, T, g in EPOCHS:
        for gam in GAMMA_GRID:
            for u0 in U0_GRID:
                th = tauhat_eddy(gam, u0)
                fb = f_break_from_tauhat(th, T, g)
                p_br = BREAK_COEFF * u0
                tk0 = 1.0 / u0
                resolved = "yes" if tk0 >= 1.0e3 else "no"
                print(f"{label:6s}{gam:9.3g}{u0:8.3g}{th:11.4g}{fb:13.4g}"
                      f"{p_br:10.4g}{tk0:11.4g}{resolved:>11s}")
    print("  [p_break = pi u_0 is INDEPENDENT of gamma and of the epoch: with an "
          "eddy-turnover")
    print("   lifetime the break never sits more than ~1.5 decades below the "
          "source-scale peak")
    print("   p_peak ~ 1-2, so the k^1 band is narrow and, per tab:ir-branch, "
          "not cleanly resolved.]")

    # swept ranges for the figure
    for label, T, g in EPOCHS:
        eddy = [f_break_from_tauhat(tauhat_eddy(gam, u0), T, g)
                for gam in GAMMA_RANGE for u0 in U0_RANGE]
        glob = [f_break_from_tauhat(eps, T, g) for eps in EPS_RANGE]
        out[f"{label}_eddy"] = (min(eddy), max(eddy))
        out[f"{label}_global"] = (min(glob), max(glob))

    print()
    print("=" * 78)
    print("swept ranges (figure bands)")
    print("=" * 78)
    for key, (lo, hi) in out.items():
        print(f"  {key:12s}  {lo:.3g} -- {hi:.3g} Hz")

    # ------------------------------------------------------------------ LISA
    f_om = _min_of(lisa_Omega_sens, 1.0e-5, 1.0)
    om_lo, om_hi = _factor_band(lisa_Omega_sens, f_om, 2.0)
    f_h = _min_of(lisa_Sn, 1.0e-5, 1.0)
    h_lo, h_hi = _factor_band(lisa_Sn, f_h, 2.0)

    print()
    print("=" * 78)
    print("LISA  (Robson-Cornish-Liu 2019 SciRD model, L = 2.5e9 m, "
          f"h = {H0_LITTLE_H})")
    print("=" * 78)
    print(f"  Omega_sens minimum   f = {f_om * 1e3:.3f} mHz   "
          f"Omega_sens = {lisa_Omega_sens(f_om):.4g}")
    print(f"  within a factor 2    f = {om_lo * 1e3:.3f} -- {om_hi * 1e3:.3f} mHz")
    print(f"  STRAIN S_n minimum   f = {f_h * 1e3:.3f} mHz   "
          f"(sqrt(f S_n) = {np.sqrt(f_h * lisa_Sn(f_h)):.3g})")
    print(f"  strain within f.2    f = {h_lo * 1e3:.3f} -- {h_hi * 1e3:.3f} mHz")
    print("  [the often-quoted 3-10 mHz is the STRAIN minimum; the f^3 in "
          "Omega_sens moves it down]")

    print()
    print("=" * 78)
    print("PTA")
    print("=" * 78)
    print(f"  1/(20 yr)        = {PTA_LO:.4g} Hz")
    print(f"  1/(2 x 2 weeks)  = {PTA_HI:.4g} Hz")

    out["lisa_f_om"] = f_om
    out["lisa_om_band"] = (om_lo, om_hi)
    out["lisa_f_h"] = f_h
    return out


# ------------------------------------------------------------------- figure
def main(name: str = "k_break_frequency"):
    apply_paper_style()
    res = scan()

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]}, constrained_layout=True)

    FLO, FHI = 1.0e-10, 1.0

    # ---------------------------------------------------------------- top: bands
    f = np.logspace(np.log10(FLO), 0.0, 4000)
    om = lisa_Omega_sens(f)
    lisa_mask = (f >= 1.0e-5) & (f <= 1.0)
    ax0.loglog(f[lisa_mask], om[lisa_mask], color=PALETTE[5], lw=1.6,
               label=r"LISA $\Omega_{\rm sens}(f)$")

    f_om = res["lisa_f_om"]
    om_lo, om_hi = res["lisa_om_band"]
    ax0.axvspan(om_lo, om_hi, color=PALETTE[5], alpha=0.16, lw=0)
    ax0.plot([f_om], [lisa_Omega_sens(f_om)], "o", color=PALETTE[5], ms=6, zorder=5)
    ax0.annotate(rf"$\Omega$ min {f_om * 1e3:.2f}\,mHz"
                 "\n" rf"(within $\times2$: {om_lo * 1e3:.1f}--{om_hi * 1e3:.1f}\,mHz)",
                 xy=(f_om, lisa_Omega_sens(f_om)), xytext=(1.0e-5, 8e-14),
                 fontsize=9.5, color=PALETTE[5], ha="left",
                 arrowprops=dict(arrowstyle="->", color=PALETTE[5], lw=0.9,
                                 connectionstyle="arc3,rad=0.15"))

    ax0.axvspan(PTA_LO, PTA_HI, color=PALETTE[3], alpha=0.18, lw=0)
    ax0.text(np.sqrt(PTA_LO * PTA_HI), 3e-14, "PTA band", fontsize=10.5,
             color=PALETTE[3], ha="center", va="bottom")

    ax0.set_ylabel(r"$\Omega_{\rm sens}(f)$")
    ax0.set_ylim(1e-14, 1e-7)
    ax0.set_xlim(FLO, FHI)
    ax0.legend(loc="upper left", fontsize=10, handlelength=1.6,
               framealpha=0.9)

    # ------------------------------------------------- bottom: k_break locations
    rows = (
        ("EW_global", r"EW, $\tau_c=\epsilon/H_*$", PALETTE[6], "-",
         f_break_from_tauhat(0.1, 100.0, 100.0)),
        ("EW_eddy", r"EW, $\tau_c=1/(k_0u_0)$", PALETTE[6], "--",
         f_break_from_tauhat(tauhat_eddy(0.01, 0.1), 100.0, 100.0)),
        ("QCD_global", r"QCD, $\tau_c=\epsilon/H_*$", PALETTE[1], "-",
         f_break_from_tauhat(0.1, 0.15, 15.0)),
        ("QCD_eddy", r"QCD, $\tau_c=1/(k_0u_0)$", PALETTE[1], "--",
         f_break_from_tauhat(tauhat_eddy(0.01, 0.1), 0.15, 15.0)),
    )
    ax1.axvspan(om_lo, om_hi, color=PALETTE[5], alpha=0.16, lw=0)
    ax1.axvspan(PTA_LO, PTA_HI, color=PALETTE[3], alpha=0.18, lw=0)

    for i, (key, lab, col, ls, fid) in enumerate(rows):
        y = 2.0 * (len(rows) - i)
        lo, hi = res[key]
        ax1.plot([lo, hi], [y, y], color=col, lw=7.0, alpha=0.30,
                 solid_capstyle="butt")
        ax1.plot([lo, hi], [y, y], color=col, lw=1.4, ls=ls)
        ax1.plot([lo, hi], [y, y], "|", color=col, ms=11, mew=1.4)
        ax1.plot([fid], [y], "o", color=col, ms=5.5, zorder=5)
        ax1.text(lo, y + 0.38, lab, fontsize=10, color=col,
                 va="bottom", ha="left")

    ax1.set_xscale("log")
    ax1.set_xlim(FLO, FHI)
    ax1.set_ylim(0.8, 2.0 * len(rows) + 1.9)
    ax1.set_yticks([])
    ax1.set_xlabel(r"observed frequency today $f$ [Hz]")
    ax1.set_ylabel(r"$f_{\rm break}=f_H/(2\hat\tau_c)$")
    ax1.text(FLO * 2.0, 2.0 * len(rows) + 1.2,
             r"swept over $\gamma\in[10^{-3},10^{-1}]$, $u_0\in[0.01,0.3]$, "
             r"$\epsilon\in[0.01,1]$; $\bullet$ = fiducial",
             fontsize=9.5, color="0.35", va="center")

    out = save_figure(fig, name)
    print(f"\nsaved {out}")
    return out


if __name__ == "__main__":
    main()
