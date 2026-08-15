#!/usr/bin/env python3
r"""Infrared branch diagram: a hard finite source lifetime turns $k^3$ into $k^1$.

Task 2.1 of ``ACTION_PLAN.md``.  Two independent referees asked for the same
figure: run the finite-coherence kernel of ``finite_coherence_gw.py`` far enough
into the infrared that the break predicted by Eq.(eq:window-factor) of
``derivation.tex`` is straddled on both sides, and measure the local logarithmic
slope across it.

What is being tested
--------------------
The source correlator is the TRIANGLE

    f(tau) = (1 - |tau|/tau_c) Theta(tau_c - |tau|),

i.e. the autocorrelation of a source switched on for a window tau_c.  (The
top-hat is inadmissible: its transform changes sign, violating Bochner, and
delivers negative GW energy.  It is available here through ``kind="tophat"`` and
is reported only as a contrast in ``--tophat``.)  The stress is quadratic in the
field, so by the product rule Eq.(eq:ft-of-product) the two-leg temporal factor
is the cosine transform of the PRODUCT of the two legs' correlators; for a common
lifetime that is the SQUARED tent, whose one-sided slope at zero lag is
f'(0+) = -2/tau_c.  The cusp theorem Eq.(eq:cusp-tail) then gives

    T(omega) -> -2 f'(0+)/omega^2 = 4/(tau_c omega^2),    omega tau_c >> 1,
    T(0)      = 2 tau_c/3,                                 omega tau_c << 1,

(divided here by tau_1 tau_2 = tau_c^2, an overall normalisation that carries no
p dependence).  On the sound cone omega = k the spatial integral is flat in the
infrared -- the white-noise floor of Sec.(sec:band-split) -- so

    Omega_GW ~ p^3 T(p tau_c k_0)  ->  p^3   below the break,
                                       p^1   above it,

the two powers being removed by the finite emission window exactly as in
Eq.(eq:window-factor).  Two exact statements follow for the squared tent and are
checked numerically below:

  * the two asymptotes intersect at omega tau_c = sqrt(6) = 2.449;
  * the local slope of T passes through -1 (i.e. Omega_GW through p^2, the
    midpoint of the crossover) at omega tau_c = pi EXACTLY -- the condition
    1 + cos u = 2 sin(u)/u is solved by u = pi.

So the break sits at k_break = pi/tau_c, a factor pi above the naive 1/tau_c.

Two lifetime models
-------------------
``tau_mode="global"``
    Every leg dies at the same time, tau_1 = tau_2 = tau_c: a genuinely hard,
    scale-independent lifetime.  This is the model of Eq.(eq:window-factor) and
    of the "constant in time, switched off abruptly" source of Auclair et al.
    and RoperPol et al.  The temporal factor then leaves the spatial integral
    entirely, so one spatial pass serves every tau_c (exercised as a validation).

``tau_mode="eddy"``
    tau_i = sqrt(x_i)/M, the scale-dependent eddy time of
    ``finite_coherence_gw.H_finite``, with M = sqrt(2 pi)/that.  This is the
    existing tool verbatim -- ``H_eddy`` is validated against it to machine
    precision in ``--validate`` -- and it smears the break because different
    legs stop radiating at different times.

The control parameter is that = tau_c k_0 as in Eq.(eq:that); for the eddy model
the quoted tau_c is the lifetime of the energy-containing modes,
tau_c = tau(k_0) = 1/M = that/sqrt(2 pi) in units of 1/k_0.

Everything spatial (grids, band masks, triangle bounds, kernel bracket) is
``band_split_gw``'s, unchanged, including the erfc correction of task 0.5 -- which
this calculation never touches, the Kraichnan temporal factor that carried the
erfc having been replaced wholesale by the finite-lifetime one.

Produces images/ir_branch_diagram.pdf.  ``--validate`` and ``--converge`` print
the checks; ``--tophat`` prints the inadmissible contrast.
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

from gw_turbulence.core import kernel_bracket  # noqa: E402
import band_split_gw as B  # noqa: E402
from finite_coherence_gw import temporal_factor  # noqa: E402

warnings.filterwarnings("ignore")

SQRT2PI = np.sqrt(2.0 * np.pi)

# Fiducial spatial input: full spectrum, Batchelor k^4 infrared band.
R_FID, R_IR_FID, IR_FID, BAND_FID = 1.0e4, 1.0e2, "batchelor", "both"
X_POINTS, Y_POINTS = 260, 200

# p range and density.  The break sits at p = pi/tau_c, so eight decades of p
# straddle every that in [0.1, 1e4] by at least 1.5 decades on the infrared side.
P_LO, P_HI, P_PER_DEC = 1.0e-6, 1.0e2, 24

THATS = (0.1, 1.0, 10.0, 100.0, 1000.0, 1.0e4)
THATS_EDDY = (1.0, 10.0, 100.0, 1000.0, 1.0e4)
_COLOR_IDX = (5, 2, 3, 1, 6, 7)

# Exact crossover constants for the squared tent (see module docstring).
U_ASYMPTOTE = np.sqrt(6.0)   # where the p^3 and p^1 asymptotes intersect
U_MIDPOINT = np.pi           # where the local slope passes through 2

# Upper edge of the windows used for the fits.  The spatial integral G(p) is flat
# to 5e-3 in log-log below p = 0.05, so a fit that stops there measures the
# temporal branch and not the roll-over towards the source-scale peak.
BAND_HI = 0.05
BAND_U_LO = 10.0             # start the band fit at omega tau_c = 10
IR_HI = 0.1                  # never fit the "infrared" above this p
SEPARATED = 0.1              # break counts as resolved only if p_trans < this


def p_grid(lo: float = P_LO, hi: float = P_HI, per_decade: int = P_PER_DEC):
    n = int(round(per_decade * np.log10(hi / lo))) + 1
    return np.geomspace(lo, hi, n)


# --------------------------------------------------------------------------- #
#  kernel: band_split's spatial integral with a finite-lifetime temporal factor
# --------------------------------------------------------------------------- #
def H_lifetime(p: float, q: float, that: float, kind: str = "triangle",
               tau_mode: str = "global", band: str = BAND_FID, R: float = R_FID,
               R_IR: float = R_IR_FID, ir: str = IR_FID,
               x_points: int = X_POINTS, y_points: int = Y_POINTS) -> float:
    r"""H(p,q) for a source with a hard finite lifetime.

    ``tau_mode``:
      ``"global"``  tau_1 = tau_2 = tau_c = that/k_0 (scale independent);
      ``"eddy"``    tau_i = sqrt(x_i)/M, M = sqrt(2 pi)/that, i.e. exactly
                    ``finite_coherence_gw.H_finite``;
      ``"none"``    temporal factor set to 1 -- the bare spatial integral G(p).
    """
    M = SQRT2PI / that
    ir_exp = B.IR_EXPONENTS[ir]
    p = max(float(p), 1e-12)
    u_floor, u_ceil = 1.0 / R_IR, R ** 0.75
    ks = B._outer_k_grid(u_floor, u_ceil, x_points, p, u_floor, u_ceil)
    xs = (ks ** (-4.0 / 3.0))[::-1]
    s1 = B.shape_band(xs ** (-0.75), ir_exp, band)
    if tau_mode == "global":
        tfac_const = temporal_factor(q, that, that, kind)
    acc = np.zeros_like(xs)
    for i, x in enumerate(xs):
        if s1[i] == 0.0:
            continue
        tk1 = x ** (-0.75)
        u_min, u_max = max(abs(tk1 - p), u_floor), min(tk1 + p, u_ceil)
        if not u_min < u_max:
            continue
        ys = B._split_grid(u_max ** (-4.0 / 3.0), u_min ** (-4.0 / 3.0), y_points)
        s2 = B.shape_band(ys ** (-0.75), ir_exp, band)
        geom = ys ** 0.75 * x ** 0.75 * kernel_bracket(p, x, ys)
        if tau_mode == "none":
            tfac = 1.0
        elif tau_mode == "global":
            tfac = tfac_const
        elif tau_mode == "eddy":
            tfac = temporal_factor(q, np.sqrt(x) / M, np.sqrt(ys) / M, kind)
        else:
            raise ValueError(f"tau_mode must be global/eddy/none, got {tau_mode!r}")
        acc[i] = np.trapezoid(geom * tfac * s1[i] * s2, ys)
    return float(np.trapezoid(acc, xs)) / p


def spatial_floor(ps, **kw):
    r"""G(p): the p-dependent spatial integral alone, temporal factor removed."""
    return np.array([H_lifetime(p, 0.0, 1.0, tau_mode="none", **kw) for p in ps])


def omega_global(ps, that: float, G, kind: str = "triangle"):
    r"""Omega_GW(p) = p^3 G(p) T(p that) -- the temporal factor factors out exactly."""
    ps = np.asarray(ps, float)
    return ps ** 3 * G * temporal_factor(ps, that, that, kind)


def omega_eddy(ps, that: float, kind: str = "triangle", **kw):
    r"""Omega_GW(p) with the scale-dependent eddy lifetimes of ``H_finite``."""
    return np.array([p ** 3 * H_lifetime(p, p, that, kind, tau_mode="eddy", **kw)
                     for p in ps])


def tau_c_of(that: float, tau_mode: str) -> float:
    """Coherence time of the energy-containing modes, in units of 1/k_0."""
    return that if tau_mode == "global" else that / SQRT2PI


# --------------------------------------------------------------------------- #
#  slope diagnostics
# --------------------------------------------------------------------------- #
def local_slope(ps, ys, half_window: int = 3):
    """d ln y / d ln p by a moving least-squares fit of +/- ``half_window`` nodes."""
    lx, ly = np.log(np.asarray(ps, float)), np.log(np.abs(np.asarray(ys, float)))
    out = np.full(lx.shape, np.nan)
    good = np.isfinite(ly)
    for i in range(lx.size):
        a, b = max(0, i - half_window), min(lx.size, i + half_window + 1)
        m = good[a:b]
        if m.sum() >= 3:
            out[i] = np.polyfit(lx[a:b][m], ly[a:b][m], 1)[0]
    return out


def fit_slope(ps, ys, lo: float, hi: float) -> float:
    """Least-squares log-log slope over p in [lo, hi]; NaN if under-sampled."""
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    m = (ps >= lo) & (ps <= hi) & (ys > 0)
    if m.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log(ps[m]), np.log(ys[m]), 1)[0])


def slope_crossing(ps, slopes, level: float = 2.0, p_max: float = 2.0) -> float:
    """First downward crossing of ``level`` by the local slope, below ``p_max``."""
    ps, s = np.asarray(ps, float), np.asarray(slopes, float)
    m = np.isfinite(s) & (ps <= p_max)
    ps, s = ps[m], s[m]
    idx = np.where((s[:-1] > level) & (s[1:] <= level))[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    f = (s[i] - level) / (s[i] - s[i + 1])
    return float(np.exp(np.log(ps[i]) + f * (np.log(ps[i + 1]) - np.log(ps[i]))))


def measure(ps, om, that: float, tau_mode: str, G=None) -> dict:
    """IR slope, intermediate-band slope, and break position for one curve.

    The infrared fit stops a decade below the break AND below p = IR_HI, so it
    can never see the source-scale roll-over.  The band fit runs from
    omega tau_c = BAND_U_LO up to p = BAND_HI for the same reason, and is
    reported only when that window is at least a quarter of a decade wide.
    ``band_lo/hi_slope`` bracket the pointwise local slope inside the window:
    the two-leg temporal factor is 2 tau_c (2/u^2 - 2 sin u / u^3), whose
    subleading term makes the local slope RING about -2 with an amplitude
    falling as 1/u, so the fit averages an oscillation that is physical (the
    ringing of an abrupt switch-off) and not numerical noise.
    """
    tc = tau_c_of(that, tau_mode)          # in units 1/k_0, so p_break = u/tc
    s = local_slope(ps, om)
    ir_hi = min(0.1 / tc, IR_HI)
    band_lo, band_hi = BAND_U_LO / tc, BAND_HI
    width = np.log10(band_hi / band_lo) if band_hi > band_lo else 0.0
    inband = (ps >= band_lo) & (ps <= band_hi) & np.isfinite(s)
    ok = width >= 0.25
    out = dict(
        that=that, tau_c=tc, p_break_naive=1.0 / tc,
        ir_slope=fit_slope(ps, om, ps[0], ir_hi),
        ir_decades=np.log10(ir_hi / ps[0]),
        band_slope=fit_slope(ps, om, band_lo, band_hi) if ok else np.nan,
        band_decades=width,
        band_lo_slope=np.min(s[inband]) if (ok and inband.any()) else np.nan,
        band_hi_slope=np.max(s[inband]) if (ok and inband.any()) else np.nan,
        p_trans=slope_crossing(ps, s),
        peak=B.peak_position(ps, om),
    )
    out["coeff"] = out["p_trans"] * tc
    out["separated"] = out["p_trans"] < SEPARATED
    if G is not None:                       # temporal branch with G(p) divided out
        out["band_slope_T"] = fit_slope(ps, om / G, band_lo, band_hi) if ok else np.nan
    return out


_HDR = ("  that     tau_c k0  1/tau_c    IR slope (dec)   band slope (dec)"
        "   band ringing    p_trans    coeff   peak")


def _row(r: dict) -> str:
    return (f"  {r['that']:<8.4g} {r['tau_c']:<9.4g}{r['p_break_naive']:<10.4g}"
            f"{r['ir_slope']:+8.3f} ({r['ir_decades']:.1f})"
            f"   {r['band_slope']:+8.3f} ({r['band_decades']:.1f})"
            f"   {r['band_lo_slope']:+5.2f},{r['band_hi_slope']:+5.2f}"
            f"  {r['p_trans']:9.4g}{r['coeff']:8.3f}{'' if r['separated'] else '*'}"
            f" {r['peak']:7.3g}")


# --------------------------------------------------------------------------- #
#  figure
# --------------------------------------------------------------------------- #
def main() -> None:
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import (
        PALETTE, apply_max_ticks, apply_paper_style, save_figure,
    )

    apply_paper_style(grid=False)
    ps = p_grid()
    colour = {t: PALETTE[i] for t, i in zip(THATS, _COLOR_IDX)}

    print(f"[ir_branch_diagram] spatial integral on {ps.size} nodes ...", flush=True)
    G = spatial_floor(ps)
    print(f"    G(p) flat to {np.ptp(G[ps < 1e-2]) / G[0]:.2e} over p < 1e-2"
          f"  (G = {G[0]:.6g})", flush=True)

    om_g = {t: omega_global(ps, t, G) for t in THATS}
    sl_g = {t: local_slope(ps, om_g[t]) for t in THATS}
    res_g = [measure(ps, om_g[t], t, "global", G=G) for t in THATS]

    om_e, sl_e, res_e = {}, {}, []
    for t in THATS_EDDY:
        print(f"[ir_branch_diagram] eddy-lifetime kernel, that={t:g} ...", flush=True)
        om_e[t] = omega_eddy(ps, t)
        sl_e[t] = local_slope(ps, om_e[t])
        res_e.append(measure(ps, om_e[t], t, "eddy"))

    print("\n" + "=" * 118)
    print("HARD GLOBAL LIFETIME  tau_1 = tau_2 = tau_c   (triangle correlator)")
    print("=" * 118 + "\n" + _HDR)
    for r in res_g:
        print(_row(r))
    print("\n  band slope with the spatial factor G(p) divided out:  "
          + ", ".join(f"that={r['that']:g}: {r['band_slope_T']:+.3f}"
                      for r in res_g if np.isfinite(r["band_slope_T"])))
    print("\n" + "=" * 118)
    print("SCALE-DEPENDENT LIFETIME  tau_i = sqrt(x_i)/M  (finite_coherence_gw.H_finite)")
    print("=" * 118 + "\n" + _HDR)
    for r in res_e:
        print(_row(r))
    print(f"\n  predicted coefficient: pi = {np.pi:.4f} (slope midpoint), "
          f"sqrt(6) = {U_ASYMPTOTE:.4f} (asymptote intersection)")
    print("  * = break not separated from the source scale (p_trans > "
          f"{SEPARATED:g}), coefficient contaminated by the peak roll-over\n")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), constrained_layout=True)

    # (a) the spectra themselves --------------------------------------------
    ax = axes[0, 0]
    norm = {}
    for t in THATS:
        y = om_g[t] / np.nanmax(om_g[t])
        norm[t] = y
        ax.loglog(ps, y, "-", color=colour[t], lw=1.3,
                  label=rf"$\hat\tau={t:g}$".replace("10000", "10^{4}"))
        pb = U_MIDPOINT / tau_c_of(t, "global")
        if ps[0] < pb < ps[-1]:
            ax.plot([pb], [np.interp(pb, ps, y)], "o", color=colour[t], ms=4)
    # guides: p^3 anchored on the long infrared of the short-lived source,
    # p^1 anchored on the wide intermediate band of the long-lived one
    for t_ref, power, span, off, tpos in ((1.0, 3, (1e-5, 3e-3), 0.03, 0.16),
                                          (1e4, 1, (1e-3, 5e-2), 25.0, 2.2)):
        m = (ps >= span[0]) & (ps <= span[1])
        anchor = np.interp(span[0], ps, norm[t_ref]) * off
        guide = anchor * (ps[m] / span[0]) ** power
        ax.loglog(ps[m], guide, ":", color="0.4", lw=1.1)
        j = m.sum() // 2
        ax.text(ps[m][j], guide[j] * tpos, rf"$p^{{{power}}}$", color="0.3",
                fontsize=13, ha="center", va="center")
    ax.set_xlim(ps[0], ps[-1])
    ax.set_ylim(1e-19, 8.0)
    ax.set_xlabel(r"$p = k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p)/\Omega_{\rm GW}^{\rm peak}$")
    ax.set_title(r"(a) hard lifetime, peak-normalised", fontsize=13)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    apply_max_ticks(ax, n=6)

    # (b) the local slope, the key panel ------------------------------------
    ax = axes[0, 1]
    ax.axhline(3.0, color="0.6", lw=0.9, ls=":")
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    for t in THATS:
        ax.semilogx(ps, sl_g[t], "-", color=colour[t], lw=1.3,
                    label=rf"$\hat\tau={t:g}$")
        pb = U_MIDPOINT / tau_c_of(t, "global")
        if ps[0] < pb < ps[-1]:
            ax.axvline(pb, color=colour[t], lw=0.7, ls="--", alpha=0.55)
    ax.set_xlim(ps[0], 3.0)
    ax.set_ylim(-0.5, 3.6)
    ax.set_xlabel(r"$p = k/k_0$")
    ax.set_ylabel(r"$d\ln\Omega_{\rm GW}/d\ln p$")
    ax.set_title(r"(b) local slope; dashes at $\pi/\tau_c$", fontsize=13)
    apply_max_ticks(ax, n=6)

    # (c) collapse in omega tau_c, both lifetime models ---------------------
    ax = axes[1, 0]
    ax.axhline(3.0, color="0.6", lw=0.9, ls=":")
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.axvline(U_MIDPOINT, color="0.35", lw=1.0)
    for t in THATS:
        ax.semilogx(ps * tau_c_of(t, "global"), sl_g[t], "-",
                    color=colour[t], lw=1.3)
    for t in THATS_EDDY:
        ax.semilogx(ps * tau_c_of(t, "eddy"), sl_e[t], "--",
                    color=colour[t], lw=1.1, alpha=0.85)
    ax.plot([], [], "-", color="0.35", lw=1.3, label="hard global lifetime")
    ax.plot([], [], "--", color="0.35", lw=1.1, label=r"eddy lifetime $\tau(k)$")
    ax.set_xlim(1e-3, 3e2)
    ax.set_ylim(-0.5, 3.6)
    ax.set_xlabel(r"$\omega\tau_c = p\,\tau_ck_0$")
    ax.set_ylabel(r"$d\ln\Omega_{\rm GW}/d\ln p$")
    ax.set_title(r"(c) collapse; vertical line $\omega\tau_c=\pi$", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    apply_max_ticks(ax, n=6)

    # (d) break slides, peak does not --------------------------------------
    ax = axes[1, 1]
    tt = np.array([r["tau_c"] for r in res_g], float)
    te = np.array([r["tau_c"] for r in res_e], float)
    tg = np.geomspace(min(tt.min(), te.min()), max(tt.max(), te.max()), 50)
    ax.loglog(tg, U_MIDPOINT / tg, "-", color="0.4", lw=1.0, label=r"$\pi/\tau_c$")
    ax.loglog(tt, [r["p_trans"] for r in res_g], "o", ms=5, color=PALETTE[5],
              label=r"measured break (global)")
    ax.loglog(te, [r["p_trans"] for r in res_e], "s", ms=5, mfc="none",
              color=PALETTE[6], label=r"measured break (eddy)")
    ax.loglog(tt, [r["peak"] for r in res_g], "^", ms=5, color=PALETTE[1],
              label=r"peak (global)")
    ax.loglog(te, [r["peak"] for r in res_e], "v", ms=5, mfc="none",
              color=PALETTE[3], label=r"peak (eddy)")
    ax.axhspan(1.0, 3.0, color="0.87", zorder=0)
    ax.text(1.3e3, 1.15, r"source scale", fontsize=9, color="0.35")
    ax.set_xlabel(r"$\tau_c k_0$")
    ax.set_ylabel(r"$p$")
    ax.set_title(r"(d) break slides, peak stays", fontsize=13)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    apply_max_ticks(ax, n=6)

    out = save_figure(fig, "ir_branch_diagram")
    print(f"[ir_branch_diagram] wrote {out}")


# --------------------------------------------------------------------------- #
#  checks
# --------------------------------------------------------------------------- #
def _validate() -> None:
    import finite_coherence_gw as F

    print("=" * 78)
    print("VALIDATION (1): tau_mode='eddy' == finite_coherence_gw.H_finite")
    print("=" * 78)
    print(f"  {'p':>9}{'this':>18}{'H_finite':>18}{'rel.diff':>12}")
    for p in (1e-4, 1e-2, 0.3, 1.0, 5.0):
        a = H_lifetime(p, p, 10.0, tau_mode="eddy")
        b = F.H_finite(p, p, 10.0, "triangle")
        print(f"  {p:9.3g}{a:18.9e}{b:18.9e}{abs(a / b - 1):12.2e}")

    print("\n" + "=" * 78)
    print("VALIDATION (2): tau_mode='global' factorises, H = G(p) T(p tau_c)")
    print("=" * 78)
    print(f"  {'p':>9}{'H(global)':>18}{'G*T':>18}{'rel.diff':>12}")
    for p in (1e-4, 1e-2, 1.0):
        a = H_lifetime(p, p, 10.0, tau_mode="global")
        b = H_lifetime(p, 0.0, 1.0, tau_mode="none") * temporal_factor(
            p, 10.0, 10.0, "triangle")
        print(f"  {p:9.3g}{a:18.9e}{b:18.9e}{abs(a / b - 1):12.2e}")

    print("\n" + "=" * 78)
    print("VALIDATION (3): squared-tent temporal factor against the cusp theorem")
    print("=" * 78)
    print("  T(0) tau_c = 2/3 ?   ", end="")
    print(f"{temporal_factor(1e-9, 1.0, 1.0, 'triangle'):.6f}  (exact 0.666667)")
    print(f"  {'omega tau_c':>12}{'T omega^2 tau_c^3':>20}   (exact 4 as u -> inf)")
    for u in (10.0, 30.0, 100.0, 300.0, 1000.0):
        print(f"  {u:12g}{temporal_factor(u, 1.0, 1.0, 'triangle') * u ** 2:20.4f}")
    print("\n  root of 1 + cos u = 2 sin(u)/u  (local slope of T equals -1):")
    uu = np.geomspace(1.0, 6.0, 200001)
    ff = 1.0 + np.cos(uu) - 2.0 * np.sin(uu) / uu
    i = int(np.argmin(np.abs(ff)))
    print(f"    numerical u = {uu[i]:.6f}   pi = {np.pi:.6f}")


def _converge() -> None:
    print("=" * 96)
    print("CONVERGENCE: quadrature mesh and p-grid density, triangle, that = 1000")
    print("=" * 96)
    that = 1000.0
    print(f"  {'mode':>7}{'x_pts':>7}{'y_pts':>7}{'p/dec':>7}"
          f"{'IR slope':>11}{'band slope':>12}{'p_trans':>11}{'coeff':>9}")
    rows = {}
    for mode in ("global", "eddy"):
        for xp, yp, ppd in ((130, 100, 24), (260, 200, 24), (520, 400, 24),
                            (260, 200, 16), (260, 200, 48)):
            ps = p_grid(per_decade=ppd)
            if mode == "global":
                G = spatial_floor(ps, x_points=xp, y_points=yp)
                om = omega_global(ps, that, G)
            else:
                om = omega_eddy(ps, that, x_points=xp, y_points=yp)
            r = measure(ps, om, that, mode)
            rows.setdefault(mode, []).append(r)
            print(f"  {mode:>7}{xp:7d}{yp:7d}{ppd:7d}"
                  f"{r['ir_slope']:+11.4f}{r['band_slope']:+12.4f}"
                  f"{r['p_trans']:11.5g}{r['coeff']:9.4f}", flush=True)
    print()
    for mode, rr in rows.items():
        for key in ("ir_slope", "band_slope", "coeff"):
            v = np.array([r[key] for r in rr], float)
            print(f"  {mode:>7}  {key:<11} spread = {np.ptp(v):.4f}"
                  f"   (min {v.min():+.4f}, max {v.max():+.4f})")

    print("\n  The residual spread in the transition coefficient is the width of the"
          "\n  moving slope estimator, not the quadrature: doubling x_pts/y_pts moves"
          "\n  it by <0.004, changing the p-density moves it by 0.13.  Because the"
          "\n  global model factorises exactly and G(p) is flat to 1e-3 across the"
          "\n  break, the estimator can be refined at no quadrature cost:")
    print(f"  {'p/dec':>8}{'coeff (G frozen)':>20}")
    for ppd in (16, 24, 48, 96, 192, 384):
        ps = p_grid(per_decade=ppd)
        om = omega_global(ps, that, np.ones_like(ps))
        print(f"  {ppd:8d}{slope_crossing(ps, local_slope(ps, om)) * that:20.4f}")
    print(f"  {'exact':>8}{np.pi:20.4f}   (root of 1 + cos u = 2 sin u / u)")


def _tophat() -> None:
    print("=" * 78)
    print("CONTRAST: the top-hat correlator is INADMISSIBLE (Bochner)")
    print("=" * 78)
    ps = p_grid()
    G = spatial_floor(ps)
    print(f"  {'that':>8}{'Omega<0 fraction':>20}{'IR slope':>11}"
          f"{'band slope':>12}   (both meaningless where Omega<0)")
    for t in (1.0, 10.0, 100.0, 1000.0):
        om = omega_global(ps, t, G, kind="tophat")
        r = measure(ps, om, t, "global")
        print(f"  {t:8g}{np.mean(om < 0):20.1%}{r['ir_slope']:+11.3f}"
              f"{r['band_slope']:+12.3f}")
    print("\n  The transform of the top-hat product is 2 sin(omega a)/omega:"
          "\n  it changes sign at every omega a = n pi, so Omega_GW is negative on"
          "\n  roughly half the axis and no slope may be quoted.  The triangle is"
          "\n  used everywhere else in this script.")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        _validate()
    elif "--converge" in sys.argv:
        _converge()
    elif "--tophat" in sys.argv:
        _tophat()
    else:
        main()
