#!/usr/bin/env python3
r"""Band-split experiment: which part of the fluid spectrum sources the GW infrared?

The fluid input of Eq.(full-input-spectrum) has two bands separated by the
stirring scale k0,

    E(k) ~ k^{s}      (k_IR <= k <  k0)   causal / Batchelor band, s = 4
    E(k) ~ k^{-5/3}   (k0   <= k <= k_d)  Kolmogorov inertial range,

and the GW source is the *quadratic* stress T_ij ~ v_i v_j, so the stress
spectrum is the SELF-CONVOLUTION of the fluid spectrum.  The hypothesis under
test is that for k << k0 that convolution is saturated by the antiparallel pairs
k1 ~ -k2 with |k1| ~ |k2| near the top of whichever band is populated, so the
stress floors at WHITE NOISE no matter how steep (or absent) the fluid infrared
is -- and hence Omega_GW ~ k^3 universally.

We therefore run the same stationary kernel three times, feeding it

    band = "ir"        only the k^4 band          (S = 0 for k >= k0)
    band = "inertial"  only the k^{-5/3} range    (S = 0 for k <  k0)
    band = "both"      the full spectrum          (= _fullspectrum_kernel.H_full)

Implementation
--------------
Everything kinematic and temporal is core's, verbatim, exactly as in
``_fullspectrum_kernel.H_full``; the ONLY change is the spectral shape factor
S(k) = A(k)/A_Kol(k) applied identically to both convolution legs k1 and u.
Band selection multiplies S by the band's indicator.  The physical integration
bounds -- the triangle inequality |k1 - p| <= u <= k1 + p intersected with the
spectral support [k_IR, k_d] -- are NEVER narrowed; the band is imposed by
zeroing the integrand, which is safe.  Because that zeroing puts a step at
k = k0 in both integration variables, the quadrature grids carry a node pair
straddling the step (``_split_grid``), which is a refinement of the mesh and not
a change of the domain.

Invariant: ``H_band(..., band="both", R_IR=1)`` == ``core.H_pq`` to <~1e-3, and
``H_band(..., band="inertial")`` == ``core.H_pq`` for ANY R_IR (masking the IR
band away must reproduce the paper's original inertial-range-only calculation).
Both are checked in ``_validate()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import special

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from gw_turbulence.core import _h_prefactor, kernel_bracket  # noqa: E402
from _fullspectrum_kernel import IR_EXPONENTS  # noqa: E402

BANDS = ("ir", "inertial", "both")

BAND_LABELS = {
    "ir": r"$k^{4}$ band only",
    "inertial": r"$k^{-5/3}$ band only",
    "both": r"full spectrum",
}

# Fiducial parameters of the experiment (quoted in the caption, not on the plot).
M_FID = 0.5
R_FID = 1.0e4
R_IR_FID = 1.0e2

_EPS = 1.0e-9  # half-width of the node pair straddling a step in the integrand


# --------------------------------------------------------------------------- #
#  spectral shape factor with a band mask
# --------------------------------------------------------------------------- #
def shape_band(tilde_k, ir_exp: float, band: str = "both"):
    r"""S(k) = A(k)/A_Kol(k) restricted to ``band``;  tilde_k = k/k0.

    ``both``      -> 1 above k0, (k/k0)^{ir_exp} below  (continuous at k0)
    ``inertial``  -> 1 above k0, 0 below
    ``ir``        -> 0 above k0, (k/k0)^{ir_exp} below
    """
    tk = np.asarray(tilde_k, dtype=float)
    upper = tk >= 1.0
    if band == "both":
        return np.where(upper, 1.0, tk ** ir_exp)
    if band == "inertial":
        return np.where(upper, 1.0, 0.0)
    if band == "ir":
        return np.where(upper, 0.0, tk ** ir_exp)
    raise ValueError(f"band must be one of {BANDS}, got {band!r}")


def _split_grid(lo: float, hi: float, n: int, edge: float = 1.0) -> np.ndarray:
    """Geometric grid on [lo, hi] with a node pair straddling ``edge`` if interior.

    Refines the quadrature mesh across the band step; the domain is unchanged.
    """
    if not (lo < edge < hi):
        return np.geomspace(lo, hi, n)
    lo_e, hi_e = edge * (1.0 - _EPS), edge * (1.0 + _EPS)
    span_lo = np.log(lo_e / lo)
    span_hi = np.log(hi / hi_e)
    frac = span_lo / (span_lo + span_hi)
    n_lo = int(np.clip(round(frac * n), 16, n - 16))
    n_hi = n - n_lo
    return np.concatenate([
        np.geomspace(lo, lo_e, n_lo),
        np.geomspace(hi_e, hi, n_hi),
    ])


def _outer_k_grid(k_lo: float, k_hi: float, n: int, p: float,
                  u_floor: float, u_ceil: float, n_ref: int = 64) -> np.ndarray:
    r"""Outer-leg grid in k1/k0, log-uniform plus local refinement at the kinks.

    The inner (u) integral runs over ``[max(|k1-p|, k_IR), min(k1+p, k_d)]`` and
    carries a step in S(u) at u = k0, so as a function of k1 it has weak kinks
    where a clamp activates or where the window crosses the band edge, i.e. at
    k1 = k0 +/- p, k1 = |k_IR -/+ p| and k1 = k_d -/+ p, plus the step in S(k1)
    itself at k1 = k0.  Those features are O(p) wide, so at small p a plain
    log-uniform mesh under-resolves them; we add a narrow cluster around each.
    """
    grids = [np.geomspace(k_lo, k_hi, n)]
    criticals = {1.0, 1.0 - p, 1.0 + p,
                 u_floor + p, abs(u_floor - p),
                 u_ceil - p, u_ceil + p}
    for kc in criticals:
        if not (k_lo < kc < k_hi):
            continue
        half = max(3.0 * p, 1e-3 * kc)
        a, b = max(k_lo, kc - half), min(k_hi, kc + half)
        if a < b:
            grids.append(np.geomspace(a, b, n_ref))
    ks = np.unique(np.concatenate(grids))
    if k_lo < 1.0 < k_hi:                       # straddle the band step exactly
        ks = ks[ks != 1.0]
        ks = np.unique(np.concatenate([ks, [1.0 - _EPS, 1.0 + _EPS]]))
    return ks


# --------------------------------------------------------------------------- #
#  kernel
# --------------------------------------------------------------------------- #
def H_band(p: float, q: float, M: float = M_FID, R: float = R_FID,
           R_IR: float = R_IR_FID, ir: str = "batchelor", band: str = "both",
           temporal: str = "sweeping",
           x_points: int = 320, y_points: int = 240) -> float:
    r"""Stationary GW kernel H(p,q) sourced by ``band`` of the fluid spectrum only.

    Identical to ``_fullspectrum_kernel.H_full`` except that the shape factor is
    ``shape_band(..., band)``.  ``R_IR = k0/k_IR``, ``R = (k_d/k0)^{4/3}``.

    ``temporal``:
      ``"sweeping"``  Kraichnan; the Gaussian, the erfc and the (x+y)^{-1/2}
                      weight are all present.  ``q`` is used.
      ``"delta"``     the source fires at a single epoch,
                      Pi(k;t1,t2) = P(k) delta(t1-t0) delta(t2-t0).  Both deltas
                      collapse and cos[k(t1-t2)] -> cos 0 = 1, so the temporal
                      factor is identically unity FOR ANY t0 -- the burst epoch
                      cancels exactly and ``q`` is ignored.  Note this drops the
                      (x+y)^{-1/2} weight as well: that factor is the
                      sqrt(pi/(A+B)) of the Gaussian omega_1 convolution and is
                      part of the temporal model, not of the spatial one.  Hence
                      "delta" is NOT the q -> 0 limit of "sweeping"; the two
                      agree only where k1 and u are pinned to a single shell.
    """
    if temporal not in ("sweeping", "delta"):
        raise ValueError(f"temporal must be 'sweeping' or 'delta', got {temporal!r}")
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    ir_exp = IR_EXPONENTS[ir]
    p = max(float(p), 1e-10)

    u_floor = 1.0 / R_IR                        # k_IR/k0
    u_ceil = R ** 0.75                          # k_d/k0
    # x = (k1/k0)^{-4/3} in [1/R, R_IR^{4/3}]  <->  k1/k0 in [k_IR/k0, k_d/k0]
    ks = _outer_k_grid(u_floor, u_ceil, x_points, p, u_floor, u_ceil)
    xs = (ks ** (-4.0 / 3.0))[::-1]                       # ascending in x
    s1 = shape_band(xs ** (-0.75), ir_exp, band)          # S(k1) on the outer grid
    x_integrand = np.zeros_like(xs)

    for i, x in enumerate(xs):
        if s1[i] == 0.0:
            continue
        tk1 = x ** (-0.75)                                # k1/k0
        u_min = max(abs(tk1 - p), u_floor)
        u_max = min(tk1 + p, u_ceil)
        if not (u_min < u_max):
            continue
        y_min, y_max = u_max ** (-4.0 / 3.0), u_min ** (-4.0 / 3.0)
        ys = _split_grid(y_min, y_max, y_points)          # y = (u/k0)^{-4/3}
        s2 = shape_band(ys ** (-0.75), ir_exp, band)      # S(u)
        ss = x + ys
        geom = ys ** 0.75 * x ** 0.75 * kernel_bracket(p, x, ys)
        if temporal == "delta":
            tfac = 1.0
        else:
            tfac = (ss ** (-0.5)
                    * np.exp(-2.0 * x * ys / ss * q ** 2 / M ** 2)
                    * special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(ss))))
        x_integrand[i] = np.trapezoid(geom * tfac * s1[i] * s2, ys)

    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def omega_gw(p, M: float = M_FID, R: float = R_FID, R_IR: float = R_IR_FID,
             ir: str = "batchelor", band: str = "both", **kw):
    r"""Omega_GW(p) ~ p^3 H(p,p) on the sound-cone diagonal q = p (arbitrary units)."""
    ps = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.array([pp ** 3 * H_band(pp, pp, M=M, R=R, R_IR=R_IR, ir=ir,
                                     band=band, **kw) for pp in ps])
    return out if np.ndim(p) else float(out[0])


# --------------------------------------------------------------------------- #
#  fluid input spectrum (panel a) and diagnostics
# --------------------------------------------------------------------------- #
def source_spectrum(tilde_k, ir: str = "batchelor", band: str = "both",
                    R: float = R_FID, R_IR: float = R_IR_FID):
    r"""E(k)/E(k0) fed into the kernel: S(k) (k/k0)^{-5/3} on the band's support."""
    tk = np.asarray(tilde_k, dtype=float)
    support = (tk >= 1.0 / R_IR) & (tk <= R ** 0.75)
    e = shape_band(tk, IR_EXPONENTS[ir], band) * tk ** (-5.0 / 3.0)
    return np.where(support, e, np.nan)


def fit_slope(ps, ys, lo: float, hi: float) -> float:
    """Least-squares log-log slope of ys(ps) over ps in [lo, hi]."""
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    m = (ps >= lo) & (ps <= hi) & (ys > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(ps[m]), np.log(ys[m]), 1)[0])


def peak_position(ps, ys) -> float:
    """Peak of Omega_GW by a parabolic fit in log-log around the discrete maximum."""
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    i = int(np.nanargmax(ys))
    if i in (0, len(ps) - 1):
        return float(ps[i])
    lx, ly = np.log(ps[i - 1:i + 2]), np.log(ys[i - 1:i + 2])
    a, b, _ = np.polyfit(lx, ly, 2)
    return float(np.exp(-b / (2.0 * a))) if a < 0 else float(ps[i])


# --------------------------------------------------------------------------- #
#  figure
# --------------------------------------------------------------------------- #
def main(n_p: int = 46, p_lo: float = 1e-3, p_hi: float = 3e1,
         ir_fit: tuple[float, float] = (2e-3, 2e-2)) -> None:
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import (
        PALETTE, apply_max_ticks, apply_paper_style, save_figure,
    )

    apply_paper_style()
    ps = np.geomspace(p_lo, p_hi, n_p)
    colours = {"ir": PALETTE[5], "inertial": PALETTE[6], "both": PALETTE[0]}
    styles = {"ir": "--", "inertial": "-.", "both": "-"}

    # Kraichnan sweeping on the diagonal q = p, versus a source that fires at a
    # single epoch, Pi(k;t1,t2) = P(k) delta(t1-t0) delta(t2-t0).  For the latter
    # both deltas collapse and cos[k(t1-t2)] -> 1, so the temporal factor is
    # identically unity FOR ANY t0.  Same spatial integral either way, so the
    # pair isolates exactly what the temporal model controls.
    def _omega_delta(ps_, band):
        return np.array([pp ** 3 * H_band(pp, 0.0, band=band, temporal="delta")
                         for pp in ps_])

    spectra, slopes, peaks = {}, {}, {}
    for band in BANDS:
        print(f"[band_split_gw] computing band={band!r} ...", flush=True)
        spectra[band] = omega_gw(ps, band=band)
        slopes[band] = fit_slope(ps, spectra[band], *ir_fit)
        peaks[band] = peak_position(ps, spectra[band])
        print(f"    IR slope over p in [{ir_fit[0]:g}, {ir_fit[1]:g}]"
              f" = {slopes[band]:+.3f}   peak p = {peaks[band]:.3g}"
              f"   Omega(peak) = {np.nanmax(spectra[band]):.3e}", flush=True)

    nocorr, nc_slopes = {}, {}
    for band in ("ir", "inertial"):
        print(f"[band_split_gw] computing band={band!r} with delta(t-t0) source ...",
              flush=True)
        nocorr[band] = _omega_delta(ps, band)
        nc_slopes[band] = fit_slope(ps, nocorr[band], *ir_fit)
        print(f"    IR slope = {nc_slopes[band]:+.3f}"
              f"   peak p = {peak_position(ps, nocorr[band]):.3g}", flush=True)

    def _pos(y):                       # blank out the empty support on a log axis
        y = np.asarray(y, float).copy()
        y[~(y > 0)] = np.nan
        return y

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))

    # (a) the three fluid inputs
    ax = axes[0]
    ks = np.geomspace(1.0 / R_IR_FID, R_FID ** 0.75, 800)
    ax.loglog(ks, _pos(source_spectrum(ks, band="both")), "-",
              color=colours["both"], lw=3.4, alpha=0.30, label=BAND_LABELS["both"])
    for band in ("ir", "inertial"):
        ax.loglog(ks, _pos(source_spectrum(ks, band=band)), styles[band],
                  color=colours[band], lw=1.7, label=BAND_LABELS[band])
    ax.axvline(1.0, color="0.55", lw=0.8, ls=":")
    ax.set_xlabel(r"$k/k_0$")
    ax.set_ylabel(r"$E(k)/E(k_0)$")
    ax.set_title(r"(a) fluid input")
    ax.set_ylim(3e-9, 6.0)
    apply_max_ticks(ax)
    ax.legend(loc="lower center", frameon=False, fontsize=10)

    # (b) the three GW spectra
    ax = axes[1]
    ax.loglog(ps, _pos(spectra["both"]), "-", color=colours["both"], lw=3.4,
              alpha=0.30, label=rf"{BAND_LABELS['both']}: $p^{{{slopes['both']:.2f}}}$")
    for band in ("ir", "inertial"):
        ax.loglog(ps, _pos(spectra[band]), "-", color=colours[band], lw=1.7,
                  label=rf"{BAND_LABELS[band]} $+\,T_{{\rm sw}}$: $p^{{{slopes[band]:.2f}}}$")
    for band in ("ir", "inertial"):
        ax.loglog(ps, _pos(nocorr[band]), "--", color=colours[band], lw=1.4, alpha=0.85,
                  label=rf"{BAND_LABELS[band]} $+\,\delta(t-t_0)$: $p^{{{nc_slopes[band]:.2f}}}$")
    i_ref = int(np.argmin(np.abs(ps - 3e-3)))
    guide = spectra["ir"][i_ref] * (ps / ps[i_ref]) ** 3
    sel = (ps >= 1.2e-3) & (ps <= 4e-2)
    ax.loglog(ps[sel], 0.30 * guide[sel], color="0.5", lw=1.0, ls=":")
    ax.text(4.5e-2, 0.30 * guide[sel][-1] * 0.9, r"$p^{3}$", color="0.35",
            fontsize=12, ha="left", va="top")
    ax.set_xlabel(r"$p = k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p) \propto p^{3}\,H(p,p)$")
    ax.set_title(r"(b) sourced GW spectrum")
    ax.set_ylim(1e-12, 3e-1)
    apply_max_ticks(ax, n=6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)

    fig.tight_layout()
    out = save_figure(fig, "band_split_gw")
    print(f"[band_split_gw] wrote {out}")

    print("\n  band        IR slope    peak p    Omega(peak)")
    for band in BANDS:
        print(f"  {band:<10}{slopes[band]:+9.3f}{peaks[band]:10.3g}"
              f"{np.nanmax(spectra[band]):14.3e}")


# --------------------------------------------------------------------------- #
#  validation
# --------------------------------------------------------------------------- #
def _validate() -> None:
    from gw_turbulence.core import H_pq
    from _fullspectrum_kernel import H_full

    print("=" * 74)
    print('VALIDATION (1): band="both", R_IR=1  ==  core.H_pq')
    print("=" * 74)
    print(f"  {'p':>6}{'H_band':>16}{'core.H_pq':>16}{'rel.diff':>12}")
    for p in (0.01, 0.3, 1.0, 3.0):
        a = H_band(p, p, M=1.0, R=1e4, R_IR=1.0, band="both")
        b = H_pq(p, p, M=1.0, R=1e4)
        print(f"  {p:6.2f}{a:16.6e}{b:16.6e}{abs(a / b - 1):12.2e}")

    print("\n" + "=" * 74)
    print('VALIDATION (2): band="inertial" is R_IR-independent and == core.H_pq')
    print("=" * 74)
    print(f"  {'p':>6}{'R_IR=1':>16}{'R_IR=100':>16}{'core.H_pq':>16}{'rel.diff':>12}")
    for p in (0.01, 0.3, 1.0):
        a = H_band(p, p, M=1.0, R=1e4, R_IR=1.0, band="inertial")
        c = H_band(p, p, M=1.0, R=1e4, R_IR=100.0, band="inertial")
        b = H_pq(p, p, M=1.0, R=1e4)
        print(f"  {p:6.2f}{a:16.6e}{c:16.6e}{b:16.6e}{abs(c / b - 1):12.2e}")

    print("\n" + "=" * 74)
    print('VALIDATION (3): band="both" == H_full (same shape factor, refined mesh)')
    print("=" * 74)
    print(f"  {'p':>6}{'H_band':>16}{'H_full':>16}{'rel.diff':>12}")
    for p in (0.01, 0.3, 1.0):
        a = H_band(p, p, M=1.0, R=1e4, R_IR=100.0, band="both")
        b = H_full(p, p, M=1.0, R=1e4, R_IR=100.0)
        print(f"  {p:6.2f}{a:16.6e}{b:16.6e}{abs(a / b - 1):12.2e}")

    print("\n" + "=" * 74)
    print("VALIDATION (4): quadrature convergence (mesh doubled), M=0.5")
    print("=" * 74)
    print(f"  {'band':>10}{'p':>7}{'coarse':>16}{'fine':>16}{'rel.diff':>12}")
    for band in BANDS:
        for p in (3e-3, 1.0):
            a = H_band(p, p, band=band)
            b = H_band(p, p, band=band, x_points=640, y_points=480)
            print(f"  {band:>10}{p:7.3g}{a:16.6e}{b:16.6e}{abs(a / b - 1):12.2e}")


def _diagnostics() -> None:
    """Extra checks on the white-noise-floor claim beyond the figure."""
    ps = np.geomspace(1e-3, 3e1, 46)
    ir_fit = (2e-3, 2e-2)
    sp = {b: omega_gw(ps, band=b) for b in BANDS}

    print("=" * 74)
    print("DIAGNOSTIC (1): additivity of the two bands (cross term)")
    print("=" * 74)
    print(f"  {'p':>8}{'IR only':>13}{'inertial':>13}{'sum':>13}{'full':>13}{'full/sum':>10}")
    for p in (3e-3, 1e-2, 1e-1, 0.5, 1.0, 3.0):
        i = int(np.argmin(np.abs(ps - p)))
        a, b, c = sp["ir"][i], sp["inertial"][i], sp["both"][i]
        print(f"  {ps[i]:8.3g}{a:13.3e}{b:13.3e}{a + b:13.3e}{c:13.3e}{c / (a + b):10.3f}")

    print("\n" + "=" * 74)
    print('DIAGNOSTIC (2): band="ir" GW infrared vs the IR bandwidth R_IR')
    print("=" * 74)
    print(f"  {'R_IR':>8}{'IR slope':>11}{'Omega(p=5e-3)':>16}{'peak p':>9}")
    pp = np.geomspace(1e-3, 3e1, 34)
    for r in (10.0, 30.0, 100.0, 300.0):
        y = omega_gw(pp, R_IR=r, band="ir")
        j = int(np.argmin(np.abs(pp - 5e-3)))
        print(f"  {r:8.0f}{fit_slope(pp, y, *ir_fit):+11.3f}{y[j]:16.3e}"
              f"{peak_position(pp, y):9.3g}")

    print("\n" + "=" * 74)
    print('DIAGNOSTIC (3): band="ir" GW infrared vs the fluid IR slope')
    print("=" * 74)
    print(f"  {'fluid E(k)':>22}{'GW IR slope':>14}{'Omega(p=5e-3)':>16}")
    for name, lab in (("saffman", "k^2 (Saffman)"), ("batchelor", "k^4 (Batchelor)")):
        y = omega_gw(pp, ir=name, band="ir")
        j = int(np.argmin(np.abs(pp - 5e-3)))
        print(f"  {lab:>22}{fit_slope(pp, y, *ir_fit):+14.3f}{y[j]:16.3e}")

    print("\n" + "=" * 74)
    print("DIAGNOSTIC (4): local log-log slope of the full run, decade by decade")
    print("=" * 74)
    for lo, hi in ((1e-3, 1e-2), (2e-3, 2e-2), (5e-3, 5e-2), (1e-2, 1e-1)):
        row = "".join(f"{fit_slope(ps, sp[b], lo, hi):+10.3f}" for b in BANDS)
        print(f"  p in [{lo:7.1e},{hi:8.1e}]   " + "  ".join(BANDS) + " ->" + row)


if __name__ == "__main__":
    if "--validate" in sys.argv:
        _validate()
    elif "--diagnostics" in sys.argv:
        _diagnostics()
    else:
        main()
