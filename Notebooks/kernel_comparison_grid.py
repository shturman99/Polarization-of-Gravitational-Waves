#!/usr/bin/env python3
r"""Side-by-side survey of all nine kernels of the model grid on one figure.

Produces images/kernel_comparison_grid.pdf: a 3x3 grid, one panel per entry of
Table~\ref{tab:model-grid}, showing the dimensionless GW spectrum

    Omega_GW(p)  ~  p^3 H[S,T](p, p),        p = k / k0,

on the sound-cone diagonal q = p, so the abscissa of every panel is the GW
wavenumber in units of the source (initial-power) peak k0.

Control parameter
-----------------
Every temporal model carries one characteristic time.  We use the SAME control
in all panels,

    that = tau_c * k0                        (source coherence time in units of
                                              the light-crossing time of 1/k0)

so that the panels are directly comparable.  On the diagonal omega = p*k0, so
the dimensionless temporal argument of every kernel is simply

    omega * tau_c = p * that.

For the sweeping (Kraichnan) models tau_c = tau_{k0} = sqrt(2 pi)/(M k0), i.e.
    M = sqrt(2 pi) / that,
and for the finite burst tau_c = Dt, giving the low-pass |g~|^2 = exp(-p^2 that^2)
for a Gaussian profile of width Dt.

  * STATIONARY / time-independent kernels (T_sw, T_imp) -> a single line at
    that = 1 (coherence time equal to the light-crossing time of the source
    scale; M = sqrt(2 pi) = 2.51 for the sweeping panels).
  * TIME-DEPENDENT kernels (T_dec, T_burst) -> five lines,
    that = 0.01, 0.1, 1, 10, 100.

Normalisation
-------------
Each boxed H drops a different dimensional prefactor, so absolute amplitudes are
not comparable across panels.  Every panel is normalised by the PEAK of its own
reference curve (that = 1), which keeps the amplitude trend with that visible
within a panel while making shapes comparable between panels.
"""
from __future__ import annotations

import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gw_turbulence import core  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)
from _fullspectrum_kernel import H_full  # noqa: E402

R_FID = 1e4
SQRT2PI = np.sqrt(2.0 * np.pi)
THATS = (0.01, 0.1, 1.0, 10.0, 100.0)
THAT_REF = 1.0

#: p grid.  The two monochromatic spatial models vanish for p > 2 (triangle
#: inequality), so the grid is dense below 2 and still resolves the K41/white UV.
P_GRID = np.geomspace(1e-3, 20.0, 44)

_TCOLOR = {0.01: PALETTE[5], 0.1: PALETTE[2], 1.0: PALETTE[0],
           10.0: PALETTE[1], 100.0: PALETTE[6]}


# ---------------------------------------------------------------- kernels ----
def _k41_sw(p, that):
    return core.H_pq(p, p, M=SQRT2PI / that, R=R_FID)


def _k41_dec(p, that):
    return core.H_pq_decaying(p, p, M=SQRT2PI / that, R=R_FID)


def _dk_sw(p, that):
    # H_delta_k_kraichnan takes Omega = omega/eta0 = p*that
    return float(core.H_delta_k_kraichnan(p, p * that))


def _dk_dec(p, that):
    return core.H_delta_k_decay(p, p * that)


def _dr_sw(p, that):
    return core.H_white_kraichnan(p, p * that, R=R_FID)


def _dr_dec(p, that):
    return core.H_white_decay(p, p * that, R=R_FID)


def _dk_imp(p, that):
    return core.K0_p(p) / p if 0 < p <= 2.0 else 0.0


def _dk_burst(p, that):
    if not (0 < p <= 2.0):
        return 0.0
    return core.K0_p(p) / p * np.exp(-(p * that) ** 2)


def _full_sw(p, that):
    return H_full(p, p, M=SQRT2PI / that, R=R_FID, R_IR=1e2, ir="batchelor")


#: (key, callable, LaTeX title, time-dependent?)
POWERLAW_UV = {"k41_dec", "dk_dec", "dr_dec"}

PANELS = [
    ("k41_sw",   _k41_sw,   r"$H[S_{\rm K41},T_{\rm sw}]$",   False),
    ("dk_sw",    _dk_sw,    r"$H[S_{\delta k},T_{\rm sw}]$",  False),
    ("dr_sw",    _dr_sw,    r"$H[S_{\delta r},T_{\rm sw}]$",  False),
    ("k41_dec",  _k41_dec,  r"$H[S_{\rm K41},T_{\rm dec}]$",  True),
    ("dk_dec",   _dk_dec,   r"$H[S_{\delta k},T_{\rm dec}]$", True),
    ("dr_dec",   _dr_dec,   r"$H[S_{\delta r},T_{\rm dec}]$", True),
    ("full_sw",  _full_sw,  r"$H[S_{\rm full},T_{\rm sw}]$",  False),
    ("dk_imp",   _dk_imp,   r"$H[S_{\delta k},T_{\rm imp}]$", False),
    ("dk_burst", _dk_burst, r"$H[S_{\delta k},T_{\rm burst}]$", True),
]


def _omega(fn, that, ps):
    """Omega_GW(p) = p^3 H(p,p) for one curve."""
    return np.array([p ** 3 * fn(p, that) for p in ps])


def _omega_par(fn, that, ps, workers=8):
    with Pool(workers) as pool:
        vals = pool.map(partial(fn, that=that), ps)
    return ps ** 3 * np.array(vals)


# --------------------------------------------------------------- analysis ----
def _slope(ps, ys, lo, hi):
    """Local log-log slope fitted over p in [lo, hi]."""
    m = (ps >= lo) & (ps <= hi) & (ys > 0) & np.isfinite(ys)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(ps[m]), np.log(ys[m]), 1)[0])


def _peak(ps, ys):
    m = (ys > 0) & np.isfinite(ys)
    return float(ps[m][np.argmax(ys[m])]) if m.any() else np.nan


# ------------------------------------------------------------------ figure ---
def load(cache: Path):
    z = np.load(cache)
    ps = z["ps"]
    data: dict[str, dict[float, np.ndarray]] = {}
    for k in z.files:
        if k == "ps":
            continue
        key, that = k.split("|")
        data.setdefault(key, {})[float(that)] = z[k]
    return ps, data


def build(cache: Path | None = None):
    ps = P_GRID
    data: dict[str, dict[float, np.ndarray]] = {}
    for key, fn, _, tdep in PANELS:
        thats = THATS if tdep else (THAT_REF,)
        data[key] = {}
        for that in thats:
            print(f"  {key}  that={that}", flush=True)
            if key == "k41_dec":                    # only slow kernel
                data[key][that] = _omega_par(fn, that, ps)
            else:
                data[key][that] = _omega(fn, that, ps)
    if cache is not None:
        np.savez(cache, **{f"{k}|{t}": v for k, d in data.items() for t, v in d.items()},
                 ps=ps)
    return ps, data


def _limits(ps, data):
    """Axis limits that put every curve of every panel on scale, identical in all
    panels.  The floor is set four decades below the FAINTEST curve peak (not
    below the faintest data point): the Gaussian-cutoff kernels dive to 1e-277,
    so bracketing their tails is neither possible nor meaningful, whereas every
    curve's peak and the structure around it must be visible."""
    peaks = []
    xlo, xhi = np.inf, -np.inf
    for key, _, _, tdep in PANELS:
        curves = data[key]
        allv = np.concatenate(list(curves.values()))
        norm = np.nanmax(allv[allv > 0])
        for that in (THATS if tdep else (THAT_REF,)):
            y = curves[that] / norm
            m = (y > 0) & np.isfinite(y)
            if m.any():
                peaks.append(y[m].max())
                xlo = min(xlo, ps[m].min())
                xhi = max(xhi, ps[m].max())
    ymin = 10.0 ** np.floor(np.log10(min(peaks)) - 4.0)
    return (xlo / 1.3, xhi * 1.3), (ymin, 3.0)


def plot(ps, data):
    apply_paper_style(grid=False)
    xlim, ylim = _limits(ps, data)
    print(f"  common axes: x in [{xlim[0]:.3g}, {xlim[1]:.3g}], "
          f"y in [{ylim[0]:.3g}, {ylim[1]:.3g}] "
          f"({np.log10(ylim[1] / ylim[0]):.0f} decades)")
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.6), sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, (key, _, title, tdep) in zip(axes.flat, PANELS):
        curves = data[key]
        ref = curves[THAT_REF]
        allv = np.concatenate([v for v in curves.values()])
        norm = np.nanmax(allv[allv > 0]) if np.any(allv > 0) else 1.0
        for that in (THATS if tdep else (THAT_REF,)):
            y = curves[that] / norm
            yy = np.where(y > 0, y, np.nan)
            ax.loglog(ps, yy, color=_TCOLOR[that], lw=1.3,
                      label=rf"$\hat\tau={that:g}$" if tdep else None)
            pk = _peak(ps, curves[that])
            if np.isfinite(pk):
                ax.plot([pk], [np.nanmax(yy)], "o", ms=3.0, color=_TCOLOR[that])
        # p^3 causal reference, anchored well inside the IR
        i0 = 2
        if ref[i0] > 0:
            ax.loglog(ps, (ref[i0] / norm) * (ps / ps[i0]) ** 3,
                      ls=":", color="0.55", lw=0.8)
        sir = _slope(ps, ref, ps[0], 3e-2)
        pk = _peak(ps, ref)
        txt = rf"IR $p^{{{sir:.2f}}}$" if np.isfinite(sir) else ""
        # A fitted UV slope is only meaningful where the temporal factor has a
        # power-law tail.  The sweeping/burst kernels cut off as a Gaussian, for
        # which a local log-log slope is a function of the fit window, not a
        # property of the model -- so it is reported as such, never as a number.
        if key in POWERLAW_UV:
            suv = _slope(ps, ref, min(2.5 * pk, 4.0), 18.0) if np.isfinite(pk) else np.nan
            if np.isfinite(suv):
                txt += "\n" + rf"UV $p^{{{suv:.2f}}}$"
        elif key == "dk_imp":
            txt += "\n" + r"UV: $\Theta(2-p)$"
        elif key in ("dk_sw", "dk_burst"):
            txt += "\n" + r"UV: Gaussian, $\Theta(2-p)$"
        else:
            txt += "\n" + r"UV: Gaussian"
        if np.isfinite(pk):
            txt += "\n" + (r"$p_{\rm pk}$: UV-limited" if pk >= ps[-1] * 0.99
                            else rf"$p_{{\rm pk}}={pk:.2f}$")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left", color="0.25")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)
        apply_max_ticks(ax, n=7, axes=("y",))

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$p=k/k_0$", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Omega_{\rm GW}(p)\propto p^3H(p,p)$", fontsize=8)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.025))
    out = save_figure(fig, "kernel_comparison_grid")
    print(f"saved {out}")


def main():
    cache = Path(__file__).resolve().parent / "kernel_grid_cache.npz"
    if cache.exists() and "--rebuild" not in sys.argv:
        print(f"loading {cache.name} (pass --rebuild to recompute)")
        ps, data = load(cache)
    else:
        ps, data = build(cache=cache)
    plot(ps, data)


if __name__ == "__main__":
    main()
