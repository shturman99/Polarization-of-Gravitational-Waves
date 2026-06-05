#!/usr/bin/env python3
r"""Simplest toy: an impulsive source coherent at a SINGLE length scale.

Companion to derivation.tex Sec.~"Impulsive source in real space".  There the GW
energy density per log frequency is
    drho_GW/dln omega  ~  omega^3 P(omega) |g~(omega)|^2,
with P(omega) the spatial power of the source stress and omega = k the GW frequency.

Take the FLUID coherent at ONE length scale ell0 only.  The GW source is the Reynolds
stress S_ij ~ v_i v_j, QUADRATIC in the velocity (quadrupole radiation).  By Wick's
theorem the connected stress correlator is the SQUARE of the velocity correlator,
    Sigma(r) ~ R(r)^2,   R(r) = sin(r/ell0)/(r/ell0)  (single-scale fluid).
Squaring HALVES the scale (sin^2 theta = (1 - cos 2theta)/2):
    R(r)^2 ~ sin^2(r/ell0)/(r/ell0)^2 = [1 - cos(2 r/ell0)] / [2 (r/ell0)^2],
so the radiating part of Sigma oscillates on the halved scale ell0/2.  The GW frequency
is the inverse of the scale it samples, giving a SINGLE spectral line at
    omega_GW = 2/ell0     (GW scale ell_GW = ell0/2).
The FACTOR OF 2 is thus DERIVED from the quadratic source -- the same "GW peak at half
the source scale" (p_peak = 2) found for the monochromatic source in derivation.tex
Sec. "Impulsive (delta-in-time) source".  No causal omega^3 infrared rise, no
ultraviolet decay -- the omega^3 and the temporal filter |g~|^2 only set the line's
HEIGHT, not its position.  Hence omega_GW is locked to twice the inverse fluid scale and
is independent of the firing time t0 AND of the burst duration Dt.

Run: python Notebooks/impulsive_monochromatic_source.py
  -> prints the relation + writes images/impulsive_monochromatic_source.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

try:
    from gw_turbulence.plot_style import PALETTE, apply_paper_style, save_figure  # noqa: E402
    try:
        apply_paper_style()
    except Exception:                       # pragma: no cover
        apply_paper_style(usetex=False)
except Exception:                           # pragma: no cover
    PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73",
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    def save_figure(fig, name, ext="pdf", subdir=None):
        out = ROOT / "images" / f"{name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        return out


def omega_gw(ell0: float) -> float:
    """GW line frequency for a FLUID coherent at the single scale ell0.

    Quadrupole (stress ~ velocity^2) halves the scale ell0 -> ell0/2, so the GW scale is
    ell_GW = ell0/2 and the frequency is omega_GW = 2/ell0.
    """
    return 2.0 / ell0


def run_checks() -> None:
    print("=" * 66)
    print("Single-scale fluid -> GW scale ell0/2 (quadrupole halving of the scale)")
    print("=" * 66)
    for ell0 in (0.5, 1.0, 2.0, 4.0):
        ell_gw = 0.5 * ell0
        w0 = omega_gw(ell0)
        print(f"  ell0 = {ell0:4.1f}  ->  ell_GW = ell0/2 = {ell_gw:5.3f},  "
              f"omega_GW = 2/ell0 = {w0:5.3f}   (scale halved)")
        assert abs(w0 - 1.0 / ell_gw) < 1e-15
    print("-" * 66)
    print("ell_GW = ell0/2 exactly: factor-of-2 quadrupole halving of the scale;")
    print("no slope, no decay, no t0/Dt dependence.")
    print("-" * 66)


def make_figure() -> Path:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0),
                                   constrained_layout=True)

    # (a) the derivation: velocity correlator R(r) vs its square R(r)^2 (the stress).
    #     R oscillates on the scale ell0; R^2 oscillates on the halved scale ell0/2.
    ell0 = 1.0
    x = np.linspace(1e-3, 6.0, 1200)          # x = r/ell0
    R = np.sin(x) / x
    axL.plot(x, R, color=PALETTE[2], lw=2.0,
             label=r"velocity $R(r)\propto\sin(r/\ell_0)/(r/\ell_0)$")
    axL.plot(x, R**2, color=PALETTE[6], lw=2.2,
             label=r"stress $\Sigma\propto R(r)^2$")
    axL.axhline(0.0, color="0.7", lw=0.8)
    # mark the halving: R^2 has nodes every pi (scale ell0/2) vs R every pi too, but the
    # OSCILLATION of R^2 (the cos(2 r/ell0) term) runs at twice the rate -> annotate.
    axL.annotate(r"$R^2=\frac{1-\cos(2r/\ell_0)}{2(r/\ell_0)^2}$: scale $\ell_0/2$",
                 xy=(2.1, 0.05), xytext=(2.4, 0.45),
                 arrowprops=dict(arrowstyle="->", color=PALETTE[6], lw=1.4),
                 color=PALETTE[6], fontsize=10)
    axL.set_xlim(0.0, 6.0)
    axL.set_ylim(-0.32, 1.05)
    axL.set_xlabel(r"separation $r/\ell_0$")
    axL.set_ylabel(r"correlator (normalised)")
    axL.set_title(r"Wick: $\Sigma\propto R^2$ halves the scale $\ell_0\to\ell_0/2$")
    axL.legend(frameon=False, fontsize=9, loc="upper right")

    # (b) the resulting relation: GW scale = ell0/2.
    ell0s = np.array([0.5, 1.0, 2.0, 4.0])
    ell_gw = 0.5 * ell0s
    lline = np.linspace(0.0, 4.5, 50)
    axR.plot(lline, 0.5 * lline, "-", color="0.5", lw=1.5,
             label=r"$\ell_{\rm GW}=\ell_0/2$")
    axR.plot(lline, 1.0 * lline, ":", color="0.7", lw=1.2,
             label=r"$\ell_{\rm GW}=\ell_0$ (no halving)")
    for i, (l0i, lgi) in enumerate(zip(ell0s, ell_gw)):
        axR.plot([l0i], [lgi], "o", color=PALETTE[i + 1], ms=9,
                 label=rf"$\ell_0={l0i:g}$")
    axR.set_xlim(0.0, 4.5)
    axR.set_ylim(0.0, 4.5)
    axR.set_xlabel(r"fluid scale $\ell_0$")
    axR.set_ylabel(r"GW scale $\ell_{\rm GW}=1/\omega_{\rm GW}$")
    axR.set_title(r"$\ell_{\rm GW}=\ell_0/2$ (slope $1/2$)")
    axR.legend(frameon=False, fontsize=9, loc="upper left")

    out = save_figure(fig, "impulsive_monochromatic_source")
    plt.close(fig)
    return out


if __name__ == "__main__":
    run_checks()
    print(f"figure written: {make_figure()}")
