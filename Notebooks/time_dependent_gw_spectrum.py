#!/usr/bin/env python3
r"""Time-dependent GW spectrum from a turbulent/magnetic source: analytic model + code.

Companion code to Notebooks/time_dependent_gw_derivation.md.  Implements two
things and checks they agree:

  1. THE EXACT SOLUTION.  Each GW Fourier mode is a driven oscillator
        h_k'' + k^2 h_k = S_k(t),
     solved with the retarded Green's function.  Its energy is
        E_k(t) = C(t)^2 + D(t)^2,  C=int cos(k t') S dt', D=int sin(k t') S dt',
     and the spectrum is Omega_GW(k,t) ~ k^3 E_k(t).  We build C,D by cumulative
     quadrature for an arbitrary source S_k(t) and read off the spectrum at any t.

  2. THE ANALYTIC TIME-DEPENDENT SPECTRUM.  For a statistically stationary but
     finite-lived source, the ensemble energy factorises into a spatial stress
     shape and a temporal build-up window,
        Omega_GW(k,t) = Omega_sat(k) * B(k, t-t0),
     with the coherent-source build-up  B = 1 - sinc(k(t-t0))  (numpy sinc).
     -> (k dt)^2/6 at small argument (causal k^3), -> 1 at saturation (flat).

The main() routine (a) verifies the exact solver reproduces the closed form
4 sin^2(k dt/2)/k^2 for a constant source, and (b) plots the analytic spectrum
evolving in time, showing the causal-k^3 knee sweeping to low k.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE, apply_max_ticks, apply_paper_style, save_figure,
)


# ----------------------------------------------------------------------------
# 1. EXACT SOLUTION: E_k(t) = C(t)^2 + D(t)^2 for an arbitrary source S_k(t)
# ----------------------------------------------------------------------------
def exact_energy(k, t, S):
    """E_k(t) = |int_{t0}^t e^{i k t'} S(t') dt'|^2 for one wavenumber k.

    t : 1-D time grid (t[0]=t0).  S : source S_k(t') on that grid.
    Returns E_k(t) on the whole grid via cumulative trapezoid of cos/sin*S.
    """
    ct = np.cos(k * t) * S
    st = np.sin(k * t) * S
    C = np.concatenate([[0.0], np.cumsum(0.5 * (ct[1:] + ct[:-1]) * np.diff(t))])
    D = np.concatenate([[0.0], np.cumsum(0.5 * (st[1:] + st[:-1]) * np.diff(t))])
    return C**2 + D**2


def coherent_window_closed(k, dt):
    """Closed form for a CONSTANT source: I(k,dt) = 4 sin^2(k dt/2)/k^2."""
    return 4.0 * np.sin(0.5 * k * dt) ** 2 / k**2


# ----------------------------------------------------------------------------
# 2. ANALYTIC TIME-DEPENDENT SPECTRUM  Omega_GW(k,t) = Omega_sat(k) B(k,t-t0)
# ----------------------------------------------------------------------------
def build_up(k, dt):
    """Time-averaged coherent build-up fraction B(k,dt) = 1 - sinc(k dt).

    Derived in the notes: <I(k,dt)> = (2/k^2)(1 - sin(k dt)/(k dt)); normalised
    to its saturated value 2/k^2 this is 1 - sinc(k dt).  numpy.sinc(x)=sin(pi x)/(pi x).
    """
    return np.clip(1.0 - np.sinc(k * dt / np.pi), 0.0, None)


def omega_sat(k, kp, ir_slope=1.0, uv_slope=-11.0 / 3.0):
    """Saturated GW spectrum shape Omega_sat(k): flat/causal below kp, power-law UV.

    A white-stress plateau (Omega_GW ~ k^{ir_slope}, ir_slope=1 => flat E_GW)
    with a peak near kp = 2 k0 and a Kolmogorov ultraviolet Omega_GW ~ k^{uv_slope}.
    Written as a smooth broken power law.
    """
    x = k / kp
    return x**ir_slope / (1.0 + x ** (ir_slope - uv_slope))


def omega_gw(k, t, kp, t0=0.0):
    """Time-dependent spectrum Omega_GW(k,t) = Omega_sat(k) * B(k, t-t0)."""
    return omega_sat(k, kp) * build_up(k, t - t0)


# ----------------------------------------------------------------------------
def main():
    apply_paper_style()

    # (a) verify the exact solver against the closed form for a constant source
    print("VERIFY exact E_k = C^2+D^2  vs  closed form 4 sin^2(k dt/2)/k^2:")
    t = np.linspace(0.0, 6.0, 60001)
    S = np.ones_like(t)                     # constant source turned on at t0=0
    for k in (0.5, 2.0, 8.0):
        Ee = exact_energy(k, t, S)[-1]
        Ec = coherent_window_closed(k, t[-1] - t[0])
        print(f"  k={k:4.1f}:  exact={Ee:.6e}  closed={Ec:.6e}  ratio={Ee/Ec:.6f}")

    # (b) the analytic time-dependent spectrum evolving in time
    k = np.geomspace(0.05, 200.0, 400)
    kp = 8.0                                 # peak at 2 k0
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for j, dt in enumerate((0.02, 0.1, 0.5, 3.0, 50.0)):
        ax.loglog(k, omega_gw(k, dt, kp), color=PALETTE[j + 1], lw=1.5,
                  label=rf"$t-t_0={dt:g}$")
    ax.loglog(k, 3e-3 * (k / k[0]) ** 3, color=PALETTE[0], ls=":", lw=1.0)
    ax.text(0.09, 2e-3, r"$k^3$ (causal)", fontsize=7, color=PALETTE[0], rotation=34)
    ax.loglog(k, omega_sat(k, kp), color="0.5", ls="--", lw=1.0, label="saturated")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(k,t)$")
    ax.set_ylim(1e-8, 3)
    ax.set_title(r"time-dependent GW spectrum: causal knee sweeps to low $k$", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    apply_max_ticks(ax)
    out = save_figure(fig, "time_dependent_gw_spectrum")
    print(f"\nsaved {out}")
    print("At each time the knee sits at k ~ 1/(t-t0): modes above it are flat")
    print("(saturated), below it still rise as k^3; the knee moves left with time.")


if __name__ == "__main__":
    main()
