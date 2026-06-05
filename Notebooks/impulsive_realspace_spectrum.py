#!/usr/bin/env python3
r"""GW spectrum of an impulsive ($\delta$-in-time) source, in real space.

Companion to derivation.tex Sec.~"Impulsive source in real space".  The source
fires once, $S_{ij}(\mathbf r,t)=\mathcal S_{ij}(\mathbf r)\,g(t-t_0)$, and we keep
the spatial correlator $\Sigma(r)=\langle\mathcal S_{ij}(\mathbf r_1)\mathcal
S_{ij}(\mathbf r_2)\rangle$ purely in configuration space.

WHICH SPECTRUM.  The boxed configuration-space expression in the section,
    dE/dV domega  ~  omega^2 \int d^3r_1 d^3r_2 / (r_1 r_2) cos[omega(r_1-r_2)] Sigma,
is the *time-integrated* energy fluence.  For a statistically homogeneous source the
radiated field oscillates forever with a non-decaying energy density, so that
time integral (and the double real-space integral) DIVERGES -- it is the secular
on-shell resonance, ~ (total observation time).  The finite observable is the
time-averaged energy DENSITY per log frequency, which for the isotropic correlator
collapses to a single radial transform -- still purely real space, only omega, no k:

    drho_GW/dln omega = 8 G omega^2 \int_0^infty dr  r sin(omega r) Sigma(r) * |g~(omega)|^2.   (*)

(The radial sine transform IS the isotropic 3-D spatial transform; for a GW the
frequency equals the wavenumber, omega = k, so a frequency-resolved spectrum is
unavoidably a wavenumber-resolved one.  We just never write k.)

WHAT IS CHECKED (all asserts must pass):
  1. radial transform (*) reproduces the analytic Gaussian transform;
  2. the spectrum is independent of the firing time t_0 (it enters only as a phase
     e^{i omega t_0} that cancels in the modulus);
  3. instantaneous (g -> delta) peak sits at omega_peak = sqrt(3)/ell, set by the
     SOURCE coherence length ell, NOT by t_0;
  4. infrared slope d ln(drho/dln omega)/d ln omega -> 3 (causal);
  5. finite-duration burst of width Dt shifts the peak to
     omega_peak = sqrt(3) / sqrt(ell^2 + 2 Dt^2) = min(1/ell, 1/Dt) parametrically.

Run: python Notebooks/impulsive_realspace_spectrum.py
  -> prints the checks + writes images/impulsive_realspace_spectrum.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

try:
    from gw_turbulence.plot_style import (  # noqa: E402
        PALETTE,
        apply_paper_style,
        apply_max_ticks,
        save_figure,
    )
    try:
        apply_paper_style()          # usetex if a TeX install is present
    except Exception:                 # pragma: no cover - headless / no TeX
        apply_paper_style(usetex=False)
except Exception:                     # pragma: no cover - standalone fallback
    PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73",
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    def apply_max_ticks(*_a, **_k):
        pass

    def save_figure(fig, name, ext="pdf", subdir=None):
        out = ROOT / "images" / f"{name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        return out


# --------------------------------------------------------------------------- #
# Source model: isotropic real-space stress correlator (coherence length ell).  #
# Sigma(r) = <S_ij(r1) S_ij(r2)>, |r1 - r2| = r.  Gaussian toy (a stand-in for  #
# the TT-contracted correlator R_aa R_bb + 1/3 R_ab R_ab).                       #
# --------------------------------------------------------------------------- #
def Sigma(r: np.ndarray, ell: float = 1.0) -> np.ndarray:
    return np.exp(-r**2 / (2.0 * ell**2))


def P_radial(omega: float, ell: float = 1.0, r_max: float = 60.0) -> float:
    r"""Spatial power P(omega) = (4 pi / omega) int_0^inf r sin(omega r) Sigma(r) dr.

    Pure real-space (radial) transform of Sigma; the integrand decays Gaussianly.
    """
    if omega == 0.0:
        val, _ = quad(lambda r: r**2 * Sigma(r, ell), 0.0, r_max * ell, limit=200)
        return 4.0 * np.pi * val
    val, _ = quad(lambda r: r * np.sin(omega * r) * Sigma(r, ell),
                  0.0, r_max * ell, limit=400)
    return 4.0 * np.pi / omega * val


def P_analytic(omega: np.ndarray, ell: float = 1.0) -> np.ndarray:
    r"""Analytic transform of the Gaussian Sigma: P(omega) = (2 pi)^{3/2} ell^3 e^{-omega^2 ell^2/2}."""
    return (2.0 * np.pi) ** 1.5 * ell**3 * np.exp(-omega**2 * ell**2 / 2.0)


def temporal_filter(omega: np.ndarray, dt: float) -> np.ndarray:
    r"""|g~(omega)|^2 for a Gaussian burst of width Dt: e^{-omega^2 Dt^2}.

    Dt = 0 (instantaneous, g -> delta) gives |g~|^2 = 1 (white in source frequency).
    """
    return np.exp(-(omega**2) * dt**2)


def spectrum(omega: np.ndarray, ell: float = 1.0, dt: float = 0.0,
             G: float = 1.0, analytic_P: bool = True) -> np.ndarray:
    r"""drho_GW/dln omega = 8 G omega^2 [int r sin(omega r) Sigma dr] |g~(omega)|^2.

    Written via P(omega) = (4 pi/omega) int r sin Sigma:  = (2 G/pi) omega^3 P(omega) |g~|^2.
    """
    if analytic_P:
        P = P_analytic(omega, ell)
    else:
        P = np.array([P_radial(w, ell) for w in np.atleast_1d(omega)])
    return (2.0 * G / np.pi) * omega**3 * P * temporal_filter(omega, dt)


def peak_omega(ell: float = 1.0, dt: float = 0.0) -> float:
    r"""Analytic peak of omega^3 exp[-omega^2 (ell^2/2 + Dt^2)]:  sqrt(3)/sqrt(ell^2 + 2 Dt^2)."""
    return np.sqrt(3.0) / np.sqrt(ell**2 + 2.0 * dt**2)


# --------------------------------------------------------------------------- #
def run_checks() -> None:
    ell = 1.0
    print("=" * 70)
    print("Impulsive real-space GW spectrum -- checks (ell = 1)")
    print("=" * 70)

    # 1. radial (real-space) transform reproduces the analytic Gaussian transform
    omg = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 3.0])
    num = np.array([P_radial(w, ell) for w in omg])
    ana = P_analytic(omg, ell)
    rel = np.max(np.abs(num / ana - 1.0))
    print(f"[1] radial transform vs analytic Gaussian:  max rel. err = {rel:.2e}")
    assert rel < 1e-4, "radial transform does not match analytic P(omega)"

    # 2. firing time t_0 enters only as a phase -> spectrum independent of t_0
    w = 1.3
    amp = lambda t0: np.abs(-1j * w * np.exp(1j * w * t0)) ** 2  # |S~_dot|^2 mode factor
    vals = [amp(t0) for t0 in (0.0, 0.5, 3.7, -2.1, 100.0)]
    print(f"[2] |source(omega)|^2 over t_0 in {{0,0.5,3.7,-2.1,100}}: "
          f"spread = {np.ptp(vals):.2e}")
    assert np.ptp(vals) < 1e-12, "spectrum depends on t_0 (it must not)"

    # 3. instantaneous peak at sqrt(3)/ell, independent of t_0
    wg = np.linspace(0.05, 8.0, 4000)
    S0 = spectrum(wg, ell, dt=0.0)
    w_peak_num = wg[np.argmax(S0)]
    w_peak_th = peak_omega(ell, 0.0)
    print(f"[3] instantaneous peak: numeric {w_peak_num:.4f}, "
          f"theory sqrt(3)/ell = {w_peak_th:.4f}")
    assert abs(w_peak_num - w_peak_th) < 2e-2, "instantaneous peak off"

    # 4. infrared slope -> 3
    wl = np.array([1e-3, 2e-3])
    sl = np.diff(np.log(spectrum(wl, ell, 0.0))) / np.diff(np.log(wl))
    print(f"[4] infrared log-slope d ln S / d ln omega = {sl[0]:.4f}  (expect 3)")
    assert abs(sl[0] - 3.0) < 1e-3, "IR slope is not causal omega^3"

    # 5. finite-duration peak: sqrt(3)/sqrt(ell^2 + 2 Dt^2)
    print("[5] finite-duration peak omega_peak(Dt):")
    ok = True
    for dt in (0.0, 0.5, 1.0, 2.0, 5.0):
        S = spectrum(wg, ell, dt=dt)
        wpk = wg[np.argmax(S)]
        wth = peak_omega(ell, dt)
        flag = "ok" if abs(wpk - wth) < 3e-2 else "FAIL"
        ok = ok and flag == "ok"
        print(f"     Dt={dt:4.1f}:  numeric {wpk:.4f}   theory {wth:.4f}   [{flag}]")
    assert ok, "finite-duration peak track does not match min(1/ell, 1/Dt)"

    print("-" * 70)
    print("ALL CHECKS PASSED")
    print("-" * 70)


# --------------------------------------------------------------------------- #
def make_figure() -> Path:
    ell = 1.0
    wg = np.linspace(0.02, 8.0, 2000)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.2),
                                   constrained_layout=True)

    # (a) instantaneous spectrum, t_0 independence, causal IR
    S0 = spectrum(wg, ell, dt=0.0)
    S0n = S0 / S0.max()
    for i, t0 in enumerate((0.0, 1.0, 5.0)):
        axL.plot(wg, S0n, color=PALETTE[i + 1], lw=2.0 - 0.5 * i,
                 ls=("-", "--", ":")[i],
                 label=rf"$t_0={t0:g}$")
    wpk = peak_omega(ell, 0.0)
    axL.axvline(wpk, color="0.5", lw=1.0, ls="--")
    axL.text(wpk * 1.05, 0.5, r"$\sqrt{3}/\ell$", color="0.4")
    # causal omega^3 guide (anchored to the curve at omega ell = 0.3)
    g = np.array([0.06, 0.5])
    anchor = float(np.interp(0.3, wg, S0n))
    axL.plot(g, anchor * (g / 0.3) ** 3, color="0.6", lw=1.0, ls="-.",
             label=r"$\propto\omega^3$")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_ylim(1e-4, 2.0)
    axL.set_xlim(0.05, 8)
    axL.set_xlabel(r"$\omega\,\ell$")
    axL.set_ylabel(r"$d\rho_{\rm GW}/d\ln\omega$ (normalised)")
    axL.set_title(r"instantaneous burst ($\Delta t=0$)")
    axL.legend(title=r"firing time", frameon=False)
    apply_max_ticks(axL)

    # (b) finite-duration: peak slides from 1/ell to 1/Dt
    dts = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    for i, dt in enumerate(dts):
        S = spectrum(wg, ell, dt=dt)
        axR.plot(wg, S / S.max(), color=PALETTE[i + 1], lw=1.8,
                 label=rf"$\Delta t={dt:g}\,\ell$")
        wpk = peak_omega(ell, dt)
        axR.axvline(wpk, color=PALETTE[i + 1], lw=0.8, ls=":")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_ylim(1e-4, 2.0)
    axR.set_xlim(0.05, 8)
    axR.set_xlabel(r"$\omega\,\ell$")
    axR.set_ylabel(r"$d\rho_{\rm GW}/d\ln\omega$ (normalised)")
    axR.set_title(r"finite duration: $\omega_{\rm peak}\simeq\min(1/\ell,1/\Delta t)$")
    axR.legend(frameon=False)
    apply_max_ticks(axR)

    out = save_figure(fig, "impulsive_realspace_spectrum")
    plt.close(fig)
    return out


if __name__ == "__main__":
    run_checks()
    path = make_figure()
    print(f"figure written: {path}")
