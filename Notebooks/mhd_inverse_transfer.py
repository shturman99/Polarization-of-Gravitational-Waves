#!/usr/bin/env python3
r"""Helical-MHD inverse transfer and the simulated infrared GW slope k^1.

EFFECTIVE-MODEL / MECHANISM DEMONSTRATION (not a first-principles MHD closure).
-----------------------------------------------------------------------------
Section IV of the paper localizes the simulation-vs-analytic IR discrepancy
(analytic causal Omega_GW~k^3 vs simulated k^1) to MHD: the first-principles
full-spatial *hydrodynamic* self-similar kernel
(Notebooks/fullspatial_selfsimilar_exact.py) shows freely-decaying HD turbulence
does NOT flatten the IR -- its decorrelation is the k-dependent sweeping time
tau_c ~ 1/(M k^{2/3}), short relative to the emission lifetime, and the growing
coherence as the flow decays is cancelled by the amplitude decay.

The physical claim of Sec. IV is that helical MHD is different: magnetic-helicity
conservation drives an INVERSE TRANSFER that moves magnetic energy to ever larger
scales and SUSTAINS a coherent large-scale field over the GW emission time T_em.
Such a field has a large-scale (Alfvenic) correlation time that is (i) essentially
k-INDEPENDENT (set by the large-scale field, not by k-dependent sweeping) and
(ii) LONG -- comparable to or exceeding T_em.

This tool isolates that mechanism with the SAME full-spatial kernel geometry as
the HD calculation (core.kernel_bracket / _integration_bounds / _h_prefactor,
k1=x^{-3/4}, k2=y^{-3/4}), changing ONLY the temporal factor to:
  * a k-INDEPENDENT base correlation time tau_c0 = coh / M  (Alfvenic large scale),
  * self-similar growth tau_c(T) = tau_c0 (1 + T/tau_st)^alpha during the window,
  * per-leg amplitude a(t) = (1 + t/tau_st)^{-gamma}.
"coh" = tau_c0 * M is the number of eddy times the large-scale field stays coherent
at t=0; the sustainment axis is s = tau_c0 / T_em. SUSTAINED + COHERENT corresponds
to small gamma / large tau_st and large coh (helical inverse transfer); the HD-like
limit is small coh (short k-set coherence) with decay.

RESULT (printed + plotted): as the source stays coherent over a larger fraction of
its lifetime (s = tau_c0/T_em -> 0.5..1), the GW IR slope flattens from the causal
k^2.7 straight through the simulated k^1 and below -- and this survives a realistic
helical amplitude decay gamma = 1/3 (B^2 ~ t^{-2/3}). So sustained large-scale
magnetic coherence, which helical inverse transfer supplies and HD decay does not,
quantitatively produces the simulated k^1.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Notebooks"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
from gw_turbulence.core import _h_prefactor, _integration_bounds, kernel_bracket


def _amp(t, tau_st, gamma):
    return (1.0 + t / tau_st) ** (-gamma)


def _window_factor(omega, T, T_em, tau1_0, tau2_0, tau_st, alpha, gamma):
    """Causal-window double-time factor for one emission epoch T."""
    w = min(T, T_em - T)
    if w <= 0:
        return 0.0
    half = 2.0 * w
    tau1 = tau1_0 * (1 + T / tau_st) ** alpha
    tau2 = tau2_0 * (1 + T / tau_st) ** alpha

    def g(dt):
        return (_amp(T + dt / 2, tau_st, gamma) * _amp(T - dt / 2, tau_st, gamma)
                * (1 + dt / tau1) ** (-2.0 / 3.0) * (1 + dt / tau2) ** (-2.0 / 3.0))

    if omega <= 0:
        v, _ = integrate.quad(g, 0, half, limit=200)
    else:
        v, _ = integrate.quad(g, 0, half, weight="cos", wvar=omega, limit=200)
    return 2.0 * float(v)


def _temporal(omega, M, tau_st, T_em, alpha, gamma, coh, n_T):
    """Emission-time-averaged temporal factor; k-INDEPENDENT (Alfvenic) base time."""
    tau1_0 = tau2_0 = coh / M
    Ts = np.linspace(0.0, T_em, n_T)
    inner = np.array([_window_factor(omega, T, T_em, tau1_0, tau2_0, tau_st, alpha, gamma)
                      for T in Ts])
    return np.pi / (tau1_0 * tau2_0 * T_em) * np.trapezoid(inner, Ts)


def H_mhd(p, q, M=1.0, R=1e4, tau_st=1e9, T_em=40.0, alpha=1.0, gamma=0.0,
          coh=1.0, x_points=24, y_points=24, n_T=40):
    """Full-spatial GW kernel with a sustained, coherent (Alfvenic) large-scale source."""
    xs = np.geomspace(1.0 / R, 1.0, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        b = _integration_bounds(x, p, R)
        if b is None:
            continue
        y_min, y_max = b
        ys = np.geomspace(y_min, y_max, y_points)
        vals = [yv ** 0.75 * (x + yv) ** (-0.5) * x ** 0.75 * kernel_bracket(p, x, yv)
                * _temporal(q, M, tau_st, T_em, alpha, gamma, coh, n_T) for yv in ys]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def _ir_slope(band, spectrum):
    good = spectrum > 0
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(band[good]), np.log(spectrum[good]), 1)[0])


# coherence values scanned (coh = tau_c0 * M); T_em = 40 eddy times, M = 1.
COH_VALUES = (0.3, 1.0, 3.0, 8.0, 20.0, 50.0, 120.0)
T_EM = 40.0
IR_BAND = np.geomspace(0.06, 0.45, 6)


def sustainment_scan(gamma, tau_st, n_T=40):
    """IR slope vs base coherence tau_c0 (=coh/M) for given amplitude-decay setup."""
    out = []
    for coh in COH_VALUES:
        spec = np.array([p ** 3 * H_mhd(p, p, M=1.0, tau_st=tau_st, T_em=T_EM,
                                        alpha=1.0, gamma=gamma, coh=coh, n_T=n_T)
                         for p in IR_BAND])
        out.append((coh / T_EM, _ir_slope(IR_BAND, spec), bool((spec > 0).all())))
    return out


def _figure(name="mhd_inverse_transfer"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import (PALETTE, apply_max_ticks, apply_paper_style,
                                          save_figure)

    apply_paper_style()
    # Panel (a) data: IR slope vs sustainment for sustained (gamma=0) and helical (gamma=1/3).
    sust = sustainment_scan(gamma=0.0, tau_st=1e12)
    heli = sustainment_scan(gamma=1.0 / 3.0, tau_st=8.0)
    s_s, sl_s = np.array([r[0] for r in sust]), np.array([r[1] for r in sust])
    s_h, sl_h = np.array([r[0] for r in heli]), np.array([r[1] for r in heli])

    # Panel (b) data: example IR spectra at increasing coherence (sustained).
    pb = np.geomspace(0.04, 0.6, 8)
    curves = []
    for coh in (1.0, 20.0, 120.0):
        sp = np.array([p ** 3 * H_mhd(p, p, M=1.0, tau_st=1e12, T_em=T_EM,
                                      alpha=1.0, gamma=0.0, coh=coh) for p in pb])
        curves.append((coh, sp / sp.max()))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)

    axa.axhline(3.0, color="0.55", ls=":", lw=1.2)
    axa.axhline(1.0, color="0.25", ls="--", lw=1.2)
    axa.text(0.011, 3.05, r"causal $k^{3}$ (HD)", fontsize=7, color="0.4")
    axa.text(0.011, 1.05, r"simulated $k^{1}$", fontsize=7, color="0.2")
    axa.semilogx(s_s, sl_s, "o-", color=PALETTE[1], lw=1.5, ms=4.5,
                 label=r"sustained ($\gamma=0$)")
    axa.semilogx(s_h, sl_h, "s-", color=PALETTE[2], lw=1.5, ms=4.0,
                 label=r"helical decay ($\gamma=\tfrac13$)")
    axa.set_xlabel(r"$\tau_c/T_{\rm em}$")
    axa.set_ylabel(r"IR slope of $\Omega_{\rm GW}(k)$")
    axa.set_ylim(0.2, 3.2)
    axa.legend(fontsize=7, loc="upper right")
    apply_max_ticks(axa)

    # reference slope guides (reference, not data)
    pr = np.array([0.05, 0.16])
    axb.loglog(pr, 6e-3 * (pr / pr[0]) ** 3, color="0.55", ls=":", lw=1.2)
    axb.loglog(pr, 0.18 * (pr / pr[0]) ** 1, color="0.25", ls="--", lw=1.2)
    axb.text(0.052, 1.1e-2, r"$k^{3}$", fontsize=7, color="0.4")
    axb.text(0.052, 0.30, r"$k^{1}$", fontsize=7, color="0.2")
    for (coh, sp), c in zip(curves, (PALETTE[0], PALETTE[1], PALETTE[3])):
        axb.loglog(pb, sp, "-", color=c, lw=1.6,
                   label=rf"$\tau_c/T_{{\rm em}}={coh / T_EM:.3g}$")
    axb.set_xlabel(r"$k/k_0$")
    axb.set_ylabel(r"$\Omega_{\rm GW}(k)$  (peak-normalised)")
    axb.legend(fontsize=7, loc="lower right")
    apply_max_ticks(axb)

    out = save_figure(fig, name)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main():
    print("Helical-MHD inverse transfer: GW IR slope vs source coherence-over-lifetime")
    print(f"T_em = {T_EM} eddy times, M = 1, full-spatial kernel, IR band p in [0.06,0.45]\n")
    print("(A) SUSTAINED amplitude (gamma=0, tau_st=inf) -- helical inverse-transfer limit")
    print(f"   {'tau_c/T_em':>11}{'IR slope':>10}{'all Om>0':>10}")
    for s, sl, ok in sustainment_scan(gamma=0.0, tau_st=1e12):
        print(f"   {s:>11.4g}{sl:>10.2f}{str(ok):>10}")
    print("\n(B) realistic helical amplitude decay (gamma=1/3, B^2~t^-2/3, tau_st=8)")
    print(f"   {'tau_c/T_em':>11}{'IR slope':>10}")
    for s, sl, ok in sustainment_scan(gamma=1.0 / 3.0, tau_st=8.0):
        print(f"   {s:>11.4g}{sl:>10.2f}")
    print("\nVERDICT: sustained large-scale coherence (tau_c/T_em >~ 0.5) flattens the GW")
    print("IR from causal k^3 through the simulated k^1; survives helical amplitude decay.")


if __name__ == "__main__":
    main()
    _figure()
