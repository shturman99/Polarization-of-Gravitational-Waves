#!/usr/bin/env python3
# =============================================================================
# DEPRECATED for the Sec. IV IR-slope conclusion (2026-05-25 audit).
# -----------------------------------------------------------------------------
# This validator checks the OLD "coherence-resolution" story (checks C5/C6 below
# exercise fullspatial_selfsimilar.py's coherence KNOB and the tau_c/T_em
# narrative).  Per the Sec. IV reframe, that resolution is superseded: self-
# similar HD decay does NOT flatten the IR to k^1.  The AUTHORITATIVE Sec. IV
# kernel is Notebooks/fullspatial_selfsimilar_exact.py.
#
# NOTE: checks C1-C4 here are STILL VALID and useful -- they are independent
# from-scratch re-validations of the AEROACOUSTIC self-similar source and of
# selfsimilar_hybrid.H_exact (C2 confirms brute-force == H_exact to <0.5%).  Only
# the coherence-resolution interpretation in C5/C6 is deprecated.  As of
# 2026-05-25 all 13/13 checks still PASS.
# =============================================================================
r"""INDEPENDENT validation of every numerical claim in derivation.tex
Sec.~principal-discrepancy "Resolution".

Strategy: re-implement the aeroacoustic self-similar GW source FROM SCRATCH here
(no import of selfsimilar_hybrid) using a brute-force direct double-time grid
integral -- a different integration route than the production tools -- and confirm
(a) it reproduces the production kernels, and (b) the slopes quoted in the paper.
Full-spatial claims are cross-checked against the independent core.py kernels.

Run: python Notebooks/validate_ir_resolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "Notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


# ===========================================================================
# Independent from-scratch self-similar model (Loitsiansky), aeroacoustic source
# ===========================================================================
P_DECAY, Q_GROW, S_SHAPE = 10 / 7, 2 / 7, 4


def _model(tau_st, u0=1.0, l0=1.0):
    """Return closures (Phi2, tau_c0) for the self-similar BK2016 model."""
    def u(t):  return u0 * (1 + t / tau_st) ** (-P_DECAY / 2)
    def L(t):  return l0 * (1 + t / tau_st) ** Q_GROW
    def eps(t): return P_DECAY * u0 ** 2 / (2 * tau_st) * (1 + t / tau_st) ** (-P_DECAY - 1)
    def tau1(k, t): return 1.0 / (eps(t) ** (1 / 3) * k ** (2 / 3))
    def phi(kp): return kp ** S_SHAPE / (1 + kp) ** (S_SHAPE + 5 / 3)
    def Phi_eq(k, t): return (u(t) ** 2 * L(t) * phi(k * L(t))) / (4 * np.pi * k ** 2)

    def Phi2(k, T1, T2):
        Tm = 0.5 * (T1 + T2)
        R = (1.0 + np.abs(T1 - T2) / tau1(k, Tm)) ** (-2 / 3)
        amp = np.sqrt(Phi_eq(k, T1) * Phi_eq(k, T2))
        return (amp * R) ** 2

    return Phi2, tau1(1.0, 0.0)


def H_aeroacoustic_bruteforce(omega, T_em, tau_st, n_t=240, n_k=80,
                              k_min=1e-2, k_max=1e3):
    """7/(3pi^2) int dk k^2 int_[0,Tem]^2 dt1 dt2 cos(w(t1-t2)) Phi(k;t1,t2)^2.

    Direct double-time grid integral (independent of selfsimilar_hybrid's Wigner
    method).  Vectorised over the time grid; loops over k.
    """
    Phi2, _ = _model(tau_st)
    ts = np.linspace(0.0, T_em, n_t)
    ks = np.geomspace(k_min, k_max, n_k)
    T1, T2 = np.meshgrid(ts, ts, indexing="ij")
    cosfac = np.cos(omega * (T1 - T2))
    kint = np.empty(n_k)
    for j, k in enumerate(ks):
        integ = Phi2(k, T1, T2) * cosfac
        kint[j] = np.trapezoid(np.trapezoid(integ, ts, axis=1), ts)
    return 7.0 / (3.0 * np.pi ** 2) * np.trapezoid(ks ** 2 * kint, ks)


def omega_aero(p, T_em, tau_st, **kw):
    return p ** 3 * H_aeroacoustic_bruteforce(p, T_em, tau_st, **kw)


def band_slope(spec, ps):
    y = np.array([spec(p) for p in ps])
    g = y > 0
    return float(np.polyfit(np.log(ps[g]), np.log(y[g]), 1)[0])


def verdict(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got:+.3f}  (expect {lo}..{hi})")
    return ok


# ===========================================================================
def main():
    results = []
    print("=" * 74)
    print("C1. Top-hat duration identity   int_[0,T]^2 cos(w dt) = 4 sin^2(wT/2)/w^2")
    print("=" * 74)
    bad = 0.0
    for w, T in [(0.3, 5.0), (1.0, 8.0), (2.5, 3.0)]:
        ts = np.linspace(0, T, 4000)
        A, B = np.meshgrid(ts, ts, indexing="ij")
        num = np.trapezoid(np.trapezoid(np.cos(w * (A - B)), ts, axis=1), ts)
        ana = 4 * np.sin(w * T / 2) ** 2 / w ** 2
        bad = max(bad, abs(num / ana - 1))
        print(f"  w={w} T={T}: numeric={num:.5f} analytic={ana:.5f} rel.err={abs(num/ana-1):.2e}")
    results.append(verdict("identity holds (<1e-3)", bad, 0.0, 1e-3))
    print("  Consequence: coherent white source Omega = w^3 * const * <4sin^2/w^2>")
    print("    -> w^3 * (2/w^2) = w^1 (wT>>1) ;  -> w^3 * T^2 (wT<<1) = w^3.  [analytic]")

    print("\n" + "=" * 74)
    print("C2. Brute-force aeroacoustic kernel == selfsimilar_hybrid.H_exact (independent")
    print("    integration routes agree -> both trustworthy)")
    print("=" * 74)
    import selfsimilar_hybrid as ss
    worst = 0.0
    for tau_st, w, Tem in [(1.0, 0.6, 8.0), (0.3, 1.2, 6.0), (1e3, 0.3, 30.0)]:
        pars = dict(u0=1.0, l0=1.0, tau_st=tau_st, p=P_DECAY, q=Q_GROW, s=S_SHAPE)
        a = H_aeroacoustic_bruteforce(w, Tem, tau_st, n_t=260, n_k=80)
        b = ss.H_exact(w, Tem, n_T=160, n_k=80, **pars)
        worst = max(worst, abs(a / b - 1))
        print(f"  tau_st={tau_st:<6g} w={w} Tem={Tem}: brute={a:.4e} H_exact={b:.4e} ratio={a/b:.4f}")
    results.append(verdict("brute vs H_exact agree (<5%)", worst, 0.0, 0.05))

    print("\n" + "=" * 74)
    print("C3. Aeroacoustic IR slope (band, brute-force) vs paper claims")
    print("=" * 74)
    # coherent over lifetime: frozen amplitude (tau_st huge), long window
    Tem = 60.0
    band = np.geomspace(3.0 / Tem, 0.5, 9)
    s_coh = band_slope(lambda p: omega_aero(p, Tem, 1e6, n_t=300, n_k=70), band)
    print(f"  coherent (tau_st=1e6, Tem=60):  slope = {s_coh:+.2f}   [paper: ~1.1, analytic 1]")
    results.append(verdict("coherent slope ~ 1 (k^1)", s_coh, 0.7, 1.5))

    # self-similar decay, slow and fast
    s_slow = band_slope(lambda p: omega_aero(p, Tem, 1e3, n_t=260, n_k=70), band)
    s_fast = band_slope(lambda p: omega_aero(p, Tem, 0.25, n_t=200, n_k=70), band)
    print(f"  slow decay (tau_st=1e3 -> eps0~7e-4): slope = {s_slow:+.2f}   [paper: ~1.9]")
    print(f"  fast decay (tau_st=0.25 -> eps0=4):   slope = {s_fast:+.2f}   [paper: ~2.5]")
    results.append(verdict("slow-decay slope ~1.9", s_slow, 1.6, 2.2))
    results.append(verdict("fast-decay slope ~2.5", s_fast, 2.2, 2.8))
    results.append(verdict("monotonic coherent<slow<fast", 1.0 if s_coh < s_slow < s_fast else 0.0, 1, 1))

    print("\n" + "=" * 74)
    print("C4. Deep-IR (p->0) is causal k^3 (brute-force, p=1e-3..1e-2)")
    print("=" * 74)
    deep = np.geomspace(1e-3, 1e-2, 6)
    s_deep = band_slope(lambda p: omega_aero(p, 8.0, 1.0, n_t=160, n_k=70), deep)
    print(f"  deep-IR slope = {s_deep:+.2f}   [paper: causal ~3, anchor 2.93]")
    results.append(verdict("deep-IR ~ 3 (causal)", s_deep, 2.7, 3.2))

    print("\n" + "=" * 74)
    print("C5. Full-spatial: H_window == fullspatial_decay (Tem->inf); deep-IR; peak")
    print("=" * 74)
    from fullspatial_decay import H_decay_fast
    from fullspatial_selfsimilar import H_window, omega_gw as omega_win
    worst = 0.0
    for p in (0.5, 1.0, 2.0):
        a = H_window(p, p, M=1.0, R=1e4, T_em=np.inf)
        b = H_decay_fast(p, p, M=1.0, R=1e4)
        worst = max(worst, abs(a / b - 1))
    print(f"  H_window(inf) vs H_decay_fast: worst rel.diff = {worst:.2e}")
    results.append(verdict("window reduces to quasi-stationary (<1e-3)", worst, 0.0, 1e-3))

    fs_deep = np.geomspace(1e-3, 1e-2, 5)
    s_fsdeep = band_slope(lambda p: p ** 3 * H_decay_fast(p, p, M=1.0, R=1e4), fs_deep)
    print(f"  full-spatial deep-IR slope (independent H_decay_fast) = {s_fsdeep:+.2f}  [causal 3]")
    results.append(verdict("full-spatial deep-IR ~3", s_fsdeep, 2.6, 3.2))

    def fs_peak(coh):
        ps = np.geomspace(0.8, 6.0, 22)
        sp = np.array([omega_win(p, 1.0, T_em=20.0, coherence=coh) for p in ps])
        i = int(np.argmax(sp)); il, ir = max(i-1, 0), min(i+1, len(ps)-1)
        c = np.polyfit(np.log(ps[il:ir+1]), np.log(sp[il:ir+1]), 2)
        return float(np.exp(-c[1] / (2 * c[0])))
    band2 = np.geomspace(0.1, 0.7, 7)
    for coh in (1.0, 4.0, 16.0):
        sl = band_slope(lambda p: omega_win(p, 1.0, T_em=20.0, coherence=coh), band2)
        pk = fs_peak(coh)
        print(f"  coherence={coh:<4g}: IR slope={sl:+.2f}  peak p={pk:.2f}")
    pk1, pk16 = fs_peak(1.0), fs_peak(16.0)
    results.append(verdict("peak stays source-scale (1.7..3) for coherence 1", pk1, 1.7, 3.0))
    results.append(verdict("peak stays source-scale (1.7..3) for coherence 16", pk16, 1.7, 3.0))

    print("\n" + "=" * 74)
    print("C6. tau_c/T_em is NOT a single control: lifetime and correlation time each")
    print("    flatten the slope, so the ratio moves OPPOSITELY along the two routes.")
    print("=" * 74)
    fixed_band = np.geomspace(0.05, 0.45, 8)  # band fixed for ALL points (fair comparison)

    def aslope(T_em, tau_st):
        y = np.array([omega_aero(p, T_em, tau_st, n_t=max(160, int(8 * T_em)), n_k=60)
                      for p in fixed_band])
        g = y > 0
        return float(np.polyfit(np.log(fixed_band[g]), np.log(y[g]), 1)[0])

    sT = [aslope(T, 2.0) for T in (15, 60, 120)]          # vary lifetime, fixed decay
    sD = [aslope(60, ts) for ts in (0.5, 8.0, 40.0)]      # vary decay, fixed lifetime
    print(f"  vary T_em (tau_st=2):   T_em=15/60/120 -> slope {sT[0]:.2f}/{sT[1]:.2f}/{sT[2]:.2f}"
          f"  (longer lifetime -> flatter)")
    print(f"  vary decay (T_em=60):   tau_st=0.5/8/40 -> slope {sD[0]:.2f}/{sD[1]:.2f}/{sD[2]:.2f}"
          f"  (slower decay -> flatter)")
    results.append(verdict("longer lifetime flattens", sT[0] - sT[2], 0.05, 2.0))
    results.append(verdict("slower decay flattens", sD[0] - sD[2], 0.05, 2.0))

    print("\n" + "=" * 74)
    n_pass = sum(results)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed.")
    print("=" * 74)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
