#!/usr/bin/env python3
r"""Stationary GW kernel with a FULL fluid input spectrum (IR band + Kolmogorov range).

core.H_pq integrates the Gogoberidze Appendix-A form Eq.(Hijij-AppA) over the
Kolmogorov inertial range only (k1, u in [k0, k_d]); the fluid spectrum enters
purely through A(k) ~ k^{-11/3}, i.e. the k1^{-10/3} u^{-10/3} powers of the
integrand (= A(k) k^{1/3} per leg, the k^{1/3} being measure x (1/eta_k)).

To feed an IR band k < k0 into the GW kernel faithfully we keep every kinematic
and temporal factor of core verbatim and change only the spectrum: replace
A(k) -> A_full(k) with

    E(k) ~ k^{-5/3}                (k >= k0, Kolmogorov inertial range)
    E(k) ~ k^{s}                   (k <  k0, IR band; s=2 Saffman, s=4 Batchelor)

so A = E/(4 pi k^2) and the *shape factor* relative to the extrapolated Kolmogorov
law A_Kol ~ k^{-11/3} is

    S(k) = A(k)/A_Kol(k) = 1                  (k >= k0)
                         = (k/k0)^{s + 5/3}   (k <  k0),     continuous at k0.

In the substituted variable x = (k1/k0)^{-4/3} this is k1/k0 = x^{-3/4}, so the
inertial range is x <= 1 and the IR band is 1 < x <= R_IR^{4/3} with
R_IR = k0/k_IR the IR dynamic range.  The integrand is core's, times S(k1) S(u);
the u-clamp k0 is relaxed to k_IR.  For R_IR = 1 (no IR band) this is identically
core.H_pq -- verified in _validate().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from gw_turbulence.core import _h_prefactor, kernel_bracket  # noqa: E402

# IR fluid energy-spectrum slopes E(k)~k^s -> shape exponent s + 5/3 below k0.
IR_EXPONENTS = {"saffman": 2.0 + 5.0 / 3.0, "batchelor": 4.0 + 5.0 / 3.0}


def _shape(tilde_k, ir_exp: float):
    """S(k) = A(k)/A_Kol(k); tilde_k = k/k0.  1 in the inertial range, rising IR power."""
    return np.where(tilde_k >= 1.0, 1.0, np.asarray(tilde_k, float) ** ir_exp)


def H_full(p: float, q: float, M: float = 1.0, R: float = 1e4,
           R_IR: float = 1.0, ir: str = "batchelor",
           x_points: int = 160, y_points: int = 160) -> float:
    r"""Stationary GW kernel H(p,q) with IR band k in [k0/R_IR, k0] + inertial [k0, k_d].

    R_IR = k0/k_IR (IR dynamic range); ir selects the IR energy slope.  R_IR=1 -> core.H_pq.
    """
    ir_exp = IR_EXPONENTS[ir]
    x_lo, x_hi = 1.0 / R, R_IR ** (4.0 / 3.0)          # x<=1 inertial, x>1 IR band
    u_floor = 1.0 / R_IR                                # k_IR/k0
    u_ceil = R ** 0.75                                  # k_d/k0
    xs = np.geomspace(x_lo, x_hi, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        tk1 = x ** (-0.75)                              # k1/k0
        u_min = max(abs(tk1 - p), u_floor)
        u_max = min(tk1 + p, u_ceil)
        if not (u_min < u_max):
            continue
        y_min, y_max = u_max ** (-4.0 / 3.0), u_min ** (-4.0 / 3.0)
        ys = np.geomspace(y_min, y_max, y_points)       # inner integral vectorised over u
        tus = ys ** (-0.75)                             # u/k0
        ss = x + ys
        geom = ys ** 0.75 * ss ** (-0.5) * x ** 0.75 * kernel_bracket(p, x, ys)
        expo = np.exp(-2.0 * x * ys / ss * q ** 2 / M ** 2)
        erfc = special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(ss)))
        vals = geom * expo * erfc * _shape(tk1, ir_exp) * _shape(tus, ir_exp)
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def omega_gw_over_k(p, M=1.0, R=1e4, R_IR=1.0, ir="batchelor", **kw):
    """Omega_GW(k)/k = p^2 H(p,p) in the Roper Pol convention."""
    return p ** 2 * H_full(p, p, M=M, R=R, R_IR=R_IR, ir=ir, **kw)


def _validate():
    from gw_turbulence.core import H_pq

    print("=" * 70)
    print("VALIDATION (1): R_IR=1 reduces H_full to core.H_pq")
    print("=" * 70)
    print(f"  {'p':>6}{'H_full(R_IR=1)':>18}{'core.H_pq':>16}{'rel.diff':>11}")
    for p in (0.3, 1.0, 3.0):
        a = H_full(p, p, M=1.0, R=1e4, R_IR=1.0, x_points=160, y_points=160)
        b = H_pq(p, p, M=1.0, R=1e4)
        print(f"  {p:6.1f}{a:18.6e}{b:16.6e}{abs(a/b-1):11.2e}")

    print("\n" + "=" * 70)
    print("VALIDATION (2): effect of the IR band on the GW spectrum (M=1, R_IR=100)")
    print("=" * 70)
    print(f"  {'p':>6}{'no IR':>14}{'Saffman':>14}{'Batchelor':>14}")
    for p in (0.03, 0.1, 0.3, 1.0, 2.0):
        n = omega_gw_over_k(p, R_IR=1.0)
        s = omega_gw_over_k(p, R_IR=100.0, ir="saffman")
        b = omega_gw_over_k(p, R_IR=100.0, ir="batchelor")
        print(f"  {p:6.2f}{n:14.4e}{s:14.4e}{b:14.4e}")


if __name__ == "__main__":
    _validate()
