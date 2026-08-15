r"""Quantify the viscous-cusp floor promised in Sec.~\ref{sec:outlook}.

The outlook section of ``derivation.tex`` asserts, without computing, that a
viscously damped mode decorrelates as :math:`e^{-\nu k^{2}|\tau|}`, that this
supplies the zero-lag kink demanded by the cusp theorem
Eq.~(eq:cusp-tail) even when the sweeping correlator is the exact Gaussian of
Auclair et al., and that the resulting on-cone temporal factor tends to the
constant :math:`2\nu` -- a strictly power-law floor beneath the super-exponential
sweeping cutoff, linear in the viscosity.  This script turns that assertion into
numbers.

Conventions (all from ``derivation.tex``)
-----------------------------------------
Sweeping correlator (Sec.~sec:kraichnan-k41):

    f(eta_k, tau) = exp(-(pi/4) eta_k^2 tau^2),
    eta_k = eps^{1/3} k^{2/3} / sqrt(2 pi).

Viscous factor appended to it:

    f_nu(k, tau) = exp(-(pi/4) eta_k^2 tau^2) * exp(-nu k^2 |tau|).

Dimensionless variables of Eq.~(eq:appA-dimless):

    p = k/k0,  q = omega/(c k0),  M = (eps/k0)^{1/3}/c,
    k_d/k0 = R^{3/4},  x = (k1/k0)^{-4/3},  y = (u/k0)^{-4/3}.

With the Kolmogorov relation nu = eps^{1/3} k_d^{-4/3} this gives the exact
dimensionless viscous rate

    nu k^2 / (c k0) = M p^2 / R,

so the viscosity is expressed entirely through the paper's own Reynolds number
R = (k_d/k_0)^{4/3}.

Two-leg temporal factor
-----------------------
The stress is quadratic, so what enters the kernel is the transform of the
PRODUCT of the two legs (k1, u), exactly as the squared tent does in
Sec.~sec:ir-branch:

    F(tau) = f_nu(k1,tau) f_nu(u,tau)
           = exp(-a tau^2) exp(-b |tau|),
    a = (pi/4)(eta_{k1}^2 + eta_u^2),   b = nu (k1^2 + u^2),

    T(omega) = 2 int_0^inf cos(omega tau) F(tau) dtau
             = sqrt(pi/a) Re w((omega + i b)/(2 sqrt a)),

with w the Faddeeva function.  Setting b = 0 returns sqrt(pi/a) exp(-omega^2/4a),
i.e. exactly the Gaussian factor of Eq.~(eq:Hijij-AppA-dimless).  In the (x,y)
variables

    X = omega/(2 sqrt a)  = sqrt(2) (q/M) sqrt(x y/(x+y)),
    Y = b     /(2 sqrt a) = (sqrt(2)/R) (x^{-3/2}+y^{-3/2}) sqrt(x y/(x+y)),

and note that M cancels out of Y: the viscous kink is controlled by R alone.
The kernel of Eq.~(eq:Hijij-AppA-dimless) is therefore reproduced verbatim with
the single replacement

    exp(-2 x y q^2 / (M^2 (x+y)))  ->  Re w(X + i Y),

which is what ``integrand_y_visc`` below does; the (x+y)^{-1/2} weight, the
erfc, the geometric bracket and the prefactor are untouched (imported from
``gw_turbulence.core``).  The floor is the UV asymptote Re w -> Y/(sqrt(pi)X^2),
equivalently T -> 2 nu (k1^2+u^2)/omega^2.

Outputs
-------
* console: analytic verification, the C_eff coefficient, the R-table;
* ``images/viscous_cusp_floor.pdf``.

Run:  .venv/bin/python Notebooks/viscous_cusp_floor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_turbulence.core import (  # noqa: E402
    H_pq,
    _h_prefactor,
    _integration_bounds,
    kernel_bracket,
)
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

SQRT2 = np.sqrt(2.0)
SQRTPI = np.sqrt(np.pi)


# --------------------------------------------------------------------------
# 1.  Analytic verification of the cusp claim
# --------------------------------------------------------------------------
def two_leg_correlator(tau, a, b):
    """F(tau) = exp(-a tau^2) exp(-b|tau|)."""
    return np.exp(-a * tau**2) * np.exp(-b * np.abs(tau))


def T_closed(omega, a, b):
    """T(omega) = 2 int_0^inf cos(omega t) e^{-a t^2 - b t} dt, in closed form."""
    return np.sqrt(np.pi / a) * np.real(special.wofz((omega + 1j * b) / (2.0 * np.sqrt(a))))


def T_quad(omega, a, b):
    value, _ = integrate.quad(
        lambda t: 2.0 * np.exp(-a * t**2 - b * t),
        0.0,
        np.inf,
        weight="cos",
        wvar=omega,
        limit=400,
    )
    return value


def verify_analytics() -> None:
    print("=" * 78)
    print("1.  ANALYTIC VERIFICATION")
    print("=" * 78)

    # (a) the one-sided slope of a single viscously damped leg
    print("\n(a) single leg  f(tau) = exp(-(pi/4) eta^2 tau^2) exp(-nu k^2 |tau|)")
    print("    f'(0+) should be -nu k^2 exactly (the Gaussian contributes nothing)")
    h = 1e-7
    for eta, g in [(1.0, 1e-1), (2.0, 1e-2), (0.5, 1.0), (10.0, 1e-3)]:
        a = np.pi / 4 * eta**2
        num = (two_leg_correlator(h, a, g) - 1.0) / h
        print(f"    eta={eta:5.2f}  nu k^2={g:8.1e}   f'(0+)={num:+.8f}   -nu k^2={-g:+.8f}")

    # (b) closed form vs direct quadrature
    print("\n(b) closed form  T = sqrt(pi/a) Re w((w+ib)/(2 sqrt a))  vs quadrature")
    worst = 0.0
    for a, b in [(1.0, 0.1), (0.7, 0.01), (2.0, 1.0), (0.3, 1e-3)]:
        for w in [0.1, 1.0, 5.0, 20.0, 100.0]:
            rel = abs(T_closed(w, a, b) / T_quad(w, a, b) - 1.0)
            worst = max(worst, rel)
    print(f"    max relative deviation over the grid: {worst:.2e}")

    # (c) the cusp tail
    print("\n(c) cusp theorem  T(w) w^2 / (-2 F'(0+)) -> 1 with F'(0+) = -b")
    a, b = np.pi / 4, 0.05
    for w in [10, 30, 100, 300, 1000]:
        print(f"    w={w:6.0f}   T w^2/(2b) = {T_closed(w, a, b) * w**2 / (2 * b):.6f}")

    # (d) the two-leg product rule
    print("\n(d) two-leg product  F = f_nu(k1) f_nu(u):  F'(0+) = -nu (k1^2 + u^2)")
    nu = 0.02
    for k1, u, eta1, eta2 in [(0.7, 1.3, 0.9, 1.4), (1.0, 1.0, 1.0, 1.0), (0.05, 1.0, 0.2, 1.0)]:
        a = np.pi / 4 * (eta1**2 + eta2**2)
        b = nu * (k1**2 + u**2)
        num = (two_leg_correlator(h, a, b) - 1.0) / h
        print(
            f"    k1={k1:4.2f} u={u:4.2f}:  F'(0+)={num:+.8f}   "
            f"-nu(k1^2+u^2)={-b:+.8f}"
        )

    print("\n    => on the GW cone omega = c k, T -> 2 nu (k1^2 + u^2) / (c^2 k^2).")
    print("       k1 = u = k        (allowed by the triangle inequality):  T -> 4 nu")
    print("       k1 << k, u -> k   (the UV-dominant soft/hard pairing) :  T -> 2 nu")
    print("       k1 = u = k/2      (the minimum of k1^2+u^2 on the cone):  T ->  nu")
    print("       The paper's '2 nu' is the middle case only.")


# --------------------------------------------------------------------------
# 2.  The kernel with the viscous correlator
# --------------------------------------------------------------------------
def _XY(x, y, p, q, M, R):
    r = np.sqrt(x * y / (x + y))
    X = SQRT2 * (q / M) * r
    Y = (SQRT2 / R) * (x ** (-1.5) + y ** (-1.5)) * r
    return X, Y


def integrand_y_visc(y, x, p, q, M, R, mode):
    """Eq.~(eq:Hijij-AppA-dimless) integrand with the temporal factor replaced.

    ``mode`` is one of

    * ``"gauss"`` -- exp(-X^2), i.e. core.integrand_y verbatim;
    * ``"visc"``  -- Re w(X + iY), the Gaussian times exp(-nu(k1^2+u^2)|tau|);
    * ``"floor"`` -- Y/(sqrt(pi)(X^2+Y^2)), the UV asymptote of ``visc``;
    * ``"const"`` -- the paper's claim T -> 2 nu, i.e. (k1^2+u^2)/k^2 -> 1.
    """
    s = x + y
    if s <= 0:
        return 0.0
    bracket = kernel_bracket(p, x, y)
    erfc_factor = special.erfc(-SQRT2 * q * y / (M * np.sqrt(s)))
    X, Y = _XY(x, y, p, q, M, R)

    if mode == "gauss":
        ttilde = np.exp(-(X**2))
    elif mode == "visc":
        ttilde = np.real(special.wofz(X + 1j * Y))
    elif mode == "floor":
        ttilde = Y / (SQRTPI * (X**2 + Y**2))
    elif mode == "const":
        # T = 2 nu / c^2 exactly: strip the (k1^2+u^2)/k^2 = (x^-3/2+y^-3/2)/p^2
        # factor from the floor, keeping every other weight identical.
        ttilde = Y / (SQRTPI * (X**2 + Y**2)) * p**2 / (x ** (-1.5) + y ** (-1.5))
    else:  # pragma: no cover
        raise ValueError(mode)

    return x**0.75 * y**0.75 * s**-0.5 * bracket * ttilde * erfc_factor


def _inner(x, p, q, M, R, mode, epsabs, epsrel):
    bounds = _integration_bounds(x, p, R)
    if bounds is None:
        return 0.0
    y_min, y_max = bounds
    value, _ = integrate.quad(
        integrand_y_visc,
        y_min,
        y_max,
        args=(x, p, q, M, R, mode),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return value


def H_cone(p, M, R, mode, epsabs=1e-12, epsrel=1e-7):
    """H_ijij(p, q=p) with the chosen temporal factor.  Outer integral in ln x."""
    p = max(p, 1e-10)

    def outer(t):
        x = np.exp(t)
        return x * _inner(x, p, p, M, R, mode, epsabs, epsrel)

    value, _ = integrate.quad(
        outer, -np.log(R), 0.0, epsabs=epsabs, epsrel=epsrel, limit=200
    )
    return _h_prefactor(p, M, 1.0) * value


def omega_gw(p, M, R, mode):
    return p**3 * H_cone(p, M, R, mode)


# --------------------------------------------------------------------------
# 3.  How large is the coefficient really?  C_eff = <(k1^2+u^2)>/k^2
# --------------------------------------------------------------------------
def c_eff(p, M, R):
    """Ratio of the true floor to the paper's 'T -> 2 nu' claim, kernel-weighted."""
    num = H_cone(p, M, R, "floor")
    den = H_cone(p, M, R, "const")
    return num / den


# --------------------------------------------------------------------------
# 4.  Crossover and floor/peak
# --------------------------------------------------------------------------
def crossover_p(M, R, p_lo=1.0, p_hi=None, n=32):
    """Smallest p where the viscous floor exceeds the Gaussian sweeping curve."""
    if p_hi is None:
        p_hi = R**0.75  # stay inside the source's own support, k < k_d

    def diff(lp):
        p = np.exp(lp)
        g = omega_gw(p, M, R, "gauss")
        f = omega_gw(p, M, R, "floor")
        if f <= 0.0:
            return -np.inf
        if g <= 0.0:
            return np.inf
        return np.log(f) - np.log(g)

    grid = np.linspace(np.log(p_lo), np.log(p_hi), n)
    vals = np.array([diff(t) for t in grid])
    sign_change = np.where((vals[:-1] < 0) & (vals[1:] > 0))[0]
    if vals[0] > 0:
        return float(p_lo)
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])
    root = optimize.brentq(diff, grid[i], grid[i + 1], xtol=1e-5)
    return float(np.exp(root))


def peak_of_gaussian(M, R, n=61):
    ps = np.geomspace(1e-2, 30.0, n)
    vals = np.array([omega_gw(p, M, R, "gauss") for p in ps])
    i = int(np.argmax(vals))
    return float(ps[i]), float(vals[i])


# --------------------------------------------------------------------------
# 5.  Driver
# --------------------------------------------------------------------------
def main() -> None:
    verify_analytics()

    M = 1.0

    print()
    print("=" * 78)
    print("2.  KERNEL CROSS-CHECK  (mode='gauss' must reproduce core.H_pq)")
    print("=" * 78)
    for p in [0.05, 0.5, 2.0, 5.0]:
        mine = H_cone(p, M, 1e4, "gauss")
        ref = H_pq(p, p, M=M, R=1e4)
        print(f"    p={p:5.2f}   H_cone={mine:.8e}   core.H_pq={ref:.8e}   ratio={mine/ref:.10f}")

    print()
    print("=" * 78)
    print("3.  THE COEFFICIENT:  T_floor = 2 nu C_eff(p),  C_eff = <k1^2+u^2>/k^2")
    print("=" * 78)
    print("    (the paper claims C_eff = 1, i.e. T -> 2 nu)")
    for R in [1e4, 1e6]:
        for p in [2.0, 10.0, 100.0, 1e3]:
            if p > 2 * R**0.75:
                continue
            print(f"    R={R:.0e}  p={p:8.1f}   C_eff={c_eff(p, M, R):.4f}")

    print()
    print("=" * 78)
    print("4.  R-TABLE:  crossover and floor/peak")
    print("=" * 78)
    Rs = [1e3, 1e4, 1e5, 1e6, 1e8, 1e10]
    rows = []
    for R in Rs:
        p_peak, om_peak = peak_of_gaussian(M, R)
        p_x = crossover_p(M, R)
        p_end = 2.0 * R**0.75
        om_x = omega_gw(p_x, M, R, "floor") if np.isfinite(p_x) else np.nan
        # the floor is a decreasing power law, so its largest value is at p_x;
        # record the ratio there and at the spectrum's hard edge.
        om_end = omega_gw(0.9 * p_end, M, R, "floor")
        rows.append((R, p_peak, om_peak, p_x, om_x / om_peak, p_end, om_end / om_peak))
        print(
            f"    R={R:9.0e}  p_peak={p_peak:6.3f}  p_x={p_x:8.2f}  "
            f"floor/peak at p_x = {om_x/om_peak:.3e}   "
            f"p_end={p_end:.3e}  floor/peak at p_end = {om_end/om_peak:.3e}"
        )

    # parametric law: fit floor/peak ~ R^-alpha
    Rarr = np.array([r[0] for r in rows])
    ratio = np.array([r[4] for r in rows])
    alpha = np.polyfit(np.log10(Rarr), np.log10(ratio), 1)[0]
    print(f"\n    d log10(floor/peak at p_x) / d log10 R = {alpha:.4f}")
    px = np.array([r[3] for r in rows])
    print(f"    crossover p_x grows only logarithmically: {px}")

    print()
    print("=" * 78)
    print("5.  FIGURE")
    print("=" * 78)

    apply_paper_style(usetex=False)
    fig, axes = __import__("matplotlib.pyplot", fromlist=["x"]).subplots(
        1, 2, figsize=(FIGSIZES["large"][0] * 1.15, FIGSIZES["large"][1] * 0.52)
    )

    # ---- panel (a): the two-leg temporal factor at a single (k1,u) pair
    ax = axes[0]
    a = 1.0
    ws = np.geomspace(1e-1, 1e3, 600)
    T0 = T_closed(1e-6, a, 0.0)
    ax.loglog(ws, [T_closed(w, a, 0.0) / T0 for w in ws], color=PALETTE[0], lw=1.8,
              label=r"Gaussian, $\nu=0$")
    for i, b in enumerate([1e-2, 1e-4, 1e-6]):
        ax.loglog(
            ws,
            [T_closed(w, a, b) / T0 for w in ws],
            color=PALETTE[i + 1],
            lw=1.4,
            label=rf"$b/\sqrt{{a}}=10^{{{int(np.log10(b))}}}$",
        )
        ax.loglog(ws, 2 * b / ws**2 / T0, color=PALETTE[i + 1], ls=":", lw=1.0)
    ax.set_xlabel(r"$\omega/\sqrt{a}$")
    ax.set_ylabel(r"$T(\omega)/T(0)$")
    ax.set_ylim(1e-14, 3.0)
    ax.set_title(r"(a) two-leg temporal factor", fontsize=12)
    ax.legend(fontsize=7, loc="lower left")
    apply_max_ticks(ax)

    # ---- panel (b): the GW spectra
    ax = axes[1]
    for i, R in enumerate([1e4, 1e6, 1e8]):
        p_peak, om_peak = peak_of_gaussian(M, R)
        ps = np.geomspace(0.05, min(2 * R**0.75, 3e5), 44)
        og = np.array([omega_gw(p, M, R, "gauss") for p in ps])
        of = np.array([omega_gw(p, M, R, "floor") for p in ps])
        ax.loglog(ps, og / om_peak, color=PALETTE[i + 1], lw=1.6,
                  label=rf"$R=10^{{{int(np.log10(R))}}}$ sweeping")
        ax.loglog(ps, of / om_peak, color=PALETTE[i + 1], ls="--", lw=1.3,
                  label=rf"$R=10^{{{int(np.log10(R))}}}$ viscous floor")
    ax.set_xlabel(r"$p=k/k_0$")
    ax.set_ylabel(r"$\Omega_{\rm GW}(p)/\Omega_{\rm GW}^{\rm peak}$")
    ax.set_ylim(1e-22, 5.0)
    ax.set_title(r"(b) GW spectrum, $M=1$", fontsize=12)
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    apply_max_ticks(ax)

    fig.tight_layout()
    out = save_figure(fig, "viscous_cusp_floor")
    print(f"    wrote {out}")


if __name__ == "__main__":
    main()
