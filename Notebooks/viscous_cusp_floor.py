"""The viscous cusp: does a damped mode's kink put a power-law floor under the
Gaussian sweeping cutoff, and how big is it?

Sec. `sec:outlook` of derivation.tex asserts, without computing, that a viscously
damped mode decorrelates as exp(-nu k^2 |tau|), whose one-sided slope at zero lag
is -nu k^2 rather than zero, so viscosity SUPPLIES the kink that the cusp theorem
Eq. (eq:cusp-tail) requires even when the sweeping correlator is the exact Gaussian
measured by Auclair et al.  This script checks that assertion.  Two results.

(1) THE COEFFICIENT IS 4 nu, NOT 2 nu.  The source is quadratic, so the temporal
    factor is the transform of the *squared* correlator -- the same two-leg rule
    that gives f'(0+) = -2/tau_c rather than -1/tau_c for the tent in
    Sec. `sec:ir-branch`.  With f1 = exp(-(pi/4) eta_k^2 tau^2) exp(-nu k^2 |tau|),

        f2 = f1^2,   f2'(0+) = 2 f1(0) f1'(0+) = -2 nu k^2,

    so by the cusp theorem T -> 4 nu k^2 / omega^2, which ON THE GW CONE omega = k
    is the constant 4 nu.  The manuscript's "2 nu" is the one-leg value.

(2) IN THE PAPER'S OWN VARIABLES THE FLOOR IS That -> 4 M / R.  With
    k_d = (eps/nu^3)^{1/4} and R = (k_d/k0)^{4/3},

        nu = eps^{1/3} k0^{-4/3} R^{-1},   u0 = (eps/k0)^{1/3} = M   (c = 1),

    so the dimensionless viscous rate is (nu k^2)/k0 = M p^2 / R, and the
    dimensionless temporal factor floors at That = T k0 -> 4 nu k0 = 4 M / R,
    INDEPENDENT of p.

The consequence is not the one the manuscript anticipates.  On the cone the
two-leg Gaussian factor is ~ exp(-p^{2/3}/M^2), which for subsonic flow is
astronomically small by p ~ M, so a p-independent floor does not sit "beneath the
cutoff" -- it REPLACES it, and the crossover falls below the sweeping peak
1.488 M for M <~ 0.1.  Whether that survives in the full two-wavevector kernel is
NOT settled here; this is the temporal factor alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

SQRT2PI = np.sqrt(2.0 * np.pi)


def eta_hat(p: float, M: float) -> float:
    """Sweeping rate in units of k0:  eta_k = eps^{1/3} k^{2/3} / sqrt(2 pi)."""
    return M * p ** (2.0 / 3.0) / SQRT2PI


def nu_rate(p: float, M: float, R: float) -> float:
    """Viscous rate (nu k^2) in units of k0, = M p^2 / R."""
    return M * p ** 2 / R


def T_hat(p: float, M: float, R: float, viscous: bool = True) -> float:
    """Two-leg temporal factor That(q) = 2 int_0^inf cos(q t) f1(t)^2 dt on the
    cone q = p, with f1 the sweeping Gaussian times the viscous exponential."""
    e2 = eta_hat(p, M) ** 2
    nr = nu_rate(p, M, R) if viscous else 0.0
    f2 = lambda t: np.exp(-0.5 * np.pi * e2 * t ** 2 - 2.0 * nr * np.abs(t))
    val, _ = integrate.quad(lambda t: np.cos(p * t) * f2(t), 0.0, 2000.0,
                            limit=2000, epsabs=1e-300, epsrel=1e-12)
    return 2.0 * val


def log_T_hat_gauss(p: float, M: float) -> float:
    """log of the pure-Gaussian two-leg factor, in closed form so it stays
    representable where the value itself underflows:
        That = sqrt(2)/eta * exp(-p^2 / (2 pi eta^2)),  p^2/(2 pi eta^2) = p^{2/3}/M^2.
    """
    return np.log(np.sqrt(2.0) / eta_hat(p, M)) - p ** (2.0 / 3.0) / M ** 2


def crossover(M: float, R: float) -> float:
    """p where the Gaussian factor falls to the viscous floor 4M/R."""
    target = np.log(4.0 * M / R)
    f = lambda lp: log_T_hat_gauss(np.exp(lp), M) - target
    try:
        return float(np.exp(optimize.brentq(f, np.log(1e-6), np.log(1e4))))
    except ValueError:
        return float("nan")


def main(name: str = "viscous_cusp_floor"):
    apply_paper_style()

    print("=" * 78)
    print("(1) two-leg coefficient: is the floor 2 nu or 4 nu?")
    print("=" * 78)
    eta, nk2 = 1.0, 0.05
    f1 = lambda t: np.exp(-0.25 * np.pi * eta ** 2 * t ** 2 - nk2 * np.abs(t))
    for legs, f in ((1, f1), (2, lambda t: f1(t) ** 2)):
        for w in (100.0, 1000.0):
            v, _ = integrate.quad(lambda t: np.cos(w * t) * f(t), 0.0, 400.0,
                                  limit=800, epsabs=1e-14, epsrel=1e-13)
            print(f"  {legs} leg(s)  omega={w:6.0f}   T*omega^2 = {2 * v * w ** 2:.6f}"
                  f"   (predicted {2 * nk2 if legs == 1 else 4 * nk2})")
    print("  => the quadratic source gives 4 nu, not the 2 nu of sec:outlook.\n")

    print("=" * 78)
    print("(2) the floor in the paper's variables:  That -> 4 M / R")
    print("=" * 78)
    print(f"  {'M':>6} {'R':>8} {'4M/R':>12} {'That(p=3)':>12} {'That(p=30)':>12} {'ratio':>8}")
    for M in (0.5, 0.1, 0.05):
        for R in (1e3, 1e4):
            fl = 4 * M / R
            a, b = T_hat(3.0, M, R), T_hat(30.0, M, R)
            print(f"  {M:6.2f} {R:8.0e} {fl:12.4e} {a:12.4e} {b:12.4e} {b / fl:8.4f}")

    print("\n" + "=" * 78)
    print("(3) where the floor overtakes the sweeping cutoff, vs the peak 1.488 M")
    print("=" * 78)
    print(f"  {'M':>6} {'R':>8} {'p_cross':>10} {'p_peak=1.488M':>15}   verdict")
    rows = []
    for M in (0.5, 0.3, 0.1, 0.05):
        for R in (1e3, 1e4, 1e5):
            pc, pk = crossover(M, R), 1.488 * M
            rows.append((M, R, pc, pk))
            v = "floor takes over BELOW the peak" if pc < pk else "floor sits above the peak"
            print(f"  {M:6.2f} {R:8.0e} {pc:10.4g} {pk:15.4g}   {v}")
    print("\n  On the cone the Gaussian is exp(-p^{2/3}/M^2), so for subsonic flow it is")
    print("  already negligible by p ~ M and a p-independent floor does not sit under")
    print("  the cutoff -- it replaces it.  NOT yet checked in the full 2-wavevector")
    print("  kernel; this is the temporal factor alone.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)

    ps = np.geomspace(1e-2, 1e2, 220)
    for c, (M, R) in enumerate(((0.5, 1e4), (0.1, 1e4), (0.05, 1e4))):
        col = PALETTE[(c + 1) % len(PALETTE)]
        gauss = np.array([np.exp(log_T_hat_gauss(p, M)) for p in ps])
        axA.loglog(ps, np.maximum(gauss, 1e-300), color=col, lw=1.4,
                   label=rf"$M={M:g}$, no $\nu$")
        axA.axhline(4 * M / R, color=col, ls="--", lw=1.0)
        axA.axvline(1.488 * M, color=col, ls=":", lw=0.8)
    axA.set_xlabel(r"$p=k/k_0$")
    axA.set_ylabel(r"$\hat T$ on the cone $q=p$")
    axA.set_title(r"(a) Gaussian cutoff vs viscous floor $4M/R$")
    axA.set_ylim(1e-12, 1e3)
    axA.legend(loc="lower left", fontsize=7.5, handlelength=1.3)
    apply_max_ticks(axA)

    Ms = np.geomspace(0.02, 1.0, 60)
    for c, R in enumerate((1e3, 1e4, 1e5)):
        col = PALETTE[(c + 2) % len(PALETTE)]
        axB.loglog(Ms, [crossover(M, R) for M in Ms], color=col, lw=1.4,
                   label=rf"$R=10^{{{int(np.log10(R))}}}$")
    axB.loglog(Ms, 1.488 * Ms, color="0.35", ls="--", lw=1.2,
               label=r"peak $\xi_\ast M$")
    axB.set_xlabel(r"$M$")
    axB.set_ylabel(r"$p$")
    axB.set_title(r"(b) crossover vs the sweeping peak")
    axB.legend(loc="upper left", fontsize=7.5, handlelength=1.4)
    apply_max_ticks(axB)

    out = save_figure(fig, name)
    print(f"\nsaved {out}")
    return out


if __name__ == "__main__":
    main()
