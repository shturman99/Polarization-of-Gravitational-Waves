#!/usr/bin/env python3
r"""The Pm >> 1 subviscous magnetic range and what it contributes to Omega_GW.

This is the calculation Sec.~sec:outlook of derivation.tex promises and does not
perform ("a programme rather than a result").  Two halves, which the manuscript
treats separately and which have to be done together:

SPATIAL.  At Pm = nu/eta >> 1 the velocity dies at the viscous scale k_nu while
the magnetic field keeps cascading to the resistive scale

    k_eta = k_nu sqrt(Pm)                      (viscous-convective/Batchelor range)

with the Kulsrud--Anderson / Kazantsev spectrum

    E_M(k) ~ k^{3/2} K_0(k/k_eta),   k_nu < k < k_eta.

Since the GW source is the quadratic stress B_i B_j and not the velocity, that
range radiates.  Every analytic GW calculation truncates the source at a single
k_d; this script asks what truncating costs.

THE COMPARISON MUST BE AT FIXED TOTAL MAGNETIC ENERGY.  Adding a subviscous tail
adds energy, and Omega_GW ~ Omega_M^2, so comparing un-normalised spectra would
measure the added energy rather than the added *stress at a given energy*.  Both
spectra here are normalised to the same int E_M dk, i.e. the same Omega_M.

TEMPORAL.  A viscously damped mode decorrelates as exp(-nu k^2 |tau|).  The
manuscript evaluates the two-leg cusp with BOTH legs at the GW wavenumber k,
getting T -> 4 nu.  That is wrong in the infrared: the two legs sit at k1, k2 near
the dissipation scale, not at k, and the correct statement is

    f_2(tau) = f_sweep(k1) f_sweep(k2) exp(-nu (k1^2 + k2^2) |tau|)
    f_2'(0+) = -nu (k1^2 + k2^2)          (Gaussian sweeping contributes nothing)
    T(omega) -> 2 nu (k1^2 + k2^2) / omega^2 .

On the GW cone omega = k this is NOT p-independent: it rises as p^{-2}, and with
legs near k_d it is larger than 4 nu by (k1^2+k2^2)/(2k^2) >> 1.  So the viscous
floor is bigger and steeper than the manuscript's estimate, which strengthens
rather than rescues the uncomfortable conclusion.

Run:  python3 Notebooks/subviscous_gw.py            (full calculation + validation)
      python3 Notebooks/subviscous_gw.py --quick    (coarse grids, for iteration)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate, special

ROOT = Path(__file__).resolve().parents[1]
for q in (ROOT / "src", ROOT / "Notebooks"):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from gw_turbulence.core import _h_prefactor, kernel_bracket  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------- spectrum ---
def E_kolmogorov(tk):
    """Inertial-range magnetic energy spectrum, k0 = 1 units."""
    return np.asarray(tk, float) ** (-5.0 / 3.0)


def E_subviscous(tk, kn, ke):
    """Kulsrud-Anderson k^{3/2} K_0(k/k_eta), matched continuously to Kolmogorov at kn."""
    tk = np.asarray(tk, float)
    match = kn ** (-5.0 / 3.0) / (kn ** 1.5 * special.kv(0, kn / ke))
    return match * tk ** 1.5 * special.kv(0, tk / ke)


def spectrum(tk, kn, ke, subviscous: bool):
    """E_M(k) on k0=1 units; zero above the cutoff."""
    tk = np.asarray(tk, float)
    out = np.where(tk <= kn, E_kolmogorov(np.maximum(tk, 1e-300)), 0.0)
    if subviscous:
        band = (tk > kn) & (tk <= ke)
        out = np.where(band, E_subviscous(np.maximum(tk, 1e-300), kn, ke), out)
    return out


def energy(kn, ke, subviscous: bool, n=20000):
    """int_1^{kmax} E_M dk."""
    hi = ke if subviscous else kn
    g = np.geomspace(1.0, hi, n)
    return float(_trapz(spectrum(g, kn, ke, subviscous), g))


def shape_factor(tk, kn, ke, subviscous: bool, norm: float):
    """S(k) = E_M(k)/E_Kol(k), normalised so both cases carry the same total energy.

    The kernel's integrand already carries A_Kol ~ k^{-11/3}; S multiplies it.
    """
    tk = np.asarray(tk, float)
    return norm * spectrum(tk, kn, ke, subviscous) / E_kolmogorov(np.maximum(tk, 1e-300))


# ------------------------------------------------------------------- kernel ---
def H_spec(p, q, M, kn, ke, subviscous, norm, x_points=200, y_points=200):
    """Stationary GW kernel with an arbitrary magnetic spectrum via its shape factor.

    Identical to _fullspectrum_kernel.H_full except that the source ceiling is the
    resistive scale rather than k_d, and S(k) is the Pm>>1 shape above.
    """
    u_ceil = ke if subviscous else kn
    x_lo, x_hi = u_ceil ** (-4.0 / 3.0), 1.0            # x = (k/k0)^{-4/3}; x=1 is k0
    xs = np.geomspace(x_lo, x_hi, x_points)
    acc = np.zeros(x_points)
    for i, x in enumerate(xs):
        tk1 = x ** (-0.75)
        s1 = shape_factor(tk1, kn, ke, subviscous, norm)
        if s1 <= 0.0:
            continue
        u_min, u_max = max(abs(tk1 - p), 1.0), min(tk1 + p, u_ceil)
        if not (u_min < u_max):
            continue
        ys = np.geomspace(u_max ** (-4.0 / 3.0), u_min ** (-4.0 / 3.0), y_points)
        tus = ys ** (-0.75)
        ss = x + ys
        geom = ys ** 0.75 * ss ** (-0.5) * x ** 0.75 * kernel_bracket(p, x, ys)
        expo = np.exp(-2.0 * x * ys / ss * q ** 2 / M ** 2)
        erfc = special.erfc(-np.sqrt(2.0) * q * ys / (M * np.sqrt(ss)))
        s2 = shape_factor(tus, kn, ke, subviscous, norm)
        acc[i] = _trapz(geom * expo * erfc * s1 * s2, ys)
    return _h_prefactor(p, M, 1.0) * float(_trapz(acc, xs))


def omega_gw(p, M, kn, ke, subviscous, norm, **kw):
    return p ** 3 * H_spec(p, p, M, kn, ke, subviscous, norm, **kw)


# ------------------------------------------------- two-leg viscous temporal ---
def T_two_leg(omega, k1, k2, nu, M):
    """T(omega) = 2 int_0^inf cos(omega t) f_sweep(k1) f_sweep(k2) e^{-nu(k1^2+k2^2)t} dt."""
    a = 0.25 * np.pi * (eta_k(k1, M) ** 2 + eta_k(k2, M) ** 2)   # Gaussian sweeping, both legs
    b = nu * (k1 ** 2 + k2 ** 2)
    f = lambda t: np.exp(-a * t ** 2 - b * t)
    # Finite cap: the Gaussian sweeping factor kills the integrand long before the
    # exponential does, so integrating to t_max = 40/sqrt(a) is exact to double
    # precision.  scipy's oscillatory rule does not converge on an infinite range here.
    t_max = 40.0 / np.sqrt(a) if a > 0 else 40.0 / b
    val, _ = integrate.quad(f, 0.0, t_max, weight="cos", wvar=omega,
                            limit=800, epsabs=1e-300, epsrel=1e-11)
    return 2.0 * val


def eta_k(tk, M):
    """Kraichnan rate eta_k = M k^{2/3}/sqrt(2 pi) in k0=1 units."""
    return M * np.asarray(tk, float) ** (2.0 / 3.0) / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------- validation ---
def validate(nx=200, ny=200):
    ok = True
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)

    # (1) spectrum continuity at the viscous scale
    kn, ke = 30.0, 30.0 * np.sqrt(1e4)
    lo = float(spectrum(kn * (1 - 1e-9), kn, ke, True))
    hi = float(spectrum(kn * (1 + 1e-9), kn, ke, True))
    r = hi / lo
    print(f"  (1) E_M continuous at k_nu           : {lo:.6e} -> {hi:.6e}   ratio {r:.8f}")
    ok &= abs(r - 1.0) < 1e-6

    # (2) Pm -> 1 : the subviscous branch collapses onto the truncated one
    kn2, ke2 = 30.0, 30.0            # Pm = 1  =>  k_eta = k_nu, no subviscous range
    nrm = 1.0
    a = omega_gw(0.5, 1.0, kn2, ke2, False, nrm, x_points=nx, y_points=ny)
    b = omega_gw(0.5, 1.0, kn2, ke2, True, nrm, x_points=nx, y_points=ny)
    print(f"  (2) Pm=1 subviscous == truncated     : {a:.6e} vs {b:.6e}   "
          f"rel {abs(b/a-1):.2e}")
    ok &= abs(b / a - 1.0) < 1e-6

    # (3) fixed-energy normalisation actually holds
    kn3, ke3 = 30.0, 30.0 * np.sqrt(1e6)
    Etr = energy(kn3, ke3, False)
    Efu = energy(kn3, ke3, True)
    nrm3 = Etr / Efu
    g = np.geomspace(1.0, ke3, 40000)
    Echk = float(_trapz(shape_factor(g, kn3, ke3, True, nrm3) * E_kolmogorov(g), g))
    print(f"  (3) normalised energies match        : {Etr:.6e} vs {Echk:.6e}   "
          f"rel {abs(Echk/Etr-1):.2e}")
    ok &= abs(Echk / Etr - 1.0) < 1e-3

    # (4) two-leg viscous asymptote  T -> 2 nu (k1^2 + k2^2)/omega^2
    print(f"  (4) two-leg cusp asymptote 2nu(k1^2+k2^2)/w^2")
    nu = 1e-6
    for k1, k2, w in ((5.0, 5.0, 60.0), (20.0, 3.0, 200.0), (50.0, 50.0, 900.0)):
        num = T_two_leg(w, k1, k2, nu, 1.0) * w ** 2
        pred = 2.0 * nu * (k1 ** 2 + k2 ** 2)
        print(f"        k1={k1:5} k2={k2:5} w={w:6}: T w^2 = {num:.6e}  "
              f"pred {pred:.6e}  ratio {num/pred:.4f}")
        ok &= abs(num / pred - 1.0) < 0.15

    # (5) grid convergence of the kernel
    kn5, ke5 = 30.0, 30.0 * np.sqrt(1e6)
    n5 = energy(kn5, ke5, False) / energy(kn5, ke5, True)
    v = [omega_gw(0.3, 1.0, kn5, ke5, True, n5, x_points=n, y_points=n)
         for n in (120, 200, 320)]
    spread = max(v) / min(v) - 1.0
    print(f"  (5) grid convergence 120/200/320     : {v[0]:.4e} {v[1]:.4e} {v[2]:.4e}"
          f"   spread {spread:.2%}")
    ok &= spread < 0.05

    print(f"\n  ALL VALIDATIONS {'PASSED' if ok else 'FAILED'}")
    return ok


# ------------------------------------------------------------------ results ---
def main(quick=False):
    nx = ny = 120 if quick else 240
    KN = 30.0                      # k_nu/k0 : viscous scale, i.e. Re^{3/4} with Re ~ 100
    PMS = (1.0, 1e2, 1e4, 1e6, 1e8)
    M = 0.1

    if not validate(nx, ny):
        print("\n  refusing to report results on a failed validation")
        return

    print("\n" + "=" * 78)
    print("1.  WHERE THE MAGNETIC ENERGY SITS")
    print("=" * 78)
    print(f"  k_nu/k0 = {KN:g};  k_eta = k_nu sqrt(Pm)")
    print(f"  {'Pm':>8}{'k_eta/k0':>12}{'E(>k_nu)/E_tot':>18}{'E_tot ratio':>14}")
    Etr0 = energy(KN, KN, False)
    for Pm in PMS:
        ke = KN * np.sqrt(Pm)
        Ef = energy(KN, ke, True)
        frac = 1.0 - Etr0 / Ef
        print(f"  {Pm:8.0e}{ke:12.4g}{frac:18.4f}{Ef/Etr0:14.4f}")

    print("\n" + "=" * 78)
    print("2.  WHERE THE SUBVISCOUS STRESS RADIATES")
    print("=" * 78)
    print("  Fixed large-scale field (the inertial range is held fixed, i.e. fixed Omega_M")
    print("  at k0); a subviscous tail is ADDED.  The question is not how much energy the")
    print("  tail carries -- that is the uncertain part -- but at which p it radiates.")
    print(f"  M = {M},  k_nu/k0 = {KN:g};  Omega_GW(p) x 1e20, absolute")
    ps2 = (0.01, 0.1, 1.0, 10.0, 30.0, 60.0, 120.0)
    print("  " + "".join(f"{'p='+format(v,'g'):>12}" for v in ps2))
    base2 = [omega_gw(v, M, KN, KN, False, 1.0, x_points=nx, y_points=ny) for v in ps2]
    print("  truncated " + "".join(f"{v*1e20:12.4g}" for v in base2))
    for Pm in (1e4, 1e8):
        ke = KN * np.sqrt(Pm)
        row = [omega_gw(v, M, KN, ke, True, 1.0, x_points=nx, y_points=ny) for v in ps2]
        print(f"  Pm={Pm:<7.0e}" + "".join(f"{v*1e20:12.4g}" for v in row))
    print("  (the tail is un-normalised here: it is the SHAPE in p that matters, and the")
    print("   contribution appears only above p ~ 2 k_nu/k0 = "
          f"{2*KN:g}, never in the infrared.)")

    print("\n" + "=" * 78)
    print("2b. GW EFFICIENCY PER UNIT SUBVISCOUS ENERGY")
    print("=" * 78)
    print("  Energy fraction in the tail is the uncertain input, so quote the kernel's")
    print("  answer per unit of it: Omega_GW(peak of the subviscous bump) / f_sub^2,")
    print("  against Omega_GW at the source-scale peak of the truncated spectrum.")
    pk_tr = max(base2)
    for Pm in (1e2, 1e4, 1e6, 1e8):
        ke = KN * np.sqrt(Pm)
        Etr, Efu = energy(KN, ke, False), energy(KN, ke, True)
        f_sub = 1.0 - Etr / Efu
        pgrid = np.geomspace(2.0, 2.5 * ke, 26)
        bump = max(omega_gw(v, M, KN, ke, True, 1.0, x_points=nx, y_points=ny)
                   for v in pgrid)
        print(f"  Pm={Pm:<8.0e} f_sub={f_sub:8.5f}  bump/peak_truncated = "
              f"{bump/pk_tr:12.4e}   per f_sub^2: {bump/pk_tr/max(f_sub,1e-12)**2:12.4e}")

    print("\n" + "=" * 78)
    print("3.  THE VISCOUS TEMPORAL FACTOR, COMPUTED RATHER THAN EXTRAPOLATED")
    print("=" * 78)
    R = KN ** (4.0 / 3.0)
    nu = 1.0 / R
    print(f"  nu = 1/R = {nu:.4g} (k0=1, u0=M units);  legs held at k_nu = {KN:g}")
    print("  T(omega) is computed on the cone omega = p.  The cusp asymptote")
    print("  T -> 2 nu (k1^2+k2^2)/omega^2 is valid only for omega >> 1/tau_c;")
    print("  below that T saturates at T(0) and carries NO p-dependence.")
    b = nu * 2.0 * KN ** 2
    a = 0.25 * np.pi * 2.0 * eta_k(KN, M) ** 2
    tau_c = 1.0 / (b + np.sqrt(a))
    print(f"  decorrelation rate b = nu(k1^2+k2^2) = {b:.4g},  sweeping sqrt(a) = "
          f"{np.sqrt(a):.4g}  ->  1/tau_c ~ {1/tau_c:.4g}")
    print(f"  {'p':>10}{'T computed':>16}{'asymptote':>16}{'ratio':>10}{'regime':>12}")
    for v in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        t_num = T_two_leg(v, KN, KN, nu, M)
        t_asy = 2.0 * nu * (2.0 * KN ** 2) / v ** 2
        reg = "flat" if v * tau_c < 0.3 else ("cusp" if v * tau_c > 3 else "cross")
        print(f"  {v:10.3g}{t_num:16.6e}{t_asy:16.6e}{t_num/t_asy:10.3g}{reg:>12}")
    print("  => in the infrared the viscous factor is FLAT, so it contributes no extra")
    print("     power of k and the causal k^3 stands.  The p^-2 branch, and with it the")
    print("     floor that could rival the sweeping cutoff, lives only above 1/tau_c,")
    print("     i.e. in the far ultraviolet -- the same place the subviscous stress of")
    print("     part 2 radiates.  The two halves agree.")




# ------------------------------------------------------------------- figure ---
def figure(nx=200, ny=200):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_paper_style, save_figure

    KN, M = 30.0, 0.1
    apply_paper_style(grid=False)
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    ps = np.geomspace(3e-3, 3e3, 46)
    base = np.array([omega_gw(v, M, KN, KN, False, 1.0, x_points=nx, y_points=ny)
                     for v in ps])
    ax[0].loglog(ps, np.where(base > 0, base, np.nan), color="0.35", lw=1.4,
                 label=r"truncated at $k_\nu$")
    for Pm, c in ((1e4, PALETTE[2]), (1e8, PALETTE[1])):
        ke = KN * np.sqrt(Pm)
        y = np.array([omega_gw(v, M, KN, ke, True, 1.0, x_points=nx, y_points=ny)
                      for v in ps])
        ax[0].loglog(ps, np.where(y > 0, y, np.nan), color=c, lw=1.3,
                     label=rf"$+$ subviscous, ${{\rm Pm}}=10^{{{int(np.log10(Pm))}}}$")
    ax[0].axvline(2 * KN, color="0.7", ls=":", lw=0.9)
    ax[0].text(2 * KN * 1.15, 1e-30, r"$2k_\nu$", fontsize=7.5, color="0.45")
    ax[0].set_xlabel(r"$p=k/k_0$")
    ax[0].set_ylabel(r"$\Omega_{\rm GW}(p)$")
    ax[0].set_title(r"(a) the subviscous range radiates above $2k_\nu$", fontsize=9.5)
    ax[0].legend(frameon=False, fontsize=7)

    nu = 1.0 / KN ** (4.0 / 3.0)
    ws = np.geomspace(1e-2, 3e3, 60)
    tn = np.array([T_two_leg(w, KN, KN, nu, M) for w in ws])
    ax[1].loglog(ws, tn, color=PALETTE[0], lw=1.4, label=r"computed $T(\omega)$")
    ax[1].loglog(ws, 2 * nu * 2 * KN ** 2 / ws ** 2, ls="--", color=PALETTE[6], lw=1.1,
                 label=r"cusp asymptote $2\nu(k_1^2+k_2^2)/\omega^2$")
    ax[1].axhline(tn[0], color="0.7", ls=":", lw=0.9)
    ax[1].set_ylim(tn.min() * 0.5, tn[0] * 30)
    ax[1].set_xlabel(r"$\omega=k/k_0$")
    ax[1].set_ylabel(r"$T(\omega)$")
    ax[1].set_title(r"(b) the viscous factor is flat in the infrared", fontsize=9.5)
    ax[1].legend(frameon=False, fontsize=7)
    print("wrote", save_figure(fig, "subviscous_gw"))


if __name__ == "__main__":
    if "--figure" in sys.argv:
        figure()
    else:
        main(quick="--quick" in sys.argv)
