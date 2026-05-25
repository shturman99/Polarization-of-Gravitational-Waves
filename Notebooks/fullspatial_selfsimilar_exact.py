#!/usr/bin/env python3
r"""First-principles full-spatial self-similar decaying GW kernel.

This is the rigorous version of ``fullspatial_selfsimilar.py``: it replaces the
illustrative coherence *knob* with the genuine self-similar slow-time integration
on the full-spatial kernel.  The geometry (geometric bracket, triangle-inequality
bounds, prefactor, and the leg wavenumbers k1 = x^{-3/4}, k2 = y^{-3/4}) is reused
verbatim from ``core`` / ``fullspatial_decay``; only the temporal factor changes.

Temporal factor
---------------
The quasi-stationary factor of ``fullspatial_decay`` is the single-emission-epoch
Fourier transform of the product of the two legs' BK2016 decorrelations,

    conv_ftprod(q; t1, t2) = (2 pi / t1 t2) int_0^inf cos(q t) R1(t) R2(t) dt,
    R_i(t) = (1 + t/tau_i)^{-2/3},   tau_i = sqrt(x or y)/M.

The self-similar source instead acts over a finite emission window [0, T_em] while
the turbulence decays, so both the amplitude and the decorrelation time evolve with
the (slow) emission time T.  The genuine temporal factor is the full double-time
integral over the window,

    T_SS = (pi / (t1_0 t2_0 T_em)) int_0^{T_em} int_0^{T_em} dt1 dt2
             cos(omega (t1-t2)) rho1(t1,t2) rho2(t1,t2),

    rho_i(t1,t2) = sqrt(a(t1) a(t2)) (1 + |t1-t2| / tau_i(T))^{-2/3},   T=(t1+t2)/2,

with the self-similar (Loitsiansky p=10/7, q=2/7) evolution

    tau_i(T) = tau_{i,0} (1 + T/tau_st)^{17/21}      (decorrelation time GROWS)
    a(t)     = (1 + t/tau_st)^{-34/21}               (Phi_eq amplitude DECAYS, inertial)

The normalisation pi/(t1_0 t2_0 T_em) makes T_SS reduce EXACTLY to conv_ftprod (and
H_ss_exact reduce to fullspatial_decay) in the no-decay limit tau_st -> inf,
T_em >> tau_c.  This is the non-negotiable cross-check, verified in _validate().

The double-time integral is evaluated in causal-window (T, dtau) coordinates
[t1=T+dtau/2, t2=T-dtau/2; |dtau| <= 2 min(T, T_em-T)] so the inner grid resolves
the eddy/oscillation scale without wasting points on the empty off-diagonal.

Units (M-coupling): tau_{i,0} = sqrt(x or y)/M is the sweeping time; t and T_em are
in the same units; the decay time tau_st sets eps0 ~ 1/(M tau_st) (decay time in
eddy times).  This is the inertial-range model, consistent with the pure-Kolmogorov
geometric weight of fullspatial_decay.

Run: python Notebooks/fullspatial_selfsimilar_exact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "Notebooks") not in sys.path:
    sys.path.insert(0, str(ROOT / "Notebooks"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from gw_turbulence.core import _h_prefactor, _integration_bounds, kernel_bracket  # noqa: E402

# Loitsiansky self-similar exponents
ALPHA = 17.0 / 21.0   # tau_c(T) ~ (1+T/tau_st)^ALPHA   (decorrelation-time growth)
GAMMA = 34.0 / 21.0   # Phi_eq(t)/Phi_eq(0) ~ (1+t/tau_st)^{-GAMMA}  (inertial amplitude decay)


def _amp(t, tau_st):
    """Phi_eq(k,t)/Phi_eq(k,0) in the inertial range (k-independent)."""
    return (1.0 + t / tau_st) ** (-GAMMA)


def _wigner_pair(omega, T, T_em, tau1_0, tau2_0, tau_st):
    """Inner causal-window integral int dtau cos(omega dtau) a(t1) a(t2) R1 R2.

    Both times t1=T+dtau/2, t2=T-dtau/2 are kept in [0, T_em] via the causal window
    |dtau| <= 2 min(T, T_em-T).  Evaluated with scipy's oscillatory cosine quadrature
    (exact for the cos weight) over [0, half] (the integrand is even in dtau, so the
    two-sided integral is twice the one-sided one).  This captures the heavy
    oscillatory tail of the BK2016 decorrelation that a capped uniform grid truncated,
    which is what made the no-decay reduction converge to 1.15x instead of 1x.
    """
    w = min(T, T_em - T)
    if w <= 0.0:
        return 0.0
    half = 2.0 * w
    tau1 = tau1_0 * (1.0 + T / tau_st) ** ALPHA
    tau2 = tau2_0 * (1.0 + T / tau_st) ** ALPHA

    def g(dt):  # even integrand for dt >= 0 (smooth; no |.| cusp on the half-line)
        return (_amp(T + dt / 2.0, tau_st) * _amp(T - dt / 2.0, tau_st)
                * (1.0 + dt / tau1) ** (-2.0 / 3.0) * (1.0 + dt / tau2) ** (-2.0 / 3.0))

    if omega <= 0:
        val, _ = integrate.quad(g, 0.0, half, limit=200)
    else:
        val, _ = integrate.quad(g, 0.0, half, weight="cos", wvar=omega, limit=200)
    return 2.0 * float(val)


def _conv_ss_exact(omega, x, y, M, tau_st, T_em, n_T=60):
    """Self-similar temporal factor; reduces to fullspatial_decay._conv_ftprod as tau_st->inf."""
    tau1_0 = np.sqrt(x) / M
    tau2_0 = np.sqrt(y) / M
    Ts = np.linspace(0.0, T_em, n_T)
    inner = np.array([_wigner_pair(omega, T, T_em, tau1_0, tau2_0, tau_st) for T in Ts])
    dbl = np.trapezoid(inner, Ts)                      # int_0^Tem dT int dtau (...)
    return np.pi / (tau1_0 * tau2_0 * T_em) * dbl


def H_ss_exact(p, q, M=1.0, R=1e4, tau_st=1e9, T_em=40.0,
               x_points=26, y_points=26, n_T=60):
    """Full-spatial self-similar GW kernel H(p,q; M,R, tau_st, T_em)."""
    xs = np.geomspace(1.0 / R, 1.0, x_points)
    x_integrand = np.zeros(x_points)
    for i, x in enumerate(xs):
        bounds = _integration_bounds(x, p, R)
        if bounds is None:
            continue
        y_min, y_max = bounds
        ys = np.geomspace(y_min, y_max, y_points)
        vals = [
            yy**0.75 * (x + yy) ** (-0.5) * x**0.75
            * kernel_bracket(p, x, yy)
            * _conv_ss_exact(q, x, yy, M, tau_st, T_em, n_T=n_T)
            for yy in ys
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def omega_gw(p, M=1.0, R=1e4, tau_st=1e9, T_em=40.0, **kw):
    return p**3 * H_ss_exact(p, p, M=M, R=R, tau_st=tau_st, T_em=T_em, **kw)


# ---------------------------------------------------------------------------
def _peak(spec, plo=0.3, phi=8.0, n=30):
    ps = np.geomspace(plo, phi, n)
    sp = np.array([spec(pp) for pp in ps])
    i = int(np.argmax(sp))
    il, ir = max(i - 1, 0), min(i + 1, len(ps) - 1)
    c = np.polyfit(np.log(ps[il:ir + 1]), np.log(sp[il:ir + 1]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


def _band_slope(spec, p_lo, p_hi, n=8):
    ps = np.geomspace(p_lo, p_hi, n)
    sp = np.array([spec(pp) for pp in ps])
    g = sp > 0
    return float(np.polyfit(np.log(ps[g]), np.log(sp[g]), 1)[0])


def _validate():
    from fullspatial_decay import H_decay_fast  # noqa: E402

    print("=" * 74)
    print("VALIDATION  (first-principles full-spatial self-similar kernel)")
    print("=" * 74)

    print("\n(1) NO-DECAY reduction -> fullspatial_decay; ratio must -> 1 as T_em grows")
    print("    (with exact oscillatory inner quad; the old capped trapz drifted to ~1.15)")
    b = H_decay_fast(1.0, 1.0, M=1.0, R=1e4)
    for Tem in (60.0, 150.0, 300.0):
        a = H_ss_exact(1.0, 1.0, M=1.0, R=1e4, tau_st=1e12, T_em=Tem,
                       x_points=24, y_points=24, n_T=int(Tem / 3))
        print(f"    T_em={Tem:5.0f}:  ratio={a/b:.4f}")

    print("\n(2) IR slope (band 0.08-0.5) and positivity vs decay rate (M=1, T_em=40):")
    print(f"    {'tau_st':>8}{'eps0~':>8}{'IR slope':>10}{'all Om>0?':>10}")
    ps_band = np.geomspace(0.08, 0.5, 6)
    ps_full = np.geomspace(0.08, 5.0, 14)
    for tau_st in (1e12, 1.0, 0.25):
        og_b = np.array([p**3 * H_ss_exact(p, p, M=1.0, tau_st=tau_st, T_em=40.0,
                                           x_points=22, y_points=22, n_T=40) for p in ps_band])
        og_f = np.array([p**3 * H_ss_exact(p, p, M=1.0, tau_st=tau_st, T_em=40.0,
                                           x_points=22, y_points=22, n_T=40) for p in ps_full])
        sl = float(np.polyfit(np.log(ps_band), np.log(og_b), 1)[0])
        eps0 = 1.0 / tau_st
        lab = "inf" if tau_st > 1e6 else f"{tau_st:.2f}"
        print(f"    {lab:>8}{eps0:>8.2g}{sl:>10.2f}{str(bool((og_f > 0).all())):>10}")
    print("    -> test: does the IR slope stay causal (~2.5-2.8) and decay-independent?")


def _figure(name="fullspatial_selfsimilar_exact"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)

    # Panel (a): rigorous full-spatial self-similar spectra at several decay rates.
    # The infrared slopes OVERLAP (causal, decay-independent) -- decay does not flatten
    # them toward k^1; the source-scale peak is common.
    # Peak region (p>~1.5) requires fine resolution: the kernel bracket changes sign
    # and is under-resolved at lower grids (spurious dips/negatives near the peak; the
    # fast-decay eps0=4 case needs 60x60/n_T=90 to stay positive through the peak).
    # Resolution must be UNIFORM across p -- the kernel's absolute scale shifts with grid,
    # so mixing resolutions would put a spurious step in the curve.
    ps = np.geomspace(0.04, 3.0, 13)
    cases = [(1e12, PALETTE[0], r"no decay ($\varepsilon_0\!\to\!0$)"),
             (1.0, PALETTE[2], r"$\varepsilon_0=1$"),
             (0.25, PALETTE[1], r"$\varepsilon_0=4$ (fast)")]
    for tau_st, col, lab in cases:
        sp = np.array([omega_gw(p, 1.0, tau_st=tau_st, T_em=40.0,
                                x_points=60, y_points=60, n_T=90) for p in ps])
        ax0.plot(ps, sp / sp.max(), "-", color=col, lw=1.8, label=lab)
    pr = np.geomspace(0.06, 0.5, 8)
    ax0.plot(pr, 0.55 * (pr / 0.5) ** 3, ":", color="0.45", lw=1.2, label=r"$k^3$ (causal)")
    ax0.plot(pr, 0.55 * (pr / 0.5) ** 1, "--", color="0.45", lw=1.2, label=r"$k^1$ (DNS)")
    ax0.axvspan(1.8, 2.7, color="0.7", alpha=0.18, lw=0)
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_ylim(1e-4, 2)
    ax0.set_xlabel(r"$p=k/k_0$")
    ax0.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,\Omega_{\rm GW}^{\rm peak}$")
    ax0.set_title(r"full-spatial self-similar spectra", fontsize=10)
    ax0.legend(fontsize=7, loc="lower right")

    # Panel (b): IR slope vs decay rate -- flat at the causal value, NOT interpolating to 1.
    eps0s = np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 4.0])
    band = np.geomspace(0.06, 0.45, 6)
    slopes = []
    for e0 in eps0s:
        og = np.array([p**3 * H_ss_exact(p, p, M=1.0, tau_st=1.0 / e0, T_em=40.0,
                                         x_points=24, y_points=24, n_T=38) for p in band])
        slopes.append(np.polyfit(np.log(band), np.log(og), 1)[0])
    ax1.semilogx(eps0s, slopes, "o-", color=PALETTE[3], lw=1.8, ms=4,
                 label="full-spatial self-similar")
    ax1.axhline(3.0, color="0.45", ls=":", lw=1.2)
    ax1.axhline(1.0, color="0.45", ls="--", lw=1.2)
    ax1.text(eps0s[0], 2.7, r"causal $k^3$", fontsize=8, color="0.35")
    ax1.text(eps0s[0], 1.1, r"DNS $k^1$", fontsize=8, color="0.35")
    ax1.set_xlabel(r"decay rate $\varepsilon_0=\tau_{\rm eddy}/\tau_{\rm st}$")
    ax1.set_ylabel(r"IR slope $d\ln\Omega_{\rm GW}/d\ln k$")
    ax1.set_title(r"IR stays causal at every decay rate", fontsize=10)
    ax1.set_ylim(0.6, 3.4)
    for ax in (ax0, ax1):
        apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"\nwrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    _validate()
    _figure()
