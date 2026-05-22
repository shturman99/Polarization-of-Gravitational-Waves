#!/usr/bin/env python3
r"""
Self-similar decaying-turbulence GW spectrum: hybrid exact / O(eps^2) kernel.

Background
----------
The self-similar derivation (``decaying_selfsimilar_derivation.ipynb``) writes the
aeroacoustic GW source at slow time T as

    H(0, omega; T) = 7/(3 pi^2) int dk k^2 W_PhiPhi(k; omega, T),

with W_PhiPhi the temporal (Wigner) transform of the squared two-time velocity
correlator Phi(k; t1, t2)^2.  The notebook evaluates W_PhiPhi by a slow/fast
gradient expansion truncated at O(eps^2),

    W_PhiPhi ~ (Phi^2 tau_1 / 2pi) [ G_BK(q) - (xi/4) G_BK''(q) ],   q = omega tau_1,

with xi = tau_1^2 d^2 ln Phi / dT^2 ~ eps^2 and eps = tau_1 / (tau_* + T).

AUDIT FINDING (2026-05-22).  The expansion parameter is only small in the deep
inertial range (kL >> 1), but the k-integral is dominated by the spectral peak
kappa ~ 1, where for the cosmologically relevant FAST decay (decay time ~ eddy
time, eps_0 = tau_{e,0}/tau_* ~ 1) the parameter xi is O(1).  Truncating at
O(eps^2) then drives the GW source NEGATIVE (unphysical).

FIX (this module).  Compute W_PhiPhi EXACTLY as the Wigner transform of Phi^2 over
the CAUSAL window (both times >= 0, the source switches on at t=0):

    W_PhiPhi(k; omega, T) = int_{|tau| <= 2 min(T, T_em - T)} dtau cos(omega tau)
                            Phi(k; T + tau/2, T - tau/2)^2.

Phi^2 is a positive-type, even-in-tau kernel, so this is real and >= 0 by
construction.  ``H_total`` dispatches: cheap O(eps^2) kernel for slow decay
(eps_0 <= eps_switch), exact Wigner otherwise.

Run ``python Notebooks/selfsimilar_hybrid.py`` for the validation table and the
cross-check figure ``images/selfsimilar_hybrid_crosscheck.pdf``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if not hasattr(np, "trapezoid"):  # numpy < 2.0
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

# Loitsiansky default class (p, q, s); pass overrides via **pars (Saffman: 6/5, 2/5, 2).
LOITSIANSKY = dict(u0=1.0, l0=1.0, tau_st=1.0, p=10 / 7, q=2 / 7, s=4)


# ---------------------------------------------------------------------------
# Self-similar model (verbatim from decaying_selfsimilar_derivation.ipynb)
# ---------------------------------------------------------------------------
def u_t(t, **pars):
    return pars["u0"] * (1 + t / pars["tau_st"]) ** (-pars["p"] / 2)


def L_t(t, **pars):
    return pars["l0"] * (1 + t / pars["tau_st"]) ** pars["q"]


def eps_t(t, **pars):
    u0, tau_st, p = pars["u0"], pars["tau_st"], pars["p"]
    return p * u0**2 / (2 * tau_st) * (1 + t / tau_st) ** (-p - 1)


def tau1(k, t, **pars):
    return 1.0 / (eps_t(t, **pars) ** (1 / 3) * k ** (2 / 3))


def phi(kappa, s=4):
    return kappa**s / (1 + kappa) ** (s + 5 / 3)


def E_kt(k, t, **pars):
    L = L_t(t, **pars)
    return u_t(t, **pars) ** 2 * L * phi(k * L, s=pars.get("s", 4))


def Phi_eq(k, t, **pars):
    return E_kt(k, t, **pars) / (4 * np.pi * k**2)


def R_decorr(k, t1, t2, **pars):
    T = 0.5 * (t1 + t2)
    return (1.0 + np.abs(t1 - t2) / tau1(k, T, **pars)) ** (-2 / 3)


def Phi(k, t1, t2, **pars):
    return np.sqrt(Phi_eq(k, t1, **pars) * Phi_eq(k, t2, **pars)) * R_decorr(k, t1, t2, **pars)


def eps0_of(**pars) -> float:
    """Decay-strength parameter eps_0 = tau_{e,0}/tau_* = (L_0/u_0)/tau_*."""
    return (pars["l0"] / pars["u0"]) / pars["tau_st"]


def pars_from_M(M, eps0=1.0):
    """Mach-coupled parameters: M enters via the sweeping eddy time tau_1 ~ 1/(M k^{2/3}).

    Set u_0 = M (c_s = 1), L_0 = 1, and the decay time in eddy-time units
    tau_* = tau_{e,0}/eps_0 = 1/(M eps_0).  Then eps(0) ~ M^3 and
    tau_1(k) ~ 1/(M k^{2/3}), matching derivation.tex's sweeping
    tau_k = sqrt(2pi)/(M k_0^{1/3} k^{2/3}).  With the emission window scaled in
    the same eddy-time units (T_em = N/(M eps_0)) the whole problem is
    self-similar in M:  H_exact(omega; M) = M^2 H_exact(omega/M; 1), so the
    aeroacoustic GW spectrum obeys  Omega_GW(p; M) = M^5 Otil(p/M).
    """
    return dict(u0=float(M), l0=1.0, tau_st=1.0 / (M * eps0), p=10 / 7, q=2 / 7, s=4)


# ---------------------------------------------------------------------------
# Truncated O(eps^2) kernel:  G_SS(q; xi) = G_BK(q) - (xi/4) G_BK''(q)
# ---------------------------------------------------------------------------
def _ghat_curve(qs):
    """Two-sided BK2016 kernel g_hat(q) = 2 int_0^inf cos(q sigma)(1+sigma)^{-2/3} dsigma.

    Uses scipy's oscillatory cosine quadrature (accurate; matches the corrected
    core.g_decaying to machine precision).  Singular ~|q|^{-1/3} at q=0, so the
    table grid is chosen to avoid the exact origin.
    """
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        if q == 0.0:
            out[i] = np.inf
        else:
            val, _ = integrate.quad(
                lambda s: (1 + s) ** (-2 / 3), 0, np.inf, weight="cos", wvar=abs(q), limit=200
            )
            out[i] = 2.0 * val
    return out


def _make_GSS_table(q_max=40.0, n_q=800):
    qs = np.linspace(-q_max, q_max, n_q)  # even n_q -> grid straddles, never hits, q=0
    g = _ghat_curve(qs)
    dq = qs[1] - qs[0]
    GBK = np.convolve(g, g, mode="same") * dq          # G_BK = g_hat * g_hat
    GBK2 = np.gradient(np.gradient(GBK, dq), dq)        # G_BK''
    return qs, GBK, GBK2


_QS, _GBK, _GBK2 = _make_GSS_table()


def G_SS(q, xi):
    GBK = np.interp(q, _QS, _GBK, left=0.0, right=0.0)
    GBK2 = np.interp(q, _QS, _GBK2, left=0.0, right=0.0)
    return GBK - (xi / 4.0) * GBK2


def _d2lnPhi_dT2(k, T, **pars):
    dT = 0.02 * (pars["tau_st"] + T)
    f0 = np.log(Phi_eq(k, T, **pars))
    fp = np.log(Phi_eq(k, T + dT, **pars))
    fm = np.log(Phi_eq(k, max(T - dT, -0.9 * pars["tau_st"]), **pars))
    return (fp - 2 * f0 + fm) / dT**2


def xi_param(k, T, **pars):
    return tau1(k, T, **pars) ** 2 * _d2lnPhi_dT2(k, T, **pars)


def H_trunc_T(omega, T, k_min=1e-2, k_max=1e3, n_k=200, include_xi=True, **pars):
    """Instantaneous O(eps^2) source at slow time T (notebook's H_ss_T)."""
    ks = np.geomspace(k_min, k_max, n_k)
    E = np.array([E_kt(k, T, **pars) for k in ks])
    t1 = np.array([tau1(k, T, **pars) for k in ks])
    xi = np.array([xi_param(k, T, **pars) for k in ks]) if include_xi else np.zeros_like(ks)
    integrand = (E**2 / ks**2) * t1 * G_SS(omega * t1, xi)
    return 7.0 / (96.0 * np.pi**5) * np.trapezoid(integrand, ks)


def H_trunc_total(omega, T_em, n_T=41, include_xi=True, **pars):
    Ts = np.linspace(0.0, T_em, n_T)
    return np.trapezoid([H_trunc_T(omega, T, include_xi=include_xi, **pars) for T in Ts], Ts)


# ---------------------------------------------------------------------------
# Exact causal-window Wigner  (valid at all eps_0, manifestly >= 0)
# ---------------------------------------------------------------------------
def _wigner_causal(k, omega, T, T_em, tail=100.0, **pars):
    """int dtau cos(omega tau) Phi(k; T+tau/2, T-tau/2)^2 over the causal window.

    Window |tau| <= 2 min(T, T_em-T) keeps both times in [0, T_em]; it is further
    capped at ``tail`` eddy times where the kernel has decayed to <1%.  The grid
    resolves both the eddy time tau_1 and the oscillation 1/omega.
    """
    w = min(T, T_em - T)
    if w <= 0.0:
        return 0.0
    t1 = tau1(k, T, **pars)
    half = min(2.0 * w, tail * t1)
    dtau = min(t1, np.pi / (4.0 * omega)) if omega > 0 else t1
    n = int(2.0 * half / dtau) + 1
    n = max(201, min(n, 40001)) | 1                      # odd for symmetric grid
    taus = np.linspace(-half, half, n)
    ph = Phi(k, T + taus / 2.0, T - taus / 2.0, **pars)  # vectorised, even in tau
    return float(np.trapezoid(ph * ph * np.cos(omega * taus), taus))


def H_exact(omega, T_em, k_min=1e-2, k_max=1e3, n_k=90, n_T=80, **pars):
    """Exact aeroacoustic GW source H(0, omega) integrated over emission [0, T_em]."""
    ks = np.geomspace(k_min, k_max, n_k)
    Ts = np.linspace(0.0, T_em, n_T)
    kint = np.empty(n_k)
    for j, k in enumerate(ks):
        perT = np.array([_wigner_causal(k, omega, T, T_em, **pars) for T in Ts])
        kint[j] = np.trapezoid(perT, Ts)
    return 7.0 / (3.0 * np.pi**2) * np.trapezoid(ks**2 * kint, ks)


# ---------------------------------------------------------------------------
# Hybrid dispatcher
# ---------------------------------------------------------------------------
def H_total(omega, T_em, *, eps_switch=0.3, method="auto", trunc_kw=None, exact_kw=None, **pars):
    """GW source H(0, omega).

    method='auto' uses the cheap O(eps^2) kernel for slow decay (eps_0 <= eps_switch)
    and the exact causal Wigner otherwise.  Force with method in {'trunc','exact'}.
    """
    use_exact = method == "exact" or (method == "auto" and eps0_of(**pars) > eps_switch)
    if use_exact:
        return H_exact(omega, T_em, **(exact_kw or {}), **pars)
    return H_trunc_total(omega, T_em, **(trunc_kw or {}), **pars)


# ---------------------------------------------------------------------------
# Validation + cross-check figure
# ---------------------------------------------------------------------------
def _H_2d(omega, T_em, n_t=220, n_k=70, k_min=1e-2, k_max=1e3, **pars):
    """Gold-standard exact source: 7/(3pi^2) int dk k^2 int_[0,Tem]^2 dt1 dt2
    cos(omega(t1-t2)) Phi(k;t1,t2)^2.  Direct double-time integral, no Wigner
    window -- the unambiguous reference for H_exact."""
    ts = np.linspace(0.0, T_em, n_t)
    ks = np.geomspace(k_min, k_max, n_k)
    T1, T2 = np.meshgrid(ts, ts, indexing="ij")
    cos = np.cos(omega * (T1 - T2))
    out = np.empty(n_k)
    for j, k in enumerate(ks):
        M = Phi(k, T1, T2, **pars)
        out[j] = np.trapezoid(np.trapezoid(M * M * cos, ts, axis=1), ts)
    return 7.0 / (3.0 * np.pi**2) * np.trapezoid(ks**2 * out, ks)


def _validate():
    from gw_turbulence.core import g_decaying

    print("=" * 74)
    print("VALIDATION")
    print("=" * 74)

    # (1) truncated-path kernel vs the corrected core.py BK2016 kernel.
    qs = np.array([0.5, 1.0, 2.0, 4.0])
    mine = _ghat_curve(qs)
    core = 2.0 * np.array([complex(g_decaying(q)).real for q in qs])
    print("\n(1) two-sided g_hat vs 2*Re core.g_decaying (regression for the -1/3 fix):")
    for q, a, b in zip(qs, mine, core):
        print(f"    q={q:4.1f}:  here={a:+.5f}  core={b:+.5f}  match={abs(a - b) < 1e-3}")

    # (2) exact integrator vs the gold-standard direct 2D-time integral.
    print("\n(2) H_exact vs gold-standard 2D-time integral (ratio -> 1):")
    print(f"    {'regime':>14}{'omega':>7}{'H_exact':>13}{'H_2d':>13}{'ratio':>8}")
    for lab, p, Tem in [("fast eps0=2", {**LOITSIANSKY, "tau_st": 0.5}, 1.0),
                        ("slow eps0=1e-4", {**LOITSIANSKY, "tau_st": 1e4}, 30.0)]:
        for om in (0.5, 1.5):
            he = H_exact(om, Tem, **p)
            h2 = _H_2d(om, Tem, **p)
            print(f"    {lab:>14}{om:7.2f}{he:13.3e}{h2:13.3e}{he / h2:8.3f}")

    # (3) the headline: fast decay -- the O(eps^2) truncation is erratic and can go
    #     negative (kernel G_SS < 0 for xi >~ 1 at the peak), while exact stays >= 0.
    pars = {**LOITSIANSKY, "tau_st": 0.5}  # eps_0 = 2
    oms = np.geomspace(0.15, 8.0, 16)
    He = np.array([H_exact(om, 1.0, **pars) for om in oms])
    Ht = np.array([H_trunc_total(om, 1.0, include_xi=True, **pars) for om in oms])
    print(f"\n(3) fast decay eps_0={eps0_of(**pars):.1f}, T_em=1:  truncation breakdown")
    print(f"    exact: all >= 0 ? {bool((He >= 0).all())}   (min {He.min():.2e})")
    print(f"    trunc: any  < 0 ? {bool((Ht < 0).any())}    (min {Ht.min():.2e})")
    print(f"    trunc/exact ratio over the omega grid: {(Ht / He).min():+.2f} .. {(Ht / He).max():+.2f}")
    print("    -> O(eps^2) wildly unreliable for fast decay; use the exact Wigner.")

    # (4) Mach coupling via sweeping tau_1: exact self-similar scaling + IR slope.
    eps0, N = 1.0, 4.0
    print("\n(4) Mach coupling (sweeping tau_1):  H_exact(omega;M) =?= M^2 H_exact(omega/M;1)")
    for M in (2.0, 3.0):
        pM, p1 = pars_from_M(M, eps0), pars_from_M(1.0, eps0)
        hM = H_exact(0.6, N / (M * eps0), **pM)
        h1 = M**2 * H_exact(0.6 / M, N / eps0, **p1)
        print(f"    M={M:.0f}:  H={hM:.4e}  M^2 H(./M;1)={h1:.4e}  ratio={hM / h1:.4f}")
    p1 = pars_from_M(1.0, eps0)
    ps = np.array([0.05, 0.1, 0.2])
    Om = ps**3 * np.array([H_exact(p, N, **p1) for p in ps])
    slope = np.diff(np.log(Om)) / np.diff(np.log(ps))
    print(f"    => Omega_GW(p;M) = M^5 Otil(p/M).  IR slope d ln Omega/d ln p = "
          f"{slope.mean():.2f} (causal p^3); IR amplitude ~ M^2.")
    print("    NOTE: aeroacoustic (p<~1) gives the IR only; the peak needs the full-spatial kernel.")


def _figure(name="selfsimilar_hybrid_crosscheck"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.9), constrained_layout=True)

    # Panel (a): MECHANISM -- the truncated kernel G_SS(q; xi) goes negative for xi >~ 1.
    qg = np.linspace(0.0, 8.0, 400)
    for c, xi in zip((PALETTE[7], PALETTE[2], PALETTE[4], PALETTE[1]), (0.0, 0.5, 1.0, 1.5)):
        ax0.plot(qg, G_SS(qg, xi), color=c, lw=1.6, label=rf"$\xi={xi}$")
    ax0.axhline(0.0, color="0.5", lw=0.8)
    ax0.set_title(r"$O(\epsilon^2)$ kernel $G_{\rm SS}(q;\xi)$ turns negative", fontsize=10)
    ax0.set_xlabel(r"$q=\omega\tau_1$")
    ax0.set_ylabel(r"$G_{\rm SS}(q;\xi)$")
    ax0.set_ylim(-4, 13)
    ax0.legend(fontsize=8, title=r"$\xi=\tau_1^2\,\partial_T^2\ln\Phi$", title_fontsize=8)
    ax0.text(0.55, -3.3, r"$\xi\sim1$ at the peak for $\epsilon_0\gtrsim1$", fontsize=8, color=PALETTE[1])

    # Panel (b): CONSEQUENCE -- fast-decay GW source: exact (correct) vs truncation (erratic).
    omegas = np.geomspace(0.15, 8.0, 13)
    fast = {**LOITSIANSKY, "tau_st": 0.5}   # eps_0 = 2.0 (cosmological-like fast decay)
    He = np.array([H_exact(om, 1.0, **fast) for om in omegas])
    Ht = np.array([H_trunc_total(om, 1.0, include_xi=True, **fast) for om in omegas])
    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=3e-8)
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.plot(omegas, He, "-o", color=PALETTE[0], ms=3.5, label=r"exact Wigner ($\geq0$, validated)")
    ax1.plot(omegas, Ht, "--s", color=PALETTE[1], ms=3.5, label=r"$O(\epsilon^2)$ truncation")
    neg = Ht < 0
    if neg.any():
        ax1.plot(omegas[neg], Ht[neg], "x", color=PALETTE[1], ms=8, mew=1.6, label="trunc.\\ $<0$")
    ax1.set_title(r"fast decay $\epsilon_0=2$: source $H_{ijij}(0,\omega)$", fontsize=10)
    ax1.set_xlabel(r"$\omega\,\tau_{e,0}$")
    ax1.set_ylabel(r"$H_{ijij}(0,\omega)$")
    ax1.legend(fontsize=8, loc="upper right")
    for ax in (ax0, ax1):
        apply_max_ticks(ax)

    path = save_figure(fig, name)
    print(f"\nwrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    _validate()
    _figure()
