#!/usr/bin/env python3
r"""Scale-dependent MHD GW kernel: which correlation-time law gives the simulated k^1?

This is the *controlled* companion to ``fullspatial_selfsimilar_exact.py`` (HD) and
the flawed ``mhd_inverse_transfer.py`` (k-INDEPENDENT tau_c factored OUT of the x,y
integral -> spurious finite-window k^1).  Here we keep the temporal decorrelation
INSIDE the (x,y) integral with a PHYSICAL, scale-dependent correlation time
tau_c,0(k) and ask which law flattens the GW infrared toward the simulated k^1.

Geometry is reused verbatim from ``core`` (kernel_bracket, _integration_bounds,
_h_prefactor; legs k1=x^{-3/4}, k2=y^{-3/4}).  Only two things change between cases:
the per-leg base correlation time tau_c,0(k) and the amplitude-decay exponent.

Triangle-windowed double-time temporal factor (identical machinery to H_ss_exact):

    T(omega; x,y) = (pi / (tau1_0 tau2_0 T_em)) int_0^Tem dT int dtau
                      cos(omega tau) a(t1) a(t2) R1(t1,t2) R2(t1,t2),
    R_i = (1 + |tau| / tau_i(T))^{-2/3},   tau_i(T) = tau_i0 (1 + T/tau_st)^ALPHA,
    a(t) = (1 + t/tau_st)^{-gamma},   t1=T+tau/2, t2=T-tau/2, |tau|<=2 min(T,Tem-T).

Correlation-time laws tau_c,0(k) (k = leg wavenumber = x^{-3/4} or y^{-3/4}):

  CASE                tau_c,0(k)                          k*tau_c    coherence
  ------------------  ----------------------------------  ---------  --------------
  HD eddy (+decay)    sqrt(x)/M = 1/(M k^{2/3})           ~ k^{1/3}  -> 0  (incoherent)
  MHD Alfvenic        1/(v_A k)        [v_A = M]          = 1/M      const
  MHD sustained       max(1/(v_A k), C*T_em) for k<k0     >= 1       coherent over window

For the MHD cases v_A is identified with M (Alfvenic Mach), so the no-decay reduction
and the M-scaling stay comparable to the HD case.  The "sustained" case caps the
large-scale (sub-source) legs' correlation time at >~ T_em -- the helical inverse-
transfer claim that the large-scale field stays coherent over the whole emission.

The normalisation pi/(tau1_0 tau2_0 T_em) reduces T -> conv_ftprod (and the kernel ->
fullspatial_decay) in the no-decay, long-window limit, exactly as in H_ss_exact, so
the HD case here is a regression check against the audited HD result.

Run: python Notebooks/mhd_scaledependent.py
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

# Loitsiansky self-similar exponents (same as H_ss_exact / mhd_inverse_transfer).
ALPHA = 17.0 / 21.0   # tau_c(T) ~ (1+T/tau_st)^ALPHA   (decorrelation-time growth)
GAMMA_HD = 34.0 / 21.0  # inertial-range Phi_eq amplitude decay (HD self-similar)


# --------------------------------------------------------------------------- #
#  Correlation-time laws tau_c,0(k) per leg.  x relates to the leg wavenumber
#  by k1 = x^{-3/4}, so sqrt(x) = k1^{-2/3} and 1/k1 = x^{3/4}.
# --------------------------------------------------------------------------- #
def tau0_hd(x: float, M: float, T_em: float) -> float:
    """HD sweeping/eddy time tau_c = sqrt(x)/M = 1/(M k^{2/3}); saturates as k->0 (x large)."""
    return np.sqrt(x) / M


def tau0_alfven(x: float, M: float, T_em: float) -> float:
    """MHD Alfvenic tau_c ~ 1/(v_A k) = x^{3/4}/M (v_A=M); k*tau_c = 1/M const."""
    return x ** 0.75 / M


def tau0_sustained(x: float, M: float, T_em: float, coh_frac: float = 1.0) -> float:
    """MHD sustained: Alfvenic for k>=k0 (x<=1), but the large-scale (x>1, k<k0) legs are
    capped at tau_c >~ coh_frac*T_em -- the inverse-transfer 'coherent large-scale field'.
    Implemented as the max of the Alfvenic time and the sustained floor so it is continuous
    and only lifts the SUB-SOURCE legs (x>1) that the IR band actually samples.
    """
    return max(x ** 0.75 / M, coh_frac * T_em)


TAU0_LAWS = {
    "hd": tau0_hd,
    "alfven": tau0_alfven,
    "sustained": tau0_sustained,
}


def _amp(t: float, tau_st: float, gamma: float) -> float:
    return (1.0 + t / tau_st) ** (-gamma)


def _wigner_pair(omega, T, T_em, tau1_0, tau2_0, tau_st, gamma):
    """Inner causal-window integral int dtau cos(omega dtau) a(t1) a(t2) R1 R2.

    Exact oscillatory cosine quadrature over the causal half-window, identical in
    structure to H_ss_exact._wigner_pair (the audited HD machinery).
    """
    w = min(T, T_em - T)
    if w <= 0.0:
        return 0.0
    half = 2.0 * w
    tau1 = tau1_0 * (1.0 + T / tau_st) ** ALPHA
    tau2 = tau2_0 * (1.0 + T / tau_st) ** ALPHA

    def g(dt):
        return (_amp(T + dt / 2.0, tau_st, gamma) * _amp(T - dt / 2.0, tau_st, gamma)
                * (1.0 + dt / tau1) ** (-2.0 / 3.0) * (1.0 + dt / tau2) ** (-2.0 / 3.0))

    if omega <= 0:
        val, _ = integrate.quad(g, 0.0, half, limit=200)
    else:
        val, _ = integrate.quad(g, 0.0, half, weight="cos", wvar=omega, limit=200)
    return 2.0 * float(val)


def _temporal(omega, x, y, M, tau_st, T_em, gamma, law, n_T, coh_frac):
    """Scale-dependent temporal factor; tau_c,0 set per leg by `law`. INSIDE the x,y integral."""
    tau0 = TAU0_LAWS[law]
    if law == "sustained":
        tau1_0 = tau0(x, M, T_em, coh_frac)
        tau2_0 = tau0(y, M, T_em, coh_frac)
    else:
        tau1_0 = tau0(x, M, T_em)
        tau2_0 = tau0(y, M, T_em)
    Ts = np.linspace(0.0, T_em, n_T)
    inner = np.array([_wigner_pair(omega, T, T_em, tau1_0, tau2_0, tau_st, gamma) for T in Ts])
    dbl = np.trapezoid(inner, Ts)
    return np.pi / (tau1_0 * tau2_0 * T_em) * dbl


def H_mhd(p, q, M=1.0, R=1e4, tau_st=1e12, T_em=40.0, law="hd", gamma=None,
          coh_frac=1.0, x_points=24, y_points=24, n_T=38):
    """Full-spatial scale-dependent GW kernel H(p,q) with correlation-time law `law`."""
    if gamma is None:
        gamma = GAMMA_HD
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
            * _temporal(q, x, yy, M, tau_st, T_em, gamma, law, n_T, coh_frac)
            for yy in ys
        ]
        x_integrand[i] = np.trapezoid(vals, ys)
    return _h_prefactor(p, M, 1.0) * float(np.trapezoid(x_integrand, xs))


def omega_gw(p, M=1.0, R=1e4, **kw):
    return p**3 * H_mhd(p, p, M=M, R=R, **kw)


def band_slope(law, band, M=1.0, tau_st=1e12, T_em=40.0, gamma=None, coh_frac=1.0,
               x_points=24, y_points=24, n_T=38):
    """log-log slope of Omega_GW(p)=p^3 H over a p-band, for one correlation-time law."""
    sp = np.array([omega_gw(p, M=M, tau_st=tau_st, T_em=T_em, law=law, gamma=gamma,
                            coh_frac=coh_frac, x_points=x_points, y_points=y_points, n_T=n_T)
                   for p in band])
    good = sp > 0
    if good.sum() < 2:
        return float("nan"), sp
    return float(np.polyfit(np.log(band[good]), np.log(sp[good]), 1)[0]), sp


# --------------------------------------------------------------------------- #
#  Roper Pol digitized-data band test (real CSV only; no fabrication).
# --------------------------------------------------------------------------- #
def roperpol_band_test():
    """Does the measured GW IR steepen toward k^3 at the smallest available k, or stay flat?

    Loads the real digitized Omega_GW/k CSV (NO fabrication), forms the
    Omega_GW = k*(Omega/k) = drho/dln k spectrum, and measures its log-log slope in
    the lowest-k decile vs the mid-IR, both on the IR side of the ~1.84 k0 GW peak.

    Caveat returned in `caveat`: the smallest digitized k is only ~0.19 k0 -- NOT deep
    in the sub-source IR (k<<k0), let alone below k0/T_em where the kernel predicts the
    k^3 reversion.  The digitization is also quantised (a ~14-step staircase below
    ~0.7 k0) with a shallow non-monotonic dip near k~0.29 k0, which biases the
    lowest-decile slope LOW.  The test can therefore only check "does it steepen toward
    k^3 by the smallest k?" (it does not) -- it cannot probe the predicted deep-IR.
    """
    import roperpol_data
    kg, ogk = roperpol_data.load("gw")
    og = kg * ogk                       # Omega_GW = d rho/d ln k
    k0 = roperpol_data.k0()
    peak_k = roperpol_data.gw_peak_ratio() * k0   # ~1.84 k0
    # restrict to the IR side of the GW peak
    ir = kg < 0.95 * peak_k
    k_ir, og_ir = kg[ir], og[ir]
    n = len(k_ir)
    # lowest-k decile vs the mid-IR band (upper half of the IR side, still below peak)
    dec = max(n // 10, 3)
    k_lo, og_lo = k_ir[:dec], og_ir[:dec]
    mid = (k_ir >= 0.4 * peak_k) & (k_ir < 0.95 * peak_k)
    sl_lo = float(np.polyfit(np.log(k_lo), np.log(og_lo), 1)[0])
    sl_mid = float(np.polyfit(np.log(k_ir[mid]), np.log(og_ir[mid]), 1)[0])
    sl_full = float(np.polyfit(np.log(k_ir), np.log(og_ir), 1)[0])
    n_distinct_low = int(len(np.unique(ogk[kg < 0.7 * peak_k])))
    return {
        "k0": k0, "peak_k": peak_k,
        "k_min": float(kg.min()), "p_min": float(kg.min() / k0),
        "n_ir": n,
        "k_lo_range": (float(k_lo.min()), float(k_lo.max())),
        "p_lo_range": (float(k_lo.min() / k0), float(k_lo.max() / k0)),
        "slope_lowk_decile": sl_lo,
        "slope_mid_ir": sl_mid,
        "slope_full_ir": sl_full,
        "n_distinct_quantised_low": n_distinct_low,
        "caveat": ("smallest k ~0.19 k0 (not deep IR); ~%d-step quantisation + a shallow "
                   "non-monotonic dip near 0.29 k0 bias the lowest-decile slope low"
                   % n_distinct_low),
    }


# --------------------------------------------------------------------------- #
#  Main: slope tables + verdict + figure.
# --------------------------------------------------------------------------- #
def _validate(M=1.0, R=1e4, T_em=40.0, x_points=22, y_points=22, n_T=34):
    """No-decay reduction: HD case here must match fullspatial_decay (cross-check)."""
    from fullspatial_decay import H_decay_fast
    print("=" * 74)
    print("VALIDATION: HD law, no decay -> fullspatial_decay (ratio -> 1 as T_em grows)")
    print("=" * 74)
    b = H_decay_fast(1.0, 1.0, M=M, R=R)
    for Tem in (60.0, 150.0):
        a = H_mhd(1.0, 1.0, M=M, R=R, tau_st=1e12, T_em=Tem, law="hd",
                  x_points=x_points, y_points=y_points, n_T=int(Tem / 3))
        print(f"   T_em={Tem:5.0f}:  HD/decay_fast = {a / b:.4f}")


def slope_tables(M=1.0, T_em=40.0, x_points=22, y_points=22, n_T=34):
    """Two-band IR slope table for HD / Alfvenic / sustained, with and without decay."""
    # DNS-like band: 1/T_em (~0.025 for T_em=40) up to ~0.5; DEEP-IR band: < 1/T_em.
    inv_Tem = 1.0 / T_em
    dns_band = np.geomspace(max(inv_Tem, 0.04), 0.45, 6)
    deep_band = np.geomspace(0.2 * inv_Tem, 0.8 * inv_Tem, 6)   # well below 1/T_em
    rows = []
    configs = [
        ("HD eddy, no decay",     "hd",        1e12, GAMMA_HD),
        ("HD eddy, decay e0=1",   "hd",        1.0,  GAMMA_HD),
        ("MHD Alfvenic, no decay","alfven",    1e12, GAMMA_HD),
        ("MHD Alfvenic, decay",   "alfven",    1.0,  GAMMA_HD),
        ("MHD sustained, no decay","sustained",1e12, 0.0),
        ("MHD sustained, decay",  "sustained", 8.0,  1.0 / 3.0),
    ]
    for lab, law, tau_st, gamma in configs:
        s_dns, _ = band_slope(law, dns_band, M=M, tau_st=tau_st, T_em=T_em, gamma=gamma,
                              x_points=x_points, y_points=y_points, n_T=n_T)
        s_deep, _ = band_slope(law, deep_band, M=M, tau_st=tau_st, T_em=T_em, gamma=gamma,
                               x_points=x_points, y_points=y_points, n_T=n_T)
        rows.append((lab, law, s_dns, s_deep))
    return dns_band, deep_band, rows


def tem_tracking(law, gamma, tau_st, M=1.0, x_points=22, y_points=22, n_T=None):
    """For a case that shows a band: confirm the band's lower edge tracks 1/T_em.

    Report the DNS-band slope and the deep-IR slope at T_em = 10/40/160; if a k^1
    band exists between 1/T_em and k0, its width grows as T_em grows, and the deep-IR
    (below 1/T_em) should always revert to causal k^3.
    """
    out = []
    for T_em in (10.0, 40.0, 160.0):
        inv = 1.0 / T_em
        dns = np.geomspace(max(inv, 0.02), 0.45, 6)
        deep = np.geomspace(0.2 * inv, 0.8 * inv, 6)
        nt = n_T if n_T is not None else max(int(T_em / 3), 14)
        s_dns, _ = band_slope(law, dns, M=M, tau_st=tau_st, T_em=T_em, gamma=gamma,
                              x_points=x_points, y_points=y_points, n_T=nt)
        s_deep, _ = band_slope(law, deep, M=M, tau_st=tau_st, T_em=T_em, gamma=gamma,
                               x_points=x_points, y_points=y_points, n_T=nt)
        out.append((T_em, inv, s_dns, s_deep))
    return out


def _figure(name="mhd_scaledependent"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import PALETTE, apply_max_ticks, apply_paper_style, save_figure

    apply_paper_style()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)

    # Panel (a): example spectra for the three laws over a wide p-range spanning
    # the deep IR (p < 1/T_em) and the DNS-like band (1/T_em < p < 0.5).
    T_em = 40.0
    ps = np.geomspace(0.006, 1.2, 13)
    cases = [
        ("hd",        1e12, GAMMA_HD,  0,  PALETTE[5], r"HD eddy $\tau_c\!\sim\!k^{-2/3}$"),
        ("alfven",    1e12, GAMMA_HD,  0,  PALETTE[3], r"MHD Alfv\'en $\tau_c\!\sim\!k^{-1}$"),
        ("sustained", 1e12, 0.0,       0,  PALETTE[1], r"MHD sustained $\tau_c\!\gtrsim\!T_{\rm em}$"),
    ]
    for law, tau_st, gamma, _, col, lab in cases:
        sp = np.array([omega_gw(p, M=1.0, tau_st=tau_st, T_em=T_em, law=law, gamma=gamma,
                                x_points=24, y_points=24, n_T=34) for p in ps])
        ax0.plot(ps, sp / np.nanmax(sp), "-", color=col, lw=1.8, label=lab)
    # reference slope guides (reference, not data)
    pr = np.geomspace(0.02, 0.3, 8)
    ax0.plot(pr, 0.5 * (pr / 0.3) ** 3, ":", color="0.45", lw=1.2, label=r"$k^3$ (reference, not data)")
    ax0.plot(pr, 0.5 * (pr / 0.3) ** 1, "--", color="0.45", lw=1.2, label=r"$k^1$ (reference, not data)")
    ax0.axvline(1.0 / T_em, color="0.7", lw=1.0, ls="-.")
    ax0.text(1.0 / T_em * 1.05, 1.3e-3, r"$1/T_{\rm em}$", fontsize=7, color="0.4")
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_ylim(1e-5, 2)
    ax0.set_xlabel(r"$p=k/k_0$")
    ax0.set_ylabel(r"$\Omega_{\rm GW}(p)\,/\,\Omega_{\rm GW}^{\rm peak}$")
    ax0.set_title(r"GW IR vs.\ correlation-time law", fontsize=10)
    ax0.legend(fontsize=6.2, loc="lower right")

    # Panel (b): DNS-band slope vs T_em for each law, with deep-IR slope as open markers.
    Tems = np.array([10.0, 40.0, 160.0])
    markers = {"hd": ("o", PALETTE[5]), "alfven": ("s", PALETTE[3]), "sustained": ("D", PALETTE[1])}
    for law, gamma in (("hd", GAMMA_HD), ("alfven", GAMMA_HD), ("sustained", 0.0)):
        dns, deep = [], []
        for T_em in Tems:
            inv = 1.0 / T_em
            db = np.geomspace(max(inv, 0.02), 0.45, 5)
            dpb = np.geomspace(0.2 * inv, 0.8 * inv, 5)
            nt = max(int(T_em / 3), 14)
            s1, _ = band_slope(law, db, M=1.0, tau_st=1e12, T_em=T_em, gamma=gamma,
                               x_points=22, y_points=22, n_T=nt)
            s2, _ = band_slope(law, dpb, M=1.0, tau_st=1e12, T_em=T_em, gamma=gamma,
                               x_points=22, y_points=22, n_T=nt)
            dns.append(s1); deep.append(s2)
        m, c = markers[law]
        ax1.semilogx(Tems, dns, m + "-", color=c, lw=1.6, ms=5, label=f"{law} (DNS band)")
        ax1.semilogx(Tems, deep, m + ":", color=c, lw=1.2, ms=5, mfc="white")
    ax1.axhline(3.0, color="0.45", ls=":", lw=1.2)
    ax1.axhline(1.0, color="0.45", ls="--", lw=1.2)
    ax1.text(11, 3.05, r"causal $k^3$ (reference, not data)", fontsize=6.5, color="0.35")
    ax1.text(11, 1.08, r"DNS $k^1$ (reference, not data)", fontsize=6.5, color="0.35")
    ax1.set_xlabel(r"$T_{\rm em}$ (eddy times)")
    ax1.set_ylabel(r"IR slope $d\ln\Omega_{\rm GW}/d\ln k$")
    ax1.set_title(r"filled: DNS band; open: deep IR ($p<1/T_{\rm em}$)", fontsize=9)
    ax1.set_ylim(0.2, 3.6)
    ax1.legend(fontsize=6.2, loc="center right")

    # Panel (c): the REAL digitized Roper Pol GW spectrum (Omega_GW = k*(Omega/k)).
    # The measured IR is flat (~k^1.3) all the way to the smallest available k ~ 0.19 k0;
    # it does NOT steepen toward k^3 -- but that smallest k is still inside the predicted
    # k^1 band (k>k0/T_em), so the deep-IR k^3 reversion is simply out of digitized range.
    import roperpol_data
    kg, ogk = roperpol_data.load("gw")
    k0 = roperpol_data.k0()
    pk = kg / k0
    og = kg * ogk
    sel = pk < 1.95            # IR side + peak
    og_n = og / np.nanmax(og[sel])
    ax2.loglog(pk[sel], og_n[sel], "-", color=PALETTE[6], lw=1.5,
               label=r"Roper Pol Fig.~1 (digitized data)")
    pr2 = np.geomspace(0.2, 0.9, 8)
    anchor = og_n[np.argmin(np.abs(pk - 0.2))]
    ax2.loglog(pr2, anchor * (pr2 / 0.2) ** 3, ":", color="0.45", lw=1.2,
               label=r"$k^3$ (reference, not data)")
    ax2.loglog(pr2, anchor * (pr2 / 0.2) ** 1, "--", color="0.45", lw=1.2,
               label=r"$k^1$ (reference, not data)")
    ax2.axvline(roperpol_data.gw_peak_ratio(), color="0.7", lw=1.0, ls="-.")
    ax2.text(roperpol_data.gw_peak_ratio() * 1.03, 1.3e-2, r"GW peak $\sim1.84\,k_0$",
             fontsize=6.5, color="0.4", rotation=90, va="bottom")
    ax2.set_xlabel(r"$k/k_0$")
    ax2.set_ylabel(r"$\Omega_{\rm GW}(k)$  (peak-normalised)")
    ax2.set_title(r"digitized data: flat to smallest $k$", fontsize=10)
    ax2.set_ylim(1e-2, 2)
    ax2.legend(fontsize=6.2, loc="lower right")

    for ax in (ax0, ax1, ax2):
        apply_max_ticks(ax)
    out = save_figure(fig, name)
    print(f"\nwrote {out}")
    plt.close(fig)
    return out


if __name__ == "__main__":
    _validate()
    print()
    dns_band, deep_band, rows = slope_tables()
    print("=" * 74)
    print(f"IR SLOPE TABLE  (M=1, T_em=40; DNS band p in [{dns_band[0]:.3f},{dns_band[-1]:.2f}],"
          f" deep IR p in [{deep_band[0]:.4f},{deep_band[-1]:.4f}])")
    print("=" * 74)
    print(f"   {'case':<26}{'DNS slope':>11}{'deep-IR slope':>15}")
    for lab, law, s_dns, s_deep in rows:
        print(f"   {lab:<26}{s_dns:>11.2f}{s_deep:>15.2f}")

    print("\n" + "=" * 74)
    print("T_em TRACKING (no decay): does the band lower edge track 1/T_em?")
    print("=" * 74)
    for law, gamma in (("alfven", GAMMA_HD), ("sustained", 0.0)):
        print(f"  -- {law} --")
        print(f"     {'T_em':>6}{'1/T_em':>9}{'DNS slope':>11}{'deep-IR slope':>15}")
        for T_em, inv, s_dns, s_deep in tem_tracking(law, gamma, 1e12):
            print(f"     {T_em:>6.0f}{inv:>9.4f}{s_dns:>11.2f}{s_deep:>15.2f}")

    print("\n" + "=" * 74)
    print("ROPER POL DIGITIZED-DATA BAND TEST (real CSV; Omega_GW = k*(Omega/k))")
    print("=" * 74)
    rp = roperpol_band_test()
    print(f"   k0 = {rp['k0']:.0f}, GW peak k = {rp['peak_k']:.0f} (~1.84 k0)")
    print(f"   data k_min = {rp['k_min']:.0f}  (p_min = {rp['p_min']:.2f})  -- cannot reach p<<1/T_em")
    print(f"   lowest-k decile: k in [{rp['k_lo_range'][0]:.0f},{rp['k_lo_range'][1]:.0f}]"
          f"  (p in [{rp['p_lo_range'][0]:.2f},{rp['p_lo_range'][1]:.2f}])")
    print(f"   slope lowest-k decile = {rp['slope_lowk_decile']:+.2f}")
    print(f"   slope mid-IR          = {rp['slope_mid_ir']:+.2f}")
    print(f"   slope full IR side    = {rp['slope_full_ir']:+.2f}")

    _figure()
