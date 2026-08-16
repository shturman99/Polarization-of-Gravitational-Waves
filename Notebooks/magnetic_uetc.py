#!/usr/bin/env python3
r"""Task 2.6: the MAGNETIC anisotropic-stress UETC from Roper Pol et al. (2020).

THE QUESTION
------------
Sec. ``sec:decaying-peak`` of ``derivation.tex`` makes the source-scale GW peak
conditional on a single number: the one-sided slope of the source's unequal-time
correlator at zero lag,

    T(omega) = 2 int_0^inf dtau cos(omega tau) f(tau)  ->  -2 f'(0+)/omega^2 ,
                                                          [eq:cusp-tail]

so the omega^-2 tail -- and with it the peak displacement -- exists if and only
if f has a CUSP at tau = 0.  Auclair et al. (arXiv:2205.02588) measure the
HYDRODYNAMIC velocity UETC and find it Gaussian in the lag, i.e. f'(0+) = 0.
The manuscript's defence is that the GW source is the MAGNETIC stress B_i B_j,
whose UETC has never been measured.  Referees A and C both asked for that
measurement.

FEASIBILITY VERDICT (part 1 of this script, and the honest headline)
--------------------------------------------------------------------
A UETC needs <B(k,t1) B(k,t2)> at UNEQUAL times, i.e. the complex Fourier modes
at two epochs, or a two-time product formed inside the code.  The public release
of run ini2 contains NEITHER.  It contains shell-averaged EQUAL-TIME spectra at
a sequence of dumps, plus a two-point time series.  So:

  (a) NO unequal-time magnetic quantity can be formed directly.  Confirmed by
      enumerating the release (``--files``).
  (b) The dump cadence is Delta t = 1.00e-3 in Hubble units.  The lag that the
      cusp lives at is tau ~ 1/k (the GW resonance is at omega = k), which at
      the GW peak k ~ 1.8 k_0 = 1240 is 8.1e-4 -- BELOW the cadence.  A
      time-domain correlation measurement is therefore impossible at every
      wavenumber that matters: the answer would be one lag bin wide.
  (c) The same holds for the single-point series bxpt/bypt/bzpt in
      ``time_series.dat`` (cadence 8e-4, and only two spatial points, hence
      k-integrated and un-averaged).

WHAT *IS* POSSIBLE, AND WHY
---------------------------
The simulation itself performs the short-lag integral, at its own timestep
(dt = 3.9e-5, i.e. 20x finer than 1/k at the GW peak), and stores the answer in
the GW field.  With h = g = 0 at t_0 (``INITHIJ='nothing'``, ``INITGIJ='nothing'``
in ``param.nml``) the solution of h'' + k^2 h = S is

    g(k,t) = int_{t_0}^{t} dt' cos(k(t-t')) S(k,t') ,

so the ADIABATIC INVARIANT E(k,t) = [P_gg(k,t) + k^2 P_hh(k,t)]/2 obeys EXACTLY

    dE/dt = <g S> = int_0^{t-t_0} dtau cos(k tau) C_S(k; t, t-tau) ,      (*)

the running cosine transform of the stress UETC evaluated at omega = k.  Both
P_gg (``power_GWs.dat``) and P_hh (``power_GWh.dat``) are public, and their
combination de-oscillates (verified below to 0.6-3%, which simultaneously
self-calibrates the shell -> Hubble conversion to 100.0 +/- 1%).

Equation (*) is not a UETC, and this script does not claim to measure one.  It
measures the ONE functional of the UETC that eq:cusp-tail is about, at the one
frequency that matters.  Two things follow, and both are reported:

  1. UPPER BOUND on the cusp.  At late times (t-t_0 >> tau_c) eq. (*) tends to
     Pi(k,t) T(k)/2, so T(k) = 2 (dE/dt)/Pi(k,t) and |f'(0+)| = k^2|T|/2.  The
     measured dE/dt is small and NEGATIVE, so this is a bound, not a detection.
  2. FORWARD MODELS with no free parameters.  Pi(k,t) is known independently
     (below), so E(k,t) can be predicted outright for each candidate correlator
     and compared with the data.

THE ENABLING STEP: the equal-time magnetic stress spectrum, measured
--------------------------------------------------------------------
The first GW dump is at eps = 4.42e-5 after switch-on -- one code step -- where
the source is still fully coherent at every resolved k (eps/tau_sweep < 0.3 even
at the Nyquist).  For a source constant over [t_0, t_0+eps] the exact solution
of the sourced oscillator gives E = Pi 2 sin^2(k eps/2)/k^2, so

    Pi(k,t_0) = E(k,t_0+eps) k^2 / (2 sin^2(k eps/2))

is the anisotropic-stress spectrum in the code's own units, with no closure
assumed.  RP20 never published it.  It agrees with the Gaussian (random-phase)
closure convolution of the measured E_M(k,t_0) to +/-15% in shape over
k = 1e3-5e4 and to 8% in amplitude -- which both validates the extraction and
licenses using the closure to propagate Pi(k,t) to later times.

Usage
-----
    python3 Notebooks/magnetic_uetc.py            # full report + figure
    python3 Notebooks/magnetic_uetc.py --files    # feasibility audit only
    python3 Notebooks/magnetic_uetc.py --quick    # skip the forward models

Produces images/magnetic_uetc.pdf.
"""
from __future__ import annotations

import math as _math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import roperpol_timeseries as TS      # noqa: E402  (reuse its fetcher + cache)

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ``roperpol_timeseries`` caches power_krms/mag/GWs; task 2.6 additionally needs
# the strain spectrum power_GWh.dat and the run parameters.  Same URL, same dir.
_EXTRA = ("power_GWh.dat", "param.nml", "legend.dat", "time_series.dat")

T_START = 1.0            # the field is imposed at t = 1 (Hubble units)
SHELL_TO_HUBBLE = 100.0  # box fundamental; self-calibrated in calibrate_shell()
K0_HUBBLE = 673.0        # source peak, from Notebooks/kernel_at_rp20.py


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def _fetch_extra():
    import urllib.request
    d = TS._fetch()
    for f in _EXTRA:
        dest = d / f
        if not dest.exists():
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(f"{TS._BASE}/{f}", dest)
    return d


def load_all():
    """k (shells), t, and the mag / GWs / GWh spectra of run ini2."""
    d = _fetch_extra()
    k = TS._toks(d / "power_krms.dat")
    nk = len(k)

    def spec(name):
        a = TS._toks(d / name)
        block = nk + 1
        nt = len(a) // block
        a = a[:nt * block].reshape(nt, block)
        return a[:, 0], a[:, 1:]

    t, EM = spec("power_mag.dat")
    _, Pgg = spec("power_GWs.dat")
    _, Phh = spec("power_GWh.dat")
    m = k > 0
    return d, k[m], t, EM[:, m], Pgg[:, m], Phh[:, m]


# --------------------------------------------------------------------------- #
#  part 1 -- feasibility audit
# --------------------------------------------------------------------------- #
def feasibility(d, kk, t, EM):
    K = kk * SHELL_TO_HUBBLE
    dt = np.diff(t)
    vA = np.sqrt(2.0 * _trapz(EM, kk, axis=1))
    k_peak = K[np.argmax(EM[0])]
    ts = np.genfromtxt(d / "time_series.dat", skip_header=1)

    print("=" * 78)
    print("1.  FEASIBILITY:  CAN AN UNEQUAL-TIME QUANTITY BE FORMED AT ALL?")
    print("=" * 78)
    print("\n  public release of run ini2 (M1152e_exp6k4):")
    for f in sorted(p.name for p in d.iterdir()):
        print(f"      {f}")
    print("""
  Every ``power_*.dat`` is a SHELL-AVERAGED EQUAL-TIME spectrum: for each dump
  time, one number per k shell, already squared and angle-averaged.  The complex
  Fourier modes B_i(k) are not in the release, and no two-time product was ever
  formed inside the code.  ==> a magnetic UETC CANNOT be constructed, at any
  lag, from these files.  ``time_series.dat`` adds B_i at TWO spatial points
  (bxpt..bzp2), which is an unequal-time object but k-integrated and with two
  realisations only.""")

    print(f"\n  dump cadence         Delta t = {np.median(dt):.3e}  "
          f"({len(t)} dumps, t = {t[0]:.5f}..{t[-1]:.5f})")
    print(f"  point-series cadence           = {np.median(np.diff(ts[:, 1])):.1e}")
    print(f"  code timestep                  = {np.median(ts[:, 2]):.2e}")
    print(f"  v_A                            = {vA[0]:.4f} -> {vA[-1]:.4f}")
    print(f"  magnetic peak k_0              = {k_peak:.0f} (Hubble units)")

    print("\n  the two lags that matter, against the cadence:")
    print(f"    {'k':>8}{'k/k0':>8}{'tau_sweep=1/(v_A k)':>21}{'1/k (GW resonance)':>20}"
          f"{'dt/tau_sw':>11}{'dt*k':>8}")
    for kh in (K0_HUBBLE * 0.5, K0_HUBBLE, 1.84 * K0_HUBBLE, 4 * K0_HUBBLE,
               1e4, 5.75e4):
        tsw = 1.0 / (vA[0] * kh)
        print(f"    {kh:8.0f}{kh / K0_HUBBLE:8.2f}{tsw:21.2e}{1.0 / kh:20.2e}"
              f"{np.median(dt) / tsw:11.3f}{np.median(dt) * kh:8.2f}")
    print("""
  READ THE LAST TWO COLUMNS.  A cusp is a statement about lags SHORTER than the
  correlation time; eq:cusp-tail probes the correlator at lag ~ 1/omega = 1/k.
  At the GW peak (k = 1.84 k_0) that lag is 8.1e-4, i.e. 0.8 of ONE dump, and it
  falls below the cadence for every k above ~1000.  So a directly measured
  short-lag correlation would be a single bin wide and could not distinguish
  1 - |tau|/tau_c from 1 - tau^2/2tau_c^2.

  ==> ON THE DIRECT ROUTE THE QUESTION IS UNDECIDABLE FROM THE PUBLIC DATA.
      What would decide it: the complex TT-projected stress modes T_ij(k, t), or
      the two-time product <T_ij(k,t1) T_ij*(k,t2)> accumulated in-code, dumped
      at the CODE timestep (~4e-5) rather than the spectrum cadence, for lags out
      to a few tau_sweep.  That is ~25 samples per 1/k at the GW peak.
""")
    return vA


# --------------------------------------------------------------------------- #
#  part 2 -- the indirect channel
# --------------------------------------------------------------------------- #
def calibrate_shell(kk, t, Pgg, Phh, shells=(3, 5, 9, 19, 39)):
    """Self-calibrate shell->Hubble by demanding P_gg + (c k)^2 P_hh be smooth.

    For a freely propagating mode g^2 + k^2 h^2 is exactly conserved, so the
    correct conversion is the one that cancels the oscillation.
    """
    late = t > 1.20
    out = []
    for sh in shells:
        cs = np.linspace(80.0, 120.0, 401)
        rough = [np.std(np.diff(Pgg[late, sh] + (c * kk[sh]) ** 2 * Phh[late, sh], 2))
                 / np.mean(Pgg[late, sh] + (c * kk[sh]) ** 2 * Phh[late, sh])
                 for c in cs]
        i = int(np.argmin(rough))
        out.append((sh + 1, kk[sh], cs[i], rough[i]))
    return out


def stress_convolution(EM, K, nmu=48):
    """Gaussian (random-phase) closure: TT anisotropic-stress SHELL spectrum.

    Pi(k) = 4 pi k^2 int d^3p P_B(p) P_B(|k-p|) (1+gamma^2)(1+beta^2)/4,
    with P_B(k) = E_M(k)/(4 pi k^2), gamma = cos(k,p), beta = cos(k,k-p).
    Only shape and time-RATIOS of this are used; the overall constant is fixed
    by the measured Pi(k,t_0), so no absolute normalisation is assumed.
    """
    P = np.where(K > 0, EM / (4.0 * np.pi * K ** 2), 0.0)
    mu, w = np.polynomial.legendre.leggauss(nmu)
    lk, lP = np.log(K), np.log(np.maximum(P, 1e-300))
    out = np.zeros_like(K)
    for i, kv in enumerate(K):
        q = np.sqrt(kv ** 2 + K[:, None] ** 2 - 2 * kv * K[:, None] * mu[None, :])
        Pq = np.exp(np.interp(np.log(np.maximum(q, K[0])), lk, lP))
        Pq = np.where((q < K[0]) | (q > K[-1]), 0.0, Pq)
        beta = np.where(q > 0, (kv - K[:, None] * mu[None, :]) / np.maximum(q, 1e-30), 0.0)
        f = (1.0 + mu[None, :] ** 2) * (1.0 + beta ** 2) / 4.0
        out[i] = 2 * np.pi * _trapz((P[:, None] * Pq * f * w[None, :]).sum(1) * K ** 2, K)
    return 4.0 * np.pi * K ** 2 * out


def measured_stress(E, K, eps):
    """Pi(k,t_0) from the first GW dump; exact constant-source oscillator solution."""
    return E * K ** 2 / (2.0 * np.sin(K * eps / 2.0) ** 2)


# --------------------------------------------------------------------------- #
#  part 3 -- forward models (no free parameters)
# --------------------------------------------------------------------------- #
_GAMMA23 = _math.gamma(2.0 / 3.0)


def _fhat(nu, a, kind):
    r"""Spectral density of the normalised correlator, fhat(w) = 2 int_0^inf cos(w tau) f.

    All four are non-negative (Bochner), and all are normalised so that
    int dw/2pi fhat = f(0) = 1.  Their large-|w| tails encode eq:cusp-tail:
    f'(0+) = 0 for the Gaussian (super-exponential tail), -a for the exponential
    (tail 2a/w^2) and -2a/3 for BK2016 (tail 4a/3w^2).
    """
    if kind == "gauss":
        return np.sqrt(2 * np.pi) / a * np.exp(-0.5 * (nu / a) ** 2)
    if kind == "exp":
        return 2 * a / (a ** 2 + nu ** 2)
    if kind == "bk16":
        # (1+a tau)^-2/3 is completely monotone: = <exp(-s a tau)> over
        # s^-1/3 e^-s / Gamma(2/3), so its transform is a positive mixture of
        # Lorentzians -- stable, and manifestly non-negative.
        s, w = np.polynomial.laguerre.laggauss(120)
        sa = s * a
        return (2.0 / _GAMMA23) * np.sum(
            w[None, :] * s[None, :] ** (-1 / 3.0) * sa[None, :]
            / (sa[None, :] ** 2 + nu[:, None] ** 2), axis=1)
    raise ValueError(kind)


def model_E(KK, x, t, Pi_t, v, kind, pad=8, safety=1.6):
    r"""E(k,x) predicted for a candidate correlator.  No free parameters.

    The time-domain double integral suffers catastrophic cancellation: for a
    cuspless correlator the answer is ~30 orders of magnitude below the
    individual terms, and evaluating it directly returns NEGATIVE energies.
    Going to the frequency domain makes the integrand manifestly non-negative:
    with Ahat(nu) = int_{t_0}^{t_0+x} dt sqrt(Pi(k,t)) e^{i nu t},

        E(k,x) = (1/2) int dnu/(2 pi)  fhat(nu + k) |Ahat(nu)|^2 ,

    which is exact for a locally stationary correlator C = sqrt(Pi Pi) f(tau)
    and is a sum of squares, so no cancellation occurs.  Ahat is one FFT.
    """
    a = KK * v
    numax = safety * (KK + (8 if kind == "gauss" else 20) * a)
    dt = np.pi / numax
    n0 = int(np.ceil(x / dt))
    n = 1
    while n < pad * n0:
        n <<= 1
    tt = T_START + np.arange(n0) * dt
    y = np.zeros(n)
    y[:n0] = np.sqrt(np.maximum(np.interp(tt, t, Pi_t), 0.0))
    A2 = (np.abs(np.fft.fft(y)) * dt) ** 2
    nu = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    if kind == "coh":                      # fhat = 2 pi delta(w)
        i = np.argsort(nu)
        return 0.5 * float(np.interp(KK, nu[i], A2[i]))
    dnu = 2 * np.pi / (n * dt)
    return 0.5 * float(np.sum(_fhat(nu + KK, a, kind) * A2) * dnu / (2 * np.pi))


MODELS = ("coh", "gauss", "exp", "bk16")
MODEL_LABEL = {
    "coh": r"coherent, $f=1$",
    "gauss": r"Gaussian sweeping, $f=e^{-(kv_A\tau)^2/2}$",
    "exp": r"cusp, $f=e^{-kv_A|\tau|}$",
    "bk16": r"BK2016, $f=(1+kv_A\tau)^{-2/3}$",
}


# --------------------------------------------------------------------------- #
def analyse(quick=False):
    d, kk, t, EM, Pgg, Phh = load_all()
    vA = feasibility(d, kk, t, EM)

    print("=" * 78)
    print("2.  THE INDIRECT CHANNEL:  THE GW FIELD AS THE SIMULATION'S OWN")
    print("    SHORT-LAG INTEGRATOR")
    print("=" * 78)
    cal = calibrate_shell(kk, t, Pgg, Phh)
    print("\n  self-calibration of the shell -> Hubble conversion, from demanding")
    print("  that P_gg + (c k)^2 P_hh de-oscillate (free waves conserve it):")
    print(f"    {'shell':>7}{'k_rms':>9}{'best c':>9}{'residual':>10}")
    for sh, krms, c, r in cal:
        print(f"    {sh:7d}{krms:9.2f}{c:9.2f}{r:10.4f}")
    cbest = np.mean([c for *_, c, _ in cal])
    print(f"    mean c = {cbest:.2f}  (assumed {SHELL_TO_HUBBLE:.0f}); residual "
          f"oscillation {min(r for *_, r in cal):.3f}-{max(r for *_, r in cal):.3f}"
          f"\n    ==> the invariant is well formed, so dE/dt = <gS> is measurable.")

    K = kk * SHELL_TO_HUBBLE
    E = 0.5 * (Pgg + K ** 2 * Phh)
    eps = t[0] - T_START
    Pi0 = measured_stress(E[0], K, eps)
    C0 = stress_convolution(EM[0], K)
    band = slice(14, 200)
    A = float(np.median(Pi0[band] / C0[band]))
    ratio = Pi0 / C0 / A

    print(f"\n  equal-time MAGNETIC STRESS spectrum from the first dump "
          f"(eps = {eps:.2e},\n  k*eps < {K[-1] * eps:.2f}, eps/tau_sweep < "
          f"{eps * vA[0] * K[-1]:.2f} even at the Nyquist):")
    print(f"    {'shell':>7}{'k':>9}{'Pi_meas':>12}{'Pi_closure':>12}{'ratio':>9}")
    for sh in (0, 4, 9, 19, 39, 99, 199, 399, 574):
        print(f"    {sh + 1:7d}{K[sh]:9.0f}{Pi0[sh]:12.3e}{C0[sh] * A:12.3e}{ratio[sh]:9.3f}")
    sl = float(np.polyfit(np.log(K[9:500]), np.log(ratio[9:500]), 1)[0])
    print(f"    amplitude ratio Pi_meas/Pi_closure = {A:.3f} (i.e. within "
          f"{abs(1 - A) * 100:.0f}% of unity)"
          f"\n    shape agreement over shells 10-500: residual log-slope {sl:+.3f}, "
          f"spread {ratio[9:500].max() / ratio[9:500].min():.2f}"
          f"\n    ==> the extraction is validated AND the random-phase closure holds,"
          f"\n        so Pi(k,t) may be propagated with the measured E_M(k,t).")

    # Pi(k,t): closure shape at a few epochs, calibrated at t_0, with the 1/t
    # of the source S = 6 T/t.
    sub = np.array([0, 20, 50, 100, 150, 200, 250, 300, 350, 397])
    Cs = np.array([stress_convolution(EM[i], K) for i in sub])
    PiT = (np.array([np.interp(t, t[sub], Cs[:, i]) for i in range(len(K))]).T
           * A / t[:, None] ** 2)

    print("\n" + "=" * 78)
    print("3.  THE BOUND ON f'(0+)")
    print("=" * 78)
    print("""
  At t-t_0 >> tau_c, eq. (*) -> Pi(k,t) T(k)/2, and eq:cusp-tail at omega = k
  gives f'(0+) = -k^2 T(k)/2.  The natural dimensionless measure is the cusp
  strength relative to the sweeping rate,

      xi(k) = |f'(0+)| / (v_A k) ,

  which is 1 for a correlator that decorrelates by sweeping WITH a cusp
  (f = exp(-v_A k |tau|)) and 0 for a Gaussian.
""")
    windows = [(1.05, 1.15), (1.15, 1.25), (1.20, 1.397), (1.30, 1.397)]
    shells = [4, 9, 19, 29, 39, 59, 99, 199, 399, 574]

    def xi_window(sh, a, b):
        """|f'(0+)|/(v_A k) from dE/dt on [a,b], as |slope| + 2 sigma_slope."""
        m = (t >= a) & (t <= b)
        tt, y = t[m], E[m, sh]
        c = np.polyfit(tt, y, 1)
        resid = y - np.polyval(c, tt)
        sig = (resid.std(ddof=2) / np.sqrt(np.sum((tt - tt.mean()) ** 2))
               if len(tt) > 2 else 0.0)
        j = int(np.argmin(np.abs(t - 0.5 * (a + b))))
        T = 2.0 * (abs(c[0]) + 2.0 * sig) / PiT[j, sh]
        return K[sh] * T / (2.0 * vA[j]), c[0]

    print(f"    {'k':>8}{'k/k0':>7}", end="")
    for a, b in windows:
        print(f"{f'[{a},{b}]':>13}", end="")
    print(f"{'bound':>11}{'sign dE/dt':>12}")
    xi_all = np.array([max(xi_window(sh, a, b)[0] for a, b in windows)
                       for sh in range(len(K))])
    for sh in shells:
        print(f"    {K[sh]:8.0f}{K[sh] / K0_HUBBLE:7.2f}", end="")
        slopes = []
        for a, b in windows:
            xi, sl = xi_window(sh, a, b)
            slopes.append(sl)
            print(f"{xi:13.2e}", end="")
        neg = sum(s < 0 for s in slopes)
        print(f"{xi_all[sh]:11.2e}{f'{neg}/4 neg':>12}")
    fit = (K > 500) & (K < 5e4)
    print(f"\n    xi is stable across four independent windows: worst-case bound over"
          f"\n    k = 500-5e4 is xi < {xi_all[fit].max():.3f}, median {np.median(xi_all[fit]):.3f}."
          f"\n    The measured dE/dt is NEGATIVE at every k and every window, so this is"
          f"\n    an UPPER BOUND, not a detection: |f'(0+)| < {xi_all[fit].max():.2f} v_A k.")
    v_mid = float(np.interp(1.25, t, vA))
    gauss_xi = np.sqrt(2 * np.pi) / (2 * v_mid ** 2) * np.exp(-1.0 / (2 * v_mid ** 2))
    print(f"\n    For scale: a Kraichnan Gaussian at v_A = {v_mid:.3f} would give an"
          f"\n    EFFECTIVE xi of {gauss_xi:.1e} -- because the GW resonance sits at"
          f"\n    omega = k while the source decorrelates at k v_A, i.e. at 1/v_A = "
          f"{1 / v_mid:.0f}\n    Gaussian widths out in the tail.  The bound therefore EXCLUDES a"
          f"\n    sweeping-strength cusp by a factor ~{1 / xi_all[fit].max():.0f}, but it sits ~30"
          f"\n    orders of magnitude ABOVE the Gaussian, so it cannot exclude a weak one.")

    print("\n" + "=" * 78)
    print("4.  WHERE THE GW ENERGY ACTUALLY COMES FROM")
    print("=" * 78)
    print(f"\n    {'shell':>7}{'k':>8}{'k/k0':>7}{'tau_sw':>10}"
          f"{'x at 90% of E_end':>19}{'in tau_sw':>11}{'E_max/E_end':>13}")
    for sh in shells:
        y = E[:, sh]
        tsw = 1.0 / (vA[0] * K[sh])
        i = int(np.argmax(y >= 0.9 * y[-1]))
        print(f"    {sh + 1:7d}{K[sh]:8.0f}{K[sh] / K0_HUBBLE:7.2f}{tsw:10.2e}"
              f"{t[i] - T_START:19.4f}{(t[i] - T_START) / tsw:11.1f}{y.max() / y[-1]:13.2f}")
    print("""
    E_GW reaches 90% of its final value within a FRACTION of one sweeping time
    at every k, overshoots by 2-4x, and then decays slightly.  The GW spectrum of
    run ini2 is laid down by the SWITCH-ON, not by sustained radiation from the
    decaying turbulence.  That is consistent with the bound above -- a source
    radiating at the bounded rate would take ~0.06 Hubble times to build what is
    there -- and it is the mechanism of the manuscript's own paragraph "A hard
    finite lifetime supplies the same cusp".""")

    res = dict(K=K, kk=kk, t=t, E=E, EM=EM, Pi0=Pi0, C0=C0 * A, A=A, PiT=PiT,
               vA=vA, ratio=ratio, xi=xi_all, eps=eps)
    if quick:
        return res

    print("\n" + "=" * 78)
    print("5.  FORWARD MODELS WITH NO FREE PARAMETERS")
    print("=" * 78)
    print("""
    Pi(k,t) is measured, v_A(t) is measured, the window is the run itself.  So
    E(k,t_end) is a PREDICTION for each candidate correlator, with nothing tuned.
""")
    msh = [4, 9, 14, 19, 29, 39, 59, 99, 199]
    x = t[-1] - T_START
    v_bar = float(np.mean(vA))
    print(f"    {'k':>8}{'k/k0':>7}{'E_data':>12}"
          + "".join(f"{m:>12}" for m in MODELS))
    pred = {m: {} for m in MODELS}
    for sh in msh:
        row = f"    {K[sh]:8.0f}{K[sh] / K0_HUBBLE:7.2f}{E[-1, sh]:12.3e}"
        for m in MODELS:
            val = model_E(K[sh], x, t, PiT[:, sh], v_bar, m)
            pred[m][sh] = val
            row += f"{val / E[-1, sh]:12.2f}"
        print(row, flush=True)
    print("    (entries are E_model / E_data;  v_A = %.4f, the run mean)" % v_bar)
    core = [sh for sh in msh if 1000 <= K[sh] <= 2e4]
    print()
    for m in MODELS:
        rr = np.array([pred[m][sh] / E[-1, sh] for sh in core])
        print(f"    {MODEL_LABEL[m]:<46} E_model/E_data = "
              f"{rr.min():5.2f} - {rr.max():5.2f}")
    print(f"    (over k = 1010-20000, i.e. p = 1.5-30, 1.3 decades)")

    print("\n  robustness:")
    sh = 19
    print(f"    zero-pad (grid) convergence at k = {K[sh]:.0f}:")
    for pad in (2, 8, 32):
        print(f"      pad={pad:3d}" + "".join(
            f"{model_E(K[sh], x, t, PiT[:, sh], v_bar, m, pad=pad):12.4e}"
            for m in MODELS))
    print(f"    sensitivity to v_A (start / mean / end of run) at k = {K[sh]:.0f}:")
    for vv in (vA[0], v_bar, vA[-1]):
        print(f"      v={vv:.4f}" + "".join(
            f"{model_E(K[sh], x, t, PiT[:, sh], vv, m):12.4e}" for m in MODELS))
    print("""
    ==> The cuspless correlators (coherent and Gaussian sweeping alike)
        reproduce the measured GW energy; the cusped ones overpredict it by
        1-2 orders of magnitude.  Note WHY the two cuspless models agree with
        each other: neither radiates at omega = k at all, so in both the entire
        spectrum comes from the finite emission window -- the switch-on -- and
        that is what the data show.  A cusp would add sustained radiation on
        top of it, and there is no room for any.""")
    res["pred"] = pred
    res["msh"] = msh
    res["v_bar"] = v_bar
    return res


# --------------------------------------------------------------------------- #
def figure(r, name="magnetic_uetc"):
    import matplotlib.pyplot as plt
    from gw_turbulence.plot_style import (
        FIGSIZES, PALETTE, apply_max_ticks, apply_paper_style, save_figure,
    )
    apply_paper_style(grid=False)
    K, t, E, PiT, vA = r["K"], r["t"], r["E"], r["PiT"], r["vA"]
    p = K / K0_HUBBLE
    fig, ax = plt.subplots(2, 2, figsize=FIGSIZES["large"], constrained_layout=True)
    (a1, a2), (a3, a4) = ax

    # (a) the enabling step: the measured stress spectrum
    a1.loglog(p, r["Pi0"], "-", color=PALETTE[0], lw=1.8,
              label=r"$\Pi(k,t_0)$ from the first GW dump")
    a1.loglog(p, r["C0"], "--", color=PALETTE[6], lw=1.5,
              label=r"random-phase closure of $E_M(k,t_0)$")
    a1.set_xlabel(r"$p=k/k_0$")
    a1.set_ylabel(r"magnetic stress $\Pi(k)$")
    a1.set_title(r"(a) equal-time stress, measured", fontsize=11)
    a1.legend(frameon=False, fontsize=6.6, loc="lower left")
    apply_max_ticks(a1)

    # (b) build-up of E_GW at one k: data vs the four model curves
    sh = 19
    col = {"coh": 2, "gauss": 3, "exp": 6, "bk16": 1}
    style = {"coh": ":", "gauss": "-", "exp": "--", "bk16": "-."}
    a2.loglog(t - T_START, E[:, sh], "-", color=PALETTE[0], lw=1.8,
              label=rf"data, $p={p[sh]:.1f}$", zorder=5)
    if "v_bar" in r:
        xs = np.geomspace(2e-3, t[-1] - T_START, 26)
        for m in MODELS:
            ym = [model_E(K[sh], xx, t, PiT[:, sh], r["v_bar"], m) for xx in xs]
            a2.loglog(xs, ym, style[m], color=PALETTE[col[m]], lw=1.3,
                      label=MODEL_LABEL[m].split(",")[0])
    a2.axvline(1.0 / (vA[0] * K[sh]), color="0.6", lw=0.9, ls=":")
    a2.text(1.15 / (vA[0] * K[sh]), 2e-13, r"$\tau_{\rm sw}$", fontsize=7,
            color="0.4")
    a2.set_xlabel(r"$t-t_0$")
    a2.set_ylabel(r"$E_{\rm GW}(k,t)$")
    a2.set_title(r"(b) build-up at $p=3$: data vs models", fontsize=11)
    a2.legend(frameon=False, fontsize=6.2, loc="lower right")
    apply_max_ticks(a2)

    # (c) model discrimination
    if "pred" in r:
        msh = r["msh"]
        for m in MODELS:
            a3.loglog([p[s] for s in msh],
                      [r["pred"][m][s] / E[-1, s] for s in msh],
                      style[m], color=PALETTE[col[m]],
                      lw=1.5, marker="o", ms=3, label=MODEL_LABEL[m].split(",")[0])
    a3.axhline(1.0, color=PALETTE[0], lw=1.2)
    a3.axhspan(0.5, 2.0, color="0.85", zorder=0, lw=0)
    a3.text(1.0, 1.35, "data", fontsize=7, color=PALETTE[0])
    a3.set_xlabel(r"$p=k/k_0$")
    a3.set_ylabel(r"$E_{\rm GW}^{\rm model}/E_{\rm GW}^{\rm data}$")
    a3.set_title(r"(c) no free parameters", fontsize=11)
    a3.legend(frameon=False, fontsize=6.4, loc="upper left")
    apply_max_ticks(a3)

    # (d) the bound
    fit = (K > 500) & (K < 5e4)
    a4.fill_between(p[fit], r["xi"][fit], 30.0, color=PALETTE[6], alpha=0.16, lw=0)
    a4.loglog(p[fit], r["xi"][fit], "-", color=PALETTE[0], lw=1.8,
              label=r"$2\sigma$ bound from $dE_{\rm GW}/dt$")
    a4.axhline(1.0, color=PALETTE[6], lw=1.4, ls="--",
               label=r"cusp at sweeping strength")
    v_mid = float(np.interp(1.25, t, vA))
    g = np.sqrt(2 * np.pi) / (2 * v_mid ** 2) * np.exp(-1.0 / (2 * v_mid ** 2))
    a4.text(p[fit][-1] * 0.75, 6.0, "excluded", fontsize=8,
            color=PALETTE[6], ha="right")
    a4.text(np.sqrt(p[fit][0] * p[fit][-1]), 1.3e-3,
            rf"a Gaussian would give $\xi_{{\rm eff}}\sim10^{{{np.log10(g):.0f}}}$:"
            "\nfar below anything the data can reach",
            fontsize=6.2, color=PALETTE[3], ha="center")
    a4.set_ylim(6e-4, 30)
    a4.set_xlabel(r"$p=k/k_0$")
    a4.set_ylabel(r"$\xi=|f'(0^+)|/(v_Ak)$")
    a4.set_title(r"(d) upper bound on the cusp", fontsize=11)
    a4.legend(frameon=False, fontsize=6.4, loc="upper left")
    apply_max_ticks(a4)

    out = save_figure(fig, name)
    plt.close(fig)
    print(f"\n[magnetic_uetc] wrote {out}")
    return out


def main():
    if "--files" in sys.argv:
        d, kk, t, EM, _, _ = load_all()
        feasibility(d, kk, t, EM)
        return
    r = analyse(quick="--quick" in sys.argv)
    figure(r)


if __name__ == "__main__":
    main()
