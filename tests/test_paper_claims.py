#!/usr/bin/env python3
r"""Unit tests that VERIFY every quantitative claim made in derivation.tex.

Each test recomputes a claim from first principles -- the GW kernels (core / Notebooks)
and the digitized Roper Pol CSVs -- and asserts the value the paper/figures state.  No
test trusts a hardcoded number: the digitized-data numbers come from roperpol_data.py
(the single source of truth that the figures also use), and the kernel numbers are
recomputed here.  Coverage:

  * numerical correctness   : g_decaying closed form vs its defining integral;
                              H_full(R_IR=1) == core.H_pq; exact M^4 factorisation.
  * physics interpretation  : causal k^3 infrared of both kernels; data k^1 (not k^3);
                              fluid k^4 -> k^-5/3; GW UV ~ k^-11/3.
  * Mach dependence         : stationary peak ~1.47 M; decaying peak ~2.4 (M-indep);
                              IR amplitude M^4 (decaying) vs M^3 (stationary);
                              peak amplitude steeper (~M^5.3).
  * data-derived claims     : k0, GW peak ~1.84 k0 (above source), Omega_M, UV bracket,
                              and no drift between the figure scripts and roperpol_data.

Fast by construction: the stationary kernel uses the vectorised H_full(R_IR=1)
(verified == core.H_pq), the decaying kernel uses H_decay_fast; grids are coarse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "Notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

import roperpol_data as rpd  # noqa: E402
from _fullspectrum_kernel import H_full, omega_gw_over_k  # noqa: E402
from fullspatial_decay import H_decay_fast  # noqa: E402
from fullspatial_selfsimilar import H_window  # noqa: E402
from gw_turbulence.core import H_pq  # noqa: E402


# ---- fast kernels in Omega_GW = p^3 H and Omega_GW/k = p^2 H conventions --------
def stat_H(p, M, R=1e4):
    """Stationary Kraichnan kernel via H_full(R_IR=1) == core.H_pq (fast, vectorised)."""
    return H_full(p, p, M=M, R=R, R_IR=1.0, x_points=90, y_points=90)


def dec_H(p, M, R=1e4):
    return H_decay_fast(p, p, M=M, R=R)


def _fit(x, y, lo, hi):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x >= lo) & (x <= hi)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def _powerlaw_exp(M, y):
    return float(np.polyfit(np.log(M), np.log(y), 1)[0])


def _peak(spec, plo, phi, n=26):
    ps = np.geomspace(plo, phi, n)
    sp = np.array([spec(p) for p in ps])
    i = int(np.argmax(sp))
    il, ir = max(i - 1, 0), min(i + 1, n - 1)
    c = np.polyfit(np.log(ps[il:ir + 1]), np.log(sp[il:ir + 1]), 2)
    return float(np.exp(-c[1] / (2 * c[0])))


# =================================================================================
# 1. NUMERICAL CORRECTNESS
# =================================================================================
def test_g_decaying_closed_form_matches_defining_integral():
    """core.g_decaying(q) == int_0^inf e^{+i q s} (1+s)^{-2/3} ds  (confirms the -1/3 exponent).

    The integrand is only conditionally convergent, so it is evaluated with the oscillatory
    (cos/sin weight) quadrature rather than a naive quad over [0, inf).
    """
    from gw_turbulence.core import g_decaying
    for q in (0.3, 1.0, 3.0):
        re, _ = integrate.quad(lambda s: (1 + s) ** (-2 / 3), 0, np.inf,
                               weight="cos", wvar=q, limit=200)
        im, _ = integrate.quad(lambda s: (1 + s) ** (-2 / 3), 0, np.inf,
                               weight="sin", wvar=q, limit=200)
        ref = re + 1j * im
        got = complex(g_decaying(q))
        assert abs(got - ref) < 1e-6 * abs(ref), (q, got, ref)


def test_Hfull_reduces_to_core_Hpq():
    """The full-spectrum kernel with no IR band (R_IR=1) reproduces core.H_pq to <1%."""
    for p in (0.3, 1.0, 3.0):
        a = H_full(p, p, M=1.0, R=1e4, R_IR=1.0, x_points=160, y_points=160)
        b = H_pq(p, p, M=1.0, R=1e4)
        assert abs(a / b - 1) < 0.01, (p, a, b)


def test_decaying_M4_factorisation_is_exact():
    """H_decay_fast(p, wM, M)/M^4 is M-independent (the exact factorisation, Eq. decay-amplitude)."""
    for p, w in ((0.5, 1.0), (2.4, 0.5)):
        vals = [H_decay_fast(p, w * M, M=M, R=1e4) / M ** 4 for M in (0.4, 0.8, 1.6)]
        assert max(vals) / min(vals) - 1 < 2e-3, (p, w, vals)


# =================================================================================
# 2. INFRARED SLOPES (physics interpretation)
# =================================================================================
def test_stationary_infrared_is_causal_k3():
    ps = np.geomspace(0.02, 0.2, 6)
    og = [p ** 3 * stat_H(p, 1.0) for p in ps]
    assert 2.7 < _fit(ps, og, ps[0], ps[-1]) < 3.3


def test_decaying_infrared_approaches_causal_k3():
    """Decaying Omega_GW -> causal k^3 as p->0 (slow, pre-asymptotic), and is NOT the data's k^1.

    The local slope rises monotonically toward 3 as the window deepens; at p~0.01-0.08 it is
    ~2.8.  The physically meaningful statement -- causal-steep (~3), not flat -- is what
    distinguishes every analytic kernel from the simulated k^1 (Sec. principal-discrepancy).
    """
    sh = np.geomspace(0.05, 0.4, 6)
    dp = np.geomspace(0.01, 0.08, 6)
    s_sh = _fit(sh, [p ** 3 * dec_H(p, 1.0) for p in sh], sh[0], sh[-1])
    s_dp = _fit(dp, [p ** 3 * dec_H(p, 1.0) for p in dp], dp[0], dp[-1])
    assert 2.6 < s_dp < 3.2, s_dp          # causal ~3 (pre-asymptotic)
    assert s_dp > s_sh                      # steepens toward 3 as p -> 0
    assert s_dp > 2.3                       # clearly NOT the data's k^1


def test_data_GW_infrared_is_k1_not_k3():
    """The discrepancy: simulated Omega_GW ~ k^1, clearly NOT the analytic causal k^3."""
    s = rpd.gw_ir_slope_omega_gw()
    assert 0.7 < s < 1.8, s
    assert s < 2.3, f"data IR slope {s} must be distinguishable from causal k^3"


def test_data_fluid_slopes_batchelor_kolmogorov():
    assert 3.6 < rpd.fluid_ir_slope() < 4.6      # Omega_M/k = E(k) ~ k^4 (Batchelor)
    assert -1.9 < rpd.fluid_uv_slope() < -1.4     # ~ k^-5/3 (Kolmogorov)


# ---- RESOLUTION: finite source duration bridges k^3 (analytic) and k^1 (sim) -----
def _win_H(p, coh, T_em, n=20):
    return H_window(p, p, M=1.0, R=1e4, T_em=T_em, coherence=coh, x_points=n, y_points=n)


def test_window_kernel_reduces_to_quasistationary():
    """T_em->inf, coherence=1 reproduces fullspatial_decay exactly (the build is correct)."""
    for p in (0.5, 1.0, 2.0):
        a = H_window(p, p, M=1.0, R=1e4, T_em=np.inf)
        b = H_decay_fast(p, p, M=1.0, R=1e4)
        assert abs(a / b - 1) < 1e-3, (p, a, b)


def test_finite_duration_flattens_IR_while_peak_survives():
    """A source coherent over its lifetime flattens the full-spatial infrared toward the
    simulated k^1 while the source-scale peak stays put (Sec. principal-discrepancy)."""
    band = np.geomspace(0.1, 0.7, 5)

    def slope(coh):
        return _fit(band, [p ** 3 * _win_H(p, coh, 20.0) for p in band], band[0], band[-1])

    s_qs, s_coh = slope(1.0), slope(16.0)
    assert s_qs > 2.2, s_qs                       # quasi-stationary: causal-steep
    assert s_coh < 1.8, s_coh                      # coherent: flattened toward k^1
    assert s_qs - s_coh > 0.5, (s_qs, s_coh)
    for coh in (1.0, 16.0):                        # peak stays source-scale in both
        pk = _peak(lambda p: p ** 3 * _win_H(p, coh, 20.0), 0.8, 6.0, n=16)
        assert 1.7 < pk < 3.0, (coh, pk)


def test_deep_IR_of_finite_duration_kernel_is_causal_k3():
    """The strict k->0 tail of the finite-duration kernel is still causal k^3 (analyticity)."""
    ps = np.geomspace(1e-3, 1e-2, 5)
    og = [p ** 3 * _win_H(p, 1.0, np.inf) for p in ps]
    assert 2.6 < _fit(ps, og, ps[0], ps[-1]) < 3.2


def test_coherence_ratio_controls_aeroacoustic_IR_slope():
    """Aeroacoustic IR slope falls from causal ~3 (fast decay, tau_c<<T_em) toward the
    simulated ~1 (coherent, tau_c>~T_em) as the coherence-over-lifetime ratio grows."""
    import selfsimilar_hybrid as ss
    LOITS = dict(u0=1.0, l0=1.0, p=10 / 7, q=2 / 7, s=4)
    T_em = 40.0
    band = np.geomspace(3.0 / T_em, 0.5, 6)

    def slope(tau_st):
        pars = {**LOITS, "tau_st": tau_st}
        og = [p ** 3 * ss.H_exact(p, T_em, n_T=110, n_k=64, **pars) for p in band]
        return _fit(band, og, band[0], band[-1])

    s_fast, s_slow = slope(0.3), slope(1e5)
    assert s_fast > 2.2, s_fast                    # fast decay -> causal-steep
    assert s_slow < 1.8, s_slow                    # coherent -> flat toward k^1
    assert s_fast - s_slow > 0.5, (s_fast, s_slow)


# =================================================================================
# 3. MACH-NUMBER DEPENDENCE
# =================================================================================
def test_stationary_peak_scales_as_1p47_M():
    Ms = np.array([0.3, 0.5, 0.7, 1.0])
    pk = np.array([_peak(lambda p: p ** 3 * stat_H(p, M), 0.05, 4.0) for M in Ms])
    n = _powerlaw_exp(Ms, pk)
    A = float(np.exp(np.polyfit(np.log(Ms), np.log(pk), 1)[1]))
    assert 0.9 < n < 1.1, f"exponent {n}"
    assert 1.30 < A < 1.65, f"prefactor {A}"


def test_decaying_peak_is_M_independent_near_2p4():
    Ms = (0.3, 0.5, 1.0, 2.0)
    pk = [_peak(lambda p: p ** 3 * dec_H(p, M), 0.5, 8.0) for M in Ms]
    assert all(2.1 < v < 2.8 for v in pk), pk
    assert max(pk) / min(pk) < 1.2, pk            # essentially flat in M


def test_decaying_IR_amplitude_scales_M4():
    Ms = np.array([0.3, 0.5, 1.0, 2.0])
    A = np.array([dec_H(0.1 * M, M) for M in Ms])  # Omega_GW/p^3 = H at fixed p/M
    assert 3.7 < _powerlaw_exp(Ms, A) < 4.3


def test_stationary_IR_amplitude_scales_M3():
    Ms = np.array([0.3, 0.5, 1.0, 2.0])
    A = np.array([stat_H(0.1 * M, M) for M in Ms])
    assert 2.7 < _powerlaw_exp(Ms, A) < 3.3


def test_decaying_peak_amplitude_steeper_than_IR():
    Ms = np.array([0.3, 0.5, 1.0, 2.0])
    pk = [_peak(lambda p: p ** 3 * dec_H(p, M), 0.5, 8.0) for M in Ms]
    Apk = np.array([(pp ** 3) * dec_H(pp, M) for pp, M in zip(pk, Ms)])
    n = _powerlaw_exp(Ms, Apk)
    assert 4.8 < n < 5.8, f"peak-amplitude exponent {n} (paper: ~5.3)"


# =================================================================================
# 4. ROPER POL DATA-DERIVED CLAIMS (and no drift vs the figure scripts)
# =================================================================================
def test_k0_value():
    assert 650 < rpd.k0() < 700                   # paper quotes ~673


def test_gw_peak_above_source_scale_near_2k0():
    r = rpd.gw_peak_ratio()
    assert r > 1.0, "GW peak must lie ABOVE the source scale k0"
    assert 1.6 < r < 2.1                          # ~1.84, the '2x source' claim


def test_energy_fraction_and_subsonic_mach():
    assert 1e-3 < rpd.energy_fraction() < 3e-3    # ~1.8e-3
    assert rpd.effective_mach()[1] < 0.1          # deeply subsonic


def test_data_UV_is_k_minus_11_3():
    assert -3.95 < rpd.gw_uv_slope_over_k() < -3.2  # ~ -11/3 = -3.67


def test_no_hardcoded_drift_in_figure_scripts():
    """Figure scripts must use the COMPUTED data numbers, not stale hardcoded ones."""
    import gw_peak_vs_mach as gwm
    import roperpol_comparison as cmp
    assert abs(cmp.K0 - rpd.k0()) < 1e-6
    assert abs(gwm.DATA_PEAK - rpd.gw_peak_ratio()) < 1e-6
    assert tuple(gwm.DATA_M) == pytest.approx(rpd.effective_mach())


# =================================================================================
# 5. THE KERNELS BRACKET THE DATA (the section-IV/peak claims)
# =================================================================================
def test_UV_slope_brackets_the_data():
    """Stationary UV too steep, decaying too shallow; data k^-11/3 lies between."""
    ps = np.geomspace(2.0, 14.0, 9)
    stat_uv = _fit(ps, [omega_gw_over_k(p, M=0.5, R=1e4, R_IR=100.0) for p in ps], 3, 12)
    dec_uv = _fit(ps, [p ** 2 * dec_H(p, 0.5) for p in ps], 3, 12)
    data_uv = rpd.gw_uv_slope_over_k()
    assert stat_uv < data_uv < dec_uv, (stat_uv, data_uv, dec_uv)


def test_peak_discriminator_stationary_below_decaying_above():
    """At the data's subsonic Mach: stationary peak below k0, decaying above (and ~data)."""
    M = float(np.sqrt(rpd.effective_mach()[0] * rpd.effective_mach()[1]))
    p_stat = _peak(lambda p: p ** 3 * stat_H(p, M), 0.01, 1.0)
    p_dec = _peak(lambda p: p ** 3 * dec_H(p, M), 0.5, 8.0)
    assert p_stat < 0.5, f"stationary peak {p_stat} should be far below source scale"
    assert p_dec > 1.5, f"decaying peak {p_dec} should be above source scale"
    assert abs(p_dec - rpd.gw_peak_ratio()) < 1.0  # decaying ~ data peak (~1.8-2.4)
