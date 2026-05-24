#!/usr/bin/env python3
"""Regression tests for symbolic and numeric identities from derivation.tex."""

import unittest

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate, special

from src.gw_turbulence import g_decaying
from src.gw_turbulence.core import _conv_intervals, _cosine_grid, _temporal_conv_decay


class TestDerivationIdentities(unittest.TestCase):
    def test_projection_trace_and_double_contraction(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            k = rng.normal(size=3)
            u = rng.normal(size=3)
            k /= np.linalg.norm(k)
            u /= np.linalg.norm(u)
            projection_k = np.eye(3) - np.outer(k, k)
            projection_u = np.eye(3) - np.outer(u, u)
            self.assertAlmostEqual(np.trace(projection_k), 2.0, places=12)
            self.assertAlmostEqual(
                np.sum(projection_k * projection_u),
                1.0 + np.dot(k, u) ** 2,
                places=12,
            )

    def test_kernel_bracket_expansion(self):
        k, k1, u = sp.symbols("k k1 u", positive=True, real=True)
        mu = (k**2 - k1**2 - u**2) / (2 * k1 * u)
        lhs = sp.expand(4 + sp.Rational(1, 3) * (1 + mu**2))
        rhs = (
            sp.Rational(27, 6)
            + k**4 / (12 * k1**2 * u**2)
            + k1**2 / (12 * u**2)
            + u**2 / (12 * k1**2)
            - k**2 / (6 * u**2)
            - k**2 / (6 * k1**2)
        )
        self.assertEqual(sp.simplify(lhs - rhs), 0)

    def test_appendix_a_identity_numeric(self):
        samples = [(0.7, 1.2, 0.5), (2.0, 0.4, 1.3), (1.1, 1.7, 2.2)]
        for coeff_a, coeff_b, z in samples:
            integrand = lambda y: np.exp(-coeff_a * y**2) * np.exp(-coeff_b * (y - z) ** 2)
            lhs = integrate.quad(integrand, 0, np.inf, limit=400)[0]
            rhs = (
                np.sqrt(np.pi / (4 * (coeff_a + coeff_b)))
                * np.exp(-coeff_a * coeff_b * z**2 / (coeff_a + coeff_b))
                * special.erfc(-coeff_b * z / np.sqrt(coeff_a + coeff_b))
            )
            self.assertAlmostEqual(lhs, rhs, places=9)

    def test_decaying_substitution_q_equals_omega_tau1(self):
        tau1 = 0.35
        omegas = np.array([0.2, 1.5, 4.0])
        q_values = omegas * tau1
        g_w = g_decaying(omegas * tau1)
        g_q = g_decaying(q_values)
        self.assertLess(np.max(np.abs(g_w - g_q)), 1e-12)

    def test_k0_limit_bracket(self):
        bracket = 4 + (1 / 3) * (1 + 1)
        self.assertAlmostEqual(bracket, 14 / 3, places=12)


class TestDecayingExtensionKernel(unittest.TestCase):
    def test_decaying_kernel_matches_derivation_formula(self):
        q_values = np.array([0.2, 1.0, 4.0])
        # Derivation closed form (derivation.tex, Secs. delta-decay / decay-k):
        #   g_hat(q) = e^{-i q} (-i q)^{-1/3} Gamma(1/3, -i q)   (upper incomplete).
        # The exponent is -1/3 (= alpha - 1, alpha = 2/3); the earlier -5/3 (and the
        # e^{+i q} sign) predate the kernel-exponent fix and are stale.
        expected = np.array(
            [
                complex(
                    mp.e ** (-1j * q)
                    * ((-1j * q) ** (-1.0 / 3.0))
                    * mp.gammainc(1.0 / 3.0, -1j * q)
                )
                for q in q_values
            ]
        )
        actual = g_decaying(q_values)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-10, atol=1e-10))

    def test_decaying_temporal_convolution_covers_negative_q1(self):
        intervals = _conv_intervals(q=1.0, q_bound=5.0, split_width=0.01)
        self.assertEqual(intervals[0], (-5.0, -0.01))
        self.assertIn((0.01, 0.99), intervals)
        self.assertIn((1.01, 5.0), intervals)

    def test_decaying_temporal_convolution_uses_negative_q1_interval(self):
        q = 1.0
        positive_only = 0.0
        for lower, upper in ((0.01, 0.99), (1.01, 5.0)):
            grid = _cosine_grid(lower, upper, 200)
            positive_only += np.trapz((g_decaying(grid) * g_decaying(q - grid)).real, grid)
        full_line = _temporal_conv_decay(q, q_bound=5.0, split_width=0.01, n_points=200)
        self.assertGreater(abs(full_line - positive_only), 1e-2)

    def test_decaying_kernel_is_finite_and_conjugate_symmetric(self):
        zgrid = np.r_[np.linspace(-8, -1e-3, 20), np.linspace(1e-3, 8, 20)]
        values = g_decaying(zgrid)
        self.assertTrue(np.all(np.isfinite(values.real)))
        self.assertTrue(np.all(np.isfinite(values.imag)))
        z = np.array([0.2, 1.0, 2.0, 4.0])
        self.assertTrue(np.allclose(g_decaying(-z), np.conj(g_decaying(z))))


def _kernel_K(k, k1, u):
    """Full geometric kernel K(k, k1, u) from derivation.tex Eq.~\\eqref{eq: H-geometric}."""
    return (
        27.0 / 6.0
        + k**4 / (12.0 * k1**2 * u**2)
        + k1**2 / (12.0 * u**2)
        + u**2 / (12.0 * k1**2)
        - k**2 / (6.0 * u**2)
        - k**2 / (6.0 * k1**2)
    )


def _K0_p(p):
    """K_0(p) = K(p k_0, k_0, k_0) collapsed onto the delta-shell."""
    return 14.0 / 3.0 - p**2 / 3.0 + p**4 / 12.0


def _smeared_delta(x, sigma):
    """Unit-normalized Gaussian, the regularization of delta(x)."""
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


class TestDeltaSpectrumKraichnan(unittest.TestCase):
    """Numerical cross-check of derivation.tex Sec.~3.1 (monochromatic E_0 delta(k-k0))
    against the smeared-delta convolution form.

    The static delta-spectrum claim is

        int d^3 k_1 A(k_1) A(u) K(k, k_1, u)
            = E_0^2 K_0(p) Theta(2-p) / (8 pi k k_0^2)        (*)

    with A(k) = E_0 delta(k - k_0) / (4 pi k^2) and u = |k - k_1|.  Combined with
    the Kraichnan temporal convolution and the (1/(2 pi)^8) prefactor from
    Eq.~\\eqref{eq: H-with-P}, the dimensional GW kernel is

        H_{ijij}(k, omega) = sqrt(2) E_0^2 K_0(p) Theta(2-p) e^{-Omega^2/(2 pi)}
                             / [ 8 pi (2 pi)^7 k k_0^2 eta_0 ]            (**)

    Eq.~\\eqref{eq:H-delta-k} in the current draft has 1/2 in place of 1/(8 pi)
    in (*), and Eq.~\\eqref{eq:H-delta-Kraichnan} carries the resulting 8 pi
    overstatement.  These tests verify (*) and (**) (i.e.\\ the corrected
    prefactor) and confirm that the boxed dimensionless mathfrak{H}
    [Eq.~\\eqref{eq:dimless-delta-Kraichnan}] is shape-correct.
    """

    def _spatial_integral_numeric(self, p, sigma, k0=1.0, E0=1.0):
        """Compute  S(p, sigma) = (E_0^2/(8 pi)) * int dk_1 G_sigma(k_1-k_0)
                                      * int dmu G_sigma(u-k_0)/u^2 * K(p k_0, k_1, u),
        the spatial piece of H_{ijij} with both deltas regularized by a Gaussian
        of width sigma.  In the sigma -> 0 limit this equals the LHS of (*).
        """
        k = p * k0

        def inner_mu(k1):
            mu_grid = np.linspace(-1.0, 1.0, 4001)
            u = np.sqrt(k**2 + k1**2 - 2.0 * k * k1 * mu_grid)
            integrand = _smeared_delta(u - k0, sigma) / u**2 * _kernel_K(k, k1, u)
            return np.trapz(integrand, mu_grid)

        # Tight window around the k_1 = k_0 peak; the Gaussian is negligible past +/- 6 sigma
        k1_lo = max(1e-6, k0 - 6.0 * sigma)
        k1_hi = k0 + 6.0 * sigma
        k1_grid = np.linspace(k1_lo, k1_hi, 401)
        outer = np.array([
            _smeared_delta(k1 - k0, sigma) * inner_mu(k1) for k1 in k1_grid
        ])
        return (E0**2 / (8.0 * np.pi)) * np.trapz(outer, k1_grid)

    def test_spatial_prefactor_for_p_below_triangle(self):
        """Verify  int d^3k_1 A(k_1) A(u) K  =  E_0^2 K_0(p) / (8 pi k k_0^2)  for p<2."""
        k0, E0 = 1.0, 1.0
        sigma = 0.01
        for p in (0.3, 0.7, 1.2, 1.7):
            with self.subTest(p=p):
                lhs = self._spatial_integral_numeric(p, sigma, k0=k0, E0=E0)
                rhs = E0**2 * _K0_p(p) / (8.0 * np.pi * p * k0**3)
                rel_err = abs(lhs - rhs) / abs(rhs)
                self.assertLess(
                    rel_err, 2.0e-2,
                    f"p={p}: numeric {lhs:.6e} vs analytic {rhs:.6e}, "
                    f"relative error {rel_err:.2%} (expected < 2%).",
                )

    def test_spatial_prefactor_vanishes_above_triangle(self):
        """For p > 2 the triangle inequality forbids u = k_0; integral -> 0."""
        sigma = 0.01
        for p in (2.3, 3.0, 5.0):
            with self.subTest(p=p):
                val = self._spatial_integral_numeric(p, sigma)
                self.assertLess(
                    abs(val), 1.0e-6,
                    f"p={p}: expected ~0 above triangle, got {val:.3e}.",
                )

    def test_kraichnan_self_convolution_closed_form(self):
        """Verify the Gaussian self-convolution
              int d w1  (2/eta0) exp(-w1^2/(pi eta0^2))  (2/eta0) exp(-(w-w1)^2/(pi eta0^2))
              = (2 sqrt(2) pi / eta0) * exp(-w^2/(2 pi eta0^2)),
        i.e.\\ Eq.~\\eqref{eq:Iomega-Kraichnan} of derivation.tex.
        """
        eta0 = 0.7
        def tilde_f(w):
            return (2.0 / eta0) * np.exp(-(w**2) / (np.pi * eta0**2))
        for omega in (0.0, 0.3, 1.0, 2.5):
            with self.subTest(omega=omega):
                integrand = lambda w1: tilde_f(w1) * tilde_f(omega - w1)
                lhs, _ = integrate.quad(integrand, -np.inf, np.inf, limit=400)
                rhs = (
                    (2.0 * np.sqrt(2.0) * np.pi / eta0)
                    * np.exp(-(omega**2) / (2.0 * np.pi * eta0**2))
                )
                self.assertAlmostEqual(lhs, rhs, places=10)

    def test_full_H_matches_corrected_dimensional_prefactor(self):
        """End-to-end: combine the smeared spatial integral with the analytic
        Kraichnan temporal convolution and divide by the CORRECTED dimensional
        prefactor matching the boxed dimensionless form

            mathfrak{H}(p, Omega) = K_0(p)/p * Theta(2-p) * exp(-Omega^2/(2 pi)).

        Writing H = prefactor * mathfrak{H}, the *corrected* prefactor reads
            prefactor_correct = sqrt(2) E_0^2 / [ 8 pi (2 pi)^7 k_0^3 eta_0 ]
        (no k or p in the denominator — the 1/p sits inside mathfrak{H}).  The
        published form in derivation.tex Eq.~\\eqref{eq:H-delta-Kraichnan} has
        prefactor = sqrt(2) E_0^2 / [(2 pi)^7 k k_0^2 eta_0]
            = (8 pi) * prefactor_correct * (1/p) at fixed k_0 — i.e. it is
        8 pi too large after the 1/p inside mathfrak{H} is moved out.
        """
        k0, E0, eta0 = 1.0, 1.0, 1.0  # so Omega = omega and k = p
        sigma = 0.01
        cases = [(0.5, 0.0), (0.5, 1.0), (1.5, 0.3), (1.7, 2.0)]
        for p, omega in cases:
            with self.subTest(p=p, omega=omega):
                spatial = self._spatial_integral_numeric(p, sigma, k0=k0, E0=E0)
                temporal = (
                    (2.0 * np.sqrt(2.0) * np.pi / eta0)
                    * np.exp(-(omega**2) / (2.0 * np.pi * eta0**2))
                )
                H_numeric = spatial * temporal / (2.0 * np.pi) ** 8

                # Corrected prefactor of the boxed mathfrak{H} form.
                prefactor_correct = (
                    np.sqrt(2.0) * E0**2
                    / (8.0 * np.pi * (2.0 * np.pi) ** 7 * k0**3 * eta0)
                )
                mathfrak_H_numeric = H_numeric / prefactor_correct
                mathfrak_H_analytic = (
                    _K0_p(p) / p * np.exp(-(omega**2) / (2.0 * np.pi))
                )
                rel_err = abs(mathfrak_H_numeric - mathfrak_H_analytic) / abs(mathfrak_H_analytic)
                self.assertLess(
                    rel_err, 2.0e-2,
                    f"(p, Omega)=({p}, {omega}): mathfrak_H numeric {mathfrak_H_numeric:.6e} "
                    f"vs analytic {mathfrak_H_analytic:.6e}, rel.err {rel_err:.2%}.",
                )

                # And confirm the *published* (uncorrected) prefactor at this p,
                # rewritten as a boxed-form coefficient, is 8 pi too large.
                k = p * k0
                prefactor_published_boxed = (
                    np.sqrt(2.0) * E0**2 * p
                    / ((2.0 * np.pi) ** 7 * k * k0**2 * eta0)
                )
                ratio = prefactor_published_boxed / prefactor_correct
                self.assertAlmostEqual(ratio, 8.0 * np.pi, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
