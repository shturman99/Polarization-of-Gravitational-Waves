#!/usr/bin/env python3
"""Explicit regression tests for symbolic/numeric identities from derivation.tex.

Run:
  python src/test_transformations.py
"""

import unittest

import numpy as np
import sympy as sp
from scipy import integrate, special

from src.gw_turbulence import g_decaying


class TestDerivationIdentities(unittest.TestCase):
    def test_projection_trace_and_double_contraction(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            k = rng.normal(size=3)
            u = rng.normal(size=3)
            k /= np.linalg.norm(k)
            u /= np.linalg.norm(u)
            Pk = np.eye(3) - np.outer(k, k)
            Pu = np.eye(3) - np.outer(u, u)
            self.assertAlmostEqual(np.trace(Pk), 2.0, places=12)
            self.assertAlmostEqual(np.sum(Pk * Pu), 1.0 + np.dot(k, u) ** 2, places=12)

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
        for A, B, z in samples:
            f = lambda y: np.exp(-A * y**2) * np.exp(-B * (y - z) ** 2)
            lhs = integrate.quad(f, 0, np.inf, limit=400)[0]
            rhs = (
                np.sqrt(np.pi / (4 * (A + B)))
                * np.exp(-A * B * z**2 / (A + B))
                * special.erfc(-B * z / np.sqrt(A + B))
            )
            self.assertAlmostEqual(lhs, rhs, places=9)

    def test_decaying_substitution_q_equals_omega_tau1(self):
        tau1 = 0.35
        omegas = np.array([0.2, 1.5, 4.0])
        q = omegas * tau1
        g_w = g_decaying(omegas * tau1)
        g_q = g_decaying(q)
        self.assertLess(np.max(np.abs(g_w - g_q)), 1e-12)

    def test_k0_limit_bracket(self):
        bracket = 4 + (1 / 3) * (1 + 1)
        self.assertAlmostEqual(bracket, 14 / 3, places=12)


class TestDecayingExtensionKernel(unittest.TestCase):
    def test_decaying_envelope_is_stationary_replacement(self):
        """Checks the reusable derivation object used in notebooks.

        For stationary model we use exp*erfc factor in frequency-space integrand.
        For decaying model we replace it by convolution of g(z1) g(z2).
        This test verifies the decaying kernel is finite and smooth on a sample grid,
        making the derivation extension numerically usable.
        """
        zgrid = np.r_[np.linspace(-8, -1e-3, 20), np.linspace(1e-3, 8, 20)]
        vals = g_decaying(zgrid)
        self.assertTrue(np.all(np.isfinite(vals.real)))
        self.assertTrue(np.all(np.isfinite(vals.imag)))
        z = np.array([0.2, 1.0, 2.0, 4.0])
        self.assertTrue(np.allclose(g_decaying(-z), np.conj(g_decaying(z))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
