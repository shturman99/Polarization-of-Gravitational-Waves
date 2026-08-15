#!/usr/bin/env python3
"""Tests for reusable numerical kernels."""

import unittest
import warnings
from unittest import mock

import numpy as np

from src.gw_turbulence import core
from src.gw_turbulence.core import (
    H_k0_analytic,
    H_pq_decaying_grid,
    LiveStatusLogger,
    _cosine_grid,
    _conv_intervals,
    g_decaying,
    kernel_bracket,
)

# NumPy 2.0 renamed ``np.trapz`` -> ``np.trapezoid`` and removed the old name;
# requirements.txt pins 1.26.4, which only has ``np.trapz``.  Support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class TestCoreHelpers(unittest.TestCase):
    def test_kernel_bracket_is_symmetric_in_x_and_y(self):
        value_xy = kernel_bracket(0.7, 0.3, 0.9)
        value_yx = kernel_bracket(0.7, 0.9, 0.3)
        self.assertAlmostEqual(value_xy, value_yx, places=12)

    def test_g_decaying_scalar_and_vector_inputs_match(self):
        points = np.array([0.2, 0.5, 1.0])
        vector_values = g_decaying(points)
        scalar_values = np.array([g_decaying(point) for point in points])
        self.assertTrue(np.allclose(vector_values, scalar_values))

    def test_g_decaying_is_finite_for_small_nonzero_arguments(self):
        points = np.array([1e-8, 1e-6, 1e-4, 1e-2])
        values = g_decaying(points)
        self.assertTrue(np.all(np.isfinite(values.real)))
        self.assertTrue(np.all(np.isfinite(values.imag)))

    def test_g_decaying_matches_direct_fourier_transform(self):
        # Regression for the kernel-exponent bug: g is the forward FT of the
        # decorrelation (1+sigma)^{-2/3}, so Re g(q) must equal
        #   int_0^inf cos(q sigma) (1+sigma)^{-2/3} dsigma.
        # The earlier closed form used exponent -5/3 (correct is -1/3) and was
        # off by factors of ~2-3 with the wrong sign.
        from scipy import integrate

        for q in (0.5, 1.0, 2.0, 4.0):
            ref, _ = integrate.quad(
                lambda s: (1.0 + s) ** (-2.0 / 3.0),
                0.0,
                np.inf,
                weight="cos",
                wvar=q,
                limit=200,
            )
            self.assertAlmostEqual(g_decaying(q).real, ref, places=4)

    def test_g_decaying_small_q_power_law_is_minus_one_third(self):
        # |g(q)| ~ q^{-1/3} as q -> 0, so |g(q)|/|g(2q)| -> 2^(1/3) ~ 1.26.
        # The -5/3 bug would give 2^(5/3) ~ 3.17 -- a clean discriminator.
        ratio = abs(g_decaying(1e-4)) / abs(g_decaying(2e-4))
        self.assertAlmostEqual(ratio, 2.0 ** (1.0 / 3.0), delta=0.05)

    def test_temporal_conv_decay_is_finite_and_order_unity(self):
        # The corrected kernel makes the self-convolution converge to O(1);
        # the -5/3 bug made it non-integrable (~1e6, sign-flipping).
        for q in (0.5, 1.5, 4.0):
            val = float(core._temporal_conv_decay(q))
            self.assertTrue(np.isfinite(val))
            self.assertLess(abs(val), 50.0)
            self.assertGreater(abs(val), 1e-3)

    def test_temporal_conv_decay_matches_independent_cosine_transform(self):
        # Regression pin for the UV-tail defect.  The temporal factor must equal
        # pi * cosT(q), with cosT the cosine transform of the SQUARED two-sided
        # correlation [(1+|tau|)^{-2/3}]^2.  Reference is a brute-force
        # real-space trapezoid -- deliberately a different quadrature from the
        # weight='cos' route in core, so this is an independent check.
        #
        # The old truncated q1-convolution passed at small q but failed in the
        # UV: ~2x low at q=8 and sign-flipped by q=16.  q=8,16,32 are the pins
        # that matter; a regression there reintroduces the bug.
        tau = np.linspace(-4000.0, 4000.0, 8_000_001)
        squared = (1.0 + np.abs(tau)) ** (-4.0 / 3.0)
        for q in (0.3, 1.0, 4.0, 8.0, 16.0, 32.0):
            reference = np.pi * float(_trapz(squared * np.cos(q * tau), tau))
            value = float(core._temporal_conv_decay(q))
            self.assertAlmostEqual(value / reference, 1.0, delta=2e-3, msg=f"q={q}")
            self.assertGreater(value, 0.0, msg=f"q={q} must stay positive")

    def test_temporal_conv_decay_uv_tail_follows_eight_thirds_over_q_squared(self):
        # The cusp of (1+|tau|)^{-4/3} at tau=0 fixes the UV tail analytically:
        # cosT(q) -> (8/3)/q^2, so the temporal factor -> pi*(8/3)/q^2.  The
        # truncated convolution decayed far faster than this and changed sign.
        for q in (50.0, 100.0, 200.0):
            self.assertAlmostEqual(
                q**2 * float(core._temporal_conv_decay(q)) / (np.pi * 8.0 / 3.0),
                1.0,
                delta=0.01,
                msg=f"q={q}",
            )

    def test_ft_product_decay_closed_form_matches_quadrature_branch(self):
        # Equal tau takes the closed form, unequal tau the Feynman-parameter
        # Gauss-Jacobi rule.  Approaching tau_b -> tau_a the two must meet
        # (analytically they do: C * B(2/3, 2/3) = 1).
        for q in (0.5, 3.0, 11.0):
            closed = core.ft_product_decay(q, 1.0, 1.0)
            quad = core.ft_product_decay(q, 1.0, 1.0 + 1e-7)
            self.assertAlmostEqual(closed / quad, 1.0, delta=1e-5, msg=f"q={q}")

    def test_ft_product_decay_zero_frequency_and_tau_scaling(self):
        # F(0) = 2 int_0^inf (1+s)^{-4/3} ds = 6, and s -> tau*u gives the
        # scaling F(q; tau, tau) = tau * F(q*tau; 1, 1).
        self.assertAlmostEqual(core.ft_product_decay(0.0), 6.0, places=6)
        # The approach to 6 is only O(q^{1/3}) -- slow, but a clean power law.
        # Pin the exponent: a decade in q must shrink the deficit by 10^{1/3}.
        deficits = [6.0 - core.ft_product_decay(q) for q in (1e-6, 1e-7, 1e-8, 1e-9)]
        for coarse, fine in zip(deficits, deficits[1:]):
            self.assertAlmostEqual(coarse / fine, 10.0 ** (1.0 / 3.0), delta=0.01)
        for tau, q in ((2.0, 1.5), (0.3, 7.0)):
            self.assertAlmostEqual(
                core.ft_product_decay(q, tau, tau),
                tau * core.ft_product_decay(q * tau),
                places=9,
                msg=f"tau={tau}",
            )

    def test_ft_product_decay_unequal_tau_matches_direct_integral(self):
        # The unequal-tau (Feynman-parameter) branch against a direct real-space
        # evaluation of 2 int_0^inf cos(q s) (1+s/a)^{-2/3}(1+s/b)^{-2/3} ds.
        # Truncation at S costs ~2*3*(ab)^{1/3} S^{-1/3}, so S is taken large and
        # the residual tail is added back analytically.
        for q, a, b in ((1.0, 1.0, 2.0), (0.3, 0.05, 4.0), (7.0, 1.0, 2.0)):
            s = np.linspace(0.0, 2.0e4, 4_000_001)
            amp = (1.0 + s / a) ** (-2.0 / 3.0) * (1.0 + s / b) ** (-2.0 / 3.0)
            head = 2.0 * float(_trapz(np.cos(q * s) * amp, s))
            self.assertAlmostEqual(
                core.ft_product_decay(q, a, b) / head, 1.0, delta=5e-3,
                msg=f"q={q} a={a} b={b}",
            )

    def test_ft_product_decay_is_finite_across_the_sampled_domain(self):
        # Regression: the oscillatory-quadrature implementations of this factor
        # failed (or silently returned garbage) whenever the cosine period 1/q
        # was far larger than the amplitude scales -- exactly the deep IR.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for M in (0.1, 1.0, 3.0):
                for x in (1e-3, 1.0):
                    for y in (1e-3, 1.0):
                        for q in (0.0, 1e-9, 1e-3, 1.0, 1e3):
                            value = core.integrand_y_decaying(y, x, 0.7, q, M)
                            self.assertTrue(np.isfinite(value), msg=f"{M} {x} {y} {q}")

    def test_cos_transform_spline_matches_exact_closed_form(self):
        for z in (1e-5, 1e-2, 0.3, 1.0, 4.0, 16.0, 100.0, 1e4, 1e6):
            exact = core._cos_transform_sq_decay(z)
            spline = float(core._cos_transform_sq_decay_many(z)[0])
            self.assertAlmostEqual(spline / exact, 1.0, delta=1e-6, msg=f"z={z}")

    def test_integration_bounds_return_none_for_empty_region(self):
        self.assertIsNone(core._integration_bounds(x=1.0, p=5.0, R=1.0))

    def test_integration_bounds_are_ordered_for_valid_region(self):
        bounds = core._integration_bounds(x=0.5, p=0.4, R=1e4)
        self.assertIsNotNone(bounds)
        y_min, y_max = bounds
        self.assertLess(y_min, y_max)

    def test_h_k0_analytic_scalar_and_vector_results_agree(self):
        q_values = np.array([0.1, 0.2, 0.4])
        vector_result = H_k0_analytic(q_values, M=0.3, R=100)
        scalar_result = np.array([H_k0_analytic(q, M=0.3, R=100) for q in q_values])
        self.assertTrue(np.allclose(vector_result, scalar_result))

    def test_h_k0_analytic_rejects_non_positive_q_with_nan(self):
        result = H_k0_analytic(np.array([-1.0, 0.0, 0.2]), M=0.3, R=100)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isfinite(result[2]))

    def test_cosine_grid_endpoints_and_monotonicity(self):
        grid = _cosine_grid(0.5, 3.0, 50)
        self.assertAlmostEqual(grid[0], 0.5, places=12)
        self.assertAlmostEqual(grid[-1], 3.0, places=12)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_conv_intervals_splits_around_singularities(self):
        intervals = _conv_intervals(q=1.0, q_bound=5.0, split_width=0.01)
        self.assertEqual(len(intervals), 3)
        self.assertAlmostEqual(intervals[0][0], -5.0)
        self.assertAlmostEqual(intervals[0][1], -0.01)
        self.assertAlmostEqual(intervals[1][0], 0.01)
        self.assertAlmostEqual(intervals[1][1], 0.99)
        self.assertAlmostEqual(intervals[2][0], 1.01)
        self.assertAlmostEqual(intervals[2][1], 5.0)

    def test_conv_intervals_fallback_for_small_q(self):
        intervals = _conv_intervals(q=0.001, q_bound=5.0, split_width=0.01)
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0][0], -5.0)
        self.assertAlmostEqual(intervals[0][1], -0.01)
        self.assertAlmostEqual(intervals[1][0], 0.011)
        self.assertAlmostEqual(intervals[1][1], 5.0)


class TestDecayingGrid(unittest.TestCase):
    def test_decaying_grid_shape_matches_input_axes(self):
        ps = np.array([0.1, 0.2])
        qs = np.array([0.3, 0.4, 0.5])
        with mock.patch("src.gw_turbulence.core.H_pq_decaying", side_effect=lambda p, q, **_: p + q):
            grid = H_pq_decaying_grid(ps, qs, M=0.1, R=10)
        self.assertEqual(grid.shape, (len(qs), len(ps)))
        self.assertTrue(np.allclose(grid, [[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]))

    def test_decaying_grid_marks_failures_as_nan(self):
        def fake_h(p, q, **_):
            if np.isclose(p, 0.2) and np.isclose(q, 0.4):
                raise RuntimeError("boom")
            return p * q

        ps = np.array([0.1, 0.2])
        qs = np.array([0.3, 0.4])
        with mock.patch("src.gw_turbulence.core.H_pq_decaying", side_effect=fake_h):
            grid = H_pq_decaying_grid(ps, qs, M=0.1, R=10)
        self.assertTrue(np.isfinite(grid[0, 0]))
        self.assertTrue(np.isnan(grid[1, 1]))

    def test_stationary_and_decaying_integrands_short_circuit_for_non_positive_sum(self):
        self.assertEqual(core.integrand_y(-1.0, 1.0, 0.2, 0.3, 0.4), 0.0)
        self.assertEqual(core.integrand_y_decaying(-1.0, 1.0, 0.2, 0.3, 0.4), 0.0)

    def test_decaying_grid_reports_status_messages(self):
        messages = []

        def recorder(message, force=False):
            messages.append(message)

        with mock.patch("src.gw_turbulence.core.H_pq_decaying", side_effect=lambda p, q, **_: p + q):
            H_pq_decaying_grid(
                np.array([0.1, 0.2]),
                np.array([0.3]),
                M=0.1,
                R=10.0,
                status=recorder,
            )
        self.assertTrue(any("grid start" in message for message in messages))
        self.assertTrue(any("row 1/1 complete" in message for message in messages))

    def test_live_status_logger_accepts_force_keyword(self):
        logger = LiveStatusLogger(every_seconds=10.0)
        logger("first", force=True)
        logger("second", force=True)


class TestMPIGatherGrid(unittest.TestCase):
    def test_gather_grid_non_root_returns_zeros_array_not_none(self):
        from src.gw_turbulence.mpi import MPIContext, gather_grid
        mock_comm = mock.MagicMock()
        mock_comm.gather.return_value = None
        context = MPIContext(comm=mock_comm, rank=1, size=2)
        result = gather_grid({1: np.array([0.1, 0.2])}, (3, 2), context)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
