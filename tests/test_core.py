#!/usr/bin/env python3
"""Tests for reusable numerical kernels."""

import unittest
from unittest import mock

import numpy as np

from src.gw_turbulence import core
from src.gw_turbulence.core import H_k0_analytic, H_pq_decaying_grid, g_decaying, kernel_bracket


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
