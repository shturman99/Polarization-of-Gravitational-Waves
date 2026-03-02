#!/usr/bin/env python3
"""Tests for plotting helpers and the CLI workflow."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from src.gw_turbulence import cli, plotting


class TestPlottingHelpers(unittest.TestCase):
    def test_scan_and_plot_grid_writes_tagged_paths_and_grid_payload(self):
        ps = np.array([0.1, 0.2])
        qs = np.array([0.3, 0.4])
        with (
            mock.patch("src.gw_turbulence.plotting.np.save") as save_mock,
            mock.patch("src.gw_turbulence.plotting.plt.savefig") as savefig_mock,
            mock.patch("src.gw_turbulence.plotting.plt.figure"),
            mock.patch("src.gw_turbulence.plotting.plt.pcolormesh"),
            mock.patch("src.gw_turbulence.plotting.plt.xscale"),
            mock.patch("src.gw_turbulence.plotting.plt.yscale"),
            mock.patch("src.gw_turbulence.plotting.plt.xlabel"),
            mock.patch("src.gw_turbulence.plotting.plt.ylabel"),
            mock.patch("src.gw_turbulence.plotting.plt.title"),
            mock.patch("src.gw_turbulence.plotting.plt.colorbar"),
            mock.patch("src.gw_turbulence.plotting.plt.tight_layout"),
            mock.patch("src.gw_turbulence.plotting.plt.close"),
        ):
            plotting.scan_and_plot_grid(
                lambda p, q, **_: p + q,
                M=0.5,
                R=10,
                ps=ps,
                qs=qs,
                out_png="tmp/scan.png",
                out_npy="tmp/scan.npy",
            )

        save_path, payload = save_mock.call_args.args
        self.assertEqual(save_path, "tmp/scan_M5.00em01_R1.00ep01.npy")
        self.assertTrue(np.array_equal(payload["ps"], ps))
        self.assertTrue(np.array_equal(payload["qs"], qs))
        self.assertTrue(np.allclose(payload["H"], [[0.4, 0.5], [0.5, 0.6]]))
        self.assertEqual(savefig_mock.call_args.args[0], "tmp/scan_M5.00em01_R1.00ep01.png")

    def test_plot_p0_spectra_params_writes_expected_file_name(self):
        with (
            mock.patch("src.gw_turbulence.plotting.H_k0_analytic", side_effect=lambda q, **_: q),
            mock.patch("src.gw_turbulence.plotting.plt.savefig") as savefig_mock,
            mock.patch("src.gw_turbulence.plotting.plt.figure"),
            mock.patch("src.gw_turbulence.plotting.plt.loglog"),
            mock.patch("src.gw_turbulence.plotting.plt.xlabel"),
            mock.patch("src.gw_turbulence.plotting.plt.ylabel"),
            mock.patch("src.gw_turbulence.plotting.plt.title"),
            mock.patch("src.gw_turbulence.plotting.plt.legend"),
            mock.patch("src.gw_turbulence.plotting.plt.tight_layout"),
            mock.patch("src.gw_turbulence.plotting.plt.close"),
        ):
            plotting.plot_p0_spectra_params([0.1, 1.0], nq=4, R=10, out_png="tmp/spectra.png")

        self.assertEqual(savefig_mock.call_args.args[0], "tmp/spectra_Ms1.00em01-1.00ep00_R10.png")

    def test_plot_hqq_decaying_writes_expected_file_name(self):
        with (
            mock.patch("src.gw_turbulence.plotting.H_pq_decaying", side_effect=lambda p, q, **_: p + q),
            mock.patch("src.gw_turbulence.plotting.plt.savefig") as savefig_mock,
            mock.patch("src.gw_turbulence.plotting.plt.figure"),
            mock.patch("src.gw_turbulence.plotting.plt.loglog"),
            mock.patch("src.gw_turbulence.plotting.plt.xlabel"),
            mock.patch("src.gw_turbulence.plotting.plt.ylabel"),
            mock.patch("src.gw_turbulence.plotting.plt.title"),
            mock.patch("src.gw_turbulence.plotting.plt.legend"),
            mock.patch("src.gw_turbulence.plotting.plt.tight_layout"),
            mock.patch("src.gw_turbulence.plotting.plt.close"),
        ):
            plotting.plot_Hqq_decaying([0.1], nq=4, R=10, out_png="tmp/diag.png")

        self.assertEqual(savefig_mock.call_args.args[0], "tmp/diag_Ms1.00em01_R10.png")


class TestCliWorkflow(unittest.TestCase):
    def test_build_parser_defaults(self):
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.mach_values, [0.001, 0.01, 0.1, 1.0])
        self.assertFalse(args.skip_scans)
        self.assertFalse(args.skip_decaying_diagonal)

    def test_cli_calls_expected_workflow_functions(self):
        with (
            mock.patch("src.gw_turbulence.cli.plot_p0_spectra_params") as plot_p0,
            mock.patch("src.gw_turbulence.cli.plot_scans_for_M_list") as plot_scans,
            mock.patch("src.gw_turbulence.cli.plot_Hqq_decaying") as plot_hqq,
        ):
            result = cli.main(["--mach-values", "0.001", "0.02", "--scan-points", "5"])

        self.assertEqual(result, 0)
        plot_p0.assert_called_once()
        plot_scans.assert_called_once()
        plot_hqq.assert_called_once()
        self.assertEqual(plot_scans.call_args.kwargs["R"], 1e4)
        self.assertEqual(plot_scans.call_args.args[0], [0.02])
        self.assertEqual(len(plot_scans.call_args.kwargs["ps"]), 5)

    def test_cli_skip_flags_avoid_expensive_steps(self):
        with (
            mock.patch("src.gw_turbulence.cli.plot_p0_spectra_params") as plot_p0,
            mock.patch("src.gw_turbulence.cli.plot_scans_for_M_list") as plot_scans,
            mock.patch("src.gw_turbulence.cli.plot_Hqq_decaying") as plot_hqq,
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = cli.main(["--skip-scans", "--skip-decaying-diagonal"])

        self.assertEqual(result, 0)
        plot_p0.assert_called_once()
        plot_scans.assert_not_called()
        plot_hqq.assert_not_called()
        self.assertIn("Requested plots generated.", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
