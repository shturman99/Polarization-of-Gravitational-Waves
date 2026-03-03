"""Command-line workflow entry points for common study outputs."""

from __future__ import annotations

import argparse

import numpy as np

from .plotting import plot_Hqq_decaying, plot_p0_spectra_params, plot_scans_for_M_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standard outputs for the gravitational-wave turbulence study."
    )
    parser.add_argument(
        "--mach-values",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0],
        help="Mach numbers used for spectra and scans.",
    )
    parser.add_argument("--R", type=float, default=1e4, help="Outer-scale ratio.")
    parser.add_argument("--k0", type=float, default=1.0, help="Reference wave number.")
    parser.add_argument("--scan-points", type=int, default=30, help="Grid size for coarse 2D scans.")
    parser.add_argument(
        "--use-mpi",
        action="store_true",
        help="Distribute decaying-grid evaluations across MPI ranks when launched under mpiexec.",
    )
    parser.add_argument(
        "--convolution-points",
        type=int,
        default=48,
        help="Number of samples for the fast decaying convolution approximation.",
    )
    parser.add_argument(
        "--x-points",
        type=int,
        default=16,
        help="Number of x samples for sampled decaying integration.",
    )
    parser.add_argument(
        "--y-points",
        type=int,
        default=16,
        help="Number of y samples for sampled decaying integration.",
    )
    parser.add_argument(
        "--skip-scans",
        action="store_true",
        help="Skip expensive 2D scans and generate only 1D spectra products.",
    )
    parser.add_argument(
        "--skip-decaying-diagonal",
        action="store_true",
        help="Skip H(q,q) decaying spectra.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print("=" * 60)
    print("H(p,q) Numerical Evaluator")
    print("=" * 60)

    print("\n[1] Generating analytic p->0 spectra (log-log)...")
    plot_p0_spectra_params(
        args.mach_values,
        qmin=1e-4,
        qmax=1e1,
        nq=200,
        R=args.R,
        k0=args.k0,
        out_png="outputs/H_p0_params.png",
    )

    if not args.skip_scans:
        print("[2] Generating 2D scans for stationary and decaying models (coarse)...")
        ps = np.logspace(-2, 1, args.scan_points)
        qs = np.logspace(-2, 1, args.scan_points)
        scan_machs = [value for value in args.mach_values if value >= 0.01]
        if not scan_machs:
            scan_machs = args.mach_values
        plot_scans_for_M_list(
            scan_machs,
            R=args.R,
            k0=args.k0,
            ps=ps,
            qs=qs,
            decaying_kwargs={
                "use_mpi": args.use_mpi,
                "convolution_method": "trapz",
                "convolution_points": args.convolution_points,
                "integration_method": "sampled",
                "x_points": args.x_points,
                "y_points": args.y_points,
            },
        )

    if not args.skip_decaying_diagonal:
        print("[3] Generating H(q,q) spectra (decaying, p=q) (coarse)...")
        scan_machs = [value for value in args.mach_values if value >= 0.01]
        if not scan_machs:
            scan_machs = args.mach_values
        plot_Hqq_decaying(
            scan_machs,
            qmin=1e-2,
            qmax=10.0,
            nq=60,
            R=args.R,
            k0=args.k0,
            out_png="outputs/Hqq_decaying.png",
            convolution_method="trapz",
            convolution_points=args.convolution_points,
            integration_method="sampled",
            x_points=args.x_points,
            y_points=args.y_points,
        )

    print("\n" + "=" * 60)
    print("Requested plots generated.")
    print("Files are in the outputs/ directory.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
