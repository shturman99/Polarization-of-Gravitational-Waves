"""Plotting and scan helpers built on the numerical kernels."""

from __future__ import annotations

import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from .core import H_k0_analytic, H_pq, H_pq_decaying


PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "lines.linewidth": 2,
    "savefig.dpi": 200,
}

DEFAULT_FIGURE_SIZES = {
    "grid": (7, 5),
    "spectrum": (8, 6),
    "compact_spectrum": (6, 4),
}


def _parameter_tag(value: float) -> str:
    return f"{value:.2e}".replace("+", "p").replace("-", "m")


def _ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


@contextmanager
def _plot_style_context():
    with plt.rc_context(PLOT_STYLE):
        yield


def _finalize_plot(out_png: str) -> None:
    _ensure_parent(out_png)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def scan_and_plot_grid(
    Hfunc,
    M,
    R: float = 1e4,
    k0: float = 1.0,
    ps=None,
    qs=None,
    out_png: str = "outputs/H_pq_scan.png",
    out_npy: str = "outputs/Hgrid.npy",
):
    if ps is None:
        ps = np.logspace(-2, 1, 40)
    if qs is None:
        qs = np.logspace(-2, 1, 40)

    grid = np.zeros((qs.size, ps.size))
    for i, q in enumerate(qs):
        for j, p in enumerate(ps):
            try:
                grid[i, j] = Hfunc(p, q, M=M, R=R, k0=k0)
            except Exception:
                grid[i, j] = np.nan

    mstr = _parameter_tag(M)
    rstr = _parameter_tag(R)
    out_npy = f"{os.path.splitext(out_npy)[0]}_M{mstr}_R{rstr}{os.path.splitext(out_npy)[1]}"
    out_png = f"{os.path.splitext(out_png)[0]}_M{mstr}_R{rstr}{os.path.splitext(out_png)[1]}"

    _ensure_parent(out_npy)
    np.save(out_npy, {"ps": ps, "qs": qs, "H": grid})

    _ensure_parent(out_png)
    with _plot_style_context():
        plt.figure(figsize=DEFAULT_FIGURE_SIZES["grid"])
        plt.pcolormesh(ps, qs, grid, shading="auto", cmap="viridis")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("p = k/k0")
        plt.ylabel(r"q = $\omega$/k0")
        plt.title(f"H(p,q) 2D scan ({'decaying' if Hfunc is H_pq_decaying else 'stationary'}); M={M}, R={R}, k0={k0}")
        plt.colorbar(label="H")
        _finalize_plot(out_png)
    print(f"Wrote {out_png} and {out_npy}")


def plot_scans_for_M_list(M_list, R: float = 1e4, k0: float = 1.0, ps=None, qs=None):
    for M in M_list:
        scan_and_plot_grid(
            H_pq,
            M,
            R=R,
            k0=k0,
            ps=ps,
            qs=qs,
            out_png="outputs/H_pq_stationary.png",
            out_npy="outputs/Hgrid_stationary.npy",
        )
        scan_and_plot_grid(
            H_pq_decaying,
            M,
            R=R,
            k0=k0,
            ps=ps,
            qs=qs,
            out_png="outputs/H_pq_decaying.png",
            out_npy="outputs/Hgrid_decaying.npy",
        )


def plot_Hqq_decaying(
    M_list,
    qmin: float = 1e-2,
    qmax: float = 10.0,
    nq: int = 80,
    R: float = 1e4,
    k0: float = 1.0,
    out_png: str = "outputs/Hqq_decaying.png",
):
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    tagged = "-".join(_parameter_tag(M) for M in M_list)
    out_png = f"{os.path.splitext(out_png)[0]}_Ms{tagged}_R{R}{os.path.splitext(out_png)[1]}"
    with _plot_style_context():
        plt.figure(figsize=DEFAULT_FIGURE_SIZES["spectrum"])
        for M in M_list:
            values = np.zeros_like(qs)
            for i, q in enumerate(qs):
                try:
                    values[i] = H_pq_decaying(q, q, M=M, R=R, k0=k0)
                except Exception:
                    values[i] = np.nan
            plt.loglog(qs, (qs * values) ** 0.5, label=f"M={M}")
        plt.xlabel(r"q = $\omega$/k0 (and p=q)")
        plt.ylabel("h_c (decaying, p=q)")
        plt.title(f"H(q,q) spectra (decaying); R={R}, k0={k0}, nq={nq}")
        plt.legend()
        _finalize_plot(out_png)
    print(f"Wrote {out_png}")


def example_scan_and_plot(
    out_png: str = "outputs/H_pq.png",
    out_npy: str = "outputs/Hgrid.npy",
    M: float = 1.0,
    R: float = 1e4,
    k0: float = 1.0,
):
    ps = np.logspace(-2, 1, 60)
    qs = np.logspace(-2, 1, 60)
    scan_and_plot_grid(H_pq, M=M, R=R, k0=k0, ps=ps, qs=qs, out_png=out_png, out_npy=out_npy)


def plot_p0_spectra_params(
    M_list,
    qmin: float = 1e-4,
    qmax: float = 1e1,
    nq: int = 200,
    R: float = 1e4,
    k0: float = 1.0,
    out_png: str = "outputs/H_p0_params.png",
):
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    tagged = "-".join(_parameter_tag(M) for M in M_list)
    out_png = f"{os.path.splitext(out_png)[0]}_Ms{tagged}_R{R}{os.path.splitext(out_png)[1]}"
    with _plot_style_context():
        plt.figure(figsize=DEFAULT_FIGURE_SIZES["spectrum"])
        for M in M_list:
            values = np.array([H_k0_analytic(q, M=M, k0=k0, R=R) for q in qs])
            plt.loglog(qs, (qs * values) ** 0.5, label=f"M={M}")
        plt.xlabel(r"q = $\omega$/k0")
        plt.ylabel("h_c (p->0 analytic)")
        plt.title(f"p->0 analytic spectra; k0={k0}, R={R}, nq={nq}")
        plt.legend()
        _finalize_plot(out_png)
    print(f"Wrote {out_png}")


def plot_spectra_M(
    M_list,
    qmin: float = 1e-3,
    qmax: float = 10.0,
    nq: int = 200,
    out_png: str = "outputs/H_spectra_M.png",
):
    qs = np.logspace(qmin, qmax, nq)
    tagged = "-".join(_parameter_tag(M) for M in M_list)
    out_png = f"{os.path.splitext(out_png)[0]}_Ms{tagged}{os.path.splitext(out_png)[1]}"
    with _plot_style_context():
        plt.figure(figsize=DEFAULT_FIGURE_SIZES["compact_spectrum"])
        for M in M_list:
            values = H_k0_analytic(qs, M=M)
            plt.loglog(qs, (qs * values) ** 0.5, label=f"M={M}")
        plt.xlabel(r"q = $\omega$/k0")
        plt.ylabel("h_c (analytic p->0 scaling)")
        plt.title("Spectra for various Mach numbers (p->0 analytic)")
        plt.legend()
        _finalize_plot(out_png)
    print(f"Wrote {out_png}")


def plot_spectra_M_analytic(
    M_list,
    qmin: float = 1e-4,
    qmax: float = 1e1,
    nq: int = 300,
    out_png: str = "outputs/H_spectra_analytic.png",
    R: float = 1e6,
):
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    tagged = "-".join(_parameter_tag(M) for M in M_list)
    out_png = f"{os.path.splitext(out_png)[0]}_Ms{tagged}_R{_parameter_tag(R)}{os.path.splitext(out_png)[1]}"
    with _plot_style_context():
        plt.figure(figsize=DEFAULT_FIGURE_SIZES["spectrum"])
        for M in M_list:
            values = np.array([H_k0_analytic(q, M=M, k0=1.0, R=R) for q in qs])
            plt.loglog(qs, (qs * values) ** 0.5, label=f"M={M}")
        plt.xlabel(r"q = $\omega$/k0")
        plt.ylabel("H(0, q)")
        plt.title("Analytic p->0 spectra for various Mach numbers")
        plt.ylim(bottom=1e-21)
        plt.legend()
        _finalize_plot(out_png)
    print(f"Wrote {out_png}")
