"""Difference between the two temporal-kernel models in the f_peak^GW(tau_*, M_0)
heatmaps of ``f_peak_vs_tau_mach.py``.

The "pairs" are the two temporal-correlation kernels evaluated on the *same*
(tau_*, M_0) grid and the *same* BK2016 self-similar decay class:

    Model A : random-phase BK2016   R_A(sigma) = (1+|sigma|)^{-2/3}
    Model B : coherent eddy          R_B(sigma) = cos(sigma) e^{-alpha|sigma|}

For each of the four decay classes we plot the signed log-ratio

    Delta(tau_*, M_0) = log10[ f_peak^A / f_peak^B ]

on the (M_0, tau_*) plane with a diverging colormap centred on 0.  Delta > 0
(red) means Model A peaks at a *higher* GW frequency than Model B, Delta < 0
(blue) the reverse.

The physics shortcut: both peak frequencies inherit the *same* overall
characteristic-frequency scaling with (tau_*, M_0); the kernels differ only in
their dimensionless peak position q_peak, so the ratio is expected to be a
constant offset per class.  This script tests that expectation numerically and
shows it on the plane.

This reuses the model functions from ``f_peak_vs_tau_mach.py`` verbatim; only the
inner Omega_GW evaluation is vectorised (and G_A cached as an interpolant) so a
fine grid is affordable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import f_peak_vs_tau_mach as fp  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    apply_max_ticks,
    apply_paper_style,
    pcolormesh_rasterized,
    save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Cached dimensionless kernel G_A(q) (Model A is otherwise the slow path)
# ---------------------------------------------------------------------------

_QGRID = np.linspace(0.0, 60.0, 6000)
_GA_TAB = fp.G_random_phase(_QGRID)  # even in q


def G_A_fast(q):
    return np.interp(np.abs(np.asarray(q, float)), _QGRID, _GA_TAB)


def G_of(model: str, q, alpha: float = 0.15):
    if model == "A":
        return G_A_fast(q)
    if model == "B":
        return fp.G_coherent(q, alpha=alpha)
    raise ValueError(model)


# ---------------------------------------------------------------------------
# Vectorised peak-frequency scan (same formulas as f_peak_vs_tau_mach)
# ---------------------------------------------------------------------------

def f_peak_grid(
    tau_grid: np.ndarray,
    M0_grid: np.ndarray,
    *,
    cls: fp.DecayClass,
    model: str,
    kp0: float = 1.0,
    T_em_in_tau: float = 5.0,
    n_omega: int = 600,
    n_T: int = 80,
    alpha_coherent: float = 0.15,
) -> np.ndarray:
    peaks = np.zeros((len(tau_grid), len(M0_grid)))
    for i, tau_star in enumerate(tau_grid):
        T_em = T_em_in_tau * tau_star
        T = np.linspace(0.0, T_em, n_T)
        te = fp.tau_e_of_T(T, u0=1.0, kp0=kp0, tau_star=tau_star)  # u0 cancels below
        for j, M0 in enumerate(M0_grid):
            u0 = fp.u0_from_mach(M0, kp0=kp0, tau_star=tau_star, p=cls.p)
            te_j = te / u0  # tau_e_of_T scales as 1/u0
            w = (fp.A_of_T(T, u0=u0, tau_star=tau_star, p=cls.p) ** 2
                 / fp.k0_of_T(T, kp0=kp0, tau_star=tau_star, q=cls.q) ** 2) * te_j
            omega_e0 = u0 * kp0
            omega_lo = 0.02 * omega_e0 / (1.0 + T_em / tau_star)
            omega_hi = 4.0 * omega_e0
            omega = np.linspace(omega_lo, omega_hi, n_omega)
            arg = omega[:, None] * te_j[None, :]            # (n_omega, n_T)
            G = G_of(model, arg, alpha=alpha_coherent)
            spec = _trapz(w[None, :] * G, T, axis=1)        # (n_omega,)
            peaks[i, j] = omega[np.argmax(omega * spec)] / (2.0 * np.pi)
    return peaks


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_difference(
    tau_grid: np.ndarray,
    M0_grid: np.ndarray,
    name: str = "f_peak_model_difference",
):
    apply_paper_style()
    log_ratios = {}
    for cls in fp.CLASSES:
        pA = f_peak_grid(tau_grid, M0_grid, cls=cls, model="A")
        pB = f_peak_grid(tau_grid, M0_grid, cls=cls, model="B")
        lr = np.log10(pA / pB)
        log_ratios[cls.name] = lr
        print(f"{cls.name:16s}  log10(fA/fB): "
              f"min {lr.min():+.4f}  max {lr.max():+.4f}  "
              f"=> fA/fB in [{10**lr.min():.3f}, {10**lr.max():.3f}]")

    vmax = max(np.abs(lr).max() for lr in log_ratios.values())
    vmax = max(vmax, 1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), sharex=True, sharey=True)
    pcm = None
    for ax, cls in zip(axes.ravel(), fp.CLASSES):
        lr = log_ratios[cls.name]
        pcm = pcolormesh_rasterized(
            ax, M0_grid, tau_grid, lr,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(cls.name)
        med = np.median(lr)
        ax.text(0.05, 0.92, rf"$\langle f_A/f_B\rangle={10**med:.2f}$",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=11, bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0.5))
        apply_max_ticks(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$M_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\tau_\ast$")

    cb = fig.colorbar(pcm, ax=axes, shrink=0.85, aspect=30, pad=0.02)
    cb.set_label(r"$\log_{10}\!\left(f_{\rm peak}^{A}/f_{\rm peak}^{B}\right)$")
    return save_figure(fig, name)


def _omega_spectrum(omega, *, cls, model, u0, kp0, tau_star, T_em_in_tau, n_T, alpha):
    """Time-integrated omega * Omega_GW(omega) at fixed (tau_*, M_0)."""
    T = np.linspace(0.0, T_em_in_tau * tau_star, n_T)
    te = fp.tau_e_of_T(T, u0=u0, kp0=kp0, tau_star=tau_star)
    w = (fp.A_of_T(T, u0=u0, tau_star=tau_star, p=cls.p) ** 2
         / fp.k0_of_T(T, kp0=kp0, tau_star=tau_star, q=cls.q) ** 2) * te
    arg = omega[:, None] * te[None, :]
    G = G_of(model, arg, alpha=alpha)
    spec = _trapz(w[None, :] * G, T, axis=1)
    return omega * spec


def plot_spectra_overlay(
    tau_star: float = 1.0,
    M0: float = 0.3,
    name: str = "f_peak_model_spectra",
    *,
    kp0: float = 1.0,
    T_em_in_tau: float = 5.0,
    n_omega: int = 1200,
    n_T: int = 120,
    alpha: float = 0.15,
):
    """omega * Omega_GW(omega) for Models A and B, one panel per decay class.

    Each curve is normalised to its own peak so the *horizontal* offset between
    the two peaks is the model difference quantified by ``plot_difference``.
    """
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), sharex=True)
    for ax, cls in zip(axes.ravel(), fp.CLASSES):
        u0 = fp.u0_from_mach(M0, kp0=kp0, tau_star=tau_star, p=cls.p)
        omega_e0 = u0 * kp0
        omega = np.logspace(np.log10(2e-3 * omega_e0), np.log10(8.0 * omega_e0), n_omega)
        f = omega / (2.0 * np.pi)
        for model, color in (("A", "C5"), ("B", "C1")):
            y = _omega_spectrum(omega, cls=cls, model=model, u0=u0, kp0=kp0,
                                tau_star=tau_star, T_em_in_tau=T_em_in_tau,
                                n_T=n_T, alpha=alpha)
            ypk = y / y.max()
            fpk = f[np.argmax(y)]
            ax.loglog(f, ypk, color=color, label=f"Model {model}")
            ax.axvline(fpk, color=color, ls=":", lw=1.0)
        ax.set_ylim(3e-3, 2.0)
        ax.set_title(cls.name)
        apply_max_ticks(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$f = \omega/2\pi$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f\,\Omega_{\rm GW}/\max$")
    axes[0, 0].legend(loc="lower center")
    fig.suptitle(rf"$\tau_\ast={tau_star:g},\ M_0={M0:g}$ "
                 r"(peak frequencies dotted)")
    return save_figure(fig, name)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all",
                        choices=("all", "heatmap", "spectra"))
    args = parser.parse_args()
    if args.which in ("all", "heatmap"):
        tau_grid = np.logspace(-1, 1, 24)
        M0_grid = np.logspace(-1.5, 0.5, 28)
        print(f"saved {plot_difference(tau_grid, M0_grid)}")
    if args.which in ("all", "spectra"):
        print(f"saved {plot_spectra_overlay()}")


if __name__ == "__main__":
    main()
