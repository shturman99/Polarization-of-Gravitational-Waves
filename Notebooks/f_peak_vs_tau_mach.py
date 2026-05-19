"""GW peak frequency from a delta-shell decaying source.

E(k, t) = A(t) * delta(k - k_0(t)) with the BK2016 self-similar parametrization
from spectra_explorer.ipynb:

    xi(t)   = (1 + t / tau_*)^q
    k_0(t)  = k_{p,0} * xi(t)^{-1} = k_{p,0} (1 + t / tau_*)^{-q}
    u(t)    = u_0 (1 + t / tau_*)^{-p/2}
    A(t)    = u(t)^2 = u_0^2 (1 + t / tau_*)^{-p}
    epsilon(t) = (p u_0^2 / (2 tau_*)) (1 + t / tau_*)^{-(p+1)}

Self-similarity constraint: p = 2 (1 - q),  equivalently  p = (1 + beta) q,
beta = 2/q - 3.  Then both omega_e(t) = u(t) k_0(t) and eta_0(t) ~
eps(t)^{1/3} k_0(t)^{2/3} scale as (1 + t/tau_*)^{-1} — class-independent.

Mach number (Gogoberidze 2007 convention):

    M(t) = (epsilon(t) / k_0(t))^{1/3} = M_0 (1 + t / tau_*)^{-(1-q)}
    M_0  = (p u_0^2 / (2 tau_* k_{p,0}))^{1/3}.

Instantaneous GW spectrum at k_GW = 0, from delta_spectrum_selfsimilar.ipynb:

    H(0, omega; T) = C * A(T)^2 / k_0(T)^2 * tau_e(T) * G(omega tau_e(T))

with C a universal dimensional constant and G(q) = FT of R^2(sigma) for whichever
temporal-correlation model:

    Model A (random-phase BK2016):   R_A(sigma) = (1 + |sigma|)^{-2/3}
        G_A(q) = 2 Re int_0^infty e^{i q sigma} (1+sigma)^{-4/3} d sigma
    Model B (coherent eddy, finite lifetime alpha = tau_e/tau_L):
        R_B(sigma) = cos(sigma) e^{-alpha |sigma|}
        G_B(q) = three Lorentzians of width 2 alpha at q = 0, +/- 2

Time-integrated:

    Omega_GW(omega) ~ int_0^{T_em} dT H(0, omega; T)

Peak frequency: argmax_{omega} [omega * Omega_GW(omega)]  (energy / log f).

This script scans the (tau_*, M_0) plane and plots f_peak for both models,
for the four BK2016 self-similar classes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    apply_max_ticks,
    apply_paper_style,
    pcolormesh_rasterized,
    save_figure,
)

# numpy < 2.0 calls this np.trapz; keep one alias so the rest of the file works
# on both old and new numpy releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# BK2016 self-similar decay classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayClass:
    name: str
    beta: float
    q: float

    @property
    def p(self) -> float:
        # self-similarity: p = (1 + beta) q  =  2 (1 - q)
        return 2.0 * (1.0 - self.q)


CLASSES = [
    DecayClass("HD Loitsiansky",  beta=4.0, q=2.0 / 7.0),
    DecayClass("HD Saffman",      beta=3.0, q=1.0 / 3.0),
    DecayClass("nonhelical MHD",  beta=1.0, q=1.0 / 2.0),
    DecayClass("helical MHD",     beta=0.0, q=2.0 / 3.0),
]


# ---------------------------------------------------------------------------
# Dimensionless scaling functions (k_{p,0} = 1, u_0 = 1, tau_* = 1 by default;
# all results can be rescaled afterwards)
# ---------------------------------------------------------------------------

def f_factor(T: np.ndarray | float, tau_star: float = 1.0) -> np.ndarray | float:
    return 1.0 + np.asarray(T) / tau_star


def k0_of_T(T, kp0=1.0, tau_star=1.0, q=2.0 / 7.0):
    return kp0 * f_factor(T, tau_star) ** (-q)


def u_of_T(T, u0=1.0, tau_star=1.0, p=10.0 / 7.0):
    return u0 * f_factor(T, tau_star) ** (-p / 2.0)


def A_of_T(T, u0=1.0, tau_star=1.0, p=10.0 / 7.0):
    return u0**2 * f_factor(T, tau_star) ** (-p)


def epsilon_of_T(T, u0=1.0, tau_star=1.0, p=10.0 / 7.0):
    return (p * u0**2 / (2.0 * tau_star)) * f_factor(T, tau_star) ** (-(p + 1.0))


def omega_e_of_T(T, u0=1.0, kp0=1.0, tau_star=1.0):
    # By the self-similarity closure omega_e = u_0 k_{p,0} / (1 + T/tau_*)
    return u0 * kp0 / f_factor(T, tau_star)


def tau_e_of_T(T, u0=1.0, kp0=1.0, tau_star=1.0):
    return 1.0 / omega_e_of_T(T, u0, kp0, tau_star)


def mach_of_T(T, u0=1.0, kp0=1.0, tau_star=1.0, p=10.0 / 7.0, q=2.0 / 7.0):
    """M(T) = (eps(T) / k_0(T))^{1/3} = M_0 (1 + T/tau_*)^{-(1-q)}.

    M_0 = (p u_0^2 / (2 tau_* k_{p,0}))^{1/3}.
    """
    return (epsilon_of_T(T, u0, tau_star, p) / k0_of_T(T, kp0, tau_star, q)) ** (1.0 / 3.0)


def mach_initial(u0=1.0, kp0=1.0, tau_star=1.0, p=10.0 / 7.0) -> float:
    return (p * u0**2 / (2.0 * tau_star * kp0)) ** (1.0 / 3.0)


def u0_from_mach(M0: float, kp0=1.0, tau_star=1.0, p=10.0 / 7.0) -> float:
    """Invert M_0 = (p u_0^2 / (2 tau_* k_{p,0}))^{1/3}."""
    return np.sqrt(M0**3 * 2.0 * tau_star * kp0 / p)


# ---------------------------------------------------------------------------
# Dimensionless kernels  G(q) = FT of R^2(sigma)
# ---------------------------------------------------------------------------

def G_random_phase(q: np.ndarray | float, sigma_max: float = 300.0, n: int = 8001) -> np.ndarray | float:
    """Model A: R_A = (1+|sigma|)^{-2/3}, so G_A(q) = 2 Re int_0^infty e^{i q s} (1+s)^{-4/3} ds.

    Broad fall-off, no side peaks.  G_A(0) = 6.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    sig = np.linspace(0.0, sigma_max, n)
    weight = (1.0 + sig) ** (-4.0 / 3.0)
    # 2 Re int = 2 int cos(q sigma) (1+s)^{-4/3} ds
    result = 2.0 * np.array([_trapz(np.cos(qi * sig) * weight, sig) for qi in q])
    return result if result.size > 1 else float(result[0])


def G_coherent(q: np.ndarray | float, alpha: float = 0.15) -> np.ndarray | float:
    """Model B: R_B = cos(sigma) e^{-alpha |sigma|}, so R_B^2 = e^{-2 alpha |sigma|} [1/2 + (1/2) cos(2 sigma)].

    G_B(q) = sum of three Lorentzians at q = 0, +/- 2, each of width 2 alpha.
    """
    a = 2.0 * alpha
    return (
        a / (a**2 + q**2)
        + 0.5 * a / (a**2 + (q - 2.0) ** 2)
        + 0.5 * a / (a**2 + (q + 2.0) ** 2)
    ) * 2.0  # matches the Re-FT normalisation of G_random_phase


# ---------------------------------------------------------------------------
# Instantaneous H(omega; T) and time-integrated Omega_GW(omega)
# ---------------------------------------------------------------------------

def H_instant(omega, T, model: str, *, u0=1.0, kp0=1.0, tau_star=1.0, p=10.0 / 7.0,
              q=2.0 / 7.0, alpha_coherent: float = 0.15) -> np.ndarray:
    """Instantaneous GW source kernel up to the universal dimensional constant C.

    H(0, omega; T) propto A^2(T) / k_0^2(T) * tau_e(T) * G(omega tau_e(T)).
    """
    A_T = A_of_T(T, u0=u0, tau_star=tau_star, p=p)
    k0_T = k0_of_T(T, kp0=kp0, tau_star=tau_star, q=q)
    te = tau_e_of_T(T, u0=u0, kp0=kp0, tau_star=tau_star)
    qq = np.asarray(omega) * te
    if model == "A":
        G = G_random_phase(qq)
    elif model == "B":
        G = G_coherent(qq, alpha=alpha_coherent)
    else:
        raise ValueError(f"Unknown model {model!r}; use 'A' or 'B'.")
    return (A_T**2 / k0_T**2) * te * G


def Omega_GW(omega_grid, T_grid, model: str, **kwargs) -> np.ndarray:
    """Time-integrated GW spectrum at k_GW = 0: int_0^{T_em} dT H_inst(omega, T)."""
    omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=float))
    spec = np.zeros_like(omega_grid)
    for j, om in enumerate(omega_grid):
        H_vs_T = np.array([H_instant(om, T, model, **kwargs) for T in T_grid])
        spec[j] = _trapz(H_vs_T, T_grid)
    return spec


def find_peak_frequency(omega_grid, spectrum, weighted: bool = True) -> float:
    """Argmax of omega * spectrum if weighted (energy / log f convention), else of spectrum."""
    arr = omega_grid * spectrum if weighted else spectrum
    return float(omega_grid[np.argmax(arr)])


# ---------------------------------------------------------------------------
# (tau_*, M_0) parameter scan
# ---------------------------------------------------------------------------

def scan_f_peak(
    tau_grid: np.ndarray,
    M0_grid: np.ndarray,
    *,
    cls: DecayClass,
    model: str,
    kp0: float = 1.0,
    T_em_in_tau: float = 5.0,
    n_omega: int = 200,
    n_T: int = 80,
    alpha_coherent: float = 0.15,
) -> np.ndarray:
    """Compute f_peak (= omega_peak / (2 pi)) on a 2-D (tau_*, M_0) grid."""
    peaks = np.zeros((len(tau_grid), len(M0_grid)))
    for i, tau_star in enumerate(tau_grid):
        T_em = T_em_in_tau * tau_star
        T_grid = np.linspace(0.0, T_em, n_T)
        for j, M0 in enumerate(M0_grid):
            u0 = u0_from_mach(M0, kp0=kp0, tau_star=tau_star, p=cls.p)
            # frequency window: from the late-time eddy frequency to a few times the initial one
            omega_e0 = u0 * kp0
            omega_lo = 0.02 * omega_e0 / (1.0 + T_em / tau_star)
            omega_hi = 4.0 * omega_e0
            omega_grid = np.linspace(omega_lo, omega_hi, n_omega)
            spec = Omega_GW(
                omega_grid, T_grid, model,
                u0=u0, kp0=kp0, tau_star=tau_star,
                p=cls.p, q=cls.q, alpha_coherent=alpha_coherent,
            )
            peaks[i, j] = find_peak_frequency(omega_grid, spec) / (2.0 * np.pi)
    return peaks


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_kernels(name: str = "f_peak_kernels"):
    qs = np.linspace(-5, 5, 600)
    GA = G_random_phase(qs)
    GB = G_coherent(qs, alpha=0.15)
    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    ax.plot(qs, GA, label=r"Model A")
    ax.plot(qs, GB, label=r"Model B")
    ax.axvline( 2, color="0.6", ls=":", lw=1.0)
    ax.axvline(-2, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\mathcal{G}(q)$")
    ax.legend()
    apply_max_ticks(ax)
    return save_figure(fig, name)


def plot_spectra_at_tau(
    cls: DecayClass,
    tau_star: float,
    M0_list: np.ndarray,
    name: str,
    *,
    kp0: float = 1.0,
    T_em_in_tau: float = 5.0,
    n_omega: int = 200,
    n_T: int = 80,
    omega_lo: float = 1e-3,
    omega_hi: float = 3e1,
    alpha_coherent: float = 0.15,
):
    """Full ``omega * Omega_GW(omega)`` spectrum at fixed tau_*, Mach overlay.

    For each ``M_0`` in ``M0_list`` two curves are drawn: Model A solid, Model B
    dotted. Curves at the same Mach share a color from the Okabe-Ito palette.
    """
    omega_grid = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_omega)
    T_em = T_em_in_tau * tau_star
    T_grid = np.linspace(0.0, T_em, n_T)

    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    for j, M0 in enumerate(M0_list):
        u0 = u0_from_mach(M0, kp0=kp0, tau_star=tau_star, p=cls.p)
        color = f"C{j}"
        for model, ls in (("A", "-"), ("B", ":")):
            spec = Omega_GW(
                omega_grid, T_grid, model,
                u0=u0, kp0=kp0, tau_star=tau_star,
                p=cls.p, q=cls.q, alpha_coherent=alpha_coherent,
            )
            mach_label = rf"$M_0 = {M0:g}$" if model == "A" else None
            ax.loglog(omega_grid, omega_grid * spec,
                      color=color, linestyle=ls, label=mach_label)

    # Two legends: colour = Mach, line style = model.
    mach_legend = ax.legend(loc="upper right", title=r"")
    ax.add_artist(mach_legend)
    style_handles = [
        plt.Line2D([], [], color="black", linestyle="-",  label="Model A"),
        plt.Line2D([], [], color="black", linestyle=":",  label="Model B"),
    ]
    ax.legend(handles=style_handles, loc="lower left")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\omega\,\Omega_{\rm GW}(\omega)$")
    ax.set_title(rf"{cls.name}: $\tau_\ast = {tau_star:g}$")
    apply_max_ticks(ax)
    return save_figure(fig, name)


def plot_scan(peaks: np.ndarray, tau_grid: np.ndarray, M0_grid: np.ndarray,
              cls: DecayClass, model: str, name: str):
    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    pcm = pcolormesh_rasterized(
        ax, M0_grid, tau_grid, peaks,
        cmap="viridis", norm="log",
    )
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$f_{\rm peak}^{\rm GW}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_0$")
    ax.set_ylabel(r"$\tau_\ast$")
    apply_max_ticks(ax)
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_heatmaps():
    apply_paper_style()
    out = plot_kernels()
    print(f"saved {out}")

    tau_grid = np.logspace(-1, 1, 6)   # 6 decay times
    M0_grid  = np.logspace(-1.5, 0.5, 7)  # 7 Mach values (0.03 to 3.2)

    for cls in CLASSES:
        for model in ("A", "B"):
            peaks = scan_f_peak(tau_grid, M0_grid, cls=cls, model=model)
            stem = f"f_peak_{cls.name.replace(' ', '_').lower()}_model{model}"
            out = plot_scan(peaks, tau_grid, M0_grid, cls, model, name=stem)
            print(f"saved {out}")
            print(f"  peak range: {peaks.min():.3e} to {peaks.max():.3e}  [k_p0 / (2 pi)]")

    print("heatmaps done.")


def main_spectra():
    """Full ``omega * Omega_GW(omega)`` spectra at fixed tau_*, Mach overlay."""
    apply_paper_style()
    tau_list = [0.1, 1.0, 10.0, 100.0]
    M0_list = np.array([0.03, 0.1, 0.3, 1.0, 3.0])

    for cls in CLASSES:
        for tau_star in tau_list:
            stem = f"f_spectrum_{cls.name.replace(' ', '_').lower()}_tau{tau_star:g}"
            out = plot_spectra_at_tau(cls, tau_star, M0_list, name=stem)
            print(f"saved {out}")
    print("spectra done.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all",
                        choices=("all", "heatmaps", "spectra"))
    args = parser.parse_args()
    if args.which in ("all", "heatmaps"):
        main_heatmaps()
    if args.which in ("all", "spectra"):
        main_spectra()


if __name__ == "__main__":
    main()
