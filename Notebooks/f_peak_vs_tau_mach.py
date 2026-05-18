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

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


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
    result = 2.0 * np.array([np.trapezoid(np.cos(qi * sig) * weight, sig) for qi in q])
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
        spec[j] = np.trapezoid(H_vs_T, T_grid)
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

def plot_kernels(savepath: str | None = None):
    qs = np.linspace(-5, 5, 600)
    GA = G_random_phase(qs)
    GB = G_coherent(qs, alpha=0.15)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(qs, GA, label=r"Model A: random-phase BK2016")
    ax.plot(qs, GB, label=r"Model B: coherent eddy ($\alpha=0.15$)")
    ax.axvline( 2, color="0.6", ls=":", lw=1.0)
    ax.axvline(-2, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"$q = \omega\,\tau_e(T)$")
    ax.set_ylabel(r"$\mathcal{G}(q)$")
    ax.set_title("Dimensionless temporal kernel  $\\mathcal{G}(q) = \\widehat{R^2}(q)$")
    ax.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def plot_scan(peaks: np.ndarray, tau_grid: np.ndarray, M0_grid: np.ndarray,
              cls: DecayClass, model: str, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(
        M0_grid, tau_grid, peaks,
        shading="auto", cmap="viridis", norm="log",
    )
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$f_{\rm peak}^{\rm GW}\,[k_{p,0}]$  (units of $k_{p,0}/(2\pi)$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial Mach $M_0 = (p\,u_0^2 / 2\tau_\ast k_{p,0})^{1/3}$")
    ax.set_ylabel(r"decay time $\tau_\ast\,[k_{p,0}^{-1}]$")
    ax.set_title(
        f"{cls.name}  ($q={cls.q:.3f}$, $p={cls.p:.3f}$, $\\beta={cls.beta:.0f}$)\n"
        f"Model {model}"
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = "Notebooks"
    plot_kernels(savepath=f"{out_dir}/f_peak_kernels.pdf")

    tau_grid = np.logspace(-1, 1, 6)   # 6 decay times
    M0_grid  = np.logspace(-1.5, 0.5, 7)  # 7 Mach values (0.03 to 3.2)

    for cls in CLASSES:
        for model in ("A", "B"):
            peaks = scan_f_peak(tau_grid, M0_grid, cls=cls, model=model)
            stem = f"f_peak_{cls.name.replace(' ', '_').lower()}_model{model}"
            plot_scan(peaks, tau_grid, M0_grid, cls, model,
                      savepath=f"{out_dir}/{stem}.pdf")
            print(f"saved {out_dir}/{stem}.pdf")
            print(f"  peak range: {peaks.min():.3e} to {peaks.max():.3e}  [k_p0 / (2 pi)]")

    print("done.")


if __name__ == "__main__":
    main()
