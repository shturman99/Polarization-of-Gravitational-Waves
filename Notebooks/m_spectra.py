"""GW spectra of the delta-function models with explicit Mach-number coupling.

Implements the four dimensionless kernels of derivation.tex Sec.~2 after the
Mach-number fix (k-dependent sweeping rate eta_k = (M/sqrt(2pi)) k0^{1/3} k^{2/3}):

  monochromatic + Kraichnan : H = K0(p)/p Theta(2-p) exp(-q^2/M^2)
  monochromatic + decay     : H = K0(p)/p Theta(2-p) T_dec(q;M)
  white-noise  + Kraichnan  : H = int dz dy (z/p) y Ktilde (z^{4/3}+y^{4/3})^{-1/2}
                                      exp(-2 q^2 / (M^2 (z^{4/3}+y^{4/3})))
  white-noise  + decay      : H = int dz dy (z/p) y Ktilde Jtilde(q;z,y,M)

Decay temporal factor — IMPORTANT
---------------------------------
The boxed decay kernels are written as the frequency convolution
int dq1 g(q1) g(qbar - q1).  Computed naively this DIVERGES: g(q) ~ |q|^{-5/3}
is singular at the origin and the discrete convolution scales with grid spacing.
By the convolution theorem the convolution equals 2*pi times the FT of the
PRODUCT of the time-domain correlations, which is finite and smooth:

    int dq1 g(.tau_a) g(.tau_b) = 2 int_0^inf dsigma cos(q sigma)
                                    (1 + sigma/tau_a)^{-2/3} (1 + sigma/tau_b)^{-2/3}

with f(t) = (1 + t/tau)^{-2/3} the decaying-turbulence correlation.  We evaluate
the decay factors this way (the divergent core.py `_temporal_conv_decay` should
not be used).

Smoothness: the (z,y) integrals use Gauss-Legendre quadrature on the *exact*
[a,b] intervals so the nodes track the moving bounds (a fixed linspace jitters
when a bound crosses a grid point).  The p-grid is dense.

Plot: Omega_GW(p) ~ p^3 H(p, q=p; M) on the sound-cone diagonal, one curve per M.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.core import K0_p  # noqa: E402
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz
SQRT2PI = np.sqrt(2.0 * np.pi)

MACH_LIST = (0.1, 0.3, 1.0, 3.0)
R_WHITE = 100.0

N_P_DELTA = 160
N_P_WHITE = 80
GL_N = 40

# FT-of-product quadrature for the decay temporal factor.
_SIGMA_MAX = 400.0
_N_SIGMA = 8000
_SIG = np.linspace(0.0, _SIGMA_MAX, _N_SIGMA)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(GL_N)


def _gl(a, b):
    xm = 0.5 * (b - a) * _GL_X + 0.5 * (a + b)
    wm = 0.5 * (b - a) * _GL_W
    return xm, wm


def Ktilde(p, z, y):
    return (
        27.0 / 6.0
        + p**4 / (12.0 * z**2 * y**2)
        + z**2 / (12.0 * y**2)
        + y**2 / (12.0 * z**2)
        - p**2 / (6.0 * y**2)
        - p**2 / (6.0 * z**2)
    )


def _ft_product(q, tau_a, tau_b):
    """2 int_0^inf cos(q s) (1+s/tau_a)^{-2/3}(1+s/tau_b)^{-2/3} ds  (finite, smooth)."""
    integrand = (
        np.cos(q * _SIG)
        * (1.0 + _SIG / tau_a) ** (-2.0 / 3.0)
        * (1.0 + _SIG / tau_b) ** (-2.0 / 3.0)
    )
    return 2.0 * float(_trapz(integrand, _SIG))


# ---------------------------------------------------------------------------
# Monochromatic E = E0 delta(k-k0)   (k1 = u = k0; tau at the single scale k0)
# ---------------------------------------------------------------------------

def h_delta_kraichnan(p, q, M):
    if p <= 0.0 or p > 2.0:
        return 0.0
    return K0_p(p) / p * np.exp(-(q**2) / M**2)


def h_delta_decay(p, q, M):
    if p <= 0.0 or p > 2.0:
        return 0.0
    tau = SQRT2PI / M  # z = 1
    return K0_p(p) / p * _ft_product(q, tau, tau)


# ---------------------------------------------------------------------------
# White-noise R_ij ~ delta^3(r)
# ---------------------------------------------------------------------------

def h_white_kraichnan(p, q, M, R=R_WHITE):
    Rd34 = R**0.75
    zg, zw = _gl(1.0, Rd34)
    total = 0.0
    for z, wz in zip(zg, zw):
        ylo = max(abs(p - z), 1.0)
        yhi = min(p + z, Rd34)
        if ylo >= yhi:
            continue
        yg, yw = _gl(ylo, yhi)
        s = z ** (4.0 / 3.0) + yg ** (4.0 / 3.0)
        integ = yg * Ktilde(p, z, yg) * s ** (-0.5) * np.exp(-2.0 * q**2 / (M**2 * s))
        total += wz * (z / p) * np.sum(yw * integ)
    return float(total)


def h_white_decay(p, q, M, R=R_WHITE):
    Rd34 = R**0.75
    zg, zw = _gl(1.0, Rd34)
    total = 0.0
    for z, wz in zip(zg, zw):
        ylo = max(abs(p - z), 1.0)
        yhi = min(p + z, Rd34)
        if ylo >= yhi:
            continue
        tau_z = SQRT2PI / (M * z ** (2.0 / 3.0))
        yg, yw = _gl(ylo, yhi)
        row = 0.0
        for yj, wj in zip(yg, yw):
            tau_y = SQRT2PI / (M * yj ** (2.0 / 3.0))
            row += wj * yj * Ktilde(p, z, yj) * _ft_product(q, tau_z, tau_y)
        total += wz * (z / p) * row
    return float(total)


# ---------------------------------------------------------------------------
# Plot: Omega_GW(p) ~ p^3 H(p, q=p; M) vs p, one curve per M
# ---------------------------------------------------------------------------

def _plot(name, kernel, ps, *, white=False):
    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    for c, M in enumerate(MACH_LIST):
        spec = np.array([
            p**3 * (kernel(p, p, M, R_WHITE) if white else kernel(p, p, M))
            for p in ps
        ])
        m = (spec > 0) & np.isfinite(spec)
        ax.loglog(ps[m], spec[m], color=PALETTE[c % len(PALETTE)],
                  label=rf"$M={M:g}$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$p^{3}\,\mathfrak{H}(p,p)$")
    ax.legend()
    apply_max_ticks(ax)
    out = save_figure(fig, name)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main():
    apply_paper_style()
    ps_delta = np.logspace(-2, np.log10(1.98), N_P_DELTA)
    ps_white = np.logspace(-1.5, 1.0, N_P_WHITE)

    _plot("m_spectrum_delta_kraichnan", h_delta_kraichnan, ps_delta)
    _plot("m_spectrum_delta_decay",     h_delta_decay,     ps_delta)
    _plot("m_spectrum_white_kraichnan", h_white_kraichnan, ps_white, white=True)
    _plot("m_spectrum_white_decay",     h_white_decay,     ps_white, white=True)
    print("done.")


if __name__ == "__main__":
    main()
