#!/usr/bin/env python3
r"""Explicit reproduction of Cai, Pi & Sasaki, PRL 2020 (arXiv:1909.13728).

"Universal infrared scaling of gravitational wave background spectra."

They prove Omega_GW ~ k^3 in the infrared, and identify exactly which assumptions
buy that exponent.  Two of their exceptions are the two decay models of
derivation.tex, so the theorem is a genuine external check on our kernels rather
than a restatement of them.  This script reproduces both, from scratch, with no
dependence on core.py.

THEIR THEOREM.  Omega_GW ~ k^3 provided
  (I)   0 < int dl [source spectra]^2 < infinity                     (finiteness)
  (II)  k is below EVERY source scale: the peak k_*, the peak width Dk,
        AND the inverse duration Dt_s^-1 = (a_* Deta_s)^-1            (separation)
  (III) the modes re-enter during radiation domination.
The k^3 itself is pure counting: at horizon crossing of 1/k_* there are
N = (k_*/k)^3 causally disconnected patches, so Poisson statistics give
<h_k h_k> = N^-1 |h_*|^2 = (k/k_*)^3 |h_*|^2.

WHAT IS REPRODUCED HERE
  (A) Spectral width.  The anisotropic stress is the convolution of two source
      shells.  Reduced to one dimension it carries an explicit 1/k prefactor,

          Pi(k) = (2 pi / k) int dp p P(p) int_{|k-p|}^{k+p} dq q P(q).

      For a source of FINITE width the inner interval shrinks as 2k when k->0,
      cancelling the prefactor and leaving Pi(0) finite -> k^3.  For a
      DELTA-function shell the inner integral does not shrink, the 1/k survives,
      and the spectrum is k^{3-1} = k^2.  This is their delta-function exception,
      and it is the same 1/k pole that appears in our monochromatic kernel as
      K_0(p)/p.
  (B) Duration.  With a source coherent over a window Deta the temporal factor is
      |int_0^Deta e^{i k eta} deta|^2 = 4 sin^2(k Deta/2)/k^2, which tends to
      Deta^2 for k Deta << 1 but averages to 2/k^2 above -- removing two powers
      and turning k^3 into k^1.  This is condition (II)'s duration clause, and it
      is the origin of the intermediate k^1 band discussed in Sec. IV.

Figure: cai_pi_sasaki_reproduction (two panels, log-log).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gw_turbulence.plot_style import (  # noqa: E402
    FIGSIZES,
    PALETTE,
    apply_max_ticks,
    apply_paper_style,
    save_figure,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz

K_STAR = 1.0            # source peak wavenumber
GL_N = 200              # Gauss-Legendre nodes per exact interval


def _gl(a, b, n=GL_N):
    """Gauss-Legendre nodes/weights on the EXACT interval [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _shell(p, width):
    """Isotropic source power spectrum: a shell at K_STAR of log-width `width`.

    Normalised so the total power is width-independent, letting width -> 0
    approach a delta shell without changing the amplitude.
    """
    return np.exp(-0.5 * (np.log(p / K_STAR) / width) ** 2) / (p * width * np.sqrt(2 * np.pi))


def stress_convolution(k, width, n_sigma=10.0):
    """Pi(k) = (2 pi/k) int dp p P(p) int_{|k-p|}^{k+p} dq q P(q).

    The 1/k prefactor is explicit; whether it survives is decided entirely by
    how the inner integral behaves as k -> 0.

    Both integrals are restricted to the shell's actual support, ln p within
    n_sigma*width of ln k_*.  A fixed grid spanning many decades would badly
    under-resolve a narrow shell (and silently return nonsense), whereas this
    keeps the resolution per unit width fixed as width -> 0.
    """
    ln_lo, ln_hi = np.log(K_STAR) - n_sigma * width, np.log(K_STAR) + n_sigma * width
    ps, wp = _gl(ln_lo, ln_hi)
    ps, wp = np.exp(ps), wp * np.exp(ps)          # integrate in log p
    total = 0.0
    for p, w in zip(ps, wp):
        q_lo = max(abs(k - p), np.exp(ln_lo))
        q_hi = min(k + p, np.exp(ln_hi))
        if q_lo >= q_hi:
            continue
        qs, wq = _gl(q_lo, q_hi)
        inner = np.sum(wq * qs * _shell(qs, width))
        total += w * p * _shell(p, width) * inner
    return 2.0 * np.pi * total / k


def stress_delta_shell(k, k_star=K_STAR):
    """Pi(k) for an EXACT delta shell, in closed form.

    With P(p) = A delta(p-k_*)/(4 pi k_*^2) both integrals collapse and

        Pi(k) = A^2 / (8 pi k k_*^2),      0 < k < 2 k_*,

    i.e. a 1/k pole that survives all the way to k -> 0 because a zero-width
    shell has no lower scale to cut it off.  Hence Omega_GW ~ k^3 * (1/k) = k^2
    over the whole infrared -- Cai-Pi-Sasaki's delta-function exception.
    """
    return np.where(k < 2.0 * k_star, 1.0 / (8.0 * np.pi * k * k_star**2), 0.0)


def temporal_factor(k, duration):
    """|int_0^Deta e^{i k eta} deta|^2 = 4 sin^2(k Deta/2) / k^2."""
    return 4.0 * np.sin(0.5 * k * duration) ** 2 / k**2


def temporal_factor_averaged(k, duration):
    """Oscillation-averaged temporal factor, <sin^2> = 1/2  ->  2/k^2."""
    return 2.0 / k**2


def _slope(k, y, lo, hi):
    m = (k >= lo) & (k <= hi) & (y > 0) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[m]), np.log(y[m]), 1)[0])


def panel_a_width():
    """k^3 for finite width, k^2 for a delta shell -- their delta exception.

    A shell of log-width sigma introduces its OWN scale.  Their condition (II)
    demands k below every source scale, so k^3 is recovered only for k << sigma;
    in the band sigma << k << k_* the 1/k pole is unscreened and the slope is 2.
    A true delta shell has sigma = 0, so that band extends to k -> 0.
    """
    ks = np.geomspace(1e-5, 5e-1, 44)
    out = {}
    for width in (0.01, 0.001):
        out[width] = ks**3 * np.array([stress_convolution(k, width) for k in ks])
    out["delta"] = ks**3 * stress_delta_shell(ks)
    return ks, out


def panel_b_duration():
    """k^3 below the inverse duration, k^1 above -- their duration clause."""
    ks = np.geomspace(1e-2, 3e2, 900)
    return ks, ks**3 * temporal_factor(ks, 1.0), ks**3 * temporal_factor_averaged(ks, 1.0)


def main():
    apply_paper_style()

    ks_a, curves_a = panel_a_width()
    ks_b, spec_b, spec_b_avg = panel_b_duration()

    print("=" * 74)
    print("REPRODUCTION OF Cai, Pi & Sasaki (arXiv:1909.13728)")
    print("=" * 74)
    print("\n(A) spectral width -> the delta-function exception (their k^2)")
    print("    A shell of log-width sigma is itself a source scale, so k^3 needs")
    print("    k << sigma; between sigma and k_* the 1/k pole is unscreened.")
    print(f"    {'shell':>10}{'slope k<<sigma':>16}{'slope sigma<<k<<k_*':>22}")
    for width, spec in curves_a.items():
        if width == "delta":
            print(f"    {'delta':>10}{'--':>16}{_slope(ks_a, spec, 1e-5, 3e-1):22.3f}"
                  f"   <- k^2 everywhere (no lower scale)")
        else:
            deep = _slope(ks_a, spec, 1e-5, 0.2 * width)
            band = _slope(ks_a, spec, 10.0 * width, 3e-1)
            print(f"    {width:10.3f}{deep:16.3f}{band:22.3f}")
    print("    -> finite width: 3 deep in the IR, 2 in the unscreened band.")
    print("       delta shell: 2 all the way down -- zero width violates their")
    print("       condition (II), exactly their k^{3-1} = k^2.")

    print("\n(B) source duration -> their condition (II), k << 1/Deta")
    s_lo = _slope(ks_b, spec_b, 1e-2, 2e-1)
    s_hi = _slope(ks_b, spec_b_avg, 3e1, 3e2)
    print("    Deta = 1, so the condition reads k << 1")
    print(f"    slope for k << 1/Deta            : {s_lo:+.3f}   expected +3")
    print(f"    slope for k >> 1/Deta (osc. avg) : {s_hi:+.3f}   expected +1")
    print("    -> k^3 holds ONLY below the inverse duration.  Above it the finite")
    print("       emission window contributes k^-2 and the slope drops to 1.")
    print("       (Cai-Pi-Sasaki quote k^9 above 1/Deta for SOUND WAVES, where a")
    print("        coherent oscillating source resonates with the Green function;")
    print("        for an incoherent window like ours the loss is two powers.)")

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZES["large"][0], FIGSIZES["small"][1]))

    ax = axes[0]
    for i, key in enumerate((0.01, 0.001, "delta")):
        spec = curves_a[key]
        lbl = r"$\delta$-shell" if key == "delta" else rf"$\sigma_{{\ln k}}={key:g}$"
        ax.loglog(ks_a, spec / spec[-1], color=PALETTE[i + 1], lw=1.5, label=lbl)
        if key != "delta":
            ax.axvline(key, color=PALETTE[i + 1], lw=0.8, ls="-.", alpha=0.5)
    ref = ks_a / ks_a[-1]
    ax.loglog(ks_a, ref**3, color=PALETTE[0], ls=":", lw=1.2, label=r"$k^{3}$")
    ax.loglog(ks_a, ref**2, color=PALETTE[0], ls="--", lw=1.2, label=r"$k^{2}$")
    ax.set_xlabel(r"$k/k_{*}$")
    ax.set_ylabel(r"$\Omega_{\rm GW}$ (arb.)")
    ax.set_title(r"(a) spectral width", fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    apply_max_ticks(ax)

    ax = axes[1]
    ax.loglog(ks_b, spec_b, color=PALETTE[5], lw=1.0, alpha=0.6,
              label=r"$k^3|\tilde W|^2$")
    ax.loglog(ks_b, spec_b_avg, color=PALETTE[3], lw=1.5, label=r"osc.\ average")
    ax.loglog(ks_b, ks_b**3, color=PALETTE[0], ls=":", lw=1.2, label=r"$k^{3}$")
    ax.loglog(ks_b, 2.0 * ks_b, color=PALETTE[0], ls="--", lw=1.2, label=r"$k^{1}$")
    ax.axvline(1.0, color=PALETTE[6], lw=1.0, ls="-.")
    ax.text(1.4, 3e-4, r"$k=1/\Delta\eta$", fontsize=7, color=PALETTE[6])
    ax.set_xlabel(r"$k\,\Delta\eta$")
    ax.set_ylim(1e-5, 1e3)
    ax.set_title(r"(b) source duration", fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    apply_max_ticks(ax)

    fig.tight_layout()
    out = save_figure(fig, "cai_pi_sasaki_reproduction")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
