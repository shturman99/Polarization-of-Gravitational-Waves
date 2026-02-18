#!/usr/bin/env python3
"""
Numerical evaluator for H(p,q) from derivation.tex.

Usage:
  python src/compute_H.py        # runs an example scan and writes outputs/H_pq.png and outputs/Hgrid.npy
Functions:
  H_pq(p, q, M=1.0, R=1e6, k0=1.0, epsabs=1e-8, epsrel=1e-6)
"""
import os
import numpy as np
from scipy import integrate, special
import matplotlib.pyplot as plt

def kernel_bracket(p, x, y):
    xp32 = x**1.5
    yp32 = y**1.5
    return (27.0
            - p**2 * xp32
            - p**2 * yp32
            + 0.5 * p**4 * xp32 * yp32
            + 0.5 * x**(-1.5) * yp32
            + 0.5 * y**(-1.5) * xp32)

def integrand_y(y, x, p, q, M):
    s = x + y
    if s <= 0:
        return 0.0
    pref = x**0.75 * y**0.75 * s**(-0.5)
    br = kernel_bracket(p, x, y)
    expo = np.exp(-2.0 * x * y / s * (q**2) / (M**2))
    er = special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(s)))
    return pref * br * expo * er

def inner_integral(x, p, q, M, R, epsabs, epsrel):
    tilde_k1 = x**(-0.75)
    u_min = max(abs(tilde_k1 - p), 1.0)
    u_max = min(tilde_k1 + p, R**(0.75))
    if not (u_min < u_max):
        return 0.0
    y_min = u_max**(-4.0/3.0)
    y_max = u_min**(-4.0/3.0)
    if not (y_min < y_max):
        return 0.0
    val, err = integrate.quad(integrand_y, y_min, y_max,
                              args=(x, p, q, M),
                              epsabs=epsabs, epsrel=epsrel, limit=200)
    return val

def H_pq(p, q, M=1.0, R=1e6, k0=1.0, epsabs=1e-8, epsrel=1e-6):
    """
    Compute H(p,q) (dimensionful overall prefactor included as in derivation).
    - p: dimensionless wavenumber p = k/k0 (must be > 0; very small p handled by a floor)
    - q: dimensionless frequency q = omega/k0
    - M, R, k0: parameters from derivation (defaults mimic a wide inertial range)
    Returns: float H(p,q)
    Note: For extremely small p we use a small floor to avoid division by zero.
    """
    p_floor = max(p, 1e-8)
    x_lo = R**(-1)
    x_hi = 1.0

    def outer_x(xx):
        return inner_integral(xx, p_floor, q, M, R, epsabs, epsrel)

    I, err_x = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=200)

    pref = 3.0 * (M**3) * (k0**(-4)) / (256.0 * (2.0 * np.pi)**1.5 * p_floor)
    return pref * I

def example_scan_and_plot(out_png="outputs/H_pq.png", out_npy="outputs/Hgrid.npy",
                          M=1.0, R=1e4, k0=1.0):
    ps = np.logspace(-2, 1, 60)   # p in [1e-2, 10]
    qs = np.logspace(-2, 1, 60)  # q in [1e-2, 10]
    Hgrid = np.zeros((qs.size, ps.size))
    for i, q in enumerate(qs):
        for j, p in enumerate(ps):
            try:
                Hgrid[i, j] = H_pq(p, q, M=M, R=R, k0=k0)
            except Exception:
                Hgrid[i, j] = np.nan

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    np.save(out_npy, {"ps": ps, "qs": qs, "H": Hgrid})
    plt.figure(figsize=(7,5))
    plt.pcolormesh(ps, qs, Hgrid, shading='auto', cmap='viridis')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('p = k/k0')
    plt.ylabel('q = ω/k0')
    plt.title('H(p,q) (dimensionful prefactor included)')
    plt.colorbar(label='H')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Wrote {out_png} and {out_npy}")


def H_k0_analytic(q, M=1.0, k0=1.0, R=1e6):
    """Analytic p->0 limit: H_ijij(0, q) from derivation.
    
    H_ijij(0, q) ≃ (7 M^3 k0^{-4}) / (16 π^{3/2})
                   * ∫_1^R^{-1} dx x^{11/4} exp(-q̄² x) erfc(-q̄ x^{1/2})
    
    where q̄ = q/M (dimensionless frequency scaled by M).
    This evaluation includes the prefactor shown in derivation.
    """
    q_array = np.atleast_1d(q)
    out = np.zeros_like(q_array, dtype=float)
    
    x_lo = 1.0
    x_hi = R**(-1)
    
    for idx, q_val in enumerate(q_array):
        if q_val <= 0:
            out[idx] = np.nan
            continue
        
        q_bar = q_val / M
        
        def integrand_x(x):
            # x^{11/4} * exp(-q̄² x) * erfc(-q̄ x^{1/2})
            return x**(11.0/4.0) * np.exp(-q_bar**2 * x) * special.erfc(-q_bar * np.sqrt(x))
        
        try:
            integral, err = integrate.quad(integrand_x, x_hi, x_lo, epsabs=1e-10, epsrel=1e-8, limit=300)
        except Exception:
            integral = 0.0
        
        # prefactor: (7 M^3 k0^{-4}) / (16 π^{3/2})
        pref = (7.0 * M**3 * k0**(-4)) / (16.0 * np.pi**1.5)
        out[idx] = pref * integral
    
    # return scalar if input was scalar
    if np.isscalar(q):
        return out[0]
    return out


def plot_spectra_M(M_list, qmin=1e-3, qmax=10.0, nq=200, out_png='outputs/H_spectra_M.png'):
    qs = np.logspace(qmin, qmax, nq)
    plt.figure(figsize=(6,4))
    for M in M_list:
        Hvals = H_k0_analytic(qs, M=M)
        plt.loglog(qs, (qs * Hvals)**(0.5), label=f'M={M}')
    plt.xlabel('q = ω/k0')
    plt.ylabel('h_c (analytic p->0 scaling)')
    plt.title('Spectra for various Mach numbers (p->0 analytic)')
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png}')


def plot_spectra_M_analytic(M_list, qmin=1e-4, qmax=1e1, nq=300, out_png='outputs/H_spectra_analytic.png', R=1e6):
    """Plot the correct analytic p->0 spectra (no grid, log-log)."""
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    plt.figure(figsize=(8, 6))
    for M in M_list:
        Hvals = np.array([H_k0_analytic(q, M=M, k0=1.0, R=R) for q in qs])
        plt.loglog(qs,(qs * Hvals) ** (0.5), label=f'M={M}', linewidth=2)
    plt.xlabel('q = ω/k0', fontsize=12)
    plt.ylabel('H(0, q)', fontsize=12)
    plt.title('Analytic p→0 spectra for various Mach numbers', fontsize=12)
    plt.ylim(bottom=1e-21)
    plt.legend(fontsize=10)
    # no grid per request
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png}')


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("H(p,q) Numerical and Analytic Evaluator")
    print("=" * 60)
    
    M_values = [0.001, 0.01, 0.1, 1.0]
    
    # Option 1: Plot the correct analytic p->0 formula
    print("\n[1] Generating analytic p→0 spectra...")
    plot_spectra_M_analytic(M_values, qmin=1e-4, qmax=1e1, nq=300,
                            out_png='outputs/H_spectra_analytic.png', R=1e4)
    
    # Option 2: Optional—can uncomment to run full 2D scans (slower)
    # print("[2] Generating 2D H(p,q) scan...")
    # example_scan_and_plot(out_png='outputs/H_pq.png', out_npy='outputs/Hgrid.npy', M=1.0, R=1e4)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)