#!/usr/bin/env python3
"""
Numerical evaluator for H(p,q) from derivation.tex.

Usage:
  python src/compute_H.py        
Functions:
  H_pq(p, q, M=1.0, R=1e6, k0=1.0, epsabs=1e-8, epsrel=1e-6)
"""
import os
import numpy as np
from scipy import integrate, special
import matplotlib.pyplot as plt
import mpmath as mp

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
    pref = y**0.75 * s**(-0.5) * x**(3/4)
    br = kernel_bracket(p, x, y)
    expo = np.exp(-2.0 * x * y / s * (q**2) / (M**2))
    er = special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(s)))
    return pref * br * expo * er


def g_decaying(z):
    """
    Dimensionless temporal kernel g(z) = e^{iz} (-iz)^{-5/3} Gamma(1/3,-iz).
    Accepts complex or real `z` and returns complex values.
    """
    z = np.asarray(z, dtype=complex)

    def _g_scalar(zz):
        arg = -1j * zz
        # upper incomplete gamma for complex argument via mpmath
        gam_up = mp.gammainc(1.0 / 3.0, arg, mp.inf)
        return complex(mp.e ** (1j * zz) * ((-1j * zz) ** (-5.0 / 3.0)) * gam_up)

    if z.ndim == 0:
        return _g_scalar(complex(z))
    return np.vectorize(_g_scalar, otypes=[complex])(z)


def integrand_y_decaying(y, x, p, q, M, epsabs=1e-6, epsrel=1e-4):
    """Integrand over y for the decaying-spectrum temporal model.
    This replaces the exp*erfc temporal factor by a convolution of `g` functions
    following the derivation in `derivation.tex` ("Making Decaying Spectrum Dimensionless").
    """
    s = x + y
    if s <= 0:
        return 0.0
    pref = y**0.75 * s**(-0.5) * x**(3/4)
    br = kernel_bracket(p, x, y)

    # convolution over Q1 (dimensionless frequency variable Q1 = omega1/k0)
    # mapping to g-arguments: z1 = Q1 * x^{1/2} / M, z2 = (q - Q1) * y^{1/2} / M
    def conv_integrand(Q1):
        z1 = Q1 * np.sqrt(x) / M
        z2 = (q - Q1) * np.sqrt(y) / M
        g1 = g_decaying(z1)
        g2 = g_decaying(z2)
        return (g1 * g2).real

    # integrate over all Q1 (allow negative and positive frequencies)
    try:
        conv_val, conv_err = integrate.quad(conv_integrand, -np.inf, np.inf,
                                            epsabs=epsabs, epsrel=epsrel, limit=200)
    except Exception:
        conv_val = 0.0

    return pref * br * conv_val


def inner_integral_decaying(x, p, q, M, R, epsabs, epsrel):
    tilde_k1 = x**(-0.75)
    u_min = max(abs(tilde_k1 - p), 1.0)
    u_max = min(tilde_k1 + p, R**(0.75))
    if not (u_min < u_max):
        return 0.0
    y_min = u_max**(-4.0/3.0)
    y_max = u_min**(-4.0/3.0)
    if not (y_min < y_max):
        return 0.0
    val, err = integrate.quad(integrand_y_decaying, y_min, y_max,
                              args=(x, p, q, M, epsabs, epsrel),
                              epsabs=epsabs, epsrel=epsrel, limit=200)
    return val


def H_pq_decaying(p, q, M=1.0, R=1e6, k0=1.0, epsabs=1e-6, epsrel=1e-4):
    """Compute H(p,q) for the decaying turbulence temporal model.

    This function implements the "Making Decaying Spectrum Dimensionless"
    derivation: the temporal dependence is given by a convolution of the
    dimensionless `g` kernels. The spatial kernel and prefactor follow the
    stationary case so results are comparable up to normalization.
    """
    p_floor = max(p, 1e-10)
    x_lo = R**(-1)
    x_hi = 1.0

    def outer_x(xx):
        return inner_integral_decaying(xx, p_floor, q, M, R, epsabs, epsrel)

    I, err_x = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=200)

    pref = 3.0 * (M**3) * (k0**(-4)) / (256.0 * (2.0 * np.pi)**1.5 * p_floor)
    return pref * I


def plot_p0_spectra_params(M_list, qmin=1e-4, qmax=1e1, nq=200,
                           R=1e4, k0=1.0, out_png='outputs/H_p0_params.png'):
    """Plot p->0 analytic spectra for given M_list and include parameters in title."""
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    plt.figure(figsize=(8,6))
    for M in M_list:
        Hvals = np.array([H_k0_analytic(q, M=M, k0=k0, R=R) for q in qs])
        plt.loglog(qs, (qs * Hvals)**0.5, label=f'M={M}')
    plt.xlabel('q = ω/k0')
    plt.ylabel('h_c (p->0 analytic)')
    title = f'p->0 analytic spectra; k0={k0}, R={R}, nq={nq}'
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    out_png_base, out_png_ext = os.path.splitext(out_png)
    mlist_str = '-'.join([f"{M:.2e}".replace('+','p').replace('-','m') for M in M_list])
    out_png = f"{out_png_base}_Ms{mlist_str}_R{R}{out_png_ext}"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png}')


def scan_and_plot_grid(Hfunc, M, R=1e4, k0=1.0, ps=None, qs=None,
                       out_png='outputs/H_pq_scan.png', out_npy='outputs/Hgrid.npy'):
    """Compute Hgrid using Hfunc(p,q,...) and save 2D plot + npy. Hfunc must accept (p,q,M,R,k0).
    """
    if ps is None:
        ps = np.logspace(-2, 1, 40)
    if qs is None:
        qs = np.logspace(-2, 1, 40)
    Hgrid = np.zeros((qs.size, ps.size))
    for i, q in enumerate(qs):
        for j, p in enumerate(ps):
            try:
                Hgrid[i, j] = Hfunc(p, q, M=M, R=R, k0=k0)
            except Exception:
                Hgrid[i, j] = np.nan

    mstr = f"{M:.2e}".replace('+','p').replace('-', 'm')
    rstr = f"{R:.2e}".replace('+','p').replace('-', 'm')
    out_npy_base, out_npy_ext = os.path.splitext(out_npy)
    out_png_base, out_png_ext = os.path.splitext(out_png)
    out_npy = f"{out_npy_base}_M{mstr}_R{rstr}{out_npy_ext}"
    out_png = f"{out_png_base}_M{mstr}_R{rstr}{out_png_ext}"
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, {"ps": ps, "qs": qs, "H": Hgrid})
    plt.figure(figsize=(7,5))
    plt.pcolormesh(ps, qs, Hgrid, shading='auto', cmap='viridis')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('p = k/k0')
    plt.ylabel('q = ω/k0')
    mode = 'decaying' if Hfunc is H_pq_decaying else 'stationary'
    plt.title(f'H(p,q) 2D scan ({mode}); M={M}, R={R}, k0={k0}')
    plt.colorbar(label='H')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png} and {out_npy}')


def plot_scans_for_M_list(M_list, R=1e4, k0=1.0, ps=None, qs=None):
    """Generate 2D scans for each M for stationary and decaying models (coarse grids)."""
    for M in M_list:
        scan_and_plot_grid(H_pq, M, R=R, k0=k0, ps=ps, qs=qs,
                           out_png='outputs/H_pq_stationary.png', out_npy='outputs/Hgrid_stationary.npy')
        scan_and_plot_grid(H_pq_decaying, M, R=R, k0=k0, ps=ps, qs=qs,
                           out_png='outputs/H_pq_decaying.png', out_npy='outputs/Hgrid_decaying.npy')


def plot_Hqq_decaying(M_list, qmin=1e-2, qmax=10.0, nq=80, R=1e4, k0=1.0,
                      out_png='outputs/Hqq_decaying.png'):
    """Plot spectra for H(q,q) (p=q) using the decaying turbulence model for each M."""
    qs = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    plt.figure(figsize=(8,6))
    for M in M_list:
        Hvals = np.zeros_like(qs)
        for i, q in enumerate(qs):
            try:
                Hvals[i] = H_pq_decaying(q, q, M=M, R=R, k0=k0)
            except Exception:
                Hvals[i] = np.nan
        plt.loglog(qs, (qs * Hvals)**0.5, label=f'M={M}')
    plt.xlabel('q = ω/k0 (and p=q)')
    plt.ylabel('h_c (decaying, p=q)')
    plt.title(f'H(q,q) spectra (decaying); R={R}, k0={k0}, nq={nq}')
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    mlist_str = '-'.join([f"{M:.2e}".replace('+','p').replace('-','m') for M in M_list])
    out_png_base, out_png_ext = os.path.splitext(out_png)
    out_png = f"{out_png_base}_Ms{mlist_str}_R{R}{out_png_ext}"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png}')

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
    Compute H(p,q)
    - p: dimensionless wavenumber p = k/k0 
    - q: dimensionless frequency q = omega/k0
    - M, R, k0: parameters from derivation 
    Returns: float H(p,q)
    Note: For extremely small p we use a small floor to avoid division by zero.
    """
    p_floor = max(p, 1e-10)
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

    # include M and R in output filenames for traceability (2-digit scientific notation)
    mstr = f"{M:.2e}".replace('+', 'p').replace('-', 'm')
    rstr = f"{R:.2e}".replace('+', 'p').replace('-', 'm')
    out_npy_base, out_npy_ext = os.path.splitext(out_npy)
    out_png_base, out_png_ext = os.path.splitext(out_png)
    out_npy = f"{out_npy_base}_M{mstr}_R{rstr}{out_npy_ext}"
    out_png = f"{out_png_base}_M{mstr}_R{rstr}{out_png_ext}"
    
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    # save as a numpy file containing a dict for easy loading
    np.save(out_npy, {"ps": ps, "qs": qs, "H": Hgrid})
    plt.figure(figsize=(7,5))
    plt.pcolormesh(ps, qs, Hgrid, shading='auto', cmap='viridis')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'p = k/k_0')
    plt.ylabel(r'q = $\omega/k_0$')
    plt.title('H(p,q) ')
    plt.colorbar(label='H')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Wrote {out_png} and {out_npy}")

def H_k0_analytic(q, M=1.0, k0=1.0, R=1e4):
    """
    Analytic p->0 limit: H_ijij(0, q) from derivation.
    
    H_ijij(0, q) ≃ (7 M^3 k_0^{-4}) / (16 \pi^{3/2})
                   * 
                   \int_1^R^{-1} dx x^{11/4} exp(- \bar{q}^2 x) erfc(- \bar{q} x^{1/2})
    
    where  \bar{q} = q/M (dimensionless frequency scaled by M).
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
    # include M list in filename (2-digit scientific notation)
    mlist_str = '-'.join([f"{M:.2e}".replace('+', 'p').replace('-', 'm') for M in M_list])
    out_png_base, out_png_ext = os.path.splitext(out_png)
    out_png = f"{out_png_base}_Ms{mlist_str}{out_png_ext}"
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
    plt.title('Analytic p->0 spectra for various Mach numbers', fontsize=12)
    plt.ylim(bottom=1e-21)
    plt.legend(fontsize=10)
    # no grid per request
    plt.tight_layout()
    # include M list and R in filename (2-digit scientific notation)
    mlist_str = '-'.join([f"{M:.2e}".replace('+', 'p').replace('-', 'm') for M in M_list])
    rstr = f"{R:.2e}".replace('+', 'p').replace('-', 'm')
    out_png_base, out_png_ext = os.path.splitext(out_png)
    out_png = f"{out_png_base}_Ms{mlist_str}_R{rstr}{out_png_ext}"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Wrote {out_png}')


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("H(p,q) Numerical Evaluator")
    print("=" * 60)
    
    M_values = [0.001, 0.01, 0.1, 1.0]

    # 1) p -> 0 analytic spectra (log-log)
    print("\n[1] Generating analytic p->0 spectra (log-log)...")
    plot_p0_spectra_params(M_values, qmin=1e-4, qmax=1e1, nq=200, R=1e4, k0=1.0,
                           out_png='outputs/H_p0_params.png')

    # 2) 2D scans H(p,q) for stationary and decaying cases (coarse grids, log-log axes)
    print("[2] Generating 2D scans for stationary and decaying models (coarse)...")
    ps = np.logspace(-2, 1, 30)
    qs = np.logspace(-2, 1, 30)
    # limit to two representative Mach numbers to keep runtime reasonable
    scan_Ms = [0.01, 0.1, 1.0]
    plot_scans_for_M_list(scan_Ms, R=1e4, k0=1.0, ps=ps, qs=qs)

    # 3) Spectra for H(q,q) (p=q) for decaying turbulence (log-log)
    print("[3] Generating H(q,q) spectra (decaying, p=q) (coarse)...")
    plot_Hqq_decaying(scan_Ms, qmin=1e-2, qmax=10.0, nq=60, R=1e4, k0=1.0,
                      out_png='outputs/Hqq_decaying.png')

    print("\n" + "=" * 60)
    print("Requested plots generated (coarse).")
    print("Files are in the outputs/ directory. Re-run with finer grids if desired.")
    print("=" * 60)
