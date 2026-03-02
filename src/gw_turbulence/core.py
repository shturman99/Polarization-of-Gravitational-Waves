"""Numerical kernels for stationary and decaying turbulence models."""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy import integrate, special


def kernel_bracket(p: float, x: float, y: float) -> float:
    xp32 = x**1.5
    yp32 = y**1.5
    return (
        27.0
        - p**2 * xp32
        - p**2 * yp32
        + 0.5 * p**4 * xp32 * yp32
        + 0.5 * x**(-1.5) * yp32
        + 0.5 * y**(-1.5) * xp32
    )


def integrand_y(y: float, x: float, p: float, q: float, M: float) -> float:
    s = x + y
    if s <= 0:
        return 0.0
    pref = y**0.75 * s**(-0.5) * x**0.75
    bracket = kernel_bracket(p, x, y)
    expo = np.exp(-2.0 * x * y / s * q**2 / M**2)
    erfc_factor = special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(s)))
    return pref * bracket * expo * erfc_factor


def g_decaying(z):
    """Dimensionless temporal kernel for the decaying-spectrum model."""
    if np.isscalar(z):
        return _g_decaying_scalar(complex(z))
    array = np.asarray(z)
    values = np.empty(array.shape, dtype=complex)
    for index, value in np.ndenumerate(array):
        values[index] = _g_decaying_scalar(complex(value))
    return values


@lru_cache(maxsize=1024)
def _g_decaying_scalar(z: complex) -> complex:
    arg = -1j * z
    gamma_upper = mp.gammainc(1.0 / 3.0, arg, mp.inf)
    return complex(mp.e ** (1j * z) * ((-1j * z) ** (-5.0 / 3.0)) * gamma_upper)


def integrand_y_decaying(
    y: float,
    x: float,
    p: float,
    q: float,
    M: float,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
) -> float:
    s = x + y
    if s <= 0:
        return 0.0
    pref = y**0.75 * s**(-0.5) * x**0.75
    bracket = kernel_bracket(p, x, y)
    if abs(bracket) < 1e-12 or abs(pref) < 1e-14:
        return 0.0

    scale = np.sqrt(x * y) / M if M > 0 else 1.0
    q_cutoff = max(15.0 * scale, 50.0)

    def conv_integrand(q1: float) -> float:
        z1 = q1 * np.sqrt(x) / M
        z2 = (q - q1) * np.sqrt(y) / M
        return (g_decaying(z1) * g_decaying(z2)).real

    try:
        conv_val, _ = integrate.quad(
            conv_integrand,
            -q_cutoff,
            q_cutoff,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=100,
        )
    except Exception:
        conv_val = 0.0

    return pref * bracket * conv_val


def _integration_bounds(x: float, p: float, R: float) -> tuple[float, float] | None:
    tilde_k1 = x**(-0.75)
    u_min = max(abs(tilde_k1 - p), 1.0)
    u_max = min(tilde_k1 + p, R**0.75)
    if not (u_min < u_max):
        return None
    y_min = u_max ** (-4.0 / 3.0)
    y_max = u_min ** (-4.0 / 3.0)
    if not (y_min < y_max):
        return None
    return y_min, y_max


def inner_integral(
    x: float,
    p: float,
    q: float,
    M: float,
    R: float,
    epsabs: float,
    epsrel: float,
) -> float:
    bounds = _integration_bounds(x, p, R)
    if bounds is None:
        return 0.0
    y_min, y_max = bounds
    value, _ = integrate.quad(
        integrand_y,
        y_min,
        y_max,
        args=(x, p, q, M),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return value


def inner_integral_decaying(
    x: float,
    p: float,
    q: float,
    M: float,
    R: float,
    epsabs: float,
    epsrel: float,
) -> float:
    bounds = _integration_bounds(x, p, R)
    if bounds is None:
        return 0.0
    y_min, y_max = bounds
    value, _ = integrate.quad(
        integrand_y_decaying,
        y_min,
        y_max,
        args=(x, p, q, M, epsabs, epsrel),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=100,
    )
    return value


def _h_prefactor(p: float, M: float, k0: float) -> float:
    p_floor = max(p, 1e-10)
    return 3.0 * M**3 * k0**(-4) / (256.0 * (2.0 * np.pi) ** 1.5 * p_floor)


def H_pq(
    p: float,
    q: float,
    M: float = 1.0,
    R: float = 1e6,
    k0: float = 1.0,
    epsabs: float = 1e-8,
    epsrel: float = 1e-6,
) -> float:
    p_floor = max(p, 1e-10)
    x_lo = R**(-1)
    x_hi = 1.0

    def outer_x(xx: float) -> float:
        return inner_integral(xx, p_floor, q, M, R, epsabs, epsrel)

    value, _ = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=200)
    return _h_prefactor(p_floor, M, k0) * value


def H_pq_decaying(
    p: float,
    q: float,
    M: float = 1.0,
    R: float = 1e6,
    k0: float = 1.0,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
) -> float:
    p_floor = max(p, 1e-10)
    x_lo = R**(-1)
    x_hi = 1.0

    def outer_x(xx: float) -> float:
        return inner_integral_decaying(xx, p_floor, q, M, R, epsabs, epsrel)

    value, _ = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=80)
    return _h_prefactor(p_floor, M, k0) * value


def H_pq_decaying_grid(
    ps,
    qs,
    M: float = 1.0,
    R: float = 1e4,
    k0: float = 1.0,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
    verbose: bool = False,
) -> np.ndarray:
    ps = np.atleast_1d(ps)
    qs = np.atleast_1d(qs)
    grid = np.zeros((len(qs), len(ps)))

    for i, q in enumerate(qs):
        if verbose:
            print(f"  qs: {i + 1}/{len(qs)} (q={q:.4e})")
        for j, p in enumerate(ps):
            try:
                grid[i, j] = H_pq_decaying(
                    p,
                    q,
                    M=M,
                    R=R,
                    k0=k0,
                    epsabs=epsabs,
                    epsrel=epsrel,
                )
            except Exception as exc:
                if verbose:
                    print(f"    Error at (p,q)=({p:.4e},{q:.4e}): {exc}")
                grid[i, j] = np.nan

    return grid


def H_k0_analytic(q, M: float = 1.0, k0: float = 1.0, R: float = 1e4):
    q_array = np.atleast_1d(q)
    output = np.zeros_like(q_array, dtype=float)

    x_lo = 1.0
    x_hi = R**(-1)
    prefactor = 7.0 * M**3 * k0**(-4) / (16.0 * np.pi**1.5)

    for index, q_value in enumerate(q_array):
        if q_value <= 0:
            output[index] = np.nan
            continue

        q_bar = q_value / M

        def integrand_x(x: float) -> float:
            return x ** (11.0 / 4.0) * np.exp(-q_bar**2 * x) * special.erfc(-q_bar * np.sqrt(x))

        try:
            integral, _ = integrate.quad(
                integrand_x,
                x_hi,
                x_lo,
                epsabs=1e-10,
                epsrel=1e-8,
                limit=300,
            )
        except Exception:
            integral = 0.0
        output[index] = prefactor * integral

    if np.isscalar(q):
        return output[0]
    return output
