"""Numerical evaluation of the Hijij integral.

This module implements a Python equivalent of the Mathematica H_ijij integral with
an optional analytic attempt for the μ integration using sympy. It provides safe
numerical handling around u ≈ 0 and k1 ≈ 0 to avoid singular denominators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import signal
from typing import Optional

import numpy as np
import sympy as sp
from scipy import integrate


_TWO_PI = 2.0 * math.pi


@dataclass
class HijijParams:
    Ck: float = 1.0
    eps: float = 1.0
    k1_max: float = 20.0
    om1_max: float = 20.0
    u_reg: float = 0.0
    use_analytic_mu: bool = True
    progress_enabled: bool = False
    progress_print_step_percent: float = 0.01
    progress_target: int = 1_000_000
    progress_max_prints: int | None = None
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    max_subdiv: int = 200
    method: str = "scipy_nquad"


class ProgressTracker:
    """Best-effort progress tracker for adaptive quadrature.

    Since adaptive integrators do not expose percent completion, we proxy progress
    based on the total number of integrand evaluations. Each evaluation increments
    a counter, and a print occurs when the percent threshold advances by
    progress_print_step_percent.
    """

    def __init__(self, params: HijijParams) -> None:
        self.enabled = params.progress_enabled
        self.step_percent = params.progress_print_step_percent
        self.target = max(1, params.progress_target)
        self.max_prints = params.progress_max_prints
        self.start_time = time.monotonic()
        self.count = 0
        self.next_percent = self.step_percent
        self.prints = 0

    def update(self, k1: float, mu: float, om1: float, u_val: float, integrand: float) -> None:
        if not self.enabled:
            return
        self.count += 1
        percent = (self.count / self.target) * 100.0
        if percent + 1e-12 < self.next_percent:
            return
        if self.max_prints is not None and self.prints >= self.max_prints:
            return
        elapsed = time.monotonic() - self.start_time
        print(
            f"{percent:.4f}% | k1={k1:.4e} mu={mu:.4e} om1={om1:.4e} "
            f"u={u_val:.4e} f={integrand:.4e} elapsed={elapsed:.2f}s"
        )
        self.prints += 1
        while self.next_percent <= percent + 1e-12:
            self.next_percent += self.step_percent


def eta(k: float, eps: float) -> float:
    return (1.0 / math.sqrt(_TWO_PI)) * eps ** (1.0 / 3.0) * k ** (2.0 / 3.0)


def E(k: float, Ck: float, eps: float) -> float:
    return Ck * eps ** (2.0 / 3.0) * k ** (-5.0 / 3.0)


def g(k: float, om: float, Ck: float, eps: float) -> float:
    eta_val = eta(k, eps)
    prefactor = (2.0 * E(k, Ck, eps)) / (k**2 * eta_val)
    return prefactor * math.exp(-(om**2) / (math.pi * eta_val**2))


def u(k: float, k1: float, mu: float, u_reg: float) -> float:
    val = k**2 + k1**2 - 2.0 * k * k1 * mu + u_reg**2
    if val < 0.0:
        val = 0.0
    return math.sqrt(val)


def kernel(k: float, k1: float, u_val: float) -> float:
    eps = 1e-12
    k1_sq = max(k1**2, eps)
    u_sq = max(u_val**2, eps)
    return (
        27.0 / 6.0
        + k**4 / (12.0 * k1_sq * u_sq)
        + k1_sq / (12.0 * u_sq)
        + u_sq / (12.0 * k1_sq)
        - k**2 / (6.0 * u_sq)
        - k**2 / (6.0 * k1_sq)
    )


class _Timeout:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._old_handler = None

    def __enter__(self) -> None:
        if hasattr(signal, "SIGALRM"):
            self._old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)

    @staticmethod
    def _handle_timeout(signum, frame) -> None:
        raise TimeoutError("Sympy integration timed out")


def mu_integral_sympy(
    k: float,
    k1: float,
    om: float,
    om1: float,
    params: HijijParams,
    timeout_s: float = 0.5,
) -> Optional[float]:
    """Attempt analytic μ integration using sympy.

    Returns a float on success, or None on failure/timeout.
    """

    mu = sp.symbols("mu")
    u_expr = sp.sqrt(k**2 + k1**2 - 2.0 * k * k1 * mu + params.u_reg**2)

    eta_k1 = eta(k1, params.eps)
    eta_u = (1.0 / sp.sqrt(2.0 * sp.pi)) * params.eps ** (sp.Rational(1, 3)) * u_expr ** (sp.Rational(2, 3))

    g_k1 = (2.0 * E(k1, params.Ck, params.eps)) / (k1**2 * eta_k1)
    g_k1 *= sp.exp(-(om1**2) / (sp.pi * eta_k1**2))

    g_u = (2.0 * (params.Ck * params.eps ** (sp.Rational(2, 3)) * u_expr ** (sp.Rational(-5, 3)))) / (
        u_expr**2 * eta_u
    )
    g_u *= sp.exp(-((om - om1) ** 2) / (sp.pi * eta_u**2))

    kernel_expr = (
        27.0 / 6.0
        + k**4 / (12.0 * k1**2 * u_expr**2)
        + k1**2 / (12.0 * u_expr**2)
        + u_expr**2 / (12.0 * k1**2)
        - k**2 / (6.0 * u_expr**2)
        - k**2 / (6.0 * k1**2)
    )

    integrand = g_k1 * g_u * kernel_expr

    try:
        with _Timeout(timeout_s):
            result = sp.integrate(integrand, (mu, -1, 1))
    except (TimeoutError, Exception):
        return None

    if isinstance(result, sp.Integral):
        return None

    try:
        return float(result.evalf())
    except (TypeError, ValueError):
        return None


def _safe_u_for_denominator(u_val: float, u_reg: float) -> float:
    if u_val == 0.0 and u_reg == 0.0:
        return 1e-12
    return u_val


def _integrand(
    k: float,
    om: float,
    k1: float,
    mu: float,
    om1: float,
    params: HijijParams,
    tracker: Optional[ProgressTracker],
) -> float:
    k1_safe = max(k1, 1e-12)
    u_val = u(k, k1_safe, mu, params.u_reg)
    u_val = _safe_u_for_denominator(u_val, params.u_reg)
    g1 = g(k1_safe, om1, params.Ck, params.eps)
    g2 = g(u_val, om - om1, params.Ck, params.eps)
    kern = kernel(k, k1_safe, u_val)
    value = k1_safe**2 * g1 * g2 * kern
    if tracker is not None:
        tracker.update(k1_safe, mu, om1, u_val, value)
    return value


def hijij(k: float, om: float, params: HijijParams = HijijParams()) -> float:
    """Compute the Hijij integral for given k and ω.

    Numerical safeguards:
    - The k1 integration starts at 1e-12 to avoid division by zero in the kernel.
    - u is regularized with params.u_reg; if u_reg=0 and u hits 0, a small
      epsilon is used in denominators.
    """

    tracker = ProgressTracker(params) if params.progress_enabled else None

    k1_min = 1e-12
    k1_max = params.k1_max
    om1_min = -params.om1_max
    om1_max = params.om1_max

    prefactor = (_TWO_PI) ** (-7)

    analytic_ok = False
    if params.use_analytic_mu:
        sample_k1 = 0.5 * (k1_min + (k1_max if np.isfinite(k1_max) else 1.0))
        sample_om1 = 0.0
        analytic_ok = mu_integral_sympy(k, sample_k1, om, sample_om1, params) is not None

    if analytic_ok:
        def integrand_2d(om1: float, k1: float) -> float:
            k1_safe = max(k1, 1e-12)
            mu_val = mu_integral_sympy(k, k1, om, om1, params)
            if mu_val is None:
                inner = integrate.quad(
                    lambda mu: _integrand(k, om, k1_safe, mu, om1, params, tracker) / (k1_safe**2),
                    -1.0,
                    1.0,
                    epsabs=params.abs_tol,
                    epsrel=params.rel_tol,
                    limit=params.max_subdiv,
                )[0]
            else:
                inner = mu_val
                if tracker is not None:
                    tracker.update(k1_safe, 0.0, om1, u(k, k1_safe, 0.0, params.u_reg), inner)
            return k1_safe**2 * inner

        result = integrate.nquad(
            integrand_2d,
            [(om1_min, om1_max), (k1_min, k1_max)],
            opts=[{"epsabs": params.abs_tol, "epsrel": params.rel_tol, "limit": params.max_subdiv}]
            * 2,
        )[0]
        return prefactor * result

    if params.method == "scipy_quad":
        def inner_mu(mu: float, om1: float, k1: float) -> float:
            return _integrand(k, om, k1, mu, om1, params, tracker)

        def inner_om1(om1: float, k1: float) -> float:
            return integrate.quad(
                lambda mu: inner_mu(mu, om1, k1),
                -1.0,
                1.0,
                epsabs=params.abs_tol,
                epsrel=params.rel_tol,
                limit=params.max_subdiv,
            )[0]

        def inner_k1(k1: float) -> float:
            return integrate.quad(
                lambda om1: inner_om1(om1, k1),
                om1_min,
                om1_max,
                epsabs=params.abs_tol,
                epsrel=params.rel_tol,
                limit=params.max_subdiv,
            )[0]

        result = integrate.quad(
            inner_k1,
            k1_min,
            k1_max,
            epsabs=params.abs_tol,
            epsrel=params.rel_tol,
            limit=params.max_subdiv,
        )[0]
    else:
        result = integrate.nquad(
            lambda om1, mu, k1: _integrand(k, om, k1, mu, om1, params, tracker),
            [(om1_min, om1_max), (-1.0, 1.0), (k1_min, k1_max)],
            opts=[{"epsabs": params.abs_tol, "epsrel": params.rel_tol, "limit": params.max_subdiv}]
            * 3,
        )[0]

    return prefactor * result
