"""Debug script to compare analytic and numeric mu integrals."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import numpy as np
from scipy import integrate
import sympy as sp
from hijij import (
    HijijParams, eta, E, g, u, kernel,
    mu_integral_sympy, _integrand
)

def numeric_mu_integral(k, k1, om, om1, params):
    """Compute mu integral numerically."""
    def integrand_mu(mu):
        return _integrand(k, om, k1, mu, om1, params, None) / (k1**2)
    
    result, _ = integrate.quad(
        integrand_mu,
        -1.0, 1.0,
        epsabs=params.abs_tol,
        epsrel=params.rel_tol,
        limit=params.max_subdiv,
    )
    return result

def manual_analytic_mu(k, k1, om, om1, params):
    """Manually construct and integrate the analytic mu integral."""
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
    
    print(f"\n=== Computing analytic mu integral for k={k}, k1={k1}, om={om}, om1={om1} ===")
    print(f"Integrand (symbolic):\n{integrand}")
    print(f"\nAttempting symbolic integration...")
    
    try:
        result = sp.integrate(integrand, (mu, -1, 1))
        print(f"Symbolic result: {result}")
        if isinstance(result, sp.Integral):
            print("Result is still an Integral - integration failed")
            return None
        numeric = float(result.evalf())
        print(f"Numeric evaluation: {numeric}")
        return numeric
    except Exception as e:
        print(f"Error during integration: {e}")
        return None

# Test parameters
params = HijijParams(
    k1_max=2.0,
    om1_max=2.0,
    u_reg=0.1,
    abs_tol=1e-4,
    rel_tol=1e-4,
    use_analytic_mu=True,
)

k = 1.0
k1 = 0.5
om = 0.5
om1 = 0.3

print("=" * 80)
print("Testing mu integral consistency")
print("=" * 80)

print(f"\nParameters: k={k}, k1={k1}, om={om}, om1={om1}")
print(f"u_reg={params.u_reg}, eps={params.eps}, Ck={params.Ck}")

# Test numeric
numeric_result = numeric_mu_integral(k, k1, om, om1, params)
print(f"\nNumeric mu integral: {numeric_result}")

# Test via mu_integral_sympy
sympy_result = mu_integral_sympy(k, k1, om, om1, params, timeout_s=2.0)
print(f"\nmu_integral_sympy result: {sympy_result}")

# Manual analytic
manual_result = manual_analytic_mu(k, k1, om, om1, params)
print(f"\nManual analytic result: {manual_result}")

if sympy_result is not None and numeric_result is not None:
    rel_error = abs(sympy_result - numeric_result) / abs(numeric_result)
    print(f"\nRelative error (sympy vs numeric): {rel_error:.6e}")
    if rel_error > 1e-3:
        print("⚠️  WARNING: Large discrepancy between analytic and numeric!")
    else:
        print("✓ Good agreement between analytic and numeric")
