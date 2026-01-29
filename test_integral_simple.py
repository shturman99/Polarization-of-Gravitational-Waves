"""Test the simplified version of mu_integral_sympy."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import numpy as np
from scipy import integrate
import sympy as sp
from hijij import (
    HijijParams, eta, E
)

def test_simpler_integral():
    """Try a simpler symbolic integration to debug."""
    
    # Simple test: integrate exp(-mu^2) from -1 to 1
    mu = sp.symbols("mu")
    simple_integrand = sp.exp(-(mu**2))
    
    print("Simple integral test:")
    print(f"Integrand: {simple_integrand}")
    result = sp.integrate(simple_integrand, (mu, -1, 1))
    print(f"Result: {result}")
    print(f"Numeric: {float(result.evalf())}")
    print()
    
    # Now test with sqrt
    u_expr = sp.sqrt(1.0 + mu**2)
    integrand2 = 1.0 / u_expr**2
    print("Integral with sqrt:")
    print(f"Integrand: {integrand2}")
    result2 = sp.integrate(integrand2, (mu, -1, 1))
    print(f"Result: {result2}")
    print(f"Numeric: {float(result2.evalf())}")
    print()
    
    # Test with fractional power
    u_expr3 = sp.sqrt(1.0 + mu**2)
    integrand3 = u_expr3 ** (sp.Rational(-4, 3))
    print("Integral with u^(-4/3):")
    print(f"Integrand: {integrand3}")
    try:
        result3 = sp.integrate(integrand3, (mu, -1, 1))
        print(f"Result: {result3}")
        print(f"Numeric: {float(result3.evalf())}")
    except Exception as e:
        print(f"Integration failed: {e}")
    print()
    
    # Test the actual form
    k = 1.0
    k1 = 0.5
    params = HijijParams(u_reg=0.1, eps=1.0, Ck=1.0)
    
    u_expr4 = sp.sqrt(k**2 + k1**2 - 2.0*k*k1*mu + params.u_reg**2)
    
    eta_k1 = eta(k1, params.eps)
    eta_u = (1.0 / sp.sqrt(2.0 * sp.pi)) * params.eps ** (sp.Rational(1, 3)) * u_expr4 ** (sp.Rational(2, 3))
    
    # Just the g_u part without the kernel
    E_u = params.Ck * params.eps ** (sp.Rational(2, 3)) * u_expr4 ** (sp.Rational(-5, 3))
    g_u_simple = (2.0 * E_u) / (u_expr4**2 * eta_u)
    
    print("Actual g_u expression (simplified):")
    print(f"u_expr: {u_expr4}")
    print(f"eta_u: {eta_u}")
    print(f"g_u: {g_u_simple}")
    print()
    print("Attempting symbolic integration of g_u only...")
    try:
        result4 = sp.integrate(g_u_simple, (mu, -1, 1), timeout=2)
        print(f"Result: {result4}")
        if isinstance(result4, sp.Integral):
            print("Integration did not complete")
        else:
            print(f"Numeric: {float(result4.evalf())}")
    except Exception as e:
        print(f"Integration failed: {type(e).__name__}: {e}")

test_simpler_integral()
