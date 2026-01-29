"""Test with the actual g_u expression including exponential."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import sympy as sp
from hijij import HijijParams, eta, E

def test_with_exponential():
    """Test integration with exponential included."""
    
    k = 1.0
    k1 = 0.5
    om = 0.5
    om1 = 0.3
    params = HijijParams(u_reg=0.1, eps=1.0, Ck=1.0)
    
    mu = sp.symbols("mu")
    u_expr = sp.sqrt(k**2 + k1**2 - 2.0*k*k1*mu + params.u_reg**2)
    
    eta_k1 = eta(k1, params.eps)
    eta_u = (1.0 / sp.sqrt(2.0 * sp.pi)) * params.eps ** (sp.Rational(1, 3)) * u_expr ** (sp.Rational(2, 3))
    
    # g_k1 (independent of mu)
    g_k1 = (2.0 * E(k1, params.Ck, params.eps)) / (k1**2 * eta_k1)
    g_k1 *= sp.exp(-(om1**2) / (sp.pi * eta_k1**2))
    
    # g_u (dependent on mu)
    E_u = params.Ck * params.eps ** (sp.Rational(2, 3)) * u_expr ** (sp.Rational(-5, 3))
    g_u = (2.0 * E_u) / (u_expr**2 * eta_u)
    g_u *= sp.exp(-((om - om1) ** 2) / (sp.pi * eta_u**2))
    
    # Just the product g_k1 * g_u
    integrand = g_k1 * g_u
    
    print("Testing integration with exponential:")
    print(f"Integrand structure: g_k1 * g_u")
    print(f"g_k1 (numeric): {float(g_k1.evalf())}")
    print()
    
    print("Attempting to integrate g_k1 * g_u from -1 to 1...")
    try:
        result = sp.integrate(integrand, (mu, -1, 1))
        print(f"Result: {result}")
        if isinstance(result, sp.Integral):
            print("Result is still Integral - integration failed")
            numeric_result = None
        else:
            numeric_result = float(result.evalf())
            print(f"Numeric result: {numeric_result}")
        return numeric_result
    except Exception as e:
        print(f"Integration failed: {type(e).__name__}: {e}")
        return None

result = test_with_exponential()
