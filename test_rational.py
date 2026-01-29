"""Test using rational coefficients instead of floats."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import sympy as sp
from fractions import Fraction

def test_with_rationals():
    """Test integration using symbolic rather than floating point coefficients."""
    
    # Use exact rationals
    k = sp.Rational(1)
    k1 = sp.Rational(1, 2)
    om = sp.Rational(1, 2)
    om1 = sp.Rational(3, 10)
    u_reg = sp.Rational(1, 10)
    eps = sp.Rational(1)
    Ck = sp.Rational(1)
    
    mu = sp.symbols("mu")
    u_expr = sp.sqrt(k**2 + k1**2 - 2*k*k1*mu + u_reg**2)
    
    # eta_k1
    eta_k1 = (1 / sp.sqrt(2*sp.pi)) * eps**(sp.Rational(1, 3)) * k1**(sp.Rational(2, 3))
    
    # E(k1)
    E_k1 = Ck * eps**(sp.Rational(2, 3)) * k1**(-sp.Rational(5, 3))
    
    # eta_u
    eta_u = (1 / sp.sqrt(2*sp.pi)) * eps**(sp.Rational(1, 3)) * u_expr**(sp.Rational(2, 3))
    
    # g_k1
    g_k1 = (2 * E_k1) / (k1**2 * eta_k1)
    g_k1 *= sp.exp(-(om1**2) / (sp.pi * eta_k1**2))
    
    # E(u)
    E_u = Ck * eps**(sp.Rational(2, 3)) * u_expr**(-sp.Rational(5, 3))
    
    # g_u
    g_u = (2 * E_u) / (u_expr**2 * eta_u)
    g_u *= sp.exp(-((om - om1)**2) / (sp.pi * eta_u**2))
    
    integrand = g_k1 * g_u
    
    print("Testing integration with rational coefficients:")
    print(f"Attempting to integrate from -1 to 1...")
    try:
        result = sp.integrate(integrand, (mu, -1, 1))
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        if isinstance(result, sp.Integral):
            print("Result is still Integral - integration failed")
            return None
        else:
            try:
                numeric_result = float(result.evalf())
                print(f"Numeric result: {numeric_result}")
                return numeric_result
            except:
                print(f"Could not convert to float")
                return None
    except Exception as e:
        print(f"Integration failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

result = test_with_rationals()
