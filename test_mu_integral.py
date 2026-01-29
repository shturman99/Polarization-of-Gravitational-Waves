"""Quick test of the mu integral function."""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import numpy as np
from scipy import integrate
from hijij import (
    HijijParams, mu_integral_sympy, _integrand, u
)

def numeric_mu_integral_reference(k, k1, om, om1, params):
    """Compute mu integral numerically as reference."""
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

params = HijijParams(
    k1_max=2.0,
    om1_max=2.0,
    u_reg=0.1,
    abs_tol=1e-5,
    rel_tol=1e-5,
)

test_cases = [
    (1.0, 0.5, 0.5, 0.3),
    (2.0, 1.0, 0.1, 0.2),
    (0.5, 0.2, 1.0, 0.5),
]

print("Testing mu_integral_sympy function:")
print("=" * 80)

for k, k1, om, om1 in test_cases:
    print(f"\nTest: k={k}, k1={k1}, om={om}, om1={om1}")
    
    # Compute reference (numeric)
    try:
        ref = numeric_mu_integral_reference(k, k1, om, om1, params)
        print(f"  Reference (numeric):     {ref:.10e}")
    except Exception as e:
        print(f"  Reference failed: {e}")
        ref = None
    
    # Compute with mu_integral_sympy
    try:
        result = mu_integral_sympy(k, k1, om, om1, params)
        print(f"  mu_integral_sympy result: {result:.10e}")
        
        if ref is not None and result is not None:
            rel_error = abs(result - ref) / abs(ref)
            print(f"  Relative error: {rel_error:.6e}")
            if rel_error < 1e-3:
                print(f"  ✓ PASS")
            else:
                print(f"  ⚠️  WARNING: Large discrepancy")
    except Exception as e:
        print(f"  mu_integral_sympy failed: {e}")
        result = None

print("\n" + "=" * 80)
print("Test complete")
