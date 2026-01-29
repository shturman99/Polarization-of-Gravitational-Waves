"""Verify the corrected mu integral implementation."""

import sys
sys.path.insert(0, 'src')

from hijij import HijijParams, mu_integral_sympy, _integrand, u
from scipy import integrate
import numpy as np

def numeric_mu_integral_reference(k, k1, om, om1, params):
    """Compute mu integral numerically for verification."""
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

print("=" * 80)
print("VERIFICATION: Correctness of mu_integral_sympy implementation")
print("=" * 80)

print("""
MATHEMATICAL BASIS:
According to derivation.tex, the H_ijij integral is:

    H_{ijij} = (1/(2π)^7) ∫ dk₁ k₁² ∫ dμ ∫ dω₁ g(k₁,ω₁) g(u,ω-ω₁) 𝒦(k,k₁,u)

where:
    - u = √(k² + k₁² - 2kk₁μ) depends on μ
    - g(k,ω) = 2E(k)/(k²ηₖ) exp(-ω²/(πηₖ²))
    - E(k) = Cₖ ε^(2/3) k^(-5/3)
    - ηₖ = (1/√(2π)) ε^(1/3) k^(2/3)
    - 𝒦 is the kernel term depending on k, k₁, u

IMPLEMENTATION DETAILS:
The mu_integral_sympy function:
1. Takes k, k1, om, om1 as parameters (all treated as constants during integration)
2. Constructs the integrand: g(k₁,ω₁) g(u,ω-ω₁) 𝒦(k,k₁,u) symbolically
3. Does NOT include the k₁² Jacobian factor (it's applied later)
4. Converts to numerical function using sp.lambdify
5. Integrates numerically from μ ∈ [-1, 1] using scipy.integrate.quad

The integrand_2d in hijij function then:
- Calls mu_integral_sympy to get ∫ g(k₁,ω₁) g(u,ω-ω₁) 𝒦 dμ
- Multiplies by k₁² to get the full μ-integrated expression
- This is then integrated over ω₁ and k₁

VERIFICATION:
""")

params = HijijParams(
    u_reg=0.1,
    abs_tol=1e-5,
    rel_tol=1e-5,
    Ck=1.0,
    eps=1.0,
)

test_cases = [
    (1.0, 0.5, 0.5, 0.3),
    (2.0, 1.0, 0.1, 0.2),
]

all_pass = True
for k, k1, om, om1 in test_cases:
    print(f"\nTest case: k={k}, k1={k1}, ω={om}, ω₁={om1}")
    
    # Reference from direct numerical integration
    ref = numeric_mu_integral_reference(k, k1, om, om1, params)
    print(f"  Reference (scipy.quad): {ref:.10e}")
    
    # Our implementation
    result = mu_integral_sympy(k, k1, om, om1, params)
    print(f"  mu_integral_sympy:      {result:.10e}")
    
    if result is not None:
        rel_error = abs(result - ref) / (abs(ref) + 1e-15)
        print(f"  Relative error:         {rel_error:.6e}")
        
        if rel_error < 1e-3:
            print(f"  ✓ PASS (error < 0.1%)")
        else:
            print(f"  ⚠️  MARGINAL (error > 0.1%)")
            all_pass = False
    else:
        print(f"  ✗ FAIL (returned None)")
        all_pass = False

print("\n" + "=" * 80)
if all_pass:
    print("✓ All verification tests PASSED")
else:
    print("⚠️  Some tests did not pass")
print("=" * 80)
