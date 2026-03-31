# Problem Sets: Gravitational Wave Turbulence Kernels

These problems test understanding of the core physics and numerical methods in this codebase. Each problem asks you to *implement* a function from scratch (without looking at the source), then verify it against the existing implementation.

---

## Problem Set 1 — Foundations

### P1.1: The Geometric Kernel

The GW stress-energy tensor kernel from Gogoberidze et al. (2007) Appendix A is:

```
B(p, x, y) = 54 - 2p²x^(3/2) - 2p²y^(3/2) + p⁴ x^(3/2) y^(3/2) + x^(-3/2) y^(3/2) + y^(-3/2) x^(3/2)
```

where `p = k/k₀`, `x = k₁^(-4/3)`, `y = u^(-4/3)` in the Gogoberidze substitution variables.

**Task:** Implement `kernel_bracket(p, x, y)` and verify:

- `kernel_bracket(0, 1, 1)` = 54 + 2 = 56 (check manually)
- `kernel_bracket(1, 1, 1)` = 54 - 2 - 2 + 1 + 1 + 1 = 53 (check manually)
- For `p = 0`, the result should only depend on the last two terms

**Verification:**
```python
from src.gw_turbulence.core import kernel_bracket
assert abs(kernel_bracket(0, 1, 1) - 56.0) < 1e-10
assert abs(kernel_bracket(1, 1, 1) - 53.0) < 1e-10
```

---

### P1.2: The Monochromatic Geometric Kernel K₀(p)

When the turbulent energy spectrum is a delta function `E(k) = E₀ δ(k − k₀)`, the wavevector integration collapses. The remaining geometric factor at `k₁ = u = k₀` is:

```
K₀(p) = 14/3 − p²/3 + p⁴/12
```

**Tasks:**

**(a)** Implement `K0_p(p)`.

**(b)** Show that `K₀(0) = 14/3`. What does this value represent physically (hint: it is the isotropic average of the tensor structure when the source is at rest relative to the GW)?

**(c)** Find the value of `p` at which `K₀(p) = 0`. What does `K₀(p) = 0` at `p = 2` (triangle inequality limit) imply physically?

**(d)** Verify numerically that `K₀(2) > 0` (it is not zero—the zero is elsewhere). Compute `K₀(2)` exactly.

**Expected:** `K₀(2) = 14/3 − 4/3 + 16/12 = 14/3 − 4/3 + 4/3 = 14/3`... re-derive this.

---

### P1.3: The Kraichnan Decorrelation Kernel (Closed Form)

For the monochromatic spectrum with **Kraichnan** (stationary) decorrelation:

```
H(p, Ω) = K₀(p)/p · Θ(2 − p) · exp(−Ω² / (2π))
```

where `Θ` is the Heaviside step function, `p = k/k₀`, and `Ω = ω/η₀`.

**Tasks:**

**(a)** Implement `H_delta_k_kraichnan(p, Omega)` that is vectorized (works on arrays).

**(b)** Verify the Heaviside cutoff: `H(p > 2, Ω)` must be exactly zero. Why does the triangle inequality `|k₁ − u| ≤ k ≤ k₁ + u` with `k₁ = u = k₀` impose `k ≤ 2k₀`?

**(c)** Show the frequency dependence is purely Gaussian. What is the characteristic frequency scale (where the kernel drops to `e⁻¹` of its peak)?

**(d)** Plot `H(p, Ω=0)` vs `p ∈ [0, 2]` and `H(p=1, Ω)` vs `Ω ∈ [-5, 5]`.

**Verification:**
```python
from src.gw_turbulence.core import H_delta_k_kraichnan
import numpy as np
assert H_delta_k_kraichnan(2.5, 0.0) == 0.0          # outside triangle
assert H_delta_k_kraichnan(1.0, 0.0) > 0              # nonzero inside
assert abs(H_delta_k_kraichnan(1.0, 0.0) - H_delta_k_kraichnan(1.0, 0.0)) < 1e-12
```

---

## Problem Set 2 — Integration Infrastructure

### P2.1: Integration Bounds from the Triangle Inequality

In Gogoberidze variables, the integration over `k₁` uses the substitution `x = (k₁/k₀)^(−4/3)` so `k̃₁ = x^(−3/4)`. For given `p = k/k₀` and outer-scale ratio `R`, the inner variable `y = (u/k₀)^(−4/3)` is bounded by the triangle inequality:

```
u ∈ [max(|k̃₁ − p|, 1),  min(k̃₁ + p, R^(3/4))]
```

Converting back to `y`:

```
y_min = u_max^(−4/3),   y_max = u_min^(−4/3)
```

**Tasks:**

**(a)** Implement `_integration_bounds(x, p, R) -> tuple[float, float] | None` that returns `(y_min, y_max)` or `None` if the interval is empty (when `u_min ≥ u_max`).

**(b)** Check that for `p = 1.5`, `R = 100`, `x = 1.0` (i.e., `k̃₁ = 1`), the bounds are non-trivial. Work through the arithmetic by hand.

**(c)** For what values of `x` (given `p = 0.1`, `R = 1e4`) does the function return `None`? How does this relate to the triangle inequality?

**Verification:**
```python
from src.gw_turbulence.core import _integration_bounds
result = _integration_bounds(x=1.0, p=1.5, R=100)
assert result is not None
y_min, y_max = result
assert y_min < y_max
assert _integration_bounds(x=0.01, p=0.01, R=10) is None   # may be None at extreme ratios
```

---

### P2.2: Cosine-Spaced Grid

Near the singularities of the temporal convolution integrand, uniform grids are inaccurate. A **cosine grid** on `[a, b]` uses:

```
t_i = a + (b−a)/2 · (1 − cos(π·i/(N−1))),   i = 0, …, N−1
```

This clusters sample points near both endpoints.

**Tasks:**

**(a)** Implement `_cosine_grid(lower, upper, count)` returning an `np.ndarray` of length `count`.

**(b)** Verify the endpoints exactly: `grid[0] = lower`, `grid[-1] = upper`.

**(c)** Compare the density of points near the endpoints between a cosine grid and `np.linspace` for `N = 20` on `[0, 1]`. Specifically, count how many points fall in `[0, 0.1]` for each.

**(d)** Explain why clustering near singularities improves trapezoid-rule accuracy for integrands that blow up at the endpoints.

**Verification:**
```python
from src.gw_turbulence.core import _cosine_grid
import numpy as np
g = _cosine_grid(2.0, 5.0, 50)
assert abs(g[0] - 2.0) < 1e-12
assert abs(g[-1] - 5.0) < 1e-12
assert len(g) == 50
assert np.all(np.diff(g) > 0)   # strictly increasing
```

---

### P2.3: Singular Interval Decomposition

The temporal convolution `∫ g(q₁) g(q − q₁) dq₁` has integrable singularities at `q₁ = 0` and `q₁ = q`. To handle these, the integration domain `[−q_bound, q_bound]` is split to *exclude* small windows around the singular points.

**Tasks:**

**(a)** Implement `_conv_intervals(q, q_bound, split_width)` that returns a list of `(lower, upper)` sub-intervals covering `[−q_bound, q_bound]` with windows of width `split_width` around `0` and `q` removed.

**(b)** For `q = 2.0`, `q_bound = 5.0`, `split_width = 0.1`, enumerate all returned intervals by hand.

**(c)** What happens when `q = 0`? The singular points coincide — verify your implementation merges the windows correctly (only one gap, not two overlapping ones).

**(d)** What happens when `q > 2 * q_bound`? The point `q₁ = q` is outside the domain — verify it is simply ignored.

**Verification:**
```python
from src.gw_turbulence.core import _conv_intervals
intervals = _conv_intervals(q=2.0, q_bound=5.0, split_width=0.1)
# Singular points at 0 and 2.0, so gaps are [-0.1, 0.1] and [1.9, 2.1]
# Expected intervals cover [-5, -0.1], [0.1, 1.9], [2.1, 5.0]
total_covered = sum(hi - lo for lo, hi in intervals)
assert abs(total_covered - (10.0 - 0.2 - 0.2)) < 1e-10
```

---

## Problem Set 3 — Decaying Turbulence

### P3.1: The Decaying Temporal Kernel g(q)

For turbulence with energy decaying as `E(t) ∝ (1 + t/τ₁)^(−2/3)`, the Fourier-space temporal kernel is:

```
g(q) = e^{iq} · (−iq)^{−5/3} · Γ(1/3, −iq)
```

where `Γ(a, z)` is the **lower** incomplete gamma function and `q = ωτ₁`.

**Tasks:**

**(a)** Implement `g_decaying(z)` for scalar complex input using `mpmath.gammainc`. Note: `mpmath.gammainc(a, z)` returns the *lower* incomplete gamma `γ(a, z)` when called with two arguments (verify this in the mpmath docs).

**(b)** Verify that `g(0)` diverges (power-law singularity at the origin) — your implementation should handle this gracefully by returning a very large number rather than `NaN`.

**(c)** Plot `Re[g(q)]` and `Im[g(q)]` for `q ∈ [0.01, 30]`. Describe the qualitative behavior: does it oscillate? decay? at what rate?

**(d)** For large real `q`, the kernel should decay as `q^{-5/3}` (up to oscillatory prefactors). Verify this numerically by computing `|g(q)| * q^{5/3}` for `q ∈ [10, 1000]` and checking it approaches a constant.

**Verification:**
```python
from src.gw_turbulence.core import g_decaying
import numpy as np
# Test that g is complex-valued
val = g_decaying(1.0)
assert np.iscomplex(val) or isinstance(val, complex)
# Test large-q power law decay
q_large = 100.0
ratio = abs(g_decaying(q_large)) * q_large**(5/3)
q_larger = 1000.0
ratio2 = abs(g_decaying(q_larger)) * q_larger**(5/3)
assert abs(ratio - ratio2) / abs(ratio) < 0.15   # roughly constant
```

---

### P3.2: Temporal Convolution for the Decaying Model

The GW spectrum from decaying turbulence requires the convolution:

```
C(q) = ∫ dq₁  Re[g(q₁) · g(q − q₁)]
```

This integral is computed numerically over a truncated domain `[−q_bound, q_bound]`.

**Tasks:**

**(a)** Implement `_temporal_conv_decay(q, q_bound=None, split_width=None, n_points=200)` using the `_cosine_grid` and `_conv_intervals` helpers and `np.trapz`.

**(b)** Verify that `C(q)` is a real number (taking `Re[...]` before integrating, as in the code).

**(c)** Compute `C(q)` for `q ∈ [0.1, 20]` and plot it. How does it compare qualitatively to the Kraichnan Gaussian `exp(−Ω²/(2π))`?

**(d)** The convolution must be positive everywhere (it represents a power spectrum). Verify this numerically for a range of `q` values.

**Verification:**
```python
from src.gw_turbulence.core import _temporal_conv_decay
c0 = _temporal_conv_decay(q=0.0)
c5 = _temporal_conv_decay(q=5.0)
assert c0 > 0
assert c5 > 0
assert c0 > c5   # decays from peak
```

---

### P3.3: Monochromatic Decaying Kernel H_delta_k_decay

The full monochromatic decaying kernel combines the geometric factor `K₀(p)` with the temporal convolution:

```
H(p, q) = K₀(p)/p · Θ(2 − p) · C(q)
```

where `C(q) = ∫ dq₁ Re[g(q₁) g(q − q₁)]` from P3.2.

**Tasks:**

**(a)** Implement `H_delta_k_decay(p, q)`.

**(b)** Verify the `p`-dependence is identical to the Kraichnan model (same `K₀(p)/p · Θ(2−p)` factor). What is different between the two models?

**(c)** Implement a grid version `H_delta_k_decay_grid(ps, qs)` that computes `C(q)` **once per q value** (not once per `(p, q)` pair) to avoid redundant work. How much faster is this than a naive loop?

**(d)** Compare the frequency shapes: for fixed `p = 1.0`, plot `H_kraichnan(p, Ω)` and `H_decay(p, q)` on the same axes. Which has heavier tails?

---

## Problem Set 4 — The Full Kolmogorov Model

### P4.1: The Stationary Integrand integrand_y

For the full Kolmogorov power-law spectrum with Kraichnan decorrelation, the integrand over the inner variable `y` is:

```
f(y; x, p, q, M) = y^(3/4) · (x+y)^(-1/2) · x^(3/4) · B(p,x,y) · exp(−2xy/(x+y) · q²/M²) · erfc(−√2 q / (M √(x+y)))
```

where `erfc` is the complementary error function.

**Tasks:**

**(a)** Implement `integrand_y(y, x, p, q, M)`.

**(b)** For `q = 0`, the `exp` factor is 1 and `erfc(−∞) → 2`. Verify this limiting behavior in your implementation.

**(c)** The term `erfc(−√2 q/(M√s))` with `s = x + y` enforces causality. For large `q/M`, this term → 0. Verify numerically that `integrand_y` decays rapidly as `q → ∞` for fixed `x`, `y`.

**(d)** For what sign convention does `erfc(-z) → 2` as `z → +∞`? Confirm that the code's use of `erfc` is consistent with this (look at `scipy.special.erfc`).

**Verification:**
```python
from src.gw_turbulence.core import integrand_y
# At q=0: erfc(-0) = 1, exp(0) = 1
val_q0 = integrand_y(y=1.0, x=1.0, p=0.5, q=0.0, M=1.0)
assert val_q0 > 0
# Large q: should be much smaller than q=0
val_large_q = integrand_y(y=1.0, x=1.0, p=0.5, q=100.0, M=1.0)
assert val_large_q < val_q0 * 1e-3
```

---

### P4.2: The p → 0 Analytic Limit H_k0_analytic

When `k → 0` (long-wavelength GWs), the full double integral simplifies. The analytic `p → 0` limit is:

```
H_k0(q) = (7 M³ k₀^{-4}) / (16 π^{3/2})  ∫_{R^{-1}}^{1} x^{11/4} exp(−q²x/M²) erfc(−q√x/M) dx
```

**Tasks:**

**(a)** Implement `H_k0_analytic(q, M=1.0, k0=1.0, R=1e4)` using `scipy.integrate.quad` for the 1D integral.

**(b)** Verify that for small `q`, the result grows (as less suppression by the exponential), and for large `q` it decays. Find the approximate peak frequency `q*` numerically.

**(c)** This function should match `H_pq(p → 0, q)` asymptotically. Test this by calling `H_pq(p=1e-4, q, ...)` and comparing to `H_k0_analytic(q, ...)` for several `q` values. What relative error do you observe?

**(d)** The prefactor `7/16` comes from the angular integration in 3D. Derive where the factor of 7 comes from by considering the isotropic tensor contraction (hint: it involves `∫ dΩ P_ij(k̂) P_ij(k̂) ...`).

---

### P4.3: The Full H(p, q) Prefactor

The overall normalization of `H(p, q)` is:

```
prefactor(p, M, k₀) = 3 M³ k₀^{-4} / (256 (2π)^{3/2} p)
```

**Tasks:**

**(a)** Implement `_h_prefactor(p, M, k0)`.

**(b)** The factor `1/p` diverges as `p → 0`. The code uses `p_floor = max(p, 1e-10)` to avoid this. Why is it physically correct that `H ∝ 1/p` at small `p`? (Hint: think about the GW power spectrum `Ω_GW(k) ∝ k H(k, ω)`.)

**(c)** The prefactor scales as `M³`. This comes from the cubic dependence of the source on the velocity field (two powers from the stress-energy tensor, one from the Kolmogorov normalization). Verify the `M³` scaling numerically: compute `H_pq(p, q, M=0.1) / H_pq(p, q, M=1.0)` and check it equals `0.1³ = 0.001`.

**(d)** Implement `H_pq(p, q, M, R, k0)` that chains together `_h_prefactor`, `inner_integral`, and the outer `scipy.integrate.quad`. Verify it returns a positive number for `p = 0.5`, `q = 1.0`, `M = 1.0`.

---

## Problem Set 5 — White-Noise Spatial Model

### P5.1: The White-Noise Geometric Kernel

For `δ³(r)` spatial correlations (white noise), the Kolmogorov substitution variables are not needed. Working directly with `z = k₁/k₀` and `y = u/k₀`:

```
K_zy(p, z, y) = 54 − 2p²/z² − 2p²/y² + p⁴/(z²y²) + z²/y² + y²/z²
```

**Tasks:**

**(a)** Implement `kernel_bracket_zy(p, z, y)`.

**(b)** Verify that `kernel_bracket_zy(p, x^{-3/4}, y^{-3/4})` equals `kernel_bracket(p, x, y)` up to a constant factor. What is that factor, and why does it arise from the change of variables?

**(c)** For `p → 0`, the kernel simplifies. Show analytically that in this limit:
   `K_zy(0, z, y) = z²/y² + y²/z²`

---

### P5.2: White-Noise Spatial Integral

The 2D spatial integral for white-noise correlations is:

```
S(p, R) = ∫_1^{R^{3/4}} (z/p) dz ∫_{max(|p−z|, 1)}^{min(p+z, R^{3/4})} y dy · K_zy(p, z, y)
```

**Tasks:**

**(a)** Implement `_white_spatial_integral(p, R)` using nested `scipy.integrate.quad` calls.

**(b)** The inner `y` bounds come from the triangle inequality `|k − k₁| ≤ u ≤ k + k₁` with `k = p k₀`, `k₁ = z k₀`, `u = y k₀`. Verify these bounds algebraically.

**(c)** The outer `z` integral starts at `z = 1` (not `z = 0`) because modes with `k₁ < k₀` are outside the inertial range. For `R = 100` and `p = 1.5`, what fraction of the integral comes from `z ∈ [1, 2]` vs `z ∈ [2, R^{3/4}]`?

**(d)** The white-noise Kraichnan kernel `H_white_kraichnan(p, Ω) = exp(−Ω²/(2π)) · S(p, R)` factorizes perfectly. Why does the full Kolmogorov kernel not factorize this way?

---

## Problem Set 6 — Challenge Problems

### P6.1: LRU Caching Strategy

The function `_g_decaying_scalar` uses `@lru_cache(maxsize=1024)`. During a grid computation over `N_q × N_p` points, the temporal convolution at each `(x, y, q)` triplet calls `g_decaying(q₁)` for many `q₁` values.

**Tasks:**

**(a)** Explain why caching `_g_decaying_scalar` (which takes a `complex` argument) is effective here. When are cache hits expected?

**(b)** The cache is keyed on the exact floating-point value of the complex argument. Design a scenario where two calls that should hit the same cache entry miss because of floating-point arithmetic. How does the code guard against this?

**(c)** Estimate the memory usage of the cache for `maxsize=1024` entries, given each complex value is 16 bytes and Python object overhead is ~50 bytes.

**(d)** For the `"trapz"` method with `convolution_points=160` and a `24×24` grid, estimate the total number of `g_decaying` evaluations and the expected cache hit rate.

---

### P6.2: MPI Row Distribution

The `H_pq_decaying_grid` function supports MPI parallelism by distributing rows across ranks. The `split_row_indices` function assigns a subset of row indices to each MPI rank.

**Tasks:**

**(a)** Implement `split_row_indices(n_rows, rank, size)` that returns a list of row indices for the given rank. Use a simple round-robin distribution (row `i` goes to rank `i % size`).

**(b)** For `n_rows = 10`, `size = 3`, list the rows assigned to ranks 0, 1, and 2. Verify that each row appears exactly once across all ranks.

**(c)** After each rank computes its local rows, `gather_grid` assembles the full grid on rank 0. Describe the data that must be transmitted: how many floats does each rank send? For `N_q = 30`, `N_p = 30`, `size = 4`, compute the total bytes transmitted (assuming float64).

**(d)** For a workload where different rows have very different compute times (e.g., small `q` is cheaper than large `q`), is round-robin optimal? Propose a better assignment strategy.

---

### P6.3: Numerical Stability at Singularities

The kernel `g(q₁)` diverges as `q₁ → 0`. In `integrand_y_decaying`, this is handled by substituting a small imaginary perturbation:

```python
if z1 == 0:
    z1 = (split_width * np.sqrt(x) / M) * 1j
```

**Tasks:**

**(a)** Explain why `g(iε)` for small real `ε > 0` is finite even though `g(0)` diverges. Use the asymptotic form of the incomplete gamma function `Γ(1/3, z)` as `z → 0`.

**(b)** What is the order of the singularity at `q₁ = 0`? I.e., does `|g(q₁)| ∼ |q₁|^α` for some `α` as `q₁ → 0`? Determine `α` from the formula.

**(c)** The singularity is described as "integrable." Verify analytically that `∫₀^δ |g(q₁)| dq₁ < ∞` for the `α` you found in (b).

**(d)** Compare two numerical strategies for evaluating `∫₀^1 g(q₁) g(1-q₁) dq₁`:
   - Strategy A: uniform grid with `N = 1000` points, exclude endpoints
   - Strategy B: cosine grid with `N = 200` points, exclude endpoints

   Implement both and compare accuracy vs the "quad" method result. Which wins?

---

### P6.4: Dimensional Analysis and Physical Units

The dimensionless variable `p = k/k₀` measures wavenumber in units of the peak turbulence wavenumber `k₀`. The variable `q = ωτ₁` (for decaying) or `Ω = ω/η₀` (for Kraichnan) measures frequency.

**Tasks:**

**(a)** The Kraichnan decorrelation time for mode `k` is `τ_k = 1/η_k` where `η_k = (ε^(1/3)/√(2π)) k^(2/3)`. At the peak scale `k = k₀`, express `η₀` in terms of `ε` and `k₀`. Then show that `Ω = ωτ_k₀`.

**(b)** In `integrand_y`, the argument of the exponential is `2xy/(x+y) · q²/M²`. The factor `2xy/(x+y)` is a harmonic-mean-like combination. Show this equals `2/(k₁^{-4/3} + u^{-4/3}) · ... ` and interpret it as `2 k̃₁² ũ²/(k̃₁² + ũ²)` in the Kraichnan model.

**(c)** The GW energy density spectrum is `Ω_GW(k) ∝ k⁴ H(k, ω=0)`. Given that `H(p, q) ∝ M³ k₀^{-4} / p`, what is `Ω_GW(k)` as a function of `k` and physical parameters? What `k`-scaling does this predict?

**(d)** The prefactor `3/(256 (2π)^{3/2})` contains purely geometric factors. Trace through the GW production integral (see `derivation.tex` if available, or Gogoberidze 2007) to understand where each factor comes from. What physical process generates the factor of 3?

---

## Appendix: Useful References

- `src/gw_turbulence/core.py` — all implementations above live here
- `tests/test_derivations.py` — symbolic identity checks you can compare against
- `tests/test_core.py` — numerical regression tests for many of the functions above
- Gogoberidze et al. (2007) — theoretical basis for the `kernel_bracket` formula
- `src/numerical_benchmarks.ipynb` — timing and convergence demonstrations

## How to Run Verifications

```bash
cd /home/mgurgeni/programming/Polarization-of-Gravitational-Waves
MPLBACKEND=Agg python -m unittest discover -s tests -v

# Or in a notebook/REPL:
import sys; sys.path.insert(0, "src")
from gw_turbulence.core import *
```
