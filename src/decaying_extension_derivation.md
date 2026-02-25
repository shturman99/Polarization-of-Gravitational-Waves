# Extending the Stationary Derivation to the Decaying Case

This note provides an explicit bridge from the stationary temporal factor used in `derivation.tex` to the decaying-spectrum kernel implemented in `compute_H.py` and notebooks.

## 1) Stationary temporal block

In the stationary treatment, after integrating over the time lag one obtains a frequency factor of the schematic form

\[
\exp\!\left[-\frac{2xy}{x+y}\frac{q^2}{M^2}\right]\,\operatorname{erfc}\!\left(-\frac{\sqrt{2}q}{M\sqrt{x+y}}\right).
\]

This is exactly what appears in `integrand_y`.

## 2) Decaying replacement rule

For the decaying spectrum model, the temporal correlator is replaced by

\[
g(z)=e^{iz}(-iz)^{-5/3}\Gamma\!\left(\frac13,-iz\right),
\]

with dimensionless argument \(z=\omega\tau_1\).

The extension is:

\[
\text{stationary temporal factor}
\quad\Longrightarrow\quad
\int_{-\infty}^{\infty} dQ_1\, g\!\left(Q_1\frac{\sqrt{x}}{M}\right)
\,g\!\left((q-Q_1)\frac{\sqrt{y}}{M}\right).
\]

That replacement is exactly the operation used in `integrand_y_decaying`.

## 3) Why the same spatial kernel is retained

The geometric and spectral parts are unchanged by this substitution:

- Jacobian block: \(y^{3/4}(x+y)^{-1/2}x^{3/4}\)
- Geometric bracket: `kernel_bracket(p,x,y)`

Only the temporal block changes. Therefore, this is a strict extension of the same derivation structure, and can be tested against the stationary expression at shared parameter points.

## 4) Explicit testable consequences

1. **Dimensional invariance:** with \(q_i=\omega_i\tau_1\), `g` is evaluated on dimensionless arguments only.
2. **Regularity:** for finite real \(z\), \(g(z)\) must remain finite (tested on sample grids).
3. **Continuity with workflow:** all stationary integration bounds and the \((x,y)\) transform remain unchanged.

These are captured in `src/test_transformations.py` and in the notebook sections for decaying-model checks.
