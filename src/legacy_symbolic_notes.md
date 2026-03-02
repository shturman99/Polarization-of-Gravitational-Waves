# Legacy Symbolic Notes

This file keeps the useful parts of the older symbolic scratch notebook in a compact reference form.

## Core Definitions

The symbolic work used the same building blocks as the numerical package:

- \(A(k) = C_k \epsilon^{2/3} k^{-11/3} / (4\pi)\)
- \(\eta_k = \epsilon^{1/3} k^{2/3} / \sqrt{2\pi}\)
- \(\tilde f(\omega, \eta_k) = 2 \eta_k^{-1} \exp[-\omega^2 / (\pi \eta_k^2)]\)
- \(P_{ij}(\mathbf{k}) = \delta_{ij} - k_i k_j / k^2\)

## Stationary Integrand Structure

The symbolic notebook repeatedly reduced the stationary source to the form

\[
H_{ijij}(k,\omega) \propto \int d^3k_1 \, d\omega_1 \,
A(k_1) A(u)
\tilde f(\omega_1,\eta_{k_1})
\tilde f(\omega-\omega_1,\eta_u)
\mathcal{K}(k,k_1,u),
\]

with \(u = |\mathbf{k} - \mathbf{k}_1|\) and the same geometric kernel used in the packaged implementation.

## Dimensionless Split Form

The exploratory derivation also used the split representation

\[
\mathfrak{H}(\Omega)
= \int_0^\infty dq \, q^{-10/3}
\int_{-\infty}^{\infty} d\Omega_1 \,
e^{-2\Omega_1^2 q^{-4/3}}
\sum_{\alpha \in \{+1,-1,+3\}} c_\alpha(q) I_\alpha(q,\Omega,\Omega_1),
\]

where

\[
I_\alpha(q,\Omega,\Omega_1)
= \int_{|1-q|}^{1+q} ds \,
s^{-10/3+\alpha}
\exp[-2(\Omega-\Omega_1)^2 s^{-4/3}].
\]

This is useful as a comparison form, but the maintained implementation lives in `src/gw_turbulence/`.

## Practical Rule

Keep new symbolic checks in `src/verify_transformations.ipynb` if they need interactivity.
If they become stable and testable, move them into `tests/test_derivations.py`.
