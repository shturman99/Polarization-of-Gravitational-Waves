# Review-response action plan

Derived from the five-agent review of 2026-08-14 (`REVIEW_2026-08-14.md`).
Tasks are ordered by (value ÷ effort). Status is updated as work proceeds.

**Markup convention in `derivation.tex`:** `\add{...}` = red, added in response to a
report; `\del{...}` = red strikeout, removed; `\why{...}` = red bracketed note naming
the point addressed. Compile a clean copy by setting `\reviewmarkupfalse` in the
preamble.

---

## Tier 0 — mechanical defects (no physics judgement needed)

| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| 0.1 | `bib.bib` — add the pre-empting and missing references | citation audit | — | **DONE** (43→84) |
| 0.2 | Kraichnan sweeping mis-citation → `Kraichnan:1964` | citation audit C2 | — | **DONE** |
| 0.3 | "BK2016" cited 10× with no entry → `Brandenburg:2016odr` | citation audit C1 | — | **DONE** |
| 0.4 | erfc printed in two forms — correct the main-text equation | Referee C | — | **DONE** |
| 0.5 | **erfc in `core.py:integrand_y`** — last home of the defect | Referee C | 1 h | **DONE** |
| 0.6 | §V LISA-sensitivity sentence is factually wrong | Referee A | 1 h | **DONE** |
| 0.7 | Reconcile author lists between `main.tex` and `derivation.tex` | Referee C | 5 min | **AUTHORIAL** |
| 0.8 | Decide whether citations belong in the abstract (PRD style) | — | 5 min | **AUTHORIAL** |

## Tier 1 — framing corrections (protect priority; no new computation)

| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| 1.1 | Abstract: qualify factorisation to *asymptotes* | Referee B, prior-art | — | **DONE** |
| 1.2 | Abstract: white-noise floor "we show" → "known, we verify" | Referee B | — | **DONE** |
| 1.3 | Abstract: drop Ω_T ∝ Ω_M, replace with the band statement | Referee B/C | — | **DONE** |
| 1.4 | Appendix G corollary: restrict to **stationary** UETCs | Referee C | — | **DONE** |
| 1.5 | Cite `RoperPol:2022iel` in §V (was intro-only) | both lit agents | — | **DONE** |
| 1.6 | Conclusions 1–5 rewritten to match | Referee B/C | — | **DONE** |
| 1.7 | Concede the cusp tail is the standard Abelian theorem | Referee B | — | **DONE** |
| 1.8 | Cite Auclair as *support* for finite lifetime, not only objection | Referee B | — | **DONE** |
| 1.9 | Credit `Niksa:2018ofa` for the k⁻¹-vs-k⁻²ᐟ³ objection | prior-art | — | **DONE** |
| 1.10 | §VII: state that the subviscous calculation is not performed | prior-art | — | **DONE** |
| 1.11 | **Reclassify `T_burst`** — duration is a third ingredient, not a UETC | Referee C | 2 h | **DONE** |
| 1.12 | Reconcile impulsive-onset vs sustained-coherent in §V | Referee C | 1 d | **TODO** |
| 1.13 | Replace pre-Hosking q=1/2 magnetic decay class | citation audit C6 | 3 h | **CODE+FIGS DONE**, caption number pending |

## Tier 2 — calculations that change the paper's standing

| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| 2.1 | **Branch diagram**: IR slope 3→1 | **both referees, converged** | 1–2 d | **DONE** |
| 2.2 | Viscous-cusp floor in §VII — turn the one novel claim into a result | Referee B, prior-art | 3–5 d | **TODO** |
| 2.3 | k_break in Hz vs LISA/PTA sensitivity curves | Referee A/C | 1 d | **TODO** |
| 2.4 | Report ωT_em and T_em/τ_c for Fig. `ir_resolution`, or re-run | Referee C | 2 d–1 wk | **TODO** |
| 2.5 | Run our kernel at RP20's own T_em, τ_c, k-range and show k^1.x | Referee C | 3 d | **TODO** |
| 2.6 | Measure the **magnetic** UETC from RP20 Zenodo data | Referee A/C | 1–3 wk | **TODO** |

## Tier 3 — scope

| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| 3.1 | Split: letter (dissipation gap, once quantified) + long companion | Referee C | 1 wk | **AUTHORIAL** |
| 3.2 | Defer §VI/§VII to a third paper | Referee C | — | **AUTHORIAL** |
| 3.3 | Write `main.tex` abstract and sections | — | — | **AUTHORIAL** |

---

## Why 2.1 is next

Both referees, working blind to each other, named the same task. It costs one to two
days with a tool that is already written, and it does three things at once:

1. It exhibits a UETC that **does** move the infrared, which is what makes the
   Appendix-G corollary defensible rather than merely patched.
2. It shows the cusp peak-pinning result and the k¹ result are the **same mechanism** —
   a hard finite lifetime. The manuscript currently treats these in two disconnected
   sections and never notices they are one.
3. It replaces the retracted Ω_T ∝ Ω_M localisation with a controlled, reproducible
   statement, which is the thing the reversal history most demands.

---

## Log

- **2026-08-14** — Tier 0 and Tier 1 framing items closed (0.1–0.6, 1.1–1.10).
  Markup macros installed in `derivation.tex`.
- **2026-08-14, task 0.5** — erfc corrected in **three** call sites, not one:
  `core.py:integrand_y`, `Notebooks/_fullspectrum_kernel.py`,
  `Notebooks/band_split_gw.py`. Verified the headline band-split result is unaffected:
  IR slopes now **+2.996 / +2.998 / +3.007** (were +3.00 / +3.01 / +3.02). The
  inertial-only band — zero infrared power — still radiates p³, so the paper's sharpest
  claim survives the correction.
- **2026-08-14** — two defects were sitting in `derivation.tex` as **invisible LaTeX
  comments** and are now rendered notes:
  1. the erfc inconsistency (now resolved, note records the fix and the numbers);
  2. **a 4π error in the dimensional prefactor** of Eq. `eq:H-delta-k`, propagating to
     the dimensional result of §`sec:delta-kraichnan` as 8π. Shapes, slopes and peaks
     are unaffected; **absolute amplitudes are not**. Flagged, not fixed — see new
     task 0.9.
  While surfacing (2) it emerged that the comment referenced `eq:H-delta-Kraichnan`,
  a label that **has never existed** in the document.

### Added after the first pass

| # | Task | Effort | Status |
|---|---|---|---|
| 0.9 | Fix the 4π/8π dimensional prefactor, consistently with §III normalisation | 0.5 d | **DONE** |
| 0.10 | Repair the test environment | 1 h | **DONE** — 68 passed, 0 failed |

- **2026-08-14, task 0.10** — root cause was that this repo has **no venv of its own**;
  `python3` is the `chiral-mhd` venv's interpreter (numpy 2.4.4), while
  `requirements.txt` pins numpy 1.26.4. Fixed by binding
  `_trapz = getattr(np, "trapezoid", None) or np.trapz` per module in `core.py`,
  `test_core.py`, `test_derivations.py`, `roperpol_data.py` — this works under both
  numpy 1.x and 2.x. (A blanket rename to `np.trapezoid` would have broken the pinned
  environment: `trapezoid` arrived in numpy **2.0**, not 1.22.) **17 failed → 0 failed,
  68 passed.** All 17 were the same AttributeError; no physics failure was hiding behind
  them. `decaying_gw_modular.ipynb` still has 4 bare `np.trapz` calls — not collected by
  pytest, left alone.
- **2026-08-14, task 0.9** — the 4π flag was **correct**, and re-derivation found a
  **second, unflagged** slip. (i) the radial collapse
  ∫k₁²dk₁A(k₁)=E₀/(4π) had been applied as ×E₀ ⇒ `eq:H-delta-k` and `A:Hdeltak` were 4π
  too large; (ii) a further factor 2 was lost substituting into
  `eq:dimless-delta-Kraichnan` ⇒ that equation was 8π too large. The flag reached the
  right endpoint number by luck. Confirmed against the document's *own*
  `eq:angular-collapse`, which was correct all along — the two displays had disagreed
  with each other. Shapes, slopes, peaks, edges, Mach scalings, figures and code are
  **unaffected**; only absolute amplitude of the monochromatic branch.

- **2026-08-14, task 2.1 — DONE, and it sharpened the prediction.** New
  `Notebooks/ir_branch_diagram.py`, figure `images/ir_branch_diagram.pdf`, new
  §`sec:ir-branch` with Table `tab:ir-branch`. Results: IR slope **+3.000** in every
  case; intermediate band **+0.974 / +0.995** (global lifetime) and **+0.965 / +0.996**
  (eddy lifetime) for τ̂ = 10³, 10⁴.
  **The break is at π/τ_c, not 1/τ_c.** Because the stress is quadratic the temporal
  factor is the transform of the *squared* tent, f′(0⁺) = −2/τ_c; the asymptotes cross
  at ωτ_c = √6 and the slope midpoint is at ωτ_c = π *exactly* (1+cos u = 2 sin u/u).
  Measured coefficients 3.07 (global) and 3.17 (eddy) against π = 3.1416.
  Honest caveats now in the text: the linear band is resolved only for τ̂ ≳ 10³, and it
  is not a clean power law — the sin² ringing makes the local slope oscillate between
  0.90 and 1.25. Peak stays at p ≈ 1.1–2.2, confirming that one finite lifetime produces
  **both** the k¹ band and the source-scale peak.
- **2026-08-14, citation verification** — nine attributions checked against primary
  sources; **five of my own edits were wrong** and are corrected: (i) LISA's Ω-sensitivity
  minimum is **2.5 mHz**, not 3–10 mHz (that is the *strain* minimum) — independently
  recomputed, so the discriminating band *overlaps* LISA's best sensitivity rather than
  sitting below it; (ii) the band-split is **not** a limit Brandenburg & Boldyrev failed
  to take — their Eq. (14)/Fig. 1 is exactly a band-limited input; what is ours is the
  GW kernel and the retained temporal sector; (iii) the Kulsrud caution was backwards —
  the **bulk** of subviscous magnetic energy is at the resistive scale; (iv)
  `RoperPol:2025lgc` is scope-setting, not a concession, and contains **zero** mention of
  Prandtl number; (v) `Caprini:2009fx` already links regularity to the **peak**, so the
  peak-novelty claim was demoted to the object it applies to. Added `Perez:2020`
  (arXiv:2004.11458), an MHD Eulerian UETC measurement that favours sweeping and
  therefore cuts *against* the cusp.

### Remaining

| # | Task | Status |
|---|---|---|
| 1.13 caption | Re-read ⟨f_A/f_B⟩ for the regenerated nonhelical (Hosking) panel; caption still says 0.83 | audit agent running |
| 1.12 | Reconcile impulsive-onset vs sustained-coherent in §V | TODO |
| 2.2 | Viscous-cusp floor calculation | TODO |
| 2.3 | k_break in Hz vs LISA/PTA — now more valuable, since the band overlaps LISA's best sensitivity | TODO |
| 2.4–2.6 | ir_resolution branch parameters; kernel at RP20 parameters; magnetic UETC | TODO |

- **2026-08-14, numerical audit (resumed).** Three outcomes.
  1. **`f_peak` caption corrected.** The Hosking swap moves the non-helical MHD ratio
     **0.83 → 0.40**; control re-run with the old (β=1, q=1/2) parameters reproduces
     0.8301 exactly, so the shift is entirely the class change. Composite figure
     regenerated (it had been left stale). **The HD-vs-MHD contrast in the prose no
     longer holds**: at 0.40 the non-helical class now sits with the HD classes, and
     helical MHD alone stands apart. Prose rewritten accordingly.
  2. **The erfc fix is independently validated.** `core.H_k0_analytic`, the closed-form
     aeroacoustic p→0 limit derived separately and never edited, is now reproduced by
     `H_pq` to **1.00000 at every q**; the uncorrected kernel disagreed with its own
     limit by up to **8%**. Added to the manuscript.
  3. **One audit claim did not survive checking.** The agent reported the post-fix
     band-split slopes `+2.996/+2.998/+3.007` as unreproducible. Re-run at the
     documented parameters gives **+2.9963/+2.9979/+3.0073** on [2e-3, 2e-2], identical
     at n=7 and n=25 — grid-independent. The manuscript values stand; the agent
     evidently imported a pre-fix copy from its scratch tree.

### TOP REMAINING ITEM — the 1.47 constant moved

| # | Task | Status |
|---|---|---|
| 0.11 | **ξ\* = 1.47 → 1.488**, and p_peak = 1.48 M^1.00 → **1.515 M^1.007** | **TODO — do first** |

The erfc correction moves the paper's most-quoted number. Scope: ~26 occurrences in
`derivation.tex`, plus hardcoded guide lines in `stationary_fixed_epsilon.py`,
`finite_coherence_gw.py`, `band_split_mach.py`, `gw_peak_vs_mach.py`. Related numbers
that also move: 1.49→1.53 at M=1, 0.77→0.744 at M=3.4, fixed-ε law 1.47/M²→1.53/M².
**The test suite will not catch this**: `test_stationary_peak_scales_as_1p47_M` asserts
only 1.30 < A < 1.65, so it passes at both 1.48 and 1.52. Tighten it once the value is
settled.

Other numbers confirmed changed by the erfc fix and needing an update pass: model-grid
UV −4.75→−4.73; Mach exponent +0.974→+0.970; IR-band uplift 1.9→2.0;
Saffman/Batchelor 1.4→1.49; source band 0.67–1.35→0.659–1.318; band-split-Mach range
2.98–3.08→2.99–3.06; M^-3 flattening 13%/19%→16.6%/14.0%.

Pre-existing inconsistencies found (unrelated to erfc): delta-control peaks 0.47/0.74
vs computed 1.366/2.503; "1.26 at M=3" vs 1.327; "peak from 1.5k₀ to k₀" vs computed
1.11; "1.49 at M=1" (§III) vs "1.53" (App. G) for the same quantity.

---

## Notes for an automated follow-up run

**Abort check.** If `REVIEW_2026-08-14.md` is missing, or `bib.bib` has ~43 entries rather
than ~88, the prior session's work was never pushed. Stop and report that; do not redo it.

**Task 0.11 in detail.** ~26 occurrences of the 1.47 family in `derivation.tex`, ~33 across
`Notebooks/*.py` and `src/gw_turbulence/*.py`, including hardcoded plot guide lines in
`stationary_fixed_epsilon.py`, `finite_coherence_gw.py`, `band_split_mach.py`,
`gw_peak_vs_mach.py`. **The digits play three distinct roles** — the universal constant ξ\*,
the law p_peak = 1.47 M, and the fixed-ε form 1.47/M². **Never blind find-and-replace;
inspect every site.** Recompute rather than trusting the quoted deltas: the generating
script is `Notebooks/stationary_peak_analysis.py` (~90 s). Related values that move:
1.49→1.53 at M=1, 0.77→0.744 at M=3.4, 1.47/M²→1.53/M². Afterwards tighten
`test_stationary_peak_scales_as_1p47_M`, which asserts only 1.30 < A < 1.65 and will not
catch this drift.

**House rules.**
- Mark every change `\add{}` / `\del{}` / `\why{}` (see the markup convention above).
- After each batch: `pdflatex ×3 + bibtex`, and require 0 errors, 0 undefined references,
  0 undefined citations. Also keep `python3 -m pytest` green (currently 68 passed).
- Verify numbers by running the generating script. Several claims from subagents proved
  wrong today when checked — do not accept a reported number without reproducing it.
- Cross-reference each change against `REVIEW_2026-08-14.md`; do not re-introduce a claim
  a referee showed to be pre-empted (esp. Ω_T ∝ Ω_M, and "we show" for the white-noise floor).
- Do **not** push to `main`. Commit to a branch `review-round-2` and open a PR.
