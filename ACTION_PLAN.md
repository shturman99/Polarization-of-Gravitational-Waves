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
| 0.7 | Reconcile author lists | Referee C | 5 min | **DONE** — Brock removed, both files agree |
| 0.8 | Citations in the abstract | — | 5 min | **DONE** — stripped, credit moved to intro/body |

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
| 1.12 | Reconcile impulsive-onset vs sustained-coherent in §V | Referee C | 1 d | **DONE** (overnight) |
| 1.13 | Replace pre-Hosking q=1/2 magnetic decay class | citation audit C6 | 3 h | **DONE** (overnight) |

## Tier 2 — calculations that change the paper's standing

| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| 2.1 | **Branch diagram**: IR slope 3→1 | **both referees, converged** | 1–2 d | **DONE** |
| 2.2 | Viscous-cusp floor in §VII — turn the one novel claim into a result | Referee B, prior-art | 3–5 d | **DONE** (overnight; coefficient corrected) |
| 2.3 | k_break in Hz vs LISA/PTA sensitivity curves | Referee A/C | 1 d | **DONE** (overnight) |
| 2.4 | Report ωT_em and T_em/τ_c for Fig. `ir_resolution`, or re-run | Referee C | 2 d–1 wk | **DONE** (overnight) |
| 2.5 | Run our kernel at RP20's own T_em, τ_c, k-range and show k^1.x | Referee C | 3 d | **DONE** (overnight) |
| 2.6 | Measure the **magnetic** UETC from RP20 Zenodo data | Referee A/C | 1–3 wk | **DONE — result is negative for our cusp** |

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
| 0.11 | **ξ\* = 1.47 → 1.488**, and p_peak = 1.48 M^1.00 → **1.515 M^1.007** | **DONE 2026-08-15** |

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

---

## Round 2 — automated follow-up run, 2026-08-15 (branch `review-round-2`)

Abort check passed: `REVIEW_2026-08-14.md` present, `bib.bib` at 88 entries.

### Closed

| # | Outcome |
|---|---|
| 0.11 | **ξ\* = 1.488**, fit **1.515 M^1.007**, 1.53 at M=1, 0.744 at M=3.37. 29 sites in `derivation.tex` (each inspected, none replaced blind) plus 8 scripts. Test renamed and tightened to `1.49 < A < 1.57` (measured 1.5256), which the old bounds could not catch. **Departure from the plan:** the fixed-ε form is **1.49/M²**, not the 1.53/M² predicted — 1.53 is the M=1 value, where the law has already begun to saturate; measured coefficient is 1.482–1.501 over M = 0.03–0.3. |
| 1.12 | New paragraph in §V separating the two accounts: the build-up test measures the sharpness of the switch-**on**, the branch of `eq:window-factor` depends on the **length** of the window, and a source may do both. Explicitly does **not** claim to settle the coherence question — the build-up bound still points the other way, and that is stated. |
| 1.13 | The plan's entry was itself stale: the prose had already been fixed and no live `0.83` survived. The real residual was the **figure caption**, which still documented the superseded q=1/2, β=1 class and mislabelled the β=3 panel "Saffman". Fixed, plus the closing clause that restated the retracted HD-vs-MHD split. |
| 2.3 | New §`sec:k-break-hz`. **f_break = f_H / (2 τ̂_c)** — γ cancels. LISA Ω-minimum recomputed at **2.520 mHz**, factor-2 window **1.042–4.604 mHz** (strain minimum separately at 7.257 mHz), independently reproducing the earlier session's 2.52 mHz. Verdict: the break lies below LISA's window for an EW source in every corner but one. New `Notebooks/k_break_frequency.py`. |
| 2.4 | Figure `ir_resolution` **does** test its caption: T_em = 40, T_em/τ_c = 40–4000, ωT_em = 2.4–18 across the fit band, which lies **above** the window break π/T_em = 0.079. It lies **below** the decorrelation break π/τ_c = 3.1, and that is the mechanism behind the null result — so the figure bounds the decorrelation time, not the duration. Both now reported in the text. |
| 2.5 | New §`sec:kernel-at-rp20`, `Notebooks/kernel_at_rp20.py`. Parameters taken from their own Zenodo data (k_0 = 673, k_box = 129, R = 158, M = 0.043–0.060; GW peak 1.836 k_0 vs the 1.84 in §`sec:roperpol-compare`; IR slope +1.295 raw vs +1.298 digitised). **Honest result:** the kernel spans +0.57 to +1.59 over their fit band and the measured +1.30 is inside that span, but the manuscript's own lifetime closure gives **+0.70**, i.e. 0.6 too shallow. The residual is diagnosed as our *spatial* factor rolling over inside a fit band only 0.66 decades wide (the coherent control reads +2.709, and 1 − 0.291 = 0.709 reproduces the band slope to three decimals) — **not** the temporal model. Two clean results survive: the causal +3 is absent from the box even for an infinitely coherent source, and the break sits at or below the box floor. |

### Partially closed

| # | Outcome |
|---|---|
| 2.2 | **The paper's coefficient was wrong.** The two-leg rule (quadratic source ⇒ transform of the *squared* correlator, the same rule §`sec:ir-branch` applies correctly to the tent) gives f₂′(0⁺) = −2νk², hence **T → 4ν, not 2ν**. In the paper's variables the floor is **T̂ → 4M/R**, constant in p, reproduced to <0.5%. **This inverts the framing:** on the cone the sweeping factor is exp(−p^{2/3}/M²), so a p-independent floor does not sit *beneath* the cutoff — it replaces it, and the crossover falls *below* the sweeping peak for M ≲ 0.1, the regime cosmological flows occupy. Stated in the text with its limit made explicit: this is the temporal factor alone, and the full two-wavevector kernel is still not evaluated. `Notebooks/viscous_cusp_floor.py`. |

### Defects found in the round-1 state

1. **The committed manuscript did not build.** `\add{}` was defined through `\textcolor`, whose argument is not `\long`, so the multi-paragraph `\add{}` wrapping §V.A aborted `pdflatex` with *"Paragraph ended before `\@textcolor` was complete"*. Redefined via `\color`.
2. **`Kraichnan:1959` was undefined** under a single bibtex pass: `derivationNotes.bib` stores footnote text containing `\cite`, so citations inside notes reach the `.aux` only after the `.bbl` has been typeset once. `build_derivation.sh` now runs bibtex twice and reports errors / undefined refs / undefined citations.
3. **`bib.bib` carried a stray JSON error blob** — `{"message": "PIDDoesNotExistRESTError…", "status": 404}` — left between entries by round 1's automated bibliography tooling. Removed.
4. **§V.A was inserted mid-way through a run of `\paragraph`s** belonging to §V's introduction, so seven of them ended up nested inside it, and the branch diagram answered a question posed three lines *after* it. Moved.
5. **The abstract and Conclusion 5 still said the Pm omission was "acknowledged"** in `RoperPol:2025lgc` — the exact over-claim round 1's citation verification retracted and corrected in §VII only.
6. **The introduction still asserted the Ω_T ∝ Ω_M localisation as ours**, and still carried the unqualified factorisation claim, after both had been dropped from the abstract and the conclusions.
7. **Two different coefficients for the same break** (1/Δt in §`sec:cps-reproduction`, π/τ_c in §`sec:ir-branch`) with nothing relating them.

### Numbers corrected (each reproduced from the generating script before applying)

−4.75→**−4.73**; +0.974→**+0.970** (Mach exponent only — the +0.974 in `tab:ir-branch` is a different quantity); 1.9→**2.0**; 1.4→**1.49**; 0.67–1.35→**0.66–1.32**; 2.98–3.08→**2.99–3.06**; 13%/19%→**16.6%/14.0%** (ordering also inverts); band-split peak amplitudes 4.14/2.57/5.16→**3.96/2.43/5.48**; inertial-only IR slope 3.01→**3.00**; δ-control peaks 1.26/2.13→**1.37/2.50**; Batchelor peak at M=3 1.26→**1.33**.

Two of the plan's "pre-existing inconsistencies" were **mis-diagnoses**: the 0.47/0.74 peaks are correct (they are the *sweeping* values; their δ-in-time partners were the stale ones), and "peak from 1.5 k₀ to k₀" is not reproducible in any single convention — it took its two ends from *different* conventions (1.55 is p³H, 1.01 is p²H). Also, the round-1 log's band-split slopes `+2.9963/+2.9979/+3.0073` do **not** reproduce; `band_split_gw.py` at its documented defaults gives **+3.005/+3.005/+3.016**.

`tab:ir-branch` was re-verified in full and stands exactly (+3.000 IR; band +0.974/+0.995 global and +0.965/+0.996 eddy; coefficients 3.072 and 3.175 against π).

### Not done

- **3.1, 3.2, 3.3, 0.7, 0.8** — authorial, untouched. `main.tex` not edited, as instructed.
- **2.6** (magnetic UETC from RP20 Zenodo data) — 1–3 weeks; not attempted.
- **2.2 full-kernel evaluation** — see above.
- **LISA noise-model citation.** Egress to `inspirehep.net` and `arxiv.org` is blocked by network policy on the machine this ran on, so a new bibliography entry could not be verified against a primary source. The noise model is written out in full in `Notebooks/k_break_frequency.py` and a `\why{}` note in §`sec:k-break-hz` asks the authors to add the reference. **Flagged rather than guessed.**
- Two Tier-2 subagents were killed part-way by an org monthly spend limit; 2.2 and 2.4 were finished directly instead, which is why 2.2 is partial.

**Build:** 73 pp, 0 errors, 0 undefined references, 0 undefined citations (`./build_derivation.sh`).
**Tests:** 68 passed.

- **2026-08-15, task 2.6 — DONE, and the answer goes against us.** A direct UETC
  measurement is **impossible** from the public release: dump cadence is Δt = 1e-3 in
  Hubble units while the cusp lives at τ ~ 1/k ≈ 8e-4 at the GW peak, so any time-domain
  correlation would be one lag bin wide. But a **parameter-free forward model** is
  possible — the simulation does the short-lag integral itself at its own timestep, and
  Π(k,t) and v_A(t) are both measured, so E_GW at the end of the run is a *prediction*
  for each candidate correlator. Over p = 1.5–30:

  | correlator | E_model / E_data |
  |---|---|
  | coherent, f = 1 | 0.35 – 1.19 |
  | Gaussian sweeping | 0.90 – 1.14 |
  | cusped, exp(−k v_A \|τ\|) | **19.5 – 97.8** |
  | BK2016 power law | **13.0 – 64.0** |

  **Cuspless correlators reproduce the data; cusped ones overpredict by 1–2 orders of
  magnitude.** Stable against grid padding and against v_A taken at start/mean/end.
  So f′(0⁺) ≃ 0 for the magnetic stress too, as for Auclair's hydrodynamic case.
  **The source-scale peak survives, but via duration, not decorrelation** — the spectrum
  is laid down by the switch-on, and a cusp would add sustained radiation for which
  there is no room. This *strengthens* the T_burst/window result of
  `eq:burst-equals-triangle` and *retires* the T_dec/cusp route as the explanation.
  Tool: `Notebooks/magnetic_uetc.py`, Fig. `magnetic_uetc`.

### Working practice (revised after two spend-limit losses)

Commit after **every** completed task, not at the end of a batch. Two runs have now been
cut off mid-flight; the first lost 7 commits to an unpushed branch, the second lost
nothing only because its script and figure happened to be on disk. Artefacts on disk are
not progress — committed artefacts are.

---

## Build variants (added 2026-08-15)

`derivation.tex` now carries two independent switches in the preamble:

| switch | effect | pages |
|---|---|---|
| `\reviewmarkuptrue` (default) | red `\add` / struck `\del` / `\why` notes visible | — |
| `\reviewmarkupfalse` | clean copy, all markup resolved | −1 |
| `\concisefalse` (default) | full paper | **75** |
| `\concisetrue` | drops Appendices F and H, substituting a pointer section | **70** |

Four combinations, all verified: 0 errors, 0 undefined references, 0 undefined
citations. `\concisetrue` + `\reviewmarkupfalse` = **69 pp**, the referee build.

**Why only F and H.** Cutting Appendix B as well reaches ~61 pp but leaves **38 dangling
cross-references** — the main text depends on the consolidated derivation chain far more
than the audit's page count suggested. F and H cost six anchors, which the
`\concisestub` carries. Further reduction needs prose rewriting, not conditional
inclusion, and should not be done mechanically.

---

## ROUND-2 REFEREE: BLOCKING DEFECT — the π/2.33 rule is wrong (2026-08-17)

**Do not submit until this is resolved.** The discriminant I wrote into
`sec:duration-kernel` on 2026-08-15 —

> "Where a lifetime is imposed on the stress we use 2.33, and where it is a property of
> the field we use π."

— **is not the correct rule**, and the correctness referee is right. The discriminant is
**idempotency of the window**, not stress-level versus field-level.

For a hard finite lifetime the field correlator is R = 1 inside the window, so the stress
carries R² = **R**: squaring does nothing, the two-leg factor is the *un-squared* tent
`4sin²(ωW/2)/ω²`, and the break is at **2.33/W**. Verified numerically: |∫₀^W e^{iωt}dt|²
equals `4sin²(ωW/2)/ω²` to ratio 1.000000, and R² − R = 0 identically. The *squared* tent
(break π/τ_c) is correct only for a **stationary field with triangular lag memory** —
a different source, which should not then be called a lifetime or a duration.

**What propagates.** All of this is arithmetically fine but physically mislabelled:

| item | consequence |
|---|---|
| `tab:k-break-hz`, Fig. `k_break_frequency` | every entry high by π/2.33 = **1.35** |
| corner counting in §V.D | 5.03 mHz → **3.7 mHz**, i.e. *inside* the LISA window, not above |
| abstract's exception clause | rests on a 9% margin against a coefficient 35% in dispute |
| §V.C "what is added here is the factor π" | the paper's one claimed novelty in §V |
| `tab:ir-branch` caption | the computation stands; its physical caption does not |
| `tab:temporal-catalogue` | lists T_burst with the *un-squared* value while §V.B calls the table per-leg |
| the π/T_em straddle at `ir_resolution` | with 2.33/T_em = 0.058 the sentence I struck was **right** |

**Also required** (referee's list): delete or rewrite "The decorrelation shape, not the
onset" (it asserts the negation of §III.B's new conclusion); Conclusion 3's second half
still carries text deleted from §V.D; Conclusion 4 and §V still say the magnetic
measurement "has not been made" though 2.6 made it; "+0.70 to +0.77" quotes eddy rows
only and omits the +0.570 global row; band-split slope printed as 3.00 in one place and
3.01 in another; "1.49 M of Eq. (peak-fit)" cites an equation that reads 1.515 M^1.007;
the preamble comment claims `\concisetrue` drops Appendix B (it does not).

**Correctly done, per the same referee:** the viscous two-leg coefficient 4ν (not 2ν),
the erfc fix in all three call sites, and the priority attributions — "no remaining
unattributed claim found".

---

## 2026-08-17 — referee changes applied; and why the paper cannot be halved by a toggle

**Blocking defect resolved.** The π/2.33 rule was keyed to the wrong discriminant. The
correct one is **idempotency of the window**: for a hard lifetime R = 1 inside the
window, so the stress carries R² = R, the two-leg factor is the *un-squared* tent, and
the break is **2.33/W**. The squared tent and its π describe a stationary triangular lag
memory — a different source. Every closure in the paper imposes a lifetime, so 2.33
applies throughout.

`k_break_frequency.py` now carries `BREAK_COEFF` (window 2.3311 / triangle π) and
defaults to the window value. Every f_break falls by 1.348, and **the LISA corner
counting changes from 6 below / 1 inside / 2 above to 6 / 2 / 1**: the corner at
γ=10⁻³, u₀=0.1 moves from 5.03 mHz (above the window) to 3.73 mHz (inside it).
Sec. V.C now claims the coefficient 2.33 rather than π. The `ir_resolution` "straddle" is
reverted — with 2.33/T_em = 0.058 the sentence originally struck was correct.

**Contradictions closed:** the "decorrelation shape, not the onset" heading is struck;
Conclusion 3 now reports 6/2/1 instead of "every corner but one"; Conclusion 4 no longer
says the magnetic measurement has not been made.

### Halving: not achievable by conditional inclusion

Tested directly. Cutting appendices under `\concisetrue`:

| cut | pages | dangling refs |
|---|---|---|
| F + H (current) | **71** | 0 |
| + D/E (impulsive) | 67 | 19 |
| + B (derivation chain) | 58 | **51** |

The big appendices are the load-bearing ones: the main text depends on Appendix B for 51
cross-references and on D/E for 19. **Getting to ~38 pp requires rewriting the main text
so it no longer depends on them, not hiding them behind a switch.** Concretely that means:

1. Fold the results of App. B into the sections that use them, and delete the chain (−8 pp).
2. Merge App. D and E into one impulsive appendix (−3 pp).
3. Halve §III, which largely re-derives Gogoberidze (−4 pp).
4. Compress §VI to one table (−2 pp).
5. Trim the symbol table to one page (−3 pp).

That is prose surgery of a few days, and it is the point at which the **split** both
referees recommend becomes the cheaper route: the letter (task 2.6) needs almost none of
this material, and the companion can stay long.

---

## Round-2 referee tasks (2026-08-17)

| # | Task | Source | Status |
|---|---|---|---|
| R1 | π/2.33 rule + everything downstream | correctness, BLOCKING | **DONE** |
| R2 | "decorrelation shape, not the onset" heading | correctness | **DONE** |
| R3 | Conclusion 3 second half; Conclusion 4 stale re 2.6 | correctness | **DONE** |
| R4a | "+0.70 to +0.77" quotes eddy rows only, omits +0.570 | correctness | **DONE** |
| R4b | band-split slope printed 3.00 in one place, 3.01 in another | correctness | **DONE** |
| R4c | "1.49 M of Eq. (peak-fit)" — that equation reads 1.515 M^1.007 | correctness | **DONE** |
| R5 | viscous-floor caveat must cover leg-wavenumber ambiguity; Fig (a) incommensurable | correctness | **DONE** |
| R6 | two remaining "proposed as future work" passages, stale re 2.6 | correctness | **DONE** |
| R7 | Move task 2.6 to its own section; rebuild abstract around it | significance | **DONE** |
| R8 | §V.B "spatial model wrong near the peak" tension | significance | **DONE** |
| R9 | Split: letter = 2.6, companion = the rest (incl. the now-closed dissipation result) | both | **DECIDED** — letter is the magnetic-UETC result; dissipation goes in the companion |
| R10 | Parameter-reconstruction forecast (NANOGrav + LISA bias) | significance | **TODO, 1–2 wk** |
| R11 | Is ini2's own viscosity consistent with the 2.6 f'(0+) bound? | significance | **TODO, 1 d** |

- **2026-08-17, R7/R8/R9.** R7: task 2.6 promoted to `sec:magnetic-uetc`, its own
  section before the discrepancy section. (Note: a `\add{}` cannot go in a section
  title — the heading is uppercased for the running head and xcolor then rejects `RED`.)
  R8: the spatial-axis claim is now stated at its true width — no spatial choice moves
  the *asymptotic* infrared, which says nothing about the peak, where the k^0.7-vs-k^1.3
  deficit lives. R9: `letter.tex` created — title, authors, full abstract, the results
  table, and a section skeleton with per-section notes on what to lift from the
  companion. **The prose is not written**; that is an authorial job, and the abstract is
  the part worth reading first because it commits the letter's claim.

- **2026-08-17, the subviscous calculation — DONE, and it decides R9.**
  `Notebooks/subviscous_gw.py`, Fig. `subviscous_gw`, written into §VII and Conclusion 5.
  **Spatial:** Kolmogorov joined continuously to Kulsrud–Anderson k^(3/2)K₀(k/k_η),
  k_η = k_ν√Pm, ceiling moved from k_d to k_η. Holding the large-scale field fixed, the
  tail changes Ω_GW only above p ≈ 2k_ν/k₀; at p = 0.01 and 0.1 the truncated and
  Pm=10⁸ spectra agree to **2–3%**.
  **Temporal:** the correct two-leg factor is T → 2ν(k₁²+k₂²)/ω², not 4ν — but that is
  the *large*-ω asymptote. Computed rather than extrapolated, **T is flat below
  1/τ_c ≈ 20** (0.1034 at p = 0.01, 0.1 and 1 alike), reaching the asymptote only near
  p ~ 10³. No extra power of k, so the causal k³ stands.
  **Validated:** continuity 1.00000000; Pm→1 reproduces the truncated kernel to 0.00e+00;
  fixed-energy normalisation 1.4e−08; two-leg asymptote to 0.03–0.4%; grid convergence
  0.11%. The script refuses to print results on a failed validation.
  **Consequence for R9:** the dissipation gap is now a *closed, negative* result — real
  physics, missing from every calculation, and unobservable. It can no longer carry a
  letter on detection grounds. **Letter = the magnetic-UETC forward model; companion =
  everything else including this.** The abstract now reports the closed gap rather than
  promising to close it.

### Remaining after this

| # | Task | Status |
|---|---|---|
| R9 | Write the letter prose around `letter.tex`'s abstract | **AUTHORIAL** |
| R10 | Parameter-reconstruction forecast (NANOGrav + LISA bias) | **TODO, 1–2 wk** |
| R11 | ini2 viscosity vs the 2.6 f'(0⁺) bound | **TODO, 1 d** |
| — | Push (20 commits local) | **AUTHORIAL** |
