# Project status — 2026-08-17

One-page answer to "where is this and what is left". Detail lives in
`ACTION_PLAN.md` (task log) and `REVIEW_2026-08-14.md` (round-one review).

---

## Where the project is

**Two rounds of review are complete and answered.** Round one (five agents: prior-art,
citation audit, three referees) found the abstract inverted — two of its headline claims
were already published. Round two (two referees on the revised text) returned **major
revision** from both, with one blocking defect that is now fixed.

| | |
|---|---|
| `derivation.tex` | **76 pp** full, **71 pp** concise; 0 errors, 0 undefined refs/cites |
| `letter.tex` | skeleton + full abstract, builds; **prose not written** |
| `main.tex` | 2 pp; empty abstract, no sections past the intro |
| `bib.bib` | **88 entries**, all INSPIRE/Crossref verified |
| tests | **68 passed**, 0 failed |
| git | `main` clean, **19 commits ahead of origin** — not pushed |

### Build switches in `derivation.tex`

| switch | effect |
|---|---|
| `\reviewmarkuptrue` (default) | red `\add` / struck `\del` / `\why` notes |
| `\reviewmarkupfalse` | clean submission copy |
| `\concisetrue` | drops Appendices F and H (76 → 71 pp) |
| `\refcommentstrue` (default) | boxed round-2 referee comments, red = blocking |

---

## What the science now says

The paper is **narrower and more negative** than it was two rounds ago, and that is the
correct direction.

- **Conceded, not claimed:** the white-noise stress floor (Brandenburg & Boldyrev 2020),
  the k³→k¹ break at the inverse source duration (Roper Pol 2022, 2024), the Abelian
  ω⁻² tail and its peak consequence (Caprini et al. 2009), the causal k³ theorem
  (Cai–Pi–Sasaki), the UV-from-temporal framework (Rubinstein & Zhou 2000). Round-two
  referee: *"no remaining unattributed claim found."*
- **Genuinely new:** the parameter-free magnetic-UETC forward model (§`sec:magnetic-uetc`);
  the break coefficient 2.33; "the causal +3 is not in the box"; the viscous cusp
  coefficient 4ν; the band-split as the sharpest test of a known theorem.
- **Refuted by our own test:** the cusp mechanism. Cuspless correlators reproduce the
  radiated energy of the one run we can address; cusped ones overpredict by 1–2 orders of
  magnitude. The source-scale peak survives but is a **window** effect, not a
  decorrelation effect.
- **Still uncomputed:** the Pm ≫ 1 subviscous range. This is the only claim no one can
  pre-empt (INSPIRE full-text for *Prandtl* ∧ *gravitational wave* still returns zero) and
  the paper says plainly it is "a programme rather than a result".

---

## What needs to be done

### Decisions only the authors can make

1. **R9 — split or not.** Both referees recommend letter (the magnetic-UETC result) plus
   long companion, and **against** holding for the subviscous calculation. This decision
   gates everything else: splitting makes the length problem disappear, not splitting
   means restructuring a 76-page document.
2. **Does the letter's abstract overstate?** `letter.tex` claims the magnetic correlator
   is "no more cusped than the hydrodynamic one" from **one Pm = 1 run with an imposed
   initial field**. If that is too strong, it is much cheaper to find out now than after
   four pages are written around it.
3. **Push.** 19 commits sit unpushed.

### Work with a clear specification

| # | Task | Effort |
|---|---|---|
| R10 | Parameter-reconstruction forecast: fit a QCD source under k³ and under k¹, report the bias in (T\*, τ_c, Ω_M, γ) against the NANOGrav 15 yr posterior and our LISA curve. **Highest impact** — without it, PTA analysis papers cite Roper Pol rather than us. Every ingredient is built. | 1–2 wk |
| R11 | Is `ini2`'s own viscosity consistent with the \|f′(0⁺)\| bound the forward model yields? The only calculation joining our one unpre-emptable claim (§VII) to our one new measurement. Both tools written. | 1 d |
| — | The subviscous two-wavevector integral: replace 𝒮(k) with one continuing past k_d. Novelty-rich, impact-uncertain — our own Kulsrud–Anderson caveat suggests it may come out negligible. | 2–4 wk |
| — | `main.tex`: abstract and sections. | authorial |

### Known limitations, stated in the text, not defects

- The forward model uses the same Gaussian closure as the rest of the paper.
- It excludes a cusp but does **not** separate coherent from Gaussian sweeping, so it
  bounds rather than measures the correlator.
- One run, Pm = 1, R = 158, imposed initial field — the switch-on is a property of the
  setup, not of a phase transition.
- The viscous floor is the temporal factor alone; with outer-scale legs it would go as
  4M/(Rp²), so the p-scaling is undetermined until the full kernel is done.
- Halving the paper is **not** achievable with the concise switch: the main text depends
  on Appendix B for 51 cross-references and D/E for 19. Getting to ~38 pp means rewriting
  the sections that cite them.

---

## For whoever picks this up next

Read `ACTION_PLAN.md` first, then build with markup on and read the boxed referee
comments — the blocking one at §V.B is the single most instructive thing in the file.

Two habits this project earned the hard way: **reproduce every number before quoting it**
(five separate edits and two agent-reported numbers had to be reverted across the two
rounds), and **commit after each completed task** (two cloud runs were cut off by spend
limits; the first lost seven commits).
