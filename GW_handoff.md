# Handoff — Pencil Code GW solver investigation

**Deliverable:** `GW_solver_workflow.md` (repo root, ~1050 lines, untracked). Complete
technical doc on how the Pencil Code solves the GW equations, plus the physics needed
to interpret its output and how it relates to the analytic literature. Read it before
continuing; this file is only the compressed state.

**Working dir:** `/home/mgurgeni/pencil-code`, branch `master`, git clean apart from
two untracked files (`GW_solver_workflow.md`, `GW_handoff.md`). Nothing in the repo
was modified.

---

## 1. Established facts about the code (all verified against source)

**Module:** `src/special/gravitational_waves_hTXk.f90` (production). Alternatives:
`gravitational_waves.f90` (MVAR 6, real-space `del2` + RK3),
`gravitational_waves_hij6.f90` (MVAR 12, same). Selected via `SPECIAL=` in
`src/Makefile.local`.

**Method — hybrid, not what either naive guess suggests:**

1. Quadratic stress `T_ij` built in **real space** on the same pencils as the MHD
   (`calc_pencils_special`, line 963), gated `if (lfirst)` → first RK substep only.
2. `dspecial_dt` (1165) deposits `6/a · T_ij` into aux slots. **It never touches
   `df`** — the GW sector is entirely outside the Runge–Kutta integrator.
3. `special_after_timestep` (1390), also gated on `lfirst`, calls
   `compute_gT_and_gX_from_gij` (3086): one forward FFT of all 6 components.
4. `solve_and_stress` (2523), per k-mode: TT projection `Λ = P_ip P_jq − ½P_ij P_pq`,
   decomposition onto `e⁺/e^×`, then an **exact analytic propagator** with
   `ω² = k² − a″/a`, source frozen over `Δt`:
   `h(t+Δt) = (h − S/ω²)cos ωΔt + (g/ω)sin ωΔt + S/ω²`. `cosh/sinh` branch when
   `ω² < 0`. **No spatial differencing, no RK, hence no GW CFL constraint.**
5. Inverse FFT only if `lreal_space_*_as_aux` requested — otherwise `h` never exists
   in real space during the run.

`h,g` live permanently in k-space as **MAUX** slots; `f(nghost+ikx,…,ihhT)` is a
k-space index triple despite looking like a grid access.

**Known source defect (unreported upstream):** call at line 3242 passes
`(f,S_T_re,S_X_re,S_T_im,S_X_im,dt)` against dummy list at 2523
`(f,S_T_re,S_T_im,S_X_re,S_X_im,dt)` — args 3/4 transposed. Currently harmless (all
four are scratch, zeroed on entry, results go to `f`), but a trap for anyone reading
`S_T_im` back in the caller.

---

## 2. Physics conclusions reached

- `T_ij(x)` is genuinely a **one-point** object; GR field equations are local. Correct
  as coded.
- Two nonlocalities exist, both handled: **retardation** is a property of the wave
  *solution* (the `cos ωΔt`/`sin ωΔt` propagator *is* the retarded Green's function
  mode by mode); the **TT projection** is genuinely nonlocal in real space and is the
  actual reason the module uses FFTs — exact in a periodic box.
- **GW energy is irreducibly two-point** (not localizable; Isaacson/Brill–Hartle).
  Enters *only* in diagnostics: `EEGW` is the correlator at zero separation (Parseval,
  forward FFT carries `1/nwgrid`, `fourier_fftpack.f90:242`); the `GWs/GWh/Str` spectra
  *are* the correlator by Wiener–Khinchin. Volume average over the box = the Isaacson
  average, licensed by ergodicity.
- **No UETC anywhere** — verified three ways: not a solver input; the one stored past
  step (`StT/StX…`) feeds only `ΔS/Δt` for `itorder_GW=2`, a two-time *difference* not
  a *product*; diagnostics are equal-time. Recoverable in post-processing from
  high-cadence dumps of `S_T/S_X` (already exposed as slices, line 3504).

---

## 3. Literature comparison (GKK07 vs RP20)

GKK07 = Gogoberidze/Kahniashvili/Kosowsky, PRD 76, 083002 (arXiv:0705.1733).
RP20 = Roper Pol et al., PRD 102, 083512 (arXiv:1903.08585).

**The single root cause of every disagreement.** User supplied RP20's slope table:

| slope of | ana | sim | Kol | Gol |
|---|---|---|---|---|
| `Ω_M`  | 5 | 5 | −2/3 | −8/3 |
| `Ω_GW` | 3 | 1 | −8/3 | −14/3 |
| `h_c`  | 1/2 | −1/2 | −7/3 | −10/3 |

Two rules reproduce every cell: `Ω_GW = Ω_T − 2` and `h_c = (Ω_GW − 2)/2`. Note
`Ω_M = 5` in **both** columns (same Batchelor field — disagreement is *not* about the
turbulence) and `Kol/Gol` carry single values (inertial range: the approaches agree).

Conflict is one cell-pair: subinertial `Ω_GW`, **3 vs 1**. Both use `Ω_GW = Ω_T − 2`,
so they differ on the **stress** spectrum: `ana` assumes `Ω_T ∝ Ω_M ∝ k⁵`; sim finds
`Ω_T ∝ k³` (white noise). Cause: `T ~ B²` is **quadratic**, so its spectrum is the
magnetic spectrum **self-convolved**; below the peak the integral is dominated by
`k₁ ≈ −k₂` near the peak and floors at white noise regardless of how steep `Ω_M` is.
**Holds even for Gaussian `B`** (Wick: `⟨B²B²⟩ = 2⟨BB⟩²` *is* that convolution) — so
Gaussianity is *not* the culprit; the lone bad step is `Ω_T ∝ Ω_M`.

**Peak location.** GKK07: `~M·k_*`, set by eddy turnover time, via the aeroacoustic
substitution `H_ijij(k=ω,ω) → H_ijij(k=0,ω)` which deletes spatial-wavenumber
dependence. RP20: `2k_*`, because free GWs obey `ω = k` and the stress peaks at `2k_*`.
Ratio `~2/M`. The **factor 2 is kinematic** — the upper support edge of the same
self-convolution that produces the IR floor; duration-independent, universal across all
(quadratic) source types in the module. Depends on source spectral *width*, and the
`1/k²` weighting pulls the GW peak slightly below the stress peak. Duration enters only
via `k_*(t)` drifting under inverse transfer (`k_GW(t) ≈ 2k_*(t)` holds instantaneously).

**Discriminant:** simulated peak is sensitive to *decay history*; GKK07's to *Mach
number*. A run series varying `M` separates them by ~an order of magnitude in peak
frequency — sharper than amplitude comparison, which is entangled with a normalization
correction (arXiv v5 note: "corrected normalization error").

**`k¹→k³` break** at `k_break ≈ 1/Δt_source` (impulsive vs oscillatory branch of the
same propagator; both carry the same white-noise stress). For `samples/GravitationalWaves`
this is `k ≈ 1–10` vs box fundamental `k₁ = 100` — the `k³` regime is **absent, not
under-resolved**. Reaching it needs boxes 100–1000× larger. Hence universal `k³` must be
imported analytically (Caprini/Durrer, arXiv:1909.13728), never measured.

**Frequency bands where it matters** (EW source, `k_* ~ 100 H_*`): worst in
**0.15–3 mHz**, where GKK07 is falling as `f^(−13/4)` while RP20 still rises as `f¹` —
*opposite slopes*, not a normalization quibble. That is LISA's most sensitive decade.

---

## 4. Corrections already made — do not re-introduce

- Inertial range (Kolmogorov) is `Ω_GW ∝ f^(−8/3)` ⟺ `h_c ∝ f^(−7/3)`. An earlier
  claim of `h_c ∝ f^(−13/6)` was wrong and is purged from the doc (verified: no
  `13/6` remains). The user's table settled it.
- Always state which convention a slope is in; `Ω_GW = (2π²/3H₀²) f² h_c²`.

---

## 5. Open / unverified — flag rather than assert

- Quoted RP20/GKK07 numbers come from **HTML renders and abstracts**, not line-by-line
  PDF reads (PDF fetch returned compressed binary). Weakest: GKK07's `f^(−13/4)` slope
  and `exp(−2ω²/k₀²M²R)` cutoff.
- The `2/M` peak-ratio factor, the aeroacoustic-substitution diagnosis, and the
  resonance reading of the 200× acoustic-efficiency result are **synthesis**, not
  quoted. RP20 never diagnose the peak mismatch explicitly — they fold amplitude and
  frequency into one sentence (`0.7×10⁻²¹` @3 mHz vs `4×10⁻²⁰` @1 mHz).
- Unknown whether a measured UETC has been published for these MHD runs.
- Doc has a section-numbering wart: `3.10a` inserted after `3.10`. Cosmetic.

**Style the user expects:** verify in source before asserting; mark synthesis vs.
quotation explicitly; use `file.f90:line` markdown links; correct errors plainly
without ceremony.
