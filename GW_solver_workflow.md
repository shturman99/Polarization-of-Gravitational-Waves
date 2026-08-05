# How the Pencil Code solves the gravitational-wave equations

**Module:** `src/special/gravitational_waves_hTXk.f90`
(the production module used for early-universe MHD → GW runs; the two alternatives are
compared in §11)

**One-line answer to "real space or Fourier space?"**
The *source* is built in real space, on exactly the same pencils as the MHD.
The *wave equation* is then solved in Fourier space, **mode by mode, with an exact
analytic propagator** — no finite differences in space, no Runge–Kutta in time.
The strains `h` never exist in real space during the run; they live permanently as
k-space arrays in the `f`-array and are inverse-transformed only for output.

---

## 1. The equations, in the order the code applies them

### 1.1 The physical problem

Linearised Einstein equations in a spatially flat FRW background, in conformal time
`t` and comoving coordinates, for the transverse-traceless metric perturbation:

```
  h̄''_ij  +  ( -c² ∇²  -  a''/a ) h̄_ij  =  (16πG/(a c⁴)) T^TT_ij
```

where `h̄_ij = a·h_ij` and `'` = d/d(conformal time). The code works with
normalised variables in which the whole right-hand-side prefactor collapses to a
single number `stress_prefactor` (default **6**), so what is actually integrated is

```
  ∂²h/∂t²  +  ( k² - a''/a ) h  =  S                              (★)
```

for each of the two polarisations, at each Fourier mode **k**, independently.

### 1.2 Step-by-step chain

| # | Quantity | Space | Where |
|---|---|---|---|
| 1 | `T_ij(x)` from `u`, `B`, `E`, `∇φ` | real | `calc_pencils_special` |
| 2 | `S^raw_ij(x) = (6/a)·T_ij(x)` | real | `dspecial_dt` |
| 3 | `T̃_pq(k) = FFT[S^raw_pq]` | → Fourier | `compute_gT_and_gX_from_gij` |
| 4 | `S̃_ij = Λ_ij,pq T̃_pq` (TT projection) | Fourier | `solve_and_stress` |
| 5 | `S_T, S_X` (+ / × amplitudes) | Fourier | `solve_and_stress` |
| 6 | advance `h_T,h_X,g_T,g_X` by exact propagator | Fourier | `solve_and_stress` |
| 7 | optional inverse FFT for output only | → real | `compute_gT_and_gX_from_gij` |

---

## 2. Step 1 — the stress tensor, in real space

**Location:** [`calc_pencils_special`, line 963](src/special/gravitational_waves_hTXk.f90#L963)

This runs inside the ordinary Pencil `m`/`n` loop, using the *same* pencils
`p%uu`, `p%bb`, `p%rho` that the momentum and induction equations use in that
same sweep. This is the only place where the GW module touches real-space fields.

```
  T_ij  =   (4/3) ρ γ² u_i u_j        (Reynolds,  if lreynolds)
          -  B_i B_j                  (Maxwell,   if luse_mag)
          -  E_i E_j                  (electric,  if lelectmag)
          +  a² ∂_i φ ∂_j φ           (scalar,    if lscalar_phi)
          -  δ_ij · trace_factor · ( ... )       (trace removal)
```

**Every term above is evaluated at one and the same grid point and one and the same
time.** See §3 for why that is correct GR and where the `|x−y|` nonlocality you might
expect actually lives.

Code, verbatim:

```fortran
if (lreynolds) p%stress_ij(:,ij) = p%stress_ij(:,ij) + p%uu(:,i)*p%uu(:,j)*prefactor*p%rho
if (luse_mag)  p%stress_ij(:,ij) = p%stress_ij(:,ij) - p%bb(:,i)*p%bb(:,j)
if (lelectmag) p%stress_ij(:,ij) = p%stress_ij(:,ij) - p%el(:,i)*p%el(:,j)
```

with

```fortran
prefactor = fourthird_factor/(1.-p%u2)     ! if lgamma_factor=T  →  (4/3)γ²
prefactor = fourthird_factor               ! otherwise           →   4/3
```

The `4/3` is `ρ+p` for a radiation fluid (`p = ρ/3`). Note the deliberate **opposite
signs** for `u` and `B` (the code comments on this). Only the 6 independent
components are stored, indexed through `ij_table`:

```
  ij_table:   (1,1)→1  (2,2)→2  (3,3)→3  (1,2)=(2,1)→4  (2,3)=(3,2)→5  (3,1)=(1,3)→6
```

**Trace removal.** On the diagonal the code subtracts `trace_factor × (u², b², e²)`
with `trace_factor = 1/3` by default (`ctrace_factor='1/3'`). This is cosmetic:
the TT projection in step 4 annihilates any `δ_ij` piece anyway, so the result is
independent of `ctrace_factor`.

**Relativistic / conservative branch.** If `lconservative=T` (relativistic bulk
motion in `hydro`), the Reynolds part is *not* rebuilt here — the full `T_ij` is
read straight from the `Tij` slots of the `f`-array:

```fortran
if (lconservative) p%stress_ij(:,ij) = p%stress_ij(:,ij) + f(l1:l2,m,n,iTij-1+ij)
```

**Optional time modulations** applied at the end of the same routine:
`lstress_ramp` (linear ramp over `tstress_ramp`), `lturnoff` (hard zero after
`tturnoff`), `lstress_upscale` (power-law growth).

> **Crucial gate:** the whole body is wrapped in `if (lfirst)`, i.e. it executes
> **only on the first of the three Runge–Kutta substeps**. On substeps 2 and 3 the
> stress is not recomputed.

---

## 3. Locality: why `T_ij` is a one-point object, and where `|x−y|` really enters

This section exists because the pointwise definition above looks, at first sight,
incompatible with the retarded-integral picture of GW generation. It is not.

### 3.1 The field equations are local

```
  G_μν(x)  =  8πG T_μν(x)
```

Both sides are evaluated at the **same event**. `T_μν` is a *field*, constructed
algebraically from other fields at that point:

```
  T_ij(x) = (ρ+p) γ² u_i(x)u_j(x) - B_i(x)B_j(x) - E_i(x)E_j(x) + p δ_ij + …
```

There is no `x−y` anywhere in the definition, and there should not be. The linearised
equation is likewise a local PDE:

```
  □ h̄_ij  =  −16πG T_ij
```

So `p%uu(:,i)*p%uu(:,j)*p%rho` — an element-wise product on a single pencil — is the
faithful discretisation.

### 3.2 Nonlocality #1: retardation — handled by the wave operator

The familiar expression

```
  h_ij(t,x)  ∝  ∫ d³y  T_ij( t − |x−y|/c , y ) / |x−y|
```

is the **solution** of the wave equation, not the equation. The code never imposes
it; it integrates the PDE, and retardation emerges from the dynamics.

In fact the propagator of §7,

```
  h(t+Δt) = (h − S/ω²) cos ωΔt + (g/ω) sin ωΔt + S/ω²
```

with `ω = k`, **is** the retarded Green's function of `□`, written mode by mode:
`cos(kΔt)` and `sin(kΔt)/k` are precisely the Fourier transform of the light-cone
kernel `δ(t−|x−y|)/|x−y|`. The retarded integral is being evaluated — in the basis
where it is diagonal rather than a convolution.

### 3.3 Nonlocality #2: the TT projection — genuinely nonlocal, and the reason for the FFT

In real space the transverse-traceless part of the source is a convolution with a
long-range kernel:

```
  h^TT_ij(x)  =  ∫ d³y  Λ_ij,pq(x−y)  T_pq(y)
```

with `Λ_ij,pq = P_ip P_jq − ½ P_ij P_pq`, `P_ij = δ_ij − k̂_i k̂_j`. Because `Λ`
depends on the *direction* `k̂`, its real-space kernel does not fall off in any
convenient local way. **This is the actual reason the module works in Fourier
space.** There the convolution collapses to a per-mode matrix multiply (§6.2), and
in a periodic box it is *exact*, not an approximation — the discrete Fourier modes
are exact eigenfunctions of the projector.

Equivalently: one may either solve `□h̄_ij = −16πG T_ij` with a **local** source and
project afterwards, or solve `□h^TT_ij = −16πG Λ_ij,pq T_pq` with a **nonlocal**
source. The code does the latter, which is why the local `T_ij` of §2 and the k-space
projection of §6.2 sit on either side of a single FFT.

### 3.4 What is deliberately *not* done

The quadrupole formula `Q_ij = ∫ ρ x_i x_j d³x` is a far-field multipole expansion,
valid for an observer at `r ≫` source size. It is **not** used and does not apply
here: these runs compute `h_ij` *inside* the source region, everywhere in the box,
with the source active throughout. Solving the full wave equation is strictly more
general; no near-zone/far-zone split is made and no observer at infinity is assumed.

### 3.5 GW *energy* is a two-point object — where that enters

In the literature the energy of a homogeneous/stochastic GW field is always written
with a two-point correlation function. That is not a stylistic choice: **GW energy is
not localizable.** By the equivalence principle the field can be transformed away at
any single point, so there is no pointwise `t_μν^GW`. The Isaacson (1968) stress
tensor exists only under a coarse-graining average,

```
  t_μν^GW  =  (c⁴/32πG) ⟨ ∂_μ h_ij ∂_ν h_ij ⟩
```

with `⟨⟩` the Brill–Hartle average over a region large compared with the GW
wavelength but small compared with the background curvature radius. It is
**quadratic**, hence intrinsically a two-point object even at zero separation.

The code keeps the two kinds of object cleanly separated:

| object | nature | needs a correlator? | where |
|---|---|---|---|
| equation of motion `□h̄ = −16πG T` | linear, local, deterministic | **no** | §2–§8 |
| energy / spectrum | quadratic, averaged | **yes** | §11 only |

So the correlator enters **exactly and only in the diagnostics**, in two places.

**(a) Zero separation — the energy** ([line 1275](src/special/gravitational_waves_hTXk.f90#L1275)):

```fortran
call sum_mn_name((ggT**2+ggTim**2+ggX**2+ggXim**2)*nwgrid*EGWpref, idiag_EEGW)
```

The forward FFT carries a `1/N` normalisation ([`fourier_fftpack.f90:242`](src/fourier_fftpack.f90#L242)), so Parseval reads

```
  Σ_k |ĝ(k)|²  =  (1/N) Σ_x |g(x)|²  =  ⟨ ḣ² ⟩_volume
```

giving `EEGW = EGWpref·⟨ḣ_T² + ḣ_X²⟩`. **That volume average over the periodic box
IS the Isaacson average**, and it is the two-point correlator at zero separation.
(`sum_mn_name` divides by `nwgrid`; the explicit `*nwgrid` undoes it, leaving the bare
sum over modes.) Note `h_ij h_ij = 2(h_T² + h_X²)`, since `e⁺_ij e⁺_ij =
e^×_ij e^×_ij = 2` with vanishing cross term — that factor of 2 is absorbed into
`EGWpref`.

**(b) All separations — the spectrum** ([line 1885](src/special/gravitational_waves_hTXk.f90#L1885)):

```fortran
spectra%GWs(ik) = spectra%GWs(ik) + f(...,iggX)**2 + f(...,iggXim)**2 &
                                  + f(...,iggT)**2 + f(...,iggTim)**2
```

By the **Wiener–Khinchin theorem** this binned `Σ_{|k|∈bin}|ĝ(k)|²` *is* the Fourier
transform of the two-point autocorrelation `⟨ḣ_ij(x) ḣ_ij(x+r)⟩`. The power spectrum
and the correlation function are the same information in two bases — so the
correlator is not missing from the code, the whole `GWs`/`GWh`/`Str` output *is* the
correlator. Integrating `GWs(k)` over `k` returns `EEGW`, i.e. the `r → 0` limit.

### 3.6 What licenses the volume average — and its two failure modes

Statistical homogeneity plus ergodicity: for a homogeneous stochastic field, the
spatial average over a large volume estimates the ensemble average. The code evolves
**one realisation** and uses the box average as a proxy for `⟨⟩`. There is no
ensemble averaging anywhere in the module. Two consequences:

- **Cosmic variance.** Each `k`-bin holds `~4πk²Δk·V/(2π)³` modes, so the spectrum
  carries fractional scatter `~1/√N_modes`. Worst at low `k`, where bins are nearly
  empty. This is a genuine statistical error, not a numerical one.
- **Scale separation.** The Isaacson average requires `λ_GW ≪ L_box`; for modes near
  `k₁ = 2π/L` this fails outright.

Together with the `om2_min` threshold of §7.4, that is *three* independent reasons
not to trust the lowest-`k` end of these spectra.

### 3.7 Why the analytic literature *must* start from the correlator

Analytic treatments (Caprini & Durrer; Roper Pol et al.) cannot evolve a realisation,
so they begin from the source **unequal-time correlator** (UETC) and convolve it with
the Green's function:

```
  ⟨|h(k,t)|²⟩  =  ∫dt₁ ∫dt₂  G(t,t₁) G(t,t₂) ⟨ S(k,t₁) S*(k,t₂) ⟩
```

The UETC then has to be *modelled* — usually assuming Gaussianity plus some
decorrelation ansatz.

The simulation inverts this logic. It propagates the actual realisation through the
exact Green's function (the `cos ωΔt`/`sin ωΔt` propagator of §7.2, which is that same
`G`), so the UETC is reproduced automatically and exactly, including non-Gaussianity
and the true time-decorrelation of the turbulence. The correlator is needed only at
the very end, to turn one realisation of `h` into an observable — which is precisely
why it appears in the diagnostics and nowhere else.

**Does the code use a UETC anywhere? No — verified, in all three places it could:**

1. **As solver input: never.** The propagator sees only `S(k,t)` at the current
   instant. There is no `∫dt₁∫dt₂ G G ⟨SS⟩` construction, and no unequal-time or
   time-lag machinery exists in the module.
2. **Stored history: exactly one step, and not a correlator.** The `StT/StX/StTim/
   StXim` slots hold the previous step's `S_T, S_X` ([lines 3074–3077](src/special/gravitational_waves_hTXk.f90#L3074))
   and feed exactly one expression, `dS_T_re = S_T_re - f(...,iStressT)`
   ([line 2977](src/special/gravitational_waves_hTXk.f90#L2977)) — a finite-difference
   *derivative* `Ṡ ≈ ΔS/Δt` for the `itorder_GW=2` correction (§7.3), i.e. a two-time
   **difference**, not a two-time **product**. Their registration confirms it:
   `rhs = itorder_GW==2` ([line 335](src/special/gravitational_waves_hTXk.f90#L335)).
3. **As a diagnostic: also no.** `Str` and `Stg` are **equal-time**, `|S(t)|²` and
   `S(t)·g(t)` ([lines 2093–2121](src/special/gravitational_waves_hTXk.f90#L2093)). No lag.

The absence is the design. The UETC is what you need when you *cannot* evolve a
realisation; the simulation has the dynamics, so it needs no statistical summary. The
true unequal-time structure is present implicitly in the trajectory `S(k,t)` and is
convolved with the Green's function step by step — but equally, nothing is *computed*
that could be held up beside GKK07's `exp(−(π/4)η_k²τ²)` ansatz.

**Measuring it in post-processing.** Since this is exactly the assumption the peak
discrepancy of §3.9 hangs on, it is worth knowing it is within reach. `S_T, S_X` live
in the f-array in k-space and are already exposed as slice/video output
([line 3504](src/special/gravitational_waves_hTXk.f90#L3504)). Dump them at high
cadence and form

```
  UETC(k,t₁,t₂) = ⟨ S_T(k,t₁)S_T*(k,t₂) + S_X(k,t₁)S_X*(k,t₂) ⟩
```

shell-averaged over `|k|`. The binding constraint is that the dump cadence must
resolve the eddy turnover time at the spectral peak. That would be a direct test of
GKK07's central ansatz.

### 3.8 Where this actually changed the physics: analytics vs. simulations

This is not an academic distinction — it is the origin of the main quantitative
disagreement in the literature. Both approaches solve the *same* local linear wave
equation with the *same* Green's function, and both form the *same* Isaacson-averaged
quadratic observable. **The only difference is whether the source correlator is
supplied or generated.**

**(a) The infrared slope — `k¹` vs `k³`.** Roper Pol, Mandal, Brandenburg,
Kahniashvili & Kosowsky (PRD **102**, 083512, arXiv:1903.08585) state that

> "the scaling of Ω_GW(k) with k³ obtained in previous analytical estimates as in,
> e.g., [Gogoberidze et al. 2007], is not expected for the turbulent developed
> spectrum"

and that "the subinertial power law Ω_GW(k) ∼ k is a novel result from our
simulations that was not obtained in previous analytical estimates."

The mechanism is purely a statement about the **stress two-point function**: for a
magnetic field with a Batchelor `k⁴` spectrum, the spectrum of `B²` — i.e. of the
source `T_ij` — is **white noise, `∝ k²`**, in the subinertial range. With
`Ω_GW ~ Ω_T/k²` this gives a spectrum far shallower than the assumed `k³`. In strain
terms: `h_c ∝ f^(-1/2)` below the peak, `h_c ∝ f^(-7/3)` in the inertial range
(Kolmogorov).

**The whole disagreement is two powers of `k`, in one place.** The slope table from
RP20 reads, with subinertial columns `ana`/`sim` and inertial-range columns for a
Kolmogorov and a Golitsyn magnetic spectrum:

| slope of | ana | sim | Kol | Gol |
|---|---|---|---|---|
| `Ω_M`   |   5 |    5 | −2/3 | −8/3 |
| `Ω_GW`  |   3 |    1 | −8/3 | −14/3 |
| `h_c`   | 1/2 | −1/2 | −7/3 | −10/3 |

Two rules reproduce every entry: `Ω_GW = Ω_T − 2` (dividing by `k²` in the wave
equation) and `h_c = (Ω_GW − 2)/2` (from `Ω_GW ∝ f²h_c²`). Check the conversions:
`(3−2)/2 = 1/2`, `(1−2)/2 = −1/2`, `(−8/3−2)/2 = −7/3`, `(−14/3−2)/2 = −10/3`. And
the inertial columns follow from `Ω_M`: Kolmogorov `E ∝ k^(−5/3) ⟹ Ω_M = −2/3`,
`−2/3−2 = −8/3`; Golitsyn `E ∝ k^(−11/3) ⟹ Ω_M = −8/3`, `−8/3−2 = −14/3`.

Note what *agrees*:

- `Ω_M = 5` in **both** columns — both have a Batchelor field, `E_M ∝ k⁴`. The
  disagreement is **not** about the turbulence.
- `Kol` and `Gol` carry a **single** value each, no ana/sim split — in the inertial
  range the two approaches agree.

They differ in exactly one cell-pair: subinertial `Ω_GW`, 3 vs 1 (with `h_c` merely
restating it). Since both use `Ω_GW = Ω_T − 2`, they must differ on the **stress**
spectrum `Ω_T`:

| | `Ω_T` below the peak | `Ω_GW` |
|---|---|---|
| ana | assumed `Ω_T ∝ Ω_M ∝ k⁵` | `5 − 2 = 3` |
| sim | found `Ω_T ∝ k³` (white noise) | `3 − 2 = 1` |

**Why the analytic step fails.** `T ~ B²` is *quadratic*, so its spectrum is the
magnetic spectrum **convolved with itself**. For `k` below the peak that convolution
is dominated by pairs `k₁ ≈ −k₂` near the peak, whose contribution does not vanish as
`k → 0`. The result is white noise — `E_T ∝ k²`, `Ω_T ∝ k³` — *regardless* of how
steep `Ω_M` is at small `k`. A `k⁵` field cannot produce a `k⁵` stress; the quadratic
operation floors the slope at `k³`.

This is purely kinematic and holds **even for perfectly Gaussian `B`**: Wick gives
`⟨B²B²⟩ = 2⟨BB⟩²`, which is exactly that self-convolution. **Gaussianity is not the
culprit** — the single assumption `Ω_T ∝ Ω_M` is.

Two corollaries worth internalising:

- The discrepancy is confined *below* the peak because the convolution floor only
  binds where the true spectrum is steeper than white noise. Above the peak nothing
  is steeper than `k³`, so `Kol`/`Gol` need no ana/sim split.
- The code cannot get this wrong by construction: the white-noise floor emerges
  automatically from evolving `B(x,t)` and forming `B_iB_j` pointwise (§2). It is
  only reachable by error if you write `Ω_T ∝ Ω_M` by hand.
Observational consequence: more low-frequency GW energy, extending LISA
detectability to lower frequencies and hence higher energy scales.

**(b) Amplitude.** Gogoberidze et al. predicted `h_c ≈ 4×10⁻²⁰` at 1 mHz; run `ini1`
gives `h_c ≈ 0.7×10⁻²¹`. *Treat with care* — the arXiv record for 1903.08585 carries
the comment "corrected normalization error", so amplitudes shifted between versions.
The slope result is robust; the absolute normalisation is the fragile part.

**(c) Dependence on the driving — invisible to the analytics.** For the *same total
source energy*, the simulations find forced acoustic (irrotational) driving yields
**~200×** more GW energy than a suddenly-imposed magnetic field (`ini1–3`), and
forced non-helical magnetic driving **~10×** more. A calculation that begins from
"assume a stationary Kolmogorov spectrum acting for duration τ" integrated the
driving history away and cannot see this.

**(d) Which assumption did it.**

| Analytic assumption (GKK07 and successors) | What the simulation does instead |
|---|---|
| stationary Kolmogorov spectrum given by hand | spectrum evolves self-consistently, and decays |
| temporal correlation of `u` **assumed** (sweeping ansatz) | realisation evolved; decorrelation is whatever MHD produces |
| Gaussian source → Wick factorisation of `⟨TT⟩` | no Gaussianity assumed |
| duration = fixed fraction of Hubble time, expansion neglected | expansion carried in `a''/a` inside `ω²` (§7.1) |
| aeroacoustic / long-wavelength limit | not needed |

Note GKK07 explicitly *validated* their own aeroacoustic approximation as "an
excellent approximation" — so that is **not** the source of the discrepancy. The
difference comes from the assumed source spectrum and its temporal correlation, i.e.
exactly the UETC.

**(e) Reconciliation.** `k³` is not simply wrong. Caprini, Durrer et al.
(PRD **102**, 083528, arXiv:1909.13728) show `k³` is a *universal* IR scaling when the
source is bounded in **both** frequency and time, for `k` below **all** physical scales
of the source. The `k¹` applies in the turbulence's subinertial range, which lies
above that asymptotic regime. The two describe different `k` intervals; the practical
question is which covers the detector band, and the simulations say it is the
shallower one.

**(f) Same authors, both sides.** Kahniashvili, Brandenburg, Gogoberidze, Mandal &
Roper Pol, PRResearch **3**, 013193 (2021) is the direct successor — the analytic
authors moved onto the simulation side. This is a group superseding its own earlier
work, not a standing dispute.

For the dedicated study of the object at the centre of all this, see
Brandenburg & Boldyrev, "The Turbulent Stress Spectrum in the Inertial and
Subinertial Ranges", ApJ (2020).

### 3.9 Complete difference list, and why the spectral PEAK does not match

**Full comparison.** GKK07 = Gogoberidze, Kahniashvili & Kosowsky, PRD **76**, 083002
(arXiv:0705.1733). RP20 = Roper Pol et al., PRD **102**, 083512 (arXiv:1903.08585).

| # | Aspect | GKK07 (analytic) | RP20 (simulation) |
|---|---|---|---|
| 1 | Method | UETC × Green's function convolution | direct evolution of one realisation |
| 2 | Source spectrum | stationary Kolmogorov, imposed | self-consistent, decaying MHD |
| 3 | Time decorrelation | Kraichnan square-exponential `exp(−(π/4)η_k²τ²)`, assumed | whatever MHD produces |
| 4 | Gaussianity | assumed (Wick factorisation of `⟨TT⟩`) | not assumed |
| 5 | Aeroacoustic limit | used (`k→0` in `H_ijij`) | not used |
| 6 | Expansion during source | neglected | included via `a''/a` in `ω²` (§7.1) |
| 7 | Source duration | fixed fraction of Hubble time | actual, self-consistent |
| 8 | **Peak location** | `~M·k_*` (eddy turnover frequency) | `2k_*` (stress spectrum peak) |
| 9 | IR slope | `h_c ∝ f^(1/2)` ⟺ `Ω_GW ∝ f³` | `h_c ∝ f^(−1/2)` ⟺ `Ω_GW ∝ f¹` |
| 10 | Falloff above peak | `h_c ∝ f^(−13/4)` ⟺ `Ω_GW ∝ f^(−9/2)` | Kolmogorov: `Ω_GW ∝ f^(−8/3)` ⟺ `h_c ∝ f^(−7/3)` |
| 11 | High-f cutoff | **exponential**, `exp(−2ω²/k₀²M²R)` | power law, no exponential cutoff |
| 12 | Peak amplitude | `h_c ≈ 4×10⁻²⁰` at 1 mHz | `h_c ≈ 0.7×10⁻²¹` at 3 mHz |
| 13 | Driving dependence | none — integrated away | acoustic ~200×, nonhel. magnetic ~10× vs. sudden `ini` |
| 14 | Mach dependence of peak | strong | none |

Rows 9–10 use `Ω_GW = (2π²/3H₀²) f² h_c²` to convert; the papers quote in different
conventions, which is an easy place to get confused.

**Why the peak differs.** The two calculations tie the peak to *different physical
quantities*:

- **GKK07** — "the peak frequency is thus proportional to the inverse of the
  circulation time on the stirring scale". The **aeroacoustic (long-wavelength)
  approximation** replaces `H_ijij(k=ω, ω) → H_ijij(k=0, ω)`, which deliberately
  deletes the source's spatial-wavenumber dependence. The emitted GW frequency is
  then fixed purely by the source's *temporal* correlation spectrum — the eddy
  turnover / sweeping rate `ω_* ~ M k_*`. This is standard Lighthill aeroacoustics:
  a subsonic flow radiates at the frequency of its temporal fluctuations.
- **RP20** — no such substitution is available, because the code never forms
  `H_ijij`. It solves `ḧ(k) + k²h(k) = S(k,t)` per mode (§7.2), and free GWs obey
  `ω = k` exactly, so power at wavenumber `k` is observed at `f = k/2πa`. The
  spectrum in `k` therefore tracks the **stress** spectrum in `k`, which peaks at
  `2k_*` because the stress is *quadratic* in `B` and self-convolution shifts the
  peak: "the peak of the stress spectrum shifts to `2k_*`, being `k_*` the position
  of the spectral peak of the magnetic field."

**Consequence: the peaks differ by roughly `2/M`.** They coincide only near `M ~ 1`;
for subsonic turbulence the analytic peak sits well below the simulated one — GKK07's
peak slides to `~10⁻⁴ Hz` at `M = 0.1` while RP20's stays in the mHz band set by
`k_* ~ 100/H_*`.

**Which to trust here.** The simulation does *not* neglect the temporal structure — it
integrates the true `S(k,t)`. If the source at `k` is off-resonance (varying at
`Mk ≪ k`), the code produces exactly the resulting small `ḣ`. The resonance physics
GKK07 model is *included* in the simulation, not omitted from it; the aeroacoustic
step is an approximation made on one side only. Independent check: acoustic driving
gives ~200× more GW energy, and acoustic motions decorrelate at `ω ~ c_s k` with
`c_s = 1/√3` — far nearer the `ω = k` resonance than vortically-swept turbulence at
`v ≪ c`. The resonance picture predicts exactly that ordering.

> **Provenance.** The peak *definitions* in rows 8 and the two bullets are verbatim
> from the respective papers. The `2/M` factor, the aeroacoustic-substitution
> diagnosis, and the resonance interpretation of the 200× acoustic result are a
> synthesis of the two papers, not a quotation from either. No passage was found in
> which RP20 diagnose the peak-location discrepancy as such — their explicit
> comparison conflates amplitude and frequency in one sentence. Rows 10–11 come from
> an HTML render of GKK07 rather than a line-by-line PDF read.

### 3.10 Where the `k¹ → k³` break sits, and why the box never contains it

The `k¹` branch is bounded below. The break follows directly from the propagator of
§7.2 — start from `h = g = 0` and drive with constant `S` for a time `Δt`:

| regime | expansion | result |
|---|---|---|
| `kΔt ≪ 1` (impulsive) | `sin(ωΔt) ≈ ωΔt` ⟹ `g ≈ S·Δt`, **k-independent** | `Ω_GW ∝ Ω_T·Δt² ∝ k³` |
| `kΔt ≫ 1` (oscillatory) | response saturates at `S/ω` ⟹ `|g|² ~ |S|²/k²` | `Ω_GW ∝ Ω_T/k² ∝ k¹` |

Both regimes carry the *same* white-noise stress `Ω_T ∝ k³` (§3.8). Only the GW
*response* changes — whether the mode completes oscillations while the source lives.
Hence

```
  k_break  ≈  1 / Δt_source
```

**For the sample run's parameters** (`samples/GravitationalWaves`): `tstart=1.` sets
`t_* = 1`, and radiation-era `ℋ = 1/t` puts the horizon at generation at `k = 1`.
`wav1=100.` sets `Lxyz = 2π/100` ([`start.f90:215`](src/start.f90#L215)), so `k₁ = 100`;
`nxgrid=16` gives Nyquist `k = 800`.

| quantity | k |
|---|---|
| `k_break`, source lasting one Hubble time (`Δt ~ 1`) | ~1 |
| `k_break`, source decaying on eddy turnover `1/(k_*v_*)`, `k_*=100`, `v_*~0.1` | ~10 |
| **box fundamental `k₁`** | **100** |
| Nyquist | 800 |

**The `k³` regime is therefore not under-resolved — it is absent.** The break lies
10–100× *below* the largest mode the box can hold. Every mode in the simulation sits
in the `kΔt ≫ 1` branch, which is exactly why the measured spectra show a clean `k¹`
with no turnover: the box lies entirely inside the `k¹` window by construction.

**Cost to reach it.** You need `k₁ ≲ k_break`, plus a decade below that to *measure*
a slope rather than infer it — so a box 100–1000× larger. Holding the small-scale
physics fixed (same Nyquist) scales `N` by the same factor: this 16³ sample becomes
1600³–16000³; a production 1152³ run becomes ~10⁵–10⁶ per dimension. Capturing the
`k³` tail means simulating a comoving Hubble volume *while* resolving the inertial
range — a dynamic range of 10⁴–10⁶ per dimension. This is why the universal `k³` must
be imported analytically (Caprini/Durrer causality) rather than measured: simulations
establish the `k¹` branch and its origin, and no simulation reaches below `1/Δt`.

### 3.10a Is the factor 2 in `k_GW ≈ 2k_*` universal?

**Kinematic in origin, therefore duration-independent.** `T_ij` is *quadratic* in the
fields — every term in §2 (`u_iu_j`, `B_iB_j`, `E_iE_j`, `∂_iφ∂_jφ`). So in Fourier
space the stress is a **self-convolution**, `T(k) = ∫d³q B(q)B(k−q)`. With `B`
concentrated near `|q| = k_*`, the reachable sums `q₁+q₂` fill a ball of radius
`2k_*`: **no stress mode can exceed twice the field's wavenumber.** Since
`E_T = 4πk²P_T` grows as `k²`, the stress spectrum peaks right against that upper
support edge. This involves one instant only — no `Δt`, no propagator. It is
universal across *all* source types in this module, because all are quadratic.

(The same integral at the opposite end gives the white-noise floor of §3.8:
`P_T(k→0) = ∫d³q P_B(q)² = const`. The two results are the two edges of one
convolution.)

**What it does depend on:**

- **Spectral width.** A crisp `2k_*` needs `B` dominated by a narrow band. For
  developed turbulence with a broad inertial range the edge smears — which is exactly
  why the `Kol`/`Gol` columns of the §3.8 table show `Ω_T ∝ Ω_M` there instead of a
  doubling.
- **The `1/k²` weighting.** `Ω_GW = Ω_T/k²` pulls the maximum *below* the stress peak.
  RP20 quote `≈2k_*` for both, consistent with a sharply peaked source where the hard
  convolution edge beats the smooth `k²` division. For a broad source the GW peak
  would sit noticeably below the stress peak.

**Two indirect routes by which duration enters:**

1. *Impulsive vs oscillatory.* If the peak sat in the impulsive branch (§3.10),
   `Ω_GW ∝ Ω_T·Δt²` and the GW peak would coincide **exactly** with the stress peak,
   with no `1/k²` shift. Requires `Δt ≪ 1/(2k_*)`; since `2k_* ≫ 1/Δt` for any
   realistic parameters, this route exists but is never activated.
2. *A moving `k_*` — this one bites.* In decaying MHD turbulence with inverse
   transfer, `k_*` drifts to lower wavenumbers. The GW spectrum accumulates over the
   whole source lifetime, so the final peak is a weighted integral over `k_*(t)`; a
   longer run samples lower `k_*` and shifts the peak down. But what moves is **`k_*`
   itself, not the factor 2** — `k_GW(t) ≈ 2k_*(t)` holds instantaneously throughout,
   and the observed peak is set by whichever epoch dominates GW production (early, for
   decaying turbulence).

Consequence for §3.9: the simulated peak position is sensitive to the **decay
history**, whereas GKK07's `M·k_*` is sensitive to the **Mach number**. Two different
sensitivities — another handle for telling the predictions apart.

**Structural remark (synthesis, not in either paper).** In the decaying case
`k_break ≈ k_*·v_*` — the eddy turnover rate. That is the *same combination* as
GKK07's peak `~M·k_*` (§3.9), and not by coincidence: the turnover rate is the
source's temporal frequency, and `k = ω_source` is exactly where the response crosses
from impulsive to resonant. **GKK07 place their spectral peak precisely where the
simulation's spectrum merely changes slope.** Their aeroacoustic treatment turns the
spectrum over there; the simulation keeps climbing as `k¹` for another 1–2 decades up
to `2k_*`. That one structural difference generates *both* the peak-location
discrepancy (§3.9) and the IR-slope discrepancy (§3.8).

---

## 4. Step 2 — scaling and deposition into the `f`-array

**Location:** [`dspecial_dt`, line 1165](src/special/gravitational_waves_hTXk.f90#L1165)

Despite the name, **this routine does not contribute to `df` at all** — it calls
`keep_compiler_quiet(df)`. The GW variables are therefore *completely outside* the
Runge–Kutta integrator. All it does is:

```fortran
call compute_scl_factor
stress_prefactor2 = stress_prefactor/scale_factor          ! = 6/a

do ij=1,6
  f(l1:l2,m,n,iStress_ij+ij-1) = stress_prefactor2*p%stress_ij(:,ij)
enddo
```

### Scale factor `a(t)`

[`compute_scl_factor`, line 1076](src/special/gravitational_waves_hTXk.f90#L1076):

| switch | `a(t)` |
|---|---|
| default | `(t + tshift)^nscale_factor_conformal` (`=1` for radiation era, `nscale_factor_conformal=1`) |
| `lreheating_GW` | `¼(t+1)²` |
| `lmatter_GW` | `t²/t_equality` |
| `ldark_energy_GW` | `t_acceleration³/(t·t_equality)` |
| `lscalar` | `exp(f_ode(ilna))` |
| `lread_scl_factor_file` | log-log interpolation from `a_vs_eta.dat` ([`read_scl_factor`, line 1144](src/special/gravitational_waves_hTXk.f90#L1144)) |

Extra optional multipliers on `stress_prefactor2`:
`tau_stress_comp` (compensate decaying turbulence: `×(1+(t-tstart)/τ)^exp`) and
`tau_stress_kick` (stepwise kicks).

### `stress_prefactor` / `EGWpref` table

`cstress_prefactor` selects both the source normalisation and the GW-energy
normalisation ([line 460](src/special/gravitational_waves_hTXk.f90#L460)):

| `cstress_prefactor` | `stress_prefactor` | `EGWpref` |
|---|---|---|
| `'6'` *(default, the one to use)* | 6 | 1/6 |
| `'1'` | 1 | 8π |
| `'2'` | 2 | 1/2 |
| `'16pi'` | 16π | 1/(32π) |
| `'16piG/c^2'` | 16πG/c² | c²/(32πG) |

(The in-source comment `EGWpref=.5*16*pi/stress_prefactor**2` reproduces only the
`'1'` and `'16pi'` rows; the others are historical/back-compatibility values. Use
`'6'`.)

---

## 5. Step 3 — the forward FFT

**Location:** [`compute_gT_and_gX_from_gij`, line 3086](src/special/gravitational_waves_hTXk.f90#L3086)

```fortran
Tpq_re(:,:,:,:) = f(l1:l2,m1:m2,n1:n2,iStress_ij:iStress_ij+5)
Tpq_im          = 0.0
call fft_xyz_parallel(Tpq_re(:,:,:,:), Tpq_im(:,:,:,:))    ! one call, all 6 components
```

That is **one forward 3-D parallel FFT per timestep** (of a 6-component field).
Everything after this point is per-k-mode arithmetic.

*Optional nonlinear source* (`lnonlinear_source=T`): before this, `compute_Hijk`
builds `∂_k h_ij` in k-space, inverse-transforms it, forms
`T^nl_pq = Σ_ij (∂_p h_ij)(∂_q h_ij)` in real space, then transforms it forward and
adds it to `S̃`.

---

## 6. Steps 4–5 — TT projection and polarisation decomposition

**Location:** [`solve_and_stress`, line 2523](src/special/gravitational_waves_hTXk.f90#L2523)

Triple loop over `ikz, iky, ikx`. Per mode:

### 6.1 Wavevector and projector

```
  k = (kx_fft(ikx+ipx*nx), ky_fft(iky+ipy*ny), kz_fft(ikz+ipz*nz))
  k²  = k1²+k2²+k3²
  P_ij = δ_ij - k_i k_j / k²
```

### 6.2 TT (Lambda) projection

```
  Λ_ij,pq  =  P_ip P_jq  -  ½ P_ij P_pq
  S̃_ij     =  Λ_ij,pq  T̃_pq
```

verbatim:

```fortran
Sij_re(ij) = Sij_re(ij) + (Pij(ip)*Pij(jq)-.5*Pij(ij)*Pij(pq))*Tpq_re(ikx,iky,ikz,pq)
```
(and identically for the imaginary part).

This single line is the discrete, exact realisation of the real-space convolution
`∫d³y Λ_ij,pq(x−y) T_pq(y)` discussed in §3.3.

### 6.3 Polarisation basis

Two orthonormal vectors `e1, e2 ⊥ k` are constructed by picking the *smallest*
component of `k` as preferred direction (avoids the degeneracy when `k` aligns with
an axis), e.g. for `|k1|` smallest:

```
  e1 = (0, -k3, +k2) / |…|
  e2 = (k2²+k3², -k2k1, -k3k1) / |…|
```

then

```
  e⁺_ij = e1_i e1_j - e2_i e2_j
  e^×_ij = e1_i e2_j + e2_i e1_j
```

Sign convention: with `lswitch_sign_e_X=T` (default) `e^×` is flipped for
`k3<0` (or `k2<0` if `k3=0`, or `k1<0` if `k2=k3=0`) so that helicity is
consistently defined over the whole half-space.

### 6.4 Projected source amplitudes

```
  S_T = ½ e⁺_ij S̃_ij          S_X = ½ e^×_ij S̃_ij
```

Both are complex (`S_T_re`, `S_T_im`, `S_X_re`, `S_X_im`).

---

## 7. Step 6 — the time advance: an **exact** propagator, not an integrator

This is the heart of the method and why the scheme has **no GW-related CFL
constraint** (RoperPol et al. 2020, *GApFD* **114**, 130).

### 7.1 The dispersion relation

```fortran
om2 = ksqr - appa_om          ! ω² = k² - a''/a
om  = sqrt(om2)
```

`appa_om = a''/a` is zero in the pure radiation era (default), or interpolated from
`a_vs_eta.dat` when `lread_scl_factor_file=T`. Special cases:

| switch | `ω²` |
|---|---|
| default | `k² - a''/a` |
| `linflation` | `4k² - 2/t²` |
| `lreheating_GW` | `k² - 2/(t+1)²` |
| `lmatter_GW`, `ldark_energy_GW` | `k² - 2/t²` |
| `lscalar` | `k² - ddotam` |
| `delkt≠0` | `k² + δk(t)² - a''/a` |
| `lhorndeski` | `(1+α_T)k² + δk² - α_M² term - a''/a` |

If `ω² < 0` (super-horizon / tachyonic modes) the flag `lsign_om2=.false.` and the
trigonometric functions are replaced by hyperbolic ones — see below.

### 7.2 The propagator

Over one step `Δt` the source `S` is held **constant**. Then (★) is a driven
harmonic oscillator with the exact solution

```
  h(t+Δt) = ( h - S/ω² ) cos(ωΔt)  +  (g/ω) sin(ωΔt)  +  S/ω²
  g(t+Δt) =   g cos(ωΔt)           -  ω( h - S/ω² ) sin(ωΔt)
```

with `g ≡ dh/dt`. In the code ([line 2881](src/special/gravitational_waves_hTXk.f90#L2881)), with
`om12 = 1/ω²`, `om1 = 1/ω`:

```fortran
coefAre = hhTre - om12*S_T_re(ikx,iky,ikz)      ! A = h - S/ω²
coefBre = ggTre*om1                             ! B = g/ω

f(...,ihhT) = coefAre*cosot     + coefBre*sinot       + om12*S_T_re(...)
f(...,iggT) = coefBre*cosot*om  + coefAre*om*sinot_minus
```

and identically for the `X` polarisation and for the imaginary parts.

```fortran
if (lsign_om2) then                ! ω² > 0
  cosot = cos(om*dt);  sinot = sin(om*dt);  sinot_minus = -sinot
else                               ! ω² < 0  →  |ω| = sqrt(|ω²|)
  cosot = cosh(om*dt); sinot = sinh(om*dt); sinot_minus = +sinot
endif
```

The sign flip in `sinot_minus` is exactly what turns the rotation matrix into a
boost matrix — one branch of code, both regimes.

**No `Δt` appears anywhere except inside `cos(ωΔt)`/`sin(ωΔt)`.** The step is exact
for arbitrarily large `ωΔt`; accuracy is limited only by how well "`S` = const over
`Δt`" holds. This is the entire reason `h` is kept in k-space.

### 7.3 Second-order source correction (`itorder_GW=2`)

Instead of freezing `S`, assume it varies linearly across the step,
`Ṡ ≈ ΔS/Δt`, and add the corresponding particular-solution correction:

```fortran
dS_T_re = S_T_re(...) - f(...,iStressT)          ! ΔS from the previous step
f(...,ihhT) += dS_T_re*om12*(1. - om1*dt1*sinot)
f(...,iggT) += dS_T_re*om12*dt1*(1. - cosot)
```

This is why the *previous* step's `S_T, S_X` are stored in the aux slots
`iStressT/X(im)` at the end of the routine.

### 7.4 The `k = 0` mode

Guarded by `lswitch_om2_min_condition`:

```fortran
if (lnew_switch_om2_min) then
  lswitch_om2_min_condition = om2 /= 0.
else
  lswitch_om2_min_condition = om2 > om2_min       ! om2_min = (om2_min_factor*kmin)²
endif
```

If false, the mode is simply zeroed (`h = g = 0` at the origin `(1,1,1)` on the root
processor). The default `lnew_switch_om2_min=F` uses the historical threshold
`om2 > om2_min`, which introduces a small artificial jump in the GW spectrum at
low `k`; setting `lnew_switch_om2_min=T` uses the correct condition `ω² ≠ 0`.

### 7.5 Horndeski / modified-gravity branch

For `lhorndeski=T` the equation acquires friction,

```
  h'' + α_M h' + ω² h = S
```

solved by diagonalising the 2×2 system exactly:

```
  λ± = ½( -α_M ± sqrt(α_M² - 4ω²) )
```

```fortran
explam1t = exp(lam1*dt);   explam2t = exp(lam2*dt);   det1 = 1./discrim
cosoth = det1*(lam1*explam2t - lam2*explam1t)
sinoth = -det1*(explam2t - explam1t)*om_cmplx
hcomplex_new = cosoth*coefA + sinoth*coefB + om12*cmplx(S_T_re, S_T_im)
```

i.e. the same "exact propagator" idea, but with a complex matrix exponential.

---

## 8. Step 7 — back to real space (output only)

At the end of [`compute_gT_and_gX_from_gij`](src/special/gravitational_waves_hTXk.f90#L3244), and **only if requested**:

| flag | what is inverse-FFT'd | into aux slots |
|---|---|---|
| `lreal_space_hTX_as_aux` | `h_T, h_X` | `hhT_realspace`, `hhX_realspace` |
| `lreal_space_gTX_as_aux` | `g_T, g_X` | `ggT_realspace`, `ggX_realspace` |
| `lreal_space_hij_as_aux` | all 6 `h_ij` (via `compute_hij`, un-projecting T/X back to ij) | `h11..h31_realspace` |

If none of these is set, **no inverse FFT is ever performed** and `h` exists only in
k-space for the entire run.

---

## 9. The per-timestep control flow

`special_after_timestep` is called by the timestepper after *every* RK substep
([`src/timestep.f90:329`](src/timestep.f90#L329)), but the GW work is gated:

```fortran
if (lfirst) then                                    ! itsub == 1 only
  dt_GW = dt_GW + dt
  if (mod(it+1,ntimesteps_per_GW_step) == 0) then
    call compute_gT_and_gX_from_gij(f,'St',dt_GW)
    dt_GW = 0.0
  endif
endif
```

So one full timestep looks like:

```
timestep it
├── itsub = 1   (lfirst = .true.)
│   ├── pde → calc_pencils_special   : build T_ij(x) from u, B      [REAL SPACE]
│   ├──       dspecial_dt            : f(...,iStress_ij..) = 6/a·T_ij
│   │                                  (df untouched — GW not in RK)
│   ├── RK substep updates u, B, ρ, … as usual
│   └── special_after_timestep       : dt_GW += dt
│       └── compute_gT_and_gX_from_gij(f,'St',dt_GW)
│           ├── forward FFT of the 6 stress components          [→ FOURIER]
│           └── solve_and_stress
│               ├── per k: P_ij, Λ_ij,pq, e⁺, e^×  →  S_T, S_X
│               ├── per k: ω² = k² − a″/a
│               ├── per k: EXACT propagator  h,g ← h,g,S  over dt_GW
│               └── per k: store S_T,S_X in iStressT/X for itorder_GW=2
│           └── optional inverse FFT for *_realspace aux slots   [→ REAL]
├── itsub = 2   (lfirst = .false.) : stress NOT recomputed, GW NOT advanced
└── itsub = 3   (lfirst = .false.) : idem
```

`ntimesteps_per_GW_step > 1` subcycles: the GW system is advanced once every N MHD
steps with the accumulated `dt_GW`. This is legitimate precisely because the
propagator is exact in `Δt`.

> **Known bookkeeping subtlety** (documented in the source): because the stress is
> built at substep 1 (time `t`) but the GW fields end up at `t+dt`, the *stress*
> spectrum is written out labelled `t+dt` although it belongs to `t`. The *GW*
> spectra are correctly at `t+dt`. With `lspec_first=T` spectra are output at both.

---

## 10. The `f`-array slots

Registered in [`register_special`, line 301](src/special/gravitational_waves_hTXk.f90#L301). All are **MAUX**
(auxiliary), not MVAR — confirming they are never touched by the RK integrator.
Requires `MAUX CONTRIBUTION 18` in `src/cparam.local`.

| slot | meaning | space |
|---|---|---|
| `hhT, hhTim, hhX, hhXim` | strain `h_+`, `h_×` (Re/Im) | **Fourier** |
| `ggT, ggTim, ggX, ggXim` | `ḣ_+`, `ḣ_×` (Re/Im) | **Fourier** |
| `Str` (6 components) | `6/a · T_ij` | **real** |
| `StT, StTim, StX, StXim` | projected `S_T, S_X` from previous step | **Fourier** |
| `hhT_realspace, …` | optional diagnostics | real |

A single array in the `f`-array is thus indexed as a k-space array:
`f(nghost+ikx, nghost+iky, nghost+ikz, ihhT)` — the grid indices are reinterpreted
as `(kx,ky,kz)` indices, with `kz` the fastest index in the parallel FFT layout.

---

## 11. Diagnostics and spectra

**Energy** ([`calc_diagnostics_special`, line 1258](src/special/gravitational_waves_hTXk.f90#L1258)):

```
  E_GW  =  EGWpref · Σ_k ( g_T,re² + g_T,im² + g_X,re² + g_X,im² )
```

```fortran
call sum_mn_name((ggT**2+ggTim**2+ggX**2+ggXim**2)*nwgrid*EGWpref, idiag_EEGW)
```

**Strain rms:** `hrms = sqrt( Σ_k (h_T² + h_X² ) )`.

**Spectra** ([`make_spectra`, line 1452](src/special/gravitational_waves_hTXk.f90#L1452)), binned by `ik = 1 + nint(|k|/kscale_factor)`:

| kind | built from | helical partner |
|---|---|---|
| `GWs` | `g_T, g_X` (energy) | `GWshel` = `2(g_X,im g_T,re − g_X,re g_T,im)` |
| `GWh` | `h_T, h_X` (strain) | `GWhhel` |
| `Str` | `S_T, S_X` (stress) | `Strhel` |
| `Stg` | `S·g` cross-spectrum | — |

Because everything is already in k-space, these spectra cost essentially nothing.
`Str` is the k-space realisation of the two-point source correlator of §3.5.

---

## 12. The two alternative modules (for contrast)

| file | `h` stored as | method |
|---|---|---|
| `gravitational_waves_hTXk.f90` | MAUX 18, **k-space** | exact propagator, no CFL |
| `gravitational_waves.f90` | MVAR 6, real space | `del2` + RK3 |
| `gravitational_waves_hij6.f90` | MVAR 12, real space | `del2` + RK3 |

`gravitational_waves_hij6.f90` is the honest "same footing as the MHD" version
([lines 559–578](src/special/gravitational_waves_hij6.f90#L559-L578)):

```fortran
call del2(f,jhij,del2hij(:,ij))
GW_rhs = c_light2*del2hij(:,ij) + stress_prefactor2*p%stress_ij(:,ij)
df(l1:l2,m,n,jhij) = df(l1:l2,m,n,jhij) + f(l1:l2,m,n,jgij)
df(l1:l2,m,n,jgij) = df(l1:l2,m,n,jgij) + GW_rhs
```

Here `h_ij` are genuine MVAR variables, spatial derivatives are 6th-order finite
differences, and the whole system rides the RK3 timestepper — which means the GW
sector **does** impose a CFL condition (`c·dt/dx`), and artificial diffusivities
(`diffhh`, `diffgg`, `diffhh_hyper3`) are available/needed. Fourier transforms are
used there only to obtain the TT-projected stress, not to solve the equation.

Select the module in your run's `src/Makefile.local`:

```
SPECIAL = special/gravitational_waves_hTXk
```

---

## 13. Minimal checklist to reproduce a run

1. `src/Makefile.local`: `SPECIAL = special/gravitational_waves_hTXk`
2. `src/cparam.local`: `MAUX CONTRIBUTION 18` (more if you enable the
   `*_realspace` slots)
3. `start.in` → `&special_init_pars`: `initGW`, `amplGW`, `kpeak_GW`,
   `lStress_as_aux=T`, `lggTX_as_aux=T`, `lhhTX_as_aux=T`
4. `run.in` → `&special_run_pars`: `cstress_prefactor='6'`,
   `nscale_factor_conformal=1.` (radiation era), `lnew_switch_om2_min=T`,
   optionally `itorder_GW=2`
5. `print.in`: `EEGW`, `hrms`, `gg2m`
6. `power_spectrum.in`: `GWs`, `GWh`, `Str`
7. Put `tstart=1.` in `start.in` to avoid `t=0` division in `a(t)=(t+tshift)^n`
8. Working reference: `samples/GravitationalWaves/`

---

## 14. One thing to watch in the source

The call at [line 3242](src/special/gravitational_waves_hTXk.f90#L3242)

```fortran
call solve_and_stress(f, S_T_re, S_X_re, S_T_im, S_X_im, dt)
```

does not match the dummy-argument order at [line 2523](src/special/gravitational_waves_hTXk.f90#L2523)

```fortran
subroutine solve_and_stress(f, S_T_re, S_T_im, S_X_re, S_X_im, dt)
```

— arguments 3 and 4 are transposed. It is currently **harmless**, because all four
arrays are pure scratch: they are zeroed at the top of `solve_and_stress`, all
results are written straight into the `f`-array, and the caller reassigns them from
`f` before reusing them as inverse-FFT workspace. But it will silently produce
wrong output for anyone who later tries to read `S_T_im` back in the caller.
