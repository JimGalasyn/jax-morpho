# jax-morpho.evodevo — design & calibration spec

Status: draft. Goal of this document is to fix the architecture *before*
building beyond the calibration anchor, so the work stays disciplined.

## 0. Thesis & honesty frame

Build the **reverse-mode, GPU-scale, mechanical-engine realization of a
development-derived G-matrix**, and use it to close the loop
`genotype → development → phenotype → selection → response`.

This is a **sufficiency / construction** contribution, not an "explains
evolution" claim. Milocco & Uller (2026 PNAS) own the concept — sensitivity-
derived quantitative-genetic parameters and the breeder's-equation validation.
Our unoccupied ground is: (a) reverse-mode autodiff instead of forward
variational sensitivity, (b) a general high-dimensional **mechanical**
morphogenesis engine instead of low-dimensional ODE toys, (c) organism scale,
and (d) the same artifact doubles as an honest evolution-game loop. Every claim
must be forward/forbidding with a number — never a fit (no menu-fit).

We are **building the tool and validating it against known results**, not yet
attempting a novel scientific study.

## 1. The core object and the non-negotiable fix

Empirically established (this repo): **naive reverse-mode autodiff through the
unrolled relaxation is not the developmental sensitivity** — it disagrees with
finite differences. The correct object is the **implicit sensitivity of the
developmental equilibrium**.

> **Mechanism corrected (Phase 1).** This document originally attributed that
> failure to reverse-mode *degrading with unroll length*. That diagnosis was
> wrong, and Phase 0b already saw the first crack in it: on the M-U ODE, a
> 2000-step unroll did **not** degrade at all. The real cause is upstream of
> autodiff. `center_based.relax` **never reaches a fixed point**: its clipped
> fixed step turns the linear instability of the stiffest mode into a *stable
> period-2 limit cycle*. Measured on a 12-cell blob — effective step 0.0142–0.0197
> straddling the stability limit `2/λ_max = 0.01398` — the iterate orbits
> forever: `|∇E| = 3.34` at 200 steps and still 3.34 at 100 000, with
> `|p_{k+2} − p_k| = 1e-16` while `|p_{k+1} − p_k| = 0.05`. The energy alternates
> between two values. It *looks* converged (bounded energy, constant gradient
> norm) and is not.
>
> So autodiff was never differentiating an equilibrium — it was differentiating
> a limit cycle, and longer unrolls simply backprop through more orbits. There is
> no "long unrolls break reverse-mode" law here. **The conclusion survives; the
> argument for it is replaced.** Implicit-diff remains the correct core tool, now
> for its actual reasons: it is defined by `F(x*,θ)=0` rather than by a step
> count, so it *cannot* be fooled by a relaxation that silently failed to
> converge — and it forces the convergence question into the open, since it needs
> a real fixed point as input. Pinned by
> `test_fixed_point.py::test_relax_never_converges_period_2_limit_cycle`.

Write development as a fixed point `F(x*, θ) = 0`. Then, by the implicit
function theorem,

    ∂x*/∂θ = −[∂F/∂x]⁻¹ [∂F/∂θ]

and the phenotype sensitivity is `J = (∂r/∂x*)·(∂x*/∂θ)` for readout
`z = r(x*)`. For an energy relaxation, `F = ∇ₓE` and `∂F/∂x` is the Hessian;
each reverse-mode VJP costs **one linear solve** (matrix-free CG via
Hessian-vector products) and is **independent of relaxation step count**. This
is the discrete analog of Milocco–Uller's variational sensitivity (their Eq. 3).

Architectural rule: write the sensitivity engine against a **general fixed
point** `F(x*,θ)=0`, with energy-minimization (`F=∇E`) as the default instance.
That one seam covers our potential/relaxation development *and* M-U's general
(non-potential) dynamics, and leaves room for active/growth-driven development.

## 2. Layered architecture

```
genome (evolving genes)
  └─▶ [A: GRN/MLP — fixed developmental map]      weights fixed for now
       └─▶ θ (developmental parameters, per-cell/region field)
            └─▶ [B: developmental dynamics → equilibrium x*]   ← implicit-diff sensitivity
                 └─▶ [C: landmarks by cell-index]              ← correspondence seam
                      └─▶ [Procrustes → shape z]                ← geometric morphometrics
                           └─▶ [D: quant-gen — J, G=JMJᵀ, β, Δz̄=Gβ]
                                └─▶ [E: evolution loop — select · reproduce · repeat]
```

The two load-bearing interfaces are **θ** (developmental parameters) and **z**
(phenotype). Get those clean and every layer is swappable.

### Extension points (locked decisions in **bold**)
Status: A ✅ (`genome_map`), B ✅ (`mechanical`), C ✅ (`phenotype`),
D ✅ (`quantgen` + `response`), E ✅ (`evolution` — the loop, gate #4, §5c seams).

- **A — genome→θ map.** **Full GRN/MLP** (nonlinear, multi-parameter), *not*
  affine. **Genome = the network's inputs (the evolving genes); the network is
  the fixed, conserved developmental map** (matches M-U's `f`). Start
  feedforward (MLP); recurrent/dynamical GRN is the first extension (gives two
  nested equilibria; the same fixed-point sensitivity engine handles it).
  Evolving the *weights* (evolvability of development) is a separate, later axis.
- **B — developmental dynamics.** **Equilibrium sensitivity first** (fixed
  point), to reach Fig 3C. Seam for **trajectory sensitivity `s(t)`** along
  development (M-U's "the whole trajectory carries information") — turned on
  after equilibrium works, and mandatory once development is non-equilibrium
  (growth, ongoing division). Pluggable linear solver (dense small / CG large).
- **C — phenotype = landmarks.** **Landmarks** (biologically real; there is no
  bundle of shape descriptors in a genome). In the equilibrium phase, all
  individuals develop from the same initial tissue, so **cell-index gives free
  homology** → landmarks = a fixed subset of cells. When differential growth is
  added, homology breaks and the **correspondence seam becomes the
  Thompson-transformation / attention engine**. Readout ends with **Procrustes**
  alignment → geometric-morphometric shape space (differentiable; first smoke
  test may use raw centroid-normalized landmarks).
- **D — quant-gen.** `J` (implicit-diff) → `G = J M Jᵀ` (or `Σ σᵢ² sᵢ sᵢᵀ`), β,
  `Δz̄ = Gβ`. Reverse-mode VJP path `Jᵀβ` gives the response without forming G.
  Pluggable fitness/selection.
- **E — evolution loop.** Population → develop (vmap, GPU) → select → reproduce
  → repeat. Seams: reproduction model, drift, gene flow, **selection = env or
  player**. This layer *is* the science artifact and the game loop.

  > **⚠ vmap-over-population does not survive the organism scale (measured
  > 2026-07-16, RTX 4090 **Laptop** 16 GB — *not* a desktop 4090; see §5b's
  > provenance warning — `scale.relax_neighbor_list`).** It costs
  > **~2.2 KB/cell**, so a million-cell organism peaks at **2.21 GB → ~7
  > organisms per 16 GB laptop card** (**~10 per rented 24 GB desktop 4090**),
  > and 2 M cells OOMs outright on 16 GB. A population of
  > 1000 million-cell organisms needs **2.2 TB** — ~138 such GPUs *just to hold
  > one generation*. So "population = vmap axis" is a **small-organism design**
  > (fine at Phase 2's 19 cells, where 400 individuals vmap trivially; dead at
  > 10⁶). At organism scale the parallelism **inverts**: one organism saturates a
  > device and the population becomes *outer* — many independent runs across many
  > devices. The run unit changes with size: at 10²–10³ cells a run is a campaign
  > leg with the population vmapped inside; at 10⁶ cells a run is **one
  > individual's development**.
  >
  > Budget at 1 M cells, from measured throughput (2.08e7 cell-steps/s;
  > development ≈ 2·N·relax_steps ≈ 4e8 cell-steps ≈ **19 s/organism**):
  > `P=100, G=100` → 53 GPU-h/lineage; `P=1000, G=1000` → 5342 GPU-h/lineage,
  > ×30 replicates ×4 arms ≈ **641 000 GPU-h (73 GPU-years)**. This is fleet
  > territory, not one-box territory — see the `run-farm` thread.
  >
  > **⚠ The 73-GPU-year case is not purchasable, at any price.** Live survey by
  > the `run-farm` session (2026-07-16, 13 real rentals, $0.19): **Vast lists 598
  > GPUs total** across all types and box sizes, and only **~42% of created
  > instances ever reach `running`** → **~251 usable**. So 641 000 GPU-h is
  > **~106 days of wall-clock**, not a fleet-week. The modest end (6410 GPU-h)
  > is ~25 hours and is fine. Treat the worst case as **out of scope for this
  > marketplace** rather than merely expensive; an elastic provider (Modal) is
  > the only path to the top of the range.

  > **✅ The dense solver was the first wall — now fixed.** `equilibrate` now
  > defaults to matrix-free **CG-Newton** (`newton_solver="cg"`; `"pinv"` kept as
  > the reference it is gated against). The dense path's real cost turned out to
  > be **O(N³) memory, not O(N²)**: the Hessian is only 51 MB at N=1261, but
  > `jax.hessian` differentiates *through* `field_morse_energy`'s O(N²) pair
  > matrix and builds a `(2N, N, N)` intermediate — **32 GB at N=1261, 348 GB at
  > N=2791**. Measured on the 16 GB card: `pinv` OOMs at **N=1261** where `cg`
  > runs, and where both fit (N=331) CG is ~3.6× faster. The two reach the *same
  > organism*: they differ by ~1e-5 living **entirely** in the rigid modes
  > (physical component a million times smaller), and the Procrustes phenotype is
  > identical to 2.6e-16 — **the phenotype is solver-independent**, which is a
  > third independent confirmation of the anholonomy.
  >
  > **⚠ The next wall is not where it was predicted.** The expectation was that
  > removing the solver wall would expose the energy's O(N²) pair matrix (dying
  > ~N=20 000, fixable with the neighbour lists `scale.py` already implements).
  > **It doesn't. The binding constraint is the descent stage.** At N=1261
  > CG-Newton still fails (`max|F| ~ 1`, 5000 descent iterations exhausted):
  > gradient descent needs O(N) iterations to relax a large blob's
  > long-wavelength breathing mode, and a hex lattice at `spacing = r_eq` is not
  > the Morse equilibrium (second neighbours at √3 ≈ 1.73 sit inside
  > `r_max = 1.8`), so the whole blob must relax collectively. Nor can descent be
  > dropped: `newton_tol=1e2` ("always Newton, damped") **stalls at
  > `max|F| ~ 0.9` after 10–20 iterations** — far from the minimum H is not
  > positive-definite, so `H⁻¹∇E` is not a descent direction and damping cannot
  > rescue it. Newton is mesh-independent only *inside* the basin.
  >
  > So organism scale next needs a better **globalisation** (accelerated or
  > preconditioned descent — FIRE / Nesterov, standard in molecular statics,
  > O(√N) instead of O(N)), *then* the neighbour-list energy. The pipeline may
  > partly dodge this, since individuals are small perturbations of a reference
  > genome whose equilibrium is known — but **warm-starting from that reference
  > is a cheat**: it biases which basin development lands in and would suppress
  > exactly the multistability §3c measures. Development starts from a
  > primordium, or it isn't development.

### Placement (decision #4)
Everything lives in **`jax_morpho.evodevo`** for now (refactor later). The
implicit-diff sensitivity is a candidate to promote into core (it would also fix
`inverse_design`'s Jacobian), but is developed in `evodevo` first.

## 3. Phase 0 — exact Milocco–Uller calibration (before any mechanical engine)

Reference: their MATLAB at `github.com/lisandromilocco/DevStat-Bridge`.
Their development is a bistable toggle-switch gene network, integrated to t=50:

    ġ₁ = (2 + θ₁)/(1 + (g₂/2)²) − 0.4·g₁
    ġ₂ = (2 + θ₂)/(1 + (g₁/(3+u))²) − 0.4·g₂

phenotype = (g₁, g₂) at t=50; θ₁,θ₂ genetic, u environmental. Fig 3C: 5000
individuals, 10 loci/parameter, γ~N(0,1e-4), u~N(0,1.5e-3), θ₁ minor-allele-freq
fixed 0.5, θ₂ swept 0.5/2ᵏ; regression average effects → `G=Σ2pq αα ᵀ`;
`P=cov(phenotype)`; truncation-select 50% closest to optimum (4,4); compare the
observed recombinant response to `Δz̄=Gβ` (via G) vs `s` (naive, via P).

### Calibration ladder
- **0a — reproduce Fig 3C.** ✅ (`reference_mu.py`) — G predicts the response
  (small angle) while P misaligns at low allele frequency. *(Bit-identical is
  impossible across MATLAB/Python RNG; calibration = matching the figure's
  quantitative pattern — the G-vs-P angle crossover.)*
- **0b — reproduce Fig 1C and validate *our* sensitivity three ways.** ✅
  (`sensitivity.py`, `test_sensitivity.py`). On their exact ODE, forward-mode
  autodiff, reverse-mode autodiff, and implicit-diff at the steady state all
  agree (forward≡reverse to machine precision; implicit to ~1e-8 in float64),
  and `s·γ ≈ regression α` (ratio ~1.000, their Fig 1C). **Refined finding:**
  reverse-mode did **not** degrade here (2000-step ODE, float64, clean
  convergence) — so the mechanical-relaxation failure was a
  **float32/convergence/conditioning** issue, not a universal "long unrolls
  break reverse-mode" law. Implicit-diff stays the correct *core* tool
  (precision- and convergence-robust, scalable); on the mechanical engine we
  will use it and cross-check against reverse-mode with clean numerics.
- **0c — unify.** ✅ (`build_G_sensitivity`, `TestPhase0cSensitivityG`). G built
  from *our* sensitivity-derived average effects (`α = γ·s`, using the
  implicit-diff Jacobian at the population mean) equals their regression G
  (relative Frobenius ~0.01 at high MAF, ≤0.1 at low MAF) and predicts the
  response equally well (angle_G_sens ≈ angle_G at every frequency), both
  beating P at low MAF.

**Phase 0 complete: our autodiff/implicit sensitivity → G → breeder's equation
reproduces Milocco–Uller Fig 3C end-to-end.**

## 3b. Phase 1 — the sensitivity engine on the mechanical engine ✅

Swaps their ODE for our mechanical development, and gates it against finite
differences (validation-ladder rung 1). `mechanical.py` + `fixed_point.py`,
16 tests.

**θ is now a per-cell field** — adhesion `D[i]` and preferred spacing `r_eq[i]`,
mixed pairwise (`D_ij = √(D_i D_j)`, `r_eq_ij = (r_eq_i+r_eq_j)/2`). Differential
adhesion is the load-bearing sorting mechanism, so this is the seam layer A's
GRN writes into. Uniform fields reproduce `center_based`'s *forces* exactly
(the energies differ by a constant per pair — the shift `center_based` pays for
its C⁰ truncation; here a quintic switch makes the potential C², which the
Hessian requires).

### Gate #1 — implicit-diff ⟺ finite differences
**Relative difference `8.44e-10`** on the gauge-invariant subspace, 19-cell
tissue, non-uniform θ (38 parameters). Reproduce with
`python examples/demo_phase1_gate.py`. Three findings were load-bearing:

- **The equilibrium must be real.** Gate #1 divides an equilibrium difference by
  `2·eps`, so the FD reference is only as good as the solve. A textbook Armijo
  line search stalls at `max|F| ~ 1e-7` and then drifts *upward*: near the
  minimum the per-step energy decrease (~1e-14) falls below float64's absolute
  resolution of E (~2e-15) and the sufficient-decrease test becomes roundoff
  noise. That would have put ~5% noise on the reference and made the gate
  vacuous. `equilibrate` therefore hands off from Armijo descent to **projected
  Newton** (which tests gradients, never energy differences) and reaches
  `max|F| ~ 1e-14`.
- **The Hessian is exactly singular** — 3 zero modes in 2D (2 translations + 1
  rotation), spectrum `[0, 0, 0, 6.36, 9.56, ...]` against `λ_max ~ 263`. Not
  ill-conditioning: a degeneracy, forced by rigid-invariance of the energy.
  `jnp.linalg.solve` is the wrong call; the pseudo-inverse (equivalently,
  projected CG — the two agree to `5.7e-11`, and CG never forms the Hessian) is
  the right one.
- **A Gaussian cloud is not a tissue.** Scattered at the scale of `r_max` it
  relaxes into disconnected fragments; each carries its own zero modes, their
  relative placement is energetically free, and **∂x*/∂θ does not exist**. The
  first gate attempt failed this way (>3 zero modes, `|J| ~ 49`). Development
  must start from a cohesive blob — biologically obvious, and the condition for
  the IFT to apply.

### The gauge is anholonomic (why Procrustes is load-bearing)
Gate #1 passes on the *gauge-invariant* subspace, and the discarded part is not
a rounding detail — it is **entirely rotation**, and it is large (`0.446` against
a Jacobian of scale ~1.3). Decomposed: translation components `5.7e-08`/`3.7e-07`
(= FD noise; the centre of mass is conserved along the path to `4.6e-14`),
rotation `0.446`.

Each gradient step carries zero net torque, yet net rotation *accumulates*,
because the modes rotate with the shape as it deforms — a geometric phase, the
falling-cat effect. **The equilibrium form is a function of θ; its orientation is
a functional of the whole developmental trajectory.** No fixed-point method can
reproduce it, and none should.

> The claim then confirmed itself by accident. Phase 2 damped the Newton stage
> and moved the Armijo→Newton handoff, changing the *path* to the same
> equilibrium — and the rotational component fell from `5.17` to `0.446` while
> the gauge-invariant gate was unaffected (it improved, `4.6e-09` → `8.4e-10`).
> A quantity that changes when only the path changes is a property of the path.
> The magnitude of the anholonomy is therefore not a stable number to quote; its
> *presence*, and its confinement to the rotational mode, are the findings.

This promotes §2C's Procrustes decision from convenience to necessity: alignment
is not tidying-up before comparing shapes, it is what makes the phenotype a
well-defined function of the genotype at all. Absent it, "phenotype" carries a
path-dependent geometric-phase term. Pinned by `TestGaugeIsRotationOnly`.

## 3c. Phase 2 — GRN genome map, Procrustes phenotype, gate #2 ✅

Wires the rest of the spine: `genome_map.py` (layer A), `phenotype.py` (layer C),
`pipeline.py` (the composed chain), `quantgen.py` (layer D's opening move).
25 tests. Reproduce with `python examples/demo_phase2_gate.py`.

    a ──[A: GRN/MLP]──▶ θ ──[B: relax]──▶ x* ──[C: landmarks+Procrustes]──▶ z
    ∂z/∂a = (∂z/∂x*) · (∂x*/∂θ) · (∂θ/∂a)
             autodiff    implicit     autodiff

Only the middle factor needs the IFT; the outer two are explicit functions.

### The Phase-1 payoff: the composed chain is gauge-invariant
**`∂z/∂a` matches finite differences to `4.84e-09` — RAW, with no gauge
projection.** Phase 1's raw `∂x*/∂θ` vs FD failed at ~0.7 relative because of
the anholonomic rotation; gate #1 had to be stated on the gauge-invariant
subspace. Composing the Procrustes readout removes that contamination entirely:
`|∂z/∂x* · Z| = 1.4e-17` — the readout annihilates the rigid modes, so whatever
gauge the implicit solve chose cannot reach the phenotype. This is the
measurement that turns §2C's Procrustes decision from a convenience into a
structural requirement.

### Gate #2 — `G = J M Jᵀ` ⟺ empirical `Cov(z)`
19-cell tissue, 4 genes, n=400, common random numbers (same ξ at every σ, and G
built from the *empirical* covariance of the drawn genomes — so sampling noise
cancels to leading order and what remains is the map's nonlinearity):

| σ | rel. diff |
|---|---|
| 1e-2 | 1.05e-02 |
| 5e-3 | 4.80e-03 |
| 2.5e-3 | 3.15e-03 |
| **1.25e-3** | **1.85e-03** |

The discrepancy is controlled by σ, as a local claim must be. **The fitted
convergence *order* is deliberately not gated**: two error terms compete here —
an O(σ²) truncation (the naive O(σ) cancels because Gaussian genetic variation
has zero third moment) and an O(σ) finite-sample term — so the fit swings between
0.81 (n=400) and 2.03 (n=800). The magnitudes are robust; the order is not, and
reporting the flattering fit would be cherry-picking. Not a numerical artefact:
tightening the equilibrium tolerance from 1e-10 to 1e-13 changes the result in
no digit.

`rank(G) = 4 = n_genes`: development cannot express more independent directions
of variation than the genome supplies.

### Development is multistable — G is a *within-basin* object
At σ=0.05 the form displacement `|x* − x*(a0)|` is **bimodal**: ~0.02 for most
individuals, ~0.43 for a couple (2/120). Those are neighbour exchanges — the
tissue rearranges into a different packing, so **the phenotype is discontinuous
in the genome** and no local Jacobian can describe the crossing. G describes the
response *within* a basin and is silent about jumps between them. None occur at
σ ≤ 1.25e-3, which is why gate #2 is entitled to be a purely local claim there.

This is not an embarrassment for the framework: Milocco & Uller's own reference
model is a *bistable toggle switch*. Their development is multistable by
construction; ours turns out to be multistable by consequence.

Ruled out as explanations: Procrustes symmetry-branch flipping on the
near-C6-symmetric blob (alignment angles stay < 0.14° across the population) and
solver noise (tolerance-independent).

### Two instruments corrected along the way
- **Delaunay is the wrong basin fingerprint here.** A hex lattice is *maximally
  cocircular*, so its triangulation flips diagonals under infinitesimal
  perturbation with nothing physical changing: measured 23/120 flagged, **21
  false positives**. `mechanical.contact_topology` uses contacts instead.
  (`center_based.interior_side_counts` keeps Delaunay legitimately — it measures
  *disordered* packings, where cocircularity is measure-zero rather than the norm.)
- **Shape space is degenerate by 4, but not uniformly.** Three dimensions go
  *exactly* (centring ×2 and Procrustes optimality `Σ zᵢ × refᵢ = 0` are linear
  in z → singular values at machine zero); the fourth goes only
  *asymptotically*, because unit scale is nonlinear — at finite spread ε the
  cloud pokes out of the tangent plane by O(ε²), leaving a singular value of
  relative size O(ε). So the finite-ε linear rank is 2k−3, and 2k−4 is the ε→0
  tangent dimension where G lives.

## 3d. Phase 3 — the quant-gen layer and gate #3 ✅

`genetics.py` (diploid loci → genome), `response.py` (the one-generation
protocol), `quantgen.py` (β, `Δz̄ = Gβ`), plus the tangent shape space and the
implicit VJP/JVP. 16 tests; `examples/demo_phase3_gate.py`.

### Gate #3 — the Fig-3C pattern, on our development
Stated against the measurement's own resolution rather than an arbitrary
threshold. A response estimated at signal-to-noise `snr` can be tilted
`arcsin(1/snr)` by noise alone, so nothing can be *shown* to align better:

| p | angle_G | angle_P | snr | noise floor |
|---|---|---|---|---|
| 0.5 | **1.6–3.6°** | 20–29° | 19–23 | ~2.8° |
| 0.25 | **1.9–5.3°** | 53–114° | 11–14 | ~4–5° |

**G's error sits at the noise floor — consistent with exact.** P's is an order of
magnitude above it: a genuine systematic error. The claim reproduces.

### Three things that had to be right first
- **P is singular in ambient shape coordinates**, so `β = P⁻¹s` is meaningless
  there — not merely ill-conditioned. Fixed by working in the Procrustes
  **tangent space** (`phenotype.tangent_basis`, 2k−4 dims, Kendall's standard
  move): rank(P) goes 5-of-8 → 4-of-4.
- **Environment is load-bearing.** With no non-heritable variance `P = G` exactly
  and the comparison is vacuous. M-U's `u` exists for this reason; ours is extra
  non-heritable inputs to the same GRN.
- **Two protocol errors, both "I let something adapt that they hold fixed":**
  sweeping *every* gene's MAF (they hold θ₁ at 0.5 and sweep θ₂ — the anisotropy
  *is* the mechanism; without it G just shrinks uniformly and no contrast
  appears), and recomputing the optimum per population (selection then always
  pulls along whatever variance exists, so `s` can never misalign). Both produced
  plausible-looking nulls.

### What is *not* claimed
**The monotone degradation of P as p → 0 does not reproduce.** That tail is where
our SNR dies: the response shrinks ∝ 2pq while the noise floor is set by the
environment-dominated phenotypic sd and does not. M-U buy it with 5000
individuals × 50 replays ≈ 5e5 developments per point; we spend ~1e4.

**And it cannot be bought by raising the genetic variance.** Measured: σ_γ
0.02 → 0.08 lifts SNR but degrades `angle_G` 3.3° → 18.8°, because larger
perturbations leave the linear regime G is defined in. **Gate #2's
small-perturbation constraint and gate #3's SNR pull against each other** — sit
where G is accurate and admit the MAF range that costs. Closing that gap is a
compute problem (§5b), not a modelling one.

### The reverse-mode path — and why it needs the IFT, not autodiff
`Δz̄ = J M Jᵀβ` in **two solves** (`pipeline.lande_response_vjp`), independent of
gene *and* trait count, versus one solve per gene to form J. It cannot be written
as `jax.vjp` through the genome→phenotype map, for two independent reasons:
(1) it does not run — `equilibrate` is a `lax.while_loop`, which reverse-mode
autodiff does not support; (2) it would be wrong if it did — unrolling
differentiates the relaxation *path*, not the fixed point (§1). So the implicit
transpose (`fixed_point.implicit_vjp`, verified to 4.1e-11 against the dense
Jacobian) is what makes a reverse-mode path exist here at all. Pinned by a test
that asserts the naive route still raises.

## 3e. Phase 4 — the evolution loop and gate #4 ✅

`evolution.py`: population → develop → select → reproduce → repeat, with the §5c
seams built in. 13 tests; `examples/demo_phase4_gate.py`. This is the layer where
Phases 1–3 stop being a one-generation measurement and become a population moving
through time — and where the compounding question, which only exists over time,
gets asked.

### Gate #4a — neutral drift and the effective population size
With no selection, heterozygosity decays geometrically (log-linear R² > 0.98) at
`1/(2Ne)`. **The load-bearing number is `Ne ≈ 2N`, not `N`** — measured `Ne/N` =
1.9–2.0 across N = 20/40/80. The default mating gives every pair exactly two
offspring, so family-size variance is zero and `Ne ≈ 2N − 1` (the equal-family
case). Drift is half as strong as naive Wright-Fisher. This is a property of the
reproduction model, not the genetics; it was *measured before asserting*, which
is the only reason the gate reads Ne≈2N instead of "fixing" the code to hit N.

### Gate #4b — does the one-generation prediction compound? *Only while variance lasts.*
The honest finding, and a better one than "it compounds perfectly." Under strong
truncation selection:

- **The mean advances on the optimum and plateaus** — a heritability-limited
  selection limit. Heterozygosity collapses (0.50 → ~0.02 over 12 generations);
  the response saturates exactly as the variance runs out.
- **Per-generation `Gβ` tracks the realised response while variance is healthy,
  then fails.** 8-seed binned medians: **21°** at het≈0.48, **40°** at het≈0.40,
  **78°** at het≈0.30. The cause is specific: `β = P⁻¹s` destabilises as
  selection drives P toward singular. Phase 3 validated `Δz̄ = Gβ` at *constant*
  high heterozygosity; iterating it *through variance collapse* is a strictly
  harder claim, and this is where our tooling's boundary is.
- **G must be recomputed each generation.** A G frozen at generation 0 describes a
  population that no longer exists — its (fixed) G times a growing β overshoots
  the recomputed prediction by the end of the run.

This is forward/forbidding with a number: the prediction is good above het≈0.42
and gone below ≈0.30, and the response is bounded by the cumulative genetic
variance. It is not a menu-fit — the failure boundary is a prediction, tested.

### The §5c seams — a testbed for viral punctuation
The variation operator is a seam, not a hardcoded step, because the punctuation
experiment is exactly a swap of it:

- **`point_mutation(rate)`** — gradualism: each locus independently re-drawn. A
  small, within-basin perturbation (Phase 2/3 regime, G predicts).
- **`retroviral_insertion(rate)`** — punctuation: an infected offspring gets a
  donor lineage's **entire gene**, wholesale. A coordinated multi-locus jump,
  which is Phase 2's *basin-crossing* regime — the phenotype is discontinuous
  there and no local Jacobian describes it. **Requires a donor pool** (it raises
  without one), because horizontal transfer has to transfer *from somewhere* —
  which is why §5c also requires multi-lineage populations, and why
  `Architecture.gene_of_locus` was kept (it supplies the gene block an insertion
  overwrites). The operator exists so the seam is real and exercised; the
  punctuation *study* (does viral macromutation reach basins gradualism cannot?)
  is the science this testbed now supports.

The loop takes `develop=False` for the neutral null (drift needs no phenotype)
and rejects `measure_response=True` without development (nothing to measure a
response in) — both to keep a wrong call loud rather than silent.

Next: the deferred axes (§5) — trajectory sensitivity, real-DNA genome,
non-potential/growth-driven development, endogenous virus emergence — and the
punctuation study the seams now make possible. Organism scale still wants the
FIRE/Nesterov globalisation flagged in §2E.

## 4. Validation ladder (each rung a known-answer gate)
1. implicit-diff sensitivity ⟺ finite-difference Jacobian. ✅ `8.44e-10` (§3b)
2. `G = J M Jᵀ` ⟺ empirical `Cov(phi(a))` in the small-perturbation regime.
   ✅ `1.85e-03` at σ=1.25e-3, shrinking with σ (§3c)
3. reproduce M-U Fig 3C / Fig 1C on their model (Phase 0). ✅ — and the
   *pattern* with our own development (Phase 3, §3d): G at the noise floor,
   P an order of magnitude above it
4. multi-generation loop matches quantitative-genetic expectations. ✅ Ne≈2N;
   Gβ tracks the response above het≈0.42, fails below ≈0.30 (§3e)
No layer ships without its number.

## 5. Sequencing
0. **Phase 0 calibration** (0a ✅ → 0b ✅ → 0c ✅).
1. implicit-diff sensitivity engine + gate #1. ✅ (§3b — rel. diff `8.44e-10`)
2. GRN genome map + landmark/Procrustes phenotype (non-degenerate) + gate #2.
   ✅ (§3c — gate #2 `1.85e-03`; chain vs FD raw `4.84e-09`)
3. quant-gen layer on the mechanical engine; reproduce the Fig-3C *pattern* with
   our development + gate #3. ✅ (§3d — G's error at the measurement noise floor,
   P's an order of magnitude above it)
4. evolutionary loop (the closed loop / game substrate) + gate #4 — built to
   serve §5c's viral-punctuation testbed. Prerequisite met: CG-Newton (§2E).
5. deferred: trajectory `s(t)`, real-DNA genome, non-potential development, the
   game presentation layer, endogenous virus emergence (§5c).

## 5b. Architecting for massive campaigns

Design target: campaigns at organism scale (10⁶ cells) across populations,
generations, replicate lineages and arms. §2E records the wall; this section
records what to do about it. **The binding constraint is money, not wall-clock**
— so the optimisation target is *cost per unit of science*, and cost has to be a
first-class quantity in the architecture rather than a post-hoc surprise.

> **⚠ Partly falsified by the live survey (2026-07-16), and worth keeping visible
> rather than quietly editing.** "Money, not wall-clock" was written against an
> assumed ~1000-GPU spot pool. The real marketplace has **~251 usable GPUs**, which
> is a *hard* ceiling no budget raises — so above roughly $5k the two constraints
> swap and **wall-clock binds**: $20k buys ~11 days, and §2E's worst case is ~106.
> Money still binds in the $1k–$5k band this section is aimed at, so the framing
> survives *where it is used*. But cost-per-unit-of-science is now the right target
> only below a capacity ceiling that the architecture must also model.

### Who this is for (it decides the requirements)
Not only us on one box. The intended user includes **funded researchers — the
Milocco–Uller community itself — spending grant money at $1k–$20k per campaign**.
From the measured cost model that envelope is not aspirational:

| budget | GPU-h (spot ~$0.30) | campaign it buys at 10⁶ cells | wall-clock @ ~251 usable |
|---|---|---|---|
| $1 000 | 3 300 | P=G≈175, 10 replicates × 2 arms | ~13 h |
| $5 000 | 16 700 | P=G≈390, 10 replicates × 2 arms | ~2.8 days |
| $20 000 | 66 700 | P=G≈790, 10 replicates × 2 arms | ~11 days |

(The 73-GPU-year worst case in §2E is ~$190k — *above* grant scale, **and not
purchasable on Vast at any price**: ~106 days of wall-clock against a live-surveyed
~251 usable GPUs. The band that matters is $1k–$20k, and it lands squarely on a
publishable design.)

**The wall-clock column is not the GPU-h column divided by 1000.** The live survey
(§2E) found **598 GPUs listed, ~251 usable** — that is the whole marketplace, not
our share of it. A campaign estimator that reports GPU-h without reporting
*elapsed* will let someone buy a feasible-looking price for an infeasible schedule.
Report both.

Grant money carries accountability, which turns four soft nice-to-haves into
hard requirements:
- **A pre-launch estimate and a hard spend cap.** Someone spending a grant cannot
  discover the cost afterwards. `(P, G, N, replicates, arms) → GPU-h, $` must be
  callable *before* launch, and the cap must be enforced, not advisory.
- **Restart-exactness.** A $5k campaign that dies at 80% and cannot resume is a
  $4k loss, not an inconvenience.
- **Provenance as a published artifact.** Config hash, engine version, and
  **cost** belong in the record: "this result cost $X on Y hardware" is
  reproducibility metadata, not trivia.
- **Portability.** Brokers and credentials must not assume our box.

And one that is ours alone, and follows from the honesty frame (§0, §6):
- **Ship the Milocco–Uller reference as a built-in campaign arm.** `reference_mu`
  already reproduces their Fig 3C end-to-end. Making it an arm any campaign can
  include means **every campaign carries its own known-answer gate** — the
  validation ladder travels with the tool instead of living in our test suite. If
  an external scientist is to trust a G computed by this engine, the calibration
  should re-run alongside their result, not be cited from a paper.

### Measured cost model (RTX 4090 **Laptop**, 16 GB, 2026-07-16)

> **⚠ Provenance warning — read before quoting any row.** This was measured on an
> **RTX 4090 Laptop GPU**, which is **not the card any of this runs on in
> production**. It is a different die with roughly half the machine:
>
> | | laptop 4090 (the box these were measured on) | desktop 4090 (what Vast rents) |
> |---|---|---|
> | die | AD103 | AD102 |
> | CUDA cores | 9 728 | 16 384 |
> | VRAM | **16 GB** | **24 GB** |
> | bandwidth | ~576 GB/s | ~1008 GB/s |
>
> The original header said "RTX 4090, 16 GB" — honest about the memory, and
> misleading about everything else, since "RTX 4090" reads as the desktop part.
> The `run-farm` session read it that way, correctly, and derived ~10 organisms
> per card against this table's ~7. **They were right.** Caught 2026-07-16 only
> because the two numbers disagreed across sessions.
>
> **The general lesson, and it is the one this section exists to enforce: the
> provenance of a number decays faster than the number.** `2.08e7 cell-steps/s`
> survived fine; *"on what?"* fell off within hours, under a header that already
> said "reproduce before trusting". Every row below is **pessimistic** for a
> rented card by an unknown factor — pessimistic is the safe direction for a
> budget, but the attribution was still wrong.

| quantity | measured (laptop 4090) | on a rented desktop 4090 |
|---|---|---|
| memory | **~2.2 KB/cell** (float32, incl. neighbour list + autodiff tape) | same (hardware-independent) |
| throughput | **2.08e7 cell-steps/s** at 1 M cells (47.98 ms/relax step) | **unmeasured** — see below |
| 1 M-cell organism | 2.21 GB → **~7 per 16 GB card**; 2 M cells OOMs | **~10 per 24 GB card** (= 24/2.21; solid) |
| development, primordium→adult | ≈ 2·N·relax_steps ≈ 4e8 cell-steps ≈ **19 s** | **unmeasured** — see below |
| float64 | **~4x slower, 2x memory** (bandwidth-bound — *not* the 64x the 1:64 FP64:FP32 spec ratio suggests; measured, because the spec-sheet guess is wrong by 16x) | ratio likely similar (both bandwidth-bound), magnitude unmeasured |

**Why "unmeasured" and not an extrapolation.** The obvious move is to scale
throughput by the bandwidth ratio (576 → 1008 GB/s ⇒ ~1.75x), since the row above
establishes the inner loop *is* bandwidth-bound. **Do not.** That same float64 row
is a monument to exactly this reasoning failing: the spec ratio predicted 64x and
the measurement said 4x — **wrong by 16x**. A spec-sheet guess is what this table
exists to replace, and one guess per table is one too many. The number is cheap to
get — one already-billed desktop 4090, `examples/demo_gpu_scaling.py` at 1 M cells
— and it is requested from the `run-farm` session, who are renting anyway.

Everything below follows from those numbers. A campaign estimator should be
a function of them, run *before* launch — `(P, G, N, replicates, arms) → GPU-h, $`
— not a scratch calculation after the bill. **It should take the hardware as an
argument**, not bake in the box it was written on; that is the whole content of the
provenance warning above.

### The levers, in order of leverage
1. **Keep float64 out of the inner loop.** The sensitivity engine needs it (§1's
   Armijo finding is *about* float64's absolute resolution). Development and
   fitness do not. Run development in float32 and compute `J`/`G` in float64
   **rarely** — at the population mean, not per individual. Costs 4x time and
   halves the organisms per card wherever it leaks.
2. **Adaptive fidelity, keyed on the basin criterion.** §3c measured the
   threshold: *zero* crossings at σ ≤ 1.25e-3, ~2% at σ=0.05, and within a basin
   `G` reproduces the phenotype covariance to 1.85e-3. So most offspring are
   linearly predictable from the ancestor's `J` and do not need developing.
   Spend full development where the predicted perturbation approaches the
   crossing threshold, and use `G` elsewhere.
   > **This must stay multi-fidelity, never surrogate-replacement.** Predicting
   > phenotypes with `G` and then selecting on them is quantitative genetics
   > wearing the engine's clothes: it would make `G` self-confirming and destroy
   > the ability to observe the one thing the mechanistic engine exists to
   > observe — where the linearisation *fails*. Always develop a random
   > subsample and measure the drift. The saving is real; the check is what keeps
   > it honest (§6, and `insight_morphospace_falsifiability`).
3. **Checkpoint at generation granularity, not mid-development.** The evolution
   loop's state is the *genome population* — kilobytes, not field arrays — so
   checkpointing it is nearly free. Mid-development checkpointing is pointless:
   at 19 s/organism a preempted development is cheaper to redo than to persist.
   A preempted 5000-GPU-hour lineage is not.
4. **Resolution ladder.** 10⁶ cells is a *target*, not a requirement of every
   fitness evaluation. Establish the coarsest resolution at which the selected
   phenotype is resolution-invariant, and pay for full resolution only where the
   science needs it. Unmeasured; worth a gate of its own.

### Parallelism, restated
Population-as-vmap-axis holds only while an organism is small (§2E). At organism
scale the run unit is **one individual's development**, the population is outer,
and the fleet is the parallelism. The campaign axes `(arm, replicate_seed, …)`
are the same in both regimes; only the granularity moves. Design the run config
so both fit — see the `run-farm` thread.

> **⚠ "A run is one individual's development" is a claim about the *science*, not
> about *dispatch*, and reading it as dispatch buys the expensive design.** One
> development is the indivisible unit of work (it cannot be split across devices,
> and §5b lever 3 says a preempted one is cheaper to redo than persist). It is
> **not** the unit of rental. The `run-farm` session measured the reason
> (2026-07-16, 13 live rentals): **the failure tax is fixed per host *acquired*
> — ~$0.02–0.05 to get one host that boots — independent of how long the work then
> runs**, because ~58% of created instances never reach `running`. At 19 s per
> development, renting per run pays ~100x the tax over the work; amortised over a
> long leg it is noise.
>
> **So the run unit and the dispatch unit must be allowed to differ**: a leg
> batches many individuals onto one acquired host. The multi-GPU case compounds it
> — the tax is per *rental*, not per GPU, so an 8x box is ~8x cheaper per GPU, and
> at **~10 organisms per 24 GB card** that is **~80 organisms in flight per single
> acquisition**. One boot-lottery ticket, eighty organisms.
>
> The sentence above is the hazard, not the model: it reads naturally and leads
> straight to renting per 19-second run.

## 5c. Standing requirement — a testbed for viral-punctuated evolution

Jim's requirement, and it is a design constraint on layer E rather than a later
feature: **whatever we build must be a capable testbed for Villarreal's theory of
retrovirus-punctuated evolution.** Morphospace already has the empirical hook —
its macromutation A/B test found viral > point-mutation fitness at **8–9×
across 3 seeds** (Kauffman/Villarreal), on the DNA-string GRN stack.

### Phase 2 already found the mechanism this needs
This is not a bolt-on to the G-matrix programme; it is the other half of it.

* **Point mutation** is a *small* perturbation → stays in a developmental basin
  → §3c's gate #2 regime → **G predicts the response** (gate #3: G's error sits
  at the measurement noise floor).
* **A retroviral insertion is not a small perturbation.** It is a large,
  coordinated, multi-gene jump — exactly §3c's basin-crossing regime, where the
  phenotype is **discontinuous in the genome** and no local Jacobian can describe
  the transition.

So **the basin structure is a candidate mechanism for punctuation itself**: viral
macromutation jumps between developmental attractors; point mutation diffuses
within one. That reframes the 8–9× viral advantage as *access to basins gradualism
cannot reach*, and it is forward/forbidding rather than a fit — the crossing rate
as a function of insertion size is a prediction, and the replay measures it.
G-predicts-the-response and G-is-silent-across-basins are the same finding seen
from two sides.

### What the architecture must therefore preserve
- **A pluggable variation operator.** Mendelian `recombine` (`genetics.py`) is
  one; point mutation is another; **retroviral insertion is a third** — a block
  of loci overwritten *from a donor lineage* (horizontal transfer), not a local
  perturbation. `Architecture.gene_of_locus` already gives the block structure an
  insertion needs; keep the operator a seam, not a hardcoded step of the loop.
- **Multi-lineage populations with contact between them.** Horizontal transfer is
  meaningless within a single panmictic pool — the donor has to come from
  somewhere. This is the same replicate-lineage substrate the campaign work wants
  (see the `run-farm` thread), which is convenient: one mechanism, two motives.
- **Do not assume small perturbations anywhere structural.** The cost levers in
  §5b are the live risk: adaptive fidelity keyed on the basin threshold would
  silently *skip developing* exactly the viral macromutations that are the
  experiment. Multi-fidelity, never surrogate-replacement — and viral variants
  always develop.

### Bonus / north star: an evolutionary loop that *creates* viruses
Jim's stretch goal, and honestly labelled as one. Endogenous emergence — rather
than an imposed insertion operator — is a different class of model: it needs
genome elements that replicate on their **own** schedule (not the host's),
transmit horizontally, and are selected at a level below the individual. That is
a multilevel-selection substrate, not a variation operator, and nothing here
currently supplies it.

It is worth naming anyway, because it changes what "genome" must mean. The
current genome is a fixed-length vector feeding a fixed MLP; a genome that can
*acquire* elements is variable-length, which the MLP forbids and a
recurrent/attention-over-genes map would allow. That is already flagged in §2A as
the first extension to layer A — so the two roads meet. **Not a Phase 4 goal;
recorded so Phase 4 does not build something that precludes it.**

## 6. Guardrails / non-goals
- Falsifiability: every layer validated against a known answer; no fitting-as-
  evidence.
- Honesty labels: "sufficiency/construction," never "explains." Credit M-U.
- Verify sources against primary PDFs/code (we were burned once by a
  confabulated summary).
- Non-goal (now): claiming this is how real organisms evolve — a constructive,
  not causal, claim.
