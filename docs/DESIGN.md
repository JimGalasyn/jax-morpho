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
D partial (`quantgen` — G only; β and Δz̄ are Phase 3), E not started.

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
  > 2026-07-16, RTX 4090 16 GB, `scale.relax_neighbor_list`).** It costs
  > **~2.2 KB/cell**, so a million-cell organism peaks at **2.21 GB → ~7
  > organisms per 16 GB card**, and 2 M cells OOMs outright. A population of
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

  > **⚠ The dense solver dies here too.** `mechanical.equilibrate` polishes with
  > Newton via `jnp.linalg.pinv` on a **dense** (2N × 2N) Hessian — at 1 M cells
  > that matrix is **16 TB**. The matrix-free projected-CG path in `fixed_point`
  > (Hessian-vector products only, never forms H) is the *only* solver that
  > reaches organism scale. It is built and validated (agrees with `pinv` to
  > 1.3e-11) but **at 19 cells**, and `equilibrate` does not use it yet.
  > Promoting the Newton stage to CG-Newton is a prerequisite for layer E at
  > scale, not an optimisation.

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

Next: Phase 3 — the quant-gen layer proper (β, `Δz̄ = Gβ`, the reverse-mode `Jᵀβ`
path) + gate #3: reproduce the Fig-3C *pattern* with our development.

## 4. Validation ladder (each rung a known-answer gate)
1. implicit-diff sensitivity ⟺ finite-difference Jacobian. ✅ `8.44e-10` (§3b)
2. `G = J M Jᵀ` ⟺ empirical `Cov(phi(a))` in the small-perturbation regime.
   ✅ `1.85e-03` at σ=1.25e-3, shrinking with σ (§3c)
3. reproduce M-U Fig 3C / Fig 1C on their model (Phase 0). ✅
4. multi-generation loop matches quantitative-genetic expectations.
No layer ships without its number.

## 5. Sequencing
0. **Phase 0 calibration** (0a ✅ → 0b ✅ → 0c ✅).
1. implicit-diff sensitivity engine + gate #1. ✅ (§3b — rel. diff `8.44e-10`)
2. GRN genome map + landmark/Procrustes phenotype (non-degenerate) + gate #2.
   ✅ (§3c — gate #2 `1.85e-03`; chain vs FD raw `4.84e-09`)
3. quant-gen layer on the mechanical engine; reproduce the Fig-3C *pattern* with
   our development + gate #3.
4. evolutionary loop (the closed loop / game substrate) + gate #4.
5. deferred: trajectory `s(t)`, real-DNA genome, non-potential development, the
   game presentation layer.

## 6. Guardrails / non-goals
- Falsifiability: every layer validated against a known answer; no fitting-as-
  evidence.
- Honesty labels: "sufficiency/construction," never "explains." Credit M-U.
- Verify sources against primary PDFs/code (we were burned once by a
  confabulated summary).
- Non-goal (now): claiming this is how real organisms evolve — a constructive,
  not causal, claim.
