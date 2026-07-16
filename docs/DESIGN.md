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
**Relative difference `4.58e-09`** on the gauge-invariant subspace, 19-cell
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
  projected CG — the two agree to `1.3e-11`, and CG never forms the Hessian) is
  the right one.
- **A Gaussian cloud is not a tissue.** Scattered at the scale of `r_max` it
  relaxes into disconnected fragments; each carries its own zero modes, their
  relative placement is energetically free, and **∂x*/∂θ does not exist**. The
  first gate attempt failed this way (>3 zero modes, `|J| ~ 49`). Development
  must start from a cohesive blob — biologically obvious, and the condition for
  the IFT to apply.

### The gauge is anholonomic (why Procrustes is load-bearing)
Gate #1 passes on the *gauge-invariant* subspace, and the discarded part is not
a rounding detail — it is **entirely rotation**, and it is large (~5.2 against a
Jacobian of scale ~1). Decomposed: translation components `1.7e-07`/`7.6e-08`
(= FD noise; the centre of mass is conserved along the path to `3.8e-15`),
rotation `5.17`.

Each gradient step carries zero net torque, yet net rotation *accumulates*,
because the modes rotate with the shape as it deforms — a geometric phase, the
falling-cat effect. **The equilibrium form is a function of θ; its orientation is
a functional of the whole developmental trajectory.** No fixed-point method can
reproduce it, and none should.

This promotes §2C's Procrustes decision from convenience to necessity: alignment
is not tidying-up before comparing shapes, it is what makes the phenotype a
well-defined function of the genotype at all. Absent it, "phenotype" carries a
path-dependent geometric-phase term. Pinned by `TestGaugeIsRotationOnly`.

Next: Phase 2 — GRN genome map + landmark/Procrustes phenotype + gate #2.

## 4. Validation ladder (each rung a known-answer gate)
1. implicit-diff sensitivity ⟺ finite-difference Jacobian. ✅ `4.58e-09` (§3b)
2. `G = J M Jᵀ` ⟺ empirical `Cov(phi(a))` in the small-perturbation regime.
3. reproduce M-U Fig 3C / Fig 1C on their model (Phase 0). ✅
4. multi-generation loop matches quantitative-genetic expectations.
No layer ships without its number.

## 5. Sequencing
0. **Phase 0 calibration** (0a ✅ → 0b ✅ → 0c ✅).
1. implicit-diff sensitivity engine + gate #1. ✅ (§3b — rel. diff `4.58e-09`)
2. GRN genome map + landmark/Procrustes phenotype (non-degenerate) + gate #2.
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
