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
finite differences and *degrades with unroll length*. The correct object is the
**implicit sensitivity of the developmental equilibrium**.

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
reproduces Milocco–Uller Fig 3C end-to-end.** Next: Phase 1 swaps the ODE for
the mechanical morphogenesis engine, sensitivity method already proven.

## 4. Validation ladder (each rung a known-answer gate)
1. implicit-diff sensitivity ⟺ finite-difference Jacobian.
2. `G = J M Jᵀ` ⟺ empirical `Cov(phi(a))` in the small-perturbation regime.
3. reproduce M-U Fig 3C / Fig 1C on their model (Phase 0).
4. multi-generation loop matches quantitative-genetic expectations.
No layer ships without its number.

## 5. Sequencing
0. **Phase 0 calibration** (0a ✅ → 0b → 0c).
1. implicit-diff sensitivity engine + gate #1.
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
