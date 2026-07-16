# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

## [Unreleased] — Phases 1 & 2: our development, end to end

Phase 0 calibrated the sensitivity machinery against Milocco & Uller's ODE.
Phase 1 swaps in **our** development — a tissue relaxed to mechanical
equilibrium — and gates it against finite differences. Phase 2 wires the rest of
the spine (genome → θ → x* → landmarks → shape) and gates the G-matrix against a
developed population.

### Added — Phase 2: the composed genotype→phenotype map
- **`genome_map`** (layer A) — a nonlinear GRN/MLP from genome to per-cell θ
  field, via positional information (`θ_i = MLP([a, u_i])`): the same conserved
  network read against each cell's position. The genome is the network's *input*;
  the network is the fixed developmental map. Deliberately not affine — an affine
  map would assume away the question a development-derived G exists to answer.
  Outputs are sigmoid-bounded so no genome can tear the tissue into fragments
  (where `∂x*/∂θ` would not exist).
- **`phenotype`** (layer C) — landmarks (homologous by cell index) + a
  differentiable Procrustes shape, using the 2D closed form (`atan2`) rather than
  an SVD, whose gradient is singular when singular values coincide — not a
  hypothetical for a near-round tissue.
- **`pipeline`** — the composed chain and its Jacobian
  `∂z/∂a = (∂z/∂x*)(∂x*/∂θ)(∂θ/∂a)`, plus `develop_population` /
  `phenotype_population`: the `vmap`'d GPU path.
- **`quantgen`** — `G = J M Jᵀ` (Milocco & Uller Eq. 12 in matrix form).
- **Gate #2**: `G = J M Jᵀ` vs the empirical covariance of a developed
  population = **`1.85e-03`** at σ=1.25e-3, shrinking with σ. 25 tests;
  `examples/demo_phase2_gate.py`.

### Phase 1's payoff, measured
The composed `∂z/∂a` matches finite differences to **`4.84e-09` raw — with no
gauge projection**, where Phase 1's raw `∂x*/∂θ` failed at ~0.7 because of the
developmental anholonomy. The Procrustes readout annihilates the rigid modes
(`|∂z/∂x* · Z| = 1.4e-17`), so the gauge cannot reach the phenotype. This is what
promotes the Procrustes readout from a convenience to a structural requirement.

### Notes — Phase 2
- **Development is multistable; G is a within-basin object.** At σ=0.05 the form
  displacement is bimodal (~0.02 vs ~0.43): the tissue rearranges its neighbours
  and the phenotype is **discontinuous in the genome**, so no local Jacobian can
  describe the crossing. None occur at σ ≤ 1.25e-3, which is what entitles gate
  #2 to be a local claim. Fittingly, Milocco & Uller's reference model is itself
  a bistable toggle switch — their development is multistable by construction,
  ours by consequence.
- **The convergence order of gate #2 is deliberately not gated.** Two error terms
  compete (an O(σ²) truncation — the naive O(σ) cancels because Gaussian genetic
  variation has zero third moment — and an O(σ) finite-sample term), so the
  fitted order swings between 0.81 and 2.03 with sample size. The magnitudes are
  robust; the order is not.
- **Delaunay is the wrong basin fingerprint on a lattice.** A hex packing is
  maximally cocircular, so its triangulation flips diagonals under infinitesimal
  perturbation with nothing physical changing: 23/120 flagged, **21 false
  positives**. `mechanical.contact_topology` uses contacts instead.
  (`center_based.interior_side_counts` keeps Delaunay legitimately — it measures
  disordered packings, where cocircularity is measure-zero.)
- **Shape space is degenerate by 4, but not uniformly**: three dimensions go
  exactly (linear constraints → machine zero), the fourth only asymptotically
  (unit scale is nonlinear → a singular value of relative size O(ε)). So finite-ε
  linear rank is 2k−3; 2k−4 is the tangent dimension where G lives.

### Added — Phase 1: `jax_morpho.evodevo`
- **`mechanical`** — development as relaxation of a tissue to mechanical
  equilibrium, with **θ a per-cell field** (adhesion `D[i]`, preferred spacing
  `r_eq[i]`) rather than a couple of global scalars. Differential adhesion is the
  load-bearing sorting mechanism, so this is the seam a genome can write into.
  `equilibrate()` reaches a *genuine* fixed point (`max|F| ~ 1e-14`) and reports
  its residual; a quintic switch makes the truncated potential C², which the
  Hessian requires.
- **`fixed_point`** — implicit sensitivity of a general fixed point `F(x*,θ)=0`,
  with energy minimisation as the default instance. Dense pseudo-inverse and
  matrix-free projected CG (Hessian-vector products only) paths, plus
  `rigid_modes` and a finite-difference reference.
- **Gate #1** (validation-ladder rung 1): implicit-diff ⟺ finite differences to
  **`8.44e-10`**. 16 tests; `examples/demo_phase1_gate.py`.
- `equilibrate`'s Newton stage is **damped** — an undamped step is only
  trustworthy inside the quadratic basin, and handing over too early made the
  solve *freeze* at the handoff residual instead of converging slowly. A guess
  about where the basin starts should degrade, not fail.

### Fixed / corrected
- **`center_based.relax` does not converge**, and v0.2.0's explanation of why
  autodiff failed through it was wrong. It is not that reverse-mode "degrades
  with unroll length" — Phase 0b already found a 2000-step unroll that didn't
  degrade at all. The clipped fixed step turns the stiffest mode's linear
  instability into a *stable period-2 limit cycle*: `|∇E| = 3.34` at 200 steps
  and still 3.34 at 100 000, `|p_{k+2} − p_k| = 1e-16`. Autodiff was
  differentiating an orbit, not an equilibrium. The conclusion (use implicit
  diff) stands; the argument is replaced. `relax`'s behaviour is unchanged — the
  epithelial topology calibration is tuned against it — but it is now documented
  as unfit for equilibrium work, and `evodevo.mechanical.equilibrate` is the
  supported path. See docs/DESIGN.md §1.

### Notes
- **The Hessian at a tissue equilibrium is exactly singular** (3 zero modes in 2D:
  2 translations + 1 rotation) — a degeneracy forced by rigid-invariance, not
  ill-conditioning. `linalg.solve` is the wrong tool; the pseudo-inverse is the
  right one.
- **The developmental gauge is anholonomic.** Relaxation conserves the centre of
  mass exactly (~1e-14), but *not* orientation: zero net torque at every step
  still accumulates net rotation as the shape deforms — a geometric phase (the
  falling-cat effect). The equilibrium **form** is a function of θ; its
  **orientation** is a functional of the whole developmental path. This makes the
  planned Procrustes readout load-bearing rather than cosmetic: it is what makes
  the phenotype a well-defined function of the genotype.
- A Gaussian cloud of cells is not a tissue — it relaxes into disconnected
  fragments and ∂x*/∂θ genuinely does not exist. Development must start cohesive.

## [0.2.0] — Closing the developmental loop: calibrated evo-devo

This release adds **`jax_morpho.evodevo`**, the layer that turns jax-morpho from
a differentiable tissue engine into a substrate for **connecting development to
evolutionary dynamics** — building the mechanistic, differentiable
genotype→phenotype map that quantitative genetics normally treats as a black
box (the additive-genetic covariance, or "G-matrix"), with autodiff and at
scale.

Rather than assert that our engine can do this, v0.2.0 **calibrates the tooling
against a published result where the answer is known** before any of our own
developmental model is substituted: Milocco & Uller (2026, *PNAS*), "Bridging
developmental and statistical approaches to variation and evolution." Their
concept — a development-derived G and its use in the breeder's equation — is
reproduced end-to-end through our differentiable sensitivity machinery. Credit
for the concept is theirs; our contribution is the reverse-mode / implicit-diff
/ scalable realization.

### Added — `jax_morpho.evodevo`
- **`reference_mu`** — a faithful port of their bistable toggle-switch gene
  network and **Figure 3C**. Reproduces the central result: a development-derived
  G predicts the one-generation response to selection across allele frequencies
  (small angle to the observed recombinant response), while the phenotypic
  covariance P — often used as a proxy for G — misaligns badly at low
  minor-allele frequency. Development runs in JAX (RK4, `vmap` over the
  population, GPU-parallel and autodiff-ready).
- **`sensitivity`** — the developmental sensitivity ∂phenotype/∂parameter (the
  Jacobian of the genotype→phenotype map, the object a G-matrix is built from),
  computed three ways: forward-mode autodiff (≡ the forward-variational method),
  reverse-mode autodiff, and **implicit differentiation at the developmental
  equilibrium** (`∂x*/∂θ = −(∂f/∂x)⁻¹(∂f/∂θ)` — one linear solve, independent of
  how the equilibrium was reached; the scalable, numerically robust core tool).
  Validated: all three agree on the reference model, and sensitivity ×
  allelic-effect equals the Fisher regression average effect (**Figure 1C**).
- **`build_G_sensitivity`** — G assembled from our sensitivity-derived average
  effects (α = γ·s) equals their regression-derived G and predicts the response
  equally well, both beating P at low allele frequency. The loop is closed
  through our own differentiable tooling.
- **`docs/DESIGN.md`** — the architecture and calibration ladder (0a→0b→0c), the
  layered pipeline (genome → GRN/MLP → developmental parameters → equilibrium →
  landmarks → shape → G → selection), and the locked design decisions.

### Notes
- A recorded finding worth its own line: naive reverse-mode autodiff through an
  *under-converged* iterative developmental solve can yield a wrong sensitivity
  that degrades with unroll length; on a cleanly-converged solve all three
  methods agree. The equilibrium **implicit-diff** path sidesteps this and is
  the recommended tool.
- Phase 0 is calibration only — the mechanical morphogenesis engine replaces the
  toggle switch in the next phase.

### Tests & demos
10 `evodevo` tests (~7 s); runnable demos `examples/demo_mu_fig3c.py` and
`examples/demo_mu_sensitivity.py`.

## [0.1.1]

### Added
- Zenodo concept + version DOIs and PyPI badges.

## [0.1.0]

### Added
- Initial release: differentiable center-based tissue engine (Morse relaxation,
  cell division, growth, topology + gyration shape descriptors); `jax_md`
  neighbor-list scaling to 1–2M cells on GPU; gradient-based inverse design of
  tissue shape; and a genome→mechanics→form map differentiable to the genome.
