# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

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
