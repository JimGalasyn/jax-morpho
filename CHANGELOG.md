# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

## [0.2.0]

### Added
- **`jax_morpho.evodevo`** — closing the genotype→development→selection loop.
  Phase 0 calibrates our tooling against Milocco & Uller (2026 PNAS) on the
  system where the answer is known, before substituting our own developmental
  engine:
  - `reference_mu` — faithful port of their bistable toggle-switch model and
    **Fig 3C**: a development-derived G predicts the one-generation response to
    selection while the phenotypic covariance P misaligns at low minor-allele
    frequency.
  - `sensitivity` — the developmental sensitivity ∂phenotype/∂parameter three
    ways (forward-mode autodiff, reverse-mode autodiff, and **implicit-diff at
    the developmental equilibrium** — the scalable core tool). Validated to
    agree, and that sensitivity × allelic-effect = the Fisher regression
    average effect (**Fig 1C**).
  - `build_G_sensitivity` — G built from our sensitivity (α = γ·s) equals their
    regression G and predicts the response end-to-end.
- `docs/DESIGN.md` — architecture + calibration ladder for the evo-devo layer.

## [0.1.1]

### Added
- Zenodo concept + version DOIs and PyPI badges.

## [0.1.0]

### Added
- Initial release: differentiable center-based tissue engine (Morse relaxation,
  cell division, growth, topology + gyration shape descriptors); `jax_md`
  neighbor-list scaling to 1–2M cells on GPU; gradient-based inverse design of
  tissue shape; and a genome→mechanics→form map differentiable to the genome.
