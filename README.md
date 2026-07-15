# jax-morpho

[![CI](https://github.com/JimGalasyn/jax-morpho/actions/workflows/ci.yml/badge.svg)](https://github.com/JimGalasyn/jax-morpho/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JimGalasyn/jax-morpho/branch/main/graph/badge.svg)](https://codecov.io/gh/JimGalasyn/jax-morpho)
[![CodeQL](https://github.com/JimGalasyn/jax-morpho/actions/workflows/codeql.yml/badge.svg)](https://github.com/JimGalasyn/jax-morpho/actions/workflows/codeql.yml)
[![release](https://img.shields.io/github/v/release/JimGalasyn/jax-morpho?include_prereleases&label=release)](https://github.com/JimGalasyn/jax-morpho/releases)
[![PyPI](https://img.shields.io/pypi/v/jax-morpho)](https://pypi.org/project/jax-morpho/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383756.svg)](https://doi.org/10.5281/zenodo.21383756)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/JimGalasyn/jax-morpho)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status: alpha (0.1.x).** The API is being designed in the open and may
> change without notice until 0.2.

**Differentiable, GPU-scale developmental morphogenesis in JAX.** Cells are
points with a Morse adhesion/repulsion potential; relaxation is autodiff
gradient descent, and growth is cell division. An evolvable genome grows a
tissue form — and gradients reach all the way back to the genome.

## Why

Mature tissue simulators (tyssue, Chaste, CompuCell3D) are CPU vertex- or
Potts-based and forward-only. The differentiable-morphogenesis work (Growing
NCA, VertAX, Deshpande et al. 2024) is autodiff-native but tops out at ~10²
cells. `jax-morpho` occupies the gap: a **differentiable, GPU-scale,
genome-driven** center-based engine.

- **Reproduces real-epithelium statistics.** Coupling proliferation to
  relaxation lands the polygon-side distribution on the Gibson-2006
  epithelium (~50% hexagons), tunable between random (Poisson–Voronoi) and
  crystalline packings by the relaxation-per-division ratio.
- **Scales.** The same Morse energy over a `jax_md` cell-list reaches **1–2
  million cells on a single GPU** (~32 ms/step on an RTX 4090), verified to
  the same energy minimum as the dense O(N²) path.
- **Differentiable to the genome.** Autodiff through the whole relaxation
  recovers the mechanical program — or the genome — that produces a target
  form (gradient-based inverse design / evo-devo), beating gradient-free
  search at equal budget.

## Install

```bash
pip install jax-morpho              # CPU jax
pip install 'jax[cuda12]'           # add GPU support (CUDA 12)
pip install 'jax-morpho[scale]'     # + jax_md neighbor lists for 10^5–10^6 cells
```

## Quickstart

```python
import jax_morpho as jm

# Grow a tissue by coupled division + relaxation; measure its packing.
pos, alive = jm.grow_relax(n_max=1400, n_start=20, target=1000, relax_steps=8)
sides = jm.interior_side_counts(pos, alive)
print("hexagon fraction:", jm.side_distribution(sides)[6])   # ~0.5, near Gibson

# Invert a target form into the genome that grows it.
# (see examples/demo_genome_mechanics.py)
```

Runnable demos:

```bash
python examples/demo_center_based.py        # emergent Gibson statistics + differentiability
python examples/demo_inverse_design.py      # gradient-based shape inverse design
python examples/demo_genome_mechanics.py    # genome -> form, gradients to the genome
python examples/demo_gpu_scaling.py         # 1–2M cells on GPU (needs [scale] + jax[cuda12])
```

## Modules

| Module | What |
| --- | --- |
| `jax_morpho.center_based` | Morse engine: relaxation, division, growth, topology + shape descriptors |
| `jax_morpho.scale` | `jax_md` cell-list relaxation for O(N) scaling (needs `[scale]`) |
| `jax_morpho.inverse_design` | gradient-based inverse design of tissue shape |
| `jax_morpho.genome` | genome → mechanics → form (differentiable to the genome) |
| `jax_morpho.stats` | epithelial topology references (Poisson–Voronoi, Gibson) + L1 distance |

## Development

```bash
pip install -e '.[test]'
pytest -q -n auto --cov=jax_morpho
```

## Citing

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI is minted on the first
tagged release.

## License

MIT — see [`LICENSE`](LICENSE).
