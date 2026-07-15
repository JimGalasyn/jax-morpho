"""Genome -> mechanics -> form, and gradient-based evo-devo.

Part 1 (forward): amino-acid genomes produce distinct tissue forms through
the center-based engine, using the real aa_substrate_token chemistry.

Part 2 (inverse): starting from a target FORM, gradient descent recovers
the GENOME that produces it -- gradients flow genome -> mechanics ->
relaxation -> form -- and beats random search.

Run:  .venv/bin/python examples/demo_genome_mechanics.py
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jax_morpho.genome import (
    axial_sigmas, develop, genome_morphology,
    hydro_scalar, nearest_amino_acids, sequence_to_genome,
)
from jax_morpho.inverse_design import adam, relax_in_field

N = 350


def disk(seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = np.sqrt(rng.uniform(0, 1, N)) * np.sqrt(N) * 0.55
    return jnp.asarray(np.column_stack(
        [rad * np.cos(ang), rad * np.sin(ang)]).astype(np.float32))


def main():
    pos0 = disk(0)

    print("=== 1. FORWARD: amino-acid genome -> tissue form ===")
    print("   (F hydrophobic +1, G neutral 0, K hydrophilic -1)")
    print(f"   {'genome':>10} {'h_x,h_y':>9} {'sigma_x':>8} {'sigma_y':>8} "
          f"{'aspect':>7}  shape")
    for ax, ay in [("F", "K"), ("K", "F"), ("G", "G"), ("F", "F"), ("F", "G")]:
        g = sequence_to_genome(ax, ay)
        pos = relax_in_field(pos0, develop(g), 220)
        sx, sy = (float(v) for v in axial_sigmas(pos))
        _, aspect = (float(v) for v in genome_morphology(g, pos0, 220))
        shape = ("elongated-y" if sy > sx * 1.1 else
                 "elongated-x" if sx > sy * 1.1 else "round")
        print(f"   ({ax},{ay})      {hydro_scalar(ax):+.0f},{hydro_scalar(ay):+.0f}"
              f"      {sx:8.2f} {sy:8.2f} {aspect:7.2f}  {shape}")

    print("\n=== 2. INVERSE (evo-devo): target form -> the genome for it ===")
    # Orientation-aware target (per-axis extent) so the correct genome is
    # identifiable (size+aspect alone is mirror-degenerate).
    g_true = sequence_to_genome("F", "K")
    sx_t, sy_t = (float(v) for v in
                  axial_sigmas(relax_in_field(pos0, develop(g_true), 220)))
    print(f"   target form (from ground-truth genome F,K): "
          f"sigma_x={sx_t:.2f} sigma_y={sy_t:.2f}")

    def loss(g):
        sx, sy = axial_sigmas(relax_in_field(pos0, develop(g), 220))
        return ((sx - sx_t) / sx_t) ** 2 + ((sy - sy_t) / sy_t) ** 2

    genome, history = adam(loss, jnp.array([0.0, 0.0], dtype=jnp.float32),
                           lr=0.08, n_steps=41)
    aa_x, aa_y = nearest_amino_acids(genome)
    print(f"   gradient descent: loss {history[0]:.4f} -> {history[-1]:.5f} "
          f"({history[0]/max(history[-1],1e-9):.0f}x)")
    print(f"   recovered genome h=({float(genome[0]):+.2f},{float(genome[1]):+.2f})"
          f" -> nearest amino acids ({aa_x},{aa_y})   [ground truth: F,K]")

    rs = np.random.default_rng(3)
    best = np.inf
    for _ in range(41):
        g = jnp.asarray(rs.uniform(-1.2, 1.2, 2).astype(np.float32))
        best = min(best, float(loss(g)))
    print(f"   random search (41 evals): best loss {best:.5f}")
    print(f"\n   -> gradients reach the genome: a target FORM is turned into "
          f"the GENOME that grows it.")


if __name__ == "__main__":
    main()
