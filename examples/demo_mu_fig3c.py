"""Phase-0a calibration demo: reproduce Milocco & Uller (2026 PNAS) Fig 3C.

Their toggle-switch developmental model, ported faithfully. Sweeps the theta2
minor-allele frequency and shows the key result: the development-derived G
predicts the one-generation response to selection (small angle to the observed
recombinant response) while the phenotypic covariance P misaligns at low allele
frequency.

Run:  .venv/bin/python examples/demo_mu_fig3c.py
"""
from __future__ import annotations

import numpy as np

from jax_morpho.evodevo.reference_mu import run_fig3c


def main():
    print("Milocco & Uller 2026 Fig 3C — reproduced (toggle-switch development)\n")
    res = run_fig3c(n_ind=5000, n_replays=30, dt=0.05, seed=0)
    print(f"  {'p2 (MAF)':>10} {'obs∠Gβ (reg)':>13} {'obs∠Gβ (sens)':>14} "
          f"{'obs∠Pβ':>9} {'G_reg vs G_sens':>16}")
    for r in res:
        print(f"  {r['p2']:>10.4f} {r['angle_G']:>12.1f}° {r['angle_G_sens']:>13.1f}° "
              f"{r['angle_P']:>8.1f}° {r['G_rel_frob']:>15.3f}")
    print("\n  (Phase 0c: 'sens' G is built from our autodiff/implicit sensitivity,")
    print("   α = γ·s; it matches the regression G and predicts equally well.)")
    lo = [r for r in res if r["p2"] < 0.02]
    hi = [r for r in res if r["p2"] >= 0.2]
    print(f"\n  low-freq  mean: G {np.mean([r['angle_G'] for r in lo]):.1f}°  "
          f"P {np.mean([r['angle_P'] for r in lo]):.1f}°   (P fails)")
    print(f"  high-freq mean: G {np.mean([r['angle_G'] for r in hi]):.1f}°  "
          f"P {np.mean([r['angle_P'] for r in hi]):.1f}°   (G ≈ P, proportional)")
    print("\n  → development-derived G predicts the response across allele")
    print("    frequencies; P is a good proxy only when G ≈ P (high MAF).")


if __name__ == "__main__":
    main()
