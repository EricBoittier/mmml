# Constraining the neural interaction toward the MM prior

## The problem

The SpookyNet model carries a physical MM prior (CGenFF Lennard-Jones + point-charge
electrostatics) *inside* the total-energy prediction. Nothing constrains how large the
neural contribution may be on top of that prior, and measured on the 15-pair dimer scan
(checkpoint `from_step2000_lowlr/step-00005000`, binding window −10 < E_ref < +5):

| coverage | mean \|neural interaction\| |
|---|---|
| pairs with ≥300 training structures | 0.93 kcal/mol |
| pairs with <15 training structures | 3.85 kcal/mol |

**The neural term is 4.1× *larger* where there is no data to justify it** — backwards
from what a well-behaved residual should do. On ACE–BENZ (2 training structures) the
neural interaction is 7.76 kcal/mol sitting on an LJ prior of 0.007, and the resulting
PES is +11.9 kcal/mol RMSE from PBE0-D3BJ where plain CGenFF is 0.52. See
`plots/neural_vs_coverage.png`.

A separate finding compounds it: the learned LJ scaling had driven `global_vdw_scale`
to 0.14 with per-element scales of 0.10 (C) / 0.24 (H), i.e. the prior itself was scaled
to ~1% of its physical value. So the model was both erasing the prior and replacing it
with an unconstrained neural surface.

## Two mechanisms, both added as flags

### `--fixed-cgenff-vdw`

Pins the CGenFF LJ term at its published parameters by removing all three learned
scaling paths (per-atom predicted, global, per-element). The prior becomes a fixed
physical baseline the network can only add to, never scale away. Commit `ef5caf76`.

### `--neural-interaction-l2 <λ>`

Ridge shrinkage of the neural **interaction** energy toward zero:

    loss += λ · mean[ (E_neural(AB) − E_neural(A) − E_neural(B))² ]

The target stays the total energy — this only regularises how loudly the neural term
speaks on top of the prior. The zero-evidence limit becomes CGenFF instead of an
invented surface. Commit `85555e04`.

**The monomer reference** `E_neural(A) + E_neural(B)` is computed exactly by one
forward-only pass with the inter-monomer message-passing edges removed. This required a
real edge gate: `batch_mask` only scales the pairwise prior terms, not the neural graph,
so an `edge_mask` was added to `SpookyPhysNet` that zeroes the message-passing basis for
masked edges (default `None` = no-op). Verified: the masked neural interaction is nonzero
at contact and decays to exactly 0 beyond the 6 Å cutoff (`tests/unit/test_neural_interaction_l2.py`).

## Experiment (in progress)

Restarting from `step-00005000` on the v1 cache (571,708 structures), LR 1e-5 cosine to
25k steps, all with `--fixed-cgenff-vdw`:

| run | λ | node |
|---|---|---|
| `intl2_20260715/l2_0p01` | 0.01 | gpu08:0 |
| `intl2_20260715/l2_0p1` | 0.1 | gpu08:1 |
| `intl2_20260715/l2_1p0` | 1.0 | gpu01:1 |

Plus two `--fixed-cgenff-vdw`-only controls (`fixedprior_20260715/`, λ = 0).

**Success criterion:** sparse pairs (ACE–BENZ 11.9, DCM–DCM 6.4) collapse toward the
CGenFF baseline (0.52, 0.94) while well-covered pairs (n≥300: 0.54) hold. If a global λ
buys sparse-pair robustness only by wrecking the rich pairs, that failure is the evidence
that the shrinkage strength must itself be evidence-dependent (per-chemistry / hierarchical λ).

Baselines to beat, mean binding-window RMSE over the 14 pairs with training data:

    CGenFF (force field)                 1.07 kcal/mol
    best ML checkpoint (step 5000)       3.63
    150k annealed, no constraint         4.80   (training longer made it worse)

Score any checkpoint with:

    python local_validation/rescore.py \
        "CGenFF=surfaces/cgenff_direct.csv" \
        "L2 0.1=surfaces/<scan>.csv"
