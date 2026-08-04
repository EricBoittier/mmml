# Hybrid ML/MM potentials

How a hybrid potential is **assembled**: which region each atom belongs to, how
the ML and MM terms are combined, where charges come from, and how long-range
electrostatics are closed off.

This section is about the *shape* of the potential. Training the ML part is
[Training & sampling](training-sampling.md); running it is
[MD & campaigns](md-campaigns.md).

## The decisions, in order

1. **Regions and cutoffs** — which atoms are ML, which are MM, and what happens
   in the switching region.
2. **Charges** — fixed from the force field, latent from the model, or both.
3. **Repulsion/dispersion** — whether LJ parameters stay fixed or are trained.
4. **Long range** — how electrostatics beyond the cutoff are handled.

## What's here

**Start here** — [Cutoffs, regions & LR solvers](../hybrid-potential-regions.md)
covers decision 1 and frames the rest.

**How-to**

- [Hybrid MM charges](../hybrid-mm-charges.md) — fixed / latent / fixed+latent.
- [Trainable hybrid MM LJ scales](../hybrid-mm-lj-scales.md) — per-type σ / ε.
- [Hybrid ML/MM decomposition](../hybrid-mlmm-decomposition.md) — mapping each
  energy term back to its PyCHARMM counterpart. The page to read when a hybrid
  energy disagrees with CHARMM and you need to find *which* term.
- [Bonded intra and rigid-water stabilization](../hybrid-bonded-intra.md) —
  operational controls for ML models trained on rigid monomers: when to use
  `--ml-potential-mode bonded_intra`, when to use jax-md `--rigid-water`, and
  what to check before NPT density runs.
- [Interaction-prior constraints & trust map](../interaction-prior-constraints.md)
  — constraining the model where you have no data.
- [MLpot settings](../mlpot-settings.md) — the COM handoff switches and the
  runtime knobs.
- [DCMNet calculators](../dcmnet_calculators.md) — distributed charge models.

**Long range** — [Long-range solver tutorial](../long-range-solver-tutorial.md)
for the worked path, [Long-range electrostatics (ScaFaCoS)](../mlpot-long-range-electrostatics.md)
for the solver details.

## Verifying an assembly

The [calculator capability matrix](../calculator-capabilities.md) records which
combinations are actually supported — check it before designing around one.
[CHARMM CGenFF JAX clone](../cgenff-jax-clone.md) and
[jax-mm-spoof vs native CHARMM](../jax-mm-spoof-charmm-parity.md) are the
standing parity reports against PyCHARMM.
