# Message-passing net inventory and ownership

This inventory prevents PhysNet-family and related e3x models from becoming
invisible or being mistaken for interchangeable APIs. It records ownership as of
2026-07-20 and the phased harmonization plan (shared kernels first; no
god-model rewrite).

## Classification

- **Canonical**: supported model path for training / hybrid MD / evaluation.
- **Supporting library**: shared numerical kernels used by canonical models.
- **Adapter**: ASE / checkpoint glue; must not own scientific behavior.
- **Deprecated**: retained for provenance; do not extend.

## Canonical models

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.model.PhysNet` | Canonical | EF without charge/spin conditioning. Prefer for plain E/F training. |
| `mmml.models.physnetjax.physnetjax.models.spooky_model.SpookyPhysNet` | Canonical | Production hybrid path: E/F + Q/S conditioning. Prefer for liquid/hybrid checkpoints. |
| `mmml.models.efield.model.EFieldPhysNet` | Canonical | External electric-field conditioned head. Keep train/eval CLI under `efield-*`. |
| `mmml.models.dcmnet.dcmnet.modules.DCMNetCharges` | Canonical | Distributed-charge (ESP) head. Do not fold into PhysNet energy heads. |

## Supporting kernels

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.mpnn_kernels` | Supporting library | Shared pair geometry, radial/spherical basis, electrostatic switches, and pair Coulomb assembly for PhysNet / SpookyPhysNet. Extend here instead of copy-pasting into model forks. |
| `mmml.models.physnetjax.physnetjax.models.zbl` | Supporting library | ZBL short-range repulsion already shared by PhysNet family. |
| `mmml.models.physnetjax.physnetjax.models.euclidean_fast_attention` | Supporting library | Optional EFA attention blocks for PhysNet family. |

## Adapters and loaders

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.calc.helper_mlp` | Adapter | Resolves architecture and builds ASE calculators from checkpoints. |
| `mmml.models.physnetjax.physnetjax.calc.ase_calculator` | Adapter | ASE wrapper around PhysNet-family apply functions. |
| `mmml.interfaces.calculators.checkpoint_loading` | Adapter | Portable JSON / Orbax load paths; must stay load-compatible across kernel extractions. |
| `mmml.models.spookynet_calc.SpookyNetCalculator` | Adapter | External / legacy SpookyNet ASE path; not the Flax SpookyPhysNet production model. |
| `mmml.models.dcmnet.dcmnet_ase` | Adapter | ASE calculator for DCMNetCharges. |

## Forks and deprecated paths

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.model_charge_spin.PhysNetChargeSpin` | Deprecated candidate | Third PhysNet fork with Q/S. Absorb into SpookyPhysNet (or thin alias) in Phase 2; do not add features here. Alias `EF_ChargeSpinConditioned` is already deprecated. |
| `mmml.models.EF` | Deprecated | Import shim → `mmml.models.efield`. |
| `spooky_model.EF` / `model.EF` aliases | Deprecated | Prefer `SpookyPhysNet` / `PhysNet` names. |

## What must stay separate

- **Checkpoint parameter trees** (Orbax / portable JSON): kernel extraction must not rename Flax modules.
- **Training loops**: `physnetjax/training`, `efield/training`, `dcmnet/training` until model cores share encode APIs.
- **Hybrid ML/MM assembly**: `mmml.models.hybrid_energy` and MLpot calculator remain calculator-neutral.
- **DCM / external-field heads**: Phase 3 only — shared encode, separate readout.

## Harmonization phases

1. **Shared kernels** (`mpnn_kernels`) used by PhysNet + SpookyPhysNet — current work.
2. **Collapse PhysNet family** into one flagged Flax module with thin aliases.
3. **Pluggable heads** for DCMNet and EFieldPhysNet on the shared encode path.

## Maintenance rule

Update this inventory in the same pull request that adds, removes, supersedes,
or changes the ownership of a message-passing model. A new supported energy
model requires an explicit explanation of why PhysNet / SpookyPhysNet cannot be
extended (or why a new head is required).
