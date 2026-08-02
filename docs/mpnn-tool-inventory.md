# Message-passing net inventory and ownership

This inventory prevents PhysNet-family and related e3x models from becoming
invisible or being mistaken for interchangeable APIs. It records ownership as of
2026-07-20 and the phased harmonization plan (shared kernels + family facade;
no god-model Flax merge that would break checkpoints).

## Classification

- **Canonical**: supported model path for training / hybrid MD / evaluation.
- **Supporting library**: shared numerical kernels / family facade used by canonical models.
- **Adapter**: ASE / checkpoint glue; must not own scientific behavior.
- **Deprecated**: retained for provenance; do not extend.

## Canonical models

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.model.PhysNet` | Canonical | EF without charge/spin conditioning. Prefer for plain E/F training. Inherits `PhysNetFamilyMixin`. |
| `mmml.models.physnetjax.physnetjax.models.spooky_model.SpookyPhysNet` | Canonical | Production hybrid path: E/F + Q/S conditioning. Prefer for liquid/hybrid checkpoints. Inherits `PhysNetFamilyMixin`. |
| `mmml.models.efield.model.EFieldPhysNet` | Canonical | External electric-field conditioned head. Uses shared `encode_geometry_and_basis` (reciprocal Bernstein); field coupling stays local. |
| `mmml.models.dcmnet.dcmnet.modules.DCMNetCharges` | Canonical | Distributed-charge (ESP) head. Uses shared geometry/basis encode; DCM readout stays local. |

## Supporting kernels and family facade

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.mpnn_kernels` | Supporting library | Pair geometry, radial/spherical basis (`radial_fn` selectable), `encode_geometry_and_basis`, electrostatic switches, pair Coulomb, molecular dipoles. Extend here instead of copy-pasting. |
| `mmml.models.physnetjax.physnetjax.models.physnet_family` | Supporting library | `PhysNetFamilyConfig`, `resolve_physnet_class`, `PhysNetFamilyMixin`. Selects PhysNet vs SpookyPhysNet without merging Flax parameter trees. |
| `mmml.models.physnetjax.physnetjax.models.zbl` | Supporting library | ZBL short-range repulsion already shared by PhysNet family. |
| `mmml.models.physnetjax.physnetjax.models.euclidean_fast_attention` | Supporting library | Optional EFA attention blocks for PhysNet family. |

## Adapters and loaders

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.calc.helper_mlp` | Adapter | Resolves architecture and builds ASE calculators from checkpoints. Spooky detection uses `type(model).__module__` containing `spooky_model` — keep SpookyPhysNet defined in that module. |
| `mmml.models.physnetjax.physnetjax.calc.ase_calculator` | Adapter | ASE wrapper around PhysNet-family apply functions. |
| `mmml.interfaces.calculators.checkpoint_loading` | Adapter | Portable JSON / Orbax load paths; must stay load-compatible across kernel extractions. |
| `mmml.models.spookynet_calc.SpookyNetCalculator` | Adapter | External / legacy SpookyNet ASE path; not the Flax SpookyPhysNet production model. |
| `mmml.models.dcmnet.dcmnet_ase` | Adapter | ASE calculator for DCMNetCharges. |

## Forks and deprecated paths

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.models.physnetjax.physnetjax.models.model_charge_spin.PhysNetChargeSpin` | Deprecated | Third PhysNet fork with discrete Q/S embeddings. Emits `DeprecationWarning` on `setup`. Use SpookyPhysNet for new Q/S work. Alias `EF_ChargeSpinConditioned` remains. |
| `mmml.models.EF` | Deprecated | Import shim → `mmml.models.efield`. |
| `spooky_model.EF` / `model.EF` aliases | Deprecated | Prefer `SpookyPhysNet` / `PhysNet` names. |

## What must stay separate

- **Checkpoint parameter trees** (Orbax / portable JSON): do not rename Flax modules or move `Embed` / `MessagePass` into shared helpers.
- **Training loops**: `physnetjax/training`, `efield/training`, `dcmnet/training` until heads share a richer encode API.
- **Hybrid ML/MM assembly**: `mmml.models.hybrid_energy` and MLpot calculator remain calculator-neutral.
- **Head-specific readout**: DCM distributed sites, EF field coupling, Spooky CGenFF vdW — stay in their modules.

## Harmonization status

1. **Shared kernels** (`mpnn_kernels`) — done (Phase 1).
2. **Family facade + mixin** (`physnet_family`) — done (Phase 2). Full single-Flax-module merge deferred (would break checkpoints).
3. **Shared geometry/basis encode for DCM / EF** — done (Phase 3, non-parametric only). Parametric MP loops remain per-head.

## Maintenance rule

Update this inventory in the same pull request that adds, removes, supersedes,
or changes the ownership of a message-passing model. A new supported energy
model requires an explicit explanation of why PhysNet / SpookyPhysNet cannot be
extended (or why a new head is required).
