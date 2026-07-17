# Hybrid MM charge modes — example YAMLs

Example configs for the three MM Coulomb charge modes (dimer-only for B/C).
See [`docs/hybrid-mm-charges.md`](../../docs/hybrid-mm-charges.md).

| Mode | Formula | Train YAML | MD YAML |
|------|---------|------------|---------|
| A `fixed` | `q_MM = q_CGenFF` | [`train_fixed.yaml`](train_fixed.yaml) | [`md_fixed.yaml`](md_fixed.yaml) |
| A + PME | fixed + `nvalchemiops_pme` Coulomb (LJ off) | [`train_fixed_nvalchemiops_pme.yaml`](train_fixed_nvalchemiops_pme.yaml) | — |
| B `latent` | `q_MM = neutralize(q_ML)` | [`train_latent.yaml`](train_latent.yaml) | [`md_latent.yaml`](md_latent.yaml) |
| C `fixed_plus_latent` | `q_CGenFF + neutralize(q_ML)` | [`train_fixed_plus_latent.yaml`](train_fixed_plus_latent.yaml) | [`md_fixed_plus_latent.yaml`](md_fixed_plus_latent.yaml) |

## Train

NPZ must carry CGenFF fields (`cgenff_type_idx`, `mol_id`, `cgenff_charge`,
`cgenff_master_*`). Modes B/C need `charges: true`.

YAML keys match CLI flags (`lr_solver`, `pme_box_length`, `pme_accuracy`,
`mm_include_lj`, …). CLI overrides the config when both are set.

```bash
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_nvalchemiops_pme.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_latent.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_plus_latent.yaml
```

## MD (dimer vacuum smoke)

Modes B/C are **dimer-only** (`composition: "DCM:2"`). Do not use liquid
boxes with `mm_charge_mode: latent` / `fixed_plus_latent` yet. Use
`lr_solver: mic` (or omit) — jax-pme is refused for B/C.

```bash
# After training + orbax-to-json (or point checkpoint at an existing JSON)
mmml md-system --config examples/hybrid_mm_charges/md_fixed.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_latent.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_fixed_plus_latent.yaml --run-all
```

Edit `defaults.checkpoint` to your Mode A/B/C checkpoint. Train and MD modes
must match (`hybrid_mm.json` sidecar records the training mode).

## Parity gate (cluster / live CHARMM)

MIC LJ+Coulomb hybrid (default train/MD)::

```bash
python scripts/check_hybrid_train_md_parity.py \
  --checkpoint /path/to/ckpts/hybrid/... \
  --data /path/to/energies_forces_dipoles_test.npz \
  --mm-charge-mode fixed   # or latent / fixed_plus_latent
```

Full-box nvalchemiops PME (train `lr_solver: nvalchemiops_pme` ↔ MD
`periodic_external` many-to-many; no CHARMM needed for the PME kernel / `e_mm`
layers)::

```bash
python scripts/check_nvalchemiops_train_md_pme_parity.py \
  --data /path/to/energies_forces_dipoles_test.npz \
  --pme-box-length 30
```
