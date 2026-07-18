# Hybrid MM charge modes — example YAMLs

Example configs for the MM Coulomb charge modes (dimer-only for B/C; D is
liquid-compatible). See
[`docs/hybrid-mm-charges.md`](../../docs/hybrid-mm-charges.md).

| Mode | Formula | Train YAML | MD YAML |
|------|---------|------------|---------|
| A `fixed` | `q_MM = q_CGenFF` | [`train_fixed.yaml`](train_fixed.yaml) | [`md_fixed.yaml`](md_fixed.yaml) |
| A + PME (nvalchemiops) | fixed + `nvalchemiops_pme` Coulomb (LJ off) | [`train_fixed_nvalchemiops_pme.yaml`](train_fixed_nvalchemiops_pme.yaml) | [`md_fixed_nvalchemiops_pme.yaml`](md_fixed_nvalchemiops_pme.yaml) |
| A + PME (native ewald) | fixed + `ewald` Coulomb (LJ off, pure JAX, no CUDA) | [`train_fixed_ewald.yaml`](train_fixed_ewald.yaml) | [`md_fixed_ewald.yaml`](md_fixed_ewald.yaml) |
| B `latent` | `q_MM = neutralize(q_ML)` | [`train_latent.yaml`](train_latent.yaml) | [`md_latent.yaml`](md_latent.yaml) |
| C `fixed_plus_latent` | `q_CGenFF + neutralize(q_ML)` | [`train_fixed_plus_latent.yaml`](train_fixed_plus_latent.yaml) | [`md_fixed_plus_latent.yaml`](md_fixed_plus_latent.yaml) |
| D `latent_mean` (liquid) | `q_MM = tile(mean(neutralize(q_ML)))` | same checkpoint as B | `--mm-charge-mode latent_mean --mm-latent-charge-template <path>` (see below) |
| E `latent_dynamic` (liquid) | `q_MM = neutralize(weighted_mean_over_active_dimers(q_ML))` | same checkpoint as B | `--mm-charge-mode latent_dynamic` (no precompute) |

## Train

NPZ must carry CGenFF fields (`cgenff_type_idx`, `mol_id`, `cgenff_charge`,
`cgenff_master_*`). Modes B/C need `charges: true`.

YAML keys match CLI flags (`lr_solver`, `pme_box_length`, `pme_accuracy`,
`mm_include_lj`, …). CLI overrides the config when both are set.

```bash
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_nvalchemiops_pme.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_ewald.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_latent.yaml
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_plus_latent.yaml
```

## MD (dimer vacuum smoke)

Modes B/C are **dimer-only** (`composition: "DCM:2"`). Do not use liquid
boxes with `mm_charge_mode: latent` / `fixed_plus_latent` — use `latent_mean`
(Mode D, below) for liquids instead. Use `lr_solver: mic` (or omit) for B/C —
jax-pme is refused for B/C (Mode D has no such restriction).

```bash
# After training + orbax-to-json (or point checkpoint at an existing JSON)
mmml md-system --config examples/hybrid_mm_charges/md_fixed.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_fixed_nvalchemiops_pme.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_fixed_ewald.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_latent.yaml --run-all
mmml md-system --config examples/hybrid_mm_charges/md_fixed_plus_latent.yaml --run-all
```

Full-box PME MD flags (CLI equivalent of the nvalchemiops train config)::

```bash
mmml md-system --setup pbc_nvt --backend pycharmm \
  --composition DCM:20 --box-size 30 \
  --mm-nonbond-mode periodic_external \
  --lr-solver nvalchemiops_pme \
  --no-periodic-charmm-vdw \
  --mm-charge-mode fixed \
  --checkpoint /path/to/params.json
```

Same, with the pure-JAX native Ewald solver (no external PME library, no CUDA
requirement — drop-in wherever `nvalchemiops` isn't installed)::

```bash
mmml md-system --setup pbc_nvt --backend pycharmm \
  --composition DCM:20 --box-size 30 \
  --mm-nonbond-mode periodic_external \
  --lr-solver ewald \
  --no-periodic-charmm-vdw \
  --mm-charge-mode fixed \
  --checkpoint /path/to/params.json
```

### Mode D: latent-charge liquid MD with native Ewald

`latent`/`fixed_plus_latent` (B/C) cannot run on a 20-monomer liquid box —
their `q_ML` needs a live AB-dimer forward, undefined once there are more
than 2 monomers. Mode D uses a checkpoint trained with `--mm-charge-mode
latent --charges` (Mode B), but instead of a live forward, precomputes one
monomer's mean latent charge offline and tiles it across the box:

```bash
# 1) Precompute the template once, from the checkpoint + its training data
python scripts/compute_latent_monomer_charges.py \
  --checkpoint ./ckpts/mp2_nms/mp2nms_ewald \
  --data /path/to/mp2_nms15_clean_train.npz \
  --resid DCM \
  --out ./ckpts/mp2_nms/latent_charge_template_DCM.npz

# 2) Run the liquid box with it, same lr_solver as training (ewald)
mmml md-system --setup pbc_nvt --backend pycharmm \
  --composition DCM:20 --box-size 30 \
  --mm-nonbond-mode periodic_external \
  --lr-solver ewald \
  --no-periodic-charmm-vdw \
  --mm-charge-mode latent_mean \
  --mm-latent-charge-template ./ckpts/mp2_nms/latent_charge_template_DCM.npz \
  --checkpoint /path/to/params.json
```

See [Mode D in `docs/hybrid-mm-charges.md`](../../docs/hybrid-mm-charges.md#mode-d--latent_mean-md-only-liquid-compatible)
for the v1 limitation (homogeneous liquids only) and what it is not (a live,
geometry-dependent charge model).

### Mode E: live latent-charge liquid MD (no precompute)

`latent_dynamic` recomputes charges every step instead of freezing them:
each monomer's charge is a live, `ml_switch_scale`-weighted average of
`q_ML` over every currently active ML-dimer partner (no template to
generate first). Trade-off: no training-set averaging to smooth out
per-pair noise, and atoms with no active partner within `mm_switch_on` get
charge 0 (only appropriate where every monomer reliably has neighbors —
see [Mode E's v1 limitation](../../docs/hybrid-mm-charges.md#mode-e--latent_dynamic-md-only-liquid-compatible-live)):

```bash
mmml md-system --setup pbc_nvt --backend pycharmm \
  --composition DCM:20 --box-size 30 \
  --mm-nonbond-mode periodic_external \
  --lr-solver ewald \
  --no-periodic-charmm-vdw \
  --mm-charge-mode latent_dynamic \
  --checkpoint /path/to/params.json
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

Full-box native Ewald (train `lr_solver: ewald` ↔ MD `periodic_external`
many-to-many; pure JAX, no external package needed at all)::

```bash
python scripts/check_ewald_train_md_pme_parity.py \
  --data /path/to/energies_forces_dipoles_test.npz \
  --pme-box-length 30
```
