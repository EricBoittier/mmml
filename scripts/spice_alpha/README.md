# SPICE-α → efield polarizability

No downloads. Point these scripts at the unzipped Zenodo 19205036 tree
(`SPICE-alpha.zip` already extracted).

Polarizability is trained on the **efield** model as an extra loss term:
`α = dμ/dEf` evaluated at **Ef = 0** (`mmml.models.efield.model_functions`).
Labels are converted from the release unit `e·Å²/V` to Bohr³.

`train_efield_polar.sh` **forces `JAX_ENABLE_X64=0`**. SciCORE's
`scripts/scicore_env.sh` defaults x64 on and will crash `EFieldPhysNet.init`.
Do not source that prolog for this job. No CHARMM.

Checkpoints now write `run_meta.json` + `history.jsonl` every epoch, a rich
`best-valid-<uuid>.json` (polar/energy/force MAE, not just loss), and Orbax
weights under `orbax/` (JSON `params-best-*.json` still written; `--save-format
orbax|json|both`). A missing `params-best.json` while the job is running is
**not** proof that validation never improved — older trainers only created
that symlink on exit. Audit from the login node:

```bash
python scripts/spice_alpha/audit_efield_job.py \
  --ckpt $HOME/mmml/ckpts/spice_ef_polar_big \
  --log artifacts/spice_ef_polar/slurm-22826285.out \
  --job 22826285
```

```bash
# 1. Inner HDF5 + NPZ splits (256 frames = smoke; skips the 1.3 GB dimers)
# Published DES370K HDF5 may have an empty file-level units_map; convert
# still reads the first molecule group, then (if that is also empty)
# assumes SPICE-α README Å/eV units. Explicit Hartree/Bohr on that group
# is refused unless --allow-atomic-units.
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ./spice_mmml 256
# dimers later: INCLUDE_DIMERS=1 scripts/spice_alpha/prepare_efield_dataset.sh ... 0
python scripts/spice_alpha/check_efield_npz.py \
  ./spice_mmml/splits_des_mono/energies_forces_dipoles_{train,valid}.npz

# 2. Full DES370K monomers (new tree; skip dimers). Prefer a CPU sbatch.
SKIP_DIMERS=1 scripts/spice_alpha/prepare_efield_dataset.sh \
  ~/data/spicealpha ~/data/spicealpha/mmml_efield_full 0
# or: sbatch scripts/spice_alpha/prepare_efield_dataset.sbatch

# 3. GPU smoke, then full. MODE=full defaults POLAR_WEIGHT=100 (1 left
# valid polar mae frozen on the 256-frame extract).
sbatch scripts/spice_alpha/train_efield_polar.sbatch
# 256-frame extract: BATCH_SIZE=8 (valid n=13; B=64 → 0 valid batches)
sbatch --time=06:00:00 --qos=rtx4090-6hours \
  --export=ALL,MODE=full,EPOCHS=100,BATCH_SIZE=8 \
  scripts/spice_alpha/train_efield_polar.sbatch
# all monomers. B=16 F=64 and B=64 F=32 both died in polar-JVP XLA autotune.
sbatch --partition=rtx4090 --qos=rtx4090-6hours --time=06:00:00 \
  --export=ALL,MODE=big,EPOCHS=100,BATCH_SIZE=4,SPLITS=$HOME/data/spicealpha/mmml_efield_full/splits_des_mono,CKPT=$HOME/mmml/ckpts/spice_ef_polar_big \
  scripts/spice_alpha/train_efield_polar.sbatch
# after a TIMEOUT: new CKPT dir, same weights
# RESTART=$HOME/mmml/ckpts/spice_ef_polar_big/params-best-441a9161-9eca-4563-9957-04c9d2ec5a34.json
```

Interactive GPU (after an allocation):

```bash
BATCH_SIZE=8 FEATURES=16 MAX_DEGREE=1 \
  scripts/spice_alpha/train_efield_polar.sh ./spice_mmml/splits_des_mono ./ckpts/spice_ef_polar 2
```

Equivalent one-liners: `docs/spice-alpha.md`.
