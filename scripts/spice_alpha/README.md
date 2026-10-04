# SPICE-α → efield polarizability

No downloads. Point these scripts at the unzipped Zenodo 19205036 tree
(`SPICE-alpha.zip` already extracted).

Polarizability is trained on the **efield** model as an extra loss term:
`α = dμ/dEf` evaluated at **Ef = 0** (`mmml.models.efield.model_functions`).
Labels are converted from the release unit `e·Å²/V` to Bohr³.

`train_efield_polar.sh` **forces `JAX_ENABLE_X64=0`**. SciCORE's
`scripts/scicore_env.sh` defaults x64 on and will crash `EFieldPhysNet.init`.
Do not source that prolog for this job. No CHARMM.

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
```

Interactive GPU (after an allocation):

```bash
BATCH_SIZE=8 FEATURES=16 MAX_DEGREE=1 \
  scripts/spice_alpha/train_efield_polar.sh ./spice_mmml/splits_des_mono ./ckpts/spice_ef_polar 2
```

Equivalent one-liners: `docs/spice-alpha.md`.

## Full SPICE-α (packed batches, H200)

See `docs/spice-alpha.md` § "Full dataset". `convert_full_ragged.sbatch`
writes ragged shards; `train_full_packed.sbatch OUT_DIR [opts]` trains with
packed batches and a molecule split. `eval_polar_dipole.py` and
`relative_errors.py` report errors in MACE-MDP units / relative to the label
spread. Note the SiLU charge floor (docs § "Charge head") — prefer
`--charge-activation linear` for dipoles.

Analysing packed checkpoints (training overwrites `params-best.json` each
epoch, so copy it to `CKPT/analysis/epNN/params.json` first):

- `eval_full_packed.{py,sbatch} CKPT PARAMS OUT` — whole test/valid splits,
  RMSE/MAE per subset, per-frame predictions, charge statistics by element.
- `eval_detail_packed.{py,sbatch} CKPT PARAMS OUT` — per-atom charges,
  atomic dipoles and forces on a test subsample, plus captured layer
  features for one example frame per subset.
- `chiral_survey.py H5_DIR RAGGED_DIR SPLIT OUT` — frame → HDF5 group map and
  RDKit stereocentres per molecule.
- `symmetry_tests.{py,sbatch} CKPT PARAMS FRAMES_JSON OUT` — rotation,
  inversion, mirror, translation and permutation tests at several fields,
  pseudoscalar parity per layer, conservation and μ vs −∂U/∂E
  (docs § "Parity").

The `.sbatch` wrappers default to the `l40s` partition.
