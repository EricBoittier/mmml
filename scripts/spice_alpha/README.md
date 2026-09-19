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
# Published DES370K HDF5 may have an empty units_map; convert treats that
# as unknown and still writes Å/eV labels from the SPICE-α README.
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ./spice_mmml 256
# dimers later: INCLUDE_DIMERS=1 scripts/spice_alpha/prepare_efield_dataset.sh ... 0
python scripts/spice_alpha/check_efield_npz.py \
  ./spice_mmml/splits_des_mono/energies_forces_dipoles_{train,valid}.npz

# 2. Full DES370K monomers (max_frames=0)
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ./spice_mmml 0

# 3. GPU smoke, then full
sbatch scripts/spice_alpha/train_efield_polar.sbatch
# 256-frame extract: BATCH_SIZE=8 (valid n=13; B=64 → 0 valid batches)
sbatch --time=06:00:00 --qos=rtx4090-6hours \
  --export=ALL,MODE=full,EPOCHS=100,BATCH_SIZE=8 \
  scripts/spice_alpha/train_efield_polar.sbatch
```

Interactive GPU (after an allocation):

```bash
BATCH_SIZE=8 FEATURES=16 MAX_DEGREE=1 \
  scripts/spice_alpha/train_efield_polar.sh ./spice_mmml/splits_des_mono ./ckpts/spice_ef_polar 2
```

Equivalent one-liners: `docs/spice-alpha.md`.
