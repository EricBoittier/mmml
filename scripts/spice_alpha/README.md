# SPICE-α → efield polarizability

No downloads. Point these scripts at the unzipped Zenodo 19205036 tree
(`SPICE-alpha.zip` already extracted).

Polarizability is trained on the **efield** model as an extra loss term:
`α = dμ/dEf` evaluated at **Ef = 0** (`mmml.models.efield.model_functions`).
Labels are converted from the release unit `e·Å²/V` to Bohr³.

```bash
# 1. Inner HDF5 + NPZ splits (256 frames = smoke)
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ./spice_mmml 256

# 2. Full DES370K monomers (max_frames=0)
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ./spice_mmml 0

# 3. Train (GPU)
scripts/spice_alpha/train_efield_polar.sh ./spice_mmml/splits_des_mono ./ckpts/spice_ef_polar 100
```

Equivalent one-liners without the shell wrappers are in `docs/spice-alpha.md`.
