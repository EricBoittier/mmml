# Metatomic Hub models (PET-MAD / UPET)

CHARMM-free smoke: export a TorchScript AtomisticModel from metatensor/UPET
checkpoints, then evaluate it through MMML’s ASE loader and CHARMM MLpot adapter.

CI does **not** download these files. No MD.

## Export

```bash
uv sync --extra metatomic
uv pip install 'metatrain[pet]' huggingface_hub

mkdir -p /tmp/mmml-metatomic-models
cd /tmp/mmml-metatomic-models
mtt export lab-cosmo/upet models/pet-mad-s-v1.0.2.ckpt -o pet-mad-s-v1.0.2.pt
mtt export lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt -o pet-mad-xs-v1.5.0.pt
mtt export lab-cosmo/upet models/pet-mols-s-v1.0.0.ckpt -o pet-mols-s-v1.0.0.pt
```

## Evaluate

```bash
MMML_METATOMIC_DEVICE=cpu \
uv run python tests/functionality/metatomic/eval_hub_models.py \
  --model-dir /tmp/mmml-metatomic-models
```

Pass: each `.pt` loads as `metatomic_ase.MetatomicCalculator`; water monomer and
dimer energies/forces are finite; dummy `calculate_charmm` kcal/mol matches
`E_eV * EV_TO_KCAL_MOL`.

See `docs/metatomic.md`.
