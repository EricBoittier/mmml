# PhysNet student on PET-MAD acetone labels

See `docs/metatomic.md` (PET-MAD teacher → PhysNet student).

```bash
uv run mmml pet-physnet-distill \
  --checkpoint /path/to/pet-mad-xs-v1.5.0.pt \
  --out-dir ./acetone_pet_distill --preset md

uv run mmml physnet-train --config ./acetone_pet_distill/physnet-train.yaml
```

The YAML written next to the NPZ warm-starts
`examples/ckpts_json/DESdimers_params.json` and trains on eV / eV/Å interaction
labels. `--distill` stays off: the NPZ is the teacher.
