# QCML many-body dispersion

This module trains an E3x surrogate for the QCML/libMBD correction. It predicts
the molecular dispersion energy, conservative forces, and positive per-atom
`C6` coefficients and polarizabilities. QCML uses atomic units: positions are
in `a0`, energy in `Eh`, and polarizabilities in `a0³`.

Prepare the joined cache:

```console
python scripts/cache_qcml_mbd_orbax.py \
  --data-dir . \
  --cache-dir orbax_cache/qcml_mbd
```

Train energy, force, C6, and polarizability targets:

```console
python scripts/train_qcml_mbd.py \
  --cache orbax_cache/qcml_mbd \
  --workdir artifacts/qcml_mbd \
  --epochs 100 \
  --batch-size 16 \
  --bucket-width 8 \
  --max-atoms 64 \
  --max-structures 100000
```

The four loss weights are configurable with `--energy-weight`,
`--force-weight`, `--c6-weight`, and `--alpha-weight`. C6 and alpha use
`log1p` losses to handle their dynamic range.
Atom-count bucketing crops positions and every atomic target to the bucket
ceiling. `--max-atoms` filters oversized structures before the random split.
For manifest caches, the final `--test-shards` are never restored during
training. Validation uses the preceding `--validation-shards`, and the full
partition is saved as `data_split.json`.

Audit geometry scales, positivity, and force/torque conservation with:

```console
python scripts/audit_qcml_shards.py \
  --cache orbax_cache/qcml_mbd \
  --kind mbd \
  --max-shards 3 \
  --samples-per-shard 2000 \
  --output artifacts/qcml_mbd/shard_audit.json
```

`qdo_pairwise_dispersion` implements the damped pairwise C6/C8/C10 expression
as a separately testable baseline. It is not presented as equivalent to the
full coupled-oscillator libMBD target used by QCML.
