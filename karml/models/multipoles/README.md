# E3x molecular multipoles

`E3xMultipoleModel` predicts the 16 real spherical components for molecular
multipoles through `l=3` from coordinates `R`, atomic numbers `Z`, total charge
`Q`, and spin/multiplicity `S`. Convert predictions on demand with
`irrep_blocks_to_traceless` to obtain symmetric traceless Cartesian tensors
with shapes `(3,)`, `(3, 3)`, and `(3, 3, 3)`.

Inputs use flattened atoms and E3x sparse pair indices:

```python
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(Z))
variables = model.init(key, R, Z, Q[None], S[None], dst_idx, src_idx)
prediction = model.apply(variables, R, Z, Q[None], S[None], dst_idx, src_idx)
loss = jnp.mean((prediction["multipoles"] - target_multipoles) ** 2)
```

For batches, concatenate atoms and edges, offset each molecule's edge indices,
and pass `batch_segments` plus the static `batch_size`. Coordinates and targets
must refer to the same molecular origin because multipoles of charged systems
are origin dependent.

The QCML cache utility is:

```console
python scripts/cache_qcml_multipoles_orbax.py \
  --data-dir . \
  --cache-dir orbax_cache/qcml_multipoles_traceless
```

It stores padded `R`, `Z`, `atom_mask`, `Q`, `S`, packed spherical multipoles,
individual irrep blocks, and traceless Cartesian tensors.

Train from that cache with:

```console
python scripts/train_qcml_multipoles.py \
  --cache orbax_cache/qcml_multipoles_traceless \
  --workdir artifacts/qcml_multipoles \
  --epochs 100 \
  --batch-size 32 \
  --bucket-width 8 \
  --max-atoms 64 \
  --max-structures 100000
```

The loss gives equal weight to each degree rather than allowing the seven
octupole components to dominate the scalar monopole. Checkpoints contain model
parameters, optimizer state, configuration, and train/validation metrics.
Atom-count buckets crop each batch to the bucket ceiling, reducing complete
graph cost from the cache-wide maximum. `--max-atoms` excludes larger
structures before splitting; omit it to retain all structures.
Manifest caches reserve the final `--test-shards` shards untouched and the
preceding `--validation-shards` shards for validation. The exact paths are
recorded in `data_split.json`.

Audit sampled shards before training:

```console
python scripts/audit_qcml_shards.py \
  --cache orbax_cache/qcml_multipoles_traceless \
  --kind multipoles \
  --max-shards 3 \
  --samples-per-shard 2000 \
  --output artifacts/qcml_multipoles/shard_audit.json
```

Generate metrics, per-structure CSV data, reference/prediction scatters, and
error distributions with:

```console
python scripts/analyze_qcml_multipoles.py \
  --cache orbax_cache/qcml_multipoles_traceless/0 \
  --checkpoint artifacts/qcml_multipoles/epoch-0100 \
  --output-dir artifacts/qcml_multipoles/report \
  --split validation
```

The report covers both the native spherical traceless components and their
Cartesian traceless tensors. QCML does not declare units for these fields in
its published schema, so labels default to `QCML native`. Use `--scale-lN` and
`--unit-lN` to apply known display conversions for each degree.
