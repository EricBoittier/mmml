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
  --cache orbax_cache/qcml_mbd/0 \
  --workdir artifacts/qcml_mbd \
  --epochs 100 \
  --batch-size 16
```

The four loss weights are configurable with `--energy-weight`,
`--force-weight`, `--c6-weight`, and `--alpha-weight`. C6 and alpha use
`log1p` losses to handle their dynamic range.

`qdo_pairwise_dispersion` implements the damped pairwise C6/C8/C10 expression
as a separately testable baseline. It is not presented as equivalent to the
full coupled-oscillator libMBD target used by QCML.
