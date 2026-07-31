# QM & data

Producing **reference data** and turning it into training sets: DFT/MP2 energies,
forces, dipoles and ESP; rigid dimer and internal-coordinate scans; and the
NPZ conversion, validation, and splitting steps in between.

## Happy path

```bash
mmml pyscf-evaluate -i traj.npz -o out.npz --EF --esp   # label geometries
mmml validate out.npz                                   # check against schema
mmml fix-and-split --efd out.npz --output-dir ./splits  # unit fixes + splits
```

Converting from other sources:

```bash
mmml xml2npz molpro.xml -o data.npz     # Molpro XML -> NPZ
mmml npz2traj data.npz -o traj.traj     # NPZ -> ASE trajectory
```

## What's here

**How-to**

- [QC cross-check](../qc-cross-check.md) — independently verifying a QM pipeline
  before you train on its output.
- [Preparing hybrid ML/MM datasets](../hybrid-mm-dataset-preparation.md) —
  assigning CGenFF types and charges to a dimer NPZ.
- [Dimer scans (DCM / ACO)](../functionality/dimer_scans/README.md) and
  [Orientation scan plots](../functionality/orient_scan_plots.md) — generating
  and reading rigid scan surfaces.

**Commands** — the `pyscf-*` family, `dimer-scan`, `ic-scan`, `mode-check`,
`fix-and-split`, `validate`, plus the ORCA external interface.

## A note on trust

Reference data is the one input nothing downstream can correct for. The
[scientific claim evidence policy](../evidence-policy.md) covers what has to be
recorded for a number to be quotable, and `mmml cross-check` /
`mmml compare-npz` exist to make disagreement visible early.

## Where this leads

Split NPZs feed [Training & sampling](training-sampling.md). Scans are also the
validation target for [Hybrid ML/MM potentials](hybrid-potentials.md).
