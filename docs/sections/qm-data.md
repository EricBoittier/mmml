# QM & data

Producing **reference data** and turning it into training sets: DFT/MP2 energies,
forces, dipoles and ESP; rigid dimer and internal-coordinate scans (rigid or constrained-relax); and the
NPZ conversion, validation, and splitting steps in between.

How-to: [internal-coordinate scans](../ic-scan-design.md) (`mmml ic-scan`),
[dimer scans](../functionality/dimer_scans/README.md),
[PET-MAD interaction slices](../examples/pet-mad-etoh-pbc.md#2b-interaction-slices-and-surfaces-no-md)
(`mmml pet-interaction-pes`).

```bash
mmml pyscf-evaluate -i traj.npz -o out.npz --EF --esp   # label geometries
mmml validate out.npz                                   # check against schema
mmml fix-and-split --efd out.npz --output-dir ./splits  # unit fixes + splits
mmml ic-scan --config examples/ic_scan/acem_dihedrals_relaxed.yaml \
  --output artifacts/ic_scan/acem --overwrite
```

Converting from other sources:

```bash
mmml xml2npz molpro.xml -o data.npz     # Molpro XML -> NPZ
mmml npz2traj data.npz -o traj.traj     # NPZ -> ASE trajectory
# already eV / eV/Å / e·Å (SPICE-α, distill): do not reconvert
mmml fix-and-split --efd data.npz -o ./splits --preserve-units
```

What trainers accept (keys, units, total vs interaction, gradient sign):
[Training NPZ contract](../training-npz-contract.md). External DFT dump:
[SPICE-α → PhysNet](../spice-alpha.md).

## What's here

**How-to**

- [QC cross-check](../qc-cross-check.md) — independently verifying a QM pipeline
  before you train on its output.
- [Training NPZ contract](../training-npz-contract.md) — ingest vs train
  units, energy meaning, padding, hybrid extras.
- [SPICE-α (Zenodo 19205036)](../spice-alpha.md) — `mmml.data.spice_alpha`
  (flip gradients), then `fix-and-split --preserve-units`.
- [Preparing hybrid ML/MM datasets](../hybrid-mm-dataset-preparation.md) —
  assigning CGenFF types and charges to a dimer NPZ.
- [Dimer scans (DCM / ACO)](../functionality/dimer_scans/README.md) and
  [Orientation scan plots](../functionality/orient_scan_plots.md) — generating
  and reading rigid scan surfaces.
- [Internal-coordinate scans](../ic-scan-design.md) — bond / angle / dihedral
  how-to (`mmml ic-scan`). Default is rigid; `geometry_mode: constrained-relax`
  for a methyl rotor.

**Commands** — the `pyscf-*` family, `dimer-scan`, `pet-interaction-pes`, `ic-scan`, `mode-check`,
`fix-and-split`, `validate`, plus the ORCA external interface.

## A note on trust

Reference data is the one input nothing downstream can correct for. The
[scientific claim evidence policy](../evidence-policy.md) covers what has to be
recorded for a number to be quotable, and `mmml cross-check` /
`mmml compare-npz` exist to make disagreement visible early.

## Where this leads

Split NPZs feed [Training & sampling](training-sampling.md). Scans are also the
validation target for [Hybrid ML/MM potentials](hybrid-potentials.md).
