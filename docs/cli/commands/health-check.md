# `mmml health-check`

Validate MMML/PyCHARMM/JAX interface health.


## Usage

```bash
mmml health-check --help
```

## Options

```text
usage: mmml health-check [-h] [--only CHECK [CHECK ...]]
                         [--skip CHECK [CHECK ...]] [--live]
                         [--checkpoint CHECKPOINT]
                         [--live-residue LIVE_RESIDUE]
                         [--live-n-molecules LIVE_N_MOLECULES] [--require-gpu]
                         [--json] [--strict] [--prelaunch] [--tier2]

Validate MMML interface health before PyCHARMM / MLpot jobs: imports, JAX devices, libcharmm, MLpot symbols, Packmol, checkpoint, MPI.

options:
  -h, --help            show this help message and exit
  --only CHECK [CHECK ...]
                        Run subset of: core, jax, charmm, mlpot, packmol,
                        checkpoint, mpi, live
  --skip CHECK [CHECK ...]
                        Skip checks from the default set.
  --live                Run live MLpot registration + CHARMM energy (implies
                        charmm + checkpoint).
  --checkpoint CHECKPOINT
                        PhysNet checkpoint (default: MMML_CKPT).
  --live-residue LIVE_RESIDUE
                        Residue for --live smoke (default: DCM).
  --live-n-molecules LIVE_N_MOLECULES
                        Monomer count for --live smoke (default: 2).
  --require-gpu         Fail if JAX does not see a CUDA device.
  --json                Emit machine-readable JSON.
  --strict              Treat warnings as errors.
  --prelaunch           Relax MPI prelaunch warnings (serial health-check
                        before mpirun).
  --tier2               Also run spatial-MPI GPU checks inside the mpi
                        section.

Examples:
  # Fast preflight on a GPU node (no CHARMM energy eval):
  mmml health-check --require-gpu

  # Under the MPI launcher (recommended on MPI-linked libcharmm):
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --require-gpu --strict

  # Include live MLpot registration + CHARMM ENER on DCM:2:
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --live --checkpoint "$MMML_CKPT"

  # Smallest liquid-DCM density smoke (after modules + CHARMM build):
  mmml liquid-box --composition DCM:20 --target-density-g-cm3 1.326 \
    --profile standard -o boxes/dcm20 --charmm-sd-steps 50 --charmm-abnr-steps 50
  MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system \
    --from-psf boxes/dcm20/model.psf --from-crd boxes/dcm20/model.crd \
    --checkpoint "$MMML_CKPT" --md-stages mini --mini-nstep 20 --no-echeck --quiet

Checks: core, jax, charmm, mlpot, packmol, checkpoint, mpi (+ live with --live).
```


## Related docs

- [MLpot settings](../../mlpot-settings.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
