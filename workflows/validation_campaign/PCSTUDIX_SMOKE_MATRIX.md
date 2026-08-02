# pc-studix calculator and backend smoke matrix

This is the maintained, capability-aware replacement for the older ad hoc
pc-studix sweeps. It exercises each calculator through the canonical dimer-scan
API and each propagation backend through the unified MD runner. Every case runs
in a fresh subprocess and writes `request.json`, `status.json`, `stdout.log`,
and `stderr.log`; the matrix writes `summary.json` with explicit `PASS`, `FAIL`,
and `BLOCKED` counts.

## Coverage

| Area | Cases |
|---|---|
| Learned calculators | PhysNet/joint (inferred and explicit charge/spin), SpookyNet, learned MBD, learned multipoles, EField |
| Quantum calculators | xTB/tblite, PySCF HF, DFTB3-D4 |
| MM and hybrid | JAX CGenFF spoof, PyCHARMM live MM, unified ML-intra + MM-nonbonded |
| Propagation | JAX-MD FIRE, NVE, NVT, NPT, rigid-body MC, PyCHARMM |
| Charge/long range | fixed, latent and fixed-plus-latent charge modes; MIC, Ewald/JAX-PME and rejection/lowering contracts |

The NPT case is retained and tagged `known_pcstudix_risk`: it has historically
failed while materializing XLA symbols on pc-studix. A real failure remains a
failure in the receipt; it is not silently removed from the matrix.

## Run on pc-studix

From the repository root on the login node:

```bash
sbatch workflows/validation_campaign/launch/pcstudix_smoke_matrix.slurm.sh
```

Run a subset by tag or exact case:

```bash
MMML_SMOKE_TAG=calculator \
  sbatch workflows/validation_campaign/launch/pcstudix_smoke_matrix.slurm.sh

MMML_SMOKE_CASE=jaxmd_nve \
  sbatch workflows/validation_campaign/launch/pcstudix_smoke_matrix.slurm.sh
```

List the matrix without running calculations:

```bash
.venv/bin/python -m mmml.validation.smoke_matrix \
  workflows/validation_campaign/pcstudix_smoke_matrix.yaml \
  --output-root /tmp/mmml-smoke --list
```

Results are stored under
`artifacts/validation_campaign/<run-id>/pcstudix/calculator_backend_matrix/`.
Runs never overwrite one another unless `MMML_SMOKE_RUN_ID` is explicitly
reused.

## Optional dependencies

Cases declare requirements and become `BLOCKED` when they are absent. Supply
these paths before submission when applicable:

```bash
export MMML_SMOKE_EFIELD_CHECKPOINT=/path/to/efield-params
export MMML_SMOKE_EFIELD_CONFIG=/path/to/efield-config.json
export MMML_SMOKE_DFTB_SLAKO_DIR=/path/to/3ob-3-1
```

xTB accepts either `xtb-python` or `tblite`; PySCF requires the `pyscf` module;
DFTB3-D4 also requires `dftb+` on `PATH`. By default, blocked optional cases are
reported but do not make the batch job fail. Set `MMML_SMOKE_STRICT_BLOCKED=1`
when the node is expected to provide the complete environment.

## Scientific contract

Calculator cases request a one-point rigid dimer calculation because the goal
is interface and force-contract coverage, not a production surface. Successful
dimer cases must produce the versioned manifest and both machine-readable
trajectory formats. Backend cases use one fixed seed and the same small TIP3
system. Larger scans and sampling campaigns belong in their dedicated
workflows; they should consume the same calculator and backend APIs rather than
forking this runner.
