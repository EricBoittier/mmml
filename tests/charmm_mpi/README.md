# CHARMM MPI test suite

Fast unit tests (mocked) run in the **build** CI job. Live PyCHARMM tests run in the **charmm** job under `mpirun`.

## Layout

| File | CI job | Needs CHARMM |
|------|--------|--------------|
| `test_mpi_bootstrap.py` | build | no |
| `test_rank0_trajectory_io.py` | build | no |
| `test_tier3_domdec.py` | build | no |
| `test_spatial_mpi_yaml_campaign.py` | build | no |
| `test_mpi_live_energy.py` | charmm | yes |

## Related unit tests (outside this directory)

- `tests/unit/test_charmm_mpi.py` — `charmm_mpi` module
- `tests/unit/test_mpi_check.py` — `mmml mpi-check`
- `tests/unit/test_mlpot_spatial_mpi_integration.py` — Tier 2 callback path

## Local commands

```bash
# Fast suite (no CHARMM)
pytest tests/charmm_mpi/ -m "not pycharmm" -q

# Full suite on CHARMM node
mmml mpi-check --tier2 --tier3
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh pytest tests/charmm_mpi/ -q
```

`mmml mpi-check --tier3` is informational: it can exit 0 while reporting Tier 3 production as blocked. Use `mmml mpi-check --tier3 --strict` when a script should fail until PyCHARMM exposes local/ghost atom metadata.

See [`docs/pycharmm-mpi.md`](../../docs/pycharmm-mpi.md).
