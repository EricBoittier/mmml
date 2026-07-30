# JAX-MD + CGenFF spoof smokes (DCM / ACO)

Small **infrastructure** jobs that exercise the hybrid **jaxmd** path with
`--jax-mm-spoof` (JAX CGenFF bonded clone in place of PhysNet). No trained ML
weights are required for the energy/force evaluation.

| Job | Composition | Setup |
|-----|-------------|-------|
| `dcm_vac_nve` | `DCM:4` | `free_nve` |
| `dcm_pbc_nve` | `DCM:4` | `pbc_nve` |
| `aco_vac_nve` | `ACO:4` | `free_nve` |
| `aco_pbc_nve` | `ACO:4` | `pbc_nve` |

## Clone / branch

Work only in the dedicated clone:

```bash
cd ~/mmml_cursor
git checkout cursor/jaxmd-cgenff-spoof-dcm-aco-b59b
```

## Run

```bash
cd ~/mmml_cursor
bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all.sh
# or one job:
bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all.sh dcm_vac_nve
```

Outputs land under `artifacts/jaxmd_cgenff_spoof_smoke/{job_id}/` with
`job.yaml` + `smoke_report.json`.

```bash
python workflows/jaxmd_cgenff_spoof_smoke/scripts/report.py
```
