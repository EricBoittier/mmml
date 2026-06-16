# DCM:5 cross-backend MD benchmark report

- Composition: `DCM:5`
- Target length: **2.0 ps** (8000 steps @ 0.25 fs)
- Jobs: 15 (pass=2, warn=0, fail/missing=13)

| job | backend | setup | PBC | integrator | status | nsteps | temp_mean_K | notes |
|-----|---------|-------|-----|------------|--------|--------|-------------|-------|
| ase_pbc_nve | ase | pbc_nve | True | nve | missing |  |  | suite_summary.json not found |
| ase_pbc_nvt_langevin | ase | pbc_nvt | True | nvt_langevin | missing |  |  | suite_summary.json not found |
| ase_pbc_nvt_nhc | ase | pbc_nvt | True | nvt_nhc | missing |  |  | suite_summary.json not found |
| ase_vac_nve | ase | free_nve | False | nve | missing |  |  | suite_summary.json not found |
| ase_vac_nvt_langevin | ase | free_nvt | False | nvt_langevin | missing |  |  | suite_summary.json not found |
| ase_vac_nvt_nhc | ase | free_nvt | False | nvt_nhc | missing |  |  | suite_summary.json not found |
| jaxmd_pbc_npt | jaxmd | pbc_npt | True | npt_nhc | missing |  |  | suite_summary_jaxmd.json not found |
| jaxmd_pbc_nve | jaxmd | pbc_nve | True | nve | missing |  |  | suite_summary_jaxmd.json not found |
| jaxmd_pbc_nvt | jaxmd | pbc_nvt | True | nvt_nhc | missing |  |  | suite_summary_jaxmd.json not found |
| jaxmd_vac_nve | jaxmd | free_nve | False | nve | missing |  |  | suite_summary_jaxmd.json not found |
| jaxmd_vac_nvt | jaxmd | free_nvt | False | nvt_nhc | pass | 8000 | 300.0 |  |
| pycharmm_pbc_nve | pycharmm | pbc_nve | True | nve | fail | 100 |  | restart_step=100 |
| pycharmm_vac_heat_hoover | pycharmm | free_nvt | False | heat_hoover | pass | 8000 |  |  |
| pycharmm_vac_heat_scale | pycharmm | free_nvt | False | heat_scale | fail | 2000 |  | possible echeck abort |
| pycharmm_vac_nve | pycharmm | free_nve | False | nve | fail | 2000 |  | possible echeck abort |

