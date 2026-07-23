# Liquid box report

**Status:** PASS
**Profile:** dense

## System

- Composition: `TIP3:903`
- Molecules: 903
- Atoms: 2709

## Box

- Target cubic side: 30.000 Å
- Final cubic side: 30.000 Å
- Target density: 1.0000 g/cm³
- Final density: 1.0005 g/cm³
- Density relative error: 0.05%

## Geometry certification (MM)

- Worst inter-monomer contact: 1.169 Å
- Prep floor: 0.450 Å
- Dynamics overlap reference: 0.450 Å
- CHARMM MM GRMS: 0.0407 kcal/mol/Å

## Steps applied

- packmol_cluster
- mc_density_skipped_hold_box
- save_model_topology
- charmm_mm_pre_minimize
- pre_mlpot:monomer_repack_skipped_clean
- pre_mlpot:mc_density_skipped_hold_box
- write_model_crd

## Artifacts

- `model.psf`
- `model.crd`
- `model.pdb`
- `box.json`
- `prep_ladder/` (checkpoints)

## Next step

```bash
mmml md-system \
  --from-psf /mmhome/boittier/home/mmml/scratch/tip3_physnet_ewald_ir/tip3_90_box_opt/liquid_box/model.psf \
  --from-crd /mmhome/boittier/home/mmml/scratch/tip3_physnet_ewald_ir/tip3_90_box_opt/liquid_box/model.crd \
  --checkpoint /path/to/checkpoint.json \
  --backend jaxmd --setup pbc_nve \
  --output-dir runs/liquid_box_nve
# or: --backend pycharmm --md-stages mini,heat,equi
```
