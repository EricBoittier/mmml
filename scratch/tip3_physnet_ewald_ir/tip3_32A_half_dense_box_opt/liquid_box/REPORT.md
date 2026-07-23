# Liquid box report

**Status:** PASS
**Profile:** dense

## System

- Composition: `TIP3:548`
- Molecules: 548
- Atoms: 1644

## Box

- Target cubic side: 32.000 Å
- Final cubic side: 32.000 Å
- Target density: 0.5000 g/cm³
- Final density: 0.5003 g/cm³
- Density relative error: 0.06%

## Geometry certification (MM)

- Worst inter-monomer contact: 1.673 Å
- Prep floor: 0.450 Å
- Dynamics overlap reference: 0.450 Å
- CHARMM MM GRMS: 0.3568 kcal/mol/Å

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
  --from-psf /mmhome/boittier/home/mmml/scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/liquid_box/model.psf \
  --from-crd /mmhome/boittier/home/mmml/scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/liquid_box/model.crd \
  --checkpoint /path/to/checkpoint.json \
  --backend jaxmd --setup pbc_nve \
  --output-dir runs/liquid_box_nve
# or: --backend pycharmm --md-stages mini,heat,equi
```
