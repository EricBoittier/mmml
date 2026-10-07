# Examples

Copy-paste invocations, grouped the same way as `karml examples`.
Run `karml <command> --help` for the full flag list of any of these.

!!! note
    This page is generated from `karml.cli.help_text.EXAMPLE_BLOCKS`,
    so it always matches what `karml examples` prints.

## Residues & boxes

```bash
karml make-res --list-residues
karml make-res --res CYBZ
karml make-box --res CYBZ --n 50 --box-size 25.0
karml liquid-box --composition DCM:206 --target-density-g-cm3 1.326 -o boxes/dcm206
karml health-check --require-gpu
```

## MD & campaigns

```bash
karml configure
karml md-system --setup pbc_npt --composition MEOH:5,TIP3:5 --temperature 300
karml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml --job-id nve_smoke
karml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
karml md-system --config campaign.yaml --run-all
karml warmup-mlpot-jax --checkpoint "$KARML_CKPT" --n-monomers 20
karml analyze-liquid --campaign-dir artifacts/lj_scales/liquid_dcm -o analysis/
```

## QM pipeline

```bash
karml fix-and-split --efd data.npz --output-dir ./splits
karml fix-and-split --efd spice.npz -o ./splits --preserve-units
karml npz2traj data.npz -o trajectory.traj
karml pyscf-evaluate -i traj.npz -o out.npz --EF --esp
karml compare-charmm-ml --checkpoint ~/ckpts/eg_joint --valid-efd splits/energies_forces_dipoles_test.npz --valid-esp splits/grids_esp_test.npz --pdb pdb/initial.pdb --n-samples 50 --out-dir charmm_ml_comparison
karml physnet-train --config train.yaml
karml label-acquire --config workflows/label_acquisition/config.smoke.yaml all
karml efield-train --train-npz splits/energies_forces_dipoles_train.npz --valid-npz splits/energies_forces_dipoles_valid.npz --polar_weight 1 --polar-at-zero-field
karml pet-physnet-distill --checkpoint pet-mad.pt --out-dir ./acetone_pet_distill --preset smoke
karml pet-interaction-pes --checkpoint "$PET_MAD_CKPT"
karml mode-check --composition TIP3:1 --checkpoint "$KARML_CKPT" --output-dir ./mode_tip3_1
karml mode-check --composition TIP3:2 --checkpoint "$KARML_CKPT" --output-dir ./mode_tip3_2 --checks minimize,fd,bond-scan,vibrations,kick
karml mode-check --pbc-fd --checkpoint "$KARML_CKPT" --output artifacts/fd_force_check.json
karml neb --config examples/m/yaml/neb.yaml --overwrite
karml neb --checkpoint examples/m/kl.json --initial examples/m/neb/reag_0_opt.xyz --final examples/m/neb/prod_0_opt.xyz --output-dir artifacts/nh3_ch3cl/neb --n-images 11 --fmax 0.05
karml dmc --natm 20 --nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 --alpha 1200.0 --checkpoint "$KARML_CKPT" --input karml/generate/dmc/examples/acetone_dmc.extxyz
```

Interactive setup for YAML and Snakemake scaffolds: `karml configure`.

See also: [How the CLI is organized](cli/index.md).
