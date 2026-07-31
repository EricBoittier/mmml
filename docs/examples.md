# Examples

Copy-paste invocations, grouped the same way as `mmml examples`.
Run `mmml <command> --help` for the full flag list of any of these.

!!! note
    This page is generated from `mmml.cli.help_text.EXAMPLE_BLOCKS`,
    so it always matches what `mmml examples` prints.

## Residues & boxes

```bash
mmml make-res --list-residues
mmml make-res --res CYBZ
mmml make-box --res CYBZ --n 50 --box-size 25.0
mmml liquid-box --composition DCM:206 --target-density-g-cm3 1.326 -o boxes/dcm206
mmml health-check --require-gpu
```

## MD & campaigns

```bash
mmml configure
mmml md-system --setup pbc_npt --composition MEOH:5,TIP3:5 --temperature 300
mmml md-system --config campaign.yaml --run-all
mmml warmup-mlpot-jax --checkpoint "$MMML_CKPT" --n-monomers 20
```

## QM pipeline

```bash
mmml fix-and-split --efd data.npz --output-dir ./splits
mmml npz2traj data.npz -o trajectory.traj
mmml pyscf-evaluate -i traj.npz -o out.npz --EF --esp
mmml compare-charmm-ml --checkpoint ~/ckpts/eg_joint --valid-efd splits/energies_forces_dipoles_test.npz --valid-esp splits/grids_esp_test.npz --pdb pdb/initial.pdb --n-samples 50 --out-dir charmm_ml_comparison
mmml physnet-train --config train.yaml
mmml mode-check --composition TIP3:1 --checkpoint "$MMML_CKPT" --output-dir ./mode_tip3_1
mmml mode-check --composition TIP3:2 --checkpoint "$MMML_CKPT" --output-dir ./mode_tip3_2 --checks minimize,fd,bond-scan,vibrations,kick
mmml mode-check --pbc-fd --checkpoint "$MMML_CKPT" --output artifacts/fd_force_check.json
mmml neb --config examples/m/yaml/neb.yaml --overwrite
mmml neb --checkpoint examples/m/kl.json --initial examples/m/neb/reag_0_opt.xyz --final examples/m/neb/prod_0_opt.xyz --output-dir artifacts/nh3_ch3cl/neb --n-images 11 --fmax 0.05
mmml dmc --natm 20 --nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 --alpha 1200.0 --checkpoint "$MMML_CKPT" --input mmml/generate/dmc/examples/acetone_dmc.extxyz
```

Interactive setup for YAML and Snakemake scaffolds: `mmml configure`.

See also: [How the CLI is organized](cli/index.md).
