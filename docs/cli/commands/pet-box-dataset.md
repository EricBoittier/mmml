# `mmml pet-box-dataset`

Many-seed PET box dataset: random packing, FIRE intermediates, NVT (extxyz).


## Usage

```bash
mmml pet-box-dataset --help
```

## Options

```text
usage: mmml pet-box-dataset [-h] --checkpoint CHECKPOINT --out-dir OUT_DIR
                            [--seeds SEEDS] [--residue RESIDUE]
                            [--monomer-xyz MONOMER_XYZ] [--box-size BOX_SIZE]
                            [--n-molecules N_MOLECULES]
                            [--target-density-g-cm3 TARGET_DENSITY_G_CM3]
                            [--temperatures TEMPERATURES] [--dt-fs DT_FS]
                            [--friction FRICTION] [--fire-steps FIRE_STEPS]
                            [--fire-every FIRE_EVERY] [--fire-fmax FIRE_FMAX]
                            [--md-steps MD_STEPS] [--md-every MD_EVERY]
                            [--com-jitter-frac COM_JITTER_FRAC]
                            [--max-force MAX_FORCE]

Many-seed periodic PET dataset: random packing, FIRE intermediates and Langevin
NVT frames, labelled with the driving model (extxyz).

Input & configuration:
  --checkpoint CHECKPOINT
                        metatomic .pt
  --residue RESIDUE     bulk density lookup

Scientific model:
  --target-density-g-cm3 TARGET_DENSITY_G_CM3
  --temperatures TEMPERATURES
                        NVT targets in K, cycled over seeds (e.g. 300,350,400)
  --max-force MAX_FORCE
                        drop frames with max |F| above this (eV/Å)

Execution:
  --seeds SEEDS         e.g. 0-15 or 0,3,9-12
  --dt-fs DT_FS
  --fire-steps FIRE_STEPS
  --md-steps MD_STEPS

Output & artifacts:
  --out-dir OUT_DIR

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --monomer-xyz MONOMER_XYZ
                        default: ethanol
  --box-size BOX_SIZE
  --n-molecules N_MOLECULES
                        default: bulk density
  --friction FRICTION   Langevin 1/fs
  --fire-every FIRE_EVERY
                        keep every Nth FIRE step
  --fire-fmax FIRE_FMAX
  --md-every MD_EVERY
  --com-jitter-frac COM_JITTER_FRAC
                        random COM shift, fraction of lattice spacing
```


## Related docs

- [Metatomic in MMML](../../metatomic.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
