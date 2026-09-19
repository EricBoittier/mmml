# `mmml metatomic-pbc-md`

CHARMM-free metatomic ASE MD in a cubic liquid box (NVT/NVE).


CHARMM-free cubic-box ASE MD through a metatomic `.pt` (PET-MAD by default).
Does **not** call `setup_calculator` or CHARMM. Default recipe: 32 Å ethanol at
0.789 g/cm³ (ETOH:338), 300 K, 0.5 fs. `--ensemble nve` with FIRE mini is the
conservation path.

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
```

Worked example: [PET-MAD ethanol PBC](../../examples/pet-mad-etoh-pbc.md).

## Usage

```bash
mmml metatomic-pbc-md --help
```

## Options

```text
usage: mmml metatomic-pbc-md [-h] [--checkpoint CHECKPOINT] [--residue RESIDUE]
                             [--monomer-xyz MONOMER_XYZ] [--box-size BOX_SIZE]
                             [--target-density-g-cm3 TARGET_DENSITY_G_CM3]
                             [--n-molecules N_MOLECULES]
                             [--temperature TEMPERATURE] [--dt-fs DT_FS]
                             [--n-steps N_STEPS] [--seed SEED]
                             [--ensemble {nve,nvt}] [--friction FRICTION]
                             [--minimize-steps MINIMIZE_STEPS]
                             [--minimize-fmax MINIMIZE_FMAX]
                             [--log-every LOG_EVERY] [--traj-every TRAJ_EVERY]
                             [--output-dir OUTPUT_DIR] [--json-out JSON_OUT]

CHARMM-free metatomic ASE MD in a cubic liquid box. Default: 32 Å ethanol at
experimental density, 300 K, 0.5 fs.

Input & configuration:
  --checkpoint CHECKPOINT
                        TorchScript AtomisticModel (.pt). Default:
                        $PET_MAD_CKPT.
  --residue RESIDUE     CGenFF residue name for bulk-density count (default:
                        ETOH).

Scientific model:
  --target-density-g-cm3 TARGET_DENSITY_G_CM3
                        Bulk liquid density (default:
                        SOLVENT_BULK_PROPS[residue]).
  --temperature TEMPERATURE
  --ensemble {nve,nvt}  nve=VelocityVerlet; nvt=Langevin at --temperature.

Execution:
  --dt-fs DT_FS
  --n-steps N_STEPS
  --seed SEED
  --minimize-steps MINIMIZE_STEPS
                        FIRE steps before assigning velocities (0 skips mini).

Output & artifacts:
  --log-every LOG_EVERY
                        Record PE/KE/Etot every N MD steps (always includes step
                        0).
  --traj-every TRAJ_EVERY
                        Write a labelled periodic frame (E, F, cell) to <output-
                        dir>/traj.extxyz every N MD steps (0 = off). Training
                        data for metatrain / pet-physnet-distill --from-box-
                        extxyz.
  --output-dir OUTPUT_DIR
                        Output directory (default:
                        scratch/pet_mad_etoh_pbc/ase_smoke).
  --json-out JSON_OUT   Report path (default: <output-dir>/report.json).

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --monomer-xyz MONOMER_XYZ
                        Monomer xyz (default:
                        examples/pet_mad_etoh_pbc/etoh.xyz).
  --box-size BOX_SIZE
  --n-molecules N_MOLECULES
                        Override molecule count (default: density → N in --box-
                        size).
  --friction FRICTION   ASE Langevin friction (1/fs) for --ensemble nvt.
  --minimize-fmax MINIMIZE_FMAX
                        FIRE force threshold (eV/Å).
```


## Related docs

- [Metatomic in MMML](../../metatomic.md)
- [PET-MAD ethanol PBC example](../../examples/pet-mad-etoh-pbc.md)
- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
