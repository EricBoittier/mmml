# `mmml pet-physnet-distill`

PET-MAD teacher → PhysNet NPZ (acetone dataset + synthetic pool).


## Usage

```bash
mmml pet-physnet-distill --help
```

## Options

```text
usage: mmml pet-physnet-distill [-h] [--checkpoint CHECKPOINT] --out-dir OUT_DIR
                                [--preset {smoke,md}] [--seed SEED]
                                [--energy-mode {interaction,total}]
                                [--geometries-only]
                                [--extra-extxyz [EXTRA_EXTXYZ ...]]
                                [--from-box-extxyz FROM_BOX_EXTXYZ [FROM_BOX_EXTXYZ ...]]
                                [--atoms-per-monomer ATOMS_PER_MONOMER]
                                [--reference-monomer-xyz REFERENCE_MONOMER_XYZ]
                                [--frame-stride FRAME_STRIDE]
                                [--dimer-com-cutoff DIMER_COM_CUTOFF]
                                [--max-monomers-per-frame MAX_MONOMERS_PER_FRAME]
                                [--max-dimers-per-frame MAX_DIMERS_PER_FRAME]
                                [--valid-fraction VALID_FRACTION]
                                [--teacher-backend {torchscript,ase}]
                                [--max-atoms-per-batch MAX_ATOMS_PER_BATCH]
                                [--max-systems-per-batch MAX_SYSTEMS_PER_BATCH]
                                [--student-yaml | --no-student-yaml]

Build an acetone geometry pool (dataset + noise/scans), label it with a
metatomic PET teacher, and write PhysNet-train NPZ in eV.

Input & configuration:
  --checkpoint CHECKPOINT
                        Teacher AtomisticModel (.pt). Required unless
                        --geometries-only.

Scientific model:
  --energy-mode {interaction,total}
                        interaction: monomer E-E_ref and unswitched dimer E_int
                        (default, hybrid MD)
  --dimer-com-cutoff DIMER_COM_CUTOFF
                        Å, box pool dimers

Execution:
  --preset {smoke,md}
  --seed SEED
  --teacher-backend {torchscript,ase}
                        torchscript: batched AtomisticModel forward over many
                        structures (default); ase: one MetatomicCalculator call
                        per structure
  --max-atoms-per-batch MAX_ATOMS_PER_BATCH
                        torchscript backend: atom budget per forward (lower for
                        larger PETs)
  --max-systems-per-batch MAX_SYSTEMS_PER_BATCH
                        torchscript backend: structure budget per forward

Output & artifacts:
  --out-dir OUT_DIR     Train/valid NPZ + report.json
  --extra-extxyz [EXTRA_EXTXYZ ...]
                        Additional ASE extxyz frames (10-atom monomers or
                        20-atom dimers)
  --from-box-extxyz FROM_BOX_EXTXYZ [FROM_BOX_EXTXYZ ...]
                        Periodic MD frames (extxyz with cell, e.g. metatomic-
                        pbc-md --traj-every). Replaces the acetone pool with
                        monomers and COM-close dimers cut out by minimum image.
                        Needs --atoms-per-monomer.

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --geometries-only     Write unlabeled R/Z/N NPZ (no teacher). For pool
                        inspection.
  --atoms-per-monomer ATOMS_PER_MONOMER
  --reference-monomer-xyz REFERENCE_MONOMER_XYZ
                        Gas-phase monomer for E_ref in interaction mode (box
                        pool only)
  --frame-stride FRAME_STRIDE
                        Use every Nth box frame
  --max-monomers-per-frame MAX_MONOMERS_PER_FRAME
  --max-dimers-per-frame MAX_DIMERS_PER_FRAME
  --valid-fraction VALID_FRACTION
  --student-yaml, --no-student-yaml
                        Write physnet-train.yaml next to the NPZ (warm-start
                        DESdimers)
```


## Related docs

- [Metatomic in MMML](../../metatomic.md)
- [Bayesian PES design](../../bayesian-pes-design.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
