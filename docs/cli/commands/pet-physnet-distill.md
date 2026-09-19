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

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --geometries-only     Write unlabeled R/Z/N NPZ (no teacher). For pool
                        inspection.
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
