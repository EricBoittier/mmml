# `mmml physnet-md`

PhysNet MD sampling.


## Usage

```bash
mmml physnet-md --help
```

## Options

```text
usage: mmml physnet-md [-h] --checkpoint CHECKPOINT (--structure STRUCTURE |
                       --data DATA) [-o OUTPUT_DIR] [--temperature TEMPERATURE]
                       [--timestep TIMESTEP] [--nsteps-ase NSTEPS_ASE]
                       [--nsteps-jaxmd NSTEPS_JAXMD] [--printfreq PRINTFREQ]
                       [--skip-jaxmd] [--n-replicas B]
                       [--use-ema | --no-use-ema]

PhysNet MD sampling with ASE and JAX-MD.

Input & configuration:
  --checkpoint CHECKPOINT
                        Path to PhysNet checkpoint directory (e.g.
                        out/ckpts/cybz_physnet)
  --structure STRUCTURE
                        Initial structure (XYZ, PDB, etc.)
  --data DATA           NPZ with R, Z (e.g.
                        splits/energies_forces_dipoles_train.npz)

Scientific model:
  --temperature TEMPERATURE
                        Temperature in K (default: 300)

Execution:
  --nsteps-ase NSTEPS_ASE
                        ASE Langevin steps (default: 100)
  --nsteps-jaxmd NSTEPS_JAXMD
                        JAX-MD Nose-Hoover steps (default: 200)

Output & artifacts:
  -o, --output-dir OUTPUT_DIR
                        Output directory (default: .)

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --timestep TIMESTEP   Timestep in fs (default: 0.5)
  --printfreq PRINTFREQ
                        Print/save interval (default: 25)
  --skip-jaxmd          Skip JAX-MD (ASE only)
  --n-replicas B        Number of independent replicas to run in parallel. ASE:
                        ProcessPoolExecutor; JAX-MD: batched GPU. With --data,
                        uses first B structures as initial geometries if
                        available. (default: 1)
  --use-ema, --no-use-ema
                        Deploy the checkpoint's EMA params (default: on). Use
                        --no-use-ema for the live training weights.

CLI for PhysNet molecular dynamics sampling with ASE and JAX-MD. Runs NVT
Langevin (ASE) and NVT Nose-Hoover (JAX-MD) using a trained PhysNet checkpoint
as the energy/force calculator. Usage: mmml physnet-md --checkpoint
out/ckpts/cybz_physnet --structure molecule.xyz -o out/ mmml physnet-md
--checkpoint out/ckpts/cybz_physnet --data
splits/energies_forces_dipoles_train.npz -o out/ mmml physnet-md --checkpoint
out/ckpts/cybz_physnet --data splits/train.npz -o out/ --n-replicas 4
```

## Visual examples

![Energy conservation with force snapshots](../../images/povray-overlays/water_nve_with_povray.png)


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
