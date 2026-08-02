# `mmml dmc`

Diffusion Monte Carlo with PhysNetJax (batched walkers).


Diffusion Monte Carlo on a PhysNetJax potential. Walker energies are evaluated
in parallel with `jax.vmap` (chunked by `--max-batch`).

## Example (acetone dimer)

Bundled geometry: `mmml/generate/dmc/examples/acetone_dmc.extxyz` (20 atoms).

Smoke run (short equilibration, few production steps):

```bash
mmml env   # resolve $MMML_CKPT if you use the bundled checkpoint

mmml dmc \
  --natm 20 \
  --nwalker 64 \
  --stepsize 5e-4 \
  --nstep 200 \
  --eqstep 50 \
  --alpha 1200.0 \
  --max-batch 64 \
  --seed 0 \
  --checkpoint "$MMML_CKPT" \
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz \
  --output-dir runs/dmc_acetone_smoke
```

Production-style settings (more walkers / longer averaging):

```bash
mmml dmc \
  --natm 20 \
  --nwalker 512 \
  --stepsize 5e-4 \
  --nstep 5000 \
  --eqstep 1000 \
  --alpha 1200.0 \
  --max-batch 512 \
  --seed 0 \
  --checkpoint "$MMML_CKPT" \
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz \
  --output-dir runs/dmc_acetone
```

Outputs under `--output-dir` (or CWD):

- `acetone_dmc.pot` — reference energy vs step (hartree and cm⁻¹)
- `acetone_dmc.log` — run metadata + average energy
- `configs_acetone_dmc.traj` — last 10 steps of surviving walkers
- `defective_acetone_dmc.xyz` — geometries flagged below the reference minimum

See the [Diffusion Monte Carlo guide](../../dmc.md) for inputs, units, and
memory tips.

## Usage

```bash
mmml dmc --help
```

## Options

```text
usage: mmml dmc [-h] --natm NATM --nwalker NWALKER --stepsize STEPSIZE
                --nstep NSTEP --eqstep EQSTEP --alpha ALPHA [--fbohr {0,1}]
                --checkpoint CHECKPOINT [--model {physnet,kernnn}]
                [--max-batch MAX_BATCH] [--minimize-fmax MINIMIZE_FMAX]
                [--minimize-steps MINIMIZE_STEPS] [--random-sigma RANDOM_SIGMA]
                [--seed SEED] -i INPUT [--output-dir OUTPUT_DIR]

Diffusion Monte Carlo with PhysNetJax energies (batched walker evaluation via
jax.vmap).

Input & configuration:
  --checkpoint CHECKPOINT
                        PhysNetJax or KerNN checkpoint (JSON / Orbax).
  -i, --input INPUT     Geometry file (XYZ/EXTXYZ/anything ASE can read).

Scientific model:
  --model {physnet,kernnn}
                        Energy backend (default: auto-detect KerNN JSON vs
                        PhysNet).

Execution:
  --stepsize STEPSIZE   Imaginary-time stepsize (atomic units).
  --max-batch MAX_BATCH
                        Maximum walker geometries evaluated per JAX energy batch
                        (default: 512).
  --minimize-steps MINIMIZE_STEPS
                        Maximum ASE BFGS steps for the reference geometry
                        (default: 200).
  --seed SEED           RNG seed (default: wall-clock time).

Output & artifacts:
  --output-dir OUTPUT_DIR
                        Directory for .pot/.log/.traj outputs (default: current
                        working directory).

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --natm NATM           Number of atoms per configuration (must match the input
                        frame).
  --nwalker NWALKER     Number of walkers in the simulation.
  --nstep NSTEP         Total number of diffusion steps.
  --eqstep EQSTEP       Equilibration steps discarded before energy averaging.
  --alpha ALPHA         Feedback parameter (typically proportional to
                        1/stepsize).
  --fbohr {0,1}         1 if input geometry is already in Bohr; 0 if Angstrom
                        (default).
  --minimize-fmax MINIMIZE_FMAX
                        ASE BFGS force convergence criterion in eV/Å (default:
                        1e-3).
  --random-sigma RANDOM_SIGMA
                        Gaussian noise (Å) applied to the minimised geometry for
                        x0 (default: 0.02).

Example (acetone dimer smoke): mmml dmc --natm 20 --nwalker 64 --stepsize 5e-4
--nstep 200 --eqstep 50 --alpha 1200.0 \ --checkpoint "$MMML_CKPT" \ --input
mmml/generate/dmc/examples/acetone_dmc.extxyz \ --output-dir
runs/dmc_acetone_smoke Docs: docs/dmc.md | mmml dmc --help
```


## Related docs

- [Diffusion Monte Carlo guide](../../dmc.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
