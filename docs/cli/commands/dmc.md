# `mmml dmc`

Diffusion Monte Carlo with PhysNetJax (batched walkers).


## Usage

```bash
mmml dmc --help
```

## Options

```text
usage: mmml dmc [-h] --natm NATM --nwalker NWALKER --stepsize STEPSIZE
                --nstep NSTEP --eqstep EQSTEP --alpha ALPHA [--fbohr {0,1}]
                --checkpoint CHECKPOINT [--max-batch MAX_BATCH]
                [--minimize-fmax MINIMIZE_FMAX]
                [--minimize-steps MINIMIZE_STEPS] [--random-sigma RANDOM_SIGMA]
                [--seed SEED] -i INPUT [--output-dir OUTPUT_DIR]

Diffusion Monte Carlo with PhysNetJax energies (batched walker evaluation via
jax.vmap).

Input & configuration:
  --checkpoint CHECKPOINT
                        PhysNetJax checkpoint directory (experiment or epoch
                        path).
  -i, --input INPUT     Geometry file (XYZ/EXTXYZ/anything ASE can read).

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

Diffusion Monte Carlo driver using PhysNetJax energies. Originally adapted from
the TensorFlow-based implementation by Silvan Kaeser. This version evaluates
walker energies with the PhysNetJax model (batched via ``jax.vmap``) to stay
consistent with the rest of the MMML tooling. CLI:: mmml dmc \ --natm 20
--nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 \ --alpha 1200.0
--checkpoint path/to/epoch-NNNNNN \ --input
mmml/generate/dmc/examples/acetone_dmc.extxyz
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
