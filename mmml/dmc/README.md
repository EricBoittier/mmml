# Diffusion Monte Carlo Driver

This directory contains a standalone diffusion Monte Carlo (DMC) driver that
uses the PhysNetJax potential to evaluate walker energies.  The script was
adapted from an earlier TensorFlow implementation by Silvan Kaeser but now runs
entirely on JAX to stay in step with the rest of the MMML stack.

## Features

- Evaluates walker energies with PhysNetJax checkpoints (GPU/TPU/CPU via JAX).
- Minimises the reference geometry with the PhysNetJax ASE calculator before
  sampling and generates a perturbed starting geometry automatically.
- Streams the last 10 Monte Carlo steps to an ASE `.traj` file for inspection.
- Accepts multi-frame XYZ/extended-XYZ input through ASE, so existing MMML
  datasets can be fed in directly.

## Requirements

- Python environment with the MMML package (this repository) installed.
- Optional but recommended: GPU-enabled JAX build for performance.
- Runtime dependencies: `jax`, `ase`, `numpy`, and PhysNetJax checkpoints.

## Input expectations

The script understands any file format readable by `ase.io.read`.  For
reproducibility we ship an acetone dimer example in
`examples/acetone_dmc.extxyz`.  The file contains two frames:

1. the equilibrium geometry (used for the minimisation stage), and
2. a slightly distorted geometry (kept for reference, but a fresh random
   distortion is generated at runtime).

## Quick start example

```bash
# From the project root
python -m mmml.dmc.dmc \
  --natm 20 \
  --nwalker 512 \
  --stepsize 5e-4 \
  --nstep 5000 \
  --eqstep 1000 \
  --alpha 1200.0 \
  --checkpoint mmml/physnetjax/ckpts/<your-experiment>/epoch-000123 \
  --max-batch 512 \
  --input mmml/dmc/examples/acetone_dmc.extxyz
```

Replace `<your-experiment>/epoch-000123` with the checkpoint you want to run.
The script will:

1. minimise the first frame using the PhysNetJax calculator (BFGS, tolerances
   controlled via `--minimize-fmax` and `--minimize-steps`),
2. perturb the minimised geometry by a Gaussian noise with standard deviation
   `--random-sigma` (default 0.02 Å), and
3. run standard DMC branching/diffusion for the number of walkers/steps you
   provide.

## Outputs

For an input named `acetone_dmc.extxyz`, the script produces:

- `acetone_dmc.pot` – reference potential history (hartree and cm⁻¹).
- `acetone_dmc.log` – run metadata and final average energies.
- `defective_acetone_dmc.xyz` – problematic geometries (if any) flagged during
  branching.
- `configs_acetone_dmc.traj` – ASE trajectory capturing the last 10 DMC steps
  for all surviving walkers.

All files are written alongside the input unless you run the script from a
different working directory.

## Command-line reference

| Flag | Description |
| ---- | ----------- |
| `--natm` | Number of atoms per configuration (must match input frame). |
| `--nwalker` | Number of walkers sampled in parallel. |
| `--stepsize` | Imaginary time step in atomic units. |
| `--nstep` | Total diffusion steps. |
| `--eqstep` | Steps discarded for equilibration before averaging. |
| `--alpha` | Feedback parameter (typically ∝ 1/stepsize). |
| `--fbohr` | Set to `1` if input is already in Bohr; defaults to Å. |
| `--checkpoint` | PhysNetJax checkpoint (experiment or epoch directory). |
| `--max-batch` | Upper bound on the number of geometries per energy batch. |
| `--minimize-fmax` | Force convergence goal for ASE BFGS (default `1e-3` eV/Å). |
| `--minimize-steps` | Max BFGS steps (default `200`). |
| `--random-sigma` | Magnitude of the random perturbation applied to the starting geometry. |
| `-i/--input` | Path to the geometry file (XYZ/EXTXYZ/anything readable by ASE). |

Run `python -m mmml.dmc.dmc --help` for the full argument list.

## Advanced usage tips

- **Memory tuning:** If you hit device memory limits, lower `--max-batch`; the
  script automatically slices walker evaluations into manageable chunks.
- **Determinism:** The random distortion is seeded with the current wall clock.
  Set `PYTHONHASHSEED` and call `np.random.seed(...)` early in the script if you
  need fully deterministic results.
- **Alternate inputs:** You can pass multi-frame trajectories (e.g. `.traj` or
  `.extxyz`). Only the first frame is used for minimisation; the last frame can
  serve as a visual reference.

## Further reading

See the original DMC references cited at the top of `dmc.py` for algorithmic
background, and explore the scripts under `examples/` for larger MMML
workflows.
