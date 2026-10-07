# Diffusion Monte Carlo Driver

This directory contains the diffusion Monte Carlo (DMC) driver that uses the
PhysNetJax potential to evaluate walker energies. Energies are evaluated in
parallel via ``jax.vmap`` over walker geometries (chunked by ``--max-batch``).

## CLI

```bash
mmml dmc \
  --natm 20 \
  --nwalker 512 \
  --stepsize 5e-4 \
  --nstep 5000 \
  --eqstep 1000 \
  --alpha 1200.0 \
  --checkpoint mmml/models/physnetjax/ckpts/<your-experiment>/epoch-000123 \
  --max-batch 512 \
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz
```

Equivalent module form: ``python -m mmml.generate.dmc.dmc ...``.

Replace ``<your-experiment>/epoch-000123`` with the checkpoint you want to run.
The command will:

1. minimise the first frame using the PhysNetJax calculator (BFGS; tolerances
   via ``--minimize-fmax`` / ``--minimize-steps``),
2. perturb the minimised geometry by Gaussian noise with std ``--random-sigma``
   (default 0.02 Å), and
3. run standard DMC branching/diffusion for the walkers/steps you provide.

## Input expectations

Any format readable by ``ase.io.read``. The acetone example
``examples/acetone_dmc.extxyz`` has two frames (equilibrium + distorted
reference); only the first frame is minimised.

Supported atom types: H, C, O.

## Outputs

For an input named ``acetone_dmc.extxyz`` (written to ``--output-dir`` or CWD):

- ``acetone_dmc.pot`` – reference potential history (hartree and cm⁻¹)
- ``acetone_dmc.log`` – run metadata and final average energies
- ``defective_acetone_dmc.xyz`` – problematic geometries flagged during branching
- ``configs_acetone_dmc.traj`` – ASE trajectory of the last 10 DMC steps

## Command-line reference

| Flag | Description |
| ---- | ----------- |
| ``--natm`` | Number of atoms per configuration (must match input frame). |
| ``--nwalker`` | Number of walkers. |
| ``--stepsize`` | Imaginary time step in atomic units. |
| ``--nstep`` | Total diffusion steps. |
| ``--eqstep`` | Steps discarded before averaging. |
| ``--alpha`` | Feedback parameter (typically ∝ 1/stepsize). |
| ``--fbohr`` | ``1`` if input is already in Bohr; default Å. |
| ``--checkpoint`` | PhysNetJax checkpoint (experiment or epoch directory). |
| ``--max-batch`` | Upper bound on geometries per energy batch (vmap chunk). |
| ``--minimize-fmax`` | ASE BFGS force goal (default ``1e-3`` eV/Å). |
| ``--minimize-steps`` | Max BFGS steps (default ``200``). |
| ``--random-sigma`` | Perturbation applied to the starting geometry (Å). |
| ``--seed`` | RNG seed (default: wall clock). |
| ``--output-dir`` | Directory for output artifacts (default: CWD). |
| ``-i/--input`` | Geometry file (XYZ/EXTXYZ/ASE-readable). |

Run ``mmml dmc --help`` for the full argument list.

## Tips

- **Memory:** lower ``--max-batch`` if device memory is tight; walker energies
  are still evaluated with the parallel ``vmap`` path in chunks.
- **Determinism:** pass ``--seed`` for reproducible noise / branching draws.
