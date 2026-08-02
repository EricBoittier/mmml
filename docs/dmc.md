# Diffusion Monte Carlo (`mmml dmc`)

Estimate vibrational ground-state energies with diffusion Monte Carlo on a
PhysNetJax potential. Walker energies are evaluated in parallel via
`jax.vmap` (chunked by `--max-batch`).

CLI reference: [`mmml dmc`](cli/commands/dmc.md).

## Quick example (acetone dimer)

The repo ships a 20-atom acetone dimer geometry at
`mmml/generate/dmc/examples/acetone_dmc.extxyz`.

```bash
# Resolve a PhysNetJax checkpoint path (sets hints for $MMML_CKPT)
mmml env

mkdir -p runs/dmc_acetone_smoke

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

For a longer production-style run:

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

### What the command does

1. Minimises the first input frame with the PhysNetJax ASE calculator (BFGS).
2. Perturbs that geometry with Gaussian noise (`--random-sigma`, default 0.02 Å).
3. Diffuses / branches walkers for `--nstep` imaginary-time steps.
4. Averages the reference energy after `--eqstep` equilibration steps.

### Outputs

Written to `--output-dir` (default: current working directory), using the input
stem (e.g. `acetone_dmc`):

| File | Contents |
|------|----------|
| `*.pot` | Step, alive walkers, \(V_\mathrm{ref}\) (hartree and cm⁻¹) |
| `*.log` | Run metadata and average energy |
| `configs_*.traj` | ASE trajectory of the last 10 steps |
| `defective_*.xyz` | Walkers that dipped below the reference minimum |

### Pass / fail smoke check

- Process exits 0 and prints `nalive` each 10 steps.
- `*.pot` has `nstep + 1` lines (including step 0).
- Average energy in `*.log` is finite (cm⁻¹ column populated).

## Inputs

- Any ASE-readable geometry (`xyz`, `extxyz`, …).
- `--natm` must match the atom count of the first frame.
- Currently supported element types: **H, C, N, O, Cl** (NH₃–CH₃Cl uses H, C, N, Cl).
- `--fbohr 1` if coordinates are already in Bohr (default assumes Å).

## Tunables

| Flag | Typical role |
|------|----------------|
| `--stepsize` | Imaginary-time step (a.u.); smaller is stabler / slower |
| `--alpha` | Population feedback; often ≈ `1/stepsize` scale |
| `--nwalker` / `--max-batch` | Walkers vs GPU/CPU memory; lower `--max-batch` if OOM |
| `--seed` | Reproducible noise + branching draws |

## See also

- Package README: `mmml/generate/dmc/README.md`
- [`mmml physnet-md`](cli/commands/physnet-md.md) — classical MD sampling on the same potential
