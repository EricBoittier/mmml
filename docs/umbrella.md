# Batched umbrella sampling (NVT + MBAR)

`mmml umbrella-sample` supports two engines:

| `engine` | System | Energy |
|----------|--------|--------|
| `packed_ml` (default) | Vacuum / free-space molecule | Pure ML; \(K\) windows packed into one JAX-MD NVT |
| `hybrid_jaxmd` | Explicit solvent (PSF + box) | Mechanical embedding: **ML reactive complex** + **MM solvent**; one window at a time |

Both add a harmonic distance restraint
\(W_k = \tfrac12 k_k(\xi-\xi_{0,k})^2\). Free-energy differences come from
**pymbar MBAR** on the saved snapshots.

This is **not** CHARMM ADUMB (adaptive umbrella / WHAM). For ADUMB on hybrid
CHARMM+ML systems see [examples/m](examples/nh3-ch3cl-results.md). For MEP /
barrier paths see [NEB](neb.md).

## When to use

| Goal | Tool |
|------|------|
| Canonical PMF along a bond / contact distance (pure ML, gas) | `mmml umbrella-sample` (`packed_ml`) + `mmml umbrella-mbar` |
| Same PMF with explicit solvent (ML solute + MM solvent) | `mmml umbrella-sample --engine hybrid_jaxmd` + `mmml umbrella-mbar` |
| Alchemical λ free energy (hybrid MMML) | `md-system --setup lambda_ti` + `mmml lambda-mbar` |
| Adaptive umbrella in CHARMM | ADUMB via `pycharmm_pre_dynamics_lingo` |
| Minimum-energy path | [`mmml neb`](neb.md) |

## How packing works

```text
one XYZ  →  tile K copies  →  flat (K·N) atoms
           batch_segments + offset pair list
           E = Σ_k E_ML(R_k) + Σ_k ½ k_k (r_ij^(k) − ξ₀,k)²
           jax_md.simulate.nvt_langevin(…, center_velocity=False)
```

Langevin is the default thermostat for packed multi-window runs: a shared
Nose-Hoover chain couples replicas, so one hot window can spike the batch T and
NaN everyone else. Per-window kinetic temperatures are printed each
`--printfreq` and abort above `--max-window-temp` (default 5×T).

### Replica exchange

`--replica-exchange` enables Hamiltonian RE between neighbor windows (bias-only
Metropolis; ML energy cancels). Even/odd neighbor pairs on the 1D chain or 2D
grid are proposed every `--rex-freq` steps (default 100). Cumulative acceptance
is written to `umbrella_summary.json`.

The packed layout matches multi-replica [`mmml physnet-md`](cli/commands/physnet-md.md)
batching. Use `hybrid_jaxmd` for PBC / explicit solvent (MIC restraints via the
shared `smd` energy term).

### Hybrid mechanical embedding (`engine: hybrid_jaxmd`)

```text
PSF + PDB + box → MolecularSystem
  mol_id: ML-region atoms share one id  →  MM drops solute–solute pairs
  E = E_ML(solute complex) + E_MM(pairs with solvent) + W(ξ)
  JaxmdDriver NVT per window (not packed)
```

- ML region defaults to residue names `AMM1,CH3CL` (one PhysNet evaluation;
  matches a 9-atom dimer checkpoint such as `examples/m/kl.json`).
- Solute–solvent and solvent–solvent stay in `mm_nonbonded`.
- Snapshots store `energies_unbiased_ev` (= \(E_{\mathrm{ML}}+E_{\mathrm{MM}}\))
  so MBAR does not need to rebuild the hybrid Hamiltonian.
- v1: 1D CV only; no replica exchange.

Example: [`examples/m/yaml/umbrella_nc_tip3.yaml`](../examples/m/yaml/umbrella_nc_tip3.yaml).

## Workflow

```bash
# 1a) Gas-phase sample (requires jax-md + a PhysNet/Spooky checkpoint)
# Fix C (2), translate NH₃ (N+H) rigidly along N–C; default dt=0.1 fs
mmml umbrella-sample \
  --checkpoint examples/m/kl.json \
  --structure examples/m/neb/reag_0_opt.xyz \
  --atoms 2,1 --move-with 1,3,4,5 \
  --xi-min 1.5 --xi-max 3.5 --n-windows 11 \
  --k 20 --timestep 0.1 --temperature 300 --nsteps 20000 --savefreq 100 \
  -o artifacts/umbrella --overwrite

# 1b) Solvated mechanical-embedding sample (after make-box)
mmml umbrella-sample --config examples/m/yaml/umbrella_nc_tip3.yaml --overwrite
# or: bash examples/m/14_umbrella_sample_sol.sh

# NPZ with R/Z (optional --seed-mode frames for pre-generated window seeds):
# mmml umbrella-sample --checkpoint ckpt.json --structure data.npz \
#   --atoms 0,1 --targets 1.8,2.0,2.2 --seed-mode frames -o artifacts/umbrella

# 2) MBAR (requires: uv sync --extra mbar)
mmml umbrella-mbar --run-dir artifacts/umbrella
```

`--structure` accepts **XYZ, PDB, or NPZ** (`R`, `Z`). Default `--seed-mode stretch`
fixes `atom_i` and translates `atom_j` (plus `--move-with` for a rigid group) to
each window ξ₀. Without `--move-with`, dangling H atoms left behind after a
large stretch can blow up forces — use the group for NH₃/CH₃ fragments, prefer
`dt ≤ 0.1 fs` for H-containing systems, or seed from NEB frames.

### 2D umbrella

Pass a second distance CV with `--atoms2 K,L` and a Y grid
(`--yi-min/--yi-max/--n-windows-y` or `--targets-y`). Windows are the product
grid (``nx × ny``), batched in one JAX-MD NVT. MBAR writes
`pmf_rel_kcal_mol_2d` reshaped to `grid_shape`.

```bash
mmml umbrella-sample \
  --checkpoint examples/m/kl.json \
  --structure examples/m/neb/reag_0_opt.xyz \
  --atoms 0,2 --atoms2 1,2 \
  --move-with2 1,3,4,5 --invert-with 6,7,8 \
  --xi-min 1.8 --xi-max 3.0 --n-windows 4 \
  --yi-min 1.8 --yi-max 3.0 --n-windows-y 4 \
  --k 10 --ky 10 --timestep 0.1 --nsteps 5000 \
  --replica-exchange --rex-freq 100 \
  -o artifacts/umbrella2d --overwrite
```

CV1 is Cl–C; CV2 is N–C with `--move-with2` for NH₃. `--invert-with` Walden-blends
CH₃ hydrogens along the SN2 progress. Avoid the (1.5, 1.5) corner from the reactant
geometry — both ligands bonded without a proper TS seed blows up forces
(`--max-seed-force` aborts those windows). Prefer ≥1.8 Å grids, an SN2 corridor
(`--targets` / `--targets-y` along ClC+NC≈const), or NEB `--seed-mode frames`.

Artifacts in the run directory:

- `umbrella_snapshots.npz` — positions `(K, N_frames, N, 3)`, per-frame
  `energies_ev` (`E_ML+W`), `xi0`, `k_ev_A2`, …
- `umbrella_bin_minima.traj` — ASE trajectory with the lowest-`E_ML+W`
  structure per window (mass-weighted CoM at the origin)
- `umbrella_windowXXX.xyz` — optional per-window trajectory
  (`--write-window-xyz`; each frame CoM-centered at the origin)
- `umbrella_summary.json` — run args + MBAR block after step 2

## MBAR formula

For samples \(R_k^n\) from window \(k\):

\[
u_{kln} = \beta\bigl(U_{\mathrm{unbiased}}(R_k^n) + W_l(R_k^n)\bigr),\quad
W_l(R)=\tfrac12 k_l\bigl(\|r_i-r_j\|-\xi_{0,l}\bigr)^2.
\]

For `packed_ml`, \(U_{\mathrm{unbiased}}=U_{\mathrm{ML}}\) is recomputed from the
checkpoint. For `hybrid_jaxmd`, snapshots store
`energies_unbiased_ev` \(=U_{\mathrm{ML}}+U_{\mathrm{MM}}\) at sample time
(biases remain analytic; MIC distances when a box is present).
The reported PMF is window free energy relative to the minimum window.

## CLI reference

- [`mmml umbrella-sample`](cli/commands/umbrella-sample.md)
- [`mmml umbrella-mbar`](cli/commands/umbrella-mbar.md)

## Library API

```python
from mmml.umbrella import UmbrellaConfig, run_umbrella_nvt, run_umbrella_mbar
from mmml.umbrella.config import UmbrellaMbarConfig

cfg = UmbrellaConfig(
    checkpoint="ckpt",
    structure="mol.xyz",
    output_dir="out",
    atom_i=0,
    atom_j=1,
    xi_min=1.5,
    xi_max=3.5,
    n_windows=11,
    k_ev_A2=20.0,
    nsteps=1000,
    overwrite=True,
)
run_umbrella_nvt(cfg)
run_umbrella_mbar(UmbrellaMbarConfig(run_dir="out"))
```
