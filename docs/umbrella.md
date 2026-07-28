# Batched umbrella sampling (NVT + MBAR)

Pure-ML **distance umbrella sampling** packs \(K\) copies of one molecule into a
single PhysNet / SpookyNet batch, adds a different harmonic restraint
\(W_k = \tfrac12 k_k(\xi-\xi_{0,k})^2\) on each copy, and integrates them together
with **JAX-MD NVT Nose-Hoover**. Free-energy differences come from
**pymbar MBAR** on the saved snapshots.

This is **not** CHARMM ADUMB (adaptive umbrella / WHAM). For ADUMB on hybrid
CHARMM+ML systems see [examples/m](examples/nh3-ch3cl-results.md). For MEP /
barrier paths see [NEB](neb.md).

## When to use

| Goal | Tool |
|------|------|
| Canonical PMF along a bond / contact distance (pure ML) | `mmml umbrella-sample` + `mmml umbrella-mbar` |
| Alchemical λ free energy (hybrid MMML) | `md-system --setup lambda_ti` + `mmml lambda-mbar` |
| Adaptive umbrella in CHARMM | ADUMB via `pycharmm_pre_dynamics_lingo` |
| Minimum-energy path | [`mmml neb`](neb.md) |

## How packing works

```text
one XYZ  →  tile K copies  →  flat (K·N) atoms
           batch_segments + offset pair list
           E = Σ_k E_ML(R_k) + Σ_k ½ k_k (r_ij^(k) − ξ₀,k)²
           jax_md.simulate.nvt_nose_hoover(energy_sum, …)
```

The layout matches multi-replica [`mmml physnet-md`](cli/commands/physnet-md.md)
batching. Vacuum / free space only (no PBC MIC restraints in v1).

## Workflow

```bash
# 1) Sample (requires jax-md + a PhysNet/Spooky checkpoint)
# Fix C (2), translate NH₃ (N+H) rigidly along N–C; default dt=0.1 fs
mmml umbrella-sample \
  --checkpoint examples/m/kl.json \
  --structure examples/m/neb/reag_0_opt.xyz \
  --atoms 2,1 --move-with 1,3,4,5 \
  --xi-min 1.5 --xi-max 3.5 --n-windows 11 \
  --k 20 --timestep 0.1 --temperature 300 --nsteps 20000 --savefreq 100 \
  -o artifacts/umbrella --overwrite

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
  --xi-min 1.5 --xi-max 3.0 --n-windows 4 \
  --yi-min 1.5 --yi-max 3.0 --n-windows-y 4 \
  --k 10 --ky 10 --timestep 0.1 --nsteps 5000 \
  -o artifacts/umbrella2d --overwrite
```

CV1 is Cl–C (atom 0 alone); CV2 is N–C with `--move-with2` translating NH₃ rigidly.
`--invert-with` Walden-blends the CH₃ hydrogens as the SN2 progress advances.
Shared hub C (2) stays fixed when seeding both stretches. Full product grids can
still hit harsh corners — soften `--k`, raise `--max-seed-force` only if you
understand the risk, or seed from NEB frames instead.

Artifacts in the run directory:

- `umbrella_snapshots.npz` — positions `(K, N_frames, N, 3)`, `xi0`, `k_ev_A2`, …
- `umbrella_windowXXX.xyz` — per-window trajectory
- `umbrella_summary.json` — run args + MBAR block after step 2

## MBAR formula

For samples \(R_k^n\) from window \(k\):

\[
u_{kln} = \beta\bigl(U_{\mathrm{ML}}(R_k^n) + W_l(R_k^n)\bigr),\quad
W_l(R)=\tfrac12 k_l\bigl(\|r_i-r_j\|-\xi_{0,l}\bigr)^2.
\]

\(U_{\mathrm{ML}}\) is evaluated once per sample; biases are analytic.
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
