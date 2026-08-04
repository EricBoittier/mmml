# Batched umbrella sampling (NVT + MBAR)

`mmml umbrella-sample` supports two engines:

| `engine` | System | Energy |
|----------|--------|--------|
| `packed_ml` (default) | Vacuum / free-space molecule | Pure ML; \(K\) windows packed into one JAX-MD NVT |
| `hybrid_jaxmd` | Explicit solvent (PSF + box) | Mechanical embedding: **ML reactive complex** + **MM solvent**; one window at a time |

Both add a harmonic restraint
\(W_k = \tfrac12 k_k(\xi-\xi_{0,k})^2\). For **distances**, \(\xi\) is in Å and
\(k\) in eV/Å². For **dihedrals** (`cv_x.kind: dihedral`), \(\xi\) is in degrees
(periodic shortest-arc Δφ) and \(k\) in eV/deg² — see the
[peptide φ/ψ teaching exercise](examples/tria-phi-psi-scan.md).
Free-energy differences come from **pymbar MBAR** on the saved snapshots.

This is **not** CHARMM ADUMB (adaptive umbrella / WHAM). For ADUMB on hybrid
CHARMM+ML systems see [examples/m](examples/nh3-ch3cl-results.md). For MEP /
barrier paths see [NEB](neb.md).

## When to use

| Goal | Tool |
|------|------|
| Canonical PMF along a bond / contact distance (pure ML, gas) | `mmml umbrella-sample` (`packed_ml`) + `mmml umbrella-mbar` |
| Same PMF with explicit solvent (ML solute + MM solvent) | `mmml umbrella-sample --engine hybrid_jaxmd` + `mmml umbrella-mbar` |
| Backbone φ/ψ constrained maps + gas dihedral PMF | [Teaching exercise](examples/tria-phi-psi-scan.md) (`DihedralCV`, `seed_mode: frames`) |
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
  E = E_ML(solute complex) + E_MM_bonded(solvent) + E_MM(pairs with solvent) + W(ξ)
  JaxmdDriver NVT per window (not packed)
```

- ML region defaults to residue names `AMM1,CH3CL` (one PhysNet evaluation;
  matches a 9-atom dimer checkpoint such as `examples/m/model_ext.json`).
- Solvent intramolecular CGenFF bonded is required (`mm_bonded`); without it,
  TIP3 has no O–H restoring forces and NVE/NVT explodes.
- Solute–solvent and solvent–solvent stay in `mm_nonbonded`.
- Snapshots store `energies_unbiased_ev` (= \(E_{\mathrm{ML}}+E_{\mathrm{MM}}\))
  so MBAR does not need to rebuild the hybrid Hamiltonian.
- v1: 1D CV only; no replica exchange.

Example: [`examples/m/yaml/umbrella_nc_tip3.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/m/yaml/umbrella_nc_tip3.yaml).

#### Pair lists: static or rebuilt

`hybrid_jaxmd` has two ways to feed MM pairs to the device, chosen by
`static_pairs` (default **on**):

| `static_pairs` | What happens | Use when |
|---|---|---|
| `true` (default) | The complete intermolecular list is built once and uploaded once. The switching functions cull by distance on the GPU, so no host rebuild happens and none of the per-block transfer cost is paid. | Up to ~4 800 atoms |
| `false` | `make_intermolecular_neighbor_fn` rebuilds a padded list on the host, with `nl_skin_A` of Verlet skin and a block size from `mmml.md.nl_cadence`. | Above ~4 800 atoms, where the O(N²) energy costs more than the rebuild saves |

**Correctness is identical**, which is the precondition for treating this as a
pure performance choice. The switched force field makes pairs beyond `ctofnb`
contribute exactly zero, so the complete list and the cutoff list evaluate to
the same number. Measured across 300–15 000 atoms, complete list versus a list
built at the production cutoff (12 Å = `ctofnb`):

| | worst case over all sizes |
|---|---|
| \|ΔE\| | 2.5 × 10⁻¹² eV, on totals of order 200 eV |
| max \|ΔF\| | 6.4 × 10⁻¹⁴ eV/Å |

Reproduce with
[`scripts/bench_static_vs_neighbor_pairs.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/bench_static_vs_neighbor_pairs.py)
(CHARMM-free; TIP3P water at experimental density).

##### Where the crossover is

Per-step cost, energy + forces, with the host rebuild amortised over a 20-step
block. A100 and a single CPU core; ratio > 1 means the static list is faster:

| atoms | GPU static (ms) | GPU rebuilt (ms) | GPU ratio | faster |
|---:|---:|---:|---:|---|
| 300 | 1.02 | 1.16 | 1.14× | static |
| 600 | 1.65 | 2.48 | 1.50× | static |
| 1 200 | 3.84 | 9.58 | **2.49×** | static |
| 2 625 | 15.69 | 36.58 | **2.33×** | static |
| 4 800 | 50.54 | 51.80 | 1.02× | level |
| 7 200 | 104.07 | 75.00 | 0.72× | **rebuilt** |
| 10 500 | 227.35 | 111.34 | 0.49× | **rebuilt** |
| 15 000 | 502.72 | 155.18 | 0.31× | **rebuilt** |

So the crossover is at **~4 800 atoms**. Below it the static list wins by up to
2.5×; above it the rebuilt list wins, reaching 3.2× by 15 000 atoms. The
default targets the regime this engine is for — a solute in a few thousand
solvent atoms — and the sampler prints a note if a run starts past the
crossover with `static_pairs` still on.

**Below about twice the cutoff, a neighbour list prunes nothing.** At 300 atoms
the box is 14.4 Å against a 12 Å cutoff, and the list holds 44 548 of the
44 550 possible intermolecular pairs. The rebuild is pure overhead, and the two
paths converge — which is what the 1.14× at the top of the table shows.

!!! note "These numbers moved when the capacity bug was fixed"
    The rebuilt path's capacity used to be sized by
    `n_atoms × shell_capacity(...)`. That double-counts — `shell_capacity`
    returns the neighbours of *one* atom, and an unordered pair list holds each
    pair once — and it came from a cutoff-sphere estimate that ignores the box,
    so a 300-atom system that can hold at most 44 550 pairs was allocated
    434 400 slots. Padding is not free: masked slots are still evaluated.

    Fixing both moved the crossover from ~7 000 to ~4 800 atoms. An earlier
    revision of this page said the padding would not move it; that was measured
    with only the box bound applied, before the double-count was removed.

The CPU crossover is earlier still (~2 600 atoms) because the O(N²) work has no
parallelism to hide behind. Set `static_pairs: false` for large CPU runs.

##### What the static list cannot get wrong

Beyond speed, it removes two failure modes that the rebuilt path has:

- **A list built below `ctofnb` is silently wrong.** Against the complete list
  as reference, a 2 625-atom box gives −33 meV/atom at a 9 Å build cutoff,
  +27 at 10 Å and −5 at 11 Å, reaching exact agreement only at 12 Å = `ctofnb`.
  The static list has no cutoff to get wrong.
- **A rebuilt list is only correct at the configuration it was built at.** After
  1 Å RMS drift it is off by 3.2 meV/atom and 0.45 eV/Å in the worst force.
  At the per-block drift a 20-step block actually produces this is negligible
  (~0.1 Å → 6 × 10⁻³ eV/Å), but the margin is a function of the block size, and
  a 10 ps equilibration block is what caused two failures in the Menshutkin
  campaign — see [the campaign record](examples/menshutkin-campaign-record.md).
  The static list cannot go stale.

#### Pre-equilibration and window chaining

A Packmol box has the right density from the first step but no liquid structure,
and the first solvation shell around a charged solute takes tens to hundreds of
picoseconds to form. Windows started from it sample a solvent that is still
relaxing, which under-solvates the solute and biases the barrier high.

| Field | Default | What it does |
|---|---|---|
| `pre_equilibrate_ps` | `0.0` | Picoseconds of NVT on the packed box *before any window runs*, restrained at the schedule point nearest the base geometry. Cached to `<output_dir>/../equilibrated_<n>atoms_seed<s>_<ps>ps.npz`, so the cost is paid once per box and survives `--resume`. |
| `heat_stages` | `0` | Stages over which pre-equilibration raises T from `heat_start_fraction · temperature_K` to the target. A packed box carries no kinetic energy, so assigning full-target velocities in one step is a thermal shock. |
| `heat_start_fraction` | `0.2` | Where that ramp starts. Must be in (0, 1] — a 0 K start is the shock the staging exists to prevent. |
| `seed_from_previous_window` | `false` | Seed each window from the previous window's final frame instead of re-stretching from the one base structure, so a relaxed solvation shell travels along the ladder. The solute is still stretched to the new centre; only the solvent is inherited. |

Chaining is applied **only when the full ladder runs in order**. Under
`--resume` or `--windows` the set of windows that happen to be missing would
decide what "previous" means, so a chained window would sample a different
ensemble on a resumed run than on a fresh one; the sampler disables chaining in
that case and says so. A tell-tale that chaining is off is an identical
`seed_max|F|` on every window.

#### Reaction-channel restraints

A bias on `ξ = r(A,B) − r(C,D)` fixes the *difference* of two distances and
leaves their *sum* free. `FlatBottomWall` on the sum and `BondRetentionWall` on
`min(r)` each bound that direction, but with a **constant** bound — and a
constant bound is a box, while the reaction path is a line inside it.

`ReactionChannelRestraint` bounds the deviation from the reference path
*evaluated at the configuration's own ξ*:

\[
W = \tfrac12 k \max\bigl(|s(R) - s_\mathrm{ref}(\xi(R))| - \mathrm{tol},\ 0\bigr)^2
\]

with `s_ref` a linear interpolation through `(xi_grid, sum_grid)` — normally the
median sum per ξ bin of the training set.

Two properties matter for the analysis. It is flat-bottomed within `tol`, so it
costs nothing on the path itself; and because `s_ref` is evaluated at the
configuration's own ξ rather than at a window's target, it is one fixed function
of the coordinates, identical in every window, and **cancels in the MBAR
reduced-potential differences** exactly as the other walls do. A restraint aimed
at each window's ξ₀ would not cancel and would force a two-dimensional MBAR.

From the CLI, `--wall-channel A,B,C,D,JSON,GRIDKEY,TOL[,K]` (repeatable). The
JSON supplies `xi_grid` plus the grid named by `GRIDKEY`: `sum_grid` restrains
`r(A,B) + r(C,D)`, `cn_grid` restrains `r(C,D)` alone. In YAML, any `walls`
mapping carrying an `xi_grid` key resolves to this restraint. Atom references
must be integer indices — unlike the other wall kinds, channel specs are not
run through hybrid atom-name binding.

#### Paired 2D window centres

`targets_xy` takes explicit `((x0, y0), (x1, y1), …)` centres for a 2D umbrella
instead of the full `targets_A × targets_y_A` grid, giving one window per pair
rather than a lattice (`uses_paired_windows` reports which mode is active).

A reaction is a one-dimensional path through a two-dimensional space, so the
grid is mostly wasted: for a methyl transfer, windows with both r(C–Cl) and
r(C–N) large are a dissociated methyl and windows with both small are a
five-coordinate carbon. Neither is on the path, neither is in the training data,
and sampling them puts the model far outside the region it was fitted on. The
natural source of centres is the reaction path the model was trained on.

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

### Dihedral collective variables

```yaml
cv_x:
  kind: dihedral
  atoms: [14, 16, 18, 24]   # 0-based; φ for TRIA central ALA
seed_mode: frames           # required — stretch seeding is distance-only
k_ev_A2: 0.05               # eV/deg²
xi_min: -180
xi_max: 180
n_windows: 13
```

Worked TRIA path (gas scan → seeds → sample → MBAR → plot):
[Peptide φ/ψ scan → umbrella PMF](examples/tria-phi-psi-scan.md).

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
