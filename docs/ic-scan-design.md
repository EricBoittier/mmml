# Reproducible internal-coordinate scan design

## Status and scope

`mmml ic-scan` is the supported interface for bond / angle / dihedral scans on
an arbitrary molecular structure identified by **0-based ASE atom indices**.
It prepares geometries for QM or ML evaluation and can optionally evaluate
energies/forces in-process with the same calculator set as `dimer-scan`.

## Goals

- Config-driven DoFs: kind, atom indices, grid (`start`/`stop`/`n_points` or
  explicit `values`).
- Combine scans as:
  - `scan_mode: product` — one exhaustive N-D grid over all DoFs
  - `scan_mode: individual` — separate 1D scans per DoF (others at reference)
  - `scans:` — explicit named jobs selecting DoF subsets (1D or N-D each)
- Prepare-only mode (`evaluate: none` or `--prepare-only`) for external QM/ML.
- Calculator-neutral evaluation via the ASE calculator contract.
- Self-describing result bundles (`manifest.json`, `data.csv`, trajectories).

## Non-goals (v1)

- Constrained relaxation (`FixInternals` / CHARMM `CONS DIHE`) — planned as
  `geometry_mode: constrained-relax`.
- Periodic / MIC internal coordinates.
- Fragment interaction energies (use `dimer-scan` for that).
- Replacing the trialanine-specific φ/ψ campaign script.

## Config sketch

```yaml
structure: mol.xyz
calculator: xtb          # or physnet/spookynet/pyscf/...; omit with evaluate: none
evaluate: energy         # energy | none
scan_mode: product       # product | individual (used when scans omitted)
dofs:
  - name: omega
    kind: dihedral       # bond | angle | dihedral
    atoms: [0, 4, 6, 8]  # a1 a2 a3 a4 — see mask rules below
    # mask: omit to use covalent-topology default (recommended)
    start: -180
    stop: 180
    n_points: 36
scans:                   # optional explicit jobs
  - name: omega_1d
    dofs: [omega]
  - name: omega_methyl_2d
    dofs: [omega, n_methyl]
```

## Dihedral atom order and `mask` (read this)

ASE `Atoms.set_dihedral(a1, a2, a3, a4, angle, indices=…)`:

1. Rotates about the **central bond a2–a3**.
2. Moves only atoms listed in `indices` / `mask`.
3. The geometric target is the torsion **a1–a2–a3–a4**, so **a4 must be in the
   moving set** and must lie on the **a3** side of a2–a3.

```text
 a1          a4
  \          /
   a2 ---- a3     ← rotate about this bond
```

### Default mask (when `mask` is omitted)

MMML builds a covalent bond graph and takes every atom on the **a3 side** of
bond a2–a3 (BFS from a3, blocked at a2). That fragment always includes a4 when
the atom order is chemically correct.

**Do not** rely on “indices from a3 to n−1” — PSF order is not a topological
side of the bond.

### Explicit `mask`

If you set `mask`, it **must include a4**. Common failure for NMA amide
`CL–C–N–CR` (`atoms: [0, 4, 6, 8]`):

| Mask | Result |
|------|--------|
| omitted / `[6,7,8,9,10,11]` / `[7,8,9,10,11]` | OK |
| `[9,10,11]` (HR* only, **no CR**) | **Broken** — a4=CR never moves |
| methyl order `HL1–CL–C–N` with methyl-H mask | **Broken** — wrong a3/a4 side |

`ic-scan` validates masks up front and re-checks requested vs actual angles
after each geometry (multi-pass for N-D). Mismatches raise a clear error
instead of writing a silent wrong trajectory.

### NMA examples (CGenFF `make-res` order)

| Name | Chemically | `atoms` | Default moving fragment |
|------|------------|---------|-------------------------|
| Amide C–C–N–C | `CL–C–N–CR` | `[0, 4, 6, 8]` | N side: N, H, CR, HR* |
| Acetyl methyl | `N–C–CL–HL1` | `[6, 4, 0, 1]` | CL, HL* |
| N-methyl | `C–N–CR–HR1` | `[4, 6, 8, 9]` | CR, HR* |

Bundled configs:

- `examples/ic_scan/nma_methyl.yaml` — both methyl 1D scans
- `examples/ic_scan/nma_omega_methyl_2d.yaml` — amide ω 1D + N-methyl 1D + 2D product

### N-D / 2D scans

A `scans` job with two dihedrals builds the cartesian product. Coupled torsions
are applied in several passes until every active DoF matches. If two DoFs fight
(impossible rigid combination), preparation fails with the residual errors
listed — fix atom order / masks rather than trusting `actual_*` in the CSV.

## Public API

```python
from mmml.ic_scan import IcScanConfig, run_ic_scan

config = IcScanConfig.from_dict(yaml.safe_load(path.read_text()))
result = run_ic_scan(config)
result.write("artifacts/ic_scan_out")
```

```bash
mmml ic-scan --config examples/ic_scan/nma_omega_methyl_2d.yaml \
  --prepare-only --output artifacts/nma_omega_2d
```

## Relation to other tools

| Tool | Role |
|------|------|
| `mmml dimer-scan` | Rigid intermolecular separation |
| `scripts/scan_trialanine_phi_psi_pes.py` | Peptide-specific constrained φ/ψ PES |
| `mmml mode-check` | Diagnostic X–H bond scans |
| `mmml ic-scan` | General monomer IC grids for QM/ML |
| [NMA end-to-end tutorial](examples/nma-workflow.md) | make-res → methyl/`ω` `ic-scan` → train → dimer → MD |
