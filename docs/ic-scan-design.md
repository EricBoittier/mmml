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
  - name: phi
    kind: dihedral       # bond | angle | dihedral
    atoms: [14, 16, 18, 24]
    start: -180
    stop: 180
    n_points: 36
scans:                   # optional explicit jobs
  - name: phi_1d
    dofs: [phi]
```

## Public API

```python
from mmml.ic_scan import IcScanConfig, run_ic_scan

config = IcScanConfig.from_dict(yaml.safe_load(path.read_text()))
result = run_ic_scan(config)
result.write("artifacts/ic_scan_out")
```

```bash
mmml ic-scan --config examples/ic_scan/butane_like.yaml \
  --prepare-only --output artifacts/ic_prep
```

## Relation to other tools

| Tool | Role |
|------|------|
| `mmml dimer-scan` | Rigid intermolecular separation |
| `scripts/scan_trialanine_phi_psi_pes.py` | Peptide-specific constrained φ/ψ PES |
| `mmml mode-check` | Diagnostic X–H bond scans |
| `mmml ic-scan` | General monomer IC grids for QM/ML |
| [NMA end-to-end tutorial](examples/nma-workflow.md) | make-res → methyl `ic-scan` → train → dimer → MD |

### Methyl rotations (ASE atom order)

`Atoms.set_dihedral(a1, a2, a3, a4, …)` rotates about **a2–a3** with **a4** on
the a3 side. For a methyl rotor, put the **methyl carbon as a3** and one methyl
hydrogen as **a4**, and set `mask` to all three methyl hydrogens. Example for
CGenFF NMA: acetyl `N–C–CL–HL1` → atoms `[6, 4, 0, 1]`, mask `[1, 2, 3]`.
