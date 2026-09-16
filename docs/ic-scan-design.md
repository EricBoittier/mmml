# Internal-coordinate scans (`ic-scan`)

![ACEM methyl: O–C–CC–HC1 atoms, rigid vs relaxed profiles on identical axes](images/plots/acem-methyl-scan.png)

`mmml ic-scan` scans **bonds**, **angles**, or **dihedrals** by 0-based ASE
index. Default `geometry_mode: rigid` rotates a fragment and evaluates. That
is why the ACEM methyl above is not 3-fold until you relax: the CGenFF XYZ has
unequal H–C–C angles, so 120° does not permute equivalent hydrogens.

Kinds are only `bond` | `angle` | `dihedral`. There is no user-defined CV
(linear combinations, COM distances, …). What you *can* customize is the
grid (`values:`), the moving fragment (`mask:`), and inactive-DoF
`reference:` values.

```bash
mmml ic-scan --config CONFIG.yaml --output artifacts/ic_scan/out --overwrite
```

`--prepare-only` writes geometries without energies (incompatible with
`constrained-relax`).

## Bonds

`kind: bond` takes two atoms. Default mask: the covalent fragment on the
**a1** side of a0–a1 (a1 must move). Lengths in Å, must be positive.

```yaml
dofs:
  - name: r12
    kind: bond
    atoms: [0, 1]
    start: 1.40
    stop: 1.70
    n_points: 4
scans:
  - name: bond_1d
    dofs: [r12]
```

Example: `examples/ic_scan/butane_like.yaml`.

## Angles

`kind: angle` takes three atoms **a1–a2–a3**. Default mask: fragment on the
**a3** side of a2–a3. Values in degrees.

```yaml
dofs:
  - name: theta
    kind: angle
    atoms: [0, 1, 2]     # C–C–C on the butane-like chain
    start: 90
    stop: 130
    n_points: 5
scans:
  - name: angle_1d
    dofs: [theta]
```

Same file: `examples/ic_scan/butane_like.yaml` (`angle_1d`).

## Dihedrals

`kind: dihedral` takes four atoms. ASE rotates about **a2–a3**. **a4 must be
on the a3 side** and in `mask`.

```text
 a1          a4
  \          /
   a2 ---- a3     ← rotate about this bond
```

Omit `mask` unless you have a reason: default is the covalent fragment on the
a3 side of a2–a3. PSF index order is not that fragment. If you set `mask`, it
must include a4.

```yaml
dofs:
  - name: methyl
    kind: dihedral
    atoms: [5, 1, 0, 6]   # ACEM O–C–CC–HC1
    start: -180
    stop: 180
    n_points: 37
```

| Rotor (CGenFF `make-res`) | `atoms` |
|---------------------------|---------|
| ACEM methyl `O–C–CC–HC1` | `[5, 1, 0, 6]` |
| NMA amide `CL–C–N–CR` | `[0, 4, 6, 8]` |
| NMA acetyl methyl `N–C–CL–HL1` | `[6, 4, 0, 1]` |
| NMA N-methyl `C–N–CR–HR1` | `[4, 6, 8, 9]` |

NMA ω with `mask: [9, 10, 11]` (HR* only, no CR) is broken — a4 never moves.

## Relaxed scans

`geometry_mode: constrained-relax` holds **active** DoFs with `FixInternals`
and FIRE/BFGS-minimizes the rest. A 1D methyl job does not freeze the amide.
Needs `evaluate: energy` (no `--prepare-only`).

```yaml
evaluate: energy
geometry_mode: constrained-relax
relax_fmax_ev_A: 0.05
relax_steps: 200
relax_maxstep_A: 0.03
relax_optimizer: fire    # or bfgs if FixInternals steps fail
relax_chain: true
scans:
  - name: methyl_1d
    dofs: [methyl]
```

```bash
mmml ic-scan \
  --config examples/ic_scan/acem_dihedrals_relaxed.yaml \
  --output artifacts/ic_scan/acem_xtb_relaxed \
  --overwrite
```

Pass: `methyl_1d` repeats every 120° with three equal wells. Peptide φ/ψ with
CHARMM `CONS DIHE` is still `scripts/scan_trialanine_phi_psi_pes.py`.

## 2D scans

A `scans` job with two (or more) DoFs is the cartesian product. 1D jobs in the
same YAML still run as 1D. `scan_mode: product` (default when `scans:` is
omitted) is one N-D grid over every DoF; `individual` is a 1D sweep per DoF
with the others held at the reference geometry.

```yaml
# examples/ic_scan/nma_omega_methyl_2d.yaml
scans:
  - name: omega_1d
    dofs: [omega]
  - name: n_methyl_1d
    dofs: [n_methyl]
  - name: omega_methyl_2d
    dofs: [omega, n_methyl]   # 13 × 13
```

```yaml
# examples/ic_scan/butane_like.yaml — mixed kinds
scans:
  - name: bond_dihedral_2d
    dofs: [r12, phi]
```

Coupled internals are re-applied until they match. If they fight, preparation
fails — fix atom order / mask. 1D plots go to `energy_*.png`; 2D is
`data.csv` + `trajectory.traj`.

```bash
mmml ic-scan --config examples/ic_scan/nma_omega_methyl_2d.yaml \
  --prepare-only --output ic_scan/omega_methyl_2d --overwrite
ase gui ic_scan/omega_methyl_2d/trajectory.traj
```

## Custom grids

Not a new `kind`. Use an explicit list instead of `start` / `stop` /
`n_points` (do not mix the two). Units are still Å or degrees.

```yaml
dofs:
  - name: r12
    kind: bond
    atoms: [0, 1]
    values: [1.40, 1.45, 1.50, 1.70]    # irregular spacing
  - name: phi
    kind: dihedral
    atoms: [0, 1, 2, 3]
    values: [-180, -60, 0, 60, 180]     # skip the rest of the circle
```

Optional:

```yaml
reference:
  r12: 1.54          # inactive DoFs in 1D / subset scans (else measured)
  phi: 180.0
dofs:
  - name: omega
    kind: dihedral
    atoms: [0, 4, 6, 8]
    mask: [6, 7, 8, 9, 10, 11]   # must include a4; omit to use topology
    values: [0, 180]
```

## Output

| File | |
|------|-|
| `manifest.json` | counts, resolved config, checksums |
| `data.csv` | coordinates, `status`, energies |
| `trajectory.extxyz` / `trajectory.traj` | frames (`ase gui …`) |
| `energy_*.png` | 1D plots when `evaluate: energy` |

```python
from mmml.ic_scan import IcScanConfig, run_ic_scan

config = IcScanConfig.from_dict(yaml.safe_load(path.read_text()))
result = run_ic_scan(config)
result.write("artifacts/ic_scan_out")
```

Examples: `examples/ic_scan/butane_like.yaml`, `nma_methyl.yaml`,
`nma_omega_methyl_2d.yaml`, `acem_dihedrals.yaml`,
`acem_dihedrals_relaxed.yaml`. Related: [`dimer-scan`](cli/commands/dimer-scan.md),
[NMA tutorial](examples/nma-workflow.md).
