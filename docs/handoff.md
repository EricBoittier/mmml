# Cross-backend MD handoff (PyCHARMM ↔ JAX-MD ↔ ASE)

Campaign jobs chain via `depends_on`. Each completed stage writes
`output_dir/handoff/state.npz` (and optionally `handoff/final.res`) containing:

| Field | Description |
|-------|-------------|
| `positions` | Cartesian coordinates (Å) |
| `velocities` | Atomic velocities (Å/ps) when available |
| `cell` | 3×3 periodic cell from PBC equilibration |
| `metadata` | Prior backend, stages, source path |

See [`MdHandoffState`](../mmml/cli/run/md_handoff.py).

## Default policy on handoff

When continuing from a handoff (`depends_on` or `--continue-from`):

1. **Pre-minimization is skipped** unless `handoff_pre_minimize: true`.
2. **Overlap rescue** at placement is skipped (geometry is assumed equilibrated).
3. **Cutoffs** must match the predecessor job (`defaults:` in campaign YAML).
4. **Box:** handoff `cell` wins over campaign `box_size`; a warning is printed if both differ.
5. **Velocities:** used when present and `continue_velocities: true` (default), unless pre-min ran.

## When to enable pre-minimization

| YAML flag | Use when |
|-----------|----------|
| `handoff_pre_minimize: true` | Always relax on the MMML surface after PyCHARMM equil |
| `handoff_quality_gate: true` | Only pre-min if initial MMML \|F\| > `handoff_quality_fmax_eVA` (default 1.0 eV/Å) |

```yaml
defaults:
  mm_switch_on: 9.0
  mm_switch_width: 1.5
  ml_switch_width: 1.0

runs:
  pycharmm_equil:
    backend: pycharmm
    setup: pbc_npt
    md_stages: "mini,heat,equi"

  jaxmd_prod:
    backend: jaxmd
    setup: pbc_nvt
    depends_on: pycharmm_equil
    handoff_quality_gate: true
    handoff_quality_fmax_eVA: 1.0
    jaxmd_minimize_steps: 500
```

## Diagnostics

JAX-MD / ASE write `handoff_policy.json` in the job output directory with box source,
velocity policy, cutoffs, and initial MMML energy/forces.

## Interpreting initial MMML energy

Positive total MMML energy (eV) is normal — the hybrid calculator is not
zero-referenced like CHARMM. High **\|F\|** (≫ 1 eV/Å) after handoff usually means
cutoff mismatch or missing pre-min on the MMML surface.

## Velocity carry-over (JAX-MD)

Handoff velocities are applied to the JAX-MD integrator momentum (`mass × v`) when:

- `continue_velocities: true` (default)
- Pre-minimization did **not** run (positions unchanged)

After pre-min, velocities are re-thermalized at `--temperature`.

Drift removal (`handoff_velocity_remove_drift: true`, default) zeros net momentum
and angular momentum before dynamics.
