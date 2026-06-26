# Cross-backend MD handoff (PyCHARMM ↔ JAX-MD ↔ ASE)

Campaign jobs chain via `depends_on`. Each completed stage writes
`output_dir/handoff/state.npz` (and optionally `handoff/final.res`) containing:

| Field | Description |
|-------|-------------|
| `positions` | Cartesian coordinates (Å) |
| `velocities` | Atomic velocities (Å/ps) when available |
| `cell` | 3×3 periodic cell from PBC equilibration |
| `metadata` | Prior backend, stages, source path |

See [`MdHandoffState`](https://github.com/EricBoittier/mmml/blob/main/mmml/cli/run/md_handoff.py).

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

## PyCHARMM runtime guards

Topology, parameter, PSF, and coordinate reads use a relaxed `BOMLEV`/`WRNLEV`
context only while the read is active, then restore the prior levels before the
next CHARMM command. This avoids leaving `bomlev 0` pinned after benign read
warnings, which can later abort MLpot registration or dynamics setup.

Before production dynamics, MMML clears CHARMM `COMP` coordinates and scalar
components so stale comparison data is never interpreted as velocities when
`iasvel=0` or restart paths are reused. `clear_comp_for_production()` preserves
its `quiet` argument: normal calls are visible by default, and staged workflows
can opt into quiet CHARMM housekeeping when running with reduced log noise.

## Interpreting initial MMML energy

Positive total MMML energy (eV) is normal — the hybrid calculator is not
zero-referenced like CHARMM. High **\|F\|** (≫ 1 eV/Å) after handoff usually means
cutoff mismatch or missing pre-min on the MMML surface.

## Box and velocities on write (PyCHARMM)

At the end of `pycharmm_equil`, `handoff_from_charmm` resolves the cubic cell as:

1. Live CHARMM `pbound_get_size` (during/after CPT)
2. `!CRYSTAL PARAMETERS` in the last stage restart (`equi_*.res`)
3. Campaign `--box-size` fallback

Velocities are taken from the live CHARMM coordinate set, then from the last restart `!VELOCITIES` block if needed.

If an older `handoff/state.npz` lacks cell or velocities, `load_dependency_handoff` enriches from staged `.res` files in the predecessor job directory.

## PyCHARMM continuation (JAX-MD → prod)

When a later campaign job runs PyCHARMM with a handoff, `prepare_pycharmm_handoff_continuation` patches coordinates (and velocities when `continue_velocities` is true) into a restart file using, in order:

1. `--handoff-template-res`
2. `restart_path` metadata on the handoff
3. `handoff/final.res` next to `continue_from` / `state.npz`
4. Local staged `heat` / `equi` / `prod` restarts

The patched `handoff/continue_seed.res` is `READ restart` into CHARMM so the first dynamics stage uses `restart=True` (not `new` + Boltzmann assign).

## Velocity carry-over (JAX-MD)

Handoff velocities are applied to the JAX-MD integrator momentum (`mass × v`) when:

- `continue_velocities: true` (default)
- Pre-minimization did **not** run (positions unchanged)

After pre-min, velocities are re-thermalized at `--temperature`.

Drift removal (`handoff_velocity_remove_drift: true`, default) zeros net momentum
and angular momentum before dynamics.
