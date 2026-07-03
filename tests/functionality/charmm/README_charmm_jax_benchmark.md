# CHARMM vs JAX-MM energy benchmark

Compare **native PyCHARMM** ``ENER FORCE`` against **MMML JAX** loaders
(``cgenff_bonded``, ``mm_system_energy``) for supported CGENFF systems.

## Supported cases

| Case | System | Layers |
|------|--------|--------|
| `tip3_monomer` | CGENFF TIP3 from committed fixture PDB (`pycharmmETC/pdb/initial.pdb`) | bonded |
| `tip3_water_box` | 10× TIP3 in 28 Å cube (grid placement) | bonded, nonbonded, total_mm |
| `trialanine_water` | TRIA peptide + 10× TIP3 (bundled RTF) | bonded, nonbonded, total_mm |

Nonbonded and total layers use CHARMM switched VDW/Coulomb with MIC PBC
(same tolerances as ``test_trialanine_water_box_mm.py``).

## Run (CHARMM node)

```bash
export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
JAX_PLATFORMS=cpu uv run python scripts/benchmark_charmm_jax_energy.py -o /tmp/charmm_jax_bench
```

Subset:

```bash
JAX_PLATFORMS=cpu uv run python scripts/benchmark_charmm_jax_energy.py \\
  --cases tip3_monomer trialanine_water -o /tmp/charmm_jax_bench
```

Outputs:

- `benchmark.md` — per-term table (CHARMM, JAX, Δ, rel Δ) + force RMS
- `benchmark.json` — machine-readable report

Exit code `0` when all layers pass existing pytest tolerances; `1` otherwise.

## Pytest cross-checks

The benchmark reuses the same reference helpers as:

```bash
pytest tests/functionality/charmm/test_cgenff_bonded_pycharmm.py -m pycharmm -k tip3 -v
pytest tests/functionality/charmm/test_trialanine_water_box_mm.py -m pycharmm -v
```

Unit tests (no CHARMM):

```bash
pytest tests/unit/test_charmm_jax_energy_benchmark.py -q
```

## Notes

- **CMAP** on TRIA uses bundled ``par_trialanine_backbone_cmap.prm`` (alanine backbone grid with CGENFF type headers); included in bonded and total-MM comparisons.
- **Urey–Bradley** 1–3 distance terms are included in the JAX bonded kernel (parsed from CHARMM PRM angle lines).
- Long-range PME is not benchmarked here; only ``lr_solver=mic`` truncated Coulomb.

## Trajectory demo (per-frame component errors)

Compare live PyCHARMM ``ENER FORCE`` vs the JAX CGenFF clone on every frame of a
short NVT segment (or an existing DCD). No CHARMM BLOCK scripts — assumes full
CGENFF is already loaded (box build) or pass ``--reregister-cgenff`` after MLpot.

```bash
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python scripts/demo_charmm_jax_trajectory_energy.py \\
      -o /tmp/charmm_jax_traj
```

Outputs ``trajectory_report.md`` and ``trajectory_report.json`` with per-term
ΔE and force RMS for bond, angle, Urey–Bradley, torsion, improper, CMAP, VDW,
electrostatics, and totals. Use ``--synthetic`` for a fast coordinate series
without MD, or ``--dcd PATH --skip-dynamics`` to analyze a saved trajectory.

Unit tests (no CHARMM):

```bash
pytest tests/unit/test_charmm_jax_trajectory_energy.py -q
```
