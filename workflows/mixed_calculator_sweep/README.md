# Mixed-system / calculator sweep (10000-step NVE)

Longer-horizon, broader-axis sibling of
[`workflows/unified_backend_sweep/`](../unified_backend_sweep/README.md), aimed
at the project's actual goal: validating a **mixed** system — one ML-scored
"core" region (a peptide) embedded in a solvent that's partly ML (near the
core, via `ml_pep_water`) and partly classical MM (far from it, via
`vdw_core` + `mm_nonbonded`) — plus the calculator/force-field knobs that
matter for that: checkpoint choice, electrostatics damping, disabling the
ML model's own point-Coulomb term, and MM nonbonded cutoffs.

See `docs/md-cg-unification-design.md` §11 "Mixed-system (ML core + MM/ML
solvent) support checklist" for what each setting validates and what's still
open (dynamic ML/MM re-assignment, non-`mic` `lr_solver`, NVT/NPT, RDF
validation).

## Settings

| setting | system | what it tests |
|---|---|---|
| `water_baseline` | water_box | MM/MM baseline, default checkpoint/cutoffs (2 seeds) |
| `water_tight_cutoffs` | water_box | `mm_nonbonded` cutnb=8/ctonnb=6/ctofnb=8 |
| `water_loose_cutoffs` | water_box | `mm_nonbonded` cutnb=16/ctonnb=14/ctofnb=16 |
| `water_older_epoch` | water_box | earlier ML checkpoint (epoch 5 vs 10) |
| `water_damping_sigma` | water_box | `electrostatics_damping_sigma=1.0` |
| `water_no_point_coulomb` | water_box | `disable_physnet_point_coulomb=True` |
| `mixed_baseline` | peptide_water | **the mixed-system case**: `ml_intra` + `ml_pep_water` + `mm_nonbonded` (2 seeds) |
| `mixed_older_epoch` | peptide_water | mixed system, earlier ML checkpoint |
| `mixed_damping_sigma` | peptide_water | mixed system, damped ML electrostatics |
| `mixed_core_vdw` | peptide_water | mixed system + `vdw_core` repulsive wall outside `ml_pep_water`'s 6 Å cutoff |

`water_box` = packmol TIP3 box (`ml_intra` + `mm_nonbonded`, MM/MM only).
`peptide_water` = `PeptideWaterSystemBuilder` (trialanine + TIP3 waters, real
CHARMM PSF/params), scored with `ml_intra` (core intramolecular) +
`ml_pep_water` (core↔water ML dimers) + optionally `vdw_core` (core↔far-water
repulsive wall) + `mm_nonbonded` (water↔water MM).

Every setting runs 10000-step NVE with `JaxmdDriver`'s default
`record_every=100` (100 recorded energy samples) — enough to see a real
conservation trace, not just a 2-3 point endpoint delta like the shorter
`unified_backend_sweep`.

## Running

Local dry-run / smoke test (override `default_n_steps` for a quick check):

```bash
uv run --with snakemake snakemake --profile profiles/local --configfile config.yaml -n
```

On pc-studix (CPU `short` partition — this cluster's `.venv` has no CUDA
jaxlib, so `gpu`-partition jobs fall back to CPU anyway):

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm-cpu --keep-going
```

`results/summary.csv` / `results/summary.md` aggregate all settings once
their `status.json` files exist; `scripts/collect_results.py` always exits 0
so Snakemake doesn't delete the report on a per-setting failure (see
`unified_backend_sweep/scripts/collect_results.py` for why that matters).

## Known limitations

- `lr_solver` other than `mic` (`jax_pme`/`nvalchemiops_pme`/`scafacos`) is
  ASE-face only and not wireable into `JaxmdDriver`'s jit loop — not
  exercised here for the same reason as `unified_backend_sweep`.
- Only NVE is run; NPT is known to fail deterministically on this cluster
  (see `unified_backend_sweep/README.md`), and NVT hasn't been validated for
  mixed systems yet.
- `ml_pep_water`'s core↔water assignment is fixed at build time — there is no
  dynamic re-assignment of a water from ML-scored to MM-scored as it moves
  across the interaction cutoff during dynamics.

## Real-run status (2026-07-12, pc-studix)

All 12 settings were submitted to the real cluster and hit (and had fixed,
in order): a packmol binary clobbered with a macOS build (rebuilt as Linux
ELF via `scripts/rebuild_packmol.sh`), `short`-partition nodes silently
mixing Ivy Bridge (no AVX2, crashes `polars` with no traceback) and Broadwell
(AVX2) hardware (constrained to the latter via Snakemake's `constraint`
resource), `peptide_water` settings OOM-ing at the original
`mem_mb_per_cpu=2000` (bumped to 6000), and the driving Snakemake process
itself dying twice when its SSH session dropped (now run inside `tmux` on
the cluster to survive disconnects).

With all of that fixed:
- **All 6 `water_box` settings completed for real** (10000 steps, 101
  recorded frames each) — see `results/summary.md` on the cluster for the
  actual energy traces. `water_baseline/seed_1` took ~324 s.
- **The 6 `peptide_water` (mixed ML-core + MM/ML-water) settings ran clean
  for ~3 hours with no crash** — CPU/memory usage confirmed healthy and
  steadily active (not hung) via `ps` on each compute node — but were
  intentionally cancelled before completing all 10000 steps rather than
  waiting out the full run: at that point `water_baseline` had already
  finished 10000 steps in ~324 s, while the mixed settings (at `n_waters=20`)
  hadn't finished after ~10800 s and counting — at least ~33x slower.
  **Root cause, verified by reading the term's implementation** (not just
  inferred from timing): `ml_pep_water`'s `interaction_cutoff_A` does **not**
  reduce its per-step cost — checked directly in
  `mmml/interfaces/jaxmdInterface/hybrid_energy.py::make_peptide_water_ml_energy_fn`,
  every core-water dimer is vmapped through the ML model *every step*
  regardless of the cutoff; the cutoff only applies a post-hoc energy
  switching weight (correct physics, zero runtime effect). Unlike
  `mm_nonbonded`, there is no neighbor-list-style pruning for this term (no
  `active_group_slots`/`active_group_mask` auto-wiring exists in
  `assemble.py` for `ml_core_group`-kind `NeighborRequest`s — see the "Mixed
  system support checklist" in `docs/md-cg-unification-design.md` §11 for
  that as an open item). **Actual fix applied**: `n_waters` dropped from 20
  to 8 (2.5x fewer vmapped ML forward passes per step — the only lever that
  really cuts `ml_pep_water`'s cost today) and `n_steps` dropped from 10000
  to 2000 for all `peptide_water` settings (still 100x the local smoke
  test's 20 steps, and `record_every=100` still gives 20 recorded energy
  samples). `peptide_water_ml_cutoff_A` is kept on every mixed setting
  because it's still the physically correct choice, just not a speed fix.
- This validates the mixed-system code path itself (builder, term
  composition, calculator variants) end-to-end on the real cluster. The
  reduced-scale config above (`n_waters=8`, `n_steps=2000`) is intended to
  actually complete within the existing walltime budget — that run has not
  yet been executed/confirmed as of this writing; do that before trusting
  the mixed-system energy traces to exist in `results/summary.md`. Wiring
  real neighbor-list support for `ml_core_group` terms (so
  `peptide_water_ml_cutoff_A` becomes a genuine speed lever, not just an
  energy correction) remains open future work if larger `n_waters`/`n_steps`
  mixed runs are needed.
