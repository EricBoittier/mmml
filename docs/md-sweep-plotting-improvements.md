# MD sweep plotting improvements (2026-07-13)

Summary of the visualization/analysis pass over
`workflows/unified_backend_sweep/` and `workflows/mixed_calculator_sweep/`,
following feedback that the initial `plot_results.py` scripts used the wrong
energy-drift metric, invented their own plot style instead of the repo's,
and didn't use the existing structural-analysis module.

## 1. Energy drift: fluctuation + tendency, not a bare endpoint delta

**Problem**: both sweeps originally reported `energy_drift_ev = E[last frame]
- E[first frame]` — misleading, since it can be large from single-frame
noise in an otherwise-flat trace, or small while the trace trends steadily
between two coincidentally-close endpoints.

**Fix**: [`mmml/md/results.py`](../mmml/md/results.py) gains
`energy_drift_metrics(energies)`, returning:

- `energy_fluctuation_std_ev` — std over the whole trace (noise floor).
- `energy_trend_ev_per_frame` — slope of a linear least-squares fit vs. frame
  index (the systematic tendency).
- `energy_trend_total_ev` — the trend line's implied total change.

Wired into both `scripts/run_setting.py` (computed and written to
`status.json`), both `scripts/collect_results.py` (new CSV/markdown columns,
with a caveat note on `unified_backend_sweep`'s table that its 2-3-frame
traces make these numbers much less statistically meaningful than
`mixed_calculator_sweep`'s 100-sample ones), and both `scripts/plot_results.py`
(fitted trend line overlaid on every energy trace; summary bars now show
fluctuation/tendency instead of the bare delta). Regression-tested in
[`tests/unit/test_md_results.py`](../tests/unit/test_md_results.py).

Old `summary.csv` rows from before this change don't have the new columns;
`plot_results.py` falls back to recomputing the metrics directly from
`trajectory.npz` when the CSV lacks them, so existing result directories
don't need to be re-run to get correct plots.

## 2. Structural analysis: reused the existing module, didn't reinvent it

[`mmml/utils/plotting/trajectory_structure.py`](../mmml/utils/plotting/trajectory_structure.py)
(added 2026-07-11) already computes bond-length/angle/dihedral distributions
and periodic element-pair RDFs from `Sequence[ase.Atoms]`. New:

- **`JaxmdDriver`/`RigidBodySampler` now save `Z` and `box`** alongside
  `positions`/`energies` in `trajectory.npz` (`mmml/md/drivers/jaxmd.py`,
  `mmml/md/samplers/rigid.py`) — needed to reconstruct `ase.Atoms` per frame
  without re-running any dynamics (topology is static per run).
- **New script**: [`workflows/mixed_calculator_sweep/scripts/plot_structure.py`](../workflows/mixed_calculator_sweep/scripts/plot_structure.py) —
  loads one setting's `trajectory.npz`, reconstructs `Atoms` per frame
  (rebuilding topology on the fly for older `trajectory.npz` files that
  predate the `Z`/`box` fix — a fast, dynamics-free system rebuild, not a
  re-run), and calls `element_pair_rdfs` / `internal_coordinate_distributions`,
  plotted via the existing `scripts/plot_trajectory_structure.py`'s
  `plot_rdfs`/`plot_internal` (reused directly, not re-implemented).
- **Fixed a real crash** in `scripts/plot_trajectory_structure.py::plot_internal`:
  it called `np.concatenate` on an empty coordinate group unconditionally,
  which crashed for any system with zero dihedrals (e.g. a pure water box —
  water has no 4-atom dihedral chains). Now shows "none found" for that panel
  instead.

Validated on real completed runs: `water_baseline` (TIP3 water) shows the
correct O-H bond length (~0.96 Å) and H-O-H angle (~104.5°) with zero
dihedrals; `mixed_baseline` (trialanine + water) shows a full peptide
bond/angle/dihedral distribution.

## 3. Plot style: adopted the house module instead of inventing one

**Problem**: the initial `plot_results.py` scripts used ad-hoc hex color
lists and no font/dpi conventions, not matching how the rest of the repo
plots.

**Fix**: [`mmml/utils/plotting/styles.py`](../mmml/utils/plotting/styles.py) —
already the repo's style module (`PlotStyle` presets: `nature`, `google`,
`xmgrace`, `tron`, `mpl_classic`) — is now used via `apply_plot_style("nature")`
+ `comparison_colors(style, n)` in both workflows' `plot_results.py`. New doc,
[`docs/plotting-style-guide.md`](plotting-style-guide.md), documents:

- Which preset to use when (`nature` for scientific analysis figures,
  `google` for training curves, etc.).
- Concrete DPI/figure-size/gridline/legend conventions taken from real
  precedent (`scripts/plot_mlpot_settings.py`, `scripts/plot_trajectory_structure.py`),
  not invented.
- The `energy_drift_metrics` guidance from §1.
- How to use `trajectory_structure.py` for structural analysis from §2.

## 4. Bonus finding: the recurring "packmol not found" failures had a real, fixable cause

Not part of the original ask, but found while re-syncing the cluster during
this pass: `mmml/generate/packmol/packmol` (the platform-specific compiled
binary) was **tracked in git** despite already being listed in `.gitignore`
— it had been committed before that rule was added, so the gitignore entry
never took effect. Every `git pull`/`merge` on pc-studix was silently
overwriting a freshly Linux-rebuilt binary with the tracked macOS one, which
is why "packmol not found for this platform" recurred three separate times
across the `mixed_calculator_sweep`/`unified_backend_sweep` validation runs
this session, each time misattributed to an unexplained one-off clobber.
Fixed with `git rm --cached` (removes from tracking, leaves each platform's
local rebuild untouched) — this should not recur.

## What's still open

- `unified_backend_sweep`'s summary table's fluctuation/trend numbers are on
  a 2-3 frame basis (its short smoke-test step counts) — much coarser than
  `mixed_calculator_sweep`'s 100-sample traces; noted in its `summary.md`
  header, not fixed (would need longer runs, see
  `workflows/unified_backend_sweep/README.md` "Note on NPT" for the same
  step-count tension already documented there).
- `plot_structure.py` currently only demonstrated on `water_baseline` and
  `mixed_baseline`; running it over every completed setting (and wiring it
  into the Snakemake `collect` rule as an optional post-processing step)
  is a natural next step if per-setting structural plots are wanted for the
  full sweep rather than on demand.
- The rebuilt-topology fallback path in `plot_structure.py` (for
  `trajectory.npz` files predating the `Z`/`box` fix) requires PyCHARMM and
  the workflow's `config.yaml`; it will not work standalone on a
  `trajectory.npz` copied without that context — only relevant for the
  already-completed runs from earlier in this session, not new ones (which
  save `Z`/`box` directly).
