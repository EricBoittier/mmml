# Plotting style guide

**There is already a house style module — use it instead of ad-hoc hex lists.**
This doc exists because a recent pass (`workflows/*/scripts/plot_results.py`)
initially invented its own colors/fonts instead of using it; this is the
correction, so it doesn't happen again.

## The one module to import

[`mmml/utils/plotting/styles.py`](../mmml/utils/plotting/styles.py) defines the
house `PlotStyle` presets and is the single source of truth for fonts, colors,
and line weights. Use it at the top of any new plotting script:

```python
from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

style = apply_plot_style("nature")     # sets matplotlib rcParams
colors = comparison_colors(style, n=len(settings))  # fixed categorical order
```

- `apply_plot_style(name)` calls `plt.rcParams.update(...)` for the named
  preset and returns the resolved `PlotStyle`. Call it once per script, before
  creating any figure.
- `comparison_colors(style, n)` returns `n` colors from the style's
  `comparison_palette`, cycling only if you genuinely need more series than
  the palette has (rare — reconsider the chart if you do). **Assign each
  series a color once, from this call, and keep it across every panel/rerun**
  — never re-derive colors ad hoc per plot, or the same setting ends up a
  different color in two figures.

### Which preset

| Preset | When |
|---|---|
| `"icml"` | **Default for sweep/analysis figures** (energy traces, RDFs, bond/angle histograms, anything going in a writeup) — clean sans-serif, muted "seaborn deep"-style categorical colors (`#4C72B0`/`#DD8452`/`#55A868`/...), moderate type (`axes.labelsize=13`), no top/right spine. The closest preset to a modern ML-conference (ICML/NeurIPS) figure — less soft/rounded than `"google"`, less serif/journal than the `editorial_*` family. Pairs with `legend_outside()` (see below). Used by `workflows/*/scripts/plot_results.py` and `plot_structure.py`. |
| `"editorial_dejavu_sans"` / `"editorial_dejavu_serif"` / `"editorial_stix"` / `"editorial_cm"` | An alternative "read from across the room" family (e.g. a talk slide) — large type (`axes.labelsize=15`, `axes.titlesize=17` bold), thick lines (`lines.linewidth=2.8`), LaTeX-style math via `mathtext.fontset`, no top/right spine, faint dotted grid. The four variants share this axis/spacing treatment and differ only in typeface — see [`docs/plot-style-gallery.md`](plot-style-gallery.md) for a rendered comparison. |
| `"nature"` (alias `"pub"`/`"publication"`) | Compact journal-figure alternative when `"icml"`'s or the editorial family's type doesn't fit a dense multi-panel grid — sans-serif, `axes.labelsize=9`. |
| `"google"` (module default) | Training-curve dashboards (loss/lr curves) — this is what most existing training-plot scripts assume implicitly. |
| `"mpl_classic"` | Quick throwaway diagnostics where house branding doesn't matter. |
| `"xmgrace"` / `"tron"` | Not for new work — legacy/novelty presets. |

Run `mmml.utils.plotting.styles.list_plot_styles()` to see all registered
names/aliases.

**On the name**: these are *not* "the Tufte style" — Tufte described a set of
principles (data-ink ratio, redundant encoding instead of decoration, small
multiples), not one font or palette, so no single preset should claim his
name. `"tufte"` still resolves as a backward-compat alias (→
`editorial_stix`) since an earlier pass mistakenly registered it that way,
but new code should name the `editorial_*` variant directly.

## Semantic color, not palette index

**A color should mean the same thing every time it appears, not just be "the
next one in the cycle."** `comparison_colors(style, n)` is fine for truly
interchangeable series (e.g. random seeds of the same setting), but once
your series fall into meaningful groups, assign color by group membership —
fixed and hand-picked, not generated:

```python
# Good: color means something and is stable across every figure.
_SYSTEM_COLORS = {"water_box": "#1A5276", "peptide_water": "#943126"}
color = _SYSTEM_COLORS[row["system"]]

# Bad: the 3rd setting in whatever order happened to sort this run.
color = comparison_colors(style, n)[i]
```

Concrete house examples (both from `workflows/*/scripts/plot_results.py`):

- **`mixed_calculator_sweep`**: color = system (`water_box` = deep blue "MM
  only", `peptide_water` = brick red "mixed ML/MM"); a distinct **marker
  shape** per individual setting shares that color rather than getting its
  own hue, so identity is still readable (redundant color+shape coding)
  without diluting what the color itself means.
- **`unified_backend_sweep`**: color = *kind of physics* the backend
  represents — `jaxmd_min` (deterministic minimization) neutral gray,
  `jaxmd_nve` (energy-conserving reference) deep blue, `jaxmd_nvt`
  (thermostatted — deliberately exchanges heat) forest green, `jaxmd_npt`
  (documented deterministic failure on this cluster) brick red, `rigid_mc`
  (stochastic sampler) muted purple.
- **Element coloring** (`scripts/plot_trajectory_structure.py`'s `plot_rdfs`)
  already does this correctly via `ase.data.colors.jmol_colors` — an O atom
  is the same red everywhere because "oxygen" is the semantic category, not
  because oxygen happened to be plotted first.
- **Coordinate-type coloring** (`plot_internal` in the same file): bonds,
  angles, and dihedrals are three different physical quantities (different
  stiffness, different units) and each gets its own fixed color
  (`_COORDINATE_TYPE_COLORS`) — they should never all render as the same
  default blue just because no one assigned them a color explicitly.

When in doubt: if you can name *why* a series has the color it has (its
system, its physics, its pass/fail status) in one short phrase, it's
semantic. If the only answer is "it's next in the list," fix it.

## Legends live outside the plot

**A legend never overlaps the data.** Use
[`mmml.utils.plotting.styles.legend_outside(target, side="auto", **kwargs)`](../mmml/utils/plotting/styles.py)
instead of `axis.legend(loc="best", ...)`.

**Which side depends on the figure's longest dimension, not a fixed default:**

- **Single-column, multi-row (stacked) figures** — e.g. 2-3 subplots stacked
  vertically, the figure is *taller than wide* — put the legend **below the
  whole figure**, not to the side. Pass the `Figure` (not an `Axes`) so one
  combined legend covers every panel instead of stacking several smaller
  ones:
  ```python
  legend_outside(fig, handles=all_handles, labels=all_labels, side="bottom", fontsize=12)
  ```
  A bottom legend wraps into a small grid (`ncol` auto-picked, ≤4) rather
  than one long row, so it stays roughly as wide as the figure itself.
- **Multi-column figures** — e.g. two panels side by side — **each column
  gets its own legend on its own outer edge**: the left panel's legend
  attaches further left, the right panel's further right. Never stack both
  panels' legends on one side.
  ```python
  legend_outside(ax_left, side="left", fontsize=10)
  legend_outside(ax_right, side="right", fontsize=10)
  ```
- **`side="auto"`** (default) picks between right/bottom by comparing the
  *figure's* width to height (`fig.get_size_inches()`) — use it for a
  single-axes figure where there's no column/row structure to reason about;
  set the side explicitly for anything with subplots, per the two cases
  above.

Because a legend is outside, **it's free to grow long** — a 12-setting
sweep's legend reads like a small table (marker/color/seed-symbol → setting
name) rather than needing to be trimmed to fit inside the plot. Don't
economize on legend entries to make them fit inside the axes; if the legend
needs to be big, let it be big outside instead.

**Verify there's no overlap by actually looking at the rendered PNG** — a
bottom legend can still collide with rotated x-tick labels, and a
side-by-side figure's two titles can run into each other if the panels are
too narrow for the font size. `bbox_inches="tight"` (house convention on
every `savefig`) prevents *clipping*, but not two artists rendering on top of
each other; widen the figure (`figsize`) or add `fig.subplots_adjust(wspace=...)`
if titles or labels are cramped, then re-render and look again.

## Symbols over small text

When a repeated label would otherwise shrink to fit (e.g. "(seed 3)" tacked
onto a dozen x-tick labels), replace it with a symbol instead of making the
text smaller. House convention:
[`mmml.utils.plotting.styles.seed_symbol(seed)`](../mmml/utils/plotting/styles.py)
returns a filled-dot count — `seed_symbol(1) == "●"`, `seed_symbol(3) ==
"●●●"` — instead of the text `"(seed 3)"`. (Unicode die faces U+2680-2685
were tried first and rejected: they render as generic missing-glyph boxes on
common sans-serif fonts, including the ones this cluster and this house
style actually use — verify any "clever" glyph choice actually renders
before committing to it, don't assume Unicode support.) Reuse `seed_symbol`
rather than inventing another per-script seed convention.

More generally: a marker *shape* (see "Redundant coding" below) or a fixed
color is preferable to a text tag wherever one is already semantically
assigned — only fall back to text for identifiers that don't have a natural
symbol (most setting names).

## Overlaid semi-transparent bars, not more panels

When two bar-chart metrics share units and scale (e.g. energy fluctuation
$\sigma$ and trend $|dE/dn|$, both eV-scale) don't give them two separate
stacked panels — overlay them as semi-transparent bars in **one** panel
instead, distinguished by fill style (solid vs. `hatch="///"`) at different
alpha:

```python
ax.bar(x, fluctuation, alpha=0.9, width=0.6, label="fluctuation")
ax.bar(x, trend, alpha=0.45, hatch="///", width=0.6, label="tendency")
```

This reads their relative size directly (one glance, not a vertical scan
between two panels) and keeps a multi-panel sweep figure shorter — which
also means less pressure on the "legend below" rule above, since fewer
stacked rows makes the figure less tall relative to its width. Reserve a
separate panel for a metric with genuinely different units/scale (e.g.
wall-clock seconds) that wouldn't overlay meaningfully.

## Figure conventions (from real precedent, not invented)

- **DPI**: `150` for a quick per-run diagnostic plot regenerated often;
  `200` for `editorial_*`-styled sweep figures (`workflows/*/scripts/plot_results.py`);
  `300` for anything meant to be read very closely or reused in a writeup
  (RDFs, bond/angle/dihedral histograms — see
  `scripts/plot_trajectory_structure.py`'s `plot_rdfs`/`plot_internal`, both
  `dpi=300`).
- **Uncertainty as shading, not just error bars**: `ax.fill_between(x, trend
  - sigma, trend + sigma, alpha=0.1-0.15)` behind a line reads faster than a
  legend annotation — see `plot_energy_traces` in
  `workflows/mixed_calculator_sweep/scripts/plot_results.py`.
- **Redundant coding for multi-category series**: color for the group, a
  distinct marker shape (`o`, `s`, `^`, `D`, `v`, ...) for the individual
  series within it — never rely on color alone to carry both. See "Semantic
  color, not palette index" below.
- **Always**: `fig.savefig(path, dpi=..., bbox_inches="tight")` then
  `plt.close(fig)` — every plotting function in this repo returns the output
  `Path` and closes its own figure; never leave figures open across calls.
- **Element/atom coloring**: use `ase.data.colors.jmol_colors` +
  `ase.data.atomic_numbers`, not an invented per-element palette — see
  `plot_rdfs` in `scripts/plot_trajectory_structure.py`.
- **Multi-panel structural plots**: one row of 3 panels (bonds / angles /
  dihedrals) at `figsize=(15, 4.5)` is the established layout
  (`plot_internal` in `scripts/plot_trajectory_structure.py`) — reuse it
  rather than inventing a new grid.
- **Grid**: `alpha=0.18-0.3` on gridlines (recessive, never full-strength) —
  consistent across every plotting script surveyed.
- **Legends**: only when there are ≥2 series; `fontsize=7-9`; a single-series
  panel needs no legend box (the axis title/label already names it) — this
  matches the dataviz skill's non-negotiables and is not repo-specific.
- **One y-axis per panel.** Never a dual-axis (twinx) chart — split into two
  stacked panels (`plt.subplots(2, 1, ...)`) instead, as
  `workflows/*/scripts/plot_results.py` do for energy vs. elapsed time.

## Energy drift: fluctuation + tendency, not a bare endpoint delta

`energy_drift_ev = E[last frame] - E[first frame]` (the original metric in
both sweep workflows) is misleading: it can be large purely from single-frame
noise in an otherwise-flat trace, or small while the trace trends steadily in
one direction between two coincidentally-close endpoints. Use
[`mmml.md.results.energy_drift_metrics`](../mmml/md/results.py) instead, which
reports:

- `energy_fluctuation_std_ev` — std over the *whole* trace (the noise floor).
- `energy_trend_ev_per_frame` — slope of a linear least-squares fit vs. frame
  index (the systematic tendency).
- `energy_trend_total_ev` — the trend line's implied total change over the
  run (`slope * (n_frames - 1)`), comparable in scale to the old endpoint
  delta but reflecting the fit rather than one noisy pair of samples.

Both `workflows/unified_backend_sweep/scripts/run_setting.py` and
`workflows/mixed_calculator_sweep/scripts/run_setting.py` compute all three
alongside the original fields (kept for backward compatibility with existing
`summary.csv` consumers); `scripts/plot_results.py` in each workflow plots the
fitted trend line over the raw trace rather than just annotating two
endpoints.

## Structural analysis (bonds/angles/dihedrals/RDF)

Don't re-derive these — [`mmml/utils/plotting/trajectory_structure.py`](../mmml/utils/plotting/trajectory_structure.py)
already has them, operating on `Sequence[ase.Atoms]`:

- `element_pair_rdfs(frames, r_max=8.0, bins=160)` — periodic RDF per element
  pair (minimum-image, orthorhombic cells only).
- `internal_coordinate_distributions(frames, indices)` — bonds/angles/proper
  dihedrals (inferred from covalent radii) as an `InternalCoordinates`
  dataclass of per-labeled-coordinate arrays.
- `infer_bonds`, `water_tetrahedrality`, `hydrogen_bond_analysis`,
  `radius_of_gyration_and_diffusion` — see the module docstring for the rest.

These take ASE `Atoms`, not raw position arrays — `JaxmdDriver` and
`RigidBodySampler` now save `Z` (and `box`, for fixed-box runs; `boxes` for
NPT) alongside `positions`/`energies` in `trajectory.npz` specifically so a
downstream script can reconstruct `Atoms` per frame without re-running the
simulation:

```python
import numpy as np
from ase import Atoms

data = np.load("trajectory.npz")
frames = [
    Atoms(numbers=data["Z"], positions=pos, cell=data["box"], pbc=True)
    for pos in data["positions"]
]
```

`scripts/plot_trajectory_structure.py` is the reference CLI wrapper —
`plot_rdfs`, `plot_internal`, `plot_hydrogen_bond_timeseries`, etc. — reuse
its plotting functions directly rather than re-plotting the same arrays with
different styling.

## Rendering ASE `Atoms` (molecule structure drawings)

For drawing actual molecular geometry (not a data curve) — e.g. a dimer
snapshot alongside an energy-scan plot — reuse
[`scripts/plot_utils.py::render_dimer_atoms`](../scripts/plot_utils.py),
the good precedent already in the repo (used by the SpookyNet dimer-scan
figures in `scripts/plot_1d_slices_by_offset.py` and `scripts/plot_2d_pes.py`):

- Atom size from `ase.data.covalent_radii` (scaled by `radii_scale`), color
  from `ase.data.colors.jmol_colors` — the same "semantic color" principle
  as the RDF element coloring above, applied to the atoms themselves.
- Camera angle as an `ase.io.utils.rotate`-style string (e.g.
  `"15x,-20y,0z"`), projected to 2D — not a fixed top-down view.
- Bonds drawn as lines only within a fragment (same-monomer atom pairs
  closer than `bond_cutoff_scale * (sum of covalent radii)`), so
  intermolecular contacts in a dimer aren't mistaken for bonds.
- Atoms drawn back-to-front (sorted by projected depth) with per-atom alpha
  as a depth cue, so overlapping monomers stay legible instead of a flat
  jumble.
- Optional force-arrow overlay via `ax.quiver`, projected through the same
  rotation matrix as the atoms.

`plot_2d_pes.py`'s `_plot_atoms_filmstrip` (a strip of geometry snapshots
along a scan coordinate) and `plot_1d_slices_by_offset.py`'s per-slice
minimum-energy-geometry inset are the reference patterns for combining a
structure drawing with an energy curve in one figure — reuse those, don't
re-derive the atom-rendering logic for a new script.
