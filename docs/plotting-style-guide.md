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
| `"editorial_dejavu_sans"` / `"editorial_dejavu_serif"` / `"editorial_stix"` / `"editorial_cm"` | **Default family for sweep/analysis figures meant to be read, not just glanced at** (energy traces, RDFs, bond/angle histograms, anything going in a writeup) — large type (`axes.labelsize=15`, `axes.titlesize=17` bold), thick lines (`lines.linewidth=2.8`), LaTeX-style math via `mathtext.fontset` (no TeX install needed — `r"$E(t) - E(0)$"` just works), no top/right spine, faint dotted grid. The four variants share this **same axis/spacing treatment** and differ only in typeface — see [`docs/plot-style-gallery.md`](plot-style-gallery.md) for a rendered side-by-side comparison before picking one. Used by both `workflows/*/scripts/plot_results.py` and `plot_structure.py` (currently defaulted to `editorial_dejavu_serif`, pending a final pick). |
| `"nature"` (alias `"pub"`/`"publication"`) | Compact journal-figure alternative when the editorial family's larger type doesn't fit a multi-panel grid — sans-serif, `axes.labelsize=9`. |
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
