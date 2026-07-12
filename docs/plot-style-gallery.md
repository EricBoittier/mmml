# Plot style gallery

Renders of the same example figure under each registered style
(`mmml.utils.plotting.styles`), so a look can be picked by eye instead of
from a description. Regenerate with `python scripts/render_plot_style_gallery.py`.

Every example also demonstrates the **legend-outside-the-plot** rule (see
`docs/plotting-style-guide.md` "Legends live outside the plot") via
`legend_outside(ax)` — never overlapping the data, and free to grow long
(reads like a small table of series → color/marker) without crowding
anything.

**Current pick: `"icml"`** — clean sans-serif, muted "seaborn deep"-style
categorical colors, moderate (not oversized) type; the closest preset to a
modern ML-conference figure. This is what `workflows/*/scripts/plot_results.py`
and `plot_structure.py` use by default now.

## `icml`

![icml](plot-style-gallery-assets/icml.png)

## The `editorial_*` family (large-type/thick-line/no-spine variants)

These four share the **same axes/spacing treatment** — large type, thick
lines, no top/right spine, faint grid — only the font differs. Kept in the
gallery as an alternative "read from across the room" look for a different
kind of figure (e.g. a talk slide) than `"icml"`'s conference-paper density.

**Note on naming**: these were previously grouped under one preset literally
called `"tufte"`. That was a category error — Tufte described *principles*
(data-ink ratio, redundant encoding, small multiples), not a single font or
palette, so no one preset should claim his name. The axis/spacing choices
below (no chart junk, minimal spines) draw on those principles; the presets
are named `editorial_*` instead. `"tufte"` still resolves as a backward-compat
alias to `editorial_stix`, but new code should name the variant directly.

Each example also demonstrates the "semantic color, not palette index"
principle from [`docs/plotting-style-guide.md`](plotting-style-guide.md):
the two energy traces are colored by *system* (not palette order), and the
right-hand panel gives bonds/angles/dihedrals **three different colors**
because they're three different physical quantities — not the same blue
reused three times (the bug this pass fixed in
`scripts/plot_trajectory_structure.py::plot_internal`).

## `editorial_dejavu_sans`

Matplotlib's bundled sans-serif (DejaVu Sans) — renders identically on every
machine, no font-substitution risk.

![editorial_dejavu_sans](plot-style-gallery-assets/editorial_dejavu_sans.png)

## `editorial_dejavu_serif`

Matplotlib's bundled serif (DejaVu Serif) — same reliability guarantee as
above, traditional serif feel.

![editorial_dejavu_serif](plot-style-gallery-assets/editorial_dejavu_serif.png)

## `editorial_stix`

STIX serif — a journal-typeset look (closest to what most physics/chemistry
papers use). This was the previous `"tufte"` preset.

![editorial_stix](plot-style-gallery-assets/editorial_stix.png)

## `editorial_cm`

DejaVu Serif body text with Computer-Modern-style math
(`mathtext.fontset="cm"`) — the classic LaTeX-paper look for equations
specifically, while regular text stays DejaVu.

![editorial_cm](plot-style-gallery-assets/editorial_cm.png)

## Chart types (not just fonts)

Same style (`"icml"`), different chart *forms* — regenerate with
`python scripts/render_chart_type_gallery.py`. Each one names the specific
Tufte principle it demonstrates, since "in line with Tufte's teachings" is
about more than axis minimalism — several of his central ideas (small
multiples, the range-frame, data-ink ratio, redundant encoding) are concrete,
nameable techniques, not just "make it clean." Use these to judge whether a
*form*, not just a font, fits before committing to it for a real figure.

### Radial (polar)

A circular histogram of a periodic quantity (dihedral angle). Use the
geometry the data actually has — an angle wraps around; forcing it onto a
linear 0-360° axis hides the wraparound at the boundary.

![radial](plot-style-gallery-assets/chart_radial.png)

### 3D surface

A dimer PES as an explicit 3D surface. Included because it was asked for,
but flagged honestly: 3D is data-ink-*expensive* (occlusion, projection
distortion, no way to read exact values) — Tufte generally preferred 2D +
color/contour for exactly this reason. The `matshow` and `range_frame`
panels below are the better-Tufte alternative for two-variable-vs-response
data of this kind.

![3d surface](plot-style-gallery-assets/chart_3d_surface.png)

### XY scatter with confidence interval

A fitted trend with a shaded 95% CI band, raw points still visible
underneath. The band carries the uncertainty; no per-point error bars
cluttering every marker.

![scatter with CI](plot-style-gallery-assets/chart_scatter_ci.png)

### Time series with distributions above and below

Not seaborn's `jointplot` (marginals top+right, sharing the plot's
*x*-axis) — here the two extra panels sit above and below the middle one
and share its **y-axis** instead, showing how the value distribution itself
shifts between the first and second half of a run. Two related
distributions placed for direct visual comparison, Tufte's small-multiples
idea applied to margins instead of a legend/caption.

![time series with marginals](plot-style-gallery-assets/chart_timeseries_marginals.png)

### Lollipop chart

A thin stem + a dot instead of a filled bar — the most literal data-ink-ratio
win in this gallery: nearly all of a bar's ink minus the fill, for the same
information (position along the stem = value). See `docs/plotting-style-guide.md`
"Overlaid semi-transparent bars, not more panels" for the sibling rule about
when *not* to add another panel instead.

![lollipop](plot-style-gallery-assets/chart_lollipop.png)

### Matrix heatmap (`matshow`)

A pairwise atom-distance matrix. Color encodes magnitude directly on the
(i, j) grid the data already has — no need to invent x/y positions for it
the way a scatter would.

![matshow](plot-style-gallery-assets/chart_matshow.png)

### Small multiples

Tufte's signature form, by name: many small, identically-scaled panels, with
almost all chart furniture (ticks, boxes, axis labels) stripped since the
*shape* is the point, not reading an exact value off any one panel. Compare
directly to the “everything overlaid in one legend-heavy panel” approach
used in `workflows/*/scripts/plot_results.py` — small multiples is the
alternative worth considering when a sweep has few enough series to lay out
as a grid.

![small multiples](plot-style-gallery-assets/chart_small_multiples.png)

### Range-frame

Tufte's own named contribution (*The Visual Display of Quantitative
Information*): axis lines span only the data's actual range, not an
arbitrary origin-anchored box, with ticks only at min/median/max. Worth
calling out explicitly since it's a specific technique, not a generic
"minimalist" choice.

![range frame](plot-style-gallery-assets/chart_range_frame.png)

### Schematic diagram

Not a data plot — boxes + thin connecting lines for composition/flow (here:
how `HybridEnergy` sums its terms). Same principle applies to diagrams as to
charts: no drop shadows/gradients/3D bevels, labels placed directly on the
elements they describe rather than pulled out into a legend.

![diagram](plot-style-gallery-assets/chart_diagram.png)

## How to pick

```python
from mmml.utils.plotting.styles import apply_plot_style

apply_plot_style("icml")  # or editorial_dejavu_serif / _dejavu_sans / _stix / _cm
```

`_STYLE_NAME` in `workflows/*/scripts/plot_results.py` and `plot_structure.py`
is currently `"icml"` — see [`docs/plotting-style-guide.md`](plotting-style-guide.md)
for the rest of the convention (semantic color, DPI, uncertainty shading,
legends outside).

(There was a 5th candidate, `editorial_stixsans`, dropped from the gallery:
matplotlib has no real "STIX Sans" font bundled, so it silently fell back to
the same rendering as `editorial_dejavu_sans` — not a meaningfully distinct
option.)
