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
