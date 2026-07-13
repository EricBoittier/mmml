from __future__ import annotations

from pathlib import Path

import pytest

from mmml.cli.misc.extract_checkpoint_metrics import (
    plot_training_comparison,
    plot_training_metrics,
)
from mmml.utils.plotting.styles import (
    DEFAULT_PLOT_STYLE,
    LINE_STYLE_CYCLE,
    MARKER_CYCLE,
    MULTI_CMAP_SHORTLIST,
    OKABE_ITO_PALETTE,
    PLOT_STYLES,
    STATUS_COLORS,
    STATUS_HATCHES,
    STYLE_DISPLAY_ORDER,
    apply_plot_style,
    assert_no_text_overlap,
    comparison_colors,
    find_overlapping_text,
    get_plot_style,
    latex_available,
    legend_outside,
    list_plot_styles,
    render_latex_table,
    status_color,
    status_hatch,
    timeseries_with_distribution,
)


def _synthetic_metrics(n: int = 40, seed: int = 0) -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n + 1, dtype=float)
    valid_loss = 1e10 * np.exp(-0.25 * epochs) + 8.0 + 0.05 * rng.normal(size=n)
    train_loss = valid_loss * 1.1
    valid_energy = 0.2 * np.exp(-0.12 * epochs) + 0.15
    valid_forces = 0.08 * np.exp(-0.1 * epochs) + 0.05
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_energy_mae": valid_energy * 1.05,
        "valid_energy_mae": valid_energy,
        "train_forces_mae": valid_forces * 1.05,
        "valid_forces_mae": valid_forces,
        "train_dipole_mae": np.full(n, np.nan),
        "valid_dipole_mae": np.full(n, np.nan),
        "lr_eff": np.full(n, 1e-3),
        "best_loss": np.minimum.accumulate(valid_loss),
    }


@pytest.mark.parametrize("style_name", list_plot_styles())
def test_each_plot_style_renders_training_curves(tmp_path: Path, style_name: str) -> None:
    out = tmp_path / f"{style_name}.png"
    plot_training_metrics(
        _synthetic_metrics(),
        out,
        ckpt_name=f"demo-{style_name}",
        log_loss=True,
        verbose=False,
        ef_only=True,
        plot_style=style_name,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_plot_style_aliases_resolve() -> None:
    assert get_plot_style("science").name == "science"
    assert get_plot_style("dark").name == "dark"
    assert get_plot_style("grace").name == "xmgrace"


def test_unknown_plot_style_raises() -> None:
    with pytest.raises(ValueError, match="Unknown plot style"):
        get_plot_style("not-a-real-style")


def test_comparison_colors_length() -> None:
    colors = comparison_colors("tron", 8)
    assert len(colors) == 8
    assert all(isinstance(c, str) for c in colors)


def test_apply_plot_style_returns_named_preset() -> None:
    style = apply_plot_style(DEFAULT_PLOT_STYLE)
    assert style.name in PLOT_STYLES


@pytest.mark.parametrize("style_name", ["nature", "google", "tron"])
def test_comparison_plot_accepts_style(tmp_path: Path, style_name: str) -> None:
    runs = [
        ("dcm1-aaaa-bbbb", _synthetic_metrics(30, seed=1)),
        ("dcm1-cccc-dddd", _synthetic_metrics(35, seed=2)),
    ]
    out = tmp_path / f"compare_{style_name}.png"
    written = plot_training_comparison(runs, out, ef_only=True, verbose=False, plot_style=style_name)
    assert len(written) == 3
    assert (tmp_path / f"compare_{style_name}_valid_loss.png").is_file()


def test_okabe_ito_palette_is_eight_distinct_hex_colors() -> None:
    assert len(OKABE_ITO_PALETTE) == 8
    assert len(set(OKABE_ITO_PALETTE)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in OKABE_ITO_PALETTE)


def test_icml_uses_okabe_ito_as_its_default_cycle() -> None:
    style = get_plot_style("icml")
    assert style.comparison_palette == OKABE_ITO_PALETTE


def test_icml_okabe_ito_alias_resolves_to_icml() -> None:
    assert get_plot_style("icml_okabe_ito") is get_plot_style("icml")


def test_every_preset_has_no_legend_border() -> None:
    for name, style in PLOT_STYLES.items():
        assert style.rc_params.get("legend.frameon") is False, name
        assert "legend.edgecolor" not in style.rc_params, name


def test_every_preset_wires_comparison_palette_into_prop_cycle() -> None:
    for name, style in PLOT_STYLES.items():
        cycle_colors = [c["color"] for c in style.rc_params["axes.prop_cycle"]]
        assert cycle_colors == list(style.comparison_palette), name


def test_icml_bare_plot_calls_cycle_through_okabe_ito() -> None:
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    fig, ax = plt.subplots()
    colors = [ax.plot([0, 1], [i, i + 1])[0].get_color() for i in range(3)]
    plt.close(fig)
    assert colors == list(OKABE_ITO_PALETTE[:3])


def test_multi_cmap_shortlist_covers_all_three_kinds() -> None:
    assert set(MULTI_CMAP_SHORTLIST) == {"sequential", "diverging", "cyclic"}
    for kind, names in MULTI_CMAP_SHORTLIST.items():
        assert len(names) >= 2, kind
        assert len(set(names)) == len(names), f"duplicate colormap name in {kind}"


def test_line_and_marker_cycles_are_nonempty_and_distinct() -> None:
    assert len(set(LINE_STYLE_CYCLE)) == len(LINE_STYLE_CYCLE) > 0
    assert len(set(MARKER_CYCLE)) == len(MARKER_CYCLE) > 0


@pytest.mark.skipif(not latex_available(), reason="requires pdflatex + pdftocairo on PATH")
def test_render_latex_table_produces_a_png(tmp_path: Path) -> None:
    out = tmp_path / "table.png"
    result = render_latex_table(
        [["monopole", "+0.289", "e"], ["atom_id_5%", "0.9", "e|bohr"]],
        col_labels=["quantity", "value", "units"],
        out_path=out,
    )
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_every_preset_sets_a_title_pad() -> None:
    # A bold/enlarged title with matplotlib's default 6pt titlepad sits close
    # enough to the axes box to visually collide with the top y-tick label.
    for name, style in PLOT_STYLES.items():
        assert "axes.titlepad" in style.rc_params, f"{name} is missing axes.titlepad"
        assert style.rc_params["axes.titlepad"] >= 8.0, name


def _legend_side(legend, ax) -> str:
    bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    return "bottom" if bbox.y0 < 0 else "right"


def test_legend_outside_auto_picks_bottom_for_a_small_legend_either_aspect() -> None:
    # A single short entry costs almost nothing added to figure height at the
    # bottom, but a full extra column of width on the side -- min-area picks
    # bottom regardless of whether the figure itself is wide or tall.
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    for figsize in [(10, 4), (4, 10)]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], label="series")
        legend = legend_outside(ax, side="auto")
        assert _legend_side(legend, ax) == "bottom", figsize
        plt.close(fig)


def test_legend_outside_auto_picks_right_for_a_large_legend_either_aspect() -> None:
    # Many long-label entries would wrap into several wide columns at the
    # bottom (adding more height than the side adds width) -- min-area picks
    # "right" regardless of the figure's own aspect ratio.
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    for figsize in [(12, 4), (4, 10)]:
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(8):
            ax.plot([0, 1], [i, i + 1], label=f"a fairly long series label number {i}")
        legend = legend_outside(ax, side="auto")
        assert _legend_side(legend, ax) == "right", figsize
        plt.close(fig)


def test_legend_outside_auto_matches_explicit_min_area_side() -> None:
    # The "auto" choice should always equal whichever of "right"/"bottom"
    # actually produces the smaller total figure bounding box -- check this
    # directly rather than trusting the heuristic reasoning above alone.
    import matplotlib.pyplot as plt

    from mmml.utils.plotting.styles import _legend_footprint_in

    apply_plot_style("icml")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(4):
        ax.plot([0, 1], [i, i + 1], label=f"series {i}")
    fig_w, fig_h = fig.get_size_inches()

    areas = {}
    for side in ("right", "bottom"):
        legend_w, legend_h = _legend_footprint_in(ax, fig, side, None, {})
        if side == "right":
            areas[side] = (fig_w + legend_w) * max(fig_h, legend_h)
        else:
            areas[side] = max(fig_w, legend_w) * (fig_h + legend_h)
    expected_side = min(areas, key=areas.get)

    legend = legend_outside(ax, side="auto")
    assert _legend_side(legend, ax) == expected_side
    plt.close(fig)


def test_style_display_order_covers_every_registered_style() -> None:
    assert set(STYLE_DISPLAY_ORDER) == set(PLOT_STYLES)
    assert STYLE_DISPLAY_ORDER[0] == "icml"


def test_status_colors_and_hatches_share_the_same_keys() -> None:
    assert set(STATUS_COLORS) == set(STATUS_HATCHES) == {
        "good", "warning", "serious", "critical", "neutral",
    }
    assert len(set(STATUS_COLORS.values())) == 5  # all distinct hex colors


def test_status_color_resolves_aliases() -> None:
    assert status_color("success") == status_color("good") == STATUS_COLORS["good"]
    assert status_color("fail") == status_color("critical") == STATUS_COLORS["critical"]


def test_status_color_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown status level"):
        status_color("not-a-real-level")


def test_status_hatch_good_is_unhatched() -> None:
    assert status_hatch("good") == ""
    assert status_hatch("critical") != ""


def test_find_overlapping_text_clean_figure_has_no_overlaps() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style("icml")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    ax.set_title("A reasonably sized figure")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    assert find_overlapping_text(fig) == []
    assert_no_text_overlap(fig)  # should not raise
    plt.close(fig)


def test_legend_outside_left_does_not_collide_with_ylabel() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style("icml")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(50)
    ax.plot(x, -75 + 0.1 * np.sin(x / 5), label="a fairly long series label")
    ax.set_xlim(0, 49)
    ax.set_ylabel("E(t) - E(0)  (eV)")
    legend_outside(ax, side="left", fontsize=10)
    overlaps = find_overlapping_text(fig)
    # The regression this guards: the legend's own text used to sit right on
    # top of the y-tick labels/ylabel when anchored at the old -0.02 offset.
    label_related = [(a, b) for a, b in overlaps
                      if "series label" in a or "series label" in b
                      or "E(t)" in a or "E(t)" in b]
    assert label_related == [], overlaps
    plt.close(fig)


def test_find_overlapping_text_detects_a_real_collision() -> None:
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_title("Very Long Overlapping Title Text Here", fontsize=22, y=0.85)
    ax.set_ylabel("y")
    overlaps = find_overlapping_text(fig)
    assert overlaps, "expected a real overlap in a deliberately cramped figure"
    with pytest.raises(AssertionError, match="Overlapping text"):
        assert_no_text_overlap(fig)
    plt.close(fig)


def test_timeseries_with_distribution_centers_by_default() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style("icml")
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 1)
    t = np.linspace(0, 10, 50)
    y = 5.0 + np.sin(t)  # offset series -- centering should remove the 5.0
    ax_series, ax_hist = timeseries_with_distribution(fig, gs[0, 0], t, y, color="#4C72B0")
    # the plotted series (not the raw input) should be mean-centered
    plotted_y = ax_series.lines[0].get_ydata()
    assert abs(float(np.mean(plotted_y))) < 1e-9
    assert ax_hist is not ax_series
    plt.close(fig)


def test_timeseries_with_distribution_no_center_keeps_raw_values() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style("icml")
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 1)
    t = np.linspace(0, 10, 50)
    y = 5.0 + np.sin(t)
    ax_series, _ = timeseries_with_distribution(fig, gs[0, 0], t, y, color="#4C72B0", center=False)
    plotted_y = ax_series.lines[0].get_ydata()
    assert abs(float(np.mean(plotted_y)) - 5.0) < 0.5
    plt.close(fig)
