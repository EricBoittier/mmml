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
    apply_plot_style,
    comparison_colors,
    get_plot_style,
    latex_available,
    list_plot_styles,
    render_latex_table,
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


def test_icml_okabe_ito_style_uses_the_palette() -> None:
    style = get_plot_style("icml_okabe_ito")
    assert style.comparison_palette == OKABE_ITO_PALETTE


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
