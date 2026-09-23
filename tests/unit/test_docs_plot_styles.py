"""Keep MkDocs image publishers on the shared plotting baseline."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


def _import_clause_from_styles(source: str) -> str:
    """Return the ``mmml.utils.plotting.styles`` import clause.

    Combined single-line imports are fine (``apply_plot_style, comparison_colors``).
    Parenthesized multiline imports include names after the opening ``(``.
    """
    marker = "from mmml.utils.plotting.styles import"
    if marker not in source:
        return ""
    clause = source.split(marker, 1)[1]
    stripped = clause.lstrip()
    if stripped.startswith("("):
        return stripped.split(")", 1)[0]
    return clause.split("\n", 1)[0]


@pytest.mark.parametrize(
    "source",
    (
        "from mmml.utils.plotting.styles import apply_plot_style, comparison_colors\n",
        "from mmml.utils.plotting.styles import (\n    apply_plot_style,\n    comparison_colors,\n)\n",
        "from mmml.utils.plotting.styles import (  # noqa: E402\n    apply_plot_style,\n)\n",
    ),
)
def test_import_clause_finds_apply_plot_style_across_styles(source: str):
    assert "apply_plot_style" in _import_clause_from_styles(source)


def test_first_line_only_misses_parenthesized_multiline_import():
    source = (
        "from mmml.utils.plotting.styles import (\n"
        "    apply_plot_style,\n"
        "    comparison_colors,\n"
        ")\n"
    )
    first_line = source.split("from mmml.utils.plotting.styles import", 1)[1]
    first_line = first_line.split("\n", 1)[0]
    assert "apply_plot_style" not in first_line
    assert "apply_plot_style" in _import_clause_from_styles(source)


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/generate_docs_figures.py",
        "scripts/analyze_aaa_ama_dataset.py",
        "scripts/collect_md_embedding_docs_results.py",
        "scripts/plot_trajectory_structure.py",
    ),
)
def test_docs_image_publishers_use_shared_icml_style(relative_path: str):
    source = (REPO / relative_path).read_text(encoding="utf-8")
    # Combined imports are fine (`apply_plot_style, comparison_colors`); the
    # Sep 16 CI failure required the name to be the first imported symbol.
    assert "from mmml.utils.plotting.styles import" in source
    assert "apply_plot_style" in _import_clause_from_styles(source)
    assert 'apply_plot_style("icml")' in source
