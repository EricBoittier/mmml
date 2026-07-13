"""Keep MkDocs image publishers on the shared plotting baseline."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


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
    assert "from mmml.utils.plotting.styles import apply_plot_style" in source
    assert 'apply_plot_style("icml")' in source
