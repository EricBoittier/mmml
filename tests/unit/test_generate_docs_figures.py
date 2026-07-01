"""Tests for documentation figure generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EXPECTED_IMAGES = (
    "docs/images/structures/make-res-aco.png",
    "docs/images/structures/make-box-acetone.png",
    "docs/images/structures/build-crystal.png",
    "docs/images/plots/liquid-box-density-ladder.png",
    "docs/images/plots/structure-builder-sizes.png",
)


def test_doc_figures_exist():
    for rel in EXPECTED_IMAGES:
        path = REPO / rel
        assert path.is_file(), f"missing doc figure: {rel}"
        assert path.stat().st_size > 1000


def test_generate_docs_figures_check_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_docs_figures.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
