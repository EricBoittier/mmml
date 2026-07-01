"""Unit tests for crystal literature comparison helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_metrics_from_dcm_cif():
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_dcm_crystal_cif

    m = metrics_from_cif(default_dcm_crystal_cif(), space_group=60)
    assert m.natoms == 20
    assert m.space_group == 60
    assert m.density_g_cm3 == pytest.approx(1.976, rel=0.01)
    assert m.lengths_a[0] == pytest.approx(3.924, rel=0.01)


def test_metrics_from_benzene_cif():
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_benzene_crystal_cif

    m = metrics_from_cif(default_benzene_crystal_cif(), space_group=14)
    assert m.natoms == 24
    assert m.space_group == 14
    assert m.density_g_cm3 == pytest.approx(1.202, rel=0.01)
    assert m.angles_deg[1] == pytest.approx(110.55, rel=0.01)


def test_comparison_table_markdown_with_built():
    from mmml.interfaces.crystal_reference import (
        CrystalMetrics,
        comparison_table_markdown,
    )

    lit = CrystalMetrics(
        label="lit",
        natoms=20,
        lengths_a=(4.0, 8.0, 9.0),
        angles_deg=(90.0, 90.0, 90.0),
        volume_a3=288.0,
        density_g_cm3=2.0,
        space_group=60,
    )
    built = CrystalMetrics(
        label="built",
        natoms=20,
        lengths_a=(4.0, 8.0, 9.0),
        angles_deg=(90.0, 90.0, 90.0),
        volume_a3=288.0,
        density_g_cm3=2.0,
        space_group=60,
    )
    md = comparison_table_markdown(
        lit,
        built,
        literature_citation="Test citation.",
        built_caption="Test build.",
    )
    assert "| ρ (g/cm³) | 2.000 | 2.000 | +0.0% |" in md


def test_literature_comparison_markdown_runs():
    from mmml.interfaces.crystal_reference import literature_comparison_markdown

    md = literature_comparison_markdown()
    assert "COD 2100015" in md
    assert "COD 4501704" in md
    assert "Literature cross-check" in md
    assert "make-res+CIF" in md


def test_generate_crystal_lit_compare_script():
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate_crystal_lit_compare.py")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
