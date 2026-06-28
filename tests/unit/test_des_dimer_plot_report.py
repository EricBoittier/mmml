"""Unit tests for des dimer pair scan plotting and report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "des_dimer_pair_scans"
sys.path.insert(0, str(WORKFLOW / "scripts"))

import build_report  # noqa: E402
import plot_pair_scan  # noqa: E402


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    d = np.linspace(3.0, 6.0, 4)
    d1, d2 = np.meshgrid(d, d, indexing="ij")
    energy = (d1 - 4.0) ** 2 + (d2 - 4.5) ** 2
    path = tmp_path / "scan_2d.npz"
    np.savez_compressed(
        path,
        d01_grid=d,
        d02_grid=d,
        label="test pair",
        charmm_ENER_kcal=energy,
    )
    return path


def test_plot_pair_scan_npz(synthetic_npz: Path, tmp_path: Path):
    out = tmp_path / "fig.png"
    assert plot_pair_scan.plot_pair_scan_npz(synthetic_npz, out) is True
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_pending_pair(tmp_path: Path):
    out = tmp_path / "pending.png"
    plot_pair_scan.plot_pending_pair(out, pair_tag="a__b", label="A + B")
    assert out.is_file()


def test_build_report_all_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = WORKFLOW / "config.smoke.yaml"
    figures = tmp_path / "figures"
    html_out = tmp_path / "report.html"

    # Point one pair at synthetic data via output_root override in temp artifacts.
    import scan_lib as sl

    art = tmp_path / "artifacts"
    pair_dir = art / "aco__meoh"
    pair_dir.mkdir(parents=True)
    d = np.linspace(3.0, 6.0, 3)
    np.savez_compressed(
        pair_dir / "scan_2d.npz",
        d01_grid=d,
        d02_grid=d,
        label="acetone + methanol",
        charmm_ENER_kcal=np.ones((3, 3)),
    )

    smoke_cfg = sl.load_config(cfg)
    smoke_cfg["output_root"] = str(art)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.dump(smoke_cfg), encoding="utf-8")

    stats = build_report.build_report(
        cfg_path=cfg_path,
        figures_dir=figures,
        output_html=html_out,
        dpi=80,
    )
    assert stats["total"] == 78
    assert html_out.is_file()
    assert len(list(figures.glob("*.png"))) == 78
    text = html_out.read_text(encoding="utf-8")
    assert "aco__meoh" in text
    assert "DES dimer pair 2D scans" in text
