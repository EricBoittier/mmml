"""Tests for DCM NVE scaling result collector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "dcm_nve_scaling"
_SCRIPTS = _WORKFLOW / "scripts"


def _load_collect_module():
    path = _SCRIPTS / "collect_scaling.py"
    spec = importlib.util.spec_from_file_location("collect_scaling", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_scaling"] = mod
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def collect_mod():
    return _load_collect_module()


def test_load_com_npz_reads_object_arrays(tmp_path: Path, collect_mod) -> None:
    """Regression: object-dtype fail_reasons broke allow_pickle=False loads."""
    path = tmp_path / "com_analysis.npz"
    np.savez(
        path,
        max_disp_per_monomer=np.array([[1.0, 2.0], [1.5, 2.5]]),
        ok=np.array(True),
        n_frames=np.array(2),
        max_cluster_com_disp_A=np.array(1.2),
        mean_msd_cluster_A2=np.array(0.3),
        max_internal_rmsd_A=np.array(0.4),
        fail_reasons=np.array(["cluster_disp"], dtype=object),
        dcd_header=np.array(["hdr"], dtype=object),
    )
    loaded = collect_mod._load_com_npz(path)
    assert loaded is not None
    assert "_load_error" not in loaded
    assert float(np.max(loaded["max_disp_per_monomer"])) == pytest.approx(2.5)


def test_row_for_size_passes_with_fixture(tmp_path: Path, collect_mod, monkeypatch) -> None:
    cfg = {"composition_prefix": "DCM", "output_root": "results"}
    out = tmp_path / "results" / "dcm_3_nve" / "inbfrq_neg1"
    out.mkdir(parents=True)
    (out / "done.txt").write_text("", encoding="utf-8")
    np.savez(
        out / "com_analysis.npz",
        max_disp_per_monomer=np.array([0.5, 0.6, 0.7]),
        ok=np.array(True),
        n_frames=np.array(10),
        max_cluster_com_disp_A=np.array(0.8),
        mean_msd_cluster_A2=np.array(0.1),
        max_internal_rmsd_A=np.array(0.2),
        fail_reasons=np.array([], dtype=object),
    )

    def _paths(_cfg, _n, *, inbfrq):
        assert inbfrq == -1
        return {
            "out_dir": out,
            "com_npz": out / "com_analysis.npz",
            "audit_json": out / "audit.json",
            "forces_npz": out / "forces.npz",
        }

    monkeypatch.setattr(collect_mod, "paths_for_size", _paths)
    row = collect_mod._row_for_size(cfg, 3, inbfrq=-1)
    assert row["status"] == "pass"
    assert row["n_frames"] == 10
