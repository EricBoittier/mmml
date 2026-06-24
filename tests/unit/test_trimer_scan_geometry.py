"""Unit tests for trimer 2D scan geometry (no CHARMM)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
    atoms_per_monomer_from_psf,
    com_distances,
    distance_report,
    place_trimer,
    run_scan_2d,
)

_REPO = Path(__file__).resolve().parents[2]
_SCAN_SCRIPT = _REPO / "scripts" / "scan_mlpot_dimer_2d_pycharmm.py"


def _load_scan_script():
    spec = importlib.util.spec_from_file_location("scan_mlpot_dimer_2d_pycharmm", _SCAN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_place_trimer_sets_target_com_distances() -> None:
    atoms_per = [2, 2, 2]
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [1.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    pos = place_trimer(ref, atoms_per, d01=4.0, d02=5.0, angle_02_rad=0.0)
    d = com_distances(pos, atoms_per)
    assert d[0] == pytest.approx(4.0, abs=1e-9)
    assert d[1] == pytest.approx(5.0, abs=1e-9)


def test_run_scan_2d_collects_metrics() -> None:
    atoms_per = [1, 1, 1]
    ref = np.eye(3, dtype=float)
    d_grid = np.array([3.0, 4.0])

    def eval_fn(pos: np.ndarray) -> dict[str, float]:
        dist = distance_report(pos, atoms_per)
        return {
            "energy_kcal": float(dist["com_d01"] + dist["com_d02"]),
            "min_inter_01": dist["min_inter_01"],
        }

    out = run_scan_2d(
        eval_fn,
        ref,
        atoms_per,
        d_grid,
        d_grid,
        angle_02_deg=90.0,
        metric_keys=("energy_kcal", "min_inter_01"),
    )
    assert out["energy_kcal"].shape == (2, 2)
    assert np.all(np.isfinite(out["energy_kcal"]))
    assert np.all(out["min_inter_01"] > 0.0)


def test_run_scan_2d_accepts_dimer_cluster() -> None:
    atoms_per = [1, 1]
    ref = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    d1_grid = np.array([3.0, 4.0])
    d2_grid = np.array([5.0, 6.0])

    def eval_fn(pos: np.ndarray) -> dict[str, float]:
        return {"energy_kcal": float(distance_report(pos, atoms_per)["com_d01"])}

    out = run_scan_2d(
        eval_fn,
        ref,
        atoms_per,
        d1_grid,
        d2_grid,
        angle_02_deg=90.0,
        metric_keys=("energy_kcal", "min_inter_01", "min_inter_02"),
    )
    assert out["energy_kcal"].shape == (2, 2)
    assert np.all(out["energy_kcal"] == d1_grid[:, None])
    assert np.all(out["min_inter_01"] == d1_grid[:, None])
    assert np.all(np.isnan(out["min_inter_02"]))


def test_run_scan_2d_accepts_more_than_three_monomers() -> None:
    atoms_per = [1, 1, 1, 1]
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 0.0],
        ],
        dtype=float,
    )

    out = run_scan_2d(
        lambda pos: {"energy_kcal": float(distance_report(pos, atoms_per)["com_d03"])},
        ref,
        atoms_per,
        np.array([3.0]),
        np.array([4.0]),
        angle_02_deg=90.0,
        metric_keys=("energy_kcal", "min_inter_03", "min_inter_23"),
    )
    assert out["energy_kcal"][0, 0] == pytest.approx(np.sqrt(162.0))
    assert out["min_inter_03"][0, 0] == pytest.approx(np.sqrt(162.0))
    assert out["min_inter_23"][0, 0] == pytest.approx(np.sqrt(106.0))


def test_atoms_per_monomer_from_psf_uses_ibase(monkeypatch) -> None:
    fake_psf = types.SimpleNamespace(
        get_natom=lambda: 15,
        get_ibase=lambda: [5, 10, 15],
        get_resid=lambda: [1, 2, 3],
    )
    fake_pycharmm = types.SimpleNamespace(psf=fake_psf)
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.psf", fake_psf)

    assert atoms_per_monomer_from_psf() == [5, 5, 5]


def test_atoms_per_monomer_from_psf_accepts_leading_zero_ibase(monkeypatch) -> None:
    fake_psf = types.SimpleNamespace(
        get_natom=lambda: 15,
        get_ibase=lambda: [0, 5, 10, 15],
        get_resid=lambda: [1, 2, 3],
    )
    fake_pycharmm = types.SimpleNamespace(psf=fake_psf)
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.psf", fake_psf)

    assert atoms_per_monomer_from_psf() == [5, 5, 5]


def test_atoms_per_monomer_from_psf_rejects_residue_table_as_atom_resids(monkeypatch) -> None:
    fake_psf = types.SimpleNamespace(
        get_natom=lambda: 15,
        get_resid=lambda: [1, 2, 3],
    )
    fake_pycharmm = types.SimpleNamespace(psf=fake_psf)
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.psf", fake_psf)

    with pytest.raises(ValueError, match="get_ibase is required"):
        atoms_per_monomer_from_psf()


def test_scan_mlpot_dimer_2d_pycharmm_batch_parse() -> None:
    mod = _load_scan_script()
    assert mod._parse_batch_compositions("DCM:2, ACO:4") == ["DCM:2", "ACO:4"]


def test_scan_mlpot_dimer_2d_pycharmm_help_accepts_dimer_example(capsys) -> None:
    mod = _load_scan_script()
    with pytest.raises(SystemExit) as exc:
        mod._parse_args(
            [
                "DCM:2",
                "--output-dir",
                "artifacts/pycharmm_mlpot/dimer_2d_scan/dcm2",
                "-h",
            ]
        )
    assert exc.value.code == 0
    assert "DCM:2" in capsys.readouterr().out
