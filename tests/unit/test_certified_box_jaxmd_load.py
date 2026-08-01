"""Certified liquid-box PSF/CRD must load on ase/jaxmd (not Packmol rebuild)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.cli.run.md_pbc_suite.ase import (
    certified_box_geometry_requested,
    cluster_geometry_from_certified_artifacts,
    resolve_cluster_geometry,
)
from mmml.cli.run.md_system import build_command


def test_ase_suite_import_does_not_eagerly_load_pycharmm():
    env = os.environ.copy()
    env["MMML_WARMUP_MLPOT_JAX_ONLY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mmml.cli.run.md_pbc_suite.ase as suite; "
                "assert suite.read is None"
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_certified_box_geometry_requested_requires_both():
    assert not certified_box_geometry_requested(argparse.Namespace(from_psf=None, from_crd=None))
    assert not certified_box_geometry_requested(
        argparse.Namespace(from_psf=Path("a.psf"), from_crd=None)
    )
    assert certified_box_geometry_requested(
        argparse.Namespace(from_psf=Path("a.psf"), from_crd=Path("a.crd"))
    )


def test_maybe_apply_certified_box_json_overrides_box_size(tmp_path: Path):
    from mmml.cli.run.md_pbc_suite.ase import _maybe_apply_certified_box_json

    crd = tmp_path / "model.crd"
    crd.write_text("dummy\n", encoding="utf-8")
    (tmp_path / "box.json").write_text(
        json.dumps({"box_side_A": 28.167, "density_g_cm3": 1.30}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        box_size=28.0,
        box_auto="density",
        target_density_g_cm3=1.36,
        bulk_density_fraction=None,
        quiet=True,
    )
    side = _maybe_apply_certified_box_json(args, crd)
    assert side == pytest.approx(28.167)
    assert args.box_size == pytest.approx(28.167)
    assert args.box_auto is None
    assert args.target_density_g_cm3 is None


def test_maybe_apply_certified_box_json_falls_back_to_box_size(tmp_path: Path):
    from mmml.cli.run.md_pbc_suite.ase import _maybe_apply_certified_box_json

    crd = tmp_path / "model.crd"
    crd.write_text("dummy\n", encoding="utf-8")
<<<<<<< HEAD
    args = argparse.Namespace(box_size=28.0, quiet=True)
    side = _maybe_apply_certified_box_json(args, crd)
    assert side == pytest.approx(28.0)
    assert args.box_size == pytest.approx(28.0)
=======
    args = argparse.Namespace(
        box_size=30.0,
        box_auto="density",
        target_density_g_cm3=1.3,
        bulk_density_fraction=0.5,
        quiet=True,
    )
    side = _maybe_apply_certified_box_json(args, crd)
    assert side == pytest.approx(30.0)
    assert args.box_size == pytest.approx(30.0)
    assert args.box_auto is None
>>>>>>> 3dc82b323 (feat(md_campaign): add new CLI override keys for certified box deployment)


def test_maybe_apply_certified_box_json_requires_box_json_without_box_size(
    tmp_path: Path,
):
    from mmml.cli.run.md_pbc_suite.ase import _maybe_apply_certified_box_json

    crd = tmp_path / "model.crd"
    crd.write_text("dummy\n", encoding="utf-8")
    args = argparse.Namespace(box_size=None, quiet=True)
    with pytest.raises(FileNotFoundError, match="box.json"):
        _maybe_apply_certified_box_json(args, crd)


def test_resolve_cluster_geometry_uses_certified_artifacts(monkeypatch):
    z = np.array([6, 1, 1, 17, 17] * 2, dtype=int)
    r0 = np.arange(len(z) * 3, dtype=float).reshape(-1, 3)
    args = argparse.Namespace(
        from_psf=Path("boxes/dcm/model.psf"),
        from_crd=Path("boxes/dcm/model.crd"),
        composition="DCM:2",
        quiet=True,
        box_size=28.0,
    )

    def _fake_load(_args):
        return z, r0, 2, "dcm2"

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.load_cluster_from_artifacts",
        _fake_load,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.trimer_scan.atoms_per_monomer_from_psf",
        lambda: [5, 5],
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_pbc_suite.ase._maybe_apply_certified_box_json",
        lambda *_a, **_k: 28.167,
    )
    packmol_calls = {"n": 0}

    def _fail_build(*_a, **_k):
        packmol_calls["n"] += 1
        raise AssertionError("Packmol rebuild must not run for certified geometry")

    monkeypatch.setattr(
        "mmml.cli.run.md_pbc_suite.ase.build_initial_cluster_from_args",
        _fail_build,
    )

    out_z, out_r, atoms_per, labels, summary = resolve_cluster_geometry(args, None)
    assert packmol_calls["n"] == 0
    assert np.allclose(out_z, z)
    assert np.allclose(out_r, r0)
    assert atoms_per == [5, 5]
    assert labels == ["DCM", "DCM"]
    assert summary == {"DCM": 2}


def test_cluster_geometry_from_certified_requires_matching_atom_counts(monkeypatch):
    args = argparse.Namespace(
        from_psf=Path("a.psf"),
        from_crd=Path("a.crd"),
        composition=None,
        quiet=True,
        box_size=None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.load_cluster_from_artifacts",
        lambda _a: (np.zeros(5, dtype=int), np.zeros((5, 3)), 1, "x"),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.trimer_scan.atoms_per_monomer_from_psf",
        lambda: [3, 3],
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_pbc_suite.ase._maybe_apply_certified_box_json",
        lambda *_a, **_k: 28.0,
    )
    with pytest.raises(ValueError, match="resid atom counts"):
        cluster_geometry_from_certified_artifacts(args)


def test_validate_psf_charges_works_when_module_psf_cache_is_none(monkeypatch):
    """Certified-box load never fills ase.psf; validation must import pycharmm.psf."""
    import types

    import mmml.cli.run.md_pbc_suite.ase as ase_mod

    fake_psf = types.SimpleNamespace(
        get_atype=lambda: np.array(["CG331", "HGA3", "CG331", "HGA3"], dtype=str),
    )
    monkeypatch.setattr(ase_mod, "psf", None)
    monkeypatch.setitem(sys.modules, "pycharmm.psf", fake_psf)
    monkeypatch.setattr(
        ase_mod,
        "_get_actual_psf_charges",
        lambda n: np.array([0.1, -0.1, 0.1, -0.1], dtype=float)[:n],
    )

    summary = ase_mod._validate_psf_charges(
        monomer_offsets=np.array([0, 2, 4], dtype=int),
        residue_labels=["ACO", "ACO"],
        total_atoms=4,
    )
    assert summary["total_charge_e"] == pytest.approx(0.0)
    assert summary["residues"]["ACO"]["n_atoms"] == 2


def test_run_charmm_minimize_loads_module_cache_when_none(monkeypatch):
    """Certified-box path leaves ase.coor None; pre-min must hydrate the cache."""
    import types

    import mmml.cli.run.md_pbc_suite.ase as ase_mod
    from ase import Atoms

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    fake_coor = types.SimpleNamespace(
        set_positions=lambda _df: None,
        get_positions=lambda: types.SimpleNamespace(
            to_numpy=lambda dtype=float: pos.copy()
        ),
    )
    fake_min = types.SimpleNamespace(
        run_sd=lambda **_k: None,
        run_abnr=lambda **_k: None,
    )
    load_calls = {"n": 0}

    def _fake_load() -> None:
        load_calls["n"] += 1
        ase_mod.coor = fake_coor
        ase_mod.charmm_minimize = fake_min

    monkeypatch.setattr(ase_mod, "coor", None)
    monkeypatch.setattr(ase_mod, "charmm_minimize", None)
    monkeypatch.setattr(ase_mod, "_load_pycharmm_modules", _fake_load)
    monkeypatch.setattr(ase_mod, "reset_block", lambda: None)
    monkeypatch.setattr(ase_mod.pyci, "pycharmm_quiet", lambda: None)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_vacuum_nbonds",
        lambda **_k: None,
    )

    atoms = Atoms("HH", positions=pos)
    ase_mod._run_charmm_minimize(
        atoms,
        nstep_sd=1,
        nstep_abnr=0,
        tolenr=1e-3,
        tolgrd=1e-3,
    )
    assert load_calls["n"] == 1
    assert np.allclose(atoms.get_positions(), pos)


def test_build_command_jaxmd_forwards_from_psf_crd_and_skips_packmol():
    args = parse_md_system_minimal(
        from_psf=Path("boxes/dcm206/model.psf"),
        from_crd=Path("boxes/dcm206/model.crd"),
    )
    backend, cmd = build_command(args)
    assert backend == "jaxmd"
    assert "--from-psf" in cmd
    assert "boxes/dcm206/model.psf" in cmd
    assert "--from-crd" in cmd
    assert "boxes/dcm206/model.crd" in cmd
    assert "--packmol" not in cmd


def parse_md_system_minimal(**overrides):
    """Build a Namespace sufficient for build_command(jaxmd)."""
    from mmml.cli.run.md_system import parse_md_system_args

    base = [
        "--setup",
        "pbc_nve",
        "--backend",
        "jaxmd",
        "--composition",
        "DCM:206",
        "--box-size",
        "28.0",
        "--checkpoint",
        "ckpts/dummy",
        "--ps",
        "1.0",
    ]
    if "from_psf" in overrides:
        base.extend(["--from-psf", str(overrides["from_psf"])])
    if "from_crd" in overrides:
        base.extend(["--from-crd", str(overrides["from_crd"])])
    with mock.patch(
        "mmml.cli.run.md_system._validate_packmol_args",
        lambda _a: None,
    ), mock.patch(
        "mmml.cli.run.md_system._validate_builder_args",
        lambda _a: None,
    ):
        return parse_md_system_args(base)
