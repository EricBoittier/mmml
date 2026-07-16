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


def test_maybe_apply_certified_box_json_requires_box_json(tmp_path: Path):
    from mmml.cli.run.md_pbc_suite.ase import _maybe_apply_certified_box_json

    crd = tmp_path / "model.crd"
    crd.write_text("dummy\n", encoding="utf-8")
    args = argparse.Namespace(box_size=28.0, quiet=True)
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
