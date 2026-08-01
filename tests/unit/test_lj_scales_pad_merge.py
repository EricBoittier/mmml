"""Unit tests for examples/lj_scales pad-merge and campaign wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from mmml.cli.run.md_system import parse_md_system_args

_REPO = Path(__file__).resolve().parents[2]
_PAD_MERGE = _REPO / "examples/lj_scales/_pad_merge_npz.py"


def _load_pad_merge():
    spec = importlib.util.spec_from_file_location("lj_pad_merge", _PAD_MERGE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_npz(path: Path, *, n_frames: int, n_atoms: int, pad: int) -> None:
    R = np.zeros((n_frames, pad, 3))
    Z = np.zeros((n_frames, pad), dtype=np.int32)
    F = np.zeros((n_frames, pad, 3))
    mol = np.full((n_frames, pad), -1, dtype=np.int32)
    tidx = np.full((n_frames, pad), -1, dtype=np.int32)
    chg = np.zeros((n_frames, pad))
    pattern = np.array([6, 17, 17, 1, 1], dtype=np.int32)
    for i in range(n_frames):
        z_row = np.resize(pattern, n_atoms)
        Z[i, :n_atoms] = z_row
        mol[i, :n_atoms] = 0
        tidx[i, :n_atoms] = 1
    np.savez(
        path,
        R=R,
        Z=Z,
        N=np.full(n_frames, n_atoms, dtype=np.int32),
        E=np.zeros((n_frames, 1)),
        F=F,
        D=np.zeros((n_frames, 3)),
        mol_id=mol,
        cgenff_type_idx=tidx,
        cgenff_charge=chg,
        res_name=np.array(["DCM"] * n_frames),
    )


def test_pad_frames_to_marks_padding() -> None:
    mod = _load_pad_merge()
    raw = {
        "R": np.ones((2, 10, 3)),
        "Z": np.ones((2, 10), dtype=np.int32),
        "F": np.ones((2, 10, 3)),
        "mol_id": np.zeros((2, 10), dtype=np.int32),
        "cgenff_type_idx": np.ones((2, 10), dtype=np.int32),
        "cgenff_charge": np.zeros((2, 10)),
        "N": np.array([10, 10], dtype=np.int32),
        "E": np.zeros((2, 1)),
    }
    out = mod.pad_frames_to(raw, 20)
    assert out["R"].shape == (2, 20, 3)
    assert out["Z"].shape == (2, 20)
    assert np.all(out["Z"][:, 10:] == 0)
    assert np.all(out["mol_id"][:, 10:] == -1)
    assert np.all(out["cgenff_type_idx"][:, 10:] == -1)
    assert np.all(out["N"] == 10)


def test_merge_npz_paths_concat(tmp_path: Path) -> None:
    mod = _load_pad_merge()
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _tiny_npz(a, n_frames=3, n_atoms=10, pad=10)
    _tiny_npz(b, n_frames=2, n_atoms=15, pad=15)
    merged = mod.merge_npz_paths([a, b], pad_to=20)
    assert merged["R"].shape == (5, 20, 3)
    assert merged["N"].tolist() == [10, 10, 10, 15, 15]
    assert np.all(merged["mol_id"][:, 15:] == -1)


def test_pad_merge_cli(tmp_path: Path) -> None:
    mod = _load_pad_merge()
    a = tmp_path / "a.npz"
    out = tmp_path / "out.npz"
    _tiny_npz(a, n_frames=2, n_atoms=10, pad=10)
    assert mod.main([str(a), "-o", str(out), "--pad-to", "20"]) == 0
    d = np.load(out)
    assert d["R"].shape == (2, 20, 3)


def test_make_dimer_scan_rejects_single_conformer(tmp_path: Path) -> None:
    import subprocess
    import sys

    src = tmp_path / "src.npz"
    # Minimal stub — argparse should fail before needing real chemistry.
    np.savez(
        src,
        R=np.zeros((1, 5, 3)),
        Z=np.array([[6, 17, 17, 1, 1]], dtype=np.int32),
        N=np.array([5], dtype=np.int32),
        res_name=np.array(["DCM"]),
        cgenff_type_idx=np.zeros((1, 5), dtype=np.int32),
        cgenff_charge=np.zeros((1, 5)),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/make_dimer_scan_dataset.py",
            "--data",
            str(src),
            "--resids",
            "DCM",
            "--monomer-conformers",
            "1",
            "--geometry-only",
            "--out",
            str(tmp_path / "geoms.npz"),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "monomer-conformers must be >= 2" in (proc.stderr + proc.stdout)


def test_lj_scales_liquid_campaign_yaml_parses() -> None:
    cfg = _REPO / "examples/hybrid_mm_charges/md_lj_scales_liquid_campaign.yaml"
    assert cfg.is_file()
    raw = yaml.safe_load(cfg.read_text())
    assert "runs" in raw
    assert set(raw["runs"]) >= {
        "jaxmd_settle",
        "pycharmm_npt",
        "jaxmd_nvt",
        "jaxmd_nve",
    }
    assert raw["runs"]["pycharmm_npt"]["depends_on"] == "jaxmd_settle"
    assert raw["runs"]["jaxmd_nvt"]["depends_on"] == "pycharmm_npt"
    assert raw["runs"]["jaxmd_nve"]["depends_on"] == "jaxmd_nvt"
    assert raw["defaults"]["mm_nonbond_mode"] == "jax_mic"

    # parse_md_system_args applies campaign ``defaults``; per-job backend/setup
    # merge happens later in run_campaign.
    args = parse_md_system_args(
        [
            "--config",
            str(cfg),
            "--job-id",
            "jaxmd_settle",
            "--checkpoint",
            "/tmp/ckpt.json",
            "--from-psf",
            "/tmp/model.psf",
            "--from-crd",
            "/tmp/model.crd",
            "--composition",
            "DCM:8",
            "--mm-lj-scales-file",
            "/tmp/hybrid_mm.json",
        ]
    )
    assert args.mm_nonbond_mode == "jax_mic"
    assert args.composition == "DCM:8"
    assert str(args.mm_lj_scales_file).endswith("hybrid_mm.json")
    assert str(args.from_psf).endswith("model.psf")
    assert args.job_id == "jaxmd_settle"
