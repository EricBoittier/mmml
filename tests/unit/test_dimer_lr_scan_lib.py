"""Unit tests for dimer LR scan plotting (no PyCHARMM)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCAN_SCRIPT = REPO / "scripts" / "scan_mlpot_dimer_2d_pycharmm.py"


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("scan_mlpot_dimer_2d_pycharmm", _SCAN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_synthetic_scan_npz(path: Path, *, composition: str, scan_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d1 = np.linspace(4.0, 12.0, 5)
    hybrid = (d1 * 0.1).reshape(1, -1)
    ml2b = hybrid * 0.6
    mm = hybrid * 0.3
    np.savez_compressed(
        path,
        composition=np.array(composition),
        scan_tag=np.array(scan_tag),
        lr_solver_active=np.array(scan_tag),
        mm_nonbond_mode=np.array("jax_mic"),
        jax_pme_method=np.array("ewald"),
        scan_2d_d01_A=d1,
        scan_2d_hybrid_energy_kcal=hybrid,
        scan_2d_ml_2b_E_kcal=ml2b,
        scan_2d_mm_E_kcal=mm,
    )


def test_scan_parser_accepts_lr_solver_flags():
    mod = _load_scan_module()
    args = mod._parse_args(
        [
            "DCM:2",
            "--scan-tag",
            "test_jax_pme",
            "--lr-solver",
            "jax_pme",
            "--jax-pme-method",
            "pme",
            "--box-size",
            "36",
            "--mlpot-pbc",
            "--scan-1d",
            "--output",
            "/tmp/scan.npz",
        ]
    )
    assert args.composition == "DCM:2"
    assert args.scan_tag == "test_jax_pme"
    assert args.lr_solver == "jax_pme"
    assert args.jax_pme_method == "pme"
    assert args.scan_1d is True


def test_resolve_output_path_includes_scan_tag():
    mod = _load_scan_module()

    class NS:
        output = None
        output_dir = Path("/tmp/out")
        scan_tag = "pbc_jax_pme_ewald"
        scan_1d = True

    p = mod._resolve_output_path(NS(), tag="dcm_2", ckpt=Path("/ckpt/model"), batch=False)
    assert "pbc_jax_pme_ewald" in str(p)
    assert p.name == "scan_1d.npz"


def test_plot_dimer_lr_scan_compare_on_synthetic(tmp_path: Path):
    root = tmp_path / "scans"
    _write_synthetic_scan_npz(
        root / "ckpt" / "dcm_2" / "vacuum_mic" / "scan_1d.npz",
        composition="DCM:2",
        scan_tag="vacuum_mic",
    )
    _write_synthetic_scan_npz(
        root / "ckpt" / "dcm_2" / "pbc_jax_pme_ewald" / "scan_1d.npz",
        composition="DCM:2",
        scan_tag="pbc_jax_pme_ewald",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "plot_dimer_lr_scan_compare.py"),
            "--root",
            str(root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    plots = root / "plots"
    assert (plots / "dcm_2_lr_compare.png").is_file()
    index = json.loads((plots / "scan_index.json").read_text(encoding="utf-8"))
    assert len(index) == 2


def test_add_mlpot_lr_nonbond_args_on_parser():
    import argparse

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import add_mlpot_lr_nonbond_args

    p = argparse.ArgumentParser()
    add_mlpot_lr_nonbond_args(p)
    args = p.parse_args(["--lr-solver", "scafacos", "--mm-nonbond-mode", "periodic_external"])
    assert args.lr_solver == "scafacos"
    assert args.mm_nonbond_mode == "periodic_external"
