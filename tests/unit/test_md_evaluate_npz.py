"""Unit tests for md-system --evaluate-npz."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.cli.run.md_evaluate_npz import (
    EvaluateNpzPayload,
    load_evaluate_npz,
    resolve_evaluate_use_pbc,
    run_evaluate_npz,
)
from mmml.cli.run.md_handoff import MdHandoffState


def test_load_evaluate_npz_optional_mm_fields(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).random((6, 3))
    z = np.array([6, 1, 1, 6, 1, 1], dtype=np.int32)
    charges = np.linspace(-0.3, 0.3, 6)
    iac = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
    path = tmp_path / "geom.npz"
    np.savez_compressed(
        path,
        positions=pos,
        atomic_numbers=z,
        pbc=False,
        metadata=json.dumps({"source": "test"}),
        charges=charges,
        iac=iac,
    )

    payload = load_evaluate_npz(path)
    assert isinstance(payload, EvaluateNpzPayload)
    assert payload.charges is not None
    assert payload.at_codes is not None
    assert int(payload.at_codes.min()) == 0
    np.testing.assert_allclose(payload.charges, charges)
    np.testing.assert_allclose(payload.handoff.positions, pos)


def test_load_evaluate_npz_positions_only(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).random((10, 3))
    path = tmp_path / "geom.npz"
    np.savez_compressed(path, positions=pos, pbc=False, metadata="{}")

    payload = load_evaluate_npz(path)
    np.testing.assert_allclose(payload.handoff.positions, pos)
    assert payload.handoff.atomic_numbers.size == 0


def test_evaluate_int_arg_treats_none_as_default() -> None:
    from mmml.cli.run.md_evaluate_npz import _evaluate_int_arg

    args = Namespace(max_pairs=None, jax_md_update_interval=None)
    assert _evaluate_int_arg(args, "max_pairs", 20_000) == 20_000
    assert _evaluate_int_arg(args, "jax_md_update_interval", 1) == 1
    assert _evaluate_int_arg(Namespace(), "max_pairs", 20_000) == 20_000


def test_resolve_evaluate_use_pbc_from_setup() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.ones(3, dtype=int),
        pbc=False,
    )
    assert resolve_evaluate_use_pbc(Namespace(setup="free_nve"), handoff) is False
    assert resolve_evaluate_use_pbc(Namespace(setup="pbc_nvt"), handoff) is True


def test_md_system_parser_accepts_evaluate_npz() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(
        [
            "--evaluate-npz",
            "/tmp/geom.npz",
            "--composition",
            "DCM:2",
            "--backend",
            "ase",
            "--setup",
            "free_nve",
            "--evaluate-frame",
            "2",
        ]
    )
    assert Path(args.evaluate_npz) == Path("/tmp/geom.npz")
    assert args.evaluate_frame == 2


def test_md_system_parser_accepts_optimize_cutoffs() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(
        [
            "--optimize-cutoffs",
            "--reference-npz",
            "/tmp/traj.npz",
            "--composition",
            "DCM:2",
            "--ml-cutoff-grid",
            "1.5,2.0",
            "--max-frames",
            "10",
        ]
    )
    assert args.optimize_cutoffs is True
    assert Path(args.reference_npz) == Path("/tmp/traj.npz")
    assert args.ml_switch_width_grid == "1.5,2.0"


def test_run_evaluate_npz_ase_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    pos = np.zeros((4, 3))
    z = np.array([6, 1, 1, 1], dtype=np.int32)
    npz = tmp_path / "state.npz"
    np.savez_compressed(
        npz,
        positions=pos,
        atomic_numbers=z,
        pbc=False,
        metadata="{}",
    )
    out_dir = tmp_path / "out"
    args = Namespace(
        evaluate_npz=npz,
        evaluate_output=None,
        output_dir=out_dir,
        composition="ACO:1",
        n_molecules=1,
        checkpoint=None,
        backend="ase",
        setup="free_nve",
        box_size=None,
        handoff_require_cell=False,
        quiet=True,
        ml_cutoff=0.1,
        jax_md_capacity_multiplier=1.25,
        jax_md_capacity_growth_factor=1.5,
        jax_md_max_overflow_retries=4,
        jax_md_disable_fallback=False,
        jax_md_update_interval=1,
        jax_md_skin_distance=0.2,
        max_pairs=1000,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        min_com_restraint_distance=None,
        min_com_restraint_k=1.0,
        ml_batch_size=None,
        ml_max_active_dimers=None,
        ml_compute_dtype=None,
        verbose_calc=False,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        mm_cutoff=5.0,
    )

    fake_atoms = MagicMock()
    fake_atoms.get_positions.return_value = pos
    fake_ase_mod = MagicMock()
    fake_ase_mod._cubic_box_length.return_value = 40.0
    fake_ase_mod._parse_composition.return_value = [("ACO", 1)]

    with (
        patch.dict(sys.modules, {"mmml.cli.run.md_pbc_suite.ase": fake_ase_mod}),
        patch(
            "mmml.cli.run.md_evaluate_npz.cluster_geometry_from_handoff",
            return_value=(z, pos, [4], ["ACO"], {"ACO": 1}),
        ),
        patch("mmml.cli.run.md_evaluate_npz.ensure_psf_for_handoff_cluster"),
        patch(
            "mmml.cli.run.md_evaluate_npz._build_atoms_for_evaluate",
            return_value=fake_atoms,
        ),
        patch(
            "mmml.cli.run.md_evaluate_npz._evaluate_ase_mmml",
            return_value={
                "energy_eV": -1.25,
                "max_force_eV_A": 0.01,
                "rms_force_eV_A": 0.01,
                "n_atoms": 4,
                "n_monomers": 1,
                "pbc": False,
                "box_A": None,
            },
        ),
        patch(
            "mmml.cli.base.resolve_checkpoint_paths",
            return_value=(Path("/tmp/ckpt"), Path("/tmp/ckpt")),
        ),
    ):
        code = run_evaluate_npz(args)

    assert code == 0
    result_path = out_dir / "evaluate.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "ase"
    assert payload["metrics"]["energy_eV"] == pytest.approx(-1.25)


def test_md_system_evaluate_npz_manifest_exit_code_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from mmml.cli.run import md_system

    out_dir = tmp_path / "eval_smoke"
    out_dir.mkdir()
    npz = tmp_path / "geom.npz"
    np.savez_compressed(npz, positions=np.zeros((10, 3)), pbc=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml",
            "--evaluate-npz",
            str(npz),
            "--composition",
            "DCM:2",
            "--backend",
            "ase",
            "--setup",
            "free_nve",
            "--output-dir",
            str(out_dir),
        ],
    )
    monkeypatch.setattr("mmml.cli.run.md_evaluate_npz.run_evaluate_npz", lambda _args: 0)

    assert md_system.main() == 0

    manifest_path = tmp_path / "artifacts" / "md_system" / "jobs" / "eval_smoke.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["exit_code"] == 0
    assert manifest["backend"] == "ase"
