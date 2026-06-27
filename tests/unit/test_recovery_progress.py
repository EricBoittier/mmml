"""Unit tests for prep_ladder/ and cleanup/ recovery artifact folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
    CLEANUP_SUBDIR,
    PREP_LADDER_SUBDIR,
    RecoveryProgressStore,
    resolve_cleanup_dir,
    resolve_prep_ladder_dir,
    slugify_step_label,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        output_dir=Path("/tmp/mmml_run"),
        prep_ladder_dir=PREP_LADDER_SUBDIR,
        cleanup_dir=CLEANUP_SUBDIR,
        no_recovery_artifacts=False,
        quiet=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_prep_ladder_dir_default_subfolder(tmp_path: Path) -> None:
    out = tmp_path / "job"
    args = _args(output_dir=out)
    assert resolve_prep_ladder_dir(args) == (out / PREP_LADDER_SUBDIR).resolve()


def test_resolve_cleanup_dir_custom_name(tmp_path: Path) -> None:
    out = tmp_path / "job"
    args = _args(output_dir=out, cleanup_dir="geometry_cleanup")
    assert resolve_cleanup_dir(args) == (out / "geometry_cleanup").resolve()


def test_no_recovery_artifacts_disables_dirs(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path / "job", no_recovery_artifacts=True)
    assert resolve_prep_ladder_dir(args) is None
    assert resolve_cleanup_dir(args) is None


def test_slugify_step_label() -> None:
    assert slugify_step_label("round1:monomer_repack") == "round1_monomer_repack"
    assert slugify_step_label("  ") == "step"


def test_recovery_progress_store_writes_journal_and_latest(tmp_path: Path) -> None:
    root = tmp_path / "prep_ladder"
    store = RecoveryProgressStore(root, "prep_ladder", "Test ladder", quiet=True)

    fake_written = {
        "pdb": root / "001_initial.pdb",
        "crd": root / "001_initial.crd",
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.recovery_progress._save_geometry_checkpoint",
        return_value=fake_written,
    ) as save_mock:
        store.record_step("initial", grms_kcalmol_A=42.0, box_side_A=30.0)
        save_mock.assert_called_once()

    assert (root / "journal.json").is_file()
    journal = json.loads((root / "journal.json").read_text(encoding="utf-8"))
    assert journal["category"] == "prep_ladder"
    assert len(journal["steps"]) == 1
    assert journal["steps"][0]["label"] == "initial"
    assert journal["steps"][0]["hybrid_grms_kcalmol_A"] == pytest.approx(42.0)

    store.finish({"ok": True})
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["ok"] is True
    assert summary["steps"][0]["stem"] == "001_initial"


def test_for_prep_ladder_factory_returns_none_without_output_dir() -> None:
    assert RecoveryProgressStore.for_prep_ladder(argparse.Namespace()) is None


def test_build_pycharmm_command_forwards_recovery_artifact_flags() -> None:
    from mmml.cli.run.md_system import build_pycharmm_command
    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    cmd = build_pycharmm_command(
        _pycharmm_args(
            prep_ladder_dir="my_ladder",
            cleanup_dir="my_cleanup",
            no_recovery_artifacts=True,
        )
    )
    idx = cmd.index("--prep-ladder-dir")
    assert cmd[idx + 1] == "my_ladder"
    idx = cmd.index("--cleanup-dir")
    assert cmd[idx + 1] == "my_cleanup"
    assert "--no-recovery-artifacts" in cmd
