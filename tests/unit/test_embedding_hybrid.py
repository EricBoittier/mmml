"""Unit tests for embedding hybrid setup (no PyCHARMM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.embedding_hybrid import (
    EmbeddingValidationResult,
    export_embedding_checkpoint,
    validate_embedding_monomer_potential,
)


def test_export_embedding_checkpoint_calls_orbax_to_json(tmp_path: Path) -> None:
    epoch = tmp_path / "epoch-49"
    epoch.mkdir()
    out_json = tmp_path / "params.json"
    with mock.patch(
        "mmml.utils.model_checkpoint.orbax_to_json",
        return_value=out_json,
    ) as orbax_fn, mock.patch(
        "mmml.cli.base.resolve_checkpoint_paths",
        return_value=(tmp_path, epoch),
    ):
        path = export_embedding_checkpoint(epoch, out_json)
    assert path == out_json
    orbax_fn.assert_called_once()


def test_validate_embedding_monomer_potential_parses_metrics(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text("{}", encoding="utf-8")
    valid = tmp_path / "valid.npz"
    valid.write_bytes(b"")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "metrics.json").write_text(
        json.dumps(
            {
                "energy": {"mae_kcal_mol": 0.42},
                "forces": {"mae_kcal_mol": 1.23},
            }
        ),
        encoding="utf-8",
    )
    with mock.patch("subprocess.run") as run_fn:
        result = validate_embedding_monomer_potential(ckpt, valid, eval_dir, repo_root=tmp_path)
    run_fn.assert_called_once()
    assert isinstance(result, EmbeddingValidationResult)
    assert result.energy_mae_kcal_mol == pytest.approx(0.42)
    assert result.force_mae_kcal_mol_A == pytest.approx(1.23)
