"""Unit tests for embedding hybrid setup (no PyCHARMM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.embedding_hybrid import (
    EmbeddingValidationResult,
    export_embedding_checkpoint,
    validate_embedding_monomer_potential,
)


def test_tria_psf_atom_names_must_not_map_as_cgenff_types() -> None:
    """Regression: embedding used get_atype() names with a type→Z table.

    TRIA atom names like HY1 / CAY are not CGenFF types (HGA3 / CG331). Looking
    them up in a type table defaults to Z=6 and yields ~1e68 Spooky energies.
    Mass-based Z (get_Z_from_psf) is required.
    """
    import ase.data

    # ACE methyl: CAY (CG331) + HY1–3 (HGA3) — names as CHARMM stores them
    names = ["CAY", "HY1", "HY2", "HY3"]
    masses = np.array([12.011, 1.008, 1.008, 1.008], dtype=float)
    type_table = {
        "HGA3": 1,
        "CG331": 6,
        "H": 1,
        "C": 6,
    }
    z_from_names = [type_table.get(n.upper(), 6) for n in names]
    ase_m = ase.data.atomic_masses_common
    z_from_mass = [int(np.argmin((ase_m - float(m)) ** 2)) for m in masses]
    assert z_from_names == [6, 6, 6, 6]
    assert z_from_mass == [6, 1, 1, 1]


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
    assert orbax_fn.call_args.kwargs["params_key"] == "ema_params"


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
