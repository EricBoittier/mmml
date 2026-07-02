"""Unit tests for protein CHARMM build helpers (no live PyCHARMM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_protein_toppar_paths_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.CHARMM_HOME",
        str(tmp_path),
    )
    from mmml.interfaces.pycharmmInterface.protein_charmm_build import protein_toppar_paths

    with pytest.raises(FileNotFoundError, match="Protein toppar"):
        protein_toppar_paths()


def test_alad_dataclass_fields() -> None:
    from mmml.interfaces.pycharmmInterface.protein_charmm_build import AladBuildResult

    pos = np.zeros((3, 3))
    result = AladBuildResult(positions=pos, n_atoms=3)
    assert result.n_atoms == 3
    assert result.segment == "ALAD"
