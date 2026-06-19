"""PyCHARMM integration tests write pdb/psf/packmol relative to cwd — isolate in tmp."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.functionality.pycharmmETC._paths import PYCHARMMETC_DIR

# Committed inputs copied into each isolated workdir when present.
_SEED_PDBS = ("initial.pdb", "init-packmol.pdb", "aco.pdb", "init-tip3.pdb", "tip3.pdb")


@pytest.fixture
def pycharmm_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary cwd with seed PDBs; PyCHARMM outputs stay out of the git tree."""
    for sub in ("pdb", "psf", "packmol", "res", "dcd", "xyz"):
        (tmp_path / sub).mkdir()
    for name in _SEED_PDBS:
        src = PYCHARMMETC_DIR / "pdb" / name
        if src.is_file():
            shutil.copy2(src, tmp_path / "pdb" / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path
