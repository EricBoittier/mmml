"""CRD parsing for validate_mlpot_sparse_dimers (PyCHARMM EXT cards)."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO
    / "tests/functionality/mlpot/output/minimize/mini_full_mlpot.crd"
)


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="mlpot minimize fixture CRD missing")
def test_load_positions_crd_pycharmm_ext():
    from scripts.validate_mlpot_sparse_dimers import _load_positions_crd

    pos = _load_positions_crd(_FIXTURE)
    assert pos.shape == (20, 3)
    assert pos[0, 0] == pytest.approx(0.3308962950)


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="mlpot minimize fixture CRD missing")
def test_validate_script_accepts_pycharmm_crd():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts/validate_mlpot_sparse_dimers.py"),
            "--crd",
            str(_FIXTURE),
            "--n-monomers",
            "2",
            "--atoms-per-monomer",
            "10",
            "--mm-switch-on",
            "7.0",
            "--free-space",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stderr[-500:]
