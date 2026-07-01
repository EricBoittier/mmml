"""PBC nbonds helpers must use KEY_LIBRARY C API (no ``nbonds`` script)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
def test_apply_pbc_nbonds_uses_nbonds_api() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds

    with patch(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_nbonds_kwargs"
    ) as mock_apply:
        cuts = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=32.0)
    mock_apply.assert_called_once()
    assert cuts.cubic_box_side_A == pytest.approx(32.0)


def test_restore_crystal_lattice_runs_image_byres_before_nbonds() -> None:
    from pathlib import Path

    src = Path(
        "mmml/interfaces/pycharmmInterface/mlpot/pbc_env.py"
    ).read_text(encoding="utf-8")
    body = src.split("def restore_charmm_cubic_crystal_lattice")[1].split("\ndef ")[0]
    assert "_image_setup_byres_all" in body
    assert body.index("_image_setup_byres_all") < body.index("apply_pbc_nbonds")


def test_apply_vacuum_nbonds_uses_nbonds_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prepare_charmm_vacuum",
        lambda: None,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_nbonds_kwargs"
    ) as mock_apply:
        apply_vacuum_nbonds(nbxmod=5)
    mock_apply.assert_called_once()
