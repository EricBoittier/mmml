"""PBC nbonds helpers must use KEY_LIBRARY C API (no ``nbonds`` script)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
def test_apply_pbc_nbonds_uses_nbonds_api() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        pbc_nbond_cutoffs_from_mlpot_switches,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_nbonds_kwargs"
    ) as mock_apply:
        cuts = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=32.0)
    mock_apply.assert_called_once()
    assert cuts.cubic_box_side_A == pytest.approx(32.0)
    expected = pbc_nbond_cutoffs_from_mlpot_switches(
        32.0,
        mm_switch_on=DEFAULT_MM_SWITCH_ON,
        mm_switch_width=DEFAULT_MM_SWITCH_WIDTH,
    )
    assert cuts == expected


def test_restore_crystal_lattice_runs_prepare_pbc_before_nbonds() -> None:
    from pathlib import Path

    src = Path(
        "mmml/interfaces/pycharmmInterface/mlpot/pbc_env.py"
    ).read_text(encoding="utf-8")
    body = src.split("def restore_charmm_cubic_crystal_lattice")[1].split("\ndef ")[0]
    assert "prepare_charmm_pbc" in body
    assert body.index("prepare_charmm_pbc") < body.index("apply_pbc_nbonds")


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
