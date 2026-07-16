"""EXT CRD parse used when loading certified liquid-box artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    read_crd_coordinates,
)


def test_read_crd_coordinates_ext_liquid_box_style(tmp_path: Path) -> None:
    crd = tmp_path / "model.crd"
    crd.write_text(
        "\n".join(
            [
                "* liquid-box certified",
                "*",
                "          2  EXT",
                "         1         1  DCM   C            -1.0000000000       -2.0000000000       -3.0000000000  CLST           1          1.0000000000",
                "         2         1  DCM   H1           -1.1000000000       -2.1000000000       -3.1000000000  CLST           1          1.0000000000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pos = read_crd_coordinates(crd)
    assert pos is not None
    assert pos.shape == (2, 3)
    assert pos[0, 0] == pytest.approx(-1.0)
    assert pos[1, 2] == pytest.approx(-3.1)


def test_read_crd_coordinates_live_dcm27_box_if_present() -> None:
    path = Path.home() / "tests/boxes/dcm27_rho100/model.crd"
    if not path.is_file():
        pytest.skip("certified dcm27 box not present")
    pos = read_crd_coordinates(path)
    assert pos is not None
    assert pos.shape[0] == 160
    assert pos.shape[1] == 3
