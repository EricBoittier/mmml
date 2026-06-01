"""Unit conversion for CHARMM MLpot vs ASE."""

from __future__ import annotations

import pytest

ase = pytest.importorskip("ase")

from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol
from mmml.models.physnetjax.physnetjax.calc.helper_mlp import pycharmm_conversion


def test_pycharmm_conversion_matches_ev2kcalmol():
    assert pycharmm_conversion["energy"] == pytest.approx(ev2kcalmol, rel=1e-6)
    assert pycharmm_conversion["forces"] == pytest.approx(ev2kcalmol, rel=1e-6)
