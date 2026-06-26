"""Unit tests for liquid-density NL geometry helpers."""

from __future__ import annotations

import numpy as np


def test_liquid_density_box_side_aco_n16() -> None:
    from tests.functionality.neighbor_lists._common import (
        build_liquid_density_synthetic_case,
        liquid_density_box_side_for_composition,
        liquid_density_synthetic_cases,
    )

    side, rho = liquid_density_box_side_for_composition(
        {"ACO": 16},
        bulk_density_fraction=1.0,
    )
    assert rho == 0.784
    assert 11.0 < side < 14.0

    case = {c["name"]: c for c in liquid_density_synthetic_cases()}["synthetic_aco_liquid_n16"]
    pos, cell, offsets, monomer_id, cutoff, _desc, side2, rho2 = (
        build_liquid_density_synthetic_case(case)
    )
    assert pos.shape[0] == 16 * 5
    assert monomer_id.shape[0] == pos.shape[0]
    assert offsets.shape[0] == 17
    assert abs(side2 - side) < 1e-6
    assert rho2 == rho
    assert cutoff == 13.0
    assert np.allclose(cell, np.diag([side2, side2, side2]))


def test_liquid_density_box25_derives_monomer_count() -> None:
    from tests.functionality.neighbor_lists._common import (
        _composition_dict_from_liquid_case,
        liquid_density_synthetic_cases,
    )

    case = {c["name"]: c for c in liquid_density_synthetic_cases()}["synthetic_aco_liquid_box25"]
    comp = _composition_dict_from_liquid_case(case)
    assert list(comp.keys()) == ["ACO"]
    assert comp["ACO"] == 32
