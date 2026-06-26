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


def test_higher_density_smaller_box_than_bulk() -> None:
    from tests.functionality.neighbor_lists._common import (
        build_liquid_density_synthetic_case,
        effective_mass_density_g_cm3,
        liquid_density_synthetic_cases,
    )

    cases = {c["name"]: c for c in liquid_density_synthetic_cases()}
    _, _, _, _, _, _, side_bulk, _ = build_liquid_density_synthetic_case(
        cases["synthetic_aco_liquid_n32"]
    )
    _, _, _, _, _, _, side_150, rho_150 = build_liquid_density_synthetic_case(
        cases["synthetic_aco_liquid_n32_rho150"]
    )
    comp = {"ACO": 32}
    assert side_150 < side_bulk
    assert effective_mass_density_g_cm3(comp, side_150) > rho_150 * 0.99


def test_motion_step_box_scale() -> None:
    from tests.functionality.neighbor_lists._common import (
        apply_motion_step,
        motion_stress_steps,
    )

    pos = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    cell = 10.0 * np.eye(3)
    step = next(s for s in motion_stress_steps() if s["name"] == "box_shrink_0.97")
    new_pos, new_cell = apply_motion_step(pos, cell, step, rng=np.random.default_rng(0))
    assert abs(float(new_cell[0, 0]) - 9.7) < 1e-9
    assert abs(float(new_pos[0, 0]) - 0.97) < 1e-9


def test_liquid_density_box25_derives_monomer_count() -> None:
    from tests.functionality.neighbor_lists._common import (
        _composition_dict_from_liquid_case,
        liquid_density_synthetic_cases,
    )

    case = {c["name"]: c for c in liquid_density_synthetic_cases()}["synthetic_aco_liquid_box25"]
    comp = _composition_dict_from_liquid_case(case)
    assert list(comp.keys()) == ["ACO"]
    assert comp["ACO"] == 32
