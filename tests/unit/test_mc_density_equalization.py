"""Unit tests for post-build MC density equalization."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        mc_density_equalize=True,
        mc_density_target_g_cm3=None,
        mc_density_steps=80,
        mc_density_step_scale=0.02,
        mc_density_temperature=0.02,
        mc_density_seed=7,
        mc_density_min_scale=0.5,
        mc_density_max_scale=2.0,
        target_density_g_cm3=None,
        bulk_density_fraction=None,
        box_size=None,
        composition="DCM:2",
        seed=123,
        min_intermonomer_atom_distance=0.1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_mc_density_target_defaults_to_single_residue_bulk_table():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        resolve_mc_density_target_g_cm3,
    )

    target, source = resolve_mc_density_target_g_cm3(_args(), {"DCM": 2})
    assert target == pytest.approx(1.326)
    assert source == "bulk_density_table"


def test_mc_density_equalization_moves_box_toward_target_density():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
        density_g_cm3_for_box,
    )

    args = _args(mc_density_target_g_cm3=1.326)
    # Two one-atom "monomers" in a loose 12 A box; DCM mass makes this underdense.
    pos = np.array([[5.0, 6.0, 6.0], [7.0, 6.0, 6.0]], dtype=float)
    initial_density = density_g_cm3_for_box({"DCM": 2}, 12.0)
    new_pos, new_L, summary = apply_mc_density_equalization(
        args,
        pos,
        atoms_per_list=[1, 1],
        composition={"DCM": 2},
        box_side_A=12.0,
        use_pbc=True,
    )

    assert summary.ran
    assert summary.attempted_moves == 80
    assert new_L is not None
    assert new_L < 12.0
    assert summary.final_density_g_cm3 > initial_density
    assert abs(summary.final_density_g_cm3 - 1.326) < abs(initial_density - 1.326)
    assert new_pos.shape == pos.shape


def test_mc_density_equalization_preserves_intramonomer_geometry():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )

    args = _args(mc_density_target_g_cm3=1.326)
    pos = np.array(
        [
            [4.5, 6.0, 6.0],
            [5.5, 6.0, 6.0],
            [7.5, 6.0, 6.0],
            [8.5, 6.0, 6.0],
        ],
        dtype=float,
    )
    new_pos, _new_L, summary = apply_mc_density_equalization(
        args,
        pos,
        atoms_per_list=[2, 2],
        composition={"DCM": 2},
        box_side_A=12.0,
        use_pbc=True,
    )

    assert summary.ran
    assert np.linalg.norm(new_pos[1] - new_pos[0]) == pytest.approx(1.0)
    assert np.linalg.norm(new_pos[3] - new_pos[2]) == pytest.approx(1.0)


def test_mc_density_equalization_skips_fixed_box_by_default():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )

    args = _args(box_size=12.0)
    pos = np.array([[5.0, 6.0, 6.0], [7.0, 6.0, 6.0]], dtype=float)
    new_pos, new_L, summary = apply_mc_density_equalization(
        args,
        pos,
        atoms_per_list=[1, 1],
        composition={"DCM": 2},
        box_side_A=12.0,
        use_pbc=True,
    )

    assert not summary.ran
    assert summary.reason == "fixed_box"
    assert new_L == 12.0
    np.testing.assert_allclose(new_pos, pos)


def test_mc_density_equalization_skips_unknown_mixed_density_without_target():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )

    args = _args(composition="UNK:1,DCM:1")
    pos = np.array([[5.0, 6.0, 6.0], [7.0, 6.0, 6.0]], dtype=float)
    new_pos, new_L, summary = apply_mc_density_equalization(
        args,
        pos,
        atoms_per_list=[1, 1],
        composition={"UNK": 1, "DCM": 1},
        box_side_A=12.0,
        use_pbc=True,
    )

    assert not summary.ran
    assert summary.reason == "no_density_target"
    assert new_L == 12.0
    np.testing.assert_allclose(new_pos, pos)


def test_mc_density_equalization_skips_unknown_mass_even_with_target():
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )

    args = _args(composition="UNK:1", mc_density_target_g_cm3=1.0)
    pos = np.array([[5.0, 6.0, 6.0]], dtype=float)
    new_pos, new_L, summary = apply_mc_density_equalization(
        args,
        pos,
        atoms_per_list=[1],
        composition={"UNK": 1},
        box_side_A=12.0,
        use_pbc=True,
    )

    assert not summary.ran
    assert summary.reason == "no_mass_metadata"
    assert new_L == 12.0
    np.testing.assert_allclose(new_pos, pos)
