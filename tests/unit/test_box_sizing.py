"""Unit tests for PBC box sizing helpers."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


def test_cubic_box_side_from_target_density_dcm60():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        cubic_box_length_from_geometry,
        cubic_box_side_from_target_density,
        total_mass_g_for_composition,
    )

    comp = {"DCM": 60}
    mass = total_mass_g_for_composition(comp)
    side = cubic_box_side_from_target_density(
        n_molecules=60,
        total_mass_g=mass,
        target_density_g_cm3=1.326,
    )
    pos = np.zeros((10, 3))
    floor = cubic_box_length_from_geometry(pos, ml_cutoff=12.0)
    assert side > 15.0
    assert cubic_box_side_from_target_density(
        n_molecules=60,
        total_mass_g=mass,
        target_density_g_cm3=1.326,
        min_side_A=floor,
    ) >= floor


def test_resolve_initial_pbc_box_side_density_mode():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_initial_pbc_box_side,
    )

    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    args = argparse.Namespace(
        box_size=None,
        box_auto="density",
        target_density_g_cm3=1.326,
        bulk_density_fraction=None,
        composition="DCM:8",
        n_molecules=8,
        ml_cutoff=12.0,
    )
    side, source = resolve_initial_pbc_box_side(args, pos)
    assert source == "density"
    assert side > 15.0


def test_resolve_density_packmol_cube_side_from_composition():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_density_packmol_cube_side,
    )

    args = argparse.Namespace(
        box_auto="density",
        target_density_g_cm3=0.9,
        bulk_density_fraction=None,
        composition="DCM:100",
        n_molecules=100,
    )
    side = resolve_density_packmol_cube_side(args)
    assert 20.0 < side < 30.0


def test_resolve_packmol_cube_side_from_args_uses_density_auto():
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_cube_side_from_args,
    )

    args = argparse.Namespace(
        box_size=None,
        packmol_box_size=None,
        packmol_box_padding=None,
        packmol_radius=None,
        flat_bottom_radius=None,
        packmol_tolerance=2.0,
        spacing=5.0,
        ml_cutoff=12.0,
        quiet=True,
        box_auto="density",
        target_density_g_cm3=0.9,
        bulk_density_fraction=None,
        composition="DCM:100",
        n_molecules=100,
    )
    side = resolve_packmol_cube_side_from_args(args)
    sim = float(args._cold_start_sim_cell_side_A)
    assert side < sim
    assert 10.0 < side < 30.0
    assert 20.0 < sim < 35.0


def test_resolve_packmol_cube_side_smaller_than_explicit_sim_cell():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_packmol_cube_side_for_sim_cell,
    )

    args = argparse.Namespace(
        packmol_box_size=None,
        packmol_box_padding=10.0,
        packmol_tolerance=2.0,
        spacing=5.0,
        ml_cutoff=12.0,
        box_auto=None,
    )
    sim = 80.0
    packmol = resolve_packmol_cube_side_for_sim_cell(args, sim)
    assert packmol == pytest.approx(60.0)
    assert packmol < sim


def test_cubic_side_from_cluster_extent_uses_max_axis_span():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        cubic_side_from_cluster_extent,
    )

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [30.0, 5.0, 2.0],
            [1.0, 20.0, 3.0],
        ],
        dtype=float,
    )
    side = cubic_side_from_cluster_extent(pos, margin_A=5.0, ml_cutoff=12.0)
    assert side == pytest.approx(40.0)


def test_resolve_packmol_cube_side_from_args_explicit_box_size():
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_cube_side_from_args,
    )

    args = argparse.Namespace(
        box_size=80.0,
        packmol_box_size=None,
        packmol_box_padding=10.0,
        packmol_radius=None,
        flat_bottom_radius=None,
        packmol_tolerance=2.0,
        spacing=5.0,
        ml_cutoff=12.0,
        quiet=True,
        box_auto=None,
        composition="DCM:200",
        n_molecules=200,
    )
    side = resolve_packmol_cube_side_from_args(args)
    assert args._cold_start_sim_cell_side_A == pytest.approx(80.0)
    assert side == pytest.approx(60.0)


def test_resolve_packmol_box_padding_defaults_small_for_fixed_box_composition():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        FIXED_BOX_COMPOSITION_PACKMOL_PADDING_A,
        resolve_packmol_box_padding_A,
        resolve_packmol_cube_side_for_sim_cell,
    )

    args = argparse.Namespace(
        packmol_box_padding=None,
        box_size=30.0,
        composition="DCM:127",
        n_molecules=127,
        packmol_tolerance=1.0,
        spacing=4.0,
        ml_cutoff=12.0,
        box_auto=None,
    )
    assert resolve_packmol_box_padding_A(args) == pytest.approx(
        FIXED_BOX_COMPOSITION_PACKMOL_PADDING_A
    )
    packmol = resolve_packmol_cube_side_for_sim_cell(args, 30.0)
    assert packmol == pytest.approx(28.0)
    assert packmol > 25.0


def test_n_molecules_for_target_density_in_fixed_box_dcm32():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        n_molecules_for_target_density_in_fixed_box,
        total_mass_g_for_composition,
    )

    scaled = n_molecules_for_target_density_in_fixed_box(
        composition={"DCM": 1},
        box_side_A=32.0,
        target_density_g_cm3=1.326,
    )
    assert scaled == {"DCM": 308}
    mass = total_mass_g_for_composition(scaled)
    vol_cm3 = (32.0 * 1.0e-8) ** 3
    assert mass / vol_cm3 == pytest.approx(1.326, rel=0.01)


def test_apply_box_auto_count_composition_mutates_args():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        apply_box_auto_count_composition,
    )

    args = argparse.Namespace(
        box_auto="count",
        box_size=32.0,
        target_density_g_cm3=1.326,
        bulk_density_fraction=None,
        composition="DCM:60",
        box_auto_count_min_molecules=1,
        box_auto_count_max_molecules=None,
        quiet=True,
    )
    scaled = apply_box_auto_count_composition(args)
    assert scaled["DCM"] == 308
    assert args.composition == "DCM:308"
    assert args.n_molecules == 308


def test_box_auto_count_mixed_stoichiometry():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        n_molecules_for_target_density_in_fixed_box,
        total_mass_g_for_composition,
    )

    scaled = n_molecules_for_target_density_in_fixed_box(
        composition={"DCM": 2, "ACO": 1},
        box_side_A=28.0,
        target_density_g_cm3=0.9,
    )
    assert scaled["DCM"] == 2 * scaled["ACO"]
    mass = total_mass_g_for_composition(scaled)
    vol_cm3 = (28.0 * 1.0e-8) ** 3
    assert mass / vol_cm3 == pytest.approx(0.9, rel=0.02)


def test_resolve_initial_pbc_box_side_explicit_box_size():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_initial_pbc_box_side,
    )

    args = argparse.Namespace(box_size=40.0, box_auto=None)
    side, source = resolve_initial_pbc_box_side(
        args,
        np.zeros((3, 3)),
    )
    assert side == 40.0
    assert source == "explicit"


def test_bulk_density_fraction_requires_single_species():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_target_density_g_cm3,
    )

    with pytest.raises(ValueError, match="single-species"):
        resolve_target_density_g_cm3(
            argparse.Namespace(
                target_density_g_cm3=None,
                bulk_density_fraction=0.85,
            ),
            {"DCM": 4, "MEOH": 4},
        )


def test_should_run_mini_box_equil_skips_when_pretreat_npt():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        should_run_mini_box_equil,
    )

    args = argparse.Namespace(
        mini_box_equil_ps=5.0,
        box_size=None,
        mini_box_equil_allow_fixed_box=False,
        charmm_mm_pretreat_ps_equi=10.0,
    )
    assert not should_run_mini_box_equil(
        args,
        charmm_pbc=True,
        pretreat_mm=True,
        stages=["mini", "heat"],
    )


def test_should_run_mini_box_equil_true_for_pbc_mini():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        should_run_mini_box_equil,
    )

    args = argparse.Namespace(
        mini_box_equil_ps=2.0,
        box_size=None,
        mini_box_equil_allow_fixed_box=False,
        charmm_mm_pretreat_ps_equi=0.0,
    )
    assert should_run_mini_box_equil(
        args,
        charmm_pbc=True,
        pretreat_mm=False,
        stages=["mini", "heat"],
    )


def test_resolve_mini_box_equil_durations_splits_total_evenly():
    from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
        resolve_mini_box_equil_durations_ps,
    )

    args = argparse.Namespace(mini_box_equil_ps=200.0)
    assert resolve_mini_box_equil_durations_ps(args) == pytest.approx((100.0, 100.0))


def test_resolve_mini_box_equil_durations_explicit_legs():
    from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
        resolve_mini_box_equil_durations_ps,
    )

    args = argparse.Namespace(
        mini_box_equil_ps=200.0,
        mini_box_equil_ps_heat=120.0,
        mini_box_equil_ps_cool=80.0,
    )
    assert resolve_mini_box_equil_durations_ps(args) == pytest.approx((120.0, 80.0))


def test_resolve_mini_box_equil_hot_temp_default():
    from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
        resolve_mini_box_equil_hot_temp_K,
    )

    args = argparse.Namespace(mini_box_equil_hot_temp=None)
    assert resolve_mini_box_equil_hot_temp_K(args, target_K=300.0) == pytest.approx(450.0)


def test_resolve_mini_box_equil_hot_temp_explicit():
    from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
        resolve_mini_box_equil_hot_temp_K,
    )

    args = argparse.Namespace(mini_box_equil_hot_temp=500.0)
    assert resolve_mini_box_equil_hot_temp_K(args, target_K=300.0) == pytest.approx(500.0)


def test_apply_certified_box_size_from_box_json(tmp_path) -> None:
    import json

    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        apply_certified_box_size_from_artifacts,
    )

    box_dir = tmp_path / "boxes" / "dcm103"
    box_dir.mkdir(parents=True)
    (box_dir / "box.json").write_text(
        json.dumps({"box_side_A": 55.229, "composition": "DCM:103"}),
        encoding="utf-8",
    )
    psf = box_dir / "model.psf"
    psf.write_text("stub", encoding="utf-8")
    args = argparse.Namespace(
        from_psf=str(psf),
        from_crd=str(box_dir / "model.crd"),
        quiet=True,
    )
    side = apply_certified_box_size_from_artifacts(args)
    assert side == pytest.approx(55.229)
    assert args.box_size == pytest.approx(55.229)
