"""Unit tests for monomer-aware GRMS thresholds and pre-MLpot geometry gate."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


def test_per_monomer_grms_from_forces():
    from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
        per_monomer_grms_from_forces,
    )

    forces = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    per = per_monomer_grms_from_forces(forces, [2, 2])
    assert per.shape == (2,)
    assert per[0] == pytest.approx(float(np.sqrt((9.0 + 16.0) / 6.0)))
    assert per[1] == pytest.approx(float(np.sqrt(3.0 / 6.0)))


def test_resolve_grms_thresholds_from_stats_scales_with_hybrid_tail():
    from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
        MonomerGrmsStats,
        resolve_grms_thresholds_from_stats,
    )

    stats = MonomerGrmsStats(
        charmm_per_monomer=np.array([1.0, 1.2, 1.1, 1.0]),
        hybrid_per_monomer=np.array([2.0, 3.0, 120.0, 2.5]),
        charmm_total=1.1,
        hybrid_total=40.0,
    )
    thresholds = resolve_grms_thresholds_from_stats(
        stats,
        n_monomers=4,
        n_atoms=20,
        pbc=True,
        base_max_grms=50.0,
    )
    assert thresholds.intervention_grms > 5.0
    assert thresholds.intervention_grms < thresholds.max_grms_before_dyn
    assert thresholds.max_grms_before_dyn >= 50.0


def test_resolve_grms_thresholds_caps_intervention_for_geometry_stress():
    from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
        MonomerGrmsStats,
        resolve_grms_thresholds_from_stats,
    )

    # Per-monomer tails can be huge while the live hybrid total is moderate (DCM:52 case).
    stats = MonomerGrmsStats(
        charmm_per_monomer=np.full(52, 0.13),
        hybrid_per_monomer=np.concatenate(
            [np.full(51, 2.0), np.array([601.72])]
        ),
        charmm_total=0.1346,
        hybrid_total=144.0367,
    )
    thresholds = resolve_grms_thresholds_from_stats(
        stats,
        n_monomers=52,
        n_atoms=260,
        pbc=True,
        base_max_grms=50.0,
    )
    assert thresholds.intervention_grms == pytest.approx(0.85 * 144.0367, rel=1e-3)
    assert thresholds.intervention_grms < 144.0367


def test_resilient_defaults_use_conservative_bulk_fraction_for_large_clusters():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
    )

    args = argparse.Namespace(
        liquid_prep=True,
        density_prep_ladder=None,
        box_size=None,
        target_density_g_cm3=None,
        bulk_density_fraction=None,
        composition="DCM:100",
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        mini_nstep=20,
        bonded_mm_mini_steps=200,
        mini_lattice_abnr_steps=0,
        mini_box_equil_ps=0.0,
        calculator_pre_minimize=True,
    )
    apply_density_prep_resilient_defaults(args)
    assert args.bulk_density_fraction == 0.55


def test_assert_pre_mlpot_intermonomer_geometry_aborts_on_overlap():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        assert_pre_mlpot_intermonomer_geometry,
    )

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    with pytest.raises(RuntimeError, match="inter-monomer atom overlap"):
        assert_pre_mlpot_intermonomer_geometry(
            positions,
            [2, 2],
            min_distance_A=1.0,
            box_side=20.0,
            use_pbc=False,
        )


def test_run_pre_mlpot_geometry_gate_disabled_without_liquid_prep():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        run_pre_mlpot_geometry_gate,
    )

    args = argparse.Namespace(liquid_prep=False, quiet=True)
    pos = np.zeros((4, 3), dtype=float)
    out_pos, side, summary = run_pre_mlpot_geometry_gate(
        args,
        positions=pos,
        atoms_per_list=[2, 2],
        composition={"DCM": 2},
        box_side=20.0,
        charmm_pbc=False,
        n_mol=2,
        n_atoms=4,
    )
    assert summary.enabled is False
    assert np.allclose(out_pos, pos)
    assert side == 20.0
