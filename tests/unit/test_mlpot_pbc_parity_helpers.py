"""Unit tests for PBC parity helpers (LR defaults, image dimers, MM eterm split)."""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    decompose_mlpot_mm_nb_eterms_kcalmol,
)
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_jax_pme_sr_cutoff_for_mlpot,
    resolve_lr_solver_for_mlpot,
    resolve_mlpot_use_pbc,
    warn_if_mic_pbc_without_lr,
)
from mmml.interfaces.pycharmmInterface.mlpot.charmm_eterm_routing import (
    route_mlpot_callback_energy_kcalmol,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    image_aware_dimer_com_distance_numpy,
    mic_displacement_numpy,
)


def test_resolve_lr_solver_defaults_jax_pme_for_pbc_setup():
    args = argparse.Namespace(
        setup="pbc_nve",
        lr_solver=None,
        free_space=False,
        mlpot_pbc=False,
    )
    assert resolve_mlpot_use_pbc(args) is True
    assert resolve_lr_solver_for_mlpot(args, mlpot_pbc=True, mm_nonbond_mode="jax_mic") == "jax_pme"


def test_resolve_lr_solver_explicit_mic_wins():
    args = argparse.Namespace(lr_solver="mic")
    assert resolve_lr_solver_for_mlpot(args, mlpot_pbc=True) == "mic"


def test_jax_pme_sr_cutoff_matches_switched_mm_outer_edge():
    cp = CutoffParameters(mm_switch_on=8.0, mm_switch_width=5.0)
    assert resolve_jax_pme_sr_cutoff_for_mlpot(None, cp) == pytest.approx(13.0)


def test_warn_if_mic_pbc_without_lr_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_mic_pbc_without_lr(lr_solver="mic", mlpot_pbc=True)
    assert any("truncated MIC Coulomb" in str(w.message) for w in caught)


def test_image_aware_dimer_com_distance_across_box_face():
    side = 20.0
    cell = np.diag([side, side, side])
    # Monomer A near +x face, monomer B near -x face (MIC distance ~2 Å)
    pos = np.zeros((6, 3), dtype=np.float64)
    pos[:3, 0] = side * 0.5 - 0.5
    pos[3:, 0] = -side * 0.5 + 0.5
    d_mic = float(np.linalg.norm(mic_displacement_numpy(pos[0], pos[3], cell)))
    d_img = image_aware_dimer_com_distance_numpy(
        pos, np.arange(6, dtype=np.int32), 3, 3, cell
    )
    assert d_img == pytest.approx(1.0, abs=0.05)
    assert d_mic == pytest.approx(1.0, abs=0.05)


def test_decompose_mm_nb_primary_vs_image_buckets():
    side = 30.0
    cell = np.diag([side, side, side])
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [side - 1.0, 0.0, 0.0],
            [side + 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pair_idx = np.array([[0, 2], [1, 3]], dtype=np.int32)
    pair_mask = np.array([True, True], dtype=bool)
    charges = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
    rmins = np.full(4, 1.8, dtype=np.float64)
    eps = np.full(4, 0.1, dtype=np.float64)
    monomer_id = np.array([0, 0, 1, 1], dtype=np.int32)
    out = decompose_mlpot_mm_nb_eterms_kcalmol(
        pos,
        pair_idx,
        pair_mask,
        cell,
        charges_e=charges,
        rmins_A=rmins,
        epsilons_kcal=eps,
        monomer_id=monomer_id,
        mm_switch_on=20.0,
        mm_switch_width=5.0,
    )
    assert out["mm_total"] == pytest.approx(
        out["vdw_primary"] + out["vdw_image"] + out["elec_primary"] + out["elec_image"]
    )
    assert out["vdw_primary"] + out["elec_primary"] >= 0.0
    assert out["vdw_image"] + out["elec_image"] >= 0.0


def test_route_mlpot_callback_energy_subtracts_mm_from_user(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ROUTE_MM_ETERMS", "1")
    components = {
        "vdw_primary": 1.0,
        "vdw_image": 0.5,
        "elec_primary": 2.0,
        "elec_image": 0.25,
        "mm_total": 3.75,
    }
    e_user = route_mlpot_callback_energy_kcalmol(10.0, components, route=True)
    assert e_user == pytest.approx(6.25)


def test_route_mlpot_callback_keeps_user_when_routing_would_zero_hybrid(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ROUTE_MM_ETERMS", "1")
    components = {
        "vdw_primary": -15000.0,
        "vdw_image": -10000.0,
        "elec_primary": -6000.0,
        "elec_image": -729.4,
        "mm_total": -31729.4,
    }
    e_user = route_mlpot_callback_energy_kcalmol(-31729.4, components, route=True)
    assert e_user == pytest.approx(-31729.4)
