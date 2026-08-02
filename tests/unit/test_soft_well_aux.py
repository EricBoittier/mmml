"""Unit tests for soft-well E_int aux loss (lit DCM window + contact filter)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.data.units import EV_TO_KCAL_MOL, KCAL_MOL_TO_EV
from mmml.models.physnetjax.physnetjax.training.soft_well_aux import (
    SoftWellConfig,
    SoftWellGeometryPool,
    extract_monomer_from_hybrid_frame,
    soft_well_e_int_loss,
)


def test_soft_well_loss_zero_inside_window():
    import jax.numpy as jnp

    # −4 kcal/mol → eV
    e = jnp.asarray([-4.0 * KCAL_MOL_TO_EV])
    loss = float(soft_well_e_int_loss(e, center_weight=0.0))
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_soft_well_loss_penalises_underbind_and_deep_wells():
    import jax.numpy as jnp

    shallow = jnp.asarray([-1.0 * KCAL_MOL_TO_EV])  # underbind vs −3
    deep = jnp.asarray([-20.0 * KCAL_MOL_TO_EV])  # past hard floor
    lit = jnp.asarray([-4.0 * KCAL_MOL_TO_EV])
    loss_shallow = float(soft_well_e_int_loss(shallow, center_weight=0.0))
    loss_deep = float(soft_well_e_int_loss(deep, center_weight=0.0))
    loss_lit = float(soft_well_e_int_loss(lit, center_weight=0.0))
    assert loss_lit == pytest.approx(0.0, abs=1e-6)
    assert loss_shallow > 1.0  # (2 kcal)^2
    assert loss_deep > loss_shallow


def test_soft_well_loss_caps_outlier_samples():
    import jax.numpy as jnp

    # Pathological deep well must soft-cap near per_sample_cap (tanh).
    deep = jnp.asarray([-200.0 * KCAL_MOL_TO_EV])
    loss = float(soft_well_e_int_loss(deep, center_weight=0.0, per_sample_cap=64.0))
    assert 60.0 <= loss <= 64.0


def test_soft_well_loss_units_match_ev_to_kcal():
    import jax.numpy as jnp

    # 1 kcal too shallow: e = -2 kcal → under = 1 → loss 1 with center_weight=0
    e = jnp.asarray([(-3.0 + 1.0) * KCAL_MOL_TO_EV])
    loss = float(soft_well_e_int_loss(e, center_weight=0.0))
    assert loss == pytest.approx(1.0, abs=1e-5)
    # Sanity: EV_TO_KCAL_MOL is inverse of KCAL_MOL_TO_EV
    assert EV_TO_KCAL_MOL * KCAL_MOL_TO_EV == pytest.approx(1.0, rel=1e-6)


def test_geometry_pool_filters_contact_ok():
    # Tiny DCM-like monomer (5 atoms)
    mono = {
        "R": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.5, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        ),
        "Z": np.array([6, 17, 17, 1, 1], dtype=np.int32),
        "cgenff_type_idx": np.array([0, 1, 1, 2, 2], dtype=np.int32),
        "cgenff_charge": np.zeros(5, dtype=np.float64),
    }
    cfg = SoftWellConfig(
        enabled=True,
        n_directions=8,
        n_orientations=6,
        n_r=6,
        r_min=3.4,
        r_max=6.0,
        min_contact=2.0,
        batch_size=8,
        seed=0,
    )
    pool = SoftWellGeometryPool(mono, cfg)
    assert pool.n > 0
    assert pool.pad == 10
    # All stored geometries must be contact-ok by construction.
    from mmml.analysis.dimer_scans import intermolecular_min_distance

    for i in range(min(pool.n, 32)):
        Ra = pool.R[i, :5]
        Rb = pool.R[i, 5:]
        assert intermolecular_min_distance(Ra, Rb) >= 2.0 - 1e-9


def test_extract_monomer_from_hybrid_frame():
    data = {
        "R": np.zeros((2, 10, 3)),
        "Z": np.tile(np.array([6, 17, 17, 1, 1, 6, 17, 17, 1, 1]), (2, 1)),
        "mol_id": np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), (2, 1)),
        "cgenff_type_idx": np.tile(np.arange(10), (2, 1)),
        "cgenff_charge": np.zeros((2, 10)),
    }
    data["R"][0, :5] = np.arange(15).reshape(5, 3) * 0.1
    mono = extract_monomer_from_hybrid_frame(data, frame=0)
    assert mono["R"].shape == (5, 3)
    assert np.allclose(mono["R"].mean(axis=0), 0.0, atol=1e-8)


def test_soft_well_config_coerce():
    cfg = SoftWellConfig.coerce(
        {"enabled": True, "steps_per_epoch": 4, "batch_size": 8}
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.steps_per_epoch == 4
    assert SoftWellConfig.coerce(None) is None
