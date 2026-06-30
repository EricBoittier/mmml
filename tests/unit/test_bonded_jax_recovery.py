"""Unit tests for JAX bonded recovery dispatch."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    BondedMmMiniConfig,
    bonded_mm_mini_config_from_namespace,
    minimize_bonded_mm_recovery,
)


def test_bonded_mm_mini_config_from_namespace_defaults():
    cfg = bonded_mm_mini_config_from_namespace(
        Namespace(bonded_mm_mini_steps=75, bonded_recovery_backend="jax", quiet=False)
    )
    assert cfg.nstep_sd == 75
    assert cfg.backend == "jax"
    assert cfg.verbose is True


def test_minimize_bonded_mm_recovery_uses_jax_when_auto_succeeds():
    ctx = MagicMock()
    cfg = BondedMmMiniConfig(nstep_sd=10, backend="auto", verbose=False)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((2, 3)),
    ):
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_recovery",
            return_value=3.5,
        ) as jax_mini:
            with patch(
                "mmml.interfaces.pycharmmInterface.mlpot.dynamics._minimize_bonded_charmm_recovery",
            ) as charmm_mini:
                with patch(
                    "mmml.interfaces.pycharmmInterface.mlpot.dynamics._print_bonded_recovery_geometry_diff",
                ):
                    grms = minimize_bonded_mm_recovery(ctx, cfg)
    assert grms == pytest.approx(3.5)
    jax_mini.assert_called_once()
    charmm_mini.assert_not_called()


def test_minimize_bonded_mm_recovery_falls_back_to_charmm_on_jax_error():
    ctx = MagicMock()
    cfg = BondedMmMiniConfig(nstep_sd=10, backend="auto", verbose=False)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((2, 3)),
    ):
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_recovery",
            side_effect=RuntimeError("no psf"),
        ):
            with patch(
                "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._mlpot_covers_all_atoms",
                return_value=False,
            ):
                with patch(
                    "mmml.interfaces.pycharmmInterface.mlpot.dynamics._minimize_bonded_charmm_recovery",
                    return_value=2.0,
                ) as charmm_mini:
                    with patch(
                        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._print_bonded_recovery_geometry_diff",
                    ):
                        grms = minimize_bonded_mm_recovery(ctx, cfg)
    assert grms == pytest.approx(2.0)
    charmm_mini.assert_called_once()


def test_minimize_bonded_mm_recovery_backend_charmm_skips_jax():
    ctx = MagicMock()
    cfg = BondedMmMiniConfig(nstep_sd=10, backend="charmm", verbose=False)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_recovery",
    ) as jax_mini:
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._mlpot_covers_all_atoms",
            return_value=False,
        ):
            with patch(
                "mmml.interfaces.pycharmmInterface.mlpot.dynamics._minimize_bonded_charmm_recovery",
                return_value=1.0,
            ) as charmm_mini:
                with patch(
                    "mmml.interfaces.pycharmmInterface.mlpot.dynamics._print_bonded_recovery_geometry_diff",
                ):
                    minimize_bonded_mm_recovery(ctx, cfg)
    jax_mini.assert_not_called()
    charmm_mini.assert_called_once()


def test_minimize_bonded_jax_recovery_all_ml_returns_none_for_charmm_fallback():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        minimize_bonded_jax_recovery,
    )

    ctx = MagicMock(ml_selection=MagicMock(get_atom_indexes=lambda: list(range(4))))
    cfg = BondedMmMiniConfig(nstep_sd=10, backend="auto", verbose=False)
    system = MagicMock()
    system.topology.bonds = np.zeros((0, 2), dtype=np.int32)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.load_bonded_system_for_recovery",
        return_value=(system, MagicMock(cleanup=lambda: None)),
    ):
        assert minimize_bonded_jax_recovery(ctx, cfg) is None


def test_bonded_forces_grms_kcalmol_A():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        bonded_forces_grms_kcalmol_A,
    )

    forces = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64)
    assert bonded_forces_grms_kcalmol_A(forces) == pytest.approx(np.sqrt(12.5))


def test_resolve_recovery_psf_source_prefers_topology_psf(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        resolve_recovery_psf_source,
    )

    psf = tmp_path / "cluster.psf"
    psf.write_text("psf\n")
    ctx = MagicMock(topology_psf_path=None)
    source = resolve_recovery_psf_source(ctx, topology_psf=psf)
    assert source.path == psf.resolve()
    assert source.temporary is False


def test_apply_frozen_positions_to_fire_state_uses_dataclass_replace():
    """jax-md ``FireDescentState`` is a dataclass; ``_replace`` is not available."""
    from jax_md import minimize as jax_minimize
    from jax_md import space

    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        _apply_frozen_positions_to_fire_state,
    )

    def force_fn(pos):
        return -pos

    _, shift_fn = space.free()
    init_fn, _step_fn = jax_minimize.fire_descent(
        force_fn,
        shift_fn,
        dt_start=0.05,
        dt_max=0.05,
    )
    state = init_fn(jnp.zeros((3, 3)))
    freeze_idx = jnp.asarray([0, 1], dtype=jnp.int32)
    pos0_frozen = jnp.ones((2, 3))
    updated = _apply_frozen_positions_to_fire_state(
        state,
        pos0_frozen=pos0_frozen,
        freeze_idx=freeze_idx,
    )
    assert np.allclose(np.asarray(updated.position[0]), 1.0)
    assert np.allclose(np.asarray(updated.position[1]), 1.0)
    assert np.allclose(np.asarray(updated.position[2]), 0.0)
