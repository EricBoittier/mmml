"""Unit tests for per-monomer JAX bonded preflight."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_maybe_run_per_monomer_preflight_skips_single_monomer() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _maybe_run_per_monomer_bonded_jax_preflight,
    )

    ctx = MagicMock()
    config = MagicMock(n_monomers=1, rescue=MagicMock())
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_per_monomer_recovery",
    ) as mini:
        _maybe_run_per_monomer_bonded_jax_preflight(ctx, config, context="test")
    mini.assert_not_called()


def test_maybe_run_per_monomer_preflight_calls_jax() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _maybe_run_per_monomer_bonded_jax_preflight,
    )

    ctx = MagicMock()
    config = MagicMock(
        n_monomers=4,
        monomer_health=MagicMock(enabled=False),
        rescue=MagicMock(
            nstep_sd=20,
            nprint=5,
            tolenr=1e-3,
            tolgrd=1e-3,
            verbose=False,
        ),
        topology_psf=None,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_per_monomer_recovery",
    ) as mini:
        _maybe_run_per_monomer_bonded_jax_preflight(ctx, config, context="test")
    mini.assert_called_once()
    assert mini.call_args.kwargs["n_monomers"] == 4
