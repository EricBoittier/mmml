"""Unit tests for umbrella Hamiltonian replica exchange."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.rex import (
    RexStats,
    attempt_replica_exchanges,
    bias_energy_matrix,
    metropolis_exchange_delta,
    neighbor_exchange_pairs,
)


def test_neighbor_pairs_1d_even_odd():
    assert neighbor_exchange_pairs((5,), 0) == [(0, 1), (2, 3)]
    assert neighbor_exchange_pairs((5,), 1) == [(1, 2), (3, 4)]


def test_neighbor_pairs_2d_phases():
    # 2x3 grid, indexing ix*ny+iy
    assert neighbor_exchange_pairs((2, 3), 0) == [(0, 1), (3, 4)]  # horiz even
    assert neighbor_exchange_pairs((2, 3), 1) == [(1, 2), (4, 5)]  # horiz odd
    assert neighbor_exchange_pairs((2, 3), 2) == [(0, 3), (1, 4), (2, 5)]  # vert even
    assert neighbor_exchange_pairs((2, 3), 3) == []  # vert odd: no ix=1 pairs


def test_bias_matrix_and_delta_identity():
    cv = np.array([[1.0], [2.0], [3.0]])
    targets = [[1.0, 2.0, 3.0]]
    ks = [[10.0, 10.0, 10.0]]
    w = bias_energy_matrix(cv, targets, ks)
    np.testing.assert_allclose(np.diag(w), 0.0)
    # Swap 0 and 2: configs at targets of each other
    delta = metropolis_exchange_delta(w, 0, 2)
    # W_0(R_2)=0.5*10*(3-1)^2=20, W_2(R_0)=20, W_0(R_0)=0, W_2(R_2)=0
    assert delta == pytest.approx(40.0)


def test_attempt_exchanges_swaps_when_delta_zero():
    # Two windows already at each other's centers → Δ=0 → always accept
    pos = np.zeros((2, 2, 3))
    pos[0, 0] = 0.0
    pos[0, 1] = 1.0  # r=1
    pos[1, 0] = 0.0
    pos[1, 1] = 2.0  # r=2
    # After measuring cv from positions... use explicit cv matching targets swapped
    cv = np.array([[2.0], [1.0]])  # each sits at the other's center
    targets = [[1.0, 2.0]]
    ks = [[5.0, 5.0]]
    rng = np.random.default_rng(0)
    stats = RexStats()
    pos_out, _, _, att, acc = attempt_replica_exchanges(
        positions_packed=pos.reshape(4, 3),
        momenta_packed=None,
        forces_packed=None,
        cv=cv,
        targets_per_cv=targets,
        k_per_cv=ks,
        grid_shape=(2,),
        phase=0,
        beta=1.0,
        rng=rng,
        n_atoms=2,
        stats=stats,
    )
    assert att == 1 and acc == 1
    assert stats.acceptance == pytest.approx(1.0)
    out = pos_out.reshape(2, 2, 3)
    np.testing.assert_allclose(out[0, 1, 0], 2.0)
    np.testing.assert_allclose(out[1, 1, 0], 1.0)


def test_attempt_exchanges_writable_from_readonly_view():
    pos = np.zeros((2, 2, 3))
    pos[0, 1, 0] = 1.0
    pos[1, 1, 0] = 2.0
    packed = pos.reshape(4, 3)
    packed.setflags(write=False)
    cv = np.array([[2.0], [1.0]])
    pos_out, _, _, att, acc = attempt_replica_exchanges(
        positions_packed=packed,
        momenta_packed=None,
        forces_packed=None,
        cv=cv,
        targets_per_cv=[[1.0, 2.0]],
        k_per_cv=[[5.0, 5.0]],
        grid_shape=(2,),
        phase=0,
        beta=1.0,
        rng=np.random.default_rng(0),
        n_atoms=2,
    )
    assert att == 1 and acc == 1
    assert pos_out.flags.writeable


def test_cli_replica_exchange_flag():
    from mmml.cli.misc.umbrella_sample import _config_from_args, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--structure",
            "mol.xyz",
            "-o",
            "out",
            "--atoms",
            "0,1",
            "--targets",
            "1.0,1.5",
            "--replica-exchange",
            "--rex-freq",
            "50",
            "--overwrite",
        ]
    )
    cfg = _config_from_args(args)
    assert cfg.replica_exchange is True
    assert cfg.rex_freq == 50
