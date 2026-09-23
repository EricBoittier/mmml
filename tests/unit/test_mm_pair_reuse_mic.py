"""MM pair-list Verlet reuse: minimum-image displacement and MLpot skin resolution."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import (
    DEFAULT_MLPOT_MM_SKIN_A,
    resolve_mlpot_mm_skin_A,
)
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    max_displacement_since_build_A,
    neighbor_pair_cache_should_reuse,
)

L = 20.0
CELL = np.diag([L, L, L])


def _reuse(R, last_R, *, skin=1.0, cell=CELL):
    return neighbor_pair_cache_should_reuse(
        calls=7,
        interval=1,
        skin=skin,
        R=R,
        last_R=last_R,
        box=CELL,
        last_box=CELL,
        have_cache=True,
        cell=cell,
    )


def test_lattice_shift_is_not_a_displacement():
    rng = np.random.default_rng(0)
    R0 = rng.random((30, 3)) * L
    R1 = R0.copy()
    R1[3:6] += np.array([L, 0.0, -L])  # molecule re-wrapped into the primary cell
    assert max_displacement_since_build_A(R1, R0, CELL) == pytest.approx(0.0, abs=1e-12)
    assert max_displacement_since_build_A(R1, R0, None) == pytest.approx(L * np.sqrt(2))
    assert _reuse(R1, R0)
    # Without a cell the old behaviour (plain displacement) is kept.
    assert not _reuse(R1, R0, cell=None)


def test_real_move_past_half_skin_still_rebuilds():
    R0 = np.zeros((4, 3)) + 5.0
    R1 = R0.copy()
    R1[2, 0] += 0.49
    assert _reuse(R1, R0, skin=1.0)
    R1[2, 0] += 0.02
    assert not _reuse(R1, R0, skin=1.0)
    # Across the boundary: real 0.3 A move plus a wrap by -L.
    R0 = np.array([[L - 0.1, 1.0, 1.0]])
    R1 = np.array([[0.2, 1.0, 1.0]])
    assert max_displacement_since_build_A(R1, R0, CELL) == pytest.approx(0.3)


def test_cell_formats_equivalent():
    rng = np.random.default_rng(1)
    R0 = rng.random((10, 3)) * L
    R1 = R0 + rng.normal(scale=0.1, size=R0.shape) + np.round(rng.random((10, 3))) * L
    ref = max_displacement_since_build_A(R1, R0, CELL)
    assert max_displacement_since_build_A(R1, R0, np.array([L, L, L])) == pytest.approx(ref)
    assert max_displacement_since_build_A(R1, R0, L) == pytest.approx(ref)


def _mic_pairs_within(R, cutoff):
    d = R[None, :, :] - R[:, None, :]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=-1)
    i, j = np.nonzero(np.triu(r < cutoff, k=1))
    return set(zip(i.tolist(), j.tolist()))


def test_verlet_list_stays_complete_under_mic_reuse():
    """Every pair inside the cutoff is listed while the MIC reuse check passes."""
    rng = np.random.default_rng(2)
    n_mol, per = 60, 3
    rc, skin = 5.0, 1.0
    com = rng.random((n_mol, 1, 3)) * L
    R0 = (com + rng.normal(scale=0.6, size=(n_mol, per, 3))).reshape(-1, 3)
    listed = _mic_pairs_within(R0, rc + skin)
    checked = 0
    for trial in range(200):
        step = rng.normal(size=R0.shape)
        step *= (rng.random((len(R0), 1)) * 0.5 * skin) / np.linalg.norm(step, axis=1, keepdims=True)
        R1 = R0 + step
        # Re-wrap whole molecules by COM (what the MLpot callback does to its copy).
        Rm = R1.reshape(n_mol, per, 3)
        Rm -= L * np.floor(Rm.mean(axis=1, keepdims=True) / L)
        R1 = Rm.reshape(-1, 3)
        if not _reuse(R1, R0, skin=skin):
            continue
        checked += 1
        assert _mic_pairs_within(R1, rc) <= listed
    assert checked > 150


def test_resolve_mlpot_mm_skin(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_MM_SKIN_A", raising=False)
    assert resolve_mlpot_mm_skin_A(None) == DEFAULT_MLPOT_MM_SKIN_A
    args = argparse.Namespace(jax_md_skin_distance=0.25)
    # Parser default (not explicit) does not override the MLpot default.
    assert resolve_mlpot_mm_skin_A(args) == DEFAULT_MLPOT_MM_SKIN_A
    args._cli_explicit = {"jax_md_skin_distance"}
    args.jax_md_skin_distance = 0.4
    assert resolve_mlpot_mm_skin_A(args) == pytest.approx(0.4)
    monkeypatch.setenv("MMML_MLPOT_MM_SKIN_A", "0.8")
    assert resolve_mlpot_mm_skin_A(args) == pytest.approx(0.8)
