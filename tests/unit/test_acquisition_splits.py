"""Grouped splits must not leak neighboring frames or duplicate geometries."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.acquisition.splits import (
    SPLIT_CANDIDATE,
    SPLIT_SEED,
    SPLIT_TEST,
    SPLIT_VALID,
    assert_split_isolation,
    build_pool,
    deduplicate,
    records_from_npz,
)
from mmml.acquisition.synthetic import default_seed_groups, make_smoke_pool


def _two_traj_npz():
    # 2 trajectories × 3 frames, plus an exact duplicate of frame 0.
    z = np.array([8, 1, 1], dtype=np.int32)
    r0 = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    frames = []
    seeds = []
    idx = []
    for g in (1, 2):
        for f in range(3):
            stretch = r0.copy()
            stretch[1, 0] += 0.05 * f
            stretch[2, 1] += 0.04 * g
            frames.append(stretch)
            seeds.append(g)
            idx.append(f)
    frames.append(frames[0].copy())  # duplicate geometry of first frame
    seeds.append(99)
    idx.append(0)
    n = len(frames)
    pad = 3
    R = np.zeros((n, pad, 3))
    Z = np.zeros((n, pad), dtype=np.int32)
    for i, r in enumerate(frames):
        R[i] = r
        Z[i] = z
    return {
        "R": R,
        "Z": Z,
        "N": np.full(n, 3, dtype=np.int32),
        "group_seed": np.asarray(seeds),
        "frame_index": np.asarray(idx),
        "temperature": np.full(n, 300.0),
    }


def test_duplicates_are_removed_but_mapped():
    recs = records_from_npz(_two_traj_npz())
    unique, dup = deduplicate(recs)
    assert len(unique) == len(recs) - 1
    assert dup
    kept_ids = {r.structure_id for r in unique}
    for src_index, dst in dup.items():
        assert dst in kept_ids
        assert int(src_index) not in {r.index for r in unique}


def test_grouped_splits_keep_a_trajectory_on_one_side():
    data = _two_traj_npz()
    man = build_pool(data, valid_fraction=0.5, test_fraction=0.0, seed=0)
    assert_split_isolation(man.records)
    groups = {}
    for rec in man.records:
        groups.setdefault(rec.group, set()).add(rec.split)
    for splits in groups.values():
        assert len(splits) == 1


def test_seed_groups_are_excluded_from_candidate_and_held_out():
    pool = make_smoke_pool(n_traj_per_stratum=2, n_frames=4, seed=0)
    seed_groups = default_seed_groups(pool, n_groups=2)
    man = build_pool(
        pool,
        valid_fraction=0.25,
        test_fraction=0.25,
        seed=1,
        seed_groups=seed_groups,
    )
    assert_split_isolation(man.records)
    seed_recs = man.by_split(SPLIT_SEED)
    assert seed_recs
    assert all(r.group in seed_groups for r in seed_recs)
    for split in (SPLIT_CANDIDATE, SPLIT_VALID, SPLIT_TEST):
        assert all(r.group not in seed_groups for r in man.by_split(split))


def test_isolation_detector_catches_a_leaked_group():
    pool = make_smoke_pool(n_traj_per_stratum=1, n_frames=3, seed=0)
    man = build_pool(pool, valid_fraction=0.3, test_fraction=0.3, seed=0)
    leaked = man.records[0]
    leaked.split = SPLIT_TEST if leaked.split != SPLIT_TEST else SPLIT_VALID
    # Force the same group onto two splits.
    man.records[1].split = SPLIT_CANDIDATE if leaked.split != SPLIT_CANDIDATE else SPLIT_VALID
    man.records[1].group = leaked.group
    with pytest.raises(ValueError, match="split isolation"):
        assert_split_isolation(man.records)
