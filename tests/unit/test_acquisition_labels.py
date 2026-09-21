"""Label cache reuses calculations; mock backend is deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.acquisition.ids import structure_id
from mmml.acquisition.labels import LabelCache, MockMorseReference, labels_to_npz
from mmml.acquisition.splits import StructureRecord


def _rec(i: int = 0) -> StructureRecord:
    z = np.array([8, 1, 1], dtype=np.int32)
    r = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]) + 0.01 * i
    sid = structure_id(r, z)
    return StructureRecord(
        index=i,
        structure_id=sid,
        geometry_fingerprint=sid,
        composition="H2O1",
        stratum="H2O1|T300",
        group="g0",
        n_atoms=3,
        atomic_numbers=z,
        positions=r,
    )


def test_cache_hit_skips_recompute(tmp_path: Path):
    backend = MockMorseReference()
    cache = LabelCache(tmp_path / "cache")
    rec = _rec()
    a = cache.compute_or_cached(rec, backend)
    assert cache.misses == 1 and cache.hits == 0
    b = cache.compute_or_cached(rec, backend)
    assert cache.hits == 1
    assert a.energy == b.energy
    np.testing.assert_allclose(a.forces, b.forces)


def test_different_method_settings_do_not_collide(tmp_path: Path):
    cache = LabelCache(tmp_path / "cache")
    rec = _rec()
    r1 = cache.compute_or_cached(rec, MockMorseReference(de=4.0))
    r2 = cache.compute_or_cached(rec, MockMorseReference(de=8.0))
    assert r1.energy != r2.energy
    assert cache.misses == 2


def test_failures_are_accounted(tmp_path: Path):
    rec = _rec()
    backend = MockMorseReference(fail_ids={rec.structure_id})
    cache = LabelCache(tmp_path / "cache")
    result = cache.compute_or_cached(rec, backend)
    assert not result.success
    assert cache.failures == 1
    npz = labels_to_npz([rec], [result])
    assert not bool(npz["label_success"][0])
