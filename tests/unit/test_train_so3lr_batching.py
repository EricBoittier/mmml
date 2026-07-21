"""Throughput helpers for scripts/train_so3lr_spooky_extxyz.py.

Covers widened atom buckets, pair-budget sizing, prefetch, and deferred metric sync.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_so3lr_spooky_extxyz.py"


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("train_so3lr_spooky_extxyz", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flat_data(natoms: list[int]) -> dict[str, np.ndarray]:
    offsets = [0]
    z_parts = []
    r_parts = []
    f_parts = []
    for n in natoms:
        z_parts.append(np.arange(1, n + 1, dtype=np.int32) % 8 + 1)
        r_parts.append(np.random.default_rng(n).normal(size=(n, 3)))
        f_parts.append(np.zeros((n, 3), dtype=np.float64))
        offsets.append(offsets[-1] + n)
    n_mol = len(natoms)
    return {
        "mol_offsets": np.asarray(offsets, dtype=np.int64),
        "R": np.concatenate(r_parts, axis=0),
        "Z": np.concatenate(z_parts, axis=0),
        "F": np.concatenate(f_parts, axis=0),
        "E": np.arange(n_mol, dtype=np.float64).reshape(-1, 1),
        "N": np.asarray(natoms, dtype=np.int32).reshape(-1, 1),
        "Q": np.zeros((n_mol, 1), dtype=np.float64),
        "S": np.ones((n_mol, 1), dtype=np.float64),
        "D": np.zeros((n_mol, 3), dtype=np.float64),
    }


def test_pad_atoms_for_count(trainer):
    assert trainer.pad_atoms_for_count(7, 1) == 7
    assert trainer.pad_atoms_for_count(7, 4) == 8
    assert trainer.pad_atoms_for_count(8, 4) == 8
    assert trainer.pad_atoms_for_count(9, 4) == 12


def test_bucket_width_groups_nearby_sizes(trainer):
    data = _flat_data([5, 6, 7, 8, 9])
    buckets = trainer.bucket_indices_by_natoms(
        data, np.arange(5, dtype=np.int64), bucket_width=4
    )
    assert set(buckets) == {8, 12}
    assert sorted(buckets[8].tolist()) == [0, 1, 2, 3]
    assert buckets[12].tolist() == [4]


def test_pairs_budget_batch_size(trainer):
    assert (
        trainer.pairs_budget_batch_size(
            77, per_device_batch_size=64, max_pairs_per_device=18000
        )
        == 3
    )
    assert (
        trainer.pairs_budget_batch_size(
            10, per_device_batch_size=8, max_pairs_per_device=18000
        )
        == 8
    )


def test_resolve_batch_sizes_heuristic(trainer):
    buckets = {10: np.arange(20), 77: np.arange(20)}
    sizes = trainer.resolve_batch_sizes(
        buckets,
        per_device_batch_size=64,
        max_pairs_per_device=18000,
        auto_batch=False,
    )
    assert sizes[10] == 64  # capped by pair budget? 18000/100=180 -> min(64,180)=64
    assert sizes[77] == 3


def test_iter_device_batches_uses_resolved_sizes(trainer):
    buckets = {8: np.arange(16, dtype=np.int64)}
    batches = list(
        trainer.iter_device_batches(
            buckets,
            batch_sizes={8: 4},
            n_devices=2,
            rng=np.random.default_rng(0),
        )
    )
    assert len(batches) == 2  # 16 / (4*2) = 2
    pad, device_indices = batches[0]
    assert pad == 8
    assert device_indices.shape == (2, 4)


def test_iter_device_batches_sorted_pads_by_default(trainer):
    buckets = {
        8: np.arange(8, dtype=np.int64),
        12: np.arange(8, 16, dtype=np.int64),
    }
    pads = [
        pad
        for pad, _ in trainer.iter_device_batches(
            buckets,
            batch_sizes={8: 2, 12: 2},
            n_devices=2,
            rng=np.random.default_rng(0),
            shuffle_pad_buckets=False,
        )
    ]
    assert pads == [12, 12, 8, 8]


def test_apply_auto_batch_margin(trainer):
    out = trainer.apply_auto_batch_margin({84: 6, 120: 2, 8: 8}, margin=1)
    assert out == {84: 5, 120: 1, 8: 7}


def test_shuffle_pad_buckets_mixes_order(trainer):
    buckets = {
        8: np.arange(8, dtype=np.int64),
        120: np.arange(8, 16, dtype=np.int64),
    }
    pads = [
        pad
        for pad, _ in trainer.iter_device_batches(
            buckets,
            batch_sizes={8: 2, 120: 2},
            n_devices=2,
            rng=np.random.default_rng(1),
            shuffle_pad_buckets=True,
        )
    ]
    assert set(pads) == {8, 120}
    # Sorted default is always large-first; shuffle must sometimes start small.
    started_small = False
    for seed in range(20):
        first = next(
            trainer.iter_device_batches(
                buckets,
                batch_sizes={8: 2, 120: 2},
                n_devices=2,
                rng=np.random.default_rng(seed),
                shuffle_pad_buckets=True,
            )
        )[0]
        if first == 8:
            started_small = True
            break
    assert started_small


def test_max_batches_per_bucket_visit_bounds_consecutive_run(trainer):
    """Neither bucket may monopolize more than max_batches_per_bucket_visit
    consecutive batches while the other still has work left -- regression
    test for the observed failure mode where a >1M-step bucket gave the
    model zero exposure to other structure populations for well over a
    million steps. (Both buckets are sized well above the cap here so
    neither exhausts early -- once a bucket legitimately runs out, the
    other finishing its tail alone is expected, not a bound violation.)"""
    buckets = {
        8: np.arange(40, dtype=np.int64),  # 20 batches at B=2
        120: np.arange(40, dtype=np.int64),  # 20 batches at B=2 -- the "huge" one
    }
    pads = [
        pad
        for pad, _ in trainer.iter_device_batches(
            buckets,
            batch_sizes={8: 2, 120: 2},
            n_devices=1,
            rng=np.random.default_rng(0),
            shuffle_pad_buckets=False,
            max_batches_per_bucket_visit=3,
        )
    ]
    # Every maximal run of one pad value must be <= 3 batches long.
    run_len = 1
    for prev, cur in zip(pads, pads[1:]):
        run_len = run_len + 1 if cur == prev else 1
        assert run_len <= 3
    # All batches from both buckets still get produced exactly once overall.
    assert pads.count(8) == 20
    assert pads.count(120) == 20


def test_max_batches_per_bucket_visit_none_matches_unbounded_default(trainer):
    """max_batches_per_bucket_visit=None (the default) must reproduce the
    original fully-drain-one-bucket-then-move-on behavior exactly."""
    buckets = {
        8: np.arange(8, dtype=np.int64),
        12: np.arange(8, 16, dtype=np.int64),
    }
    kwargs = dict(
        buckets=buckets,
        batch_sizes={8: 2, 12: 2},
        n_devices=2,
        shuffle_pad_buckets=False,
    )
    unbounded = list(
        trainer.iter_device_batches(rng=np.random.default_rng(0), **kwargs)
    )
    explicit_none = list(
        trainer.iter_device_batches(
            rng=np.random.default_rng(0),
            max_batches_per_bucket_visit=None,
            **kwargs,
        )
    )
    pads_unbounded = [p for p, _ in unbounded]
    pads_explicit = [p for p, _ in explicit_none]
    assert pads_unbounded == pads_explicit == [12, 12, 8, 8]


def test_max_batches_per_bucket_visit_cli_default_is_none(trainer):
    args = trainer.build_parser().parse_args([])
    assert args.max_batches_per_bucket_visit is None
    args_set = trainer.build_parser().parse_args(
        ["--max-batches-per-bucket-visit", "2000"]
    )
    assert args_set.max_batches_per_bucket_visit == 2000


def test_prefetch_matches_eager_stack(trainer):
    data = _flat_data([3, 3, 3, 3, 3, 3, 3, 3])
    buckets = {4: np.arange(8, dtype=np.int64)}
    index_list = list(
        trainer.iter_device_batches(
            buckets,
            batch_sizes={4: 2},
            n_devices=2,
            rng=np.random.default_rng(1),
        )
    )
    eager = [
        trainer.stack_device_batches(data, di, pad_atoms=pad)
        for pad, di in index_list
    ]
    prefetched = list(
        trainer.prefetch_stacked_batches(
            iter(index_list),
            data,
            depth=2,
        )
    )
    assert len(prefetched) == len(eager)
    for (pad, bsz, batch), eager_batch in zip(prefetched, eager):
        assert pad == 4
        assert bsz == 2
        np.testing.assert_allclose(
            np.asarray(batch["R"]), np.asarray(eager_batch["R"])
        )


def test_finalize_metrics_averages_once(trainer):
    m0 = {"loss": jnp.asarray(2.0), "energy_mae": jnp.asarray(4.0)}
    m1 = {"loss": jnp.asarray(4.0), "energy_mae": jnp.asarray(6.0)}
    total = trainer.accumulate_metrics(None, m0)
    total = trainer.accumulate_metrics(total, m1)
    out = trainer.finalize_metrics(total, 2)
    assert out["loss"] == pytest.approx(3.0)
    assert out["energy_mae"] == pytest.approx(5.0)


def test_stack_device_batches_pads_and_masks(trainer):
    data = _flat_data([3, 4])
    device_indices = np.array([[0, 1]], dtype=np.int64)  # 1 device, B=2
    batch = trainer.stack_device_batches(data, device_indices, pad_atoms=4)
    # Leading device axis from stack.
    assert batch["Z"].shape[0] == 1
    assert batch["Z"].shape[1] == 8  # 2 mols * 4 pad
    am = np.asarray(batch["atom_mask"][0])
    assert am.tolist() == [1, 1, 1, 0, 1, 1, 1, 1]
    bm = np.asarray(batch["batch_mask"][0])
    assert np.all((bm == 0) | (bm == 1))
    assert np.any(bm == 0)  # padding edges zeroed


def test_cli_defaults_enable_throughput_path(trainer):
    args = trainer.build_parser().parse_args([])
    assert args.auto_batch is True
    assert args.atom_bucket_width == 4
    assert args.prefetch_batches == 2
    assert args.batch_size_per_device == 64
    args_off = trainer.build_parser().parse_args(["--no-auto-batch", "--atom-bucket-width", "1"])
    assert args_off.auto_batch is False
    assert args_off.atom_bucket_width == 1


def test_is_oom_error_matches_bfc_message(trainer):
    assert trainer._is_oom_error(
        RuntimeError("Allocator (GPU_0_bfc) ran out of memory trying to allocate")
    )
    assert not trainer._is_oom_error(ValueError("shape mismatch"))


def test_auto_batch_cache_roundtrip(trainer, tmp_path):
    args = trainer.build_parser().parse_args(
        ["--batch-size-per-device", "32", "--atom-bucket-width", "4"]
    )

    class _FakeDev:
        device_kind = "gpu"

        def memory_stats(self):
            return {"bytes_limit": 32 << 30}

    devices = [_FakeDev()]
    fp = trainer.auto_batch_cache_fingerprint(args, devices, [8, 12, 120])
    path = tmp_path / "auto_batch_sizes.json"
    sizes = {8: 32, 12: 24, 120: 2}
    trainer.save_auto_batch_cache(path, fp, sizes)
    loaded = trainer.load_auto_batch_cache(path, fp)
    assert loaded == sizes
    # Different cap → miss
    args2 = trainer.build_parser().parse_args(
        ["--batch-size-per-device", "64", "--atom-bucket-width", "4"]
    )
    fp2 = trainer.auto_batch_cache_fingerprint(args2, devices, [8, 12, 120])
    assert trainer.load_auto_batch_cache(path, fp2) is None


def test_resolve_batch_sizes_respects_seed_and_only_pads(trainer):
    buckets = {8: np.arange(16), 12: np.arange(16), 120: np.arange(16)}
    sizes = trainer.resolve_batch_sizes(
        buckets,
        per_device_batch_size=32,
        max_pairs_per_device=18000,
        auto_batch=True,
        data=None,  # forces heuristic for probed pads
        only_pads={12},
        seed_sizes={8: 30, 120: 2},
    )
    assert sizes[8] == 30
    assert sizes[120] == 2
    # pad 12 was "probed" but data=None → heuristic
    assert sizes[12] == trainer.pairs_budget_batch_size(
        12, per_device_batch_size=32, max_pairs_per_device=18000
    )


def test_force_auto_batch_cli(trainer):
    args = trainer.build_parser().parse_args(["--force-auto-batch"])
    assert args.force_auto_batch is True
    assert trainer.build_parser().parse_args([]).force_auto_batch is False
