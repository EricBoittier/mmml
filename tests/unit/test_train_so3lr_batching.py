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
