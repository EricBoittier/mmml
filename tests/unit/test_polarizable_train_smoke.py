"""Smoke test of examples/polarizable_dcmnet/train_field.py on tiny synthetic data (scratch init, 1 epoch).  Marked slow."""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "polarizable_dcmnet" / "train_field.py"
Z10 = np.array([6, 6, 8, 6, 1, 1, 1, 1, 1, 1], np.int32)


def _geom(rng):
    R = rng.normal(size=(10, 3)) * 1.2
    return R - R.mean(0)


def _make_data(root, rng):
    G = 40
    for i in range(6):
        frame = 1 + i if i < 4 else (46 if i == 4 else 60)  # 4 train, 1 val, 1 test
        A, B = _geom(rng), _geom(rng) + np.array([5.0, 0.0, 0.0])
        grid = rng.normal(size=(G, 3)) * 3.0 + np.array([2.5, 0, 0])
        np.savez(
            root / f"dim_{i:04d}.npz",
            Z=np.concatenate([Z10, Z10]),
            R=np.concatenate([A, B]),
            grid_ang=grid,
            scale=np.full(G, 1.4),
            esp_AB=rng.normal(size=G) * 0.01,
            esp_A=rng.normal(size=G) * 0.01,
            esp_B=rng.normal(size=G) * 0.01,
            centroid=5.0,
            frame=frame,
            E_int_kcal=0.0,
        )
    N = 12
    np.savez(
        root / "mono.npz",
        Z=Z10,
        R=np.stack([_geom(rng) for _ in range(N)]),
        grid=rng.normal(size=(N, G, 3)) * 3.0,
        gmask=np.ones((N, G), bool),
        mmask=np.ones((N, G), bool),
        esp=rng.normal(size=(N, G)) * 0.01,
        dip_au=rng.normal(size=(N, 3)) * 0.1,
        q_cgenff=rng.normal(size=10) * 0.1,
        q_qm=rng.normal(size=(N, 10)) * 0.1,
        split=np.array([0] * 8 + [1] * 4),
    )


@pytest.mark.slow
def test_train_field_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location("train_field", SCRIPT)
    tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf)
    _make_data(tmp_path, np.random.default_rng(0))
    out = tmp_path / "run"
    with jax.enable_x64(False):
        tf.main(
            [
                "--data",
                str(tmp_path),
                "--mono_data",
                str(tmp_path / "mono.npz"),
                "--mono_test",
                str(tmp_path / "mono.npz"),
                "--out",
                str(out),
                "--scratch",
                "--features",
                "8",
                "--num_iterations",
                "2",
                "--epochs",
                "1",
                "--bs",
                "2",
                "--rbs",
                "2",
            ]
        )
    final = json.load(open(out / "final.json"))
    assert (out / "best_params.msgpack").exists() and np.isfinite(final["gas_val"]["rmse_dESP"])
