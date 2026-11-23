"""CPU smoke test of the timing scripts (tiny N, random-init model)."""

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import e3x
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from karml.models.dcmnet.polarizable import NATOMS, FieldMessagePassingModel

EX = Path(__file__).resolve().parents[2] / "examples" / "polarizable_dcmnet"
CFG = dict(
    features=8,
    max_degree=1,
    num_iterations=2,
    num_basis_functions=8,
    cutoff=4.0,
    max_atomic_number=17,
    n_dcm=3,
    dc_scale=0.3,
)
Z10 = np.array([6, 6, 8, 6, 1, 1, 1, 1, 1, 1], np.int32)


def _load(name):
    sys.path.insert(0, str(EX))
    spec = importlib.util.spec_from_file_location(name, EX / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_bench_scripts(tmp_path):
    rng = np.random.default_rng(0)
    R = rng.normal(size=(10, 3)) * 1.2
    R -= R.mean(0)
    with jax.enable_x64(False):
        m = FieldMessagePassingModel(**CFG)
        dst, src = e3x.ops.sparse_pairwise_indices(NATOMS)
        z = jnp.zeros(NATOMS)
        params = m.init(
            jax.random.PRNGKey(0),
            atomic_numbers=jnp.zeros(NATOMS, jnp.int32),
            positions=jnp.zeros((NATOMS, 3)),
            dst_idx=dst,
            src_idx=src,
            field_V=z,
            field_E=jnp.zeros((NATOMS, 3)),
        )
        run = tmp_path / "run"
        run.mkdir()
        (run / "model_config.json").write_text(json.dumps({"architecture": CFG}))
        (run / "best_params.msgpack").write_bytes(
            serialization.msgpack_serialize(jax.tree_util.tree_map(np.asarray, params))
        )
        q = rng.normal(size=10) * 0.1
        np.savez(tmp_path / "mono.npz", Z=Z10, q_cgenff=q, R=R[None])
        bc = _load("bench_cluster")
        bc.main(
            [
                "--params",
                str(run),
                "--mono_data",
                str(tmp_path / "mono.npz"),
                "--ns",
                "2,3",
                "--reps",
                "2",
                "--json",
                str(tmp_path / "c.json"),
            ]
        )
        assert json.load(open(tmp_path / "c.json"))["results"][0]["fwdgrad_ms"] > 0
        N, L = 4, 12.0
        Rb = np.stack([R + np.array([i % 2, (i // 2) % 2, 0]) * 6.0 for i in range(N)]).astype(np.float32)
        np.savez(tmp_path / "boxes.npz", t4_R=Rb, t4_L=L, t4_Z=Z10, t4_q=q.astype(np.float32))
        bb = _load("bench_box")
        bb.main(
            [
                "--params",
                str(run),
                "--boxes",
                str(tmp_path / "boxes.npz"),
                "--systems",
                "t4",
                "--Ks",
                "2",
                "--reps",
                "2",
                "--json",
                str(tmp_path / "b.json"),
            ]
        )
        assert np.isfinite(json.load(open(tmp_path / "b.json"))["results"][0]["fwdgrad_ms"])
