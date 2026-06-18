#!/usr/bin/env python3
"""Suggest / benchmark MLpot PhysNet batch sizes for DCM-scale clusters.

Usage (GPU node with checkpoint):
  python scripts/benchmark_mlpot_ml_batch.py --checkpoint path/to/ckpt --n-monomers 90

Without a checkpoint, prints tuning guidance only (safe on CPU CI).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def sparse_effective_batch(n_monomers: int) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        resolve_max_active_dimers,
    )

    n_dimers = n_monomers * (n_monomers - 1) // 2
    cap = resolve_max_active_dimers(n_monomers, n_dimers)
    return n_monomers + cap


def print_tuning_table(n_monomers: int) -> None:
    eff = sparse_effective_batch(n_monomers)
    print(f"n_monomers={n_monomers} -> ~{eff} PhysNet systems/step (sparse dimers)")
    print("Single-GPU (--ml-batch-size); aim for 1-3 chunks after warmup:")
    for bs in (64, 128, 256, 512, 600, eff):
        if bs > eff:
            continue
        chunks = (eff + bs - 1) // bs
        print(f"  batch_size={bs:4d} -> {chunks} chunk(s)")
    print(
        "\nMulti-GPU: --ml-gpu-count N (or MMML_MLPOT_N_GPUS) with CUDA_VISIBLE_DEVICES "
        "listing N devices; keep batch_size so each GPU fits in VRAM."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-monomers", type=int, default=90)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 256, 512, 600],
    )
    parser.add_argument(
        "--ml-gpu-count",
        type=int,
        default=1,
        help="Parallel PhysNet chunks across N local GPUs (default 1).",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print_tuning_table(args.n_monomers)

    if args.checkpoint is None:
        print("\n(no --checkpoint; skipping timed ML eval)")
        return 0

    from mmml.interfaces.pycharmmInterface.jax_device_policy import apply_mlpot_jax_platform_env

    apply_mlpot_jax_platform_env()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import build_decomposed_mlpot_model

    n = args.n_monomers
    n_atoms = n * 10
    z = np.array([6, 1, 1, 1, 1, 1, 1, 1, 1, 1] * n, dtype=int)
    per = [10] * n
    r = np.random.default_rng(0).standard_normal((n_atoms, 3)) * 2.0

    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context

    pos = jnp.asarray(r)
    z_j = jnp.asarray(z)
    for bs in args.batch_sizes:
        model = build_decomposed_mlpot_model(
            args.checkpoint,
            z,
            per,
            n,
            ml_batch_size=bs,
            ml_gpu_count=args.ml_gpu_count,
            verbose=False,
        )
        times = []
        for _ in range(args.repeats + 1):
            t0 = time.perf_counter()
            with mlpot_jax_device_context():
                out = model._spherical_fn(
                    positions=pos,
                    atomic_numbers=z_j,
                    n_monomers=n,
                    cutoff_params=model._cutoff_params,
                    doML=True,
                    doMM=False,
                    doML_dimer=True,
                )
            jax.block_until_ready(out.energy)
            times.append(time.perf_counter() - t0)
        warm = times[0]
        med = sorted(times[1:])[len(times[1:]) // 2]
        print(f"batch_size={bs}: warmup={warm:.3f}s median={med:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
