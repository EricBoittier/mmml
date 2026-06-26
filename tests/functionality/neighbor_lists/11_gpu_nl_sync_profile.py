#!/usr/bin/env python3
"""Profile GPU NL communication: D2H sync vs CPU build vs H2D pairs vs CuPy+DLPack.

Run on a GPU node with ``uv sync --extra gpu`` and ``MMML_MM_NL_DEVICE=gpu`` for
the full comparison. On CPU-only hosts, reports D2H + CPU Vesin timings only.

Examples
--------
  uv run python tests/functionality/neighbor_lists/11_gpu_nl_sync_profile.py
  MMML_MM_NL_DEVICE=gpu uv run python tests/functionality/neighbor_lists/11_gpu_nl_sync_profile.py \\
      --case synthetic_aco_liquid_n32 --repeat 30
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import (
    build_liquid_density_synthetic_case,
    liquid_density_synthetic_cases,
    print_header,
)
from mmml.interfaces.pycharmmInterface.nl_gpu import (
    gpu_nl_path_available,
    have_cupy,
    profile_nl_sync_components,
    resolve_mm_nl_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="synthetic_aco_liquid_n32",
        help="Liquid-density synthetic case name",
    )
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    print_header("GPU NL sync profile")
    print(f"  MMML_MM_NL_DEVICE={resolve_mm_nl_device()}")
    print(f"  cupy={have_cupy()}  gpu_path={gpu_nl_path_available()}")

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        print(f"SKIP: JAX required ({exc})")
        return 0

    synthetic_all = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    if args.case not in synthetic_all:
        print(f"Unknown case {args.case!r}; choose from: {', '.join(sorted(synthetic_all))}")
        return 1

    positions, cell, offsets, _mid, cutoff, _desc, _side, _rho = build_liquid_density_synthetic_case(
        synthetic_all[args.case]
    )

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    device = gpu_devices[0] if gpu_devices else jax.devices()[0]
    positions_jax = jax.device_put(jnp.asarray(positions), device)

    timings = profile_nl_sync_components(
        positions_jax,
        cell,
        cutoff=cutoff,
        monomer_offsets=offsets,
        repeat=int(args.repeat),
        warmup=int(args.warmup),
    )

    print(f"\n--- {args.case} (n={positions.shape[0]}, device={device}) ---")
    for key, ms in timings.items():
        label = key.replace("_", " ")
        if np.isnan(ms):
            print(f"  {label:28s}  n/a (GPU path unavailable)")
        else:
            print(f"  {label:28s}  {ms:8.3f} ms (median)")

    if not gpu_nl_path_available():
        print(
            "\nNote: set MMML_MM_NL_DEVICE=gpu with cupy + vesin>=0.5 on a CUDA node "
            "to benchmark gpu_vesin_dlpack_ms."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
