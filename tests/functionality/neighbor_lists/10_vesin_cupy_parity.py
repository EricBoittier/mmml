#!/usr/bin/env python3
"""Parity: CPU vesin_mic_pairs vs vectorized filter vs optional GPU Vesin path."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import (
    build_liquid_density_synthetic_case,
    liquid_density_synthetic_cases,
    print_fail,
    print_header,
    print_pass,
)
from mmml.interfaces.pycharmmInterface.nl_gpu import gpu_nl_path_available, rebuild_vesin_pairs_gpu
from mmml.interfaces.pycharmmInterface.nl_reference import (
    compare_pair_sets,
    extract_valid_pairs,
    filter_vesin_half_list_vectorized,
    vesin_mic_pairs,
    vesin_raw_half_list,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="synthetic_aco_liquid_n16")
    args = parser.parse_args()

    print_header("Vesin CuPy / vectorized filter parity")
    synthetic_all = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    if args.case not in synthetic_all:
        print_fail(f"unknown case {args.case!r}")
        return 1

    positions, cell, offsets, monomer_id, cutoff, _desc, _side, _rho = build_liquid_density_synthetic_case(
        synthetic_all[args.case]
    )
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int32)

    ref = vesin_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )

    i_raw, j_raw, dist_raw = vesin_raw_half_list(positions, cell, cutoff)
    i_vec, j_vec = filter_vesin_half_list_vectorized(
        i_raw,
        j_raw,
        dist_raw,
        cutoff,
        monomer_id,
        positions,
        cell,
        monomer_offsets=offsets,
    )
    vec_set = {(int(a), int(b)) for a, b in zip(i_vec, j_vec, strict=False)}
    cmp_vec = compare_pair_sets(ref, vec_set)
    if cmp_vec.match:
        print_pass(f"vectorized filter matches vesin_mic_pairs ({len(ref)} pairs)")
    else:
        print_fail(f"vectorized filter mismatch: {cmp_vec.summary()}")
        return 1

    if gpu_nl_path_available():
        import jax
        import jax.numpy as jnp

        device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
        pos_jax = jax.device_put(jnp.asarray(positions), device)
        pair_idx, pair_mask, label = rebuild_vesin_pairs_gpu(
            pos_jax,
            cell,
            cutoff=cutoff,
            monomer_offsets=offsets,
            total_atoms=positions.shape[0],
        )
        gpu_set = extract_valid_pairs(
            np.asarray(jax.device_get(pair_idx)),
            np.asarray(jax.device_get(pair_mask), dtype=bool),
        )
        cmp_gpu = compare_pair_sets(ref, gpu_set)
        if cmp_gpu.match:
            print_pass(f"{label} matches CPU oracle ({len(ref)} pairs)")
        else:
            print_fail(f"GPU path mismatch: {cmp_gpu.summary()}")
            return 1
    else:
        print("SKIP: GPU Vesin path (set MMML_MM_NL_DEVICE=gpu + cupy + vesin on CUDA node)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
