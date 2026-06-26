"""GPU neighbor-list path: Vesin + CuPy + DLPack interchange with JAX.

Enabled when ``MMML_MM_NL_DEVICE=gpu``, CuPy and ``vesin>=0.5`` are available,
and positions already reside on the JAX GPU (e.g. ``jaxmd_runner`` block boundary).
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from mmml.interfaces.pycharmmInterface.nl_backend import (
    _resolve_max_pairs,
    pick_static_rebuild_backend,
)
from mmml.interfaces.pycharmmInterface.nl_reference import (
    cell_matrix_3x3,
    filter_vesin_half_list_vectorized,
    have_vesin,
    monomer_id_from_offsets,
    pad_pair_arrays,
    vesin_raw_half_list,
)

MmNlDeviceName = Literal["cpu", "gpu"]

_HAVE_CUPY = False
try:
    import cupy as cp

    _HAVE_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]


def have_cupy() -> bool:
    return _HAVE_CUPY


def resolve_mm_nl_device(name: str | None = None) -> MmNlDeviceName:
    """Resolve NL device from argument or ``MMML_MM_NL_DEVICE`` env (default ``cpu``)."""
    raw = (name or os.environ.get("MMML_MM_NL_DEVICE", "cpu")).strip().lower()
    if raw == "gpu":
        return "gpu"
    if raw != "cpu":
        raise ValueError(f"MMML_MM_NL_DEVICE must be cpu|gpu; got {raw!r}")
    return "cpu"


def gpu_nl_path_available() -> bool:
    return resolve_mm_nl_device() == "gpu" and have_cupy() and have_vesin()


def _jax_array_module():
    import jax.numpy as jnp

    return jnp


def positions_to_cupy(positions) -> "cp.ndarray":
    """Export positions to CuPy without host round-trip when already on GPU."""
    if not have_cupy():
        raise RuntimeError("CuPy is not installed")
    if isinstance(positions, cp.ndarray):
        return positions
    if hasattr(positions, "__dlpack_device__"):
        return cp.from_dlpack(positions)
    return cp.asarray(positions)


def cupy_to_jax(arr):
    """Import CuPy array to JAX via DLPack (zero-copy on same GPU)."""
    jnp = _jax_array_module()
    if hasattr(arr, "__dlpack__"):
        return jnp.from_dlpack(arr)
    return jnp.asarray(arr)


def rebuild_vesin_pairs_gpu(
    positions,
    box: np.ndarray,
    *,
    cutoff: float,
    monomer_offsets: np.ndarray,
    mm_r_min: float | None = None,
    max_pairs: int | None = None,
    cell_list_safety_factor: float = 2.5,
    cell_list_density_estimate: float | None = None,
    total_atoms: int | None = None,
    debug: bool = False,
) -> Tuple[object, object, str]:
    """Build MM pairs on GPU; return ``(pair_idx_jax, pair_mask_jax, backend_label)``."""
    if not gpu_nl_path_available():
        raise RuntimeError("GPU NL path requires MMML_MM_NL_DEVICE=gpu, cupy, and vesin>=0.5")

    pos_cp = positions_to_cupy(positions)
    n_atoms = int(total_atoms if total_atoms is not None else pos_cp.shape[0])
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    monomer_id = monomer_id_from_offsets(offsets, n_atoms)
    cell_mat = cell_matrix_3x3(np.asarray(box, dtype=np.float64))
    box_cp = cp.asarray(cell_mat)

    i_raw, j_raw, dist_raw = vesin_raw_half_list(
        pos_cp,
        cell_mat,
        cutoff,
        points_module=cp,
    )
    i_filt, j_filt = filter_vesin_half_list_vectorized(
        i_raw,
        j_raw,
        dist_raw,
        cutoff,
        monomer_id,
        pos_cp,
        cell_mat,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )
    capacity = _resolve_max_pairs(
        total_atoms=n_atoms,
        box=cell_mat,
        cutoff=cutoff,
        max_pairs=max_pairs,
        cell_list_safety_factor=cell_list_safety_factor,
        cell_list_density_estimate=cell_list_density_estimate,
    )
    pair_i, pair_j, mask, n_valid = pad_pair_arrays(
        i_filt,
        j_filt,
        max_pairs=capacity,
    )
    if debug:
        print(f"[nl_gpu:vesin] n_valid={n_valid} capacity={capacity}")

    pair_idx = cupy_to_jax(cp.stack([pair_i, pair_j], axis=1))
    pair_mask = cupy_to_jax(mask)
    return pair_idx, pair_mask, "vesin_gpu"


def profile_nl_sync_components(
    positions_jax,
    box: np.ndarray,
    *,
    cutoff: float,
    monomer_offsets: np.ndarray,
    mm_r_min: float | None = None,
    repeat: int = 20,
    warmup: int = 3,
) -> dict[str, float]:
    """Time D2H sync, CPU Vesin rebuild, H2D pairs, and GPU Vesin+DLPack path (ms)."""
    import statistics
    import time

    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend

    jax.block_until_ready(positions_jax)

    def _median_ms(fn, *, n: int) -> float:
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return float(statistics.median(samples))

    pos_host = np.asarray(jax.device_get(positions_jax), dtype=np.float64)
    box_np = np.asarray(box, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=np.int32)

    d2h_ms = _median_ms(
        lambda: np.asarray(jax.device_get(positions_jax), dtype=np.float64),
        n=repeat,
    )

    def _cpu_build_result():
        return build_mm_pairs_with_backend(
            pick_static_rebuild_backend(use_jax_md_neighbor_list=False),
            positions=pos_host,
            box=box_np,
            cutoff=cutoff,
            monomer_offsets=offsets,
            mm_r_min=mm_r_min,
            total_atoms=pos_host.shape[0],
        )

    cpu_build_ms = _median_ms(_cpu_build_result, n=repeat)

    def _h2d_pairs():
        cl_i, cl_j, cl_mask, *_ = _cpu_build_result()
        idx = jnp.stack([jnp.asarray(cl_i), jnp.asarray(cl_j)], axis=1)
        mask = jnp.asarray(cl_mask)
        jax.block_until_ready(idx)
        jax.block_until_ready(mask)

    h2d_pairs_ms = _median_ms(_h2d_pairs, n=repeat)

    gpu_ms = float("nan")
    if gpu_nl_path_available():
        gpu_ms = _median_ms(
            lambda: jax.block_until_ready(
                rebuild_vesin_pairs_gpu(
                    positions_jax,
                    box_np,
                    cutoff=cutoff,
                    monomer_offsets=offsets,
                    mm_r_min=mm_r_min,
                    total_atoms=pos_host.shape[0],
                )[0]
            ),
            n=max(3, repeat // 2),
        )

    return {
        "d2h_positions_ms": d2h_ms,
        "cpu_vesin_build_ms": cpu_build_ms,
        "h2d_pairs_ms": h2d_pairs_ms,
        "gpu_vesin_dlpack_ms": gpu_ms,
    }
