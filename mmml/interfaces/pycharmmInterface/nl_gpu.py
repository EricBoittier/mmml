"""GPU neighbor-list path: Vesin + CuPy + DLPack interchange with JAX.

Enabled when ``MMML_MM_NL_DEVICE=gpu``, CuPy and ``vesin>=0.5`` are available,
and positions already reside on the JAX GPU (e.g. ``jaxmd_runner`` block boundary).

Contract: callers pass Cartesian Å positions on device and a scalar, ``(3,)``,
or ``(3, 3)`` Å cell. The returned JAX arrays are padded ``pair_idx`` with shape
``(capacity, 2)`` and boolean ``pair_mask`` with shape ``(capacity,)``. Only
``mask == True`` entries are valid; pair order is not stable API.

CUDA toolkit note
-----------------
CuPy NVRTC adds ``-I$CUDA_PATH/include``. On some HPC images
``/usr/local/cuda`` is a symlink to an ancient toolkit (e.g. CUDA 9.0) whose
``cuda_fp16.hpp`` does ``#include <utility>`` under NVRTC and fails with
``cannot open source file "utility"``. Pip ``nvidia-cuda-runtime-cu12`` wheels
ship modern headers that work with NVRTC; :func:`ensure_cupy_cuda_path`
points ``CUDA_PATH`` at those wheels when the system toolkit looks unusable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple

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

_CUPY_RUNTIME_OK: bool | None = None
_CUDA_PATH_ENSURED = False


def have_cupy() -> bool:
    return _HAVE_CUPY


def _nvidia_wheel_cuda_runtime_root() -> str | None:
    """Return ``.../nvidia/cuda_runtime`` (or cu13 layout) that has ``include/cuda_fp16.hpp``."""
    import importlib.metadata

    candidates = (
        ("nvidia-cuda-runtime-cu12", "cuda_runtime"),
        ("nvidia-cuda-runtime-cu13", "cuda_runtime"),
        ("nvidia-cuda-runtime", "cu13"),
        ("nvidia-cuda-runtime", "cu12"),
    )
    for pkg_name, dir_name in candidates:
        try:
            dist = importlib.metadata.distribution(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        root = Path(dist.locate_file(f"nvidia/{dir_name}"))
        if (root / "include" / "cuda_fp16.hpp").is_file():
            return str(root.resolve())
    # Fallback: walk site-packages next to cupy.
    if _HAVE_CUPY and cp is not None:
        site = Path(cp.__file__).resolve().parents[1]
        for dir_name in ("cuda_runtime", "cu13", "cu12"):
            root = site / "nvidia" / dir_name
            if (root / "include" / "cuda_fp16.hpp").is_file():
                return str(root)
    return None


def _fp16_header_path(cuda_root: str) -> Path | None:
    root = Path(cuda_root)
    for candidate in (
        root / "include" / "cuda_fp16.hpp",
        root / "targets" / "x86_64-linux" / "include" / "cuda_fp16.hpp",
    ):
        if candidate.is_file():
            return candidate
    return None


def cuda_path_looks_broken(cuda_path: str | None) -> bool:
    """True when CuPy NVRTC would likely fail with this ``CUDA_PATH``."""
    if not cuda_path:
        return True
    root = Path(cuda_path)
    if not root.exists():
        return True
    real = str(root.resolve()).lower()
    if any(tag in real for tag in ("cuda-8", "cuda-9", "cuda-10.0", "cuda-10.1")):
        return True
    fp16 = _fp16_header_path(str(root))
    if fp16 is None:
        return True
    try:
        head = fp16.read_text(encoding="utf-8", errors="ignore")[:8000]
    except OSError:
        return True
    # CUDA ≤9 style: top-level ``#include <utility>`` without NVRTC exclusion.
    # Modern wheel headers do not pull host ``<utility>`` for NVRTC.
    util = head.find("#include <utility>")
    if util < 0:
        return False
    before = head[:util]
    # If an NVRTC-only / host-skip guard wraps the include, accept it.
    window = before[-400:]
    if "CUDACC_RTC" in window and ("ifndef" in window or "if !" in window):
        return False
    return True


def ensure_cupy_cuda_path(*, force: bool = False, quiet: bool = False) -> str | None:
    """Ensure ``CUDA_PATH`` points at NVRTC-usable CUDA headers.

    Returns the effective CUDA root (wheel or existing), or ``None``.
    """
    global _CUDA_PATH_ENSURED
    if _CUDA_PATH_ENSURED and not force:
        return os.environ.get("CUDA_PATH") or None

    current = (os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "").strip()
    if not current:
        # Mirror CuPy's discovery so we can decide whether to override.
        try:
            import cupy._environment as cupy_env

            current = cupy_env.get_cuda_path() or ""
        except Exception:
            if Path("/usr/local/cuda").exists():
                current = "/usr/local/cuda"

    wheel_root = _nvidia_wheel_cuda_runtime_root()
    if wheel_root and (force or cuda_path_looks_broken(current or None)):
        os.environ["CUDA_PATH"] = wheel_root
        os.environ["CUDA_HOME"] = wheel_root
        # Prefer wheel NVRTC libs when present.
        nvrtc_lib = Path(wheel_root).parent / "cuda_nvrtc" / "lib"
        if nvrtc_lib.is_dir():
            prev = os.environ.get("LD_LIBRARY_PATH", "")
            prefix = str(nvrtc_lib)
            if prefix not in prev.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{prefix}:{prev}" if prev else prefix
                )
        try:
            import cupy._environment as cupy_env

            cupy_env._cuda_path = wheel_root
        except Exception:
            pass
        _patch_cupy_wheel_includes(str(Path(wheel_root) / "include"))
        if not quiet:
            old = current or "(unset)"
            print(
                f"[nl_gpu] CUDA_PATH {old} → {wheel_root} "
                f"(pip nvidia-cuda-runtime headers for CuPy NVRTC)",
                flush=True,
            )
        _CUDA_PATH_ENSURED = True
        return wheel_root

    if current:
        try:
            import cupy._environment as cupy_env

            cupy_env._cuda_path = current
        except Exception:
            pass
    _CUDA_PATH_ENSURED = True
    return current or None


def _patch_cupy_wheel_includes(wheel_include: str) -> None:
    """Prepend pip runtime ``-I`` so NVRTC never picks stale toolkit headers first."""
    if not _HAVE_CUPY or not wheel_include or not Path(wheel_include).is_dir():
        return
    try:
        from cupy.cuda import compiler
    except Exception:
        return
    flag = f"-I{wheel_include}"
    existing = getattr(compiler, "_get_extra_include_dir_opts", None)
    if existing is None or getattr(existing, "_mmml_wheel_include", None) == flag:
        return

    def _wrapped():
        opts = tuple(existing())
        if flag not in opts:
            opts = (flag,) + opts
        return opts

    _wrapped._mmml_wheel_include = flag  # type: ignore[attr-defined]
    # Clear memoized empty include-dir results from before the patch.
    cache = getattr(existing, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    compiler._get_extra_include_dir_opts = _wrapped  # type: ignore[assignment]


def cupy_runtime_ok(*, force: bool = False) -> bool:
    """Return True if CuPy can JIT a trivial kernel on this host.

    Runs :func:`ensure_cupy_cuda_path` first. Some CUDA/NVRTC + stale
    ``/usr/local/cuda`` setups import CuPy but fail on the first kernel
    compile (``#include <utility>``). Probe once and cache so
    ``MMML_MM_NL_DEVICE=gpu`` can fall back cleanly instead of crashing MD.
    """
    global _CUPY_RUNTIME_OK
    if not force and _CUPY_RUNTIME_OK is not None:
        return _CUPY_RUNTIME_OK
    if not have_cupy():
        _CUPY_RUNTIME_OK = False
        return False

    ensure_cupy_cuda_path(force=force)

    def _probe() -> bool:
        x = cp.arange(4, dtype=cp.float32)
        y = x + cp.asarray(1, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        float(y.sum())
        return True

    try:
        _CUPY_RUNTIME_OK = _probe()
    except Exception:
        # One retry after a forced path repair (covers import-order races).
        try:
            ensure_cupy_cuda_path(force=True)
            _CUPY_RUNTIME_OK = _probe()
        except Exception:
            _CUPY_RUNTIME_OK = False
    return bool(_CUPY_RUNTIME_OK)


def resolve_mm_nl_device(name: str | None = None) -> MmNlDeviceName:
    """Resolve NL device from argument or ``MMML_MM_NL_DEVICE`` env (default ``cpu``)."""
    raw = (name or os.environ.get("MMML_MM_NL_DEVICE", "cpu")).strip().lower()
    if raw == "gpu":
        return "gpu"
    if raw != "cpu":
        raise ValueError(f"MMML_MM_NL_DEVICE must be cpu|gpu; got {raw!r}")
    return "cpu"


def gpu_nl_path_available() -> bool:
    return (
        resolve_mm_nl_device() == "gpu"
        and have_cupy()
        and have_vesin()
        and cupy_runtime_ok()
    )


def _jax_array_module():
    import jax.numpy as jnp

    return jnp


def positions_to_cupy(positions) -> "cp.ndarray":
    """Export positions to CuPy without host round-trip when already on GPU."""
    if not have_cupy():
        raise RuntimeError("CuPy is not installed")
    ensure_cupy_cuda_path()
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
    """Build padded MM pairs on GPU from Cartesian Å coordinates."""
    if not gpu_nl_path_available():
        raise RuntimeError(
            "GPU NL path requires MMML_MM_NL_DEVICE=gpu, working CuPy JIT, and vesin>=0.5"
        )

    pos_cp = positions_to_cupy(positions)
    n_atoms = int(total_atoms if total_atoms is not None else pos_cp.shape[0])
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    monomer_id = monomer_id_from_offsets(offsets, n_atoms)
    cell_mat = cell_matrix_3x3(np.asarray(box, dtype=np.float64))
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
