"""GPU neighbor-list path: Vesin + CuPy + DLPack interchange with JAX.

Selected automatically (``MMML_MM_NL_DEVICE=auto``, the default) when CuPy and
``vesin>=0.6.1`` work and JAX's default device is a GPU (``pip install
'mmml[nl-gpu]'``); ``MMML_MM_NL_DEVICE=cpu`` forces the Vesin + NumPy path.
Positions may already be on the JAX GPU (no host copy) or on the host (one small
H2D copy); the padded pairs are handed to JAX via DLPack either way.

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
    have_vesin,
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


def _cupy_device_ctx(ordinal: int | None):
    """``cp.cuda.Device(ordinal)`` context, or a no-op when ``ordinal`` is None."""
    import contextlib

    if ordinal is None or cp is None:
        return contextlib.nullcontext()
    return cp.cuda.Device(int(ordinal))


def cupy_runtime_ok(*, force: bool = False, device_ordinal: int | None = None) -> bool:
    """Return True if CuPy can JIT a trivial kernel on this host.

    Runs :func:`ensure_cupy_cuda_path` first. Some CUDA/NVRTC + stale
    ``/usr/local/cuda`` setups import CuPy but fail on the first kernel
    compile (``#include <utility>``). Probe once and cache so the GPU pair list
    can fall back cleanly instead of crashing MD. The probe runs on
    ``device_ordinal`` (the JAX device's CUDA ordinal) so it never opens a
    context on another, possibly exclusive-process-busy, GPU.
    """
    global _CUPY_RUNTIME_OK
    if not force and _CUPY_RUNTIME_OK is not None:
        return _CUPY_RUNTIME_OK
    if not have_cupy():
        _CUPY_RUNTIME_OK = False
        return False

    ensure_cupy_cuda_path(force=force, quiet=True)

    def _probe() -> bool:
        with _cupy_device_ctx(device_ordinal):
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
            ensure_cupy_cuda_path(force=True, quiet=True)
            _CUPY_RUNTIME_OK = _probe()
        except Exception:
            _CUPY_RUNTIME_OK = False
    return bool(_CUPY_RUNTIME_OK)


MmNlDeviceRequest = Literal["cpu", "gpu", "auto"]


def resolve_mm_nl_device_request(name: str | None = None) -> MmNlDeviceRequest:
    """Requested NL device: argument > ``MMML_MM_NL_DEVICE`` > ``auto``.

    ``auto`` (default) uses the GPU pair list when CuPy + ``vesin>=0.6.1`` work
    and JAX's default device is a GPU; otherwise the CPU (Vesin + NumPy) path.
    ``gpu`` asks for the same path but warns once if it is unavailable;
    ``cpu`` never touches CuPy.
    """
    raw = (name or os.environ.get("MMML_MM_NL_DEVICE") or "auto").strip().lower()
    if raw in ("cpu", "gpu", "auto"):
        return raw  # type: ignore[return-value]
    raise ValueError(f"MMML_MM_NL_DEVICE must be auto|cpu|gpu; got {raw!r}")


def resolve_mm_nl_device(name: str | None = None) -> MmNlDeviceName:
    """Effective NL device (``cpu`` or ``gpu``) after resolving ``auto``."""
    if resolve_mm_nl_device_request(name) == "cpu":
        return "cpu"
    return "gpu" if gpu_nl_path_available(name) else "cpu"


_VESIN_GPU_OK: bool | None = None


def _vesin_version_tuple() -> tuple[int, ...]:
    try:
        import importlib.metadata

        raw = importlib.metadata.version("vesin")
    except Exception:
        return ()
    parts: list[int] = []
    for tok in raw.split("."):
        digits = "".join(ch for ch in tok if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _vesin_gpu_version_ok() -> bool:
    """Blackwell (sm_120) needs vesin>=0.6.1; older wheels reject the arch."""
    if not have_vesin():
        return False
    ver = _vesin_version_tuple()
    if not ver:
        # Metadata unavailable: let vesin fail at compute time.
        return True
    return ver >= (0, 6, 1)


def _jax_target_device(positions=None):
    """JAX device the pair arrays must land on (positions' device, else default)."""
    devs = getattr(positions, "devices", None)
    if callable(devs):
        try:
            ds = list(devs())
            if len(ds) == 1:
                return ds[0]
        except Exception:
            pass
    import jax

    try:
        from jax._src import config as _jax_config

        dev = _jax_config.default_device.value  # honours ``with jax.default_device``
    except Exception:
        dev = None
    if isinstance(dev, str):
        try:
            dev = jax.devices(dev)[0]
        except Exception:
            dev = None
    if dev is None:
        dev = jax.devices()[0]
    return dev


def _cuda_ordinal(device) -> int | None:
    """CUDA ordinal of a JAX GPU device, or None for non-GPU devices."""
    if device is None or getattr(device, "platform", "") != "gpu":
        return None
    ordinal = getattr(device, "local_hardware_id", None)
    if ordinal is None:
        ordinal = getattr(device, "id", 0)
    return int(ordinal)


_WARNED_GPU_UNAVAILABLE = False


def gpu_nl_path_available(name: str | None = None, *, positions=None) -> bool:
    """True when the GPU pair-list rebuild can run for the current JAX device.

    Requires: request ``auto``/``gpu``, CuPy importable and able to JIT on the
    JAX device, ``vesin>=0.6.1``, and a JAX GPU target device (pairs are handed
    to JAX via DLPack on that device).
    """
    global _VESIN_GPU_OK, _WARNED_GPU_UNAVAILABLE
    req = resolve_mm_nl_device_request(name)
    if req == "cpu":
        return False
    reason = None
    if not have_cupy():
        reason = "cupy not installed (pip install 'mmml[nl-gpu]')"
    elif not have_vesin():
        reason = "vesin not installed"
    else:
        if _VESIN_GPU_OK is None:
            _VESIN_GPU_OK = _vesin_gpu_version_ok()
        if not _VESIN_GPU_OK:
            reason = "vesin<0.6.1"
    ordinal = None
    if reason is None:
        try:
            ordinal = _cuda_ordinal(_jax_target_device(positions))
        except Exception as exc:  # pragma: no cover - defensive
            reason = f"JAX device lookup failed ({exc})"
        if reason is None and ordinal is None:
            reason = "JAX default device is not a GPU"
    if reason is None and not cupy_runtime_ok(device_ordinal=ordinal):
        reason = "CuPy runtime probe failed"
    if reason is None:
        return True
    if req == "gpu" and not _WARNED_GPU_UNAVAILABLE:
        _WARNED_GPU_UNAVAILABLE = True
        print(f"[nl_gpu] MMML_MM_NL_DEVICE=gpu but GPU pair list unavailable: {reason}; using CPU", flush=True)
    return False


def _jax_array_module():
    import jax.numpy as jnp

    return jnp


def is_device_array(x) -> bool:
    """True for JAX/CuPy buffers, not host NumPy.

    NumPy ≥1.23 implements ``__dlpack_device__``, so that attribute is not
    evidence of a GPU array. Treating host coordinates as JAX arrays and
    running them through ``jnp.asarray`` with x64 disabled silently rounds
    float64 → float32 (12.4699999 → 12.470000267) and can flip pair
    membership at a strict cutoff.
    """
    return x is not None and not isinstance(x, np.ndarray) and hasattr(x, "__dlpack_device__")


def positions_to_cupy(positions) -> "cp.ndarray":
    """Export positions to CuPy without host round-trip when already on GPU.

    Host NumPy stays float64 on the H2D copy. Device arrays use DLPack.
    """
    if not have_cupy():
        raise RuntimeError("CuPy is not installed")
    ensure_cupy_cuda_path(quiet=True)
    if isinstance(positions, np.ndarray):
        return cp.asarray(positions, dtype=cp.float64)
    if isinstance(positions, cp.ndarray):
        return positions
    if is_device_array(positions):
        return cp.from_dlpack(positions)
    return cp.asarray(positions, dtype=cp.float64)


def cupy_to_jax(arr):
    """Import CuPy array to JAX via DLPack (zero-copy on same GPU)."""
    jnp = _jax_array_module()
    if hasattr(arr, "__dlpack__"):
        return jnp.from_dlpack(arr)
    return jnp.asarray(arr)


_VESIN_CALCULATORS: dict[float, object] = {}


def _vesin_calculator(cutoff: float):
    from vesin import NeighborList

    calc = _VESIN_CALCULATORS.get(float(cutoff))
    if calc is None:
        calc = NeighborList(cutoff=float(cutoff), full_list=False)
        _VESIN_CALCULATORS[float(cutoff)] = calc
    return calc


def vesin_mic_pair_keys_cupy(
    pos_cp,
    cell_mat: np.ndarray,
    cutoff: float,
    monomer_offsets: np.ndarray,
    *,
    mm_r_min: float | None = None,
):
    """Sorted unique half-list pair keys ``i * (n + 1) + j`` (``i < j``) on the GPU.

    Same rule as :func:`nl_reference.vesin_mic_pair_arrays` (the CPU path):
    ``dist < cutoff``, canonical ``i < j``, inter-monomer only, dimer COM
    distance ``>= mm_r_min`` by MIC, lexicographic order without duplicates.
    """
    from mmml.interfaces.pycharmmInterface.nl_reference import unique_mic_orthorhombic

    n = int(pos_cp.shape[0])
    cutoff = float(cutoff)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    n_mono = len(counts)
    mid = cp.asarray(np.repeat(np.arange(n_mono, dtype=np.int32), counts))
    quantities = "ijd" if unique_mic_orthorhombic(cell_mat, cutoff) else "ijSd"
    out = _vesin_calculator(cutoff).compute(
        points=pos_cp, box=cp.asarray(cell_mat), periodic=True, quantities=quantities
    )
    i = cp.asarray(out[0], dtype=cp.int64)
    j = cp.asarray(out[1], dtype=cp.int64)
    dist = cp.asarray(out[-1], dtype=cp.float64)
    lo = cp.minimum(i, j)
    hi = cp.maximum(i, j)
    keep = (dist < cutoff) & (lo != hi) & (mid[lo] != mid[hi])
    if mm_r_min is not None and n_mono > 1:
        R = pos_cp[: int(offsets[-1])]
        # Whole molecules before centroids (see nl_reference.mm_pair_filter_mask).
        anchor = cp.asarray(np.repeat(np.asarray(offsets[:-1], dtype=np.int64), np.asarray(counts, dtype=np.int64)))
        frac_a = (R - R[anchor]) @ cp.asarray(np.linalg.inv(cell_mat).T)
        R = R[anchor] + (frac_a - cp.round(frac_a)) @ cp.asarray(cell_mat)
        if int(counts.min()) == int(counts.max()):
            coms = R.reshape(n_mono, int(counts[0]), 3).sum(axis=1) / float(counts[0])
        else:
            csum = cp.concatenate([cp.zeros((1, 3), dtype=R.dtype), cp.cumsum(R, axis=0)])
            off_cp = cp.asarray(offsets)
            coms = (csum[off_cp[1:]] - csum[off_cp[:-1]]) / cp.asarray(counts, dtype=R.dtype)[:, None]
        dcom = coms[None, :, :] - coms[:, None, :]
        frac = dcom @ cp.asarray(np.linalg.inv(cell_mat).T)
        dcom = (frac - cp.round(frac)) @ cp.asarray(cell_mat)
        com_ok = cp.linalg.norm(dcom, axis=2) >= float(mm_r_min)
        cp.fill_diagonal(com_ok, False)
        keep &= com_ok[mid[lo], mid[hi]]
    key = lo[keep] * (n + 1) + hi[keep]
    return cp.unique(key), n


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
    check_available: bool = True,
) -> Tuple[object, object, str]:
    """Build padded MM pairs on GPU from Cartesian Å coordinates.

    ``positions`` may be a JAX GPU array (DLPack, no host copy), a CuPy array,
    or a host NumPy array (one small H2D copy). The pair set and order are
    identical to the CPU Vesin rebuild; the padded arrays go to JAX via DLPack
    without a device-to-host round trip. Only the pair count is synchronized.
    """
    if check_available and not gpu_nl_path_available(positions=positions):
        raise RuntimeError(
            "GPU NL path requires MMML_MM_NL_DEVICE=auto|gpu, working CuPy JIT, "
            "vesin>=0.6.1 (Blackwell/sm_120), and a JAX GPU device"
        )
    ordinal = _cuda_ordinal(_jax_target_device(positions))
    cell_mat = cell_matrix_3x3(np.asarray(box, dtype=np.float64))
    with _cupy_device_ctx(ordinal):
        pos_cp = cp.asarray(positions_to_cupy(positions), dtype=cp.float64)
        n_atoms = int(total_atoms if total_atoms is not None else pos_cp.shape[0])
        key, n = vesin_mic_pair_keys_cupy(
            pos_cp[:n_atoms], cell_mat, cutoff, monomer_offsets, mm_r_min=mm_r_min
        )
        n_valid = int(key.shape[0])
        capacity = _resolve_max_pairs(
            total_atoms=n_atoms,
            box=cell_mat,
            cutoff=cutoff,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            cell_list_density_estimate=cell_list_density_estimate,
        )
        if n_valid > capacity:
            from mmml.interfaces.pycharmmInterface.cell_list import PairListTruncationError

            raise PairListTruncationError(n_valid, capacity)
        pair_idx = cp.zeros((int(capacity), 2), dtype=cp.int32)
        pair_idx[:n_valid, 0] = (key // (n + 1)).astype(cp.int32)
        pair_idx[:n_valid, 1] = (key % (n + 1)).astype(cp.int32)
        mask = cp.arange(int(capacity)) < n_valid
        if debug:
            print(f"[nl_gpu:vesin] n_valid={n_valid} capacity={capacity}")
        return cupy_to_jax(pair_idx), cupy_to_jax(mask), "vesin_gpu"


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
