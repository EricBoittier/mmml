"""JAX device and compilation-cache setup for MLpot with OpenMPI-linked CHARMM.

GPU selection
-------------
- ``MMML_MLPOT_DEVICE``: ``gpu`` (default) or ``cpu``.
- ``CUDA_VISIBLE_DEVICES``: restrict which physical GPUs JAX sees (e.g. ``0`` or ``0,1``).
- ``MMML_MLPOT_N_GPUS`` / ``--ml-gpu-count``: parallel PhysNet *chunks* across local GPUs
  (default 1). Does not split CHARMM integration across devices.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def mlpot_local_gpu_count() -> int:
    """Number of visible JAX GPU devices (0 when running on CPU only)."""
    try:
        import jax

        return len(jax.devices("gpu"))
    except Exception:
        return 0


def mlpot_jax_device_name() -> str:
    """``cpu`` or ``gpu`` for MLpot energy/force evaluation."""
    mode = (os.environ.get("MMML_MLPOT_DEVICE") or "gpu").strip().lower()
    if mode in ("cpu", "gpu"):
        return mode
    if mode == "auto":
        return "gpu"
    return "gpu"


def mlpot_jax_compilation_cache_dir() -> Path | None:
    """Persistent JIT cache directory (``None`` when disabled)."""
    if _truthy("MMML_NO_JAX_COMPILATION_CACHE"):
        return None
    override = (os.environ.get("JAX_COMPILATION_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    cache_home = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    if cache_home:
        base = Path(cache_home).expanduser()
    else:
        base = Path.home() / ".cache"
    return base / "mmml" / "jax-compilation-cache"


def apply_mlpot_jax_compilation_cache_env(*, quiet: bool = False) -> Path | None:
    """Enable JAX persistent compilation cache before the first ``import jax``."""
    cache_dir = mlpot_jax_compilation_cache_dir()
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    # Reuse GPU autotuning across runs (safe default for MLpot PhysNet JIT).
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    if not quiet and not _truthy("MMML_QUIET"):
        print(f"mmml: JAX compilation cache -> {cache_dir}", flush=True)
    return cache_dir


def apply_mlpot_jax_platform_env(*, quiet: bool = False) -> str:
    """Set ``JAX_PLATFORMS`` and compilation cache before the first ``import jax``."""
    device = mlpot_jax_device_name()
    os.environ.setdefault("JAX_PLATFORMS", device)
    apply_mlpot_jax_compilation_cache_env(quiet=quiet)
    if not quiet and not _truthy("MMML_QUIET") and device == "cpu":
        print(
            "mmml: MLpot JAX runs on CPU (MMML_MLPOT_DEVICE=cpu). "
            "Unset or set MMML_MLPOT_DEVICE=gpu for GPU.",
            flush=True,
        )
    return device


def jax_warmup_device_name() -> str:
    """Warmup backend; follows :func:`mlpot_jax_device_name` unless overridden."""
    mode = (os.environ.get("MMML_JAX_WARMUP_DEVICE") or "auto").strip().lower()
    if mode in ("cpu", "gpu"):
        return mode
    if mode == "auto":
        return mlpot_jax_device_name()
    return "gpu"


@contextmanager
def mlpot_jax_device_context() -> Iterator[Any]:
    """Run MLpot JAX work on the selected device."""
    import jax

    name = mlpot_jax_device_name()
    devices = jax.devices(name)
    if not devices:
        devices = jax.devices("cpu")
    with jax.default_device(devices[0]):
        yield devices[0]


@contextmanager
def jax_cpu_until_mlpot_registered() -> Iterator[Any]:
    """Keep JAX array placement on CPU until CHARMM MLpot ``upinb`` completes."""
    import jax

    cpu = jax.devices("cpu")
    with jax.default_device(cpu[0]):
        yield cpu[0]
