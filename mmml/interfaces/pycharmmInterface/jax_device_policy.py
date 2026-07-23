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
    # UNVERIFIED HEURISTIC [evidence: jax_persistent_cache_policy]. User env wins.
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    if not quiet and not _truthy("MMML_QUIET"):
        print(f"mmml: JAX compilation cache -> {cache_dir}", flush=True)
    return cache_dir


def apply_mlpot_jax_platform_env(*, quiet: bool = False) -> str:
    """Set ``JAX_PLATFORMS`` and compilation cache before the first ``import jax``."""
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        apply_jax_compile_xla_flags,
    )

    apply_jax_compile_xla_flags(quiet=quiet)
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


# Set by the most recent real (non-mocked) call to `mlpot_jax_device_context`;
# lets callers such as `DecomposedMlpotModel._finalize_jax_factory` verify the
# device actually used rather than trusting the request. Process-global by
# design: MLpot's device choice is a single process-wide policy, not
# per-thread. `reset_mlpot_device_fallback_flag` lets a caller establish a
# known baseline immediately before invoking (a possibly test-mocked)
# `mlpot_jax_device_context`.
_last_call_fell_back_to_cpu: bool = False


def reset_mlpot_device_fallback_flag() -> None:
    """Clear the fallback flag; call immediately before ``mlpot_jax_device_context()``."""
    global _last_call_fell_back_to_cpu
    _last_call_fell_back_to_cpu = False


def mlpot_device_context_fell_back_to_cpu() -> bool:
    """Whether the most recent real ``mlpot_jax_device_context()`` call fell back to CPU."""
    return _last_call_fell_back_to_cpu


@contextmanager
def mlpot_jax_device_context() -> Iterator[Any]:
    """Run MLpot JAX work on the selected device.

    Falls back to CPU when ``gpu`` was requested but JAX sees no GPU device
    (e.g. a CPU-only jaxlib install, or a long-running process whose jaxlib
    was loaded before a later ``uv sync --extra gpu``). The fallback itself
    is intentional -- CPU-only dev/CI machines default to
    ``MMML_MLPOT_DEVICE=gpu`` and must not hard-fail -- but callers
    (``DecomposedMlpotModel._finalize_jax_factory``) track "on GPU" state
    from this fallback outcome (see :func:`mlpot_device_context_fell_back_to_cpu`),
    not from the request, precisely because a silent fallback here previously
    let ``_jax_on_gpu`` stay ``True`` while the actual compute ran on CPU.
    Emit one unmistakable warning every time the fallback triggers so it can
    never look like a successful GPU run.
    """
    import jax

    global _last_call_fell_back_to_cpu

    name = mlpot_jax_device_name()
    requested_gpu = name == "gpu"
    fell_back_to_cpu = False
    if requested_gpu:
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            fell_back_to_cpu = True
            devices = jax.devices("cpu")
    else:
        devices = jax.devices("cpu")
    _last_call_fell_back_to_cpu = fell_back_to_cpu
    if fell_back_to_cpu:
        _warn_gpu_requested_but_unavailable()
    with jax.default_device(devices[0]):
        yield devices[0]


def _warn_gpu_requested_but_unavailable() -> None:
    message = (
        "MMML_MLPOT_DEVICE=gpu requested but JAX sees no GPU device "
        "(jax.devices('gpu') is empty) -- falling back to CPU. If a GPU "
        "should be present: uv sync --extra gpu, then restart this process "
        "(a running interpreter keeps whatever jaxlib it already imported; "
        "resyncing on disk does not change it). Set MMML_MLPOT_DEVICE=cpu "
        "to silence this if CPU is intentional."
    )
    try:
        from mmml.utils.rich_report import get_reporter

        get_reporter().status("warning", message)
    except Exception:
        print(f"mmml WARNING: {message}", flush=True)


@contextmanager
def jax_cpu_until_mlpot_registered() -> Iterator[Any]:
    """Keep JAX array placement on CPU until CHARMM MLpot ``upinb`` completes."""
    import jax

    cpu = jax.devices("cpu")
    with jax.default_device(cpu[0]):
        yield cpu[0]
