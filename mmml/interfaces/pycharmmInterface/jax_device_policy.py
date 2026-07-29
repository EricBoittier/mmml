"""JAX device and compilation-cache setup for MLpot with OpenMPI-linked CHARMM.

GPU selection
-------------
- ``MMML_MLPOT_DEVICE``: ``gpu`` (default) or ``cpu``.
- ``CUDA_VISIBLE_DEVICES``: restrict which physical GPUs JAX sees (e.g. ``0`` or ``0,1``).
- ``MMML_MLPOT_N_GPUS`` / ``--ml-gpu-count``: parallel PhysNet *chunks* across local GPUs
  (default 1). Does not split CHARMM integration across devices.
- ``JAX_PLATFORMS``: default ``gpu,cpu`` when ``MMML_MLPOT_DEVICE=gpu`` (CPU kept for
  MPI defer / fallback). GPU-only values like ``cuda`` are expanded to ``cuda,cpu``
  before the first ``import jax``.
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


def sanitize_stale_jax_platforms_env(*, prefer_cuda: bool | None = None) -> str | None:
    """Drop unusable ``rocm`` tokens from ``JAX_PLATFORMS`` (NVIDIA studix nodes).

    Mixed AMD/NVIDIA clusters often export ``JAX_PLATFORMS=rocm`` into every
    shell. JAX builds with only ``cuda`` then abort on the first ``import jax``
    / ``jax_md`` with ``Backend 'rocm' is not in the list of known backends``.

    Parameters
    ----------
    prefer_cuda
        When True and stripping leaves the list empty, set ``cuda``.
        When False, leave unset (JAX auto-selects). When None (default), set
        ``cuda`` if a GPU allocation is visible (``CUDA_VISIBLE_DEVICES``,
        ``SLURM_JOB_GPUS``, or a GPU Slurm partition), otherwise leave unset.

    Returns
    -------
    The resulting ``JAX_PLATFORMS`` value, or ``None`` if unset.
    """
    # Deprecated alias still seen in old profiles.
    legacy = (os.environ.get("JAX_PLATFORM_NAME") or "").strip().lower()
    if legacy == "rocm":
        os.environ.pop("JAX_PLATFORM_NAME", None)

    existing = (os.environ.get("JAX_PLATFORMS") or "").strip()
    if not existing:
        return None

    parts = [p.strip() for p in existing.split(",") if p.strip()]
    cleaned = [p for p in parts if p.lower() != "rocm"]
    if cleaned == parts:
        return existing

    if cleaned:
        os.environ["JAX_PLATFORMS"] = ",".join(cleaned)
        return os.environ["JAX_PLATFORMS"]

    os.environ.pop("JAX_PLATFORMS", None)
    if prefer_cuda is None:
        prefer_cuda = bool(
            (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
            or (os.environ.get("SLURM_JOB_GPUS") or "").strip()
            or "gpu" in str(os.environ.get("SLURM_JOB_PARTITION", "")).lower()
        )
    if prefer_cuda:
        os.environ["JAX_PLATFORMS"] = "cuda"
        return "cuda"
    return None


def mlpot_jax_platforms_for_device(device: str) -> str:
    """``JAX_PLATFORMS`` value for ``cpu`` / ``gpu`` MLpot device selection.

    GPU mode keeps ``cpu`` registered as well: MPI-defer registration,
    jax-pme host paths, and GPU→CPU fallback all call ``jax.devices("cpu")``.
    With ``JAX_PLATFORMS=gpu`` alone, JAX reports only ``['cuda']`` and those
    calls raise ``Unknown backend cpu``.

    Prefer the explicit ``cuda`` token over the alias ``gpu`` so a stale ROCm
    install cannot be selected on NVIDIA nodes.

    When no ``jax-cuda*-plugin`` is installed, return ``cpu`` even if GPU was
    requested so CPU-only envs do not hard-fail on a missing CUDA backend.
    """
    name = (device or "gpu").strip().lower()
    if name == "cpu":
        return "cpu"
    try:
        from mmml.utils.jax_gpu_warmup import _installed_jax_cuda_plugins

        if not _installed_jax_cuda_plugins():
            return "cpu"
    except Exception:
        return "cpu"
    # CUDA first so it remains the default device; CPU stays available.
    return "cuda,cpu"


def _expand_gpu_platforms_to_include_cpu(existing: str) -> str | None:
    """If ``existing`` is GPU-only, return an expanded ``gpu|cuda,cpu`` string."""
    parts = [p.strip().lower() for p in existing.split(",") if p.strip()]
    if not parts or "cpu" in parts:
        return None
    if not any(p in ("gpu", "cuda") for p in parts):
        return None
    return f"{existing},cpu"


def apply_mlpot_jax_platform_env(*, quiet: bool = False) -> str:
    """Set ``JAX_PLATFORMS`` and compilation cache before the first ``import jax``."""
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        apply_jax_compile_xla_flags,
    )

    apply_jax_compile_xla_flags(quiet=quiet)
    device = mlpot_jax_device_name()
    wanted = mlpot_jax_platforms_for_device(device)
    sanitize_stale_jax_platforms_env(prefer_cuda=(device == "gpu" and wanted != "cpu"))
    existing = (os.environ.get("JAX_PLATFORMS") or "").strip()
    if not existing:
        os.environ["JAX_PLATFORMS"] = wanted
    elif device == "gpu" and wanted != "cpu":
        expanded = _expand_gpu_platforms_to_include_cpu(existing)
        if expanded is not None:
            import sys

            if "jax" in sys.modules:
                if not quiet and not _truthy("MMML_QUIET"):
                    print(
                        "mmml WARNING: JAX_PLATFORMS="
                        f"{existing!r} is GPU-only and jax is already imported; "
                        "cannot register the CPU backend. Restart with "
                        f"JAX_PLATFORMS={expanded!r} (or unset it).",
                        flush=True,
                    )
            else:
                os.environ["JAX_PLATFORMS"] = expanded
                if not quiet and not _truthy("MMML_QUIET"):
                    print(
                        f"mmml: JAX_PLATFORMS expanded {existing!r} → {expanded!r} "
                        "(CPU kept for MLpot defer/fallback)",
                        flush=True,
                    )
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


def jax_registered_backend_names() -> list[str]:
    """Backend names JAX actually registered (empty if jax is not imported yet)."""
    import sys

    if "jax" not in sys.modules:
        return []
    try:
        import jax

        # Prefer the public-ish bridge helper when present.
        from jax._src import xla_bridge

        return sorted(xla_bridge.backends().keys())
    except Exception:
        names: list[str] = []
        for platform in ("cpu", "gpu", "cuda", "tpu"):
            try:
                import jax

                if jax.devices(platform):
                    names.append(platform)
            except Exception:
                continue
        return names


def jax_cpu_backend_available() -> bool:
    """True when ``jax.devices('cpu')`` works with the already-initialized runtime."""
    import sys

    if "jax" not in sys.modules:
        # Not initialized yet — CPU will be available unless platforms exclude it.
        platforms = (os.environ.get("JAX_PLATFORMS") or "").strip().lower()
        if not platforms:
            return True
        return "cpu" in {p.strip() for p in platforms.split(",") if p.strip()}
    try:
        import jax

        return bool(jax.devices("cpu"))
    except Exception:
        return False


def jax_gpu_backend_available() -> bool:
    """True when at least one JAX GPU/CUDA device is visible."""
    import sys

    if "jax" not in sys.modules:
        return False
    try:
        import jax

        return bool(jax.devices("gpu"))
    except Exception:
        try:
            import jax

            return bool(jax.devices("cuda"))
        except Exception:
            return False


@contextmanager
def jax_cpu_until_mlpot_registered() -> Iterator[Any]:
    """Keep JAX array placement on CPU until CHARMM MLpot ``upinb`` completes.

    If JAX was already initialized without a CPU backend (common when an early
    import used ``JAX_PLATFORMS=gpu`` / ``cuda`` only, then the env was later
    expanded to ``gpu,cpu``), fall back to GPU instead of aborting. MPI defer
    prefers CPU to keep XLA off-GPU during ``upinb``, but a working GPU is
    safer than a hard failure when CPU is unreachable.
    """
    import jax

    try:
        cpu = jax.devices("cpu")
    except RuntimeError as exc:
        platforms = (os.environ.get("JAX_PLATFORMS") or "").strip() or "(unset)"
        registered = jax_registered_backend_names()
        gpu: list[Any] = []
        try:
            gpu = list(jax.devices("gpu"))
        except Exception:
            try:
                gpu = list(jax.devices("cuda"))
            except Exception:
                gpu = []
        if gpu:
            # Default GPU policy: CPU defer is best-effort. Missing CPU usually
            # means jax was imported earlier with a GPU-only platform list; the
            # env may already say gpu,cpu but that cannot be fixed in-process.
            want_cpu = mlpot_jax_device_name() == "cpu"
            level = "WARNING" if want_cpu else "NOTE"
            print(
                f"mmml {level}: JAX CPU backend not registered "
                f"(JAX_PLATFORMS={platforms!r}, registered={registered or ['?']}); "
                "using GPU. To register CPU as well, restart with "
                "JAX_PLATFORMS=gpu,cpu before any import jax.",
                flush=True,
            )
            with jax.default_device(gpu[0]):
                yield gpu[0]
            return
        raise RuntimeError(
            "MLpot deferred/CPU path needs the JAX CPU backend, but it is not "
            f"registered (JAX_PLATFORMS={platforms!r}, registered="
            f"{registered or ['?']}). Restart with JAX_PLATFORMS=gpu,cpu "
            "(GPU runs) or JAX_PLATFORMS=cpu (CPU-only) before importing jax."
        ) from exc
    with jax.default_device(cpu[0]):
        yield cpu[0]
