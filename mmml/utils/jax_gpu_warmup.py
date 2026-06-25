"""Warm up JAX/XLA on GPU before timed kernels run.

XLA's CUDA timer uses a short "delay" kernel to calibrate GPU timing. Without a
prior completed GPU execution, that kernel can time out and log::

    Delay kernel timed out: measured time has sub-optimal accuracy.
    There may be a missing warmup execution

Call :func:`ensure_xla_gpu_warmed` once per process before the first large
``jax.jit`` evaluation (e.g. hybrid MMML calculator).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_xla_gpu_warmed = False
_ptxas_path_configured = False
_jax_cuda_runtime_libs_configured = False


@dataclass
class JAXCompileTiming:
    """One JAX warmup pass (first pass is compile+run; later passes are mostly run)."""

    label: str
    pass_index: int
    wall_seconds: float


@dataclass
class JAXCompileTimerSession:
    """Collected warmup timings for an optional end-of-phase summary."""

    entries: list[JAXCompileTiming] = field(default_factory=list)

    def record(self, label: str, pass_index: int, wall_seconds: float) -> None:
        self.entries.append(
            JAXCompileTiming(label=label, pass_index=pass_index, wall_seconds=wall_seconds)
        )

    def summary_lines(self) -> list[str]:
        if not self.entries:
            return []
        by_label: dict[str, list[JAXCompileTiming]] = {}
        for entry in self.entries:
            by_label.setdefault(entry.label, []).append(entry)
        lines: list[str] = []
        total_compile = 0.0
        total_run = 0.0
        for label, passes in sorted(by_label.items()):
            passes = sorted(passes, key=lambda e: e.pass_index)
            if len(passes) >= 2:
                run_s = passes[-1].wall_seconds
                compile_s = max(0.0, passes[0].wall_seconds - run_s)
                total_compile += compile_s
                total_run += run_s
                lines.append(
                    f"  {label}: compile≈{compile_s:.2f}s, run≈{run_s:.2f}s "
                    f"(pass1={passes[0].wall_seconds:.2f}s)"
                )
            else:
                total_run += passes[0].wall_seconds
                lines.append(
                    f"  {label}: {passes[0].wall_seconds:.2f}s (single pass)"
                )
        header = (
            f"mmml: JAX compile timers — estimated compile={total_compile:.2f}s, "
            f"run={total_run:.2f}s"
        )
        return [header, *lines]


_COMPILE_TIMER_SESSION = JAXCompileTimerSession()


def jax_compile_timers_enabled() -> bool:
    """Enable with ``MMML_JAX_COMPILE_TIMERS=1`` or ``MMML_MLPOT_PROFILE=1``."""
    for key in ("MMML_JAX_COMPILE_TIMERS", "MMML_MLPOT_PROFILE"):
        if (os.environ.get(key) or "").strip().lower() in ("1", "yes", "true"):
            return True
    return False


def reset_jax_compile_timers() -> None:
    global _COMPILE_TIMER_SESSION
    _COMPILE_TIMER_SESSION = JAXCompileTimerSession()


def get_jax_compile_timer_session() -> JAXCompileTimerSession:
    return _COMPILE_TIMER_SESSION


def maybe_log_jax_compile_timers(*, quiet: bool = False) -> None:
    if quiet or not jax_compile_timers_enabled():
        return
    lines = _COMPILE_TIMER_SESSION.summary_lines()
    if not lines:
        return
    for line in lines:
        print(line, flush=True)


def _log_jax_compile_pass(label: str, pass_index: int, wall_seconds: float) -> None:
    _COMPILE_TIMER_SESSION.record(label, pass_index, wall_seconds)
    phase = "compile+run" if pass_index == 0 else "run"
    print(
        f"mmml: JAX compile timer [{label}] pass {pass_index + 1} ({phase}): "
        f"{wall_seconds:.2f}s",
        flush=True,
    )


def run_jax_warmup_passes(
    label: str,
    n_passes: int,
    run_once: Callable[[], Any],
    *,
    block: Callable[[Any], None] | None = None,
) -> None:
    """Run ``n_passes`` warmup executions; log wall time per pass when timers are on."""
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        jax_compile_threads_context,
    )

    block_fn = block or (lambda result: block_jax_values(result))
    with jax_compile_threads_context():
        if not jax_compile_timers_enabled():
            for _ in range(n_passes):
                block_fn(run_once())
            return
        for pass_index in range(n_passes):
            t0 = time.perf_counter()
            result = run_once()
            block_fn(result)
            wall = time.perf_counter() - t0
            _log_jax_compile_pass(label, pass_index, wall)
    if n_passes >= 2:
        entries = [
            e
            for e in _COMPILE_TIMER_SESSION.entries
            if e.label == label and e.pass_index < n_passes
        ]
        if len(entries) >= 2:
            entries = sorted(entries, key=lambda e: e.pass_index)
            run_s = entries[-1].wall_seconds
            compile_s = max(0.0, entries[0].wall_seconds - run_s)
            print(
                f"mmml: JAX compile timer [{label}] summary: "
                f"compile≈{compile_s:.2f}s, run≈{run_s:.2f}s",
                flush=True,
            )


def _site_package_roots() -> list[Path]:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots: list[Path] = [Path(sys.prefix) / f"lib/python{ver}/site-packages"]
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass
    try:
        import sysconfig

        for key in ("purelib", "platlib"):
            lib = sysconfig.get_path(key)
            if lib:
                roots.append(Path(lib))
    except Exception:
        pass
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _ptxas_candidates_in_tree(root: Path) -> Path | None:
    rel_bins = (
        "nvidia/cu13/bin",
        "nvidia/cu12/bin",
        "nvidia/cuda_nvcc/bin",
    )
    for rel in rel_bins:
        candidate = (root / rel).resolve()
        if (candidate / "ptxas").is_file():
            return candidate
    return None


def _ptxas_from_nvidia_namespace() -> Path | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        found = _ptxas_candidates_in_tree(Path(root))
        if found is not None:
            return found
    return None


def _ptxas_from_importlib_metadata() -> Path | None:
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:
        return None
    for dist_name in (
        "nvidia-cuda-nvcc",
        "nvidia-cuda-nvcc-cu13",
        "nvidia-cuda-nvcc-cu12",
    ):
        try:
            dist = distribution(dist_name)
        except PackageNotFoundError:
            continue
        for path in (dist.locate_file("nvidia"), dist.locate_file("")):
            found = _ptxas_candidates_in_tree(Path(path))
            if found is not None:
                return found
    return None


def _ptxas_from_system_cuda() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        cuda_home = (os.environ.get(env_name) or "").strip()
        if cuda_home:
            candidates.append(Path(cuda_home) / "bin")
    candidates.extend(
        Path(p)
        for p in (
            "/usr/local/cuda/bin",
            "/opt/cuda/bin",
            "/usr/lib/cuda/bin",
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "ptxas").is_file():
            return resolved
    return None


@lru_cache(maxsize=1)
def find_bundled_nvidia_lib_dirs() -> list[Path]:
    """Wheel-shipped NVIDIA runtime libs (cuDNN, cuBLAS, cuSPARSE, …) for JAX CUDA plugins."""
    rel_libs = (
        "nvidia/cudnn/lib",
        "nvidia/cu13/lib",
        "nvidia/cu12/lib",
        "nvidia/cusparse/lib",
        "nvidia/cublas/lib",
        "nvidia/cusolver/lib",
        "nvidia/curand/lib",
        "nvidia/cufft/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_cupti/lib",
        "nvidia/nccl/lib",
    )
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in _site_package_roots():
        for rel in rel_libs:
            candidate = (root / rel).resolve()
            if candidate.is_dir() and candidate not in seen:
                seen.add(candidate)
                dirs.append(candidate)
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                try:
                    resolved = lib_dir.resolve()
                except OSError:
                    continue
                if resolved in seen or not resolved.is_dir():
                    continue
                if any(resolved.glob("*.so*")) or any(resolved.glob("*.dll")):
                    seen.add(resolved)
                    dirs.append(resolved)
    return dirs


def _jax_is_imported() -> bool:
    return "jax" in sys.modules


def _installed_jax_cuda_plugins() -> list[str]:
    try:
        from importlib.metadata import distributions
    except ImportError:
        return []
    names: list[str] = []
    for dist in distributions():
        name = (dist.metadata.get("Name") or "").strip()
        lower = name.lower()
        if lower.startswith("jax-cuda") and "plugin" in lower:
            names.append(name)
    return sorted(names)


def diagnose_jax_cuda_toolchain() -> dict[str, str | bool | None | list[str]]:
    """Notebook-friendly snapshot of JAX GPU toolchain discovery."""
    ptxas_dir = find_bundled_ptxas_dir()
    bundled = [str(p) for p in find_bundled_nvidia_lib_dirs()]
    diag: dict[str, str | bool | None | list[str]] = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "jax_imported": _jax_is_imported(),
        "jax_cuda_plugins": _installed_jax_cuda_plugins(),
        "bundled_nvidia_lib_dirs": bundled,
        "ptxas_on_path": shutil.which("ptxas"),
        "ptxas_dir": str(ptxas_dir) if ptxas_dir is not None else None,
        "cuda_home": (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "").strip()
        or None,
    }
    if _jax_is_imported():
        try:
            import jax

            diag["jax_devices"] = [str(d) for d in jax.devices()]
            diag["jax_default_backend"] = str(jax.default_backend())
        except Exception as exc:
            diag["jax_devices"] = [f"(error: {exc})"]
    return diag


@lru_cache(maxsize=1)
def find_bundled_ptxas_dir() -> Path | None:
    """Return the directory containing wheel-shipped or system ``ptxas``, if any."""
    for root in _site_package_roots():
        found = _ptxas_candidates_in_tree(root)
        if found is not None:
            return found
    for finder in (_ptxas_from_nvidia_namespace, _ptxas_from_importlib_metadata, _ptxas_from_system_cuda):
        found = finder()
        if found is not None:
            return found
    return None


def _ptxas_missing_error() -> RuntimeError:
    diag = diagnose_jax_cuda_toolchain()
    return RuntimeError(
        "ptxas not found for JAX GPU JIT compilation. "
        "JAX can list CUDA devices without ptxas, but the first jax.jit compile needs it.\n"
        f"  python: {diag['python']}\n"
        f"  CUDA_HOME/CUDA_PATH: {diag['cuda_home'] or '(unset)'}\n"
        "Fix (pick one):\n"
        "  1. In this kernel's env: uv sync --extra gpu  (installs nvidia-cuda-nvcc wheel)\n"
        "  2. module load cuda && export CUDA_HOME=$CUDA_HOME  (system toolkit bin/ptxas)\n"
        "  3. Notebook first cell: from mmml.utils.jax_gpu_warmup import prepare_jax_gpu_notebook; "
        "prepare_jax_gpu_notebook()\n"
        "Verify: import shutil; print(shutil.which('ptxas'))"
    )


def _jax_gpu_env_too_late_error() -> RuntimeError:
    diag = diagnose_jax_cuda_toolchain()
    return RuntimeError(
        "JAX was already imported before GPU runtime libraries were configured. "
        "cuSPARSE/cuBLAS from pip wheels must be on LD_LIBRARY_PATH *before* "
        "the first `import jax`, or the CUDA plugin falls back to CPU.\n"
        f"  python: {diag['python']}\n"
        f"  jax_cuda_plugins: {diag.get('jax_cuda_plugins')}\n"
        f"  bundled_nvidia_lib_dirs: {len(diag.get('bundled_nvidia_lib_dirs') or [])} dirs\n"
        "Fix:\n"
        "  1. Restart the Jupyter kernel.\n"
        "  2. First cell only:\n"
        "       from mmml.interfaces.pycharmmInterface.mlpot.cli_common import prepare_jax_gpu_notebook\n"
        "       prepare_jax_gpu_notebook()\n"
        "  3. Then import jax / setup_calculator in later cells.\n"
        "  4. In this venv: cd ~/mmml && uv sync --extra gpu  (CUDA 13; RTX 5090)\n"
        "     Avoid mixing jax-cuda12-plugin with the default gpu extra."
    )


def prepare_jax_gpu_notebook(*, required: bool = True) -> bool:
    """Prep PATH/LD_LIBRARY_PATH for JAX GPU JIT in Jupyter (call once per kernel).

    Must run in the **first notebook cell**, before ``import jax`` or any mmml import
    that pulls JAX in transitively.
    """
    if _jax_is_imported():
        try:
            import jax

            if jax.default_backend() == "cpu" and _installed_jax_cuda_plugins():
                if required:
                    raise _jax_gpu_env_too_late_error()
                return False
        except Exception:
            pass

    bundled = ensure_jax_cuda_runtime_libs(quiet=True)
    if not bundled and required and _installed_jax_cuda_plugins():
        raise RuntimeError(
            "JAX CUDA plugin(s) are installed but no pip NVIDIA runtime libs were found "
            f"under {sys.prefix}. Run: cd ~/mmml && uv sync --extra gpu"
        )
    ok = ensure_jax_cuda_toolchain(required=False)
    if ok:
        return True
    if required:
        raise _ptxas_missing_error()
    return False


def ensure_jax_cuda_runtime_libs(*, quiet: bool = False) -> list[str]:
    """Prefer pip-shipped cuDNN/cuBLAS over older system modules on ``LD_LIBRARY_PATH``.

    JAX CUDA 13 rejects cuDNN < 9.10.1 on multi-GPU nodes when cuBLAS >= 12.9 is
    visible. ``uv sync --extra gpu`` installs ``nvidia-cudnn-cu13`` under
    ``site-packages``, but cluster ``module load cudnn/9.4`` often wins unless these
    directories are prepended *after* MPI/CHARMM library setup as well.
    """
    global _jax_cuda_runtime_libs_configured
    bundled = [str(path) for path in find_bundled_nvidia_lib_dirs()]
    if not bundled:
        return []

    bundled_set = set(bundled)
    cur_parts = [
        part
        for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part and part not in bundled_set
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(bundled + cur_parts)
    _jax_cuda_runtime_libs_configured = True
    if not quiet and not os.environ.get("MMML_QUIET", "").strip().lower() in (
        "1",
        "yes",
        "true",
    ):
        logger.debug("Prepended JAX CUDA runtime libs to LD_LIBRARY_PATH: %s", bundled)
    return bundled


def ensure_jax_cuda_toolchain(*, required: bool = False) -> bool:
    """Put bundled CUDA runtime libs and ``ptxas`` on ``PATH`` for XLA GPU use.

    JAX CUDA wheels ship ``ptxas`` under ``site-packages/nvidia/cu13/bin`` (or
    ``cu12``). XLA fails with ``INTERNAL: Failed to launch ptxas`` when that
    directory is not on ``PATH`` — common after ``uv sync --extra gpu``.

    Returns True when ``ptxas`` is available on ``PATH`` after this call.
    """
    global _ptxas_path_configured
    ensure_jax_cuda_runtime_libs(quiet=True)
    if shutil.which("ptxas"):
        _ptxas_path_configured = True
        return True

    ptxas_dir = find_bundled_ptxas_dir()
    if ptxas_dir is None:
        if required:
            raise _ptxas_missing_error()
        return False

    path = str(ptxas_dir)
    cur = os.environ.get("PATH", "")
    if path not in cur.split(os.pathsep):
        os.environ["PATH"] = f"{path}{os.pathsep}{cur}"
        logger.debug("Prepended JAX CUDA toolchain to PATH: %s", path)

    _ptxas_path_configured = True
    if not shutil.which("ptxas") and required:
        raise RuntimeError(f"ptxas exists at {ptxas_dir / 'ptxas'} but is not executable")
    return shutil.which("ptxas") is not None


def sync_jax_gpu_before_charmm(*, phase: str = "before CHARMM") -> None:
    """Drain JAX GPU work and release the CUDA context before Fortran MPI/CHARMM ENER.

    JAX XLA compile (especially with raised OMP thread counts) can leave OpenMPI's
    registered-memory pool in a bad state; the next CHARMM ``gete`` may segfault in
    ``send_coord_to_recip`` / ``PMPI_Free_mem`` even after ``domdec off``.
    """
    del phase
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    try:
        if jax.default_backend() != "gpu":
            return
    except Exception:
        return
    try:
        for device in jax.local_devices():
            with jax.default_device(device):
                jax.block_until_ready(jnp.sum(jnp.ones((1,), dtype=jnp.float32)))
    except Exception:
        pass


def apply_xla_cuda_timer_log_filter() -> None:
    """Suppress XLA ``cuda_timer.cc`` delay-kernel timeout noise (harmless autotuner warnings).

  Set env ``MMML_SUPPRESS_XLA_CUDA_TIMER=1`` (default) to raise ``TF_CPP_MIN_LOG_LEVEL``
  to 3 when it is unset. Set ``MMML_SUPPRESS_XLA_CUDA_TIMER=0`` to leave logging unchanged.
    """
    if os.environ.get("MMML_SUPPRESS_XLA_CUDA_TIMER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _jax_warmup_backend() -> str:
    """``cpu`` or ``gpu`` for JAX compile warmup."""
    from mmml.interfaces.pycharmmInterface.jax_device_policy import jax_warmup_device_name

    return jax_warmup_device_name()


def ensure_xla_gpu_warmed(*, force: bool = False) -> bool:
    """Run a tiny JITted reduction on GPU and block until complete.

    Returns True if a GPU warmup ran, False if JAX is missing or no GPU backend.
    Idempotent unless ``force=True``.
    """
    global _xla_gpu_warmed
    if _xla_gpu_warmed and not force:
        return False

    if _jax_warmup_backend() == "cpu":
        _xla_gpu_warmed = True
        return False

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        _xla_gpu_warmed = True
        return False

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        _xla_gpu_warmed = True
        return False

    ensure_jax_cuda_toolchain(required=True)

    @jax.jit
    def _warmup_kernel(x: jnp.ndarray) -> jnp.ndarray:
        # Reduction + matmul: schedules real GPU work (not just a host alloc).
        y = jnp.sum(x * 1.001)
        m = jnp.ones((32, 32), dtype=x.dtype)
        return y + jnp.sum(m @ m)

    x = jnp.ones((256,), dtype=jnp.float32)

    def _run_kernel() -> Any:
        return _warmup_kernel(x)

    run_jax_warmup_passes(
        "xla_gpu_delay_kernel",
        2,
        _run_kernel,
        block=lambda out: jax.block_until_ready(out),
    )

    _xla_gpu_warmed = True
    logger.debug("XLA GPU delay-kernel warmup completed on %s", gpu_devices[0])
    return True


def block_jax_values(*values: Any) -> None:
    """Block until JAX array leaves are ready (no-op if JAX is unavailable)."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    for value in values:
        try:
            jax.block_until_ready(jnp.asarray(value))
        except Exception:
            pass


_hybrid_spherical_warmup_keys: set[tuple[Any, ...]] = set()


def reset_hybrid_spherical_warmup_cache() -> None:
    """Clear process-wide hybrid warmup dedup (tests only)."""
    global _hybrid_spherical_warmup_keys
    _hybrid_spherical_warmup_keys = set()


def _hybrid_warmup_key(
    *,
    positions: Any,
    atomic_numbers: Any,
    n_monomers: int,
    doML: bool,
    doMM: bool,
    doML_dimer: bool,
    debug: bool,
    mm_pair_idx: Any,
    mm_pair_mask: Any,
    box: Any,
) -> tuple[Any, ...]:
    """JIT-relevant warmup fingerprint (skip identical repeat compiles)."""
    pair_shape = getattr(mm_pair_idx, "shape", None)
    return (
        int(n_monomers),
        bool(doML),
        bool(doMM),
        bool(doML_dimer),
        bool(debug),
        getattr(positions, "shape", None),
        str(getattr(positions, "dtype", None)),
        getattr(atomic_numbers, "shape", None),
        str(getattr(atomic_numbers, "dtype", None)),
        pair_shape,
        getattr(mm_pair_mask, "shape", None),
        box is not None,
    )


def warmup_hybrid_spherical_cutoff(
    spherical_cutoff_calculator: Any,
    *,
    atomic_numbers: Any,
    positions: Any,
    n_monomers: int,
    cutoff_params: Any,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    debug: bool = False,
    mm_pair_idx: Any = None,
    mm_pair_mask: Any = None,
    box: Any = None,
) -> None:
    """Compile and run one hybrid MMML eval; block until GPU work completes.

    Call after PyCHARMM/MM setup (e.g. CGENFF drudes) and before timed JAX-MD compiles.
    Identical repeat calls in one process are skipped (MLpot used to warm twice).
    """
    kwargs = dict(
        positions=positions,
        atomic_numbers=atomic_numbers,
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        doML=doML,
        doMM=doMM,
        doML_dimer=doML_dimer,
        debug=debug,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        box=box,
    )
    key = _hybrid_warmup_key(
        positions=positions,
        atomic_numbers=atomic_numbers,
        n_monomers=n_monomers,
        doML=doML,
        doMM=doMM,
        doML_dimer=doML_dimer,
        debug=debug,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        box=box,
    )
    if key in _hybrid_spherical_warmup_keys:
        return

    backend = _jax_warmup_backend()
    if backend == "gpu":
        ensure_jax_cuda_toolchain(required=True)
        ensure_xla_gpu_warmed(force=False)
    try:
        import jax
    except ImportError:
        return

    device_ctx = (
        jax.default_device(jax.devices("cpu")[0])
        if backend == "cpu"
        else nullcontext()
    )

    def _block_spherical_result(result: Any) -> None:
        block_jax_values(getattr(result, "energy", None), getattr(result, "forces", None))

    # Two runs: first compiles/autotunes; second calibrates XLA's delay kernel.
    with device_ctx:
        run_jax_warmup_passes(
            "spherical_cutoff",
            2,
            lambda: spherical_cutoff_calculator(**kwargs),
            block=_block_spherical_result,
        )
    _hybrid_spherical_warmup_keys.add(key)


def warmup_ase_mmml_energy_forces(atoms: Any, *, include_forces: bool = True) -> None:
    """JIT-warm an ASE calculator attached to ``atoms`` (energy, optionally forces)."""
    ensure_xla_gpu_warmed(force=True)
    energy = atoms.get_potential_energy()
    block_jax_values(energy)
    if include_forces:
        forces = atoms.get_forces()
        block_jax_values(forces)


# Apply log filter on import so CLI entry points that import this module early
# can suppress cuda_timer noise even if JAX was not imported yet.
apply_xla_cuda_timer_log_filter()
ensure_jax_cuda_toolchain()
