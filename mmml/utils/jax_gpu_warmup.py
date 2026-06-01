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
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_xla_gpu_warmed = False
_ptxas_path_configured = False


def _site_package_roots() -> list[Path]:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots: list[Path] = [Path(sys.prefix) / f"lib/python{ver}/site-packages"]
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            unique.append(resolved)
    return unique


@lru_cache(maxsize=1)
def find_bundled_ptxas_dir() -> Path | None:
    """Return the directory containing wheel-shipped ``ptxas``, if any."""
    rel_bins = (
        "nvidia/cu13/bin",
        "nvidia/cu12/bin",
        "nvidia/cuda_nvcc/bin",
    )
    for root in _site_package_roots():
        for rel in rel_bins:
            candidate = (root / rel).resolve()
            if (candidate / "ptxas").is_file():
                return candidate
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = (os.environ.get(env_name) or "").strip()
        if not cuda_home:
            continue
        candidate = (Path(cuda_home) / "bin").resolve()
        if (candidate / "ptxas").is_file():
            return candidate
    return None


def ensure_jax_cuda_toolchain(*, required: bool = False) -> bool:
    """Put bundled ``ptxas`` on ``PATH`` for XLA GPU compilation.

    JAX CUDA wheels ship ``ptxas`` under ``site-packages/nvidia/cu13/bin`` (or
    ``cu12``). XLA fails with ``INTERNAL: Failed to launch ptxas`` when that
    directory is not on ``PATH`` — common after ``uv sync --extra gpu-cuda13``.

    Returns True when ``ptxas`` is available on ``PATH`` after this call.
    """
    global _ptxas_path_configured
    if shutil.which("ptxas"):
        _ptxas_path_configured = True
        return True

    ptxas_dir = find_bundled_ptxas_dir()
    if ptxas_dir is None:
        if required:
            raise RuntimeError(
                "ptxas not found for JAX GPU compilation. Install CUDA extras, e.g. "
                "`uv sync --extra gpu-cuda13` or `pip install nvidia-cuda-nvcc`, "
                "or put a CUDA toolkit bin directory containing ptxas on PATH."
            )
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


def ensure_xla_gpu_warmed(*, force: bool = False) -> bool:
    """Run a tiny JITted reduction on GPU and block until complete.

    Returns True if a GPU warmup ran, False if JAX is missing or no GPU backend.
    Idempotent unless ``force=True``.
    """
    global _xla_gpu_warmed
    if _xla_gpu_warmed and not force:
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
    # Two executions: first may compile; second satisfies delay-kernel calibration.
    for _ in range(2):
        out = _warmup_kernel(x)
        jax.block_until_ready(out)

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
    """
    ensure_jax_cuda_toolchain(required=True)
    ensure_xla_gpu_warmed(force=True)
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
    # Two untimed runs: first compiles/autotunes; second calibrates XLA's delay kernel.
    for _ in range(2):
        result = spherical_cutoff_calculator(**kwargs)
        block_jax_values(getattr(result, "energy", None), getattr(result, "forces", None))


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
