"""Unit tests for CuPy CUDA_PATH repair (GPU Vesin NVRTC)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.nl_gpu import (
    cuda_path_looks_broken,
    ensure_cupy_cuda_path,
)


def test_cuda_path_looks_broken_for_missing_and_ancient(tmp_path: Path, monkeypatch):
    assert cuda_path_looks_broken(None)
    assert cuda_path_looks_broken(str(tmp_path / "missing"))

    ancient = tmp_path / "cuda-9.0"
    ancient.mkdir()
    (ancient / "include").mkdir()
    (ancient / "include" / "cuda_fp16.hpp").write_text(
        "#if __cplusplus >= 201103L\n#include <utility>\n#endif\n"
        "#if defined(__CUDACC_RTC__)\n#define X 1\n#endif\n",
        encoding="utf-8",
    )
    assert cuda_path_looks_broken(str(ancient))


def test_cuda_path_ok_for_modern_wheel_style_header(tmp_path: Path):
    modern = tmp_path / "cuda_runtime"
    (modern / "include").mkdir(parents=True)
    (modern / "include" / "cuda_fp16.hpp").write_text(
        "/* modern wheel header: no host <utility> for NVRTC */\n"
        "#if defined(__CUDACC_RTC__)\n#define X 1\n#endif\n",
        encoding="utf-8",
    )
    assert not cuda_path_looks_broken(str(modern))


def test_ensure_cupy_cuda_path_overrides_broken_system_path(tmp_path: Path, monkeypatch):
    import mmml.interfaces.pycharmmInterface.nl_gpu as nl_gpu

    monkeypatch.setattr(nl_gpu, "_CUDA_PATH_ENSURED", False)
    ancient = tmp_path / "cuda-9.0"
    (ancient / "include").mkdir(parents=True)
    (ancient / "include" / "cuda_fp16.hpp").write_text(
        "#include <utility>\n#if defined(__CUDACC_RTC__)\n#endif\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "wheel_runtime"
    (wheel / "include").mkdir(parents=True)
    (wheel / "include" / "cuda_fp16.hpp").write_text(
        "/* ok */\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_PATH", str(ancient))
    monkeypatch.setattr(nl_gpu, "_nvidia_wheel_cuda_runtime_root", lambda: str(wheel))
    monkeypatch.setattr(nl_gpu, "_patch_cupy_wheel_includes", lambda _inc: None)

    out = ensure_cupy_cuda_path(force=True, quiet=True)
    assert out == str(wheel)
    assert Path(out).name == "wheel_runtime"
    assert not cuda_path_looks_broken(out)


def test_cupy_runtime_ok_after_ensure_when_gpu_present():
    """Integration: with pip CUDA wheels, probe should pass on a CUDA node."""
    import os

    cupy = pytest.importorskip("cupy")
    jax = pytest.importorskip("jax")
    try:
        if not jax.devices("gpu"):
            pytest.skip("no JAX GPU")
    except Exception:
        pytest.skip("no JAX GPU")

    import mmml.interfaces.pycharmmInterface.nl_gpu as nl_gpu

    if nl_gpu._nvidia_wheel_cuda_runtime_root() is None:
        pytest.skip("nvidia-cuda-runtime wheel headers not installed")

    nl_gpu._CUDA_PATH_ENSURED = False
    nl_gpu._CUPY_RUNTIME_OK = None
    # Simulate the broken cluster default (/usr/local/cuda → cuda-9.0).
    if Path("/usr/local/cuda").exists():
        os.environ["CUDA_PATH"] = "/usr/local/cuda"
    assert nl_gpu.cupy_runtime_ok(force=True), (
        "CuPy JIT still failing after CUDA_PATH repair; "
        f"CUDA_PATH={os.environ.get('CUDA_PATH')} "
        f"wheel={nl_gpu._nvidia_wheel_cuda_runtime_root()}"
    )
    x = cupy.arange(4, dtype=cupy.float32)
    assert float((x + 1).sum()) == 10.0
