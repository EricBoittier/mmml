"""Device-selection precedence in ``examples/m/_env.sh``.

The regression these guard: a stale ``export JAX_PLATFORMS=cpu`` in a login
profile used to outrank ``MMML_EXAMPLE_DEVICE=gpu``, so the documented device
knob was the one setting that could not change the device and a GPU run silently
executed on CPU.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SH = REPO_ROOT / "examples" / "m" / "_env.sh"

_REPORT = (
    'printf "%s|%s|%s|%s|%s" '
    '"$JAX_PLATFORMS" "$MMML_MLPOT_DEVICE" "$MMML_JAX_WARMUP_DEVICE" '
    '"$MMML_EXAMPLE_DEVICE_EXPLICIT" "$MMML_EXAMPLE_DEVICE_FORCED"'
)


def _resolve(**env: str) -> dict[str, str]:
    """Source ``_env.sh`` in a clean shell and report the resolved device vars."""
    clean = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        # The banner is not exercised here; skip its `uv run` backend probe.
        "MMML_EXAMPLE_SKIP_DEVICE_PROBE": "1",
    }
    clean.update(env)
    proc = subprocess.run(
        ["bash", "-c", f"source {ENV_SH} >/dev/null && {_REPORT}"],
        cwd=REPO_ROOT,
        env=clean,
        capture_output=True,
        text=True,
        check=True,
    )
    platforms, mlpot, warmup, explicit, forced = proc.stdout.split("|")
    return {
        "JAX_PLATFORMS": platforms,
        "MMML_MLPOT_DEVICE": mlpot,
        "MMML_JAX_WARMUP_DEVICE": warmup,
        "explicit": explicit,
        "forced": forced,
    }


def test_defaults_to_cpu() -> None:
    got = _resolve()
    assert got["JAX_PLATFORMS"] == "cpu"
    assert got["MMML_MLPOT_DEVICE"] == "cpu"
    assert got["explicit"] == "0"
    assert got["forced"] == ""


def test_explicit_gpu_on_clean_env() -> None:
    got = _resolve(MMML_EXAMPLE_DEVICE="gpu")
    assert got["JAX_PLATFORMS"] == "cuda"
    assert got["MMML_MLPOT_DEVICE"] == "gpu"
    assert got["MMML_JAX_WARMUP_DEVICE"] == "gpu"
    assert got["forced"] == ""


def test_explicit_gpu_overrides_stale_cpu_profile() -> None:
    """The reported bug: a profile exporting cpu must not win over an explicit gpu."""
    got = _resolve(
        MMML_EXAMPLE_DEVICE="gpu",
        JAX_PLATFORMS="cpu",
        MMML_MLPOT_DEVICE="cpu",
        MMML_JAX_WARMUP_DEVICE="cpu",
    )
    assert got["JAX_PLATFORMS"] == "cuda"
    assert got["MMML_MLPOT_DEVICE"] == "gpu"
    assert got["MMML_JAX_WARMUP_DEVICE"] == "gpu"
    # Every discarded value is named so the banner can point at the profile.
    for var in ("JAX_PLATFORMS=cpu", "MMML_MLPOT_DEVICE=cpu", "MMML_JAX_WARMUP_DEVICE=cpu"):
        assert var in got["forced"]


def test_explicit_cpu_overrides_stale_gpu_profile() -> None:
    got = _resolve(MMML_EXAMPLE_DEVICE="cpu", JAX_PLATFORMS="cuda")
    assert got["JAX_PLATFORMS"] == "cpu"
    assert got["MMML_MLPOT_DEVICE"] == "cpu"
    assert "JAX_PLATFORMS=cuda" in got["forced"]


def test_agreeing_platforms_value_is_kept_verbatim() -> None:
    """``cuda,cpu`` implies gpu and must survive: the cpu token is the fallback."""
    got = _resolve(MMML_EXAMPLE_DEVICE="gpu", JAX_PLATFORMS="cuda,cpu")
    assert got["JAX_PLATFORMS"] == "cuda,cpu"
    assert got["MMML_MLPOT_DEVICE"] == "gpu"
    assert got["forced"] == ""


@pytest.mark.parametrize(
    ("env", "var", "expected"),
    [
        ({"JAX_PLATFORMS": "cuda"}, "JAX_PLATFORMS", "cuda"),
        ({"MMML_MLPOT_DEVICE": "gpu"}, "MMML_MLPOT_DEVICE", "gpu"),
    ],
)
def test_per_variable_override_without_example_device(
    env: dict[str, str], var: str, expected: str
) -> None:
    """Setting only the low-level knobs stays a supported override."""
    got = _resolve(**env)
    assert got[var] == expected
    assert got["explicit"] == "0"
    assert got["forced"] == ""


def test_nested_step_of_gpu_run_keeps_gpu() -> None:
    """`run_all.sh` exports these into `bash 0X_*.sh`; the child must agree."""
    got = _resolve(
        MMML_EXAMPLE_DEVICE="gpu",
        MMML_EXAMPLE_DEVICE_EXPLICIT="1",
        JAX_PLATFORMS="cuda",
        MMML_MLPOT_DEVICE="gpu",
    )
    assert got["JAX_PLATFORMS"] == "cuda"
    assert got["MMML_MLPOT_DEVICE"] == "gpu"
    assert got["forced"] == ""


def test_nested_step_does_not_clobber_per_variable_override() -> None:
    """A child sees our exported MMML_EXAMPLE_DEVICE=cpu but explicit=0 wins.

    Without the inherited-marker check, the child would read the exported
    default as a user request and discard the parent's JAX_PLATFORMS=cuda.
    """
    got = _resolve(
        MMML_EXAMPLE_DEVICE="cpu",
        MMML_EXAMPLE_DEVICE_EXPLICIT="0",
        JAX_PLATFORMS="cuda",
    )
    assert got["JAX_PLATFORMS"] == "cuda"
    assert got["forced"] == ""


def test_invalid_device_is_rejected() -> None:
    proc = subprocess.run(
        ["bash", "-c", f"source {ENV_SH}"],
        cwd=REPO_ROOT,
        env={
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "MMML_EXAMPLE_DEVICE": "tpu",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "must be 'cpu' or 'gpu'" in proc.stderr
