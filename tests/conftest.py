"""Shared pytest hooks and environment probes for MMML test selection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent

# Paths (relative to tests/) that require a live PyCHARMM build.
_PYCHARMM_PATH_PREFIXES = (
    "functionality/pycharmmETC/",
    "functionality/mlpot/test_mlpot_energy_matches_ase.py",
    "functionality/mlpot/test_mlpot_dynamics_smoke.py",
    "functionality/mlpot/test_comp_velocities_integration.py",
    "functionality/mmml_tests/test_mmml_calc.py",
    "functionality/mmml_tests/test_ase_jaxmd_pbc_consistency.py",
    "misc/test_charmm.py",
    "integration/test_dcm_charmm_regression.py",
)

# Subset that loads ML checkpoints / benefits from JAX on GPU.
_GPU_PATH_PREFIXES = (
    "functionality/mlpot/test_mlpot_energy_matches_ase.py",
    "functionality/mlpot/test_mlpot_dynamics_smoke.py",
    "functionality/mmml_tests/test_mmml_calc.py",
    "functionality/mmml_tests/test_ase_jaxmd_pbc_consistency.py",
    "functionality/pycharmmETC/test_physnetjax_calc.py",
    "functionality/pycharmmETC/test_spookynetjax_calc.py",
    "misc/test_orbax_json_checkpoint.py",
)

_MLPOT_PATH_PREFIXES = (
    "functionality/mlpot/test_mlpot_energy_matches_ase.py",
    "functionality/mlpot/test_mlpot_dynamics_smoke.py",
    "functionality/mlpot/test_comp_velocities_integration.py",
)


def _rel_test_path(item: pytest.Item) -> str:
    path = Path(str(item.fspath))
    try:
        return path.relative_to(_TESTS_ROOT).as_posix()
    except ValueError:
        return path.name


def _matches_any(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == p or rel.startswith(p) for p in prefixes)


def can_import_pycharmm() -> bool:
    try:
        __import__("pycharmm")
        return True
    except Exception:
        return False


def charmm_env_configured() -> bool:
    home = os.environ.get("CHARMM_HOME")
    lib = os.environ.get("CHARMM_LIB_DIR")
    if not home or not lib:
        return False
    return os.path.exists(home) and os.path.exists(lib)


def jax_gpu_available() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        rel = _rel_test_path(item)
        if _matches_any(rel, _PYCHARMM_PATH_PREFIXES):
            item.add_marker(pytest.mark.pycharmm)
        if _matches_any(rel, _GPU_PATH_PREFIXES):
            item.add_marker(pytest.mark.gpu)
        if _matches_any(rel, _MLPOT_PATH_PREFIXES):
            item.add_marker(pytest.mark.mlpot)
