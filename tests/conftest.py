"""Shared pytest hooks and environment probes for MMML test selection."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.functionality.pycharmmETC._paths import PYCHARMMETC_DIR

_TESTS_ROOT = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    """Avoid blocking ``dlopen(libcharmm)`` while collecting tests.

    MPI-linked CHARMM can hang for minutes (or forever) when pytest imports
    ``mmml_calculator`` / ``hybrid_mlpot`` in a plain serial shell.  Live
    PyCHARMM jobs use ``mmml-charmm-mpirun.sh`` or import CHARMM inside the
    test body after bootstrap.  Override with ``MMML_WARMUP_MLPOT_JAX_ONLY=0``.
    """
    if os.environ.get("MMML_WARMUP_MLPOT_JAX_ONLY", "").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    os.environ.setdefault("MMML_WARMUP_MLPOT_JAX_ONLY", "1")

# Committed inputs copied into each isolated PyCHARMM workdir when present.
_PYCHARMM_SEED_PDBS = (
    "initial.pdb",
    "init-packmol.pdb",
    "aco.pdb",
    "init-tip3.pdb",
    "tip3.pdb",
)
_PYCHARMM_SEED_PSFS = (
    "aco-1.psf",
)

# Paths (relative to tests/) that require a live PyCHARMM build.
_PYCHARMM_PATH_PREFIXES = (
    "functionality/pycharmmETC/",
    "functionality/charmm/",
    "charmm_mpi/test_mpi_live",
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
        from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_available

        return charmm_lib_available()
    except Exception:
        return False


def charmm_env_configured() -> bool:
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_available
        from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths

        home, lib = resolve_charmm_paths()
        if not home or not lib:
            return False
        return os.path.exists(home) and os.path.exists(lib) and charmm_lib_available()
    except Exception:
        return False


def jax_gpu_available() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def bonded_block_hangs_under_mpi_mpirun() -> bool:
    """Selective COEFF BLOCK scripts stall on MPI-linked libcharmm under mpirun.

    Affects ``apply_bonded_mm_only_block`` (ELEC/VDW off) and
    ``setup_nonbonded_only_charmm`` (bonded terms off). Full ``reset_block`` /
    ``apply_charmm_mm_block`` are fine.
    """
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        return bool(charmm_lib_links_mpi() and _under_mpirun())
    except Exception:
        return False


@pytest.fixture
def pycharmm_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary cwd with seed PDBs; PyCHARMM outputs stay out of the git tree."""
    for sub in ("pdb", "psf", "packmol", "res", "dcd", "xyz"):
        (tmp_path / sub).mkdir()
    for name in _PYCHARMM_SEED_PDBS:
        src = PYCHARMMETC_DIR / "pdb" / name
        if src.is_file():
            shutil.copy2(src, tmp_path / "pdb" / name)
    for name in _PYCHARMM_SEED_PSFS:
        src = PYCHARMMETC_DIR / "psf" / name
        if src.is_file():
            shutil.copy2(src, tmp_path / "psf" / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        rel = _rel_test_path(item)
        if rel.startswith("charmm_mpi/"):
            item.add_marker(pytest.mark.charmm_mpi)
        if _matches_any(rel, _PYCHARMM_PATH_PREFIXES):
            item.add_marker(pytest.mark.pycharmm)
        if _matches_any(rel, _GPU_PATH_PREFIXES):
            item.add_marker(pytest.mark.gpu)
        if _matches_any(rel, _MLPOT_PATH_PREFIXES):
            item.add_marker(pytest.mark.mlpot)

    if any(item.get_closest_marker("pycharmm") is not None for item in items):
        try:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import _under_mpirun
            from mmml.interfaces.pycharmmInterface.import_pycharmm import (
                ensure_pycharmm_loaded,
            )

            if _under_mpirun():
                ensure_pycharmm_loaded()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _jax_enable_x64_for_pycharmm_tests(request: pytest.FixtureRequest) -> None:
    """CHARMM cross-checks need float64 for rtol=1e-4 bonded/improper agreement."""
    if request.node.get_closest_marker("pycharmm") is not None:
        import jax

        jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _charmm_default_levels_for_pycharmm_tests(request: pytest.FixtureRequest) -> None:
    """Live PyCHARMM tests load CHARMM outside ``import_pycharmm`` when ``MMML_WARMUP_MLPOT_JAX_ONLY=1``.

    Unit-test collection skips ``apply_charmm_verbosity(bomlev=-2)``; ensure relaxed
    BOMLEV before the first ``read`` / ``nbonds`` in each live test body.
    """
    if request.node.get_closest_marker("pycharmm") is None:
        return
    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import (
            ensure_pycharmm_loaded,
        )

        ensure_pycharmm_loaded()
        from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

        apply_charmm_verbosity(prnlev=5, warnlev=5, bomlev=-2)
    except Exception:
        pass
