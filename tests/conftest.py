"""Shared pytest hooks and environment probes for MMML test selection."""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"


def _sanitize_jax_platforms_env() -> None:
    """Drop GPU-class JAX backends that cannot initialize from ``JAX_PLATFORMS``.

    A stale ``JAX_PLATFORMS=rocm`` (or ``cuda`` / ``gpu``) inherited from the
    shell makes ``jax.default_backend()`` raise at import / collection time
    ("Unable to initialize backend 'rocm': ... not in the list of known
    backends") whenever that backend is not usable here — the plugin may be
    absent *or* pip-installed but non-functional (no ROCm/CUDA runtime). We
    can't just check whether the plugin dist is installed, so probe the backend
    for real in a subprocess (importing jax in-process would crash the
    collector on a bad backend). If it fails, keep only ``cpu`` / ``tpu`` so the
    suite runs on CPU; real GPU machines pass the probe and are left untouched.
    """
    raw = (os.environ.get("JAX_PLATFORMS") or "").strip()
    if not raw:
        return
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    # Only GPU-class backends can fail to initialize; cpu/tpu need no probe.
    if not any(t.lower() in ("gpu", "cuda", "rocm") for t in tokens):
        return

    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import jax; jax.default_backend()"],
            env={**os.environ, "JAX_PLATFORMS": raw},
            capture_output=True,
            timeout=180,
        )
        backend_ok = proc.returncode == 0
    except Exception:
        backend_ok = False
    if backend_ok:
        return

    safe = [t for t in tokens if t.lower() in ("cpu", "tpu")]
    if safe:
        os.environ["JAX_PLATFORMS"] = ",".join(safe)
    else:
        # Nothing usable was requested; let JAX auto-select an available backend.
        os.environ.pop("JAX_PLATFORMS", None)


_sanitize_jax_platforms_env()


def _block_pycharmm_imports_when_disabled() -> None:
    """Make ``import pycharmm`` fail outright under ``MMML_DISABLE_CHARMM=1``.

    ``make test-ci`` exists to reproduce the CI ``build`` job -- which has no
    libcharmm -- on a machine that does have one. Hiding the library from
    ``charmm_paths`` is not enough: several tests guard themselves with a bare
    ``__import__("pycharmm")``, and the ``pycharmm`` package finds and dlopens
    the library through its own search path. Those tests then run for real, and
    a native CHARMM ``STOP`` at interpreter teardown ends the session early
    *with exit status 0* -- the local run stops at 2% and still looks green.

    Installing a meta-path blocker makes the guard checks answer "no", so the
    same tests skip locally that skip in CI.
    """
    import sys

    if (os.environ.get("MMML_DISABLE_CHARMM") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return

    class _PycharmmBlocker:
        """A ``sys.meta_path`` finder that refuses the ``pycharmm`` package."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "pycharmm" or fullname.startswith("pycharmm."):
                raise ImportError(
                    f"{fullname} is blocked by MMML_DISABLE_CHARMM=1 "
                    "(see tests/conftest.py)"
                )
            return None

    sys.meta_path.insert(0, _PycharmmBlocker())
    for name in [n for n in sys.modules if n == "pycharmm" or n.startswith("pycharmm.")]:
        del sys.modules[name]


_block_pycharmm_imports_when_disabled()

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
    "functionality/mlpot/test_live_optimizers_dynamics.py",
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
    "functionality/mlpot/test_live_optimizers_dynamics.py",
    "functionality/mmml_tests/test_mmml_calc.py",
    "functionality/mmml_tests/test_ase_jaxmd_pbc_consistency.py",
    "functionality/pycharmmETC/test_physnetjax_calc.py",
    "functionality/pycharmmETC/test_spookynetjax_calc.py",
    "misc/test_orbax_json_checkpoint.py",
)

_MLPOT_PATH_PREFIXES = (
    "functionality/mlpot/test_mlpot_energy_matches_ase.py",
    "functionality/mlpot/test_mlpot_dynamics_smoke.py",
    "functionality/mlpot/test_live_optimizers_dynamics.py",
    "functionality/mlpot/test_comp_velocities_integration.py",
)

# Live CHARMM functionality tests mutate global CHARMM state in-process and are
# unsafe under MPI-linked libcharmm + mpirun smoke selection.
_CHARMM_SERIAL_PATH_PREFIXES: tuple[str, ...] = (
    "functionality/charmm/",
)


def charmm_rebuild_psf_unsafe_under_mpirun() -> bool:
    """True when a second in-process PSF/CGENFF read is unsafe (MPI-linked libcharmm)."""
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        return bool(_under_mpirun() and charmm_lib_links_mpi())
    except Exception:
        return False


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
            selective_bonded_block_unsafe_under_mpi,
        )

        return selective_bonded_block_unsafe_under_mpi()
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
        if _matches_any(rel, _CHARMM_SERIAL_PATH_PREFIXES):
            item.add_marker(pytest.mark.charmm_serial)

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


# Substrings that identify a failure caused purely by ``libcharmm`` being
# absent (unbuilt), rather than a genuine defect in the code under test.
_CHARMM_UNAVAILABLE_SIGNATURES = (
    "libcharmm.so: cannot open shared object",
    "libcharmm.dylib",
    "No module named 'pycharmm.",
    "'pycharmm' is not a package",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Skip (don't fail) tests that only break because ``libcharmm`` is unbuilt.

    The fast ``build`` CI job installs the package without compiling
    ``libcharmm``. Unit tests that reach production code importing
    ``pycharmm.lib`` then raise ``OSError``/``ModuleNotFoundError`` at runtime.
    When a real CHARMM build *is* present (``charmm`` job, developer machines)
    the guard is inert, so genuine regressions still surface there.
    """
    outcome = yield
    excinfo = getattr(outcome, "excinfo", None)
    if excinfo is None:
        return
    exc = excinfo[1]
    if not isinstance(exc, (OSError, ImportError)):
        return
    if charmm_env_configured():
        return  # CHARMM available: a failure here is a real bug, keep it.
    message = str(exc)
    if any(sig in message for sig in _CHARMM_UNAVAILABLE_SIGNATURES):
        outcome.force_exception(
            pytest.skip.Exception(
                f"libcharmm unavailable; skipping CHARMM-dependent test ({message.splitlines()[0]})"
            )
        )


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


# --- Synthetic systems for the checkpoint-free jaxmd-unified suite ----------
# A TIP3-like water box built directly as a MolecularSystem (no CHARMM build,
# no ML checkpoint), so the unified assemble -> compose -> jaxmd path can be
# exercised for regression and profiling on any machine with jax/jax-md.


def build_synthetic_water_box(n_waters: int = 8, box_len: float = 18.0, seed: int = 0):
    """A periodic MolecularSystem of ``n_waters`` rigid TIP3-geometry waters.

    Charges/LJ are the standard TIP3P values so ``mm_nonbonded`` produces
    physically-scaled (non-degenerate) energies; molecules are placed at random
    non-edge centres for reproducible-but-nontrivial dynamics.
    """
    import numpy as np

    from mmml.md.system import FFParams, MolecularSystem

    rng = np.random.default_rng(seed)
    geom = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    coords = []
    for _ in range(n_waters):
        centre = rng.uniform(2.0, box_len - 2.0, size=3)
        coords.append(geom + centre)
    R = np.concatenate(coords, axis=0)
    Z = np.tile([8, 1, 1], n_waters)
    mol_id = np.repeat(np.arange(n_waters), 3).astype(np.int32)
    ff = FFParams(
        charges=np.tile([-0.834, 0.417, 0.417], n_waters),
        epsilon=np.tile([0.152, 0.046, 0.046], n_waters),
        rmin_half=np.tile([1.768, 0.2245, 0.2245], n_waters),
        at_codes=np.tile([0, 1, 1], n_waters).astype(np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=R,
        Z=Z,
        box=np.diag([box_len, box_len, box_len]),
        mol_id=mol_id,
        monomer_indices=[np.arange(3 * g, 3 * g + 3) for g in range(n_waters)],
        water_indices=[np.arange(3 * g, 3 * g + 3) for g in range(n_waters)],
        ff_params=ff,
    )


@pytest.fixture
def synthetic_water_box():
    """Factory fixture: ``synthetic_water_box(n_waters=, box_len=, seed=)``."""
    return build_synthetic_water_box
