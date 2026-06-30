import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# current directory of current file
cwd = Path(__file__).parent
_REPO_ROOT = Path(__file__).resolve().parents[3]


from mmml.interfaces.pycharmmInterface.charmm_paths import bootstrap_charmm_env

CHARMM_HOME, CHARMM_LIB_DIR = bootstrap_charmm_env(repo_root=_REPO_ROOT)


def _ensure_vendored_pycharmm_on_path() -> None:
    """Prefer mmml's patched ``pycharmm`` over ``$CHARMM_HOME/tool/pycharmm``.

    ``sys.path.append(tool/pycharmm)`` alone lets an older CHARMM install shadow the
    vendored package (missing ``MLpot.skip_iblo_inb_update`` for PBC registration).
    """
    root = str(_REPO_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


_ensure_vendored_pycharmm_on_path()
if CHARMM_HOME:
    chmhp = Path(CHARMM_HOME) / "tool" / "pycharmm"
    if str(chmhp) not in sys.path:
        sys.path.append(str(chmhp))

CGENFF_RTF = cwd / ".." / ".." / "data" / "charmm" / "top_all36_cgenff.rtf"
CGENFF_RTF = CGENFF_RTF.resolve()
CGENFF_PRM = cwd / ".." / ".." / "data" / "charmm" / "par_all36_cgenff.prm"
CGENFF_PRM = CGENFF_PRM.resolve()

CGENFF_RTF = str(CGENFF_RTF)
CGENFF_PRM = str(CGENFF_PRM)

from mmml.interfaces.pycharmmInterface.charmm_mpi import (  # noqa: E402
    charmm_lib_available,
    prepare_serial_charmm_mpi_env,
    _under_mpirun,
)

_WARMUP_JAX_ONLY = (os.environ.get("MMML_WARMUP_MLPOT_JAX_ONLY") or "").strip().lower() in (
    "1",
    "yes",
    "true",
)

PYCHARMM_AVAILABLE = charmm_lib_available() and not _WARMUP_JAX_ONLY

if not _WARMUP_JAX_ONLY:
    prepare_serial_charmm_mpi_env()

pycharmm: Any = None
coor: Any = None
energy: Any = None
read: Any = None
settings: Any = None
psf: Any = None
minimize: Any = None

if PYCHARMM_AVAILABLE:
    import pycharmm as _pycharmm
    import pycharmm.coor as _coor
    import pycharmm.energy as _energy
    import pycharmm.minimize as _minimize
    import pycharmm.psf as _psf
    import pycharmm.read as _read
    import pycharmm.settings as _settings

    pycharmm = _pycharmm
    coor = _coor
    energy = _energy
    read = _read
    settings = _settings
    psf = _psf
    minimize = _minimize

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        charmm_lib_links_mpi,
        ensure_mpi4py_after_charmm_init,
    )

    if charmm_lib_links_mpi():
        ensure_mpi4py_after_charmm_init(phase="during import_pycharmm")


def _report_charmm_import_paths() -> None:
    if not PYCHARMM_AVAILABLE:
        return
    if (os.environ.get("MMML_QUIET") or "").strip().lower() in ("1", "yes", "true"):
        return
    try:
        from mmml.utils.rich_report import emit_charmm_env

        emit_charmm_env(
            cgenff_rtf=CGENFF_RTF,
            cgenff_prm=CGENFF_PRM,
            charmm_home=CHARMM_HOME,
            charmm_lib_dir=CHARMM_LIB_DIR,
        )
    except Exception:
        print(CGENFF_RTF)
        print(CGENFF_PRM)
        print("CHARMM_HOME", CHARMM_HOME)
        print("CHARMM_LIB_DIR", CHARMM_LIB_DIR)


_report_charmm_import_paths()


def get_block(a, b):
    block = f"""BLOCK
CALL 1 SELE .NOT. (RESID {a} .OR. RESID {b}) END
CALL 2 SELE (RESID {a} .OR. RESID {b}) END
COEFF 1 1 0.0
COEFF 2 2 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0
COEFF 1 2 0.0
END
"""
    return block


def reset_block() -> None:
    if not PYCHARMM_AVAILABLE:
        return
    if os.environ.get("MMML_SKIP_CHARMM_RESET_BLOCK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    block = """BLOCK 
        CALL 1 SELE ALL END
          COEFF 1 1 1.0 
        END
        """
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet
    from mmml.utils.rich_report import emit_charmm_block, is_verbose

    run_charmm_script_quiet(block)
    if is_verbose():
        try:
            emit_charmm_block("full MM (all atoms, COEFF 1.0)", verbose=True)
        except Exception:
            print("CHARMM BLOCK: full MM (all atoms, COEFF 1.0)", flush=True)


def should_skip_charmm_energy_show() -> bool:
    """
    True if CHARMM energy.show() / get_energy() should be skipped to avoid segfault.

    Segfaults in CHARMM's bond routines (e.g. ebondfs) are known on SLURM, some
    cluster nodes, or with certain MPI/threading. Set SKIP_CHARMM_ENERGY_SHOW=1
    (or "yes"/"true") to skip. When SLURM_JOB_ID is set, skip by default unless
    RUN_CHARMM_ENERGY_SHOW=1 is set. On macOS (darwin), skip by default due to
    bus errors in CHARMM's native code; set RUN_CHARMM_ENERGY_SHOW=1 to force.
    """
    ev = (os.environ.get("SKIP_CHARMM_ENERGY_SHOW") or "").strip().lower()
    if ev in ("1", "yes", "true"):
        return True
    force = (os.environ.get("RUN_CHARMM_ENERGY_SHOW") or "").strip().lower()
    if force in ("1", "yes", "true"):
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return True
    if sys.platform == "darwin":
        return True
    # CHARMM energy.show() can segfault under pytest on some Linux builds.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    # OpenMPI-linked CHARMM builds can segfault in domdec during print_energy.
    if os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE"):
        return True
    return False


def print_charmm_energy_summary() -> None:
    """Print ENER/USER in kcal/mol and eV after CHARMM ``energy.show()``."""
    if not PYCHARMM_AVAILABLE:
        return
    import math

    from mmml.data.units import format_energy_kcal_ev

    try:
        parts: list[str] = []
        for term in ("ENER", "USER"):
            try:
                val = float(energy.get_property_by_name(term))
            except Exception:
                continue
            if not math.isfinite(val):
                continue
            if term == "USER" and abs(val) < 1e-8:
                continue
            parts.append(f"{term}={format_energy_kcal_ev(val)}")
        if parts:
            print(f"  CHARMM energy summary: {', '.join(parts)}", flush=True)
    except Exception:
        pass


def safe_energy_show():
    """
    Call energy.show() unless the environment requests skipping it to avoid segfault.
    Use when CHARMM energy evaluation may crash (e.g. under SLURM / on some clusters).
    """
    if not PYCHARMM_AVAILABLE:
        return
    if should_skip_charmm_energy_show():
        print("Skipping energy.show() (SKIP_CHARMM_ENERGY_SHOW, SLURM, or macOS).")
    else:
        energy.show()
        print_charmm_energy_summary()


def reset_block_no_internal():
    # block = f"""BLOCK 
    #     CALL 1 SELE ALL END
    #       COEFF 1 1 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 
    #     END
    #     """
    # _ = pycharmm.lingo.charmm_script(block)
    pass


def view_atoms(atoms):
    from ase.visualize import view

    return view(atoms, viewer="x3d")

def get_forces_pycharmm():
    positions = coor.get_positions()
    force_command = """coor force sele all end"""
    _ = pycharmm.lingo.charmm_script(force_command)
    forces = coor.get_positions()
    coor.set_positions(positions)
    return forces

import pandas as pd
def set_pycharmm_xyz(atom_positions):
    xyz = pd.DataFrame(atom_positions, columns=["x", "y", "z"])
    coor.set_positions(xyz)


def capture_neighbour_list():
    # Print something
    distance_command = """
    open unit 1 write form name total.dmat
    
    COOR DMAT SINGLE UNIT 1 SELE ALL END SELE ALL END
    
    close unit 1"""
    _ = pycharmm.lingo.charmm_script(distance_command)

    with open("total.dmat") as f:
        output_dmat = f.read()

    atom_number_type_dict = {}
    atom_number_resid_dict = {}

    pair_distance_dict = {}
    pair_resid_dict = {}

    for _ in output_dmat.split("\n"):
        if _.startswith("*** "):
            _, n, resid, resname, at, _ = _.split()

            n = int(n.split("=")[0]) - 1
            atom_number_type_dict[n] = at
            atom_number_resid_dict[n] = int(resid) - 1

    for _ in output_dmat.split("\n"):
        if _.startswith("  "):
            a, b, dist = _.split()
            a = int(a) - 1
            b = int(b) - 1
            dist = float(dist)
            if atom_number_resid_dict[a] < atom_number_resid_dict[b]:
                pair_distance_dict[(a, b)] = dist
                pair_resid_dict[(a, b)] = (
                    atom_number_resid_dict[a],
                    atom_number_resid_dict[b],
                )

    return {
        "atom_number_type_dict": atom_number_type_dict,
        "atom_number_resid_dict": atom_number_resid_dict,
        "pair_distance_dict": pair_distance_dict,
        "pair_resid_dict": pair_resid_dict,
    }


def _launcher_mpi_size() -> int:
    """MPI world size from OpenMPI/PMI env (no mpi4py import)."""
    size_raw = (
        os.environ.get("OMPI_COMM_WORLD_SIZE")
        or os.environ.get("PMIX_SIZE")
        or os.environ.get("PMI_SIZE")
        or "1"
    )
    try:
        return max(1, int(size_raw))
    except ValueError:
        return 1


def _maybe_reset_block_at_import() -> None:
    """Run the default BLOCK reset once per process when safe under MPI.

    Under ``mpirun -np > 1``, ranks finish this module at different times
    (``import ase``, etc.).  ``eval_charmm_script`` during import deadlocks
    MPI-linked CHARMM: early ranks spin in Fortran receive while late ranks
    are still in Python.  Skip import-time ``reset_block`` for np>1; callers
    run it after a barrier or from rank-synchronized setup (see
    ``MMML_SKIP_CHARMM_RESET_BLOCK`` to force-skip on np=1 diagnostics).
    """
    if not PYCHARMM_AVAILABLE:
        return
    if os.environ.get("MMML_SKIP_CHARMM_RESET_BLOCK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    if _under_mpirun() and _launcher_mpi_size() > 1:
        return
    reset_block()


_maybe_reset_block_at_import()


def _init_charmm_default_levels() -> None:
    """Match ``mmml md-system`` defaults; ``bomlev 0`` aborts minimize in notebooks."""
    if not PYCHARMM_AVAILABLE:
        return
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

        apply_charmm_verbosity(prnlev=5, warnlev=5, bomlev=-2)
    except Exception:
        pass


_domdec_vacuum_disabled = False
_domdec_disabled_early = False


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def _should_run_domdec_off() -> bool:
    """Whether to send ``domdec off`` to CHARMM.

    Default **skip**. ``q_domdec`` starts false on DOMDEC builds; MM pretreat ``gete``
  works without ``domdec off``. Calling ``domdec off`` runs ``uninit_domdec`` even when
    DOMDEC was never active, which can corrupt OpenMPI pools and segfault the next
    ``gete`` in ``send_coord_to_recip`` / ``PMPI_Free_mem`` (benz100 MLpot SD on gpu09).

    Set ``MMML_FORCE_DOMDEC_OFF=1`` only if your stream explicitly enabled domdec.
    Prefer ``./scripts/rebuild_charmm_mlpot.sh --no-domdec`` for MPI MLpot campaigns.
    """
    if _truthy_env("MMML_NO_CHARMM_DOMDEC_OFF"):
        return False
    return _truthy_env("MMML_FORCE_DOMDEC_OFF")


def disable_charmm_domdec(*, when: str = "early") -> bool:
    """Turn off domdec once per process (repeat ``domdec off`` segfaults on DOMDEC builds).

    Skipped by default — see :func:`_should_run_domdec_off`. When enabled via
    ``MMML_FORCE_DOMDEC_OFF=1``, defer the single call until MLpot SD/dynamics
    (``when="mlpot_energy"``).
    """
    global _domdec_vacuum_disabled, _domdec_disabled_early
    if not _should_run_domdec_off():
        return False
    if _domdec_vacuum_disabled:
        if when == "mlpot_energy" and _domdec_disabled_early:
            print(
                "mmml: domdec off already ran before MLpot JAX warmup; cannot repeat "
                "domdec off on DOMDEC builds. Sync mmml (defer setup-time domdec off) "
                "and launch via scripts/mmml-charmm-mpirun.sh.",
                file=sys.stderr,
                flush=True,
            )
        return False
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    try:
        with charmm_relaxed_bomlev():
            pycharmm.lingo.charmm_script("domdec off")
    except Exception as exc:
        print(
            f"mmml: domdec off failed ({when}): {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    _domdec_vacuum_disabled = True
    if when != "mlpot_energy":
        _domdec_disabled_early = True
    return True


def ensure_domdec_off_for_mlpot_energy(*, context: str = "MLpot energy") -> bool:
    """Optional ``domdec off`` before MLpot SD / dynamics (off by default)."""
    from mmml.interfaces.pycharmmInterface.charmm_mpi import recover_mpi_for_charmm_after_jax

    ok = disable_charmm_domdec(when="mlpot_energy")
    if ok:
        recover_mpi_for_charmm_after_jax(phase=f"after domdec off ({context})")
    return ok


def crystal_free_charmm() -> None:
    """Clear periodic image state (safe to repeat)."""
    if not PYCHARMM_AVAILABLE:
        return
    try:
        pycharmm.lingo.charmm_script("crystal free")
    except Exception:
        pass


def force_charmm_vacuum_mode() -> None:
    """Vacuum helpers: defer domdec off to MLpot energy; crystal free may repeat."""
    crystal_free_charmm()


def _init_vacuum_charmm_state() -> None:
    # Do not call disable_charmm_domdec() here. MPI-linked libcharmm.so can
    # re-enable domdec after MPI_Init; domdec off is once-only per process.
    crystal_free_charmm()


_vacuum_charmm_synced = False


def init_vacuum_charmm_state_mpi() -> None:
    """Run ``crystal free`` once; under ``mpirun`` barrier so all ranks enter together.

    Module-import ``crystal free`` on one rank while others are still importing
    Python modules deadlocks MPI-linked CHARMM (rank 0 prints, rank 1 never arrives).
    """
    global _vacuum_charmm_synced
    if _vacuum_charmm_synced:
        return
    if not PYCHARMM_AVAILABLE:
        return
    if _under_mpirun():
        try:
            from mpi4py import MPI

            if MPI.Is_initialized():
                MPI.COMM_WORLD.Barrier()
        except Exception:
            pass
    _init_vacuum_charmm_state()
    _vacuum_charmm_synced = True


_init_charmm_default_levels()


def _skip_import_vacuum_init() -> bool:
    flag = os.environ.get("MMML_SKIP_VACUUM_CHARMM_INIT", "").strip().lower()
    return flag in ("1", "true", "yes")


if PYCHARMM_AVAILABLE and not _under_mpirun() and not _skip_import_vacuum_init():
    init_vacuum_charmm_state_mpi()


def ase_from_pycharmm_state():
    import ase

    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    Z = get_Z_from_psf()
    R = coor.get_positions()
    return ase.Atoms(Z, R)

def view_pycharmm_state():
    return view_atoms(ase_from_pycharmm_state())


def _apply_print_levels(prnlev: int, wrnlev: int) -> None:
    """Set CHARMM PRNLev/WRNLev via stream API (no command echo)."""
    if not PYCHARMM_AVAILABLE:
        return
    settings.set_verbosity(int(prnlev))
    settings.set_warn_level(int(wrnlev))


def pycharmm_quiet() -> None:
    _apply_print_levels(0, 0)


def pycharmm_soft() -> None:
    _apply_print_levels(1, 1)


def pycharmm_verbose() -> None:
    _apply_print_levels(5, 5)


def pycharmm_loud() -> None:
    _apply_print_levels(9, 9)


@contextmanager
def charmm_print_level(prnlev: int = 0, wrnlev: int | None = None):
    """Temporarily set CHARMM print/warning levels; restore on exit."""
    if not PYCHARMM_AVAILABLE:
        yield
        return
    if wrnlev is None:
        wrnlev = prnlev
    old_prn = settings.set_verbosity(int(prnlev))
    old_wrn = settings.set_warn_level(int(wrnlev))
    try:
        yield
    finally:
        settings.set_verbosity(old_prn)
        settings.set_warn_level(old_wrn)


from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev  # noqa: E402

if PYCHARMM_AVAILABLE:
    pycharmm_quiet()