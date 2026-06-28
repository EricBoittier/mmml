"""ctypes wrapper around the ScaFaCoS ``libfcs`` C API.

ScaFaCoS solves periodic Coulomb (electrostatic / gravitational) problems with
PME, P³M, P²NFFT, MSM, and related methods behind a single frontend.  MMML uses
it as an **optional long-range backend** for hybrid ML/MM electrostatics when
the shared library is installed (see README in this directory).

API reference (upstream):
  - User's guide: ``doc/manual.pdf`` in the ScaFaCoS source tree
  - Doxygen: https://www.scafacos.de/doxygen/
  - C header: ``fcs_interface_p.h`` (installed with the library)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Common ScaFaCoS method strings (configure-time availability may vary).
SCAFACOS_METHODS = (
    "p2nfft",
    "p3m",
    "p2m",
    "ewald",
    "direct",
    "memd",
    "msm",
    "mpe2",
    "pe2",
    "vm2",
)

_DEFAULT_LIB_NAMES = (
    "libfcs.so",
    "libfcs.dylib",
    "libfcs.dll",
    "libscafacos.so",
)


class ScaFaCoSUnavailable(RuntimeError):
    """Raised when ``libfcs`` cannot be loaded or a call fails at init."""


@dataclass(frozen=True)
class CoulombFieldResult:
    """Electrostatic energy and per-atom forces from one ``fcs_run`` call."""

    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray  # shape (n_atoms, 3)


def resolve_scafacos_library_path() -> Path | None:
    """Return the first existing ScaFaCoS shared library on the search path."""
    env = os.environ.get("SCAFACOS_LIB", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
    for directory in (
        os.environ.get("SCAFACOS_ROOT", ""),
        "/usr/local/lib",
        "/usr/lib",
    ):
        if not directory:
            continue
        root = Path(directory).expanduser()
        for name in _DEFAULT_LIB_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
    for stem in ("fcs", "scafacos"):
        found = ctypes.util.find_library(stem)
        if found:
            return Path(found).resolve()
    return None


def load_scafacos_library(path: Path | str | None = None) -> ctypes.CDLL:
    """Load ``libfcs`` with required function signatures bound."""
    if path is None:
        resolved = resolve_scafacos_library_path()
        if resolved is None or not resolved.resolve().is_file():
            raise ScaFaCoSUnavailable(
                "ScaFaCoS shared library not found. Set SCAFACOS_LIB to libfcs.so "
                "or install from https://github.com/scafacos/scafacos"
            )
        path = resolved

    # Shared ScaFaCoS defers solver symbols to plugin .so files loaded at runtime.
    # Lazy + global binding lets fcs_init dlopen plugins after libfcs is mapped.
    mode = getattr(ctypes, "RTLD_GLOBAL", 256) | getattr(ctypes, "RTLD_LAZY", 1)
    lib = ctypes.CDLL(str(path), mode=mode)

    lib.fcs_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.fcs_init.restype = ctypes.c_char_p

    lib.fcs_destroy.argtypes = [ctypes.c_void_p]
    lib.fcs_destroy.restype = ctypes.c_char_p

    lib.fcs_set_common.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.fcs_set_common.restype = ctypes.c_char_p

    lib.fcs_set_parameters.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.fcs_set_parameters.restype = ctypes.c_char_p

    lib.fcs_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.fcs_run.restype = ctypes.c_char_p

    return lib


def have_scafacos() -> bool:
    """True when ``libfcs`` is discoverable and loads without error."""
    try:
        load_scafacos_library()
        return True
    except (ScaFaCoSUnavailable, OSError):
        return False


def _check_fcs_result(err: bytes | None, *, where: str) -> None:
    if err is None:
        return
    text = err.decode("utf-8", errors="replace").strip()
    if not text or text.upper() in ("NULL", "FCS_SUCCESS", "SUCCESS"):
        return
    raise ScaFaCoSUnavailable(f"ScaFaCoS {where} failed: {text}")


def _orthorhombic_box_vectors(box_length_A: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = float(box_length_A)
    a = np.array([L, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, L, 0.0], dtype=np.float64)
    c = np.array([0.0, 0.0, L], dtype=np.float64)
    return a, b, c


class ScaFaCoSSession:
    """One ScaFaCoS solver instance (``fcs_init`` … ``fcs_destroy`` lifecycle)."""

    def __init__(
        self,
        *,
        method: str = "p2nfft",
        mpi_comm: int | None = None,
        library_path: Path | str | None = None,
    ) -> None:
        self._lib = load_scafacos_library(library_path)
        self._handle = ctypes.c_void_p()
        if mpi_comm is None:
            try:
                from mpi4py import MPI  # noqa: WPS433 — initializes MPI for ScaFaCoS

                mpi_comm = int(MPI.COMM_WORLD.py2f())
            except Exception:
                mpi_comm = 0
        err = self._lib.fcs_init(
            ctypes.byref(self._handle),
            method.encode("ascii"),
            int(mpi_comm),
        )
        _check_fcs_result(err, where="fcs_init")
        self._destroyed = False

    def close(self) -> None:
        if self._destroyed or not self._handle.value:
            return
        err = self._lib.fcs_destroy(self._handle)
        _check_fcs_result(err, where="fcs_destroy")
        self._destroyed = True

    def __enter__(self) -> ScaFaCoSSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def set_parameter(self, key: str, value: str | float | int) -> None:
        """Set one solver parameter via ``fcs_set_parameters`` (``key,value`` string)."""
        param_str = f"{key},{value}"
        err = self._lib.fcs_set_parameters(
            self._handle,
            param_str.encode("ascii"),
            0,
        )
        _check_fcs_result(err, where=f"fcs_set_parameters({key!r})")

    def configure_cubic_box(
        self,
        *,
        box_length_A: float,
        n_atoms: int,
        periodicity: Sequence[int] = (1, 1, 1),
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        short_range_flag: int = 0,
    ) -> None:
        """Call ``fcs_set_common`` for a cubic orthorhombic cell (Å)."""
        a, b, c = _orthorhombic_box_vectors(box_length_A)
        offset_arr = np.asarray(offset, dtype=np.float64)
        periodicity_arr = np.asarray(periodicity, dtype=np.int32)
        err = self._lib.fcs_set_common(
            self._handle,
            int(short_range_flag),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            offset_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            periodicity_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(n_atoms),
        )
        _check_fcs_result(err, where="fcs_set_common")

    def run_coulomb(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
    ) -> CoulombFieldResult:
        """Evaluate electrostatic energy and forces for local particle data.

        ``positions_A`` must be shape ``(n_atoms, 3)`` in Å.  ``charges_e`` is
        in elementary charge units.  Returned forces use kcal/mol/Å (CHARMM-like)
        after converting ScaFaCoS field output with ``332.063711`` kcal·Å/e².
        """
        pos = np.asarray(positions_A, dtype=np.float64)
        chg = np.asarray(charges_e, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be (n_atoms, 3); got {pos.shape}")
        if chg.shape[0] != pos.shape[0]:
            raise ValueError(
                f"charges length {chg.shape[0]} != n_atoms {pos.shape[0]}"
            )
        n_loc = int(pos.shape[0])
        flat_pos = np.ascontiguousarray(pos.reshape(-1))
        flat_chg = np.ascontiguousarray(chg)
        field = np.zeros(3 * n_loc, dtype=np.float64)
        potential = np.zeros(n_loc, dtype=np.float64)
        err = self._lib.fcs_run(
            self._handle,
            n_loc,
            flat_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flat_chg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            field.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            potential.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        _check_fcs_result(err, where="fcs_run")
        forces = field.reshape(n_loc, 3) * 332.063711
        energy = -0.5 * float(np.dot(chg, potential)) * 332.063711
        return CoulombFieldResult(energy_kcalmol=energy, forces_kcalmol_A=forces)


def compute_scafacos_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    method: str = "p2nfft",
    parameters: dict[str, str | float | int] | None = None,
    mpi_comm: int | None = None,
) -> CoulombFieldResult:
    """Convenience one-shot Coulomb evaluation (init → run → destroy)."""
    with ScaFaCoSSession(method=method, mpi_comm=mpi_comm) as session:
        session.configure_cubic_box(
            box_length_A=box_length_A,
            n_atoms=int(positions_A.shape[0]),
        )
        if parameters:
            for key, value in parameters.items():
                session.set_parameter(key, value)
        return session.run_coulomb(positions_A, charges_e)
