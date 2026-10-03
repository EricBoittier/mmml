#if KEY_MLMM==1

// ============================================================================
// type_pyth.cxx
// ============================================================================
//
// Persistent Python-socket ML-only backend for CHARMM MLMM/PYTH.
//
// This file keeps the current Fortran/C ABI contract unchanged:
//
//   charmm_pyth_internal_setup(
//       int pyth_in_use,
//       int pyth_use_gpu,
//       const char* model_type,
//       const char* model_name,
//       int pyth_charge,
//       int pyth_multiplicity,
//       int pyth_pt_nml,
//       const int* pyth_in_mlidx,
//       const int* pyth_in_mlZid,
//       const int* pyth_in_mlmaskid,
//       int pyth_in_natoms,
//       int* pyth_out_setup_err)
//
// The Python runner is embedded below, written to /tmp during setup, and then
// launched with python3 or MLPS_PYTHON.
//
// MODEL CONTRACT
// --------------
//   model_type : "uma", "mace", "tani", or "dummy"
//   model_name : explicit MODL string using one of:
//
//       repo:<model>
//       repo:<model>:<option>
//       local:<model>:<path>
//       local:<model>:<option>:<path>
//
//     repo  = named package/HF/cache model; download/cache may happen during SETUP
//     local = already-downloaded file; path must exist; no download attempt
//
//     <option> meaning:
//       UMA  : task name, e.g. omol, omat, oc20
//       MACE : model variant/key, e.g. large, extra_large, polar-1-m
//       TANI : no MODL option; native unit is MLPS_TANI_NATIVE_UNIT
//
//   charge       : integer total charge, optional for models that ignore it
//   multiplicity : integer spin multiplicity, optional for models that ignore it
//
// Units:
//   coordinates sent to Python : Angstrom
//   energy returned by Python  : kcal/mol
//   gradients returned         : kcal/mol/Angstrom, dE/dR, not force
//
// Socket protocol version: 2
//
// Optional environment controls:
//   MLPS_PYTHON=/path/to/python
//   MLPS_PY_STARTUP_TIMEOUT_SEC=300
//   MLPS_PY_STARTUP_SLEEP_US=100000
//   MLPS_PY_IO_TIMEOUT_MS=600000
//   MLPS_PY_PRINT_EVERY=1
//   MLPS_PY_MAX_ABS_GRAD_WARN=1.0e4
//   MLPS_MACE_DTYPE=float64
//   MLPS_TANI_NATIVE_UNIT=hartree
//
// ============================================================================

#include <algorithm>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <type_traits>
#include <vector>

#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

static_assert(sizeof(int) == 4, "C++ int must be 32-bit for Fortran integer(c_int)");
static_assert(sizeof(int32_t) == 4, "int32_t must be 32-bit");
static_assert(sizeof(double) == 8, "double must be 64-bit");

constexpr int32_t MLPS_MAGIC   = 0x53504C4D;  // bytes "MLPS" on little endian
constexpr int32_t MLPS_VERSION = 2;
constexpr int32_t CMD_SETUP    = 1;
constexpr int32_t CMD_FORCE    = 2;
constexpr int32_t CMD_STOP     = 3;
constexpr int32_t STATUS_OK    = 0;

const char EMBEDDED_MLMM_RUNNER[] = R"PYMLMMX(

#!/usr/bin/env python3
"""
Embedded Python runner for CHARMM MLMM/PYTH.

The Fortran/C ABI is unchanged. The only model-selection information from
CHARMM is still:

    model_type  = UMA | MACE | TANI | DUMMY
    model_name  = MODL string
    charge      = integer total charge
    multiplicity = integer spin multiplicity

MODL grammar
------------

    repo:<model>
    repo:<model>:<option>

    local:<model>:<path>
    local:<model>:<option>:<path>

The first field is case-insensitive. This is deliberate because some CHARMM
command paths uppercase tokens before they reach C/Python. Therefore these are
equivalent for repo models:

    repo:uma-s-1p2:omol
    REPO:UMA-S-1P2:OMOL

Meaning:

    repo   : use package/HF/cache named-model loading; download/cache may happen
             during SETUP, never during FORCE calls.
    local  : use an already-downloaded local file. The path must exist. The path
             is used exactly as received, so Fortran must preserve path case on
             case-sensitive filesystems.

The optional <option> is backend-specific:

    UMA   : task name, for example omol, omat, oc20. Defaults to omol.
    MACE  : model variant/key, for example large, extra_large, polar-1-m.
    TANI  : no MODL option is accepted. Native unit is controlled by
            MLPS_TANI_NATIVE_UNIT, but CHARMM always receives kcal/mol.

Units sent back to CHARMM
-------------------------

    energy   : kcal/mol
    gradient : kcal/mol/Angstrom, dE/dR, not force
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import struct
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple


def check_python_version() -> None:
    if sys.version_info < (3, 9) or sys.version_info >= (3, 14):
        raise RuntimeError(
            f"Unsupported Python version {sys.version.split()[0]}. "
            "Required: >=3.9 and <=3.13."
        )


try:
    check_python_version()
except Exception as exc:
    raise SystemExit(f"[PYTH] {exc}")

_np = None


def get_numpy():
    global _np
    if _np is None:
        try:
            import numpy as numpy_module
        except Exception as exc:
            raise RuntimeError(
                "Missing required Python module for common socket/numeric backend: numpy\n"
                "Install it in the Python environment used by CHARMM, for example: pip install numpy\n"
                f"Python executable: {sys.executable}\n"
                f"Original error: {exc}"
            ) from exc
        _np = numpy_module
    return _np


MAGIC = b"MLPS"
MLPS_VERSION = 2

CMD_SETUP = 1
CMD_FORCE = 2
CMD_STOP = 3

STATUS_OK = 0
STATUS_ERR = 1

EV_TO_KCAL_MOL = 23.060548867
HARTREE_TO_KCAL_MOL = 627.5094740631


@dataclass(frozen=True)
class ModlSpec:
    raw: str
    source: str                  # "repo" or "local"
    model: str                   # backend model/family label
    option: Optional[str] = None # UMA task or MACE model variant
    path: Optional[str] = None   # local file path only

    @property
    def is_repo(self) -> bool:
        return self.source == "repo"

    @property
    def is_local(self) -> bool:
        return self.source == "local"

    def describe(self) -> str:
        return (
            f"source={self.source}, model={self.model!r}, "
            f"option={self.option!r}, path={self.path!r}"
        )


@dataclass
class ModelOptions:
    model_type: str
    model_name: str
    charge: int = 0
    multiplicity: int = 1


@dataclass
class SetupState:
    natoms: int
    nml: int
    gpu: int
    options: ModelOptions
    ml_idx: Any
    ml_zid: Any
    ml_maskid: Any


# -----------------------------------------------------------------------------
# Generic validation / parsing utilities
# -----------------------------------------------------------------------------


def require_modules(module_names: list[str], context: str, install_hint: str = "") -> None:
    missing: list[str] = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        msg = f"Missing required Python module(s) for {context}: " + ", ".join(missing)
        if install_hint:
            msg += "\n" + install_hint
        msg += f"\nPython executable: {sys.executable}"
        raise RuntimeError(msg)


def require_file(path: str, context: str) -> str:
    if not path or not path.strip():
        raise RuntimeError(f"{context}: empty local model path")

    p = Path(path).expanduser()

    if not p.exists():
        extra = ""
        raw = str(path)
        # If CHARMM/Fortran uppercased the path before it reached Python, we
        # cannot recover the original case here. Make that failure mode explicit.
        if raw == raw.upper() and any(c.isalpha() for c in raw):
            extra = (
                "\nThe received path is all-uppercase. On Linux/macOS this usually means "
                "the CHARMM/Fortran command parser uppercased MODL. The Python "
                "side accepts REPO:/LOCAL: case-insensitively, but local file "
                "paths must be preserved exactly."
            )
        raise FileNotFoundError(f"{context}: local model file not found: {path}{extra}")

    if not p.is_file():
        raise FileNotFoundError(f"{context}: local model path is not a regular file: {path}")

    return str(p)


def normalize_key(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace("\\", "/")
        .split("/")[-1]
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def canonical_model_field(text: str) -> str:
    """Normalize non-path MODL fields that may have been uppercased by CHARMM."""
    value = str(text).strip()
    if not value:
        return ""
    return value.lower()


def canonical_option_field(text: Optional[str]) -> Optional[str]:
    """Normalize optional non-path MODL field; preserve underscores where useful."""
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return ""
    value = value.lower().replace(" ", "-")

    # Common spelling normalization needed because CHARMM may uppercase fields,
    # while Python package APIs often expect exact lowercase keys.
    key = normalize_key(value)
    if key in {"extra-large", "extralarge"}:
        return "extra_large"
    if key in {"polar-1-m", "polar-m"}:
        return "polar-1-m"
    if key in {"polar-1-l", "polar-l"}:
        return "polar-1-l"
    return value


def parse_modl_spec(model_type: str, modl: str) -> ModlSpec:
    """
    Parse the explicit MODL grammar while keeping the external ABI unchanged.

        repo:<model>
        repo:<model>:<option>
        local:<model>:<path>
        local:<model>:<option>:<path>

    The scheme is case-insensitive because CHARMM may uppercase command tokens.
    Non-path fields are normalized to lowercase. Local paths are preserved exactly.
    """
    mt = str(model_type).strip().lower()
    raw = str(modl).strip()

    if mt == "dummy":
        return ModlSpec(raw=raw, source="repo", model="dummy")

    if not raw:
        raise RuntimeError(f"{mt.upper()} MODL is empty. Expected repo:... or local:...")

    head, sep, tail = raw.partition(":")
    if not sep:
        raise RuntimeError(
            f"Invalid MODL {raw!r}. Use explicit repo:/local: grammar. Examples:\n"
            "  repo:uma-s-1p2:omol\n"
            "  local:uma-s-1p2:omol:/path/to/uma.pt\n"
            "  repo:mace_omol_0:extra_large\n"
            "  local:mace_omol_0:extra_large:/path/to/mace.model\n"
            "  repo:ANI2x\n"
            "  local:tani:/path/to/tani.pt"
        )

    source = head.strip().lower()
    if source not in {"repo", "local"}:
        raise RuntimeError(
            f"Invalid MODL {raw!r}. First field must be repo: or local: "
            "case-insensitive, for example REPO:UMA-S-1P2:OMOL is accepted."
        )

    if source == "repo":
        parts = tail.split(":", 1)
        model = canonical_model_field(parts[0])
        option = canonical_option_field(parts[1]) if len(parts) == 2 else None

        if not model:
            raise RuntimeError(f"Invalid MODL {raw!r}: missing model after repo:")
        if option == "":
            raise RuntimeError(f"Invalid MODL {raw!r}: empty option after trailing ':'")

        return ModlSpec(raw=raw, source="repo", model=model, option=option)

    # source == "local"
    parts = tail.split(":", 2)
    if len(parts) < 2:
        raise RuntimeError(
            f"Invalid MODL {raw!r}. Expected local:<model>:<path> "
            "or local:<model>:<option>:<path>."
        )

    model = canonical_model_field(parts[0])
    if not model:
        raise RuntimeError(f"Invalid MODL {raw!r}: missing model after local:")

    if len(parts) == 2:
        option = None
        path = parts[1].strip()
    else:
        option = canonical_option_field(parts[1])
        path = parts[2].strip()
        if option == "":
            raise RuntimeError(f"Invalid MODL {raw!r}: empty local option")

    if not path:
        raise RuntimeError(f"Invalid MODL {raw!r}: empty local path")

    checked_path = require_file(path, f"{mt.upper()} local MODL")
    return ModlSpec(raw=raw, source="local", model=model, option=option, path=checked_path)


def resolve_device(gpu: int) -> str:
    """
    C++ passes:
        gpu < 0  => CPU
        gpu >= 0 => physical GPU id

    If GPU is requested, expose only that GPU and use logical cuda.
    """
    if int(gpu) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        return "cuda"

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"


def dependency_check(options: ModelOptions) -> None:
    check_python_version()
    get_numpy()

    t = options.model_type.lower().strip()

    if t == "dummy":
        return

    spec = parse_modl_spec(t, options.model_name)

    if t == "uma":
        require_modules(
            ["ase", "torch", "fairchem"],
            context="UMA backend",
            install_hint="Install dependencies, for example: pip install ase fairchem-core torch",
        )
        return

    if t == "mace":
        require_modules(
            ["ase", "torch", "mace"],
            context="MACE backend",
            install_hint="Install dependencies, for example: pip install ase torch mace-torch",
        )
        return

    if t in {"tani", "torchani", "ani"}:
        tani_kind = normalize_key(spec.model).replace("-", "")
        if spec.option is not None:
            raise RuntimeError(
                "TANI/TorchANI MODL does not use the optional <option> field. "
                "Use repo:ANI2x or local:tani:/path/to/model.pt. "
                "Native-unit conversion is controlled internally by MLPS_TANI_NATIVE_UNIT, "
                "but CHARMM always receives kcal/mol."
            )

        if spec.is_repo:
            if tani_kind not in {"ani2x", "ani1x", "ani1ccx"}:
                raise RuntimeError("TANI repo models supported: ANI2x, ANI1x, ANI1ccx.")
            require_modules(
                ["torch", "torchani"],
                context="TorchANI built-in backend",
                install_hint="Install TorchANI and PyTorch: pip install torchani torch",
            )
        else:
            require_modules(
                ["torch"],
                context="TANI local backend",
                install_hint="Install PyTorch in this Python environment.",
            )
        return

    raise RuntimeError(f"unknown model_type={options.model_type!r}. Supported: dummy, uma, mace, tani.")


# -----------------------------------------------------------------------------
# Socket protocol helpers
# -----------------------------------------------------------------------------


def read_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes

    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed while reading {nbytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def read_i32(conn: socket.socket) -> int:
    return struct.unpack("<i", read_exact(conn, 4))[0]


def read_f64_array(conn: socket.socket, n: int):
    if n < 0:
        raise RuntimeError(f"negative float64 array length: {n}")
    np = get_numpy()
    buf = read_exact(conn, 8 * n)
    return np.frombuffer(buf, dtype="<f8").astype(np.float64, copy=True)


def read_i32_array(conn: socket.socket, n: int):
    if n < 0:
        raise RuntimeError(f"negative int32 array length: {n}")
    np = get_numpy()
    buf = read_exact(conn, 4 * n)
    return np.frombuffer(buf, dtype="<i4").astype(np.int32, copy=True)


def read_string(conn: socket.socket) -> str:
    n = read_i32(conn)
    if n < 0:
        raise RuntimeError(f"negative string length: {n}")
    if n == 0:
        return ""
    return read_exact(conn, n).decode("utf-8")


def send_i32(conn: socket.socket, value: int) -> None:
    conn.sendall(struct.pack("<i", int(value)))


def send_f64(conn: socket.socket, value: float) -> None:
    conn.sendall(struct.pack("<d", float(value)))


def send_f64_array(conn: socket.socket, array: Any) -> None:
    np = get_numpy()
    arr = np.asarray(array, dtype="<f8")
    conn.sendall(arr.tobytes(order="C"))


def send_string(conn: socket.socket, text: str) -> None:
    data = text.encode("utf-8", errors="replace")
    send_i32(conn, len(data))
    if data:
        conn.sendall(data)


def send_status_error(conn: Optional[socket.socket], message: str) -> None:
    if conn is None:
        return
    try:
        send_i32(conn, STATUS_ERR)
        send_string(conn, message)
    except Exception:
        pass


def read_header(conn: socket.socket, context: str) -> int:
    magic = read_exact(conn, 4)
    if magic != MAGIC:
        raise RuntimeError(f"bad {context} magic: got {magic!r}, expected {MAGIC!r}")

    version = read_i32(conn)
    if version != MLPS_VERSION:
        raise RuntimeError(f"bad {context} protocol version: got {version}, expected {MLPS_VERSION}")

    return read_i32(conn)


def recv_setup(conn: socket.socket) -> SetupState:
    cmd = read_header(conn, "setup")
    if cmd != CMD_SETUP:
        raise RuntimeError(f"expected setup cmd={CMD_SETUP}, got {cmd}")

    natoms = read_i32(conn)
    nml = read_i32(conn)
    gpu = read_i32(conn)

    model_type = read_string(conn).strip().lower()
    model_name = read_string(conn).strip()

    charge = read_i32(conn)
    multiplicity = read_i32(conn)

    if natoms <= 0:
        raise RuntimeError(f"invalid natoms in setup: {natoms}")
    if nml <= 0:
        raise RuntimeError(f"invalid nml in setup: {nml}")
    if not model_type:
        raise RuntimeError("empty model_type in setup")
    if not model_name and model_type != "dummy":
        raise RuntimeError("empty MODL/model_name in setup")
    if multiplicity <= 0:
        multiplicity = 1

    ml_idx = read_i32_array(conn, nml)
    ml_zid = read_i32_array(conn, nml)
    ml_maskid = read_i32_array(conn, natoms)

    options = ModelOptions(
        model_type=model_type,
        model_name=model_name,
        charge=charge,
        multiplicity=multiplicity,
    )

    print(
        "[PYTH] SETUP received: "
        f"version={MLPS_VERSION}, natoms={natoms}, nml={nml}, gpu={gpu}, "
        f"type={model_type}, MODL={model_name}, charge={charge}, multiplicity={multiplicity}, "
        f"Z={ml_zid.tolist()}",
        flush=True,
    )

    dependency_check(options)

    return SetupState(
        natoms=natoms,
        nml=nml,
        gpu=gpu,
        options=options,
        ml_idx=ml_idx,
        ml_zid=ml_zid,
        ml_maskid=ml_maskid,
    )


def validate_force_result(energy: Any, grad: Any, nml: int) -> Tuple[float, Any]:
    np = get_numpy()
    energy_f = float(energy)
    grad_a = np.asarray(grad, dtype=np.float64)

    if grad_a.shape != (nml, 3):
        raise RuntimeError(f"gradient shape mismatch: got {grad_a.shape}, expected {(nml, 3)}")

    if not np.isfinite(energy_f):
        raise RuntimeError(f"non-finite energy from predictor: {energy_f}")

    if not np.all(np.isfinite(grad_a)):
        raise RuntimeError("non-finite gradient from predictor")

    return energy_f, grad_a


def send_force_result(conn: socket.socket, energy: float, grad: Any, nml: int) -> None:
    energy, grad = validate_force_result(energy, grad, nml)

    send_i32(conn, STATUS_OK)
    send_f64(conn, energy)
    send_f64_array(conn, grad[:, 0])
    send_f64_array(conn, grad[:, 1])
    send_f64_array(conn, grad[:, 2])


# -----------------------------------------------------------------------------
# Main force-serving loop
# -----------------------------------------------------------------------------


def serve(conn: socket.socket, predictor: "Predictor", state: SetupState) -> None:
    np = get_numpy()
    nforce = 0
    print_every = int(os.environ.get("MLPS_PY_PRINT_EVERY", "1"))
    max_abs_grad_warn = float(os.environ.get("MLPS_PY_MAX_ABS_GRAD_WARN", "1.0e4"))

    while True:
        try:
            cmd = read_header(conn, "command")
        except ConnectionError as exc:
            print(f"[PYTH] C++ closed socket while waiting for command: {exc}", flush=True)
            return

        if cmd == CMD_STOP:
            print("[PYTH] STOP received.", flush=True)
            try:
                send_i32(conn, STATUS_OK)
            except OSError:
                pass
            return

        if cmd != CMD_FORCE:
            raise RuntimeError(f"unknown command: {cmd}")

        nml = read_i32(conn)
        if nml != state.nml:
            raise RuntimeError(f"nml mismatch in force call: got {nml}, expected {state.nml}")

        x = read_f64_array(conn, nml)
        y = read_f64_array(conn, nml)
        z = read_f64_array(conn, nml)
        coords = np.stack((x, y, z), axis=1).astype(np.float64, copy=False)

        nforce += 1

        try:
            energy, grad = predictor.predict(coords)
            energy, grad = validate_force_result(energy, grad, nml)

            gnorm = float(np.linalg.norm(grad))
            gmax = float(np.max(np.abs(grad)))

            if print_every > 0 and (nforce <= 5 or nforce % print_every == 0):
                print(
                    f"[PYTH] FORCE {nforce}: "
                    f"E={energy:.12f} kcal/mol |grad|={gnorm:.6e} gmax={gmax:.6e} "
                    f"x0=({coords[0,0]:.6f} {coords[0,1]:.6f} {coords[0,2]:.6f}) "
                    f"g0=({grad[0,0]:.6e} {grad[0,1]:.6e} {grad[0,2]:.6e})",
                    flush=True,
                )

            if gmax > max_abs_grad_warn:
                print(
                    f"[PYTH] WARNING: large gradient in FORCE {nforce}: "
                    f"gmax={gmax:.6e} kcal/mol/A",
                    flush=True,
                )

            send_force_result(conn, energy, grad, nml)

        except Exception:
            msg = traceback.format_exc()
            print(msg, flush=True)
            send_status_error(conn, msg)
            return


# -----------------------------------------------------------------------------
# Predictors
# -----------------------------------------------------------------------------


class Predictor(Protocol):
    def predict(self, coords_angstrom: Any) -> Tuple[float, Any]:
        ...

class DummyPredictor:
    def __init__(self,atomic_numbers: Any,gpu: int,options: ModelOptions):
        np = get_numpy()

        self.atomic_numbers = np.asarray(atomic_numbers,dtype=np.int64,)
        self.nml = int(self.atomic_numbers.size)

        # Reference coordinates are created on the first predict() call.
        self.r0: Optional[Any] = None

        # kcal/mol/Angstrom^2
        self.k = float(os.environ.get("MLPS_DUMMY_K", "1.0"))

        print("[PYTH-DUMMY] initialized: "f"nml={self.nml}, k={self.k}, dtype=float64",flush=True,)

    def predict(self,coords_angstrom: Any) -> Tuple[float, Any]:
        np = get_numpy()

        coords = np.asarray(coords_angstrom,dtype=np.float64)

        expected_shape = (self.nml, 3)
        if coords.shape != expected_shape:
            raise ValueError(
                "DUMMY coordinate shape mismatch: "
                f"got {coords.shape}, expected {expected_shape}"
            )

        # Establish the reference geometry on the first call.
        # copy() creates independent NumPy-owned storage.
        if self.r0 is None:
            self.r0 = coords.copy()

        dr = coords - self.r0

        # Coordinates: Angstrom
        # k: kcal/mol/Angstrom^2
        # Energy: kcal/mol
        energy = 0.5 * self.k * float(np.sum(dr * dr)) 

        # Gradient: kcal/mol/Angstrom
        grad = self.k * dr

        return energy, grad

# This class is not used # for internal testing with torch
class DummyPredictorTorch: 
    def __init__(self,atomic_numbers: Any,gpu: int,options: ModelOptions):
        np = get_numpy()
        import torch

        self.torch = torch
        self.device = resolve_device(gpu)
        self.dtype = torch.float64

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for DUMMY, but torch.cuda.is_available() is False.")

        atomic_numbers_np = np.asarray(atomic_numbers,dtype=np.int64)
        self.atomic_numbers = torch.tensor(atomic_numbers_np,dtype=torch.long,device=self.device)
        self.nml = int(self.atomic_numbers.numel())

        # Reference coordinates are created on the first predict() call.
        self.r0: Optional[Any] = None

        # kcal/mol/Angstrom^2
        self.k = float(os.environ.get("MLPS_DUMMY_K", "1.0"))

        print(
            "[PYTH-DUMMY] initialized: "
            f"nml={self.nml}, k={self.k}, "
            f"device={self.device}, dtype={self.dtype}",
            flush=True,
        )

    def predict(self, coords_angstrom: Any) -> Tuple[float, Any]:
        np = get_numpy()
        torch = self.torch

        coords_np = np.asarray(coords_angstrom,dtype=np.float64)

        expected_shape = (self.nml, 3)
        if coords_np.shape != expected_shape:
            raise ValueError(
                "DUMMY coordinate shape mismatch: "
                f"got {coords_np.shape}, expected {expected_shape}"
            )

        # torch.tensor() creates independent Torch-owned storage.
        coords = torch.tensor(coords_np,dtype=self.dtype,device=self.device,requires_grad=True)

        # Establish the reference geometry on the first call.
        if self.r0 is None:
            self.r0 = coords.detach().clone()

        dr = coords - self.r0

        # Energy is already in kcal/mol because:
        #   coordinates: Angstrom
        #   k: kcal/mol/Angstrom^2
        energy_tensor = self.k * torch.sum(dr * dr) # No 0.5 factor like CHARMM

        grad_tensor = torch.autograd.grad(
            outputs=energy_tensor,
            inputs=coords,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]


        energy = float(energy_tensor.detach().cpu().item())
        grad = (grad_tensor.detach().cpu().numpy().astype(np.float64, copy=False))

        return energy, grad


class UMAPredictor:
    def __init__(self, atomic_numbers: Any, gpu: int, options: ModelOptions):
        np = get_numpy()
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.device = resolve_device(gpu)
        self.spec = parse_modl_spec("uma", options.model_name)

        self.task_name = (self.spec.option or "omol").lower()
        self.model_key = self.spec.path if self.spec.is_local else self.spec.model
        self.charge = int(options.charge)
        self.spin = int(options.multiplicity)

        print(
            "[PYTH-UMA] initializing: "
            f"{self.spec.describe()}, task={self.task_name!r}, "
            f"charge={self.charge}, spin={self.spin}, device={self.device}, "
            f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'",
            flush=True,
        )

        import torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for UMA, but torch.cuda.is_available() is False.")

        from ase import Atoms
        from fairchem.core import pretrained_mlip

        try:
            from fairchem.core import FAIRChemCalculator  # type: ignore
        except Exception:
            from fairchem.core.common.relaxation.ase_utils import FAIRChemCalculator  # type: ignore

        self.Atoms = Atoms

        try:
            if self.spec.is_local:
                self.predictor = self._load_local_predictor(self.model_key)
                loader_desc = "fairchem.core.units.mlip_unit.load_predict_unit(path=...)"
            else:
                self.predictor = pretrained_mlip.get_predict_unit(self.model_key, device=self.device)
                loader_desc = "fairchem.core.pretrained_mlip.get_predict_unit(name=...)"

            self.calc = FAIRChemCalculator(self.predictor, task_name=self.task_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load UMA model through FAIR-Chem.\n"
                f"Parsed MODL: {self.spec.describe()}\n"
                f"UMA loader selected: {loader_desc if 'loader_desc' in locals() else 'unknown'}\n"
                f"UMA model_key/path: {self.model_key!r}\n"
                f"task_name passed to FAIRChemCalculator: {self.task_name!r}\n"
                "For repo: use a registered FAIR-Chem UMA model name and configured access/cache. "
                "For local: use an existing FAIR-Chem inference checkpoint compatible with load_predict_unit."
            ) from exc

        print("[PYTH-UMA] UMA model loaded.", flush=True)

    def _load_local_predictor(self, checkpoint_path: str) -> Any:
        """
        Load an already-downloaded FAIR-Chem/UMA inference checkpoint.

        Important: pretrained_mlip.get_predict_unit(...) is for registered model
        names such as 'uma-s-1p2'. For local downloaded checkpoints, FAIR-Chem
        exposes load_predict_unit(...) in fairchem.core.units.mlip_unit in current
        releases. Some versions also re-export it from pretrained_mlip, so keep
        a fallback.
        """
        try:
            from fairchem.core.units.mlip_unit import load_predict_unit  # type: ignore
        except Exception:
            try:
                from fairchem.core import pretrained_mlip
                load_predict_unit = pretrained_mlip.load_predict_unit  # type: ignore[attr-defined]
            except Exception as exc:
                raise RuntimeError(
                    "Could not import FAIR-Chem local checkpoint loader. Tried:\n"
                    "  from fairchem.core.units.mlip_unit import load_predict_unit\n"
                    "  fairchem.core.pretrained_mlip.load_predict_unit\n"
                    "Your fairchem-core version may not support direct local checkpoint loading."
                ) from exc

        # fairchem versions differ slightly: some accept path=..., others accept
        # the path as the first positional argument.
        try:
            return load_predict_unit(path=checkpoint_path, device=self.device)
        except TypeError:
            return load_predict_unit(checkpoint_path, device=self.device)

    def predict(self, coords_angstrom: Any) -> Tuple[float, Any]:
        np = get_numpy()
        coords = np.asarray(coords_angstrom, dtype=np.float64)
        atoms = self.Atoms(numbers=self.atomic_numbers, positions=coords)

        if self.task_name.lower() == "omol":
            atoms.info["charge"] = self.charge
            atoms.info["spin"] = self.spin

        atoms.calc = self.calc

        energy_ev = float(atoms.get_potential_energy())
        forces_ev_ang = np.asarray(atoms.get_forces(), dtype=np.float64)

        if forces_ev_ang.shape != coords.shape:
            raise RuntimeError(f"UMA forces shape mismatch: got {forces_ev_ang.shape}, expected {coords.shape}")

        grad_ev_ang = -forces_ev_ang
        return energy_ev * EV_TO_KCAL_MOL, grad_ev_ang * EV_TO_KCAL_MOL


def mace_family_from_model_name(model_name: str) -> str:
    """
    Detect the MACE family from the MODL model field.

    Supported force-compatible families:
      local/generic : local:mace:/path/to/model.model
      off           : MACE-OFF foundation family
      omol          : MACE-OMOL foundation family
      polar         : MACE-POLAR foundation family, zero external field here
      mp            : MACE-MP foundation family
      mh            : MACE-MH multi-head foundation family

    MDP is recognized but rejected later because it is not an energy/force
    calculator compatible with this ABI.
    """
    key = normalize_key(model_name)

    if key in {"mace", "mace-local", "local", "generic"}:
        return "local"
    if "mdp" in key:
        return "mdp"
    if "mh" in key or "multihead" in key or "multi-head" in key:
        return "mh"
    if "polar" in key:
        return "polar"
    if "omol" in key:
        return "omol"
    if "off" in key:
        return "off"
    if key in {"mp", "mace-mp"} or "mace-mp" in key:
        return "mp"

    raise RuntimeError(
        "Could not detect MACE family from MODL model field. "
        "Use model names containing OFF, OMOL, POLAR, MACE-MP, or MACE-MH, "
        "or use local:mace:/path for a generic local checkpoint. "
        f"Received model field {model_name!r}."
    )


def normalize_mace_off_variant(option: Optional[str], model_name: str) -> str:
    """Return the model= key expected by mace_off(...)."""
    raw = str(option if option is not None else model_name).strip()
    key = normalize_key(raw)

    aliases = {
        "small": "small",
        "small-off": "small",
        "mace-off23-small": "small",
        "mace-off24-small": "small",
        "medium": "medium",
        "medium-off": "medium",
        "mace-off23-medium": "medium",
        "mace-off24-medium": "medium",
        "large": "large",
        "large-off": "large",
        "mace-off23-large": "large",
        "mace-off24-large": "large",
    }

    if key in aliases:
        return aliases[key]

    # If no explicit option was given and the model field was only MACE-OFF23/24,
    # choose the documented safe middle option.
    if option is None and ("off23" in key or "off24" in key or key in {"mace-off", "off"}):
        return "medium"

    raise RuntimeError(
        f"Unknown MACE-OFF variant {raw!r}. Use small, medium, or large. "
        "Aliases like small-off/medium-off/large-off are accepted."
    )


def normalize_mace_omol_variant(option: Optional[str], model_name: str) -> str:
    raw = str(option if option is not None else model_name).strip()
    key = normalize_key(raw)

    if "extra-large" in key or "extralarge" in key:
        return "extra_large"
    if "large" in key and "extra" not in key:
        return "large"
    if "medium" in key:
        return "medium"
    if "small" in key:
        return "small"

    # Current common default for OMOL foundation usage.
    return "extra_large"


def normalize_mace_polar_variant(option: Optional[str], model_name: str) -> str:
    raw = str(option if option is not None else model_name).strip()
    key = normalize_key(raw)

    if "polar-1-l" in key or key.endswith("-l") or key in {"large", "l"}:
        return "polar-1-l"
    if "polar-1-m" in key or key.endswith("-m") or key in {"medium", "m"}:
        return "polar-1-m"
    if "polar-1" in key or "polar" in key:
        return "polar-1-m"

    raise RuntimeError(
        f"Unknown MACE-POLAR variant {raw!r}. Use polar-1-m or polar-1-l."
    )


def normalize_mace_mp_variant(option: Optional[str], model_name: str) -> str:
    raw = str(option if option is not None else model_name).strip()
    key = normalize_key(raw)

    if "small" in key:
        return "small"
    if "medium" in key:
        return "medium"
    if "large" in key:
        return "large"

    return "medium"


def normalize_mace_mh_head(option: Optional[str]) -> str:
    """
    Return the exact head string for MACE-MH/mace_mp(..., head=...).

    The input is allowed to be lowercase/uppercase and use '-' or '_'.
    Returned names keep the spelling expected by MACE where needed.
    """
    if option is None or not str(option).strip():
        return "omat_pbe"

    raw = str(option).strip()
    key = raw.lower().replace("-", "_")

    aliases = {
        "omat": "omat_pbe",
        "omat_pbe": "omat_pbe",
        "pbe": "omat_pbe",
        "omol": "omol",
        "spice": "spice_wB97M",
        "spice_wb97m": "spice_wB97M",
        "wb97m": "spice_wB97M",
        "rgd1": "rgd1_b3lyp",
        "rgd1_b3lyp": "rgd1_b3lyp",
        "b3lyp": "rgd1_b3lyp",
        "oc20": "oc20_usemppbe",
        "oc20_usemppbe": "oc20_usemppbe",
        "usemppbe": "oc20_usemppbe",
        "matpes": "matpes_r2scan",
        "matpes_r2scan": "matpes_r2scan",
        "r2scan": "matpes_r2scan",
    }

    if key not in aliases:
        raise RuntimeError(
            f"Unknown MACE-MH head {option!r}. Use one of: "
            "omat_pbe, omol, spice_wB97M, rgd1_b3lyp, "
            "oc20_usemppbe, matpes_r2scan."
        )

    return aliases[key]


def mace_default_variant(family: str, model_name: str) -> str:
    """
    Default variant/head for a MACE family.

    MLPS_MACE_MODEL remains as an escape hatch for package-version drift, but
    only for model variants. For MACE-MH heads, use the MODL option explicitly.
    """
    override = os.environ.get("MLPS_MACE_MODEL", "").strip()
    if override and family != "mh":
        return override

    if family == "off":
        return normalize_mace_off_variant(None, model_name)
    if family == "omol":
        return normalize_mace_omol_variant(None, model_name)
    if family == "polar":
        return normalize_mace_polar_variant(None, model_name)
    if family == "mp":
        return normalize_mace_mp_variant(None, model_name)
    if family == "mh":
        return normalize_mace_mh_head(None)

    return model_name


def mace_zero_external_field() -> list[float]:
    """Current clean ABI uses MACE-POLAR with deterministic zero external field."""
    return [0.0, 0.0, 0.0]


def construct_calculator(factory: Any, context: str, kwargs: dict[str, Any]) -> Any:
    attempts = [dict(kwargs)]

    if "default_dtype" in kwargs:
        reduced = dict(kwargs)
        reduced.pop("default_dtype", None)
        attempts.append(reduced)

    last_exc: Optional[BaseException] = None
    for attempt in attempts:
        try:
            return factory(**attempt)
        except TypeError as exc:
            last_exc = exc

    raise RuntimeError(
        f"Failed to initialize {context} calculator. Last attempted arguments: {attempts[-1]}. "
        "If this is a repo: model, set the optional MODL variant to the exact model key "
        "expected by your installed package."
    ) from last_exc


def construct_local_mace_calculator(path: str, device: str, default_dtype: str) -> Any:
    from mace.calculators import MACECalculator

    attempts = [
        {"model_paths": path, "device": device, "default_dtype": default_dtype},
        {"model_path": path, "device": device, "default_dtype": default_dtype},
    ]

    last_exc: Optional[BaseException] = None
    for kwargs in attempts:
        try:
            return construct_calculator(MACECalculator, "local MACE", kwargs)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "Failed to initialize local MACECalculator from downloaded/local file.\n"
        f"Path: {path}\n"
        "This file may not be compatible with generic MACECalculator in your installed mace-torch."
    ) from last_exc


class MACEPredictor:
    def _resolve_mace_variant(self) -> str:
        if self.family == "off":
            return normalize_mace_off_variant(self.spec.option, self.spec.model)
        if self.family == "omol":
            return normalize_mace_omol_variant(self.spec.option, self.spec.model)
        if self.family == "polar":
            return normalize_mace_polar_variant(self.spec.option, self.spec.model)
        if self.family == "mp":
            return normalize_mace_mp_variant(self.spec.option, self.spec.model)
        if self.family == "mh":
            return normalize_mace_mh_head(self.spec.option)
        return mace_default_variant(self.family, self.spec.model)

    def __init__(self, atomic_numbers: Any, gpu: int, options: ModelOptions):
        np = get_numpy()
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.device = resolve_device(gpu)
        self.spec = parse_modl_spec("mace", options.model_name)

        self.charge = int(options.charge)
        self.spin = int(options.multiplicity)
        self.default_dtype = os.environ.get("MLPS_MACE_DTYPE", "float64")
        self.family = mace_family_from_model_name(self.spec.model)
        self.variant = self._resolve_mace_variant()
        self.needs_external_field = self.family == "polar"
        self.external_field = mace_zero_external_field() if self.needs_external_field else None

        print(
            "[PYTH-MACE] initializing: "
            f"{self.spec.describe()}, family={self.family}, variant={self.variant!r}, "
            f"charge={self.charge}, spin={self.spin}, dtype={self.default_dtype}, "
            f"device={self.device}",
            flush=True,
        )

        import torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for MACE, but torch.cuda.is_available() is False.")

        from ase import Atoms
        self.Atoms = Atoms

        if self.family == "mdp":
            raise RuntimeError(
                "Detected MACE-MDP/MDP-like model from MODL. The current CHARMM ABI "
                "requires scalar energy and dE/dR gradients, not dipole/polarizability outputs."
            )

        if self.spec.is_local:
            assert self.spec.path is not None
            if self.family == "mh":
                from mace.calculators import mace_mp
                self.calc = construct_calculator(
                    mace_mp,
                    "MACE-MH local",
                    {
                        "model": self.spec.path,
                        "head": self.variant,
                        "device": self.device,
                        "default_dtype": self.default_dtype,
                    },
                )
                print(
                    "[PYTH-MACE] using local downloaded MACE-MH file through "
                    f"mace_mp(model=path, head={self.variant!r}).",
                    flush=True,
                )
            else:
                self.calc = construct_local_mace_calculator(
                    path=self.spec.path,
                    device=self.device,
                    default_dtype=self.default_dtype,
                )
                print("[PYTH-MACE] using local downloaded MACE file through MACECalculator.", flush=True)

        elif self.family == "off":
            from mace.calculators import mace_off
            self.calc = construct_calculator(
                mace_off,
                "MACE-OFF repo",
                {"model": self.variant, "device": self.device, "default_dtype": self.default_dtype},
            )
            print("[PYTH-MACE] using MACE-OFF repo/cache calculator.", flush=True)

        elif self.family == "omol":
            from mace.calculators import mace_omol
            self.calc = construct_calculator(
                mace_omol,
                "MACE-OMOL repo",
                {"model": self.variant, "device": self.device, "default_dtype": self.default_dtype},
            )
            print("[PYTH-MACE] using MACE-OMOL repo/cache calculator.", flush=True)

        elif self.family == "polar":
            from mace.calculators import mace_polar
            self.calc = construct_calculator(
                mace_polar,
                "MACE-POLAR repo",
                {"model": self.variant, "device": self.device, "default_dtype": self.default_dtype},
            )
            print(
                "[PYTH-MACE] using MACE-POLAR repo/cache calculator with "
                f"external_field={self.external_field}.",
                flush=True,
            )

        elif self.family == "mp":
            from mace.calculators import mace_mp
            self.calc = construct_calculator(
                mace_mp,
                "MACE-MP repo",
                {"model": self.variant, "device": self.device, "default_dtype": self.default_dtype},
            )
            print("[PYTH-MACE] using MACE-MP repo/cache calculator.", flush=True)

        elif self.family == "mh":
            from mace.calculators import mace_mp
            self.calc = construct_calculator(
                mace_mp,
                "MACE-MH repo",
                {
                    "model": "mh-1",
                    "head": self.variant,
                    "device": self.device,
                    "default_dtype": self.default_dtype,
                },
            )
            print(
                "[PYTH-MACE] using MACE-MH repo/cache calculator through "
                f"mace_mp(model='mh-1', head={self.variant!r}).",
                flush=True,
            )

        else:
            raise RuntimeError(f"Internal error: unsupported MACE family {self.family!r}")

        print("[PYTH-MACE] MACE calculator initialized.", flush=True)

    def predict(self, coords_angstrom: Any) -> Tuple[float, Any]:
        np = get_numpy()
        coords = np.asarray(coords_angstrom, dtype=np.float64)

        atoms = self.Atoms(numbers=self.atomic_numbers, positions=coords)

        # OMOL/POLAR use charge/spin metadata. For calculators that ignore the
        # fields, setting them is harmless.
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin

        if self.needs_external_field:
            atoms.info["external_field"] = list(self.external_field or [0.0, 0.0, 0.0])

        atoms.calc = self.calc

        energy_ev = float(atoms.get_potential_energy())
        forces_ev_ang = np.asarray(atoms.get_forces(), dtype=np.float64)

        if forces_ev_ang.shape != coords.shape:
            raise RuntimeError(f"MACE forces shape mismatch: got {forces_ev_ang.shape}, expected {coords.shape}")

        grad_ev_ang = -forces_ev_ang
        return energy_ev * EV_TO_KCAL_MOL, grad_ev_ang * EV_TO_KCAL_MOL


class TANIPredictor:
    def __init__(self, atomic_numbers: Any, gpu: int, options: ModelOptions):
        np = get_numpy()
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.nml = self.atomic_numbers.size
        self.device = resolve_device(gpu)
        self.spec = parse_modl_spec("tani", options.model_name)
        self.native_unit = os.environ.get("MLPS_TANI_NATIVE_UNIT", "hartree").lower()

        if self.spec.option is not None:
            raise RuntimeError(
                "TANI/TorchANI MODL does not use the optional <option> field. "
                "Use repo:ANI2x or local:tani:/path/to/model.pt."
            )

        print(
            "[PYTH-TANI] initializing: "
            f"{self.spec.describe()}, native_unit={self.native_unit}, device={self.device}",
            flush=True,
        )

        import torch
        self.torch = torch
        self.dtype = torch.float32

        if self.spec.is_repo:
            self.model_key = normalize_key(self.spec.model).replace("-", "")
            self._validate_builtin_elements(self.model_key)
            self.model = self._load_builtin_model(self.model_key)
        else:
            assert self.spec.path is not None
            self.model_key = self.spec.path
            self.model = self._load_local_model(self.spec.path)

        self.model.to(self.device)
        self.model.eval()

        print("[PYTH-TANI] TANI model initialized.", flush=True)

    def _validate_builtin_elements(self, key: str) -> None:
        allowed_by_model = {
            "ani1x": {1, 6, 7, 8},
            "ani1ccx": {1, 6, 7, 8},
            "ani2x": {1, 6, 7, 8, 9, 16, 17},
        }
        allowed = allowed_by_model.get(key.lower().replace("-", ""))
        if allowed is None:
            raise RuntimeError("TorchANI repo models supported: ANI2x, ANI1x, ANI1ccx.")
        bad = sorted({int(z) for z in self.atomic_numbers.tolist()} - allowed)
        if bad:
            raise RuntimeError(
                f"TorchANI built-in {key} does not support atomic numbers {bad}. "
                "Use a local exported model trained for this chemistry."
            )

    def _load_builtin_model(self, key: str) -> Any:
        try:
            import torchani
        except Exception as exc:
            raise RuntimeError(
                "TorchANI repo model requested, but torchani is not importable. "
                "Install it in the Python environment used by CHARMM."
            ) from exc

        name = key.lower().replace("-", "")
        if name == "ani2x":
            cls = torchani.models.ANI2x
        elif name == "ani1x":
            cls = torchani.models.ANI1x
        elif name == "ani1ccx":
            cls = torchani.models.ANI1ccx
        else:
            raise RuntimeError(f"Unsupported TorchANI repo model name: {key!r}")

        try:
            model = cls(periodic_table_index=True)
        except TypeError as exc:
            raise RuntimeError(
                "This TorchANI installation does not accept periodic_table_index=True. "
                "The embedded runner sends atomic numbers. Use a compatible TorchANI version "
                "or export a local TorchScript/full-module model."
            ) from exc

        print(f"[PYTH-TANI] loaded TorchANI repo model {key}.", flush=True)
        return model

    def _load_local_model(self, path: str) -> Any:
        try:
            model = self.torch.jit.load(path, map_location=self.device)
            print("[PYTH-TANI] loaded local model with torch.jit.load.", flush=True)
            return model
        except Exception as jit_exc:
            print(
                "[PYTH-TANI] torch.jit.load failed; trying torch.load full module. "
                f"jit_error={type(jit_exc).__name__}: {jit_exc}",
                flush=True,
            )

        try:
            try:
                model = self.torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                model = self.torch.load(path, map_location=self.device)
        except Exception as load_exc:
            raise RuntimeError("Failed to load local TANI model as TorchScript or full PyTorch module.") from load_exc

        if isinstance(model, dict):
            raise RuntimeError(
                "torch.load returned a dict/checkpoint/state_dict. This generic runner cannot "
                "infer the TANI architecture. Save/export a TorchScript model or full PyTorch module."
            )

        if not hasattr(model, "forward"):
            raise RuntimeError(f"torch.load object is not Module-like: {type(model)}")

        print("[PYTH-TANI] loaded local model with torch.load full module.", flush=True)
        return model

    def _native_to_kcal(self) -> float:
        if self.native_unit in {"hartree", "ha"}:
            return HARTREE_TO_KCAL_MOL
        if self.native_unit in {"ev", "electronvolt"}:
            return EV_TO_KCAL_MOL
        if self.native_unit in {"kcal", "kcal/mol"}:
            return 1.0
        raise ValueError(
            f"Unknown MLPS_TANI_NATIVE_UNIT={self.native_unit!r}. "
            "Allowed: hartree, ev, kcal. CHARMM output is always kcal/mol."
        )

    def _extract_energy(self, output: Any) -> Any:
        if hasattr(output, "energies"):
            return output.energies

        if isinstance(output, (tuple, list)):
            if len(output) < 2:
                raise RuntimeError("TANI tuple/list output length < 2")
            return output[1]

        if isinstance(output, dict):
            for key in ("energy", "energies", "E", "e"):
                if key in output:
                    return output[key]

        if self.torch.is_tensor(output):
            return output

        raise RuntimeError(f"Cannot extract energy from TANI output type {type(output)}")

    def _call_model(self, species: Any, coords: Any) -> Any:
        try:
            return self.model((species, coords))
        except TypeError as first_exc:
            try:
                return self.model(species, coords)
            except Exception as second_exc:
                raise RuntimeError(
                    "TANI model call failed for both signatures: model((species, coords)) "
                    "and model(species, coords). Export the local model with one of these signatures."
                ) from second_exc

    def predict(self, coords_angstrom: Any) -> Tuple[float, Any]:
        np = get_numpy()
        torch = self.torch
        coords_np = np.asarray(coords_angstrom, dtype=np.float32)

        if coords_np.shape != (self.nml, 3):
            raise ValueError(f"coords shape mismatch: got {coords_np.shape}, expected {(self.nml, 3)}")

        species = torch.tensor(
            self.atomic_numbers.reshape(1, -1),
            dtype=torch.long,
            device=self.device,
        )

        coords = torch.tensor(
            coords_np.reshape(1, self.nml, 3),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        output = self._call_model(species, coords)
        energies = self._extract_energy(output)
        e_scalar = energies.sum()

        grad = torch.autograd.grad(
            e_scalar,
            coords,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        scale = self._native_to_kcal()
        energy_native = float(energies.detach().cpu().reshape(-1)[0])
        grad_native = grad.detach().cpu().numpy()[0].astype(np.float64, copy=False)

        return energy_native * scale, grad_native * scale


def build_predictor(options: ModelOptions, atomic_numbers: Any, gpu: int) -> Predictor:
    t = options.model_type.lower().strip()

    if t == "dummy":
        return DummyPredictor(atomic_numbers, gpu, options)
    if t == "uma":
        return UMAPredictor(atomic_numbers, gpu, options)
    if t == "mace":
        return MACEPredictor(atomic_numbers, gpu, options)
    if t in {"tani", "torchani", "ani"}:
        return TANIPredictor(atomic_numbers, gpu, options)

    raise RuntimeError(f"unknown model_type={options.model_type!r}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedded CHARMM MLMM/PYTH backend")
    parser.add_argument("--socket", required=True, help="Unix-domain socket path from type_pyth.cxx")
    parser.add_argument("--gpu", default="-1", help="-1 = CPU, >=0 = physical GPU id")
    args = parser.parse_args()

    socket_path = args.socket

    print("[PYTH] EMBEDDED RUNNER ACTIVE", flush=True)
    print("[PYTH] RUNNING FILE =", __file__, flush=True)
    print("[PYTH] PYTHON =", sys.executable, flush=True)
    print("[PYTH] PYTHON VERSION =", sys.version.split()[0], flush=True)
    print(f"[PYTH] protocol: version={MLPS_VERSION}", flush=True)

    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    print(f"[PYTH] listening on {socket_path}", flush=True)

    conn: Optional[socket.socket] = None

    try:
        conn, _ = server.accept()
        print("[PYTH] C++ connected.", flush=True)

        try:
            state = recv_setup(conn)
            predictor = build_predictor(
                options=state.options,
                atomic_numbers=state.ml_zid,
                gpu=state.gpu,
            )
        except Exception:
            msg = traceback.format_exc()
            print(msg, flush=True)
            send_status_error(conn, msg)
            return 2

        send_i32(conn, STATUS_OK)
        serve(conn, predictor, state)

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        try:
            server.close()
        except Exception:
            pass

        try:
            os.unlink(socket_path)
        except Exception:
            pass

    print("[PYTH] shutdown complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)


)PYMLMMX";

struct PySocketState {
    bool active = false;

    int use_gpu = -1;
    int nml = 0;
    int natoms = 0;

    int charge = 0;
    int multiplicity = 1;

    pid_t child_pid = -1;
    int sock_fd = -1;

    std::string runf_file;   // generated temporary embedded Python file
    bool runf_is_temp = false;
    std::string socket_path;

    std::string model_type;
    std::string model_name;  // local path sent by MODL

    std::vector<int32_t> ml_idx;     // CHARMM 1-based atom indices, metadata only
    std::vector<int32_t> ml_zid;     // atomic numbers
    std::vector<int32_t> ml_maskid;  // natom-sized ML mask, 1/0
};

PySocketState S;

int getenv_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v || x <= 0 || x > INT_MAX) return fallback;
    return static_cast<int>(x);
}

std::string getenv_string(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return std::string(fallback);
    return std::string(v);
}

std::string cstr_or_empty(const char* s) {
    if (!s) return std::string();
    return std::string(s);
}

[[noreturn]] void die(const char* msg) {
    std::fprintf(stderr, "[PYTH] FATAL: %s\n", msg);
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] void die_errno(const char* msg) {
    std::fprintf(stderr, "[PYTH] FATAL: %s: %s\n", msg, std::strerror(errno));
    std::fflush(stderr);
    std::abort();
}

bool poll_fd(int fd, short events, int timeout_ms, const char* what) {
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = events;

    while (true) {
        int rc = ::poll(&pfd, 1, timeout_ms);
        if (rc > 0) {
            // If requested data is available, allow the caller to read it even
            // when POLLHUP is also set. This lets C++ receive the Python setup
            // traceback before the Python process exits.
            if (pfd.revents & events) {
                return true;
            }
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                std::fprintf(stderr, "[PYTH] socket error while waiting for %s; revents=%d\n",
                             what, static_cast<int>(pfd.revents));
                return false;
            }
            return false;
        }
        if (rc == 0) {
            std::fprintf(stderr, "[PYTH] timeout while waiting for %s\n", what);
            return false;
        }
        if (errno == EINTR) continue;
        std::fprintf(stderr, "[PYTH] poll failed while waiting for %s: %s\n",
                     what, std::strerror(errno));
        return false;
    }
}

bool write_all_fd(int fd, const void* buf, size_t nbytes, const char* what) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;

    while (done < nbytes) {
        ssize_t rc = ::write(fd, p + done, nbytes - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[PYTH] write failed for %s: %s\n", what, std::strerror(errno));
            return false;
        }
        if (rc == 0) return false;
        done += static_cast<size_t>(rc);
    }
    return true;
}

bool write_all(int fd, const void* buf, size_t nbytes, const char* what) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;

    const int timeout_ms = getenv_int("MLPS_PY_IO_TIMEOUT_MS", 600000);

    while (done < nbytes) {
        if (!poll_fd(fd, POLLOUT, timeout_ms, what)) return false;

        ssize_t rc = ::write(fd, p + done, nbytes - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[PYTH] write failed for %s: %s\n", what, std::strerror(errno));
            return false;
        }
        if (rc == 0) return false;
        done += static_cast<size_t>(rc);
    }
    return true;
}

bool read_all(int fd, void* buf, size_t nbytes, const char* what) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;

    const int timeout_ms = getenv_int("MLPS_PY_IO_TIMEOUT_MS", 600000);

    while (done < nbytes) {
        if (!poll_fd(fd, POLLIN, timeout_ms, what)) return false;

        ssize_t rc = ::read(fd, p + done, nbytes - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[PYTH] read failed for %s: %s\n", what, std::strerror(errno));
            return false;
        }
        if (rc == 0) {
            std::fprintf(stderr, "[PYTH] EOF while reading %s\n", what);
            return false;
        }
        done += static_cast<size_t>(rc);
    }
    return true;
}

template <typename T>
void send_scalar(int fd, T v, const char* what) {
    if (!write_all(fd, &v, sizeof(T), what)) die("socket write failed");
}

template <typename T>
T recv_scalar(int fd, const char* what) {
    T v{};
    if (!read_all(fd, &v, sizeof(T), what)) die("socket read failed");
    return v;
}

void send_bytes(int fd, const void* ptr, size_t nbytes, const char* what) {
    if (nbytes == 0) return;
    if (!write_all(fd, ptr, nbytes, what)) die("socket write failed");
}

void recv_bytes(int fd, void* ptr, size_t nbytes, const char* what) {
    if (nbytes == 0) return;
    if (!read_all(fd, ptr, nbytes, what)) die("socket read failed");
}

void send_string(int fd, const std::string& s, const char* what) {
    if (s.size() > static_cast<size_t>(INT32_MAX)) die("string too long");
    int32_t n = static_cast<int32_t>(s.size());
    send_scalar<int32_t>(fd, n, what);
    if (n > 0) send_bytes(fd, s.data(), static_cast<size_t>(n), what);
}

std::string recv_string(int fd, const char* what) {
    int32_t n = recv_scalar<int32_t>(fd, what);
    if (n < 0) die("received negative string length");
    std::string s(static_cast<size_t>(n), '\0');
    if (n > 0) recv_bytes(fd, &s[0], static_cast<size_t>(n), what);
    return s;
}

std::string dirname_of(const std::string& path) {
    const std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

std::string write_embedded_python_file() {
    const char* dirs[] = {"/tmp", "."};

    for (const char* dir : dirs) {
        std::string templ = std::string(dir) + "/charmm_mlmm_embedded_" +
                            std::to_string(static_cast<long>(::getpid())) + "_XXXXXX";

        std::vector<char> pathbuf(templ.begin(), templ.end());
        pathbuf.push_back('\0');

        int fd = ::mkstemp(pathbuf.data());
        if (fd < 0) continue;

        const size_t nbytes = std::strlen(EMBEDDED_MLMM_RUNNER);
        bool ok = write_all_fd(fd, EMBEDDED_MLMM_RUNNER, nbytes, "embedded Python RUNF file");
        if (::close(fd) != 0) ok = false;

        if (!ok) {
            ::unlink(pathbuf.data());
            continue;
        }

        (void)::chmod(pathbuf.data(), S_IRUSR | S_IWUSR);
        return std::string(pathbuf.data());
    }

    die("failed to write embedded Python RUNF file to /tmp or current directory");
}

std::string make_socket_path() {
    // Use the same directory as the generated Python file. This keeps the
    // fallback coherent when /tmp is not usable and the Python file is written
    // to the current directory.
    const std::string dir = dirname_of(S.runf_file);
    return dir + "/charmm_mlmm_pyth_" +
           std::to_string(static_cast<long>(::getpid())) + ".sock";
}

void close_socket() {
    if (S.sock_fd >= 0) {
        ::close(S.sock_fd);
        S.sock_fd = -1;
    }
}

void cleanup_child(bool send_stop) {
    S.active = false;

    if (S.sock_fd >= 0 && send_stop) {
        const int32_t magic = MLPS_MAGIC;
        const int32_t version = MLPS_VERSION;
        const int32_t cmd = CMD_STOP;
        (void)write_all(S.sock_fd, &magic, sizeof(magic), "STOP magic");
        (void)write_all(S.sock_fd, &version, sizeof(version), "STOP version");
        (void)write_all(S.sock_fd, &cmd, sizeof(cmd), "STOP cmd");
    }

    close_socket();

    if (!S.socket_path.empty()) {
        ::unlink(S.socket_path.c_str());
        S.socket_path.clear();
    }

    if (S.child_pid > 0) {
        int status = 0;
        pid_t rc = ::waitpid(S.child_pid, &status, WNOHANG);
        if (rc == 0) {
            ::kill(S.child_pid, SIGTERM);

            for (int i = 0; i < 20; ++i) {
                rc = ::waitpid(S.child_pid, &status, WNOHANG);
                if (rc == S.child_pid) break;
                ::usleep(50000);
            }

            if (rc == 0) {
                ::kill(S.child_pid, SIGKILL);
                (void)::waitpid(S.child_pid, &status, 0);
            }
        }
        S.child_pid = -1;
    }

    if (S.runf_is_temp && !S.runf_file.empty()) {
        ::unlink(S.runf_file.c_str());
        S.runf_file.clear();
        S.runf_is_temp = false;
    }
}

void launch_python_server() {
    S.runf_file = write_embedded_python_file();
    S.runf_is_temp = true;

    S.socket_path = make_socket_path();
    ::unlink(S.socket_path.c_str());

    std::string python_exe = getenv_string("MLPS_PYTHON", "python3");
    std::string gpu_arg = std::to_string(S.use_gpu);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(python_exe.c_str()));
    argv.push_back(const_cast<char*>(S.runf_file.c_str()));
    argv.push_back(const_cast<char*>("--socket"));
    argv.push_back(const_cast<char*>(S.socket_path.c_str()));
    argv.push_back(const_cast<char*>("--gpu"));
    argv.push_back(const_cast<char*>(gpu_arg.c_str()));
    argv.push_back(nullptr);

    std::fprintf(stderr, "[PYTH] embedded RUNF written to: %s\n", S.runf_file.c_str());
    std::fprintf(stderr, "[PYTH] launching Python executable: %s\n", python_exe.c_str());

    pid_t pid = -1;
    int rc = ::posix_spawnp(&pid, python_exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        errno = rc;
        die_errno("posix_spawnp python failed");
    }

    S.child_pid = pid;
}

void connect_with_retry() {
    const int startup_timeout_sec = getenv_int("MLPS_PY_STARTUP_TIMEOUT_SEC", 300);
    const int sleep_us = getenv_int("MLPS_PY_STARTUP_SLEEP_US", 100000);
    const int max_tries = std::max(1, (startup_timeout_sec * 1000000) / sleep_us);

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) die_errno("socket(AF_UNIX) failed");

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (S.socket_path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        die("socket path is too long");
    }
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", S.socket_path.c_str());

    for (int t = 0; t < max_tries; ++t) {
        int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc == 0) {
            S.sock_fd = fd;
            return;
        }

        if (S.child_pid > 0) {
            int status = 0;
            pid_t w = ::waitpid(S.child_pid, &status, WNOHANG);
            if (w == S.child_pid) {
                std::fprintf(stderr,
                             "[PYTH] Python server exited before socket connection. status=%d\n",
                             status);
                ::close(fd);
                S.child_pid = -1;
                die("Python server failed during startup");
            }
        }

        ::usleep(static_cast<useconds_t>(sleep_us));
    }

    ::close(fd);
    die("timed out connecting to Python socket server");
}

void send_setup_message() {
    const int32_t magic = MLPS_MAGIC;
    const int32_t version = MLPS_VERSION;
    const int32_t cmd = CMD_SETUP;
    const int32_t natoms = static_cast<int32_t>(S.natoms);
    const int32_t nml = static_cast<int32_t>(S.nml);
    const int32_t gpu = static_cast<int32_t>(S.use_gpu);
    const int32_t charge = static_cast<int32_t>(S.charge);
    const int32_t multiplicity = static_cast<int32_t>(S.multiplicity);

    send_scalar<int32_t>(S.sock_fd, magic, "SETUP magic");
    send_scalar<int32_t>(S.sock_fd, version, "SETUP version");
    send_scalar<int32_t>(S.sock_fd, cmd, "SETUP cmd");
    send_scalar<int32_t>(S.sock_fd, natoms, "SETUP natoms");
    send_scalar<int32_t>(S.sock_fd, nml, "SETUP nml");
    send_scalar<int32_t>(S.sock_fd, gpu, "SETUP gpu");

    send_string(S.sock_fd, S.model_type, "SETUP model_type");
    send_string(S.sock_fd, S.model_name, "SETUP model_name");

    send_scalar<int32_t>(S.sock_fd, charge, "SETUP charge");
    send_scalar<int32_t>(S.sock_fd, multiplicity, "SETUP multiplicity");

    send_bytes(S.sock_fd, S.ml_idx.data(),    static_cast<size_t>(S.nml)    * sizeof(int32_t), "SETUP ml_idx");
    send_bytes(S.sock_fd, S.ml_zid.data(),    static_cast<size_t>(S.nml)    * sizeof(int32_t), "SETUP ml_zid");
    send_bytes(S.sock_fd, S.ml_maskid.data(), static_cast<size_t>(S.natoms) * sizeof(int32_t), "SETUP ml_maskid");

    int32_t status = recv_scalar<int32_t>(S.sock_fd, "SETUP status");
    if (status != STATUS_OK) {
        std::string msg = recv_string(S.sock_fd, "SETUP error message");
        std::fprintf(stderr, "[PYTH] Python SETUP failed with status=%d\n%s\n",
                     static_cast<int>(status), msg.c_str());
        die("Python setup failed");
    }
}

} // namespace

extern "C" void charmm_pyth_internal_setup(
    int pyth_in_use,
    int pyth_use_gpu,
    const char* model_type,
    const char* model_name,
    int pyth_charge,
    int pyth_multiplicity,
    int pyth_pt_nml,
    const int* pyth_in_mlidx,
    const int* pyth_in_mlZid,
    const int* pyth_in_mlmaskid,
    int pyth_in_natoms,
    int* pyth_out_setup_err)
{

    // Assume failure until setup completes successfully.
    if (pyth_out_setup_err) *pyth_out_setup_err = 1;

    if (pyth_in_use != 1) {
        std::fprintf(stderr, "[PYTH] setup called with pyth_in_use=%d; disabling backend.\n",
                     pyth_in_use);
        cleanup_child(false);
        return;
    }
    if (pyth_pt_nml <= 0) {
        std::fprintf(stderr, "[PYTH] setup failed: requires nml > 0\n");
        cleanup_child(false);
        return;
    }
    if (pyth_in_natoms <= 0) {
        std::fprintf(stderr, "[PYTH] setup failed: requires natoms > 0\n");
        cleanup_child(false);
        return;
    }
    if (!pyth_in_mlidx || !pyth_in_mlZid || !pyth_in_mlmaskid) {
        std::fprintf(stderr, "[PYTH] setup failed: received null arrays\n");
        cleanup_child(false);
        return;
    }
    if (!model_type || std::strlen(model_type) == 0) {
        std::fprintf(stderr, "[PYTH] setup failed: requires model_type\n");
        cleanup_child(false);
        return;
    }


    try {

        const std::string model_type_s = cstr_or_empty(model_type);
        const std::string model_name_s = cstr_or_empty(model_name);

        if (model_name_s.empty() && model_type_s != "dummy" && model_type_s != "DUMMY") {
            std::fprintf(stderr, "[PYTH] setup failed: requires MODL/model_name for non-DUMMY models\n");
            cleanup_child(false);
            return;
        }

        cleanup_child(true);

        S.active = false;
        S.use_gpu = pyth_use_gpu;
        S.nml = pyth_pt_nml;
        S.natoms = pyth_in_natoms;
        S.model_type = model_type_s;
        S.model_name = model_name_s;

        S.charge = pyth_charge;
        S.multiplicity = (pyth_multiplicity > 0) ? pyth_multiplicity : 1;

        S.ml_idx.assign(pyth_in_mlidx, pyth_in_mlidx + S.nml);
        S.ml_zid.assign(pyth_in_mlZid, pyth_in_mlZid + S.nml);
        S.ml_maskid.assign(pyth_in_mlmaskid, pyth_in_mlmaskid + S.natoms);

        std::fprintf(stderr,
                 "[PYTH] launching embedded Python server: TYPE=%s MODL=%s CHARGE=%d MULT=%d NML=%d NATOM=%d GPU=%d\n",
                 S.model_type.c_str(),
                 S.model_name.c_str(),
                 S.charge,
                 S.multiplicity,
                 S.nml,
                 S.natoms,
                 S.use_gpu);

        launch_python_server();
        connect_with_retry();
        send_setup_message();

        S.active = true;

        // Setup completed successfully.
        if (pyth_out_setup_err) *pyth_out_setup_err = 0;

        std::fprintf(stderr, "[PYTH] Python socket backend ready: %s\n", S.socket_path.c_str());
    }
        catch (const std::exception& e) {
        std::fprintf(stderr, "[setup] setup failed: %s\n", e.what());
        S.active = false;
        cleanup_child(true);
        return;
    }
    catch (...) {
        std::fprintf(stderr, "[setup] setup failed: unknown exception\n");
        S.active = false;
        cleanup_child(true);
        return;
    }
}

extern "C" void charmm_pyth_internal_force(
    double* E_pyth_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    if (!E_pyth_c || !ml_x_c || !ml_y_c || !ml_z_c ||
        !ml_dx_c || !ml_dy_c || !ml_dz_c) {
        die("PYTH force received null pointer");
    }

    *E_pyth_c = 0.0;

    if (!S.active || S.sock_fd < 0) {
        die("PYTH force called before active setup");
    }

    const int32_t magic = MLPS_MAGIC;
    const int32_t version = MLPS_VERSION;
    const int32_t cmd = CMD_FORCE;
    const int32_t nml = static_cast<int32_t>(S.nml);

    send_scalar<int32_t>(S.sock_fd, magic, "FORCE magic");
    send_scalar<int32_t>(S.sock_fd, version, "FORCE version");
    send_scalar<int32_t>(S.sock_fd, cmd, "FORCE cmd");
    send_scalar<int32_t>(S.sock_fd, nml, "FORCE nml");

    send_bytes(S.sock_fd, ml_x_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_x");
    send_bytes(S.sock_fd, ml_y_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_y");
    send_bytes(S.sock_fd, ml_z_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_z");

    int32_t status = recv_scalar<int32_t>(S.sock_fd, "FORCE status");
    if (status != STATUS_OK) {
        std::string msg = recv_string(S.sock_fd, "FORCE error message");
        std::fprintf(stderr, "[PYTH] Python FORCE failed with status=%d\n%s\n",
                     static_cast<int>(status), msg.c_str());
        die("Python force failed");
    }

    recv_bytes(S.sock_fd, E_pyth_c, sizeof(double), "FORCE energy");
    recv_bytes(S.sock_fd, ml_dx_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dx");
    recv_bytes(S.sock_fd, ml_dy_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dy");
    recv_bytes(S.sock_fd, ml_dz_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dz");
}


#endif // defined(KEY_MLMM) && KEY_MLMM == 1