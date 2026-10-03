#!/usr/bin/env python3
"""
uma_run.py

RUNF Python backend for CHARMM MLPS PYTH mode.

The file is intentionally split into two regions:

1) CHARMM / PYTH SOCKET REGION
   - Stable protocol layer.
   - Must match type_pyth.cxx.
   - Normally DO NOT modify this region when switching ML models.

2) MODEL BACKEND REGION
   - Swappable predictor layer.
   - Modify/add predictor classes here to use UMA, MACE, ANI, custom PyTorch,
     external QM engines, etc.
   - The required predictor API is:

       predictor = SomePredictor(atomic_numbers, gpu, spec)
       energy_kcal, grad_kcal_ang = predictor.predict(coords_angstrom)

   - energy_kcal must be a scalar in kcal/mol.
   - grad_kcal_ang must be shape (nml, 3), containing dE/dR in
     kcal/mol/Angstrom, NOT forces.

Current CHARMM protocol expected from type_pyth.cxx:

SETUP from C++:
    magic      : 4 bytes, b"MLPS"
    version    : int32, expected 1
    cmd        : int32, expected 1
    natoms     : int32
    nml        : int32
    gpu        : int32, -1 = CPU, >=0 = physical GPU id
    spec_len   : int32
    spec       : char[spec_len]
    ml_idx     : int32[nml]
    ml_zid     : int32[nml]
    ml_maskid  : int32[natoms]

FORCE from C++:
    magic      : 4 bytes, b"MLPS"
    version    : int32, expected 1
    cmd        : int32, expected 2
    nml        : int32
    x          : float64[nml], Angstrom
    y          : float64[nml], Angstrom
    z          : float64[nml], Angstrom

FORCE reply to C++:
    status     : int32, 0 = OK, nonzero = error
    energy     : float64, kcal/mol
    dx         : float64[nml], dE/dx in kcal/mol/Angstrom
    dy         : float64[nml]
    dz         : float64[nml]

STOP from C++:
    magic      : 4 bytes, b"MLPS"
    version    : int32, expected 1
    cmd        : int32, expected 3

Important convention:
    CHARMM dx/dy/dz are gradients dE/dR, not forces.
    ASE/UMA returns forces F = -dE/dR.
    Therefore UMA gradients returned to CHARMM are -forces.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np


# =============================================================================
# REGION 1: CHARMM / PYTH SOCKET PROTOCOL
#
# This region is the CHARMM-facing interface. It should remain stable and should
# not need modification when switching UMA to another backend.
# =============================================================================

MAGIC = b"MLPS"
MLPS_VERSION = 1

CMD_SETUP = 1
CMD_FORCE = 2
CMD_STOP = 3

STATUS_OK = 0
STATUS_ERR = 1

# Conversion used by UMA/ASE and many atomistic ML models.
EV_TO_KCAL_MOL = 23.060548867


@dataclass
class SetupState:
    """Information sent once by CHARMM/type_pyth.cxx during PYTH setup."""

    natoms: int
    nml: int
    gpu: int
    spec_from_cpp: str
    ml_idx: np.ndarray
    ml_zid: np.ndarray
    ml_maskid: np.ndarray


def read_exact(conn: socket.socket, nbytes: int) -> bytes:
    """Read exactly nbytes from a blocking socket or raise ConnectionError."""
    chunks = []
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


def read_f64_array(conn: socket.socket, n: int) -> np.ndarray:
    if n < 0:
        raise RuntimeError(f"negative float64 array length: {n}")
    buf = read_exact(conn, 8 * n)
    return np.frombuffer(buf, dtype="<f8").astype(np.float64, copy=True)


def read_i32_array(conn: socket.socket, n: int) -> np.ndarray:
    if n < 0:
        raise RuntimeError(f"negative int32 array length: {n}")
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


def send_f64_array(conn: socket.socket, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype="<f8")
    conn.sendall(arr.tobytes(order="C"))


def read_header(conn: socket.socket, context: str) -> int:
    """Read magic/version/cmd header used by all CHARMM PYTH messages."""
    magic = read_exact(conn, 4)
    if magic != MAGIC:
        raise RuntimeError(f"bad {context} magic: got {magic!r}, expected {MAGIC!r}")

    version = read_i32(conn)
    if version != MLPS_VERSION:
        raise RuntimeError(
            f"bad {context} protocol version: got {version}, expected {MLPS_VERSION}"
        )

    cmd = read_i32(conn)
    return cmd


def recv_setup(conn: socket.socket) -> SetupState:
    """
    Receive the one-time SETUP message from type_pyth.cxx.

    Do not modify this unless the C++ protocol changes.
    """
    cmd = read_header(conn, "setup")

    if cmd != CMD_SETUP:
        raise RuntimeError(f"expected setup cmd={CMD_SETUP}, got {cmd}")

    natoms = read_i32(conn)
    nml = read_i32(conn)
    gpu = read_i32(conn)
    spec_from_cpp = read_string(conn)

    if natoms <= 0:
        raise RuntimeError(f"invalid natoms in setup: {natoms}")
    if nml <= 0:
        raise RuntimeError(f"invalid nml in setup: {nml}")

    ml_idx = read_i32_array(conn, nml)
    ml_zid = read_i32_array(conn, nml)
    ml_maskid = read_i32_array(conn, natoms)

    print(
        "[PYTH] SETUP received: "
        f"version={MLPS_VERSION}, natoms={natoms}, nml={nml}, gpu={gpu}, "
        f"spec_from_cpp='{spec_from_cpp}', Z={ml_zid.tolist()}",
        flush=True,
    )

    send_i32(conn, STATUS_OK)

    return SetupState(
        natoms=natoms,
        nml=nml,
        gpu=gpu,
        spec_from_cpp=spec_from_cpp,
        ml_idx=ml_idx,
        ml_zid=ml_zid,
        ml_maskid=ml_maskid,
    )


def validate_force_result(energy: Any, grad: Any, nml: int) -> Tuple[float, np.ndarray]:
    """Validate model output before sending it back to CHARMM."""
    energy_f = float(energy)
    grad_a = np.asarray(grad, dtype=np.float64)

    if grad_a.shape != (nml, 3):
        raise RuntimeError(f"gradient shape mismatch: got {grad_a.shape}, expected {(nml, 3)}")

    if not np.isfinite(energy_f):
        raise RuntimeError(f"non-finite energy from predictor: {energy_f}")

    if not np.all(np.isfinite(grad_a)):
        raise RuntimeError("non-finite gradient from predictor")

    return energy_f, grad_a


def send_force_result(conn: socket.socket, energy: float, grad: np.ndarray, nml: int) -> None:
    """
    Send successful force result to CHARMM.

    C++ expects the payload only when status is STATUS_OK:
        status, energy, dx[nml], dy[nml], dz[nml]
    """
    energy, grad = validate_force_result(energy, grad, nml)

    send_i32(conn, STATUS_OK)
    send_f64(conn, energy)
    send_f64_array(conn, grad[:, 0])
    send_f64_array(conn, grad[:, 1])
    send_f64_array(conn, grad[:, 2])


def send_force_error(conn: socket.socket) -> None:
    """
    Tell CHARMM that the backend failed.

    Current type_pyth.cxx should treat nonzero status as fatal and should not
    expect energy/gradient arrays after an error status.
    """
    send_i32(conn, STATUS_ERR)


def serve(conn: socket.socket, predictor: "Predictor", state: SetupState, spec: Dict[str, Any]) -> None:
    """
    Main command loop.

    This is still CHARMM protocol code. It is model-agnostic except for calling
    predictor.predict(coords).
    """
    nforce = 0
    print_every = int(spec.get("print_every", 1))
    max_abs_grad_warn = float(spec.get("max_abs_grad_warn", 1.0e4))

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
        coords = np.stack((x, y, z), axis=1)

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
            traceback.print_exc()
            try:
                send_force_error(conn)
            except OSError:
                pass
            return


# =============================================================================
# REGION 2: SPEC LOADING
#
# This region is mostly stable. You can add model-specific keys to your SPEC
# file without changing this parser.
# =============================================================================

def parse_simple_value(text: str) -> Any:
    low = text.strip().lower()

    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text.strip()


def load_spec(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a SPEC file.

    Supported formats:
        .json       JSON dictionary
        .yaml/.yml  YAML dictionary, if PyYAML is installed
        otherwise   simple key=value or key: value lines

    Example key=value SPEC:
        backend=uma
        model_name=uma-s-1p1
        task_name=omol
        charge=0
        spin=1
        print_every=1
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SPEC file not found: {path}")

    text = p.read_text().strip()
    if not text:
        return {}

    suffix = p.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
        return {} if data is None else dict(data)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "SPEC file appears to be YAML, but PyYAML is not installed. "
                "Install pyyaml or use JSON/key=value SPEC."
            ) from exc

        data = yaml.safe_load(text)
        return {} if data is None else dict(data)

    spec: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        spec[key.strip()] = parse_simple_value(value.strip())

    return spec


# =============================================================================
# REGION 3: MODEL BACKEND API
#
# This is the region you modify when replacing UMA with another model.
# The CHARMM/socket layer above should not need changes.
# =============================================================================

class Predictor(Protocol):
    """Minimal backend API required by the CHARMM socket server."""

    def predict(self, coords_angstrom: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Parameters
        ----------
        coords_angstrom:
            Array of shape (nml, 3), in Angstrom.

        Returns
        -------
        energy_kcal:
            Scalar energy in kcal/mol.

        grad_kcal_ang:
            Array of shape (nml, 3), dE/dR in kcal/mol/Angstrom.
            This must be gradients, not forces.
        """
        ...


class DummyPredictor:
    """
    Protocol/debug backend.

    This backend does not use external ML packages. Use it first to verify that:
        - CHARMM calls PYTH force.
        - socket protocol is correct.
        - returned gradients are scattered back to the correct atoms.
        - force sign convention is right.

    E = 0.5 * k * sum_i |r_i - r0_i|^2
    grad = k * (r_i - r0_i)

    Units are already kcal/mol and kcal/mol/Angstrom.
    """

    def __init__(self, atomic_numbers: np.ndarray, gpu: int, spec: Dict[str, Any]):
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.r0: Optional[np.ndarray] = None
        self.k = float(spec.get("dummy_k", 1.0))

        print(
            f"[PYTH-DUMMY] initialized: nml={self.atomic_numbers.size}, "
            f"k={self.k} kcal/mol/A^2",
            flush=True,
        )

    def predict(self, coords_angstrom: np.ndarray) -> Tuple[float, np.ndarray]:
        coords = np.asarray(coords_angstrom, dtype=np.float64)

        if coords.shape != (self.atomic_numbers.size, 3):
            raise ValueError(
                f"coords shape mismatch: got {coords.shape}, "
                f"expected {(self.atomic_numbers.size, 3)}"
            )

        if self.r0 is None:
            self.r0 = coords.copy()

        dr = coords - self.r0
        energy = 0.5 * self.k * float(np.sum(dr * dr))
        grad = self.k * dr

        return energy, grad


class UMAPredictor:
    """
    UMA backend through FAIR-Chem + ASE calculator.

    This is the only UMA-specific region in the file.

    SPEC keys used by this class:
        backend=uma
        model_name=uma-s-1p1
        task_name=omol
        charge=0
        spin=1
        device=cpu/cuda      optional override; normally omit

    UMA/ASE returns:
        energy: eV
        forces: eV/Angstrom

    This class returns:
        energy: kcal/mol
        grad  : kcal/mol/Angstrom, dE/dR = -forces
    """

    def __init__(self, atomic_numbers: np.ndarray, gpu: int, spec: Dict[str, Any]):
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)

        self.model_name = str(spec.get("model_name", "uma-s-1p1"))
        self.task_name = str(spec.get("task_name", "omol"))

        # For OMOL, FAIR-Chem/UMA expects charge and spin/multiplicity in
        # atoms.info. Here spin means multiplicity:
        #   singlet=1, doublet=2, triplet=3, ...
        self.charge = int(spec.get("charge", 0))
        self.spin = int(spec.get("spin", spec.get("multiplicity", 1)))

        # GPU convention inherited from CHARMM/type_pyth.cxx:
        #   gpu < 0  => CPU
        #   gpu >= 0 => physical GPU id
        #
        # If gpu >= 0, expose only that physical GPU and use "cuda" internally.
        # Keep this before importing torch/fairchem.
        requested_device = spec.get("device", None)

        if int(gpu) >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
            default_device = "cuda"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            default_device = "cpu"

        self.device = str(requested_device) if requested_device is not None else default_device

        print(
            "[PYTH-UMA] initializing: "
            f"model={self.model_name}, task={self.task_name}, "
            f"charge={self.charge}, spin={self.spin}, "
            f"gpu={gpu}, CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}', "
            f"device={self.device}",
            flush=True,
        )

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Failed to import torch in UMA backend.") from exc

        print(
            "[PYTH-UMA] torch: "
            f"version={getattr(torch, '__version__', '<unknown>')}, "
            f"cuda_available={torch.cuda.is_available()}",
            flush=True,
        )

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested for UMA, but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver, PyTorch CUDA wheel, and GPU id."
            )

        if self.device.startswith("cuda"):
            try:
                print(
                    f"[PYTH-UMA] CUDA device name: {torch.cuda.get_device_name(0)}",
                    flush=True,
                )
            except Exception:
                pass

        try:
            from ase import Atoms
        except Exception as exc:
            raise RuntimeError("Failed to import ASE. Install with: pip install ase") from exc

        try:
            from fairchem.core import pretrained_mlip
        except Exception as exc:
            raise RuntimeError(
                "Failed to import fairchem.core.pretrained_mlip. "
                "Install FAIR-Chem, usually with: pip install fairchem-core"
            ) from exc

        # FAIR-Chem calculator import path changed across releases.
        try:
            from fairchem.core import FAIRChemCalculator  # type: ignore
        except Exception:
            try:
                from fairchem.core.common.relaxation.ase_utils import FAIRChemCalculator  # type: ignore
            except Exception as exc:
                raise RuntimeError("Failed to import FAIRChemCalculator from FAIR-Chem.") from exc

        self.Atoms = Atoms

        try:
            self.predictor = pretrained_mlip.get_predict_unit(self.model_name, device=self.device)
            self.calc = FAIRChemCalculator(self.predictor, task_name=self.task_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load UMA model '{self.model_name}' with task '{self.task_name}' "
                f"on device '{self.device}'. Check FAIR-Chem install and Hugging Face access."
            ) from exc

        print("[PYTH-UMA] UMA model loaded and ASE calculator initialized.", flush=True)

    def predict(self, coords_angstrom: np.ndarray) -> Tuple[float, np.ndarray]:
        coords = np.asarray(coords_angstrom, dtype=np.float64)

        if coords.shape != (self.atomic_numbers.size, 3):
            raise ValueError(
                f"coords shape mismatch: got {coords.shape}, "
                f"expected {(self.atomic_numbers.size, 3)}"
            )

        atoms = self.Atoms(numbers=self.atomic_numbers, positions=coords)

        # This is essential for OMOL. Without these, FAIR-Chem warns and defaults
        # to charge=0, spin=1.
        if self.task_name.lower() == "omol":
            atoms.info["charge"] = self.charge
            atoms.info["spin"] = self.spin

        atoms.calc = self.calc

        energy_ev = float(atoms.get_potential_energy())
        forces_ev_ang = np.asarray(atoms.get_forces(), dtype=np.float64)

        if forces_ev_ang.shape != coords.shape:
            raise RuntimeError(
                f"UMA force shape mismatch: got {forces_ev_ang.shape}, expected {coords.shape}"
            )

        # ASE returns forces F = -dE/dR.
        # CHARMM wants gradients dE/dR.
        grad_ev_ang = -forces_ev_ang

        energy_kcal = energy_ev * EV_TO_KCAL_MOL
        grad_kcal_ang = grad_ev_ang * EV_TO_KCAL_MOL

        return energy_kcal, grad_kcal_ang


def build_predictor(
    backend: str,
    atomic_numbers: np.ndarray,
    gpu: int,
    spec: Dict[str, Any],
) -> Predictor:
    """
    Model backend factory.

    To add a new backend, add a new Predictor class above and register it here:

        elif backend == "my_model":
            return MyModelPredictor(atomic_numbers, gpu, spec)

    Do not modify the CHARMM socket/protocol region for new models.
    """
    backend = backend.lower().strip()

    if backend == "dummy":
        return DummyPredictor(atomic_numbers, gpu, spec)

    if backend == "uma":
        return UMAPredictor(atomic_numbers, gpu, spec)

    raise RuntimeError(f"unknown backend: {backend}")


# =============================================================================
# REGION 4: MAIN PROGRAM
#
# This wires the stable CHARMM socket layer to the swappable model backend.
# Usually you do not modify this unless command-line arguments change.
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="CHARMM MLPS PYTH RUNF backend")
    parser.add_argument("--socket", required=True, help="Unix-domain socket path from type_pyth.cxx")
    parser.add_argument("--gpu", default="-1", help="-1 = CPU, >=0 = physical GPU id")
    parser.add_argument("--spec", default="", help="Optional JSON/YAML/key=value SPEC file")
    parser.add_argument(
        "--backend",
        default="uma",
        choices=["uma", "dummy"],
        help="Backend to use when SPEC does not define backend.",
    )
    args = parser.parse_args()

    socket_path = args.socket
    gpu_from_argv = int(args.gpu)

    print("[PYTH] RUNNING FILE =", __file__, flush=True)
    print("[PYTH] PYTHON =", sys.executable, flush=True)
    print(
        f"[PYTH] protocol: version={MLPS_VERSION}, "
        f"setup={CMD_SETUP}, force={CMD_FORCE}, stop={CMD_STOP}",
        flush=True,
    )

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

        state = recv_setup(conn)

        # Prefer command-line spec. If C++ also sends a spec string over the
        # socket, use it only as fallback.
        spec_path = args.spec or state.spec_from_cpp
        spec = load_spec(spec_path)

        print(f"[PYTH] spec_path='{spec_path}'", flush=True)
        print(f"[PYTH] loaded spec={spec}", flush=True)

        backend = str(spec.get("backend", args.backend)).lower().strip()

        if gpu_from_argv != state.gpu:
            print(
                f"[PYTH] warning: --gpu={gpu_from_argv} but setup gpu={state.gpu}; "
                f"using setup gpu={state.gpu}",
                flush=True,
            )

        predictor = build_predictor(
            backend=backend,
            atomic_numbers=state.ml_zid,
            gpu=state.gpu,
            spec=spec,
        )

        serve(conn, predictor, state, spec)

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
