"""SPICE-α (Zenodo 19205036) HDF5 → PhysNet train NPZ.

The public release uses one HDF5 group per molecule and ``M`` conformers per
group (``conformations`` of shape ``(M, N, 3)``). That is not the PhysNetJAX
``mol_*`` / ``positions`` / ``total_forces`` layout in ``read_h5.py``.

Bundled ``units_map`` is already MMML **train** units (Å, eV, eV/Å, e·Å)
except ``dft_total_gradient``, which is ∇E. This module stores ``F = −∇E``.

Never download Zenodo from here. Tests use synthetic HDF5 only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

UnitsKind = Literal["canonical", "atomic", "unknown"]

# From SPICE-alpha/README.md inside the Zenodo zip (not inferred).
SPICE_ALPHA_CANONICAL_UNITS: dict[str, str] = {
    "atomic_numbers": "dimensionless",
    "conformations": "Angstrom",
    "dft_total_energy": "eV",
    "dft_total_gradient": "eV/Angstrom",
    "scf_dipole": "elementary_charge*Angstrom",
    "polarizability": "e*Angstrom^2/volt",
}

TRAIN_NPZ_UNITS: dict[str, str] = {
    "R": "angstrom",
    "E": "ev",
    "F": "ev_angstrom",
    "D": "e_angstrom",
    "force": "negated dft_total_gradient",
    "source": "zenodo-19205036",
}

DEFAULT_CHARGE_TOL = 0.15
PHYSNET_TRAIN_KEYS = ("R", "Z", "N", "E", "F")
PHYSNET_OPTIONAL_KEYS = ("D", "Q", "polar")


def normalize_unit_token(value: str) -> str:
    """Collapse unit-string spelling so README / MACE / OpenMM variants match."""
    text = str(value).strip().lower()
    for old, new in (
        ("·", "*"),
        (" ", ""),
        ("_", ""),
        ("angstroms", "angstrom"),
        ("ångstrom", "angstrom"),
        ("elementarycharge", "e"),
    ):
        text = text.replace(old, new)
    return text


def classify_units_map(units_map: Mapping[str, Any] | None) -> UnitsKind:
    """Return ``canonical`` (already train units), ``atomic`` (original SPICE), or ``unknown``."""
    if not units_map:
        return "unknown"
    tokens = {key: normalize_unit_token(str(val)) for key, val in units_map.items()}
    energy = tokens.get("dft_total_energy", "")
    coords = tokens.get("conformations", "")
    grad = tokens.get("dft_total_gradient", "")
    if "hartree" in energy or "bohr" in coords or "hartree" in grad:
        return "atomic"
    if "ev" in energy and "angstrom" in coords and "ev" in grad:
        return "canonical"
    return "unknown"


def parse_units_attr(raw: Any) -> dict[str, str]:
    """Decode an HDF5 ``units_map`` attribute (JSON string or mapping)."""
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return {str(k): str(v) for k, v in loaded.items()}
        return {}
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def read_units_map(h5: Any) -> dict[str, str]:
    """File-level ``units_map``, else the first group's."""
    parsed = parse_units_attr(h5.attrs.get("units_map"))
    if parsed:
        return parsed
    for name in h5.keys():
        group = h5[name]
        parsed = parse_units_attr(getattr(group, "attrs", {}).get("units_map"))
        if parsed:
            return parsed
    return {}


@dataclass(frozen=True)
class SpiceAlphaFrame:
    """One conformation after gradient → force conversion."""

    Z: np.ndarray
    R: np.ndarray
    E: float
    F: np.ndarray
    D: np.ndarray
    Q: float | None
    polar: np.ndarray | None
    group: str
    iconf: int


def _optional_charge(group: Any, n: int) -> np.ndarray | None:
    if "mbis_charges" not in group:
        return None
    charges = np.asarray(group["mbis_charges"][()], dtype=np.float64)
    return charges.reshape(n, -1).sum(axis=1)


def iter_spice_alpha_frames(
    h5: Any,
    *,
    flip_gradient: bool = True,
    neutral_only: bool = False,
    charge_tol: float = DEFAULT_CHARGE_TOL,
) -> Iterator[SpiceAlphaFrame]:
    """Yield frames from a SPICE-α-style HDF5 file object."""
    for name in h5.keys():
        group = h5[name]
        if "conformations" not in group or "atomic_numbers" not in group:
            continue
        numbers = np.asarray(group["atomic_numbers"][()], dtype=np.int32)
        positions = np.asarray(group["conformations"][()], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"{name}: conformations must be (M, N, 3), got {positions.shape}"
            )
        n_conf, n_atoms, _ = positions.shape
        if numbers.shape != (n_atoms,):
            raise ValueError(
                f"{name}: atomic_numbers shape {numbers.shape} != ({n_atoms},)"
            )
        if "dft_total_energy" not in group or "dft_total_gradient" not in group:
            continue
        energy = np.asarray(group["dft_total_energy"][()], dtype=np.float64).reshape(-1)
        gradient = np.asarray(group["dft_total_gradient"][()], dtype=np.float64)
        if energy.shape != (n_conf,) or gradient.shape != (n_conf, n_atoms, 3):
            raise ValueError(
                f"{name}: energy/gradient shapes {energy.shape}/{gradient.shape} "
                f"do not match conformations {positions.shape}"
            )
        if "scf_dipole" in group:
            dipole = np.asarray(group["scf_dipole"][()], dtype=np.float64)
            if dipole.shape != (n_conf, 3):
                raise ValueError(f"{name}: scf_dipole shape {dipole.shape}")
        else:
            dipole = np.full((n_conf, 3), np.nan, dtype=np.float64)
        polar = None
        if "polarizability" in group:
            polar = np.asarray(group["polarizability"][()], dtype=np.float64)
            if polar.shape != (n_conf, 3, 3):
                raise ValueError(f"{name}: polarizability shape {polar.shape}")
        charges = _optional_charge(group, n_conf)
        forces = -gradient if flip_gradient else gradient
        for i in range(n_conf):
            if not np.isfinite(energy[i]) or not np.isfinite(forces[i]).all():
                continue
            q = None if charges is None else float(charges[i])
            if neutral_only and q is not None and abs(q) > charge_tol:
                continue
            yield SpiceAlphaFrame(
                Z=numbers,
                R=positions[i],
                E=float(energy[i]),
                F=np.asarray(forces[i], dtype=np.float64),
                D=dipole[i],
                Q=q,
                polar=None if polar is None else polar[i],
                group=str(name),
                iconf=i,
            )


def pad_frames(
    frames: Sequence[SpiceAlphaFrame],
    *,
    pad_atoms: int | None = None,
) -> dict[str, np.ndarray]:
    """Stack frames into a PhysNet-padded NPZ dict (Å / eV / eV/Å / e·Å)."""
    if not frames:
        raise ValueError("no frames")
    pad = int(pad_atoms or max(int(fr.Z.shape[0]) for fr in frames))
    n = len(frames)
    R = np.zeros((n, pad, 3), dtype=np.float64)
    F = np.zeros((n, pad, 3), dtype=np.float64)
    Z = np.zeros((n, pad), dtype=np.int32)
    N = np.zeros((n,), dtype=np.int32)
    E = np.zeros((n,), dtype=np.float64)
    D = np.zeros((n, 3), dtype=np.float64)
    Q = np.zeros((n,), dtype=np.float64)
    polar = np.full((n, 3, 3), np.nan, dtype=np.float64)
    has_polar = False
    for i, fr in enumerate(frames):
        n_real = int(fr.Z.shape[0])
        if n_real > pad:
            raise ValueError(f"frame {i} has {n_real} atoms > pad_atoms={pad}")
        Z[i, :n_real] = fr.Z
        R[i, :n_real] = fr.R
        F[i, :n_real] = fr.F
        N[i] = n_real
        E[i] = fr.E
        D[i] = fr.D
        Q[i] = 0.0 if fr.Q is None else fr.Q
        if fr.polar is not None:
            polar[i] = fr.polar
            has_polar = True
    out: dict[str, np.ndarray] = {
        "R": R,
        "Z": Z,
        "N": N,
        "E": E,
        "F": F,
        "D": D,
        "Q": Q,
        "_mmml_units": np.array(json.dumps(TRAIN_NPZ_UNITS)),
    }
    if has_polar:
        out["polar"] = polar
    return out


def assert_train_npz_contract(data: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if arrays are not a PhysNet train NPZ."""
    missing = [key for key in PHYSNET_TRAIN_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing train keys: {missing}")
    r = np.asarray(data["R"])
    z = np.asarray(data["Z"])
    f = np.asarray(data["F"])
    n = np.asarray(data["N"]).reshape(-1)
    e = np.asarray(data["E"]).reshape(-1)
    if r.ndim != 3 or r.shape[-1] != 3:
        raise ValueError(f"R must be (n, pad, 3), got {r.shape}")
    n_struct, pad, _ = r.shape
    if z.shape != (n_struct, pad) or f.shape != r.shape:
        raise ValueError(f"Z/F shapes {z.shape}/{f.shape} do not match R {r.shape}")
    if e.shape != (n_struct,) or n.shape != (n_struct,):
        raise ValueError(f"E/N shapes {e.shape}/{n.shape} do not match n={n_struct}")
    if np.any(n < 0) or np.any(n > pad):
        raise ValueError("N must satisfy 0 <= N[i] <= pad")
    if "D" in data:
        d = np.asarray(data["D"])
        if d.shape != (n_struct, 3):
            raise ValueError(f"D must be (n, 3), got {d.shape}")
    units = data.get("_mmml_units")
    if units is not None:
        parsed = json.loads(str(np.asarray(units).reshape(-1)[0]))
        if parsed.get("E") not in {"ev", "eV"} or parsed.get("F") not in {
            "ev_angstrom",
            "ev/angstrom",
            "eV/Angstrom",
        }:
            raise ValueError(f"_mmml_units is not train units: {parsed}")


def max_atomic_number(data: Mapping[str, Any]) -> int:
    """Largest real (non-pad) atomic number; 0 if empty."""
    z = np.asarray(data["Z"])
    n = np.asarray(data["N"]).reshape(-1)
    found = 0
    for i, n_real in enumerate(n):
        if n_real:
            found = max(found, int(z[i, : int(n_real)].max()))
    return found


def write_physnet_npz(data: Mapping[str, Any], path: Path | str) -> Path:
    """Write a compressed NPZ after checking the train contract."""
    assert_train_npz_contract(data)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **{k: np.asarray(v) for k, v in data.items()})
    return dest


def convert_spice_alpha_hdf5(
    paths: Sequence[Path | str],
    out: Path | str,
    *,
    pad_atoms: int | None = None,
    max_frames: int = 0,
    flip_gradient: bool = True,
    neutral_only: bool = False,
    charge_tol: float = DEFAULT_CHARGE_TOL,
    require_canonical_units: bool = True,
) -> dict[str, np.ndarray]:
    """Load one or more SPICE-α HDF5 files and write a PhysNet NPZ."""
    import h5py

    frames: list[SpiceAlphaFrame] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            kind = classify_units_map(read_units_map(handle))
            if require_canonical_units and kind == "atomic":
                raise ValueError(
                    f"{path}: units_map looks like original SPICE (Bohr/Hartree). "
                    "Use fix-and-split defaults + --flip-forces, not this converter."
                )
            for frame in iter_spice_alpha_frames(
                handle,
                flip_gradient=flip_gradient,
                neutral_only=neutral_only,
                charge_tol=charge_tol,
            ):
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
        if max_frames and len(frames) >= max_frames:
            break
    data = pad_frames(frames, pad_atoms=pad_atoms)
    write_physnet_npz(data, out)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SPICE-α HDF5 to a PhysNet train NPZ (F = −∇E)."
    )
    parser.add_argument("hdf5", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--pad", type=int, default=0, help="0 = max N in this extract")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--neutral-only", action="store_true")
    parser.add_argument("--no-flip-gradient", action="store_true")
    parser.add_argument("--allow-atomic-units", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = convert_spice_alpha_hdf5(
        args.hdf5,
        args.out,
        pad_atoms=args.pad or None,
        max_frames=args.max_frames,
        flip_gradient=not args.no_flip_gradient,
        neutral_only=args.neutral_only,
        require_canonical_units=not args.allow_atomic_units,
    )
    print(
        f"wrote {args.out} n={len(data['E'])} pad={data['R'].shape[1]} "
        f"Zmax={max_atomic_number(data)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
