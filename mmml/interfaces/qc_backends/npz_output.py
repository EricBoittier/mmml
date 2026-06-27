"""Convert backend results to aligned NPZ dicts with consistent units."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.data.units import convert_energy, convert_forces, normalize_energy_unit, normalize_force_unit


BACKEND_NATIVE_UNITS: dict[str, tuple[str, str]] = {
    "pyscf": ("hartree", "hartree_bohr"),
    "ml": ("ev", "ev_angstrom"),
    "orca": ("hartree", "hartree_bohr"),
    "xtb": ("ev", "ev_angstrom"),
    "molpro": ("hartree", "hartree_bohr"),
}


def stack_frame_results(
    *,
    energies: list[float],
    forces: list[np.ndarray] | None,
    dipoles: list[np.ndarray] | None,
    frames_z: list[np.ndarray],
    frames_r: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Build a batched NPZ dict from per-frame lists."""
    n_frames = len(energies)
    n_pad = max(len(z) for z in frames_z)
    z_batch = np.zeros((n_frames, n_pad), dtype=np.int32)
    r_batch = np.zeros((n_frames, n_pad, 3), dtype=np.float64)
    n_arr = np.zeros(n_frames, dtype=np.int32)
    for i, (zi, ri) in enumerate(zip(frames_z, frames_r)):
        n = len(zi)
        n_arr[i] = n
        z_batch[i, :n] = zi
        r_batch[i, :n] = ri

    out: dict[str, np.ndarray] = {
        "E": np.asarray(energies, dtype=np.float64),
        "R": r_batch,
        "Z": z_batch,
        "N": n_arr,
    }
    if forces is not None and len(forces) == n_frames:
        f_batch = np.zeros((n_frames, n_pad, 3), dtype=np.float64)
        for i, f in enumerate(forces):
            f_batch[i, : f.shape[0]] = f
        out["F"] = f_batch
    if dipoles is not None and len(dipoles) == n_frames:
        out["Dxyz"] = np.stack(dipoles, axis=0)
    return out


def normalize_backend_npz(
    data: dict[str, np.ndarray],
    *,
    backend: str,
    target_energy_unit: str = "ev",
    target_force_unit: str = "ev_angstrom",
    source_energy_unit: str | None = None,
    source_force_unit: str | None = None,
) -> dict[str, np.ndarray]:
    """Convert backend-native units to target units for comparison."""
    native_e, native_f = BACKEND_NATIVE_UNITS.get(backend, ("ev", "ev_angstrom"))
    src_e = source_energy_unit or native_e
    src_f = source_force_unit or native_f
    out = dict(data)
    if "E" in out:
        out["E"] = np.asarray(
            convert_energy(out["E"], src_e, target_energy_unit),
            dtype=np.float64,
        )
    if "F" in out:
        out["F"] = np.asarray(
            convert_forces(out["F"], src_f, target_force_unit),
            dtype=np.float64,
        )
    return out


def infer_target_units(reference: dict[str, np.ndarray]) -> tuple[str, str]:
    """Infer comparison units from reference metadata or magnitude heuristics."""
    meta = reference.get("units")
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        meta = meta.item() if meta.ndim == 0 else meta[0]
    if isinstance(meta, dict):
        e = meta.get("energy") or meta.get("E")
        f = meta.get("forces") or meta.get("F")
        if e and f:
            return normalize_energy_unit(str(e)), normalize_force_unit(str(f))

    e_med = float(np.median(np.abs(np.asarray(reference.get("E", [0.0]), dtype=np.float64))))
    # Molecular totals: |E| ~ 1–500 Ha vs ~ 30–15000 eV for similar systems
    if e_med > 500.0:
        return "ev", "ev_angstrom"
    if e_med > 0.5:
        return "hartree", "hartree_bohr"
    return "ev", "ev_angstrom"


def write_backend_metadata(
    data: dict[str, np.ndarray],
    *,
    backend: str,
    method_label: str,
    energy_unit: str,
    force_unit: str,
) -> dict[str, np.ndarray]:
    """Attach backend metadata for reports."""
    out = dict(data)
    out["_backend"] = np.array(backend)
    out["_method_label"] = np.array(method_label)
    out["units"] = np.array(
        {
            "energy": energy_unit,
            "forces": force_unit,
            "backend": backend,
            "method": method_label,
        },
        dtype=object,
    )
    return out
