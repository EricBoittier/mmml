"""CHARMM CMAP (2D backbone correction map) for JAX bonded calculators.

Bicubic coefficient generation follows OpenMM's port of the CHARMM CMAP
interpolation (``CMAPTorsionForceImpl::calcMapDerivatives``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
from scipy.interpolate import CubicSpline

from jax_md import space
from jax_md.mm_forcefields.base import BondedParameters, Topology
from jax_md.util import normalize

_TWO_PI = 2.0 * np.pi

# OpenMM bicubic patch coefficient weights (16x16).
_WT = np.array(
    [
        1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
        0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
        0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
        0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
        0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
        0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
        0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
        0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1,
    ],
    dtype=np.float64,
).reshape(16, 16)


@dataclass(frozen=True, slots=True)
class CmapType:
    """One CHARMM CMAP grid (``resolution`` x ``resolution`` energies, kcal/mol)."""

    resolution: int
    energies: tuple[float, ...]


def parse_cmap_types_from_prm(prm_path: str | Path) -> dict[tuple[str, ...], CmapType]:
    """Parse ``CMAP`` sections from a CHARMM parameter file."""
    lines = Path(prm_path).read_text(encoding="utf-8", errors="replace").splitlines()
    section: str | None = None
    current_key: tuple[str, ...] | None = None
    current_res = 0
    current_data: list[float] = []
    out: dict[tuple[str, ...], CmapType] = {}

    def _flush() -> None:
        nonlocal current_key, current_res, current_data
        if current_key is None:
            return
        expected = current_res * current_res
        if len(current_data) != expected:
            raise ValueError(
                f"CMAP {current_key!r} in {prm_path}: expected {expected} grid "
                f"values, found {len(current_data)}"
            )
        cmap = CmapType(current_res, tuple(current_data))
        out[current_key] = cmap
        out[tuple(reversed(current_key))] = cmap
        current_key = None
        current_res = 0
        current_data = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!") or stripped.startswith("*"):
            continue
        head = stripped.split()[0].upper()
        if head in {"BOND", "ANGLE", "DIHE", "IMPR", "NONBONDED", "NBON", "END"}:
            if section == "CMAP":
                _flush()
            section = None if head == "END" else head
            continue
        if head == "CMAP":
            if section == "CMAP":
                _flush()
            section = "CMAP"
            continue
        if section != "CMAP":
            continue

        words = stripped.split()
        try:
            current_data.extend(float(w) for w in words)
            continue
        except ValueError:
            _flush()
        if len(words) < 9:
            raise ValueError(f"Malformed CMAP header in {prm_path}: {stripped!r}")
        types = tuple(w.upper() for w in words[:8])
        current_res = int(words[8])
        current_key = min(types, tuple(reversed(types)))
        current_data = []

    if section == "CMAP":
        _flush()
    return out


def calc_map_derivatives(size: int, energy_flat: Sequence[float]) -> np.ndarray:
    """Return bicubic patch coefficients, shape ``(size*size, 16)``."""
    energy = np.asarray(energy_flat, dtype=np.float64)
    if energy.size != size * size:
        raise ValueError(f"CMAP grid size mismatch: {energy.size} vs {size}x{size}")

    d1 = np.zeros((size, size), dtype=np.float64)
    d2 = np.zeros((size, size), dtype=np.float64)
    d12 = np.zeros((size, size), dtype=np.float64)
    x = np.arange(size + 1, dtype=np.float64) * _TWO_PI / size

    for i in range(size):
        y = np.empty(size + 1, dtype=np.float64)
        for j in range(size):
            y[j] = energy[j + size * i]
        y[size] = y[0]
        spline = CubicSpline(x, y, bc_type="periodic")
        for j in range(size):
            d1[i, j] = float(spline(x[j], 1))

    for j in range(size):
        y = np.empty(size + 1, dtype=np.float64)
        for i in range(size):
            y[i] = energy[i + size * j]
        y[size] = y[0]
        spline = CubicSpline(x, y, bc_type="periodic")
        for i in range(size):
            d2[i, j] = float(spline(x[i], 1))

    for i in range(size):
        y = np.empty(size + 1, dtype=np.float64)
        for j in range(size):
            y[j] = d2[i, j]
        y[size] = y[0]
        spline = CubicSpline(x, y, bc_type="periodic")
        for j in range(size):
            d12[i, j] = float(spline(x[j], 1))

    delta = _TWO_PI / size
    coeffs = np.zeros((size * size, 16), dtype=np.float64)
    for i in range(size):
        next_i = (i + 1) % size
        for j in range(size):
            next_j = (j + 1) % size
            e = np.array(
                [
                    energy[i + j * size],
                    energy[next_i + j * size],
                    energy[next_i + next_j * size],
                    energy[i + next_j * size],
                ],
                dtype=np.float64,
            )
            e1 = np.array(
                [d1[i, j], d1[next_i, j], d1[next_i, next_j], d1[i, next_j]],
                dtype=np.float64,
            )
            e2 = np.array(
                [d2[i, j], d2[next_i, j], d2[next_i, next_j], d2[i, next_j]],
                dtype=np.float64,
            )
            e12 = np.array(
                [d12[i, j], d12[next_i, j], d12[next_i, next_j], d12[i, next_j]],
                dtype=np.float64,
            )
            rhs = np.concatenate([e, e1 * delta, e2 * delta, e12 * delta * delta])
            coeffs[i + size * j] = _WT @ rhs
    return coeffs


def cmap_type_key(atom_types: Sequence[str], atom_indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(atom_types[int(i)] for i in atom_indices)


def _resolve_cmap_type_key(
    atom_types: Sequence[str],
    atom_indices: Sequence[int],
    cmap_types: dict[tuple[str, ...], CmapType],
) -> tuple[str, ...] | None:
    key = cmap_type_key(atom_types, atom_indices)
    if key in cmap_types:
        return key
    rev = tuple(reversed(key))
    if rev in cmap_types:
        return rev
    return None


def build_cmap_arrays(
    cmap_atoms: np.ndarray,
    atom_types: Sequence[str],
    prm_paths: Sequence[str | Path],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Build ``(cmap_atoms, cmap_map_idx, cmap_coeffs)`` from PSF + PRM files."""
    if cmap_atoms.size == 0:
        return None, None, None

    cmap_types: dict[tuple[str, ...], CmapType] = {}
    for path in prm_paths:
        cmap_types.update(parse_cmap_types_from_prm(path))

    unique_coeffs: list[np.ndarray] = []
    key_to_idx: dict[tuple[str, ...], int] = {}
    map_idx: list[int] = []
    kept_rows: list[np.ndarray] = []

    for row in cmap_atoms:
        key = _resolve_cmap_type_key(atom_types, row, cmap_types)
        if key is None:
            # CHARMM leaves PSF CMAP rows without PRM grids at zero energy.
            continue
        kept_rows.append(row)
        if key not in key_to_idx:
            cmap = cmap_types[key]
            key_to_idx[key] = len(unique_coeffs)
            unique_coeffs.append(
                calc_map_derivatives(cmap.resolution, cmap.energies)
            )
        map_idx.append(key_to_idx[key])

    if not kept_rows:
        return None, None, None

    stacked = np.stack(unique_coeffs, axis=0)
    return (
        np.asarray(kept_rows, dtype=np.int32),
        np.asarray(map_idx, dtype=np.int32),
        stacked,
    )


def dihedral_angle_0_2pi(
    p0: Array,
    p1: Array,
    p2: Array,
    p3: Array,
    displacement_fn: space.DisplacementFn,
) -> Array:
    """Signed dihedral wrapped to ``[0, 2π)`` (CHARMM/OpenMM CMAP convention)."""
    b0 = displacement_fn(p1, p0)
    b1 = displacement_fn(p2, p1)
    b2 = displacement_fn(p3, p2)
    b1_norm = normalize(b1)
    v = b0 - jnp.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
    w = b2 - jnp.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_norm, v) * w, axis=-1)
    return jnp.mod(jnp.arctan2(y, x) + 2.0 * jnp.pi, 2.0 * jnp.pi)


def _eval_bicubic(coeff: Array, da: Array, db: Array) -> Array:
    """Evaluate one bicubic patch (OpenMM Horner form)."""
    energy = jnp.array(0.0, dtype=coeff.dtype)
    for i in range(3, -1, -1):
        row = coeff[i * 4 + 3] * db + coeff[i * 4 + 2]
        row = row * db + coeff[i * 4 + 1]
        row = row * db + coeff[i * 4 + 0]
        energy = da * energy + row
    return energy


def cmap_energy(
    positions: Array,
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn,
) -> Array:
    """Total CMAP correction energy (kcal/mol)."""
    if (
        topology.cmap_atoms is None
        or topology.cmap_map_idx is None
        or bonded.cmap_maps is None
    ):
        return jnp.array(0.0, dtype=positions.dtype)
    if topology.cmap_atoms.shape[0] == 0:
        return jnp.array(0.0, dtype=positions.dtype)

    cmap_atoms = jnp.asarray(topology.cmap_atoms, dtype=jnp.int32)
    cmap_map_idx = jnp.asarray(topology.cmap_map_idx, dtype=jnp.int32)
    coeffs = jnp.asarray(bonded.cmap_maps, dtype=jnp.float64)
    size = int(round(float(np.sqrt(coeffs.shape[1]))))
    delta = 2.0 * jnp.pi / size

    def one_term(atoms: Array, map_idx: Array) -> Array:
        a1, a2, a3, a4, b1, b2, b3, b4 = (
            atoms[0],
            atoms[1],
            atoms[2],
            atoms[3],
            atoms[4],
            atoms[5],
            atoms[6],
            atoms[7],
        )
        angle_a = dihedral_angle_0_2pi(
            positions[a1], positions[a2], positions[a3], positions[a4], displacement_fn
        )
        angle_b = dihedral_angle_0_2pi(
            positions[b1], positions[b2], positions[b3], positions[b4], displacement_fn
        )
        s = jnp.minimum(jnp.floor(angle_a / delta).astype(jnp.int32), size - 1)
        t = jnp.minimum(jnp.floor(angle_b / delta).astype(jnp.int32), size - 1)
        da = angle_a / delta - s
        db = angle_b / delta - t
        patch_idx = s + size * t
        coeff = coeffs[map_idx, patch_idx]
        return _eval_bicubic(coeff, da, db)

    return jnp.sum(
        vmap(one_term)(cmap_atoms, cmap_map_idx),
    )


def attach_cmap_to_bonded(
    bonded: BondedParameters,
    cmap_coeffs: np.ndarray | None,
) -> BondedParameters:
    if cmap_coeffs is None:
        return bonded
    return BondedParameters(
        bond_k=bonded.bond_k,
        bond_r0=bonded.bond_r0,
        angle_k=bonded.angle_k,
        angle_theta0=bonded.angle_theta0,
        torsion_k=bonded.torsion_k,
        torsion_n=bonded.torsion_n,
        torsion_gamma=bonded.torsion_gamma,
        improper_k=bonded.improper_k,
        improper_n=bonded.improper_n,
        improper_gamma=bonded.improper_gamma,
        cmap_maps=jnp.asarray(cmap_coeffs, dtype=jnp.float64),
    )
