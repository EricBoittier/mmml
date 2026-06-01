"""Shared utilities for MM/ML calculator (indices, switching functions, etc.)."""

from __future__ import annotations

from itertools import combinations
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    from jax import Array
except ModuleNotFoundError:
    jnp = None  # type: ignore[assignment]
    Array = Any  # type: ignore[misc,assignment]

from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON


def unpack_factory_result(result: tuple) -> tuple:
    """Unpack the return value of a ``setup_calculator()`` factory call.

    Returns ``(calculator, spherical_cutoff_calculator, get_update_fn)`` where
    ``get_update_fn`` is ``None`` when the factory returns only two values.
    """
    n = len(result)
    if n == 3:
        return result[0], result[1], result[2]
    if n == 2:
        return result[0], result[1], None
    raise ValueError(f"setup_calculator factory returned {n} values, expected 2 or 3")


# Re-export for convenience
__all__ = [
    "FLAT_BOTTOM_MODES",
    "GAMMA_OFF",
    "GAMMA_ON",
    "ModelOutput",
    "apply_flat_bottom",
    "debug_print",
    "dimer_permutations",
    "epsilon",
    "indices_of_monomer",
    "indices_of_pairs",
    "jax_smooth_cutoff_cosine",
    "jax_smooth_switch_linear",
    "ml_switch_simple",
    "mm_switch_simple",
    "parse_non_int",
    "_ase_cell_to_3x3",
    "_safe_den",
    "_sharpstep",
    "_smoothstep01",
    "unpack_factory_result",
    "box_vectors_from_atoms_or_cell",
]

epsilon = 10 ** (-6)


def parse_non_int(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).lower().capitalize()


def dimer_permutations(n_mol: int) -> List[Tuple[int, int]]:
    return list(combinations(range(n_mol), 2))


# -----------------------------------------------------------------------------
# JAX-native smooth switching utilities (avoid traced-python conditionals)
# -----------------------------------------------------------------------------
def _safe_den(x: float | Array) -> Array:
    return jnp.maximum(x, 1e-6)


def jax_smooth_switch_linear(r: Array, x0: float, x1: float) -> Array:
    t = (r - x0) / _safe_den(x1 - x0)
    return jnp.clip(t, 0.0, 1.0)


def jax_smooth_cutoff_cosine(r: Array, cutoff: float) -> Array:
    t = r / _safe_den(cutoff)
    val = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < cutoff, val, 0.0)


def ml_switch_simple(r: Array, ml_cutoff: float, mm_switch_on: float) -> Array:
    """ML active at short range, tapers 1→0 over [mm_switch_on - ml_cutoff, mm_switch_on]."""
    taper_start = mm_switch_on - ml_cutoff
    t = (r - taper_start) / _safe_den(ml_cutoff)
    cosine_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < taper_start, 1.0, jnp.where(r < mm_switch_on, cosine_taper, 0.0))


def mm_switch_simple(r: Array, mm_switch_on: float, mm_cutoff: float) -> Array:
    """MM off at short range, ramps 0→1 over [mm_switch_on, mm_switch_on + mm_cutoff]."""
    ramp_end = mm_switch_on + mm_cutoff
    t = (r - mm_switch_on) / _safe_den(mm_cutoff)
    cosine_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < mm_switch_on, 0.0, jnp.where(r < ramp_end, cosine_ramp, 1.0))


def _smoothstep01(s: Array) -> Array:
    return s * s * (3.0 - 2.0 * s)


def _sharpstep(r: Array, x0: float, x1: float, gamma: float = GAMMA_ON) -> Array:
    s = jnp.clip((r - x0) / _safe_den(x1 - x0), 0.0, 1.0)
    s = s ** gamma
    return _smoothstep01(s)


def indices_of_pairs(
    a: int,
    b: int,
    n_atoms: int = 5,
    n_mol: int = 20,
    monomer_offsets: Optional[np.ndarray] = None,
    atoms_per_monomer_list: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return concatenated atom indices for monomers *a* and *b* (1-indexed).

    When *monomer_offsets* and *atoms_per_monomer_list* are provided the function
    supports heterogeneous monomer sizes; otherwise it falls back to the legacy
    uniform-size behaviour.
    """
    assert a < b, "by convention, res a must have a smaller index than res b"
    assert a >= 1, "res indices can't start from 1"
    assert b >= 1, "res indices can't start from 1"
    assert a != b, "pairs can't contain same residue"
    if monomer_offsets is not None and atoms_per_monomer_list is not None:
        off_a = monomer_offsets[a - 1]
        off_b = monomer_offsets[b - 1]
        n_a = atoms_per_monomer_list[a - 1]
        n_b = atoms_per_monomer_list[b - 1]
        return np.concatenate([
            np.arange(off_a, off_a + n_a),
            np.arange(off_b, off_b + n_b),
        ])
    return np.concatenate(
        [
            np.arange(0, n_atoms, 1) + (a - 1) * n_atoms,
            np.arange(0, n_atoms, 1) + (b - 1) * n_atoms,
        ]
    )


def indices_of_monomer(
    a: int,
    n_atoms: int = 5,
    n_mol: int = 20,
    monomer_offsets: Optional[np.ndarray] = None,
    atoms_per_monomer_list: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return atom indices for monomer *a* (1-indexed).

    When *monomer_offsets* and *atoms_per_monomer_list* are provided the function
    supports heterogeneous monomer sizes.
    """
    assert a < (n_mol + 1), "monomer index outside total n molecules"
    if monomer_offsets is not None and atoms_per_monomer_list is not None:
        off = monomer_offsets[a - 1]
        n = atoms_per_monomer_list[a - 1]
        return np.arange(off, off + n)
    return np.arange(0, n_atoms, 1) + (a - 1) * n_atoms


class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in eV (hybrid + flat-bottom)
    forces: Array  # Shape: (n_atoms, 3), forces in eV/Å
    dH: Array  # Shape: (,), total interaction energy in eV
    internal_E: Array  # Shape: (,) total internal energy in eV
    internal_F: Array
    mm_E: Array
    mm_F: Array
    ml_2b_E: Array
    ml_2b_F: Array
    hybrid_energy: Array  # ML+MM before flat-bottom (eV)
    flat_bottom_E: Array  # flat-bottom contribution (eV)
    com: Array  # mass-weighted system COM (3,); zeros in monomer mode
    com_dist: Array  # |COM - center| (Å), or max_m |COM_m - center| in monomer mode


FLAT_BOTTOM_MODES = ("system", "monomer")


def apply_flat_bottom(
    positions: Array,
    atomic_numbers: Array,
    base_forces: Array,
    *,
    radius: float,
    k: float,
    mode: str,
    monomer_offsets: Array,
    n_monomers: int,
    pbc_cell: Array | None,
    mic_fn,
) -> Tuple[Array, Array, Array, Array]:
    """Harmonic flat-bottom on COM(s). Returns (flat_E, flat_F, com, com_dist)."""
    if jnp is None:
        raise RuntimeError("apply_flat_bottom requires JAX")
    from ase.data import atomic_masses as ase_atomic_masses

    masses = jnp.take(jnp.array(ase_atomic_masses, dtype=positions.dtype), atomic_numbers)
    if pbc_cell is not None:
        center = (pbc_cell[0] + pbc_cell[1] + pbc_cell[2]) / 2.0
    else:
        center = jnp.zeros(3, dtype=positions.dtype)

    if mode == "system":
        M = jnp.sum(masses)
        com = jnp.sum(positions * masses[:, None], axis=0) / M
        if pbc_cell is not None:
            d = mic_fn(center, com, pbc_cell)
        else:
            d = com - center
        com_dist = jnp.linalg.norm(d)
        excess = jnp.maximum(0.0, com_dist - radius)
        flat_E = k * excess ** 2
        unit_d = d / (com_dist + 1e-12)
        F_com = -k * 2.0 * excess * unit_d
        flat_F = (masses[:, None] / M) * F_com[None, :]
        return flat_E, flat_F, com, com_dist

    if mode != "monomer":
        raise ValueError(f"flat_bottom mode must be one of {FLAT_BOTTOM_MODES}, got {mode!r}")

    # Static slice bounds: monomer_offsets and n_monomers are fixed at JIT compile time
    # (n_monomers is a static arg on spherical_cutoff_calculator). lax.fori_loop cannot
    # be used because dynamic slice endpoints are not allowed inside traced loops.
    mo_np = np.asarray(monomer_offsets, dtype=np.int32)
    n_mon = int(n_monomers)

    flat_E = jnp.array(0.0, dtype=positions.dtype)
    flat_F = jnp.zeros_like(base_forces)
    max_dist = jnp.array(0.0, dtype=positions.dtype)
    for m in range(n_mon):
        s = int(mo_np[m])
        e = int(mo_np[m + 1])
        pos_m = positions[s:e]
        mass_m = masses[s:e]
        M_m = jnp.sum(mass_m)
        com_m = jnp.sum(pos_m * mass_m[:, None], axis=0) / M_m
        if pbc_cell is not None:
            d = mic_fn(center, com_m, pbc_cell)
        else:
            d = com_m - center
        dist = jnp.linalg.norm(d)
        excess = jnp.maximum(0.0, dist - radius)
        E_m = k * excess ** 2
        unit_d = d / (dist + 1e-12)
        F_com = -k * 2.0 * excess * unit_d
        F_m = (mass_m[:, None] / M_m) * F_com[None, :]
        flat_F = flat_F.at[s:e].add(F_m)
        flat_E = flat_E + E_m
        max_dist = jnp.maximum(max_dist, dist)

    com = jnp.zeros(3, dtype=positions.dtype)
    return flat_E, flat_F, com, max_dist


def debug_print(debug: bool, msg: str, *args: Any, **kwargs: Any) -> None:
    """Helper function for conditional debug printing."""
    if debug:
        print(msg)
        for arg in args:
            pass
        try:
            for name, value in kwargs.items():
                print(f"{name}: {value.shape}")
        except Exception:
            pass


def box_vectors_from_atoms_or_cell(
    atoms: Any,
    setup_cell: Any = None,
) -> np.ndarray | None:
    """Return orthorhombic box lengths ``(Lx, Ly, Lz)`` from ASE atoms or setup cell."""
    cell = _ase_cell_to_3x3(atoms)
    if cell is not None:
        return np.diagonal(cell).astype(np.float64)
    if setup_cell is None:
        return None
    sc = np.asarray(setup_cell, dtype=np.float64)
    if sc.ndim == 0:
        length = float(sc)
        return np.array([length, length, length], dtype=np.float64)
    if sc.shape == (3,):
        return sc.copy()
    if sc.shape == (3, 3):
        return np.diagonal(sc).astype(np.float64)
    return None


def _ase_cell_to_3x3(atoms: Any) -> np.ndarray | None:
    """Extract 3x3 cell matrix from ASE atoms. Returns None if cell is invalid/empty."""
    try:
        import ase  # noqa: F401
    except ImportError:
        return None
    if atoms is None:
        return None
    try:
        cell = atoms.get_cell()
        if cell is None:
            return None
        arr = np.asarray(cell, dtype=np.float64)
        if arr.shape != (3, 3):
            return None
        lengths = np.linalg.norm(arr, axis=1)
        if np.any(lengths < 1e-6):
            return None
        return arr
    except Exception:
        return None
