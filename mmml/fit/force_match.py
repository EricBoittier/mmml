"""Force-matching regularizer: hybrid ML/MM forces vs PET teacher forces.

For periodic teacher frames (PET-MAD whole box with the cell) the hybrid force
is split into parts that do not depend on the fitted parameters, cached once,
and the MM part that does::

    F_hyb(theta, lam) = F_ml_mono + lam * F_ml_dimer + F_mm(theta)
    F_mm(theta)       = -d/dR E_mm(R; per_atom_lj(theta))

``F_ml_mono`` is the PhysNet monomer (intramolecular) force, ``F_ml_dimer`` the
switched PhysNet dimer interaction force, ``E_mm`` the switched CGenFF energy
(``update_mm_pairs.energy_with_lj``). ``lam`` scales the whole dimer term, so
``F_ml_dimer`` must be cached separately from the monomer force.

Loss (kcal^2 mol^-2 A^-2)::

    fm_loss = mean_{frames, atoms} |F_hyb - F_PET|^2

Full per-atom forces are matched, not per-molecule net force/torque: both
models are translation invariant, so the net force of the whole box is ~0 for
both, but per-atom forces are dominated by the stiff intramolecular part
(``F_ml_mono`` vs PET), which theta cannot change and which sets a floor on the
loss. theta/lam only move the intermolecular part, which
:func:`force_match_report` isolates with the per-molecule net-force RMSE (the
intramolecular force of a translation-invariant monomer model sums to zero on
each molecule).

Units: forces kcal/mol/A, positions and cells A. The hybrid calculator returns
eV and eV/A; PET returns eV/A; both are converted with :data:`KCAL_PER_EV`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import warnings
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mmml.fit.lj_theta import LjTypeMap, Theta, per_atom_lj

KCAL_PER_EV = 23.060548867

# (positions, pair_idx, pair_mask, cell_3x3, charges, lj_rmins, lj_epsilons) -> E kcal/mol
MmEnergyWithLj = Callable[..., jnp.ndarray]


@dataclass(frozen=True)
class ForceMatchCache:
    """Stacked per-frame arrays (F frames, N atoms, P padded MM pairs).

    Forces in kcal/mol/A. ``pair_mask`` is 0 on padding rows; padding rows
    repeat a real pair so masked distances stay finite (no NaN gradients).
    """

    positions: np.ndarray  # (F, N, 3) A, molecules whole
    cells: np.ndarray  # (F, 3, 3) A
    f_pet: np.ndarray  # (F, N, 3)
    f_ml_mono: np.ndarray  # (F, N, 3)
    f_ml_dimer: np.ndarray  # (F, N, 3) switched dimer interaction
    f_mm_ref: np.ndarray  # (F, N, 3) calculator MM force at base LJ (diagnostic)
    pair_idx: np.ndarray  # (F, P, 2) int32
    pair_mask: np.ndarray  # (F, P) float
    base_rmins: np.ndarray  # (N,) CHARMM Rmin/2, A
    base_epsilons: np.ndarray  # (N,) CHARMM epsilon (<= 0), kcal/mol
    molecule_index: np.ndarray  # (N,) int, molecule id of each atom
    fingerprint: str = ""  # hash of the build inputs (see cache_fingerprint)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    def to_device(self) -> ForceMatchCache:
        """Copy with every array as a ``jax.Array`` (one host-to-device transfer)."""
        return jax.tree_util.tree_map(jnp.asarray, self)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> ForceMatchCache:
        """Load an npz written by :meth:`save` (older files lack ``fingerprint``)."""
        out: dict[str, Any] = {}
        with np.load(path) as data:
            for f in fields(cls):
                if f.name in data.files:
                    out[f.name] = np.asarray(data[f.name])
                elif f.default is MISSING:
                    raise KeyError(f"{path}: missing {f.name!r}")
        if "fingerprint" in out:
            out["fingerprint"] = str(out["fingerprint"])
        return cls(**out)

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[dict[str, np.ndarray]],
        *,
        base_rmins: np.ndarray,
        base_epsilons: np.ndarray,
        molecule_index: np.ndarray,
        fingerprint: str = "",
    ) -> ForceMatchCache:
        """Stack per-frame dicts (keys as the per-frame fields), padding pairs."""
        if not frames:
            raise ValueError("need at least one frame")
        # energy_with_lj's switching closure (``is_inter`` from the setup-time
        # neighbour list) only broadcasts when P equals the neighbour-list
        # capacity, so keep the raw capacity rather than packing tightly.
        capacity = max(int(np.asarray(f["pair_idx"]).shape[0]) for f in frames)
        pair_idx, pair_mask = pad_pairs([f["pair_idx"] for f in frames], [f["pair_mask"] for f in frames], capacity)
        stack = {
            k: np.stack([np.asarray(f[k], dtype=np.float64) for f in frames])
            for k in ("positions", "cells", "f_pet", "f_ml_mono", "f_ml_dimer", "f_mm_ref")
        }
        return cls(
            **stack,
            pair_idx=pair_idx,
            pair_mask=pair_mask,
            base_rmins=np.asarray(base_rmins, dtype=np.float64),
            base_epsilons=np.asarray(base_epsilons, dtype=np.float64),
            molecule_index=np.asarray(molecule_index, dtype=np.int32),
            fingerprint=fingerprint,
        )


_CACHE_ARRAY_FIELDS = tuple(f.name for f in fields(ForceMatchCache) if f.name != "fingerprint")

# A pytree (arrays = leaves, fingerprint = static aux data), so a cache can be
# passed to ``jax.jit`` as an argument instead of being baked in as constants.
jax.tree_util.register_pytree_node(
    ForceMatchCache,
    lambda c: (tuple(getattr(c, k) for k in _CACHE_ARRAY_FIELDS), c.fingerprint),
    lambda fp, leaves: ForceMatchCache(**dict(zip(_CACHE_ARRAY_FIELDS, leaves)), fingerprint=fp),
)


def pad_pairs(
    pair_idx: Sequence[np.ndarray],
    pair_mask: Sequence[np.ndarray],
    capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compact per-frame pair lists to their unmasked rows and pad to one capacity.

    Returns (F, P, 2) int32 and (F, P) float with P = max(capacity, max
    unmasked count); ``capacity=None`` packs tightly. Padding rows copy the
    frame's first valid pair with mask 0, so a masked row never has a zero
    distance.
    """
    kept = []
    for idx, mask in zip(pair_idx, pair_mask):
        idx = np.asarray(idx, dtype=np.int32).reshape(-1, 2)
        mask = np.asarray(mask, dtype=np.float64).reshape(-1)
        valid = mask != 0
        kept.append((idx[valid], mask[valid]))
    cap = max(1, capacity or 0, max(len(m) for _, m in kept))
    idx_out = np.zeros((len(kept), cap, 2), dtype=np.int32)
    mask_out = np.zeros((len(kept), cap), dtype=np.float64)
    for k, (idx, mask) in enumerate(kept):
        idx_out[k] = idx[0] if len(idx) else np.array([0, 1], dtype=np.int32)
        idx_out[k, : len(idx)] = idx
        mask_out[k, : len(mask)] = mask
    return idx_out, mask_out


def _mm_forces_frame(
    theta: Theta,
    positions: jnp.ndarray,
    pair_idx: jnp.ndarray,
    pair_mask: jnp.ndarray,
    cell: jnp.ndarray,
    base_rmins: jnp.ndarray,
    base_epsilons: jnp.ndarray,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> jnp.ndarray:
    """-grad_R E_mm(R; per_atom_lj(theta)) for one frame's arrays, (N, 3) kcal/mol/A."""
    rmins, epsilons = per_atom_lj(theta, type_map, jnp.asarray(base_rmins), jnp.asarray(base_epsilons))

    def energy(r: jnp.ndarray) -> jnp.ndarray:
        return mm_energy_with_lj(r, pair_idx, pair_mask, cell, None, lj_rmins=rmins, lj_epsilons=epsilons)

    return -jax.grad(energy)(jnp.asarray(positions))


def mm_forces(
    theta: Theta,
    cache: ForceMatchCache,
    frame: int,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> jnp.ndarray:
    """F_mm(theta) = -grad_R E_mm for one cached frame, (N, 3) kcal/mol/A."""
    return _mm_forces_frame(
        theta,
        jnp.asarray(cache.positions[frame]),
        jnp.asarray(cache.pair_idx[frame]),
        jnp.asarray(cache.pair_mask[frame]),
        jnp.asarray(cache.cells[frame]),
        cache.base_rmins,
        cache.base_epsilons,
        mm_energy_with_lj,
        type_map,
    )


def hybrid_forces(
    theta: Theta,
    lam: jnp.ndarray | float,
    cache: ForceMatchCache,
    frame: int,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> jnp.ndarray:
    """F_ml_mono + lam * F_ml_dimer + F_mm(theta) for one frame, kcal/mol/A."""
    return (
        jnp.asarray(cache.f_ml_mono[frame])
        + lam * jnp.asarray(cache.f_ml_dimer[frame])
        + mm_forces(theta, cache, frame, mm_energy_with_lj, type_map)
    )


def _fm_loss_arrays(
    theta: Theta,
    lam: jnp.ndarray | float,
    cache: ForceMatchCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> jnp.ndarray:
    """fm_loss body; ``cache`` leaves may be tracers. ``lax.map`` runs frames sequentially."""

    def frame_loss(xs: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        pos, idx, mask, cell, f_pet, f_mono, f_dimer = xs
        f_mm = _mm_forces_frame(
            theta, pos, idx, mask, cell, cache.base_rmins, cache.base_epsilons, mm_energy_with_lj, type_map
        )
        diff = f_mono + lam * f_dimer + f_mm - f_pet
        return jnp.mean(jnp.sum(diff**2, axis=-1))

    xs = (
        cache.positions,
        cache.pair_idx,
        cache.pair_mask,
        cache.cells,
        cache.f_pet,
        cache.f_ml_mono,
        cache.f_ml_dimer,
    )
    return jnp.mean(jax.lax.map(frame_loss, tuple(jnp.asarray(x) for x in xs)))


# (id(mm_energy_with_lj), id(type_map)) -> (fn, type_map, jitted loss); the
# objects are kept alive so their ids cannot be reused while cached.
_JITTED_LOSS: dict[tuple[int, int], tuple[Any, Any, Callable[..., jnp.ndarray]]] = {}


def _jitted_loss(mm_energy_with_lj: MmEnergyWithLj, type_map: LjTypeMap) -> Callable[..., jnp.ndarray]:
    key = (id(mm_energy_with_lj), id(type_map))
    hit = _JITTED_LOSS.get(key)
    if hit is None or hit[0] is not mm_energy_with_lj or hit[1] is not type_map:
        jitted = jax.jit(lambda th, lam, c: _fm_loss_arrays(th, lam, c, mm_energy_with_lj, type_map))
        hit = _JITTED_LOSS[key] = (mm_energy_with_lj, type_map, jitted)
    return hit[2]


def fm_loss(
    theta: Theta,
    lam: jnp.ndarray | float,
    cache: ForceMatchCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> jnp.ndarray:
    """mean over frames and atoms of |F_hyb(theta, lam) - F_PET|^2, (kcal/mol/A)^2.

    Differentiable in ``theta`` and ``lam`` (second derivative through
    ``energy_with_lj``). The component force RMSE is ``sqrt(fm_loss / 3)``.

    The cache arrays are passed to a jitted body as arguments (one compile per
    ``(mm_energy_with_lj, type_map)`` and cache shape), so calling this eagerly
    embeds no per-frame pair list as an HLO constant. Inside a caller's own
    ``jax.jit``, pass ``cache`` as a jit argument (it is a pytree) rather than
    closing over it: closed-over arrays are baked in as constants (~190 MB per
    acetone frame). Use :meth:`ForceMatchCache.to_device` once to avoid
    re-transferring the arrays on every call.
    """
    return _jitted_loss(mm_energy_with_lj, type_map)(theta, lam, cache)


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def molecular_net_forces(forces: np.ndarray, molecule_index: np.ndarray) -> np.ndarray:
    """Sum per-atom forces (..., N, 3) over each molecule -> (..., M, 3)."""
    forces = np.asarray(forces)
    n_mol = int(np.max(molecule_index)) + 1
    onehot = np.eye(n_mol)[np.asarray(molecule_index)]  # (N, M)
    return np.einsum("...na,nm->...ma", forces, onehot)


def molecular_torques(forces: np.ndarray, positions: np.ndarray, molecule_index: np.ndarray) -> np.ndarray:
    """Per-molecule torque about each molecule's centroid, (..., M, 3) kcal/mol.

    ``positions`` (..., N, 3) A must hold whole molecules (as cached).
    """
    forces, positions = np.asarray(forces), np.asarray(positions)
    n_mol = int(np.max(molecule_index)) + 1
    onehot = np.eye(n_mol)[np.asarray(molecule_index)]  # (N, M)
    centroid = np.einsum("...na,nm->...ma", positions, onehot) / onehot.sum(0)[:, None]
    arm = positions - centroid[..., np.asarray(molecule_index), :]
    return np.einsum("...na,nm->...ma", np.cross(arm, forces), onehot)


def force_match_report(
    theta: Theta,
    lam: float,
    cache: ForceMatchCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap,
) -> dict[str, float]:
    """Component RMSEs (kcal/mol/A) of the hybrid vs PET, with split contributions.

    ``rmse_*`` are per-Cartesian-component RMSEs of full per-atom forces against
    PET and ``rms_*`` are component RMS magnitudes of each force part; both are
    dominated by the intramolecular floor (``F_ml_mono`` vs PET). Only the
    ``mol_net_*`` (per-molecule net force, kcal/mol/A) and ``mol_torque_*``
    (per-molecule torque about the centroid, kcal/mol) keys exclude it: a
    translation- and rotation-invariant monomer model has zero net force and
    torque on each molecule, so these measure the intermolecular part alone.
    """
    f_mm = np.stack(
        [np.asarray(mm_forces(theta, cache, k, mm_energy_with_lj, type_map)) for k in range(cache.n_frames)]
    )
    mono, dimer, pet = cache.f_ml_mono, lam * cache.f_ml_dimer, cache.f_pet
    hyb = mono + dimer + f_mm
    net = lambda f: molecular_net_forces(f, cache.molecule_index)  # noqa: E731
    tq = lambda f: molecular_torques(f, cache.positions, cache.molecule_index)  # noqa: E731
    pet_net, pet_tq = net(pet), tq(pet)
    return {
        "rmse_hybrid": _rmse(hyb - pet),
        "rmse_ml_mono_only": _rmse(mono - pet),
        "rmse_ml_mono_dimer": _rmse(mono + dimer - pet),
        "rmse_ml_mono_mm": _rmse(mono + f_mm - pet),
        "rms_pet": _rmse(pet),
        "rms_ml_mono": _rmse(mono),
        "rms_ml_dimer": _rmse(dimer),
        "rms_mm": _rmse(f_mm),
        "mol_net_rms_pet": _rmse(pet_net),
        "mol_net_rmse_hybrid": _rmse(net(hyb) - pet_net),
        "mol_net_rmse_ml_only": _rmse(net(mono + dimer) - pet_net),
        "mol_net_rmse_mm_only": _rmse(net(mono + f_mm) - pet_net),
        "mol_net_rms_ml_mono": _rmse(net(mono)),  # ~0: sanity check of the split
        "mol_torque_rms_pet": _rmse(pet_tq),
        "mol_torque_rmse_hybrid": _rmse(tq(hyb) - pet_tq),
        "mol_torque_rmse_ml_only": _rmse(tq(mono + dimer) - pet_tq),
        "mol_torque_rmse_mm_only": _rmse(tq(mono + f_mm) - pet_tq),
        "mol_torque_rms_ml_mono": _rmse(tq(mono)),
        # only meaningful at theta0: F_mm(theta0) vs the calculator's own MM force
        "mm_vs_calculator_max_abs": float(np.max(np.abs(f_mm - cache.f_mm_ref))),
        "fm_loss": float(np.mean(np.sum((hyb - pet) ** 2, axis=-1))),
    }


# --------------------------------------------------------------------------
# Cache building (needs the hybrid calculator and the metatomic teacher).
# --------------------------------------------------------------------------


def molecule_index_from_sizes(atoms_per_monomer: Sequence[int]) -> np.ndarray:
    """(N,) molecule id per atom for consecutive molecules of the given sizes."""
    return np.repeat(np.arange(len(atoms_per_monomer)), np.asarray(atoms_per_monomer))


def make_molecules_whole(
    positions: np.ndarray,
    atoms_per_monomer: Sequence[int],
    cell_lengths: Sequence[float],
) -> np.ndarray:
    """Unwrap each molecule about its first atom and wrap its centroid into [0, L).

    Orthorhombic cells only. The hybrid ML monomer/dimer terms need whole
    molecules; PET (whole box with PBC) is invariant to this.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    box = np.asarray(cell_lengths, dtype=np.float64).reshape(3)
    start = 0
    for n in atoms_per_monomer:
        p = pos[start : start + n]
        d = p - p[:1]
        d -= box * np.round(d / box)
        p = p[:1] + d
        pos[start : start + n] = p - box * np.floor(p.mean(axis=0) / box)
        start += n
    return pos


def _orthorhombic_lengths(cell: np.ndarray) -> np.ndarray:
    cell = np.asarray(cell, dtype=np.float64)
    if cell.shape != (3, 3) or np.any(np.abs(cell - np.diag(np.diag(cell))) > 1e-8):
        raise ValueError("force matching supports orthorhombic cells only")
    return np.diag(cell).copy()


def pet_forces_kcal(
    calculator: Any, atomic_numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Whole-box periodic teacher forces (ASE calculator, eV/A) in kcal/mol/A."""
    from mmml.interfaces.calculators.ase_fragment_hybrid import evaluate_whole_system

    res = evaluate_whole_system(calculator, atomic_numbers, positions, cell=cell)
    return np.asarray(res.forces_ev_per_angstrom, dtype=np.float64) * KCAL_PER_EV


def hybrid_force_components(
    spherical_calculator: Callable[..., Any],
    update_mm_pairs: Callable[..., Any],
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    n_monomers: int,
    cutoff_params: Any,
) -> dict[str, np.ndarray]:
    """Hybrid ML monomer / switched ML dimer / MM forces (kcal/mol/A) and MM pairs.

    ``spherical_calculator`` and ``update_mm_pairs`` come from
    ``mmml_calculator.setup_calculator`` (its returned model fn and
    ``get_update_fn(...)``); outputs are in eV and eV/A.
    """
    lengths = _orthorhombic_lengths(cell)
    pair_idx, pair_mask = update_mm_pairs(np.asarray(positions), box=lengths)
    out = spherical_calculator(
        atomic_numbers=jnp.asarray(atomic_numbers, jnp.int32),
        positions=jnp.asarray(positions, jnp.float32),
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,
        doML_dimer=True,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=jnp.asarray(lengths, jnp.float32),
    )
    to_kcal = lambda a: np.asarray(a, dtype=np.float64) * KCAL_PER_EV  # noqa: E731
    return {
        "f_ml_mono": to_kcal(out.internal_F),
        "f_ml_dimer": to_kcal(out.ml_2b_F),
        "f_mm_ref": to_kcal(out.mm_F),
        "pair_idx": np.asarray(pair_idx),
        "pair_mask": np.asarray(pair_mask),
    }


def _cutoff_signature(cutoff_params: Any) -> str:
    if cutoff_params is None:
        return "None"
    attrs = getattr(cutoff_params, "__dict__", None)
    if attrs:
        return repr(sorted((k, repr(v)) for k, v in attrs.items()))
    return repr(cutoff_params)


def _mm_signature(update_mm_pairs: Any) -> bytes:
    """Bytes of the MM object's base LJ, ``at_codes``, type names and (if exposed) charges."""
    parts = [
        np.ascontiguousarray(update_mm_pairs.lj_rmins, dtype=np.float64).tobytes(),
        np.ascontiguousarray(update_mm_pairs.lj_epsilons, dtype=np.float64).tobytes(),
    ]
    for name, dtype in (("at_codes", np.int64), ("charges", np.float64)):
        val = getattr(update_mm_pairs, name, None)
        parts.append(f"{name}:".encode())
        if val is not None:
            parts.append(np.ascontiguousarray(val, dtype=dtype).tobytes())
    names = getattr(update_mm_pairs, "atc_names", None)
    parts.append(("atc:" + ("" if names is None else ",".join(str(n) for n in names))).encode())
    return b"|".join(parts)


def cache_fingerprint(
    frames: Sequence[Any],
    atoms_per_monomer: Sequence[int],
    cutoff_params: Any = None,
    extra: str = "",
    update_mm_pairs: Any = None,
) -> str:
    """SHA-256 over the cache build inputs: frame/atom counts, atomic numbers,
    raw positions and cells (float64), molecule sizes, cutoffs, ``extra`` (the
    PhysNet checkpoint path/hash) and, if given, ``update_mm_pairs``' base LJ
    (``lj_rmins``, ``lj_epsilons``), ``at_codes``, ``atc_names`` and ``charges``
    (when exposed). :func:`build_force_match_cache` always passes
    ``update_mm_pairs``; pass it here too to predict whether a cache is reused.
    """
    h = hashlib.sha256()
    n_atoms = len(frames[0]) if len(frames) else 0
    h.update(f"frames={len(frames)};atoms={n_atoms};".encode())
    h.update(np.asarray(atoms_per_monomer, dtype=np.int64).tobytes())
    for atoms in frames:
        h.update(np.asarray(atoms.get_atomic_numbers(), dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(atoms.positions, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(atoms.cell.array, dtype=np.float64).tobytes())
    h.update(_cutoff_signature(cutoff_params).encode())
    h.update(str(extra).encode())
    if update_mm_pairs is not None:
        h.update(b"mm:" + _mm_signature(update_mm_pairs))
    return h.hexdigest()


def mm_parity_max_abs(
    cache: ForceMatchCache,
    mm_energy_with_lj: MmEnergyWithLj,
    frame: int = 0,
    base_rmins: np.ndarray | None = None,
    base_epsilons: np.ndarray | None = None,
) -> float:
    """max |F_mm(base LJ) - f_mm_ref| on one frame, kcal/mol/A.

    Checks that ``energy_with_lj`` reproduces the calculator's full MM force
    (it would not, e.g., for a ``jax_pme`` hybrid, where it returns vdW only).
    ``base_rmins`` / ``base_epsilons`` default to the cache's; pass the live
    calculator's (``update_mm_pairs.lj_rmins`` / ``.lj_epsilons``) to also
    check that the cache still matches it.
    """
    pair_idx = jnp.asarray(cache.pair_idx[frame])
    pair_mask = jnp.asarray(cache.pair_mask[frame])
    cell = jnp.asarray(cache.cells[frame])
    rmins = jnp.asarray(cache.base_rmins if base_rmins is None else base_rmins)
    epsilons = jnp.asarray(cache.base_epsilons if base_epsilons is None else base_epsilons)

    def energy(r: jnp.ndarray) -> jnp.ndarray:
        return mm_energy_with_lj(r, pair_idx, pair_mask, cell, None, lj_rmins=rmins, lj_epsilons=epsilons)

    f_mm = -jax.grad(energy)(jnp.asarray(cache.positions[frame]))
    return float(np.max(np.abs(np.asarray(f_mm) - cache.f_mm_ref[frame])))


def build_force_match_cache(
    frames: Sequence[Any],
    *,
    atoms_per_monomer: Sequence[int],
    spherical_calculator: Callable[..., Any],
    update_mm_pairs: Any,
    cutoff_params: Any,
    pet_calculator: Any,
    cache_path: str | Path | None = None,
    fingerprint_extra: str = "",
    mm_parity_tol: float | None = 1e-3,
    verbose: bool = True,
) -> ForceMatchCache:
    """PET + hybrid force components for ASE ``frames`` (with cell); cached to npz.

    An existing ``cache_path`` is reused only if its fingerprint
    (:func:`cache_fingerprint` of ``frames``, ``atoms_per_monomer``,
    ``cutoff_params``, ``fingerprint_extra`` and ``update_mm_pairs``' base LJ,
    atom types and charges) matches and its ``base_rmins`` / ``base_epsilons``
    equal ``update_mm_pairs.lj_rmins`` / ``.lj_epsilons``; otherwise it is
    rebuilt and overwritten. The PhysNet checkpoint is not visible to this
    function: pass its path (or a hash) as ``fingerprint_extra``, else a
    changed checkpoint silently reuses stale ML forces (a warning is issued
    when it is empty and ``cache_path`` is set). ``pet_calculator`` may be
    None only when the cache is reused.

    Only hybrids whose ``update_mm_pairs.energy_with_lj`` returns the full MM
    energy are supported: with ``lr_solver='jax_pme'`` it returns vdW only, so
    F_mm(theta) would silently drop the PME Coulomb force. Unless
    ``mm_parity_tol`` is None, F_mm at the base LJ is compared with the
    calculator's MM force on frame 0 (both at the calculator's current base
    LJ) and a ``ValueError`` is raised if the max abs difference exceeds
    ``mm_parity_tol`` (kcal/mol/A).
    """
    if cache_path is not None and not fingerprint_extra:
        warnings.warn(
            "build_force_match_cache: fingerprint_extra is empty, so the cache at "
            f"{cache_path} is not tied to the PhysNet checkpoint; pass the checkpoint path",
            stacklevel=2,
        )
    base_rmins = np.asarray(update_mm_pairs.lj_rmins, dtype=np.float64)
    base_epsilons = np.asarray(update_mm_pairs.lj_epsilons, dtype=np.float64)
    fingerprint = cache_fingerprint(
        frames, atoms_per_monomer, cutoff_params, fingerprint_extra, update_mm_pairs=update_mm_pairs
    )
    cache = None
    if cache_path is not None and Path(cache_path).exists():
        cached = ForceMatchCache.load(cache_path)
        if cached.fingerprint != fingerprint:
            reason = "fingerprint mismatch"
        elif not (
            cached.base_rmins.shape == base_rmins.shape
            and np.allclose(cached.base_rmins, base_rmins, rtol=0.0, atol=1e-12)
            and np.allclose(cached.base_epsilons, base_epsilons, rtol=0.0, atol=1e-12)
        ):
            reason = "base LJ differs from update_mm_pairs"
        else:
            reason = ""
            cache = cached
        if reason and verbose:
            print(f"[force_match] {cache_path}: {reason}, rebuilding", flush=True)
    if cache is None:
        if pet_calculator is None:
            raise ValueError("force-match cache must be (re)built but pet_calculator is None")
        cache = _compute_force_match_cache(
            frames,
            atoms_per_monomer=atoms_per_monomer,
            spherical_calculator=spherical_calculator,
            update_mm_pairs=update_mm_pairs,
            cutoff_params=cutoff_params,
            pet_calculator=pet_calculator,
            fingerprint=fingerprint,
            verbose=verbose,
        )
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            cache.save(cache_path)
    if mm_parity_tol is not None:
        diff = mm_parity_max_abs(cache, update_mm_pairs.energy_with_lj, 0, base_rmins, base_epsilons)
        if not diff <= mm_parity_tol:
            raise ValueError(
                f"energy_with_lj MM force differs from the calculator MM force by {diff:.3g} "
                f"kcal/mol/A (> {mm_parity_tol}); PME-Coulomb (jax_pme) hybrids are unsupported"
            )
    return cache


def _compute_force_match_cache(
    frames: Sequence[Any],
    *,
    atoms_per_monomer: Sequence[int],
    spherical_calculator: Callable[..., Any],
    update_mm_pairs: Any,
    cutoff_params: Any,
    pet_calculator: Any,
    fingerprint: str,
    verbose: bool,
) -> ForceMatchCache:
    per_frame = []
    for k, atoms in enumerate(frames):
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        z = np.asarray(atoms.get_atomic_numbers())
        x = make_molecules_whole(atoms.positions, atoms_per_monomer, _orthorhombic_lengths(cell))
        comps = hybrid_force_components(
            spherical_calculator, update_mm_pairs, z, x, cell, len(atoms_per_monomer), cutoff_params
        )
        comps.update(positions=x, cells=cell, f_pet=pet_forces_kcal(pet_calculator, z, x, cell))
        per_frame.append(comps)
        if verbose:
            print(f"[force_match] frame {k + 1}/{len(frames)} cached", flush=True)
    return ForceMatchCache.from_frames(
        per_frame,
        base_rmins=np.asarray(update_mm_pairs.lj_rmins),
        base_epsilons=np.asarray(update_mm_pairs.lj_epsilons),
        molecule_index=molecule_index_from_sizes(atoms_per_monomer),
        fingerprint=fingerprint,
    )
