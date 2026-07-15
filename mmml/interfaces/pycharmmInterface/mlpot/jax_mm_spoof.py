"""JAX CGenFF bonded clone as a stand-in for PhysNet in hybrid MLpot (testing).

Replaces checkpoint-based ML monomer/dimer evaluations with pure-JAX bonded
terms (and a simple inter-monomer repulsion on dimer batches) so box builds,
neighbor lists, COM switching, and calculator wiring can be exercised without
a PhysNet checkpoint or GPU.

Supports heterogeneous monomer sizes: each unique atom count gets its own
bonded evaluator, and dimer batches receive an explicit ``N_a`` split so a
3+3 dimer is not confused with a 6-atom monomer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md.mm_forcefields.base import BondedParameters, Topology
from jax_md.mm_forcefields.oplsaa.topology import create_topology

from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    KCAL_MOL_TO_EV,
    build_bonded_energy_fn,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    load_cgenff_bonded_from_psf,
)

BondedEvalFn = Callable[[Array], tuple[Array, Array]]


def minimal_chain_bonded_system(
    n_atoms: int,
    *,
    bond_k: float = 450.0,
    bond_r0: float = 1.54,
    angle_k: float = 60.0,
    angle_theta0: float = 1.91,
) -> tuple[Topology, BondedParameters]:
    """Harmonic chain bonded topology for synthetic toy monomers (no PSF)."""
    if n_atoms < 2:
        raise ValueError(f"n_atoms must be >= 2 for a chain, got {n_atoms}")
    bonds = jnp.stack(
        [jnp.arange(n_atoms - 1, dtype=jnp.int32), jnp.arange(1, n_atoms, dtype=jnp.int32)],
        axis=1,
    )
    angles = (
        jnp.stack(
            [
                jnp.arange(n_atoms - 2, dtype=jnp.int32),
                jnp.arange(1, n_atoms - 1, dtype=jnp.int32),
                jnp.arange(2, n_atoms, dtype=jnp.int32),
            ],
            axis=1,
        )
        if n_atoms >= 3
        else jnp.zeros((0, 3), dtype=jnp.int32)
    )
    topology = create_topology(
        n_atoms=n_atoms,
        bonds=bonds,
        angles=angles,
        torsions=jnp.zeros((0, 4), dtype=jnp.int32),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        molecule_id=jnp.zeros(n_atoms, dtype=jnp.int32),
    )
    bonded = BondedParameters(
        bond_k=jnp.full((n_atoms - 1,), float(bond_k)),
        bond_r0=jnp.full((n_atoms - 1,), float(bond_r0)),
        angle_k=jnp.full((angles.shape[0],), float(angle_k)),
        angle_theta0=jnp.full((angles.shape[0],), float(angle_theta0)),
        torsion_k=jnp.zeros(0),
        torsion_n=jnp.zeros(0, dtype=jnp.int32),
        torsion_gamma=jnp.zeros(0),
        improper_k=jnp.zeros(0),
        improper_n=jnp.zeros(0, dtype=jnp.int32),
        improper_gamma=jnp.zeros(0),
        cmap_maps=None,
    )
    return topology, bonded


def _remap_indices(indices: Array | None, start: int) -> Array | None:
    if indices is None:
        return None
    arr = np.asarray(indices)
    if arr.size == 0:
        return jnp.asarray(indices)
    return jnp.asarray(arr - int(start))


def _topology_slice_to_local(
    topology: Topology,
    *,
    start: int,
    n_atoms: int,
) -> Topology:
    """Remap global bonded indices in ``[start, start+n)`` to local ``[0, n)``."""
    return Topology(
        n_atoms=int(n_atoms),
        bonds=_remap_indices(topology.bonds, start),
        angles=_remap_indices(topology.angles, start),
        torsions=_remap_indices(topology.torsions, start),
        impropers=_remap_indices(topology.impropers, start),
        exclusion_mask=None,
        pair_14_mask=None,
        molecule_id=jnp.zeros(int(n_atoms), dtype=jnp.int32),
        cmap_atoms=_remap_indices(topology.cmap_atoms, start),
        cmap_map_idx=topology.cmap_map_idx,
        exc_pairs=None,
        nbfix_atom_type=None,
    )


def load_monomer_bonded_evaluator_from_psf(
    psf_path: Path | str,
    *,
    atoms_per_monomer: int,
    atom_offset: int = 0,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    """Bonded evaluator for one monomer slice from a cluster PSF."""
    psf = Path(psf_path).expanduser().resolve()
    n_mono = int(atoms_per_monomer)
    start = int(atom_offset)
    end = start + n_mono
    positions = jnp.zeros((max(end, 1), 3), dtype=jnp.float64)
    system = load_cgenff_bonded_from_psf(psf, positions)
    if system.n_atoms < end:
        raise ValueError(
            f"PSF {psf} has {system.n_atoms} atoms; need at least {end} "
            f"(offset={start}, atoms_per_monomer={n_mono})"
        )
    from mmml.interfaces.pycharmmInterface.cgenff_topology import filter_bonded_topology_for_mm

    mm_mask = jnp.zeros(system.n_atoms, dtype=bool)
    mm_mask = mm_mask.at[start:end].set(True)
    topology, bonded, urey_k, urey_r0 = filter_bonded_topology_for_mm(
        system.topology,
        system.bonded,
        mm_mask,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
    )
    topology = _topology_slice_to_local(topology, start=start, n_atoms=n_mono)
    bonded_fn = build_bonded_energy_fn(
        topology,
        bonded,
        urey_k=urey_k,
        urey_r0=urey_r0,
        energy_unit=energy_unit,
    )

    def evaluate(positions: Array) -> tuple[Array, Array]:
        pos = positions[:n_mono]
        components, forces = bonded_fn(pos)
        return components["total"], forces

    return evaluate


def build_minimal_chain_bonded_evaluator(
    n_atoms: int,
    *,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    n = int(n_atoms)
    if n < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n}")
    if n == 1:
        def evaluate_single(positions: Array) -> tuple[Array, Array]:
            pos = positions[:1]
            return jnp.asarray(0.0, dtype=pos.dtype), jnp.zeros_like(pos)

        return evaluate_single

    topology, bonded = minimal_chain_bonded_system(n)
    bonded_fn = build_bonded_energy_fn(topology, bonded, energy_unit=energy_unit)

    def evaluate(positions: Array) -> tuple[Array, Array]:
        components, forces = bonded_fn(positions[:n])
        return components["total"], forces

    return evaluate


def _inter_monomer_soft_repulsion(
    positions: Array,
    n_a: Array | int,
    n_b: Array | int | None = None,
    *,
    epsilon_ev: float = 0.01,
    sigma_a: float = 3.4,
    r_floor_a: float = 1.5,
) -> tuple[Array, Array]:
    """Vacuum r^-12 repulsion between atoms in monomer A vs monomer B.

    ``n_a`` / ``n_b`` may be traced integers (heterogeneous dimer batches).
    When ``n_b`` is omitted, B is taken as the trailing ``max_atoms - n_a``
    window (uniform path).

    Computed on the A×B rectangular block only (never N×N): a full pairwise
    ``1/r^12`` matrix hits the diagonal (r=0) and overflows to Inf under
    float32 before any mask can zero it.
    """
    max_n = int(positions.shape[0])
    # Cap how large a rectangular block we materialize (jit shape).
    max_mono = max_n
    na = jnp.asarray(n_a, dtype=jnp.int32)
    if n_b is None:
        nb = jnp.asarray(max_n, dtype=jnp.int32) - na
    else:
        nb = jnp.asarray(n_b, dtype=jnp.int32)

    r_ext = jnp.concatenate([positions, jnp.zeros_like(positions)], axis=0)
    pos_a = jax.lax.dynamic_slice(r_ext, (jnp.asarray(0, dtype=na.dtype), 0), (max_mono, 3))
    pos_b = jax.lax.dynamic_slice(r_ext, (na, jnp.asarray(0, dtype=na.dtype)), (max_mono, 3))
    # Mask padded slots past the true monomer sizes.
    ia = jnp.arange(max_mono, dtype=jnp.int32)
    ib = jnp.arange(max_mono, dtype=jnp.int32)
    mask_a = ia < na
    mask_b = ib < nb
    pair_mask = mask_a[:, None] & mask_b[None, :]

    diff = pos_a[:, None, :] - pos_b[None, :, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    r2 = jnp.maximum(r2, float(r_floor_a) ** 2)
    r6 = r2**3
    r12 = r6 * r6
    sig6 = float(sigma_a) ** 6
    sig12 = sig6 * sig6
    pair_e = float(epsilon_ev) * sig12 / r12
    energy = jnp.sum(jnp.where(pair_mask, pair_e, 0.0))
    coeff = -12.0 * float(epsilon_ev) * sig12 / (r12 * r2)
    coeff = jnp.where(pair_mask, coeff, 0.0)
    f_a = jnp.sum(coeff[:, :, None] * diff, axis=1)
    f_b = -jnp.sum(coeff[:, :, None] * diff, axis=0)

    forces = jnp.zeros_like(positions)
    idx = jnp.arange(max_n, dtype=jnp.int32)
    in_a = idx < na
    in_b = (idx >= na) & (idx < (na + nb))
    safe_a = jnp.where(in_a, idx, 0)
    safe_b = jnp.where(in_b, idx - na, 0)
    forces = jnp.where(in_a[:, None], f_a[safe_a], forces)
    forces = jnp.where(in_b[:, None], f_b[safe_b], forces)
    return energy, forces


def _build_size_switch_eval(
    monomer_evals: Mapping[int, BondedEvalFn],
    *,
    max_atoms: int,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """``(R, n) -> (energy, forces_padded)`` with ``forces`` shape ``(max_atoms, 3)``."""
    unique_sizes = sorted(int(s) for s in monomer_evals.keys())
    if not unique_sizes:
        raise ValueError("monomer_evals must be non-empty")
    size_arr = jnp.asarray(unique_sizes, dtype=jnp.int32)
    max_mono = max(unique_sizes)
    if max_mono > max_atoms:
        raise ValueError(
            f"largest monomer size {max_mono} exceeds max_atoms={max_atoms}"
        )

    branches: list[Callable[[Array], tuple[Array, Array]]] = []
    for sz in unique_sizes:
        eval_fn = monomer_evals[sz]

        def _branch(R: Array, sz: int = sz, eval_fn: BondedEvalFn = eval_fn) -> tuple[Array, Array]:
            e, f = eval_fn(R[:sz])
            f_pad = jnp.zeros((max_atoms, 3), dtype=R.dtype).at[:sz].set(f)
            return e, f_pad

        branches.append(_branch)

    def eval_padded(R: Array, n: Array) -> tuple[Array, Array]:
        idx = jnp.argmax(size_arr == jnp.asarray(n, dtype=jnp.int32))
        return jax.lax.switch(idx, branches, R)

    return eval_padded


def build_jax_mm_spoof_batch_apply(
    *,
    atoms_per_monomer: int | Sequence[int],
    max_atoms: int,
    monomer_eval: BondedEvalFn | Mapping[int, BondedEvalFn] | None = None,
    monomer_evals: Mapping[int, BondedEvalFn] | None = None,
    include_soft_repulsion: bool = True,
) -> Callable[..., dict[str, Array]]:
    """Return ``apply_model(Z, R, N, N_a=None)`` compatible with MLpot batching.

    ``N_a`` is the first-fragment atom count: for monomers ``N_a == N``; for
    dimers ``N_a`` is the size of monomer A and ``N == N_a + N_b``.  When
    omitted, a uniform ``atoms_per_monomer`` heuristic is used (legacy).

    ``include_soft_repulsion`` adds a toy A↔B r^-12 for dimer batches.  Disable
    it when a CGenFF PSF spoof is paired with hybrid ``doMM`` (JAX MM already
    carries inter-monomer nonbond).
    """
    if isinstance(atoms_per_monomer, (list, tuple)):
        per_list = [int(x) for x in atoms_per_monomer]
    else:
        per_list = [int(atoms_per_monomer)]

    evals: dict[int, BondedEvalFn]
    if monomer_evals is not None:
        evals = {int(k): v for k, v in monomer_evals.items()}
    elif isinstance(monomer_eval, Mapping):
        evals = {int(k): v for k, v in monomer_eval.items()}
    elif monomer_eval is not None:
        if len(set(per_list)) != 1:
            raise ValueError(
                "heterogeneous atoms_per_monomer requires monomer_evals "
                "(or a size->evaluator mapping)"
            )
        evals = {per_list[0]: monomer_eval}
    else:
        raise ValueError("monomer_eval or monomer_evals is required")

    for sz in set(per_list):
        if sz not in evals:
            raise ValueError(
                f"missing bonded evaluator for atoms_per_monomer={sz}; "
                f"have sizes {sorted(evals)}"
            )

    mono_n_uniform = per_list[0] if len(set(per_list)) == 1 else None
    eval_padded = _build_size_switch_eval(evals, max_atoms=max_atoms)
    # Pad so dynamic_slice(start=na, size=max_atoms) never runs past the end.
    slice_pad = max_atoms
    use_soft = bool(include_soft_repulsion)

    def _eval_one(R: Array, N: Array, N_a: Array) -> tuple[Array, Array]:
        na = jnp.asarray(N_a, dtype=jnp.int32)
        n_tot = jnp.asarray(N, dtype=jnp.int32)
        nb = n_tot - na
        is_dimer = n_tot > na

        def _take_mono(_):
            return eval_padded(R, n_tot)

        def _take_dimer(_):
            r_ext = jnp.concatenate([R, jnp.zeros_like(R[:slice_pad])], axis=0)
            window_b = jax.lax.dynamic_slice(
                r_ext,
                (na, jnp.asarray(0, dtype=na.dtype)),
                (max_atoms, 3),
            )
            e_a, f_a = eval_padded(R, na)
            e_b, f_b_window = eval_padded(window_b, nb)
            if use_soft:
                e_nb, f_nb = _inter_monomer_soft_repulsion(R, na, nb)
            else:
                e_nb = jnp.asarray(0.0, dtype=R.dtype)
                f_nb = jnp.zeros_like(R)
            idx = jnp.arange(max_atoms, dtype=jnp.int32)
            in_b = (idx >= na) & (idx < n_tot)
            safe_local = jnp.where(in_b, idx - na, 0)
            f_b = jnp.where(in_b[:, None], f_b_window[safe_local], 0.0)
            return e_a + e_b + e_nb, f_a + f_b + f_nb

        return jax.lax.cond(is_dimer, _take_dimer, _take_mono, operand=None)

    vmapped = jax.vmap(_eval_one, in_axes=(0, 0, 0))

    def apply_model(
        atomic_numbers: Array,
        positions: Array,
        batch_n: Array,
        batch_n_a: Array | None = None,
    ) -> dict[str, Array]:
        _ = atomic_numbers  # Z unused; topology is fixed in spoof mode
        batch_size = positions.shape[0] // max_atoms
        R = positions.reshape(batch_size, max_atoms, 3)
        N = jnp.asarray(batch_n, dtype=jnp.int32).reshape(batch_size)
        if batch_n_a is None:
            if mono_n_uniform is None:
                raise ValueError(
                    "batch_n_a is required for heterogeneous jax_mm_clone spoof"
                )
            mono_n_j = jnp.asarray(mono_n_uniform, dtype=jnp.int32)
            N_a = jnp.where(N > mono_n_j, mono_n_j, N)
        else:
            N_a = jnp.asarray(batch_n_a, dtype=jnp.int32).reshape(batch_size)
        energies, forces = vmapped(R, N, N_a)
        return {
            "energy": energies.reshape(batch_size),
            "forces": forces.reshape(batch_size * max_atoms, 3),
        }

    return apply_model


def resolve_monomer_bonded_evaluator(
    *,
    atoms_per_monomer: int,
    monomer_psf: Path | str | None = None,
    atom_offset: int = 0,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    if monomer_psf is not None:
        return load_monomer_bonded_evaluator_from_psf(
            monomer_psf,
            atoms_per_monomer=atoms_per_monomer,
            atom_offset=atom_offset,
            energy_unit=energy_unit,
        )
    return build_minimal_chain_bonded_evaluator(atoms_per_monomer, energy_unit=energy_unit)


def resolve_monomer_bonded_evaluators(
    atoms_per_monomer: Sequence[int],
    *,
    monomer_psf: Path | str | None = None,
    energy_unit: str = "eV",
) -> dict[int, BondedEvalFn]:
    """One bonded evaluator per unique monomer size (first PSF occurrence wins)."""
    per = [int(x) for x in atoms_per_monomer]
    if not per:
        raise ValueError("atoms_per_monomer must be non-empty")
    offsets = np.cumsum([0, *per])[:-1]
    out: dict[int, BondedEvalFn] = {}
    for i, n in enumerate(per):
        if n in out:
            continue
        out[n] = resolve_monomer_bonded_evaluator(
            atoms_per_monomer=n,
            monomer_psf=monomer_psf,
            atom_offset=int(offsets[i]),
            energy_unit=energy_unit,
        )
    return out


def jax_mm_spoof_enabled(args: object | None) -> bool:
    if args is None:
        return False
    if bool(getattr(args, "jax_mm_spoof", False)):
        return True
    mode = str(getattr(args, "ml_potential_mode", "") or "").strip().lower()
    return mode in {"jax_mm_clone", "jax-mm-clone", "jax_mm_spoof"}


__all__ = [
    "KCAL_MOL_TO_EV",
    "BondedEvalFn",
    "build_jax_mm_spoof_batch_apply",
    "build_minimal_chain_bonded_evaluator",
    "jax_mm_spoof_enabled",
    "load_monomer_bonded_evaluator_from_psf",
    "minimal_chain_bonded_system",
    "resolve_monomer_bonded_evaluator",
    "resolve_monomer_bonded_evaluators",
]
