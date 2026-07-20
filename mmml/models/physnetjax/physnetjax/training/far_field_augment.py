"""Synthetic "far-field replica" batches for size-extensivity training.

Motivation (see docs/... investigation of SO3LR/Spooky checkpoints, summer
2026): the model's electrostatics term is a full pairwise Coulomb sum over
predicted charges, unrestricted by the short-range message-passing cutoff --
so it has no architectural reason to stay well-behaved as system size grows,
and measurement confirmed it doesn't (charge conservation and per-atom charge
magnitude both degrade badly on >120-atom test structures, far outside the
~14-19 atom training distribution). Rather than collecting new large-structure
reference data, this module generates exact training signal for size
extensivity FOR FREE, by recombining structures already in the cache:

    E(K well-separated, non-interacting molecules) = sum_k E(molecule_k)

is true by definition once every cross-fragment atom pair is farther apart
than both terms' interaction ranges can reach, regardless of what the
molecules are. See ``SAFE_SEPARATION_ANGSTROM`` for why 12 A is sufficient.

v1 scope, deliberately: only combines EXACTLY NEUTRAL, SINGLET fragments
(``|Q| < NEUTRAL_TOL`` and ``S == 1.0``). This is what makes the dipole
superposition exact regardless of translation (a neutral fragment's dipole is
translation-invariant: mu_frag(O) = sum_i q_i*(r_i - O) = sum_i q_i*r_i since
sum_i q_i = 0, i.e. independent of the origin O), and keeps the combined
system's spin trivially singlet. Charged/open-shell composites are not
handled here.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict

import numpy as np

NEUTRAL_TOL = 1e-6
SINGLET_SPIN = 1.0
DEFAULT_SAFETY_MARGIN_ANGSTROM = 2.0


def compute_safe_separation(
    *,
    cutoff: float,
    electrostatics_off_end: float,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_ANGSTROM,
) -> float:
    """Minimum atom-atom separation guaranteeing exactly zero cross-fragment interaction.

    Deliberately takes the model's ACTUAL configured ``cutoff`` and
    ``electrostatics_off_end`` as explicit arguments rather than a hardcoded
    constant -- a bare module-level number here would silently go stale if
    either of those changes (this happened once already: this module used to
    hardcode 12.0, derived from an undocumented assumption of cutoff=6.0 and
    electrostatics_off_end=10.0 that nothing checked against the model's
    actual configuration -- see spooky_model.py's SpookyPhysNet.switch_end /
    electrostatics_off_end fields, now themselves configurable for the same
    reason instead of being hardcoded inside _calc_switches).

    Both binding constraints on interaction range are HARD (exactly zero,
    not just small), verified numerically against this model family's code:
      - message-passing radial basis (e3x.nn.smooth_cutoff(d, cutoff=...))
        is piecewise-defined as exactly 0 for d >= cutoff -- and since
        disjoint fragments have NO edge between them once every
        cross-fragment pair exceeds this cutoff, num_iterations (multi-hop
        propagation) is irrelevant: there is no chain of <=cutoff-length
        hops that can ever bridge two fragments with zero direct edges to
        begin with.
      - electrostatics gating (_calc_switches' off_dist) reaches exactly 0
        at d >= electrostatics_off_end.

    Returns ``max(cutoff, electrostatics_off_end) + safety_margin``: the
    larger of the two hard-zero points, whichever one is currently binding,
    plus a numerical safety margin.
    """
    return max(float(cutoff), float(electrostatics_off_end)) + float(safety_margin)


# Kept ONLY for callers that haven't been updated to pass explicit model
# parameters; matches the current SpookyPhysNet defaults
# (cutoff=6.0, electrostatics_off_end=10.0). Prefer compute_safe_separation().
SAFE_SEPARATION_ANGSTROM = compute_safe_separation(cutoff=6.0, electrostatics_off_end=10.0)


def _fragment_radius(positions: np.ndarray) -> float:
    """Max distance from the centroid to any atom (bounding-sphere radius)."""
    centroid = positions.mean(axis=0)
    return float(np.max(np.linalg.norm(positions - centroid, axis=1))) if len(positions) else 0.0


def eligible_source_indices(
    data: Dict[str, Any], *, max_fragment_atoms: int | None = None
) -> np.ndarray:
    """Indices of structures usable as far-field-composite fragments.

    Restricted to exactly neutral, singlet structures -- see module
    docstring for why that's what makes the construction exact.

    ``max_fragment_atoms``, if given, additionally excludes any structure
    with more atoms than this. Source structure sizes are heavy-tailed (in
    practice: median ~18 atoms, but a long tail up to the largest real
    training structures, e.g. 120-atom proteins, that happen to be exactly
    neutral and singlet) -- without this cap, ``build_far_field_composites``
    can only bound the EXPECTED composite size, not the worst case, since a
    handful of unlucky draws from the tail is enough to produce a composite
    far larger than ``k_max`` times the typical fragment size. Callers that
    need a hard, exact worst-case bound on composite atom count (e.g. to
    stay within a known-safe pad-width range and avoid probing/OOMing on
    novel shapes -- see the resolve_composite_max_atoms) must set this.
    """
    q = np.asarray(data["Q"], dtype=np.float64).reshape(-1)
    s = np.asarray(data["S"], dtype=np.float64).reshape(-1)
    mask = (np.abs(q) < NEUTRAL_TOL) & (np.abs(s - SINGLET_SPIN) < 1e-9)
    if max_fragment_atoms is not None:
        n = np.asarray(data["N"], dtype=np.int64).reshape(-1)
        mask = mask & (n <= int(max_fragment_atoms))
    return np.flatnonzero(mask)


def resolve_composite_max_atoms(*, k_max: int, max_fragment_atoms: int) -> int:
    """Exact worst-case atom count for a composite of up to ``k_max`` fragments,
    each capped at ``max_fragment_atoms`` atoms by :func:`eligible_source_indices`.

    Use this to choose ``max_fragment_atoms`` so composites stay within a
    known-safe atom-count ceiling (e.g. the largest already-probed
    auto-batch pad width), guaranteeing no novel, unprobed shape is ever
    produced -- rather than hoping ``k_max`` alone keeps composites small,
    which the heavy-tailed fragment-size distribution makes unreliable.
    """
    return int(k_max) * int(max_fragment_atoms)


def _place_fragments_on_grid(radii: list[float], safe_separation: float) -> list[np.ndarray]:
    """Grid-point translation per fragment guaranteeing no cross-fragment atom
    pair can be closer than ``safe_separation``, for any fragment orientation.

    Using bounding-sphere radii: placing centroids >= r_i + r_j + safe_separation
    apart guarantees this for every pair (i, j). A uniform cubic grid with
    spacing = 2*max(radii) + safe_separation is a simple sufficient
    (slightly conservative) construction satisfying that for all pairs at
    once.
    """
    k = len(radii)
    spacing = 2.0 * max(radii, default=0.0) + safe_separation
    n_side = max(1, int(np.ceil(k ** (1.0 / 3.0))))
    grid_points = itertools.islice(itertools.product(range(n_side), repeat=3), k)
    return [np.array(pt, dtype=np.float64) * spacing for pt in grid_points]


def build_one_far_field_composite(
    data: Dict[str, Any],
    fragment_mol_indices: np.ndarray,
    *,
    safe_separation: float = SAFE_SEPARATION_ANGSTROM,
) -> Dict[str, Any]:
    """Build one synthetic composite structure from K real structures.

    Parameters
    ----------
    data
        The flat cached dict (must contain mol_offsets, R, Z, F, E, Q, S, D).
    fragment_mol_indices
        Indices (into ``data``'s structures) of the K fragments to combine.
        Caller is responsible for restricting these to
        :func:`eligible_source_indices`.

    Returns
    -------
    dict with keys R, Z, F, mol_id (all flat, this structure's atoms only),
    E, Q, S, D, N (scalars for this one structure).
    """
    mol_offsets = np.asarray(data["mol_offsets"], dtype=np.int64)
    r_all = np.asarray(data["R"], dtype=np.float64)
    z_all = np.asarray(data["Z"], dtype=np.int32)
    f_all = np.asarray(data["F"], dtype=np.float64)
    e_all = np.asarray(data["E"], dtype=np.float64).reshape(-1)
    q_all = np.asarray(data["Q"], dtype=np.float64).reshape(-1)
    d_all = np.asarray(data.get("D"), dtype=np.float64).reshape(-1, 3) if "D" in data else None

    fragments_r, fragments_z, fragments_f, radii = [], [], [], []
    e_sum, q_sum = 0.0, 0.0
    d_sum = np.zeros(3) if d_all is not None else None
    for mi in fragment_mol_indices:
        mi = int(mi)
        a0, a1 = int(mol_offsets[mi]), int(mol_offsets[mi + 1])
        pos = r_all[a0:a1]
        centroid = pos.mean(axis=0)
        pos_centered = pos - centroid  # own local frame, centroid at origin
        fragments_r.append(pos_centered)
        fragments_z.append(z_all[a0:a1])
        fragments_f.append(f_all[a0:a1])  # forces are frame-independent under rigid translation
        radii.append(_fragment_radius(pos_centered))
        e_sum += float(e_all[mi])
        q_sum += float(q_all[mi])
        if d_all is not None:
            d_sum = d_sum + d_all[mi]  # valid because every fragment is exactly neutral (see module docstring)

    translations = _place_fragments_on_grid(radii, safe_separation)

    r_parts, z_parts, f_parts, mol_id_parts = [], [], [], []
    for frag_idx, (pos_centered, z, f, t) in enumerate(zip(fragments_r, fragments_z, fragments_f, translations)):
        r_parts.append(pos_centered + t)
        z_parts.append(z)
        f_parts.append(f)
        mol_id_parts.append(np.full(len(z), frag_idx, dtype=np.int32))

    composite = {
        "R": np.concatenate(r_parts, axis=0),
        "Z": np.concatenate(z_parts, axis=0),
        "F": np.concatenate(f_parts, axis=0),
        "mol_id": np.concatenate(mol_id_parts, axis=0),
        "E": e_sum,
        "Q": q_sum,
        "S": SINGLET_SPIN,
        "N": sum(len(z) for z in z_parts),
    }
    if d_sum is not None:
        composite["D"] = d_sum
    return composite


def build_far_field_composites(
    data: Dict[str, Any],
    rng: np.random.Generator,
    *,
    n_composites: int,
    k_min: int = 5,
    k_max: int = 50,
    safe_separation: float = SAFE_SEPARATION_ANGSTROM,
    max_fragment_atoms: int | None = None,
) -> list[Dict[str, Any]]:
    """Sample ``n_composites`` synthetic far-field structures.

    Fragment counts are drawn uniformly from ``[k_min, k_max]``; fragments
    are sampled with replacement from :func:`eligible_source_indices`.

    ``max_fragment_atoms``, if given, is forwarded to
    :func:`eligible_source_indices` so every composite's worst-case atom
    count is bounded exactly by ``k_max * max_fragment_atoms`` (see
    :func:`resolve_composite_max_atoms`) -- without it, composite size is
    only bounded in expectation, and a heavy-tailed fragment-size
    distribution can produce composites far larger than ``k_max`` times the
    typical fragment size (observed: k_max=6 with no cap produced a
    394-atom composite from a population with median fragment size 18).
    """
    eligible = eligible_source_indices(data, max_fragment_atoms=max_fragment_atoms)
    if eligible.size == 0:
        raise ValueError(
            "No exactly-neutral, singlet structures found in this cache "
            + (
                f"with <= {max_fragment_atoms} atoms "
                if max_fragment_atoms is not None
                else ""
            )
            + "-- far-field composites need at least one eligible source structure."
        )
    composites = []
    for _ in range(n_composites):
        k = int(rng.integers(k_min, k_max + 1))
        frag_indices = rng.choice(eligible, size=k, replace=True)
        composites.append(build_one_far_field_composite(data, frag_indices, safe_separation=safe_separation))
    return composites


def append_far_field_composites_to_data(
    data: Dict[str, Any],
    composites: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a NEW flat data dict with ``composites`` appended as extra structures.

    Adds ``mol_id`` (per-atom fragment id, 0 for every atom of every
    pre-existing real structure) and ``cgenff_type_idx`` (all-zero dummy --
    never paired with cgenff_master_sigmas/epsilons, so CGenFF-vdW stays
    correctly inactive; its only purpose is satisfying
    build_spooky_batch_from_flat_data's ``has_mm = "cgenff_type_idx" in
    data_dict and "mol_id" in data_dict`` gate so mol_id gets threaded
    through at all) if not already present.
    """
    mol_offsets = np.asarray(data["mol_offsets"], dtype=np.int64)
    n_atoms_orig = int(mol_offsets[-1])
    n_structures_orig = int(mol_offsets.shape[0] - 1)

    mol_id_orig = data.get("mol_id")
    if mol_id_orig is None:
        mol_id_orig = np.zeros(n_atoms_orig, dtype=np.int32)
    cgenff_orig = data.get("cgenff_type_idx")
    if cgenff_orig is None:
        cgenff_orig = np.zeros(n_atoms_orig, dtype=np.int32)

    r_new = [np.asarray(data["R"], dtype=np.float64)]
    z_new = [np.asarray(data["Z"], dtype=np.int32)]
    f_new = [np.asarray(data["F"], dtype=np.float64)]
    mol_id_new = [np.asarray(mol_id_orig, dtype=np.int32)]
    cgenff_new = [np.asarray(cgenff_orig, dtype=np.int32)]
    e_new = [np.asarray(data["E"], dtype=np.float64).reshape(-1)]
    q_new = [np.asarray(data["Q"], dtype=np.float64).reshape(-1)]
    s_new = [np.asarray(data["S"], dtype=np.float64).reshape(-1)]
    n_new = [np.asarray(data["N"], dtype=np.int32).reshape(-1)]
    has_d = "D" in data
    d_new = [np.asarray(data["D"], dtype=np.float64).reshape(-1, 3)] if has_d else None

    offsets = [mol_offsets]
    running_atom_total = n_atoms_orig

    for comp in composites:
        r_new.append(comp["R"])
        z_new.append(comp["Z"])
        f_new.append(comp["F"])
        mol_id_new.append(comp["mol_id"])
        cgenff_new.append(np.zeros(comp["N"], dtype=np.int32))
        e_new.append(np.array([comp["E"]]))
        q_new.append(np.array([comp["Q"]]))
        s_new.append(np.array([comp["S"]]))
        n_new.append(np.array([comp["N"]], dtype=np.int32))
        if has_d:
            d_new.append(comp["D"].reshape(1, 3))
        running_atom_total += comp["N"]
        offsets.append(np.array([running_atom_total], dtype=np.int64))

    out = dict(data)
    out["R"] = np.concatenate(r_new, axis=0)
    out["Z"] = np.concatenate(z_new, axis=0)
    out["F"] = np.concatenate(f_new, axis=0)
    out["mol_id"] = np.concatenate(mol_id_new, axis=0)
    out["cgenff_type_idx"] = np.concatenate(cgenff_new, axis=0)
    out["E"] = np.concatenate(e_new, axis=0).reshape(-1, 1)
    out["Q"] = np.concatenate(q_new, axis=0).reshape(-1, 1)
    out["S"] = np.concatenate(s_new, axis=0).reshape(-1, 1)
    out["N"] = np.concatenate(n_new, axis=0).reshape(-1, 1)
    if has_d:
        out["D"] = np.concatenate(d_new, axis=0)
    out["mol_offsets"] = np.concatenate(offsets, axis=0)
    out["is_far_field_composite"] = np.concatenate(
        [np.zeros(n_structures_orig, dtype=bool), np.ones(len(composites), dtype=bool)]
    )
    return out
