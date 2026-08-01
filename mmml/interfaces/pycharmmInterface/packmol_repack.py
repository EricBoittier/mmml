"""Packmol-backed monomer repack for MD recovery (selective or full)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.packmol_placement import (
    _parse_pdb_atom_records,
    execute_packmol_script,
    packmol_cube_origin,
    write_monomer_pdb_for_packmol,
)


def _cell_box_side(cell: Any | None) -> float | None:
    if cell is None:
        return None
    arr = np.asarray(cell, dtype=float)
    if arr.ndim == 0:
        side = float(arr)
        return side if side > 0.0 else None
    if arr.shape == (1,):
        side = float(arr[0])
        return side if side > 0.0 else None
    if arr.shape == (3,):
        return float(arr[0]) if float(arr[0]) > 0.0 else None
    if arr.shape == (3, 3):
        return float(arr[0, 0]) if float(arr[0, 0]) > 0.0 else None
    return None


#: Packmol's own default tolerance (Å), also used by the initial cluster pack.
PACKMOL_DEFAULT_TOLERANCE_A = 2.0

#: Max deviation (Å) between two monomers' internal distances for them to count
#: as the same rigid geometry and share one Packmol ``structure`` block.
TEMPLATE_MATCH_TOL_A = 5e-3


def _resolve_packmol_tolerance(
    *,
    min_distance: float,
    spacing: float | None = None,
    packmol_tolerance: float | None = None,
) -> float:
    """Packmol ``tolerance`` (Å) for a repack.

    Packmol's ``tolerance`` is the *minimum allowed distance between atoms of
    different structures* — a contact floor, not a packing pitch. ``spacing`` is
    a centre-to-centre COM separation (``--spacing``, default 5 Å); feeding it in
    here asks Packmol to keep every atom pair further apart than the mean
    molecular separation, which is unsatisfiable at liquid density (methanol at
    0.79 g/cm³ has ~4.1 Å between COMs and 2.8 Å O···O hydrogen bonds). It is
    accepted for call-site compatibility and deliberately ignored.
    """
    del spacing  # never a contact distance; see docstring
    if packmol_tolerance is not None and float(packmol_tolerance) > 0.0:
        tol = float(packmol_tolerance)
    else:
        tol = PACKMOL_DEFAULT_TOLERANCE_A
    # The caller's overlap floor is a genuine contact distance, so it may raise
    # the tolerance (it is ~0.45 Å in the prep ladder, i.e. normally inert).
    if np.isfinite(min_distance) and float(min_distance) > tol:
        tol = float(min_distance)
    return float(tol)


def _charmm_atom_metadata(n_atoms: int) -> tuple[list[str], np.ndarray] | None:
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.psf as psf
        from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

        names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)[: int(n_atoms)]]
        z = np.asarray(get_Z_from_psf(), dtype=int)[: int(n_atoms)]
        if len(names) != int(n_atoms) or int(z.shape[0]) != int(n_atoms):
            return None
        return names, z
    except Exception:
        return None


def _template_handedness(template: np.ndarray) -> np.ndarray:
    """Reflection-odd terms (Å³): signed volumes of consecutive atom triples.

    Distances alone are blind to reflection, so enantiomers share a distance
    fingerprint and would be collapsed into one block — silently flipping a
    molecule's hand. Each volume is invariant under proper rotation and changes
    sign under reflection, and they are kept as a vector rather than summed
    because opposite-signed terms cancel in a sum (an sp3 centre very nearly
    does). Consecutive triples in PSF order, rather than e.g. the most non-planar
    quadruple, keep this free of ``argmax`` ties — symmetric monomers such as
    methanol hit those constantly, and a tie would turn the sign into float noise
    and stop identical copies from merging at all.
    """
    arr = np.asarray(template, dtype=float).reshape(-1, 3)
    if int(arr.shape[0]) < 3:
        return np.zeros((0,), dtype=float)
    a, b, c = arr[:-2], arr[1:-1], arr[2:]
    return np.asarray(np.sum(np.cross(a, b) * c, axis=1) / 6.0, dtype=float)


def _template_key(template: np.ndarray) -> np.ndarray:
    """Rotation/translation-invariant fingerprint of a monomer's internal geometry.

    The upper triangle of the internal distance matrix in atom order, plus a
    handedness term. Raw coordinates cannot be compared directly: Packmol gives
    every copy a random orientation, so two molecules with identical rigid
    geometry have completely different coordinate arrays. Distances are invariant
    under exactly the rotation + translation Packmol applies, so they identify the
    copies that can share one ``structure ... number N`` block.
    """
    arr = np.asarray(template, dtype=float).reshape(-1, 3)
    n = int(arr.shape[0])
    if n < 2:
        return np.zeros((1,), dtype=float)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    return np.concatenate(
        [np.asarray(d[iu], dtype=float), _template_handedness(arr)]
    )


def _monomer_species_key(
    atom_count: int,
    names: list[str] | None,
    z: np.ndarray | None,
) -> tuple:
    """Chemical identity of a monomer: atom count, atom names, atomic numbers."""
    return (
        int(atom_count),
        tuple(str(x) for x in names) if names is not None else None,
        tuple(int(x) for x in np.asarray(z).reshape(-1)) if z is not None else None,
    )


def _group_identical_templates(
    indices: list[int],
    templates: list[np.ndarray],
    offsets: np.ndarray,
    *,
    atom_names: list[str] | None,
    atomic_numbers: np.ndarray | None,
    tol: float = TEMPLATE_MATCH_TOL_A,
) -> list[list[int]]:
    """Group monomers that are the same species *and* the same rigid geometry.

    One Packmol ``structure`` block per molecule is quadratically expensive at
    liquid density (Packmol treats every block as its own molecule type), so
    interchangeable copies are collapsed into a single ``number N`` block. Only
    monomers matching to within ``tol`` are merged, because the whole group is
    packed from the representative's geometry — a conformationally diverse or
    mixed-species set keeps its per-molecule blocks and its per-molecule shape.
    """
    groups: list[list[int]] = []
    reps: list[tuple[tuple, np.ndarray]] = []
    for mi in indices:
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        species = _monomer_species_key(
            e - s,
            atom_names[s:e] if atom_names is not None else None,
            atomic_numbers[s:e] if atomic_numbers is not None else None,
        )
        fingerprint = _template_key(templates[mi])
        for gi, (rep_species, rep_fingerprint) in enumerate(reps):
            if rep_species != species or rep_fingerprint.shape != fingerprint.shape:
                continue
            if float(np.max(np.abs(rep_fingerprint - fingerprint), initial=0.0)) <= tol:
                groups[gi].append(int(mi))
                break
        else:
            groups.append([int(mi)])
            reps.append((species, fingerprint))
    return groups


def _read_packmol_monomer_coords(
    pdb_path: Path,
    *,
    expected_atoms_per_monomer: list[int],
) -> list[np.ndarray]:
    """Return per-monomer coordinates in Packmol structure order.

    Packmol concatenates many ``fixed`` structures into one PDB; residue IDs are
    not unique per monomer (e.g. 198×5-atom fixed blocks → residue 1 with 990
    atoms). Split sequentially by ``expected_atoms_per_monomer`` instead.
    """
    _names, _resids, positions = _parse_pdb_atom_records(pdb_path)
    coords = np.asarray(positions, dtype=float)
    expected = [int(n) for n in expected_atoms_per_monomer]
    n_total = int(sum(expected))
    if int(coords.shape[0]) != n_total:
        raise RuntimeError(
            f"Packmol output has {int(coords.shape[0])} atoms, expected {n_total} "
            f"({len(expected)} monomers)"
        )
    out: list[np.ndarray] = []
    start = 0
    for n_expect in expected:
        block = coords[start : start + n_expect]
        if int(block.shape[0]) != n_expect:
            raise RuntimeError(
                f"Packmol atom block at offset {start} has {block.shape[0]} atoms, "
                f"expected {n_expect}"
            )
        out.append(np.asarray(block, dtype=float))
        start += n_expect
    return out


def _positions_to_packmol_primary_frame(
    positions: np.ndarray,
    box_side: float,
    monomer_offsets: np.ndarray,
    cell: Any | None,
) -> tuple[np.ndarray, bool]:
    """Map coords to jaxmd primary cell ``[0, L)`` for Packmol (CHARMM → shift by ``L/2``)."""
    from mmml.utils.geometry_checks import wrap_monomers_primary_cell

    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    if cell is not None:
        pos = wrap_monomers_primary_cell(pos, offsets, cell)
    L = float(box_side)
    if float(np.min(pos)) < -0.05 * L:
        return pos + 0.5 * L, True
    return pos, False


def _positions_from_packmol_primary_frame(
    positions: np.ndarray,
    box_side: float,
    *,
    was_charmm_centered: bool,
) -> np.ndarray:
    if not was_charmm_centered:
        return np.asarray(positions, dtype=float)
    return np.asarray(positions, dtype=float) - 0.5 * float(box_side)


def _run_packmol_repack(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    movable_indices: list[int],
    *,
    template_positions: np.ndarray | None,
    cell: Any | None,
    tolerance: float,
    seed: int | None,
    scratch_dir: Path,
    atom_names: list[str] | None,
    atomic_numbers: np.ndarray | None,
    packmol_margin_A: float | None = None,
) -> np.ndarray:
    from mmml.utils.geometry_checks import _monomer_internal_templates, wrap_monomers_primary_cell

    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    movable = sorted({int(i) for i in movable_indices})
    if not movable or len(movable) >= n_monomers:
        movable = list(range(n_monomers))
    fixed = [mi for mi in range(n_monomers) if mi not in movable]

    source = (
        np.asarray(template_positions, dtype=float)
        if template_positions is not None
        else pos
    )
    templates, _radii = _monomer_internal_templates(source, offsets)
    atoms_per = [int(offsets[i + 1] - offsets[i]) for i in range(n_monomers)]

    meta = None
    if atom_names is None or atomic_numbers is None:
        meta = _charmm_atom_metadata(int(pos.shape[0]))
    if meta is not None:
        if atom_names is None:
            atom_names = meta[0]
        if atomic_numbers is None:
            atomic_numbers = meta[1]

    scratch_dir = Path(scratch_dir).expanduser().resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    structure_lines: list[str] = []
    output_order: list[int] = []

    box_side = _cell_box_side(cell)
    was_charmm_centered = False
    if box_side is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
            MIN_PERIODIC_LIQUID_PACKMOL_PADDING_A,
            resolve_packmol_inner_cube_side_A,
        )

        pos, was_charmm_centered = _positions_to_packmol_primary_frame(
            pos, float(box_side), offsets, cell
        )
        margin = (
            float(packmol_margin_A)
            if packmol_margin_A is not None and float(packmol_margin_A) > 0.0
            else float(MIN_PERIODIC_LIQUID_PACKMOL_PADDING_A)
        )
        inner_side, margin = resolve_packmol_inner_cube_side_A(
            float(box_side),
            margin_A=margin,
        )
        center = (0.5 * float(box_side), 0.5 * float(box_side), 0.5 * float(box_side))
        x0, y0, z0 = packmol_cube_origin(center, float(inner_side))
        restraint = f"  inside cube {x0} {y0} {z0} {float(inner_side)}"
        placement_label = f"cube (inner {float(inner_side):.2f} Å, margin {margin:.2f} Å/side)"
    else:
        span = float(np.ptp(pos, axis=0).max()) if pos.size else 10.0
        radius = max(8.0, 0.6 * span + 4.0)
        com = pos.mean(axis=0)
        restraint = (
            f"  inside sphere {float(com[0])} {float(com[1])} {float(com[2])} {radius}"
        )
        placement_label = "sphere"

    for mi in fixed:
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        com = pos[s:e].mean(axis=0)
        pdb_path = scratch_dir / f"fixed_{mi:04d}.pdb"
        names_slice = atom_names[s:e] if atom_names is not None else None
        z_slice = atomic_numbers[s:e] if atomic_numbers is not None else None
        if z_slice is None:
            raise RuntimeError("Packmol repack requires atomic numbers (CHARMM PSF or ml_Z)")
        write_monomer_pdb_for_packmol(
            pdb_path,
            templates[mi],
            z_slice,
            atom_names=names_slice,
            resname=f"F{mi:03d}",
        )
        structure_lines.append(
            f"structure {pdb_path.resolve()}\n"
            f"  number 1\n"
            f"  fixed {float(com[0])} {float(com[1])} {float(com[2])} 0. 0. 0.\n"
            f"end structure"
        )
        output_order.append(mi)

    grouped = _group_identical_templates(
        movable,
        templates,
        offsets,
        atom_names=atom_names,
        atomic_numbers=atomic_numbers,
    )

    for mi_list in grouped:
        mi0 = int(mi_list[0])
        pdb_path = scratch_dir / f"movable_{mi0:04d}.pdb"
        s0, e0 = int(offsets[mi0]), int(offsets[mi0 + 1])
        names_slice = atom_names[s0:e0] if atom_names is not None else None
        z_slice = atomic_numbers[s0:e0] if atomic_numbers is not None else None
        if z_slice is None:
            raise RuntimeError("Packmol repack requires atomic numbers (CHARMM PSF or ml_Z)")
        write_monomer_pdb_for_packmol(
            pdb_path,
            templates[mi0],
            z_slice,
            atom_names=names_slice,
            resname=f"M{mi0:03d}",
        )
        structure_lines.append(
            f"structure {pdb_path.resolve()}\n"
            f"  chain A\n"
            f"  resnumbers 2\n"
            f"  number {len(mi_list)}\n"
            f"{restraint}\n"
            f"end structure"
        )
        output_order.extend(int(i) for i in mi_list)

    out_pdb = (scratch_dir / "repack.pdb").resolve()
    inp_path = scratch_dir / "repack.inp"
    randint = int(seed) if seed is not None else int(np.random.randint(1_000_000))
    packmol_input = (
        f"seed {randint}\n"
        f"output {out_pdb}\n"
        f"filetype pdb\n"
        f"tolerance {float(tolerance)}\n\n"
        + "\n\n".join(structure_lines)
        + "\n"
    )
    execute_packmol_script(packmol_input, inp_path)

    expected_atoms = [atoms_per[mi] for mi in output_order]
    packed = _read_packmol_monomer_coords(out_pdb, expected_atoms_per_monomer=expected_atoms)

    new_pos = pos.copy()
    for mi, coords in zip(output_order, packed):
        if mi in fixed:
            continue
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        new_pos[s:e] = np.asarray(coords, dtype=float)

    if cell is not None:
        new_pos = wrap_monomers_primary_cell(new_pos, offsets, cell)

    if box_side is not None:
        new_pos = _positions_from_packmol_primary_frame(
            new_pos,
            float(box_side),
            was_charmm_centered=was_charmm_centered,
        )

    print(
        f"Packmol repack ({placement_label}): patched monomers "
        f"{sorted(i + 1 for i in movable)} "
        f"(1-based; fixed {len(fixed)}, tolerance {float(tolerance):.2f} Å, "
        f"{len(grouped)} movable structure block(s))",
        flush=True,
    )
    return new_pos


def repack_monomers_clear_overlap(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float,
    spacing: float | None = None,
    margin: float = 0.2,
    seed: int | None = None,
    cell: Any | None = None,
    template_positions: np.ndarray | None = None,
    max_monomer_radius_A: float | None = None,
    random_rotations: bool = True,
    scratch_dir: Path | str | None = None,
    atom_names: list[str] | None = None,
    atomic_numbers: np.ndarray | None = None,
    packmol_tolerance: float | None = None,
    packmol_margin_A: float | None = None,
) -> np.ndarray:
    """Repack all monomers with Packmol; fall back to grid placement on failure."""
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    movable = list(range(n_monomers))
    workdir = (
        Path(scratch_dir)
        if scratch_dir is not None
        else Path("packmol_repack")
    )
    # `spacing` is a COM pitch, not a contact distance: it belongs to the grid
    # fallback below, never to the Packmol tolerance.
    tolerance = _resolve_packmol_tolerance(
        min_distance=min_distance,
        packmol_tolerance=packmol_tolerance,
    )
    try:
        return _run_packmol_repack(
            positions,
            offsets,
            movable,
            template_positions=template_positions,
            cell=cell,
            tolerance=tolerance,
            seed=seed,
            scratch_dir=workdir,
            atom_names=atom_names,
            atomic_numbers=atomic_numbers,
            packmol_margin_A=packmol_margin_A,
        )
    except Exception as exc:
        print(
            f"Packmol repack failed ({exc}); falling back to grid monomer repack",
            flush=True,
        )
        from mmml.utils.geometry_checks import repack_monomers_clear_overlap as grid_repack

        return grid_repack(
            positions,
            offsets,
            min_distance=min_distance,
            spacing=spacing,
            margin=margin,
            seed=seed,
            cell=cell,
            template_positions=template_positions,
            max_monomer_radius_A=max_monomer_radius_A,
            random_rotations=random_rotations,
        )


def repack_selected_monomers_clear_overlap(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    selected_indices: list[int] | tuple[int, ...],
    *,
    min_distance: float,
    spacing: float | None = None,
    margin: float = 0.2,
    seed: int | None = None,
    cell: Any | None = None,
    template_positions: np.ndarray | None = None,
    max_monomer_radius_A: float | None = None,
    random_rotations: bool = True,
    scratch_dir: Path | str | None = None,
    atom_names: list[str] | None = None,
    atomic_numbers: np.ndarray | None = None,
    packmol_tolerance: float | None = None,
    packmol_margin_A: float | None = None,
) -> np.ndarray:
    """Repack only selected monomers with Packmol; fall back to grid on failure."""
    offsets = np.asarray(monomer_offsets, dtype=int)
    selected = sorted({int(i) for i in selected_indices})
    workdir = (
        Path(scratch_dir)
        if scratch_dir is not None
        else Path("packmol_repack")
    )
    # `spacing` is a COM pitch, not a contact distance: it belongs to the grid
    # fallback below, never to the Packmol tolerance.
    tolerance = _resolve_packmol_tolerance(
        min_distance=min_distance,
        packmol_tolerance=packmol_tolerance,
    )
    try:
        return _run_packmol_repack(
            positions,
            offsets,
            selected,
            template_positions=template_positions,
            cell=cell,
            tolerance=tolerance,
            seed=seed,
            scratch_dir=workdir,
            atom_names=atom_names,
            atomic_numbers=atomic_numbers,
            packmol_margin_A=packmol_margin_A,
        )
    except Exception as exc:
        print(
            f"Packmol selective repack failed ({exc}); "
            f"falling back to grid monomer repack",
            flush=True,
        )
        from mmml.utils.geometry_checks import (
            repack_selected_monomers_clear_overlap as grid_repack,
        )

        return grid_repack(
            positions,
            offsets,
            selected,
            min_distance=min_distance,
            spacing=spacing,
            margin=margin,
            seed=seed,
            cell=cell,
            template_positions=template_positions,
            max_monomer_radius_A=max_monomer_radius_A,
            random_rotations=random_rotations,
        )
