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


def _resolve_packmol_tolerance(
    *,
    min_distance: float,
    spacing: float | None,
    packmol_tolerance: float | None = None,
) -> float:
    candidates = [2.0, float(min_distance)]
    if spacing is not None and float(spacing) > 0.0:
        candidates.append(float(spacing))
    if packmol_tolerance is not None and float(packmol_tolerance) > 0.0:
        candidates.append(float(packmol_tolerance))
    return float(max(candidates))


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


def _template_key(template: np.ndarray) -> tuple[float, ...]:
    arr = np.asarray(template, dtype=float).reshape(-1)
    return tuple(round(float(x), 4) for x in arr[: min(12, arr.size)])


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
) -> tuple[np.ndarray, bool]:
    """Map CHARMM image-centered coords to jaxmd primary cell ``[0, L)`` for Packmol."""
    from mmml.cli.run.md_handoff import cluster_positions_use_jaxmd_primary_cell_frame

    pos = np.asarray(positions, dtype=float)
    if cluster_positions_use_jaxmd_primary_cell_frame(pos, float(box_side)):
        return pos, False
    half = 0.5 * float(box_side)
    return pos + half, True


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

        pos, was_charmm_centered = _positions_to_packmol_primary_frame(pos, float(box_side))
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
            f"structure {pdb_path}\n"
            f"  number 1\n"
            f"  fixed {float(com[0])} {float(com[1])} {float(com[2])} 0. 0. 0.\n"
            f"end structure"
        )
        output_order.append(mi)

    grouped: dict[tuple[float, ...], list[int]] = {}
    for mi in movable:
        key = _template_key(templates[mi])
        grouped.setdefault(key, []).append(mi)

    for mi_list in grouped.values():
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
            f"structure {pdb_path}\n"
            f"  chain A\n"
            f"  resnumbers 2\n"
            f"  number {len(mi_list)}\n"
            f"{restraint}\n"
            f"end structure"
        )
        output_order.extend(int(i) for i in mi_list)

    out_pdb = scratch_dir / "repack.pdb"
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
        f"(1-based; fixed {len(fixed)}, tolerance {float(tolerance):.2f} Å)",
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
    tolerance = _resolve_packmol_tolerance(
        min_distance=min_distance,
        spacing=spacing,
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
    tolerance = _resolve_packmol_tolerance(
        min_distance=min_distance,
        spacing=spacing,
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
