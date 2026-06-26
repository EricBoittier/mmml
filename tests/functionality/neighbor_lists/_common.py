"""Shared helpers for neighbor-list functionality scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_pass(msg: str) -> None:
    from mmml.utils.rich_report import emit_status

    emit_status(True, msg)


def print_fail(msg: str) -> None:
    from mmml.utils.rich_report import emit_status

    emit_status(False, msg)


def two_dimer_cluster(
    *,
    box_side: float = 40.0,
    com_separation: float = 8.0,
    atoms_per_monomer: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, cell_3x3, monomer_offsets, monomer_id) for two dimers."""
    n_monomers = 2
    n_atoms = n_monomers * atoms_per_monomer
    offsets = np.array([0, atoms_per_monomer, n_atoms], dtype=np.int32)
    monomer_id = np.empty(n_atoms, dtype=np.int32)
    for mi in range(n_monomers):
        monomer_id[offsets[mi] : offsets[mi + 1]] = mi

    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for mi in range(n_monomers):
        start = int(offsets[mi])
        end = int(offsets[mi + 1])
        com = np.array([com_separation * mi, 0.0, 0.0], dtype=np.float64)
        for k in range(start, end):
            positions[k] = com + np.array(
                [0.3 * (k - start), 0.2 * ((k - start) % 2), 0.1 * (k - start)],
                dtype=np.float64,
            )

    cell = float(box_side) * np.eye(3, dtype=np.float64)
    return positions, cell, offsets, monomer_id


def npt_box_sequence(base_side: float = 40.0) -> list[np.ndarray]:
    """Box side vectors for NPT invalidation tests."""
    return [
        np.array([base_side, base_side, base_side], dtype=np.float64),
        np.array([base_side * 1.02, base_side * 1.02, base_side * 1.02], dtype=np.float64),
    ]


def setup_charmm_composition_cluster(
    composition: str,
    *,
    box_side: float = 40.0,
    spacing: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a CGENFF cluster from ``RES:COUNT`` composition (md-system style).

    Returns ``(positions, cell_3x3, monomer_offsets, monomer_id, atomic_numbers)``.
    Monomers are placed on a spacing grid, then centered in a cubic PBC box.
    Requires PyCHARMM + CGENFF.
    """
    import pandas as pd

    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_from_composition,
        _parse_composition,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        coor,
    )
    from mmml.interfaces.pycharmmInterface.nl_reference import monomer_id_from_offsets

    if CGENFF_PRM is None:
        raise RuntimeError("PyCHARMM/CGENFF not available")

    comp = _parse_composition(composition)
    z, positions, atoms_per_list, _residue_labels = _build_cluster_from_composition(
        composition=comp,
        spacing=float(spacing),
    )
    positions = np.asarray(positions, dtype=np.float64)
    positions = positions - positions.mean(axis=0) + np.array(
        [box_side / 2, box_side / 2, box_side / 2], dtype=np.float64
    )
    coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))

    n_monomers = len(atoms_per_list)
    offsets = np.zeros(n_monomers + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=np.int32))
    monomer_id = monomer_id_from_offsets(offsets, positions.shape[0])
    cell = float(box_side) * np.eye(3, dtype=np.float64)
    return positions, cell, offsets, monomer_id, np.asarray(z, dtype=int)


def synthetic_monomer_cluster(
    *,
    com_positions: np.ndarray,
    cell: np.ndarray,
    atoms_per_monomer: int = 5,
    monomer_atom_offsets: np.ndarray | list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic PBC cluster from monomer COM coordinates.

    Returns ``(positions, cell_3x3, monomer_offsets, monomer_id)``.

    ``monomer_atom_offsets`` may be one (N,3) array for all monomers or a per-monomer list.
    """
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        cell_matrix_3x3 as _cell_matrix_3x3,
        monomer_id_from_offsets as _monomer_id_from_offsets,
    )

    default_offsets = np.array(
        [[0.3 * k, 0.2 * (k % 2), 0.1 * k] for k in range(atoms_per_monomer)],
        dtype=np.float64,
    )

    coms = np.asarray(com_positions, dtype=np.float64).reshape(-1, 3)
    n_monomers = int(coms.shape[0])
    n_atoms = n_monomers * int(atoms_per_monomer)
    offsets = np.arange(0, n_atoms + 1, atoms_per_monomer, dtype=np.int32)
    monomer_id = _monomer_id_from_offsets(offsets, n_atoms)

    if monomer_atom_offsets is None:
        per_monomer_offsets = [default_offsets] * n_monomers
    elif isinstance(monomer_atom_offsets, list):
        per_monomer_offsets = [
            default_offsets if off is None else np.asarray(off, dtype=np.float64)
            for off in monomer_atom_offsets
        ]
    else:
        per_monomer_offsets = [np.asarray(monomer_atom_offsets, dtype=np.float64)] * n_monomers

    cell_mat = _cell_matrix_3x3(cell)
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for mi in range(n_monomers):
        start = int(offsets[mi])
        atom_off = per_monomer_offsets[mi]
        for k in range(atoms_per_monomer):
            positions[start + k] = coms[mi] + atom_off[k]
    return positions, cell_mat, offsets, monomer_id


def cell_matrix_3x3(cell: np.ndarray) -> np.ndarray:
    """Normalize scalar, (3,), or (3,3) cell spec to a 3×3 matrix (Å)."""
    from mmml.interfaces.pycharmmInterface.nl_reference import cell_matrix_3x3 as _cm

    return _cm(cell)


def monomer_id_from_offsets(monomer_offsets: np.ndarray, n_atoms: int) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.nl_reference import monomer_id_from_offsets as _mid

    return _mid(monomer_offsets, n_atoms)


def extreme_pbc_cases() -> list[dict[str, object]]:
    """Named extreme PBC geometries for NL backend stress tests."""
  # fmt: off
    return [
        {
            "name": "opposite_corners",
            "description": "Dimers at opposite cube corners (max MIC wrap path)",
            "box_side": 20.0,
            "cutoff": 13.0,
            "com_positions": np.array([[1.0, 1.0, 1.0], [19.0, 19.0, 19.0]]),
        },
        {
            "name": "tight_box_many_images",
            "description": "Small box (L≈cutoff+ε) forces many periodic images",
            "box_side": 15.0,
            "cutoff": 13.0,
            "com_positions": np.array([[2.0, 7.5, 7.5], [13.0, 7.5, 7.5]]),
        },
        {
            "name": "cutoff_near_half_box",
            "description": "Cutoff ≈ L/2 (minimum-image boundary regime)",
            "box_side": 26.0,
            "cutoff": 13.0,
            "com_positions": np.array([[6.5, 13.0, 13.0], [19.5, 13.0, 13.0]]),
        },
        {
            "name": "wrap_straddle_x",
            "description": "Monomer 1 atoms straddle the x periodic face",
            "box_side": 20.0,
            "cutoff": 13.0,
            "com_positions": np.array([[10.0, 10.0, 10.0], [2.0, 10.0, 10.0]]),
            "monomer_atom_offsets": [
                None,
                np.array(
                    [
                        [0.3, 0.0, 0.0],
                        [0.6, 0.2, 0.0],
                        [-1.7, 0.0, 0.0],
                        [17.8, 0.0, 0.0],
                        [18.1, 0.1, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ],
        },
        {
            "name": "four_dimers_dense",
            "description": "Four dimers on a 2×2 grid in a modest box",
            "box_side": 22.0,
            "cutoff": 13.0,
            "com_positions": np.array(
                [
                    [5.0, 5.0, 11.0],
                    [17.0, 5.0, 11.0],
                    [5.0, 17.0, 11.0],
                    [17.0, 17.0, 11.0],
                ],
                dtype=np.float64,
            ),
        },
        {
            "name": "orthorhombic_cell",
            "description": "Non-cubic orthorhombic cell (diagonal 28×20×16 Å)",
            "cell": np.diag([28.0, 20.0, 16.0]),
            "cutoff": 13.0,
            "com_positions": np.array([[4.0, 4.0, 4.0], [24.0, 16.0, 12.0]]),
        },
        {
            "name": "minimal_com_spacing",
            "description": "Very close monomer COMs (still inter-monomer pairs)",
            "box_side": 30.0,
            "cutoff": 13.0,
            "com_positions": np.array([[14.0, 15.0, 15.0], [16.5, 15.0, 15.0]]),
        },
        {
            "name": "edge_face_contact",
            "description": "One monomer on −x face, partner just inside +x face",
            "box_side": 24.0,
            "cutoff": 13.0,
            "com_positions": np.array([[1.2, 12.0, 12.0], [22.8, 12.0, 12.0]]),
        },
        {
            "name": "high_cutoff_fraction",
            "description": "Cutoff > L/3 in a compact box (dense pair shell)",
            "box_side": 18.0,
            "cutoff": 15.0,
            "com_positions": np.array([[6.0, 9.0, 9.0], [12.0, 9.0, 9.0], [9.0, 6.0, 9.0]]),
        },
    ]
  # fmt: on


def charmm_extreme_pbc_cases() -> list[dict[str, object]]:
    """CHARMM/CGENFF analogs of extreme PBC cases (requires PyCHARMM PSF + PBC nbonds)."""
  # fmt: off
    return [
        {
            "name": "charmm_opposite_corners",
            "synthetic_analog": "opposite_corners",
            "description": "ACO:2 wide spacing in a 20 Å cube (corner-like separation)",
            "composition": "ACO:2",
            "box_side": 20.0,
            "spacing": 12.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_tight_box_many_images",
            "synthetic_analog": "tight_box_many_images",
            "description": "ACO:2 in a 15 Å cube (L≈cutoff+ε)",
            "composition": "ACO:2",
            "box_side": 15.0,
            "spacing": 5.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_cutoff_near_half_box",
            "synthetic_analog": "cutoff_near_half_box",
            "description": "ACO:2 with spacing near L/2 in a 26 Å cube",
            "composition": "ACO:2",
            "box_side": 26.0,
            "spacing": 12.5,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_four_dimers_dense",
            "synthetic_analog": "four_dimers_dense",
            "description": "ACO:4 on a tight grid in a 22 Å cube",
            "composition": "ACO:4",
            "box_side": 22.0,
            "spacing": 6.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_minimal_com_spacing",
            "synthetic_analog": "minimal_com_spacing",
            "description": "ACO:2 with very close placement spacing",
            "composition": "ACO:2",
            "box_side": 30.0,
            "spacing": 3.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_edge_face_contact",
            "synthetic_analog": "edge_face_contact",
            "description": "ACO:2 in a 24 Å cube with tight spacing (face-adjacent MIC)",
            "composition": "ACO:2",
            "box_side": 24.0,
            "spacing": 4.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_high_cutoff_fraction",
            "synthetic_analog": "high_cutoff_fraction",
            "description": "ACO:3 in an 18 Å cube with 15 Å cutoff (dense pair shell)",
            "composition": "ACO:3",
            "box_side": 18.0,
            "spacing": 4.0,
            "cutoff": 15.0,
        },
    ]
  # fmt: on


def have_charmm_nl() -> bool:
    """Return True when PyCHARMM + CGENFF are available for NL scripts."""
    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

        return CGENFF_PRM is not None
    except Exception:
        return False


def build_extreme_pbc_case(
    case: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Materialize one extreme case. Returns geometry + cutoff + description."""
    if "cell" in case:
        cell_spec = case["cell"]
    else:
        cell_spec = float(case["box_side"])  # type: ignore[arg-type]
    cutoff = float(case["cutoff"])
    com_positions = np.asarray(case["com_positions"], dtype=np.float64)
    offsets_arg = case.get("monomer_atom_offsets")
    positions, cell, offsets, monomer_id = synthetic_monomer_cluster(
        com_positions=com_positions,
        cell=np.asarray(cell_spec, dtype=np.float64),
        monomer_atom_offsets=offsets_arg,  # type: ignore[arg-type]
    )
    desc = str(case.get("description", case.get("name", "extreme")))
    return positions, cell, offsets, monomer_id, cutoff, desc


def liquid_density_grid_com_positions(
    n_monomers: int,
    box_side: float,
    *,
    inset_fraction: float = 0.1,
) -> np.ndarray:
    """Place monomer COMs on a 3D grid inside a cubic box (liquid-like packing)."""
    n = int(n_monomers)
    if n <= 0:
        raise ValueError(f"n_monomers must be positive, got {n_monomers}")
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    inset = float(box_side) * float(inset_fraction)
    usable = float(box_side) - 2.0 * inset
    step = usable / max(n_side - 1, 1) if n_side > 1 else 0.0
    coms = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ix = i % n_side
        iy = (i // n_side) % n_side
        iz = i // (n_side * n_side)
        coms[i] = inset + step * np.array([ix, iy, iz], dtype=np.float64)
    return coms


def _resolve_liquid_target_density_g_cm3(
    composition: dict[str, int],
    *,
    bulk_density_fraction: float = 1.0,
    target_density_g_cm3: float | None = None,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import SOLVENT_BULK_PROPS

    if target_density_g_cm3 is not None:
        rho = float(target_density_g_cm3)
        if rho <= 0.0:
            raise ValueError(f"target_density_g_cm3 must be positive, got {rho}")
        return rho
    if len(composition) != 1:
        raise ValueError(
            "bulk_density_fraction requires a single-species composition "
            "(e.g. ACO:16); pass target_density_g_cm3 for mixtures."
        )
    residue = next(iter(composition))
    key = str(residue).strip().upper()
    props = SOLVENT_BULK_PROPS.get(key)
    if props is None:
        raise ValueError(f"no bulk density for residue {key!r}")
    frac = float(bulk_density_fraction)
    if frac <= 0.0:
        raise ValueError(f"bulk_density_fraction must be positive, got {frac}")
    return float(props["rho_g_cm3"]) * frac


def liquid_density_box_side_for_composition(
    composition: dict[str, int],
    *,
    bulk_density_fraction: float = 1.0,
    target_density_g_cm3: float | None = None,
    box_side: float | None = None,
    min_side_A: float | None = None,
) -> tuple[float, float]:
    """Return ``(box_side_A, target_density_g_cm3)`` for a liquid-density cubic box."""
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        cubic_box_side_from_target_density,
        total_mass_g_for_composition,
    )

    n_mol = int(sum(composition.values()))
    rho = _resolve_liquid_target_density_g_cm3(
        composition,
        bulk_density_fraction=bulk_density_fraction,
        target_density_g_cm3=target_density_g_cm3,
    )
    if box_side is not None:
        return float(box_side), rho
    mass_g = total_mass_g_for_composition(composition)
    side = cubic_box_side_from_target_density(
        n_molecules=n_mol,
        total_mass_g=mass_g,
        target_density_g_cm3=rho,
        min_side_A=min_side_A,
    )
    return side, rho


def liquid_density_spacing(box_side: float, n_monomers: int) -> float:
    """Grid spacing for liquid-density placement inside a cubic box."""
    n_side = max(int(np.ceil(int(n_monomers) ** (1.0 / 3.0))), 1)
    return float(box_side) / float(n_side) * 0.9


def effective_mass_density_g_cm3(composition: dict[str, int], box_side: float) -> float:
    """Actual mass density (g/cm³) for a composition in a cubic box."""
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import total_mass_g_for_composition

    vol_cm3 = float(box_side) ** 3 * 1e-24
    mass_g = total_mass_g_for_composition(composition)
    return mass_g / vol_cm3


def apply_mic_wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap Cartesian positions into the primary unit cell."""
    from mmml.interfaces.pycharmmInterface.nl_reference import cell_matrix_3x3

    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell)
    inv = np.linalg.inv(cell_mat)
    frac = R @ inv.T
    frac = frac - np.floor(frac)
    return frac @ cell_mat


def motion_stress_steps() -> list[dict[str, object]]:
    """Named position/box perturbations for NL motion stress tests."""
    return [
        {"name": "baseline", "kind": "identity"},
        {"name": "jitter_0.10", "kind": "jitter", "amplitude_A": 0.10},
        {"name": "jitter_0.50", "kind": "jitter", "amplitude_A": 0.50},
        {"name": "compress_0.92", "kind": "scale_com", "factor": 0.92},
        {"name": "expand_1.08", "kind": "scale_com", "factor": 1.08},
        {"name": "shift_x_2.0", "kind": "translate", "vector_A": [2.0, 0.0, 0.0]},
        {"name": "box_shrink_0.97", "kind": "box_scale", "factor": 0.97},
        {"name": "box_expand_1.03", "kind": "box_scale", "factor": 1.03},
    ]


def apply_motion_step(
    positions: np.ndarray,
    cell: np.ndarray,
    step: dict[str, object],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(new_positions, new_cell)`` after a motion-stress step."""
    kind = str(step["kind"])
    R = np.asarray(positions, dtype=np.float64).copy()
    cell_mat = np.asarray(cell, dtype=np.float64).copy()
    if kind == "identity":
        return R, cell_mat
    if kind == "jitter":
        amp = float(step["amplitude_A"])
        R = R + amp * rng.standard_normal(R.shape)
        return R, cell_mat
    if kind == "scale_com":
        factor = float(step["factor"])
        if cell_mat.ndim == 2:
            center = np.sum(cell_mat, axis=0) / 2.0
        else:
            center = np.array([float(cell_mat) / 2.0] * 3, dtype=np.float64)
        R = center + factor * (R - center)
        return R, cell_mat
    if kind == "translate":
        vec = np.asarray(step["vector_A"], dtype=np.float64)
        return R + vec, cell_mat
    if kind == "box_scale":
        factor = float(step["factor"])
        if cell_mat.ndim == 2:
            diag = np.diag(cell_mat).astype(np.float64)
        else:
            diag = np.array([float(cell_mat)] * 3, dtype=np.float64)
        new_diag = diag * factor
        R = R * factor
        return R, np.diag(new_diag)
    raise ValueError(f"unknown motion step kind {kind!r}")


def liquid_density_synthetic_cases() -> list[dict[str, object]]:
    """Synthetic toy clusters at experimental bulk liquid densities."""
  # fmt: off
    return [
        {
            "name": "synthetic_aco_liquid_n16",
            "description": "16 toy monomers in an ACO bulk-density cube (~0.784 g/cm³)",
            "composition": {"ACO": 16},
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_dcm_liquid_n16",
            "description": "16 toy monomers in a DCM bulk-density cube (~1.326 g/cm³)",
            "composition": {"DCM": 16},
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_aco_liquid_n32",
            "description": "32 toy monomers in an ACO bulk-density cube",
            "composition": {"ACO": 32},
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_dcm_liquid_n32",
            "description": "32 toy monomers in a DCM bulk-density cube",
            "composition": {"DCM": 32},
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_aco_liquid_box25",
            "description": "ACO:N capped at bulk ρ in a 25 Å cube (N≤32 for NL tests)",
            "solvent": "ACO",
            "box_side": 25.0,
            "bulk_density_fraction": 1.0,
            "max_monomers": 32,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_dcm_liquid_box25",
            "description": "DCM:N capped at bulk ρ in a 25 Å cube (N≤32)",
            "solvent": "DCM",
            "box_side": 25.0,
            "bulk_density_fraction": 1.0,
            "max_monomers": 32,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_aco_liquid_n32_rho125",
            "description": "ACO:32 at 1.25× bulk ρ (compressed liquid)",
            "composition": {"ACO": 32},
            "bulk_density_fraction": 1.25,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_aco_liquid_n32_rho150",
            "description": "ACO:32 at 1.50× bulk ρ (dense liquid)",
            "composition": {"ACO": 32},
            "bulk_density_fraction": 1.5,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_dcm_liquid_n32_rho125",
            "description": "DCM:32 at 1.25× bulk ρ",
            "composition": {"DCM": 32},
            "bulk_density_fraction": 1.25,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_dcm_liquid_n32_rho150",
            "description": "DCM:32 at 1.50× bulk ρ",
            "composition": {"DCM": 32},
            "bulk_density_fraction": 1.5,
            "cutoff": 13.0,
        },
        {
            "name": "synthetic_aco_liquid_n48_rho125",
            "description": "ACO:48 at 1.25× bulk ρ (larger dense cluster)",
            "composition": {"ACO": 48},
            "bulk_density_fraction": 1.25,
            "cutoff": 13.0,
        },
    ]
  # fmt: on


def charmm_liquid_density_cases() -> list[dict[str, object]]:
    """CGENFF clusters at bulk liquid densities (requires PyCHARMM)."""
  # fmt: off
    return [
        {
            "name": "charmm_aco_liquid_n16",
            "description": "ACO:16 CGENFF cluster, cubic box from bulk acetone ρ",
            "composition": "ACO:16",
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_dcm_liquid_n16",
            "description": "DCM:16 CGENFF cluster, cubic box from bulk DCM ρ",
            "composition": "DCM:16",
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_aco_liquid_n32",
            "description": "ACO:32 CGENFF cluster, cubic box from bulk acetone ρ",
            "composition": "ACO:32",
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_dcm_liquid_n32",
            "description": "DCM:32 CGENFF cluster, cubic box from bulk DCM ρ",
            "composition": "DCM:32",
            "bulk_density_fraction": 1.0,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_aco_liquid_box25",
            "description": "ACO:N capped at bulk ρ in a 25 Å cube (N≤32)",
            "solvent": "ACO",
            "box_side": 25.0,
            "bulk_density_fraction": 1.0,
            "max_monomers": 32,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_dcm_liquid_box25",
            "description": "DCM:N capped at bulk ρ in a 25 Å cube (N≤32)",
            "solvent": "DCM",
            "box_side": 25.0,
            "bulk_density_fraction": 1.0,
            "max_monomers": 32,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_aco_liquid_n32_rho125",
            "description": "ACO:32 CGENFF at 1.25× bulk ρ",
            "composition": "ACO:32",
            "bulk_density_fraction": 1.25,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_aco_liquid_n32_rho150",
            "description": "ACO:32 CGENFF at 1.50× bulk ρ",
            "composition": "ACO:32",
            "bulk_density_fraction": 1.5,
            "cutoff": 13.0,
        },
        {
            "name": "charmm_dcm_liquid_n32_rho150",
            "description": "DCM:32 CGENFF at 1.50× bulk ρ",
            "composition": "DCM:32",
            "bulk_density_fraction": 1.5,
            "cutoff": 13.0,
        },
    ]
  # fmt: on


def _composition_dict_from_liquid_case(case: dict[str, object]) -> dict[str, int]:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import parse_composition_dict

    if "composition" in case:
        if isinstance(case["composition"], dict):
            return {str(k): int(v) for k, v in case["composition"].items()}
        return parse_composition_dict(str(case["composition"])) or {}
    solvent = str(case["solvent"]).strip().upper()
    from workflows.pbc_solvent_burst.scripts.bulk_density import n_monomers_at_bulk_density

    n = n_monomers_at_bulk_density(
        solvent,
        float(case["box_side"]),
        float(case.get("bulk_density_fraction", 1.0)),
    )
    max_n = case.get("max_monomers")
    if max_n is not None:
        n = min(int(n), int(max_n))
    return {solvent: int(n)}


def build_liquid_density_synthetic_case(
    case: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, float, float]:
    """Build synthetic liquid-density geometry.

    Returns geometry, cutoff, description, box_side, target_density_g_cm3.
    """
    comp = _composition_dict_from_liquid_case(case)
    n_mol = int(sum(comp.values()))
    side, rho = liquid_density_box_side_for_composition(
        comp,
        bulk_density_fraction=float(case.get("bulk_density_fraction", 1.0)),
        target_density_g_cm3=case.get("target_density_g_cm3"),  # type: ignore[arg-type]
        box_side=case.get("box_side"),  # type: ignore[arg-type]
    )
    coms = liquid_density_grid_com_positions(n_mol, side)
    positions, cell, offsets, monomer_id = synthetic_monomer_cluster(
        com_positions=coms,
        cell=float(side),
    )
    cutoff = float(case["cutoff"])
    desc = str(case.get("description", case.get("name", "liquid")))
    return positions, cell, offsets, monomer_id, cutoff, desc, side, rho


def setup_charmm_liquid_density_cluster(
    case: dict[str, object],
    *,
    cutoff: float = 13.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Build CGENFF cluster in a liquid-density cubic box with PBC nbonds.

    Returns ``(positions, cell, offsets, monomer_id, atomic_numbers, eff_cutoff,
    box_side, target_density_g_cm3)``.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    comp = _composition_dict_from_liquid_case(case)
    composition = ",".join(f"{res}:{count}" for res, count in comp.items())
    n_mol = int(sum(comp.values()))
    side, rho = liquid_density_box_side_for_composition(
        comp,
        bulk_density_fraction=float(case.get("bulk_density_fraction", 1.0)),
        target_density_g_cm3=case.get("target_density_g_cm3"),  # type: ignore[arg-type]
        box_side=case.get("box_side"),  # type: ignore[arg-type]
    )
    spacing = liquid_density_spacing(side, n_mol)
    positions, cell, offsets, monomer_id, atomic_numbers = setup_charmm_composition_cluster(
        composition,
        box_side=float(side),
        spacing=float(spacing),
    )
    prepare_charmm_pbc(float(side))
    cuts = apply_pbc_nbonds(nbxmod=5, cutnb=float(cutoff), cubic_box_side_A=float(side))
    eff_cutoff = float(cuts.cutnb)
    return (
        positions,
        cell,
        offsets,
        monomer_id,
        atomic_numbers,
        eff_cutoff,
        float(side),
        rho,
    )


def setup_charmm_aco_dimer_cluster(
    *,
    box_side: float = 40.0,
    com_separation: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2×ACO dimer PSF in PyCHARMM for NL integration tests.

    Returns the same tuple as :func:`two_dimer_cluster` but with a loaded PSF
    (20 atoms, 10 per monomer) and geometry centered in a cubic box.
    """
    positions, cell, offsets, monomer_id, _z = setup_charmm_composition_cluster(
        "ACO:2",
        box_side=box_side,
        spacing=com_separation,
    )
    return positions, cell, offsets, monomer_id
