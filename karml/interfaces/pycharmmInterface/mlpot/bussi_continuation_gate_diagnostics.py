"""Structured diagnostics when the Bussi heat continuation GRMS gate fires.

Writes a JSON report under ``cleanup/`` so late-HEAT liquid blow-ups
(OH collapse, force outliers) are inspectable without re-running MD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Dump sizing (diagnostics only; not dynamics cutoffs).
BUSSI_GATE_DUMP_TOP_FORCE_ATOMS = 12
BUSSI_GATE_DUMP_TOP_BONDS = 24
BUSSI_GATE_DUMP_TOP_MONOMER_OUTLIERS = 16

# Atomic numbers used to label water-like 1–2 bonds in the dump.
ATOMIC_NUMBER_H = 1
ATOMIC_NUMBER_O = 8

# Monomer GRMS outlier: report monomers above max(floor, factor × median).
BUSSI_GATE_MONOMER_GRMS_OUTLIER_FACTOR = 3.0
BUSSI_GATE_MONOMER_GRMS_OUTLIER_FLOOR_KCALMOL_A = 5.0

# Restart-vs-live: treat coords as unchanged below this RMSD (Å).
BUSSI_GATE_RESTART_LIVE_RMSD_EQUAL_A = 1.0e-4

DIAGNOSTICS_SCHEMA = "mmml.bussi_continuation_gate_diagnostics.v1"


def bond_type_from_atomic_numbers(z_a: int, z_b: int) -> str:
    """Classify a 1–2 pair as OH / HH / OO / other from atomic numbers."""
    a, b = int(z_a), int(z_b)
    if {a, b} == {ATOMIC_NUMBER_O, ATOMIC_NUMBER_H}:
        return "OH"
    if a == ATOMIC_NUMBER_H and b == ATOMIC_NUMBER_H:
        return "HH"
    if a == ATOMIC_NUMBER_O and b == ATOMIC_NUMBER_O:
        return "OO"
    return "other"


def mic_delta_angstrom(
    delta: np.ndarray,
    box_side_A: float | None,
) -> np.ndarray:
    """Minimum-image displacement for a cubic cell (identity when ``box_side`` is None)."""
    d = np.asarray(delta, dtype=np.float64)
    if box_side_A is None:
        return d
    side = float(box_side_A)
    if not np.isfinite(side) or side <= 0.0:
        return d
    return d - side * np.round(d / side)


def nearest_neighbor_distance_angstrom(
    positions_A: np.ndarray,
    atom_index: int,
    *,
    box_side_A: float | None = None,
) -> tuple[int | None, float | None]:
    """Return ``(neighbor_index, distance_A)`` for the closest other atom."""
    pos = np.asarray(positions_A, dtype=np.float64).reshape(-1, 3)
    ai = int(atom_index)
    if pos.shape[0] < 2 or not (0 <= ai < pos.shape[0]):
        return None, None
    delta = pos - pos[ai]
    delta = mic_delta_angstrom(delta, box_side_A)
    dist = np.linalg.norm(delta, axis=1)
    dist[ai] = np.inf
    nj = int(np.argmin(dist))
    d = float(dist[nj])
    if not np.isfinite(d):
        return None, None
    return nj, d


def build_top_force_atom_records(
    forces_kcalmol_A: np.ndarray,
    positions_A: np.ndarray,
    *,
    atomic_numbers: np.ndarray | None,
    atom_to_monomer: np.ndarray | None,
    top_n: int = BUSSI_GATE_DUMP_TOP_FORCE_ATOMS,
    box_side_A: float | None = None,
) -> list[dict[str, Any]]:
    """Highest-|F| atoms with element, monomer, and nearest-neighbor distance."""
    from ase.data import chemical_symbols

    forces = np.asarray(forces_kcalmol_A, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions_A, dtype=np.float64).reshape(-1, 3)
    if forces.shape[0] == 0 or forces.shape[0] != pos.shape[0]:
        return []
    mags = np.linalg.norm(forces, axis=1)
    z = None if atomic_numbers is None else np.asarray(atomic_numbers, dtype=int).reshape(-1)
    mono = (
        None
        if atom_to_monomer is None
        else np.asarray(atom_to_monomer, dtype=int).reshape(-1)
    )
    order = np.argsort(mags)[::-1][: max(1, int(top_n))]
    out: list[dict[str, Any]] = []
    for rank, ai in enumerate(order, start=1):
        ai = int(ai)
        elem = "?"
        if z is not None and 0 <= ai < z.shape[0]:
            zi = int(z[ai])
            if 0 <= zi < len(chemical_symbols):
                elem = chemical_symbols[zi]
        nn_idx, nn_dist = nearest_neighbor_distance_angstrom(
            pos, ai, box_side_A=box_side_A
        )
        out.append(
            {
                "rank": int(rank),
                "atom_index": ai,
                "monomer_index": int(mono[ai]) if mono is not None and ai < mono.shape[0] else None,
                "element": elem,
                "force_mag_kcalmol_A": float(mags[ai]),
                "nearest_neighbor_atom_index": nn_idx,
                "nearest_neighbor_distance_A": nn_dist,
            }
        )
    return out


def build_worst_bond_records(
    positions_A: np.ndarray,
    bond_pairs: Sequence[tuple[int, int]],
    *,
    atomic_numbers: np.ndarray | None,
    atom_to_monomer: np.ndarray | None = None,
    top_n: int = BUSSI_GATE_DUMP_TOP_BONDS,
    bond_types: Sequence[str] | None = ("OH", "HH"),
) -> list[dict[str, Any]]:
    """Shortest 1–2 bonds (default OH/HH), sorted ascending by length."""
    pos = np.asarray(positions_A, dtype=np.float64).reshape(-1, 3)
    z = None if atomic_numbers is None else np.asarray(atomic_numbers, dtype=int).reshape(-1)
    mono = (
        None
        if atom_to_monomer is None
        else np.asarray(atom_to_monomer, dtype=int).reshape(-1)
    )
    want = None if bond_types is None else {str(t) for t in bond_types}
    rows: list[dict[str, Any]] = []
    for a, b in bond_pairs:
        ia, ib = int(a), int(b)
        if not (0 <= ia < pos.shape[0] and 0 <= ib < pos.shape[0]):
            continue
        if z is not None and ia < z.shape[0] and ib < z.shape[0]:
            btype = bond_type_from_atomic_numbers(int(z[ia]), int(z[ib]))
        else:
            btype = "other"
        if want is not None and btype not in want:
            continue
        length = float(np.linalg.norm(pos[ib] - pos[ia]))
        if not np.isfinite(length):
            continue
        mi = None
        if mono is not None and ia < mono.shape[0] and ib < mono.shape[0]:
            if int(mono[ia]) == int(mono[ib]):
                mi = int(mono[ia])
        rows.append(
            {
                "atom_i": ia,
                "atom_j": ib,
                "bond_type": btype,
                "distance_A": length,
                "monomer_index": mi,
            }
        )
    rows.sort(key=lambda r: float(r["distance_A"]))
    out = rows[: max(1, int(top_n))] if rows else []
    for rank, row in enumerate(out, start=1):
        row["rank"] = int(rank)
    return out


def build_monomer_grms_outlier_records(
    forces_kcalmol_A: np.ndarray,
    atoms_per_list: Sequence[int],
    *,
    top_n: int = BUSSI_GATE_DUMP_TOP_MONOMER_OUTLIERS,
    outlier_factor: float = BUSSI_GATE_MONOMER_GRMS_OUTLIER_FACTOR,
    floor_kcalmol_A: float = BUSSI_GATE_MONOMER_GRMS_OUTLIER_FLOOR_KCALMOL_A,
) -> dict[str, Any]:
    """Per-monomer GRMS vs median; list monomers well above the bulk."""
    from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
        per_monomer_fmax_from_forces,
        per_monomer_grms_from_forces,
    )

    counts = [int(x) for x in atoms_per_list]
    if not counts:
        return {
            "median_grms_kcalmol_A": None,
            "outlier_threshold_kcalmol_A": None,
            "outliers": [],
        }
    grms = per_monomer_grms_from_forces(forces_kcalmol_A, counts)
    fmax = per_monomer_fmax_from_forces(forces_kcalmol_A, counts)
    finite = grms[np.isfinite(grms)]
    median = float(np.median(finite)) if finite.size else None
    threshold = None
    if median is not None:
        threshold = max(float(floor_kcalmol_A), float(outlier_factor) * median)
    outliers: list[dict[str, Any]] = []
    for mi, (g, f) in enumerate(zip(grms, fmax, strict=True)):
        g = float(g)
        if not np.isfinite(g):
            continue
        if threshold is not None and g < threshold:
            continue
        outliers.append(
            {
                "monomer_index": int(mi),
                "grms_kcalmol_A": g,
                "fmax_kcalmol_A": float(f),
                "ratio_to_median": (g / median) if median and median > 0.0 else None,
            }
        )
    outliers.sort(key=lambda r: float(r["grms_kcalmol_A"]), reverse=True)
    return {
        "median_grms_kcalmol_A": median,
        "outlier_threshold_kcalmol_A": threshold,
        "outlier_factor": float(outlier_factor),
        "floor_kcalmol_A": float(floor_kcalmol_A),
        "outliers": outliers[: max(1, int(top_n))],
    }


def build_restart_vs_live_record(
    live_positions_A: np.ndarray,
    restart_positions_A: np.ndarray | None,
    *,
    restart_path: str | Path | None,
    equal_rmsd_A: float = BUSSI_GATE_RESTART_LIVE_RMSD_EQUAL_A,
) -> dict[str, Any]:
    """Compare live CHARMM coords to a restart frame (gate-time geometry identity)."""
    live = np.asarray(live_positions_A, dtype=np.float64).reshape(-1, 3)
    out: dict[str, Any] = {
        "restart_path": str(restart_path) if restart_path is not None else None,
        "n_atoms_live": int(live.shape[0]),
        "restart_readable": restart_positions_A is not None,
        "rmsd_angstrom": None,
        "max_abs_delta_angstrom": None,
        "coords_differ": None,
    }
    if restart_positions_A is None:
        out["note"] = "restart coordinates unavailable"
        return out
    restart = np.asarray(restart_positions_A, dtype=np.float64).reshape(-1, 3)
    out["n_atoms_restart"] = int(restart.shape[0])
    if restart.shape != live.shape:
        out["coords_differ"] = True
        out["note"] = "atom-count / shape mismatch between live and restart"
        return out
    delta = live - restart
    rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    out["rmsd_angstrom"] = rmsd
    out["max_abs_delta_angstrom"] = max_abs
    out["coords_differ"] = bool(rmsd > float(equal_rmsd_A))
    if not out["coords_differ"]:
        out["note"] = (
            "live matches restart within tolerance — early-abort restore from this "
            "file will not change geometry"
        )
    else:
        out["note"] = "live differs from restart — restore may change geometry"
    return out


def atom_to_monomer_index(
    n_atoms: int,
    atoms_per_list: Sequence[int] | None,
) -> np.ndarray:
    """Map atom index → monomer index from ``atoms_per_list`` (single fragment fallback)."""
    counts = [int(x) for x in atoms_per_list] if atoms_per_list else [int(n_atoms)]
    mapping = np.zeros(int(n_atoms), dtype=int)
    start = 0
    for mi, n in enumerate(counts):
        mapping[start : start + n] = mi
        start += n
    return mapping


def build_bussi_continuation_gate_diagnostics(
    *,
    overlap_context: str,
    global_step: int,
    gate_grms_kcalmol_A: float,
    gate_limit_kcalmol_A: float,
    forces_kcalmol_A: np.ndarray,
    positions_A: np.ndarray,
    atomic_numbers: np.ndarray | None,
    atoms_per_list: Sequence[int] | None,
    bond_pairs: Sequence[tuple[int, int]],
    microchunk_series: Sequence[Mapping[str, Any]] | None = None,
    restart_positions_A: np.ndarray | None = None,
    restart_path: str | Path | None = None,
    box_side_A: float | None = None,
) -> dict[str, Any]:
    """Assemble the serializable gate-abort diagnostic payload."""
    pos = np.asarray(positions_A, dtype=np.float64).reshape(-1, 3)
    forces = np.asarray(forces_kcalmol_A, dtype=np.float64).reshape(-1, 3)
    mono = atom_to_monomer_index(pos.shape[0], atoms_per_list)
    counts = [int(x) for x in atoms_per_list] if atoms_per_list else [int(pos.shape[0])]
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "overlap_context": str(overlap_context),
        "global_step": int(global_step),
        "gate_grms_kcalmol_A": float(gate_grms_kcalmol_A),
        "gate_limit_kcalmol_A": float(gate_limit_kcalmol_A),
        "units": {
            "force": "kcal/mol/A",
            "distance": "A",
            "temperature": "K",
            "energy": "kcal/mol",
        },
        "box_side_A": float(box_side_A) if box_side_A is not None else None,
        "top_force_atoms": build_top_force_atom_records(
            forces,
            pos,
            atomic_numbers=atomic_numbers,
            atom_to_monomer=mono,
            box_side_A=box_side_A,
        ),
        "worst_bonds": build_worst_bond_records(
            pos,
            bond_pairs,
            atomic_numbers=atomic_numbers,
            atom_to_monomer=mono,
        ),
        "microchunk_series": [dict(row) for row in (microchunk_series or ())],
        "restart_vs_live": build_restart_vs_live_record(
            pos,
            restart_positions_A,
            restart_path=restart_path,
        ),
        "monomer_grms": build_monomer_grms_outlier_records(forces, counts),
    }


def resolve_bussi_gate_diagnostics_path(
    args: Any | None,
    *,
    global_step: int,
) -> Path | None:
    """``<cleanup>/bussi_continuation_gate_step{N}.json`` or ``None`` if disabled."""
    from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
        resolve_cleanup_dir,
        resolve_output_dir,
    )

    cleanup = resolve_cleanup_dir(args)
    if cleanup is not None:
        return cleanup / f"bussi_continuation_gate_step{int(global_step)}.json"
    root = resolve_output_dir(args)
    if root is None:
        return None
    return root / f"bussi_continuation_gate_step{int(global_step)}.json"


def write_bussi_continuation_gate_diagnostics(
    payload: Mapping[str, Any],
    path: Path,
) -> Path:
    """Write diagnostics JSON; return the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out.resolve()


def dump_bussi_continuation_gate_diagnostics(
    mlpot_ctx: Any | None,
    *,
    overlap_context: str,
    global_step: int,
    gate_grms_kcalmol_A: float,
    gate_limit_kcalmol_A: float,
    microchunk_series: Sequence[Mapping[str, Any]] | None = None,
    restart_path: str | Path | None = None,
) -> Path | None:
    """Read live CHARMM state, build diagnostics, write under ``cleanup/``.

    Best-effort: never raises into the dynamics loop. Returns the written path
    or ``None`` when unavailable.
    """
    args = getattr(mlpot_ctx, "workflow_args", None) if mlpot_ctx is not None else None
    if bool(getattr(args, "no_heat_abort_force_dump", False)):
        return None
    out_path = resolve_bussi_gate_diagnostics_path(args, global_step=global_step)
    if out_path is None:
        print(
            f"{overlap_context}: Bussi gate diagnostics skipped (no output_dir)",
            flush=True,
        )
        return None

    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_grms_after_ener_force,
            charmm_positions_angstrom,
            charmm_total_forces_kcalmol_A,
        )

        charmm_grms_after_ener_force(silent=True)
        forces = np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64).reshape(
            -1, 3
        )
        positions = np.asarray(charmm_positions_angstrom(), dtype=np.float64).reshape(
            -1, 3
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"{overlap_context}: Bussi gate diagnostics unavailable ({exc})",
            flush=True,
        )
        return None

    z = getattr(mlpot_ctx, "ml_Z", None) if mlpot_ctx is not None else None
    atoms_per = getattr(args, "_cluster_atoms_per_list", None) if args is not None else None
    if atoms_per is None and mlpot_ctx is not None:
        atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)

    bond_pairs: list[tuple[int, int]] = []
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.monomer_geometry_limits import (
            psf_bond_pairs_0based,
        )

        bond_pairs = list(psf_bond_pairs_0based(exclude_1_3=False))
    except Exception:
        bond_pairs = []

    restart_pos = None
    if restart_path is not None:
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                read_restart_coordinates,
            )

            restart_pos = read_restart_coordinates(Path(restart_path))
        except Exception:
            restart_pos = None

    box_side = None
    if mlpot_ctx is not None:
        box_side = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)

    payload = build_bussi_continuation_gate_diagnostics(
        overlap_context=overlap_context,
        global_step=global_step,
        gate_grms_kcalmol_A=gate_grms_kcalmol_A,
        gate_limit_kcalmol_A=gate_limit_kcalmol_A,
        forces_kcalmol_A=forces,
        positions_A=positions,
        atomic_numbers=None if z is None else np.asarray(z, dtype=int),
        atoms_per_list=atoms_per,
        bond_pairs=bond_pairs,
        microchunk_series=microchunk_series,
        restart_positions_A=restart_pos,
        restart_path=restart_path,
        box_side_A=float(box_side) if box_side is not None else None,
    )
    try:
        written = write_bussi_continuation_gate_diagnostics(payload, out_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"{overlap_context}: Bussi gate diagnostics write failed ({exc})",
            flush=True,
        )
        return None
    print(
        f"{overlap_context}: Bussi continuation-gate diagnostics → {written}",
        flush=True,
    )
    return written


def sample_bussi_microchunk_metrics(
    *,
    global_step: int,
    target_temperature_K: float | None = None,
) -> dict[str, Any]:
    """Best-effort post-subchunk GRMS / T / E sample for the microchunk series."""
    row: dict[str, Any] = {
        "global_step": int(global_step),
        "grms_kcalmol_A": None,
        "temperature_K": None,
        "target_temperature_K": (
            float(target_temperature_K) if target_temperature_K is not None else None
        ),
        "energy_kcalmol": None,
    }
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_energy_row,
            charmm_system_is_evaluable,
            charmm_grms,
        )

        # ``charmm_energy_row`` runs ``ENER`` which fatally aborts CHARMM when no
        # system is loaded; the abort is not a catchable Python exception.
        if not charmm_system_is_evaluable():
            return row
        row["grms_kcalmol_A"] = float(charmm_grms())
        ener = charmm_energy_row()
        for key in ("ENER", "energy", "TOTE", "totener"):
            if key in ener and np.isfinite(ener[key]):
                row["energy_kcalmol"] = float(ener[key])
                break
    except Exception:
        pass
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
            charmm_masses_amu,
            charmm_velocities_akma_for_thermostat,
            estimate_kinetic_temperature_k,
        )

        live = estimate_kinetic_temperature_k(
            charmm_velocities_akma_for_thermostat(),
            charmm_masses_amu(),
        )
        if live is not None and np.isfinite(live):
            row["temperature_K"] = float(live)
    except Exception:
        pass
    return row
