"""Cross-backend MD handoff: geometry, velocities, box, and checkpoint I/O."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_handoff_in: ContextVar["MdHandoffState | None"] = ContextVar("md_handoff_in", default=None)
_handoff_out: ContextVar["MdHandoffState | None"] = ContextVar("md_handoff_out", default=None)

_FORTRAN_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+)[DEde][+-]?\d+",
    re.IGNORECASE,
)


@dataclass
class MdHandoffState:
  positions: np.ndarray
  atomic_numbers: np.ndarray
  velocities: np.ndarray | None = None
  cell: np.ndarray | None = None
  pbc: bool = False
  temperature_K: float | None = None
  pressure_atm: float | None = None
  step: int | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.positions = np.asarray(self.positions, dtype=np.float64)
    self.atomic_numbers = np.asarray(self.atomic_numbers, dtype=np.int32)
    if self.velocities is not None:
      self.velocities = np.asarray(self.velocities, dtype=np.float64)
    if self.cell is not None:
      self.cell = np.asarray(self.cell, dtype=np.float64)


def handoff_from_atoms(
    atoms: Any,
    *,
    velocities: np.ndarray | None = None,
    temperature_K: float | None = None,
    pressure_atm: float | None = None,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MdHandoffState:
    cell = None
    pbc = bool(getattr(atoms, "pbc", False) is not None and np.asarray(atoms.pbc).any())
    if pbc:
        cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    vel = velocities
    if vel is None:
        v = atoms.get_velocities()
        if v is not None:
            vel = np.asarray(v, dtype=np.float64)
    return MdHandoffState(
        positions=np.asarray(atoms.get_positions(), dtype=np.float64),
        atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
        velocities=vel,
        cell=cell,
        pbc=pbc,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        step=step,
        metadata=dict(metadata or {}),
    )


def handoff_from_charmm(
    atomic_numbers: np.ndarray,
    *,
    restart_path: Path | str | None = None,
    fallback_box_side_A: float | None = None,
    temperature_K: float | None = None,
    pressure_atm: float | None = None,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MdHandoffState:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
        read_restart_velocities,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
        _charmm_velocities_array,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    positions = get_charmm_positions_array()
    velocities = _charmm_velocities_array()
    velocities_source = "charmm_live" if velocities is not None else None

    restart_p: Path | None = None
    if restart_path is not None:
        cand = Path(restart_path).expanduser()
        if cand.is_file():
            restart_p = cand.resolve()

    # Guard: if CHARMM returned all-zero positions, try to recover from the restart
    # file before using the zeros (which would produce an invalid simulation).
    if positions.size > 0 and np.allclose(positions, 0.0):
        recovered = None
        if restart_p is not None:
            try:
                recovered = read_restart_coordinates(restart_p)
            except Exception:
                pass
        if recovered is not None and not np.allclose(recovered, 0.0):
            import warnings
            warnings.warn(
                "handoff_from_charmm: CHARMM in-memory positions are all zero; "
                f"recovered non-zero coordinates from restart file {restart_p}. "
                "This usually means CHARMM PSF was rebuilt but 'read coor' was not called.",
                stacklevel=2,
            )
            positions = recovered
        else:
            raise RuntimeError(
                "handoff_from_charmm: CHARMM returned all-zero positions "
                f"({positions.shape[0]} atoms). "
                "The PSF may have been rebuilt without loading coordinates "
                "(no 'read coor' / DCD frame read). "
                "Ensure coordinates are synced before capturing the handoff state, "
                "or pass a valid restart_path with non-zero coordinates."
                + (
                    f" Restart file {restart_p} also has zero/missing coordinates."
                    if restart_p is not None
                    else " No restart file was provided for fallback."
                )
            )

    cell = None
    pbc = False
    box_side_source: str | None = None
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            cubic_box_matrix_from_side,
            resolve_charmm_cubic_box_side_A,
        )

        side, box_side_source = resolve_charmm_cubic_box_side_A(
            restart_path=restart_p,
            fallback_side_A=fallback_box_side_A,
        )
        cell = cubic_box_matrix_from_side(side)
        pbc = True
    except Exception:
        if restart_p is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                parse_cubic_box_side_from_charmm_restart,
            )

            box_a = parse_cubic_box_side_from_charmm_restart(restart_p)
            cell = _cell_from_scalar(box_a)
            if cell is not None:
                pbc = True
                box_side_source = "restart"
        if cell is None and fallback_box_side_A is not None and float(fallback_box_side_A) > 0:
            cell = _cell_from_scalar(float(fallback_box_side_A))
            pbc = True
            box_side_source = "fallback"

    if velocities is None and restart_p is not None:
        velocities = read_restart_velocities(restart_p)
        if velocities is not None:
            velocities_source = "restart"

    meta = dict(metadata or {})
    if restart_p is not None:
        meta.setdefault("restart_path", str(restart_p))
    if box_side_source is not None:
        meta["box_side_source"] = box_side_source
    if velocities_source is not None:
        meta["velocities_source"] = velocities_source

    return MdHandoffState(
        positions=positions,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        velocities=velocities,
        cell=cell,
        pbc=pbc,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        step=step,
        metadata=meta,
    )


def _resolve_existing_file_path(raw_path: str | Path | None) -> Path | None:
    """Robustly resolve path to an existing file, handling mount/node mismatches."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()

    parts = list(path.parts)
    # Find root directories at the base of the repository
    for trigger in ("artifacts", "workflows", "examples", "results"):
        if trigger in parts:
            idx = parts.index(trigger)
            rel_path = Path(*parts[idx:])
            repo_root = Path(__file__).resolve().parents[3]
            cand = (repo_root / rel_path).resolve()
            if cand.is_file():
                return cand

            cand_cwd = (Path.cwd() / rel_path).resolve()
            if cand_cwd.is_file():
                return cand_cwd

    filename = path.name
    if len(parts) >= 3:
        parent_name = parts[-2]
        grandparent_name = parts[-3]
        repo_root = Path(__file__).resolve().parents[3]
        cand = (repo_root / "artifacts" / "pbc_solvent_burst" / grandparent_name / parent_name / filename).resolve()
        if cand.is_file():
            return cand

    return None


def _find_any_res_file_in_same_dir(raw_path: str | Path | None, handoff: MdHandoffState | None = None) -> Path | None:
    """Search for any usable .res file in the resolved directory of raw_path."""
    if not raw_path:
        return None
    try:
        path = Path(raw_path).expanduser()
    except Exception:
        return None

    dirs_to_search: list[Path] = []

    # 1. Direct parent if the directory exists
    try:
        if path.parent.is_dir():
            dirs_to_search.append(path.parent)
            if path.parent.name == "handoff" and path.parent.parent.is_dir():
                dirs_to_search.append(path.parent.parent)
    except Exception:
        pass

    # 2. Trigger-based resolution of parent directory
    try:
        parts = list(path.parts)
        for trigger in ("artifacts", "workflows", "examples", "results"):
            if trigger in parts:
                idx = parts.index(trigger)
                rel_dir = Path(*parts[idx:-1]) # up to the parent directory
                repo_root = Path(__file__).resolve().parents[3]
                cand_dir = (repo_root / rel_dir).resolve()
                if cand_dir.is_dir():
                    dirs_to_search.append(cand_dir)
                if cand_dir.parent.is_dir():
                    dirs_to_search.append(cand_dir.parent)

                cand_cwd_dir = (Path.cwd() / rel_dir).resolve()
                if cand_cwd_dir.is_dir():
                    dirs_to_search.append(cand_cwd_dir)
                if cand_cwd_dir.parent.is_dir():
                    dirs_to_search.append(cand_cwd_dir.parent)
    except Exception:
        pass

    # 3. Fallback based on grandparent/parent/filename structure
    try:
        parts = list(path.parts)
        if len(parts) >= 3:
            parent_name = parts[-2]
            grandparent_name = parts[-3]
            repo_root = Path(__file__).resolve().parents[3]
            cand_dir = (repo_root / "artifacts" / "pbc_solvent_burst" / grandparent_name / parent_name).resolve()
            if cand_dir.is_dir():
                dirs_to_search.append(cand_dir)
            if cand_dir.parent.is_dir():
                dirs_to_search.append(cand_dir.parent)
    except Exception:
        pass

    # Search the collected directories for any usable .res file
    seen_dirs = set()
    for d in dirs_to_search:
        try:
            resolved_d = d.resolve()
            if resolved_d in seen_dirs:
                continue
            seen_dirs.add(resolved_d)

            candidates = []
            expected_natom = len(handoff.positions) if handoff is not None else None
            for file_path in resolved_d.glob("*.res"):
                if file_path.is_file():
                    try:
                        if _is_usable_restart_template(file_path, expected_natom=expected_natom):
                            name = file_path.name.lower()
                            if "overlap" not in name and not name.startswith("continue_seed"):
                                candidates.append((0, file_path))
                            elif "overlap" in name:
                                candidates.append((1, file_path))
                    except Exception:
                        pass

            if candidates:
                # Sort by priority (normal res first, then overlap), then modified time (newest first)
                candidates.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
                return candidates[0][1]
        except Exception:
            pass

    return None


def resolve_handoff_restart_template(
    handoff: MdHandoffState,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> Path | None:
    """Best restart template for patching handoff coords/velocities/cell into CHARMM format."""
    explicit = getattr(args, "handoff_template_res", None)
    if explicit:
        resolved = _resolve_existing_file_path(explicit)
        if resolved is not None:
            return resolved

    expected_natom = len(handoff.positions)
    meta = handoff.metadata or {}
    for key in ("restart_path", "path"):
        raw = meta.get(key)
        if not raw:
            continue
        resolved = _resolve_existing_file_path(raw)
        if resolved is not None and resolved.suffix.lower() == ".res" and _is_usable_restart_template(resolved, expected_natom=expected_natom):
            return resolved

    continue_from = getattr(args, "continue_from", None)
    if continue_from:
        cf = _resolve_existing_file_path(continue_from)
        if cf is not None:
            if cf.suffix.lower() == ".res":
                if _is_usable_restart_template(cf, expected_natom=expected_natom):
                    return cf
            else:
                final_res = cf.parent / "final.res"
                resolved_final = _resolve_existing_file_path(final_res)
                if resolved_final is not None and _is_usable_restart_template(resolved_final, expected_natom=expected_natom):
                    return resolved_final

    for cand in (paths.get("heat_res"), paths.get("equi_res"), paths.get("prod_res")):
        if cand is not None:
            resolved = _resolve_existing_file_path(cand)
            if resolved is not None and _is_usable_restart_template(resolved, expected_natom=expected_natom):
                return resolved

    for backup in _overlap_scratch_restart_backups(paths):
        resolved = _resolve_existing_file_path(backup)
        if resolved is not None and _is_usable_restart_template(resolved, expected_natom=expected_natom):
            return resolved

    # Final fallback: search for ANY .res file in the directories of any reference paths we have
    for raw in [
        explicit,
        meta.get("restart_path"),
        meta.get("path"),
        continue_from,
    ] + list(paths.values()):
        if raw:
            found = _find_any_res_file_in_same_dir(raw, handoff)
            if found is not None:
                return found

    return None


def prepare_pycharmm_handoff_continuation(
    handoff: MdHandoffState,
    args: argparse.Namespace,
    out_dir: Path,
    paths: dict[str, Path],
    *,
    quiet: bool = False,
) -> Path | None:
    """Write ``handoff/continue_seed.res`` and ``READ restart`` it into CHARMM."""
    if getattr(args, "restart_from", None):
        path = Path(str(args.restart_from)).expanduser().resolve()
        return path if path.is_file() else None

    from dataclasses import replace

    seed = Path(out_dir) / "handoff" / "continue_seed.res"
    seed.parent.mkdir(parents=True, exist_ok=True)

    payload = handoff
    if handoff.velocities is not None and not getattr(args, "continue_velocities", True):
        payload = replace(handoff, velocities=None)

    # Whether we wrote the restart synthetically (bypasses CHARMM ``read restart``
    # since CHARMM's Fortran reader may not accept the synthetic header).
    _synthetic_restart = False

    template = resolve_handoff_restart_template(handoff, args, paths)
    if template is not None:
        save_handoff_to_res(payload, seed, template_res=template)
    else:
        try:
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                rewrite_dynamics_restart_validated,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

            sync_charmm_positions(payload.positions)
            if rewrite_dynamics_restart_validated(seed):
                # In-memory restart is valid; continue normally
                pass
            else:
                # CHARMM ``write restart`` wrote an empty coordinate section (no
                # dynamics have run in this session yet).  Fall back to synthesising
                # the restart file directly from handoff data.
                if not quiet:
                    print(
                        "Handoff continuation: in-memory restart write produced no "
                        "coordinates; writing synthetic restart from handoff positions.",
                        flush=True,
                    )
                try:
                    _write_synthetic_charmm_restart(payload, seed)
                    _synthetic_restart = True
                except Exception as exc:
                    if not quiet:
                        print(
                            f"Handoff continuation: synthetic restart write failed ({exc}); "
                            "PyCHARMM dynamics may cold-start velocities.",
                            flush=True,
                        )
                    # Try any .res in the same dir as a final fallback
                    fallback = _find_any_res_file_in_same_dir(seed, handoff)
                    if fallback is not None:
                        seed = fallback
                    else:
                        return None
        except Exception:
            if not quiet:
                print(
                    "Handoff continuation: no restart template; "
                    "PyCHARMM dynamics may cold-start velocities.",
                    flush=True,
                )
            return None

    if _synthetic_restart:
        # Coordinates already in CHARMM from sync_charmm_positions above.
        # Sync velocities if present; skip CHARMM ``read restart`` (incompatible
        # with synthetic header format).
        try:
            if payload.velocities is not None:
                try:
                    _sync_charmm_velocities(payload.velocities)
                except Exception:
                    pass
        except Exception:
            pass
        if not quiet:
            parts = ["coordinates (synthetic restart)"]
            if payload.velocities is not None:
                parts.append("velocities")
            if payload.cell is not None:
                parts.append("cell")
            print(
                f"Handoff continuation: applied {seed.name} in-memory ({', '.join(parts)})",
                flush=True,
            )
    else:
        from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
            restore_charmm_state_from_restart,
        )

        restore_charmm_state_from_restart(seed)
        if not quiet:
            parts = ["coordinates"]
            if payload.velocities is not None:
                parts.append("velocities")
            if payload.cell is not None:
                parts.append("cell")
            print(
                f"Handoff continuation: loaded {seed} ({', '.join(parts)})",
                flush=True,
            )
    seed = seed.resolve()
    args.restart_from = seed
    return seed


def cluster_layout_from_composition_string(
    composition: str,
    *,
    n_atoms: int,
) -> tuple[list[int], list[str], dict[str, int]]:
    """Per-monomer atom counts from ``RES:COUNT`` composition and handoff size."""
    parts = [p.strip() for p in str(composition).split(",") if p.strip()]
    comp: list[tuple[str, int]] = []
    for part in parts:
        if ":" not in part:
            raise ValueError(f"invalid composition token {part!r}; use RES:COUNT")
        res, cnt = part.split(":", 1)
        comp.append((res.strip().upper(), int(cnt)))
    composition_summary = {res: int(cnt) for res, cnt in comp}
    n_mol = sum(composition_summary.values())
    if n_mol <= 0:
        raise ValueError("composition must include at least one monomer")
    if int(n_atoms) % int(n_mol) != 0:
        raise ValueError(
            f"handoff has {n_atoms} atoms but composition implies {n_mol} monomers "
            "(atom count not divisible by monomer count)"
        )
    per = int(n_atoms) // int(n_mol)
    atoms_per_list = [per] * n_mol
    residue_labels = [res for res, cnt in comp for _ in range(int(cnt))]
    return atoms_per_list, residue_labels, composition_summary


def cluster_geometry_from_handoff(
    handoff: MdHandoffState,
    *,
    composition: str | None = None,
    n_molecules: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int]]:
    """Positions/Z and per-monomer layout from handoff (no Packmol)."""
    z = np.asarray(handoff.atomic_numbers, dtype=np.int32)
    r0 = np.asarray(handoff.positions, dtype=np.float64)
    if r0.size > 0 and np.allclose(r0, 0.0):
        import warnings
        warnings.warn(
            f"cluster_geometry_from_handoff: handoff positions are all zero "
            f"({len(r0)} atoms). This restart was likely saved with invalid "
            "coordinates. Check the source handoff file.",
            stacklevel=2,
        )
    if composition:
        atoms_per_list, residue_labels, composition_summary = (
            cluster_layout_from_composition_string(composition, n_atoms=len(r0))
        )
    else:
        n_mol = int(n_molecules)
        if len(r0) % n_mol != 0:
            raise ValueError(
                f"handoff has {len(r0)} atoms but n_molecules={n_mol} "
                "(atom count not divisible by monomer count)"
            )
        per = len(r0) // n_mol
        atoms_per_list = [per] * n_mol
        residue_labels = ["MEOH"] * n_mol
        composition_summary = {"MEOH": n_mol}
    if z.size != len(r0) or not int(np.asarray(z).sum()):
        if not composition:
            raise ValueError(
                "handoff has no atomic numbers; set --composition when continuing from .res"
            )
        parts: list[tuple[str, int]] = []
        for part in str(composition).split(","):
            token = part.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"invalid composition token {token!r}; use RES:COUNT")
            res, cnt = token.split(":", 1)
            parts.append((res.strip().upper(), int(cnt)))
        from mmml.cli.run.md_pbc_suite.ase import _build_cluster_psf_topology_only

        z = _build_cluster_psf_topology_only(
            parts,
            expected_atoms=len(r0),
            atoms_per_list=atoms_per_list,
            residue_labels=residue_labels,
        )
    return z, r0, atoms_per_list, residue_labels, composition_summary


def _validate_handoff_psf_layout(
    *,
    psf_atomic_numbers: np.ndarray,
    psf_atoms_per_list: list[int],
    psf_residue_labels: list[str],
    atomic_numbers: np.ndarray,
    atoms_per_list: list[int],
    residue_labels: list[str],
) -> None:
    n_atoms = int(len(atomic_numbers))
    if int(len(psf_atomic_numbers)) != n_atoms:
        raise RuntimeError(
            f"Handoff PSF atom count ({len(psf_atomic_numbers)}) does not match "
            f"handoff geometry ({n_atoms})"
        )
    if list(psf_atoms_per_list) != list(atoms_per_list):
        raise RuntimeError(
            "Handoff per-monomer atom counts do not match composition-derived PSF layout"
        )
    if list(psf_residue_labels) != list(residue_labels):
        raise RuntimeError(
            "Handoff residue label order does not match composition-derived PSF layout"
        )
    if not np.array_equal(
        np.asarray(psf_atomic_numbers, dtype=int),
        np.asarray(atomic_numbers, dtype=int),
    ):
        raise RuntimeError(
            "Handoff atomic numbers do not match composition-derived PSF (Z mismatch)"
        )


def _live_psf_atom_count() -> int:
    try:
        import pycharmm.psf as psf

        return int(len(np.asarray(psf.get_charges(), dtype=float)))
    except Exception:
        return 0


def _live_psf_matches_handoff(n_atoms: int) -> bool:
    if int(n_atoms) <= 0:
        return False
    try:
        import pycharmm.psf as psf

        charges = np.asarray(psf.get_charges(), dtype=float)
        return (
            charges.shape[0] == int(n_atoms)
            and np.all(np.isfinite(charges))
        )
    except Exception:
        return False


def ensure_psf_for_handoff_cluster(
    *,
    composition: list[tuple[str, int]],
    atomic_numbers: np.ndarray,
    atoms_per_list: list[int],
    residue_labels: list[str],
    positions: np.ndarray | None = None,
    quiet: bool = False,
) -> None:
    """Ensure CHARMM PSF exists for handoff continuation without destroying a live PSF."""
    n_atoms = int(len(atomic_numbers))
    if _live_psf_matches_handoff(n_atoms):
        if positions is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

            sync_charmm_positions(np.asarray(positions, dtype=np.float64))
        if not quiet:
            print(
                f"Reusing live CHARMM PSF for handoff continuation ({n_atoms} atoms)",
                flush=True,
            )
        return

    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_psf_topology_only

    psf_z = _build_cluster_psf_topology_only(
        composition,
        expected_atoms=n_atoms,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
    )
    _validate_handoff_psf_layout(
        psf_atomic_numbers=psf_z,
        psf_atoms_per_list=atoms_per_list,
        psf_residue_labels=residue_labels,
        atomic_numbers=atomic_numbers,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
    )
    if positions is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(np.asarray(positions, dtype=np.float64))
    if not quiet:
        print(
            f"Built CHARMM PSF from composition for handoff continuation ({n_atoms} atoms)",
            flush=True,
        )


def apply_handoff_to_atoms(atoms: Any, handoff: MdHandoffState) -> None:
    atoms.set_positions(handoff.positions)
    if handoff.atomic_numbers is not None and handoff.atomic_numbers.any():
        atoms.set_atomic_numbers(handoff.atomic_numbers)
    if handoff.cell is not None:
        atoms.set_cell(handoff.cell)
        atoms.set_pbc(True)
    elif not handoff.pbc:
        atoms.set_pbc(False)
    if handoff.velocities is not None:
        atoms.set_velocities(handoff.velocities)


def apply_handoff_geometry_to_atoms(
    atoms: Any,
    handoff: MdHandoffState,
    *,
    monomer_offsets: np.ndarray,
    sync_charmm: bool = True,
) -> None:
    """Apply handoff coords/velocities/box and wrap monomers into the primary PBC cell."""
    apply_handoff_to_atoms(atoms, handoff)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if bool(np.asarray(atoms.pbc).any()) and atoms.cell is not None:
        from mmml.utils.geometry_checks import wrap_monomers_primary_cell

        pos = wrap_monomers_primary_cell(
            pos, np.asarray(monomer_offsets, dtype=int), atoms.cell.array
        )
        atoms.set_positions(pos)
    if sync_charmm:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(np.asarray(atoms.get_positions(), dtype=np.float64))


def monomer_offsets_uniform(n_atoms: int, n_monomers: int) -> np.ndarray:
    """Cumulative atom indices for equal-sized monomers."""
    if n_monomers < 1:
        raise ValueError(f"n_monomers must be >= 1, got {n_monomers}")
    if n_atoms % n_monomers != 0:
        raise ValueError(
            f"cannot build uniform monomer offsets: {n_atoms} atoms, "
            f"{n_monomers} monomers"
        )
    per = n_atoms // n_monomers
    return np.arange(0, n_atoms + 1, per, dtype=int)


def handoff_needs_charmm_pbc_alignment(handoff: MdHandoffState) -> bool:
    """True for jaxmd (and other non-CHARMM) handoffs that use [0,L) primary-cell wraps."""
    meta = handoff.metadata or {}
    backend = str(meta.get("backend", "")).strip().lower()
    if backend in ("jaxmd", "ase"):
        return True
    source = str(meta.get("source", "")).strip().lower()
    if source in ("npz", "h5", "traj"):
        return backend != "pycharmm"
    return False


def align_handoff_positions_for_charmm_pbc(
    positions: np.ndarray,
    *,
    monomer_offsets: np.ndarray,
    box_side_A: float,
    handoff: MdHandoffState | None = None,
    quiet: bool = False,
) -> np.ndarray:
    """Map jaxmd-style coords to CHARMM ``image byres xcen/ycen/zcen 0`` convention.

    JAX-MD keeps each monomer in the primary cell via floor wraps into ``[0, L)``.
    CHARMM cubic PBC in :func:`prepare_charmm_pbc` centers the primary image on the
    origin (coordinates near ``[-L/2, L/2]``). Without this shift, the first
    ``MKIMAT2`` / dynamics pass can place residues in distant periodic images.
    """
    if handoff is not None and not handoff_needs_charmm_pbc_alignment(handoff):
        return np.asarray(positions, dtype=np.float64)

    from mmml.utils.geometry_checks import wrap_monomers_primary_cell

    L = float(box_side_A)
    if L <= 0.0:
        raise ValueError(f"box_side_A must be > 0, got {L}")
    cell = np.diag([L, L, L])
    pos = wrap_monomers_primary_cell(
        np.asarray(positions, dtype=np.float64),
        np.asarray(monomer_offsets, dtype=int),
        cell,
    )
    aligned = pos - 0.5 * L
    if not quiet:
        print(
            "Handoff PBC: aligned jaxmd primary-cell coords to CHARMM image center "
            f"(L={L:.3f} Å, xcen/ycen/zcen 0)",
            flush=True,
        )
    return aligned


def set_handoff_in(state: MdHandoffState | None) -> None:
    _handoff_in.set(state)


def get_handoff_in() -> MdHandoffState | None:
    return _handoff_in.get()


def set_handoff_out(state: MdHandoffState | None) -> None:
    _handoff_out.set(state)


def get_handoff_out() -> MdHandoffState | None:
    return _handoff_out.get()


def clear_handoff_context() -> None:
    _handoff_in.set(None)
    _handoff_out.set(None)


def detect_handoff_format(path: Path) -> str:
  p = Path(path).expanduser()
  if p.is_dir():
    if (p / "format.txt").is_file() or (p / "run_state.npz").is_file() or (p / "orbax").is_dir():
      return "run_state"
    if (p / "state.npz").is_file():
      return "npz"
    raise ValueError(f"Unrecognized handoff directory: {p}")
  suffix = p.suffix.lower()
  if suffix == ".res":
    return "res"
  if suffix in {".h5", ".hdf5"}:
    return "h5"
  if suffix == ".traj":
    return "traj"
  if suffix == ".npz":
    return "npz"
  raise ValueError(f"Unrecognized handoff file format: {p}")


def _cell_from_scalar(side_a: float | None) -> np.ndarray | None:
  if side_a is None or side_a <= 0:
    return None
  return np.diag([float(side_a), float(side_a), float(side_a)])


def load_handoff_from_res(path: Path, *, atomic_numbers: np.ndarray | None = None) -> MdHandoffState:
  from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    read_restart_coordinates,
    read_restart_last_step,
    read_restart_velocities,
  )
  from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
    parse_cubic_box_side_from_charmm_restart,
  )

  p = Path(path).expanduser().resolve()
  pos = read_restart_coordinates(p)
  if pos is None:
    raise ValueError(f"Could not read coordinates from restart: {p}")
  vel = read_restart_velocities(p)
  box_a = parse_cubic_box_side_from_charmm_restart(p)
  cell = _cell_from_scalar(box_a)
  z = atomic_numbers
  if z is None:
    z = np.zeros(len(pos), dtype=np.int32)
  return MdHandoffState(
    positions=pos,
    atomic_numbers=z,
    velocities=vel,
    cell=cell,
    pbc=cell is not None,
    step=read_restart_last_step(p),
    metadata={"source": "res", "path": str(p)},
  )


def _handoff_positions_from_npz(data: np.lib.npyio.NpzFile, *, frame: int = 0) -> np.ndarray:
  if "positions" in data.files:
    positions = np.asarray(data["positions"], dtype=np.float64)
  elif "R" in data.files:
    R = np.asarray(data["R"], dtype=np.float64)
    if R.ndim == 3:
      positions = np.asarray(R[int(frame)], dtype=np.float64)
    elif R.ndim == 2:
      positions = R
    else:
      raise ValueError(f"R must have shape (N, 3) or (F, N, 3), got {R.shape}")
  else:
    raise KeyError("positions is not a file in the archive")

  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError(f"handoff positions must have shape (N, 3), got {positions.shape}")
  return positions


def _handoff_atomic_numbers_from_npz(
    data: np.lib.npyio.NpzFile,
    *,
    frame: int = 0,
    n_atoms: int | None = None,
) -> np.ndarray:
  raw = None
  if "atomic_numbers" in data.files:
    raw = data["atomic_numbers"]
  elif "Z" in data.files:
    raw = data["Z"]

  if raw is None:
    return np.zeros(0, dtype=np.int32)

  z = np.asarray(raw, dtype=np.int32)
  if z.ndim == 2:
    z = np.asarray(z[int(frame)], dtype=np.int32)
  elif z.ndim != 1:
    raise ValueError(f"atomic_numbers/Z must have shape (N,) or (F, N), got {z.shape}")

  if n_atoms is not None and int(z.shape[0]) != int(n_atoms):
    if "N" in data.files:
      active_n = int(np.asarray(data["N"], dtype=int).reshape(-1)[int(frame)])
      if int(z.shape[0]) >= active_n:
        z = np.asarray(z[:active_n], dtype=np.int32)
    if int(z.shape[0]) != int(n_atoms):
      raise ValueError(
        f"atomic_numbers length {z.shape[0]} does not match positions ({n_atoms} atoms)"
      )
  return z


def load_handoff_from_npz(path: Path, *, frame: int = 0) -> MdHandoffState:
  data = np.load(path, allow_pickle=True)
  meta_raw = data.get("metadata")
  if meta_raw is None and "metadata.json" in data.files:
    meta_raw = data["metadata.json"]
  if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == object:
    meta = dict(meta_raw.item()) if meta_raw.shape == () else {}
  elif isinstance(meta_raw, (str, bytes)):
    meta = json.loads(meta_raw)
  else:
    meta = {}
  cell = data["cell"] if "cell" in data.files else None
  positions = _handoff_positions_from_npz(data, frame=frame)
  if "N" in data.files and positions.shape[0] > 0:
    counts = np.asarray(data["N"], dtype=int).reshape(-1)
    active_n = int(counts[int(frame)])
    if active_n < positions.shape[0]:
      positions = np.asarray(positions[:active_n], dtype=np.float64)
  atomic_numbers = _handoff_atomic_numbers_from_npz(
    data,
    frame=frame,
    n_atoms=int(positions.shape[0]),
  )
  return MdHandoffState(
    positions=positions,
    atomic_numbers=atomic_numbers,
    velocities=data["velocities"] if "velocities" in data.files else None,
    cell=cell,
    pbc=bool(data["pbc"]) if "pbc" in data.files else cell is not None,
    temperature_K=float(data["temperature_K"]) if "temperature_K" in data.files else None,
    pressure_atm=float(data["pressure_atm"]) if "pressure_atm" in data.files else None,
    step=int(data["step"]) if "step" in data.files else None,
    metadata=dict(meta),
  )


def load_handoff_from_h5(path: Path, *, frame: int = -1) -> MdHandoffState:
  from mmml.utils.hdf5_reporter import load_hdf5_trajectory

  data = load_hdf5_trajectory(path, datasets=["positions", "velocities", "box"])
  pos = np.asarray(data["positions"][frame], dtype=np.float64)
  vel = None
  if "velocities" in data:
    vel = np.asarray(data["velocities"][frame], dtype=np.float64)
  cell = None
  if "box" in data:
    cell = np.asarray(data["box"][frame], dtype=np.float64)
  attrs: dict[str, Any] = {}
  import h5py

  with h5py.File(str(path), "r") as f:
    for key in ("atomic_numbers", "temperature_target", "ensemble"):
      if key in f.attrs:
        attrs[key] = f.attrs[key]
  z = np.asarray(attrs.get("atomic_numbers", np.zeros(len(pos), dtype=np.int32)), dtype=np.int32)
  return MdHandoffState(
    positions=pos,
    atomic_numbers=z,
    velocities=vel,
    cell=cell,
    pbc=cell is not None,
    temperature_K=float(attrs["temperature_target"]) if "temperature_target" in attrs else None,
    metadata={"source": "h5", "path": str(path), "frame": int(frame)},
  )


def load_handoff_from_traj(path: Path, *, frame: int = -1) -> MdHandoffState:
  from ase.io import read as ase_read

  atoms = ase_read(str(path), index=frame)
  vel = atoms.get_velocities()
  cell = atoms.get_cell().array if atoms.pbc.any() else None
  return MdHandoffState(
    positions=np.asarray(atoms.get_positions(), dtype=np.float64),
    atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
    velocities=np.asarray(vel, dtype=np.float64) if vel is not None else None,
    cell=np.asarray(cell, dtype=np.float64) if cell is not None else None,
    pbc=bool(atoms.pbc.any()),
    metadata={"source": "traj", "path": str(path), "frame": int(frame)},
  )


def load_run_state(path: Path) -> MdHandoffState:
  from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
    load_run_state_tree,
  )

  tree = load_run_state_tree(path)
  meta = tree.get("metadata", {})
  if not isinstance(meta, dict):
    meta = {}
  return MdHandoffState(
    positions=tree["positions"],
    atomic_numbers=tree["atomic_numbers"],
    velocities=tree.get("velocities"),
    cell=None,
    pbc=False,
    temperature_K=float(meta["temperature"]) if meta.get("temperature") is not None else None,
    step=None,
    metadata={**meta, "source": "run_state", "path": str(path)},
  )


def load_handoff(
  path: Path,
  *,
  frame: int = -1,
  atomic_numbers: np.ndarray | None = None,
) -> MdHandoffState:
  fmt = detect_handoff_format(path)
  if fmt == "res":
    return load_handoff_from_res(path, atomic_numbers=atomic_numbers)
  if fmt == "npz":
    return load_handoff_from_npz(path)
  if fmt == "h5":
    return load_handoff_from_h5(path, frame=frame)
  if fmt == "traj":
    return load_handoff_from_traj(path, frame=frame)
  if fmt == "run_state":
    return load_run_state(path)
  raise ValueError(f"Unsupported handoff format: {fmt}")


_RESTART_STAGE_PREFIXES: tuple[str, ...] = (
    "prod_",
    "equi_",
    "nve_",
    "heat_",
    "charmm_mm_heat_",
    "mini_",
)


def find_latest_charmm_restart_in_dir(out_dir: Path) -> Path | None:
    """Return the best staged-workflow ``.res`` restart under a job output directory."""
    root = Path(out_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    overlap_backups: list[Path] = []
    for pattern in ("*.res", "pretreat/*.res", "handoff/*.res"):
        for path in root.glob(pattern):
            name = path.name.lower()
            if name.startswith("continue_seed"):
                continue
            if "overlap" in name:
                overlap_backups.append(path)
            else:
                candidates.append(path)
    handoff_res = root / "handoff" / "final.res"
    if handoff_res.is_file() and handoff_res not in candidates:
        candidates.append(handoff_res)
    if not candidates and not overlap_backups:
        return None

    def _sort_key(path: Path, *, overlap_backup: bool) -> tuple[int, int, float]:
        name = path.name.lower()
        if overlap_backup:
            stage_rank = 0
        else:
            stage_rank = -1
            for idx, prefix in enumerate(_RESTART_STAGE_PREFIXES):
                if prefix in name:
                    stage_rank = len(_RESTART_STAGE_PREFIXES) - idx
                    break
        seg = -1
        match = re.search(r"\.(\d+)\.res$", name)
        if match:
            seg = int(match.group(1))
        return (stage_rank, seg, path.stat().st_mtime)

    if candidates:
        return max(candidates, key=lambda p: _sort_key(p, overlap_backup=False))
    return max(overlap_backups, key=lambda p: _sort_key(p, overlap_backup=True))


def load_dependency_handoff(
    dep_dir: Path,
    *,
    quiet: bool = False,
    fallback_box_side_A: float | None = None,
) -> MdHandoffState | None:
    """Load handoff from a prior campaign job (``state.npz``, ``final.res``, or staged ``.res``)."""
    root = Path(dep_dir).expanduser().resolve()
    npz = root / "handoff" / "state.npz"
    if npz.is_file():
        handoff = load_handoff(npz)
        enriched = enrich_handoff_from_restart_files(
            handoff,
            root,
            fallback_box_side_A=fallback_box_side_A,
        )
        if enriched is not handoff and not quiet:
            print(
                "Enriched handoff from restart files "
                f"(cell={'yes' if enriched.cell is not None else 'no'}, "
                f"vel={'yes' if enriched.velocities is not None else 'no'}).",
                flush=True,
            )
        return enriched
    for res_path in (root / "handoff" / "final.res", find_latest_charmm_restart_in_dir(root)):
        if res_path is None or not res_path.is_file():
            continue
        handoff = load_handoff_from_res(res_path)
        if not quiet:
            print(
                f"Continuing from dependency restart {res_path} "
                f"({len(handoff.positions)} atoms)",
                flush=True,
            )
        return handoff
    return None


def enrich_handoff_from_restart_files(
    handoff: MdHandoffState,
    dep_dir: Path,
    *,
    fallback_box_side_A: float | None = None,
) -> MdHandoffState:
    """Fill missing cell/velocities on a loaded handoff from staged ``.res`` files."""
    need_cell = handoff.cell is None or not handoff.pbc
    need_vel = handoff.velocities is None
    if not need_cell and not need_vel:
        return handoff

    from dataclasses import replace

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_velocities,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        parse_cubic_box_side_from_charmm_restart,
    )

    root = Path(dep_dir).expanduser().resolve()
    candidates: list[Path] = []
    for key in ("path", "restart_path"):
        raw = handoff.metadata.get(key)
        if raw:
            resolved = _resolve_existing_file_path(raw)
            candidates.append(resolved if resolved is not None else Path(str(raw)))
    final_res = root / "handoff" / "final.res"
    if final_res.is_file():
        candidates.append(final_res)
    latest = find_latest_charmm_restart_in_dir(root)
    if latest is not None:
        candidates.append(latest)

    cell = handoff.cell
    pbc = handoff.pbc
    velocities = handoff.velocities
    meta = dict(handoff.metadata)
    box_side_source = meta.get("box_side_source")
    velocities_source = meta.get("velocities_source")

    for res_path in candidates:
        if not res_path.is_file():
            continue
        if need_cell and cell is None:
            box_a = parse_cubic_box_side_from_charmm_restart(res_path)
            new_cell = _cell_from_scalar(box_a)
            if new_cell is not None:
                cell = new_cell
                pbc = True
                box_side_source = "restart_enrich"
                meta["restart_path"] = str(res_path.resolve())
        if need_vel and velocities is None:
            vel = read_restart_velocities(res_path)
            if vel is not None and len(vel) == len(handoff.positions):
                velocities = vel
                velocities_source = "restart_enrich"
                meta["restart_path"] = str(res_path.resolve())
        if cell is not None and velocities is not None:
            break

    if need_cell and cell is None and fallback_box_side_A is not None and float(fallback_box_side_A) > 0:
        cell = _cell_from_scalar(float(fallback_box_side_A))
        pbc = True
        box_side_source = "yaml_fallback"

    if box_side_source is not None:
        meta["box_side_source"] = box_side_source
    if velocities_source is not None:
        meta["velocities_source"] = velocities_source

    unchanged = (
        handoff.cell is cell
        or (handoff.cell is not None and cell is not None and np.array_equal(handoff.cell, cell))
    ) and handoff.velocities is velocities and handoff.pbc == pbc
    if unchanged:
        return handoff

    return replace(
        handoff,
        cell=cell,
        pbc=pbc,
        velocities=velocities,
        metadata=meta,
    )


def handoff_to_npz_dict(handoff: MdHandoffState) -> dict[str, Any]:
  out: dict[str, Any] = {
    "positions": handoff.positions,
    "atomic_numbers": handoff.atomic_numbers,
    "pbc": np.bool_(handoff.pbc),
    "metadata": json.dumps(handoff.metadata),
  }
  if handoff.velocities is not None:
    out["velocities"] = handoff.velocities
  if handoff.cell is not None:
    out["cell"] = handoff.cell
  if handoff.temperature_K is not None:
    out["temperature_K"] = np.float64(handoff.temperature_K)
  if handoff.pressure_atm is not None:
    out["pressure_atm"] = np.float64(handoff.pressure_atm)
  if handoff.step is not None:
    out["step"] = np.int64(handoff.step)
  return out


def save_handoff_npz(handoff: MdHandoffState, path: Path) -> Path:
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **handoff_to_npz_dict(handoff))
  return path


def _write_synthetic_charmm_restart(
    handoff: "MdHandoffState",
    path: Path,
) -> Path:
    """Write a valid minimal CHARMM restart from handoff data without ``write restart``.

    CHARMM ``write restart`` only writes non-zero ``!X, Y, Z`` when dynamics
    have been run in the current session (it serialises the dynamics buffer,
    not the MAIN coordinate set).  When continuing into a fresh equi/prod stage
    with no prior dynamics in this Python process, ``write restart`` produces an
    empty coordinate section.  This function bypasses that by writing the
    restart text directly from the handoff positions/velocities/cell.
    """
    pos = np.asarray(handoff.positions, dtype=float)
    n_atoms = int(pos.shape[0])
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"handoff positions must be (N,3), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("handoff positions must be finite")

    def _fmt(v: float) -> str:
        return f"{float(v):.15E}".replace("E", "D")

    def _coord_lines(arr: np.ndarray) -> list[str]:
        flat = arr.reshape(-1)
        lines: list[str] = []
        for i in range(0, len(flat), 3):
            lines.append(" " + " ".join(_fmt(flat[j]) for j in range(i, min(i + 3, len(flat)))))
        return lines

# REST 48 1 CUBI 2 !NTITLE followed by title <<--- note format

#     # CHARMM restart header (minimal safe values)
    lines: list[str] = [
        "REST  SYNTHETIC-HANDOFF      0",
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,SEED,FIRSTT,FINALT,TBATH,TOL,IHTFRQ,IUNSAV",
        f"  {n_atoms:8d}         0         0         1         0         0"
        "         0  0.000000000000000D+00  3.000000000000000D+02"
        "  3.000000000000000D+02  1.000000000000000D-10         0        -1",
    ]
    lines.append(" !X, Y, Z")
    lines.extend(_coord_lines(pos))

    vel = handoff.velocities
    if vel is not None:
        vel = np.asarray(vel, dtype=float)
        if vel.shape == pos.shape and np.all(np.isfinite(vel)):
            lines.append(" !VELOCITIES")
            lines.extend(_coord_lines(vel))

    cell = handoff.cell
    if cell is not None and handoff.pbc:
        cell_arr = np.asarray(cell, dtype=float)
        if cell_arr.shape == (3, 3):
            # Cubic: use diagonal element
            side = float(np.mean(np.diag(cell_arr)))
        else:
            side = float(cell_arr.flat[0])
        if side > 0.0:
            lines.append(" !CRYSTAL PARAMETERS")
            # 6 angles + 6 lengths: orthorhombic cubic
            lines.append(
                f" {_fmt(side)} {_fmt(0.0)} {_fmt(0.0)}"
            )
            lines.append(
                f" {_fmt(0.0)} {_fmt(side)} {_fmt(0.0)}"
            )
            lines.append(
                f" {_fmt(0.0)} {_fmt(0.0)} {_fmt(side)}"
            )

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return path


def _format_fortran_float(value: float) -> str:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"non-finite restart value: {v}")
    return f"{v:.15E}".replace("E", "D")


def _is_overlap_scratch_restart(path: Path) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        is_overlap_scratch_restart_path,
    )

    return is_overlap_scratch_restart_path(path)


def _overlap_scratch_restart_backups(paths: dict[str, Path]) -> list[Path]:
    """Newest overlap chunk scratch restarts near staged workflow outputs (fallback only)."""
    seen: set[str] = set()
    found: list[Path] = []
    parents = {
        str(Path(p).expanduser().resolve().parent)
        for p in paths.values()
        if p is not None
    }
    for parent_s in parents:
        parent = Path(parent_s)
        if not parent.is_dir():
            continue
        for path in parent.glob("*.overlap_?.res"):
            if not _is_overlap_scratch_restart(path):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path.resolve())
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _is_usable_restart_template(path: Path, expected_natom: int | None = None) -> bool:
    """Validate restart coordinates for handoff templating (content, not filename)."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
        read_restart_natom,
        restart_has_nonfinite_coordinates,
        _restart_coordinate_values,
    )
    import sys

    if path.name.lower().startswith("continue_seed"):
        return False
    if restart_has_nonfinite_coordinates(path):
        print(f"DEBUG restart_template: {path} contains non-finite coordinates or parse failed.", file=sys.stderr, flush=True)
        return False
    natom = read_restart_natom(path)
    if natom is None:
        print(f"DEBUG restart_template: {path} NATOM could not be parsed.", file=sys.stderr, flush=True)
        return False
    if expected_natom is not None and int(natom) != int(expected_natom):
        print(f"DEBUG restart_template: {path} NATOM mismatch: parsed {natom}, expected {expected_natom}", file=sys.stderr, flush=True)
        return False
    coords = read_restart_coordinates(path)
    if coords is None:
        flat_len = len(_restart_coordinate_values(path))
        print(f"DEBUG restart_template: {path} read_restart_coordinates returned None. NATOM is {natom}, parsed {flat_len} coords (expected {3 * natom})", file=sys.stderr, flush=True)
        return False
    if int(coords.shape[0]) != int(natom):
        print(f"DEBUG restart_template: {path} coords shape mismatch: shape {coords.shape}, NATOM {natom}", file=sys.stderr, flush=True)
        return False
    return True


def _write_handoff_restart_via_charmm(
    handoff: MdHandoffState,
    path: Path,
    *,
    template_res: Path,
) -> Path:
    """Load a Fortran-valid template in CHARMM, apply handoff state, native ``write restart``."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    template = Path(template_res).expanduser().resolve()
    if not _is_usable_restart_template(template, expected_natom=len(handoff.positions)):
        raise ValueError(f"restart template unusable for handoff: {template.name}")

    restore_charmm_state_from_restart(template)
    sync_charmm_positions(handoff.positions)
    if handoff.cell is not None:
        from mmml.cli.run.md_stage_summary import cubic_box_side_from_cell
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import prepare_charmm_pbc

        side = cubic_box_side_from_cell(handoff.cell)
        if side is not None and float(side) > 0.0:
            prepare_charmm_pbc(float(side))

    if not rewrite_dynamics_restart_validated(path):
        raise ValueError(f"CHARMM restart write failed validation for {path.name}")
    return path


def _format_coord_lines(flat: np.ndarray, *, per_line: int = 3) -> list[str]:
  tokens = [_format_fortran_float(float(x)) for x in flat.reshape(-1)]
  lines: list[str] = []
  for i in range(0, len(tokens), per_line):
    lines.append(" " + " ".join(tokens[i : i + per_line]))
  return lines


def _patch_handoff_into_restart_template(
    handoff: MdHandoffState,
    template: Path,
    path: Path,
) -> None:
    """Patch handoff coords/velocities/crystal parameters into a CHARMM restart template."""
    text = template.read_text(errors="ignore")
    coord_lines = _format_coord_lines(handoff.positions)
    coord_block = " !X, Y, Z\n" + "\n".join(coord_lines) + "\n"
    if " !X, Y, Z" in text:
        text = re.sub(
            r" !X, Y, Z.*?(?=\n !|\Z)",
            coord_block.rstrip(),
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        text = text.rstrip() + "\n" + coord_block
    if handoff.velocities is not None:
        vel_lines = _format_coord_lines(handoff.velocities)
        vel_block = " !VELOCITIES\n" + "\n".join(vel_lines) + "\n"
        if " !VELOCITIES" in text:
            text = re.sub(
                r" !VELOCITIES.*?(?=\n !|\Z)",
                vel_block.rstrip(),
                text,
                count=1,
                flags=re.DOTALL,
            )
        else:
            text = text.rstrip() + "\n" + vel_block

    if handoff.cell is not None and handoff.pbc:
        cell_arr = np.asarray(handoff.cell, dtype=float)
        if cell_arr.shape == (3, 3):
            # Cubic: use diagonal element
            side = float(np.mean(np.diag(cell_arr)))
        else:
            side = float(cell_arr.flat[0])
        if side > 0.0:
            def _fmt(v: float) -> str:
                return f"{float(v):.15E}".replace("E", "D")
            crystal_block = " !CRYSTAL PARAMETERS\n" + "\n".join([
                f" {_fmt(side)} {_fmt(0.0)} {_fmt(0.0)}",
                f" {_fmt(0.0)} {_fmt(side)} {_fmt(0.0)}",
                f" {_fmt(0.0)} {_fmt(0.0)} {_fmt(side)}"
            ]) + "\n"
            if " !CRYSTAL PARAMETERS" in text:
                text = re.sub(
                    r" !CRYSTAL PARAMETERS.*?(?=\n !|\Z)",
                    crystal_block.rstrip(),
                    text,
                    count=1,
                    flags=re.DOTALL,
                )
            else:
                if " !X, Y, Z" in text:
                    text = text.replace(" !X, Y, Z", crystal_block + " !X, Y, Z", 1)
                else:
                    text = text.rstrip() + "\n" + crystal_block
    else:
        if " !CRYSTAL PARAMETERS" in text:
            text = re.sub(
                r"\n\s*!CRYSTAL PARAMETERS.*?(?=\n !|\Z)",
                "",
                text,
                count=1,
                flags=re.DOTALL,
            )
    path.write_text(text, encoding="ascii", errors="ignore")


def _find_usable_fallback_template(failed_template: Path, expected_natom: int) -> Path | None:
    """Search for a usable fallback restart template in the vicinity of failed_template."""
    try:
        search_dirs = [failed_template.parent]
        if failed_template.parent.name == "handoff":
            search_dirs.append(failed_template.parent.parent)

        curr = failed_template.parent
        for _ in range(3):
            curr = curr.parent
            if curr == curr.parent:
                break
            search_dirs.append(curr)

        candidates: list[tuple[int, Path]] = []
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for res_file in sdir.rglob("*.res"):
                if res_file.name.lower().startswith("continue_seed"):
                    continue
                try:
                    if not _is_usable_restart_template(res_file, expected_natom=expected_natom):
                        continue
                    name = res_file.name.lower()
                    if "overlap" not in name:
                        candidates.append((0, res_file))
                    else:
                        candidates.append((1, res_file))
                except Exception:
                    continue
        if candidates:
            candidates.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
            return candidates[0][1]
    except Exception:
        pass
    return None


def save_handoff_to_res(
  handoff: MdHandoffState,
  path: Path,
  *,
  template_res: Path | None = None,
) -> Path:
  """Write CHARMM ``.res`` from handoff (template patch or in-memory CHARMM)."""
  from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
      read_restart_coordinates,
  )

  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)

  if template_res is not None:
      template = Path(template_res).expanduser().resolve()
      if not template.is_file():
          raise FileNotFoundError(f"Restart template not found: {template}")

      charmm_loaded = False
      try:
          import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

          charmm_loaded = True
      except Exception:
          charmm_loaded = False

      if not _is_usable_restart_template(template, expected_natom=len(handoff.positions)):
          fallback = _find_usable_fallback_template(template, expected_natom=len(handoff.positions))
          if fallback is not None:
              print(
                  f"WARNING: template {template} is unusable, but found usable fallback: {fallback}",
                  flush=True,
              )
              template = fallback
          elif charmm_loaded:
              template = None
          else:
              raise ValueError(
                  f"restart template unusable for handoff: {template.name} "
                  "(non-finite coordinates or NATOM/coord mismatch)"
              )

      if template is not None:
          if charmm_loaded:
              try:
                  return _write_handoff_restart_via_charmm(
                      handoff,
                      path,
                      template_res=template,
                  )
              except Exception as e:
                  import sys
                  print(
                      f"WARNING: _write_handoff_restart_via_charmm failed: {e}. "
                      f"Falling back to offline restart text patching.",
                      file=sys.stderr,
                      flush=True,
                  )
          _patch_handoff_into_restart_template(handoff, template, path)
          if read_restart_coordinates(path) is None:
              raise ValueError(
                  f"patched restart {path.name} has no finite Cartesian coordinates "
                  f"(template {template.name})"
              )
          return path

  try:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
      rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(handoff.positions)
    if rewrite_dynamics_restart_validated(path):
        # Verify that the written restart contains finite coordinates
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import read_restart_coordinates
        if read_restart_coordinates(path) is None:
            raise ValueError(f"In-memory restart {path.name} has no finite coordinates after rewrite validation")
        return path
  except Exception:
    pass

  try:
      _write_synthetic_charmm_restart(handoff, path)
      from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import read_restart_coordinates
      if read_restart_coordinates(path) is not None:
          import sys
          print(f"save_handoff_to_res: fell back to synthetic restart {path.name}", file=sys.stderr, flush=True)
          return path
  except Exception:
      pass

  raise ValueError(
      "save_handoff_to_res requires template_res when CHARMM is not loaded "
      "or in-memory restart write failed validation"
  )


def save_handoff(
  handoff: MdHandoffState,
  out_dir: Path,
  *,
  template_res: Path | None = None,
  write_res: bool = True,
) -> dict[str, Path]:
  out_dir = Path(out_dir).expanduser().resolve()
  handoff_dir = out_dir / "handoff"
  handoff_dir.mkdir(parents=True, exist_ok=True)
  if template_res is None:
      template_res = find_latest_charmm_restart_in_dir(out_dir)
  paths: dict[str, Path] = {}
  paths["npz"] = save_handoff_npz(handoff, handoff_dir / "state.npz")
  if write_res:
    paths["res"] = save_handoff_to_res(
      handoff,
      handoff_dir / "final.res",
      template_res=template_res,
    )
  manifest = {k: str(v) for k, v in paths.items()}
  (handoff_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
  )
  return paths


def handoff_is_valid(path: Path) -> bool:
  p = Path(path).expanduser()
  if p.is_dir():
    return (p / "handoff" / "state.npz").is_file() or (p / "handoff" / "manifest.json").is_file()
  try:
    load_handoff(p)
    return True
  except (OSError, ValueError, KeyError):
    return False


def state_from_dict(data: dict[str, Any]) -> MdHandoffState:
  return MdHandoffState(**data)


def state_to_dict(handoff: MdHandoffState) -> dict[str, Any]:
  d = asdict(handoff)
  d["positions"] = handoff.positions.tolist()
  d["atomic_numbers"] = handoff.atomic_numbers.tolist()
  if handoff.velocities is not None:
    d["velocities"] = handoff.velocities.tolist()
  if handoff.cell is not None:
    d["cell"] = handoff.cell.tolist()
  return d


def handoff_skip_pre_min(
    handoff: MdHandoffState | None,
    *,
    handoff_pre_minimize: bool = False,
) -> bool:
    """True when continuing from handoff and pre-min was not explicitly requested."""
    return handoff is not None and not bool(handoff_pre_minimize)


def resolve_jaxmd_minimize_steps_for_handoff(
    *,
    skip_pre_min: bool,
    free_space: bool,
    jaxmd_minimize_steps: int,
    jaxmd_pbc_minimize_steps: int,
) -> tuple[int, int]:
    """JAX-MD runner FIRE step counts after handoff policy.

    Handoff continuations skip vacuum/COM and ASE/CHARMM pre-min, but still run
    PBC-aware FIRE so coordinates relax on the MMML surface in the periodic cell.
    """
    if not skip_pre_min:
        return int(jaxmd_minimize_steps), int(jaxmd_pbc_minimize_steps)
    pbc_steps = 0 if free_space else int(jaxmd_pbc_minimize_steps)
    return 0, pbc_steps


def resolve_handoff_box(
    handoff: MdHandoffState | None,
    *,
    yaml_box_size: float | None,
    free_space: bool,
    auto_box_from_geometry: float | None = None,
    mismatch_tol_A: float = 0.01,
    require_cell: bool = False,
) -> tuple[float | None, str, list[str]]:
    """Choose periodic box side (Å) and record source + warnings."""
    warnings: list[str] = []
    if free_space:
        return None, "free_space", warnings

    handoff_side: float | None = None
    if handoff is not None and handoff.cell is not None:
        from mmml.cli.run.md_stage_summary import cubic_box_side_from_cell

        side = cubic_box_side_from_cell(handoff.cell)
        if side is not None and float(side) > 0:
            handoff_side = float(side)

    yaml_side = float(yaml_box_size) if yaml_box_size is not None else None

    if handoff_side is not None:
        if yaml_side is not None and abs(yaml_side - handoff_side) > float(mismatch_tol_A):
            warnings.append(
                f"Handoff cell side {handoff_side:.4f} Å overrides campaign --box-size "
                f"{yaml_side:.4f} Å (difference {abs(yaml_side - handoff_side):.4f} Å)."
            )
        return handoff_side, "handoff_cell", warnings

    if handoff is not None:
        if yaml_side is not None:
            warnings.append(
                f"Handoff has no periodic cell; using campaign --box-size {yaml_side:.4f} Å."
            )
            return yaml_side, "yaml_fallback", warnings
        if require_cell:
            raise ValueError(
                "Handoff continuation requires a periodic cell in the handoff state "
                "(missing cell in state.npz / restart CRYSTal). "
                "Re-run the predecessor stage with PBC equilibration or set --handoff-require-cell false."
            )
        if auto_box_from_geometry is not None:
            warnings.append(
                "Handoff has no periodic cell; inferring box from initial geometry."
            )
            return float(auto_box_from_geometry), "auto_geometry", warnings

    if yaml_side is not None:
        return yaml_side, "yaml", warnings
    if auto_box_from_geometry is not None:
        return float(auto_box_from_geometry), "auto_geometry", warnings
    return None, "unset", warnings


def resolve_handoff_velocity_policy(
    handoff: MdHandoffState | None,
    *,
    continue_velocities: bool = True,
    pre_min_ran: bool = False,
) -> tuple[str, bool]:
    """Return (policy_label, use_handoff_velocities)."""
    if handoff is None or handoff.velocities is None:
        return "maxwell_boltzmann", False
    if pre_min_ran:
        return "rethermalize_after_pre_min", False
    if continue_velocities:
        return "continue", True
    return "rethermalize", False


def summarize_handoff_policy(
    handoff: MdHandoffState | None,
    *,
    skip_pre_min: bool,
    handoff_pre_minimize: bool,
    continue_velocities: bool,
    velocity_policy: str,
    use_handoff_velocities: bool,
    box_side_A: float | None,
    box_source: str,
    box_warnings: list[str] | None = None,
    ml_switch_width: float | None = None,
    mm_switch_on: float | None = None,
    mm_switch_width: float | None = None,
    initial_energy_eV: float | None = None,
    initial_fmax_eVA: float | None = None,
    quality_gate_enabled: bool = False,
    quality_gate_triggered: bool = False,
) -> dict[str, Any]:
    """Structured handoff policy record for logs and ``handoff_policy.json``."""
    meta = dict(handoff.metadata) if handoff is not None else {}
    source_path = str(meta.get("path", "")) if meta.get("path") else None
    if source_path is None and meta.get("restart_path"):
        source_path = str(meta.get("restart_path"))
    handoff_format = None
    if source_path:
        try:
            handoff_format = detect_handoff_format(Path(source_path))
        except ValueError:
            handoff_format = "unknown"
    return {
        "handoff_active": handoff is not None,
        "source_path": source_path,
        "handoff_format": handoff_format,
        "n_atoms": int(len(handoff.positions)) if handoff is not None else None,
        "prior_backend": meta.get("backend"),
        "prior_stages": meta.get("stages"),
        "box_side_source": meta.get("box_side_source"),
        "velocities_source": meta.get("velocities_source"),
        "restart_path": meta.get("restart_path"),
        "box_side_A": box_side_A,
        "box_source": box_source,
        "box_warnings": list(box_warnings or []),
        "velocities_in_handoff": bool(
            handoff is not None and handoff.velocities is not None
        ),
        "continue_velocities": bool(continue_velocities),
        "velocity_policy": velocity_policy,
        "use_handoff_velocities": bool(use_handoff_velocities),
        "skip_pre_min": bool(skip_pre_min),
        "handoff_pre_minimize": bool(handoff_pre_minimize),
        "quality_gate_enabled": bool(quality_gate_enabled),
        "quality_gate_triggered": bool(quality_gate_triggered),
        "cutoffs": {
            "ml_switch_width": ml_switch_width,
            "mm_switch_on": mm_switch_on,
            "mm_switch_width": mm_switch_width,
        },
        "initial_mmml_energy_eV": initial_energy_eV,
        "initial_mmml_fmax_eVA": initial_fmax_eVA,
    }


def write_handoff_policy_json(summary: dict[str, Any], path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def print_handoff_policy_panel(summary: dict[str, Any], *, quiet: bool = False) -> None:
    if quiet:
        return
    from mmml.utils.rich_report import emit_table

    rows: list[tuple[str, Any]] = [
        ("Handoff active", summary.get("handoff_active")),
    ]
    if summary.get("source_path"):
        rows.append(("Source", summary.get("source_path")))
    if summary.get("prior_backend"):
        rows.append(("Prior backend", summary.get("prior_backend")))
    if summary.get("box_side_source"):
        rows.append(("Box side source (write)", summary.get("box_side_source")))
    rows.extend(
        [
            ("Box side (Å)", summary.get("box_side_A")),
            ("Box source (JAX-MD)", summary.get("box_source")),
            ("Velocities in handoff", summary.get("velocities_in_handoff")),
        ]
    )
    if summary.get("velocities_source"):
        rows.append(("Velocities source (write)", summary.get("velocities_source")))
    rows.append(("Velocity policy", summary.get("velocity_policy")))
    rows.append(("Skip pre-min", summary.get("skip_pre_min")))
    cutoffs = summary.get("cutoffs") or {}
    if cutoffs:
        rows.append(
            (
                "Cutoffs (ml/mm_on/mm_w)",
                f"{cutoffs.get('ml_switch_width')} / {cutoffs.get('mm_switch_on')} / "
                f"{cutoffs.get('mm_switch_width')}",
            )
        )
    if summary.get("initial_mmml_fmax_eVA") is not None:
        rows.append(
            ("Initial MMML |F|max (eV/Å)", f"{summary.get('initial_mmml_fmax_eVA'):.4f}")
        )
    for w in summary.get("box_warnings") or []:
        rows.append(("Warning", w))
    emit_table("Handoff policy", rows, border_style="blue", quiet=quiet)
