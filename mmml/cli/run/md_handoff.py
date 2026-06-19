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


def resolve_handoff_restart_template(
    handoff: MdHandoffState,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> Path | None:
    """Best restart template for patching handoff coords/velocities/cell into CHARMM format."""
    explicit = getattr(args, "handoff_template_res", None)
    if explicit:
        path = Path(str(explicit)).expanduser()
        if path.is_file():
            return path.resolve()

    meta = handoff.metadata or {}
    for key in ("restart_path", "path"):
        raw = meta.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file() and path.suffix.lower() == ".res":
            return path.resolve()

    continue_from = getattr(args, "continue_from", None)
    if continue_from:
        cf = Path(str(continue_from)).expanduser()
        if cf.is_file():
            if cf.suffix.lower() == ".res":
                return cf.resolve()
            final_res = cf.parent / "final.res"
            if final_res.is_file():
                return final_res.resolve()

    for cand in (paths.get("heat_res"), paths.get("equi_res"), paths.get("prod_res")):
        if cand is not None and Path(cand).is_file():
            return Path(cand).resolve()
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

    template = resolve_handoff_restart_template(handoff, args, paths)
    if template is not None:
        save_handoff_to_res(payload, seed, template_res=template)
    else:
        try:
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                rewrite_dynamics_restart_from_current_state,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

            sync_charmm_positions(payload.positions)
            rewrite_dynamics_restart_from_current_state(seed)
        except Exception:
            if not quiet:
                print(
                    "Handoff continuation: no restart template; "
                    "PyCHARMM dynamics may cold-start velocities.",
                    flush=True,
                )
            return None

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
    return seed.resolve()


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


def load_handoff_from_npz(path: Path) -> MdHandoffState:
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
  return MdHandoffState(
    positions=data["positions"],
    atomic_numbers=data["atomic_numbers"],
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
    for path in root.glob("*.res"):
        name = path.name.lower()
        if "overlap" in name or name.startswith("continue_seed"):
            continue
        candidates.append(path)
    handoff_res = root / "handoff" / "final.res"
    if handoff_res.is_file():
        candidates.append(handoff_res)
    if not candidates:
        return None

    def _sort_key(path: Path) -> tuple[int, int, float]:
        name = path.name.lower()
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

    return max(candidates, key=_sort_key)


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
            candidates.append(Path(str(raw)))
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


def _format_fortran_float(value: float) -> str:
  return f"{value:.15E}".replace("E", "D")


def _format_coord_lines(flat: np.ndarray, *, per_line: int = 3) -> list[str]:
  tokens = [_format_fortran_float(float(x)) for x in flat.reshape(-1)]
  lines: list[str] = []
  for i in range(0, len(tokens), per_line):
    lines.append(" " + " ".join(tokens[i : i + per_line]))
  return lines


def save_handoff_to_res(
  handoff: MdHandoffState,
  path: Path,
  *,
  template_res: Path | None = None,
) -> Path:
  """Write CHARMM ``.res`` from handoff (template patch or in-memory CHARMM)."""
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)

  try:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
      rewrite_dynamics_restart_from_current_state,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(handoff.positions)
    rewrite_dynamics_restart_from_current_state(path)
    return path
  except Exception:
    pass

  if template_res is None:
    raise ValueError(
      "save_handoff_to_res requires template_res when CHARMM is not loaded"
    )
  template = Path(template_res).expanduser().resolve()
  if not template.is_file():
    raise FileNotFoundError(f"Restart template not found: {template}")
  text = template.read_text(errors="ignore")
  natom = int(handoff.positions.shape[0])
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
  path.write_text(text, encoding="ascii", errors="ignore")
  return path


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
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print(f"Handoff policy: {json.dumps(summary, indent=2)}", flush=True)
        return

    c = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Handoff active", str(summary.get("handoff_active")))
    if summary.get("source_path"):
        table.add_row("Source", str(summary.get("source_path")))
    if summary.get("prior_backend"):
        table.add_row("Prior backend", str(summary.get("prior_backend")))
    if summary.get("box_side_source"):
        table.add_row("Box side source (write)", str(summary.get("box_side_source")))
    table.add_row("Box side (Å)", str(summary.get("box_side_A")))
    table.add_row("Box source (JAX-MD)", str(summary.get("box_source")))
    table.add_row("Velocities in handoff", str(summary.get("velocities_in_handoff")))
    if summary.get("velocities_source"):
        table.add_row("Velocities source (write)", str(summary.get("velocities_source")))
    table.add_row("Velocity policy", str(summary.get("velocity_policy")))
    table.add_row("Skip pre-min", str(summary.get("skip_pre_min")))
    cutoffs = summary.get("cutoffs") or {}
    if cutoffs:
        table.add_row(
            "Cutoffs (ml/mm_on/mm_w)",
            f"{cutoffs.get('ml_switch_width')} / {cutoffs.get('mm_switch_on')} / "
            f"{cutoffs.get('mm_switch_width')}",
        )
    if summary.get("initial_mmml_fmax_eVA") is not None:
        table.add_row("Initial MMML |F|max (eV/Å)", f"{summary.get('initial_mmml_fmax_eVA'):.4f}")
    for w in summary.get("box_warnings") or []:
        table.add_row("Warning", w)
    c.print(Panel(table, title="[bold]Handoff policy[/bold]", border_style="blue"))
