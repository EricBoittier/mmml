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
    temperature_K: float | None = None,
    pressure_atm: float | None = None,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MdHandoffState:
    from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
        _charmm_velocities_array,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    positions = get_charmm_positions_array()
    velocities = _charmm_velocities_array()
    cell = None
    pbc = False
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            get_charmm_cubic_box_side_A,
        )

        side = get_charmm_cubic_box_side_A()
        if side is not None and float(side) > 0:
            cell = np.diag([float(side), float(side), float(side)])
            pbc = True
    except Exception:
        pass
    return MdHandoffState(
        positions=positions,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        velocities=velocities,
        cell=cell,
        pbc=pbc,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        step=step,
        metadata=dict(metadata or {}),
    )


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
            cluster_layout_from_composition_string(composition, n_atoms=len(z))
        )
    else:
        n_mol = int(n_molecules)
        if len(z) % n_mol != 0:
            raise ValueError(
                f"handoff has {len(z)} atoms but n_molecules={n_mol} "
                "(atom count not divisible by monomer count)"
            )
        per = len(z) // n_mol
        atoms_per_list = [per] * n_mol
        residue_labels = ["MEOH"] * n_mol
        composition_summary = {"MEOH": n_mol}
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
