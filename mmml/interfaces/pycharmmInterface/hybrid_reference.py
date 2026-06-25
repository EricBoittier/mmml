"""Shared reference NPZ I/O and cutoff grid-search objectives for hybrid calculators."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mmml.cli.run.md_handoff import MdHandoffState, load_handoff_from_npz
from mmml.interfaces.pycharmmInterface.cutoffs import (
    CutoffParameters,
    cutoff_search_result_dict,
)


@dataclass
class GeometryNpzPayload:
    """Single-frame geometry (+ optional MM overrides) from an NPZ file."""

    handoff: MdHandoffState
    charges: np.ndarray | None = None
    at_codes: np.ndarray | None = None
    epsilon: np.ndarray | None = None
    sigma: np.ndarray | None = None


@dataclass
class ReferenceTrajectory:
    """Multi-frame reference data for cutoff fitting (QM/QM-like NPZ)."""

    path: Path
    R: np.ndarray
    Z: np.ndarray
    E: np.ndarray | None
    F: np.ndarray | None
    frame_indices: np.ndarray
    com_distances: np.ndarray
    has_E: bool
    has_F: bool
    n_frames: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _optional_npz_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.size == 0:
        return None
    return arr


def _normalize_at_codes(raw: np.ndarray) -> np.ndarray:
    codes = np.asarray(raw, dtype=np.int32).reshape(-1)
    if codes.size == 0:
        raise ValueError("at_codes/iac array is empty")
    if int(codes.min()) >= 1:
        codes = codes - 1
    if int(codes.min()) < 0:
        raise ValueError("at_codes/iac must be 0-based or 1-based CHARMM iac indices")
    return codes


def load_geometry_npz(path: Path, *, frame: int = 0) -> GeometryNpzPayload:
    """Load single-frame geometry NPZ (``positions`` or trajectory ``R`` key)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"geometry NPZ not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        handoff = load_handoff_from_npz(path, frame=frame)
        charges = _optional_npz_array(data, "charges")
        at_raw = _optional_npz_array(data, "at_codes")
        if at_raw is None:
            at_raw = _optional_npz_array(data, "iac")
        at_codes = _normalize_at_codes(at_raw) if at_raw is not None else None
        epsilon = _optional_npz_array(data, "epsilon")
        sigma = _optional_npz_array(data, "sigma")

    n_atoms = int(handoff.positions.shape[0])
    if charges is not None and int(charges.shape[0]) != n_atoms:
        raise ValueError(f"charges length {charges.shape[0]} != {n_atoms} atoms")
    if at_codes is not None and int(at_codes.shape[0]) != n_atoms:
        raise ValueError(f"at_codes length {at_codes.shape[0]} != {n_atoms} atoms")

    return GeometryNpzPayload(
        handoff=handoff,
        charges=charges,
        at_codes=at_codes,
        epsilon=epsilon,
        sigma=sigma,
    )


def compute_com_distances(
    R_all: np.ndarray,
    *,
    n_atoms_monomer: int,
    n_monomers: int = 2,
    center_frames: bool = True,
) -> np.ndarray:
    """COM–COM distance per frame (first two monomers; general clusters use mean pair)."""
    n_per = int(n_atoms_monomer)
    n_mol = int(n_monomers)
    distances: list[float] = []
    for i in range(len(R_all)):
        frame = np.asarray(R_all[i], dtype=np.float64)
        if center_frames:
            frame = frame - frame.mean(axis=0)
        if n_mol < 2:
            distances.append(0.0)
            continue
        coms = []
        off = 0
        for _ in range(n_mol):
            coms.append(frame[off : off + n_per].mean(axis=0))
            off += n_per
        # Use min pairwise COM distance as the sort key (dimers: only one pair).
        pair_dists = []
        for a in range(len(coms)):
            for b in range(a + 1, len(coms)):
                pair_dists.append(float(np.linalg.norm(coms[a] - coms[b])))
        distances.append(min(pair_dists) if pair_dists else 0.0)
    return np.asarray(distances, dtype=np.float64)


def load_reference_trajectory_npz(
    path: Path,
    *,
    z_fallback: np.ndarray,
    n_atoms_monomer: int,
    n_monomers: int,
    max_frames: int = 200,
) -> ReferenceTrajectory:
    """Load trajectory reference NPZ with ``R`` (and optional ``E``, ``F``)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference NPZ not found: {path}")

    with np.load(path, allow_pickle=True) as dataset:
        if "R" not in dataset.files and "positions" in dataset.files:
            raise ValueError(
                f"{path.name} looks like a single-frame handoff NPZ (positions). "
                "Use --evaluate-npz for one geometry, or provide a trajectory NPZ with key 'R'."
            )
        R_all = np.array(dataset["R"], copy=True)
        Z_ds = dataset.get("Z", z_fallback)
        if hasattr(Z_ds, "ndim") and Z_ds.ndim > 1:
            Z_ds = np.array(Z_ds[0]).astype(int)
        else:
            Z_ds = np.asarray(Z_ds).astype(int)
        E_all = dataset.get("E", None)
        F_all = dataset.get("F", None)
        N_arr = dataset.get("N", None)
        meta_raw = dataset.get("metadata")
        if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == object:
            meta = dict(meta_raw.item()) if meta_raw.shape == () else {}
        elif isinstance(meta_raw, (str, bytes)):
            meta = json.loads(meta_raw)
        else:
            meta = {}

    has_E = E_all is not None and np.size(E_all) > 0
    has_F = F_all is not None and np.size(F_all) > 0
    n_frames = int(R_all.shape[0])
    n_eval = (
        min(n_frames, max_frames)
        if max_frames is not None and max_frames > 0
        else n_frames
        if max_frames == -1
        else n_frames
    )

    com_distances = compute_com_distances(
        R_all,
        n_atoms_monomer=n_atoms_monomer,
        n_monomers=n_monomers,
        center_frames=True,
    )

    if N_arr is not None:
        valid_mask = np.asarray(N_arr) == n_atoms_monomer * n_monomers
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            raise RuntimeError("No valid frames found in reference NPZ (N filter).")
    else:
        valid_idx = np.arange(n_frames)

    sorted_valid = valid_idx[np.argsort(com_distances[valid_idx])]
    n_valid = len(sorted_valid)
    stride = max(1, n_valid // max(1, n_eval))
    frame_indices = sorted_valid[::stride][:n_eval]

    return ReferenceTrajectory(
        path=path,
        R=R_all,
        Z=Z_ds,
        E=np.asarray(E_all) if has_E else None,
        F=np.asarray(F_all) if has_F else None,
        frame_indices=frame_indices,
        com_distances=com_distances,
        has_E=has_E,
        has_F=has_F,
        n_frames=n_frames,
        metadata=dict(meta),
    )


def evaluate_hybrid_mse_on_frames(
    atoms: Any,
    *,
    R_all: np.ndarray,
    frame_indices: np.ndarray,
    E_all: np.ndarray | None,
    F_all: np.ndarray | None,
    has_E: bool,
    has_F: bool,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> dict[str, float]:
    """Weighted MSE of hybrid calculator vs reference on selected frames."""
    se_e, se_f, n_e, n_f = 0.0, 0.0, 0, 0
    for i in frame_indices:
        atoms.set_positions(R_all[int(i)])
        pred_E = float(atoms.get_potential_energy())
        pred_F = np.asarray(atoms.get_forces())
        if has_E and E_all is not None:
            se_e += (pred_E - float(E_all[int(i)])) ** 2
            n_e += 1
        if has_F and F_all is not None:
            se_f += float(np.mean((np.asarray(F_all[int(i)]) - pred_F) ** 2))
            n_f += 1
    mse_e = (se_e / max(n_e, 1)) if has_E else 0.0
    mse_f = (se_f / max(n_f, 1)) if has_F else 0.0
    objective = float(energy_weight) * mse_e + float(force_weight) * mse_f
    return {
        "mse_energy": mse_e,
        "mse_forces": mse_f,
        "objective": objective,
        "n_energy_frames": float(n_e),
        "n_force_frames": float(n_f),
    }


def evaluate_cutoff_triple(
    *,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    atoms: Any,
    attach_calculator: Callable[[CutoffParameters], Any],
    R_all: np.ndarray,
    frame_indices: np.ndarray,
    E_all: np.ndarray | None,
    F_all: np.ndarray | None,
    has_E: bool,
    has_F: bool,
    energy_weight: float,
    force_weight: float,
    complementary_handoff: bool = True,
) -> dict[str, float]:
    """Evaluate one cutoff triple; ``attach_calculator`` sets ``atoms.calc``."""
    cutoff = CutoffParameters(
        ml_switch_width=float(ml_switch_width),
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
        complementary_handoff=complementary_handoff,
    )
    atoms.calc = attach_calculator(cutoff)
    metrics = evaluate_hybrid_mse_on_frames(
        atoms,
        R_all=R_all,
        frame_indices=frame_indices,
        E_all=E_all,
        F_all=F_all,
        has_E=has_E,
        has_F=has_F,
        energy_weight=energy_weight,
        force_weight=force_weight,
    )
    return cutoff_search_result_dict(
        ml_switch_width=ml_switch_width,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        mse_energy=metrics["mse_energy"],
        mse_forces=metrics["mse_forces"],
        objective=metrics["objective"],
    )


def run_cutoff_grid_search(
    *,
    ml_grid: list[float],
    mm_on_grid: list[float],
    mm_w_grid: list[float],
    atoms: Any,
    attach_calculator: Callable[[CutoffParameters], Any],
    reference: ReferenceTrajectory,
    energy_weight: float,
    force_weight: float,
    complementary_handoff: bool = True,
    verbose: bool = True,
) -> tuple[list[dict[str, float]], dict[str, float] | None]:
    """Exhaustive grid search; returns (all_results, best_result)."""
    results: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for ml_w, mm_on, mm_w in itertools.product(ml_grid, mm_on_grid, mm_w_grid):
        res = evaluate_cutoff_triple(
            ml_switch_width=ml_w,
            mm_switch_on=mm_on,
            mm_switch_width=mm_w,
            atoms=atoms,
            attach_calculator=attach_calculator,
            R_all=reference.R,
            frame_indices=reference.frame_indices,
            E_all=reference.E,
            F_all=reference.F,
            has_E=reference.has_E,
            has_F=reference.has_F,
            energy_weight=energy_weight,
            force_weight=force_weight,
            complementary_handoff=complementary_handoff,
        )
        results.append(res)
        if verbose:
            print(
                f"ml_w={res['ml_switch_width']:.3f} mm_on={res['mm_switch_on']:.3f} "
                f"mm_w={res['mm_switch_width']:.3f} -> obj={res['objective']:.6e} "
                f"(E={res['mse_energy']:.6e}, F={res['mse_forces']:.6e})",
                flush=True,
            )
        if best is None or res["objective"] < best["objective"]:
            best = res
    return results, best


def apply_npz_charges_to_psf(charges: np.ndarray) -> None:
    import pycharmm.scalar as scalar

    scalar.set_charges(np.asarray(charges, dtype=np.float64))
