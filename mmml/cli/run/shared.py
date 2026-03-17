"""
Shared utilities for ASE and JAX-MD simulation runners.

Contains common helpers: trajectory saving, simulation loop orchestration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np


def _wrap_frame_by_monomer(
    positions: np.ndarray,
    cell: np.ndarray,
    monomer_offsets: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap positions so each monomer stays intact (no atoms in another cell)."""
    from mmml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np

    cell_matrix = np.asarray(cell, dtype=np.float64)
    if cell_matrix.ndim == 1 and cell_matrix.size >= 3:
        cell_matrix = np.diag(cell_matrix.flat[:3])
    elif cell_matrix.ndim == 2 and cell_matrix.shape[0] >= 3 and cell_matrix.shape[1] >= 3:
        cell_matrix = cell_matrix[:3, :3].copy()
    return _wrap_groups_np(
        np.asarray(positions, dtype=np.float64),
        cell_matrix,
        monomer_offsets,
        masses=masses,
    )


def save_trajectory(
    out_positions: np.ndarray,
    atoms: Any,
    filename: str = "nhc_trajectory",
    format: str = "traj",
    boxes: Optional[List[Any]] = None,
    save_energy_forces: bool = True,
    trajectory_class: Any = None,
    dt_ps: Optional[float] = None,
    steps_per_frame: int = 1,
    monomer_offsets: Optional[np.ndarray] = None,
    cell: Optional[float] = None,
    masses: Optional[np.ndarray] = None,
) -> None:
    """Save trajectory in real (Cartesian) space. For NPT, pass boxes to set cell per frame.

    When save_energy_forces=True, recalculates and stores energy and forces for each frame.

    Supports format="dcd" for CHARMM-readable DCD files (pure Python, no extra deps).
    For DCD, dt_ps and steps_per_frame are used in the file header.

    Frames with NaN or Inf positions are skipped (e.g. after numerical instability).

    When monomer_offsets and cell (or boxes) are provided, each frame is wrapped by monomer
    before saving so monomers stay intact (no atoms in another cell).
    """
    out_positions = np.asarray(out_positions).reshape(-1, len(atoms), 3)

    # Wrap by monomer when PBC and monomer info available (keeps molecules intact)
    do_wrap = (
        monomer_offsets is not None
        and (cell is not None or (boxes is not None and len(boxes) > 0))
    )
    if do_wrap:
        m = np.asarray(masses, dtype=np.float64) if masses is not None else None
        for i in range(out_positions.shape[0]):
            if boxes and i < len(boxes):
                box = np.asarray(boxes[i])
            elif boxes:
                box = np.asarray(boxes[-1])
            else:
                box = np.diag([float(cell)] * 3)
            out_positions[i] = _wrap_frame_by_monomer(
                out_positions[i], box, monomer_offsets, masses=m
            )

    # Drop frames with NaN/Inf (e.g. from numerical instability before early stop)
    valid = np.all(np.isfinite(out_positions), axis=(1, 2))
    n_dropped = int(np.sum(~valid))
    if n_dropped > 0:
        out_positions = out_positions[valid]
        if boxes is not None:
            boxes = [b for b, v in zip(boxes, valid) if v]
        print(f"Dropped {n_dropped} frame(s) with NaN/Inf positions.")
    if len(out_positions) == 0:
        print("No valid frames to save; skipping trajectory.")
        return

    if format == "dcd":
        from mmml.utils.dcd_writer import save_trajectory_dcd

        path = f"{filename}.dcd"
        save_trajectory_dcd(
            path,
            out_positions,
            atoms,
            boxes=boxes,
            dt_ps=dt_ps,
            steps_per_frame=steps_per_frame,
        )
        return

    if trajectory_class is None:
        from ase.io import Trajectory
        trajectory_class = Trajectory
    trajectory = trajectory_class(f"{filename}.{format}", "a")
    for i, R in enumerate(out_positions):
        atoms.set_positions(np.asarray(R))
        if boxes is not None and i < len(boxes):
            box = np.asarray(boxes[i])
            if box.ndim == 2:
                atoms.set_cell(box)
            elif box.size >= 3:
                atoms.set_cell(np.diag(np.asarray(box).reshape(3)))
        if save_energy_forces and atoms.calc is not None:
            _ = atoms.get_potential_energy()
            _ = atoms.get_forces()
        trajectory.write(atoms)
    trajectory.close()


def run_sim_loop(
    run_sim: Any,
    sim_key: Any,
    atoms: Any,
    nsim: int = 1,
    skip_minimization: bool = False,
) -> tuple:
    """Run the simulation for the given indices and save the trajectory.

    Uses current atoms positions (after ASE MD if run) as initial positions.

    Returns:
        (out_positions, out_boxes, max_is)
    """
    out_positions = []
    out_boxes = []
    max_is = []
    pos = np.asarray(atoms.get_positions(), dtype=np.float32)
    for i in range(nsim):
        mi, pos, boxes = run_sim(sim_key, R=pos, skip_minimization=skip_minimization)
        out_positions.append(pos)
        out_boxes.append(boxes)
        max_is.append(mi)
    return out_positions, out_boxes, max_is
