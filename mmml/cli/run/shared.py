"""
Shared utilities for ASE and JAX-MD simulation runners.

Contains common helpers: trajectory saving, simulation loop orchestration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np


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
) -> None:
    """Save trajectory in real (Cartesian) space. For NPT, pass boxes to set cell per frame.

    When save_energy_forces=True, recalculates and stores energy and forces for each frame.

    Supports format="dcd" for CHARMM-readable DCD files (pure Python, no extra deps).
    For DCD, dt_ps and steps_per_frame are used in the file header.
    """
    out_positions = np.asarray(out_positions).reshape(-1, len(atoms), 3)

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
