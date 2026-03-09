"""
Notebook snippet: run MD and calculate spectra with the electric field model.

Usage in Jupyter:
    %run notebook_md_spectra.py
    # or copy cells into a notebook

Or call programmatically:
    from notebook_md_spectra import run_md, run_spectra, run_md_then_spectra
"""

import os
import sys
from pathlib import Path

# Ensure tests/EF is on path
_ef_dir = Path(__file__).resolve().parent
if str(_ef_dir) not in sys.path:
    sys.path.insert(0, str(_ef_dir))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import numpy as np
from ase.io.trajectory import Trajectory

from ase_md import get_args as get_md_args, main as run_md_main
from calc_spectra import get_args as get_spectra_args, main as run_spectra_main


def run_md(params="params.json", data="data-full.npz", steps=500, **kwargs):
    """Run MD simulation. Returns output trajectory path."""
    args = get_md_args(
        params=params,
        data=data,
        steps=steps,
        dt=0.5,
        temperature=300,
        thermostat="langevin",
        output="md_trajectory.traj",
        **kwargs,
    )
    run_md_main(args)
    return Path(args.output)


def run_spectra(params="params.json", data="data-full.npz", index=0, **kwargs):
    """Calculate IR (and optionally Raman, VCD) spectra. Returns output dir."""
    args = get_spectra_args(
        params=params,
        data=data,
        index=index,
        raman=False,
        vcd=False,
        output_dir="spectra",
        **kwargs,
    )
    run_spectra_main(args)
    return Path(args.output_dir)


def run_spectra_from_traj(traj_path, frame_index=-1, params="params.json", **kwargs):
    """
    Calculate spectra from an MD trajectory frame.
    frame_index: -1 = last frame, 0 = first frame, etc.
    """
    traj = Trajectory(str(traj_path))
    atoms = traj[frame_index]
    traj.close()

    # Build minimal npz for calc_spectra
    Z = atoms.get_atomic_numbers()
    R = atoms.get_positions()
    Ef = atoms.info.get("electric_field", np.zeros(3))
    tmp_npz = Path(traj_path).with_suffix(".frame.npz")
    np.savez(tmp_npz, Z=Z, R=R[None, ...], Ef=Ef[None, ...])

    args = get_spectra_args(
        params=params,
        data=str(tmp_npz),
        index=0,
        output_dir="spectra",
        **kwargs,
    )
    run_spectra_main(args)
    return Path(args.output_dir)


def run_md_then_spectra(
    params="params.json",
    data="data-full.npz",
    index=0,
    md_steps=500,
    spectra_from="dataset",  # "dataset" or "traj"
):
    """
    Full pipeline: run MD, then calculate spectra.
    spectra_from: "dataset" = use original geometry from data; "traj" = use last MD frame
    """
    # 1. Run MD
    args_md = get_md_args(params=params, data=data, index=index, steps=md_steps)
    run_md_main(args_md)
    traj_path = Path(args_md.output)

    # 2. Run spectra
    if spectra_from == "traj":
        return run_spectra_from_traj(traj_path, frame_index=-1, params=params)
    else:
        return run_spectra(params=params, data=data, index=index)


# =============================================================================
# Example: minimal notebook cells
# =============================================================================
"""
# Cell 1: Setup
%run tests/EF/notebook_md_spectra.py

# Cell 2: Run MD (short)
traj_path = run_md(params="params.json", data="data-full.npz", steps=500)

# Cell 3: Spectra from dataset (equilibrium structure)
run_spectra(params="params.json", data="data-full.npz", index=0)

# Cell 4 (optional): Spectra from last MD frame
run_spectra_from_traj(traj_path, frame_index=-1, params="params.json")

# Or all-in-one:
run_md_then_spectra(params="params.json", data="data-full.npz", md_steps=500, spectra_from="traj")
"""
