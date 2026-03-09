from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import ase
from ase import io as ase_io
from ase.data import atomic_masses as ase_data_masses
from dscribe.descriptors import MBTR
from MDAnalysis.analysis.distances import dist
import argparse
import random
from tqdm import tqdm
import os

from ase import Atoms

def _rect_overlap(a, b, pad=0.0):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax+aw+pad <= bx or bx+bw+pad <= ax or
                ay+ah+pad <= by or by+bh+pad <= ay)

def _resolve_pair(a, b):
    # minimal translation to separate axis-aligned rects
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx1 = (bx + bw) - ax   # push a right
    dx2 = (ax + aw) - bx   # push a left
    dy1 = (by + bh) - ay   # push a up
    dy2 = (ay + ah) - by   # push a down
    # Choose the smallest magnitude separation on x or y
    moves = np.array([[ dx1, 0.0],
                      [-dx2, 0.0],
                      [ 0.0, dy1],
                      [ 0.0,-dy2]])
    mags = np.linalg.norm(moves, axis=1)
    mv = moves[np.argmin(mags)]
    # Split the move equally between a and b (push apart)
    return mv*0.5, -mv*0.5

def adjust_inset_positions(ax, centers, box_wh=(0.0152, 0.0152),
                           pad=0.001, max_iter=500, damping=0.9):
    """
    centers: (N,2) array of target (x,y) in data coords where each inset is centered
    box_wh: (w,h) size in data coords
    Returns: list of rects [x, y, w, h] in data coords, non-overlapping.
    """
    centers = np.asarray(centers, dtype=float)
    w, h = box_wh
    rects = np.column_stack([centers[:,0]-w/2, centers[:,1]-h/2,
                             np.full(len(centers), w), np.full(len(centers), h)])
    # Axis limits (data coords) for clamping
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for _ in range(max_iter):
        moved = False
        disp = np.zeros((len(rects), 2), dtype=float)

        # Pairwise resolve overlaps
        for i in range(len(rects)):
            for j in range(i+1, len(rects)):
                if _rect_overlap(rects[i], rects[j], pad):
                    di, dj = _resolve_pair(rects[i], rects[j])
                    disp[i] += di
                    disp[j] += dj
                    moved = True

        if not moved:
            break

        # Apply displacements with damping and clamp inside axes
        disp *= damping
        rects[:,0] += disp[:,0]
        rects[:,1] += disp[:,1]

        # Clamp to stay inside axes with a tiny margin
        eps = 1e-6
        rects[:,0] = np.clip(rects[:,0], xmin+eps, xmax - w - eps)
        rects[:,1] = np.clip(rects[:,1], ymin+eps, ymax - h - eps)

    return rects.tolist()




class LazyAtomsLoader:
    def __init__(self, input):
        if isinstance(input, str):
            self.data = np.load(input, mmap_mode='r')
        else:
            self.data = input
        # else:
        #     raise ValueError(f"Invalid input type: {type(input)}")
        self.symbols = self.data['Z']
        self.positions = self.data['R']
        self.n_atoms = self.data['N']
    
    def __len__(self):
        return len(self.symbols)
    
    def __getitem__(self, idx):
        return Atoms(numbers=self.symbols[idx][:self.n_atoms[idx]],
                     positions=self.positions[idx][:self.n_atoms[idx]])


def setup_ase_atoms(atomic_numbers, positions, n_atoms):
    """Create and setup ASE Atoms object with centered positions"""
    Z = [_ for i, _ in enumerate(atomic_numbers) if i < n_atoms]
    R = np.array([_ for i, _ in enumerate(positions) if i < n_atoms])
    atoms = ase.Atoms(Z, R)
    # translate to center of mass
    # atoms.set_positions(R - R.T.mean(axis=1))
    return atoms


def concat_trajectory(files, output_path, selected=None):
    """Concatenate trajectory files.

    Args:
        files (list): List of file paths to concatenate
        selected (list): List of indices to select from each file

    Returns:
        Path: Path to concatenated trajectory
    """
    if selected is None:
        selected = list(range(len(files)))

    traj = []


def get_descriptor(system, species, plot=True):
    MIN = 2.0
    MAX = 10.0
    N = 100
    DECAY = 0.5
    desc = MBTR(
        species=species,
        geometry={"function": "distance"},
        grid={"min": MIN, "max": MAX, "sigma": 0.1, "n": N},
        weighting={"function": "exp", "scale": DECAY, "threshold": 1e-3},
        periodic=False,
        sparse=False,
        normalization="l2",
    )

    # No weighting
    mbtr_output = desc.create(system)

    # chemical symbol
    n_elements = len(desc.species)
    x = np.linspace(0, MAX, N)

    species_list = []
    descriptor_list = []

    # Plot k=2
    if plot:
        fig, ax = plt.subplots()
    for i in range(n_elements):
        for j in range(n_elements):
            if j >= i:
                i_species = desc.species[i]
                j_species = desc.species[j]
                loc = desc.get_location((i_species, j_species))
                species_list.append((i_species, j_species))
                descriptor_list.append(mbtr_output[loc])
                if plot:
                    plt.plot(
                        x,
                        mbtr_output[loc],
                        "o-",
                        label="{}-{}".format(i_species, j_species),
                    )

    if plot:
        ax.set_xlabel("Distance (Angstrom)")
        ax.legend()
        plt.show()

    species_list
    descriptor = np.array(descriptor_list)

    if plot:
        plt.matshow(descriptor, vmin=0, vmax=0.5, cmap="cubehelix_r")
        plt.show()

    return species_list, descriptor, system


def mass_to_atomic_number(mass):
    """Convert atomic mass to atomic number.

    Args:
        mass (float): Atomic mass

    Returns:
        int: Atomic number

    Raises:
        ValueError: If no matching atomic number is found
    """
    idxs = np.where(abs(ase_data_masses - mass) < 0.001)[0]
    if len(idxs) == 0:
        raise ValueError(f"No matching atomic number found for mass {mass}")
    return int(idxs[0])
