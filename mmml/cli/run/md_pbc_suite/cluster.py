"""CHARMM PSF-ordered cluster construction for md_pbc_suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_PRM,
    CGENFF_RTF,
    coor,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings

pyci.read = read
pyci.settings = settings
pyci.psf = psf


def _load_template_pdb_coords(template_pdb: Path) -> dict[str, np.ndarray]:
    """Load atom-name keyed coordinates from a PDB template."""
    coords: dict[str, np.ndarray] = {}
    for line in template_pdb.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords[atom_name] = np.array([x, y, z], dtype=float)
    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in template PDB: {template_pdb}")
    return coords


def _build_psf_ordered_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
    template_pdb: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    residue = residue.upper()
    sequence = " ".join([residue] * n_molecules)

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")

    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    pos_df = coor.get_positions()
    positions = pos_df.to_numpy(dtype=float)
    n_atoms = positions.shape[0]
    if n_atoms % n_molecules != 0:
        raise RuntimeError(
            f"Atom count {n_atoms} not divisible by n_molecules={n_molecules}; "
            "cannot form equal same-residue chunks."
        )
    atoms_per_res = n_atoms // n_molecules

    n_side = int(np.ceil(np.sqrt(n_molecules)))
    shifted = positions.copy()
    atom_names = np.asarray(psf.get_atype())
    if len(atom_names) != n_atoms:
        raise RuntimeError(f"PSF atom-name count mismatch: {len(atom_names)} vs positions {n_atoms}")

    if template_pdb is not None:
        tmpl = _load_template_pdb_coords(template_pdb)
        for i in range(n_molecules):
            start = i * atoms_per_res
            end = (i + 1) * atoms_per_res
            local_names = atom_names[start:end]
            local_coords = []
            for nm in local_names:
                if nm not in tmpl:
                    raise KeyError(
                        f"Template PDB {template_pdb} missing atom name '{nm}' required by PSF order. "
                        f"Available: {sorted(tmpl.keys())}"
                    )
                local_coords.append(tmpl[nm])
            shifted[start:end] = np.asarray(local_coords, dtype=float)

    for i in range(n_molecules):
        start = i * atoms_per_res
        end = (i + 1) * atoms_per_res
        com = shifted[start:end].mean(axis=0)
        shift = np.array([(i % n_side) * spacing, (i // n_side) * spacing, 0.0], dtype=float)
        shifted[start:end] = shifted[start:end] - com + shift

    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    z = np.asarray(get_Z_from_psf(), dtype=int)
    return z, shifted
