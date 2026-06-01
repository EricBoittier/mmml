"""CHARMM PSF-ordered cluster construction for md_pbc_suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
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

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    read_cgenff_toppar(enable_drude=False)

    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    pos_df = coor.get_positions()
    positions = pos_df[["x", "y", "z"]].to_numpy(dtype=float)
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

    if template_pdb is None and residue == "ACO":
        from mmml.paths import default_aco_template_pdb

        aco_tmpl = default_aco_template_pdb()
        if aco_tmpl.is_file():
            template_pdb = aco_tmpl

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
                        f"Template {template_pdb} missing atom '{nm}' (PSF order). "
                        f"Have: {sorted(tmpl.keys())}"
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
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(shifted)
    except Exception:
        pass

    span = np.ptp(shifted, axis=0)
    if float(span[1]) < 0.3 or float(span[2]) < 0.3:
        raise RuntimeError(
            f"Cluster geometry not 3D (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
        )

    z = np.asarray(get_Z_from_psf(), dtype=int)
    return z, shifted


def _default_template_pdb_for_residue(residue: str) -> Path | None:
    """Bundled 3D monomer templates keyed by CGenFF residue name."""
    residue = residue.upper()
    from mmml.paths import default_aco_template_pdb, default_meoh_template_pdb

    if residue == "ACO":
        path = default_aco_template_pdb()
        return path if path.is_file() else None
    if residue == "MEOH":
        path = default_meoh_template_pdb()
        return path if path.is_file() else None
    return None


def _monomer_geometry_is_3d(coords: np.ndarray, *, min_axis_span: float = 0.3) -> bool:
    span = np.max(coords, axis=0) - np.min(coords, axis=0)
    return float(span[1]) >= min_axis_span and float(span[2]) >= min_axis_span


def build_minimized_monomer_for_packmol(
    residue: str,
    *,
    nstep_sd: int = 50,
    nstep_abnr: int = 100,
    tolenr: float = 1e-3,
    tolgrd: float = 1e-3,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build and CHARMM-minimize an isolated monomer before Packmol (MM only, no MLpot)."""
    from mmml.cli.run.md_pbc_suite.ase import _generate_residue_with_make_res_recipe
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmMmMinimizeConfig,
        minimize_charmm_mm_only,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    residue = residue.upper()
    coords, atom_names, z = _generate_residue_with_make_res_recipe(residue)

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    read_cgenff_toppar(enable_drude=False)
    read.sequence_string(residue)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    psf_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    if psf_names != atom_names:
        raise RuntimeError(
            f"Atom order mismatch for {residue}: PSF {psf_names} vs relaxed {atom_names}"
        )
    sync_charmm_positions(coords)

    if verbose and (nstep_sd > 0 or nstep_abnr > 0):
        print(
            f"Packmol monomer {residue}: CHARMM MM minimize (SD={nstep_sd}, ABNR={nstep_abnr})"
        )
    if nstep_sd > 0 or nstep_abnr > 0:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(
                nstep_sd=int(nstep_sd),
                nstep_abnr=int(nstep_abnr),
                nprint=10,
                tolenr=float(tolenr),
                tolgrd=float(tolgrd),
                verbose=verbose,
                show_energy=False,
            )
        )
        coords = get_charmm_positions_array()

    if not _monomer_geometry_is_3d(coords):
        span = np.ptp(coords, axis=0)
        raise RuntimeError(
            f"Monomer {residue} not 3D after minimization "
            f"(spans Å x={span[0]:.2f} y={span[1]:.2f} z={span[2]:.2f})"
        )
    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(z.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"Atom count mismatch for {residue}: PSF {z.shape[0]} vs {len(atom_names)} names"
        )
    return coords, atom_names, z
