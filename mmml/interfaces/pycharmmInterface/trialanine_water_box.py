"""Build a minimal tri-alanine (ACE–ALA×3–CT3) periodic water box in PyCHARMM.

Uses bundled CGENFF residue ``TRIA`` (documented as TRIALANINE) plus grid-placed
TIP3 waters — no Packmol and no protein ``toppar``.

User guide: ``docs/trialanine-water-box.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs

if TYPE_CHECKING:
    from ase import Atoms

TRIA_RESI_NAME = "TRIA"  # CGENFF RESI (≤4 chars); full name TRIALANINE in docs/CGENFF.RES


@dataclass(frozen=True, slots=True)
class TrialanineWaterBox:
    """Tri-alanine + TIP3 waters in a cubic PBC cell."""

    positions: np.ndarray
    psf_path: Path
    box_side_A: float
    peptide_rtf: Path
    cgenff_prm: Path
    cmap_extra_prm_files: tuple[Path, ...]
    n_waters: int
    nbond_cutoffs: PbcNbondCutoffs

    @property
    def cell(self) -> np.ndarray:
        side = float(self.box_side_A)
        return np.diag([side, side, side])


def trialanine_cgenff_rtf_path() -> Path:
    """Supplemental RTF defining ``RESI TRIA`` (ACE–ALA×3–CT3)."""
    from mmml.paths import bundled_file

    return bundled_file("data", "charmm", "top_trialanine_cgenff.rtf")


def have_trialanine_cgenff() -> bool:
    return trialanine_cgenff_rtf_path().is_file()


def trialanine_backbone_cmap_prm_path() -> Path:
    """Bundled CMAP grid for ``RESI TRIA`` backbone (CGENFF type headers)."""
    from mmml.interfaces.pycharmmInterface.cgenff_cmap import (
        trialanine_backbone_cmap_prm_path as _cmap_prm_path,
    )

    return _cmap_prm_path()


def trialanine_cmap_extra_prm_files() -> tuple[Path, ...]:
    """Extra PRM path(s) for CMAP on the bundled TRIA residue."""
    from mmml.interfaces.pycharmmInterface.cgenff_cmap import (
        trialanine_backbone_cmap_extra_prm_files as _extra,
    )

    return _extra()


def have_trialanine_cmap_prm() -> bool:
    return trialanine_backbone_cmap_prm_path().is_file()


def _load_cgenff_with_trialanine() -> None:
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        _rtf_path_without_drude_autogen,
        read_cgenff_prm,
    )

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        read.rtf(_rtf_path_without_drude_autogen(CGENFF_RTF))
        read_cgenff_prm(bomlev=False)
        for cmap_prm in trialanine_cmap_extra_prm_files():
            read.prm(str(cmap_prm), append=True)
        read.rtf(str(trialanine_cgenff_rtf_path()), append=True)


def prepare_charmm_for_trialanine_box_psf(*, skip_reset_block: bool = True) -> None:
    """Clear CHARMM and load CGENFF+TRIA toppar before reading a saved TRIA+water PSF."""
    import pycharmm.settings as settings

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        crystal_free_charmm_for_param_append,
        pycharmm,
        reset_block,
    )

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    if not skip_reset_block:
        reset_block()
    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)


def _tip3_template() -> np.ndarray:
    """TIP3 coordinates (Å) from bundled ``tip3.pdb`` (OH2, H1, H2)."""
    from mmml.paths import bundled_file

    tip3_pdb = bundled_file("data", "charmm", "tip3.pdb")
    lines = tip3_pdb.read_text(encoding="utf-8").splitlines()
    coords: list[list[float]] = []
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if len(coords) != 3:
        raise ValueError(f"Expected 3 TIP3 atoms in {tip3_pdb}, found {len(coords)}")
    return np.asarray(coords, dtype=np.float64)


def _grid_oxygen_sites(
    *,
    n_waters: int,
    box_side_A: float,
    spacing_A: float,
    margin_A: float,
    existing: np.ndarray,
    min_dist_A: float,
    rng: np.random.Generator,
    water_template: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Place water oxygen atoms on a cubic grid, skipping overlaps with ``existing``."""
    sites: list[np.ndarray] = []
    placed_waters: list[np.ndarray] = []
    template = np.asarray(water_template, dtype=np.float64) if water_template is not None else None
    if template is not None:
        template = template - template.mean(axis=0)

    def _too_close(candidate_atoms: np.ndarray, others: np.ndarray) -> bool:
        dists = np.linalg.norm(others[:, None, :] - candidate_atoms[None, :, :], axis=-1)
        return bool(np.any(dists < min_dist_A))

    max_coord = float(box_side_A) - margin_A
    coord = margin_A
    while len(sites) < n_waters and coord < max_coord:
        for y in np.arange(margin_A, max_coord, spacing_A):
            for z in np.arange(margin_A, max_coord, spacing_A):
                oxygen = np.array([coord, float(y), float(z)], dtype=np.float64)
                oxygen += rng.normal(scale=0.05, size=3)
                if np.any(oxygen < margin_A) or np.any(oxygen > max_coord):
                    continue
                water_atoms = (
                    oxygen + template if template is not None else oxygen.reshape(1, 3)
                )
                if _too_close(water_atoms, existing):
                    continue
                if placed_waters and _too_close(
                    water_atoms,
                    np.vstack(placed_waters),
                ):
                    continue
                sites.append(oxygen)
                placed_waters.append(water_atoms)
                if len(sites) >= n_waters:
                    return sites
        coord += spacing_A
    if len(sites) < n_waters:
        raise RuntimeError(
            f"Could only place {len(sites)}/{n_waters} waters in "
            f"L={box_side_A:.1f} Å box (increase box or reduce n_waters)"
        )
    return sites


def n_peptide_atoms_in_trialanine_box(psf_path: Path | str) -> int:
    """Atom count in the ``PEPT`` segment (``RESI TRIA``) from a CHARMM PSF."""
    path = Path(psf_path)
    in_atoms = False
    count = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip().startswith("*"):
                in_atoms = False
                continue
            if "!NATOM" in line:
                in_atoms = True
                continue
            if not in_atoms:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            if parts[1] == "PEPT":
                count += 1
    if count == 0:
        raise ValueError(f"No PEPT segment atoms found in {path}")
    return count


def load_trialanine_water_atoms_for_docs() -> Atoms:
    """Bundled CHARMM-built tri-alanine + TIP3 box for MkDocs figures.

    Refresh with::

        ./scripts/mmml-charmm-mpirun.sh python scripts/export_docs_structure_assets.py
    """
    from ase.io import read

    from mmml.paths import default_trialanine_water_smoke_extxyz

    path = default_trialanine_water_smoke_extxyz()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled trialanine box at {path}. "
            "Run: ./scripts/mmml-charmm-mpirun.sh python scripts/export_docs_structure_assets.py"
        )
    return read(path)


def peptide_atoms_from_trialanine_box(
    atoms: Atoms,
    *,
    n_peptide_atoms: int | None = None,
) -> Atoms:
    """Peptide-only subset (``RESI TRIA``, 42 atoms) from a trialanine water box."""
    from ase import Atoms

    n = int(n_peptide_atoms) if n_peptide_atoms is not None else 42
    n = min(n, len(atoms))
    return Atoms(
        symbols=atoms.get_chemical_symbols()[:n],
        positions=atoms.get_positions()[:n],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def build_trialanine_water_box_in_charmm(
    *,
    n_waters: int = 12,
    box_side_A: float = 28.0,
    water_spacing_A: float = 1.85,
    min_peptide_water_dist_A: float = 1.4,
    seed: int = 42,
    workdir: Path | None = None,
    skip_reset_block: bool = False,
) -> TrialanineWaterBox:
    """Construct CGENFF ``TRIA`` + TIP3 waters in CHARMM and return PSF-ordered coordinates."""
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError(
            "PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so). "
            "Set CHARMM_LIB_DIR or unset MMML_WARMUP_MLPOT_JAX_ONLY=0 for live tests."
        )
    pycharmm = ipy.pycharmm
    reset_block = ipy.reset_block
    crystal_free_charmm_for_param_append = ipy.crystal_free_charmm_for_param_append

    if not have_trialanine_cgenff():
        raise FileNotFoundError(
            f"Missing {trialanine_cgenff_rtf_path()}. "
            "Run: ./scripts/mmml-charmm-mpirun.sh python scripts/export_trialanine_cgenff_rtf.py"
        )

    rng = np.random.default_rng(seed)
    peptide_rtf = trialanine_cgenff_rtf_path()

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    if not skip_reset_block:
        reset_block()

    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)
    read.sequence_string(TRIA_RESI_NAME)
    generate.new_segment(seg_name="PEPT", setup_ic=True)
    ic_prm_fill(replace_all=True)
    ic.build()

    pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    if np.any(np.abs(pos) > 9000.0) or float(np.std(pos)) < 0.05:
        setupRes.generate_coordinates(skip_energy_show=True, validate=True)

    peptide = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float).copy()
    peptide -= peptide.mean(axis=0)
    peptide += np.array([box_side_A / 2, box_side_A / 2, box_side_A / 2])
    coor.set_positions(pd.DataFrame(peptide, columns=["x", "y", "z"]))

    # Use packmol to pack waters
    from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable
    from ase import Atoms
    from ase.io import write as ase_write
    from ase.io import read as ase_read
    import subprocess
    import shutil
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
    
    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pep_z = get_Z_from_psf()
    pep_atoms = Atoms(pep_z, peptide)
    pep_pdb_path = out_dir / "peptide.pdb"
    ase_write(str(pep_pdb_path), pep_atoms)
    
    from mmml.paths import bundled_file
    tip3_pdb_src = bundled_file("data", "charmm", "tip3.pdb")
    shutil.copy(tip3_pdb_src, out_dir / tip3_pdb_src.name)
    
    packmol_inp_path = out_dir / "packmol.inp"
    packmol_out_path = out_dir / "packmol_out.pdb"
    
    margin = 1.5
    h_side = box_side_A - margin
    
    packmol_input = f"""
tolerance 2.0
filetype pdb
output {packmol_out_path.name}
seed {rng.integers(1000000)}

structure {pep_pdb_path.name}
  number 1
  fixed {box_side_A/2:.4f} {box_side_A/2:.4f} {box_side_A/2:.4f} 0.0 0.0 0.0
end structure

structure {tip3_pdb_src.name}
  number {n_waters}
  inside box {margin:.1f} {margin:.1f} {margin:.1f} {h_side:.1f} {h_side:.1f} {h_side:.1f}
end structure
"""
    packmol_inp_path.write_text(packmol_input)
    
    packmol_bin = packmol_executable()
    subprocess.run([packmol_bin, "-i", packmol_inp_path.name], cwd=out_dir, check=True)
    
    packed_atoms = ase_read(str(packmol_out_path))
    water_coords = packed_atoms.get_positions()[len(peptide):]

    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    all_pos = np.vstack([peptide, water_coords])
    coor.set_positions(pd.DataFrame(all_pos, columns=["x", "y", "z"]))

    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)

    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        mark_cgenff_params_full,
    )

    mark_cgenff_params_full()

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    psf_path = out_dir / "trialanine-water.psf"
    import os

    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        write.psf_card(psf_path.name)
    finally:
        os.chdir(prev_cwd)
    if not psf_path.is_file():
        raise RuntimeError(f"CHARMM did not write PSF to {psf_path}")

    crd_path = out_dir / "trialanine-water.crd"
    try:
        os.chdir(out_dir)
        write.coor_card(crd_path.name)
    finally:
        os.chdir(prev_cwd)

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    np.save(out_dir / "trialanine-water.npy", positions)
    return TrialanineWaterBox(
        positions=positions,
        psf_path=psf_path,
        box_side_A=float(box_side_A),
        peptide_rtf=peptide_rtf,
        cgenff_prm=Path(CGENFF_PRM),
        cmap_extra_prm_files=trialanine_cmap_extra_prm_files(),
        n_waters=n_waters,
        nbond_cutoffs=nbond_cutoffs,
    )


def trialanine_water_box_coords_path(workdir: Path | str) -> Path | None:
    """Return saved coordinates under ``workdir`` (``.npy`` preferred, else ``.crd``)."""
    out_dir = Path(workdir)
    npy_path = out_dir / "trialanine-water.npy"
    if npy_path.is_file():
        return npy_path
    crd_path = out_dir / "trialanine-water.crd"
    if crd_path.is_file():
        return crd_path
    return None


def reload_trialanine_water_box_in_charmm(
    workdir: Path | str,
    *,
    box_side_A: float = 28.0,
    n_waters: int = 10,
    seed: int = -1,
) -> TrialanineWaterBox:
    """Reload a saved TRIA+water PSF and coordinates into CHARMM (fast parity re-runs).

    Expects ``trialanine-water.psf`` plus ``trialanine-water.crd`` or
    ``trialanine-water.npy`` from :func:`build_trialanine_water_box_in_charmm`.
    """
    import pycharmm.coor as coor

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_psf_card_file,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        apply_crd_file_to_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    out_dir = Path(workdir)
    psf_path = out_dir / "trialanine-water.psf"
    crd_path = out_dir / "trialanine-water.crd"
    npy_path = out_dir / "trialanine-water.npy"
    if not psf_path.is_file():
        raise FileNotFoundError(f"Missing PSF: {psf_path}")
    coords_path = trialanine_water_box_coords_path(out_dir)
    if coords_path is None:
        raise FileNotFoundError(
            f"Missing coordinates under {out_dir} "
            f"(expected {crd_path.name} or {npy_path.name}). "
            "Re-run diagnose without --no-build once to refresh the workdir."
        )

    prepare_charmm_for_trialanine_box_psf()
    read_psf_card_file(psf_path)
    if coords_path.suffix.lower() == ".npy":
        set_charmm_positions(np.load(coords_path))
    else:
        # PyCHARMM read.coor_card is unreliable after PSF EXT load under mpirun.
        apply_crd_file_to_charmm(coords_path)
    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)
    mark_cgenff_params_full()

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    if positions.shape[0] == 0 or float(np.std(positions)) < 1e-6:
        raise RuntimeError(
            f"Coordinates not loaded into CHARMM from {coords_path} "
            f"(natom={positions.shape[0]}, std={float(np.std(positions)):.3g})"
        )
    return TrialanineWaterBox(
        positions=positions,
        psf_path=psf_path.resolve(),
        box_side_A=float(box_side_A),
        peptide_rtf=trialanine_cgenff_rtf_path(),
        cgenff_prm=Path(CGENFF_PRM),
        cmap_extra_prm_files=trialanine_cmap_extra_prm_files(),
        n_waters=n_waters,
        nbond_cutoffs=nbond_cutoffs,
    )
