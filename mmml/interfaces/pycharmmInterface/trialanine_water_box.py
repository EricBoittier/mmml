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
        read.rtf(str(trialanine_cgenff_rtf_path()), append=True)


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


def synthetic_trialanine_water_atoms_for_docs(
    *,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    water_spacing_A: float = 2.85,
    min_peptide_water_dist_A: float = 2.4,
    seed: int = 11,
) -> Atoms:
    """Illustrative ASE periodic system for MkDocs (no PyCHARMM).

    Regenerate figures with ``uv run python scripts/generate_docs_figures.py``.
    """
    from ase import Atoms

    centre = np.array([box_side_A / 2, box_side_A / 2, box_side_A / 2], dtype=np.float64)
    backbone_offsets = np.array(
        [
            [-4.2, -0.3, -0.8],
            [-3.5, 0.1, 0.2],
            [-2.6, -0.2, -0.5],
            [-1.8, 0.4, 0.6],
            [-0.9, -0.1, -0.4],
            [-0.1, 0.3, 0.5],
            [0.7, -0.2, -0.6],
            [1.5, 0.5, 0.7],
            [2.4, -0.1, -0.3],
            [3.2, 0.2, 0.4],
            [4.0, -0.4, -0.7],
            [4.8, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    peptide_symbols = list("CNCOCNCOCNCO")
    peptide = centre + backbone_offsets

    rng = np.random.default_rng(seed)
    tip3 = _tip3_template()
    tip3_com = tip3.mean(axis=0)
    oxygen_sites = _grid_oxygen_sites(
        n_waters=n_waters,
        box_side_A=box_side_A,
        spacing_A=water_spacing_A,
        margin_A=3.0,
        existing=peptide,
        min_dist_A=min_peptide_water_dist_A,
        rng=rng,
        water_template=tip3,
    )
    water_blocks = [site + (tip3 - tip3_com) for site in oxygen_sites]
    water_symbols = ["O", "H", "H"] * len(oxygen_sites)
    water_positions = np.vstack(water_blocks)

    symbols = peptide_symbols + water_symbols
    positions = np.vstack([peptide, water_positions])
    cell = np.diag([float(box_side_A)] * 3)
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def peptide_only_atoms_from_box(atoms: Atoms, *, n_peptide_atoms: int = 12) -> Atoms:
    """First *n_peptide_atoms* from :func:`synthetic_trialanine_water_atoms_for_docs`."""
    from ase import Atoms

    n = min(int(n_peptide_atoms), len(atoms))
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
    water_spacing_A: float = 2.85,
    min_peptide_water_dist_A: float = 2.4,
    seed: int = 11,
    workdir: Path | None = None,
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
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        crystal_free_charmm_for_param_append,
        pycharmm,
        reset_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    if not have_trialanine_cgenff():
        raise FileNotFoundError(
            f"Missing {trialanine_cgenff_rtf_path()}. "
            "Run: ./scripts/mmml-charmm-mpirun.sh python scripts/export_trialanine_cgenff_rtf.py"
        )

    rng = np.random.default_rng(seed)
    peptide_rtf = trialanine_cgenff_rtf_path()

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
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

    tip3 = _tip3_template()
    tip3_com = tip3.mean(axis=0)
    oxygen_sites = _grid_oxygen_sites(
        n_waters=n_waters,
        box_side_A=box_side_A,
        spacing_A=water_spacing_A,
        margin_A=3.0,
        existing=peptide,
        min_dist_A=min_peptide_water_dist_A,
        rng=rng,
        water_template=tip3,
    )
    water_coords = np.vstack(
        [site + (tip3 - tip3_com) for site in oxygen_sites],
    )

    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    all_pos = np.vstack([peptide, water_coords])
    coor.set_positions(pd.DataFrame(all_pos, columns=["x", "y", "z"]))

    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)

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

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    return TrialanineWaterBox(
        positions=positions,
        psf_path=psf_path,
        box_side_A=float(box_side_A),
        peptide_rtf=peptide_rtf,
        cgenff_prm=Path(CGENFF_PRM),
        n_waters=n_waters,
        nbond_cutoffs=nbond_cutoffs,
    )
