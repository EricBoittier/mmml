"""Build a minimal tri-alanine (ACE–ALA×3–CT3) periodic water box in PyCHARMM.

Grid-placed TIP3 waters avoid Packmol so the fixture runs on CHARMM CI nodes.
Protein parameters come from ``CHARMM_HOME/toppar``; TIP3 uses bundled CGENFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs


@dataclass(frozen=True, slots=True)
class TrialanineWaterBox:
    """Tri-alanine + TIP3 waters in a cubic PBC cell."""

    positions: np.ndarray
    psf_path: Path
    box_side_A: float
    protein_rtf: Path
    protein_prm: Path
    n_waters: int
    nbond_cutoffs: PbcNbondCutoffs

    @property
    def cell(self) -> np.ndarray:
        side = float(self.box_side_A)
        return np.diag([side, side, side])


def protein_toppar_paths() -> tuple[Path, Path]:
    """Return protein RTF/PRM under ``CHARMM_HOME/toppar``."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CHARMM_HOME

    base = Path(CHARMM_HOME) / "toppar"
    rtf = base / "top_all36_prot.rtf"
    prm = base / "par_all36m_prot.prm"
    if not prm.is_file():
        prm = base / "par_all36_prot.prm"
    if not rtf.is_file() or not prm.is_file():
        raise FileNotFoundError(
            f"Protein toppar not found under {base}. "
            "Set CHARMM_HOME to a full CHARMM installation."
        )
    return rtf, prm


def have_protein_toppar() -> bool:
    try:
        protein_toppar_paths()
        return True
    except FileNotFoundError:
        return False


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


def build_trialanine_water_box_in_charmm(
    *,
    n_waters: int = 12,
    box_side_A: float = 28.0,
    water_spacing_A: float = 2.85,
    min_peptide_water_dist_A: float = 2.4,
    seed: int = 11,
    workdir: Path | None = None,
) -> TrialanineWaterBox:
    """Construct tri-alanine + waters in CHARMM and return PSF-ordered coordinates."""
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm, reset_block
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    protein_rtf, protein_prm = protein_toppar_paths()
    rng = np.random.default_rng(seed)

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()

    with charmm_relaxed_bomlev():
        read.rtf(str(protein_rtf))
        read.prm(str(protein_prm))

    settings.set_verbosity(0)
    read.sequence_string("ALA ALA ALA")
    generate.new_segment(
        seg_name="TRIA",
        first_patch="ACE",
        last_patch="CT3",
        setup_ic=True,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    ic_prm_fill(replace_all=True)
    ic.build()

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

    # Load CGENFF after the protein segment is built — loading both PRMs before
    # ic.build() triggers DRUDE particle generation and aborts on some builds.
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        _rtf_path_without_drude_autogen,
        read_cgenff_prm,
    )

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        read.rtf(_rtf_path_without_drude_autogen(CGENFF_RTF))
        read_cgenff_prm(bomlev=False)

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
        protein_rtf=protein_rtf,
        protein_prm=protein_prm,
        n_waters=n_waters,
        nbond_cutoffs=nbond_cutoffs,
    )
