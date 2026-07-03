"""Build a TIP3-only periodic water box in PyCHARMM for JAX MIC parity diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.dcm_liquid_box import (
    monomer_offsets_from_atoms_per,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs

TIP3_LIQUID_PSF = "tip3-liquid.psf"
TIP3_LIQUID_CRD = "tip3-liquid.crd"
TIP3_LIQUID_NPY = "tip3-liquid.npy"
ATOMS_PER_TIP3 = 3


@dataclass(frozen=True, slots=True)
class Tip3LiquidBox:
    """CGENFF TIP3 waters in a cubic PBC cell (no peptide)."""

    positions: np.ndarray
    psf_path: Path
    box_side_A: float
    cgenff_prm: Path
    n_waters: int
    monomer_offsets: np.ndarray
    nbond_cutoffs: PbcNbondCutoffs
    seed: int

    @property
    def composition(self) -> str:
        return f"TIP3:{self.n_waters}"

    @property
    def n_monomers(self) -> int:
        return int(self.n_waters)

    @property
    def cell(self) -> np.ndarray:
        side = float(self.box_side_A)
        return np.diag([side, side, side])


def tip3_liquid_box_coords_path(workdir: Path | str) -> Path | None:
    out_dir = Path(workdir)
    npy_path = out_dir / TIP3_LIQUID_NPY
    if npy_path.is_file():
        return npy_path
    crd_path = out_dir / TIP3_LIQUID_CRD
    if crd_path.is_file():
        return crd_path
    return None


def build_tip3_liquid_box_in_charmm(
    *,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
    workdir: Path | None = None,
    skip_reset_block: bool = False,
) -> Tip3LiquidBox:
    """Grid-place ``n_waters`` TIP3 molecules in a cubic PBC cell (no TRIA peptide)."""
    import pandas as pd
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        _grid_oxygen_sites,
        _load_cgenff_with_trialanine,
        _tip3_template,
    )

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so).")
    if int(n_waters) < 2:
        raise ValueError(f"n_waters must be >= 2 for inter-monomer diagnostics, got {n_waters}")

    pycharmm = ipy.pycharmm
    reset_block = ipy.reset_block
    crystal_free_charmm_for_param_append = ipy.crystal_free_charmm_for_param_append

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    if not skip_reset_block:
        reset_block()

    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)

    rng = np.random.default_rng(seed)
    tip3 = _tip3_template()
    tip3_com = tip3.mean(axis=0)
    oxygen_sites = _grid_oxygen_sites(
        n_waters=n_waters,
        box_side_A=box_side_A,
        spacing_A=2.85,
        margin_A=3.0,
        existing=np.empty((0, 3), dtype=np.float64),
        min_dist_A=2.4,
        rng=rng,
        water_template=tip3,
    )
    water_coords = np.vstack([site + (tip3 - tip3_com) for site in oxygen_sites])

    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    coor.set_positions(pd.DataFrame(water_coords, columns=["x", "y", "z"]))

    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)
    mark_cgenff_params_full()

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    psf_path = out_dir / TIP3_LIQUID_PSF
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        write.psf_card(psf_path.name)
        write.coor_card(TIP3_LIQUID_CRD)
    finally:
        os.chdir(prev_cwd)
    if not psf_path.is_file():
        raise RuntimeError(f"CHARMM did not write PSF to {psf_path}")

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    np.save(out_dir / TIP3_LIQUID_NPY, positions)
    offsets = monomer_offsets_from_atoms_per([ATOMS_PER_TIP3] * int(n_waters))
    return Tip3LiquidBox(
        positions=positions,
        psf_path=psf_path.resolve(),
        box_side_A=float(box_side_A),
        cgenff_prm=Path(CGENFF_PRM),
        n_waters=int(n_waters),
        monomer_offsets=offsets,
        nbond_cutoffs=nbond_cutoffs,
        seed=int(seed),
    )


def reload_tip3_liquid_box_in_charmm(
    workdir: Path | str,
    *,
    box_side_A: float = 28.0,
    n_waters: int = 10,
    seed: int = -1,
) -> Tip3LiquidBox:
    """Reload a saved TIP3-only PSF and coordinates into CHARMM."""
    import pycharmm.coor as coor

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_psf_card_file,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import apply_crd_file_to_charmm
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        prepare_charmm_for_trialanine_box_psf,
    )

    out_dir = Path(workdir)
    psf_path = out_dir / TIP3_LIQUID_PSF
    if not psf_path.is_file():
        raise FileNotFoundError(f"Missing PSF: {psf_path}")
    coords_path = tip3_liquid_box_coords_path(out_dir)
    if coords_path is None:
        raise FileNotFoundError(
            f"Missing coordinates under {out_dir} "
            f"(expected {TIP3_LIQUID_CRD} or {TIP3_LIQUID_NPY})."
        )

    prepare_charmm_for_trialanine_box_psf()
    read_psf_card_file(psf_path)
    if coords_path.suffix.lower() == ".npy":
        set_charmm_positions(np.load(coords_path))
    else:
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
    n_w = int(n_waters)
    offsets = monomer_offsets_from_atoms_per([ATOMS_PER_TIP3] * n_w)
    return Tip3LiquidBox(
        positions=positions,
        psf_path=psf_path.resolve(),
        box_side_A=float(box_side_A),
        cgenff_prm=Path(CGENFF_PRM),
        n_waters=n_w,
        monomer_offsets=offsets,
        nbond_cutoffs=nbond_cutoffs,
        seed=int(seed),
    )
