"""Build a minimal DCM liquid cluster in PyCHARMM for JAX MIC parity diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs

DCM_LIQUID_PSF = "dcm-liquid.psf"
DCM_LIQUID_CRD = "dcm-liquid.crd"
DCM_LIQUID_NPY = "dcm-liquid.npy"


@dataclass(frozen=True, slots=True)
class DcmLiquidBox:
    """CGENFF DCM monomers in a cubic PBC cell."""

    positions: np.ndarray
    psf_path: Path
    box_side_A: float
    cgenff_prm: Path
    composition: str
    n_monomers: int
    monomer_offsets: np.ndarray
    nbond_cutoffs: PbcNbondCutoffs
    seed: int

    @property
    def cell(self) -> np.ndarray:
        side = float(self.box_side_A)
        return np.diag([side, side, side])


def monomer_offsets_from_atoms_per(atoms_per: list[int] | np.ndarray) -> np.ndarray:
    ap = np.asarray(atoms_per, dtype=np.int32)
    offsets = np.zeros(ap.shape[0] + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(ap)
    return offsets


def dcm_liquid_box_coords_path(workdir: Path | str) -> Path | None:
    out_dir = Path(workdir)
    npy_path = out_dir / DCM_LIQUID_NPY
    if npy_path.is_file():
        return npy_path
    crd_path = out_dir / DCM_LIQUID_CRD
    if crd_path.is_file():
        return crd_path
    return None


def build_dcm_liquid_box_in_charmm(
    *,
    n_monomers: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
    workdir: Path | None = None,
    skip_reset_block: bool = False,
) -> DcmLiquidBox:
    """Construct ``DCM:N`` on a spacing grid in CHARMM and return PSF-ordered coordinates."""
    import pycharmm.coor as coor
    import pycharmm.write as write

    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition
    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so).")
    if int(n_monomers) < 2:
        raise ValueError(f"n_monomers must be >= 2 for inter-monomer VDW diagnostics, got {n_monomers}")

    pycharmm = ipy.pycharmm
    reset_block = ipy.reset_block
    crystal_free_charmm_for_param_append = ipy.crystal_free_charmm_for_param_append

    composition = f"DCM:{int(n_monomers)}"
    n_side = max(int(np.ceil(int(n_monomers) ** (1.0 / 3.0))), 1)
    spacing = float(box_side_A) / float(n_side) * 0.9

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    if not skip_reset_block:
        reset_block()

    _z, _shifted, atoms_per_list, _labels = _build_cluster_from_composition(
        composition=[("DCM", int(n_monomers))],
        spacing=spacing,
        seed=int(seed),
    )
    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    positions = positions - positions.mean(axis=0) + np.array(
        [box_side_A / 2, box_side_A / 2, box_side_A / 2],
        dtype=np.float64,
    )
    coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))

    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cutnb=13.0, cubic_box_side_A=box_side_A)
    mark_cgenff_params_full()

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    psf_path = out_dir / DCM_LIQUID_PSF
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        write.psf_card(psf_path.name)
        write.coor_card(DCM_LIQUID_CRD)
    finally:
        os.chdir(prev_cwd)
    if not psf_path.is_file():
        raise RuntimeError(f"CHARMM did not write PSF to {psf_path}")

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    np.save(out_dir / DCM_LIQUID_NPY, positions)
    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    return DcmLiquidBox(
        positions=positions,
        psf_path=psf_path.resolve(),
        box_side_A=float(box_side_A),
        cgenff_prm=Path(CGENFF_PRM),
        composition=composition,
        n_monomers=int(n_monomers),
        monomer_offsets=offsets,
        nbond_cutoffs=nbond_cutoffs,
        seed=int(seed),
    )


def reload_dcm_liquid_box_in_charmm(
    workdir: Path | str,
    *,
    box_side_A: float = 28.0,
    n_monomers: int = 10,
    seed: int = -1,
) -> DcmLiquidBox:
    """Reload a saved DCM liquid PSF and coordinates into CHARMM."""
    import pycharmm.coor as coor
    import pycharmm.psf as psf

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
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    out_dir = Path(workdir)
    psf_path = out_dir / DCM_LIQUID_PSF
    if not psf_path.is_file():
        raise FileNotFoundError(f"Missing PSF: {psf_path}")
    coords_path = dcm_liquid_box_coords_path(out_dir)
    if coords_path is None:
        raise FileNotFoundError(
            f"Missing coordinates under {out_dir} "
            f"(expected {DCM_LIQUID_CRD} or {DCM_LIQUID_NPY})."
        )

    prepare_charmm_vacuum()
    read_psf_card_file(psf_path)
    if coords_path.suffix.lower() == ".npy":
        set_charmm_positions(np.load(coords_path))
    else:
        apply_crd_file_to_charmm(coords_path)
    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cutnb=13.0, cubic_box_side_A=box_side_A)
    mark_cgenff_params_full()

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    if positions.shape[0] == 0 or float(np.std(positions)) < 1e-6:
        raise RuntimeError(
            f"Coordinates not loaded into CHARMM from {coords_path} "
            f"(natom={positions.shape[0]}, std={float(np.std(positions)):.3g})"
        )
    natom = int(psf.get_natom())
    n_mol = int(n_monomers)
    atoms_per = natom // n_mol if n_mol > 0 else natom
    offsets = monomer_offsets_from_atoms_per([atoms_per] * n_mol)
    if int(offsets[-1]) != natom:
        offsets = monomer_offsets_from_atoms_per([atoms_per] * (n_mol - 1) + [natom - atoms_per * (n_mol - 1)])

    return DcmLiquidBox(
        positions=positions,
        psf_path=psf_path.resolve(),
        box_side_A=float(box_side_A),
        cgenff_prm=Path(CGENFF_PRM),
        composition=f"DCM:{n_monomers}",
        n_monomers=int(n_monomers),
        monomer_offsets=offsets,
        nbond_cutoffs=nbond_cutoffs,
        seed=int(seed),
    )
