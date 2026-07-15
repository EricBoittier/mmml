"""Validate jax_mm_spoof CGenFF bonded vs live PyCHARMM energy components."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    charmm_positions_xyz_array,
    compare_bonded_to_charmm,
    read_pdb_file,
    read_psf_card_file,
    run_charmm_bonded_ener_force,
    set_charmm_positions,
    setup_bonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
    load_monomer_bonded_components_from_psf,
)
from tests.conftest import bonded_block_hangs_under_mpi_mpirun, can_import_pycharmm
from tests.functionality.pycharmmETC._paths import PYCHARMMETC_DIR, workdir_pdb, workdir_psf

pytestmark = [
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
    pytest.mark.skipif(
        bonded_block_hangs_under_mpi_mpirun(),
        reason="bonded-only BLOCK hangs on MPI-linked libcharmm under mpirun",
    ),
]

ACO_PSF = PYCHARMMETC_DIR / "psf" / "aco-1.psf"
ACO_PDB = PYCHARMMETC_DIR / "pdb" / "aco.pdb"


def _perturb_positions(positions: np.ndarray, seed: int = 29) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return positions + rng.normal(scale=0.03, size=positions.shape)


def test_jax_mm_spoof_aco_components_match_pycharmm(pycharmm_workdir) -> None:
    """Spoof PSF bonded components must match CHARMM BOND/ANGL/DIHE/IMPR/UREY."""
    import pycharmm.read as read
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF

    assert ACO_PSF.is_file(), f"missing fixture PSF: {ACO_PSF}"
    assert ACO_PDB.is_file(), f"missing fixture PDB: {ACO_PDB}"

    aco_psf = Path(workdir_psf("aco-1.psf"))
    aco_pdb = Path(workdir_pdb("aco.pdb"))

    with charmm_relaxed_bomlev():
        read.rtf(CGENFF_RTF)
        read.prm(CGENFF_PRM)
        read_psf_card_file(aco_psf)
        read_pdb_file(aco_pdb, resid=True)

    positions = _perturb_positions(charmm_positions_xyz_array(), seed=29)
    n_atoms = int(positions.shape[0])

    setup_bonded_only_charmm()
    set_charmm_positions(positions)
    run_charmm_bonded_ener_force(silent=True)

    components, forces = load_monomer_bonded_components_from_psf(
        aco_psf,
        jnp.asarray(positions),
        atoms_per_monomer=n_atoms,
        energy_unit="kcal/mol",
    )
    compare_bonded_to_charmm(
        components,
        np.asarray(forces),
        energy_rtol=5e-3,
        energy_atol=5e-3,
    )
