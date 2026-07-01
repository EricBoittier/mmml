"""1:1 CGENFF bonded JAX vs PyCHARMM (bonded-only BLOCK, no MLpot)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    charmm_positions_xyz_array,
    compare_bonded_to_charmm,
    read_pdb_file,
    read_psf_card_file,
    run_charmm_bonded_ener_force,
    set_charmm_positions,
    setup_bonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    load_cgenff_bonded_from_psf,
    parse_psf_ext,
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


def _perturb_positions(positions: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return positions + rng.normal(scale=0.03, size=positions.shape)


def _load_psf_and_positions_from_charmm(residue: str) -> tuple[Path, np.ndarray]:
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )

    setupRes.main(residue)
    setupRes.generate_coordinates()
    reset_block()
    reset_block_no_internal()
    reset_block()

    psf_path = Path(f"psf/{residue.lower()}-1.psf")
    write.psf_card(str(psf_path))
    positions = charmm_positions_xyz_array()
    return psf_path, positions


def _compare_psf_bonded_to_charmm(psf_path: Path, positions: np.ndarray, **compare_kw) -> None:
    setup_bonded_only_charmm()

    set_charmm_positions(positions)
    run_charmm_bonded_ener_force(silent=True)

    system = load_cgenff_bonded_from_psf(psf_path, positions)
    components, forces = bonded_energy_and_forces(
        jnp.asarray(positions),
        system.topology,
        system.bonded,
        energy_unit="kcal/mol",
    )
    compare_bonded_to_charmm(components, np.asarray(forces), **compare_kw)


def test_committed_aco_psf_matches_pycharmm_after_charmm_load(pycharmm_workdir) -> None:
    """Regression: committed ACO PSF/PDB vs CHARMM after ``read psf`` + ``read coor pdb``.

    Runs before setupRes-based cases so mpirun smoke sees a clean CHARMM session.
    """
    import pycharmm.read as read
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF

    assert ACO_PSF.is_file(), f"missing fixture PSF: {ACO_PSF}"
    assert ACO_PDB.is_file(), f"missing fixture PDB: {ACO_PDB}"

    aco_psf = Path(workdir_psf("aco-1.psf"))
    aco_pdb = Path(workdir_pdb("aco.pdb"))
    assert aco_psf.is_file(), f"missing workdir PSF seed: {aco_psf}"
    assert aco_pdb.is_file(), f"missing workdir PDB seed: {aco_pdb}"

    with charmm_relaxed_bomlev():
        read.rtf(CGENFF_RTF)
        read.prm(CGENFF_PRM)
        read_psf_card_file(aco_psf)
        read_pdb_file(aco_pdb, resid=True)

    positions = charmm_positions_xyz_array()
    positions = _perturb_positions(positions, seed=23)
    _compare_psf_bonded_to_charmm(
        aco_psf,
        positions,
        # Committed PDB/PSF pair is not re-minimized like setupRes fixtures.
        energy_rtol=5e-3,
        energy_atol=5e-3,
    )


def test_tip3_bonded_energy_forces_match_pycharmm(pycharmm_workdir) -> None:
    psf_path, positions = _load_psf_and_positions_from_charmm("TIP3")
    positions = _perturb_positions(positions, seed=11)
    _compare_psf_bonded_to_charmm(psf_path, positions)


def test_aco_bonded_energy_forces_match_pycharmm(pycharmm_workdir) -> None:
    psf_path, positions = _load_psf_and_positions_from_charmm("ACO")
    positions = _perturb_positions(positions, seed=17)
    _compare_psf_bonded_to_charmm(psf_path, positions)
