"""Tri-alanine water box: JAX full-system MM vs PyCHARMM (no MLpot)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    compare_bonded_to_charmm,
    compare_mm_system_to_charmm,
    compare_nonbonded_to_charmm,
    run_charmm_bonded_ener_force,
    set_charmm_positions,
    setup_bonded_only_charmm,
    setup_nonbonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    load_bonded_system_from_psf,
    load_nonbonded_system_from_charmm,
    mm_system_energy_and_forces,
    nonbonded_energy_and_forces,
)
from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
    build_trialanine_water_box_in_charmm,
    have_protein_toppar,
)
from tests.conftest import bonded_block_hangs_under_mpi_mpirun, can_import_pycharmm

pytestmark = [
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
    pytest.mark.skipif(
        not have_protein_toppar(),
        reason="CHARMM_HOME/toppar protein files not available",
    ),
]


def _perturb_positions(positions: np.ndarray, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return positions + rng.normal(scale=0.02, size=positions.shape)


def _nbond_settings_from_box(box) -> CharmmNbondSettings:
    cuts = box.nbond_cutoffs
    return CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )


@pytest.fixture(scope="module")
def trialanine_water_box(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("trialanine_water")
    return build_trialanine_water_box_in_charmm(
        n_waters=10,
        box_side_A=28.0,
        seed=11,
        workdir=workdir,
    )


def test_trialanine_water_bonded_matches_pycharmm(trialanine_water_box) -> None:
    if bonded_block_hangs_under_mpi_mpirun():
        pytest.skip("bonded-only BLOCK hangs on MPI-linked libcharmm under mpirun")
    box = trialanine_water_box
    positions = _perturb_positions(box.positions, seed=23)
    set_charmm_positions(positions)

    setup_bonded_only_charmm()
    run_charmm_bonded_ener_force(silent=True)

    bonded = load_bonded_system_from_psf(
        box.psf_path,
        positions,
        prm_file=box.protein_prm,
        extra_prm_files=[CGENFF_PRM],
    )
    components, forces = bonded_energy_and_forces(
        jnp.asarray(positions),
        bonded.topology,
        bonded.bonded,
        energy_unit="kcal/mol",
    )
    compare_bonded_to_charmm(
        components,
        np.asarray(forces),
        energy_rtol=2e-4,
        force_rtol=5e-3,
    )


def test_trialanine_water_nonbonded_matches_pycharmm(trialanine_water_box) -> None:
    box = trialanine_water_box
    positions = _perturb_positions(box.positions, seed=29)
    set_charmm_positions(positions)

    setup_nonbonded_only_charmm()
    run_charmm_bonded_ener_force(silent=True)

    nbond_data = load_nonbonded_system_from_charmm(
        box.psf_path,
        box.protein_prm,
        CGENFF_PRM,
    )
    settings = _nbond_settings_from_box(box)
    components, forces = nonbonded_energy_and_forces(
        positions,
        nbond_data,
        box.cell,
        settings,
    )
    compare_nonbonded_to_charmm(components, np.asarray(forces))


@pytest.mark.parametrize("lr_solver", ["mic"])
def test_trialanine_water_total_mm_matches_pycharmm(
    trialanine_water_box,
    lr_solver: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full MM energy: bonded + truncated MIC Coulomb (``lr_solver=mic``)."""
    monkeypatch.setenv("MMML_LR_SOLVER", lr_solver)

    box = trialanine_water_box
    positions = _perturb_positions(box.positions, seed=31)
    set_charmm_positions(positions)

    apply_charmm_mm_block()
    run_charmm_bonded_ener_force(silent=True)

    bonded = load_bonded_system_from_psf(
        box.psf_path,
        positions,
        prm_file=box.protein_prm,
        extra_prm_files=[CGENFF_PRM],
    )
    nbond_data = load_nonbonded_system_from_charmm(
        box.psf_path,
        box.protein_prm,
        CGENFF_PRM,
    )
    result = mm_system_energy_and_forces(
        positions,
        bonded,
        nbond_data,
        box.cell,
        _nbond_settings_from_box(box),
    )
    compare_mm_system_to_charmm(result)
