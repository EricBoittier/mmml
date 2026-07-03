"""Tri-alanine water box: JAX full-system MM vs PyCHARMM (no MLpot)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    charmm_cmap_is_active,
    compare_bonded_to_charmm,
    compare_mm_system_to_charmm,
    compare_nonbonded_to_charmm,
    run_charmm_bonded_ener_force,
    set_charmm_positions,
    setup_bonded_only_charmm,
    setup_nonbonded_only_charmm,
    summarize_mm_system_charmm_delta,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    load_bonded_system_from_psf,
    load_nonbonded_system_from_charmm,
    mm_system_energy_and_forces,
    nonbonded_energy_and_forces,
)
from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
    build_trialanine_water_box_in_charmm,
    have_trialanine_cgenff,
    have_trialanine_cmap_prm,
)
from tests.conftest import bonded_block_hangs_under_mpi_mpirun, can_import_pycharmm

pytestmark = [
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
    pytest.mark.skipif(
        not have_trialanine_cgenff(),
        reason="bundled CGENFF TRIA (TRIALANINE) RTF not available",
    ),
    pytest.mark.skipif(
        not have_trialanine_cmap_prm(),
        reason="bundled TRIA backbone CMAP PRM not available",
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
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    if not ensure_pycharmm_loaded():
        pytest.skip("PyCHARMM not available (libcharmm / deferred import)")
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
    include_cmap = charmm_cmap_is_active()

    bonded = load_bonded_system_from_psf(
        box.psf_path,
        positions,
        prm_file=box.cgenff_prm,
        extra_prm_files=box.cmap_extra_prm_files,
    )
    components, forces = bonded_energy_and_forces(
        jnp.asarray(positions),
        bonded.topology,
        bonded.bonded,
        urey_k=bonded.urey_k,
        urey_r0=bonded.urey_r0,
        energy_unit="kcal/mol",
        include_cmap=include_cmap,
    )
    compare_bonded_to_charmm(
        components,
        np.asarray(forces),
        energy_rtol=2e-4,
        force_rtol=5e-3,
        include_cmap=include_cmap,
    )


def test_trialanine_water_nonbonded_matches_pycharmm(trialanine_water_box) -> None:
    if bonded_block_hangs_under_mpi_mpirun():
        pytest.skip("selective COEFF BLOCK hangs on MPI-linked libcharmm under mpirun")
    box = trialanine_water_box
    positions = _perturb_positions(box.positions, seed=29)
    set_charmm_positions(positions)

    setup_nonbonded_only_charmm()
    run_charmm_bonded_ener_force(silent=True)

    nbond_data = load_nonbonded_system_from_charmm(
        box.psf_path,
        box.cgenff_prm,
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
    run_charmm_bonded_ener_force(silent=True)
    include_cmap = charmm_cmap_is_active()

    bonded = load_bonded_system_from_psf(
        box.psf_path,
        positions,
        prm_file=box.cgenff_prm,
        extra_prm_files=box.cmap_extra_prm_files,
    )
    nbond_data = load_nonbonded_system_from_charmm(
        box.psf_path,
        box.cgenff_prm,
    )
    result = mm_system_energy_and_forces(
        positions,
        bonded,
        nbond_data,
        box.cell,
        _nbond_settings_from_box(box),
        include_cmap=include_cmap,
    )
    ignore_charmm_bonded = ("urey", "ub")
    print(
        f"\nTRIA MM parity ({len(nbond_data.excluded_pairs)} excluded pairs): "
        f"{summarize_mm_system_charmm_delta(result, ignore_charmm_bonded_terms=ignore_charmm_bonded)}\n",
        flush=True,
    )
    compare_mm_system_to_charmm(
        result,
        ignore_charmm_bonded_terms=ignore_charmm_bonded,
        energy_atol=0.7,
    )
