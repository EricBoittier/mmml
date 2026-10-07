"""Restart / handoff state must round-trip exactly (positions, velocities, box).

Regression tests for the restart audit of 4 Oct 2026
(metatomic-runs/training/restart_tests):

* a CHARMM leap-frog dynamics restart stores positions in ``!XOLD`` and the
  step displacement in ``!X, Y, Z``; ``--continue-from X.res`` read ``!X`` and
  collapsed the box ("Cluster not 3D");
* CHARMM ``!VX`` velocities are plain Å per AKMA time unit; the handoff read
  them as mass-weighted (a 300 K restart came back at ~5e-7 K) and metal-unit
  JAX-MD/ASE velocities as Å/ps (98x too slow), so continuations fell below
  the cold-handoff floor and silently re-drew Maxwell-Boltzmann velocities;
* the ASE backend always re-drew Maxwell-Boltzmann velocities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import units

from karml.cli.run.md_handoff import (
    CHARMM_AKMA_TIME_PS,
    MdHandoffState,
    handoff_in_charmm_velocity_units,
    handoff_velocities_as_ang_ps,
    handoff_velocities_as_charmm_akma,
    handoff_velocities_as_jaxmd_metal,
    kinetic_temperature_k_from_ang_ps_velocities,
    kinetic_temperature_k_from_jaxmd_metal_velocities,
    load_handoff_from_res,
)

STUB = Path(__file__).resolve().parents[1] / "functionality/mlpot/output/dynamics/nve_stub.res"
K_B_KCAL = 0.0019872041  # CHARMM KBOLTZ


def _section(path: Path, marker: str) -> np.ndarray:
    from karml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        _restart_section_values,
    )

    return np.asarray(_restart_section_values(path, marker), dtype=float)


def test_charmm_timfac_constant():
    # TIMFAC = sqrt(1 Å^2 amu N_A / 1 kcal) in ps
    timfac = np.sqrt(1e-20 * 1.66053906660e-27 * 6.02214076e23 / 4184.0) * 1e12
    assert CHARMM_AKMA_TIME_PS == pytest.approx(timfac, rel=1e-8)


def test_continue_from_res_reads_xold_positions_not_step_displacement():
    h = load_handoff_from_res(STUB)
    xold = _section(STUB, "!XOLD, YOLD, ZOLD")[:60].reshape(20, 3)
    disp = _section(STUB, "!X, Y, Z")[:60].reshape(20, 3)
    np.testing.assert_array_equal(h.positions, xold)
    assert np.ptp(h.positions) > 1.0  # a real geometry, not ~1e-3 Å steps
    assert np.abs(disp).max() < 0.1
    np.testing.assert_array_equal(h.velocities, _section(STUB, "!VX, VY, VZ")[:60].reshape(20, 3))


def test_charmm_restart_velocities_give_the_charmm_temperature():
    """KE from !VX in AKMA (0.5 m v^2, kcal/mol) must equal KE from the converted Å/ps."""
    rng = np.random.default_rng(0)
    z = np.array([6, 1, 1, 17, 17] * 40, dtype=np.int32)
    from ase.data import atomic_masses

    m = atomic_masses[z]
    v_akma = rng.normal(size=(len(z), 3)) * np.sqrt(K_B_KCAL * 300.0 / m)[:, None]
    t_charmm = float(np.sum(m[:, None] * v_akma**2) / (3 * len(z) * K_B_KCAL))
    h = MdHandoffState(np.zeros((len(z), 3)), z, velocities=v_akma, metadata={"backend": "pycharmm"})
    v_ang_ps = handoff_velocities_as_ang_ps(h)
    assert kinetic_temperature_k_from_ang_ps_velocities(v_ang_ps, m) == pytest.approx(t_charmm, rel=1e-9)
    v_metal = handoff_velocities_as_jaxmd_metal(h)
    assert kinetic_temperature_k_from_jaxmd_metal_velocities(v_metal, m) == pytest.approx(t_charmm, rel=1e-4)
    assert 150.0 < t_charmm < 600.0
    # back to CHARMM units is the identity up to round-off
    np.testing.assert_allclose(handoff_velocities_as_charmm_akma(h), v_akma, rtol=1e-14, atol=0)


def test_metal_handoff_velocities_are_not_read_as_ang_ps():
    z = np.array([6, 1, 17], dtype=np.int32)
    v_metal = np.array([[0.01, -0.02, 0.005], [0.05, 0.0, -0.03], [0.002, 0.004, -0.001]])
    h = MdHandoffState(np.zeros((3, 3)), z, velocities=v_metal,
                       metadata={"backend": "jaxmd", "velocity_units": "jaxmd_metal"})
    np.testing.assert_allclose(handoff_velocities_as_ang_ps(h), v_metal * 1000.0 * units.fs)
    # jaxmd -> jaxmd and ase -> ase continuations hand back the same numbers
    np.testing.assert_allclose(handoff_velocities_as_jaxmd_metal(h), v_metal, rtol=1e-14)


def test_metal_handoff_is_converted_before_it_reaches_a_charmm_restart():
    z = np.array([6, 1, 17], dtype=np.int32)
    v_metal = np.array([[0.01, -0.02, 0.005], [0.05, 0.0, -0.03], [0.002, 0.004, -0.001]])
    h = MdHandoffState(np.zeros((3, 3)), z, velocities=v_metal, metadata={"backend": "ase"})
    c = handoff_in_charmm_velocity_units(h)
    np.testing.assert_allclose(c.velocities, v_metal * 1000.0 * units.fs * CHARMM_AKMA_TIME_PS)
    assert c.metadata["velocity_units"] == "akma"
    # idempotent: a CHARMM-unit handoff is left (numerically) unchanged
    np.testing.assert_allclose(handoff_in_charmm_velocity_units(c).velocities, c.velocities, rtol=1e-14)


def test_ase_run_md_continues_handoff_velocities(tmp_path):
    """``run_md(initial_velocities=...)`` must not re-draw Maxwell-Boltzmann velocities."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    from karml.cli.run.md_pbc_suite.ase import run_md

    rng = np.random.default_rng(1)
    pos = np.stack(np.meshgrid(*[np.arange(3) * 3.5] * 3), -1).reshape(-1, 3)
    atoms = Atoms("Ar27", positions=pos, cell=[10.5] * 3, pbc=True)
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.0103, rc=5.0)
    v0 = rng.normal(scale=0.01, size=(27, 3))
    run_md(
        name="t", atoms=atoms, mode="nve", dt_fs=1.0, nsteps=0, log_every=1, traj_every=1,
        traj_chunk_frames=0, out_dir=tmp_path, nvt_temp_K=300.0, nve_temp_K=300.0,
        langevin_friction=0.01, seed=0, monomer_offsets=np.arange(28), min_intermonomer_atom_distance=0.0,
        overlap_rescue_charmm_sd_steps=0, overlap_rescue_charmm_abnr_steps=0, charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3, charmm_nbxmod=5, initial_velocities=v0,
    )
    # ASE stores momenta, so get_velocities() is p/m: equal to round-off
    np.testing.assert_allclose(atoms.get_velocities(), v0, rtol=1e-14, atol=0)


def test_pre_dynamics_seed_uses_xold_positions(tmp_path):
    """``--restart-from <out>/nve.res`` seeds CHARMM before the force gate: must be XOLD."""
    import shutil
    from unittest.mock import patch

    from karml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _seed_charmm_coords_from_dynamics_restart,
    )

    res = tmp_path / "nve.res"
    shutil.copy(STUB, res)
    with patch("karml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions") as sync, patch(
        "karml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates"
    ):
        assert _seed_charmm_coords_from_dynamics_restart(res, quiet=True) is True
    np.testing.assert_array_equal(sync.call_args[0][0], _section(STUB, "!XOLD, YOLD, ZOLD")[:60].reshape(20, 3))


def _route_args(tmp_path, src, *, setup, stages):
    import argparse

    return argparse.Namespace(backend="pycharmm", continue_from=src, restart_from=None,
                              output_dir=tmp_path / "out", md_stage=None, md_stages=stages,
                              setup=setup, n_equi_segments=1, n_prod_segments=1)


def test_continue_from_charmm_dynamics_restart_becomes_in_place_readyn(tmp_path):
    from karml.cli.run.md_system import route_pycharmm_continue_from_dynamics_restart

    args = _route_args(tmp_path, STUB, setup="pbc_nve", stages="nve")
    dst = route_pycharmm_continue_from_dynamics_restart(args)
    assert dst == tmp_path / "out" / "nve.res" and dst.read_bytes() == STUB.read_bytes()
    assert args.restart_from == dst and args.continue_from is None


def test_continue_from_routing_requires_matching_ensemble_and_single_stage(tmp_path):
    from karml.cli.run.md_system import route_pycharmm_continue_from_dynamics_restart

    # plain (non-CPT) restart must not feed a CPT stage: READYN would read no piston
    assert route_pycharmm_continue_from_dynamics_restart(
        _route_args(tmp_path, STUB, setup="pbc_npt", stages="equi")) is None
    assert route_pycharmm_continue_from_dynamics_restart(
        _route_args(tmp_path, STUB, setup="pbc_nve", stages="heat,nve")) is None
    npz = tmp_path / "state.npz"
    npz.write_bytes(b"")
    assert route_pycharmm_continue_from_dynamics_restart(
        _route_args(tmp_path, npz, setup="pbc_nve", stages="nve")) is None


def test_cpt_restart_is_recognised(tmp_path):
    from karml.cli.run.md_system import _charmm_dynamics_restart_kind

    xtl = (" !CRYSTAL PARAMETERS\n"
           " 0.300000000000000D+02 0.000000000000000D+00 0.300000000000000D+02\n"
           " 0.000000000000000D+00 0.000000000000000D+00 0.300000000000000D+02\n"
           "-0.161387886868731D-05 0.000000000000000D+00 0.000000000000000D+00\n\n")
    text = STUB.read_text()
    p = tmp_path / "equi.res"
    p.write_text(text.replace(" !NATOM", xtl + " !NATOM", 1))
    assert _charmm_dynamics_restart_kind(p) == "cpt"
    assert _charmm_dynamics_restart_kind(STUB) == "plain"
