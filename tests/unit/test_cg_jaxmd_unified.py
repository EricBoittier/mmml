"""Tests for the ``examples/cg_jaxmd_unified.py`` thin front-end.

This front-end is a validated, unified-pipeline replacement for the simulation
loop of ``examples/cg_jaxmd.py`` (see the module docstring for scope/limits).
Unlike the legacy script, it does no CHARMM/jax work at import time, so it is
imported directly here (via ``importlib``, since ``examples/`` is not a package)
rather than only regex-checked as source text.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "examples" / "cg_jaxmd_unified.py"
CKPT = REPO / "examples" / "sppoky-epoch-0010_params.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("cg_jaxmd_unified", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cg_unified():
    pytest.importorskip("jax")
    return _load_module()


# --- config validation (fast, no CHARMM) ------------------------------------


def test_check_cg_config_supported_accepts_default(cg_unified):
    cg_unified.check_cg_config_supported({})  # no toggles -> no error


def test_check_cg_config_supported_rejects_phi_psi(cg_unified):
    with pytest.raises(NotImplementedError, match="constrain_phi_psi"):
        cg_unified.check_cg_config_supported({"constrain_phi_psi": True})


def test_run_cg_config_fails_fast_on_unsupported_toggle(cg_unified):
    """The unsupported-toggle check must run before any CHARMM build."""
    with pytest.raises(NotImplementedError, match="constrain_phi_psi"):
        cg_unified.run_cg_config({"constrain_phi_psi": True, "checkpoint": "x"})


def test_run_cg_config_requires_checkpoint(cg_unified):
    with pytest.raises(ValueError, match="checkpoint"):
        cg_unified.run_cg_config({})


# --- term_kwargs_from_cg_config (pure, mocked system) -----------------------


def _fake_system(n_water_groups=1):
    from mmml.md.system import FFParams, MolecularSystem

    n_pep = 6
    n_water = n_water_groups * 3
    n = n_pep + n_water
    ff = FFParams(
        charges=np.zeros(n), epsilon=np.full(n, 0.1), rmin_half=np.full(n, 1.5),
        at_codes=np.arange(n, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    water_indices = [np.arange(n_pep + 3 * g, n_pep + 3 * g + 3) for g in range(n_water_groups)]
    return MolecularSystem(
        R=np.zeros((n, 3)), Z=np.ones(n, int), box=np.eye(3) * 20.0,
        mol_id=np.zeros(n, dtype=np.int32),
        monomer_indices=[np.arange(n_pep), *water_indices],
        water_indices=water_indices,
        ff_params=ff,
    )


def test_term_kwargs_empty_by_default(cg_unified):
    system = _fake_system()
    assert cg_unified.term_kwargs_from_cg_config({}, system) == {}


def test_term_kwargs_peptide_water_ml(cg_unified):
    system = _fake_system(n_water_groups=2)
    kwargs = cg_unified.term_kwargs_from_cg_config(
        {"peptide_water_ml": True, "peptide_water_ml_cutoff_A": 7.5}, system
    )
    assert set(kwargs) == {"ml_pep_water"}
    assert list(kwargs["ml_pep_water"]["core_indices"]) == list(range(6))
    assert len(kwargs["ml_pep_water"]["group_indices"]) == 2
    assert kwargs["ml_pep_water"]["interaction_cutoff_A"] == 7.5


def test_term_kwargs_peptide_water_ml_core_vdw(cg_unified):
    system = _fake_system(n_water_groups=1)
    kwargs = cg_unified.term_kwargs_from_cg_config(
        {"peptide_water_ml": True, "peptide_water_ml_core_vdw": True}, system
    )
    assert set(kwargs) == {"ml_pep_water", "vdw_core"}
    assert kwargs["vdw_core"]["n_core_atoms"] == 6
    assert len(kwargs["vdw_core"]["lj_epsilon"]) == system.n_atoms


def test_term_kwargs_smd(cg_unified):
    system = _fake_system()
    kwargs = cg_unified.term_kwargs_from_cg_config(
        {"smd_enable": True, "smd_atom_i": 0, "smd_atom_j": 5, "smd_d": 4.0}, system
    )
    assert kwargs["smd"] == {
        "atom_i": 0, "atom_j": 5, "k_ev_per_A2": 1.0, "target": 4.0,
    }


# --- end-to-end integration (real CHARMM build + real checkpoint) ----------


def test_end_to_end_fire_phase(cg_unified):
    try:
        import pycharmm  # noqa: F401  (triggers libcharmm load)
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")

    cfg = {
        "checkpoint": str(CKPT),
        "n_waters": 4,
        "box_size": 15.0,
        "seed": 11,
        "dt_fs": 1.0,
        "fire_total_steps": 10,
        "nvt_total_steps": 0,
        "nve_total_steps": 0,
    }
    trajectories = cg_unified.run_cg_config(cfg, phases=("fire",))
    assert set(trajectories) == {"fire"}
    energies = trajectories["fire"].metadata["energies"]
    assert np.all(np.isfinite(energies))
    assert abs(energies[-1]) < 1e4  # sane MD/MM energy scale, not a blow-up


def test_end_to_end_phase_chaining_carries_positions(cg_unified):
    try:
        import pycharmm  # noqa: F401
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")

    cfg = {
        "checkpoint": str(CKPT),
        "n_waters": 4,
        "box_size": 15.0,
        "seed": 12,
        "dt_fs": 1.0,
        "fire_total_steps": 5,
        "nvt_total_steps": 5,
        "nve_total_steps": 0,
    }
    trajectories = cg_unified.run_cg_config(cfg, phases=("fire", "nvt"))
    assert set(trajectories) == {"fire", "nvt"}
    fire_final = trajectories["fire"].metadata["energies"][-1]
    nvt_initial = trajectories["nvt"].metadata["energies"][0]
    # nvt phase 0 must start exactly where fire ended (positions carried forward)
    assert nvt_initial == pytest.approx(fire_final, rel=1e-6)


def test_end_to_end_peptide_water_ml_no_double_counting(cg_unified):
    """Regression: mm_nonbonded must exclude peptide-water pairs when
    ml_pep_water handles them, or energies double-count that interaction."""
    try:
        import pycharmm  # noqa: F401
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")

    import jax.numpy as jnp

    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    cfg = {
        "checkpoint": str(CKPT),
        "n_waters": 4,
        "box_size": 15.0,
        "seed": 13,
        "dt_fs": 1.0,
        "fire_total_steps": 3,
        "peptide_water_ml": True,
        "peptide_water_ml_core_vdw": True,
        "peptide_water_ml_cutoff_A": 8.0,
    }

    cg_unified.check_cg_config_supported(cfg)
    model, params = cg_unified._load_model(CKPT)
    ctx = EnergyContext(model=model, params=params)

    from mmml.md.assemble import build_system
    from mmml.md.lowering import runconfig_from_cg_config

    run_config = runconfig_from_cg_config(cfg, phase="fire")
    system = build_system(run_config.system)
    term_kwargs = cg_unified.term_kwargs_from_cg_config(cfg, system)
    hybrid = build_hybrid_energy(system, run_config.terms, ctx, term_kwargs)
    R = jnp.asarray(system.R)

    # correct: mm_nonbonded excludes peptide-water pairs (ml_pep_water handles them)
    inter = [r for r in hybrid.neighbor_requests if r.kind == "intermolecular"]
    cutoff = max(r.cutoff_A for r in inter)
    kw_correct = make_intermolecular_neighbor_fn(system, cutoff, None, peptide_water_ml=True)(
        np.asarray(system.R), np.asarray(system.box)
    )
    kw_buggy = make_intermolecular_neighbor_fn(system, cutoff, None, peptide_water_ml=False)(
        np.asarray(system.R), np.asarray(system.box)
    )
    # the buggy (unexcluded) pair list must include strictly more pairs, and the
    # excluded set must be exactly the peptide-water pairs (core x all groups)
    n_correct = int(kw_correct["pair_mask"].sum())
    n_buggy = int(kw_buggy["pair_mask"].sum())
    assert n_buggy > n_correct
    core_indices = system.monomer_indices[0]
    n_water_atoms = sum(len(g) for g in system.water_indices)
    assert n_buggy - n_correct == len(core_indices) * n_water_atoms

    efn = hybrid.as_jax_energy_fn()
    e_correct = float(efn(R, **{k: jnp.asarray(v) for k, v in kw_correct.items()}))
    e_buggy = float(efn(R, **{k: jnp.asarray(v) for k, v in kw_buggy.items()}))
    assert np.isfinite(e_correct)
    # double-counting the core-solvent interaction (once via ml_pep_water's
    # dimer term, again via classical mm_nonbonded) must change the energy;
    # if it didn't, the exclusion wiring would be a no-op.
    assert e_correct != pytest.approx(e_buggy, rel=1e-9)
