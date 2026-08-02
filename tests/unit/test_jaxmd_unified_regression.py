"""Regression coverage for the jaxmd-unified stack (``mmml.md`` pipeline).

These exercise the real ``assemble_and_run`` -> ``JaxmdDriver`` path end to end
on a synthetic TIP3-like water box, with **no CHARMM build and no ML
checkpoint** (the classical ``mm_nonbonded`` term and the pure-jax ``smd`` term
are enough to drive dynamics). The point is to pin behaviour that regressions
would silently break: determinism, energy conservation, ensemble coverage,
dtype handling, trajectory schema, and the driver's input guards.

Physics tests use ``mm_nonbonded`` (a real periodic LJ+Coulomb surface);
``smd`` (a single harmonic bond, jax-only) backstops the always-available path
and is a clean conservative oscillator for the NVE-drift check.

Run just this suite::

    pytest tests/unit/test_jaxmd_unified_regression.py -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md import EnsembleSpec, RunConfig, SystemSpec, assemble_and_run
from mmml.md.results import energy_drift_metrics

pytestmark = pytest.mark.unit

# jax-md is the engine under test; skip the whole module cleanly without it.
pytest.importorskip("jax")
pytest.importorskip("jax_md")


def _run(system, *, term="mm_nonbonded", ensemble="nve", n_steps=20, dt_fs=0.5,
         seed=0, temperature_K=300.0, output_dir=None, extra_params=None, term_kwargs=None,
         record_every=None):
    params = {"seed": seed}
    if extra_params:
        params.update(extra_params)
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),  # unused: system is passed in
        terms=(term,),
        ensemble=EnsembleSpec(
            ensemble=ensemble, dt_fs=dt_fs, n_steps=n_steps,
            temperature_K=temperature_K, params=params,
        ),
        backend="jaxmd",
        output_dir=output_dir,
    )
    driver = None
    if record_every is not None:
        from mmml.md.drivers import JaxmdDriver

        driver = JaxmdDriver(record_every=record_every)
    return assemble_and_run(cfg, system=system, term_kwargs=term_kwargs, driver=driver)


def _smd_kwargs():
    return {"smd": {"atom_i": 0, "atom_j": 3, "k_ev_per_A2": 1.0, "target": 3.0}}


# --- determinism ------------------------------------------------------------


def test_mm_nonbonded_run_is_deterministic(synthetic_water_box):
    """Same system + same seed -> bit-for-bit identical energy trace."""
    a = _run(synthetic_water_box(seed=0), seed=7)
    b = _run(synthetic_water_box(seed=0), seed=7)
    ea = np.asarray(a.metadata["energies"])
    eb = np.asarray(b.metadata["energies"])
    assert ea.shape == eb.shape
    assert np.array_equal(ea, eb)


def test_nvt_seed_changes_trajectory(synthetic_water_box):
    """The thermostat RNG seed must actually perturb the dynamics."""
    system = synthetic_water_box(seed=0)
    e1 = np.asarray(_run(system, ensemble="nvt", seed=1).metadata["energies"])
    e2 = np.asarray(_run(system, ensemble="nvt", seed=2).metadata["energies"])
    assert not np.allclose(e1, e2)


# --- energy conservation ----------------------------------------------------


def test_nve_energy_is_conserved_smd(synthetic_water_box):
    """A pure harmonic (smd) NVE run must conserve *total* energy.

    Conservation lives in potential + kinetic. ``metadata["energies"]`` is the
    potential surface alone, which genuinely swings for an oscillator (here
    ~17 -> ~9 eV over 100 steps) while the total holds to ~1e-4 relative -- so
    asserting on it measured the oscillation, not the integrator. That only
    looked flat while ``dt`` was ~98x too small and the dynamics were nearly
    frozen; the timestep-unit fix exposed it.
    """
    traj = _run(synthetic_water_box(seed=3), term="smd", ensemble="nve",
                n_steps=100, dt_fs=0.25, term_kwargs=_smd_kwargs(),
                record_every=10)
    total = np.asarray(traj.metadata["total_energies"])
    assert np.all(np.isfinite(total))
    assert total.size >= 10, "need a real trace, not two endpoints, to fit a trend"
    drift = energy_drift_metrics(total)
    scale = float(np.abs(total).mean()) + 1e-6
    # Symplectic NVE: systematic trend must be a small fraction of the scale.
    assert abs(drift["energy_trend_total_ev"]) < 0.05 * scale


def test_nve_potential_alone_is_not_conserved_smd(synthetic_water_box):
    """Pin the distinction: the oscillator's potential energy really does swing.

    Guards against someone "fixing" a future failure by reverting the assertion
    above to ``metadata["energies"]``.
    """
    traj = _run(synthetic_water_box(seed=3), term="smd", ensemble="nve",
                n_steps=100, dt_fs=0.25, term_kwargs=_smd_kwargs(),
                record_every=10)
    potential = np.asarray(traj.metadata["energies"])
    kinetic = np.asarray(traj.metadata["kinetic_energies"])
    total = np.asarray(traj.metadata["total_energies"])
    assert np.allclose(total, potential + kinetic)
    # Energy sloshes between the two reservoirs by far more than the 5% budget.
    assert float(np.ptp(potential)) > 0.2 * float(np.abs(total).mean())
    assert float(np.ptp(kinetic)) > 0.2 * float(np.abs(total).mean())


def test_nve_mm_nonbonded_finite_and_bounded(synthetic_water_box):
    """The physical LJ+Coulomb surface stays finite and bounded over NVE."""
    traj = _run(synthetic_water_box(seed=5), ensemble="nve", n_steps=40, dt_fs=0.5)
    energies = np.asarray(traj.metadata["energies"])
    assert np.all(np.isfinite(energies))
    # No blow-up: spread stays comparable to the mean magnitude.
    assert energies.std() < 10.0 * (abs(energies.mean()) + 1.0)


# --- ensemble coverage ------------------------------------------------------


@pytest.mark.parametrize("ensemble", ["min", "nve", "nvt", "npt"])
def test_all_ensembles_run_and_stay_finite(synthetic_water_box, ensemble):
    traj = _run(synthetic_water_box(seed=0), ensemble=ensemble, n_steps=10)
    energies = np.asarray(traj.metadata["energies"])
    assert traj.n_frames >= 1
    assert np.all(np.isfinite(energies))


@pytest.mark.parametrize("float64", [False, True])
def test_npt_runs_in_both_dtypes(synthetic_water_box, float64):
    """Regression guard for the barostat dtype fix: the driver casts kT/pressure
    **and dt** to the run dtype, so NPT runs in float32 *and* float64 under
    JAX_ENABLE_X64 (a Python-float dt promoted the Nose-Hoover chain carry and
    float32 raised ``carry component cs[0] ... float32[] ... float64[]``)."""
    traj = _run(synthetic_water_box(seed=0), ensemble="npt", n_steps=10,
                extra_params={"float64": float64})
    assert np.all(np.isfinite(np.asarray(traj.metadata["energies"])))


@pytest.mark.parametrize("ensemble", ["min", "nve", "nvt", "npt"])
def test_kinetic_and_total_energy_are_recorded(synthetic_water_box, ensemble):
    """Every ensemble records KE, so total energy is available to diagnostics."""
    traj = _run(synthetic_water_box(seed=0), ensemble=ensemble, n_steps=10)
    potential = np.asarray(traj.metadata["energies"])
    kinetic = np.asarray(traj.metadata["kinetic_energies"])
    total = np.asarray(traj.metadata["total_energies"])
    assert kinetic.shape == potential.shape == total.shape
    assert np.all(np.isfinite(kinetic)), f"{ensemble} recorded non-finite KE"
    assert np.allclose(total, potential + kinetic)
    assert np.all(kinetic >= 0.0)


# --- dtype handling ---------------------------------------------------------


@pytest.mark.parametrize("float64", [False, True])
def test_both_dtypes_run_finite(synthetic_water_box, float64):
    traj = _run(synthetic_water_box(seed=0), ensemble="nve", n_steps=10,
                extra_params={"float64": float64})
    assert np.all(np.isfinite(np.asarray(traj.metadata["energies"])))


# --- trajectory schema ------------------------------------------------------


def test_trajectory_schema_and_npz(synthetic_water_box, tmp_path):
    system = synthetic_water_box(seed=0)
    traj = _run(system, ensemble="nve", n_steps=20, output_dir=tmp_path)
    # metadata
    energies = np.asarray(traj.metadata["energies"])
    assert energies.ndim == 1 and energies.shape[0] == traj.n_frames
    assert np.all(np.isfinite(energies))
    # on-disk artifact
    npz_path = tmp_path / "trajectory.npz"
    assert npz_path.exists()
    with np.load(npz_path) as data:
        keys = set(data.files)
        assert "positions" in keys or "R" in keys
        pos_key = "positions" if "positions" in keys else "R"
        pos = data[pos_key]
        assert pos.shape[-1] == 3
        assert pos.shape[-2] == system.n_atoms
        assert np.all(np.isfinite(pos))


def test_record_every_controls_frame_count(synthetic_water_box, tmp_path):
    """Frame count must track record_every, not silently record every step."""
    from mmml.md.drivers import JaxmdDriver
    from mmml.md.assemble import build_hybrid_energy, _auto_neighbor_fn

    system = synthetic_water_box(seed=0)
    cfg = RunConfig(system=SystemSpec(builder="psf"), terms=("mm_nonbonded",),
                    ensemble=EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=20,
                                          params={"seed": 0}), backend="jaxmd")
    energy = build_hybrid_energy(system, cfg.terms)
    nfn = _auto_neighbor_fn(system, energy, cfg)
    driver = JaxmdDriver(neighbor_fn=nfn, record_every=5, output_path=tmp_path / "t.npz")
    traj = driver.run(system, energy, cfg.ensemble)
    # 20 steps at record_every=5 -> initial frame + 4 recorded = 5 frames.
    assert traj.n_frames == 5


# --- driver input guards (regression against silent misbehaviour) -----------


def test_driver_rejects_negative_steps(synthetic_water_box):
    with pytest.raises(ValueError, match="n_steps"):
        _run(synthetic_water_box(seed=0), n_steps=-1)


def test_driver_rejects_nonpositive_dt(synthetic_water_box):
    with pytest.raises(ValueError, match="dt_fs"):
        _run(synthetic_water_box(seed=0), dt_fs=0.0)


def test_driver_rejects_unknown_ensemble(synthetic_water_box):
    with pytest.raises(NotImplementedError):
        _run(synthetic_water_box(seed=0), ensemble="replica_exchange")


def test_npt_requires_box(synthetic_water_box):
    import dataclasses

    aperiodic = dataclasses.replace(synthetic_water_box(seed=0), box=None)
    with pytest.raises(ValueError, match="requires a periodic box"):
        _run(aperiodic, ensemble="npt", n_steps=5)


def test_assemble_rejects_non_jaxmd_backend(synthetic_water_box):
    cfg = RunConfig(system=SystemSpec(builder="psf"), terms=("mm_nonbonded",),
                    backend="pycharmm")
    with pytest.raises(NotImplementedError, match="jaxmd backend"):
        assemble_and_run(cfg, system=synthetic_water_box(seed=0))
