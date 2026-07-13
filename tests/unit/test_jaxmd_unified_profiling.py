"""Profiling / performance-regression harness for the jaxmd-unified stack.

Times the hot paths of the ``mmml.md`` pipeline on the synthetic water box
(no CHARMM, no checkpoint): energy-function build+compile, steady-state MD
throughput, neighbor-list rebuild, and how throughput scales with system size.

These are marked ``slow`` (they run real multi-step dynamics and JIT compiles),
so the default fast suite skips them::

    pytest tests/unit/test_jaxmd_unified_profiling.py -m slow

A JSON report is written to ``MMML_PROFILE_OUT`` if set, else a temp file, so a
CI job can diff throughput over time. The assertions are deliberately loose
floors: they catch catastrophic regressions (an accidental recompile per step,
losing jit, O(N^2) blow-ups) without being flaky on a slow shared runner.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from mmml.md import EnsembleSpec, RunConfig, SystemSpec, assemble_and_run
from mmml.md.assemble import _auto_neighbor_fn, build_hybrid_energy
from mmml.md.drivers import JaxmdDriver

pytestmark = [pytest.mark.unit, pytest.mark.slow]

pytest.importorskip("jax")
pytest.importorskip("jax_md")


def _time(fn, *, repeat=3):
    """Best-of-``repeat`` wall time (seconds); best-of reduces scheduler noise."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _block(x):
    """Force materialization of a jax result so timings aren't async-hidden."""
    import jax

    return jax.block_until_ready(x)


def _report(record: dict) -> None:
    out = os.environ.get("MMML_PROFILE_OUT")
    path = Path(out) if out else Path(os.environ.get("PYTEST_CURRENT_TEST", "")).parent
    if not out:
        return  # nothing requested; the captured stdout below is the record
    path = Path(out)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (ValueError, OSError):
            existing = []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))


def _energy_and_R(system):
    import jax.numpy as jnp

    energy = build_hybrid_energy(system, ("mm_nonbonded",))
    fn = energy.as_jax_energy_fn()
    R = jnp.asarray(system.R)
    cfg = RunConfig(system=SystemSpec(builder="psf"), terms=("mm_nonbonded",), backend="jaxmd")
    nfn = _auto_neighbor_fn(system, energy, cfg)
    nbrs = nfn(np.asarray(system.R), np.asarray(system.box))
    kw = {k: jnp.asarray(v) for k, v in nbrs.items()}
    return fn, R, kw, nfn


def test_profile_energy_build_and_eval(synthetic_water_box, capsys):
    """Compile once, then energy evals should be cheap and recompile-free."""
    system = synthetic_water_box(n_waters=16, seed=0)
    fn, R, kw, _ = _energy_and_R(system)

    t_compile = _time(lambda: _block(fn(R, **kw)), repeat=1)
    # steady-state: many evals; if it were recompiling this would blow up.
    n_eval = 50
    t_eval = _time(lambda: [_block(fn(R, **kw)) for _ in range(n_eval)], repeat=2) / n_eval

    e0 = float(fn(R, **kw))
    assert np.isfinite(e0)
    # A warm eval must be far cheaper than the cold compile (no per-call trace).
    assert t_eval < 0.5 * t_compile + 0.05
    rec = {"bench": "energy_eval", "n_atoms": system.n_atoms,
           "compile_s": t_compile, "eval_s": t_eval}
    _report(rec)
    print("\nPROFILE", json.dumps(rec))


def test_profile_md_throughput(synthetic_water_box, capsys):
    """Steady-state MD steps/second on a fixed system, past the first compile."""
    system = synthetic_water_box(n_waters=16, seed=0)

    def _n_step_run(n):
        cfg = RunConfig(
            system=SystemSpec(builder="psf"), terms=("mm_nonbonded",),
            ensemble=EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=n, params={"seed": 0}),
            backend="jaxmd",
        )
        return assemble_and_run(cfg, system=system)

    # warm up (compile), then time a longer run and subtract nothing: the
    # per-step cost is dominated by the stepping, not the one-off compile.
    _n_step_run(5)
    n = 200
    t = _time(lambda: _n_step_run(n), repeat=1)
    steps_per_s = n / t
    assert steps_per_s > 20.0, f"MD throughput regressed: {steps_per_s:.1f} steps/s"
    rec = {"bench": "md_throughput", "n_atoms": system.n_atoms,
           "n_steps": n, "wall_s": t, "steps_per_s": steps_per_s}
    _report(rec)
    print("\nPROFILE", json.dumps(rec))


def test_profile_neighbor_rebuild(synthetic_water_box, capsys):
    """Neighbor-list rebuild time (host pair build) must stay modest."""
    system = synthetic_water_box(n_waters=32, seed=0)
    _, _, _, nfn = _energy_and_R(system)
    pos = np.asarray(system.R)
    box = np.asarray(system.box)

    t = _time(lambda: nfn(pos, box), repeat=5)
    nbrs = nfn(pos, box)
    assert "pair_i" in nbrs and "pair_j" in nbrs
    assert t < 1.0, f"neighbor rebuild too slow: {t*1e3:.1f} ms"
    rec = {"bench": "neighbor_rebuild", "n_atoms": system.n_atoms,
           "rebuild_s": t, "n_pairs": int(np.asarray(nbrs["pair_i"]).shape[0])}
    _report(rec)
    print("\nPROFILE", json.dumps(rec))


@pytest.mark.parametrize("n_waters", [8, 16, 32])
def test_profile_scaling(synthetic_water_box, n_waters, capsys):
    """Throughput across sizes; recorded so a CI job can watch the trend."""
    system = synthetic_water_box(n_waters=n_waters, box_len=8.0 + n_waters, seed=0)
    energy = build_hybrid_energy(system, ("mm_nonbonded",))
    nfn = _auto_neighbor_fn(
        system, energy,
        RunConfig(system=SystemSpec(builder="psf"), terms=("mm_nonbonded",), backend="jaxmd"),
    )
    driver = JaxmdDriver(neighbor_fn=nfn, record_every=100)
    ens = EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=100, params={"seed": 0})
    driver.run(system, energy, EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=5,
                                            params={"seed": 0}))  # warm compile
    t = _time(lambda: driver.run(system, energy, ens), repeat=1)
    steps_per_s = 100 / t
    rec = {"bench": "scaling", "n_atoms": system.n_atoms, "steps_per_s": steps_per_s}
    _report(rec)
    print("\nPROFILE", json.dumps(rec))
    assert steps_per_s > 5.0
