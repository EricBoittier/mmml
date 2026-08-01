"""Neighbor-list cadence policy: behavior + parity with jaxmd_runner.

``mmml.md.nl_cadence`` is a light copy of the cadence policy that
``jaxmd_runner`` owns, so the λ driver and ``mmml.md`` driver can share it
without importing rich / HDF5 / pycharmm. These tests are what stops the two
copies drifting apart silently.
"""

from __future__ import annotations

import pytest

from mmml.md.nl_cadence import (
    ENSEMBLE_UPDATE_INTERVAL,
    resolve_block_steps,
    resolve_update_interval,
    verlet_reuse_displacement_limit_A,
)


def test_explicit_positive_interval_wins_over_ensemble_default():
    assert resolve_update_interval("nvt", 40) == 40
    assert resolve_update_interval("npt", 1) == 1


@pytest.mark.parametrize("requested", [None, 0, -5])
def test_unset_interval_falls_back_to_ensemble_default(requested):
    assert resolve_update_interval("nvt", requested) == 10
    assert resolve_update_interval("npt", requested) == 5
    assert resolve_update_interval("nve", requested) == 5


def test_free_space_batches_more_than_any_pbc_ensemble():
    free = resolve_update_interval("nvt", None, use_pbc=False)
    assert free == 100
    assert free > max(ENSEMBLE_UPDATE_INTERVAL.values())


def test_unknown_ensemble_is_conservative():
    # Refuse to batch a cadence we have no policy for.
    assert resolve_update_interval("replica_exchange_thing", None) == 1


def test_block_steps_always_divide_the_recording_interval():
    # 40 does not divide 500, so the block must fall back to a divisor rather
    # than overshooting a recording boundary.
    block = resolve_block_steps(
        steps_per_recording=500,
        use_pbc=True,
        has_update_fn=True,
        update_interval=40,
        ensemble="nvt",
    )
    assert 500 % block == 0
    assert block <= 40
    assert block == 25


def test_block_steps_never_exceed_the_recording_interval():
    block = resolve_block_steps(
        steps_per_recording=7,
        use_pbc=True,
        has_update_fn=True,
        update_interval=1000,
        ensemble="nvt",
    )
    assert block == 7


def test_block_steps_without_update_fn_ignore_ensemble_default():
    # No dynamic MM pairs: the interval only bounds the compiled block.
    block = resolve_block_steps(
        steps_per_recording=100,
        use_pbc=True,
        has_update_fn=False,
        update_interval=None,
        ensemble="npt",
    )
    assert block == 100


def test_verlet_limit_is_half_the_skin_and_clamps_negative():
    assert verlet_reuse_displacement_limit_A(0.5) == 0.25
    assert verlet_reuse_displacement_limit_A(-1.0) == 0.0


def test_parity_with_jaxmd_runner_policy():
    """The two copies of the policy must agree, or drift goes unnoticed."""
    jaxmd_runner = pytest.importorskip("mmml.cli.run.jaxmd_runner")

    assert jaxmd_runner.ENSEMBLE_JAXMD_UPDATE_INTERVAL == ENSEMBLE_UPDATE_INTERVAL

    cases = [
        ("nvt", None, True),
        ("nvt", 40, True),
        ("npt", 0, True),
        ("nve", None, True),
        ("nvt", None, False),
        ("bogus", None, True),
    ]
    for ensemble, requested, use_pbc in cases:
        assert resolve_update_interval(ensemble, requested, use_pbc=use_pbc) == (
            jaxmd_runner.resolve_ensemble_jaxmd_update_interval(
                ensemble, requested, use_pbc=use_pbc
            )
        ), f"interval drift for {ensemble=} {requested=} {use_pbc=}"

    block_cases = [
        (500, True, True, 40, "nvt"),
        (200, True, True, None, "npt"),
        (7, True, True, 1000, "nvt"),
        (100, True, False, None, "npt"),
        (1000, False, False, 0, "nve"),
    ]
    for steps_per_recording, use_pbc, has_update_fn, interval, ensemble in block_cases:
        assert resolve_block_steps(
            steps_per_recording=steps_per_recording,
            use_pbc=use_pbc,
            has_update_fn=has_update_fn,
            update_interval=interval,
            ensemble=ensemble,
        ) == jaxmd_runner.resolve_jaxmd_steps_per_loop_call(
            steps_per_recording=steps_per_recording,
            use_pbc=use_pbc,
            has_update_fn=has_update_fn,
            jax_md_update_interval=interval,
            ensemble=ensemble,
        ), f"block drift for {steps_per_recording=} {interval=} {ensemble=}"


def test_parity_with_mm_energy_forces_skin_limit():
    mm = pytest.importorskip("mmml.interfaces.pycharmmInterface.mm_energy_forces")
    for skin in (0.0, 0.25, 0.5, 2.0, -1.0):
        assert verlet_reuse_displacement_limit_A(skin) == (
            mm.verlet_reuse_displacement_limit_A(skin)
        )
