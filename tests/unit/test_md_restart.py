"""Restart payload, handoff round-trip, and position DCD."""

from pathlib import Path

import numpy as np

from karml.md.restart import (
    ang_ps_from_momenta,
    apply_integrator_restart,
    flatten_restart,
    momenta_from_ang_ps,
    snapshot_integrator,
    unflatten_restart,
    write_position_dcd,
)


class _Leaf:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set(self, **kwargs):
        data = dict(self.__dict__)
        data.update(kwargs)
        return _Leaf(**data)


def test_velocity_conversion_is_inverse():
    mass = np.array([1.0, 16.0])
    velocities = np.array([[0.1, -0.2, 0.3], [1.0, 0.0, -0.5]])
    momenta = momenta_from_ang_ps(velocities, mass)
    restored = ang_ps_from_momenta(momenta, mass)
    np.testing.assert_allclose(restored, velocities)


def test_snapshot_apply_and_npz_flatten_round_trip():
    chain = _Leaf(
        position=np.array([0.2, 0.3]),
        momentum=np.array([0.4, 0.5]),
        mass=np.array([1.0, 1.0]),
        tau=np.array(1.5),
        kinetic_energy=np.array(0.25),
        degrees_of_freedom=4,
    )
    state = _Leaf(
        position=np.zeros((2, 3)),
        momentum=np.ones((2, 3)),
        mass=np.array([1.0, 16.0]),
        force=np.zeros((2, 3)),
        chain=chain,
        box_position=np.eye(3) * 10.0,
        box_momentum=np.array([0.1, 0.0, 0.0]),
    )
    snap = snapshot_integrator(state)
    assert snap["kind"] == "nvt_nose_hoover"
    assert "position" not in snap
    assert "box_position" not in snap

    blank = _Leaf(
        position=np.full((2, 3), 3.0),
        momentum=np.zeros((2, 3)),
        mass=np.array([1.0, 16.0]),
        force=np.ones((2, 3)),
        chain=_Leaf(
            position=np.zeros(2),
            momentum=np.zeros(2),
            mass=np.ones(2),
            tau=np.array(0.0),
            kinetic_energy=np.array(0.0),
            degrees_of_freedom=4,
        ),
        box_position=np.eye(3),
        box_momentum=np.zeros(3),
    )
    restored = apply_integrator_restart(blank, unflatten_restart(flatten_restart(snap)))
    np.testing.assert_allclose(restored.momentum, state.momentum)
    np.testing.assert_allclose(restored.chain.position, chain.position)
    np.testing.assert_allclose(restored.box_momentum, state.box_momentum)
    np.testing.assert_allclose(restored.position, blank.position)
    np.testing.assert_allclose(restored.box_position, blank.box_position)
    assert restored.chain.degrees_of_freedom == 4


def test_velocities_ang_ps_fill_momentum():
    state = _Leaf(momentum=np.zeros((1, 3)), mass=np.array([2.0]))
    velocities = np.array([[1.0, 0.0, -1.0]])
    restored = apply_integrator_restart(state, {"velocities_ang_ps": velocities})
    np.testing.assert_allclose(
        restored.momentum, momenta_from_ang_ps(velocities, state.mass)
    )


def test_dcd_header_is_charmm_cord(tmp_path: Path):
    path = tmp_path / "traj.dcd"
    positions = np.zeros((2, 3, 3))
    positions[1, 0, 0] = 1.0
    write_position_dcd(path, positions, dt_fs=1.0, record_every=100)
    raw = path.read_bytes()
    assert raw[4:8] == b"CORD"


def test_handoff_npz_round_trips_integrator(tmp_path: Path):
    from karml.cli.run.md_handoff import (
        MdHandoffState,
        handoff_to_npz_dict,
        load_handoff_from_npz,
    )

    integrator = {
        "kind": "nvt_langevin",
        "momentum": np.array([[0.2, 0.0, -0.1]]),
        "mass": np.array([1.0]),
        "rng": np.array([3, 4], dtype=np.uint32),
    }
    state = MdHandoffState(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1], dtype=np.int32),
        velocities=np.ones((1, 3)),
        metadata={"backend": "jaxmd-unified", "note": "positions+box+velocities+integrator"},
        integrator=integrator,
    )
    path = tmp_path / "state.npz"
    np.savez(path, **handoff_to_npz_dict(state))
    loaded = load_handoff_from_npz(path)
    assert loaded.integrator is not None
    assert loaded.integrator["kind"] == "nvt_langevin"
    np.testing.assert_allclose(loaded.integrator["momentum"], integrator["momentum"])
    np.testing.assert_array_equal(loaded.integrator["rng"], integrator["rng"])
    assert "integrator" not in loaded.metadata


def test_restart_payload_drops_on_pre_minimize_or_fresh_velocities():
    from karml.cli.run.md_handoff import MdHandoffState, set_handoff_in
    from karml.cli.run.md_system_unified import _restart_payload_from_handoff

    state = MdHandoffState(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1], dtype=np.int32),
        velocities=np.ones((1, 3)),
        integrator={"kind": "nve", "momentum": np.ones((1, 3))},
    )
    set_handoff_in(state)
    try:
        class _Args:
            continue_velocities = True
            handoff_pre_minimize = False

        args = _Args()
        payload = _restart_payload_from_handoff(args)
        assert payload is not None
        assert payload["kind"] == "nve"
        args.handoff_pre_minimize = True
        assert _restart_payload_from_handoff(args) is None
        args.handoff_pre_minimize = False
        args.continue_velocities = False
        assert _restart_payload_from_handoff(args) is None
    finally:
        set_handoff_in(None)
