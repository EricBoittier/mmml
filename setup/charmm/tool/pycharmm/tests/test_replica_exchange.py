import math
import sys
import types

import numpy as np
import pytest

import pycharmm.replica_exchange as rex


def _clear_mpi_launch_environment(monkeypatch):
    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_STEP_NUM_TASKS",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    ):
        monkeypatch.delenv(name, raising=False)


def _state(
    label,
    ph,
    lambdas,
    biases,
    temperature=300.0,
    masses=None,
    frictions=None,
    fixed=None,
    sites=None,
):
    size = len(lambdas)
    observation = rex.PhObservation(
        temperature,
        lambdas,
        masses if masses is not None else np.ones(size),
        frictions if frictions is not None else np.ones(size) * 5.0,
        fixed if fixed is not None else np.zeros(size, dtype=bool),
        sites if sites is not None else np.arange(size),
    )
    return rex.State(rex.PhLabel(label, ph, biases), observation)


def _record(delta=0.0, observation=None, attempt=0, error=None):
    return {
        "attempt": attempt,
        "delta": delta,
        "observation": observation,
        "error": error,
    }


class _Comm:
    def __init__(self, rank, labels, kind, records=None, seed=0):
        self.rank = rank
        self.labels = labels
        self.kind = kind
        self.records = records
        self.seed = seed
        self.gathers = 0
        self.aborts = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return len(self.labels)

    def allgather(self, value):
        self.gathers += 1
        if "kind" in value:
            return [
                {
                    "kind": self.kind,
                    "seed": self.seed,
                    "label": label,
                    "error": None,
                }
                for label in self.labels
            ]
        records = [dict(record) for record in self.records]
        key = (lambda label: label.ph) if self.kind == "ph" else (lambda label: label)
        pairs = rex._neighbor_pairs(
            self.labels,
            value["attempt"],
            key,
        )
        for rank, record in enumerate(records):
            record.setdefault(
                "label",
                rex._label_token(self.kind, self.labels[rank]),
            )
            record.setdefault("partner", rex._partner_for_rank(rank, pairs))
        records[self.rank] = value
        return records

    def Abort(self, code):
        assert code == 1
        self.aborts += 1


class _Accessor:
    def __init__(self, *states):
        self.states = list(states)
        self.reads = 0
        self.applied = []

    def read_state(self, label):
        state = self.states[min(self.reads, len(self.states) - 1)]
        self.reads += 1
        assert label == state.label.label
        return state

    def apply_label(self, label):
        self.applied.append(label)


def _ph_comm(rank, states, records=None, seed=0):
    return _Comm(
        rank,
        [state.label for state in states],
        "ph",
        records=records,
        seed=seed,
    )


@pytest.fixture
def validated_temperature(monkeypatch):
    checked = []
    monkeypatch.setattr(rex, "_current_potential_energy", lambda: 0.0)
    monkeypatch.setattr(
        rex.dynamics,
        "validate_rex_temperature_direct",
        lambda temperature: checked.append(temperature) or True,
    )
    return checked


def test_ph_delta_matches_four_explicit_cross_energies():
    state_i = _state(0, 5.0, [0.75, 0.25], [1.0, -2.0])
    state_j = _state(1, 6.0, [0.20, 0.80], [3.0, -1.0])

    def energy(bias, lambdas):
        return -np.dot(bias, lambdas)

    expected = (
        energy(state_j.label.biases, state_i.observation.lambdas)
        + energy(state_i.label.biases, state_j.observation.lambdas)
        - energy(state_i.label.biases, state_i.observation.lambdas)
        - energy(state_j.label.biases, state_j.observation.lambdas)
    ) / (rex.KBOLTZ * 300.0)

    assert rex.ph_bias_delta(state_i, state_j) == pytest.approx(expected)


def test_favorable_ph_delta_is_always_accepted():
    state_i = _state(0, 5.0, [0.0, 0.0], [0.0, 1.0])
    state_j = _state(1, 6.0, [0.0, 1.0], [0.0, 0.0])

    assert rex.ph_bias_delta(state_i, state_j) < 0.0
    assert rex.ph_bias_probability(state_i, state_j) == 1.0


def test_ph_accessor_applies_only_the_complete_label(monkeypatch):
    values = {
        "ph": 5.0,
        "temperature": 300.0,
        "lambdas": np.array([1.0, 0.3, 0.7]),
        "biases": np.array([0.0, 2.0, -1.0]),
        "masses": np.ones(3),
        "frictions": np.ones(3) * 5.0,
        "fixed": np.array([False, False, True]),
        "sites": np.array([0, 1, 1]),
    }
    applied = []
    monkeypatch.setattr(rex.block, "get_ph_rex_state_direct", lambda: values)
    monkeypatch.setattr(
        rex.block,
        "set_ph_rex_label_direct",
        lambda ph, biases: applied.append((ph, biases.copy())) or True,
    )

    accessor = rex._PhStateAccessor()
    state = accessor.read_state(label=9)
    accessor.apply_label(rex.PhLabel(4, 6.0, [0.0, 4.0, -3.0]))

    assert state.label.label == 9
    assert np.array_equal(state.observation.lambdas, values["lambdas"])
    assert applied[0][0] == 6.0
    assert np.array_equal(applied[0][1], [0.0, 4.0, -3.0])


def test_ph_exchange_uses_one_collective_per_attempt(monkeypatch):
    states = [
        _state(0, 5.0, [0.0, 0.0], [0.0, 1.0]),
        _state(1, 6.0, [0.0, 1.0], [0.0, 0.0]),
    ]
    partner_delta = rex._ph_local_delta(states[1], states[0].label)
    records = [
        _record(),
        _record(partner_delta, states[1].observation),
    ]
    comm = _ph_comm(0, states, records, seed=7)
    accessor = _Accessor(states[0])
    monkeypatch.setattr(rex, "_PhStateAccessor", lambda: accessor)
    exchange = rex.ReplicaExchange.ph(
        comm=comm,
        seed=7,
    )

    result = exchange.attempt(0)

    assert result.accepted
    assert result.partner_rank == 1
    assert result.label == 6.0
    assert accessor.applied == [states[1].label]
    assert comm.gathers == 2  # one initialization, one attempt
    assert not hasattr(rex, "PHReplicaExchange")


def test_rejected_ph_exchange_keeps_label(monkeypatch):
    states = [
        _state(0, 5.0, [0.0, 1.0], [0.0, 100.0]),
        _state(1, 6.0, [0.0, 0.0], [0.0, 0.0]),
    ]
    records = [
        _record(),
        _record(
            rex._ph_local_delta(states[1], states[0].label),
            states[1].observation,
        ),
    ]
    accessor = _Accessor(states[0])
    monkeypatch.setattr(rex, "_PhStateAccessor", lambda: accessor)
    result = rex.ReplicaExchange.ph(
        comm=_ph_comm(0, states, records, seed=3),
        seed=3,
    ).attempt(0)

    assert result.attempted
    assert not result.accepted
    assert result.probability < 1.0e-50
    assert result.label == 5.0
    assert accessor.applied == []


def test_neighbor_schedule_follows_current_label_order():
    labels = [
        rex.PhLabel(0, 7.0, [0.0]),
        rex.PhLabel(1, 5.0, [0.0]),
        rex.PhLabel(2, 8.0, [0.0]),
        rex.PhLabel(3, 6.0, [0.0]),
    ]

    def key(label):
        return label.ph

    assert rex._neighbor_pairs(labels, 0, key) == [(1, 3), (0, 2)]
    assert rex._neighbor_pairs(labels, 1, key) == [(3, 0)]
    assert rex._neighbor_pairs(labels[:2], 1, key) == [(1, 0)]


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 301.0},
        {"masses": [1.0, 2.0]},
        {"frictions": [5.0, 6.0]},
        {"fixed": [False, True]},
        {"sites": [0, 2]},
    ],
)
def test_ph_protocol_mismatch_fails_collectively(changed, monkeypatch):
    states = [
        _state(0, 5.0, [0.4, 0.6], [0.0, 1.0]),
        _state(1, 6.0, [0.6, 0.4], [0.0, 2.0], **changed),
    ]
    records = [
        _record(),
        _record(0.0, states[1].observation),
    ]
    comm = _ph_comm(0, states, records)
    monkeypatch.setattr(
        rex,
        "_PhStateAccessor",
        lambda: _Accessor(states[0]),
    )
    exchange = rex.ReplicaExchange.ph(comm=comm)

    with pytest.raises(rex.Error, match="different|equal"):
        exchange.attempt(0)

    assert comm.aborts == 1


def test_temperature_exchange_updates_main_nvt_temperature(monkeypatch, validated_temperature):
    labels = [300.0, 330.0]
    comm = _Comm(0, labels, "temperature", [_record(), _record(0.0)])
    applied = []
    monkeypatch.setattr(
        rex.dynamics,
        "apply_rex_temperature_direct",
        lambda old, new: applied.append((old, new)) or True,
        raising=False,
    )
    exchange = rex.ReplicaExchange.temperature(
        300.0,
        comm=comm,
    )

    result = exchange.attempt(0)

    assert result.accepted
    assert result.label == 330.0
    assert applied == [(300.0, 330.0)]
    assert validated_temperature == [300.0]


def test_three_rank_temperature_exchange_handles_reverse_rank_pair(
    monkeypatch, validated_temperature
):
    labels = [300.0, 330.0, 315.0]
    records = [_record(attempt=1) for _ in labels]
    comm = _Comm(2, labels, "temperature", records)
    applied = []
    monkeypatch.setattr(
        rex.dynamics,
        "apply_rex_temperature_direct",
        lambda old, new: applied.append((old, new)) or True,
    )
    exchange = rex.ReplicaExchange.temperature(
        315.0,
        comm=comm,
    )

    result = exchange.attempt(1)

    assert result.accepted
    assert result.partner_rank == 1
    assert result.label == 330.0
    assert applied == [(315.0, 330.0)]
    assert validated_temperature == [315.0]


def test_three_rank_idle_replica_tracks_remote_swap(monkeypatch, validated_temperature):
    labels = [300.0, 330.0, 315.0]
    comm = _Comm(
        0,
        labels,
        "temperature",
        [_record(attempt=1) for _ in labels],
    )
    monkeypatch.setattr(
        rex.dynamics,
        "apply_rex_temperature_direct",
        lambda old, new: True,
    )
    exchange = rex.ReplicaExchange.temperature(
        300.0,
        comm=comm,
    )

    idle = exchange.attempt(1)
    comm.labels = list(exchange._labels)
    comm.records = [_record(attempt=2) for _ in labels]
    active = exchange.attempt(2)

    assert not idle.attempted
    assert active.partner_rank == 1
    assert active.label == 315.0


def test_temperature_exchange_surfaces_unsupported_dynamics(monkeypatch, validated_temperature):
    labels = [300.0, 330.0]
    comm = _Comm(0, labels, "temperature", [_record(), _record(0.0)])
    monkeypatch.setattr(
        rex.dynamics,
        "apply_rex_temperature_direct",
        lambda old, new: False,
        raising=False,
    )
    exchange = rex.ReplicaExchange.temperature(
        300.0,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="main NVT temperature"):
        exchange.attempt(0)

    assert comm.aborts == 1


def test_temperature_exchange_preserves_low_level_error(monkeypatch, validated_temperature):
    labels = [300.0, 330.0]
    comm = _Comm(0, labels, "temperature", [_record(), _record(0.0)])

    def reject_npt(old, new):
        raise RuntimeError("temperature exchange requires NVT")

    monkeypatch.setattr(
        rex.dynamics,
        "apply_rex_temperature_direct",
        reject_npt,
        raising=False,
    )
    exchange = rex.ReplicaExchange.temperature(
        300.0,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="requires NVT"):
        exchange.attempt(0)


def test_temperature_state_is_validated_before_energy(monkeypatch):
    labels = [300.0, 330.0]
    comm = _Comm(0, labels, "temperature", [_record(), _record(0.0)])
    evaluated = []

    def reject_wrong_label(temperature):
        raise RuntimeError("thermostat temperature does not match")

    monkeypatch.setattr(
        rex.dynamics,
        "validate_rex_temperature_direct",
        reject_wrong_label,
    )
    monkeypatch.setattr(
        rex,
        "_current_potential_energy",
        lambda: evaluated.append(True) or 0.0,
    )
    exchange = rex.ReplicaExchange.temperature(
        300.0,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="does not match"):
        exchange.attempt(0)

    assert evaluated == []
    assert comm.aborts == 1


def test_default_temperature_energy_is_potential_not_total(monkeypatch):
    seen = []
    monkeypatch.setattr(
        rex.energy,
        "get_eprop",
        lambda index: seen.append(index) or -12.5,
    )

    assert rex._current_potential_energy() == -12.5
    assert seen == [rex.energy.PROP_ENER]


def test_hamiltonian_exchange_restores_trial_then_applies_accepted_label():
    active = [0]
    applications = []
    evaluations = []

    def apply_label(label):
        applications.append(label)
        active[0] = label

    def reduced_potential(label):
        assert active[0] == label
        evaluations.append(label)
        return {0: 2.0, 1: 1.0}[label]

    comm = _Comm(0, [0, 1], "hamiltonian", [_record(), _record(-1.0)])
    exchange = rex.ReplicaExchange.hamiltonian(
        0,
        reduced_potential,
        apply_label,
        comm=comm,
    )

    result = exchange.attempt(0)

    assert result.accepted
    assert result.label == 1
    assert evaluations == [0, 1, 0, 1]
    assert applications == [1, 0, 1]
    assert active[0] == 1


def test_hamiltonian_cross_evaluation_failure_restores_local_label():
    active = [0]
    applications = []

    def apply_label(label):
        applications.append(label)
        active[0] = label

    def reduced_potential(label):
        if label == 1:
            raise RuntimeError("cross-energy failed")
        return 0.0

    comm = _Comm(0, [0, 1], "hamiltonian", [_record(), _record(0.0)])
    exchange = rex.ReplicaExchange.hamiltonian(
        0,
        reduced_potential,
        apply_label,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="cross-energy failed"):
        exchange.attempt(0)

    assert applications == [1, 0]
    assert active[0] == 0
    assert comm.aborts == 1


def test_hamiltonian_exchange_rejects_incomplete_restoration():
    active = [0]
    local_evaluations = [2.0, 3.0]

    def apply_label(label):
        active[0] = label

    def reduced_potential(label):
        assert active[0] == label
        if label == 0:
            return local_evaluations.pop(0)
        return 1.0

    comm = _Comm(0, [0, 1], "hamiltonian", [_record(), _record(0.0)])
    exchange = rex.ReplicaExchange.hamiltonian(
        0,
        reduced_potential,
        apply_label,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="restoration changed"):
        exchange.attempt(0)

    assert active[0] == 0
    assert comm.aborts == 1


def test_hamiltonian_commit_refresh_failure_rolls_back():
    active = [0]
    partner_evaluations = [1.0, RuntimeError("commit refresh failed")]

    def apply_label(label):
        active[0] = label

    def reduced_potential(label):
        assert active[0] == label
        if label == 0:
            return 2.0
        value = partner_evaluations.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    comm = _Comm(0, [0, 1], "hamiltonian", [_record(), _record(-1.0)])
    exchange = rex.ReplicaExchange.hamiltonian(
        0,
        reduced_potential,
        apply_label,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="commit refresh failed"):
        exchange.attempt(0)

    assert active[0] == 0
    assert comm.aborts == 1


def test_hamiltonian_commit_energy_mismatch_rolls_back():
    active = [0]
    partner_evaluations = [1.0, 1.5]

    def apply_label(label):
        active[0] = label

    def reduced_potential(label):
        if label == 0:
            return 2.0
        return partner_evaluations.pop(0)

    comm = _Comm(0, [0, 1], "hamiltonian", [_record(), _record(-1.0)])
    exchange = rex.ReplicaExchange.hamiltonian(
        0,
        reduced_potential,
        apply_label,
        comm=comm,
    )

    with pytest.raises(rex.Error, match="commit changed"):
        exchange.attempt(0)

    assert active[0] == 0
    assert comm.aborts == 1


def test_default_comm_rejects_missing_mpi4py_for_multi_rank_launch(monkeypatch):
    _clear_mpi_launch_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "2")

    with pytest.raises(rex.Error, match="mpi4py is required"):
        rex._default_comm()


@pytest.mark.parametrize("world_size", [1, 4])
def test_default_comm_rejects_world_size_mismatch(monkeypatch, world_size):
    _clear_mpi_launch_environment(monkeypatch)
    comm = types.SimpleNamespace(Get_size=lambda: world_size)
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        types.SimpleNamespace(MPI=types.SimpleNamespace(COMM_WORLD=comm)),
    )
    monkeypatch.setenv("PMI_SIZE", "2")

    with pytest.raises(rex.Error, match="active launch environment implies 2 ranks"):
        rex._default_comm()


def test_implied_mpi_size_uses_known_launch_variables():
    assert rex._implied_mpi_size_from_environment({"PMIX_SIZE": "4"}) == 4
    assert rex._implied_mpi_size_from_environment({"SLURM_STEP_NUM_TASKS": "4"}) == 4
    assert rex._implied_mpi_size_from_environment({"SLURM_NTASKS": "4", "SLURM_PROCID": "0"}) == 4
    assert rex._implied_mpi_size_from_environment({"PMI_SIZE": "invalid"}) is None


def test_allocation_only_slurm_environment_allows_serial_fallback(monkeypatch):
    _clear_mpi_launch_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setenv("SLURM_NTASKS", "4")

    assert rex._default_comm().Get_size() == 1


def test_invalid_arrays_are_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        rex.PhLabel(0, 5.0, [[0.0]])
    with pytest.raises(ValueError, match="matching length"):
        rex.PhObservation(300.0, [1.0], [1.0, 1.0], [5.0], [False], [0])
    with pytest.raises(ValueError, match="finite"):
        rex.PhLabel(0, 5.0, [math.nan])
