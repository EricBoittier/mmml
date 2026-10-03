# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

"""In-memory label exchange for pyCHARMM.

``ReplicaExchange`` provides one nearest-neighbor scheduler for pH,
temperature, and Hamiltonian labels. Coordinates remain on their MPI
rank. Accepted labels move between ranks.

The pH path evaluates the BIELAM contribution analytically and applies labels
through CHARMM's direct pH API. The temperature path exchanges the main NVT
temperature and rescales velocities through the dynamics API. Custom labels
use caller-supplied reduced-potential and label-application callbacks.
"""

from collections import namedtuple
import math
import os
import random

import numpy as np

import pycharmm.block as block
import pycharmm.dynamics as dynamics
import pycharmm.energy as energy


KBOLTZ = 0.001987191


class Error(RuntimeError):
    """Raised when replica exchange cannot proceed safely."""


_PhLabel = namedtuple("_PhLabel", "label ph biases")


class PhLabel(_PhLabel):
    """pH and BIELAM values that move together."""

    __slots__ = ()

    def __new__(cls, label, ph, biases):
        return super(PhLabel, cls).__new__(
            cls,
            int(label),
            _finite_float(ph, "ph"),
            _float_vector(biases, "biases"),
        )


_PhObservation = namedtuple(
    "_PhObservation",
    "temperature lambdas masses frictions fixed sites",
)


class PhObservation(_PhObservation):
    """Local pH/MSLD observables and protocol invariants."""

    __slots__ = ()

    def __new__(
        cls,
        temperature,
        lambdas,
        masses,
        frictions,
        fixed,
        sites,
    ):
        temperature = _positive_float(temperature, "temperature")
        lambdas = _float_vector(lambdas, "lambdas")
        masses = _float_vector(masses, "masses")
        frictions = _float_vector(frictions, "frictions")
        fixed = _vector(fixed, np.bool_, "fixed")
        sites = _vector(sites, np.int32, "sites")
        size = lambdas.size
        if any(values.size != size for values in (masses, frictions, fixed, sites)):
            raise ValueError("all pH/MSLD state arrays must have matching length")
        return super(PhObservation, cls).__new__(
            cls,
            temperature,
            lambdas,
            masses,
            frictions,
            fixed,
            sites,
        )


State = namedtuple("State", "label observation")
ExchangeResult = namedtuple(
    "ExchangeResult",
    "attempted accepted partner_rank probability random_value label",
)


class _PhStateAccessor:
    """Read local pH/MSLD state and apply one complete pH label."""

    def read_state(self, label):
        values = block.get_ph_rex_state_direct()
        if values is None:
            raise Error("CHARMM pH/MSLD replica-exchange state is unavailable")
        ph_label = PhLabel(label, values["ph"], values["biases"])
        observation = PhObservation(
            values["temperature"],
            values["lambdas"],
            values["masses"],
            values["frictions"],
            values["fixed"],
            values["sites"],
        )
        if ph_label.biases.size != observation.lambdas.size:
            raise Error("CHARMM pH bias and lambda arrays have different lengths")
        return State(ph_label, observation)

    def apply_label(self, ph_label):
        if not isinstance(ph_label, PhLabel):
            raise TypeError("pH exchange requires a PhLabel")
        if not block.set_ph_rex_label_direct(ph_label.ph, ph_label.biases):
            raise Error("failed to apply pH/BIELAM label in CHARMM")


class ReplicaExchange:
    """Nearest-neighbor label exchange with one collective per attempt.

    Construct a protocol with :meth:`ph`, :meth:`temperature`, or
    :meth:`hamiltonian`. All ranks must use the same communicator, seed,
    protocol, and attempt index.
    """

    def __init__(
        self,
        kind,
        label,
        comm=None,
        seed=0,
        state_accessor=None,
        reduced_potential=None,
        apply_label=None,
    ):
        self.kind = kind
        self.comm = comm if comm is not None else _default_comm()
        self.rank = int(self.comm.Get_rank())
        self.size = int(self.comm.Get_size())
        if self.size < 1 or self.rank < 0 or self.rank >= self.size:
            raise Error("invalid replica communicator rank/size")

        self.seed = int(seed)
        self._label = label
        self._state_accessor = state_accessor
        self._reduced_potential = reduced_potential
        self._apply_label = apply_label

        try:
            self._labels = self._initialize_labels()
        except Exception:
            self._abort()
            raise

    @classmethod
    def ph(cls, comm=None, seed=0):
        """Create direct pH/MSLD label exchange."""
        return cls(
            "ph",
            None,
            comm=comm,
            seed=seed,
            state_accessor=_PhStateAccessor(),
        )

    @classmethod
    def temperature(
        cls,
        temperature,
        comm=None,
        seed=0,
    ):
        """Create main-temperature exchange for ordinary single-bath NVT.

        The dynamics-level direct setter follows FAST REPDSTR: it changes the
        main bath, rescales atomic velocities, and preserves the effective
        lambda thermostat temperature.
        If ``TBLD`` is zero, the first accepted swap stores its effective
        pre-swap value instead of letting it follow the new main bath.

        Standard, DOMDEC, and OpenMM segments must continue from CHARMM's
        stored velocities with ``iasvel=0``. Use :attr:`label` for every
        explicit physical-temperature argument in the next segment. BLaDE
        segments must use its live ``ABIC`` state. A restart written before an
        accepted attempt contains the pre-exchange temperature and velocities.
        Call :meth:`attempt` before clearing or switching the active backend.
        """
        return cls(
            "temperature",
            temperature,
            comm=comm,
            seed=seed,
        )

    @classmethod
    def hamiltonian(
        cls,
        label,
        reduced_potential,
        apply_label,
        comm=None,
        seed=0,
    ):
        """Create fixed-temperature Hamiltonian label exchange.

        ``reduced_potential(label)`` must return a dimensionless potential for
        the active configuration and label. ``apply_label(label)`` must install
        the complete label, synchronize the active backend, and raise or return
        ``False`` on failure. It must be reversible by applying the old label.
        Each trial label is restored and reevaluated before the collective
        exchange decision.
        """
        if not callable(reduced_potential) or not callable(apply_label):
            raise TypeError("reduced_potential and apply_label must be callable")
        return cls(
            "hamiltonian",
            label,
            comm=comm,
            seed=seed,
            reduced_potential=reduced_potential,
            apply_label=apply_label,
        )

    @property
    def label(self):
        """Return the current local bookkeeping or thermodynamic label."""
        return _public_label(self.kind, self._labels[self.rank])

    def attempt(self, attempt_index):
        """Attempt one even/odd exchange between adjacent current labels."""
        try:
            return self._attempt(int(attempt_index))
        except Exception:
            self._abort()
            raise

    def _attempt(self, attempt_index):
        if attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")

        pairs = _neighbor_pairs(self._labels, attempt_index, self._sort_key)
        partner = _partner_for_rank(self.rank, pairs)
        local_label = self._labels[self.rank]
        partner_label = self._labels[partner] if partner is not None else None

        try:
            delta, observation = self._local_delta(local_label, partner_label)
            local_error = None
        except Exception as exc:
            delta, observation = 0.0, None
            local_error = "{}: {}".format(type(exc).__name__, exc)

        records = list(
            self.comm.allgather(
                {
                    "attempt": attempt_index,
                    "label": _label_token(self.kind, local_label),
                    "partner": partner,
                    "delta": delta,
                    "observation": observation,
                    "error": local_error,
                }
            )
        )
        self._validate_records(records, attempt_index)

        decisions = {}
        for left, right in pairs:
            pair_delta = records[left]["delta"] + records[right]["delta"]
            probability = metropolis_probability(pair_delta)
            random_value = _pair_random(
                self.seed,
                attempt_index,
                left,
                right,
            )
            decisions[tuple(sorted((left, right)))] = (
                random_value < probability,
                probability,
                random_value,
            )

        new_labels = list(self._labels)
        for (left, right), decision in decisions.items():
            if decision[0]:
                new_labels[left], new_labels[right] = (
                    new_labels[right],
                    new_labels[left],
                )

        if partner is None:
            self._labels = new_labels
            return ExchangeResult(False, False, None, None, None, self.label)

        pair = tuple(sorted((self.rank, partner)))
        accepted, probability, random_value = decisions[pair]
        if accepted:
            self._apply(local_label, partner_label, observation)
        self._labels = new_labels

        return ExchangeResult(
            True,
            accepted,
            partner,
            probability,
            random_value,
            self.label,
        )

    def _initialize_labels(self):
        try:
            if self.kind == "ph":
                label = self.rank if self._label is None else int(self._label)
                local_label = self._state_accessor.read_state(label).label
            elif self.kind == "temperature":
                local_label = _positive_float(self._label, "temperature")
            elif self.kind == "hamiltonian":
                hash(self._label)
                local_label = self._label
            else:
                raise ValueError("unknown replica-exchange protocol")
            local_error = None
        except Exception as exc:
            local_label = None
            local_error = "{}: {}".format(type(exc).__name__, exc)

        records = list(
            self.comm.allgather(
                {
                    "kind": self.kind,
                    "seed": self.seed,
                    "label": local_label,
                    "error": local_error,
                }
            )
        )
        if len(records) != self.size:
            raise Error("replica allgather returned the wrong number of labels")
        errors = [
            "rank {}: {}".format(rank, record["error"])
            for rank, record in enumerate(records)
            if record.get("error")
        ]
        if errors:
            raise Error("; ".join(errors))
        if any(record.get("kind") != self.kind for record in records):
            raise Error("replicas use different exchange protocols")
        if any(record.get("seed") != self.seed for record in records):
            raise Error("replicas use different exchange seeds")

        labels = [record["label"] for record in records]
        _validate_labels(self.kind, labels)
        return labels

    def _local_delta(self, local_label, partner_label):
        if self.kind == "ph":
            state = self._state_accessor.read_state(local_label.label)
            _require_current_ph_label(state.label, local_label)
            delta = (
                _ph_local_delta(state, partner_label)
                if partner_label is not None
                else 0.0
            )
            return delta, state.observation

        if self.kind == "temperature":
            dynamics.validate_rex_temperature_direct(local_label)
            potential = _finite_float(
                _current_potential_energy(),
                "potential energy",
            )
            if partner_label is None:
                return 0.0, None
            beta_change = (
                1.0 / (KBOLTZ * partner_label)
                - 1.0 / (KBOLTZ * local_label)
            )
            return beta_change * potential, None

        if partner_label is None:
            return 0.0, None
        own = _finite_float(
            self._reduced_potential(local_label),
            "reduced potential",
        )
        try:
            self._call_custom_apply(partner_label)
            cross = _finite_float(
                self._reduced_potential(partner_label),
                "reduced potential",
            )
        finally:
            self._call_custom_apply(local_label)
            restored = _finite_float(
                self._reduced_potential(local_label),
                "restored reduced potential",
            )
            if not math.isclose(restored, own, rel_tol=1.0e-10, abs_tol=1.0e-10):
                raise Error(
                    "Hamiltonian label restoration changed the reduced potential"
                )
        return cross - own, cross

    def _validate_records(self, records, attempt_index):
        if len(records) != self.size:
            raise Error("replica allgather returned the wrong number of records")
        errors = [
            "rank {}: {}".format(rank, record["error"])
            for rank, record in enumerate(records)
            if record.get("error")
        ]
        if errors:
            raise Error("; ".join(errors))
        if any(record.get("attempt") != attempt_index for record in records):
            raise Error("replicas use different exchange attempt indices")
        expected_pairs = _neighbor_pairs(
            self._labels,
            attempt_index,
            self._sort_key,
        )
        expected_partners = {
            rank: _partner_for_rank(rank, expected_pairs)
            for rank in range(self.size)
        }
        for rank, record in enumerate(records):
            if record.get("partner") != expected_partners[rank]:
                raise Error("replicas computed different exchange pairs")
            if record.get("label") != _label_token(
                    self.kind, self._labels[rank]):
                raise Error("replicas have different current label maps")
        if any(not math.isfinite(record.get("delta", math.nan)) for record in records):
            raise Error("replica reduced-potential change must be finite")
        if self.kind == "ph":
            states = [
                State(label, record["observation"])
                for label, record in zip(self._labels, records)
            ]
            _validate_ph_protocol(states)

    def _sort_key(self, label):
        if self.kind == "ph":
            return label.ph
        return label

    def _apply(self, old_label, new_label, trial_cross):
        if self.kind == "ph":
            self._state_accessor.apply_label(new_label)
            return
        if self.kind == "temperature":
            try:
                applied = dynamics.apply_rex_temperature_direct(
                    old_label,
                    new_label,
                )
            except Exception as exc:
                raise Error(str(exc)) from exc
            if not applied:
                raise Error(
                    "failed to apply the main NVT temperature in CHARMM"
                )
            return
        try:
            self._call_custom_apply(new_label)
            committed = _finite_float(
                self._reduced_potential(new_label),
                "committed reduced potential",
            )
            if not math.isclose(
                    committed, trial_cross, rel_tol=1.0e-10, abs_tol=1.0e-10):
                raise Error(
                    "Hamiltonian label commit changed the reduced potential"
                )
        except Exception as exc:
            try:
                self._call_custom_apply(old_label)
                _finite_float(
                    self._reduced_potential(old_label),
                    "rolled-back reduced potential",
                )
            except Exception as rollback_error:
                raise Error(
                    "Hamiltonian label commit and rollback both failed"
                ) from rollback_error
            raise Error(str(exc)) from exc

    def _call_custom_apply(self, label):
        if self._apply_label(label) is False:
            raise Error("Hamiltonian apply_label returned False")

    def _abort(self):
        if self.size > 1 and hasattr(self.comm, "Abort"):
            try:
                self.comm.Abort(1)
            except Exception:
                pass


def ph_bias_delta(state_i, state_j):
    """Return the dimensionless pH-label exchange energy difference."""
    _validate_ph_pair(state_i, state_j)
    return _ph_local_delta(state_i, state_j.label) + _ph_local_delta(
        state_j,
        state_i.label,
    )


def ph_bias_probability(state_i, state_j):
    """Return the pH-label Metropolis probability."""
    return metropolis_probability(ph_bias_delta(state_i, state_j))


def metropolis_probability(delta):
    """Return ``min(1, exp(-delta))`` without positive-overflow risk."""
    delta = _finite_float(delta, "delta")
    return 1.0 if delta <= 0.0 else math.exp(-delta)


def _ph_local_delta(state, partner_label):
    return float(
        np.dot(
            state.label.biases - partner_label.biases,
            state.observation.lambdas,
        )
        / (KBOLTZ * state.observation.temperature)
    )


def _validate_ph_pair(state_i, state_j):
    for state in (state_i, state_j):
        if not isinstance(state, State):
            raise Error("replica communicator returned an invalid pH state")
    _validate_ph_protocol((state_i, state_j))


def _validate_ph_protocol(states):
    if not states:
        raise Error("replica communicator returned no pH states")
    reference = states[0].observation
    for state in states:
        if not isinstance(state.label, PhLabel):
            raise Error("replica communicator returned an invalid pH label")
        if not isinstance(state.observation, PhObservation):
            raise Error("replica communicator returned invalid pH observables")
        if state.label.biases.size != state.observation.lambdas.size:
            raise Error("pH bias and lambda arrays have different lengths")
        protocol_error = _ph_protocol_error(reference, state.observation)
        if protocol_error:
            raise Error(protocol_error)


def _ph_protocol_error(left, right):
    if left.lambdas.size != right.lambdas.size:
        return "replicas have different BLOCK counts"
    if abs(left.temperature - right.temperature) > 1.0e-10:
        return "pH label exchange requires equal lambda temperatures"
    for name in ("masses", "frictions"):
        if not np.allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return "replicas have different {}".format(name)
    for name in ("fixed", "sites"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return "replicas have different {}".format(name)
    return None


def _require_current_ph_label(actual, expected):
    if (
        actual.label != expected.label
        or abs(actual.ph - expected.ph) > 1.0e-12
        or not np.allclose(
            actual.biases,
            expected.biases,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise Error("local CHARMM pH label changed outside ReplicaExchange")


def _validate_labels(kind, labels):
    if kind == "ph":
        if any(not isinstance(label, PhLabel) for label in labels):
            raise Error("replica communicator returned an invalid pH label")
        identities = [label.label for label in labels]
        values = [label.ph for label in labels]
        if len(set(identities)) != len(identities):
            raise Error("replica bookkeeping labels must be unique")
        if len(set(values)) != len(values):
            raise Error("replica pH labels must be unique")
        return

    try:
        unique = len(set(labels))
    except TypeError as exc:
        raise TypeError("replica labels must be hashable") from exc
    if unique != len(labels):
        raise Error("replica labels must be unique")


def _neighbor_pairs(labels, attempt_index, key):
    try:
        ladder = sorted(range(len(labels)), key=lambda rank: key(labels[rank]))
    except TypeError as exc:
        raise Error("replica labels must be orderable") from exc
    if len(ladder) == 2:
        return [(ladder[0], ladder[1])]
    start = attempt_index % 2
    return [
        (ladder[position], ladder[position + 1])
        for position in range(start, len(ladder) - 1, 2)
    ]


def _partner_for_rank(rank, pairs):
    for left, right in pairs:
        if rank == left:
            return right
        if rank == right:
            return left
    return None


def _public_label(kind, label):
    return label.ph if kind == "ph" else label


def _label_token(kind, label):
    return label.label if kind == "ph" else label


def _pair_random(seed, attempt_index, rank, partner):
    low, high = sorted((int(rank), int(partner)))
    key = "{}:{}:{}:{}".format(seed, attempt_index, low, high)
    return random.Random(key).random()


def _current_potential_energy():
    return energy.get_eprop(energy.PROP_ENER)


def _default_comm():
    implied_size = _implied_mpi_size_from_environment()
    try:
        from mpi4py import MPI
    except Exception as exc:
        if implied_size is not None and implied_size > 1:
            raise Error(
                "mpi4py is required for replica exchange in a multi-rank "
                "environment; detected {} ranks but mpi4py import failed: "
                "{}".format(implied_size, exc)
            )
        return _SerialComm()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    if implied_size is not None and world_size != implied_size:
        raise Error(
            "mpi4py initialized a {}-rank MPI world, but the active launch "
            "environment implies {} ranks".format(world_size, implied_size)
        )
    return comm


def _implied_mpi_size_from_environment(environ=None):
    environ = os.environ if environ is None else environ
    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_STEP_NUM_TASKS",
    ):
        try:
            size = int(environ[name])
        except (KeyError, TypeError, ValueError):
            continue
        if size > 0:
            return size
    if "SLURM_PROCID" in environ:
        try:
            size = int(environ["SLURM_NTASKS"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if size > 0:
                return size
    return None


class _SerialComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def allgather(self, value):
        return [value]


def _finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _positive_float(value, name):
    value = _finite_float(value, name)
    if value <= 0.0:
        raise ValueError("{} must be positive".format(name))
    return value


def _float_vector(values, name):
    result = _vector(values, np.float64, name)
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must contain only finite values".format(name))
    return result


def _vector(values, dtype, name):
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1 or result.size < 1:
        raise ValueError("{} must be a non-empty one-dimensional array".format(name))
    return result.copy()
