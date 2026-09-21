"""Gas-phase reference <E_gas>(T) for dHvap, from single-molecule Langevin MD.

In the hybrid, one isolated molecule sees only the PhysNet monomer term (no MM,
no switched dimer term), so <E_gas> does not depend on the fitted theta and is
sampled once per temperature.

Dynamics: BAOAB Langevin (Leimkuhler-Matthews) on ``n_replicas`` independent
copies of the molecule, vmapped and run under ``lax.scan`` (cheap on CPU for
~10 atoms). All 3N degrees of freedom are thermostatted (translation and
rotation included), so <E_kin> = 3N/2 kT; only <E_pot> carries model
information. Units: positions A, masses amu, time fs, energies kcal/mol.

Statistical error: block averaging (Flyvbjerg-Petersen halving) per replica,
combined over independent replicas, and the spread of the replica means. The
reported ``e_pot_sem`` is the larger of the two: Jonsson's blocking estimate
is biased low at the default series length (~4000 records: ~0.85x the true
SEM at lag-1 correlation 0.9, ~0.63x at 0.99), and it cannot see replicas
that sample different basins; the between-replica SEM catches both.

Chemistry check, evaluated at every MD step (equilibration and production,
as running extrema in the scan carry, not only at recorded frames): a replica
has fallen into a PES hole or dissociated if (a) its covalent-bond graph
changes, (b) a bonded pair compresses below ``bond_shrink_fraction`` x its
starting length d0 (e.g. C-H 1.09 -> < 0.87 A; a collapsed bond keeps the
graph unchanged), (c) a non-bonded pair comes closer than
``bond_shrink_fraction`` x the sum of covalent radii ((a)-(c) need atomic
numbers), or (d) any pair falls below the absolute ``min_distance_floor_A``.
The Langevin thermostat removes the released heat, so the kinetic-temperature
test alone does not catch it. Such replicas are dropped by default
(``topology_preserved=False`` and a warning); note that the remaining average
is then conditioned on staying intact (a bond guard,
:func:`flat_bottom_bond_restraint`, is the consistent remedy).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mmml.fit.reweight import KB_KCAL_MOL_K

KCAL_PER_EV = 23.060548
# (kcal/mol/A) / amu -> A/fs^2; also (kcal/mol)/amu -> (A/fs)^2.
ACCEL_KCAL_A_AMU_TO_A_FS2 = 4.184e-4

EnergyFn = Callable[[jnp.ndarray], jnp.ndarray]  # (n_atoms, 3) A -> scalar kcal/mol


# --------------------------------------------------------------------------
# dHvap
# --------------------------------------------------------------------------


def dhvap_kcal_mol(
    e_gas_mean: float | np.ndarray,
    e_liq_per_molecule_mean: float | np.ndarray,
    temperature_K: float | np.ndarray,
) -> float | np.ndarray:
    """dHvap = <E_gas> - <E_liq>/N + R T (kcal/mol).

    Ideal-gas vapour (P V_gas = R T per mole) with the liquid P V_liq term
    neglected (~1e-3 kcal/mol at 1 bar). Both energies must be the same kind:
    potential energies (kinetic terms cancel classically at equal T when
    every molecule keeps all 3N degrees of freedom), or both totals.
    """
    return (
        np.asarray(e_gas_mean, dtype=float)
        - np.asarray(e_liq_per_molecule_mean, dtype=float)
        + KB_KCAL_MOL_K * np.asarray(temperature_K, dtype=float)
    )


def dhvap_from_cohesive_kcal_mol(
    u_inter_per_molecule_mean: float | np.ndarray,
    temperature_K: float | np.ndarray,
    delta_e_intra: float | np.ndarray,
) -> float | np.ndarray:
    """dHvap = -<U_inter>/N + R T + dE_intra (kcal/mol).

    Equivalent to :func:`dhvap_kcal_mol` with E_liq/N = E_intra,liq + U_inter/N
    and ``delta_e_intra`` = <E_intra>_gas - <E_intra>_liq per molecule
    (conformational change on condensation; ~0 for a rigid molecule, but the
    liquid monomer term sees the same PES holes as the gas, so it is required
    rather than defaulted: pass 0.0 explicitly to neglect it).

    ``u_inter_per_molecule_mean`` is the liquid energy minus all monomer terms,
    per molecule, in kcal/mol. For the hybrid calculator that is
    (ml_2b_E + mm_E + wall_E) / N, where wall_E is the intermolecular
    short-range repulsive wall (``short_range_wall=True``); the hybrid reports
    these in eV, so multiply by :data:`KCAL_PER_EV`. Use this route when the
    monomer model has no stationary gas-phase ensemble (PES holes).
    """
    return (
        -np.asarray(u_inter_per_molecule_mean, dtype=float)
        + KB_KCAL_MOL_K * np.asarray(temperature_K, dtype=float)
        + np.asarray(delta_e_intra, dtype=float)
    )


# --------------------------------------------------------------------------
# Block averaging
# --------------------------------------------------------------------------


def block_average(x: np.ndarray, n_blocks: int) -> tuple[float, float]:
    """Mean and standard error from ``n_blocks`` contiguous equal blocks.

    Leftover samples that do not fill a block are dropped from the start of
    the series, so the most equilibrated data is kept.
    """
    x = np.asarray(x, dtype=float).ravel()
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    size = x.size // n_blocks
    if size < 1:
        raise ValueError(f"{x.size} samples cannot fill {n_blocks} blocks")
    xb = x[x.size - size * n_blocks :].reshape(n_blocks, size).mean(axis=1)
    return float(xb.mean()), float(xb.std(ddof=1) / np.sqrt(n_blocks))


@dataclass(frozen=True)
class BlockingResult:
    """Flyvbjerg-Petersen blocking of one time series."""

    mean: float
    sem: float  # adopted standard error of the mean
    block_sizes: np.ndarray  # samples per block at each halving level
    sems: np.ndarray  # naive SEM at each level
    sem_errors: np.ndarray  # uncertainty of each SEM estimate
    plateau: bool  # True if an uncorrelated level was found (else sem = max over levels)

    @property
    def statistical_inefficiency(self) -> float:
        """g = (sem / naive_sem)^2 ~ 2 tau_int / sampling interval."""
        return float((self.sem / self.sems[0]) ** 2) if self.sems[0] > 0 else 1.0


def blocking_analysis(x: np.ndarray, min_blocks: int = 16) -> BlockingResult:
    """Flyvbjerg-Petersen blocking with the automatic level choice of Jonsson.

    Level k averages pairs of level k-1 (block size 2^k; a leading sample is
    dropped when the count is odd). SEM_k = sqrt(var_k / n_k) (ddof=0 as in
    Jonsson), error SEM_k / sqrt(2 (n_k - 1)). The adopted level is the first
    k with M_k = sum_{i>=k} n_i ((n_i-1) var_i / n_i^2 + gamma_i)^2 / var_i^2
    below the 99% chi^2 quantile with (levels - k) dof (gamma_i: lag-1
    autocovariance; M. Jonsson, Phys. Rev. E 98, 043304, 2018), i.e. where
    the blocks are statistically uncorrelated. If no such level keeps
    >= ``min_blocks`` blocks, SEM = max over those levels (``plateau=False``).
    """
    from scipy.stats import chi2

    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 * min_blocks:
        raise ValueError(f"need >= {2 * min_blocks} samples, got {x.size}")
    mean = float(x.mean())
    y = x.copy()
    ns, vars_, gammas = [], [], []
    while y.size >= 2:
        d = y - y.mean()
        ns.append(y.size)
        vars_.append(float(np.mean(d**2)))
        gammas.append(float(np.sum(d[:-1] * d[1:]) / y.size))
        if y.size % 2:
            y = y[1:]
        y = 0.5 * (y[0::2] + y[1::2])
    n_a, v_a, g_a = np.asarray(ns, float), np.asarray(vars_), np.asarray(gammas)
    g_over_v = np.divide(g_a, v_a, out=np.zeros_like(g_a), where=v_a > 0)
    terms = n_a * ((n_a - 1.0) / n_a**2 + g_over_v) ** 2
    m_k = np.cumsum(terms[::-1])[::-1]
    n_levels = n_a.size
    q = chi2.ppf(0.99, np.arange(n_levels, 0, -1))
    keep = n_a >= min_blocks
    sems_all = np.sqrt(v_a / n_a)
    errs_all = sems_all / np.sqrt(2.0 * np.maximum(n_a - 1.0, 1.0))
    below = np.nonzero(m_k < q)[0]
    if below.size and keep[below[0]]:
        adopted, plateau = float(sems_all[below[0]]), True
    else:
        adopted, plateau = float(sems_all[keep].max()), False
    return BlockingResult(
        mean=mean,
        sem=adopted,
        block_sizes=2 ** np.arange(int(keep.sum())),
        sems=sems_all[keep],
        sem_errors=errs_all[keep],
        plateau=plateau,
    )


def replica_mean_sem(series: np.ndarray, min_blocks: int = 16) -> tuple[float, float, float]:
    """Combine independent replicas ``series`` (R, T).

    Returns (mean, sem_blocking, sem_between_replicas): the first SEM combines
    per-replica blocking SEMs (sqrt(sum sem_r^2) / R); the second is the
    spread of replica means (std / sqrt(R); nan for R = 1). Blocking is biased
    low for short series (see module docstring); report
    :func:`combined_sem` of the two.
    """
    series = np.atleast_2d(np.asarray(series, dtype=float))
    n_rep = series.shape[0]
    sems = np.asarray([blocking_analysis(s, min_blocks).sem for s in series])
    means = series.mean(axis=1)
    between = float(means.std(ddof=1) / np.sqrt(n_rep)) if n_rep > 1 else float("nan")
    return float(means.mean()), float(np.sqrt(np.sum(sems**2)) / n_rep), between


def combined_sem(sem_blocking: float, sem_between: float) -> float:
    """Conservative SEM: max(blocking, between-replica); blocking alone if R = 1."""
    if not np.isfinite(sem_between):
        return float(sem_blocking)
    return float(max(sem_blocking, sem_between))


# --------------------------------------------------------------------------
# Langevin MD
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LangevinConfig:
    """BAOAB settings. Sampling time per replica = n_steps * dt_fs."""

    dt_fs: float = 0.5
    friction_per_fs: float = 0.01  # gamma (1/fs); 0.01 = 10 ps^-1
    n_equil_steps: int = 4000
    n_steps: int = 40000
    record_every: int = 10
    n_replicas: int = 8
    seed: int = 0
    # Replicas whose mean kinetic temperature exceeds (1 + tol) T are treated as
    # blown up (integration failure / PES hole) and excluded from averages.
    unstable_temperature_tol: float = 0.5
    # Replicas whose smallest interatomic distance ever drops below this (A) are
    # treated as fused (PES hole); None disables. 0.7 A is below any real bond.
    min_distance_floor_A: float | None = 0.7
    # With atomic numbers: flag a replica if a bonded pair ever gets shorter than
    # this x its starting length, or a non-bonded pair shorter than this x the sum
    # of covalent radii (C-H 1.09 A -> 0.87 A; ~5 sigma thermal at 500 K). None disables.
    bond_shrink_fraction: float | None = 0.8
    # Covalent-graph scale for the bond-change test (x sum of covalent radii).
    bond_scale: float = 1.2
    # Drop replicas whose topology changed (else keep them, flagged).
    drop_topology_changes: bool = True


@dataclass
class GasPhaseResult:
    """Per-temperature gas-phase averages (kcal/mol per molecule)."""

    temperature_K: float
    e_pot_mean: float
    e_pot_sem: float  # max(e_pot_sem_blocking, e_pot_sem_between); use this one
    e_pot_sem_blocking: float  # combined per-replica blocking SEM (biased low)
    e_pot_sem_between: float  # SEM from spread of replica means (nan if R = 1)
    e_kin_mean: float
    e_kin_expected: float  # 3N/2 kT
    temperature_kin_K: float  # 2 <E_kin> / (3N k)
    n_samples: int  # per replica
    sampled_ps: float  # per replica
    n_replicas_used: int = 0
    # Integration failures only (non-finite or T_kin too high), whether or not dropped.
    unstable_replicas: list[int] = field(default_factory=list)
    # Every replica excluded from the averages (unstable or topology change).
    dropped_replicas: list[int] = field(default_factory=list)
    min_pair_distance_A: list[float] = field(default_factory=list)  # per replica, every step
    # Per replica, every step: min over pairs of d_ij / d_ref_ij (d_ref: starting
    # length if bonded, sum of covalent radii if not); nan without atomic numbers.
    min_distance_ratio: list[float] = field(default_factory=list)
    # False if any replica changed its bond graph or fell below a distance floor.
    topology_preserved: bool = True
    topology_changed_replicas: list[int] = field(default_factory=list)
    bond_graph_checked: bool = False  # covalent-graph test ran (atomic numbers given)
    restraint_energy_mean: float = 0.0  # <E_restraint> (not in e_pot), kcal/mol
    restraint_active_fraction: float = 0.0  # records with E_restraint > 1e-6
    config: LangevinConfig = field(default_factory=LangevinConfig)
    e_pot_series: np.ndarray | None = None  # (R, T) if kept

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k not in ("config", "e_pot_series")}
        out["config"] = dict(self.config.__dict__)
        return out


def _kinetic_kcal(v: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(masses[:, None] * v**2) / ACCEL_KCAL_A_AMU_TO_A_FS2


def make_langevin_runner(
    energy_fn: EnergyFn,
    masses_amu: np.ndarray,
    config: LangevinConfig,
    restraint_fn: EnergyFn | None = None,
    bond_cutoff_A: np.ndarray | None = None,
    reference_adjacency: np.ndarray | None = None,
    reference_distance_A: np.ndarray | None = None,
) -> Callable[..., tuple[Any, ...]]:
    """Jitted ``run(x, v, key, kT, n_records) -> ((x, v, key), e_pot, e_kin, d_min, e_rst, n_bond, ratio)``.

    ``x``, ``v``: (R, n_atoms, 3); records are (n_records, R). ``e_pot``,
    ``e_kin``, ``e_rst`` are instantaneous values every ``config.record_every``
    steps; ``d_min``, ``n_bond``, ``ratio`` are extrema over *every* step of
    the record interval. ``kT`` in kcal/mol. ``restraint_fn`` (optional) adds
    a bias force; its energy ``e_rst`` is not part of ``e_pot``.
    ``d_min``: smallest interatomic distance (A). ``n_bond``: max number of
    atom pairs whose bonded state (d < ``bond_cutoff_A``, (n, n)) differs from
    ``reference_adjacency`` (n, n bool); zeros if unset. ``ratio``: min over
    pairs of d_ij / ``reference_distance_A`` (n, n); inf if unset.
    """
    m = jnp.asarray(masses_amu, dtype=jnp.float32)
    dt = float(config.dt_fs)
    c1 = float(np.exp(-config.friction_per_fs * dt))
    c2 = float(np.sqrt(1.0 - c1**2))
    k = ACCEL_KCAL_A_AMU_TO_A_FS2
    if (bond_cutoff_A is None) != (reference_adjacency is None):
        raise ValueError("bond_cutoff_A and reference_adjacency must be given together")
    if bond_cutoff_A is not None:
        cut = jnp.asarray(bond_cutoff_A, dtype=jnp.float32)
        ref = jnp.asarray(np.triu(np.asarray(reference_adjacency, dtype=bool), k=1))
        upper = jnp.asarray(np.triu(np.ones(cut.shape, dtype=bool), k=1))

        def _bond_changes(d):
            bonded = (d < cut) & upper
            return jnp.sum(bonded != ref).astype(jnp.int32)
    else:

        def _bond_changes(d):
            return jnp.zeros((), jnp.int32)

    n_atoms = m.shape[0]
    if reference_distance_A is not None:
        inv_ref = jnp.asarray(1.0 / np.asarray(reference_distance_A, dtype=np.float64), jnp.float32)
    else:
        inv_ref = jnp.zeros((n_atoms, n_atoms), jnp.float32)
    off_diag = jnp.asarray(1e12 * np.eye(n_atoms), jnp.float32)

    def _geometry(x):
        """(d_min, n_bond_changes, min d/d_ref) of one replica."""
        d = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1) + off_diag)
        ratio = jnp.where(inv_ref > 0, d * inv_ref, jnp.inf)
        return jnp.min(d), _bond_changes(d), jnp.min(ratio)

    _geom = jax.vmap(_geometry)

    def _update(stats, x):
        dmin, nb, rt = _geom(x)
        s_d, s_nb, s_rt = stats
        # fmin/fmax would hide NaN; propagate it (a NaN replica is flagged anyway).
        return (
            jnp.where(jnp.isnan(dmin) | (dmin < s_d), dmin, s_d),
            jnp.maximum(nb, s_nb),
            jnp.where(jnp.isnan(rt) | (rt < s_rt), rt, s_rt),
        )

    def _total(x):
        e = energy_fn(x)
        er = restraint_fn(x) if restraint_fn is not None else jnp.zeros_like(e)
        return e + er, (e, er)

    _vg = jax.vmap(jax.value_and_grad(_total, has_aux=True))

    def e_and_g(x):
        (_, (e, er)), g = _vg(x)
        return (e, er), g

    def _fresh_stats(r, dtype):
        return (jnp.full((r,), jnp.inf, dtype), jnp.zeros((r,), jnp.int32), jnp.full((r,), jnp.inf, dtype))

    def step(carry, _):
        x, v, a, e, key, kT, stats = carry
        key, sub = jax.random.split(key)
        v = v + 0.5 * dt * a
        x = x + 0.5 * dt * v
        sigma = jnp.sqrt(kT * k / m)[None, :, None]
        v = c1 * v + c2 * sigma * jax.random.normal(sub, v.shape, dtype=v.dtype)
        x = x + 0.5 * dt * v
        e, g = e_and_g(x)
        a = -g * k / m[None, :, None]
        v = v + 0.5 * dt * a
        stats = _update(stats, x)
        return (x, v, a, e, key, kT, stats), None

    def record(carry, _):
        carry = (*carry[:6], _fresh_stats(carry[0].shape[0], carry[0].dtype))  # per-interval extrema
        carry, _ = jax.lax.scan(step, carry, None, length=config.record_every)
        x, v, _, (e, er), _, _, (d_min, n_bond, ratio) = carry
        e_kin = jax.vmap(_kinetic_kcal, in_axes=(0, None))(v, m)
        return carry, (e, e_kin, d_min, er, n_bond, ratio)

    @jax.jit
    def _run(x, v, key, kT, n_records_arr):
        e, g = e_and_g(x)
        a = -g * k / m[None, :, None]
        carry = (x, v, a, e, key, kT, _fresh_stats(x.shape[0], x.dtype))
        carry, (e_pot, e_kin, d_min, e_rst, n_bond, ratio) = jax.lax.scan(record, carry, n_records_arr)
        x, v, _, _, key, _, _ = carry
        return (x, v, key), e_pot, e_kin, d_min, e_rst, n_bond, ratio

    def run(x, v, key, kT, n_records: int):
        return _run(x, v, key, jnp.asarray(kT, jnp.float32), jnp.zeros(int(n_records)))

    return run


def run_gas_phase(
    energy_fn: EnergyFn,
    positions_A: np.ndarray,
    masses_amu: np.ndarray,
    temperatures_K: Sequence[float],
    config: LangevinConfig | None = None,
    *,
    min_blocks: int = 16,
    keep_series: bool = False,
    restraint_fn: EnergyFn | None = None,
    atomic_numbers: Sequence[int] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[GasPhaseResult]:
    """Sample <E_pot>, <E_kin> of one molecule at each temperature.

    ``energy_fn`` maps (n_atoms, 3) A to a scalar kcal/mol (jax-traceable).
    Replicas start from ``positions_A`` with Maxwell-Boltzmann velocities.
    Replicas that blow up (non-finite, or mean kinetic temperature above
    ``(1 + config.unstable_temperature_tol) T``) are dropped and listed in
    ``unstable_replicas``; if all do, ``FloatingPointError`` is raised.

    Replicas that, at any MD step (equilibration included), change their
    covalent graph (from ``atomic_numbers``, scale ``config.bond_scale``),
    compress a bonded pair below ``config.bond_shrink_fraction`` x its length
    in ``positions_A`` (or a non-bonded pair below that fraction x the sum of
    covalent radii), or bring any pair closer than
    ``config.min_distance_floor_A``, are listed in
    ``topology_changed_replicas`` (``topology_preserved=False``, with a
    warning) and, if ``config.drop_topology_changes``, excluded; if none
    remain, ``RuntimeError`` is raised. All excluded replicas are listed in
    ``dropped_replicas``.
    """
    cfg = config or LangevinConfig()
    x0 = np.asarray(positions_A, dtype=np.float32)
    m = np.asarray(masses_amu, dtype=np.float64)
    n_atoms = x0.shape[0]
    cut = ref_adj = ref_dist = None
    if atomic_numbers is not None:
        from ase.data import covalent_radii

        r = covalent_radii[np.asarray(atomic_numbers, dtype=int)]
        if r.shape[0] != n_atoms:
            raise ValueError(f"{r.shape[0]} atomic numbers for {n_atoms} atoms")
        cut = cfg.bond_scale * (r[:, None] + r[None, :])
        ref_adj = np.zeros((n_atoms, n_atoms), dtype=bool)
        pairs = bonded_pairs(x0, atomic_numbers, scale=cfg.bond_scale)
        ref_adj[pairs[:, 0], pairs[:, 1]] = True
        if cfg.bond_shrink_fraction is not None:
            d0 = np.linalg.norm(x0[:, None].astype(np.float64) - x0[None], axis=-1)
            adj = ref_adj | ref_adj.T
            ref_dist = np.where(adj, d0, r[:, None] + r[None, :])
            np.fill_diagonal(ref_dist, np.inf)
    run = make_langevin_runner(energy_fn, m, cfg, restraint_fn, cut, ref_adj, ref_dist)
    key = jax.random.PRNGKey(cfg.seed)
    results = []
    for T in temperatures_K:
        kT = KB_KCAL_MOL_K * float(T)
        key, kv = jax.random.split(key)
        x = jnp.broadcast_to(jnp.asarray(x0), (cfg.n_replicas, n_atoms, 3))
        sig = np.sqrt(kT * ACCEL_KCAL_A_AMU_TO_A_FS2 / m)[None, :, None]
        v = jax.random.normal(kv, x.shape, dtype=jnp.float32) * jnp.asarray(sig, jnp.float32)
        n_eq = max(1, cfg.n_equil_steps // cfg.record_every)
        (x, v, key), _, _, d_min_eq, _, n_bond_eq, ratio_eq = run(x, v, key, kT, n_eq)
        n_rec = cfg.n_steps // cfg.record_every
        (x, v, key), e_pot, e_kin, d_min, e_rst, n_bond, ratio = run(x, v, key, kT, n_rec)
        # Geometry extrema cover every step of equilibration and production.
        n_bond = np.concatenate([np.asarray(n_bond_eq), np.asarray(n_bond)]).T
        d_min = np.concatenate([np.asarray(d_min_eq), np.asarray(d_min)])
        ratio = np.concatenate([np.asarray(ratio_eq), np.asarray(ratio)])
        e_rst = np.asarray(e_rst, dtype=np.float64).T
        e_pot = np.asarray(e_pot, dtype=np.float64).T  # (R, n_rec)
        e_kin = np.asarray(e_kin, dtype=np.float64).T
        d_min = np.asarray(d_min, dtype=np.float64).T
        ratio_rep = np.min(np.asarray(ratio, dtype=np.float64).T, axis=1)  # NaN propagates
        t_kin_rep = 2.0 * e_kin.mean(axis=1) / (3.0 * n_atoms * KB_KCAL_MOL_K)
        ok = (
            np.all(np.isfinite(e_pot), axis=1)
            & np.all(np.isfinite(e_kin), axis=1)
            & (t_kin_rep <= (1.0 + cfg.unstable_temperature_tol) * float(T))
        )
        if not ok.any():
            raise FloatingPointError(f"all replicas unstable at T={T} K (reduce dt_fs?)")
        stable = ok.copy()
        d_min_rep = np.min(d_min, axis=1)  # NaN propagates
        broken = np.any(n_bond > 0, axis=1)
        if cfg.min_distance_floor_A is not None:
            broken |= ~(d_min_rep >= cfg.min_distance_floor_A)  # NaN counts as broken
        if ref_dist is not None:
            broken |= ~(ratio_rep >= cfg.bond_shrink_fraction)
        broken_ids = [int(i) for i in np.nonzero(broken)[0]]
        if broken_ids:
            action = "dropped" if cfg.drop_topology_changes else "kept"
            warnings.warn(
                f"T={T} K: replicas {broken_ids} changed covalent topology, compressed a pair "
                f"below {cfg.bond_shrink_fraction} x its reference length, or came closer "
                f"than {cfg.min_distance_floor_A} A (PES hole / dissociation); {action}",
                RuntimeWarning,
                stacklevel=2,
            )
            if cfg.drop_topology_changes:
                ok &= ~broken
                if not ok.any():
                    raise RuntimeError(f"no replica kept its topology at T={T} K (add a bond guard?)")
        mean, sem_blk, sem_between = replica_mean_sem(e_pot[ok], min_blocks)
        sem = combined_sem(sem_blk, sem_between)
        e_kin_mean = float(e_kin[ok].mean())
        res = GasPhaseResult(
            temperature_K=float(T),
            e_pot_mean=mean,
            e_pot_sem=sem,
            e_pot_sem_blocking=sem_blk,
            e_pot_sem_between=sem_between,
            e_kin_mean=e_kin_mean,
            e_kin_expected=float(1.5 * n_atoms * kT),
            temperature_kin_K=2.0 * e_kin_mean / (3.0 * n_atoms * KB_KCAL_MOL_K),
            n_samples=int(n_rec),
            sampled_ps=n_rec * cfg.record_every * cfg.dt_fs / 1000.0,
            n_replicas_used=int(ok.sum()),
            unstable_replicas=[int(i) for i in np.nonzero(~stable)[0]],
            dropped_replicas=[int(i) for i in np.nonzero(~ok)[0]],
            min_pair_distance_A=[float(d) for d in d_min_rep],
            min_distance_ratio=[float(q) if ref_dist is not None else float("nan") for q in ratio_rep],
            topology_preserved=not broken_ids,
            topology_changed_replicas=broken_ids,
            bond_graph_checked=atomic_numbers is not None,
            restraint_energy_mean=float(e_rst[ok].mean()),
            restraint_active_fraction=float((e_rst[ok] > 1e-6).mean()),
            config=cfg,
            e_pot_series=e_pot[ok] if keep_series else None,
        )
        if log is not None:
            log(
                f"T={T:.1f} K  <E_pot>={mean:.4f} +- {sem:.4f} "
                f"(blocking {sem_blk:.4f}, between {sem_between:.4f}) "
                f"kcal/mol  T_kin={res.temperature_kin_K:.1f} K  "
                f"replicas {res.n_replicas_used}/{cfg.n_replicas}  "
                f"min d={np.min(res.min_pair_distance_A):.3f} A  "
                f"min d/d_ref={np.min(res.min_distance_ratio):.3f}  "
                f"topology changed {broken_ids}"
            )
        results.append(res)
    return results


def bonded_pairs(positions_A: np.ndarray, atomic_numbers: Sequence[int], scale: float = 1.2) -> np.ndarray:
    """(P, 2) atom pairs closer than ``scale`` x sum of ASE covalent radii."""
    from ase.data import covalent_radii

    x = np.asarray(positions_A, dtype=float)
    r = covalent_radii[np.asarray(atomic_numbers, dtype=int)]
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.nonzero(np.triu(d < scale * (r[:, None] + r[None, :]), k=1))
    return np.stack([i, j], axis=1).astype(np.int32)


def flat_bottom_bond_restraint(
    positions_A: np.ndarray,
    pairs: np.ndarray,
    half_width_A: float = 0.2,
    k_kcal_A2: float = 200.0,
) -> EnergyFn:
    """Flat-bottom harmonic guard on bond lengths (kcal/mol).

    E = sum_pairs k/2 max(0, |d - d0| - w)^2 with d0 from ``positions_A``.
    Zero inside the physical basin; keeps a model with spurious short-bond or
    dissociation holes from leaving it. The same guard must be applied in the
    liquid for a consistent dHvap.
    """
    p = jnp.asarray(pairs, dtype=jnp.int32)
    x0 = np.asarray(positions_A, dtype=float)
    d0 = jnp.asarray(np.linalg.norm(x0[pairs[:, 0]] - x0[pairs[:, 1]], axis=-1), jnp.float32)
    w, kk = float(half_width_A), float(k_kcal_A2)

    def energy(x: jnp.ndarray) -> jnp.ndarray:
        d = jnp.linalg.norm(x[p[:, 0]] - x[p[:, 1]], axis=-1)
        excess = jnp.maximum(jnp.abs(d - d0) - w, 0.0)
        return 0.5 * kk * jnp.sum(excess**2)

    return energy


# --------------------------------------------------------------------------
# PhysNet monomer energy
# --------------------------------------------------------------------------


def physnet_monomer_energy_fn(
    checkpoint: str | Path,
    atomic_numbers: Sequence[int],
) -> EnergyFn:
    """Scalar PhysNet energy (kcal/mol) of one isolated molecule, free space.

    Same checkpoint loader as the hybrid MLpot (``internal_E`` monomer term,
    model output in eV); all-pairs message-passing graph, no PBC.
    """
    import e3x

    from mmml.interfaces.calculators.checkpoint_loading import load_physnet_for_hybrid_mlpot

    z = jnp.asarray(np.asarray(atomic_numbers, dtype=np.int32))
    n = int(z.shape[0])
    model, params, _ = load_physnet_for_hybrid_mlpot(checkpoint, max_padded_atoms=n)
    dst, src = e3x.ops.sparse_pairwise_indices(n)
    dst, src = jnp.asarray(dst), jnp.asarray(src)
    seg = jnp.zeros((n,), jnp.int32)
    bmask = jnp.ones_like(dst, dtype=jnp.float32)
    amask = jnp.ones((n,), jnp.float32)

    def energy_fn(positions: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(
            params,
            atomic_numbers=z,
            positions=positions,
            dst_idx=dst,
            src_idx=src,
            batch_segments=seg,
            batch_size=1,
            batch_mask=bmask,
            atom_mask=amask,
            compute_forces=False,
        )
        return jnp.reshape(out["energy"], (-1,))[0] * KCAL_PER_EV

    return energy_fn


def run_physnet_gas_phase(
    checkpoint: str | Path,
    atomic_numbers: Sequence[int],
    positions_A: np.ndarray,
    temperatures_K: Sequence[float],
    config: LangevinConfig | None = None,
    *,
    bond_guard_half_width_A: float | None = None,
    **kwargs: Any,
) -> list[GasPhaseResult]:
    """:func:`run_gas_phase` with the PhysNet monomer model and ASE masses.

    ``bond_guard_half_width_A`` adds :func:`flat_bottom_bond_restraint` on the
    covalent bonds of ``positions_A`` (reported, not included in E_pot).
    """
    from ase.data import atomic_masses

    masses = atomic_masses[np.asarray(atomic_numbers, dtype=int)]
    energy_fn = physnet_monomer_energy_fn(checkpoint, atomic_numbers)
    if bond_guard_half_width_A is not None:
        pairs = bonded_pairs(positions_A, atomic_numbers)
        kwargs["restraint_fn"] = flat_bottom_bond_restraint(positions_A, pairs, half_width_A=bond_guard_half_width_A)
    kwargs.setdefault("atomic_numbers", atomic_numbers)
    return run_gas_phase(energy_fn, positions_A, masses, temperatures_K, config, **kwargs)
