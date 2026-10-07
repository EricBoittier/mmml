"""Gas-phase monomer potential energy <E_gas> for Delta H_vap, batched.

Runs ``n_copies`` independent Langevin (BAOAB) trajectories of one molecule at
once through any evaluator with ``evaluate(structures) -> (E, F)`` in
kcal/mol and kcal/mol/A (:class:`mmml.distill.box_cohesion.PhysNetPairEvaluator`)
or a metatomic teacher wrapped by :class:`TeacherKcal`. Returns per-copy mean
potential energies after ``equil_steps`` so a mean +- std over copies is
available without block analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mmml.models.mm_nonbonded_tune import EV_TO_KCAL

KB_KCAL = 0.0019872043  # kcal/mol/K
# 1 amu*A^2/fs^2 = 2390.057 kcal/mol
AMU_A2_FS2_TO_KCAL = 2390.0573
_MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}


class TeacherKcal:
    """Adapter: BatchedMetatomicTeacher (eV) -> ``evaluate`` returning kcal/mol arrays."""

    def __init__(self, teacher) -> None:
        self.teacher = teacher

    def evaluate(self, structures):
        res = self.teacher.evaluate(structures)
        return (
            np.array([r[0] for r in res]) * EV_TO_KCAL,
            [np.asarray(r[1]) * EV_TO_KCAL for r in res],
        )


@dataclass
class GasResult:
    per_copy_mean: np.ndarray  # (n_copies,) kcal/mol
    series: np.ndarray  # (n_samples, n_copies)
    T_mean: float

    @property
    def mean(self) -> float:
        return float(self.per_copy_mean.mean())

    @property
    def sem(self) -> float:
        n = len(self.per_copy_mean)
        return float(self.per_copy_mean.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def langevin_gas_md(
    evaluator,
    numbers: np.ndarray,
    positions: np.ndarray,
    *,
    temperature_K: float = 300.0,
    dt_fs: float = 0.5,
    friction_per_fs: float = 0.01,
    n_copies: int = 32,
    n_steps: int = 4000,
    equil_steps: int = 1000,
    sample_every: int = 10,
    seed: int = 0,
) -> GasResult:
    """BAOAB Langevin for ``n_copies`` independent monomers (same start, random velocities)."""
    rng = np.random.default_rng(seed)
    z = np.asarray(numbers)
    m = np.array([_MASS[int(a)] for a in z])[None, :, None]  # amu
    x = np.repeat(np.asarray(positions, dtype=np.float64)[None], n_copies, axis=0)
    kT = KB_KCAL * temperature_K
    sig_v = np.sqrt(kT / (m * AMU_A2_FS2_TO_KCAL))  # A/fs
    v = rng.normal(size=x.shape) * sig_v
    v -= (v * m).sum(1, keepdims=True) / m.sum()

    def forces(xx) -> tuple[np.ndarray, np.ndarray]:
        e, f = evaluator.evaluate([(z, xx[k]) for k in range(len(xx))])
        return np.asarray(e), np.stack(f)

    e, f = forces(x)
    c1 = np.exp(-friction_per_fs * dt_fs)
    c2 = np.sqrt(1.0 - c1 * c1)
    samples, temps = [], []
    acc = f / (m * AMU_A2_FS2_TO_KCAL)
    for step in range(1, n_steps + 1):
        v += 0.5 * dt_fs * acc
        x += 0.5 * dt_fs * v
        v = c1 * v + c2 * sig_v * rng.normal(size=v.shape)
        x += 0.5 * dt_fs * v
        e, f = forces(x)
        acc = f / (m * AMU_A2_FS2_TO_KCAL)
        v += 0.5 * dt_fs * acc
        if step > equil_steps and step % sample_every == 0:
            samples.append(e.copy())
            ke = 0.5 * (m * v * v).sum(axis=(1, 2)) * AMU_A2_FS2_TO_KCAL
            temps.append(2.0 * ke / (3 * len(z) * KB_KCAL))
    series = np.array(samples)
    return GasResult(series.mean(0), series, float(np.mean(temps)))


def delta_hvap(e_gas: float, e_liq_per_mol: float, temperature_K: float = 298.15) -> float:
    """``<E_gas> - <E_liq>/N + RT`` (kcal/mol); potential energies, same reference."""
    return float(e_gas - e_liq_per_mol + KB_KCAL * temperature_K)


def block_mean_std(x: Sequence[float], n_blocks: int = 5) -> tuple[float, float]:
    """Mean and std of block means (``n_blocks`` contiguous blocks)."""
    x = np.asarray(x, dtype=np.float64)
    blocks = [b.mean() for b in np.array_split(x, n_blocks) if len(b)]
    return float(x.mean()), float(np.std(blocks, ddof=1)) if len(blocks) > 1 else 0.0
