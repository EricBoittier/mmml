"""Cross-monomer jax-pme long-range (one prepare, fused E_full − Σ_m E_masked).

Hybrid MM long-range needs periodic Ewald/PME on **cross-monomer** pairs only.
That equals ``E_full(q) − Σ_m E(q ⊙ mask_m)`` with the same jax-pme split,
but without ``N`` separate ``prepare`` / neighbor-list builds.

Enable with ``MMML_JAX_PME_INTRA_MODE=cross`` (default for ``ewald``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    LongRangeInteractionResult,
    jax_pme_mesh_method,
    materialize_jax_pme_host_numpy,
    resolve_jax_pme_method,
)

_PROFILE: dict[str, list[float]] = {}


def resolve_jax_pme_intra_mode(method: str | None = None) -> str:
    """``cross`` (fused masked subtraction) or ``full_minus_intra`` (legacy loop)."""
    raw = os.environ.get("MMML_JAX_PME_INTRA_MODE", "cross").strip().lower()
    if raw in ("cross", "direct", "structure_factor", "masked"):
        mode = "cross"
    elif raw in ("full_minus_intra", "legacy", "loop", "intra_loop"):
        mode = "full_minus_intra"
    else:
        raise ValueError(
            "MMML_JAX_PME_INTRA_MODE must be cross|full_minus_intra; "
            f"got {raw!r}"
        )
    if mode == "cross" and jax_pme_mesh_method(method):
        return "full_minus_intra"
    return mode


def _profile_enabled() -> bool:
    raw = os.environ.get("MMML_JAX_PME_PROFILE", "").strip().lower()
    return raw in ("1", "true", "yes", "on", "per_call")


def _record(label: str, start: float | None) -> None:
    if start is None:
        return
    _PROFILE.setdefault(label, []).append((time.perf_counter() - start) * 1000.0)


def consume_cross_monomer_profile() -> dict[str, dict[str, float]]:
    """Return and clear accumulated cross-monomer profile samples (ms)."""
    out: dict[str, dict[str, float]] = {}
    for label, samples in sorted(_PROFILE.items()):
        arr = np.asarray(samples, dtype=np.float64)
        out[label] = {
            "n": float(arr.size),
            "total_ms": float(np.sum(arr)),
            "mean_ms": float(np.mean(arr)),
        }
    _PROFILE.clear()
    return out


def monomer_id_from_offsets(total_atoms: int, monomer_offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(monomer_offsets, dtype=np.int64).reshape(-1)
    out = np.empty(int(total_atoms), dtype=np.int32)
    for m in range(int(len(offsets) - 1)):
        out[int(offsets[m]) : int(offsets[m + 1])] = m
    return out


@dataclass(frozen=True)
class _CrossMonomerJitKey:
    method_name: str
    exponent: int
    prefactor: float
    sr_cutoff_A: float
    box_length_A: float
    n_atoms: int
    n_monomers: int


def _build_cross_monomer_jit(key: _CrossMonomerJitKey):
    import jax
    import jax.numpy as jnp
    from jaxpme.kspace import generate_kvectors, get_reciprocal
    from jaxpme.potentials import potential
    from jaxpme.solvers import ewald

    pot = potential(exponent=int(key.exponent))

    def _potentials(charges, cell, positions, i, j, cell_shifts, k_grid, smearing, pbc):
        solver = ewald(pot)
        reciprocal_cell = get_reciprocal(cell)
        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            reciprocal_cell,
            k_grid.shape,
            dtype=positions.dtype,
            for_ewald=True,
        )
        from jaxpme.utils import get_distances

        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(
            smearing, charges, kvectors, positions, volume, cell, pbc
        )
        return float(key.prefactor) * (rspace + kspace)

    def _energy_for_charges(charges, cell, positions, i, j, cell_shifts, k_grid, smearing, pbc):
        phi = _potentials(charges, cell, positions, i, j, cell_shifts, k_grid, smearing, pbc)
        return jnp.sum(charges * phi)

    def _energy_cross(
        positions,
        charges,
        cell,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        pbc,
        monomer_id,
    ):
        base = _energy_for_charges(
            charges, cell, positions, i, j, cell_shifts, k_grid, smearing, pbc
        )
        n_mono = int(key.n_monomers)
        intra = jnp.asarray(0.0, dtype=positions.dtype)
        for m in range(n_mono):
            mask = (monomer_id == m).astype(charges.dtype)
            intra = intra + _energy_for_charges(
                charges * mask,
                cell,
                positions,
                i,
                j,
                cell_shifts,
                k_grid,
                smearing,
                pbc,
            )
        return base - intra

    def _energy_and_forces(
        positions,
        charges,
        cell,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        pbc,
        monomer_id,
    ):
        energy, grads = jax.value_and_grad(_energy_cross, argnums=0)(
            positions,
            charges,
            cell,
            i,
            j,
            cell_shifts,
            k_grid,
            smearing,
            pbc,
            monomer_id,
        )
        return energy, -grads

    return jax.jit(_energy_and_forces)


@lru_cache(maxsize=64)
def _cached_cross_jit(key: _CrossMonomerJitKey):
    return _build_cross_monomer_jit(key)


@dataclass(frozen=True)
class _CrossMonomerHostEvaluator:
    method_name: str
    exponent: int
    prefactor: float
    sr_cutoff_A: float
    box_length_A: float
    n_atoms: int
    n_monomers: int
    monomer_id: tuple[int, ...]

    def compute(
        self,
        positions_A: np.ndarray,
        coefficients: np.ndarray,
    ) -> LongRangeInteractionResult:
        from ase import Atoms
        from jaxpme import Ewald

        import jax.numpy as jnp

        from mmml.interfaces.pycharmmInterface.long_range_backend import (
            jax_pme_host_eval_context,
            jax_pme_pure_callback_host_context,
        )

        pos = np.asarray(positions_A, dtype=np.float64)
        coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if pos.shape != (int(self.n_atoms), 3):
            raise ValueError(f"positions shape {pos.shape} != ({self.n_atoms}, 3)")
        if coef.shape[0] != int(self.n_atoms):
            raise ValueError(f"coefficients length {coef.shape[0]} != {self.n_atoms}")

        atoms = Atoms(
            positions=pos,
            cell=np.eye(3, dtype=np.float64) * float(self.box_length_A),
            pbc=True,
        )
        smearing = float(self.sr_cutoff_A) / 5.0
        lr_wavelength = smearing / 2.0

        calc = Ewald(exponent=int(self.exponent), prefactor=float(self.prefactor))
        charges, cell_j, positions_j, i, j, cell_shifts, k_grid, smearing_v, pbc_v = (
            calc.prepare(
                atoms,
                coef,
                cutoff=float(self.sr_cutoff_A),
                smearing=smearing,
                lr_wavelength=lr_wavelength,
            )
        )
        monomer_id = jnp.asarray(self.monomer_id, dtype=jnp.int32)
        jit_key = _CrossMonomerJitKey(
            self.method_name,
            int(self.exponent),
            float(self.prefactor),
            float(self.sr_cutoff_A),
            float(self.box_length_A),
            int(self.n_atoms),
            int(self.n_monomers),
        )
        energy_forces = _cached_cross_jit(jit_key)

        host_ctx = (
            jax_pme_pure_callback_host_context
            if jax_pme_mesh_method(self.method_name)
            else jax_pme_host_eval_context
        )
        t0 = time.perf_counter() if _profile_enabled() else None
        with host_ctx():
            energy, forces = energy_forces(
                jnp.asarray(positions_j, dtype=jnp.float64),
                jnp.asarray(charges, dtype=jnp.float64),
                jnp.asarray(cell_j, dtype=jnp.float64),
                jnp.asarray(i, dtype=jnp.int32),
                jnp.asarray(j, dtype=jnp.int32),
                jnp.asarray(cell_shifts, dtype=jnp.int32),
                k_grid,
                float(smearing_v),
                jnp.asarray(pbc_v, dtype=bool),
                monomer_id,
            )
        _record("cross_monomer_eval", t0)
        e_host, f_host = materialize_jax_pme_host_numpy(energy, forces)
        return LongRangeInteractionResult(
            energy_kcalmol=e_host,
            forces_kcalmol_A=f_host,
        )


@lru_cache(maxsize=128)
def _cached_cross_evaluator(
    method_name: str,
    exponent: int,
    prefactor: float,
    sr_cutoff_A: float,
    box_length_A: float,
    n_atoms: int,
    monomer_offsets_key: tuple[int, ...],
) -> _CrossMonomerHostEvaluator:
    n_monomers = max(0, len(monomer_offsets_key) - 1)
    monomer_id = monomer_id_from_offsets(int(n_atoms), np.asarray(monomer_offsets_key))
    return _CrossMonomerHostEvaluator(
        method_name=str(method_name),
        exponent=int(exponent),
        prefactor=float(prefactor),
        sr_cutoff_A=float(sr_cutoff_A),
        box_length_A=float(box_length_A),
        n_atoms=int(n_atoms),
        n_monomers=int(n_monomers),
        monomer_id=tuple(int(v) for v in monomer_id),
    )


def compute_jax_pme_cross_monomer_power_law(
    positions_A: np.ndarray,
    coefficients: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str = "ewald",
    sr_cutoff_A: float = 6.0,
    exponent: int = 1,
    prefactor: float,
) -> LongRangeInteractionResult:
    """Cross-monomer periodic 1/r^p: one jax-pme prepare + fused masked subtraction."""
    method_name = resolve_jax_pme_method(str(method))
    if method_name != "ewald":
        raise ValueError(
            f"cross-monomer jax-pme path requires method=ewald; got {method_name!r}"
        )
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    offsets = np.asarray(monomer_offsets, dtype=np.int64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {pos.shape}")
    if coef.shape[0] != pos.shape[0]:
        raise ValueError(f"coefficients length {coef.shape[0]} != n_atoms {pos.shape[0]}")
    evaluator = _cached_cross_evaluator(
        method_name,
        int(exponent),
        float(prefactor),
        float(sr_cutoff_A),
        float(box_length_A),
        int(pos.shape[0]),
        tuple(int(v) for v in offsets),
    )
    return evaluator.compute(pos, coef)
