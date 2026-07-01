"""Cross-monomer jax-pme long-range (one prepare, fused cross-monomer sum).

Hybrid MM long-range needs periodic Ewald/PME on **cross-monomer** pairs only.
Default implementation uses a **structure-factor** reciprocal pass (ewald) plus
cross-masked real space; ``masked`` mode uses ``E_full(q) − Σ_m E(q ⊙ mask_m)``.

Env:
  ``MMML_JAX_PME_INTRA_MODE=cross|full_minus_intra`` (default ``cross``)
  ``MMML_JAX_PME_CROSS_KERNEL=structure_factor|masked`` (ewald default: structure_factor)
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
    jax_pme_mesh_spacing_A,
    materialize_jax_pme_host_numpy,
    resolve_jax_pme_method,
)

_PROFILE: dict[str, list[float]] = {}


def resolve_jax_pme_intra_mode(method: str | None = None) -> str:
    """``cross`` (fused) or ``full_minus_intra`` (legacy per-monomer loop)."""
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
    return mode


def resolve_jax_pme_cross_kernel(method: str | None = None) -> str:
    """``structure_factor`` (ewald, fast) or ``masked`` (exact reference, all methods)."""
    raw = os.environ.get("MMML_JAX_PME_CROSS_KERNEL", "auto").strip().lower()
    if raw in ("auto", ""):
        return (
            "structure_factor"
            if resolve_jax_pme_method(str(method or "ewald")) == "ewald"
            else "masked"
        )
    if raw in ("structure_factor", "sf", "fast"):
        return "structure_factor"
    if raw in ("masked", "full_minus_masked", "reference"):
        return "masked"
    raise ValueError(
        "MMML_JAX_PME_CROSS_KERNEL must be auto|structure_factor|masked; "
        f"got {raw!r}"
    )


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
    kernel: str
    exponent: int
    prefactor: float
    sr_cutoff_A: float
    box_length_A: float
    n_atoms: int
    n_monomers: int


def _correction_energy_scalar(pot, smearing, charges, volume, positions, cell, pbc):
    import jax.numpy as jnp

    corr = pot.correction(smearing, charges, volume, positions, cell, pbc)
    return jnp.sum(charges * corr) / 2.0


def _masked_cross_energy(
    pot,
    solver,
    *,
    prefactor: float,
    charges,
    cell,
    positions,
    i,
    j,
    cell_shifts,
    k_grid,
    smearing,
    pbc,
    monomer_id,
    n_mono: int,
    for_ewald: bool,
):
    import jax.numpy as jnp
    from jaxpme.kspace import generate_kvectors, get_reciprocal
    from jaxpme.utils import get_distances

    reciprocal_cell = get_reciprocal(cell)
    volume = jnp.abs(jnp.linalg.det(cell))
    kvectors = generate_kvectors(
        reciprocal_cell,
        k_grid.shape,
        dtype=positions.dtype,
        for_ewald=for_ewald,
    )

    def _energy_for_charges(q):
        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        rspace = solver.rspace(smearing, q, r, i, j)
        if for_ewald:
            kspace = solver.kspace(smearing, q, kvectors, positions, volume, cell, pbc)
        else:
            kspace = solver.kspace(
                smearing,
                q,
                reciprocal_cell,
                k_grid,
                kvectors,
                positions,
                volume,
                cell,
                pbc,
            )
        phi = prefactor * (rspace + kspace)
        return jnp.sum(q * phi)

    base = _energy_for_charges(charges)
    intra = jnp.asarray(0.0, dtype=positions.dtype)
    for m in range(n_mono):
        mask = (monomer_id == m).astype(charges.dtype)
        intra = intra + _energy_for_charges(charges * mask)
    return base - intra


def _structure_factor_cross_energy(
    pot,
    solver,
    *,
    prefactor: float,
    charges,
    cell,
    positions,
    i,
    j,
    cell_shifts,
    k_grid,
    smearing,
    pbc,
    monomer_id,
    n_mono: int,
):
    """Ewald cross energy: masked RS + |ρ|² cross k-space + correction subtraction."""
    import jax
    import jax.numpy as jnp
    from jaxpme.kspace import generate_kvectors, get_reciprocal
    from jaxpme.utils import get_distances

    reciprocal_cell = get_reciprocal(cell)
    volume = jnp.abs(jnp.linalg.det(cell))
    kvectors = generate_kvectors(
        reciprocal_cell,
        k_grid.shape,
        dtype=positions.dtype,
        for_ewald=True,
    )

    r = get_distances(cell, positions[i], positions[j], cell_shifts)
    cross_mask = (monomer_id[i] != monomer_id[j]).astype(r.dtype)
    phi_rs = solver.rspace(smearing, charges, r * cross_mask, i, j)
    e_rs = prefactor * jnp.sum(charges * phi_rs)

    trig = kvectors @ positions.T
    cos_all = jnp.cos(trig)
    sin_all = jnp.sin(trig)
    cos_w = cos_all * charges[None, :]
    sin_w = sin_all * charges[None, :]
    rho_tot_r = jnp.sum(cos_w, axis=1)
    rho_tot_i = jnp.sum(sin_w, axis=1)
    cos_nk = (cos_all.T * charges[:, None])
    sin_nk = (sin_all.T * charges[:, None])
    rho_m_r = jax.ops.segment_sum(cos_nk, monomer_id, num_segments=n_mono)
    rho_m_i = jax.ops.segment_sum(sin_nk, monomer_id, num_segments=n_mono)
    mono_sq = jnp.sum(rho_m_r**2 + rho_m_i**2, axis=0)
    cross_sq = (rho_tot_r**2 + rho_tot_i**2) - mono_sq
    k2 = jax.lax.square(kvectors).sum(axis=-1)
    G = jnp.where(k2 == 0.0, 0.0, pot.lr(smearing, k2) * 2.0)
    e_k = prefactor * jnp.sum(G * cross_sq) / volume / 2.0
    k0 = pot.lr(smearing, jnp.zeros((), dtype=positions.dtype))
    q_tot = jnp.sum(charges)
    q_mono = jax.ops.segment_sum(charges, monomer_id, num_segments=n_mono)
    e_k = e_k + prefactor * k0 * (q_tot**2 - jnp.sum(q_mono**2)) / volume / 2.0

    e_corr = _correction_energy_scalar(
        pot, smearing, charges, volume, positions, cell, pbc
    )
    for m in range(n_mono):
        mask = (monomer_id == m).astype(charges.dtype)
        e_corr = e_corr - _correction_energy_scalar(
            pot, smearing, charges * mask, volume, positions, cell, pbc
        )
    e_corr = prefactor * e_corr
    return e_rs + e_k + e_corr


def _build_cross_monomer_jit(key: _CrossMonomerJitKey):
    import jax
    import jax.numpy as jnp
    from jaxpme.potentials import potential
    from jaxpme.solvers import ewald, p3m, pme

    pot = potential(exponent=int(key.exponent))
    method = str(key.method_name)
    if method == "ewald":
        solver = ewald(pot)
        for_ewald = True
    elif method == "pme":
        solver = pme(pot)
        for_ewald = False
    elif method == "p3m":
        solver = p3m(pot)
        for_ewald = False
    else:
        raise ValueError(f"unsupported jax-pme method {method!r}")

    prefactor = float(key.prefactor)
    n_mono = int(key.n_monomers)
    use_sf = str(key.kernel) == "structure_factor" and method == "ewald"

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
        if use_sf:
            return _structure_factor_cross_energy(
                pot,
                solver,
                prefactor=prefactor,
                charges=charges,
                cell=cell,
                positions=positions,
                i=i,
                j=j,
                cell_shifts=cell_shifts,
                k_grid=k_grid,
                smearing=smearing,
                pbc=pbc,
                monomer_id=monomer_id,
                n_mono=n_mono,
            )
        return _masked_cross_energy(
            pot,
            solver,
            prefactor=prefactor,
            charges=charges,
            cell=cell,
            positions=positions,
            i=i,
            j=j,
            cell_shifts=cell_shifts,
            k_grid=k_grid,
            smearing=smearing,
            pbc=pbc,
            monomer_id=monomer_id,
            n_mono=n_mono,
            for_ewald=for_ewald,
        )

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


def _prepare_cross_inputs(
    *,
    method_name: str,
    exponent: int,
    prefactor: float,
    positions: np.ndarray,
    coefficients: np.ndarray,
    box_length_A: float,
    sr_cutoff_A: float,
):
    from ase import Atoms
    from jaxpme import Ewald, P3M, PME

    atoms = Atoms(
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.eye(3, dtype=np.float64) * float(box_length_A),
        pbc=True,
    )
    smearing = float(sr_cutoff_A) / 5.0
    calc_map = {
        "ewald": Ewald,
        "pme": PME,
        "p3m": P3M,
    }
    calc = calc_map[method_name](exponent=int(exponent), prefactor=float(prefactor))
    if method_name == "ewald":
        lr_wavelength = smearing / 2.0
        prepared = calc.prepare(
            atoms,
            np.asarray(coefficients, dtype=np.float64).reshape(-1),
            cutoff=float(sr_cutoff_A),
            smearing=smearing,
            lr_wavelength=lr_wavelength,
        )
    else:
        mesh_spacing = jax_pme_mesh_spacing_A(float(sr_cutoff_A), float(box_length_A))
        prepared = calc.prepare(
            atoms,
            np.asarray(coefficients, dtype=np.float64).reshape(-1),
            cutoff=float(sr_cutoff_A),
            mesh_spacing=mesh_spacing,
            smearing=smearing,
        )
    return prepared


@dataclass(frozen=True)
class _CrossMonomerHostEvaluator:
    method_name: str
    kernel: str
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

        charges, cell_j, positions_j, i, j, cell_shifts, k_grid, smearing_v, pbc_v = (
            _prepare_cross_inputs(
                method_name=str(self.method_name),
                exponent=int(self.exponent),
                prefactor=float(self.prefactor),
                positions=pos,
                coefficients=coef,
                box_length_A=float(self.box_length_A),
                sr_cutoff_A=float(self.sr_cutoff_A),
            )
        )
        monomer_id = jnp.asarray(self.monomer_id, dtype=jnp.int32)
        jit_key = _CrossMonomerJitKey(
            self.method_name,
            str(self.kernel),
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
        profile_label = (
            "cross_monomer_sf"
            if self.kernel == "structure_factor"
            else "cross_monomer_masked"
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
        _record(profile_label, t0)
        e_host, f_host = materialize_jax_pme_host_numpy(energy, forces)
        return LongRangeInteractionResult(
            energy_kcalmol=e_host,
            forces_kcalmol_A=f_host,
        )


@lru_cache(maxsize=128)
def _cached_cross_evaluator(
    method_name: str,
    kernel: str,
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
        kernel=str(kernel),
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
    """Cross-monomer periodic 1/r^p with one jax-pme prepare + fused cross kernel."""
    method_name = resolve_jax_pme_method(str(method))
    kernel = resolve_jax_pme_cross_kernel(method_name)
    if kernel == "structure_factor" and method_name != "ewald":
        kernel = "masked"
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    offsets = np.asarray(monomer_offsets, dtype=np.int64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {pos.shape}")
    if coef.shape[0] != pos.shape[0]:
        raise ValueError(f"coefficients length {coef.shape[0]} != n_atoms {pos.shape[0]}")
    evaluator = _cached_cross_evaluator(
        method_name,
        kernel,
        int(exponent),
        float(prefactor),
        float(sr_cutoff_A),
        float(box_length_A),
        int(pos.shape[0]),
        tuple(int(v) for v in offsets),
    )
    return evaluator.compute(pos, coef)
