"""Long-range electrostatic backend selection for hybrid ML/MM potentials.

MMML's default JAX MM path uses minimum-image Coulomb truncated at the switched-MM
outer radius (~13 Å by default).  Optional backends can supply k-space corrections:

* ``mic`` — truncated MIC Coulomb only (default)
* ``jax_pme`` — JAX-native PME via the ``jax-pme`` dependency
* ``nvalchemiops_pme`` — JAX PME via ``nvalchemiops`` electrostatics
* ``scafacos`` — ScaFaCoS ``libfcs`` (PME / P³M / P²NFFT / …)

Selection mirrors ``nl_backend.py``: CLI/YAML may pass an explicit name; otherwise
``MMML_LR_SOLVER`` or the implicit default selects ``mic``.  Set ``lr_solver: jax_pme``,
``scafacos``, or ``nvalchemiops_pme`` to opt into k-space backends.  ``auto`` is a
legacy alias for ``mic`` (it no longer auto-picks jax-pme).

See ``mlpot/LONG_RANGE_ELECTROSTATICS.md`` for how these layers interact with
CHARMM IMAGE lists and MLpot BLOCK terms.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Literal, Protocol

import numpy as np

LrSolverName = Literal["auto", "mic", "jax_pme", "nvalchemiops_pme", "ewald", "scafacos"]
JaxPmeMethod = Literal["ewald", "pme", "p3m"]

CHARMM_COULOMB_KCAL = 332.063711
DEFAULT_JAX_PME_SR_CUTOFF_A = 6.0


def resolve_jax_pme_dispersion(enabled: bool | None = None) -> bool:
    """Whether jax-pme supplies the r^-6 LJ tail in hybrid MM (default on)."""
    if enabled is not None:
        return bool(enabled)
    raw = os.environ.get("MMML_JAX_PME_DISPERSION", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def jax_pme_host_device_name() -> str:
    """Device for jax-pme work invoked from host ``pure_callback`` (default CPU)."""
    return (os.environ.get("MMML_JAX_PME_DEVICE") or "cpu").strip().lower()


@contextmanager
def jax_pme_host_eval_context(*, disable_jit: bool = False) -> Iterator[None]:
    """Run jax-pme ``energy_forces`` on a stable host device.

    Hybrid MM calls jax-pme from CHARMM MLpot ``pure_callback`` while the outer
    MLpot JAX graph may still be on CPU (MPI defer path) or sharing one GPU.
    Nested jax-pme on GPU in that window can stall indefinitely for mesh methods
    (PME/P3M) under ``mpirun``; host evaluation defaults to CPU.

    Set ``disable_jit=True`` inside ``pure_callback`` host code so nested jax-pme
  does not re-enter the parent XLA executor (deadlock on ``BlockUntilReady``).
    """
    import jax

    prefer = jax_pme_host_device_name()
    jit_ctx = jax.disable_jit() if disable_jit else nullcontext()
    if prefer == "gpu":
        with jit_ctx:
            yield
        return
    with jit_ctx, jax.default_device(jax.devices("cpu")[0]):
        yield


@contextmanager
def jax_pme_pure_callback_host_context() -> Iterator[None]:
    """jax-pme from ``jax.pure_callback`` (CPU + no nested JIT)."""
    with jax_pme_host_eval_context(disable_jit=True):
        yield


def jax_pme_mesh_method(method: str | None = None) -> bool:
    """True for k-space mesh jax-pme methods (PME / P3M)."""
    return resolve_jax_pme_method(method) in ("pme", "p3m")


def materialize_jax_pme_host_numpy(
    energy: object,
    forces: object,
) -> tuple[float, np.ndarray]:
    """Block and copy jax-pme ``energy_forces`` outputs to host ``numpy``.

    Hybrid MM returns these values from ``jax.pure_callback``; leaving pending
    JAX arrays on GPU can deadlock the parent XLA executor under ``mpirun``.
    """
    import jax
    import jax.numpy as jnp

    with jax_pme_host_eval_context(disable_jit=True):
        e_arr = jnp.asarray(energy)
        f_arr = jnp.asarray(forces)
        jax.block_until_ready(e_arr)
        jax.block_until_ready(f_arr)
        e_host = float(np.asarray(jax.device_get(e_arr), dtype=np.float64))
        f_host = np.asarray(jax.device_get(f_arr), dtype=np.float64)
    return e_host, f_host


def warmup_jax_pme_coulomb_host(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
) -> LongRangeInteractionResult:
    """Compile jax-pme on the host before the first MLpot ``ENER`` (MPI-safe)."""
    return compute_jax_pme_coulomb(
        positions_A,
        charges_e,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=float(sr_cutoff_A),
    )


def warmup_jax_pme_power_law_host(
    positions_A: np.ndarray,
    coefficients: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    exponent: int = 1,
    prefactor: float | None = None,
) -> LongRangeInteractionResult:
    """Compile/evaluate one host jax-pme power-law shape before production use."""
    return compute_jax_pme_power_law(
        positions_A,
        coefficients,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=float(sr_cutoff_A),
        exponent=int(exponent),
        prefactor=prefactor,
    )


def warmup_jax_pme_hybrid_host(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    c6_sqrt: np.ndarray | None = None,
    include_dispersion: bool | None = None,
    pbc_cell: np.ndarray | None = None,
    ml_switch_width: float = 1.0,
    mm_switch_on: float = 12.0,
    mm_switch_width: float = 1.0,
    complementary_handoff: bool = True,
    mm_r_min: float | None = None,
) -> dict[str, int]:
    """Pre-warm jax-pme hybrid shapes (cross-monomer fused or legacy intra loop)."""
    from jaxpme import prefactors as jpref

    from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
        compute_jax_pme_cross_monomer_power_law,
        resolve_jax_pme_intra_mode,
    )

    pos = np.asarray(positions_A, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64).reshape(-1)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    counts = {
        "coulomb_full": 0,
        "coulomb_intra": 0,
        "coulomb_cross": 0,
        "dispersion_full": 0,
        "dispersion_intra": 0,
        "dispersion_cross": 0,
    }
    use_cross = resolve_jax_pme_intra_mode(method) == "cross"

    def _nonzero(values: np.ndarray) -> bool:
        arr = np.asarray(values, dtype=np.float64)
        return arr.size > 0 and bool(np.any(arr != 0.0))

    def _representative_slices() -> list[slice]:
        seen: set[int] = set()
        reps: list[slice] = []
        for m in range(int(len(offsets) - 1)):
            start = int(offsets[m])
            stop = int(offsets[m + 1])
            size = stop - start
            if size <= 0 or size in seen:
                continue
            seen.add(size)
            reps.append(slice(start, stop))
        return reps

    reps = _representative_slices()
    if _nonzero(charges):
        if use_cross:
            compute_jax_pme_cross_monomer_power_law(
                pos,
                charges,
                offsets,
                box_length_A=float(box_length_A),
                method=method,
                sr_cutoff_A=float(sr_cutoff_A),
                exponent=1,
                prefactor=float(jpref.kcalmol_A),
            )
            counts["coulomb_cross"] += 1
        else:
            warmup_jax_pme_power_law_host(
                pos,
                charges,
                box_length_A=float(box_length_A),
                method=method,
                sr_cutoff_A=float(sr_cutoff_A),
                exponent=1,
                prefactor=float(jpref.kcalmol_A),
            )
            counts["coulomb_full"] += 1
            for sl in reps:
                if not _nonzero(charges[sl]):
                    continue
                warmup_jax_pme_power_law_host(
                    pos[sl],
                    charges[sl],
                    box_length_A=float(box_length_A),
                    method=method,
                    sr_cutoff_A=float(sr_cutoff_A),
                    exponent=1,
                    prefactor=float(jpref.kcalmol_A),
                )
                counts["coulomb_intra"] += 1

    if c6_sqrt is not None and resolve_jax_pme_dispersion(include_dispersion):
        c6 = np.asarray(c6_sqrt, dtype=np.float64).reshape(-1)
        if _nonzero(c6):
            if use_cross:
                compute_jax_pme_cross_monomer_power_law(
                    pos,
                    c6,
                    offsets,
                    box_length_A=float(box_length_A),
                    method=method,
                    sr_cutoff_A=float(sr_cutoff_A),
                    exponent=6,
                    prefactor=DEFAULT_JAX_PME_LJ_PREFACTOR,
                )
                counts["dispersion_cross"] += 1
            else:
                warmup_jax_pme_power_law_host(
                    pos,
                    c6,
                    box_length_A=float(box_length_A),
                    method=method,
                    sr_cutoff_A=float(sr_cutoff_A),
                    exponent=6,
                    prefactor=DEFAULT_JAX_PME_LJ_PREFACTOR,
                )
                counts["dispersion_full"] += 1
                for sl in reps:
                    if not _nonzero(c6[sl]):
                        continue
                    warmup_jax_pme_power_law_host(
                        pos[sl],
                        c6[sl],
                        box_length_A=float(box_length_A),
                        method=method,
                        sr_cutoff_A=float(sr_cutoff_A),
                        exponent=6,
                        prefactor=DEFAULT_JAX_PME_LJ_PREFACTOR,
                    )
                    counts["dispersion_intra"] += 1
    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        _com_switch_value_and_grad,
    )

    if pbc_cell is not None:
        _com_switch_value_and_grad(
            pos,
            offsets,
            np.asarray(pbc_cell, dtype=np.float64),
            ml_switch_width=float(ml_switch_width),
            mm_switch_on=float(mm_switch_on),
            mm_switch_width=float(mm_switch_width),
            complementary_handoff=bool(complementary_handoff),
            mm_r_min=mm_r_min,
        )
        counts["com_switch_jit"] = 1
    return counts


def have_jax_pme() -> bool:
    try:
        from jaxpme import Ewald, P3M, PME  # noqa: F401

        return True
    except ImportError:
        return False


def have_nvalchemiops_pme() -> bool:
    try:
        from nvalchemiops.jax.interactions.electrostatics import particle_mesh_ewald  # noqa: F401
        from nvalchemiops.jax.neighbors import neighbor_list  # noqa: F401

        return True
    except Exception:
        return False


def have_scafacos() -> bool:
    try:
        from mmml.interfaces.scafacosInterface.scafacos_session import have_scafacos as _have

        return _have()
    except Exception:
        return False


def resolve_lr_solver(name: str | None = None) -> LrSolverName:
    """Resolve solver from argument, ``MMML_LR_SOLVER`` env, or ``mic`` (default)."""
    raw = (name or os.environ.get("MMML_LR_SOLVER", "mic")).strip().lower()
    if raw in ("nvalchemiops", "nvalchemiops_pme", "nval_pme"):
        return "nvalchemiops_pme"
    if raw in ("native_ewald", "jit_ewald"):
        raw = "ewald"
    if raw in ("auto", "mic", "jax_pme", "nvalchemiops_pme", "ewald", "scafacos"):
        return raw  # type: ignore[return-value]
    raise ValueError(
        f"lr_solver must be auto|mic|jax_pme|nvalchemiops_pme|ewald|scafacos; got {name!r}"
    )


def pick_lr_solver(requested: str | None = None) -> LrSolverName:
    """Choose the active long-range electrostatic backend."""
    name = resolve_lr_solver(requested)
    if name in ("auto", "mic"):
        return "mic"
    if name == "scafacos" and not have_scafacos():
        if have_jax_pme():
            return "jax_pme"
        if have_nvalchemiops_pme():
            return "nvalchemiops_pme"
        return "mic"
    if name == "jax_pme" and not have_jax_pme():
        if have_nvalchemiops_pme():
            return "nvalchemiops_pme"
        if have_scafacos():
            return "scafacos"
        return "mic"
    if name == "nvalchemiops_pme" and not have_nvalchemiops_pme():
        if have_jax_pme():
            return "jax_pme"
        if have_scafacos():
            return "scafacos"
        return "mic"
    return name


@dataclass(frozen=True)
class LongRangeInteractionResult:
    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray


# Backward-compatible alias
LongRangeCoulombResult = LongRangeInteractionResult

DEFAULT_JAX_PME_LJ_PREFACTOR = -1.0


@lru_cache(maxsize=32)
def _cached_jax_pme_calculator(
    method_name: JaxPmeMethod,
    exponent: int,
    prefactor: float,
):
    from jaxpme import Ewald, P3M, PME

    calc_map = {"ewald": Ewald, "pme": PME, "p3m": P3M}
    return calc_map[method_name](
        exponent=int(exponent),
        prefactor=float(prefactor),
    )


def jax_pme_mesh_spacing_A(sr_cutoff_A: float, box_length_A: float) -> float:
    """Real-space mesh spacing for jax-pme PME/P3M (capped for CI memory)."""
    smearing = float(sr_cutoff_A) / 5.0
    mesh_max = int(os.environ.get("MMML_JAX_PME_MESH_MAX", "64") or "64")
    return max(smearing / 8.0, float(box_length_A) / max(mesh_max, 8))


@dataclass(frozen=True)
class _JaxPmePowerLawHostEvaluator:
    method_name: JaxPmeMethod
    exponent: int
    prefactor: float
    sr_cutoff_A: float
    box_length_A: float
    n_atoms: int

    def compute(
        self,
        positions_A: np.ndarray,
        coefficients: np.ndarray,
    ) -> LongRangeInteractionResult:
        """Evaluate one fixed-shape jax-pme power-law problem on the host."""
        from ase import Atoms
        from jaxpme import Ewald, P3M, PME  # noqa: F401

        pos = np.asarray(positions_A, dtype=np.float64)
        coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if pos.shape != (int(self.n_atoms), 3):
            raise ValueError(
                f"cached jax-pme evaluator expects positions "
                f"({self.n_atoms}, 3); got {pos.shape}"
            )
        if coef.shape[0] != int(self.n_atoms):
            raise ValueError(
                f"cached jax-pme evaluator expects {self.n_atoms} coefficients; "
                f"got {coef.shape[0]}"
            )
        atoms = Atoms(
            positions=pos,
            cell=np.eye(3, dtype=np.float64) * float(self.box_length_A),
            pbc=True,
        )
        calc = _cached_jax_pme_calculator(
            self.method_name,
            int(self.exponent),
            float(self.prefactor),
        )
        smearing = float(self.sr_cutoff_A) / 5.0
        mesh_spacing = jax_pme_mesh_spacing_A(float(self.sr_cutoff_A), float(self.box_length_A))
        lr_wavelength = smearing / 2.0
        if self.method_name == "ewald":
            inputs = calc.prepare(
                atoms,
                coef,
                cutoff=float(self.sr_cutoff_A),
                smearing=smearing,
                lr_wavelength=lr_wavelength,
            )
        else:
            inputs = calc.prepare(
                atoms,
                coef,
                cutoff=float(self.sr_cutoff_A),
                mesh_spacing=mesh_spacing,
                smearing=smearing,
            )
        host_ctx = (
            jax_pme_pure_callback_host_context
            if jax_pme_mesh_method(self.method_name)
            else jax_pme_host_eval_context
        )
        with host_ctx():
            energy, forces = calc.energy_forces(*inputs)
        e_host, f_host = materialize_jax_pme_host_numpy(energy, forces)
        return LongRangeInteractionResult(
            energy_kcalmol=e_host,
            forces_kcalmol_A=f_host,
        )


@lru_cache(maxsize=128)
def _cached_jax_pme_power_law_evaluator(
    method_name: JaxPmeMethod,
    exponent: int,
    prefactor: float,
    sr_cutoff_A: float,
    box_length_A: float,
    n_atoms: int,
) -> _JaxPmePowerLawHostEvaluator:
    return _JaxPmePowerLawHostEvaluator(
        method_name=method_name,
        exponent=int(exponent),
        prefactor=float(prefactor),
        sr_cutoff_A=float(sr_cutoff_A),
        box_length_A=float(box_length_A),
        n_atoms=int(n_atoms),
    )


def per_atom_monomer_ids(
    total_atoms: int,
    monomer_offsets: np.ndarray,
    n_monomers: int,
) -> np.ndarray:
    """Map each atom index to its 0-based monomer id."""
    out = np.empty(int(total_atoms), dtype=np.int32)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    for mi in range(int(n_monomers)):
        out[offsets[mi] : offsets[mi + 1]] = mi
    return out


def scale_per_atom_coefficients_by_monomer_lambda(
    coeffs: np.ndarray,
    monomer_ids: np.ndarray,
    lambda_monomer: np.ndarray,
) -> np.ndarray:
    """Apply hybrid ``lambda_monomer`` as sqrt(λ) per atom (pair product λ_i λ_j)."""
    lam = np.asarray(lambda_monomer, dtype=np.float64)
    mid = np.asarray(monomer_ids, dtype=np.int32)
    scaled = np.asarray(coeffs, dtype=np.float64).copy()
    scaled *= np.sqrt(np.maximum(lam[mid], 0.0))
    return scaled


def per_atom_jax_pme_c6_sqrt(
    epsilons_kcal: np.ndarray,
    rmin_half_A: np.ndarray,
) -> np.ndarray:
    """Per-atom √C6 for CHARMM LJ dispersion in jax-pme.

    CHARMM NONBONDED parameter records store ``Rmin/2``, not conventional LJ
    ``sigma`` and not full per-atom ``Rmin``.  For an identical atom pair,
    CHARMM's attractive coefficient is::

        C6_ii = 2 * |epsilon_i| * (2 * Rmin_half_i) ** 6

    The factor of two inside the sixth power is essential; treating
    ``Rmin/2`` as the full radius suppresses dispersion by 2**6 = 64.

    Products of the returned coefficients use geometric k-space combining.
    That is exact for identical atom types and an approximation to CHARMM's
    arithmetic ``Rmin/2`` combining for unlike types.  Direct-space r⁻¹²
    retains the exact CHARMM pair ``Rmin`` and epsilon combination.
    """
    ep_abs = np.abs(np.asarray(epsilons_kcal, dtype=np.float64))
    rmin_full = 2.0 * np.asarray(rmin_half_A, dtype=np.float64)
    c6 = 2.0 * ep_abs * np.power(rmin_full, 6)
    return np.sqrt(c6)


def per_atom_jax_pme_c6_sqrt_for_atoms(
    epsilons_per_atom: np.ndarray,
    rmins_per_atom: np.ndarray,
    *,
    monomer_ids: np.ndarray | None = None,
    lambda_monomer: np.ndarray | None = None,
) -> np.ndarray:
    """√C6 per atom, optionally scaled by hybrid monomer λ."""
    out = per_atom_jax_pme_c6_sqrt(epsilons_per_atom, rmins_per_atom)
    if monomer_ids is not None and lambda_monomer is not None:
        out = scale_per_atom_coefficients_by_monomer_lambda(
            out, monomer_ids, lambda_monomer
        )
    return out


def compute_jax_pme_power_law(
    positions_A: np.ndarray,
    coefficients: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    exponent: int = 1,
    prefactor: float | None = None,
) -> LongRangeInteractionResult:
    """Periodic 1/r^p interaction via jax-pme (Ewald / PME / P3M).

    ``coefficients`` are jax-pme "charges" (elementary charge for p=1, √C6 for
    attractive dispersion with ``prefactor=-1`` and p=6).
    """
    from jaxpme import prefactors as jpref

    if int(exponent) not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"jax_pme exponent must be 1..6; got {exponent}")
    if prefactor is None:
        prefactor = float(jpref.kcalmol_A) if int(exponent) == 1 else DEFAULT_JAX_PME_LJ_PREFACTOR

    method_name = resolve_jax_pme_method(str(method))
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {pos.shape}")
    if coef.shape[0] != pos.shape[0]:
        raise ValueError(f"coefficients length {coef.shape[0]} != n_atoms {pos.shape[0]}")
    L = float(box_length_A)
    evaluator = _cached_jax_pme_power_law_evaluator(
        method_name,
        int(exponent),
        float(prefactor),
        float(sr_cutoff_A),
        L,
        int(pos.shape[0]),
    )
    return evaluator.compute(pos, coef)


def compute_jax_pme_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
) -> LongRangeInteractionResult:
    """Full periodic Coulomb via jax-pme (Ewald / PME / P3M)."""
    from jaxpme import prefactors as jpref

    return compute_jax_pme_power_law(
        positions_A,
        charges_e,
        box_length_A=box_length_A,
        method=method,
        sr_cutoff_A=sr_cutoff_A,
        exponent=1,
        prefactor=float(jpref.kcalmol_A),
    )


def _nvalchemiops_pme_accuracy(accuracy: float | None = None) -> float:
    if accuracy is not None:
        return float(accuracy)
    raw = os.environ.get("MMML_NVALCHEMIOPS_PME_ACCURACY", "1e-6").strip()
    return float(raw)


def nvalchemiops_pme_device_name() -> str:
    """Device for nvalchemiops/Warp PME (default CUDA).

    Warp's neighbor-list FFI is registered on GPU, not Host.  Eager / MD
    evaluation ``device_put``s onto this device.  Jitted train steps must **not**
    run nested JAX+Warp on the same GPU as the parent XLA executable (deadlock);
    they use a spawn-isolated worker instead (see
    ``nvalchemiops_pme_train_isolate_enabled``).
    Override with ``MMML_NVALCHEMIOPS_PME_DEVICE=cpu`` only for debugging.
    """
    return (os.environ.get("MMML_NVALCHEMIOPS_PME_DEVICE") or "gpu").strip().lower()


def nvalchemiops_pme_train_isolate_enabled() -> bool:
    """Whether train-step PME runs in a spawn worker (default on).

    Nested CUDA JAX inside ``jax.pure_callback`` while the parent ``jit_train_step``
    holds the GPU XLA executor stalls forever (``time+`` frozen).  Isolation
    gives the Warp PME path its own process/JAX runtime.  Set
    ``MMML_NVALCHEMIOPS_PME_ISOLATE=0`` only for debugging (will deadlock under
    GPU train).
    """
    raw = (os.environ.get("MMML_NVALCHEMIOPS_PME_ISOLATE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# Persistent spawn worker for train-callback PME (avoids nested-GPU XLA deadlock).
_NVAL_PME_WORKER = None
_NVAL_PME_WORKER_LOCK = None


def _nval_pme_worker_lock():
    global _NVAL_PME_WORKER_LOCK
    if _NVAL_PME_WORKER_LOCK is None:
        import threading

        _NVAL_PME_WORKER_LOCK = threading.Lock()
    return _NVAL_PME_WORKER_LOCK


def _nvalchemiops_pme_worker_main(q_in, q_out) -> None:
    """Child process: eager CUDA PME with a private JAX runtime."""
    import traceback

    while True:
        msg = q_in.get()
        if msg is None:
            break
        try:
            pos, chg, box, acc, cut = msg
            # Import inside the child so the parent XLA state is not shared.
            from mmml.interfaces.pycharmmInterface.long_range_backend import (
                _nvalchemiops_pme_energy_forces_concrete,
            )

            e, f = _nvalchemiops_pme_energy_forces_concrete(
                pos,
                chg,
                box_length_A=float(box),
                accuracy=float(acc),
                real_space_cutoff_A=cut,
            )
            q_out.put(
                (
                    "ok",
                    float(e),
                    np.asarray(f, dtype=np.float64),
                )
            )
        except BaseException as exc:
            q_out.put(("err", f"{type(exc).__name__}: {exc}", traceback.format_exc()))


def _nvalchemiops_pme_ensure_worker():
    """Start the spawn PME worker once per parent process."""
    global _NVAL_PME_WORKER
    import atexit
    import multiprocessing as mp

    if _NVAL_PME_WORKER is not None:
        proc, q_in, q_out = _NVAL_PME_WORKER
        if proc.is_alive():
            return _NVAL_PME_WORKER
    ctx = mp.get_context("spawn")
    q_in = ctx.Queue()
    q_out = ctx.Queue()
    proc = ctx.Process(
        target=_nvalchemiops_pme_worker_main,
        args=(q_in, q_out),
        name="mmml-nvalchemiops-pme",
        daemon=True,
    )
    proc.start()

    def _shutdown() -> None:
        global _NVAL_PME_WORKER
        state = _NVAL_PME_WORKER
        if state is None:
            return
        p, qi, _qo = state
        try:
            qi.put(None)
        except Exception:
            pass
        try:
            p.join(timeout=5.0)
        except Exception:
            pass
        if p.is_alive():
            try:
                p.kill()
            except Exception:
                pass
        _NVAL_PME_WORKER = None

    atexit.register(_shutdown)
    _NVAL_PME_WORKER = (proc, q_in, q_out)
    return _NVAL_PME_WORKER


def _nvalchemiops_pme_energy_forces_isolated(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float | None = None,
    real_space_cutoff_A: float | None = None,
) -> tuple[float, np.ndarray]:
    """PME in a spawn child (safe under jitted train ``pure_callback``)."""
    with _nval_pme_worker_lock():
        proc, q_in, q_out = _nvalchemiops_pme_ensure_worker()
        q_in.put(
            (
                np.asarray(positions_A, dtype=np.float64),
                np.asarray(charges_e, dtype=np.float64).reshape(-1),
                float(box_length_A),
                float(_nvalchemiops_pme_accuracy(accuracy)),
                None if real_space_cutoff_A is None else float(real_space_cutoff_A),
            )
        )
        if not proc.is_alive():
            raise RuntimeError(
                "nvalchemiops PME spawn worker died before returning "
                "(check CUDA/OOM in the child process)."
            )
        status, *payload = q_out.get()
        if status == "ok":
            e, f = payload
            return float(e), np.asarray(f, dtype=np.float64)
        err, tb = payload
        raise RuntimeError(
            f"nvalchemiops PME spawn worker failed: {err}\n{tb}"
        )


def _nvalchemiops_pme_energy_forces_for_train_callback(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float | None = None,
    real_space_cutoff_A: float | None = None,
) -> tuple[float, np.ndarray]:
    """Train-callback entry: isolated by default, in-process when isolate=0."""
    if nvalchemiops_pme_train_isolate_enabled():
        return _nvalchemiops_pme_energy_forces_isolated(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            accuracy=accuracy,
            real_space_cutoff_A=real_space_cutoff_A,
        )
    return _nvalchemiops_pme_energy_forces_concrete(
        positions_A,
        charges_e,
        box_length_A=box_length_A,
        accuracy=accuracy,
        real_space_cutoff_A=real_space_cutoff_A,
    )


def warmup_nvalchemiops_pme_train_worker(
    *,
    n_atoms: int,
    box_length_A: float,
    accuracy: float | None = None,
    real_space_cutoff_A: float | None = None,
) -> None:
    """Start + compile the spawn PME worker before the first train step."""
    if not nvalchemiops_pme_train_isolate_enabled():
        return
    n = max(2, int(n_atoms))
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[1, 0] = 2.0
    chg = np.zeros(n, dtype=np.float64)
    chg[0] = 1.0
    chg[1] = -1.0
    print(
        "Warming nvalchemiops PME spawn worker "
        "(avoids nested-GPU deadlock in jit_train_step)...",
        flush=True,
    )
    _nvalchemiops_pme_energy_forces_isolated(
        pos,
        chg,
        box_length_A=float(box_length_A),
        accuracy=accuracy,
        real_space_cutoff_A=real_space_cutoff_A,
    )
    print("nvalchemiops PME spawn worker ready.", flush=True)


def _nvalchemiops_pme_resolve_device():
    """Return the JAX device used for Warp PME (CUDA by default)."""
    import jax

    prefer = nvalchemiops_pme_device_name()
    if prefer in ("cpu", "host"):
        return jax.devices("cpu")[0]
    try:
        return jax.devices("gpu")[0]
    except RuntimeError as exc:
        raise RuntimeError(
            "nvalchemiops PME requires a CUDA GPU (Warp NL FFI is not "
            "registered on Host). Train on a GPU node, or set "
            "MMML_NVALCHEMIOPS_PME_DEVICE=cpu only for debugging (Warp "
            "neighbor lists will still fail without a Host FFI handler)."
        ) from exc


@contextmanager
def nvalchemiops_pme_eval_context() -> Iterator[object]:
    """Eager nvalchemiops PME on CUDA (disable nested JIT); yields the device.

    ``jax.pure_callback`` host code defaults to the Host platform.  Warp's
    ``_compute_naive_num_shifts_*`` FFI is CUDA-only, so callers must
    ``device_put`` inputs onto the yielded device (``default_device`` alone is
    not always enough inside a callback).
    """
    import jax

    device = _nvalchemiops_pme_resolve_device()
    with jax.disable_jit(), jax.default_device(device):
        yield device


def _nvalchemiops_pme_on_device(value, device):
    """Place ``value`` on ``device`` (required for Warp FFI inside callbacks)."""
    import jax
    import jax.numpy as jnp

    return jax.device_put(jnp.asarray(value), device)


def estimate_nvalchemiops_pme_real_space_cutoff(
    *,
    box_length_A: float,
    accuracy: float | None = None,
    n_atoms: int = 2,
) -> float:
    """Host-side real-space cutoff (Å) for a cubic box (static for a training run)."""
    import jax
    import jax.numpy as jnp
    from nvalchemiops.jax.interactions.electrostatics import estimate_pme_parameters

    L = float(box_length_A)
    acc = _nvalchemiops_pme_accuracy(accuracy)
    n = max(1, int(n_atoms))
    with nvalchemiops_pme_eval_context() as device:
        pos = _nvalchemiops_pme_on_device(jnp.zeros((n, 3), dtype=jnp.float64), device)
        cell = _nvalchemiops_pme_on_device(
            (jnp.eye(3, dtype=jnp.float64) * L)[None, ...], device
        )
        params = estimate_pme_parameters(pos, cell, accuracy=acc)
        return float(
            np.asarray(jax.device_get(params.real_space_cutoff[0]), dtype=np.float64)
        )


def _nvalchemiops_pme_energy_forces_concrete(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float | None = None,
    real_space_cutoff_A: float | None = None,
) -> tuple[float, np.ndarray]:
    """Eager PME (kcal/mol) on CUDA. Safe outside ``jax.jit`` (concrete NL sizes)."""
    import jax
    import jax.numpy as jnp
    from nvalchemiops.jax.interactions.electrostatics import (
        estimate_pme_parameters,
        particle_mesh_ewald,
    )
    from nvalchemiops.jax.neighbors import neighbor_list

    pos_np = np.asarray(positions_A, dtype=np.float64)
    chg_np = np.asarray(charges_e, dtype=np.float64).reshape(-1)
    if pos_np.ndim != 2 or pos_np.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {pos_np.shape}")
    if chg_np.shape[0] != pos_np.shape[0]:
        raise ValueError(
            f"charges length {chg_np.shape[0]} != n_atoms {pos_np.shape[0]}"
        )

    L = float(box_length_A)
    acc = _nvalchemiops_pme_accuracy(accuracy)
    with nvalchemiops_pme_eval_context() as device:
        pos = _nvalchemiops_pme_on_device(pos_np, device)
        chg = _nvalchemiops_pme_on_device(chg_np, device)
        cell = _nvalchemiops_pme_on_device(
            (jnp.eye(3, dtype=jnp.float64) * L)[None, ...], device
        )
        pbc = _nvalchemiops_pme_on_device(
            jnp.array([[True, True, True]], dtype=jnp.bool_), device
        )
        if real_space_cutoff_A is None:
            params = estimate_pme_parameters(pos, cell, accuracy=acc)
            cutoff = float(
                np.asarray(jax.device_get(params.real_space_cutoff[0]), dtype=np.float64)
            )
        else:
            cutoff = float(real_space_cutoff_A)
        try:
            nl, nptr, ns = neighbor_list(
                pos,
                cutoff,
                cell=cell,
                pbc=pbc,
                return_neighbor_list=True,
            )
        except Exception as exc:
            msg = str(exc)
            if "FFI handler" in msg and "Host" in msg:
                raise RuntimeError(
                    "nvalchemiops Warp neighbor-list FFI ran on Host instead of "
                    f"CUDA (device={device}). Ensure jaxlib CUDA is installed and "
                    "MMML_NVALCHEMIOPS_PME_DEVICE is unset or 'gpu'. "
                    f"Original error: {msg}"
                ) from exc
            raise
        energies, forces = particle_mesh_ewald(
            positions=pos,
            charges=chg,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
            compute_forces=True,
            accuracy=acc,
        )
        jax.block_until_ready(energies)
        jax.block_until_ready(forces)
        energy = float(
            CHARMM_COULOMB_KCAL
            * float(np.asarray(jax.device_get(jnp.sum(energies)), dtype=np.float64))
        )
        forces_out = CHARMM_COULOMB_KCAL * np.asarray(
            jax.device_get(forces), dtype=np.float64
        )
    return energy, forces_out


def nvalchemiops_pme_coulomb_energy_jax(
    positions,
    charges,
    *,
    box_length_A: float,
    accuracy: float | None = None,
    real_space_cutoff_A: float | None = None,
    compute_forces: bool = False,
):
    """nvalchemiops PME Coulomb (kcal/mol), train- and MD-safe.

    The nvalchemiops neighbor list concretizes shift counts with ``int(...)``,
    which cannot run under ``jax.jit``.  Training therefore uses
    ``pure_callback`` + ``custom_vjp`` (analytic PME forces for the VJP).
    ``compute_forces=True`` runs the eager host path (MD / parity scripts).
    """
    import jax
    import jax.numpy as jnp

    pos = jnp.asarray(positions)
    chg = jnp.asarray(charges).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {tuple(pos.shape)}")
    if chg.shape[0] != pos.shape[0]:
        raise ValueError(f"charges length {chg.shape[0]} != n_atoms {pos.shape[0]}")

    L = float(box_length_A)
    acc = _nvalchemiops_pme_accuracy(accuracy)
    cut = None if real_space_cutoff_A is None else float(real_space_cutoff_A)

    if compute_forces:
        e_np, f_np = _nvalchemiops_pme_energy_forces_concrete(
            np.asarray(jax.device_get(pos), dtype=np.float64),
            np.asarray(jax.device_get(chg), dtype=np.float64),
            box_length_A=L,
            accuracy=acc,
            real_space_cutoff_A=cut,
        )
        return jnp.asarray(e_np, dtype=pos.dtype), jnp.asarray(f_np, dtype=pos.dtype)

    return _nvalchemiops_pme_energy_train_vjp(pos, chg, L, acc, cut)


def _nvalchemiops_pme_energy_train_vjp(
    positions,
    charges,
    box_length_A: float,
    accuracy: float,
    real_space_cutoff_A: float | None,
):
    """Differentiable PME energy for jitted train steps (callback + force VJP)."""
    import jax
    import jax.numpy as jnp

    box = float(box_length_A)
    acc = float(accuracy)
    cut = None if real_space_cutoff_A is None else float(real_space_cutoff_A)

    def _host(args):
        pos_h, chg_h = args
        # Never call concrete CUDA JAX here in-process: the parent
        # jit_train_step still holds the GPU XLA executor → deadlock.
        e_h, f_h = _nvalchemiops_pme_energy_forces_for_train_callback(
            pos_h,
            chg_h,
            box_length_A=box,
            accuracy=acc,
            real_space_cutoff_A=cut,
        )
        return (
            np.asarray(e_h, dtype=np.float64),
            np.asarray(f_h, dtype=np.float64),
        )

    @jax.custom_vjp
    def _energy(pos, chg):
        e, _f = jax.pure_callback(
            _host,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct(pos.shape, jnp.float64),
            ),
            (pos, chg),
            vmap_method="sequential",
        )
        return jnp.asarray(e, dtype=pos.dtype)

    def _energy_fwd(pos, chg):
        e, f = jax.pure_callback(
            _host,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct(pos.shape, jnp.float64),
            ),
            (pos, chg),
            vmap_method="sequential",
        )
        e = jnp.asarray(e, dtype=pos.dtype)
        f = jnp.asarray(f, dtype=pos.dtype)
        return e, (f, chg)

    def _energy_bwd(res, g):
        f_phys, chg = res
        # nvalchemiops forces are physical F = -∂E/∂R → ∂E/∂R = -F.
        return (-g * f_phys, jnp.zeros_like(chg))

    _energy.defvjp(_energy_fwd, _energy_bwd)
    return _energy(positions, charges)


def compute_nvalchemiops_pme_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float | None = None,
) -> LongRangeInteractionResult:
    """Full periodic Coulomb via nvalchemiops JAX PME.

    ``nvalchemiops`` returns electrostatic energies in e²/Å-like units for unit
    charges and Å coordinates; MMML converts to CHARMM kcal/mol with
    ``CHARMM_COULOMB_KCAL``.
    """
    energy, forces = _nvalchemiops_pme_energy_forces_concrete(
        np.asarray(positions_A, dtype=np.float64),
        np.asarray(charges_e, dtype=np.float64),
        box_length_A=float(box_length_A),
        accuracy=accuracy,
    )
    return LongRangeInteractionResult(
        energy_kcalmol=float(energy),
        forces_kcalmol_A=np.asarray(forces, dtype=np.float64),
    )


def compute_native_ewald_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
) -> LongRangeInteractionResult:
    """Full periodic Coulomb via the jit-native Ewald summation (ewald_native.py).

    Pure JAX (no external PME library, no CUDA requirement) -- the same
    operator ``lr_solver="ewald"`` trains against
    (:func:`mmml.models.ewald_hybrid_coulomb.hybrid_ewald_coulomb_energy`), so
    train and MD stay consistent. Called eagerly from the CHARMM callback
    (outside any jax.jit trace), so forces come from a plain ``jax.grad`` --
    no pure_callback/custom_vjp bridging needed here.
    """
    import jax
    import jax.numpy as jnp

    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos = jnp.asarray(positions_A, dtype=jnp.float64)
    chg = jnp.asarray(charges_e, dtype=jnp.float64)
    mol_id = jnp.zeros(pos.shape[0], dtype=jnp.int32)  # all atoms real, none padded

    def _e(p):
        return hybrid_ewald_coulomb_energy(
            p, mol_id, chg,
            box_length_A=float(box_length_A),
            accuracy=float(accuracy),
            real_space_cutoff_A=real_space_cutoff_A,
        )

    energy, grad = jax.value_and_grad(_e)(pos)
    return LongRangeInteractionResult(
        energy_kcalmol=float(energy),
        forces_kcalmol_A=np.asarray(-grad, dtype=np.float64),
    )


def compute_jax_pme_lj_dispersion(
    positions_A: np.ndarray,
    c6_sqrt: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
) -> LongRangeInteractionResult:
    """Attractive r⁻⁶ LJ tail via jax-pme (prefactor −1, exponent 6)."""
    return compute_jax_pme_power_law(
        positions_A,
        c6_sqrt,
        box_length_A=box_length_A,
        method=method,
        sr_cutoff_A=sr_cutoff_A,
        exponent=6,
        prefactor=DEFAULT_JAX_PME_LJ_PREFACTOR,
    )


class LongRangeCoulombSolver(Protocol):
    """Protocol for k-space / full-range Coulomb supplements."""

    name: str

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeInteractionResult:
        """Return total electrostatic energy and forces for all atoms."""


class MicOnlySolver:
    """Placeholder: pair-listed MIC Coulomb remains in ``mm_energy_forces.py``."""

    name = "mic"

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeInteractionResult:
        raise NotImplementedError(
            "mic backend evaluates Coulomb inside build_mm_energy_forces_fn; "
            "no separate long-range pass"
        )


def resolve_jax_pme_method(method: str | None = None) -> JaxPmeMethod:
    """Resolve jax-pme method from argument or ``JAX_PME_METHOD`` env."""
    raw = (method or os.environ.get("JAX_PME_METHOD", "ewald")).strip().lower()
    if raw in ("ewald", "pme", "p3m"):
        return raw  # type: ignore[return-value]
    raise ValueError(f"jax_pme method must be ewald|pme|p3m; got {method!r}")


def box_length_from_cell(cell: np.ndarray) -> float:
    """Cubic orthorhombic edge length (Å) from a 3-vector or 3×3 cell."""
    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.shape[0] == 3:
            return float(arr[0])
        return float(arr.reshape(-1)[0])
    return float(arr[0, 0])


class JaxPmeLongRangeSolver:
    """jax-pme Ewald / PME / P3M Coulomb backend."""

    name = "jax_pme"

    def __init__(
        self,
        *,
        method: str | None = None,
        sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    ) -> None:
        self._method = resolve_jax_pme_method(method)
        self._sr_cutoff_A = float(sr_cutoff_A)

    @property
    def method(self) -> JaxPmeMethod:
        return self._method

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        return compute_jax_pme_coulomb(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            method=self._method,
            sr_cutoff_A=self._sr_cutoff_A,
        )


class ScaFaCoSLongRangeSolver:
    name = "scafacos"

    def __init__(
        self,
        *,
        method: str | None = None,
        parameters: dict[str, str | float | int] | None = None,
    ) -> None:
        self._method = (method or os.environ.get("SCAFACOS_METHOD", "p2nfft")).strip()
        self._parameters = dict(parameters or {})

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        from mmml.interfaces.scafacosInterface.scafacos_session import compute_scafacos_coulomb

        result = compute_scafacos_coulomb(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            method=self._method,
            parameters=self._parameters or None,
        )
        return LongRangeCoulombResult(
            energy_kcalmol=result.energy_kcalmol,
            forces_kcalmol_A=result.forces_kcalmol_A,
        )


class NvalchemiopsPmeLongRangeSolver:
    name = "nvalchemiops_pme"

    def __init__(self, *, accuracy: float | None = None) -> None:
        self._accuracy = None if accuracy is None else float(accuracy)

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        return compute_nvalchemiops_pme_coulomb(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            accuracy=self._accuracy,
        )


def create_lr_solver(requested: str | None = None) -> LongRangeCoulombSolver:
    """Instantiate the resolved long-range Coulomb backend."""
    chosen = pick_lr_solver(requested)
    if chosen == "scafacos":
        return ScaFaCoSLongRangeSolver()
    if chosen == "nvalchemiops_pme":
        return NvalchemiopsPmeLongRangeSolver()
    if chosen == "jax_pme":
        return JaxPmeLongRangeSolver()
    return MicOnlySolver()


def describe_lr_solver(requested: str | None = None) -> str:
    """Human-readable summary for logs (chosen backend + availability)."""
    chosen = pick_lr_solver(requested)
    parts = [f"lr_solver={chosen}"]
    parts.append(f"scafacos={'yes' if have_scafacos() else 'no'}")
    parts.append(f"jax_pme={'yes' if have_jax_pme() else 'no'}")
    parts.append(f"nvalchemiops_pme={'yes' if have_nvalchemiops_pme() else 'no'}")
    return ", ".join(parts)


def collect_lr_solver_mapping(
    *,
    lr_solver: str | None = None,
    jax_pme_method: str | None = None,
    jax_pme_sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    jax_pme_dispersion: bool | None = None,
    scafacos_method: str | None = None,
    mm_nonbond_mode: str = "jax_mic",
    do_mm: bool = True,
    periodic_charmm_vdw: bool = True,
) -> dict[str, str]:
    """Key/value rows for the Hybrid ML/MM setup long-range Coulomb section."""
    requested = resolve_lr_solver(lr_solver)
    chosen = pick_lr_solver(lr_solver)
    mode = str(mm_nonbond_mode or "jax_mic").strip().lower()
    if mode in ("mic", "jax"):
        mode = "jax_mic"
    periodic_external = mode == "periodic_external"

    mapping: dict[str, str] = {
        "mm_nonbond_mode": mode,
        "lr_solver": chosen,
        "jax_pme_pkg": "yes" if have_jax_pme() else "no",
        "nvalchemiops_pme_pkg": "yes" if have_nvalchemiops_pme() else "no",
        "scafacos_lib": "yes" if have_scafacos() else "no",
    }
    if requested != chosen:
        mapping["lr_solver_requested"] = requested

    if periodic_external:
        if chosen == "jax_pme":
            active = "jax_pme"
            mapping["lr_solver_active"] = active
            mapping["coulomb_mode"] = "jax-pme full-box Coulomb (MLpot callback)"
            mapping["jax_pme_method"] = resolve_jax_pme_method(jax_pme_method)
            mapping["jax_pme_sr_cutoff_Å"] = f"{float(jax_pme_sr_cutoff_A):.1f}"
        elif chosen == "nvalchemiops_pme":
            active = "nvalchemiops_pme"
            mapping["lr_solver_active"] = active
            mapping["coulomb_mode"] = "nvalchemiops PME full-box Coulomb (MLpot callback)"
            mapping["nvalchemiops_pme_accuracy"] = (
                f"{_nvalchemiops_pme_accuracy():.1e}"
            )
        elif chosen == "scafacos":
            active = "scafacos"
            mapping["lr_solver_active"] = active
            mapping["coulomb_mode"] = "ScaFaCoS full-box Coulomb (MLpot callback)"
            mapping["scafacos_method"] = str(
                scafacos_method or os.environ.get("SCAFACOS_METHOD", "ewald")
            ).strip()
        else:
            active = chosen
            mapping["lr_solver_active"] = active
            mapping["coulomb_mode"] = f"{chosen} (unexpected for periodic_external)"
        mapping["charmm_vdw"] = "CHARMM IMAGE" if periodic_charmm_vdw else "off"
        return mapping

    if not do_mm:
        mapping["lr_solver_active"] = "—"
        mapping["coulomb_mode"] = "none (ML only; LR settings inactive)"
        if chosen == "nvalchemiops_pme":
            mapping["note"] = "nvalchemiops_pme applies only with periodic_external"
        elif chosen not in ("mic", "jax_pme"):
            mapping["note"] = (
                f"lr_solver={chosen} applies only with periodic_external or doMM=true"
            )
        return mapping

    active = "jax_pme" if chosen == "jax_pme" else "mic"
    mapping["lr_solver_active"] = active
    if chosen == "scafacos":
        mapping["note"] = "scafacos not wired in jax_mic; using truncated MIC"
    if chosen == "nvalchemiops_pme":
        mapping["note"] = "nvalchemiops_pme not wired in jax_mic; using truncated MIC"
    if active == "mic":
        mapping["coulomb_mode"] = "truncated MIC (switched MM pair loop)"
    else:
        mapping["coulomb_mode"] = "jax-pme k-space + pair SR (switched MM)"
        mapping["jax_pme_method"] = resolve_jax_pme_method(jax_pme_method)
        mapping["jax_pme_sr_cutoff_Å"] = f"{float(jax_pme_sr_cutoff_A):.1f}"
        mapping["jax_pme_dispersion"] = (
            "on" if resolve_jax_pme_dispersion(jax_pme_dispersion) else "off"
        )
    return mapping
