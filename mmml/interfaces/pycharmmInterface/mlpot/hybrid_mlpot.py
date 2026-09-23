"""MLpot callback via monomer/dimer PhysNet batches (``setup_calculator`` path)."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
from mmml.interfaces.pycharmmInterface.cutoffs import (
    CutoffParameters,
    cutoff_parameters_from_args,
)
from mmml.interfaces.pycharmmInterface.ml_dtypes import as_ml_array, resolve_ml_compute_dtype
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol, setup_calculator
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import (
    resolve_ml_batch_size,
    resolve_mlpot_mm_skin_A,
)
from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import (
    failstop_calculate_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import physnet_ml_atomic_numbers
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import resolve_ml_gpu_count
from mmml.interfaces.pycharmmInterface.jax_device_policy import (
    jax_cpu_until_mlpot_registered,
    mlpot_jax_device_context,
)
from mmml.utils.jax_gpu_warmup import ensure_xla_gpu_warmed

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import MetatomicMlpotModel

__all__ = [
    "resolve_ml_batch_size",
    "resolve_mm_pair_source",
    "MmPairSource",
    "DecomposedMlpotCalculator",
    "DecomposedMlpotModel",
    "build_decomposed_mlpot_model",
    "materialize_deferred_mlpot_jax_before_sd",
    "charmm_mlpot_sd_jax_cpu_guard",
    "warmup_decomposed_mlpot",
]

_DUMMY_MM_PAIR_IDX = jnp.zeros((1, 2), dtype=jnp.int32)
_DUMMY_MM_PAIR_MASK = jnp.zeros((1,), dtype=jnp.bool_)

MmPairSource = Literal["jax", "charmm_callback"]
_DEFAULT_MM_PAIR_SOURCE: MmPairSource = "charmm_callback"
# forward_fn(..., ml_eval_chunks=<this>) means "use the owner's chunk budget".
_BUDGET_DEFAULT = object()


class _CallbackPairListUnavailable(RuntimeError):
    """Raised when CHARMM callback pair lists are unusable.

    During setup (guard disarmed) ``calculate_charmm`` still returns 0.0 so
    ``assert_mlpot_user_active`` can rebind and rebuild. Once that check arms
    the guard, this exception propagates into
    ``callback_failstop.fail_closed_callback`` and the process exits 86.
    """


ALLOW_MISSING_CALLBACK_PAIRS_ENV = "MMML_MLPOT_ALLOW_MISSING_CALLBACK_PAIRS"
"""Test-only: ``1`` restores the old zero-energy return on missing pair lists."""

ALLOW_PERIODIC_COULOMB_FAILURE_ENV = "MMML_MLPOT_ALLOW_PERIODIC_COULOMB_FAILURE"
"""Test-only: ``1`` continues with an ML-only USER term if periodic Coulomb fails."""


def _callback_opt_out(env_name: str) -> bool:
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_mm_pair_source(
    args: Any | None = None,
    *,
    all_ml_jax_mic: bool = False,
    all_ml_pbc_jax_mic: bool | None = None,
) -> MmPairSource:
    """Resolve MM pair provider for decomposed MLpot (``jax`` vs Fortran callback).

    Default is ``charmm_callback`` (Fortran ``idxu/idxv`` primary pairs). All-ML
    ``jax_mic`` hybrids (vacuum or PBC) zero CHARMM ELEC/VDW via energy policy, so
    the Fortran primary list is empty — those runs default to JAX neighbor rebuild.
    Set ``MMML_MM_PAIR_SOURCE=jax`` or ``--mm-pair-source jax`` to force Vesin/cell-list.
    """
    if all_ml_pbc_jax_mic is not None:
        all_ml_jax_mic = bool(all_ml_pbc_jax_mic) or bool(all_ml_jax_mic)
    if args is not None:
        src = getattr(args, "mm_pair_source", None)
        if src is not None:
            norm = str(src).strip().lower()
            if norm in ("charmm_callback", "callback", "charmm"):
                return "charmm_callback"
            if norm == "jax":
                return "jax"
            raise ValueError(f"mm_pair_source must be jax or charmm_callback; got {src!r}")
    raw = os.environ.get("MMML_MM_PAIR_SOURCE", "").strip().lower()
    if raw == "jax":
        return "jax"
    if raw in ("charmm_callback", "callback", "charmm"):
        return "charmm_callback"
    if raw:
        raise ValueError(
            f"MMML_MM_PAIR_SOURCE must be jax or charmm_callback; got {raw!r}"
        )
    if all_ml_jax_mic:
        return "jax"
    return _DEFAULT_MM_PAIR_SOURCE


def _monomer_offsets_from_atoms_per_monomer(
    atoms_per_monomer: Sequence[int],
) -> np.ndarray:
    offsets = [0]
    for n in atoms_per_monomer:
        offsets.append(offsets[-1] + int(n))
    return np.asarray(offsets, dtype=np.int32)


def _box_cache_key(box: jnp.ndarray | None) -> bool:
    return box is not None


def _box_numpy_for_update(box: jnp.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    arr = np.asarray(box)
    if arr.ndim == 1:
        side = float(arr[0])
    else:
        side = float(arr[0, 0])
    return np.asarray([side, side, side], dtype=np.float64)


def _print_setup_calculator_factory_summary(
    factory: Any,
    *,
    checkpoint: Path,
    n_monomers: int,
    atoms_per_monomer: Sequence[int],
    do_ml: bool,
    do_mm: bool,
    do_ml_dimer: bool,
    cutoff_params: CutoffParameters,
    max_atoms_per_system: int,
    ml_batch_size: Optional[int],
    ml_gpu_count: int,
    ml_max_active_dimers: Optional[int],
    cell: Union[float, bool],
) -> None:
    """Log hybrid factory defaults after ``setup_calculator`` returns."""
    from mmml.utils.rich_report import emit_factory_summary

    cp = cutoff_params
    comp = getattr(cp, "complementary_handoff", True)
    emit_factory_summary(
        "Decomposed MLpot factory",
        {
            "factory": getattr(factory, "__name__", type(factory).__name__),
            "module": getattr(factory, "__module__", "?"),
            "model_restart_path": str(checkpoint),
            "n_monomers": n_monomers,
            "atoms_per_monomer": list(atoms_per_monomer),
            "MAX_ATOMS_PER_SYSTEM": max_atoms_per_system,
            "doML": do_ml,
            "doMM": do_mm,
            "doML_dimer": do_ml_dimer,
            "ml_switch_width": cp.ml_switch_width,
            "mm_switch_on": cp.mm_switch_on,
            "mm_switch_width": cp.mm_switch_width,
            "complementary_handoff": comp,
            "ml_batch_size": ml_batch_size,
            "ml_gpu_count": ml_gpu_count,
            "ml_max_active_dimers": ml_max_active_dimers,
            "cell": repr(cell),
        },
    )


class DecomposedMlpotCalculator:
    """CHARMM MLpot callback using padded monomer/dimer PhysNet evaluations."""

    def __init__(
        self,
        spherical_fn: Any,
        cutoff_params: CutoffParameters,
        n_monomers: int,
        atomic_numbers: np.ndarray,
        cell: Union[float, bool] = False,
        do_mm: bool = True,
        get_update_fn: Any | None = None,
        ml_compute_dtype: str | None = None,
        *,
        do_ml: bool = True,
        do_ml_dimer: bool = True,
        spatial_mpi: bool = False,
        atoms_per_monomer: Sequence[int] | None = None,
        periodic_mm_config: Any | None = None,
        mm_pair_source: MmPairSource = _DEFAULT_MM_PAIR_SOURCE,
        mm_r_min: float | None = None,
        mm_pair_capacity_hint: int | None = None,
        ml_atom_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.spherical_fn = spherical_fn
        self.cutoff_params = cutoff_params
        self.n_monomers = int(n_monomers)
        self.do_mm = bool(do_mm)
        self.do_ml = bool(do_ml)
        self.do_ml_dimer = bool(do_ml_dimer)
        self._periodic_mm_config = periodic_mm_config
        self._get_update_fn = get_update_fn
        self._ml_compute_dtype = ml_compute_dtype
        self._spatial_mpi = bool(spatial_mpi)
        if atoms_per_monomer is None:
            apm = max(1, len(atomic_numbers) // max(1, int(n_monomers)))
            self._atoms_per_monomer = [apm] * int(n_monomers)
        else:
            self._atoms_per_monomer = [int(x) for x in atoms_per_monomer]
        self.atomic_numbers = np.asarray(
            physnet_ml_atomic_numbers(atomic_numbers), dtype=np.int32
        )
        if ml_atom_indices is None:
            self._ml_atom_indices: np.ndarray | None = None
        else:
            self._ml_atom_indices = np.asarray(ml_atom_indices, dtype=int).reshape(-1)
        self.ev2kcal = float(ev2kcalmol)
        self._cell = float(cell) if cell else False
        self.last_ml_forces: np.ndarray | None = None
        self._spherical_forward_fn: Any | None = None
        self._forward_cache_key: tuple[Any, ...] | None = None
        self._mm_pair_source: MmPairSource = str(mm_pair_source)  # type: ignore[assignment]
        self._mm_r_min = float(mm_r_min) if mm_r_min is not None else None
        self._callback_pair_warned = False
        self._mm_pair_capacity: int | None = (
            int(mm_pair_capacity_hint) if mm_pair_capacity_hint else None
        )

    def _mm_pair_pad_capacity(self) -> int:
        cap = getattr(self, "_mm_pair_capacity", None)
        return max(int(cap), 1) if cap is not None else 1

    def _resolve_ml_callback_slice(self, n_charmm: int) -> np.ndarray:
        """PSF indices of the ML region for this callback (all-ML → ``0..n-1``).

        Mechanical embedding registers a model sized to the ML solute only while
        CHARMM still passes full-system coordinates (``Natom``). Without this
        slice, the hybrid forward is evaluated on solvent atoms with the wrong
        ``atoms_per_monomer`` layout and USER/GRMS explode.
        """
        expected = int(self.atomic_numbers.shape[0])
        stored = getattr(self, "_ml_atom_indices", None)
        if stored is not None:
            ml_idx = np.asarray(stored, dtype=int).reshape(-1)
        elif int(n_charmm) == expected:
            ml_idx = np.arange(expected, dtype=int)
        else:
            raise RuntimeError(
                f"Decomposed MLpot: CHARMM Natom={int(n_charmm)} != model "
                f"n_ml={expected} and ml_atom_indices was not set. Partial ML "
                "(ml_resnames) requires get_pycharmm_calculator(ml_atom_indices=...)."
            )
        if ml_idx.size != expected:
            raise RuntimeError(
                f"Decomposed MLpot: ml_atom_indices length {ml_idx.size} != "
                f"model n_ml={expected}"
            )
        if ml_idx.size == 0:
            raise RuntimeError("Decomposed MLpot: empty ml_atom_indices")
        if int(ml_idx.min()) < 0 or int(ml_idx.max()) >= int(n_charmm):
            raise RuntimeError(
                f"Decomposed MLpot: ml_atom_indices out of range for "
                f"Natom={int(n_charmm)} (min={int(ml_idx.min())}, "
                f"max={int(ml_idx.max())})"
            )
        return ml_idx

    def _invalidate_forward_jit_cache(self) -> None:
        owner = self._grad_cache_owner()
        owner._spherical_forward_fn = None
        owner._forward_cache_key = None

    def _note_mm_pair_capacity(self, pair_idx: Any) -> None:
        n = int(np.asarray(pair_idx).shape[0])
        prev = getattr(self, "_mm_pair_capacity", None)
        if prev is None or n > int(prev):
            self._mm_pair_capacity = n
            if prev is not None and n > int(prev):
                self._invalidate_forward_jit_cache()

    def _ensure_mm_pair_capacity_from_update_fn(
        self,
        pos: np.ndarray,
        box: jnp.ndarray | None,
    ) -> None:
        """Seed stable pair-buffer capacity from JAX rebuild (matches warmup JIT shapes)."""
        if getattr(self, "_mm_pair_capacity", None) is not None:
            return
        if not self.do_mm or self._get_update_fn is None:
            return
        update_fn = getattr(self, "_cached_update_fn", None)
        if update_fn is None:
            update_fn = self._get_update_fn(pos, self.cutoff_params, box=box)
            self._cached_update_fn = update_fn
        if update_fn is None:
            return
        box_np = _box_numpy_for_update(box)
        if box_np is not None:
            pair_idx, pair_mask = update_fn(pos, box=box_np)
        else:
            pair_idx, pair_mask = update_fn(pos)
        if pair_idx is not None and pair_mask is not None:
            self._note_mm_pair_capacity(pair_idx)

    def _grad_cache_owner(self) -> DecomposedMlpotCalculator | DecomposedMlpotModel:
        parent = getattr(self, "_parent_model", None)
        return parent if parent is not None else self

    def _requires_callback_pbc_box(self) -> bool:
        """True when the callback must query a live CHARMM box even if no cell was cached."""
        parent = getattr(self, "_parent_model", None)
        active = getattr(parent, "_jax_pme_lr_active", None)
        if callable(active) and bool(active()):
            return True
        cfg = getattr(self, "_periodic_mm_config", None)
        uses_jax_pme = getattr(cfg, "uses_jax_pme", False)
        return bool(uses_jax_pme() if callable(uses_jax_pme) else uses_jax_pme)

    def _callback_box_resolution_inputs(
        self,
    ) -> tuple[float | None, Path | None]:
        """Fallback side and restart for jax-pme / MIC when pbound is inactive."""
        fallback: float | None = float(self._cell) if self._cell else None
        restart_path = getattr(self, "_npt_restart_read", None)
        parent = getattr(self, "_parent_model", None)
        if parent is not None:
            if fallback is None:
                parent_cell = getattr(parent, "_cell", False)
                if parent_cell:
                    fallback = float(parent_cell)
            if fallback is None:
                charmm_box = getattr(parent, "_charmm_box_side_A", None)
                if charmm_box is not None and float(charmm_box) > 0.0:
                    fallback = float(charmm_box)
            if restart_path is None:
                restart_path = getattr(parent, "_npt_restart_read", None)
        if restart_path is not None:
            restart_path = Path(restart_path)
            if not restart_path.is_file():
                restart_path = None
        return fallback, restart_path

    def _get_spherical_forward_fn(
        self,
        *,
        n_atoms: int,
        atomic_numbers_jax: jnp.ndarray,
        box_jax: jnp.ndarray | None,
    ) -> Any:
        """Return a cached ``jit`` forward eval (energy eV, forces eV/Å from ``out.forces``).

        Matches the ASE calculator path (``backprop=False``). ``jax.value_and_grad`` on the
        energy scalar can disagree with ``out.forces`` when sparse MM pair lists are used.

        Returns ``(energy, forces, n_active_dimers)``. The keyword ``ml_eval_chunks``
        (static) defaults to the owner's :class:`MlChunkBudget` when the factory
        exposes a sparse chunk layout, so the chunk loop has a compile-time trip
        count; ``calculate_charmm`` checks the returned count against it.
        """
        dtype = resolve_ml_compute_dtype(self._ml_compute_dtype)
        box_present = box_jax is not None
        cache_key = (
            int(n_atoms),
            int(self.n_monomers),
            bool(self.do_mm),
            bool(self.do_ml),
            bool(self.do_ml_dimer),
            dtype,
            box_present,
            bool(self._spatial_mpi),
        )
        owner = self._grad_cache_owner()
        if owner._forward_cache_key == cache_key and owner._spherical_forward_fn is not None:
            return owner._spherical_forward_fn

        spherical_fn = self.spherical_fn
        cutoff_params = self.cutoff_params
        n_monomers = self.n_monomers
        do_mm = self.do_mm
        do_ml = self.do_ml
        do_ml_dimer = self.do_ml_dimer
        from mmml.interfaces.pycharmmInterface.mlpot.ml_chunk_budget import (
            MlChunkBudget,
            ml_chunk_budget_enabled,
        )

        layout = getattr(spherical_fn, "ml_chunk_layout", None)
        budget = (
            MlChunkBudget(layout)
            if layout is not None and do_ml and do_ml_dimer and ml_chunk_budget_enabled()
            else None
        )
        owner._ml_chunk_budget = budget

        def _budget_chunks(ml_eval_chunks):
            if ml_eval_chunks is _BUDGET_DEFAULT:
                return budget.current if budget is not None else None
            return ml_eval_chunks

        if box_present:

            def forward_fn(
                positions: jnp.ndarray,
                box: jnp.ndarray,
                mm_pair_idx: jnp.ndarray,
                mm_pair_mask: jnp.ndarray,
                use_mm_pairs: bool,
                spatial_monomer_indices: jnp.ndarray,
                spatial_dimer_indices: jnp.ndarray,
                use_spatial: bool,
                ml_eval_chunks: int | None = None,
                ml_dimer_candidates: jnp.ndarray | None = None,
            ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                kwargs: dict[str, Any] = dict(
                    positions=positions,
                    atomic_numbers=atomic_numbers_jax,
                    n_monomers=n_monomers,
                    cutoff_params=cutoff_params,
                    doML=do_ml,
                    doMM=do_mm,
                    doML_dimer=do_ml_dimer,
                    box=box,
                )
                if use_mm_pairs:
                    kwargs["mm_pair_idx"] = mm_pair_idx
                    kwargs["mm_pair_mask"] = mm_pair_mask
                if use_spatial:
                    kwargs["spatial_monomer_indices"] = spatial_monomer_indices
                    kwargs["spatial_dimer_indices"] = spatial_dimer_indices
                if ml_eval_chunks is not None:
                    kwargs["ml_eval_chunks"] = ml_eval_chunks
                if ml_dimer_candidates is not None:
                    kwargs["ml_dimer_candidates"] = ml_dimer_candidates
                out = spherical_fn(**kwargs)
                return (
                    jnp.reshape(out.energy, (-1,))[0],
                    out.forces,
                    jnp.asarray(getattr(out, "ml_n_active_dimers", -1), dtype=jnp.int32),
                )

            fn = jax.jit(
                forward_fn, static_argnums=(4, 7), static_argnames=("ml_eval_chunks",)
            )

            def forward_vir_fn(positions, box, *rest, ml_eval_chunks=None, ml_dimer_candidates=None):
                """Forward plus dE/d(box) at fixed positions (CPT strain virial, strain_virial.py)."""

                def energy_aux(box_arg):
                    e, f, n_act = forward_fn(
                        positions, box_arg, *rest,
                        ml_eval_chunks=ml_eval_chunks,
                        ml_dimer_candidates=ml_dimer_candidates,
                    )
                    return e, (f, n_act)

                (e, (f, n_act)), dE_dbox = jax.value_and_grad(energy_aux, has_aux=True)(box)
                return e, f, n_act, dE_dbox

            fn_vir = jax.jit(
                forward_vir_fn, static_argnums=(4, 7), static_argnames=("ml_eval_chunks",)
            )

            def _live_box():
                # Owner-level box first: this wrapper outlives the calculator that built it.
                current = getattr(owner, "_live_callback_box", None)
                if current is None:
                    current = getattr(self, "_current_box", None)
                return box_jax if current is None else current

            def wrapper(
                positions,
                mm_pair_idx,
                mm_pair_mask,
                use_mm_pairs,
                spatial_monomer_indices,
                spatial_dimer_indices,
                use_spatial,
                ml_eval_chunks=_BUDGET_DEFAULT,
                ml_dimer_candidates=None,
            ):
                current_box = _live_box()
                return fn(
                    positions,
                    current_box,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                    spatial_monomer_indices,
                    spatial_dimer_indices,
                    use_spatial,
                    ml_eval_chunks=_budget_chunks(ml_eval_chunks),
                    ml_dimer_candidates=ml_dimer_candidates,
                )

            def wrapper_vir(
                positions,
                mm_pair_idx,
                mm_pair_mask,
                use_mm_pairs,
                spatial_monomer_indices,
                spatial_dimer_indices,
                use_spatial,
                ml_eval_chunks=_BUDGET_DEFAULT,
                ml_dimer_candidates=None,
            ):
                current_box = _live_box()
                return fn_vir(
                    positions,
                    current_box,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                    spatial_monomer_indices,
                    spatial_dimer_indices,
                    use_spatial,
                    ml_eval_chunks=_budget_chunks(ml_eval_chunks),
                    ml_dimer_candidates=ml_dimer_candidates,
                )

            owner._spherical_forward_vir_fn = wrapper_vir
            owner._spherical_forward_fn = wrapper
        else:

            def forward_fn(
                positions: jnp.ndarray,
                mm_pair_idx: jnp.ndarray,
                mm_pair_mask: jnp.ndarray,
                use_mm_pairs: bool,
                spatial_monomer_indices: jnp.ndarray,
                spatial_dimer_indices: jnp.ndarray,
                use_spatial: bool,
                ml_eval_chunks: int | None = None,
                ml_dimer_candidates: jnp.ndarray | None = None,
            ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                kwargs: dict[str, Any] = dict(
                    positions=positions,
                    atomic_numbers=atomic_numbers_jax,
                    n_monomers=n_monomers,
                    cutoff_params=cutoff_params,
                    doML=do_ml,
                    doMM=do_mm,
                    doML_dimer=do_ml_dimer,
                )
                if use_mm_pairs:
                    kwargs["mm_pair_idx"] = mm_pair_idx
                    kwargs["mm_pair_mask"] = mm_pair_mask
                if use_spatial:
                    kwargs["spatial_monomer_indices"] = spatial_monomer_indices
                    kwargs["spatial_dimer_indices"] = spatial_dimer_indices
                if ml_eval_chunks is not None:
                    kwargs["ml_eval_chunks"] = ml_eval_chunks
                if ml_dimer_candidates is not None:
                    kwargs["ml_dimer_candidates"] = ml_dimer_candidates
                out = spherical_fn(**kwargs)
                return (
                    jnp.reshape(out.energy, (-1,))[0],
                    out.forces,
                    jnp.asarray(getattr(out, "ml_n_active_dimers", -1), dtype=jnp.int32),
                )

            fn_nobox = jax.jit(
                forward_fn, static_argnums=(3, 6), static_argnames=("ml_eval_chunks",)
            )

            def wrapper_nobox(*args, ml_eval_chunks=_BUDGET_DEFAULT, ml_dimer_candidates=None):
                return fn_nobox(
                    *args,
                    ml_eval_chunks=_budget_chunks(ml_eval_chunks),
                    ml_dimer_candidates=ml_dimer_candidates,
                )

            owner._spherical_forward_fn = wrapper_nobox

        owner._forward_cache_key = cache_key
        return owner._spherical_forward_fn

    def _get_value_and_grad_fn(
        self,
        *,
        n_atoms: int,
        atomic_numbers_jax: jnp.ndarray,
        box_jax: jnp.ndarray | None,
    ) -> Any:
        """Deprecated alias retained for tests; prefer :meth:`_get_spherical_forward_fn`."""
        return self._get_spherical_forward_fn(
            n_atoms=n_atoms,
            atomic_numbers_jax=atomic_numbers_jax,
            box_jax=box_jax,
        )

    def _resolve_ml_dimer_candidates(
        self,
        pos: np.ndarray,
        box: jnp.ndarray | None,
        *,
        use_spatial: bool = False,
    ) -> dict[str, Any]:
        """Forward kwargs for the centroid Verlet list of sparse ML dimers.

        Empty (all-pairs selection in-graph) unless the factory attached a
        :class:`CentroidDimerNeighborList` (``setup_calculator(ml_dimer_centroid_nl=...)``,
        env ``MMML_ML_DIMER_CENTROID_NL``) and this step uses the sparse batch.
        """
        from mmml.interfaces.pycharmmInterface.mlpot.dimer_centroid_nl import (
            CentroidDimerNeighborList,
        )

        if use_spatial or not (self.do_ml and self.do_ml_dimer):
            return {}
        nl = getattr(self.spherical_fn, "dimer_centroid_nl", None)
        if not isinstance(nl, CentroidDimerNeighborList):
            return {}
        cand = nl.update(pos, box=_box_numpy_for_update(box))
        from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
            get_mlpot_profile_stats,
            mlpot_profiling_enabled,
        )

        if mlpot_profiling_enabled():
            recorder = getattr(get_mlpot_profile_stats(), "record_dimer_centroid_nl", None)
            if recorder is not None:
                recorder(nl.stats())
        return {"ml_dimer_candidates": cand}

    def _resolve_mm_pairs(
        self,
        pos: np.ndarray,
        box: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        if not self.do_mm or self._get_update_fn is None:
            return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
        update_fn = getattr(self, "_cached_update_fn", None)
        if update_fn is None:
            update_fn = self._get_update_fn(pos, self.cutoff_params, box=box)
            self._cached_update_fn = update_fn
        if update_fn is None:
            return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
        box_np = _box_numpy_for_update(box)
        if box_np is not None:
            mm_pair_idx, mm_pair_mask = update_fn(pos, box=box_np)
        else:
            mm_pair_idx, mm_pair_mask = update_fn(pos)
        if mm_pair_idx is None or mm_pair_mask is None:
            if self._cell or box is not None:
                raise RuntimeError(
                    "PBC MM neighbor update returned no pair list; "
                    "cannot evaluate switched MM without mm_pair_idx/mm_pair_mask."
                )
            return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
        self._note_mm_pair_capacity(mm_pair_idx)
        get_stats = getattr(update_fn, "get_stats", None)
        if get_stats is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
                get_mlpot_profile_stats,
                mlpot_profiling_enabled,
            )

            if mlpot_profiling_enabled():
                get_mlpot_profile_stats().record_mm_pair_stats(get_stats())
        return jnp.asarray(mm_pair_idx), jnp.asarray(mm_pair_mask), True

    def _resolve_mm_pairs_from_callback(
        self,
        idxu,
        idxv,
        idxup,
        idxvp,
        *,
        natom: int,
        nmlmmp: int,
        pos: np.ndarray,
        box: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        from mmml.interfaces.pycharmmInterface.nl_reference import (
            apply_mm_pair_filters,
            callback_mlmm_pairs_to_half_set,
            callback_pairs_to_padded_arrays,
            filter_pairs_under_cutoff,
            inter_monomer_pair_set,
            monomer_id_from_offsets,
        )

        n_pairs = int(nmlmmp)
        if n_pairs <= 0:
            if not self.do_mm or int(self.n_monomers) <= 1:
                return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
            raise _CallbackPairListUnavailable(
                "Decomposed MLpot: charmm_callback returned zero ML/MM pairs while "
                "JAX MM is enabled for a multi-monomer system. Refusing to rebuild "
                "pairs inside the CHARMM callback because this indicates stale "
                "mlpot_update/list state before dynamics. Run CHARMM UPDATE/mlpot_update "
                "before dynamics, or opt into --mm-pair-source jax explicitly."
            )

        self._ensure_mm_pair_capacity_from_update_fn(pos, box)
        pad_capacity = self._mm_pair_pad_capacity()

        offsets = _monomer_offsets_from_atoms_per_monomer(self._atoms_per_monomer)
        mid = monomer_id_from_offsets(offsets, int(natom))
        raw = callback_mlmm_pairs_to_half_set(
            idxup,
            idxvp,
            nmlmmp=n_pairs,
            natom=int(natom),
        )
        inter = inter_monomer_pair_set(raw, monomer_id=mid)
        cutoff = float(self.cutoff_params.mm_switch_on) + float(
            self.cutoff_params.mm_switch_width
        )
        mm_r_min = self._mm_r_min
        if mm_r_min is None:
            handoff = float(self.cutoff_params.mm_switch_on) - float(
                self.cutoff_params.ml_switch_width
            )
            mm_r_min = handoff * 0.9 if handoff > 0.0 else 0.0
        box_np = _box_numpy_for_update(box)
        filtered = apply_mm_pair_filters(
            inter,
            monomer_id=mid,
            positions=pos,
            cell=box_np,
            mm_r_min=mm_r_min,
            monomer_offsets=offsets,
        )
        if box_np is not None:
            mic_filtered = filter_pairs_under_cutoff(filtered, pos, box_np, cutoff)
        else:
            mic_filtered = {
                pair
                for pair in filtered
                if float(np.linalg.norm(pos[pair[1]] - pos[pair[0]])) < cutoff
            }
        pair_idx, pair_mask = callback_pairs_to_padded_arrays(
            mic_filtered,
            min_capacity=pad_capacity,
        )
        self._note_mm_pair_capacity(pair_idx)
        return jnp.asarray(pair_idx), jnp.asarray(pair_mask), True

    def _check_ml_chunk_budget(self, budget, forward_fn, fwd_args, fwd_out, fwd_kwargs=None):
        """Validate this step's static chunk budget; re-run once if it was too small.

        The active-dimer count comes back with the forces (the host waits for
        those anyway), so this adds no device round trip inside the forward.
        """
        n_active = int(jax.device_get(fwd_out[2]))
        e_raw, forces_ev = fwd_out[0], fwd_out[1]
        if n_active < 0:  # dense or spatial batch this step: nothing was skipped
            return e_raw, forces_ev
        # Cap overflow drops interacting dimers; chunk growth cannot recover them.
        budget.raise_if_saturated(n_active)
        from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
            get_mlpot_profile_stats,
            mlpot_profiling_enabled,
        )

        if mlpot_profiling_enabled():
            get_mlpot_profile_stats().record_active_dimers(
                n_active,
                chunk_budget=budget.current,
                chunk_size=budget.layout.chunk_size,
                max_active_dimers=budget.layout.max_active_dimers,
            )
        if not budget.covers(n_active):
            # Rare (the budget keeps spare slots): grow and redo the step exactly.
            budget.update(n_active)
            self._ml_chunk_budget_reruns = getattr(self, "_ml_chunk_budget_reruns", 0) + 1
            print(
                f"MLpot chunk budget: {n_active} active dimers need "
                f"{budget.needed(n_active)} PhysNet chunks; re-evaluating the step "
                f"with {budget.current} (re-run #{self._ml_chunk_budget_reruns})",
                flush=True,
            )
            fwd_out = forward_fn(
                *fwd_args, ml_eval_chunks=budget.current, **(fwd_kwargs or {})
            )
            e_raw, forces_ev = fwd_out[0], fwd_out[1]
        budget.update(n_active)
        self._last_fwd_out = fwd_out
        return e_raw, forces_ev

    def _mlpot_eval_device_context(self):
        """CPU while MPI defer keeps the JAX factory off-GPU; else configured device."""
        parent = getattr(self, "_parent_model", None)
        if parent is not None and not getattr(parent, "_jax_on_gpu", True):
            return jax_cpu_until_mlpot_registered()
        return mlpot_jax_device_context()

    def _sync_callback_pbc_box(self):
        """Refresh ``self._cell`` from live CHARMM pbound before wrap / MIC.

        Under CPT the box changes every step. Wrapping with the previous
        ``_cell`` leaves a molecule split across the new primary cell (NPT
        ETOH: 21.4 Å raw extent, then the pair-list guard raises).
        """
        if not (self._cell or self._requires_callback_pbc_box()):
            self._set_live_callback_box(None)
            return None
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            cubic_box_matrix_from_side,
            resolve_mlpot_mic_box_side_A,
        )

        fallback_side_A, restart_path = self._callback_box_resolution_inputs()
        side, _ = resolve_mlpot_mic_box_side_A(
            fallback_side_A=fallback_side_A,
            restart_path=restart_path,
        )
        self._cell = side
        box = jnp.asarray(cubic_box_matrix_from_side(side))
        self._set_live_callback_box(box)
        return box

    def _set_live_callback_box(self, box) -> None:
        """Record the box for this ENER on this calculator and on the forward-cache owner.

        The jitted forward is cached on the owner and shared by every calculator the
        model registers; it reads the box from the owner, so a re-registered calculator
        (``refresh_mlpot_energy_and_grms``, CPT sub-chunks) cannot leave it on an old cell.
        """
        self._current_box = box
        self._grad_cache_owner()._live_callback_box = box

    def _maybe_rewrap_primary_cell_in_callback(
        self,
        pos: np.ndarray,
        n: int,
        x,
        y,
        z,
        *,
        box_side_A: float | None = None,
    ) -> np.ndarray:
        """Periodic copy of ``pos`` with each molecule's COM in the primary cell.

        Pure integer-lattice shifts per molecule, applied to a copy only. MIC
        energies/forces are unchanged by such shifts, so the evaluation sees
        tidy coordinates without touching CHARMM's state. This used to call
        ``rewrap_charmm_pbc_molecules`` and write the result into ``x/y/z``:
        its inward ``margin_A`` nudge is a real displacement, and editing the
        integrator's coordinates mid-step broke NVE (+289 kcal/mol in 0.25 ps
        on ETOH:181; conserved once removed). ``x``, ``y``, ``z`` are left
        untouched; post-SD recentering lives in ``dynamics._rewrap_mlpot_pbc_after_sd``.

        ``box_side_A`` is the live CHARMM cell (pbound). If omitted, ``self._cell``
        is used — callers that run under NPT must refresh that first.
        """
        del x, y, z
        L = float(box_side_A) if box_side_A is not None else (float(self._cell) if self._cell else 0.0)
        if L <= 0.0 or not self._atoms_per_monomer:
            return pos
        from mmml.interfaces.pycharmmInterface.mlpot.mc_density import monomer_offsets_from_atoms_per
        from mmml.utils.geometry_checks import wrap_monomers_primary_cell

        offsets = monomer_offsets_from_atoms_per(list(self._atoms_per_monomer))
        whole = np.asarray(pos[:n], dtype=np.float64)
        n_mol_atoms = int(offsets[-1])
        if n_mol_atoms <= n:
            # Rejoin molecules an engine wrapped atom by atom (jax-md, ASE wrap()) before
            # anything uses molecule COMs (this wrap, MM pair list, dimer candidates).
            # CHARMM hands over whole molecules, so this is a no-op in the callback.
            sizes = np.diff(offsets).astype(int)
            anchor = np.repeat(offsets[:-1], sizes)
            d = whole[:n_mol_atoms] - whole[anchor]
            whole = whole.copy()
            whole[:n_mol_atoms] -= np.round(d / L) * L
        # CHARMM frame is [-L/2, L/2]; wrap in [0, L) and shift back.
        wrapped = (
            wrap_monomers_primary_cell(whole + 0.5 * L, offsets, np.diag([L, L, L]))
            - 0.5 * L
        )
        out = np.array(pos, dtype=np.float64, copy=True)
        out[:n] = wrapped
        return out

    def evaluate_hybrid_ev(
        self,
        positions: np.ndarray,
        box_side_A: float,
        *,
        with_box_grad: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Hybrid energy (eV), forces (eV/Å) and optionally dE/d(box) exactly as the CHARMM callback computes them.

        Same steps as :meth:`calculate_charmm` without CHARMM: record the live box,
        rewrap each molecule's COM into the primary cell (a copy), resolve the MM pair
        list and the centroid dimer candidates, run the cached forward and honour the
        sparse chunk budget. For engines that need the callback's Hamiltonian off
        CHARMM (JAX-MD, ASE, strain finite differences). Cubic cells only.
        """
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

        if self.do_mm and self._mm_pair_source == "charmm_callback":
            raise RuntimeError("evaluate_hybrid_ev needs mm_pair_source='jax' (CHARMM pair lists exist only inside ENER)")
        side = float(box_side_A)
        pos_full = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        n = int(pos_full.shape[0])
        self._cell = side
        box = jnp.asarray(cubic_box_matrix_from_side(side))
        self._set_live_callback_box(box)
        pos_full = self._maybe_rewrap_primary_cell_in_callback(pos_full, n, None, None, None, box_side_A=side)
        ml_idx = self._resolve_ml_callback_slice(n)
        pos = pos_full[ml_idx]
        n_ml = int(ml_idx.size)
        with self._mlpot_eval_device_context():
            mm_pair_idx, mm_pair_mask, use_mm_pairs = self._resolve_mm_pairs(pos, box)
            positions_jax = as_ml_array(
                pos, dtype=resolve_ml_compute_dtype(getattr(self, "_ml_compute_dtype", None))
            )
            forward_fn = self._get_spherical_forward_fn(
                n_atoms=n_ml,
                atomic_numbers_jax=jnp.asarray(self.atomic_numbers[:n_ml]),
                box_jax=box,
            )
            if with_box_grad:
                forward_fn = getattr(self._grad_cache_owner(), "_spherical_forward_vir_fn", None)
                if forward_fn is None:
                    raise RuntimeError("box-gradient forward unavailable (no periodic box)")
            empty = jnp.zeros((0,), dtype=jnp.int32)
            fwd_args = (positions_jax, mm_pair_idx, mm_pair_mask, use_mm_pairs, empty, empty, False)
            fwd_kwargs = self._resolve_ml_dimer_candidates(pos, box, use_spatial=False)
            fwd_out = forward_fn(*fwd_args, **fwd_kwargs)
            self._last_fwd_out = fwd_out
            e_raw, forces_ev = fwd_out[0], fwd_out[1]
            budget = getattr(self._grad_cache_owner(), "_ml_chunk_budget", None)
            if budget is not None and len(fwd_out) > 2:
                e_raw, forces_ev = self._check_ml_chunk_budget(budget, forward_fn, fwd_args, fwd_out, fwd_kwargs)
            dE_dbox = (
                np.asarray(jax.device_get(self._last_fwd_out[3]), dtype=np.float64) if with_box_grad else None
            )
            e_ev = float(jax.device_get(e_raw))
            forces = np.zeros((n, 3), dtype=np.float64)
            forces[ml_idx] = np.asarray(jax.device_get(forces_ev), dtype=np.float64)
        return e_ev, forces, dE_dbox

    @failstop_calculate_charmm
    def calculate_charmm(
        self,
        Natom: int,
        Ntrans: int,
        Natim: int,
        idxp,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        Nmlp: int,
        Nmlmmp: int,
        idxi,
        idxj,
        idxjp,
        idxu,
        idxv,
        idxup,
        idxvp,
    ) -> float:
        n = int(Natom)
        from mmml.interfaces.pycharmmInterface.mlpot.callback_buffers import (
            stack_charmm_xyz,
        )

        pos_full = stack_charmm_xyz(x, y, z, n)
        pos_charmm_full = np.array(pos_full, dtype=np.float64, copy=True)
        box = self._sync_callback_pbc_box()
        live_side = float(self._cell) if self._cell else None
        pos_full = self._maybe_rewrap_primary_cell_in_callback(
            pos_full, n, x, y, z, box_side_A=live_side
        )
        ml_idx = self._resolve_ml_callback_slice(n)
        n_ml = int(ml_idx.size)
        pos = pos_full[ml_idx]
        from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
            get_mlpot_profile_stats,
            mlpot_profiling_enabled,
        )

        if mlpot_profiling_enabled():
            get_mlpot_profile_stats().record_charmm_gap()
        t0 = time.perf_counter()
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import (
            broadcast_mlpot_result,
            mpi_rank_size,
            mlpot_runs_on_this_rank,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
            spatial_mpi_enabled,
        )

        run_ml = mlpot_runs_on_this_rank()
        e_kcal = 0.0
        forces_ml = np.zeros((n_ml, 3), dtype=np.float64)
        mm_pair_idx = None
        mm_pair_mask = None
        use_mm_pairs = False
        rank, mpi_size = mpi_rank_size()
        use_spatial = (
            bool(getattr(self, "_spatial_mpi", False) or spatial_mpi_enabled())
            and mpi_size > 1
            and bool(self._cell)
        )
        if run_ml:
            with self._mlpot_eval_device_context():
                t_pairs = time.perf_counter()
                try:
                    if self._mm_pair_source == "charmm_callback":
                        mm_pair_idx, mm_pair_mask, use_mm_pairs = (
                            self._resolve_mm_pairs_from_callback(
                                idxu,
                                idxv,
                                idxup,
                                idxvp,
                                natom=n_ml,
                                nmlmmp=int(Nmlmmp),
                                pos=pos,
                                box=box,
                            )
                        )
                    else:
                        mm_pair_idx, mm_pair_mask, use_mm_pairs = self._resolve_mm_pairs(
                            pos, box
                        )
                except _CallbackPairListUnavailable as exc:
                    msg = str(exc)
                    self._last_callback_error = msg
                    self._last_callback_hybrid_energy_kcal = 0.0
                    self._last_callback_user_return_kcal = 0.0
                    self.last_ml_forces = np.zeros((n, 3), dtype=np.float64)
                    parent = getattr(self, "_parent_model", None)
                    if parent is not None:
                        parent._last_callback_error = msg
                        parent._last_ml_forces = self.last_ml_forces
                    from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import (
                        mlpot_dynamics_armed,
                    )

                    if mlpot_dynamics_armed() and not _callback_opt_out(
                        ALLOW_MISSING_CALLBACK_PAIRS_ENV
                    ):
                        # Dynamics: fail closed. The guarded entry point exits 86.
                        raise
                    if not self._callback_pair_warned:
                        print(
                            f"WARN: {msg} ({ALLOW_MISSING_CALLBACK_PAIRS_ENV}=1: "
                            "returning zero USER energy and forces)",
                            flush=True,
                        )
                        self._callback_pair_warned = True
                    return 0.0
                positions_jax = as_ml_array(
                    pos,
                    dtype=resolve_ml_compute_dtype(getattr(self, "_ml_compute_dtype", None)),
                )
                atomic_numbers_jax = jnp.asarray(self.atomic_numbers[:n_ml])
                forward_fn = self._get_spherical_forward_fn(
                    n_atoms=n_ml,
                    atomic_numbers_jax=atomic_numbers_jax,
                    box_jax=box,
                )
                mono_jax = jnp.zeros((0,), dtype=jnp.int32)
                dimer_jax = jnp.zeros((0,), dtype=jnp.int32)
                if use_spatial:
                    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.batch_builder import (
                        build_domdec_spatial_batch_indices,
                        make_domdec_aligned_grid,
                    )

                    grid = make_domdec_aligned_grid(
                        float(self._cell),
                        self.cutoff_params,
                        n_ranks_fallback=mpi_size,
                    )
                    batch_idx = build_domdec_spatial_batch_indices(
                        pos,
                        self.n_monomers,
                        self._atoms_per_monomer,
                        grid,
                        rank,
                        self.cutoff_params,
                    )
                    mono_jax = jnp.asarray(batch_idx.owned_monomers, dtype=jnp.int32)
                    dimer_jax = jnp.asarray(batch_idx.active_dimer_indices, dtype=jnp.int32)
                pair_ms = (time.perf_counter() - t_pairs) * 1000.0
                fwd_args = (
                    positions_jax,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                    mono_jax,
                    dimer_jax,
                    use_spatial,
                )
                fwd_kwargs = self._resolve_ml_dimer_candidates(
                    pos, box, use_spatial=use_spatial
                )
                from mmml.interfaces.pycharmmInterface.mlpot.strain_virial import (
                    strain_virial_enabled,
                )

                want_virial = strain_virial_enabled()
                if want_virial:
                    vir_fn = getattr(self._grad_cache_owner(), "_spherical_forward_vir_fn", None)
                    if vir_fn is None or box is None or use_spatial:
                        raise RuntimeError(
                            "CPT strain virial needs a periodic box and the non-spatial MLpot "
                            "forward; refusing to run NpT with CHARMM's central-atom virial"
                        )
                    if getattr(self, "_periodic_mm_config", None) is not None:
                        raise RuntimeError(
                            "CPT strain virial does not cover the periodic Coulomb add-on yet; "
                            "refusing NpT with an incomplete virial"
                        )
                    forward_fn = vir_fn
                t_fwd = time.perf_counter()
                fwd_out = forward_fn(*fwd_args, **fwd_kwargs)
                e_raw, forces_ev = fwd_out[0], fwd_out[1]
                self._last_fwd_out = fwd_out
                budget = getattr(self._grad_cache_owner(), "_ml_chunk_budget", None)
                if budget is not None and len(fwd_out) > 2:
                    e_raw, forces_ev = self._check_ml_chunk_budget(
                        budget, forward_fn, fwd_args, fwd_out, fwd_kwargs
                    )
                if mlpot_profiling_enabled():
                    e_raw = jax.block_until_ready(e_raw)
                    forces_ev = jax.block_until_ready(forces_ev)
                fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
                t_host = time.perf_counter()
                e_host = jax.device_get(e_raw)
                forces_host = jax.device_get(forces_ev)
                from mmml.interfaces.pycharmmInterface.mlpot.finite_guards import (
                    require_host_finite,
                )

                require_host_finite(e_host, forces_host, name="ML USER")
                if want_virial:
                    from mmml.interfaces.pycharmmInterface.mlpot.strain_virial import (
                        push_virial_to_charmm,
                        virial_correction_kcal,
                    )

                    dE_dbox = np.asarray(jax.device_get(self._last_fwd_out[3]), dtype=np.float64)
                    current_box = getattr(self, "_current_box", None)
                    cell_np = np.asarray(current_box if current_box is not None else box, dtype=np.float64)
                    correction = virial_correction_kcal(
                        dE_dcell_eV=dE_dbox,
                        cell=cell_np,
                        forces_eV_A=forces_host,
                        positions_eval=pos,
                        positions_charmm=pos_charmm_full[ml_idx],
                        ev_to_kcal=self.ev2kcal,
                    )
                    require_host_finite(0.0, correction, name="ML USER strain virial")
                    push_virial_to_charmm(correction)
                    self._last_strain_virial_kcal = correction
                    self._strain_virial_calls = getattr(self, "_strain_virial_calls", 0) + 1
                    if self._strain_virial_calls == 1 or self._strain_virial_calls % 1000 == 0:
                        vol = abs(float(np.linalg.det(cell_np)))
                        tr = float(np.trace(correction))
                        print(
                            f"MLpot strain virial (call {self._strain_virial_calls}): "
                            f"correction trace {tr:+.2f} kcal/mol -> dP {tr / (3.0 * vol) * 68568.4:+.1f} atm "
                            f"(V {vol:.0f} A^3)",
                            flush=True,
                        )
                e_kcal = float(e_host) * self.ev2kcal
                forces_ml = np.asarray(forces_host, dtype=np.float64) * self.ev2kcal
                if mlpot_profiling_enabled():
                    get_mlpot_profile_stats().record_callback_stages(
                        {
                            "mm_pairs": pair_ms,
                            "spherical_forward": fwd_ms,
                            "host_writeback": (time.perf_counter() - t_host) * 1000.0,
                            "callback_total": (time.perf_counter() - t0) * 1000.0,
                        }
                    )
                forces = np.zeros((n, 3), dtype=np.float64)
                forces[ml_idx] = forces_ml
                self.last_ml_forces = np.asarray(forces, dtype=np.float64, copy=True)
                parent = getattr(self, "_parent_model", None)
                if parent is not None:
                    parent._last_ml_forces = self.last_ml_forces
                try:
                    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                        charmm_lib_links_mpi,
                        recover_mpi_for_charmm_after_jax,
                    )

                    if charmm_lib_links_mpi():
                        recover_mpi_for_charmm_after_jax(phase="after MLpot gete")
                except Exception:
                    # Deliberately non-fatal: energy and forces are already on the
                    # host; this only re-syncs MPI/OpenMP state for CHARMM. A real
                    # MPI breakage surfaces in the next collective, not as a wrong
                    # energy.
                    pass
            parent = getattr(self, "_parent_model", None)
            if parent is not None:
                parent._maybe_promote_deferred_jax_on_hybrid_eval(self)
        else:
            forces = np.zeros((n, 3), dtype=np.float64)
        forces, e_kcal = broadcast_mlpot_result(forces, e_kcal, n)
        self.last_ml_forces = np.asarray(forces, dtype=np.float64, copy=True)
        parent = getattr(self, "_parent_model", None)
        if parent is not None:
            parent._last_ml_forces = self.last_ml_forces
        if mlpot_profiling_enabled():
            get_mlpot_profile_stats().record_ml(time.perf_counter() - t0)
        periodic_cfg = getattr(self, "_periodic_mm_config", None)
        if periodic_cfg is not None and run_ml and self._cell:
            from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm_external import (
                add_periodic_coulomb_to_callback,
            )
            from mmml.interfaces.pycharmmInterface.nl_reference import (
                monomer_id_from_offsets,
            )

            side = float(self._cell)
            offsets = _monomer_offsets_from_atoms_per_monomer(self._atoms_per_monomer)
            mid = monomer_id_from_offsets(offsets, int(n_ml))
            try:
                e_ml, forces_ml_cb = add_periodic_coulomb_to_callback(
                    pos,
                    box_side_A=side,
                    cfg=periodic_cfg,
                    energy_kcal=float(e_kcal),
                    forces_kcal=np.asarray(forces[ml_idx], dtype=np.float64),
                    mol_id=mid,
                    n_monomers=int(self.n_monomers),
                )
                e_kcal = float(e_ml)
                forces = np.asarray(forces, dtype=np.float64, copy=True)
                forces[ml_idx] = np.asarray(forces_ml_cb, dtype=np.float64)
            except Exception as exc:
                # Continuing without the periodic Coulomb term switches the
                # Hamiltonian mid-run (wrong energy and forces), so fail closed.
                if not _callback_opt_out(ALLOW_PERIODIC_COULOMB_FAILURE_ENV):
                    raise
                import sys

                print(
                    f"WARN: periodic Coulomb callback failed ({exc}); "
                    f"continuing with ML-only USER energy "
                    f"({ALLOW_PERIODIC_COULOMB_FAILURE_ENV}=1)",
                    file=sys.stderr,
                    flush=True,
                )
        from mmml.interfaces.pycharmmInterface.mlpot.callback_buffers import (
            subtract_forces_from_charmm_grad,
        )

        subtract_forces_from_charmm_grad(dx, dy, dz, forces, n)
        if run_ml and use_mm_pairs:
            hybrid_before_route = float(e_kcal)
            from mmml.interfaces.pycharmmInterface.mlpot.charmm_eterm_routing import (
                decompose_and_route_mlpot_mm_from_callback,
            )

            e_kcal = decompose_and_route_mlpot_mm_from_callback(
                self,
                pos,
                mm_pair_idx,
                mm_pair_mask,
                box,
                hybrid_before_route,
                use_mm_pairs=bool(use_mm_pairs),
            )
            self._last_callback_hybrid_energy_kcal = hybrid_before_route
        else:
            self._last_callback_hybrid_energy_kcal = float(e_kcal if run_ml else 0.0)
        self._last_callback_user_return_kcal = float(e_kcal)
        return e_kcal


class _DeferredDecomposedMlpotCalculator:
    """Defer JAX factory build until the first CHARMM ``ENER`` (after MLpot registration)."""

    def __init__(
        self,
        model: "DecomposedMlpotModel",
        *,
        ml_atomic_numbers: np.ndarray | None = None,
        ml_atom_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self._model = model
        self._ml_atomic_numbers = (
            None if ml_atomic_numbers is None else np.asarray(ml_atomic_numbers, dtype=int)
        )
        self._ml_atom_indices = (
            None
            if ml_atom_indices is None
            else np.asarray(ml_atom_indices, dtype=int).reshape(-1)
        )
        self._real: DecomposedMlpotCalculator | None = None

    @property
    def ml_atomic_numbers(self) -> np.ndarray:
        """CHARMM MLpot registration Z (before first ``ENER`` materializes the real calc)."""
        if self._ml_atomic_numbers is not None:
            return np.asarray(self._ml_atomic_numbers, dtype=int)
        return np.asarray(self._model._atomic_numbers, dtype=int)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.asarray(
            physnet_ml_atomic_numbers(self.ml_atomic_numbers), dtype=np.int32
        )

    def _ensure_real(self) -> DecomposedMlpotCalculator:
        if self._real is not None:
            return self._real
        self._real = self._model._build_registered_calculator(
            ml_atomic_numbers=self._ml_atomic_numbers,
            ml_atom_indices=self._ml_atom_indices,
        )
        return self._real

    @failstop_calculate_charmm
    def calculate_charmm(self, *args, **kwargs) -> float:
        return self._ensure_real().calculate_charmm(*args, **kwargs)


class DecomposedMlpotModel:
    def __init__(
        self,
        spherical_fn: Any | None,
        cutoff_params: CutoffParameters,
        n_monomers: int,
        atomic_numbers: np.ndarray,
        cell: Union[float, bool] = False,
        do_mm: bool = True,
        get_update_fn: Any | None = None,
        ml_compute_dtype: str | None = None,
        *,
        pending_factory: Any | None = None,
        pending_factory_z: np.ndarray | None = None,
        pending_do_ml: bool = True,
        pending_do_ml_dimer: bool = True,
        verbose: bool = False,
        spatial_mpi: bool = False,
        atoms_per_monomer: Sequence[int] | None = None,
        defer_jax_until_after_sd: bool = False,
        defer_jax_until_mlpot_registered: bool = False,
        periodic_mm_config: Any | None = None,
        lr_solver: str | None = None,
        jax_pme_method: str | None = None,
        mm_pair_source: MmPairSource = _DEFAULT_MM_PAIR_SOURCE,
        mm_r_min: float | None = None,
        mm_pair_capacity_hint: int | None = None,
    ) -> None:
        self._spherical_fn = spherical_fn
        self._cutoff_params = cutoff_params
        self._n_monomers = int(n_monomers)
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._cell = float(cell) if cell else False
        self._charmm_box_side_A: float | None = None
        self._npt_restart_read: Path | None = None
        self._do_mm = bool(do_mm)
        self._periodic_mm_config = periodic_mm_config
        self._get_update_fn = get_update_fn
        self._ml_compute_dtype = ml_compute_dtype
        self._spatial_mpi = bool(spatial_mpi)
        self._defer_jax_until_after_sd = bool(defer_jax_until_after_sd)
        self._defer_jax_until_mlpot_registered = bool(defer_jax_until_mlpot_registered)
        self._charmm_mlpot_sd_active = 0
        self._jax_on_gpu = spherical_fn is not None
        self._registered_calculator: DecomposedMlpotCalculator | None = None
        self._ml_atom_indices: np.ndarray | None = None
        if atoms_per_monomer is None:
            apm = max(1, len(self._atomic_numbers) // max(1, int(n_monomers)))
            self._atoms_per_monomer = [apm] * int(n_monomers)
        else:
            self._atoms_per_monomer = [int(x) for x in atoms_per_monomer]
        self._last_ml_forces: np.ndarray | None = None
        self._spherical_forward_fn: Any | None = None
        self._forward_cache_key: tuple[Any, ...] | None = None
        self._jax_warmup_done = False
        self._pre_sd_callback_forward_warmup_done = False
        self._pending_factory = pending_factory
        self._pending_factory_z = (
            None if pending_factory_z is None else np.asarray(pending_factory_z, dtype=int)
        )
        self._pending_do_ml = bool(pending_do_ml)
        self._pending_do_ml_dimer = bool(pending_do_ml_dimer)
        self._verbose = bool(verbose)
        self._lr_solver = lr_solver
        self._jax_pme_method = jax_pme_method
        self._mm_pair_source: MmPairSource = str(mm_pair_source)  # type: ignore[assignment]
        self._mm_r_min = float(mm_r_min) if mm_r_min is not None else None
        self._mm_pair_capacity_hint = (
            int(mm_pair_capacity_hint) if mm_pair_capacity_hint else None
        )
        self._jax_pme_hybrid_first_ener_done = not self._defer_jax_pme_gpu_promote_initial()

    def _jax_pme_lr_active(self) -> bool:
        if not self._do_mm:
            return False
        from mmml.interfaces.pycharmmInterface.long_range_backend import pick_lr_solver

        return pick_lr_solver(self._lr_solver) == "jax_pme"

    def _jax_pme_mesh_active(self) -> bool:
        from mmml.interfaces.pycharmmInterface.long_range_backend import jax_pme_mesh_method

        return self._jax_pme_lr_active() and jax_pme_mesh_method(self._jax_pme_method)

    def _defer_jax_pme_gpu_promote_initial(self) -> bool:
        """Keep hybrid on CPU through the first ENER when jax-pme uses a k-space mesh."""
        return (
            self._defer_jax_until_after_sd
            and self._jax_pme_mesh_active()
        )

    def _defer_jax_pme_gpu_promote(self) -> bool:
        return self._jax_pme_mesh_active() and not self._jax_pme_hybrid_first_ener_done

    def _finalize_jax_factory(self, *, gpu: bool = False) -> None:
        """Build ``spherical_fn`` after CHARMM MLpot ``upinb`` (``MLpot.__init__``)."""
        if self._spherical_fn is not None:
            return
        if self._pending_factory is None or self._pending_factory_z is None:
            raise RuntimeError("DecomposedMlpotModel: JAX factory was not initialized")
        from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
            jax_compile_threads_context,
        )
        from mmml.interfaces.pycharmmInterface.jax_device_policy import (
            jax_cpu_backend_available,
            jax_cpu_until_mlpot_registered,
            mlpot_device_context_fell_back_to_cpu,
            mlpot_jax_device_context,
            mlpot_jax_device_name,
            reset_mlpot_device_fallback_flag,
        )

        cpu_only = not gpu and (
            self._defer_jax_until_after_sd or mlpot_jax_device_name() == "cpu"
        )
        if cpu_only and not jax_cpu_backend_available():
            if self._verbose:
                print(
                    "Decomposed MLpot: CPU defer requested but JAX CPU backend is "
                    "unavailable (jax likely imported with GPU-only platforms); "
                    "compiling on GPU instead",
                    flush=True,
                )
            cpu_only = False

        with jax_compile_threads_context():
            if not cpu_only:
                ensure_xla_gpu_warmed()
            z = self._pending_factory_z
            r0 = np.zeros((len(z), 3), dtype=np.float64)

            device_ctx = (
                jax_cpu_until_mlpot_registered if cpu_only else mlpot_jax_device_context
            )
            if cpu_only and self._verbose:
                print(
                    "Decomposed MLpot: compiling JAX factory on CPU before MLpot SD "
                    "(MPI-linked CHARMM deferred backend promotion)",
                    flush=True,
                )
            elif gpu and self._verbose:
                print(
                    "Decomposed MLpot: compiling JAX factory on GPU after MLpot SD "
                    "(MPI defer path; ignoring registration-time CPU env)",
                    flush=True,
                )
            if not cpu_only:
                reset_mlpot_device_fallback_flag()
            with device_ctx():
                _, spherical_fn, get_update_fn = unpack_factory_result(
                    self._pending_factory(
                        atomic_numbers=jnp.asarray(z),
                        atomic_positions=jnp.asarray(r0),
                        n_monomers=self._n_monomers,
                        cutoff_params=self._cutoff_params,
                        doML=self._pending_do_ml,
                        doMM=self._do_mm,
                        doML_dimer=self._pending_do_ml_dimer,
                        backprop=False,
                        create_ase_calculator=False,
                    )
                )
        self._spherical_fn = spherical_fn
        if self._do_mm:
            self._get_update_fn = get_update_fn
        # Track the device actually used, not the request: mlpot_jax_device_context
        # silently fell back to CPU when GPU was requested but unavailable (see its
        # docstring), so `not cpu_only` alone previously left `_jax_on_gpu=True` while
        # the compute ran on CPU (the "promoted to GPU" message with 0% nvidia-smi
        # utilization bug this fixes). `reset_mlpot_device_fallback_flag` right before
        # `device_ctx()` means a test-mocked `mlpot_jax_device_context` (which never
        # touches the flag) is correctly read as "no fallback".
        self._jax_on_gpu = (not cpu_only) and not mlpot_device_context_fell_back_to_cpu()
        if not cpu_only:
            self._pending_factory = None
            self._pending_factory_z = None
        if cpu_only and self._defer_jax_until_after_sd:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                recover_mpi_for_charmm_after_jax,
            )

            recover_mpi_for_charmm_after_jax(
                phase="after deferred MLpot JAX CPU finalize",
            )
        elif (not cpu_only) and self._defer_jax_until_after_sd:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                charmm_lib_links_mpi,
                recover_mpi_for_charmm_after_jax,
            )

            if charmm_lib_links_mpi():
                recover_mpi_for_charmm_after_jax(
                    phase="after deferred MLpot JAX GPU finalize",
                )
        if self._verbose:
            backend = "CPU" if cpu_only else "GPU"
            print(
                f"Decomposed MLpot spherical_fn={spherical_fn!r} "
                f"({backend}; JIT bind doML={self._pending_do_ml} doMM={self._do_mm} "
                f"doML_dimer={self._pending_do_ml_dimer})",
                flush=True,
            )

    def _maybe_promote_deferred_jax_on_hybrid_eval(
        self,
        calc: DecomposedMlpotCalculator,
    ) -> None:
        """Promote deferred JAX to GPU after the first hybrid ENER (jax-pme mesh only).

        Blocked while CHARMM MLpot SD is active; ``maybe_warmup_deferred_decomposed_mlpot``
        promotes after SD completes. Mid-SD GPU compile on MPI-linked CHARMM can corrupt
        OpenMPI pools and segfault the next Fortran ``enbond`` step.
        """
        if not self._defer_jax_until_after_sd or self._jax_on_gpu:
            return
        if not self._defer_jax_pme_gpu_promote():
            return
        self._jax_pme_hybrid_first_ener_done = True
        self.promote_jax_factory_to_gpu(force_after_sd=False)
        if not self._jax_on_gpu:
            return
        calc._spherical_forward_fn = None
        calc._forward_cache_key = None
        if self._spherical_fn is not None:
            calc.spherical_fn = self._spherical_fn
        if self._get_update_fn is not None:
            calc._get_update_fn = self._get_update_fn
            calc._cached_update_fn = None

    def promote_jax_factory_to_gpu(self, *, force_after_sd: bool = False) -> None:
        """Rebuild ``spherical_fn`` on GPU after MLpot SD (MPI defer path)."""
        if not self._defer_jax_until_after_sd or self._jax_on_gpu:
            return
        if (
            not force_after_sd
            and int(getattr(self, "_charmm_mlpot_sd_active", 0)) > 0
        ):
            return
        if self._defer_jax_pme_gpu_promote():
            if self._verbose:
                print(
                    "Decomposed MLpot: deferring JAX GPU promote until after first "
                    "hybrid ENER (jax-pme mesh)",
                    flush=True,
                )
            return
        self._spherical_fn = None
        self._spherical_forward_fn = None
        self._forward_cache_key = None
        if self._do_mm:
            self._get_update_fn = None
        self._finalize_jax_factory(gpu=True)
        calc = self._registered_calculator
        if calc is not None:
            real = getattr(calc, "_real", calc)
            if isinstance(real, DecomposedMlpotCalculator):
                real.spherical_fn = self._spherical_fn
                real._get_update_fn = self._get_update_fn
                real._spherical_forward_fn = None
                real._forward_cache_key = None
        try:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                charmm_lib_links_mpi,
                recover_mpi_for_charmm_after_jax,
            )
            from mmml.interfaces.pycharmmInterface.jax_device_policy import (
                mlpot_jax_device_name,
            )

            if charmm_lib_links_mpi() and mlpot_jax_device_name() == "gpu":
                from mmml.utils.jax_gpu_warmup import sync_jax_gpu_before_charmm

                sync_jax_gpu_before_charmm(phase="after MLpot JAX GPU promote")
            if charmm_lib_links_mpi():
                recover_mpi_for_charmm_after_jax(phase="after MLpot JAX GPU promote")
        except Exception:
            # Non-fatal by design: MPI/GPU re-sync only; the promoted JAX factory
            # is already installed and evaluates the same energy.
            pass

    def _build_registered_calculator(
        self,
        *,
        ml_atomic_numbers: np.ndarray | None = None,
        ml_atom_indices: Sequence[int] | np.ndarray | None = None,
    ) -> DecomposedMlpotCalculator:
        self._finalize_jax_factory()
        if ml_atomic_numbers is not None:
            z = np.asarray(ml_atomic_numbers, dtype=int)
        else:
            z = self._atomic_numbers
        if ml_atom_indices is None:
            ml_atom_indices = getattr(self, "_ml_atom_indices", None)
        calc = DecomposedMlpotCalculator(
            self._spherical_fn,
            self._cutoff_params,
            self._n_monomers,
            z,
            cell=self._cell,
            do_mm=self._do_mm,
            do_ml=self._pending_do_ml,
            do_ml_dimer=self._pending_do_ml_dimer,
            get_update_fn=self._get_update_fn,
            ml_compute_dtype=self._ml_compute_dtype,
            spatial_mpi=self._spatial_mpi,
            atoms_per_monomer=self._atoms_per_monomer,
            periodic_mm_config=self._periodic_mm_config,
            mm_pair_source=self._mm_pair_source,
            mm_r_min=self._mm_r_min,
            mm_pair_capacity_hint=self._mm_pair_capacity_hint,
            ml_atom_indices=ml_atom_indices,
        )
        calc._parent_model = self
        self._registered_calculator = calc
        return calc

    def get_pycharmm_calculator(self, ml_atom_indices=None, ml_atomic_numbers=None, **kwargs):
        if ml_atom_indices is not None:
            self._ml_atom_indices = np.asarray(ml_atom_indices, dtype=int).reshape(-1)
        if self._spherical_fn is None and self._pending_factory is not None:
            if self._defer_jax_until_mlpot_registered:
                if self._defer_jax_until_after_sd:
                    # Stay fully deferred until GPU promote (calculator mini) or first ENER.
                    # Eager CPU _finalize here duplicated work: promote/warmup recompiled on GPU.
                    deferred = _DeferredDecomposedMlpotCalculator(
                        self,
                        ml_atomic_numbers=ml_atomic_numbers,
                        ml_atom_indices=getattr(self, "_ml_atom_indices", None),
                    )
                    self._registered_calculator = deferred
                    return deferred
                return self._build_registered_calculator(
                    ml_atomic_numbers=ml_atomic_numbers,
                    ml_atom_indices=getattr(self, "_ml_atom_indices", None),
                )
            deferred = _DeferredDecomposedMlpotCalculator(
                self,
                ml_atomic_numbers=ml_atomic_numbers,
                ml_atom_indices=getattr(self, "_ml_atom_indices", None),
            )
            self._registered_calculator = deferred
            return deferred
        return self._build_registered_calculator(
            ml_atomic_numbers=ml_atomic_numbers,
            ml_atom_indices=getattr(self, "_ml_atom_indices", None),
        )


def build_decomposed_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    ml_spatial_mpi: bool | None = None,
    cell: Union[float, bool] = False,
    verbose: bool = False,
    args: Any | None = None,
    ml_compute_dtype: str | None = None,
    defer_jax_until_mlpot_registered: bool = False,
    defer_jax_until_after_sd: bool = False,
) -> DecomposedMlpotModel | MetatomicMlpotModel:
    """CHARMM MLpot factory: metatomic ASE adapter or JAX PhysNet/KerNN hybrid."""
    from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import (
        maybe_build_metatomic_mlpot_model,
    )

    metatomic_model = maybe_build_metatomic_mlpot_model(
        checkpoint,
        atomic_numbers,
        atoms_per_monomer,
        int(n_monomers),
        cell=cell,
        verbose=verbose,
        args=args,
    )
    if metatomic_model is not None:
        return metatomic_model
    return _build_jax_decomposed_mlpot_model(
        checkpoint,
        atomic_numbers,
        atoms_per_monomer,
        n_monomers,
        ml_batch_size=ml_batch_size,
        ml_gpu_count=ml_gpu_count,
        ml_max_active_dimers=ml_max_active_dimers,
        ml_spatial_mpi=ml_spatial_mpi,
        cell=cell,
        verbose=verbose,
        args=args,
        ml_compute_dtype=ml_compute_dtype,
        defer_jax_until_mlpot_registered=defer_jax_until_mlpot_registered,
        defer_jax_until_after_sd=defer_jax_until_after_sd,
    )


def _load_hybrid_mm_scales(scales_file, checkpoint, verbose):
    """Load optional LJ and charge scales with explicit-file errors preserved."""
    ep_scale = sig_scale = None
    mm_charge_scale = 1.0
    from mmml.models.mm_lj_scales import resolve_md_lj_scales

    try:
        ep_scale, sig_scale = resolve_md_lj_scales(
            scales_file=scales_file,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        if scales_file is not None:
            raise
        if verbose:
            print(f"WARNING: could not load MM LJ scales: {exc}", flush=True)
    if verbose and ep_scale is not None:
        print(
            f"Loaded MM LJ scales ({len(ep_scale)} ATC types) "
            f"from hybrid_mm.json / --mm-lj-scales-file",
            flush=True,
        )
    from mmml.models.mm_lj_scales import resolve_md_charge_scale

    try:
        mm_charge_scale = resolve_md_charge_scale(
            scales_file=scales_file,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        if scales_file is not None:
            raise
        if verbose:
            print(f"WARNING: could not load MM charge scale: {exc}", flush=True)
    if verbose and mm_charge_scale != 1.0:
        print(
            f"Loaded MM charge scale {mm_charge_scale:.4f} "
            f"(Coulomb x{mm_charge_scale ** 2:.4f}) from hybrid_mm.json / --mm-lj-scales-file",
            flush=True,
        )
    return ep_scale, sig_scale, mm_charge_scale


def _build_jax_decomposed_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    ml_spatial_mpi: bool | None = None,
    cell: Union[float, bool] = False,
    verbose: bool = False,
    args: Any | None = None,
    ml_compute_dtype: str | None = None,
    defer_jax_until_mlpot_registered: bool = False,
    defer_jax_until_after_sd: bool = False,
) -> DecomposedMlpotModel:
    from mmml.models.kernnn import is_kernnn_checkpoint
    from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import (
        _jax_mm_spoof_requested,
    )

    _ckpt_probe = Path(checkpoint).expanduser() if checkpoint is not None else None
    if args is not None and getattr(args, "model_restart_path", None) is not None:
        _ckpt_probe = Path(getattr(args, "model_restart_path")).expanduser()
    _spoof = _jax_mm_spoof_requested(args)
    _kernnn = bool(_ckpt_probe) and is_kernnn_checkpoint(_ckpt_probe)
    _ml_mode = "jax_mm_clone" if _spoof else ("kernnn" if _kernnn else "physnet")
    if _spoof:
        ckpt = Path("/dev/null")
    else:
        ckpt = Path(checkpoint).expanduser().resolve()
        if not _kernnn:
            from mmml.interfaces.energy_forces.ml import assert_hybrid_ml_compatible

            assert_hybrid_ml_compatible(ckpt)
    if args is not None and ml_compute_dtype is None:
        ml_compute_dtype = getattr(args, "ml_compute_dtype", None)
    cutoff_params = (
        cutoff_parameters_from_args(args) if args is not None else CutoffParameters()
    )
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
    )

    _spatial_explicit: bool | None = ml_spatial_mpi
    if args is not None and getattr(args, "ml_spatial_mpi", None) is not None:
        _spatial_explicit = bool(args.ml_spatial_mpi)
    spatial_mpi = spatial_mpi_enabled(_spatial_explicit)
    z = np.asarray(physnet_ml_atomic_numbers(atomic_numbers), dtype=int)
    per = [int(x) for x in atoms_per_monomer]
    max_atoms = max(per) * 2
    batch_size = resolve_ml_batch_size(int(n_monomers), ml_batch_size)
    gpu_count = resolve_ml_gpu_count(ml_gpu_count)
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        resolve_max_active_dimers,
    )

    n_dimers_total = int(n_monomers) * (int(n_monomers) - 1) // 2
    free_space = cell is False or cell is None
    _box_volume = None
    _active_radius = None
    if not free_space and cell is not None:
        try:
            side = float(cell)
            if side > 0.0:
                _box_volume = side**3
                from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
                    sparse_dimer_active_radius,
                )

                margin = 0.0
                if args is not None:
                    raw_margin = getattr(args, "ml_dimer_active_margin", None)
                    if raw_margin is not None:
                        margin = float(raw_margin)
                    else:
                        margin = float(os.environ.get("MMML_ML_DIMER_ACTIVE_MARGIN") or 0.0)
                else:
                    margin = float(os.environ.get("MMML_ML_DIMER_ACTIVE_MARGIN") or 0.0)
                _active_radius = sparse_dimer_active_radius(
                    float(cutoff_params.mm_switch_on),
                    float(cutoff_params.ml_switch_width),
                    margin=margin,
                )
        except (TypeError, ValueError):
            pass
    dimer_cap = resolve_max_active_dimers(
        int(n_monomers),
        n_dimers_total,
        ml_max_active_dimers,
        free_space=free_space,
        box_volume=_box_volume,
        active_radius=_active_radius,
    )
    max_pairs = None
    if args is not None:
        max_pairs = getattr(args, "max_pairs", None)
    periodic_mm_config = None
    periodic_mode = False
    mm_nonbond_mode = "jax_mic"
    if args is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import (
            build_periodic_mm_config,
            periodic_mm_status_line,
            resolve_mm_nonbond_mode,
            resolve_periodic_charmm_vdw,
        )

        periodic_mm_config = build_periodic_mm_config(args)
        periodic_mode = resolve_mm_nonbond_mode(args) == "periodic_external"
        mm_nonbond_mode = resolve_mm_nonbond_mode(args)
    if max_pairs is None and not free_space and cell and not periodic_mode:
        from mmml.interfaces.pycharmmInterface.cell_list import estimate_max_pairs

        cutoff_a = float(cutoff_params.mm_switch_on) + float(cutoff_params.mm_switch_width)
        n_atoms = int(sum(per))
        safety = float(
            getattr(args, "cell_list_safety_factor", 3.0) or 3.0
            if args is not None
            else 3.0
        )
        max_pairs = estimate_max_pairs(
            n_atoms,
            cutoff=cutoff_a,
            safety_factor=safety,
            box_side_A=float(cell),
        )
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_local_gpu_count

    local_gpus = mlpot_local_gpu_count()
    if local_gpus > 1 and gpu_count <= 1 and verbose:
        print(
            f"Decomposed MLpot: {local_gpus} JAX GPUs visible but ml_gpu_count=1 "
            f"(only GPU:0 runs PhysNet). Use --ml-gpu-count {local_gpus} "
            f"and enough chunks (--ml-batch-size 128-256 for DCM:90).",
            flush=True,
        )
    if verbose and batch_size is not None:
        print(
            f"Decomposed MLpot: ml_batch_size={batch_size} "
            f"({int(n_monomers)} monomers; reduces JAX compile memory)",
            flush=True,
        )
    if verbose and gpu_count > 1:
        print(
            f"Decomposed MLpot: ml_gpu_count={gpu_count} (parallel PhysNet chunks)",
            flush=True,
        )
    if verbose:
        cap_note = "free-space all-pairs safe; " if free_space else ""
        print(
            f"Decomposed MLpot: max_active_dimers={dimer_cap} "
            f"({cap_note}PhysNet batch ≤ {int(n_monomers) + dimer_cap} systems/step)",
            flush=True,
        )
    if verbose and cell:
        print(
            f"Decomposed MLpot: MIC PBC cubic cell={float(cell):.3f} Å",
            flush=True,
        )
    do_ml = True if args is None else bool(getattr(args, "do_ml", True))
    include_mm = True if args is None else bool(getattr(args, "include_mm", True))
    do_mm = include_mm and not periodic_mode
    periodic_external_scales: Path | None = None

    # periodic_external takes its VDW from CHARMM IMAGE, so `do_mm` is False and
    # the JAX pair loop that applies per-type LJ scales never runs. Trained
    # scales therefore have to reach the energy through CHARMM's parameters.
    # See scaled_cgenff_prm: this is exact (per-type, pre-combining), and it is
    # a once-per-session operation -- re-deploying is a guarded no-op, and a
    # different sidecar raises rather than silently zeroing the VDW.
    if periodic_mode and include_mm and args is not None:
        from mmml.models.mm_lj_scales import find_learnable_lj_scales_sidecar

        scales_file = getattr(args, "mm_lj_scales_file", None)
        periodic_external_scales = find_learnable_lj_scales_sidecar(
            scales_file=scales_file,
            checkpoint=None if _spoof else ckpt,
        )
        if periodic_external_scales is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
                deploy_scaled_lj_into_charmm,
            )

            deploy_scaled_lj_into_charmm(periodic_external_scales, verbose=verbose)
            from mmml.models.mm_lj_scales import load_md_charge_scale

            if load_md_charge_scale(periodic_external_scales) != 1.0:
                print(
                    f"mmml WARNING: {periodic_external_scales} sets mm_charge_scale, but "
                    "periodic_external takes ELEC from CHARMM with PSF charges -- the "
                    "charge scale is NOT applied (use mm_nonbond_mode=jax_mic).",
                    file=sys.stderr,
                    flush=True,
                )
    do_ml_dimer = True if args is None else bool(getattr(args, "do_ml_dimer", True))
    if args is not None and bool(getattr(args, "skip_ml_dimers", False)):
        do_ml_dimer = False
    if verbose and periodic_mm_config is not None and cell:
        print(
            periodic_mm_status_line(periodic_mm_config, box_side_A=float(cell)),
            flush=True,
        )
    if verbose and not do_mm and not periodic_mode:
        print(
            "Decomposed MLpot: include_mm=False — ML potential only (no JAX MM LJ/Coulomb pairs)",
            flush=True,
        )
    if verbose and max_pairs is not None and not periodic_mode:
        print(
            f"Decomposed MLpot: max_pairs={int(max_pairs)} (PBC cell-list buffer)",
            flush=True,
        )
    lr_solver = None
    jax_pme_method = getattr(args, "jax_pme_method", None) if args is not None else None
    jax_pme_sr_cutoff = 6.0
    jax_pme_dispersion = getattr(args, "jax_pme_dispersion", None) if args is not None else None
    mlpot_pbc = not free_space and bool(cell)
    if args is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            resolve_jax_pme_sr_cutoff_for_mlpot,
            resolve_lr_solver_for_mlpot,
            resolve_mlpot_use_pbc,
            warn_if_mic_pbc_without_lr,
        )

        mlpot_pbc = resolve_mlpot_use_pbc(args) or mlpot_pbc
        lr_solver = resolve_lr_solver_for_mlpot(
            args,
            mlpot_pbc=mlpot_pbc,
            mm_nonbond_mode=mm_nonbond_mode,
        )
        jax_pme_sr_cutoff = resolve_jax_pme_sr_cutoff_for_mlpot(args, cutoff_params)
        warn_if_mic_pbc_without_lr(
            lr_solver=lr_solver,
            mlpot_pbc=mlpot_pbc,
            mm_nonbond_mode=mm_nonbond_mode,
            verbose=verbose,
        )
    elif mlpot_pbc:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            resolve_jax_pme_sr_cutoff_for_mlpot,
            resolve_lr_solver_for_mlpot,
            warn_if_mic_pbc_without_lr,
        )

        lr_solver = resolve_lr_solver_for_mlpot(
            args,
            mlpot_pbc=True,
            mm_nonbond_mode=mm_nonbond_mode,
        )
        jax_pme_sr_cutoff = resolve_jax_pme_sr_cutoff_for_mlpot(args, cutoff_params)
        warn_if_mic_pbc_without_lr(
            lr_solver=lr_solver,
            mlpot_pbc=True,
            mm_nonbond_mode=mm_nonbond_mode,
            verbose=verbose,
        )
    if verbose and do_mm and lr_solver:
        from mmml.interfaces.pycharmmInterface.long_range_backend import describe_lr_solver

        disp_text = (
            "env/default"
            if jax_pme_dispersion is None
            else ("on" if bool(jax_pme_dispersion) else "off")
        )
        print(
            f"Decomposed MLpot: {describe_lr_solver(lr_solver)} "
            f"(jax-pme method={jax_pme_method or 'ewald'}, sr_cutoff={jax_pme_sr_cutoff:.1f} Å; "
            f"r^-6 dispersion={disp_text} when lr_solver=jax_pme)",
            flush=True,
        )
    # Energy policy zeroes CHARMM ELEC/VDW on ML atoms for jax_mic hybrids, so
    # Fortran primary lists are empty in vacuum and PBC — use JAX pairs.
    mm_pair_source = resolve_mm_pair_source(
        args,
        all_ml_jax_mic=(
            bool(do_mm)
            and int(n_monomers) > 1
            and str(mm_nonbond_mode) == "jax_mic"
        ),
    )
    mm_r_min_arg = getattr(args, "mm_r_min", None) if args is not None else None
    if verbose and mm_pair_source == "jax":
        print(
            "Decomposed MLpot: mm_pair_source=jax "
            "(all-ML jax_mic; CHARMM callback lists are fully excluded)",
            flush=True,
        )
    if verbose and mm_pair_source == "charmm_callback":
        print(
            "Decomposed MLpot: mm_pair_source=charmm_callback "
            "(Fortran idxu/idxv primary pairs for parity diagnostics)",
            flush=True,
        )
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_name

    # Only pin setup_calculator onto CPU when we intentionally defer GPU work.
    # Default GPU runs used to always pass defer_xla_gpu_warmup=True, which
    # called jax_cpu_until_mlpot_registered during Orbax restore and spammed
    # "CPU backend is not registered" whenever jax had already initialized
    # CUDA-only — while PhysNet still belonged on GPU.
    _cpu_load = defer_jax_until_after_sd or mlpot_jax_device_name() == "cpu"
    ep_scale = None
    sig_scale = None
    mm_charge_scale = 1.0
    scales_file = getattr(args, "mm_lj_scales_file", None) if args is not None else None
    if args is not None and do_mm:
        ep_scale, sig_scale, mm_charge_scale = _load_hybrid_mm_scales(
            scales_file, None if _spoof else ckpt, verbose
        )
    elif args is not None and periodic_external_scales is None:
        # doMM off without a successful CHARMM deployment: ep_scale/sig_scale
        # feed the JAX switched-MM pair loop only. Applying nothing while the
        # user believes trained LJ is active is a silent-wrong-results failure,
        # so say so loudly for an explicit request and warn for auto-discovery.
        from mmml.models.mm_lj_scales import find_learnable_lj_scales_sidecar

        found = None
        try:
            found = find_learnable_lj_scales_sidecar(
                scales_file=scales_file,
                checkpoint=None if _spoof else ckpt,
            )
        except Exception:  # pragma: no cover - discovery must never break setup
            found = None
        mode_note = (
            f"mm_nonbond_mode={mm_nonbond_mode!r}"
            if str(mm_nonbond_mode) == "periodic_external"
            else f"include_mm=false, mm_nonbond_mode={mm_nonbond_mode!r}"
        )
        if scales_file is not None:
            raise ValueError(
                f"--mm-lj-scales-file={scales_file} was given but no LJ backend "
                "can consume it "
                f"({mode_note}), so per-type LJ scales cannot be applied: the "
                "JAX switched-MM pair loop is disabled and CHARMM parameter "
                "deployment requires --include-mm with periodic_external. Use "
                "--include-mm, or drop --mm-lj-scales-file to run without the "
                "trained LJ correction. "
                "See docs/hybrid-mm-lj-scales.md."
            )
        if found is not None:
            print(
                f"mmml WARNING: {found} carries trained MM LJ scales but JAX MM "
                f"is off ({mode_note}) — they are NOT applied. Use --include-mm "
                "to deploy them through the selected LJ backend. See "
                "docs/hybrid-mm-lj-scales.md.",
                file=sys.stderr,
                flush=True,
            )
    # Native ewald doMM: optional switched LJ beside untapered Coulomb (#139).
    from mmml.interfaces.pycharmmInterface.long_range_backend import pick_lr_solver

    _ewald_include_lj = False
    if do_mm and pick_lr_solver(lr_solver) == "ewald":
        _flag = getattr(args, "mm_include_lj", None) if args is not None else None
        if _flag is None:
            _ewald_include_lj = ep_scale is not None
        else:
            _ewald_include_lj = bool(_flag)
        if verbose:
            print(
                f"Decomposed MLpot: lr_solver=ewald include_lj={_ewald_include_lj} "
                f"(COM-switched LJ beside untapered full-box Coulomb)",
                flush=True,
            )
    factory = setup_calculator(
        ATOMS_PER_MONOMER=per,
        N_MONOMERS=int(n_monomers),
        model_restart_path=None if _spoof else str(ckpt),
        doMM=do_mm,
        doML=do_ml,
        doML_dimer=do_ml_dimer,
        verbose=verbose,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
        mm_charge_scale=mm_charge_scale,
        MAX_ATOMS_PER_SYSTEM=max_atoms,
        ml_batch_size=batch_size,
        ml_gpu_count=gpu_count,
        ml_max_active_dimers=ml_max_active_dimers,
        cell=cell,
        max_pairs=max_pairs,
        jax_md_skin_distance=resolve_mlpot_mm_skin_A(args),
        ml_compute_dtype=ml_compute_dtype,
        defer_xla_gpu_warmup=_cpu_load and defer_jax_until_mlpot_registered,
        ml_switch_width=cutoff_params.ml_switch_width,
        mm_switch_on=cutoff_params.mm_switch_on,
        mm_switch_width=cutoff_params.mm_switch_width,
        complementary_handoff=cutoff_params.complementary_handoff,
        mm_r_min=getattr(args, "mm_r_min", None) if args is not None else None,
        electrostatics_damping_sigma=(
            getattr(args, "electrostatics_damping_sigma", None) if args is not None else None
        ),
        mbd_checkpoint=(
            getattr(args, "mbd_checkpoint", None) if args is not None else None
        ),
        mbd_weight=(
            getattr(args, "mbd_weight", None) if args is not None else None
        ),
        mm_charge_correction=bool(
            getattr(args, "mm_charge_correction", False) if args is not None else False
        ),
        mm_charge_mode=(
            getattr(args, "mm_charge_mode", None) if args is not None else None
        ),
        mm_atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        min_com_restraint_distance=(
            getattr(args, "min_com_restraint_distance", None) if args is not None else None
        ),
        min_com_restraint_force_const=(
            getattr(args, "min_com_restraint_k", 1.0) if args is not None else 1.0
        ),
        lr_solver=lr_solver,
        jax_pme_method=jax_pme_method,
        jax_pme_sr_cutoff_A=jax_pme_sr_cutoff,
        jax_pme_dispersion=jax_pme_dispersion,
        ewald_include_self=(
            not bool(getattr(args, "ewald_omit_self", False))
            if args is not None
            else True
        ),
        ewald_include_intra=(
            not bool(getattr(args, "ewald_omit_self", False))
            if args is not None
            else True
        ),
        include_lj=_ewald_include_lj,
        mm_nonbond_mode=mm_nonbond_mode,
        periodic_charmm_vdw=(
            resolve_periodic_charmm_vdw(args) if args is not None else True
        ),
        ml_potential_mode=_ml_mode,
        jax_mm_spoof_psf=(
            getattr(args, "jax_mm_spoof_psf", None) if args is not None else None
        ),
    )
    if defer_jax_until_mlpot_registered:
        return DecomposedMlpotModel(
            None,
            cutoff_params,
            int(n_monomers),
            np.asarray(atomic_numbers, dtype=int),
            cell=cell,
            do_mm=do_mm,
            get_update_fn=None,
            ml_compute_dtype=ml_compute_dtype,
            pending_factory=factory,
            pending_factory_z=z,
            pending_do_ml=do_ml,
            pending_do_ml_dimer=do_ml_dimer,
            verbose=verbose,
            spatial_mpi=spatial_mpi,
            atoms_per_monomer=per,
            defer_jax_until_after_sd=defer_jax_until_after_sd,
            defer_jax_until_mlpot_registered=True,
            periodic_mm_config=periodic_mm_config,
            lr_solver=lr_solver,
            jax_pme_method=jax_pme_method,
            mm_pair_source=mm_pair_source,
            mm_r_min=mm_r_min_arg,
            mm_pair_capacity_hint=max_pairs,
        )
    r0 = np.zeros((len(z), 3), dtype=np.float64)
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context

    with mlpot_jax_device_context():
        _, spherical_fn, get_update_fn = unpack_factory_result(
            factory(
                atomic_numbers=jnp.asarray(z),
                atomic_positions=jnp.asarray(r0),
                n_monomers=int(n_monomers),
                cutoff_params=cutoff_params,
                doML=do_ml,
                doMM=do_mm,
                doML_dimer=do_ml_dimer,
                backprop=False,
                create_ase_calculator=False,
            )
        )
    if verbose:
        print(
            f"Decomposed MLpot spherical_fn={spherical_fn!r} "
            f"(JIT bind doML={do_ml} doMM={do_mm} doML_dimer={do_ml_dimer})",
            flush=True,
        )
    model = DecomposedMlpotModel(
        spherical_fn,
        cutoff_params,
        int(n_monomers),
        np.asarray(atomic_numbers, dtype=int),
        cell=cell,
        do_mm=do_mm,
        get_update_fn=get_update_fn if do_mm else None,
        ml_compute_dtype=ml_compute_dtype,
        pending_do_ml=do_ml,
        pending_do_ml_dimer=do_ml_dimer,
        spatial_mpi=spatial_mpi,
        atoms_per_monomer=per,
        periodic_mm_config=periodic_mm_config,
        lr_solver=lr_solver,
        jax_pme_method=jax_pme_method,
        mm_pair_source=mm_pair_source,
        mm_r_min=mm_r_min_arg,
        mm_pair_capacity_hint=max_pairs,
    )
    return model


def _resolve_mlpot_warmup_box_pairs(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    use_pbc: bool,
    box_A: float | None,
) -> tuple[jnp.ndarray | None, Any, Any, bool]:
    """Box matrix and MM pair buffers for callback-forward warmup."""
    box: jnp.ndarray | None = None
    mm_pair_idx = None
    mm_pair_mask = None
    use_mm_pairs = False
    if use_pbc and box_A is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side
        from mmml.interfaces.pycharmmInterface.ml_dtypes import (
            as_ml_array,
            resolve_ml_compute_dtype,
        )

        side = float(box_A)
        box_np = cubic_box_matrix_from_side(side)
        dtype = resolve_ml_compute_dtype(getattr(model, "_ml_compute_dtype", None))
        box = as_ml_array(box_np, dtype=dtype)
        if model._do_mm and model._get_update_fn is not None:
            update_fn = model._get_update_fn(
                np.asarray(positions, dtype=np.float64),
                model._cutoff_params,
                box=box,
            )
            if update_fn is not None:
                box_np = _box_numpy_for_update(box)
                pos_np = np.asarray(positions, dtype=np.float64)
                if box_np is not None:
                    mm_pair_idx, mm_pair_mask = update_fn(pos_np, box=box_np)
                else:
                    mm_pair_idx, mm_pair_mask = update_fn(pos_np)
                if mm_pair_idx is not None and mm_pair_mask is not None:
                    use_mm_pairs = True
    return box, mm_pair_idx, mm_pair_mask, use_mm_pairs


def _warmup_mlpot_callback_forward(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    box: jnp.ndarray | None,
    mm_pair_idx: Any = None,
    mm_pair_mask: Any = None,
    use_mm_pairs: bool = False,
) -> None:
    """Compile the CHARMM MLpot callback ``forward_fn`` JIT (energy + forces).

    This is the authoritative warmup for decomposed MLpot: one stable ``jax.jit``
    wrapping ``spherical_fn``. Do not also call ``warmup_hybrid_spherical_cutoff``
    here — that duplicates XLA compilation of the same hybrid graph.
    """
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context
    from mmml.utils.jax_gpu_warmup import block_jax_values, run_jax_warmup_passes

    z = np.asarray(physnet_ml_atomic_numbers(model._atomic_numbers), dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    calc = model.get_pycharmm_calculator()
    if isinstance(calc, _DeferredDecomposedMlpotCalculator):
        calc = calc._ensure_real()
    if not isinstance(calc, DecomposedMlpotCalculator):
        return
    if (
        use_mm_pairs
        and model._do_mm
        and model._get_update_fn is not None
        and (box is not None or model._cell)
        and (mm_pair_idx is None or mm_pair_mask is None)
    ):
        raise RuntimeError(
            "_warmup_value_and_grad_for_model: PBC MM warmup requires mm_pair_idx/mm_pair_mask"
        )
    if use_mm_pairs and mm_pair_idx is not None and mm_pair_mask is not None:
        pair_idx = jnp.asarray(mm_pair_idx)
        pair_mask = jnp.asarray(mm_pair_mask)
    else:
        pair_idx, pair_mask = _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK
        use_mm_pairs = False

    device_ctx = (
        calc._mlpot_eval_device_context
        if isinstance(calc, DecomposedMlpotCalculator)
        else mlpot_jax_device_context
    )
    with device_ctx():
        positions_jax = as_ml_array(
            pos,
            dtype=resolve_ml_compute_dtype(model._ml_compute_dtype),
        )
        atomic_numbers_jax = jnp.asarray(z)
        forward_fn = calc._get_spherical_forward_fn(
            n_atoms=len(z),
            atomic_numbers_jax=atomic_numbers_jax,
            box_jax=box,
        )

        # Same candidate-list kwarg as calculate_charmm, so warmup compiles the
        # graph the callback will run (also seeds the list capacity).
        fwd_kwargs = calc._resolve_ml_dimer_candidates(pos, box)

        def _run_forward():
            return forward_fn(
                positions_jax,
                pair_idx,
                pair_mask,
                use_mm_pairs,
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                False,
                **fwd_kwargs,
            )

        run_jax_warmup_passes(
            "mlpot_callback_forward",
            2,
            _run_forward,
            block=lambda out: block_jax_values(out[0], out[1]),
        )


# Backward-compatible alias (tests / older call sites).
_warmup_value_and_grad_for_model = _warmup_mlpot_callback_forward


def materialize_deferred_mlpot_jax_before_sd(
    mlpot_ctx: Any,
    *,
    verbose: bool = False,
    probe_charmm_ener: bool = False,
    force_ener_probe: bool = False,
    sync_lists: bool = False,
) -> bool:
    """Build deferred JAX on CPU before MLpot SD.

    Materialize ``spherical_fn``, pre-compile the CHARMM callback
    ``_get_spherical_forward_fn`` JIT path, then under ``mpirun`` run one
    ``ENER FORCE`` to prime USER + Fortran ``enbond`` before ``steepd``.

    Serial ``python`` on MPI-linked CHARMM is rejected by
    :func:`assert_mpi_launcher_for_mlpot_sd` (``upinb`` / MPI pool risk).
    """
    del probe_charmm_ener, force_ener_probe, sync_lists
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        assert_mpi_launcher_for_mlpot_sd,
        charmm_lib_links_mpi,
        recover_mpi_for_charmm_after_jax,
        _under_mpirun,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        mlpot_skip_charmm_ener_force_before_first_sd,
        prime_charmm_hybrid_energy_before_mlpot_sd,
        rebind_mlpot_calculator_from_pycmodel,
    )

    recover_mpi_for_charmm_after_jax(phase="before pre-MLpot SD JAX materialize")

    if not mlpot_skip_charmm_ener_force_before_first_sd(mlpot_ctx):
        return False

    import jax

    backend = str(jax.default_backend()).lower()
    device_note = (
        "CPU XLA compile may idle host CPU"
        if backend == "cpu"
        else "GPU XLA compile may idle host CPU"
    )
    print(
        "Pre-MLpot SD: JAX hybrid compile/warmup starting "
        f"({device_note}; several minutes is normal for large PBC clusters)",
        flush=True,
    )

    assert_mpi_launcher_for_mlpot_sd(context="Pre-MLpot SD materialize")
    if charmm_lib_links_mpi() and not _under_mpirun() and verbose:
        print(
            "WARN: serial python with MPI-linked CHARMM (MMML_ALLOW_SERIAL_MPI_CHARMM=1?)",
            flush=True,
        )

    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if not isinstance(pyCModel, DecomposedMlpotModel):
        return False
    if not getattr(pyCModel, "_defer_jax_until_after_sd", False):
        return False

    if getattr(mlpot_ctx, "_mlpot_sd_jax_materialized", None) is True:
        rebind_mlpot_calculator_from_pycmodel(mlpot_ctx, verbose=False)
        prime_charmm_hybrid_energy_before_mlpot_sd(
            mlpot_ctx,
            verbose=verbose,
            context="Pre-MLpot SD",
        )
        return False

    rebind_mlpot_calculator_from_pycmodel(mlpot_ctx, verbose=False)

    pos = get_charmm_positions_array()
    use_pbc = bool(getattr(mlpot_ctx, "use_pbc", False))
    box_A = getattr(mlpot_ctx, "cubic_box_side_A", None)
    if box_A is None:
        box_A = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)

    did_work = False
    if getattr(pyCModel, "_spherical_fn", None) is None:
        calc = pyCModel.get_pycharmm_calculator()
        if isinstance(calc, _DeferredDecomposedMlpotCalculator):
            calc._ensure_real()
        elif not isinstance(calc, DecomposedMlpotCalculator):
            return False

        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            mlpot_spherical_forces_ev_angstrom,
        )

        forces = mlpot_spherical_forces_ev_angstrom(
            pyCModel,
            positions=pos,
            use_pbc=use_pbc,
            box_A=float(box_A) if box_A is not None else None,
        )
        if forces is None:
            raise RuntimeError(
                "Pre-MLpot SD: failed to materialize deferred JAX hybrid calculator"
            )

        recover_mpi_for_charmm_after_jax(
            phase="after deferred MLpot JAX CPU materialize",
        )
        did_work = True
        if verbose:
            print(
                "Pre-MLpot SD: materialized deferred JAX factory on CPU",
                flush=True,
            )

    calc = pyCModel.get_pycharmm_calculator()
    if isinstance(calc, _DeferredDecomposedMlpotCalculator):
        calc = calc._ensure_real()
    if isinstance(calc, DecomposedMlpotCalculator) and calc.spherical_fn is not None:
        if getattr(pyCModel, "_pre_sd_callback_forward_warmup_done", False):
            if verbose:
                print(
                    "Pre-MLpot SD: callback forward JIT already warm (skip)",
                    flush=True,
                )
        else:
            box, mm_pair_idx, mm_pair_mask, use_mm_pairs = _resolve_mlpot_warmup_box_pairs(
                pyCModel,
                pos,
                use_pbc=use_pbc,
                box_A=float(box_A) if box_A is not None else None,
            )
            print(
                "Pre-MLpot SD: warming CHARMM callback spherical_forward JIT "
                "(same path as steepd gete)",
                flush=True,
            )
            _warmup_value_and_grad_for_model(
                pyCModel,
                pos,
                box=box,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                use_mm_pairs=use_mm_pairs,
            )
            recover_mpi_for_charmm_after_jax(
                phase="after pre-MLpot SD callback forward warmup",
            )
            pyCModel._pre_sd_callback_forward_warmup_done = True
            did_work = True
            print(
                "Pre-MLpot SD: callback forward JIT ready",
                flush=True,
            )

    setattr(mlpot_ctx, "_mlpot_sd_jax_materialized", True)
    prime_charmm_hybrid_energy_before_mlpot_sd(
        mlpot_ctx,
        verbose=verbose,
        context="Pre-MLpot SD",
    )

    return did_work


@contextmanager
def charmm_mlpot_sd_jax_cpu_guard(model: DecomposedMlpotModel | Any | None):
    """Keep deferred JAX on CPU while CHARMM ``minimize_run_sd`` is active."""
    if not isinstance(model, DecomposedMlpotModel):
        yield
        return
    model._charmm_mlpot_sd_active = int(getattr(model, "_charmm_mlpot_sd_active", 0)) + 1
    try:
        yield
    finally:
        model._charmm_mlpot_sd_active = max(
            0, int(getattr(model, "_charmm_mlpot_sd_active", 1)) - 1
        )


def maybe_warmup_deferred_decomposed_mlpot(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    cell: Union[float, bool] | None = None,
    n_monomers: int,
    verbose: bool = False,
) -> None:
    """Promote and JIT-compile deferred JAX after MLpot SD (MPI-linked CHARMM)."""
    if int(n_monomers) <= 1:
        return
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        defer_jax_warmup_until_after_mlpot_sd,
    )

    if not defer_jax_warmup_until_after_mlpot_sd():
        return
    if not isinstance(model, DecomposedMlpotModel):
        return
    if not model._defer_jax_until_after_sd or model._jax_on_gpu:
        return
    warmup_decomposed_mlpot(model, positions, cell=cell, verbose=verbose)


def warmup_decomposed_mlpot(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    cell: Union[float, bool] | None = None,
    verbose: bool = False,
) -> None:
    """JIT-compile the MLpot CHARMM callback path (single ``forward_fn`` graph).

    Compiles one stable ``jax.jit(forward_fn)`` that wraps ``spherical_fn`` — the
    same entry point ``DecomposedMlpotCalculator.calculate_charmm`` uses. Avoids
    a separate ``warmup_hybrid_spherical_cutoff`` pass that would duplicate XLA
    work (slice/mul/scatter/PhysNet/jax-pme compiled twice).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import MetatomicMlpotModel
    from mmml.utils.jax_gpu_warmup import (
        ensure_xla_gpu_warmed,
        maybe_sanitize_process_env_for_ptxas,
    )

    if isinstance(model, MetatomicMlpotModel):
        if verbose:
            print(
                "Metatomic MLpot: skipping JAX warmup (torch ASE adapter)",
                flush=True,
            )
        return

    maybe_sanitize_process_env_for_ptxas()
    if getattr(model, "_jax_warmup_done", False) and model._spherical_fn is not None:
        return
    if model._defer_jax_until_after_sd and not model._jax_on_gpu:
        model.promote_jax_factory_to_gpu(force_after_sd=True)
    else:
        model._finalize_jax_factory()

    z = np.asarray(physnet_ml_atomic_numbers(model._atomic_numbers), dtype=int)
    r = np.asarray(positions, dtype=np.float64)
    pbc_cell = cell if cell is not None else model._cell
    box = None
    if pbc_cell:
        side = float(pbc_cell)
        box = jnp.asarray([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
    mm_pair_idx = None
    mm_pair_mask = None
    use_mm_pairs = False
    if model._do_mm and model._get_update_fn is not None:
        update_fn = model._get_update_fn(r, model._cutoff_params, box=box)
        if update_fn is not None:
            box_np = _box_numpy_for_update(box)
            if box_np is not None:
                mm_pair_idx, mm_pair_mask = update_fn(r, box=box_np)
            else:
                mm_pair_idx, mm_pair_mask = update_fn(r)
            if mm_pair_idx is None or mm_pair_mask is None:
                if pbc_cell:
                    raise RuntimeError(
                        "warmup_decomposed_mlpot: PBC MM neighbor update returned no pairs"
                    )
            else:
                use_mm_pairs = True

    if verbose:
        msg = f"Decomposed MLpot JAX warmup: {len(z)} atoms, {model._n_monomers} monomers"
        if model._do_mm:
            msg += " (ML+MM)"
        if pbc_cell:
            msg += f", MIC PBC L={float(pbc_cell):.3f} Å"
        print(msg, flush=True)

    prefer_cpu = bool(model._jax_pme_lr_active() and not model._jax_on_gpu)
    if not prefer_cpu:
        ensure_xla_gpu_warmed(force=False)

    if verbose:
        print(
            "Decomposed MLpot JAX warmup: mlpot_callback_forward (single jit)...",
            flush=True,
        )
    _warmup_mlpot_callback_forward(
        model,
        r,
        box=box,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        use_mm_pairs=use_mm_pairs,
    )
    from mmml.interfaces.pycharmmInterface.charmm_mpi import recover_mpi_for_charmm_after_jax

    recover_mpi_for_charmm_after_jax(phase="after decomposed MLpot JAX warmup")
    from mmml.utils.jax_gpu_warmup import maybe_log_jax_compile_timers

    maybe_log_jax_compile_timers()
    if verbose:
        # MM warmup may have silenced CHARMM; restore visibility before MLpot registration.
        from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_verbose

        pycharmm_verbose()
        print("Decomposed MLpot JAX warmup complete", flush=True)
    model._jax_warmup_done = True
