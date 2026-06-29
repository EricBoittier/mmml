"""MLpot callback via monomer/dimer PhysNet batches (``setup_calculator`` path)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

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
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import resolve_ml_batch_size
from mmml.interfaces.pycharmmInterface.mlpot.setup import physnet_ml_atomic_numbers
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import resolve_ml_gpu_count
from mmml.interfaces.pycharmmInterface.jax_device_policy import (
    jax_cpu_until_mlpot_registered,
    mlpot_jax_device_context,
)
from mmml.utils.jax_gpu_warmup import ensure_xla_gpu_warmed

__all__ = ["resolve_ml_batch_size", "DecomposedMlpotCalculator", "DecomposedMlpotModel", "build_decomposed_mlpot_model", "warmup_decomposed_mlpot"]

_DUMMY_MM_PAIR_IDX = jnp.zeros((1, 2), dtype=jnp.int32)
_DUMMY_MM_PAIR_MASK = jnp.zeros((1,), dtype=jnp.bool_)


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
        spatial_mpi: bool = False,
        atoms_per_monomer: Sequence[int] | None = None,
        periodic_mm_config: Any | None = None,
    ) -> None:
        self.spherical_fn = spherical_fn
        self.cutoff_params = cutoff_params
        self.n_monomers = int(n_monomers)
        self.do_mm = bool(do_mm)
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
        self.ev2kcal = float(ev2kcalmol)
        self._cell = float(cell) if cell else False
        self.last_ml_forces: np.ndarray | None = None
        self._spherical_forward_fn: Any | None = None
        self._forward_cache_key: tuple[Any, ...] | None = None

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
        """
        dtype = resolve_ml_compute_dtype(self._ml_compute_dtype)
        box_present = box_jax is not None
        cache_key = (
            int(n_atoms),
            int(self.n_monomers),
            bool(self.do_mm),
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
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                kwargs: dict[str, Any] = dict(
                    positions=positions,
                    atomic_numbers=atomic_numbers_jax,
                    n_monomers=n_monomers,
                    cutoff_params=cutoff_params,
                    doML=True,
                    doMM=do_mm,
                    doML_dimer=True,
                    box=box,
                )
                if use_mm_pairs:
                    kwargs["mm_pair_idx"] = mm_pair_idx
                    kwargs["mm_pair_mask"] = mm_pair_mask
                if use_spatial:
                    kwargs["spatial_monomer_indices"] = spatial_monomer_indices
                    kwargs["spatial_dimer_indices"] = spatial_dimer_indices
                out = spherical_fn(**kwargs)
                return jnp.reshape(out.energy, (-1,))[0], out.forces

            fn = jax.jit(forward_fn, static_argnums=(4, 7))

            def wrapper(
                positions,
                mm_pair_idx,
                mm_pair_mask,
                use_mm_pairs,
                spatial_monomer_indices,
                spatial_dimer_indices,
                use_spatial,
            ):
                current_box = getattr(self, "_current_box", None)
                if current_box is None:
                    current_box = box_jax
                return fn(
                    positions,
                    current_box,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                    spatial_monomer_indices,
                    spatial_dimer_indices,
                    use_spatial,
                )

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
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                kwargs: dict[str, Any] = dict(
                    positions=positions,
                    atomic_numbers=atomic_numbers_jax,
                    n_monomers=n_monomers,
                    cutoff_params=cutoff_params,
                    doML=True,
                    doMM=do_mm,
                    doML_dimer=True,
                )
                if use_mm_pairs:
                    kwargs["mm_pair_idx"] = mm_pair_idx
                    kwargs["mm_pair_mask"] = mm_pair_mask
                if use_spatial:
                    kwargs["spatial_monomer_indices"] = spatial_monomer_indices
                    kwargs["spatial_dimer_indices"] = spatial_dimer_indices
                out = spherical_fn(**kwargs)
                return jnp.reshape(out.energy, (-1,))[0], out.forces

            owner._spherical_forward_fn = jax.jit(forward_fn, static_argnums=(3, 6))

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
            return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
        return jnp.asarray(mm_pair_idx), jnp.asarray(mm_pair_mask), True

    def _mlpot_eval_device_context(self):
        """CPU while MPI defer keeps the JAX factory off-GPU; else configured device."""
        parent = getattr(self, "_parent_model", None)
        if parent is not None and not getattr(parent, "_jax_on_gpu", True):
            return jax_cpu_until_mlpot_registered()
        return mlpot_jax_device_context()

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
        pos = np.array([x[:n], y[:n], z[:n]], dtype=np.float64).T
        box = None
        if self._cell or self._requires_callback_pbc_box():
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                cubic_box_matrix_from_side,
                resolve_mlpot_mic_box_side_A,
            )

            side, _ = resolve_mlpot_mic_box_side_A(
                fallback_side_A=float(self._cell) if self._cell else None,
                restart_path=getattr(self, "_npt_restart_read", None),
            )
            self._cell = side
            box = jnp.asarray(cubic_box_matrix_from_side(side))
        self._current_box = box
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
        forces = np.zeros((n, 3), dtype=np.float64)
        rank, mpi_size = mpi_rank_size()
        use_spatial = (
            bool(getattr(self, "_spatial_mpi", False) or spatial_mpi_enabled())
            and mpi_size > 1
            and bool(self._cell)
        )
        if run_ml:
            with self._mlpot_eval_device_context():
                mm_pair_idx, mm_pair_mask, use_mm_pairs = self._resolve_mm_pairs(pos, box)
                positions_jax = as_ml_array(
                    pos,
                    dtype=resolve_ml_compute_dtype(getattr(self, "_ml_compute_dtype", None)),
                )
                atomic_numbers_jax = jnp.asarray(self.atomic_numbers[:n])
                forward_fn = self._get_spherical_forward_fn(
                    n_atoms=n,
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
                e_raw, forces_ev = forward_fn(
                    positions_jax,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                    mono_jax,
                    dimer_jax,
                    use_spatial,
                )
                e_raw = jnp.where(jnp.isfinite(e_raw), e_raw, 0.0)
                forces_ev = jnp.where(jnp.isfinite(forces_ev), forces_ev, 0.0)
                e_kcal = float(jax.device_get(e_raw)) * self.ev2kcal
                forces = np.asarray(jax.device_get(forces_ev), dtype=np.float64) * self.ev2kcal
                self.last_ml_forces = np.asarray(forces, dtype=np.float64, copy=True)
                parent = getattr(self, "_parent_model", None)
                if parent is not None:
                    parent._last_ml_forces = self.last_ml_forces
                try:
                    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                        charmm_lib_links_mpi,
                    )
                    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
                        mlpot_jax_device_name,
                    )

                    if charmm_lib_links_mpi() and mlpot_jax_device_name() == "gpu":
                        from mmml.utils.jax_gpu_warmup import sync_jax_gpu_before_charmm

                        sync_jax_gpu_before_charmm(phase="after MLpot gete")
                except Exception:
                    pass
            parent = getattr(self, "_parent_model", None)
            if parent is not None and parent._defer_jax_pme_gpu_promote():
                parent._jax_pme_hybrid_first_ener_done = True
                if parent._defer_jax_until_after_sd and not parent._jax_on_gpu:
                    parent.promote_jax_factory_to_gpu()
                    self._spherical_forward_fn = None
                    self._forward_cache_key = None
                    if parent._spherical_fn is not None:
                        self.spherical_fn = parent._spherical_fn
                    if parent._get_update_fn is not None:
                        self._get_update_fn = parent._get_update_fn
                        self._cached_update_fn = None
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

            side = float(self._cell)
            e_kcal, forces = add_periodic_coulomb_to_callback(
                pos,
                box_side_A=side,
                cfg=periodic_cfg,
                energy_kcal=float(e_kcal),
                forces_kcal=np.asarray(forces, dtype=np.float64),
            )
        for i in range(n):
            dx[i] -= forces[i, 0]
            dy[i] -= forces[i, 1]
            dz[i] -= forces[i, 2]
        return e_kcal


class _DeferredDecomposedMlpotCalculator:
    """Defer JAX factory build until the first CHARMM ``ENER`` (after MLpot registration)."""

    def __init__(
        self,
        model: "DecomposedMlpotModel",
        *,
        ml_atomic_numbers: np.ndarray | None = None,
    ) -> None:
        self._model = model
        self._ml_atomic_numbers = (
            None if ml_atomic_numbers is None else np.asarray(ml_atomic_numbers, dtype=int)
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
        )
        return self._real

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
    ) -> None:
        self._spherical_fn = spherical_fn
        self._cutoff_params = cutoff_params
        self._n_monomers = int(n_monomers)
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._cell = float(cell) if cell else False
        self._do_mm = bool(do_mm)
        self._periodic_mm_config = periodic_mm_config
        self._get_update_fn = get_update_fn
        self._ml_compute_dtype = ml_compute_dtype
        self._spatial_mpi = bool(spatial_mpi)
        self._defer_jax_until_after_sd = bool(defer_jax_until_after_sd)
        self._defer_jax_until_mlpot_registered = bool(defer_jax_until_mlpot_registered)
        self._jax_on_gpu = spherical_fn is not None
        self._registered_calculator: DecomposedMlpotCalculator | None = None
        if atoms_per_monomer is None:
            apm = max(1, len(self._atomic_numbers) // max(1, int(n_monomers)))
            self._atoms_per_monomer = [apm] * int(n_monomers)
        else:
            self._atoms_per_monomer = [int(x) for x in atoms_per_monomer]
        self._last_ml_forces: np.ndarray | None = None
        self._spherical_forward_fn: Any | None = None
        self._forward_cache_key: tuple[Any, ...] | None = None
        self._pending_factory = pending_factory
        self._pending_factory_z = (
            None if pending_factory_z is None else np.asarray(pending_factory_z, dtype=int)
        )
        self._pending_do_ml = bool(pending_do_ml)
        self._pending_do_ml_dimer = bool(pending_do_ml_dimer)
        self._verbose = bool(verbose)
        self._lr_solver = lr_solver
        self._jax_pme_method = jax_pme_method
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
        cpu_only = self._defer_jax_until_after_sd and not gpu
        from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
            jax_compile_threads_context,
        )

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
        self._jax_on_gpu = not cpu_only
        if not cpu_only:
            self._pending_factory = None
            self._pending_factory_z = None
        if self._verbose:
            backend = "CPU" if cpu_only else "GPU"
            print(
                f"Decomposed MLpot spherical_fn={spherical_fn!r} "
                f"({backend}; JIT bind doML={self._pending_do_ml} doMM={self._do_mm} "
                f"doML_dimer={self._pending_do_ml_dimer})",
                flush=True,
            )

    def promote_jax_factory_to_gpu(self) -> None:
        """Rebuild ``spherical_fn`` on GPU after MLpot SD (MPI defer path)."""
        if not self._defer_jax_until_after_sd or self._jax_on_gpu:
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

    def _build_registered_calculator(
        self,
        *,
        ml_atomic_numbers: np.ndarray | None = None,
    ) -> DecomposedMlpotCalculator:
        self._finalize_jax_factory()
        if ml_atomic_numbers is not None:
            z = np.asarray(ml_atomic_numbers, dtype=int)
        else:
            z = self._atomic_numbers
        calc = DecomposedMlpotCalculator(
            self._spherical_fn,
            self._cutoff_params,
            self._n_monomers,
            z,
            cell=self._cell,
            do_mm=self._do_mm,
            get_update_fn=self._get_update_fn,
            ml_compute_dtype=self._ml_compute_dtype,
            spatial_mpi=self._spatial_mpi,
            atoms_per_monomer=self._atoms_per_monomer,
            periodic_mm_config=self._periodic_mm_config,
        )
        calc._parent_model = self
        self._registered_calculator = calc
        return calc

    def get_pycharmm_calculator(self, ml_atom_indices=None, ml_atomic_numbers=None, **kwargs):
        if self._spherical_fn is None and self._pending_factory is not None:
            if self._defer_jax_until_mlpot_registered:
                if self._defer_jax_until_after_sd:
                    self._finalize_jax_factory(gpu=False)
                else:
                    return self._build_registered_calculator(
                        ml_atomic_numbers=ml_atomic_numbers
                    )
            deferred = _DeferredDecomposedMlpotCalculator(
                self,
                ml_atomic_numbers=ml_atomic_numbers,
            )
            self._registered_calculator = deferred
            return deferred
        return self._build_registered_calculator(ml_atomic_numbers=ml_atomic_numbers)


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
) -> DecomposedMlpotModel:
    ckpt = Path(checkpoint).expanduser().resolve()
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
    dimer_cap = resolve_max_active_dimers(
        int(n_monomers),
        n_dimers_total,
        ml_max_active_dimers,
        free_space=free_space,
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
    do_ml = True
    include_mm = True if args is None else bool(getattr(args, "include_mm", True))
    do_mm = include_mm and not periodic_mode
    do_ml_dimer = True
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
    lr_solver = getattr(args, "lr_solver", None) if args is not None else None
    jax_pme_method = getattr(args, "jax_pme_method", None) if args is not None else None
    jax_pme_sr_cutoff = (
        float(getattr(args, "jax_pme_sr_cutoff", 6.0) or 6.0)
        if args is not None
        else 6.0
    )
    jax_pme_dispersion = (
        getattr(args, "jax_pme_dispersion", None) if args is not None else None
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
    factory = setup_calculator(
        ATOMS_PER_MONOMER=per,
        N_MONOMERS=int(n_monomers),
        model_restart_path=str(ckpt),
        doMM=do_mm,
        doML=do_ml,
        doML_dimer=do_ml_dimer,
        verbose=verbose,
        MAX_ATOMS_PER_SYSTEM=max_atoms,
        ml_batch_size=batch_size,
        ml_gpu_count=gpu_count,
        ml_max_active_dimers=ml_max_active_dimers,
        cell=cell,
        max_pairs=max_pairs,
        ml_compute_dtype=ml_compute_dtype,
        defer_xla_gpu_warmup=defer_jax_until_mlpot_registered,
        ml_switch_width=cutoff_params.ml_switch_width,
        mm_switch_on=cutoff_params.mm_switch_on,
        mm_switch_width=cutoff_params.mm_switch_width,
        complementary_handoff=cutoff_params.complementary_handoff,
        mm_r_min=getattr(args, "mm_r_min", None) if args is not None else None,
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
        mm_nonbond_mode=mm_nonbond_mode,
        periodic_charmm_vdw=(
            resolve_periodic_charmm_vdw(args) if args is not None else True
        ),
    )
    if verbose:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import report_charmm_topology_summary

        report_charmm_topology_summary()
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
        spatial_mpi=spatial_mpi,
        atoms_per_monomer=per,
        periodic_mm_config=periodic_mm_config,
        lr_solver=lr_solver,
        jax_pme_method=jax_pme_method,
    )
    return model


def _warmup_value_and_grad_for_model(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    box: jnp.ndarray | None,
    mm_pair_idx: Any = None,
    mm_pair_mask: Any = None,
    use_mm_pairs: bool = False,
) -> None:
    """Compile the CHARMM callback spherical forward path (SD / dynamics)."""
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context
    from mmml.utils.jax_gpu_warmup import block_jax_values, run_jax_warmup_passes

    z = np.asarray(physnet_ml_atomic_numbers(model._atomic_numbers), dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    calc = model.get_pycharmm_calculator()
    if use_mm_pairs and mm_pair_idx is not None and mm_pair_mask is not None:
        pair_idx = jnp.asarray(mm_pair_idx)
        pair_mask = jnp.asarray(mm_pair_mask)
    else:
        pair_idx, pair_mask = _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK
        use_mm_pairs = False

    with mlpot_jax_device_context():
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

        def _run_forward():
            return forward_fn(
                positions_jax,
                pair_idx,
                pair_mask,
                use_mm_pairs,
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                False,
            )

        run_jax_warmup_passes(
            "mlpot_spherical_forward",
            2,
            _run_forward,
            block=lambda out: block_jax_values(out[0], out[1]),
        )


def warmup_decomposed_mlpot(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    cell: Union[float, bool] | None = None,
    verbose: bool = False,
) -> None:
    """JIT-compile hybrid ML/MM on GPU (after MLpot SD when MPI defers JAX)."""
    from mmml.utils.jax_gpu_warmup import maybe_sanitize_process_env_for_ptxas

    maybe_sanitize_process_env_for_ptxas()
    if model._defer_jax_until_after_sd and not model._jax_on_gpu:
        model.promote_jax_factory_to_gpu()
    else:
        model._finalize_jax_factory()
    from mmml.utils.jax_gpu_warmup import warmup_hybrid_spherical_cutoff

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
            use_mm_pairs = True

    if verbose:
        msg = f"Decomposed MLpot JAX warmup: {len(z)} atoms, {model._n_monomers} monomers"
        if model._do_mm:
            msg += " (ML+MM)"
        if pbc_cell:
            msg += f", MIC PBC L={float(pbc_cell):.3f} Å"
        print(msg, flush=True)

    warmup_hybrid_spherical_cutoff(
        model._spherical_fn,
        atomic_numbers=jnp.asarray(z),
        positions=jnp.asarray(r),
        n_monomers=model._n_monomers,
        cutoff_params=model._cutoff_params,
        doML=True,
        doMM=model._do_mm,
        doML_dimer=True,
        box=box,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        prefer_cpu=model._jax_pme_lr_active() and not model._jax_on_gpu,
    )
    _warmup_value_and_grad_for_model(
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
