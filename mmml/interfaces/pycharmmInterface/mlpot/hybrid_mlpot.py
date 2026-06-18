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

__all__ = ["resolve_ml_batch_size", "DecomposedMlpotCalculator", "DecomposedMlpotModel", "build_decomposed_mlpot_model", "warmup_decomposed_mlpot"]

_DUMMY_MM_PAIR_IDX = jnp.zeros((1, 2), dtype=jnp.int32)
_DUMMY_MM_PAIR_MASK = jnp.zeros((1,), dtype=jnp.bool_)


def _box_cache_key(box: jnp.ndarray | None) -> tuple[float, float, float] | None:
    if box is None:
        return None
    return (float(box[0, 0]), float(box[1, 1]), float(box[2, 2]))


def _box_numpy_for_update(box: jnp.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    side = float(box[0, 0])
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
    cp = cutoff_params
    comp = getattr(cp, "complementary_handoff", True)
    print(
        "Decomposed MLpot factory (setup_calculator return):\n"
        f"  factory={factory!r}\n"
        f"  module={getattr(factory, '__module__', '?')}\n"
        f"  name={getattr(factory, '__name__', type(factory).__name__)}\n"
        f"  model_restart_path={checkpoint}\n"
        f"  n_monomers={n_monomers} atoms_per_monomer={list(atoms_per_monomer)}\n"
        f"  MAX_ATOMS_PER_SYSTEM={max_atoms_per_system}\n"
        f"  doML={do_ml} doMM={do_mm} doML_dimer={do_ml_dimer}\n"
        f"  ml_switch_width={cp.ml_switch_width} mm_switch_on={cp.mm_switch_on} "
        f"mm_switch_width={cp.mm_switch_width} complementary_handoff={comp}\n"
        f"  ml_batch_size={ml_batch_size} ml_gpu_count={ml_gpu_count} "
        f"ml_max_active_dimers={ml_max_active_dimers}\n"
        f"  cell={cell!r}",
        flush=True,
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
    ) -> None:
        self.spherical_fn = spherical_fn
        self.cutoff_params = cutoff_params
        self.n_monomers = int(n_monomers)
        self.do_mm = bool(do_mm)
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
        self._value_and_grad_fn: Any | None = None
        self._vg_cache_key: tuple[Any, ...] | None = None

    def _grad_cache_owner(self) -> DecomposedMlpotCalculator | DecomposedMlpotModel:
        parent = getattr(self, "_parent_model", None)
        return parent if parent is not None else self

    def _get_value_and_grad_fn(
        self,
        *,
        n_atoms: int,
        atomic_numbers_jax: jnp.ndarray,
        box_jax: jnp.ndarray | None,
    ) -> Any:
        """Return a cached ``jit(value_and_grad)`` for MLpot SD/dynamics callbacks."""
        dtype = resolve_ml_compute_dtype(self._ml_compute_dtype)
        cache_key = (
            int(n_atoms),
            int(self.n_monomers),
            bool(self.do_mm),
            dtype,
            _box_cache_key(box_jax),
            bool(self._spatial_mpi),
        )
        owner = self._grad_cache_owner()
        if owner._vg_cache_key == cache_key and owner._value_and_grad_fn is not None:
            return owner._value_and_grad_fn

        spherical_fn = self.spherical_fn
        cutoff_params = self.cutoff_params
        n_monomers = self.n_monomers
        do_mm = self.do_mm

        def energy_scalar(
            positions: jnp.ndarray,
            mm_pair_idx: jnp.ndarray,
            mm_pair_mask: jnp.ndarray,
            use_mm_pairs: bool,
            spatial_monomer_indices: jnp.ndarray,
            spatial_dimer_indices: jnp.ndarray,
            use_spatial: bool,
        ) -> jnp.ndarray:
            kwargs: dict[str, Any] = dict(
                positions=positions,
                atomic_numbers=atomic_numbers_jax,
                n_monomers=n_monomers,
                cutoff_params=cutoff_params,
                doML=True,
                doMM=do_mm,
                doML_dimer=True,
            )
            if box_jax is not None:
                kwargs["box"] = box_jax
            if use_mm_pairs:
                kwargs["mm_pair_idx"] = mm_pair_idx
                kwargs["mm_pair_mask"] = mm_pair_mask
            if use_spatial:
                kwargs["spatial_monomer_indices"] = spatial_monomer_indices
                kwargs["spatial_dimer_indices"] = spatial_dimer_indices
            out = spherical_fn(**kwargs)
            return jnp.reshape(out.energy, (-1,))[0]

        fn = jax.jit(
            jax.value_and_grad(energy_scalar, argnums=0),
            static_argnums=(3, 6),
        )
        owner._value_and_grad_fn = fn
        owner._vg_cache_key = cache_key
        return fn

    def _resolve_mm_pairs(
        self,
        pos: np.ndarray,
        box: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        if not self.do_mm or self._get_update_fn is None:
            return _DUMMY_MM_PAIR_IDX, _DUMMY_MM_PAIR_MASK, False
        update_fn = self._get_update_fn(pos, self.cutoff_params, box=box)
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
        if self._cell:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                cubic_box_matrix_from_side,
                resolve_charmm_cubic_box_side_A,
            )

            try:
                side, _ = resolve_charmm_cubic_box_side_A(
                    fallback_side_A=float(self._cell) if self._cell else None,
                    restart_path=getattr(self, "_npt_restart_read", None),
                )
                self._cell = side
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                side = float(self._cell)
            box = jnp.asarray(cubic_box_matrix_from_side(side))
        from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context
        from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
            get_mlpot_profile_stats,
            mlpot_profiling_enabled,
        )

        if mlpot_profiling_enabled():
            get_mlpot_profile_stats().record_charmm_gap()
        t0 = time.perf_counter()
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import (
            broadcast_mlpot_result,
            mlpot_runs_on_this_rank,
        )

        run_ml = mlpot_runs_on_this_rank()
        e_kcal = 0.0
        forces = np.zeros((n, 3), dtype=np.float64)
        if run_ml:
            with mlpot_jax_device_context():
                mm_pair_idx, mm_pair_mask, use_mm_pairs = self._resolve_mm_pairs(pos, box)
                positions_jax = as_ml_array(
                    pos,
                    dtype=resolve_ml_compute_dtype(getattr(self, "_ml_compute_dtype", None)),
                )
                atomic_numbers_jax = jnp.asarray(self.atomic_numbers[:n])
                value_and_grad_fn = self._get_value_and_grad_fn(
                    n_atoms=n,
                    atomic_numbers_jax=atomic_numbers_jax,
                    box_jax=box,
                )
                e_raw, grad = value_and_grad_fn(
                    positions_jax,
                    mm_pair_idx,
                    mm_pair_mask,
                    use_mm_pairs,
                )
                e_raw = jnp.where(jnp.isfinite(e_raw), e_raw, 0.0)
                grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
                e_kcal = float(jax.device_get(e_raw)) * self.ev2kcal
                forces = -np.asarray(jax.device_get(grad), dtype=np.float64) * self.ev2kcal
                self.last_ml_forces = np.asarray(forces, dtype=np.float64, copy=True)
                parent = getattr(self, "_parent_model", None)
                if parent is not None:
                    parent._last_ml_forces = self.last_ml_forces
        forces, e_kcal = broadcast_mlpot_result(forces, e_kcal, n)
        self.last_ml_forces = np.asarray(forces, dtype=np.float64, copy=True)
        parent = getattr(self, "_parent_model", None)
        if parent is not None:
            parent._last_ml_forces = self.last_ml_forces
        if mlpot_profiling_enabled():
            get_mlpot_profile_stats().record_ml(time.perf_counter() - t0)
        for i in range(n):
            dx[i] -= forces[i, 0]
            dy[i] -= forces[i, 1]
            dz[i] -= forces[i, 2]
        return e_kcal


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
    ) -> None:
        self._spherical_fn = spherical_fn
        self._cutoff_params = cutoff_params
        self._n_monomers = int(n_monomers)
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._cell = float(cell) if cell else False
        self._do_mm = bool(do_mm)
        self._get_update_fn = get_update_fn
        self._ml_compute_dtype = ml_compute_dtype
        self._last_ml_forces: np.ndarray | None = None
        self._value_and_grad_fn: Any | None = None
        self._vg_cache_key: tuple[Any, ...] | None = None
        self._pending_factory = pending_factory
        self._pending_factory_z = (
            None if pending_factory_z is None else np.asarray(pending_factory_z, dtype=int)
        )
        self._pending_do_ml = bool(pending_do_ml)
        self._pending_do_ml_dimer = bool(pending_do_ml_dimer)
        self._verbose = bool(verbose)

    def _finalize_jax_factory(self) -> None:
        """Build ``spherical_fn`` on GPU after CHARMM MLpot ``upinb`` (``MLpot.__init__``)."""
        if self._spherical_fn is not None:
            return
        if self._pending_factory is None or self._pending_factory_z is None:
            raise RuntimeError("DecomposedMlpotModel: JAX factory was not initialized")
        from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
            jax_compile_threads_context,
        )
        from mmml.utils.jax_gpu_warmup import ensure_xla_gpu_warmed

        with jax_compile_threads_context():
            ensure_xla_gpu_warmed()
            z = self._pending_factory_z
            r0 = np.zeros((len(z), 3), dtype=np.float64)
            from mmml.interfaces.pycharmmInterface.jax_device_policy import (
                mlpot_jax_device_context,
            )

            with mlpot_jax_device_context():
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
        self._pending_factory = None
        self._pending_factory_z = None
        if self._verbose:
            print(
                f"Decomposed MLpot spherical_fn={spherical_fn!r} "
                f"(JIT bind doML={self._pending_do_ml} doMM={self._do_mm} "
                f"doML_dimer={self._pending_do_ml_dimer})",
                flush=True,
            )

    def get_pycharmm_calculator(self, ml_atom_indices=None, ml_atomic_numbers=None, **kwargs):
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
        )
        calc._parent_model = self
        return calc


def build_decomposed_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    cell: Union[float, bool] = False,
    verbose: bool = False,
    args: Any | None = None,
    ml_compute_dtype: str | None = None,
    defer_jax_until_mlpot_registered: bool = False,
) -> DecomposedMlpotModel:
    ckpt = Path(checkpoint).expanduser().resolve()
    if args is not None and ml_compute_dtype is None:
        ml_compute_dtype = getattr(args, "ml_compute_dtype", None)
    cutoff_params = (
        cutoff_parameters_from_args(args) if args is not None else CutoffParameters()
    )
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
    do_mm = True
    do_ml_dimer = True
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
        ml_compute_dtype=ml_compute_dtype,
        defer_xla_gpu_warmup=defer_jax_until_mlpot_registered,
    )
    if verbose:
        _print_setup_calculator_factory_summary(
            factory,
            checkpoint=ckpt,
            n_monomers=int(n_monomers),
            atoms_per_monomer=per,
            do_ml=do_ml,
            do_mm=do_mm,
            do_ml_dimer=do_ml_dimer,
            cutoff_params=cutoff_params,
            max_atoms_per_system=max_atoms,
            ml_batch_size=batch_size,
            ml_gpu_count=gpu_count,
            ml_max_active_dimers=ml_max_active_dimers,
            cell=cell,
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
    """Compile the CHARMM callback ``value_and_grad`` path (SD / dynamics)."""
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
        value_and_grad_fn = calc._get_value_and_grad_fn(
            n_atoms=len(z),
            atomic_numbers_jax=atomic_numbers_jax,
            box_jax=box,
        )

        def _run_value_and_grad():
            return value_and_grad_fn(
                positions_jax,
                pair_idx,
                pair_mask,
                use_mm_pairs,
            )

        run_jax_warmup_passes(
            "mlpot_value_and_grad",
            2,
            _run_value_and_grad,
            block=lambda out: block_jax_values(out[0], out[1]),
        )


def warmup_decomposed_mlpot(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    cell: Union[float, bool] | None = None,
    verbose: bool = False,
) -> None:
    """JIT-compile hybrid ML/MM outside the CHARMM callback (before MLpot SD)."""
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
