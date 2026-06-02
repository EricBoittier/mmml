"""MLpot callback via monomer/dimer PhysNet batches (``setup_calculator`` path)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol, setup_calculator
from mmml.interfaces.pycharmmInterface.mlpot.setup import physnet_ml_atomic_numbers


def resolve_ml_batch_size(
    n_monomers: int,
    explicit: Optional[int] = None,
) -> Optional[int]:
    """Chunk size for PhysNet forward passes (limits XLA LLVM compile RAM).

    DCM:90 sparse path evaluates ~590 systems (90 monomers + 500 dimers) per step.
    Without chunking, CPU JAX JIT can exhaust memory during LLVM compilation.
    """
    if explicit is not None:
        return int(explicit)
    env = (os.environ.get("MMML_MLPOT_ML_BATCH_SIZE") or "").strip()
    if env:
        return int(env)
    n = int(n_monomers)
    if n <= 10:
        return None
    if n >= 40:
        return 64
    if n >= 20:
        return 128
    return 256


class DecomposedMlpotCalculator:
    """CHARMM MLpot callback using padded monomer/dimer PhysNet evaluations."""

    def __init__(
        self,
        spherical_fn: Any,
        cutoff_params: CutoffParameters,
        n_monomers: int,
        atomic_numbers: np.ndarray,
        cell: Union[float, bool] = False,
    ) -> None:
        self.spherical_fn = spherical_fn
        self.cutoff_params = cutoff_params
        self.n_monomers = int(n_monomers)
        self.atomic_numbers = np.asarray(
            physnet_ml_atomic_numbers(atomic_numbers), dtype=np.int32
        )
        self.ev2kcal = float(ev2kcalmol)
        self._cell = float(cell) if cell else False

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
        from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context

        with mlpot_jax_device_context():
            out = self.spherical_fn(
                positions=jnp.asarray(pos),
                atomic_numbers=jnp.asarray(self.atomic_numbers[:n]),
                n_monomers=self.n_monomers,
                cutoff_params=self.cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=True,
            )
            e_kcal = float(jax.device_get(out.energy)) * self.ev2kcal
            forces = np.asarray(jax.device_get(out.forces), dtype=np.float64) * self.ev2kcal
        for i in range(n):
            dx[i] -= forces[i, 0]
            dy[i] -= forces[i, 1]
            dz[i] -= forces[i, 2]
        return e_kcal


class DecomposedMlpotModel:
    def __init__(
        self,
        spherical_fn: Any,
        cutoff_params: CutoffParameters,
        n_monomers: int,
        atomic_numbers: np.ndarray,
        cell: Union[float, bool] = False,
    ) -> None:
        self._spherical_fn = spherical_fn
        self._cutoff_params = cutoff_params
        self._n_monomers = int(n_monomers)
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._cell = float(cell) if cell else False

    def get_pycharmm_calculator(self, ml_atom_indices=None, ml_atomic_numbers=None, **kwargs):
        if ml_atomic_numbers is not None:
            z = np.asarray(ml_atomic_numbers, dtype=int)
        else:
            z = self._atomic_numbers
        return DecomposedMlpotCalculator(
            self._spherical_fn,
            self._cutoff_params,
            self._n_monomers,
            z,
            cell=self._cell,
        )


def build_decomposed_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    ml_batch_size: Optional[int] = None,
    cell: Union[float, bool] = False,
    verbose: bool = False,
) -> DecomposedMlpotModel:
    ckpt = Path(checkpoint).expanduser().resolve()
    cutoff_params = CutoffParameters()
    z = np.asarray(physnet_ml_atomic_numbers(atomic_numbers), dtype=int)
    per = [int(x) for x in atoms_per_monomer]
    max_atoms = max(per) * 2
    batch_size = resolve_ml_batch_size(int(n_monomers), ml_batch_size)
    if verbose and batch_size is not None:
        print(
            f"Decomposed MLpot: ml_batch_size={batch_size} "
            f"({int(n_monomers)} monomers; reduces JAX compile memory)",
            flush=True,
        )
    if verbose and cell:
        print(
            f"Decomposed MLpot: MIC PBC cubic cell={float(cell):.3f} Å",
            flush=True,
        )
    factory = setup_calculator(
        ATOMS_PER_MONOMER=per,
        N_MONOMERS=int(n_monomers),
        model_restart_path=str(ckpt),
        doMM=False,
        doML=True,
        doML_dimer=True,
        verbose=verbose,
        MAX_ATOMS_PER_SYSTEM=max_atoms,
        ml_batch_size=batch_size,
        cell=cell,
    )
    r0 = np.zeros((len(z), 3), dtype=np.float64)
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context

    with mlpot_jax_device_context():
        _, spherical_fn, _ = unpack_factory_result(
            factory(
                atomic_numbers=jnp.asarray(z),
                atomic_positions=jnp.asarray(r0),
                n_monomers=int(n_monomers),
                cutoff_params=cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=True,
                backprop=False,
                create_ase_calculator=False,
            )
        )
    model = DecomposedMlpotModel(
        spherical_fn,
        cutoff_params,
        int(n_monomers),
        np.asarray(atomic_numbers, dtype=int),
        cell=cell,
    )
    return model


def warmup_decomposed_mlpot(
    model: DecomposedMlpotModel,
    positions: np.ndarray,
    *,
    cell: Union[float, bool] | None = None,
    verbose: bool = False,
) -> None:
    """JIT-compile hybrid ML outside the CHARMM callback (before MLpot SD)."""
    from mmml.utils.jax_gpu_warmup import warmup_hybrid_spherical_cutoff

    z = np.asarray(physnet_ml_atomic_numbers(model._atomic_numbers), dtype=int)
    r = np.asarray(positions, dtype=np.float64)
    pbc_cell = cell if cell is not None else model._cell
    box = None
    if pbc_cell:
        side = float(pbc_cell)
        box = jnp.asarray([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
    if verbose:
        msg = f"Decomposed MLpot JAX warmup: {len(z)} atoms, {model._n_monomers} monomers"
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
        doMM=False,
        doML_dimer=True,
        box=box,
    )
    from mmml.interfaces.pycharmmInterface.charmm_mpi import recover_mpi_for_charmm_after_jax

    recover_mpi_for_charmm_after_jax(phase="after decomposed MLpot JAX warmup")
    if verbose:
        print("Decomposed MLpot JAX warmup complete", flush=True)
