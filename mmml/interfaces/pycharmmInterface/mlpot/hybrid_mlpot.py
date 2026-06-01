"""MLpot callback via monomer/dimer PhysNet batches (``setup_calculator`` path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol, setup_calculator


class DecomposedMlpotCalculator:
    """CHARMM MLpot callback using padded monomer/dimer PhysNet evaluations."""

    def __init__(
        self,
        spherical_fn: Any,
        cutoff_params: CutoffParameters,
        n_monomers: int,
        atomic_numbers: np.ndarray,
    ) -> None:
        self.spherical_fn = spherical_fn
        self.cutoff_params = cutoff_params
        self.n_monomers = int(n_monomers)
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        self.ev2kcal = float(ev2kcalmol)

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
    ) -> None:
        self._spherical_fn = spherical_fn
        self._cutoff_params = cutoff_params
        self._n_monomers = int(n_monomers)
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)

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
        )


def build_decomposed_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    verbose: bool = False,
) -> DecomposedMlpotModel:
    ckpt = Path(checkpoint).expanduser().resolve()
    cutoff_params = CutoffParameters()
    z = np.asarray(atomic_numbers, dtype=int)
    per = [int(x) for x in atoms_per_monomer]
    max_atoms = max(per) * 2
    factory = setup_calculator(
        ATOMS_PER_MONOMER=per,
        N_MONOMERS=int(n_monomers),
        model_restart_path=str(ckpt),
        doMM=False,
        doML=True,
        doML_dimer=True,
        verbose=verbose,
        MAX_ATOMS_PER_SYSTEM=max_atoms,
    )
    r0 = np.zeros((len(z), 3), dtype=np.float64)
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
    return DecomposedMlpotModel(
        spherical_fn,
        cutoff_params,
        int(n_monomers),
        np.asarray(atomic_numbers, dtype=int),
    )
