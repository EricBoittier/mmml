"""NpT MM pair list must be built in the same frame as NVT for Cartesian input.

With ``ensemble="npt"`` the hybrid calculator builds ``update_mm_pairs`` with
``fractional_coordinates=True``, whose contract is *fractional* positions plus
the current box. The ASE calculator holds Cartesian positions, so it must
convert before refreshing the pair list; before the fix it passed Cartesian
``R`` and the pair list was built for positions scaled by the box length.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mm_energy_forces import mm_pair_update_positions

_L = 12.0
_PAIR_CUTOFF = 5.0
_ATOMS_PER_MONOMER = 2
_N_MONOMERS = 6


def _positions() -> np.ndarray:
    rng = np.random.default_rng(3)
    coms = rng.uniform(0.5, _L - 0.5, size=(_N_MONOMERS, 3))
    offs = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    return (coms[:, None, :] + offs[None, :, :]).reshape(-1, 3)


def _mic(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    return d - box * np.round(d / box)


def _fake_build_mm_factory(seen: dict):
    """Fake ``build_mm_energy_forces_fn`` that honors the real frame contract."""
    n_atoms = _ATOMS_PER_MONOMER * _N_MONOMERS
    mono = np.repeat(np.arange(_N_MONOMERS), _ATOMS_PER_MONOMER)
    ii, jj = np.triu_indices(n_atoms, k=1)
    inter = mono[ii] != mono[jj]
    all_i, all_j = ii[inter], jj[inter]

    def fake_build_mm(R0, **kwargs):
        frac = bool(kwargs["fractional_coordinates"])
        seen["fractional_coordinates"] = frac

        def update_fn(positions, box=None, **_):
            box_np = np.asarray(box, dtype=np.float64)
            P = np.asarray(positions, dtype=np.float64)
            if frac:  # contract of update_mm_pairs(fractional_coordinates=True)
                P = P * box_np
            r = np.linalg.norm(_mic(P[all_j] - P[all_i], box_np), axis=1)
            mask = r < _PAIR_CUTOFF
            return (
                jnp.asarray(np.stack([all_i, all_j], axis=1), dtype=jnp.int32),
                jnp.asarray(mask, dtype=jnp.float32),
            )

        def energy(positions, pair_idx, pair_mask, box):
            d = positions[pair_idx[:, 1]] - positions[pair_idx[:, 0]]
            d = d - box * jnp.round(d / box)
            r = jnp.sqrt(jnp.sum(d * d, axis=1))
            return jnp.sum(pair_mask * jnp.exp(-r))

        def mm_fn(positions, pair_idx, pair_mask, box_override=None, charges=None):
            box = jnp.asarray(box_override).reshape(-1)[:3]
            e, g = jax.value_and_grad(energy)(positions, pair_idx, pair_mask, box)
            return e, -g

        return mm_fn, update_fn

    return fake_build_mm


def _mm_energy_forces(ensemble: str) -> tuple[float, np.ndarray, bool]:
    from ase import Atoms

    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    pos = _positions()
    n_atoms = pos.shape[0]
    z = np.full((n_atoms,), 6, dtype=np.int32)
    seen: dict = {}
    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=_fake_build_mm_factory(seen),
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=_ATOMS_PER_MONOMER,
            N_MONOMERS=_N_MONOMERS,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=False,
            doMM=True,
            doML_dimer=False,
            MAX_ATOMS_PER_SYSTEM=_ATOMS_PER_MONOMER * 2,
            cell=_L,
            ensemble=ensemble,
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=False,
        )
        calc, _, _ = factory(
            atomic_numbers=jnp.asarray(z),
            atomic_positions=jnp.asarray(pos),
            n_monomers=_N_MONOMERS,
            cutoff_params=CutoffParameters(),
            doML=False,
            doMM=True,
            doML_dimer=False,
            backprop=False,
        )
        atoms = Atoms(numbers=z, positions=pos, cell=np.diag([_L] * 3), pbc=True)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces())
    return e, f, seen["fractional_coordinates"]


def test_mm_pair_update_positions_frames() -> None:
    R = _positions()
    box = np.array([_L, _L, 2 * _L])
    np.testing.assert_allclose(mm_pair_update_positions(R, box, True), R / box)
    assert mm_pair_update_positions(R, box, False) is R
    assert mm_pair_update_positions(R, None, True) is R
    np.testing.assert_allclose(
        np.asarray(mm_pair_update_positions(jnp.asarray(R), np.diag(box), True)),
        R / box,
        rtol=1e-6,
    )


@pytest.mark.filterwarnings("ignore")
def test_npt_and_nvt_give_same_mm_energy_forces() -> None:
    e_nvt, f_nvt, frac_nvt = _mm_energy_forces("nvt")
    e_npt, f_npt, frac_npt = _mm_energy_forces("npt")
    assert frac_nvt is False and frac_npt is True
    assert e_nvt != 0.0
    np.testing.assert_allclose(e_npt, e_nvt, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f_npt, f_nvt, rtol=1e-5, atol=1e-7)
