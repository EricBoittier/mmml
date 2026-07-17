"""lr_solver="ewald" full-box path in build_mm_energy_forces_fn.

Bypasses the switched-pair/cell-list machinery entirely (no LJ, no
exclusions, no switching -- matches training's hybrid_ewald_coulomb_energy
exactly, the same operator lr_solver="ewald" trains against). Verifies the
wired-in mm_fn actually reproduces that reference function, not just that it
runs without raising.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_ewald_branch_matches_hybrid_ewald_coulomb_energy():
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    rng = np.random.default_rng(3)
    n_atoms = 8
    n_mono = 2
    offsets = np.array([0, 4, 8], dtype=np.int32)
    atoms_per = [4, 4]
    lambda_m = np.ones(n_mono, dtype=np.float64)
    R = rng.uniform(2.0, 22.0, size=(n_atoms, 3))
    L = 24.0
    box = np.diag([L, L, L])

    charges = rng.uniform(-0.5, 0.5, size=n_atoms)
    charges[:4] -= charges[:4].mean()
    charges[4:] -= charges[4:].mean()

    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = charges
    fake_psf.get_iac.return_value = np.ones(n_atoms, dtype=np.int32)

    with patch("pycharmm.psf", fake_psf), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._get_actual_psf_charges",
        return_value=charges,
    ):
        mm_fn = build_mm_energy_forces_fn(
            R,
            total_atoms=n_atoms,
            n_monomers=n_mono,
            monomer_offsets=offsets,
            atoms_per_monomer_list=atoms_per,
            lambda_monomer=lambda_m,
            ml_switch_width=1.0,
            mm_switch_on=6.0,
            mm_switch_width=4.0,
            pbc_cell=box,
            lr_solver="ewald",
            defer_xla_gpu_warmup=True,
            debug=False,
        )

    # returned as (fn, None) or a bare callable, matching get_MM_energy_forces_fns
    assert isinstance(mm_fn, tuple)
    fn, update_fn = mm_fn
    assert update_fn is None
    assert callable(fn)

    R_j = jnp.asarray(R)
    e, forces, vdw, elec = fn(R_j, charges=jnp.asarray(charges))

    mol_id = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    e_ref = hybrid_ewald_coulomb_energy(
        R_j, mol_id, jnp.asarray(charges), box_length_A=L,
    )
    assert float(e) == pytest.approx(float(e_ref), rel=1e-9)
    assert float(vdw) == 0.0  # no LJ term at all for this lr_solver
    assert float(elec) == pytest.approx(float(e_ref), rel=1e-9)

    f_ref = -jax.grad(
        lambda r: hybrid_ewald_coulomb_energy(r, mol_id, jnp.asarray(charges), box_length_A=L)
    )(R_j)
    assert np.allclose(np.asarray(forces), np.asarray(f_ref), atol=1e-8)
    assert np.all(np.isfinite(np.asarray(forces)))


def test_ewald_branch_requires_pbc_cell():
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    n_atoms = 8
    n_mono = 2
    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = np.zeros(n_atoms)
    fake_psf.get_iac.return_value = np.ones(n_atoms, dtype=np.int32)

    with patch("pycharmm.psf", fake_psf), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._get_actual_psf_charges",
        return_value=np.zeros(n_atoms),
    ):
        with pytest.raises(ValueError, match="pbc_cell|PBC cell"):
            build_mm_energy_forces_fn(
                np.random.default_rng(1).uniform(0, 10, size=(n_atoms, 3)),
                total_atoms=n_atoms,
                n_monomers=n_mono,
                monomer_offsets=np.array([0, 4, n_atoms], dtype=np.int32),
                atoms_per_monomer_list=[4, 4],
                lambda_monomer=np.ones(n_mono),
                ml_switch_width=1.0,
                mm_switch_on=6.0,
                mm_switch_width=4.0,
                pbc_cell=None,
                lr_solver="ewald",
                defer_xla_gpu_warmup=True,
            )
