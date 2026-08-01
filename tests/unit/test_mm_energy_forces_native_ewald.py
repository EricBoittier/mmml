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
    assert float(vdw) == 0.0  # include_lj defaults False
    assert float(elec) == pytest.approx(float(e_ref), rel=1e-9)

    f_ref = -jax.grad(
        lambda r: hybrid_ewald_coulomb_energy(r, mol_id, jnp.asarray(charges), box_length_A=L)
    )(R_j)
    assert np.allclose(np.asarray(forces), np.asarray(f_ref), atol=1e-8)
    assert np.all(np.isfinite(np.asarray(forces)))


def test_ewald_branch_include_lj_adds_nonzero_vdw():
    """include_lj=True returns VDW channel from COM-switched intermolecular LJ."""
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    # Two monomers close enough that mm_switch is on (complementary handoff).
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    charges = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)
    L = 30.0
    box = np.diag([L, L, L])
    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = charges
    fake_psf.get_iac.return_value = np.ones(4, dtype=np.int32)

    with patch("pycharmm.psf", fake_psf), patch(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._get_actual_psf_charges",
        return_value=charges,
    ):
        mm_fn = build_mm_energy_forces_fn(
            R,
            total_atoms=4,
            n_monomers=2,
            monomer_offsets=np.array([0, 2, 4], dtype=np.int32),
            atoms_per_monomer_list=[2, 2],
            lambda_monomer=np.ones(2, dtype=np.float64),
            ml_switch_width=1.5,
            mm_switch_on=8.0,
            mm_switch_width=5.0,
            complementary_handoff=True,
            pbc_cell=box,
            lr_solver="ewald",
            include_lj=True,
            defer_xla_gpu_warmup=True,
            debug=False,
        )
    fn, _ = mm_fn
    e, forces, vdw, elec = fn(jnp.asarray(R), charges=jnp.asarray(charges))
    assert float(vdw) != 0.0
    assert float(e) == pytest.approx(float(elec) + float(vdw), rel=1e-9)
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


def test_ewald_include_intra_false_zeroes_single_tip3_forces():
    """``ewald_include_intra`` must reach the native Ewald kernel (not just CLI strings).

    One TIP3P in a box: with ``include_intramolecular=False`` (``--ewald-omit-self``
    path) there is no cross-monomer Coulomb, so max|F| must be ~0. With the default
    full-box Ewald, intramolecular O–H Coulomb yields large forces
    (~100 kcal/mol/Å ≈ 5 eV/Å). A wiring bug that forgets to forward
    ``ewald_include_intra=False`` into ``build_mm_energy_forces_fn`` would leave
    those intramolecular forces on.
    """
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    # Bundled tip3.pdb geometry (OH2, H1, H2) + TIP3P charges.
    R = np.array(
        [
            [1.160, 1.590, 0.942],
            [0.666, 0.833, 1.255],
            [2.041, 1.464, 1.293],
        ],
        dtype=np.float64,
    )
    charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
    L = 20.0
    box = np.diag([L, L, L])
    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = charges
    fake_psf.get_iac.return_value = np.ones(3, dtype=np.int32)

    def _max_force(*, include_intra: bool) -> float:
        with patch("pycharmm.psf", fake_psf), patch(
            "mmml.interfaces.pycharmmInterface.mm_energy_forces._get_actual_psf_charges",
            return_value=charges,
        ):
            mm_fn = build_mm_energy_forces_fn(
                R,
                total_atoms=3,
                n_monomers=1,
                monomer_offsets=np.array([0, 3], dtype=np.int32),
                atoms_per_monomer_list=[3],
                lambda_monomer=np.ones(1, dtype=np.float64),
                ml_switch_width=1.0,
                mm_switch_on=6.0,
                mm_switch_width=4.0,
                pbc_cell=box,
                lr_solver="ewald",
                # Match --ewald-omit-self: drop self + intra together.
                ewald_include_self=False,
                ewald_include_intra=include_intra,
                defer_xla_gpu_warmup=True,
                debug=False,
            )
        fn, update_fn = mm_fn
        assert update_fn is None
        _e, forces, _vdw, _elec = fn(jnp.asarray(R), charges=jnp.asarray(charges))
        return float(np.max(np.abs(np.asarray(forces))))

    max_f_omit = _max_force(include_intra=False)
    max_f_full = _max_force(include_intra=True)

    assert max_f_omit < 1.0e-6, f"omit-intra should zero single-TIP3 forces, got {max_f_omit}"
    # ~116 kcal/mol/Å on this geometry; keep a soft floor so CI catches silent drops.
    assert max_f_full > 50.0, f"full Ewald TIP3 intramolecular |F| too small: {max_f_full}"
