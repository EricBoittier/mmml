"""Small local tests for rigid-body FF presets (CGenFF / ZBL+MBD+multipoles).

No real Orbax/JSON model loads and no CHARMM — fixed multipoles and C6 arrays
are injected via ``EnergyContext.options``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from mmml.md.energy.registry import EnergyContext
from mmml.md.energy.terms.multipole import (
    HARTREE_TO_EV,
    MultipoleTerm,
    charge_dipole_pair_energy_au,
)
from mmml.md.energy.terms.zbl import DEFAULT_ZBL_CUTOFF_A, DEFAULT_ZBL_CUTON_A, ZBLTerm
from mmml.md.lowering import runconfig_from_md_system_args, terms_from_md_system_args
from mmml.md.system import MolecularSystem


def _two_diatomics(sep: float = 4.0):
    """Two rigid OH-like diatomics along x, separated by ``sep`` Å (COM–COM)."""
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = a + np.array([sep, 0.0, 0.0])
    R = np.concatenate([a, b])
    return MolecularSystem(
        R=R,
        Z=np.array([8, 1, 8, 1]),
        box=None,
        mol_id=np.array([0, 0, 1, 1]),
        monomer_indices=[np.array([0, 1]), np.array([2, 3])],
    )


def test_terms_rigid_default_cgenff():
    args = argparse.Namespace(sampler="rigid", ff=None, checkpoint=None, terms=None)
    assert terms_from_md_system_args(args) == ("mm_nonbonded",)


def test_terms_zbl_mbd_multipoles():
    args = argparse.Namespace(
        sampler="rigid", ff="zbl-mbd-multipoles", checkpoint=None, terms=None
    )
    assert terms_from_md_system_args(args) == ("zbl", "mbd", "multipole")


def test_terms_checkpoint_hybrid_overrides_rigid_default():
    args = argparse.Namespace(
        sampler="rigid", ff=None, checkpoint="ckpt.json", terms=None
    )
    assert terms_from_md_system_args(args) == ("ml_intra", "mm_nonbonded")


def test_runconfig_sampler_and_ff():
    args = argparse.Namespace(
        setup="pbc_nvt",
        dt_fs=1.0,
        ps=0.1,
        temperature=300.0,
        pressure=1.0,
        composition="TIP3:4",
        n_molecules=None,
        box_size=15.0,
        builder="packmol",
        seed=1,
        checkpoint=None,
        output_dir=None,
        sampler="rigid",
        ff="cgenff",
        mbd_checkpoint=None,
        mbd_weight=1.0,
        multipole_checkpoint=None,
    )
    cfg = runconfig_from_md_system_args(args)
    assert cfg.sampler == "rigid"
    assert cfg.terms == ("mm_nonbonded",)
    assert cfg.checkpoint is None


def test_zbl_energy_on_inside_cutoff_nonzero_outside_zero():
    import jax.numpy as jnp

    import mmml.md.energy.terms  # noqa: F401

    system = _two_diatomics(sep=0.3)  # O–O ~0.3 Å if we use atom 0 and 2
    # Place atoms 0 and 2 (oxygens of each monomer) at 0.2 Å
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]
    )
    system = MolecularSystem(
        R=R,
        Z=np.array([1, 1, 1, 1]),
        box=None,
        mol_id=np.array([0, 0, 1, 1]),
        monomer_indices=[np.array([0, 1]), np.array([2, 3])],
    )
    term = ZBLTerm(cuton_A=DEFAULT_ZBL_CUTON_A, cutoff_A=DEFAULT_ZBL_CUTOFF_A)
    fns = term.make(system, EnergyContext())
    # Bidirectional padded pairs including the close O–O-like pair
    pair_i = jnp.array([0, 2, 0, 1], dtype=jnp.int32)
    pair_j = jnp.array([2, 0, 1, 0], dtype=jnp.int32)
    e_close = float(fns.jax_energy_fn(R, pair_i=pair_i, pair_j=pair_j))
    assert e_close > 0.0

    R_far = R.copy()
    R_far[2, 0] = 2.0
    R_far[3, 0] = 3.0
    e_far = float(fns.jax_energy_fn(R_far, pair_i=pair_i, pair_j=pair_j))
    assert e_far == pytest.approx(0.0, abs=1e-12)


def test_fixed_multipole_translation_invariance():
    import jax.numpy as jnp

    system = _two_diatomics(sep=5.0)
    charges = np.array([1.0, -1.0])
    dipoles = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    opts = {
        "fixed_multipoles": {
            "charges": charges,
            "dipoles_body_bohr": dipoles,
            "ref_positions_A": system.R.copy(),
            "fragment_indices": system.monomer_indices,
        }
    }
    fns = MultipoleTerm().make(system, EnergyContext(options=opts))
    e0 = float(fns.jax_energy_fn(system.R))
    R2 = system.R + np.array([1.5, -0.7, 0.3])
    e1 = float(fns.jax_energy_fn(R2))
    assert e1 == pytest.approx(e0, rel=1e-6, abs=1e-8)


def test_fixed_multipole_matches_analytic_charge_dipole():
    """Two fragments, identity orientation → same as charge_dipole_pair_energy_au."""
    system = _two_diatomics(sep=5.0)
    charges = np.array([0.5, -0.5])
    dipoles = np.array([[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    opts = {
        "fixed_multipoles": {
            "charges": charges,
            "dipoles_body_bohr": dipoles,
            "ref_positions_A": system.R.copy(),
            "fragment_indices": system.monomer_indices,
        }
    }
    fns = MultipoleTerm().make(system, EnergyContext(options=opts))
    e = float(fns.jax_energy_fn(system.R))

    # COM of each diatomic: (0.5,0,0) and (5.5,0,0) in Å → bohr
    from mmml.md.energy.terms.multipole import ANGSTROM_TO_BOHR

    o0 = np.array([0.5, 0.0, 0.0]) * ANGSTROM_TO_BOHR
    o1 = np.array([5.5, 0.0, 0.0]) * ANGSTROM_TO_BOHR
    r_vec = o1 - o0
    e_ref = float(
        charge_dipole_pair_energy_au(r_vec, charges[0], dipoles[0], charges[1], dipoles[1])
    ) * HARTREE_TO_EV
    assert e == pytest.approx(e_ref, rel=1e-5, abs=1e-7)


def test_fixed_dispersion_qdo_changes_with_distance():
    import jax.numpy as jnp

    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy.terms.mbd import MBDDispersionTerm

    system = _two_diatomics(sep=4.0)
    coeffs = np.ones((4, 3), dtype=np.float64) * np.array([6.0, 0.0, 0.0])
    damp = np.ones(4, dtype=np.float64)
    opts = {
        "fixed_dispersion": {
            "coefficients_per_atom": coeffs,
            "damping_radii": damp,
            "weight": 1.0,
        },
        "mbd_weight": 1.0,
    }
    fns = MBDDispersionTerm(cutoff_A=12.0).make(system, EnergyContext(options=opts))
    pair_i = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    pair_j = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    e4 = float(fns.jax_energy_fn(system.R, pair_i=pair_i, pair_j=pair_j))
    R_close = system.R.copy()
    R_close[2:] -= np.array([1.5, 0.0, 0.0])
    e_close = float(fns.jax_energy_fn(R_close, pair_i=pair_i, pair_j=pair_j))
    # Closer → more negative QDO dispersion
    assert e_close < e4


def test_hybrid_zbl_mbd_multipole_with_injected_state():
    import jax.numpy as jnp

    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.assemble import build_hybrid_energy

    system = _two_diatomics(sep=5.0)
    opts = {
        "fixed_multipoles": {
            "charges": np.array([0.2, -0.2]),
            "dipoles_body_bohr": np.zeros((2, 3)),
            "ref_positions_A": system.R.copy(),
            "fragment_indices": system.monomer_indices,
        },
        "fixed_dispersion": {
            "coefficients_per_atom": np.ones((4, 3)) * np.array([1.0, 0.0, 0.0]),
            "damping_radii": np.ones(4),
            "weight": 1.0,
        },
        "zbl_cuton": DEFAULT_ZBL_CUTON_A,
        "zbl_cutoff": DEFAULT_ZBL_CUTOFF_A,
        "mbd_weight": 1.0,
    }
    energy = build_hybrid_energy(
        system, ("zbl", "mbd", "multipole"), EnergyContext(options=opts)
    )
    pair_i = jnp.array([0, 2], dtype=jnp.int32)
    pair_j = jnp.array([2, 0], dtype=jnp.int32)
    e = float(energy.as_jax_energy_fn()(system.R, pair_i=pair_i, pair_j=pair_j))
    assert np.isfinite(e)


def test_available_terms_include_qcml_set():
    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy import available_terms

    names = available_terms()
    assert "zbl" in names
    assert "mbd" in names
    assert "multipole" in names
