"""
Test that ASE calculator and JAX-MD produce consistent energies and forces for PBC.

Verifies that the ASE AseDimerCalculator (used in ASE MD) and the JAX-MD wrapped_energy_fn
(used in JAX-MD simulations) yield the same energy and forces for the same configuration.
This ensures PBC handling is aligned between both code paths.
"""
import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np


def _get_ckpt():
    ckpt_env = os.environ.get("MMML_CKPT")
    return Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")


@pytest.mark.skipif(
    importlib.util.find_spec("pycharmm") is None,
    reason="pycharmm not available in this environment",
)
@pytest.mark.skipif(
    importlib.util.find_spec("jax_md") is None,
    reason="jax_md not available in this environment",
)
def test_ase_jaxmd_pbc_energy_forces_consistency():
    """
    Compare ASE calculator energy/forces with JAX-MD wrapped_energy_fn for the same PBC config.

    Uses the same pbc_map and spherical_cutoff_calculator for both paths, mirroring the
    setup in run_sim.py. Asserts energies and forces match within tolerance.
    """
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax not available in this environment")
    if importlib.util.find_spec("e3x") is None:
        pytest.skip("e3x not available in this environment")
    if importlib.util.find_spec("ase") is None:
        pytest.skip("ase not available in this environment")

    ckpt = _get_ckpt()
    if not ckpt.exists():
        pytest.skip("No checkpoints present for ML model")

    import jax
    import jax.numpy as jnp
    from jax import jit
    import ase

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

    n_monomers = 2
    n_atoms_monomer = 10
    n_atoms = n_monomers * n_atoms_monomer
    cell_length = 40.0

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms_monomer,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        cell=cell_length,
    )

    cell_matrix = jnp.array([
        [cell_length, 0, 0],
        [0, cell_length, 0],
        [0, 0, cell_length],
    ])
    mol_id = jnp.array([
        i * jnp.ones(n_atoms_monomer, dtype=jnp.int32)
        for i in range(n_monomers)
    ], dtype=jnp.int32)
    pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=n_monomers)

    cutoff_params = CutoffParameters()
    key = jax.random.PRNGKey(42)
    R = jnp.asarray(
        jax.random.uniform(key, (n_atoms, 3), minval=2.0, maxval=cell_length - 2.0),
        dtype=jnp.float32,
    )
    Z = jnp.array([6] * n_atoms)

    calc, spherical_cutoff_calculator = factory(
        atomic_numbers=np.array(Z),
        atomic_positions=np.array(R),
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        do_pbc_map=True,
        pbc_map=pbc_map,
    )

    atoms = ase.Atoms(np.array(Z), np.array(R), cell=np.array(cell_matrix), pbc=True)
    atoms.calc = calc

    E_ase = atoms.get_potential_energy()
    F_ase = atoms.get_forces()

    @jit
    def jax_md_energy_fn(position, **kwargs):
        result = spherical_cutoff_calculator(
            atomic_numbers=Z,
            positions=jnp.array(position),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            debug=False,
        )
        return result.energy.reshape(-1)[0]

    @jax.custom_vjp
    def wrapped_energy_fn(position, **kwargs):
        pos = jnp.array(position)
        return jax_md_energy_fn(pbc_map(pos), **kwargs)

    def wrapped_energy_fn_fwd(position, **kwargs):
        pos = jnp.array(position)
        R_mapped = pbc_map(pos)
        E = jax_md_energy_fn(R_mapped, **kwargs)
        return E, (pos, R_mapped)

    def wrapped_energy_fn_bwd(res, g, **kwargs):
        pos, R_mapped = res
        result = spherical_cutoff_calculator(
            atomic_numbers=Z,
            positions=R_mapped,
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            debug=False,
        )
        F_mapped = result.forces
        F_orig = pbc_map.transform_forces(pos, F_mapped)
        return (F_orig,)

    wrapped_energy_fn.defvjp(wrapped_energy_fn_fwd, wrapped_energy_fn_bwd)
    wrapped_energy_fn = jit(wrapped_energy_fn)

    E_jax = float(wrapped_energy_fn(R))
    F_jax = np.array(-jax.grad(wrapped_energy_fn)(R))

    rtol, atol = 1e-4, 1e-3
    assert np.isclose(E_ase, E_jax, rtol=rtol, atol=atol), (
        f"ASE energy {E_ase:.6f} != JAX-MD energy {E_jax:.6f}"
    )
    assert np.allclose(F_ase, F_jax, rtol=rtol, atol=atol), (
        f"Forces differ: max |F_ase - F_jax| = {np.max(np.abs(F_ase - F_jax)):.6e}"
    )
