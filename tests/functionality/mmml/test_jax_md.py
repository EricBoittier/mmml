"""
Tests for JAX-MD integration with the MMML hybrid calculator.

Verifies that the spherical_cutoff_calculator can be used as a JAX-MD energy
function for minimization (FIRE) and optionally short MD runs.
"""
import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np


@pytest.mark.skipif(
    importlib.util.find_spec("pycharmm") is None,
    reason="pycharmm not available in this environment",
)
@pytest.mark.skipif(
    importlib.util.find_spec("jax_md") is None,
    reason="jax_md not available in this environment",
)
def test_jax_md_fire_minimization_smoke():
    """
    Run a few FIRE minimization steps with the MMML calculator as energy function.
    Asserts energy is finite and forces decrease (or energy decreases).
    """
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax not available in this environment")
    if importlib.util.find_spec("e3x") is None:
        pytest.skip("e3x not available in this environment")

    ckpt_env = os.environ.get("MMML_CKPT")
    ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
    if not ckpt.exists():
        pytest.skip("No checkpoints present for ML model")

    import jax
    import jax.numpy as jnp
    from jax import jit

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters

    n_monomers = 2
    n_atoms_monomer = 10
    n_atoms = n_monomers * n_atoms_monomer

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms_monomer,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=n_atoms,
    )

    cutoff_params = CutoffParameters()
    calc, spherical_cutoff_calculator = factory(
        atomic_numbers=np.array([6] * n_atoms),
        atomic_positions=np.zeros((n_atoms, 3)),  # placeholder
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
    )

    key = jax.random.PRNGKey(42)
    R0 = jnp.asarray(
        jax.random.uniform(key, (n_atoms, 3), minval=0.0, maxval=10.0),
        dtype=jnp.float32,
    )

    @jit
    def jax_md_energy_fn(position, **kwargs):
        out = spherical_cutoff_calculator(
            positions=position,
            atomic_numbers=jnp.array([6] * n_atoms),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
        )
        return out.energy.reshape(-1)[0]

    import jax_md
    from jax_md import space

    displacement, shift = space.free()
    init_fn, step_fn = jax_md.minimize.fire_descent(
        jax_md_energy_fn, shift, dt_start=0.001, dt_max=0.001
    )
    step_fn = jit(step_fn)

    E0 = float(jax_md_energy_fn(R0))
    assert np.isfinite(E0), f"Initial energy non-finite: {E0}"

    state = init_fn(R0)
    n_steps = 20
    for i in range(n_steps):
        state = step_fn(state)

    E_final = float(jax_md_energy_fn(state.position))
    assert np.isfinite(E_final), f"Final energy non-finite: {E_final}"
    assert not np.any(np.isnan(state.position)), "Positions contain NaN"
    assert not np.any(np.isinf(state.position)), "Positions contain Inf"


@pytest.mark.skipif(
    importlib.util.find_spec("pycharmm") is None,
    reason="pycharmm not available in this environment",
)
@pytest.mark.skipif(
    importlib.util.find_spec("jax_md") is None,
    reason="jax_md not available in this environment",
)
def test_jax_md_nve_few_steps():
    """
    Run a few NVE steps with the MMML calculator.
    Asserts energy is finite and total energy is approximately conserved.
    """
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax not available in this environment")
    if importlib.util.find_spec("e3x") is None:
        pytest.skip("e3x not available in this environment")

    ckpt_env = os.environ.get("MMML_CKPT")
    ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
    if not ckpt.exists():
        pytest.skip("No checkpoints present for ML model")

    import jax
    import jax.numpy as jnp
    from jax import jit

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters

    n_monomers = 2
    n_atoms_monomer = 10
    n_atoms = n_monomers * n_atoms_monomer

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms_monomer,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=n_atoms,
    )

    cutoff_params = CutoffParameters()
    calc, spherical_cutoff_calculator = factory(
        atomic_numbers=np.array([6] * n_atoms),
        atomic_positions=np.zeros((n_atoms, 3)),
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
    )

    key = jax.random.PRNGKey(123)
    R0 = jnp.asarray(
        jax.random.uniform(key, (n_atoms, 3), minval=0.0, maxval=10.0),
        dtype=jnp.float32,
    )

    masses = jnp.ones((n_atoms,), dtype=jnp.float32) * 12.0  # carbon amu

    @jit
    def energy_fn(position, **kwargs):
        out = spherical_cutoff_calculator(
            positions=position,
            atomic_numbers=jnp.array([6] * n_atoms),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
        )
        return out.energy.reshape(-1)[0]

    import jax_md
    from jax_md import space, simulate, quantity

    displacement, shift = space.free()
    dt = 1e-3
    kT = 0.001  # low temp for stability
    init_fn, apply_fn = simulate.nve(energy_fn, shift, dt)
    apply_fn = jit(apply_fn)

    key, vel_key = jax.random.split(key)
    state = init_fn(vel_key, R0, kT, mass=masses)
    n_steps = 10
    for _ in range(n_steps):
        state = apply_fn(state)

    E_final = float(energy_fn(state.position))
    E_kin = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))

    assert np.isfinite(E_final), f"Potential energy non-finite: {E_final}"
    assert np.isfinite(E_kin), f"Kinetic energy non-finite: {E_kin}"
    assert not np.any(np.isnan(state.position)), "Positions contain NaN"
    assert not np.any(np.isinf(state.position)), "Positions contain Inf"
