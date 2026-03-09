"""
Test that ASE calculator and JAX-MD produce consistent energies and forces for PBC.

Verifies that the ASE AseDimerCalculator and JAX-MD energy_fn yield the same energy
and forces for the same configuration. Uses MIC-only PBC (no coordinate transform).
"""
import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _can_import(name: str) -> bool:
    """Return True only if *name* can be fully imported (not just found)."""
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _can_import_e3x_nn() -> bool:
    try:
        __import__("e3x.nn.modules", fromlist=["initializers"])
        return True
    except Exception:
        return False


def _get_ckpt():
    ckpt_env = os.environ.get("MMML_CKPT")
    candidates = []
    if ckpt_env:
        candidates.append(Path(ckpt_env))
    candidates.extend(
        [
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers/epoch-1985",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
            PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "ckpts_json",
            PROJECT_ROOT / "mmml/physnetjax/ckpts",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    return None


@pytest.mark.skipif(
    not _can_import("pycharmm"),
    reason="pycharmm not available in this environment",
)
@pytest.mark.skipif(
    not _can_import("jax_md"),
    reason="jax_md not available in this environment",
)
def test_ase_jaxmd_pbc_energy_forces_consistency():
    """
    Compare ASE calculator energy/forces with JAX-MD energy_fn for the same PBC config.

    Uses MIC-only PBC (no coordinate transform). Both paths use spherical_cutoff_calculator
    with positions directly; MIC is applied internally. Asserts energies and forces match.
    """
    if not _can_import("jax"):
        pytest.skip("jax not available in this environment")
    if not _can_import_e3x_nn():
        pytest.skip("e3x.nn not available in this environment")
    if not _can_import("ase"):
        pytest.skip("ase not available in this environment")

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoints present for ML model")

    import jax
    import jax.numpy as jnp
    from jax import jit
    import ase

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters

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

    E_jax = float(jax_md_energy_fn(R))
    F_jax = np.array(-jax.grad(jax_md_energy_fn)(R))

    rtol, atol = 1e-4, 1e-3
    assert np.isclose(E_ase, E_jax, rtol=rtol, atol=atol), (
        f"ASE energy {E_ase:.6f} != JAX-MD energy {E_jax:.6f}"
    )
    assert np.allclose(F_ase, F_jax, rtol=rtol, atol=atol), (
        f"Forces differ: max |F_ase - F_jax| = {np.max(np.abs(F_ase - F_jax)):.6e}"
    )
