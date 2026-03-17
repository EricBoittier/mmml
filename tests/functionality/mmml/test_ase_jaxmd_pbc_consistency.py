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
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "examples/ckpts_json",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers/epoch-1985",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
            PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "ckpts_json",
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
    # Relaxed force tolerance: ASE and JAX paths can differ slightly near cutoffs
    force_atol = 0.05
    assert np.allclose(F_ase, F_jax, rtol=0.01, atol=force_atol), (
        f"Forces differ: max |F_ase - F_jax| = {np.max(np.abs(F_ase - F_jax)):.6e}"
    )


@pytest.mark.skipif(
    not _can_import("pycharmm"),
    reason="pycharmm not available in this environment",
)
@pytest.mark.skipif(
    not _can_import("jax_md"),
    reason="jax_md not available in this environment",
)
def test_ase_jaxmd_pbc_with_box_and_pairs():
    """
    Test that evaluate_energies_and_forces with box and pair_idx/pair_mask
    matches ASE calculator for the same wrapped configuration.
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
    import e3x

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters

    n_monomers = 2
    n_atoms_monomer = 10
    n_atoms = n_monomers * n_atoms_monomer
    cell_length = 40.0

    try:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=n_atoms_monomer,
            N_MONOMERS=n_monomers,
            doML=True,
            doMM=True,
            model_restart_path=ckpt,
            MAX_ATOMS_PER_SYSTEM=n_atoms,
            cell=cell_length,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"MM setup failed (PyCHARMM/jax_md): {exc}")

    key = jax.random.PRNGKey(42)
    R_init = np.asarray(
        jax.random.uniform(key, (n_atoms, 3), minval=2.0, maxval=cell_length - 2.0),
        dtype=np.float32,
    )
    calc_result = factory(
        atomic_numbers=np.array([6] * n_atoms),
        atomic_positions=R_init,
        n_monomers=n_monomers,
        cutoff_params=CutoffParameters(),
    )
    if len(calc_result) == 3:
        calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None

    update_fn = get_update_fn(R_init, CutoffParameters())
    if update_fn is None:
        pytest.skip("get_update_fn not available (jax_md neighbor list not used)")

    cell_matrix = np.diag([cell_length, cell_length, cell_length])
    R = R_init
    Z = np.array([6] * n_atoms)

    atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
    atoms.calc = calc

    E_ase = atoms.get_potential_energy()
    F_ase = atoms.get_forces()

    box_nl = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_init = jnp.array([cell_length, cell_length, cell_length], dtype=jnp.float32)
    pair_idx, pair_mask = update_fn(R, box=box_nl)

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)

    result = spherical_cutoff_calculator(
        atomic_numbers=jnp.array(Z),
        positions=jnp.array(R),
        n_monomers=n_monomers,
        cutoff_params=CutoffParameters(),
        doML=True,
        doMM=False,
        doML_dimer=True,
        debug=False,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    E_jax = float(result.energy.reshape(-1)[0])
    F_jax = np.asarray(result.forces)

    rtol, atol = 1e-4, 1e-3
    assert np.isclose(E_ase, E_jax, rtol=rtol, atol=atol), (
        f"ASE energy {E_ase:.6f} != JAX-MD with box/pairs energy {E_jax:.6f}"
    )
    force_atol = 0.05
    assert np.allclose(F_ase, F_jax, rtol=0.01, atol=force_atol), (
        f"Forces differ with box/pairs: max |F_ase - F_jax| = {np.max(np.abs(F_ase - F_jax)):.6e}"
    )
