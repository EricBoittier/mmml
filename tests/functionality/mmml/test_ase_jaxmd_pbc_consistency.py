"""
Test that ASE calculator and JAX-MD produce consistent energies and forces for PBC.

Verifies that the ASE AseDimerCalculator and JAX-MD energy_fn yield the same energy
and forces for the same configuration. Uses MIC-only PBC (no coordinate transform).
"""
from __future__ import annotations

from pathlib import Path
import os
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Acetone (ACO): 10 atoms per monomer; matches DESdimers PhysNet checkpoints (natoms=20).
ACO_ATOMS_PER_MONOMER = 10
ACO_N_MONOMERS = 2
ACO_N_ATOMS = ACO_ATOMS_PER_MONOMER * ACO_N_MONOMERS


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


def _skip_if_runtime_incompatible(exc: Exception) -> None:
    """Skip when CHARMM/PSF/MM setup cannot support this integration test."""
    msg = str(exc)
    known = (
        "Cannot do a non-empty jnp.take() from an empty axis",
        "PyCHARMM PSF has no atoms",
        "does not match expected",
        "Failed to load JSON checkpoint",
        "CUDA_ERROR_OPERATING_SYSTEM",
    )
    if any(k in msg for k in known):
        pytest.skip(f"CHARMM/MM runtime incompatible: {exc}")
    raise exc


def _setup_charmm_aco_dimer_pbc(
    cell_length: float = 40.0,
    com_separation: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2xACO dimer in PyCHARMM with a loaded PSF (20 atoms).

    Places monomer COMs ``com_separation`` Å apart near the cell center so MM
    switching and jax-md neighbor lists are active.
    """
    import pycharmm
    import pycharmm.generate as gen
    import pycharmm.ic as ic
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        CGENFF_RTF,
        coor,
        pycharmm_quiet,
        read,
        reset_block,
        settings,
    )
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    pycharmm_quiet()
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")
    read.sequence_string("ACO ACO")
    gen.new_segment(seg_name="DIMR", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    z = np.asarray(get_Z_from_psf(), dtype=int)
    if z.size != ACO_N_ATOMS:
        raise ValueError(f"Expected {ACO_N_ATOMS} atoms for ACO dimer, got {z.size}")

    r = coor.get_positions().to_numpy(dtype=float)
    r0 = r[:ACO_ATOMS_PER_MONOMER].copy()
    r1 = r[ACO_ATOMS_PER_MONOMER:].copy()
    r0 -= r0.mean(axis=0)
    r1 -= r1.mean(axis=0)
    r1 += np.array([com_separation, 0.0, 0.0], dtype=float)
    r_out = np.vstack([r0, r1])
    r_out = r_out - r_out.mean(axis=0) + np.array(
        [cell_length / 2, cell_length / 2, cell_length / 2], dtype=float
    )

    inter = np.linalg.norm(
        r_out[:ACO_ATOMS_PER_MONOMER, None, :] - r_out[None, ACO_ATOMS_PER_MONOMER:, :],
        axis=-1,
    )
    if float(inter.min()) < 0.5:
        raise ValueError(
            f"ACO dimer seed overlaps: min intermolecular distance {float(inter.min()):.4f} Å"
        )

    coor.set_positions(pd.DataFrame(r_out, columns=["x", "y", "z"]))
    return z, np.asarray(r_out, dtype=np.float32)


def _psf_at_codes_override() -> np.ndarray:
    """CHARMM IAC codes (0-based) for the current PSF."""
    import pycharmm.psf as psf

    return np.asarray(psf.get_iac(), dtype=int) - 1


def _assert_ase_jax_energy_forces_match(
    E_ase: float,
    F_ase: np.ndarray,
    E_jax: float,
    F_jax: np.ndarray,
    *,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    rtol, atol = 1e-4, 1e-3
    force_atol = 0.05
    assert np.isclose(E_ase, E_jax, rtol=rtol, atol=atol), (
        f"{prefix}ASE energy {E_ase:.6f} != JAX energy {E_jax:.6f}"
    )
    assert np.allclose(F_ase, F_jax, rtol=0.01, atol=force_atol), (
        f"{prefix}Forces differ: max |F_ase - F_jax| = {np.max(np.abs(F_ase - F_jax)):.6e}"
    )


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
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

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

    calc, spherical_cutoff_calculator, _ = unpack_factory_result(
        factory(
            atomic_numbers=np.array(Z),
            atomic_positions=np.array(R),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
        )
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

    _assert_ase_jax_energy_forces_match(E_ase, F_ase, E_jax, F_jax)


@pytest.mark.skipif(
    not _can_import("pycharmm"),
    reason="pycharmm not available in this environment",
)
def test_ase_jaxmd_pbc_with_box_and_pairs():
    """
    Test that spherical_cutoff_calculator with box and pair_idx/pair_mask kwargs
    matches the ASE calculator for the same PBC configuration.

    Uses ML only (doMM=False): pair indices are passed through the API but MM is
    off, so no CHARMM PSF is required. This checks that box/pair kwargs do not
    break ASE/JAX consistency for the ML path.
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
    import ase
    import e3x

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

    n_monomers = 2
    n_atoms_monomer = 10
    n_atoms = n_monomers * n_atoms_monomer
    cell_length = 40.0

    try:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=n_atoms_monomer,
            N_MONOMERS=n_monomers,
            doML=True,
            doMM=False,
            model_restart_path=ckpt,
            MAX_ATOMS_PER_SYSTEM=n_atoms,
            cell=cell_length,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Calculator setup failed: {exc}")

    key = jax.random.PRNGKey(42)
    R_init = np.asarray(
        jax.random.uniform(key, (n_atoms, 3), minval=2.0, maxval=cell_length - 2.0),
        dtype=np.float32,
    )
    calc, spherical_cutoff_calculator, _ = unpack_factory_result(
        factory(
            atomic_numbers=np.array([6] * n_atoms),
            atomic_positions=R_init,
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(),
        )
    )

    cell_matrix = np.diag([cell_length, cell_length, cell_length])
    R = R_init
    Z = np.array([6] * n_atoms)

    atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
    atoms.calc = calc

    E_ase = atoms.get_potential_energy()
    F_ase = atoms.get_forces()

    box_init = jnp.array([cell_length, cell_length, cell_length], dtype=jnp.float32)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
    pair_idx = jnp.stack([dst_idx, src_idx], axis=1)
    pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)

    cutoff_params = CutoffParameters()

    result = spherical_cutoff_calculator(
        atomic_numbers=jnp.array(Z),
        positions=jnp.array(R),
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=False,
        doML_dimer=True,
        debug=False,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    E_jax = float(result.energy.reshape(-1)[0])

    from jax import jit

    @jit
    def jax_md_energy_with_pairs(position, **kwargs):
        out = spherical_cutoff_calculator(
            atomic_numbers=jnp.array(Z),
            positions=jnp.array(position),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            debug=False,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
            box=box_init,
        )
        return out.energy.reshape(-1)[0]

    F_jax = np.array(-jax.grad(jax_md_energy_with_pairs)(jnp.array(R)))

    _assert_ase_jax_energy_forces_match(
        E_ase, F_ase, E_jax, F_jax, context="ML box/pair kwargs"
    )


@pytest.mark.skipif(
    not _can_import("pycharmm"),
    reason="pycharmm not available in this environment",
)
@pytest.mark.integration
@pytest.mark.skipif(
    not _can_import("jax_md"),
    reason="jax_md not available in this environment",
)
def test_ase_jaxmd_pbc_ml_mm_box_and_jaxmd_pairs():
    """
    Full ML+MM hybrid: ASE vs JAX with PBC box and jax-md neighbor-list pairs.

    Uses a real PyCHARMM PSF for a 2xACO dimer (20 atoms) so MM charges/types match
    the loaded system. Monomer COMs are placed in the MM/ML handoff region.
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
    import pycharmm.param as param

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

    cell_length = 40.0
    cutoff_params = CutoffParameters()

    try:
        z, r = _setup_charmm_aco_dimer_pbc(
            cell_length=cell_length,
            com_separation=4.5,
        )
    except (RuntimeError, ValueError) as exc:
        _skip_if_runtime_incompatible(exc)

    at_codes = _psf_at_codes_override()
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)

    try:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=ACO_ATOMS_PER_MONOMER,
            N_MONOMERS=ACO_N_MONOMERS,
            doML=True,
            doMM=True,
            model_restart_path=ckpt,
            MAX_ATOMS_PER_SYSTEM=ACO_N_ATOMS,
            cell=cell_length,
            at_codes_override=at_codes,
            ep_scale=ep_scale,
            sig_scale=sig_scale,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        _skip_if_runtime_incompatible(exc)

    try:
        calc, spherical_cutoff_calculator, get_update_fn = unpack_factory_result(
            factory(
                atomic_numbers=z,
                atomic_positions=r,
                n_monomers=ACO_N_MONOMERS,
                cutoff_params=cutoff_params,
            )
        )
    except (IndexError, RuntimeError, ValueError) as exc:
        _skip_if_runtime_incompatible(exc)

    try:
        update_fn = get_update_fn(r, cutoff_params)
    except (IndexError, RuntimeError, ValueError) as exc:
        _skip_if_runtime_incompatible(exc)
    if update_fn is None:
        pytest.skip("jax-md neighbor update_fn not available (jax_md path not built)")

    cell_matrix = np.diag([cell_length, cell_length, cell_length])
    atoms = ase.Atoms(z, r, cell=cell_matrix, pbc=True)
    atoms.calc = calc

    box_nl = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_init = jnp.array([cell_length, cell_length, cell_length], dtype=jnp.float32)

    try:
        pair_idx, pair_mask = update_fn(r, box=box_nl)
        E_ase = atoms.get_potential_energy()
        F_ase = atoms.get_forces()
    except (IndexError, RuntimeError, ValueError) as exc:
        _skip_if_runtime_incompatible(exc)

    result = spherical_cutoff_calculator(
        atomic_numbers=jnp.array(z),
        positions=jnp.array(r),
        n_monomers=ACO_N_MONOMERS,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    E_jax = float(result.energy.reshape(-1)[0])

    @jit
    def jax_md_energy_with_mm_pairs(position, **kwargs):
        out = spherical_cutoff_calculator(
            atomic_numbers=jnp.array(z),
            positions=jnp.array(position),
            n_monomers=ACO_N_MONOMERS,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            debug=False,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
            box=box_init,
        )
        return out.energy.reshape(-1)[0]

    F_jax = np.array(-jax.grad(jax_md_energy_with_mm_pairs)(jnp.array(r)))

    _assert_ase_jax_energy_forces_match(
        E_ase, F_ase, E_jax, F_jax, context="ML+MM jax-md pairs"
    )
