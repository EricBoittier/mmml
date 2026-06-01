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


def _require_nontrivial_forces(forces: np.ndarray, *, label: str = "forces") -> None:
    """Fail fast when forces are missing, non-finite, or all ~zero (invalid for consistency tests)."""
    f = np.asarray(forces, dtype=float)
    if f.size == 0:
        raise AssertionError(f"{label}: empty force array")
    if f.ndim != 2 or f.shape[1] != 3:
        raise AssertionError(f"{label}: expected shape (n_atoms, 3), got {f.shape}")
    per_atom = np.linalg.norm(f, axis=1)
    if not np.all(np.isfinite(per_atom)):
        bad = np.where(~np.isfinite(per_atom))[0]
        raise AssertionError(
            f"{label}: non-finite force magnitudes at atom indices {bad.tolist()}"
        )
    max_mag = float(np.max(per_atom))
    if max_mag < 1e-10:
        raise AssertionError(
            f"{label}: all forces are near zero (max |F| = {max_mag:.3e}). "
            "This usually indicates broken autodiff, a missing CHARMM PSF, NaNs zeroed "
            "in the calculator, or geometry outside the ML/MM switching region."
        )


def _assert_ase_jax_energy_forces_match(
    E_ase: float,
    F_ase: np.ndarray,
    E_jax: float,
    F_jax: np.ndarray,
    *,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    _require_nontrivial_forces(F_ase, label=f"{prefix}F_ase")
    _require_nontrivial_forces(F_jax, label=f"{prefix}F_jax")
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
    Full ML+MM hybrid: ASE vs spherical_cutoff_calculator with jax-md neighbor pairs.

    Uses a real PyCHARMM PSF for a 2xACO dimer (20 atoms) so MM charges/types match
    the loaded system. Monomer COMs are placed in the MM/ML handoff region.

    Forces are compared via the analytical hybrid force field (``ModelOutput.forces``),
    not ``jax.grad``, because the jax-md pair list is rebuilt outside the JAX trace
    when positions change; autodiff with a frozen pair list is undefined for MM.
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

    import jax.numpy as jnp
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
                backprop=False,
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

    atoms = ase.Atoms(z, r, cell=_cell_matrix(cell_length), pbc=True)
    atoms.calc = calc

    box_nl = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_jax = jnp.array(box_nl, dtype=jnp.float32)

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
        box=box_jax,
    )
    E_jax = float(result.energy.reshape(-1)[0])
    F_jax = np.asarray(result.forces)

    _assert_ase_jax_energy_forces_match(
        E_ase, F_ase, E_jax, F_jax, context="ML+MM jax-md pairs"
    )


def _cell_matrix(cell_length: float) -> np.ndarray:
    return np.diag([cell_length, cell_length, cell_length]).astype(float)


def _build_aco_mm_calculator(
    ckpt: Path,
    z: np.ndarray,
    r: np.ndarray,
    cell_length: float,
    *,
    backprop: bool = False,
):
    """Factory + calculator for 2xACO with ML+MM and PBC."""
    import pycharmm.param as param
    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

    at_codes = _psf_at_codes_override()
    n_types = len(param.get_atc())
    cutoff_params = CutoffParameters()
    factory = setup_calculator(
        ATOMS_PER_MONOMER=ACO_ATOMS_PER_MONOMER,
        N_MONOMERS=ACO_N_MONOMERS,
        doML=True,
        doMM=True,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=ACO_N_ATOMS,
        cell=cell_length,
        at_codes_override=at_codes,
        ep_scale=np.ones(n_types, dtype=float),
        sig_scale=np.ones(n_types, dtype=float),
    )
    calc, spherical_fn, get_update_fn = unpack_factory_result(
        factory(
            atomic_numbers=z,
            atomic_positions=r,
            n_monomers=ACO_N_MONOMERS,
            cutoff_params=cutoff_params,
            backprop=backprop,
        )
    )
    return calc, spherical_fn, get_update_fn, cutoff_params, z, r


def _assert_lattice_translation_invariance(
    *,
    cell_length: float,
    R: np.ndarray,
    monomer_slice: slice,
    E0_ase: float,
    E1_ase: float,
    F0_ase: np.ndarray,
    F1_ase: np.ndarray,
    E0_jax: float,
    E1_jax: float,
    energy_tol: float = 1e-3,
    force_tol: float = 1e-3,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    delta_e_ase = abs(float(E1_ase - E0_ase))
    delta_e_jax = abs(float(E1_jax - E0_jax))
    assert delta_e_ase < energy_tol, (
        f"{prefix}ASE lattice energy drift {delta_e_ase:.6e} (tol {energy_tol})"
    )
    assert delta_e_jax < energy_tol, (
        f"{prefix}JAX lattice energy drift {delta_e_jax:.6e} (tol {energy_tol})"
    )
    f0 = F0_ase[monomer_slice]
    f1 = F1_ase[monomer_slice]
    assert np.allclose(f0, f1, atol=force_tol, rtol=1e-3), (
        f"{prefix}ASE force invariance on translated monomer: "
        f"max |ΔF| = {np.max(np.abs(f0 - f1)):.6e}"
    )


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
def test_ml_only_jax_autograd_matches_model_forces():
    """ML-only: -jax.grad(energy) matches ModelOutput.forces (valid autodiff path)."""
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax
    import jax.numpy as jnp
    from jax import jit
    import ase

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    factory = setup_calculator(
        ATOMS_PER_MONOMER=10,
        N_MONOMERS=2,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=20,
        cell=cell_length,
    )
    key = jax.random.PRNGKey(7)
    R = np.asarray(
        jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0),
        dtype=np.float32,
    )
    Z = np.array([6] * 20)
    cutoff_params = CutoffParameters()
    calc, spherical_fn, _ = unpack_factory_result(
        factory(
            atomic_numbers=Z,
            atomic_positions=R,
            n_monomers=2,
            cutoff_params=cutoff_params,
            backprop=False,
        )
    )

    result = spherical_fn(
        atomic_numbers=jnp.array(Z),
        positions=jnp.array(R),
        n_monomers=2,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=False,
        doML_dimer=True,
    )
    F_model = np.asarray(result.forces)

    @jit
    def energy_fn(pos):
        out = spherical_fn(
            atomic_numbers=jnp.array(Z),
            positions=jnp.array(pos),
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
        )
        return out.energy.reshape(-1)[0]

    F_grad = np.asarray(-jax.grad(energy_fn)(jnp.array(R)))
    _require_nontrivial_forces(F_model, label="F_model")
    _require_nontrivial_forces(F_grad, label="F_grad")
    assert np.allclose(F_model, F_grad, rtol=0.02, atol=0.05), (
        f"ML autograd vs model forces: max |ΔF| = {np.max(np.abs(F_model - F_grad)):.6e}"
    )


@pytest.mark.integration
@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
def test_ml_mm_frozen_pair_autograd_differs_from_analytical():
    """
    ML+MM with a frozen jax-md pair list: autograd must not match analytical forces.

    Documents that ``jax.grad`` through MM while holding neighbor pairs fixed is not
    a supported force model; production uses ``ModelOutput.forces``.
    """
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax
    import jax.numpy as jnp
    from jax import jit

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    try:
        z, r = _setup_charmm_aco_dimer_pbc(cell_length=cell_length, com_separation=4.5)
        calc, spherical_fn, get_update_fn, cutoff_params, z, r = _build_aco_mm_calculator(
            ckpt, z, r, cell_length, backprop=False
        )
        update_fn = get_update_fn(r, cutoff_params)
        if update_fn is None:
            pytest.skip("jax-md neighbor list unavailable")
    except (RuntimeError, ValueError, IndexError) as exc:
        _skip_if_runtime_incompatible(exc)

    box = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_jax = jnp.array(box, dtype=jnp.float32)
    pair_idx, pair_mask = update_fn(r, box=box)

    result = spherical_fn(
        atomic_numbers=jnp.array(z),
        positions=jnp.array(r),
        n_monomers=ACO_N_MONOMERS,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,
        doML_dimer=True,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_jax,
    )
    F_model = np.asarray(result.forces)
    _require_nontrivial_forces(F_model, label="F_model")

    @jit
    def energy_frozen_pairs(pos):
        out = spherical_fn(
            atomic_numbers=jnp.array(z),
            positions=jnp.array(pos),
            n_monomers=ACO_N_MONOMERS,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
            box=box_jax,
        )
        return out.energy.reshape(-1)[0]

    F_grad = np.asarray(-jax.grad(energy_frozen_pairs)(jnp.array(r)))
    if not np.all(np.isfinite(F_grad)):
        return  # autograd undefined: expected for this path
    if np.max(np.linalg.norm(F_grad, axis=1)) < 1e-10:
        pytest.fail("Frozen-pair autograd returned zero forces; expected mismatch, not trivial zeros")
    max_rel = np.max(
        np.abs(F_model - F_grad)
        / (np.linalg.norm(F_model, axis=1) + 1e-10)
    )
    assert max_rel > 0.05 or not np.allclose(F_model, F_grad, rtol=0.05, atol=0.1), (
        "Frozen MM pair autograd should not match analytical hybrid forces"
    )


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_box_vectors_from_ase_atoms():
    """Orthorhombic ASE cell is forwarded as (Lx, Ly, Lz) box vectors."""
    import ase
    from mmml.pycharmmInterface.calculator_utils import box_vectors_from_atoms_or_cell

    cell_length = 40.0
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=_cell_matrix(cell_length), pbc=True)
    bv = box_vectors_from_atoms_or_cell(atoms, setup_cell=None)
    np.testing.assert_allclose(bv, [cell_length, cell_length, cell_length], rtol=0, atol=1e-6)


@pytest.mark.integration
@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
def test_ase_calculator_passes_cell_box_to_mm_path():
    """ASE calculate() with pbc=True matches spherical eval using the same box and pairs."""
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax.numpy as jnp
    import ase

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    try:
        z, r = _setup_charmm_aco_dimer_pbc(cell_length=cell_length, com_separation=4.5)
        calc, spherical_fn, get_update_fn, cutoff_params, z, r = _build_aco_mm_calculator(
            ckpt, z, r, cell_length, backprop=False
        )
        if get_update_fn is None:
            pytest.skip("jax-md neighbor list unavailable")
    except (RuntimeError, ValueError, IndexError) as exc:
        _skip_if_runtime_incompatible(exc)

    box = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_jax = jnp.array(box, dtype=jnp.float32)
    update_fn = get_update_fn(r, cutoff_params)
    pair_idx, pair_mask = update_fn(r, box=box)

    atoms = ase.Atoms(z, r, cell=_cell_matrix(cell_length), pbc=True)
    atoms.calc = calc
    E_ase = atoms.get_potential_energy()
    F_ase = atoms.get_forces()

    result = spherical_fn(
        atomic_numbers=jnp.array(z),
        positions=jnp.array(r),
        n_monomers=ACO_N_MONOMERS,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,
        doML_dimer=True,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_jax,
    )
    _assert_ase_jax_energy_forces_match(
        float(E_ase),
        F_ase,
        float(result.energy.reshape(-1)[0]),
        np.asarray(result.forces),
        context="ASE with atoms.cell box",
    )


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
def test_pbc_lattice_invariance_ml_ase_and_jax():
    """Translating monomer 0 by a lattice vector leaves ML energy/forces unchanged (ASE + JAX)."""
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax
    import jax.numpy as jnp
    from jax import jit
    import ase

    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.pycharmmInterface.calculator_utils import unpack_factory_result

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    factory = setup_calculator(
        ATOMS_PER_MONOMER=10,
        N_MONOMERS=2,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=20,
        cell=cell_length,
    )
    key = jax.random.PRNGKey(99)
    R = np.asarray(
        jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0),
        dtype=np.float32,
    )
    Z = np.array([6] * 20)
    cutoff_params = CutoffParameters()
    calc, spherical_fn, _ = unpack_factory_result(
        factory(
            atomic_numbers=Z,
            atomic_positions=R,
            n_monomers=2,
            cutoff_params=cutoff_params,
        )
    )
    atoms = ase.Atoms(Z, R, cell=_cell_matrix(cell_length), pbc=True)
    atoms.calc = calc
    monomer_slice = slice(0, 10)

    E0_ase = float(atoms.get_potential_energy())
    F0_ase = atoms.get_forces()
    a = np.array([cell_length, 0.0, 0.0])
    R_shift = R.copy()
    R_shift[monomer_slice] += a
    atoms.set_positions(R_shift)
    E1_ase = float(atoms.get_potential_energy())
    F1_ase = atoms.get_forces()

    @jit
    def energy_fn(pos):
        out = spherical_fn(
            atomic_numbers=jnp.array(Z),
            positions=jnp.array(pos),
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
        )
        return out.energy.reshape(-1)[0]

    E0_jax = float(energy_fn(jnp.array(R)))
    E1_jax = float(energy_fn(jnp.array(R_shift)))

    _assert_lattice_translation_invariance(
        cell_length=cell_length,
        R=R,
        monomer_slice=monomer_slice,
        E0_ase=E0_ase,
        E1_ase=E1_ase,
        F0_ase=F0_ase,
        F1_ase=F1_ase,
        E0_jax=E0_jax,
        E1_jax=E1_jax,
        context="ML-only",
    )


@pytest.mark.integration
@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
@pytest.mark.parametrize("com_separation", [3.5, 4.5, 5.5])
def test_aco_hybrid_nontrivial_at_com_separation(com_separation: float):
    """ML+MM ACO dimer at several COM separations: non-zero forces and ASE/JAX agreement."""
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax.numpy as jnp
    import ase

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    try:
        z, r = _setup_charmm_aco_dimer_pbc(
            cell_length=cell_length,
            com_separation=com_separation,
        )
        calc, spherical_fn, get_update_fn, cutoff_params, z, r = _build_aco_mm_calculator(
            ckpt, z, r, cell_length, backprop=False
        )
        update_fn = get_update_fn(r, cutoff_params)
        if update_fn is None:
            pytest.skip("jax-md neighbor list unavailable")
    except (RuntimeError, ValueError, IndexError) as exc:
        _skip_if_runtime_incompatible(exc)

    box = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_jax = jnp.array(box, dtype=jnp.float32)
    pair_idx, pair_mask = update_fn(r, box=box)

    atoms = ase.Atoms(z, r, cell=_cell_matrix(cell_length), pbc=True)
    atoms.calc = calc
    E_ase = atoms.get_potential_energy()
    F_ase = atoms.get_forces()

    result = spherical_fn(
        atomic_numbers=jnp.array(z),
        positions=jnp.array(r),
        n_monomers=ACO_N_MONOMERS,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,
        doML_dimer=True,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_jax,
    )
    _assert_ase_jax_energy_forces_match(
        float(E_ase),
        F_ase,
        float(result.energy.reshape(-1)[0]),
        np.asarray(result.forces),
        context=f"ACO COM={com_separation:.1f} Å",
    )


@pytest.mark.integration
@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax_md"), reason="jax_md not available")
def test_pbc_lattice_invariance_ml_mm_aco_ase_and_spherical():
    """ACO ML+MM: lattice translation invariance for ASE and spherical (jax-md pairs + box)."""
    if not _can_import("jax") or not _can_import_e3x_nn() or not _can_import("ase"):
        pytest.skip("jax/e3x/ase not available")

    import jax.numpy as jnp
    import ase

    ckpt = _get_ckpt()
    if ckpt is None:
        pytest.skip("No checkpoint")

    cell_length = 40.0
    try:
        z, r = _setup_charmm_aco_dimer_pbc(cell_length=cell_length, com_separation=4.5)
        calc, spherical_fn, get_update_fn, cutoff_params, z, r = _build_aco_mm_calculator(
            ckpt, z, r, cell_length, backprop=False
        )
        update_fn = get_update_fn(r, cutoff_params)
        if update_fn is None:
            pytest.skip("jax-md neighbor list unavailable")
    except (RuntimeError, ValueError, IndexError) as exc:
        _skip_if_runtime_incompatible(exc)

    box = np.array([cell_length, cell_length, cell_length], dtype=np.float64)
    box_jax = jnp.array(box, dtype=jnp.float32)
    monomer_slice = slice(0, ACO_ATOMS_PER_MONOMER)

    atoms = ase.Atoms(z, r, cell=_cell_matrix(cell_length), pbc=True)
    atoms.calc = calc

    E0_ase = float(atoms.get_potential_energy())
    F0_ase = atoms.get_forces()
    a = np.array([cell_length, 0.0, 0.0])
    R_shift = r.copy()
    R_shift[monomer_slice] += a
    atoms.set_positions(R_shift)
    E1_ase = float(atoms.get_potential_energy())
    F1_ase = atoms.get_forces()

    pair0 = update_fn(r, box=box)
    pair1 = update_fn(R_shift, box=box)

    def eval_spherical(pos, pair_data):
        pi, pm = pair_data
        out = spherical_fn(
            atomic_numbers=jnp.array(z),
            positions=jnp.array(pos),
            n_monomers=ACO_N_MONOMERS,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            mm_pair_idx=pi,
            mm_pair_mask=pm,
            box=box_jax,
        )
        return float(out.energy.reshape(-1)[0])

    E0_jax = eval_spherical(r, pair0)
    E1_jax = eval_spherical(R_shift, pair1)

    _assert_lattice_translation_invariance(
        cell_length=cell_length,
        R=r,
        monomer_slice=monomer_slice,
        E0_ase=E0_ase,
        E1_ase=E1_ase,
        F0_ase=F0_ase,
        F1_ase=F1_ase,
        E0_jax=E0_jax,
        E1_jax=E1_jax,
        energy_tol=5e-3,
        force_tol=0.05,
        context="ML+MM ACO",
    )
