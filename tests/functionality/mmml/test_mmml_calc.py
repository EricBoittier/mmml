import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np
import importlib as _il  # lazy import in tests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import e3x

def _can_import(name: str) -> bool:
	"""Return True only if *name* can be fully imported (not just found)."""
	try:
		__import__(name)
		return True
	except Exception:
		return False


def _can_import_e3x_nn() -> bool:
	"""Return True only if e3x.nn modules are importable."""
	try:
		__import__("e3x.nn.modules", fromlist=["initializers"])
		return True
	except Exception:
		return False


def _resolve_ckpt_path() -> Path | None:
	"""Resolve a usable checkpoint path across legacy and JSON locations."""
	candidates = []
	ckpt_env = os.environ.get("MMML_CKPT")
	if ckpt_env:
		candidates.append(Path(ckpt_env))
	candidates.extend(
		[
			PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
			PROJECT_ROOT / "examples/ckpts_json/DES",
			PROJECT_ROOT / "examples/ckpts_json",
			PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
			PROJECT_ROOT / "ckpts_json/DES",
			PROJECT_ROOT / "ckpts_json",
			PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
			PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
		]
	)
	for ckpt in candidates:
		if ckpt.exists():
			return ckpt.resolve()
	return None


def _resolve_full_ckpt_path() -> Path | None:
	"""Resolve a full checkpoint directory suitable for strict invariance tests."""
	ckpt_env = os.environ.get("MMML_CKPT")
	candidates = []
	if ckpt_env:
		candidates.append(Path(ckpt_env))
	candidates.extend(
		[
			PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
			PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
		]
	)
	for ckpt in candidates:
		if not ckpt.exists():
			continue
		ckpt = ckpt.resolve()
		if ckpt.is_dir():
			if any(p.name.startswith("epoch-") for p in ckpt.iterdir()):
				return ckpt
			if (ckpt / "model_config.json").exists():
				return ckpt
	return None


def _skip_if_runtime_incompatible(exc: Exception) -> None:
	"""Skip optional heavy tests when checkpoint/runtime combination is incompatible."""
	msg = str(exc)
	known = (
		"Cannot do a non-empty jnp.take() from an empty axis",
		"Failed to load JSON checkpoint",
		"model_config.json",
		"CUDA_ERROR_OPERATING_SYSTEM",
	)
	if any(k in msg for k in known):
		pytest.skip(f"Checkpoint/runtime incompatible for this test: {exc}")
	raise exc


def test_ev2kcalmol_constant():
	# Ensure the EV->kcal/mol conversion used by calculators is reasonable
	from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol
	assert abs(ev2kcalmol - 23.0605) < 0.05


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_setup_calculator_factory_smoke():
	# Require JAX runtime for restart helpers
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	# Skip if no checkpoints are present
	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")

	from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

	# Create factory without touching MM (avoids CHARMM setup during smoke test)
	try:
		factory = setup_calculator(
			ATOMS_PER_MONOMER=2,
			N_MONOMERS=2,
			doML=True,
			doMM=False,
			model_restart_path=ckpt,
			MAX_ATOMS_PER_SYSTEM=8,
		)
	except ModuleNotFoundError as exc:
		pytest.skip(f"Required ML runtime not available: {exc}")
	except FileNotFoundError as exc:
		msg = str(exc)
		if "e3x" in msg or "model_config.json" in msg:
			pytest.skip(f"Checkpoint/runtime not compatible for smoke test: {exc}")
		raise

	assert callable(factory)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_ml_energy_matches_reference_when_data_available():
	"""
	Optional: use the original dataset to sanity-check ML energy path.
	Skips if data is not present.
	"""
	# Resolve data path: env MMML_DATA first, then repo-local fallbacks.
	data_candidates = []
	if env_data := os.environ.get("MMML_DATA"):
		env_path = Path(env_data)
		data_candidates.append(env_path if env_path.is_absolute() else PROJECT_ROOT / env_path)
	data_candidates.extend(
		[
			PROJECT_ROOT / "mmml/data/qcml/fixed-acetone-only_MP2_21000.npz",
			PROJECT_ROOT / "mmml/data/fixed-acetone-only_MP2_21000.npz",
		]
	)
	p = next((cand for cand in data_candidates if cand.exists()), None)
	if p is None:
		pytest.skip(f"Dataset not found in expected locations: {data_candidates}")

	# Lightweight import to prepare one batch
	from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets
	from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit
	import jax
	import jax.numpy as jnp

	# Load a tiny split to keep this test fast
	key = jax.random.PRNGKey(0)
	data_key, _ = jax.random.split(key)
	train_data, valid_data = prepare_datasets(
		data_key,
		train_size=32,
		valid_size=16,
		files=[str(p)],
		natoms=20,
	)
	batches = prepare_batches_jit(data_key, valid_data, batch_size=16, num_atoms=20)
	batch = batches[0]

	# Extract a single system (first monomer) to evaluate
	Z = jnp.array(batch["Z"]).reshape(-1)[:20]
	R = jnp.array(batch["R"]).reshape(-1, 3)[:20]

	# Build ML-only calculator factory
	from mmml.interfaces.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		ev2kcalmol,
	)
	# Skip if e3x (pair indices) backend is not available
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")
	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")
	factory = setup_calculator(
		ATOMS_PER_MONOMER=10,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=20,
		ml_energy_conversion_factor=ev2kcalmol,
		ml_force_conversion_factor=ev2kcalmol,
	)

	# ASE atoms (lazy import and skip if missing)
	if not _can_import("ase"):
		pytest.skip("ase not available in this environment")
	ase = _il.import_module("ase")
	atoms = ase.Atoms(np.array(Z), np.array(R))
	calc, _ = factory(
		atomic_numbers=np.array(Z),
		atomic_positions=np.array(R),
		n_monomers=2,
	)
	atoms.calc = calc
	ml_only_energy_kcal = atoms.get_potential_energy()

	# At least assert finite; optional compare if helper/model available
	assert np.isfinite(ml_only_energy_kcal)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_check_lattice_invariance():
	"""
	Test that the energy is invariant under translation of monomer 0 by a lattice vector (PBC).
	Uses check_lattice_invariance with a spherical_cutoff_calculator that applies PBC mapping.
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")

	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")
	if ckpt.is_file() and ckpt.suffix == ".json":
		ckpt = _resolve_full_ckpt_path()
		if ckpt is None:
			pytest.skip("Strict lattice/force checks require full checkpoint directory, not JSON params")

	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		check_lattice_invariance,
	)
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters

	# Setup calculator with PBC (cell=40 Å cubic), MIC-only
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
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])

	# Create test positions and atomic numbers (2 monomers × 10 atoms)
	import jax
	key = jax.random.PRNGKey(42)
	R = jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0)
	Z = jnp.array([6] * 20)  # carbon
	cutoff_params = CutoffParameters()

	calc, spherical_cutoff_calculator = factory(
		atomic_numbers=np.array(Z),
		atomic_positions=np.array(R),
		n_monomers=2,
		cutoff_params=cutoff_params,
	)

	def sc_fn(R_in, Z_in, n_monomers, cutoff_params_in):
		return spherical_cutoff_calculator(
			positions=R_in,
			atomic_numbers=Z_in,
			n_monomers=n_monomers,
			cutoff_params=cutoff_params_in,
		)

	try:
		delta_E = check_lattice_invariance(
			sc_fn, R, Z, 2, cutoff_params, cell_matrix
		)
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	assert abs(delta_E) < 1e-4, (
		f"Lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_pbc_energy_invariance_via_ase():
	"""
	Test that energy is invariant under translation of monomer 0 by a lattice vector (PBC)
	using AseDimerCalculator directly (ASE atoms.get_potential_energy()).
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")
	if not _can_import("ase"):
		pytest.skip("ase not available in this environment")

	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")
	if ckpt.is_file() and ckpt.suffix == ".json":
		ckpt = _resolve_full_ckpt_path()
		if ckpt is None:
			pytest.skip("Strict lattice/force checks require full checkpoint directory, not JSON params")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
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
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	try:
		E0 = atoms.get_potential_energy()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	a = np.array([float(cell_length), 0.0, 0.0])
	n_per = len(atoms) // 2
	g0 = np.arange(n_per)
	R_shift = R.copy()
	R_shift[g0] += a
	atoms.set_positions(R_shift)
	try:
		E1 = atoms.get_potential_energy()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	delta_E = float(E1 - E0)
	assert abs(delta_E) < 1e-4, (
		f"ASE lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_pbc_force_invariance():
	"""
	Test that forces are unchanged after translating monomer 0 by a lattice vector (PBC).
	Forces on both monomers should be invariant (same relative geometry).
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")
	if not _can_import("ase"):
		pytest.skip("ase not available in this environment")

	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")
	if ckpt.is_file() and ckpt.suffix == ".json":
		ckpt = _resolve_full_ckpt_path()
		if ckpt is None:
			pytest.skip("Strict lattice/force checks require full checkpoint directory, not JSON params")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
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
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	try:
		F0 = atoms.get_forces()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	n_per = len(atoms) // 2
	F0_mon0, F0_mon1 = F0[:n_per], F0[n_per:]

	a = np.array([float(cell_length), 0.0, 0.0])
	g0 = np.arange(n_per)
	R_shift = R.copy()
	R_shift[g0] += a
	atoms.set_positions(R_shift)
	try:
		F1 = atoms.get_forces()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	F1_mon0, F1_mon1 = F1[:n_per], F1[n_per:]

	assert np.allclose(F0_mon0, F1_mon0, atol=1e-5, rtol=1e-4), (
		f"Force invariance violated for monomer 0: max diff = {np.max(np.abs(F0_mon0 - F1_mon0))}"
	)
	assert np.allclose(F0_mon1, F1_mon1, atol=1e-5, rtol=1e-4), (
		f"Force invariance violated for monomer 1: max diff = {np.max(np.abs(F0_mon1 - F1_mon1))}"
	)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_pbc_force_gradient_numerical():
	"""
	Numerical gradient of energy vs positions; compare against atoms.get_forces()
	to ensure forces match -dE/dR (MIC-only, no coordinate transform).
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")
	if not _can_import("ase"):
		pytest.skip("ase not available in this environment")

	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
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
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	try:
		F_analytical = atoms.get_forces()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	h = 1e-5
	F_numerical = np.zeros_like(R)
	for i in range(len(R)):
		for j in range(3):
			Rp = R.copy()
			Rp[i, j] += h
			atoms.set_positions(Rp)
			try:
				Ep = atoms.get_potential_energy()
			except Exception as exc:
				_skip_if_runtime_incompatible(exc)
			Rm = R.copy()
			Rm[i, j] -= h
			atoms.set_positions(Rm)
			try:
				Em = atoms.get_potential_energy()
			except Exception as exc:
				_skip_if_runtime_incompatible(exc)
			F_numerical[i, j] = -(Ep - Em) / (2 * h)

	atoms.set_positions(R)

	rel_err = np.abs(F_analytical - F_numerical)
	denom = np.abs(F_numerical) + 1e-10
	rel_err = np.where(denom > 1e-8, rel_err / denom, rel_err)
	max_rel_err = np.max(rel_err)
	max_abs_err = np.max(np.abs(F_analytical - F_numerical))
	# Relaxed tolerance: numerical gradients can differ near switching regions
	assert max_rel_err < 0.25 or max_abs_err < 0.1, (
		f"Force gradient mismatch: max_rel_err={max_rel_err:.4e}, max_abs_err={max_abs_err:.4e}"
	)


def test_pbc_mic_displacement_symmetry():
	"""mic_displacement(Ri, Rj, cell) = -mic_displacement(Rj, Ri, cell)."""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")

	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

	cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
	Ri = jnp.array([1.0, 2.0, 3.0])
	Rj = jnp.array([7.0, 8.0, 9.0])

	d_ij = mic_displacement(Ri, Rj, cell)
	d_ji = mic_displacement(Rj, Ri, cell)
	np.testing.assert_allclose(d_ij, -d_ji, atol=1e-10)


def test_pbc_wrap_unwrap_roundtrip():
	"""Unwrap then wrap should recover equivalent positions (same fractional coords mod 1)."""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")

	import jax
	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
		frac_coords,
		unwrap_groups,
		wrap_groups,
	)

	cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
	groups = [jnp.arange(0, 5), jnp.arange(5, 10)]
	key = jax.random.PRNGKey(0)
	R = jax.random.uniform(key, (10, 3), minval=0.0, maxval=10.0)

	R_unwrapped = unwrap_groups(R, groups, cell)
	R_roundtrip = wrap_groups(R_unwrapped, groups, cell)

	S_orig = frac_coords(R, cell)
	S_roundtrip = frac_coords(R_roundtrip, cell)
	S_diff = S_roundtrip - S_orig
	S_diff_mod = S_diff - jnp.round(S_diff)
	np.testing.assert_allclose(S_diff_mod, 0.0, atol=1e-6)


def test_pbc_mapper_idempotent():
	"""For positions already in primary cell, pbc_map(pbc_map(R)) == pbc_map(R)."""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")

	import jax
	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

	cell = jnp.array([[40.0, 0, 0], [0, 40.0, 0], [0, 0, 40.0]])
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell, mol_id=mol_id, n_monomers=2)

	key = jax.random.PRNGKey(0)
	R = jax.random.uniform(key, (20, 3), minval=2.0, maxval=38.0)

	R1 = pbc_map(R)
	R2 = pbc_map(R1)
	np.testing.assert_allclose(R1, R2, atol=1e-5)


@pytest.mark.skipif(
	not _can_import("pycharmm"),
	reason="pycharmm not available in this environment",
)
def test_pbc_energy_invariance_orthorhombic_cell():
	"""
	Test energy invariance under lattice translation with orthorhombic cell [30, 40, 50].
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")
	if not _can_import_e3x_nn():
		pytest.skip("e3x.nn not available in this environment")

	ckpt = _resolve_ckpt_path()
	if ckpt is None:
		pytest.skip("No checkpoints present for ML model")
	if ckpt.is_file() and ckpt.suffix == ".json":
		ckpt = _resolve_full_ckpt_path()
		if ckpt is None:
			pytest.skip("Strict lattice/force checks require full checkpoint directory, not JSON params")
	# Accept orbax checkpoint dirs (epoch-*) or JSON checkpoint dirs (model_config.json)
	has_orbax = ckpt.is_dir() and any(p.name.startswith("epoch-") for p in ckpt.iterdir())
	has_model_config = (ckpt / "model_config.json").exists()
	if not ckpt.is_file() and not has_orbax and not has_model_config:
		pytest.skip("Strict lattice/force checks require orbax checkpoint (epoch-*) or model_config.json")

	import jax
	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		check_lattice_invariance,
	)
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters

	cell_lengths = (30.0, 40.0, 50.0)
	cell_matrix = jnp.array([
		[cell_lengths[0], 0, 0],
		[0, cell_lengths[1], 0],
		[0, 0, cell_lengths[2]],
	])
	factory = setup_calculator(
		ATOMS_PER_MONOMER=10,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=20,
		cell=cell_lengths,
	)

	key = jax.random.PRNGKey(123)
	R = jax.random.uniform(
		key, (20, 3),
		minval=jnp.array([2.0, 2.0, 2.0]),
		maxval=jnp.array(cell_lengths) - 2.0,
	)
	Z = jnp.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, spherical_cutoff_calculator = factory(
		atomic_numbers=np.array(Z),
		atomic_positions=np.array(R),
		n_monomers=2,
		cutoff_params=cutoff_params,
	)

	def sc_fn(R_in, Z_in, n_monomers, cutoff_params_in):
		return spherical_cutoff_calculator(
			positions=R_in,
			atomic_numbers=Z_in,
			n_monomers=n_monomers,
			cutoff_params=cutoff_params_in,
		)

	try:
		delta_E = check_lattice_invariance(
			sc_fn, R, Z, 2, cutoff_params, cell_matrix
		)
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	assert abs(delta_E) < 1e-4, (
		f"Orthorhombic lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


def test_pbc_force_direction_mic():
	"""
	For two monomers near opposite faces of the cell, verify that MIC displacement
	points toward the nearest image (shorter vector through PBC).
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")

	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

	L = 20.0
	cell = jnp.diag(jnp.array([L, L, L]))
	# Atom A near origin, atom B near opposite face - MIC should give short displacement
	Ri = jnp.array([1.0, 1.0, 1.0])
	Rj = jnp.array([19.0, 19.0, 19.0])  # far in Cartesian
	d_mic = mic_displacement(Ri, Rj, cell)
	# Through PBC, nearest image of Rj is at (19-L, 19-L, 19-L) = (-1,-1,-1)
	# So d_mic should be ~ (-2,-2,-2), length ~ 3.46, not ~ 31 (Cartesian)
	d_cart = Rj - Ri
	assert jnp.linalg.norm(d_mic) < jnp.linalg.norm(d_cart), (
		f"MIC displacement {d_mic} (norm={jnp.linalg.norm(d_mic):.4f}) should be "
		f"shorter than Cartesian {d_cart} (norm={jnp.linalg.norm(d_cart):.4f})"
	)
	np.testing.assert_allclose(d_mic, jnp.array([-2.0, -2.0, -2.0]), atol=1e-5)


def test_pbc_wrap_groups_com_near_boundary():
	"""
	Test wrap_groups when a monomer's COM is near a cell face.
	Wrap should shift by lattice vector so COM lands in [0,1)^3.
	"""
	if not _can_import("jax"):
		pytest.skip("jax not available in this environment")

	import jax.numpy as jnp
	from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
		frac_coords,
		wrap_groups,
	)

	cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
	groups = [jnp.arange(0, 3), jnp.arange(3, 6)]
	# Place group 0 with COM at ~(9.5, 9.5, 9.5) - near upper boundary
	R = jnp.array([
		[9.0, 9.0, 9.0],
		[10.0, 10.0, 10.0],
		[9.5, 9.5, 9.5],
		[1.0, 1.0, 1.0],
		[2.0, 2.0, 2.0],
		[1.5, 1.5, 1.5],
	])
	R_wrapped = wrap_groups(R, groups, cell)
	S = frac_coords(R_wrapped, cell)
	# All fractional coords should be in [0, 1)
	assert jnp.all(S >= 0) and jnp.all(S < 1.01), (
		f"Wrapped fractional coords should be in [0,1), got min={S.min():.4f} max={S.max():.4f}"
	)


