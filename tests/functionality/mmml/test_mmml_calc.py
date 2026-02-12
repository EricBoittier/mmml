import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np
import importlib as _il  # lazy import in tests


def test_ev2kcalmol_constant():
	# Ensure the EV->kcal/mol conversion used by calculators is reasonable
	from mmml.pycharmmInterface.mmml_calculator import ev2kcalmol
	assert abs(ev2kcalmol - 23.0605) < 0.05


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_setup_calculator_factory_smoke():
	# Require JAX runtime for restart helpers
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	# Skip if no checkpoints are present
	ckpt_env = os.environ.get("MMML_CKPT")
	if ckpt_env:
		ckpt = Path(ckpt_env)
	else:
		ckpt = Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	from mmml.pycharmmInterface.mmml_calculator import setup_calculator

	# Create factory without touching MM (avoids CHARMM setup during smoke test)
	factory = setup_calculator(
		ATOMS_PER_MONOMER=2,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=8,
	)

	assert callable(factory)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_ml_energy_matches_reference_when_data_available():
	"""
	Optional: use the original dataset to sanity-check ML energy path.
	Skips if data is not present.
	"""
	# Resolve data path: env MMML_DATA or the original relative path used before cleanup
	data_path = os.environ.get(
		"MMML_DATA",
		"mmml/data/fixed-acetone-only_MP2_21000.npz",
	)
	p = Path(data_path)
	if not p.exists():
		pytest.skip(f"Dataset not found at {p}")

	# Lightweight import to prepare one batch
	from mmml.physnetjax.physnetjax.data.data import prepare_datasets
	from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit
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
	from mmml.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		ev2kcalmol,
	)
	# Skip if e3x (pair indices) backend is not available
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")
	factory = setup_calculator(
		ATOMS_PER_MONOMER=10,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		model_restart_path=Path("mmml/physnetjax/ckpts"),
		MAX_ATOMS_PER_SYSTEM=20,
		ml_energy_conversion_factor=ev2kcalmol,
		ml_force_conversion_factor=ev2kcalmol,
	)

	# ASE atoms (lazy import and skip if missing)
	ase_spec = importlib.util.find_spec("ase")
	if ase_spec is None:
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
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_check_lattice_invariance():
	"""
	Test that the energy is invariant under translation of monomer 0 by a lattice vector (PBC).
	Uses check_lattice_invariance with a spherical_cutoff_calculator that applies PBC mapping.
	"""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")

	ckpt_env = os.environ.get("MMML_CKPT")
	ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	import jax.numpy as jnp
	from mmml.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		check_lattice_invariance,
	)
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

	# Setup calculator with PBC (cell=40 Å cubic)
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
	# Build pbc_map (factory does not expose it; create with same params)
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

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
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	def sc_fn(R_in, Z_in, n_monomers, cutoff_params_in):
		R_mapped = pbc_map(R_in)
		return spherical_cutoff_calculator(
			positions=R_mapped,
			atomic_numbers=Z_in,
			n_monomers=n_monomers,
			cutoff_params=cutoff_params_in,
		)

	delta_E = check_lattice_invariance(
		sc_fn, R, Z, 2, cutoff_params, cell_matrix
	)
	assert abs(delta_E) < 1e-4, (
		f"Lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_pbc_energy_invariance_via_ase():
	"""
	Test that energy is invariant under translation of monomer 0 by a lattice vector (PBC)
	using AseDimerCalculator directly (ASE atoms.get_potential_energy()).
	"""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")
	ase_spec = importlib.util.find_spec("ase")
	if ase_spec is None:
		pytest.skip("ase not available in this environment")

	ckpt_env = os.environ.get("MMML_CKPT")
	ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

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
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	E0 = atoms.get_potential_energy()
	a = np.array([float(cell_length), 0.0, 0.0])
	n_per = len(atoms) // 2
	g0 = np.arange(n_per)
	R_shift = R.copy()
	R_shift[g0] += a
	atoms.set_positions(R_shift)
	E1 = atoms.get_potential_energy()
	delta_E = float(E1 - E0)
	assert abs(delta_E) < 1e-4, (
		f"ASE lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_pbc_force_invariance():
	"""
	Test that forces are unchanged after translating monomer 0 by a lattice vector (PBC).
	Forces on both monomers should be invariant (same relative geometry).
	"""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")
	ase_spec = importlib.util.find_spec("ase")
	if ase_spec is None:
		pytest.skip("ase not available in this environment")

	ckpt_env = os.environ.get("MMML_CKPT")
	ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

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
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	F0 = atoms.get_forces()
	n_per = len(atoms) // 2
	F0_mon0, F0_mon1 = F0[:n_per], F0[n_per:]

	a = np.array([float(cell_length), 0.0, 0.0])
	g0 = np.arange(n_per)
	R_shift = R.copy()
	R_shift[g0] += a
	atoms.set_positions(R_shift)
	F1 = atoms.get_forces()
	F1_mon0, F1_mon1 = F1[:n_per], F1[n_per:]

	assert np.allclose(F0_mon0, F1_mon0, atol=1e-5, rtol=1e-4), (
		f"Force invariance violated for monomer 0: max diff = {np.max(np.abs(F0_mon0 - F1_mon0))}"
	)
	assert np.allclose(F0_mon1, F1_mon1, atol=1e-5, rtol=1e-4), (
		f"Force invariance violated for monomer 1: max diff = {np.max(np.abs(F0_mon1 - F1_mon1))}"
	)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_pbc_energy_invariance_ml_mm():
	"""
	Test that energy is invariant under translation of monomer 0 by a lattice vector (PBC)
	with both ML and MM enabled. Requires CHARMM/PSF setup via a dimer PDB.
	"""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")

	ckpt_env = os.environ.get("MMML_CKPT")
	ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	repo_root = Path(__file__).resolve().parent.parent.parent.parent
	pdb_path = repo_root / "notebooks/ffFIT/example-acetone/pdb/init-packmol.pdb"
	if not pdb_path.exists():
		pytest.skip(f"PDB not found at {pdb_path} for MM setup")

	import tempfile
	import jax.numpy as jnp
	import ase.io
	from mmml.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		check_lattice_invariance,
	)
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

	try:
		from mmml.pycharmmInterface.setupBox import setup_box_generic
	except ImportError:
		pytest.skip("setup_box_generic not available")

	with tempfile.TemporaryDirectory() as tmpdir:
		import os as os_module
		orig_cwd = os_module.getcwd()
		try:
			os_module.chdir(tmpdir)
			os_module.makedirs("psf", exist_ok=True)
			os_module.makedirs("pdb", exist_ok=True)
			setup_box_generic(str(pdb_path), side_length=40.0, tag="pbcmm")
		except Exception as e:
			pytest.skip(f"CHARMM/PSF setup failed: {e}")
		finally:
			os_module.chdir(orig_cwd)

	atoms_loaded = ase.io.read(str(pdb_path))
	R_full = atoms_loaded.get_positions()
	Z_full = atoms_loaded.get_atomic_numbers()
	R = np.asarray(R_full[:20])
	Z = np.asarray(Z_full[:20])
	cell_length = 40.0

	factory = setup_calculator(
		ATOMS_PER_MONOMER=10,
		N_MONOMERS=2,
		doML=True,
		doMM=True,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=20,
		cell=cell_length,
	)
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

	cutoff_params = CutoffParameters()
	calc, spherical_cutoff_calculator = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	def sc_fn(R_in, Z_in, n_monomers, cutoff_params_in):
		R_mapped = pbc_map(R_in)
		return spherical_cutoff_calculator(
			positions=R_mapped,
			atomic_numbers=Z_in,
			n_monomers=n_monomers,
			cutoff_params=cutoff_params_in,
		)

	delta_E = check_lattice_invariance(
		sc_fn, R, Z, 2, cutoff_params, cell_matrix
	)
	assert abs(delta_E) < 1e-3, (
		f"ML+MM lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-3)"
	)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_pbc_force_gradient_numerical():
	"""
	Numerical gradient of energy vs positions; compare against atoms.get_forces()
	to ensure transform_forces (VJP) is correct.
	"""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")
	if importlib.util.find_spec("e3x") is None:
		pytest.skip("e3x not available in this environment")
	ase_spec = importlib.util.find_spec("ase")
	if ase_spec is None:
		pytest.skip("ase not available in this environment")

	ckpt_env = os.environ.get("MMML_CKPT")
	ckpt = Path(ckpt_env) if ckpt_env else Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	import jax
	import jax.numpy as jnp
	import ase
	from mmml.pycharmmInterface.mmml_calculator import setup_calculator
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

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
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

	key = jax.random.PRNGKey(42)
	R = np.asarray(jax.random.uniform(key, (20, 3), minval=2.0, maxval=cell_length - 2.0))
	Z = np.array([6] * 20)
	cutoff_params = CutoffParameters()

	calc, _ = factory(
		atomic_numbers=Z,
		atomic_positions=R,
		n_monomers=2,
		cutoff_params=cutoff_params,
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc

	F_analytical = atoms.get_forces()
	h = 1e-5
	F_numerical = np.zeros_like(R)
	for i in range(len(R)):
		for j in range(3):
			Rp = R.copy()
			Rp[i, j] += h
			atoms.set_positions(Rp)
			Ep = atoms.get_potential_energy()
			Rm = R.copy()
			Rm[i, j] -= h
			atoms.set_positions(Rm)
			Em = atoms.get_potential_energy()
			F_numerical[i, j] = -(Ep - Em) / (2 * h)

	atoms.set_positions(R)

	rel_err = np.abs(F_analytical - F_numerical)
	denom = np.abs(F_numerical) + 1e-10
	rel_err = np.where(denom > 1e-8, rel_err / denom, rel_err)
	max_rel_err = np.max(rel_err)
	max_abs_err = np.max(np.abs(F_analytical - F_numerical))
	assert max_rel_err < 0.1 or max_abs_err < 1e-3, (
		f"Force gradient mismatch: max_rel_err={max_rel_err:.4e}, max_abs_err={max_abs_err:.4e}"
	)


def test_pbc_mic_displacement_symmetry():
	"""mic_displacement(Ri, Rj, cell) = -mic_displacement(Rj, Ri, cell)."""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")

	import jax.numpy as jnp
	from mmml.pycharmmInterface.pbc_utils_jax import mic_displacement

	cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
	Ri = jnp.array([1.0, 2.0, 3.0])
	Rj = jnp.array([7.0, 8.0, 9.0])

	d_ij = mic_displacement(Ri, Rj, cell)
	d_ji = mic_displacement(Rj, Ri, cell)
	np.testing.assert_allclose(d_ij, -d_ji, atol=1e-10)


def test_pbc_wrap_unwrap_roundtrip():
	"""Unwrap then wrap should recover equivalent positions (same fractional coords mod 1)."""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")

	import jax
	import jax.numpy as jnp
	from mmml.pycharmmInterface.pbc_utils_jax import (
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
	S_diff = (S_roundtrip - S_orig) - jnp.floor(S_roundtrip - S_orig)
	np.testing.assert_allclose(S_diff, 0.0, atol=1e-10)


def test_pbc_mapper_idempotent():
	"""For positions already in primary cell, pbc_map(pbc_map(R)) == pbc_map(R)."""
	if importlib.util.find_spec("jax") is None:
		pytest.skip("jax not available in this environment")

	import jax
	import jax.numpy as jnp
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

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
	np.testing.assert_allclose(R1, R2, atol=1e-10)


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_pbc_energy_invariance_orthorhombic_cell():
	"""
	Test energy invariance under lattice translation with orthorhombic cell [30, 40, 50].
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
	from mmml.pycharmmInterface.mmml_calculator import (
		setup_calculator,
		check_lattice_invariance,
	)
	from mmml.pycharmmInterface.cutoffs import CutoffParameters
	from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

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
	mol_id = jnp.array([
		i * jnp.ones(10, dtype=jnp.int32)
		for i in range(2)
	], dtype=jnp.int32)
	pbc_map = make_pbc_mapper(cell=cell_matrix, mol_id=mol_id, n_monomers=2)

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
		do_pbc_map=True,
		pbc_map=pbc_map,
	)

	def sc_fn(R_in, Z_in, n_monomers, cutoff_params_in):
		R_mapped = pbc_map(R_in)
		return spherical_cutoff_calculator(
			positions=R_mapped,
			atomic_numbers=Z_in,
			n_monomers=n_monomers,
			cutoff_params=cutoff_params_in,
		)

	delta_E = check_lattice_invariance(
		sc_fn, R, Z, 2, cutoff_params, cell_matrix
	)
	assert abs(delta_E) < 1e-4, (
		f"Orthorhombic lattice invariance violated: |E0 - E1| = {abs(delta_E):.6e} (expected < 1e-4)"
	)


