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


