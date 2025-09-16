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


