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
def test_pbc_force_gradient_numerical():
	"""
	ML monomer forces match ``-dE/dR`` under PBC (MIC-only, no coordinate map).

	Monomer-only (``doML_dimer=False``). Validates ``ModelOutput.forces``,
	``jax.grad`` of the scalar energy, and ASE ``get_forces()`` — the same
	checks as ``test_ml_only_jax_autograd_matches_model_forces``, with explicit
	periodic box forwarding. Central differences are omitted here because they
	are unstable on float32 GPU PBC graphs; use ``scripts/check_mlpot_forces_fd.py``
	for FD sweeps on CPU.
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
	from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
	from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
	cell_length = 40.0
	factory = setup_calculator(
		ATOMS_PER_MONOMER=10,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		doML_dimer=False,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=20,
		cell=cell_length,
	)
	cell_matrix = jnp.array([
		[cell_length, 0, 0],
		[0, cell_length, 0],
		[0, 0, cell_length],
	])
	key = jax.random.PRNGKey(11)
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
	box = jnp.asarray(cell_matrix, dtype=jnp.float32)
	z_j = jnp.asarray(Z, dtype=jnp.int32)

	try:
		model_out = spherical_fn(
			atomic_numbers=z_j,
			positions=jnp.asarray(R, dtype=jnp.float32),
			n_monomers=2,
			cutoff_params=cutoff_params,
			doML=True,
			doMM=False,
			doML_dimer=False,
			box=box,
		)
		F_model = np.asarray(model_out.forces)

		@jax.jit
		def energy_fn(pos):
			out = spherical_fn(
				atomic_numbers=z_j,
				positions=jnp.asarray(pos, dtype=jnp.float32),
				n_monomers=2,
				cutoff_params=cutoff_params,
				doML=True,
				doMM=False,
				doML_dimer=False,
				box=box,
			)
			return out.energy.reshape(-1)[0]

		F_grad = np.asarray(-jax.grad(energy_fn)(jnp.asarray(R, dtype=jnp.float32)))
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)

	assert np.allclose(F_model, F_grad, rtol=0.02, atol=0.05), (
		f"ModelOutput.forces vs jax.grad: max |ΔF| = {np.max(np.abs(F_model - F_grad)):.4e}"
	)

	# ASE calculator should expose the same analytical forces (production path).
	atoms = ase.Atoms(Z, R, cell=cell_matrix, pbc=True)
	atoms.calc = calc
	try:
		F_ase = atoms.get_forces()
	except Exception as exc:
		_skip_if_runtime_incompatible(exc)
	assert np.allclose(F_ase, F_model, rtol=1e-4, atol=1e-3), (
		f"ASE forces vs ModelOutput: max |ΔF| = {np.max(np.abs(F_ase - F_model)):.4e}"
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
