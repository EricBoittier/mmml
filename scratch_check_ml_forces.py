import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ase.io import read

# Paths
workspace_dir = Path("/Users/ericboittier/mmml")
traj_path = workspace_dir / "cg_fire.traj"
ckpt_path = workspace_dir / "examples/params_aaa_long_2026-07-04_22-30-27.json"

if not traj_path.is_file():
    print(f"Error: Trajectory file {traj_path} not found.")
    sys.exit(1)

if not ckpt_path.is_file():
    print(f"Error: Checkpoint file {ckpt_path} not found.")
    sys.exit(1)

# 1. Load the first frame of the trajectory
try:
    atoms = read(str(traj_path), index=0)
    pos = jnp.array(atoms.get_positions(), dtype=jnp.float64)
    z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)
    print(f"Loaded frame from {traj_path.name}: {len(atoms)} atoms.")
except Exception as e:
    print(f"Error loading trajectory: {e}")
    sys.exit(1)

# 2. Mock pycharmm to allow importing simple_inference
from unittest import mock
fake_pycharmm = mock.MagicMock()
sys.modules["pycharmm"] = fake_pycharmm
sys.modules["pycharmm.coor"] = mock.MagicMock()
sys.modules["pycharmm.energy"] = mock.MagicMock()
sys.modules["pycharmm.select"] = mock.MagicMock()
sys.modules["pycharmm.lingo"] = mock.MagicMock()
sys.modules["mmml.interfaces.pycharmmInterface.import_pycharmm"] = mock.MagicMock()


# Import model calculator and interfaces
from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn
from jax_md import space

# 3. Create the calculator and model
try:
    calc = create_calculator_from_checkpoint(str(ckpt_path))
    model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
    params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
    print("Successfully loaded calculator and parameters from checkpoint.")
except Exception as e:
    print(f"Error loading calculator: {e}")
    sys.exit(1)

# 4. Setup monomer indices
n_trialanine = 42
monomer_indices = [np.arange(n_trialanine)]
for i in range(n_trialanine, len(pos), 3):
    monomer_indices.append(np.arange(i, i + 3))
jax_monomer_indices = [jnp.array(idx, dtype=jnp.int32) for idx in monomer_indices]

# Displacement fn
cell = np.asarray(atoms.get_cell(), dtype=np.float64)
displacement_fn, shift_fn = space.periodic(cell)

# 5. Build monomer energy function
compute_monomer_energy = make_monomer_energy_fn(
    model, params, z, jax_monomer_indices, displacement_fn
)

# 6. Compute ML forces
@jax.jit
def get_ml_forces(r):
    return -jax.grad(compute_monomer_energy)(r)

print("Compiling and evaluating ML forces...")
f_ml = np.asarray(get_ml_forces(pos))
f_ml_mag = np.linalg.norm(f_ml, axis=-1)

# Print peptide vs water force stats
pep_forces = f_ml_mag[:n_trialanine]
wat_forces = f_ml_mag[n_trialanine:]

print("\n--- ML Force Magnitude Statistics ---")
print(f"Peptide (first 42 atoms): min={pep_forces.min():.6e}, max={pep_forces.max():.6e}, mean={pep_forces.mean():.6e}")
print(f"Water (rest of atoms):    min={wat_forces.min():.6e}, max={wat_forces.max():.6e}, mean={wat_forces.mean():.6e}")

# Check if peptide forces are exactly 0
n_zero = np.sum(pep_forces < 1e-12)
print(f"\nNumber of peptide atoms with ML forces < 1e-12: {n_zero} / {n_trialanine}")
if n_zero > 0:
    print(f"Zero force indices: {np.where(pep_forces < 1e-12)[0]}")
