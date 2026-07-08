import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ase.io import read

# Enable JIT/gradient tracing to verify the compiler likes it
jax.config.update("jax_disable_jit", False)

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
from collections import defaultdict
by_size = defaultdict(list)
for idx in monomer_indices:
    by_size[len(idx)].append(idx)

group_fns = []
for size, indices_list in by_size.items():
    stacked_indices = jnp.stack(indices_list)
    n_atoms = size
    dst_idx, src_idx = np.where(~np.eye(n_atoms, dtype=bool))
    dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
    src_idx = jnp.array(src_idx, dtype=jnp.int32)
    
    # Vectorize model.apply using compute_forces=False
    vmapped_apply = jax.vmap(
        lambda pos, atomic_nums, d_idx=dst_idx, s_idx=src_idx: model.apply(
            params,
            atomic_numbers=atomic_nums,
            positions=pos,
            dst_idx=d_idx,
            src_idx=s_idx,
            compute_forces=False,
        )["energy"],
        in_axes=(0, 0)
    )
    
    group_z = z[stacked_indices]
    
    vmapped_displacement = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0)
    )
    
    def group_energy(r, stacked_idx=stacked_indices, gz=group_z, v_apply=vmapped_apply, v_disp=vmapped_displacement):
        group_pos = r[stacked_idx]
        ref_pos = group_pos[:, 0, :]
        displacements = v_disp(group_pos, ref_pos)
        unfolded_pos = ref_pos[:, None, :] + displacements
        com = jnp.mean(unfolded_pos, axis=1, keepdims=True)
        centered_pos = unfolded_pos - com
        energies = v_apply(centered_pos, gz)
        return jnp.sum(energies)
        
    group_fns.append(group_energy)

def compute_monomer_energy(r):
    return sum(fn(r) for fn in group_fns)

# 6. Compute ML forces
@jax.jit
def get_ml_forces(r):
    return -jax.grad(compute_monomer_energy)(r)

print("Compiling and evaluating ML forces...")
try:
    f_ml = np.asarray(get_ml_forces(pos))
    print("ML forces calculated successfully!")
    f_ml_mag = np.linalg.norm(f_ml, axis=-1)
    
    pep_forces = f_ml_mag[:n_trialanine]
    wat_forces = f_ml_mag[n_trialanine:]
    
    print("\n--- ML Force Magnitude Statistics ---")
    print(f"Peptide (first 42 atoms): min={pep_forces.min():.6e}, max={pep_forces.max():.6e}, mean={pep_forces.mean():.6e}")
    print(f"Water (rest of atoms):    min={wat_forces.min():.6e}, max={wat_forces.max():.6e}, mean={wat_forces.mean():.6e}")
except Exception as e:
    print(f"Failed to calculate ML forces: {e}")
    import traceback
    traceback.print_exc()
