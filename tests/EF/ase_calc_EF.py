"""
ASE calculator for the electric field model
"""

import functools

import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from flax import linen as nn

# Import model and functions from training script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from training import MessagePassingModel
from model_functions import energy_and_forces


def load_params(params_path):
    """Load parameters from JSON file."""
    with open(params_path, 'r') as f:
        params_dict = json.load(f)
    
    # Convert numpy arrays back from lists
    def convert_to_jax(obj):
        if isinstance(obj, dict):
            return {k: convert_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            arr = np.array(obj)
            if arr.dtype == np.float64:
                return jnp.array(arr, dtype=jnp.float32)
            elif arr.dtype == np.int64:
                return jnp.array(arr, dtype=jnp.int32)
            return jnp.array(arr)
        return obj
    
    params = convert_to_jax(params_dict)
    return params


def load_config(config_path):
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class AseCalculatorEF(ase_calc.Calculator):
    """ASE calculator for electric field model."""
    
    implemented_properties = ["energy", "forces", "dipole"]
    
    def __init__(self, params_path, config_path=None, electric_field=None,
                 field_scale=0.001, **kwargs):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        params_path : str or Path
            Path to parameters JSON file (e.g., params.json or params-UUID.json)
        config_path : str or Path, optional
            Path to config JSON file. If None, will try to auto-detect from params UUID.
        electric_field : array-like, shape (3,), optional
            Default electric field vector in model input units (0.001 au by default).
            If None, must be provided in atoms.info['electric_field'].
        field_scale : float, optional
            Conversion factor: Ef_physical [au] = Ef_input * field_scale.
            Default 0.001 (i.e. the model input is in milli-au).
        **kwargs
            Additional arguments passed to ase.calculators.calculator.Calculator
        """
        ase_calc.Calculator.__init__(self, **kwargs)
        self.field_scale = field_scale
        
        # Load parameters
        params_path = Path(params_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")
        self.params = load_params(params_path)
        
        # Load config (try to auto-detect from UUID if not provided)
        if config_path is None:
            # Try to extract UUID from params filename
            if params_path.stem.startswith('params-') and len(params_path.stem) > 7:
                uuid_part = params_path.stem[7:]  # Remove 'params-' prefix
                config_candidate = params_path.parent / f'config-{uuid_part}.json'
                if config_candidate.exists():
                    config_path = config_candidate
                else:
                    # Try config.json symlink
                    config_candidate = params_path.parent / 'config.json'
                    if config_candidate.exists():
                        config_path = config_candidate
        
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                config = load_config(config_path)
                # Extract model config (may be nested under 'model', 'model_config', or flat)
                model_keys = {'features', 'max_degree', 'num_iterations',
                              'num_basis_functions', 'cutoff', 'max_atomic_number',
                              'include_pseudotensors'}
                if 'model' in config and isinstance(config['model'], dict):
                    model_config = {k: v for k, v in config['model'].items()
                                    if k in model_keys}
                elif 'model_config' in config and isinstance(config['model_config'], dict):
                    model_config = {k: v for k, v in config['model_config'].items()
                                    if k in model_keys}
                else:
                    # Flat config — keys at top level
                    model_config = {k: v for k, v in config.items()
                                    if k in model_keys}
                if not model_config:
                    raise ValueError(
                        f"Could not extract model config from {config_path}. "
                        f"Expected keys {model_keys} under 'model', 'model_config', or top-level."
                    )
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            # Default model config (infer from params if possible, or use defaults)
            print("Warning: No config file found. Using default model configuration.")
            model_config = {
                'features': 64,
                'max_degree': 2,
                'num_iterations': 2,
                'num_basis_functions': 64,
                'cutoff': 10.0,
                'max_atomic_number': 55,
                'include_pseudotensors': True,
            }
        
        # Create model
        self.model = MessagePassingModel(**model_config)
        
        # Store default electric field
        if electric_field is not None:
            self.electric_field = jnp.asarray(electric_field, dtype=jnp.float32)
        else:
            self.electric_field = None
        
        # JIT compile the model apply function for efficiency
        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def model_apply(params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, 
                       batch_segments, batch_size, dst_idx=None, src_idx=None):
            return self.model.apply(params, atomic_numbers, positions, Ef,
                                  dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
                                  batch_segments=batch_segments, batch_size=batch_size,
                                  dst_idx=dst_idx, src_idx=src_idx)
        
        self.model_apply = model_apply
        
        # JIT compile energy_and_forces
        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def compute_energy_forces_dipole(atomic_numbers, positions, Ef, dst_idx, src_idx,
                                        dst_idx_flat, src_idx_flat, batch_segments, batch_size):
            return energy_and_forces(
                self.model_apply, self.params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                Ef=Ef,
                dst_idx_flat=dst_idx_flat,
                src_idx_flat=src_idx_flat,
                batch_segments=batch_segments,
                batch_size=batch_size,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
        
        self.compute_energy_forces_dipole = compute_energy_forces_dipole
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=ase_calc.all_changes):
        """
        Calculate properties for the given atoms.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        properties : list of str
            Properties to calculate (e.g., ['energy', 'forces', 'dipole'])
        system_changes : list of str
            System changes since last calculation
        """
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get atomic numbers and positions
        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        positions = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        n_atoms = len(atoms)
        
        # Get electric field from atoms.info or use default
        if 'electric_field' in atoms.info:
            Ef = jnp.asarray(atoms.info['electric_field'], dtype=jnp.float32)
        elif self.electric_field is not None:
            Ef = self.electric_field
        else:
            raise ValueError(
                "Electric field not provided. Set atoms.info['electric_field'] or "
                "provide electric_field parameter to calculator."
            )
        
        # Ensure Ef has shape (3,)
        Ef = Ef.reshape(3)
        
        # Create indices for message passing
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
        
        # Set up batch (single molecule, batch_size=1)
        batch_size = 1
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)  # All atoms in same batch
        
        # Compute flattened indices
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * n_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
        
        # Reshape for model (add batch dimension)
        atomic_numbers_batched = atomic_numbers[None, :]  # (1, N)
        positions_batched = positions[None, :, :]  # (1, N, 3)
        Ef_batched = Ef[None, :]  # (1, 3)
        
        # Compute energy, forces, and dipole
        energy, forces, dipole = self.compute_energy_forces_dipole(
            atomic_numbers=atomic_numbers_batched,
            positions=positions_batched,
            Ef=Ef_batched,
            dst_idx=dst_idx,
            src_idx=src_idx,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
        )
        
        # Convert to numpy and extract scalar values
        energy = float(np.asarray(energy)[0])  # Extract from batch (1,) -> scalar
        forces = np.asarray(forces)[0]  # Extract from batch (1, N, 3) -> (N, 3)
        dipole = np.asarray(dipole[0])  # Extract from batch (1, 3) -> (3,)
        
        # Store results
        self.results = {
            'energy': energy,
            'forces': forces,
            'dipole': dipole,
        }
    
    def set_electric_field(self, electric_field):
        """Set the default electric field for calculations."""
        self.electric_field = jnp.asarray(electric_field, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Internal helper to avoid repeating boilerplate in every method
    # ------------------------------------------------------------------
    def _prepare_inputs(self, atoms=None):
        """Prepare batched JAX inputs for a single molecule.

        Returns dict with keys:
            atomic_numbers_batched (1, N), positions_batched (1, N, 3),
            Ef_batched (1, 3), dst_idx, src_idx, dst_idx_flat, src_idx_flat,
            batch_segments, batch_size, n_atoms
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided.")

        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        positions = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        n_atoms = len(atoms)

        if 'electric_field' in atoms.info:
            Ef = jnp.asarray(atoms.info['electric_field'], dtype=jnp.float32)
        elif self.electric_field is not None:
            Ef = self.electric_field
        else:
            raise ValueError("Electric field not provided.")
        Ef = Ef.reshape(3)

        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

        batch_size = 1
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * n_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        return dict(
            atomic_numbers=atomic_numbers[None, :],
            positions=positions[None, :, :],
            Ef=Ef[None, :],
            dst_idx=dst_idx, src_idx=src_idx,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            n_atoms=n_atoms,
        )

    def _call_model_fn(self, fn, atoms=None):
        """Call a model_functions.* function with prepared inputs."""
        inp = self._prepare_inputs(atoms)
        return fn(
            self.model_apply, self.params,
            atomic_numbers=inp['atomic_numbers'],
            positions=inp['positions'],
            Ef=inp['Ef'],
            dst_idx=inp['dst_idx'], src_idx=inp['src_idx'],
            dst_idx_flat=inp['dst_idx_flat'],
            src_idx_flat=inp['src_idx_flat'],
            batch_segments=inp['batch_segments'],
            batch_size=inp['batch_size'],
        )

    # ------------------------------------------------------------------
    # Polarizability  alpha = d(mu)/d(Ef)  [physical, from predicted dipole]
    # ------------------------------------------------------------------
    def get_polarizability(self, atoms=None):
        """Polarizability tensor  alpha_{ab} = d(mu_a)/d(Ef_b).

        Computed from the Jacobian of the *predicted* dipole w.r.t. the
        electric field input.  Converted to physical (au) units using
        ``self.field_scale``.

        Returns
        -------
        alpha : np.ndarray, shape (3, 3)
            Polarizability in atomic units (Bohr³).
        """
        from model_functions import dipole_derivative_field
        raw = self._call_model_fn(dipole_derivative_field, atoms)
        # raw shape: (1, 3, 1, 3) -> extract (3, 3) and convert units
        alpha = np.asarray(raw)[0, :, 0, :] / self.field_scale
        return alpha

    # ------------------------------------------------------------------
    # Atomic Polar Tensor  APT = d(mu)/d(R)
    # ------------------------------------------------------------------
    def get_atomic_polar_tensor(self, atoms=None):
        """Atomic Polar Tensor  P_{a,s,b} = d(mu_a)/d(R_{s,b}).

        Used for IR intensities and as input for the distributed-origin AAT.

        Returns
        -------
        apt : np.ndarray, shape (3, N, 3)
            apt[a, s, b] = d(mu_a) / d(R_{s,b}).
            Units: au_dipole / Angstrom.
        """
        from model_functions import dipole_derivative_positions
        raw = self._call_model_fn(dipole_derivative_positions, atoms)
        # raw shape: (1, 3, 1, N, 3) -> (3, N, 3)
        return np.asarray(raw)[0, :, 0, :, :]

    # ------------------------------------------------------------------
    # Hessian  d²E/dR²  (for normal modes)
    # ------------------------------------------------------------------
    def get_hessian(self, atoms=None):
        """Hessian matrix  H_{si,a; sj,b} = d²E / d(R_{si,a}) d(R_{sj,b}).

        Returns
        -------
        hess : np.ndarray, shape (N, 3, N, 3)
            Units: eV / Angstrom².
        """
        from model_functions import hessian_matrix
        raw = self._call_model_fn(hessian_matrix, atoms)
        # raw shape: (1, N, 3, 1, N, 3) -> (N, 3, N, 3)
        return np.asarray(raw)[0, :, :, 0, :, :]

    # ------------------------------------------------------------------
    # Atomic charges (from model internals)
    # ------------------------------------------------------------------
    def get_atomic_charges(self, atoms=None):
        """Extract ML-predicted atomic partial charges from the model.

        These are the charges q_i used internally to compute the dipole:
            mu = sum_i q_i * (r_i - COM) + sum_i mu_i^atomic

        Returns
        -------
        charges : np.ndarray, shape (N,)
        atomic_dipoles : np.ndarray, shape (N, 3)
        """
        from model_functions import get_atomic_properties

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided.")

        inp = self._prepare_inputs(atoms)
        _energy, _dipole, charges, atomic_dipoles = get_atomic_properties(
            self.model, self.params,
            atomic_numbers=inp['atomic_numbers'],
            positions=inp['positions'],
            Ef=inp['Ef'],
            dst_idx=inp['dst_idx'], src_idx=inp['src_idx'],
            dst_idx_flat=inp['dst_idx_flat'],
            src_idx_flat=inp['src_idx_flat'],
            batch_segments=inp['batch_segments'],
            batch_size=inp['batch_size'],
        )
        return np.asarray(charges)[0], np.asarray(atomic_dipoles)[0]  # (N,), (N, 3)

    # ------------------------------------------------------------------
    # AAT — three levels of approximation
    # ------------------------------------------------------------------
    def get_aat_nuclear(self, atoms=None):
        """AAT using bare nuclear charges Z (Lorentz, distributed-origin gauge).

        M^{nuc,s}_{a,b} = -(Z_s / 4c) * eps_{a,b,g} * R_{s,g}

        This is the crudest approximation (no electronic screening).

        Returns
        -------
        aat : np.ndarray, shape (N, 3, 3)
        """
        from model_functions import aat_nuclear

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided.")

        positions = jnp.asarray(atoms.get_positions())
        Z = jnp.asarray(atoms.get_atomic_numbers())
        return np.asarray(aat_nuclear(positions, Z))

    def get_aat_born(self, atoms=None):
        """AAT using Born effective charges from the APT.

        q_eff,s = (1/3) Tr(APT^s)  — electronically screened charges.
        Then  M^s = -(q_eff / 4c) * eps * R.

        Better than nuclear AAT because q_eff captures charge redistribution.

        Returns
        -------
        aat : np.ndarray, shape (N, 3, 3)
        q_eff : np.ndarray, shape (N,)
        """
        from model_functions import aat_born

        apt = self.get_atomic_polar_tensor(atoms)
        if atoms is None:
            atoms = self.atoms
        positions = jnp.asarray(atoms.get_positions())
        aat, q_eff = aat_born(jnp.asarray(apt), positions)
        return np.asarray(aat), np.asarray(q_eff)

    def get_aat_ml_charges(self, atoms=None):
        """AAT using the model's predicted atomic charges.

        Uses the internal ML charges q_i (from the dipole prediction head)
        in the Lorentz/DO-gauge formula.  These charges represent the model's
        learned charge distribution and include all electronic effects
        captured during training.

        Returns
        -------
        aat : np.ndarray, shape (N, 3, 3)
        charges : np.ndarray, shape (N,)  — the ML charges used.
        """
        from model_functions import aat_ml_charges

        charges, _atomic_dipoles = self.get_atomic_charges(atoms)
        if atoms is None:
            atoms = self.atoms
        apt = self.get_atomic_polar_tensor(atoms)
        positions = jnp.asarray(atoms.get_positions())
        aat = aat_ml_charges(jnp.asarray(apt), positions, jnp.asarray(charges))
        return np.asarray(aat), charges

    # backward-compatible alias
    def get_aat_distributed_origin(self, atoms=None):
        """Alias for get_aat_nuclear (backward compatibility)."""
        return self.get_aat_nuclear(atoms)

    # ------------------------------------------------------------------
    # Energy-based field derivatives (kept for completeness / diagnostics)
    # ------------------------------------------------------------------
    def get_dipole_from_field(self, atoms=None):
        """Compute -dE/dEf (energy derivative w.r.t. field).

        NOTE: Because this model couples Ef through features (not via -mu*Ef),
        this does NOT give the physical dipole.  Use ``get_dipole_moment()``
        (from ``calculate``) for the physical dipole.

        Returns
        -------
        dEdEf : np.ndarray, shape (3,)
        """
        from model_functions import energy_and_dipole_from_field_derivative
        _, dEdEf = self._call_model_fn(
            energy_and_dipole_from_field_derivative, atoms
        )
        return np.asarray(dEdEf).squeeze()

    def get_polarizability_energy(self, atoms=None):
        """Compute -d²E/dEf² (energy Hessian w.r.t. field).

        NOTE: Same caveat as ``get_dipole_from_field``.  Prefer
        ``get_polarizability`` for the physical polarizability.

        Returns
        -------
        hess : np.ndarray, shape (3, 3)
        """
        from model_functions import polarizability_from_energy_hessian
        raw = self._call_model_fn(polarizability_from_energy_hessian, atoms)
        return np.asarray(raw).squeeze()


if __name__ == "__main__":
    """Example: compute all available response properties for a dataset structure."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test AseCalculatorEF — energy, forces, dipole, polarizability, APT, AAT"
    )
    parser.add_argument("--params", type=str, default="params.json")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default="data-full.npz")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--field-scale", type=float, default=0.001,
                       help="Ef_physical [au] = Ef_input * field_scale (default: 0.001)")
    args = parser.parse_args()

    # --- Load dataset --------------------------------------------------
    dataset = np.load(args.data, allow_pickle=True)
    print(f"Dataset keys: {dataset.files}")

    idx = args.index
    Z = dataset["Z"][idx]
    R = dataset["R"][idx]
    if R.ndim == 3 and R.shape[0] == 1:
        R = R.squeeze(0)
    Ef = dataset["Ef"][idx]
    E_ref = float(dataset["E"][idx])

    print(f"\nStructure {idx}:")
    print(f"  Atoms:    {Z}")
    print(f"  Positions shape: {R.shape}")
    print(f"  Ef (input units): {Ef}   (physical: {Ef * args.field_scale} au)")
    print(f"  E_ref:    {E_ref:.6f} eV")

    F_ref = D_ref = P_ref = None
    if "F" in dataset.files:
        F_ref = dataset["F"][idx]
        if F_ref.ndim == 3 and F_ref.shape[0] == 1:
            F_ref = F_ref.squeeze(0)
        print(f"  F_ref shape: {F_ref.shape}")
    if "D" in dataset.files:
        D_ref = dataset["D"][idx]
        print(f"  D_ref (au): {D_ref}")
    if "P" in dataset.files:
        P_ref = dataset["P"][idx]
        print(f"  P_ref shape: {np.array(P_ref).shape}  (polarizability reference)")

    # --- Create calculator ---------------------------------------------
    atoms = ase.Atoms(numbers=Z, positions=R)
    atoms.info['electric_field'] = Ef

    print(f"\nLoading calculator (field_scale={args.field_scale})...")
    calc = AseCalculatorEF(
        params_path=args.params, config_path=args.config,
        field_scale=args.field_scale,
    )
    atoms.calc = calc

    # --- Energy, forces, dipole ----------------------------------------
    print("\n--- Energy, Forces, Dipole ---")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    dipole = atoms.get_dipole_moment()

    print(f"  Energy:    {energy:.6f} eV  (ref: {E_ref:.6f}, err: {abs(energy-E_ref):.6f} eV)")
    print(f"  Forces:    max |F| = {np.max(np.abs(forces)):.6f} eV/A")
    if F_ref is not None:
        print(f"  Forces MAE: {np.mean(np.abs(forces - F_ref)):.6f} eV/A")
    print(f"  Dipole (au): {dipole}")
    if D_ref is not None:
        print(f"  Dipole ref:  {D_ref}")
        print(f"  Dipole MAE:  {np.mean(np.abs(dipole - D_ref)):.6f} au")

    # --- Polarizability (d mu / d Ef) ----------------------------------
    print("\n--- Polarizability  alpha = d(mu)/d(Ef)  [au] ---")
    try:
        alpha = calc.get_polarizability(atoms)
        print(f"  Shape: {alpha.shape}")
        print(f"  alpha =\n{alpha}")
        print(f"  Trace/3 (isotropic): {np.trace(alpha)/3:.4f} au")
        if P_ref is not None:
            P_ref_arr = np.array(P_ref)
            if P_ref_arr.shape == alpha.shape:
                print(f"  Reference polarizability:\n{P_ref_arr}")
                print(f"  MAE: {np.mean(np.abs(alpha - P_ref_arr)):.6f} au")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Atomic Polar Tensor (d mu / d R) ------------------------------
    print("\n--- Atomic Polar Tensor  APT = d(mu)/d(R)  [au/Angstrom] ---")
    try:
        apt = calc.get_atomic_polar_tensor(atoms)
        print(f"  Shape: {apt.shape}  (3, N, 3)")
        # Per-atom |dmu/dR| magnitude (Frobenius norm of 3x3 block)
        apt_norms = np.array([np.linalg.norm(apt[:, s, :]) for s in range(apt.shape[1])])
        print(f"  Per-atom |APT| (Frobenius):  min={apt_norms.min():.4f}  max={apt_norms.max():.4f}  mean={apt_norms.mean():.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # --- ML atomic charges -----------------------------------------------
    print("\n--- ML Atomic Charges (from model internals) ---")
    try:
        ml_charges, ml_atomic_dipoles = calc.get_atomic_charges(atoms)
        print(f"  Charges shape: {ml_charges.shape},  sum = {ml_charges.sum():.4f}")
        print(f"  Charges: {ml_charges}")
        print(f"  Atomic dipoles shape: {ml_atomic_dipoles.shape}")
    except Exception as e:
        print(f"  Error: {e}")

    # --- AAT — three levels of approximation ---------------------------
    print("\n--- AAT (Lorentz:  B_eff = -(v x E)/c²) ---")
    print("  All use  M^s = -(q_s / 4c) * eps * R_s  with different charges:")

    try:
        aat_Z = calc.get_aat_nuclear(atoms)
        norms_Z = np.array([np.linalg.norm(aat_Z[s]) for s in range(aat_Z.shape[0])])
        print(f"\n  1) Nuclear charges Z_s (crudest):")
        print(f"     |AAT|:  min={norms_Z.min():.6f}  max={norms_Z.max():.6f}  mean={norms_Z.mean():.6f}")
    except Exception as e:
        print(f"  Nuclear AAT error: {e}")

    try:
        aat_born, q_eff = calc.get_aat_born(atoms)
        norms_born = np.array([np.linalg.norm(aat_born[s]) for s in range(aat_born.shape[0])])
        print(f"\n  2) Born effective charges q_eff = (1/3)Tr(APT) (includes electronic screening):")
        print(f"     q_eff: {q_eff}")
        print(f"     |AAT|:  min={norms_born.min():.6f}  max={norms_born.max():.6f}  mean={norms_born.mean():.6f}")
    except Exception as e:
        print(f"  Born AAT error: {e}")

    try:
        aat_ml, ml_q = calc.get_aat_ml_charges(atoms)
        norms_ml = np.array([np.linalg.norm(aat_ml[s]) for s in range(aat_ml.shape[0])])
        print(f"\n  3) ML-predicted charges (from model's dipole head):")
        print(f"     q_ML: {ml_q}")
        print(f"     |AAT|:  min={norms_ml.min():.6f}  max={norms_ml.max():.6f}  mean={norms_ml.mean():.6f}")
    except Exception as e:
        print(f"  ML-charge AAT error: {e}")

    # --- Energy-based field derivatives (diagnostic) -------------------
    print("\n--- Energy-based field derivatives (diagnostic, NOT physical) ---")
    try:
        dEdEf = calc.get_dipole_from_field(atoms)
        print(f"  -dE/dEf (raw):   {dEdEf}")
        print(f"  Predicted dipole: {dipole}")
        print(f"  These differ because the model does NOT have E = E0 - mu*Ef coupling.")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        alpha_E = calc.get_polarizability_energy(atoms)
        print(f"  -d2E/dEf2 (raw): diag = {np.diag(alpha_E)}")
    except Exception as e:
        print(f"  Error: {e}")

