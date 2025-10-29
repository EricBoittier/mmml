# ================================================
# Neural-guided MCTS for DCMNET Model Selection
# ================================================
"""
MCTS-based optimization for selecting distributed charges (and their positions)
from one or more DCMNET models to minimize ESP (Electrostatic Potential) loss.

Highlights:
- Actions are tuple indices ``(atom_idx, charge_idx)`` toggling a specific charge.
- Inputs are normalized (batch dims of size 1 are squeezed):
  - ``esp_target`` shape (N,) or (1, N)
  - ``vdw_surface`` shape (N, 3) or (1, N, 3)
- Per-model predictions are accepted as dicts mapping ``model_id`` to arrays
  or as single arrays; shapes normalized to:
  - charges: (n_atoms, n_charges)
  - positions: (n_atoms, n_charges, 3)
- Ghost atoms (atomic_numbers <= 0) are ignored via ``n_atoms = sum(Z > 0)``.
- Optional target total selection: optimizer can enforce a total number of
  selected charges and search across ``target±span`` to return the best.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any
from functools import partial
import pandas as pd
import signal
import sys

# Import DCMNET components
from .dcmnet.models import DCM1, DCM2, DCM3, DCM4, dcm1_params, dcm2_params, dcm3_params, dcm4_params
from .dcmnet.loss import esp_mono_loss
from .dcmnet.electrostatics import calc_esp


CONVERSION_FACTOR = 1 #1.88973


# =========================================================
# DCMNET Model Selection Environment
# =========================================================

class DCMNETSelectionEnv:
    """
    Environment for selecting individual charges from DCMNET models.
    
    State: Binary matrix of shape (n_atoms, total_charges) indicating which charges are selected
    Actions: Toggle a charge selection via tuple ``(atom_idx, charge_idx)``
    Goal: Find charge combination that minimizes ESP loss
    
    Each model predicts charges for each atom, and we can select which specific
    charges to use from which models for each atom.

    Behavior notes:
    - Until every atom has at least one selected charge, legal actions are
      restricted to selecting (turning on) charges for atoms with none.
    - If a target total number of selected charges is enforced, legal actions
      are further restricted to only add or only remove until the target is met.
    """
    
    def __init__(self, 
                 molecular_data: Dict[str, Any],
                 esp_target: jnp.ndarray,
                 vdw_surface: jnp.ndarray,
                 model_charges: Dict[int, jnp.ndarray],
                 model_positions: Dict[int, jnp.ndarray]):
        """
        Initialize the DCMNET charge selection environment.
        
        Args:
            molecular_data: Dictionary with at least ``atomic_numbers`` (n_atoms,).
            esp_target: Target ESP (shape (N,) or (1, N)).
            vdw_surface: Surface points (shape (N, 3) or (1, N, 3)).
            model_charges: Dict[int, array] or array. Charges per model with
                shapes normalized to (n_atoms, n_charges).
            model_positions: Dict[int, array] or array. Positions per model with
                shapes normalized to (n_atoms, n_charges, 3).
        """
        self.molecular_data = molecular_data
        # Normalize targets/surfaces to 1D and (N,3) respectively
        esp_target = jnp.asarray(esp_target)
        vdw_surface = jnp.asarray(vdw_surface)
        if esp_target.ndim == 2 and esp_target.shape[0] == 1:
            esp_target = esp_target[0]
        if vdw_surface.ndim == 3 and vdw_surface.shape[0] == 1:
            vdw_surface = vdw_surface[0]
        if vdw_surface.ndim != 2 or vdw_surface.shape[-1] != 3:
            raise ValueError(f"vdw_surface must have shape (N,3); got {vdw_surface.shape}")
        self.esp_target = esp_target
        self.vdw_surface = vdw_surface
        # Normalize model_charges to shape (n_atoms, n_charges_per_model)
        self.model_charges = {}
        self.model_positions = {}
        for model_id in model_charges.keys():
            charges = np.array(model_charges[model_id])
            # Accept shapes: (n_atoms,), (n_atoms, n), (n_atoms, n, 1)
            if charges.ndim == 1:
                charges = charges[:, None]
            elif charges.ndim == 3 and charges.shape[-1] == 1:
                charges = charges.squeeze(-1)
            elif charges.ndim != 2:
                raise ValueError(f"Charges for model {model_id} must be 1D, 2D, or 3D with last dim 1; got shape {charges.shape}")
            self.model_charges[model_id] = jnp.asarray(charges)

            positions = np.array(model_positions[model_id])
            # Expect positions shape (n_atoms, n_charges, 3)
            if positions.ndim != 3 or positions.shape[-1] != 3:
                raise ValueError(f"Positions for model {model_id} must have shape (n_atoms, n_charges, 3); got {positions.shape}")
            self.model_positions[model_id] = jnp.asarray(positions)
        
        # Determine valid (non-ghost) atoms and keep an index map into original arrays
        atom_Z = np.asarray(molecular_data['atomic_numbers'])
        valid_mask = atom_Z > 0
        self.atom_index_map = np.nonzero(valid_mask)[0].astype(int)
        # Ensure robust int count even if input is JAX array
        self.n_atoms = int(self.atom_index_map.shape[0])  # ignore ghost atoms
        self.model_ids = list(model_charges.keys())
        self.n_models = len(self.model_ids)
        # Hydrogen mask in valid-atom indexing space
        Z_valid = np.asarray(atom_Z[self.atom_index_map])
        self.hydrogen_mask_atoms = (Z_valid == 1).astype(np.int8)  # (n_atoms,)
        
        # Calculate total number of charges per atom across all models
        self.n_charges_per_model = {}
        for model_id, charges in self.model_charges.items():
            self.n_charges_per_model[model_id] = int(charges.shape[1]) if charges.ndim > 1 else 1
        
        self.total_charges_per_atom = sum(self.n_charges_per_model.values())
        # Static capacity per atom for padded computations
        self.max_charges_capacity = max(int(self.total_charges_per_atom), 60)
        
        # Initialize state: binary matrix (n_atoms, total_charges_per_atom)
        # Each row represents one atom, each column represents one charge from one model
        self.selected_charges = np.zeros((self.n_atoms, self.total_charges_per_atom), dtype=np.int8)
        # Initialize to monopole solution: select one charge per atom (first global charge index)
        if self.total_charges_per_atom > 0 and self.n_atoms > 0:
            self.selected_charges[:, 0] = 1
        self.step_count = 0
        self.max_steps = self.n_atoms * self.total_charges_per_atom  # Allow toggling each charge once
        # Optional target enforcement and averaging overrides
        self.enforce_target = False
        self.target_total_selected: Optional[int] = None
        # Optional mode: actions swap entire model selection for a single atom (default ON)
        self.swap_per_atom_models: bool = True
        self.overridden_charge_values: Dict[Tuple[int, int], float] = {}
        self.overridden_charge_positions: Dict[Tuple[int, int], jnp.ndarray] = {}
        # Neutrality penalty weight (can be set by optimizer)
        self.neutrality_lambda: float = 0.0
        # Excess charge penalty weight (can be set by optimizer)
        self.excess_charge_penalty: float = 0.0
        # Optional bounds to restrict total selected charges during search
        self.total_min_bound: Optional[int] = None
        self.total_max_bound: Optional[int] = None
        # Diagnostics for NN features
        self.last_grad_stats: Tuple[float, float] = (0.0, 0.0)
        
        # Create mapping from (atom_idx, charge_idx) to model and charge within model
        self._create_charge_mapping()
        
        # Precompute candidate positions/values and kernel for fixed-shape, fast ESP
        # candidate_positions: (n_atoms, total_charges_per_atom, 3)
        candidate_positions = np.zeros((self.n_atoms, self.total_charges_per_atom, 3), dtype=float)
        candidate_values = np.zeros((self.n_atoms, self.total_charges_per_atom), dtype=float)
        for charge_idx in range(self.total_charges_per_atom):
            model_id, charge_within_model = self.charge_mapping[charge_idx]
            # Subset source arrays to valid atoms via atom_index_map to match n_atoms
            src_pos = np.asarray(self.model_positions[model_id])  # (A, n_c, 3)
            src_val = np.asarray(self.model_charges[model_id])    # (A, n_c)
            candidate_positions[:, charge_idx, :] = src_pos[self.atom_index_map, charge_within_model, :]
            candidate_values[:, charge_idx] = src_val[self.atom_index_map, charge_within_model]
        self.candidate_positions = candidate_positions  # (n_atoms, C, 3)
        self.candidate_values = candidate_values        # (n_atoms, C)
        # Build allowed charge mask per atom: for H atoms only allow monopole (charge_within_model==0), else allow all
        allowed_mask = np.ones((self.n_atoms, self.total_charges_per_atom), dtype=np.int8)
        for g in range(self.total_charges_per_atom):
            _, cwm = self.charge_mapping[g]
            if int(cwm) != 0:
                # disallow non-monopole for H atoms
                allowed_mask[self.hydrogen_mask_atoms.astype(bool), g] = 0
        self.allowed_charge_mask = allowed_mask
        
        # Precompute kernel K from surface points to candidate positions
        positions_flat = candidate_positions.reshape(-1, 3)  # (L,3), L=n_atoms*C
        sp = np.asarray(self.vdw_surface, dtype=float)       # (Ns,3)
        d = np.linalg.norm(sp[:, None, :] - positions_flat[None, :, :], axis=2)  # (Ns,L)
        d[d < 1e-6] = 1e-6
        self.K_surface_to_candidates = 1.0 / (d * CONVERSION_FACTOR)               # (Ns,L)
        # Precompute Jacobian J blocks for first-order displacement correction
        # For 1/r, grad w.r.t. charge position r_i: ∂(1/||x - r_i||)/∂r_i = (r_i - x) / ||x - r_i||^3
        diff = positions_flat[None, :, :] - sp[:, None, :]  # (Ns, L, 3)
        r = np.linalg.norm(diff, axis=2)                    # (Ns, L)
        r3 = np.maximum(r * r * r, 1e-12)
        # Flip sign to compute gradient w.r.t. surface point x in Å-units consistent with K (which is 1/(r*CF))
        # d(1/(r*CF))/dx = (1/CF) * (x - r) / r^3
        J_np = (-(diff) / (r3[..., None] * CONVERSION_FACTOR)).reshape(sp.shape[0], -1)  # (Ns, 3L)
        self.J_surface_to_candidates = J_np
        # Prepare fast loss path (JAX JIT, static shapes)
        K_jax = jnp.asarray(self.K_surface_to_candidates, dtype=jnp.float32)
        target_jax = jnp.asarray(self.esp_target, dtype=jnp.float32)

        K = jnp.asarray(self.K_surface_to_candidates, dtype=jnp.float32)
        J = jnp.asarray(self.J_surface_to_candidates, dtype=jnp.float32)
        tgt = jnp.asarray(self.esp_target, dtype=jnp.float32)
        # Precompute mask to zero displacement corrections on hydrogen atoms (size 3L)
        h_mask_flat = jnp.asarray(np.repeat(self.hydrogen_mask_atoms.astype(np.float32), self.total_charges_per_atom))
        dr_mask = 1.0 - jnp.repeat(h_mask_flat, 3)  # 0 for H entries

        @jax.jit
        def _fast_loss(s_flat: jnp.ndarray, v_flat: jnp.ndarray, lam: jnp.ndarray,
                       dq: jnp.ndarray, dr: jnp.ndarray,
                       dq_max: jnp.ndarray, dr_max: jnp.ndarray,
                       excess_penalty: jnp.ndarray, target_total: jnp.ndarray,
                       dr_mask: jnp.ndarray) -> jnp.ndarray:
            q_flat = s_flat * v_flat
            # Apply small NN deltas (clipped), project neutrality
            dq_c = jnp.clip(dq, -dq_max, dq_max)
            dr_c = jnp.clip(dr, -dr_max, dr_max)
            # Enforce no displacement corrections on hydrogens
            dr_c = dr_c * dr_mask
            q_hat = (q_flat + dq_c) * s_flat
            # Neutrality projection without boolean indexing
            sel_sum = jnp.sum(s_flat)
            mean_q = jnp.where(sel_sum > 0.0, jnp.sum(q_hat * s_flat) / sel_sum, 0.0)
            q_hat = q_hat - mean_q * s_flat
            esp0 = K @ q_hat
            q_rep = jnp.repeat(q_hat, 3)
            esp_corr = J @ (dr_c * q_rep)
            esp_pred = esp0 + esp_corr
            esp_pred = esp_pred / 2.0
            err = esp_pred - tgt
            mse = jnp.mean(err * err)
            # neutrality term uses raw selected charges (before projection)
            neutrality = jnp.sum(q_flat)
            # excess charge penalty: penalize charges beyond target (use linear penalty to avoid huge jumps)
            total_selected = jnp.sum(s_flat)
            excess = jnp.maximum(0.0, total_selected - target_total)
            excess_pen = excess_penalty * excess  # Linear instead of quadratic
            return mse + lam * (neutrality * neutrality) + excess_pen

        self._fast_loss_fn = _fast_loss
    
    def _create_charge_mapping(self):
        """Create mapping from global charge index to (model_id, charge_within_model)."""
        self.charge_mapping = {}
        global_idx = 0
        
        for model_id in self.model_ids:
            n_charges = self.n_charges_per_model[model_id]
            for charge_idx in range(n_charges):
                self.charge_mapping[global_idx] = (model_id, charge_idx)
                global_idx += 1
    
    def clone(self) -> "DCMNETSelectionEnv":
        """Create a lightweight copy without recomputing heavy precomputations.

        Avoids re-running __init__ so we don't rebuild candidate arrays and
        distance kernels on every simulation step.
        """
        new_env = DCMNETSelectionEnv.__new__(DCMNETSelectionEnv)
        # Shallow-copy immutable/heavy fields
        new_env.molecular_data = self.molecular_data
        new_env.esp_target = self.esp_target
        new_env.vdw_surface = self.vdw_surface
        new_env.model_charges = self.model_charges
        new_env.model_positions = self.model_positions
        new_env.atom_index_map = self.atom_index_map
        new_env.n_atoms = self.n_atoms
        new_env.model_ids = self.model_ids
        new_env.n_models = self.n_models
        new_env.n_charges_per_model = self.n_charges_per_model
        new_env.total_charges_per_atom = self.total_charges_per_atom
        new_env.max_charges_capacity = self.max_charges_capacity
        new_env.max_steps = self.max_steps
        new_env.charge_mapping = self.charge_mapping
        new_env.candidate_positions = self.candidate_positions
        new_env.candidate_values = self.candidate_values
        new_env.K_surface_to_candidates = self.K_surface_to_candidates
        new_env.J_surface_to_candidates = self.J_surface_to_candidates
        new_env._fast_loss_fn = self._fast_loss_fn
        new_env.allowed_charge_mask = self.allowed_charge_mask
        new_env.hydrogen_mask_atoms = self.hydrogen_mask_atoms
        new_env.neutrality_lambda = float(self.neutrality_lambda)
        new_env.excess_charge_penalty = float(self.excess_charge_penalty)
        new_env.swap_per_atom_models = bool(self.swap_per_atom_models)
        new_env.total_min_bound = (None if self.total_min_bound is None else int(self.total_min_bound))
        new_env.total_max_bound = (None if self.total_max_bound is None else int(self.total_max_bound))
        # Defaults for NN corrections
        new_env.dq_max = getattr(self, 'dq_max', 0.05)
        new_env.dr_max = getattr(self, 'dr_max', 0.1 / CONVERSION_FACTOR)
        new_env.last_grad_stats = tuple(self.last_grad_stats)
        # Deep/independent state
        new_env.selected_charges = np.array(self.selected_charges).copy()
        new_env.step_count = int(self.step_count)
        new_env.enforce_target = bool(self.enforce_target)
        new_env.target_total_selected = (None if self.target_total_selected is None
                                         else int(self.target_total_selected))
        new_env.overridden_charge_values = dict(self.overridden_charge_values)
        new_env.overridden_charge_positions = dict(self.overridden_charge_positions)
        return new_env
    
    def legal_actions(self) -> List[Tuple]:
        """Return available actions that can be applied.

        Restricts to constructive actions until every atom has ≥1 selected charge;
        optionally restricts to add-only or remove-only based on an enforced target
        total number of selections. Also exposes averaging actions per atom
        ("avg_q", atom_idx) and ("avg_vec", atom_idx) when the atom has ≥1 selection.
        """
        actions: List[Tuple] = []
        # If swap-per-atom-models mode is active, expose only per-atom model swap actions
        if getattr(self, 'swap_per_atom_models', False):
            # Actions are ("swap_model", atom_idx, model_id)
            for atom_idx in range(self.n_atoms):
                for mid in self.model_ids:
                    # If bounds set, skip actions that would violate total selection bounds
                    if self.total_min_bound is not None or self.total_max_bound is not None:
                        cur_total = int(np.sum(self.selected_charges))
                        # Predict resulting total for this atom after swap
                        # Count how many charges of this model are allowed for this atom
                        add_count = 0
                        for g in range(self.total_charges_per_atom):
                            m2, _cwm = self.charge_mapping[g]
                            if int(m2) == int(mid) and self.allowed_charge_mask[atom_idx, g]:
                                add_count += 1
                        # Current selected count on this atom
                        cur_atom = int(np.sum(self.selected_charges[atom_idx]))
                        new_total = cur_total - cur_atom + add_count
                        if self.total_min_bound is not None and new_total < int(self.total_min_bound):
                            continue
                        if self.total_max_bound is not None and new_total > int(self.total_max_bound):
                            continue
                    actions.append(("swap_model", int(atom_idx), int(mid)))
            return actions
        # If per-atom coverage is required (default when not enforcing a total target),
        # restrict actions to only selecting for atoms with no selection yet.
        charges_per_atom = np.sum(self.selected_charges, axis=1)
        if not self.enforce_target and np.any(charges_per_atom == 0):
            atoms_missing = np.where(charges_per_atom == 0)[0]
            for atom_idx in atoms_missing:
                for charge_idx in range(self.total_charges_per_atom):
                    # Enforce allowed mask (e.g., hydrogens monopole only)
                    if self.allowed_charge_mask[atom_idx, charge_idx] and self.selected_charges[atom_idx, charge_idx] == 0:
                        actions.append((atom_idx, charge_idx))
            return actions
        # Averaging actions per atom when at least one charge is selected
        # (removed - no longer wanted)
        # If enforcing a target total selection, restrict to select-only or deselect-only
        if self.enforce_target and self.target_total_selected is not None:
            total_selected = int(np.sum(self.selected_charges))
            if total_selected < int(self.target_total_selected):
                for atom_idx in range(self.n_atoms):
                    for charge_idx in range(self.total_charges_per_atom):
                        if self.selected_charges[atom_idx, charge_idx] == 0:
                            actions.append((atom_idx, charge_idx))
                return actions
            elif total_selected > int(self.target_total_selected):
                for atom_idx in range(self.n_atoms):
                    for charge_idx in range(self.total_charges_per_atom):
                        if self.selected_charges[atom_idx, charge_idx] == 1:
                            # prevent removing the last remaining charge on this atom
                            if int(np.sum(self.selected_charges[atom_idx])) > 1:
                                actions.append((atom_idx, charge_idx))
                return actions
        # Otherwise, allow toggling any charge
        for atom_idx in range(self.n_atoms):
            for charge_idx in range(self.total_charges_per_atom):
                if self.allowed_charge_mask[atom_idx, charge_idx]:
                    if self.total_min_bound is not None or self.total_max_bound is not None:
                        cur_total = int(np.sum(self.selected_charges))
                        cur_val = int(self.selected_charges[atom_idx, charge_idx])
                        new_total = cur_total + (1 - 2*cur_val)  # toggle effect
                        if self.total_min_bound is not None and new_total < int(self.total_min_bound):
                            continue
                        if self.total_max_bound is not None and new_total > int(self.total_max_bound):
                            continue
                actions.append((atom_idx, charge_idx))
        return actions
    
    def step(self, action: Tuple) -> "DCMNETSelectionEnv":
        """Apply an action.

        Supported actions:
          - (atom_idx: int, charge_idx: int): toggle selection for that charge
          - ("swap_model", atom_idx, model_id): replace this atom's selections with all charges from model_id
        """
        new_env = self.clone()
        # Swap model action
        if isinstance(action, tuple) and len(action) == 3 and action[0] == "swap_model":
            _, atom_idx, model_id = action
            atom_idx = int(atom_idx)
            if atom_idx < 0 or atom_idx >= self.n_atoms:
                raise ValueError(f"Invalid atom index in action: {action}")
            if model_id not in self.model_ids:
                raise ValueError(f"Unknown model_id in action: {action}")
            # Clear this atom's selections
            new_env.selected_charges[atom_idx, :] = 0
            # Turn on all charges corresponding to model_id for this atom
            # Identify global indices for this model's charges, filtered by allowed mask (H atoms allow only monopole)
            gidxs = []
            for g in range(self.total_charges_per_atom):
                mid, cwm = self.charge_mapping[g]
                if int(mid) == int(model_id):
                    if self.allowed_charge_mask[atom_idx, g]:
                        gidxs.append(g)
            for g in gidxs:
                new_env.selected_charges[atom_idx, int(g)] = 1
            new_env.step_count += 1
            return new_env
        # Toggle action
        if not (isinstance(action, tuple) and len(action) == 2 and all(isinstance(x, (int, np.integer)) for x in action)):
            raise ValueError(f"Invalid action: {action}")
        atom_idx, charge_idx = int(action[0]), int(action[1])
        if (atom_idx < 0 or atom_idx >= self.n_atoms or 
            charge_idx < 0 or charge_idx >= self.total_charges_per_atom):
            raise ValueError(f"Invalid action: {action}")
        # Enforce hydrogen allowed mask: ignore toggles to disallowed charges
        if not self.allowed_charge_mask[atom_idx, charge_idx]:
            return new_env
        # Support both NumPy (mutable) and JAX (immutable) arrays
        current_val = int(new_env.selected_charges[atom_idx, charge_idx])
        # Safeguard: never remove last remaining charge on this atom
        if current_val == 1 and int(np.sum(new_env.selected_charges[atom_idx])) <= 1:
            return new_env
        new_val = 1 - current_val
        if isinstance(new_env.selected_charges, jnp.ndarray):
            new_env.selected_charges = new_env.selected_charges.at[atom_idx, charge_idx].set(new_val)
        else:
            new_env.selected_charges[atom_idx, charge_idx] = new_val
        if new_env.selected_charges[atom_idx, charge_idx] == 0:
            new_env.overridden_charge_values.pop((atom_idx, charge_idx), None)
            new_env.overridden_charge_positions.pop((atom_idx, charge_idx), None)
        new_env.step_count += 1
        return new_env
    
    def is_terminal(self) -> bool:
        """Check if we've reached maximum steps or have at least one charge selected per atom."""
        if self.step_count >= self.max_steps:
            return True
        if self.enforce_target and self.target_total_selected is not None:
            total_selected = int(np.sum(self.selected_charges))
            return total_selected == int(self.target_total_selected)
        # Default criterion: per-atom coverage
        return np.all(np.sum(self.selected_charges, axis=1) > 0)
    
    def get_esp_loss(self) -> float:
        """
        Calculate ESP loss for current charge selection.
        
        Returns:
            ESP loss value (lower is better)
        """
        # Check if each atom has at least one charge selected
        charges_per_atom = np.sum(self.selected_charges, axis=1)
        # if np.any(charges_per_atom == 0):
        #     return float('inf')  # Some atoms have no charges
        
        # Fast path: use precomputed kernel when no position overrides are active
        try:
            if len(self.overridden_charge_positions) == 0:
                # Flatten selection and values to match K; apply value overrides
                s_flat = np.asarray(self.selected_charges, dtype=np.float32).reshape(-1)
                v_flat = self.candidate_values.reshape(-1).astype(np.float32)
                if self.overridden_charge_values:
                    for (atom_idx, charge_idx), override_val in self.overridden_charge_values.items():
                        flat_idx = int(atom_idx) * int(self.total_charges_per_atom) + int(charge_idx)
                        v_flat[flat_idx] = np.float32(override_val)
                lam = np.float32(self.neutrality_lambda)
                # Provide NN deltas if present, else zeros
                dq = getattr(self, 'nn_dq', np.zeros_like(v_flat)).astype(np.float32)
                dr = getattr(self, 'nn_dr', np.zeros((v_flat.shape[0]*3,), dtype=np.float32)).astype(np.float32)
                dq_max = np.float32(0.05)
                dr_max = np.float32(0.1 / CONVERSION_FACTOR)
                excess_penalty = np.float32(getattr(self, 'excess_charge_penalty', 0.0))
                # If no target is set OR we're in swap-per-atom mode, disable excess penalty
                target_total_raw = getattr(self, 'target_total_selected', None)
                swap_mode = getattr(self, 'swap_per_atom_models', False)
                if target_total_raw is None or swap_mode:
                    target_total = np.float32(1e6)  # Effectively disable excess penalty
                else:
                    target_total = np.float32(target_total_raw)
                # Create dr_mask for hydrogen constraint
                h_mask_flat = np.repeat(self.hydrogen_mask_atoms.astype(np.float32), self.total_charges_per_atom)
                dr_mask = 1.0 - np.repeat(h_mask_flat, 3)  # 0 for H entries
                loss_val = self._fast_loss_fn(jnp.asarray(s_flat), jnp.asarray(v_flat), jnp.asarray(lam),
                                              jnp.asarray(dq), jnp.asarray(dr),
                                              jnp.asarray(dq_max), jnp.asarray(dr_max),
                                              jnp.asarray(excess_penalty), jnp.asarray(target_total),
                                              jnp.asarray(dr_mask))
                return float(loss_val)
        except Exception as e:
            print(f"Error in fast path: {e}")
        
        # Slow path: fall back to dynamic positions if any position overrides exist
        selected_charge_values = []
        selected_charge_positions = []
        for atom_idx in range(self.n_atoms):
            for charge_idx in range(self.total_charges_per_atom):
                if self.selected_charges[atom_idx, charge_idx]:
                    override_key = (atom_idx, charge_idx)
                    base_val = float(self.candidate_values[atom_idx, charge_idx])
                    base_pos = np.asarray(self.candidate_positions[atom_idx, charge_idx, :], dtype=float)
                    charge_value = float(self.overridden_charge_values.get(override_key, base_val))
                    if override_key in self.overridden_charge_positions:
                        charge_position = np.asarray(self.overridden_charge_positions[override_key], dtype=float)
                    else:
                        charge_position = base_pos
                    selected_charge_values.append(charge_value)
                    selected_charge_positions.append(charge_position)
        if not selected_charge_values:
            return float('inf')
        vals = np.asarray(selected_charge_values, dtype=float)
        poss = np.asarray(selected_charge_positions, dtype=float)
        sp = np.asarray(self.vdw_surface, dtype=float)
        d = np.linalg.norm(sp[:, None, :] - poss[None, :, :], axis=2)
        d[d < 1e-6] = 1e-6
        d = d * CONVERSION_FACTOR
        esp_pred = (vals[None, :] / d).sum(axis=1)
        tgt = np.asarray(self.esp_target, dtype=float)
        loss = ((esp_pred - tgt) ** 2).mean()
        if self.neutrality_lambda > 0.0:
            neutrality = float(np.sum(vals))
            loss += self.neutrality_lambda * (neutrality ** 2)
            return float(loss)
    
    def _calculate_esp_from_distributed_charges(self, 
                                              charge_values: jnp.ndarray, 
                                              charge_positions: jnp.ndarray, 
                                              surface_points: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate ESP at surface points from distributed charges.
        
        Args:
            charge_values: Array of charge values (N_charges,)
            charge_positions: Array of charge positions (N_charges, 3)
            surface_points: Array of surface points (N_surface, 3)
            
        Returns:
            ESP values at surface points (N_surface,)
        """
        # Calculate distances between charges and surface points
        # charge_positions: (N_charges, 3), surface_points: (N_surface, 3)
        # We want distances: (N_surface, N_charges)
        
        # Expand dimensions for broadcasting
        charges_expanded = charge_positions[None, :, :]  # (1, N_charges, 3)
        surface_expanded = surface_points[:, None, :]    # (N_surface, 1, 3)
        
        # Calculate distances
        distances = jnp.linalg.norm(surface_expanded - charges_expanded, axis=2)  # (N_surface, N_charges)
        
        # Avoid division by zero
        distances = jnp.where(distances < 1e-6, 1e-6, distances)
        
        # Calculate ESP: sum(q_i / r_i) for each surface point
        esp_values = jnp.sum(charge_values[None, :] / distances, axis=1)  # (N_surface,)
        
        return esp_values
    
    def result_from(self, player: int) -> float:
        """
        Get the result (negative ESP loss) from the current player's perspective.
        Since we want to minimize loss, we return negative loss as the "value".
        """
        if not self.is_terminal():
            raise ValueError("Environment not terminal")
        
        esp_loss = self.get_esp_loss()
        # Return negative loss as value (lower loss = higher value)
        return -esp_loss if esp_loss != float('inf') else -1e6
    
    def __repr__(self) -> str:
        """String representation of the environment."""
        charges_per_atom = np.sum(self.selected_charges, axis=1)
        total_selected = np.sum(self.selected_charges)
        return f"Selected charges: {total_selected}/{self.n_atoms * self.total_charges_per_atom}, " \
               f"Charges per atom: {charges_per_atom.tolist()}, ESP Loss: {self.get_esp_loss():.6f}"

# =========================================================
# MCTS Node and Search for DCMNET
# =========================================================

@dataclass
class DCMNETNode:
    """MCTS node for DCMNET model selection."""
    key: Tuple
    parent: Optional["DCMNETNode"] = None
    parent_action: Optional[Tuple[int, int]] = None
    
    children: Dict[Tuple[int, int], "DCMNETNode"] = field(default_factory=dict)
    P: Dict[Tuple[int, int], float] = field(default_factory=dict)  # action -> prior prob
    
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    
    def expanded(self) -> bool:
        return len(self.P) > 0

def dcmnet_state_to_key(env: DCMNETSelectionEnv) -> Tuple:
    """Convert environment state to hashable key."""
    return tuple(env.selected_charges.flatten().tolist())

class DCMNET_MCTS:
    """
    MCTS for DCMNET charge selection optimization.
    
    Uses PUCT algorithm to explore combinations of individual charges
    from different DCMNET models and find the combination that minimizes ESP loss.
    """
    
    def __init__(self, 
                 policy_value_fn,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 root_noise_frac: float = 0.25,
                 rng: Optional[jax.random.PRNGKey] = None):
        self.policy_value_fn = policy_value_fn
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.root_noise_frac = float(root_noise_frac)
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.last_root = None
    
    def search(self, 
               root_env: DCMNETSelectionEnv, 
               n_simulations: int = 400, 
               temperature: float = 1.0) -> Tuple[int, int]:
        """Perform MCTS search and return best action."""
        root = DCMNETNode(key=dcmnet_state_to_key(root_env))
        self._expand(root_env, root)
        self._add_root_dirichlet_noise(root)
        
        for _ in range(n_simulations):
            self._simulate(root_env, root)
        
        self.last_root = root
        return self._select_action_from_visits(root, temperature)
    
    def _simulate(self, root_env: DCMNETSelectionEnv, root: DCMNETNode):
        """Perform one MCTS simulation."""
        node = root
        env = root_env.clone()
        path = []
        
        while True:
            if env.is_terminal():
                value = env.result_from(1)  # Always from maximizing player's POV
                self._backup(path, leaf_value=value)
                return
            
            a = self._puct_select(node)
            path.append((node, a))
            
            if a not in node.children:
                env = env.step(a)
                child = DCMNETNode(
                    key=dcmnet_state_to_key(env),
                    parent=node,
                    parent_action=a
                )
                priors, leaf_val = self.policy_value_fn(env)
                child.P = priors
                node.children[a] = child
                self._backup(path, leaf_value=leaf_val)
                return
            
            env = env.step(a)
            node = node.children[a]
    
    def _puct_select(self, node: DCMNETNode) -> Tuple[int, int]:
        """Select action using PUCT formula."""
        sum_N = max(1, sum(child.N for child in node.children.values()))
        best_a, best_score = None, -1e18
        
        for a, p in node.P.items():
            child = node.children.get(a)
            Nsa = 0 if child is None else child.N
            Qsa = 0.0 if child is None else child.Q
            u = self.c_puct * p * np.sqrt(sum_N) / (1.0 + Nsa)
            score = Qsa + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a
    
    def _expand(self, env: DCMNETSelectionEnv, node: DCMNETNode):
        """Expand node with policy priors."""
        priors, _ = self.policy_value_fn(env)
        node.P = priors
    
    def _add_root_dirichlet_noise(self, root: DCMNETNode):
        """Add Dirichlet noise to root node for exploration."""
        if not root.P:
            return
        actions = list(root.P.keys())
        alpha = self.dirichlet_alpha
        # Sample Dirichlet noise using JAX PRNG
        self.rng, subkey = jax.random.split(self.rng)
        concentration = jnp.full((len(actions),), float(alpha))
        noise = np.array(jax.random.dirichlet(subkey, concentration))
        for a, eps in zip(actions, noise):
            root.P[a] = (1 - self.root_noise_frac) * root.P[a] + self.root_noise_frac * float(eps)
    
    def _backup(self, path, leaf_value: float):
        """Backup values up the tree."""
        v = leaf_value
        for node, _ in reversed(path):
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v  # Alternate between players
    
    def _select_action_from_visits(self, root: DCMNETNode, temperature: float) -> Tuple[int, int]:
        """Select action based on visit counts."""
        actions = list(root.P.keys())
        visits = np.array([root.children[a].N if a in root.children else 0 for a in actions], dtype=float)
        
        if temperature <= 1e-8:
            return actions[int(np.argmax(visits))]
        
        # JAX numpy does not support numpy.errstate; compute directly.
        if len(actions) == 0:
            raise ValueError("No actions available at root for selection")
        pi = np.power(visits, 1.0 / max(temperature, 1e-8))
        if np.all(pi == 0):
            pi = np.ones_like(pi)
        pi = pi / np.sum(pi)
        # Sample index using JAX PRNG
        self.rng, subkey = jax.random.split(self.rng)
        chosen_index = int(jax.random.choice(subkey, jnp.arange(len(actions)), p=jnp.asarray(pi)))
        return actions[chosen_index]

# =========================================================
# Neural Network for DCMNET Model Selection
# =========================================================

def make_dcmnet_features(env: DCMNETSelectionEnv) -> jnp.ndarray:
    """
    Create feature representation for DCMNET charge selection environment.
    
    Features: (n_atoms * total_charges_per_atom) + step + esp_loss_estimate
    
    Args:
        env: DCMNETSelectionEnv instance
        
    Returns:
        Feature vector representing current state
    """
    # Flatten the charge selection matrix
    charge_features = env.selected_charges.flatten().astype(np.float32)
    
    # Step count normalized by max steps
    step_feature = np.array([env.step_count / env.max_steps], dtype=np.float32)
    
    # ESP loss estimate (normalized)
    esp_loss_estimate = env.get_esp_loss()
    esp_feature = np.array([min(esp_loss_estimate / 1000.0, 1.0)], dtype=np.float32)
    
    # Combine features
    features = np.concatenate([charge_features, step_feature, esp_feature])
    return jnp.asarray(features)

class DCMNETSelectionNet:
    """Neural network for DCMNET charge selection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_actions: int = None):
        """
        Initialize the network.
        
        Args:
            input_dim: Input feature dimension (n_atoms * total_charges + step + esp)
            hidden_dim: Hidden layer dimension
            n_actions: Number of possible actions (atom_idx, charge_idx pairs)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
    
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Simple MLP
        h = jax.nn.relu(jax.nn.Dense(self.hidden_dim)(x))
        h = jax.nn.relu(jax.nn.Dense(self.hidden_dim)(h))
        
        # Policy head
        policy_logits = jax.nn.Dense(self.n_actions)(h)  # n_actions possible actions
        
        # Value head
        value = jax.nn.tanh(jax.nn.Dense(1)(h))
        return policy_logits, value[:, 0]

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def _init_selection_net_params(input_dim: int, hidden_dim: int, n_actions: int,
                               l_size: int, dr_size: int, n_atoms: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = {
        'W1': rng.normal(0, 0.02, (input_dim, hidden_dim)).astype(np.float32),
        'b1': np.zeros((hidden_dim,), dtype=np.float32),
        'W2': rng.normal(0, 0.02, (hidden_dim, hidden_dim)).astype(np.float32),
        'b2': np.zeros((hidden_dim,), dtype=np.float32),
        'Wp': rng.normal(0, 0.02, (hidden_dim, n_actions)).astype(np.float32),
        'bp': np.zeros((n_actions,), dtype=np.float32),
        'Wv': rng.normal(0, 0.02, (hidden_dim, 1)).astype(np.float32),
        'bv': np.zeros((1,), dtype=np.float32),
        'Wdq': rng.normal(0, 0.02, (hidden_dim, l_size)).astype(np.float32),
        'bdq': np.zeros((l_size,), dtype=np.float32),
        'Wdr': rng.normal(0, 0.02, (hidden_dim, dr_size)).astype(np.float32),
        'bdr': np.zeros((dr_size,), dtype=np.float32),
        'Wrefine': rng.normal(0, 0.02, (hidden_dim, n_atoms)).astype(np.float32),
        'brefine': np.zeros((n_atoms,), dtype=np.float32),
    }
    return params

def _selection_net_forward(params: Dict[str, np.ndarray], x: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    # x: (input_dim,)
    h1 = _relu(x @ params['W1'] + params['b1'])
    h2 = _relu(h1 @ params['W2'] + params['b2'])
    logits = h2 @ params['Wp'] + params['bp']
    value = float(np.tanh(float(h2 @ params['Wv'] + params['bv'])))
    dq = h2 @ params['Wdq'] + params['bdq']
    dr = h2 @ params['Wdr'] + params['bdr']
    refine_logits = h2 @ params['Wrefine'] + params['brefine']
    return logits, value, dq, dr, refine_logits

def _jax_forward(params, x):
    h1 = jax.nn.relu(x @ params['W1'] + params['b1'])
    h2 = jax.nn.relu(h1 @ params['W2'] + params['b2'])
    logits = h2 @ params['Wp'] + params['bp']
    value = jnp.tanh((h2 @ params['Wv'] + params['bv'])[0])
    dq = h2 @ params['Wdq'] + params['bdq']
    dr = h2 @ params['Wdr'] + params['bdr']
    refine_logits = h2 @ params['Wrefine'] + params['brefine']
    return logits, value, dq, dr, refine_logits

def _params_to_jax(params_np: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in params_np.items()}

def _apply_sgd_inplace(params_np: Dict[str, np.ndarray], grads: Dict[str, jnp.ndarray], lr: float) -> None:
    for k in params_np.keys():
        params_np[k] -= lr * np.asarray(grads[k])

def _train_selection_net_step(params_np: Dict[str, np.ndarray],
                              x_np: np.ndarray,
                              legal_indices_np: np.ndarray,
                              target_pi_np: np.ndarray,
                              target_value: float,
                              # for end-to-end loss
                              K: jnp.ndarray,
                              J: jnp.ndarray,
                              s_flat: np.ndarray,
                              v_flat: np.ndarray,
                              lam: float,
                              dq_max: float,
                              dr_max: float,
                              lr: float = 1e-3,
                              alpha_e2e: float = 1e-3) -> None:
    params_j = _params_to_jax(params_np)
    x = jnp.asarray(x_np)
    legal_idx = jnp.asarray(legal_indices_np, dtype=jnp.int32)
    target_pi = jnp.asarray(target_pi_np, dtype=jnp.float32)
    target_v = jnp.asarray(target_value, dtype=jnp.float32)
    s_j = jnp.asarray(s_flat, dtype=jnp.float32)
    v_j = jnp.asarray(v_flat, dtype=jnp.float32)
    lam_j = jnp.asarray(lam, dtype=jnp.float32)
    dq_m = jnp.asarray(dq_max, dtype=jnp.float32)
    dr_m = jnp.asarray(dr_max, dtype=jnp.float32)

    def loss_fn(p):
        logits, v, dq_pred, dr_pred, refine_logits = _jax_forward(p, x)
        # Policy loss on legal indices
        legal_logits = logits[legal_idx]
        log_probs = legal_logits - jax.scipy.special.logsumexp(legal_logits)
        policy_loss = -jnp.sum(target_pi * log_probs)
        value_loss = jnp.mean((v - target_v) ** 2)
        # End-to-end ESP loss via first-order displacement and charge deltas
        dq_clipped = jnp.clip(dq_pred, -dq_m, dq_m)
        dr_clipped = jnp.clip(dr_pred, -dr_m, dr_m)
        q_hat = (v_j + dq_clipped) * s_j
        # Project neutrality
        # Neutrality projection without boolean indexing
        sel_sum = jnp.sum(s_j)
        mean_q = jnp.where(sel_sum > 0.0, jnp.sum(q_hat * s_j) / sel_sum, 0.0)
        q_hat = q_hat - mean_q * s_j
        esp0 = K @ q_hat
        # Scale dr blocks by q_hat
        q_rep = jnp.repeat(q_hat, 3)
        dr_scaled = dr_clipped * q_rep
        esp_corr = J @ dr_scaled
        esp_pred = esp0 + esp_corr
        # Target for end-to-end term is y passed via closure target E (same as optimize target)
        # For training step, we recompute y from K and v_j to avoid closing large arrays again.
        # Here, just use K and s_j, v_j as current target approximation (zero residual term weight in policy-only mode)
        # Prefer passing the real target separately if desired.
        # For now, set mse against K @ (v_j * s_j) as a proxy target.
        y_proxy = K @ (v_j * s_j)
        mse = jnp.mean((esp_pred - y_proxy) ** 2)
        reg = jnp.mean(dq_clipped * dq_clipped) + jnp.mean(dr_clipped * dr_clipped)
        e2e = mse + lam_j * (jnp.sum(q_hat) ** 2) + 1e-4 * reg
        # Refine head loss (no target for now, just regularization)
        refine_reg = jnp.mean(refine_logits * refine_logits)
        return policy_loss + 0.5 * value_loss + alpha_e2e * e2e + 1e-5 * refine_reg

    grads = jax.grad(loss_fn)(params_j)
    _apply_sgd_inplace(params_np, grads, lr)

def make_dcmnet_policy_value_fn(model, params):
    """Create policy-value function for DCMNET MCTS."""
    
    def policy_value_fn(env: DCMNETSelectionEnv):
        """Policy-value function for DCMNET environment."""
        # Get legal actions
        legal_actions = env.legal_actions()
        
        # Create features
        features = make_dcmnet_features(env)[None, ...]  # Add batch dimension
        
        # Get model predictions
        policy_logits, value = model.apply(params, features)
        
        # Convert to numpy
        policy_logits = np.array(policy_logits[0])
        value = float(np.array(value[0]))
        
        # Create policy distribution over legal actions
        # For NN policy, we only have toggle logits; assign small fixed prior to non-toggle actions
        toggles = []
        toggle_indices = []
        for a in legal_actions:
            if isinstance(a, tuple) and len(a) == 2 and not isinstance(a[0], str):
                atom_idx, charge_idx = a
                toggles.append((int(atom_idx), int(charge_idx)))
                toggle_indices.append(int(atom_idx) * int(env.total_charges_per_atom) + int(charge_idx))
        priors = {}
        if toggle_indices:
            legal_logits = policy_logits[toggle_indices]
            exp_logits = np.exp(legal_logits - np.max(legal_logits))
            probs = exp_logits / np.sum(exp_logits)
            for i, (atom_idx, charge_idx) in enumerate(toggles):
                priors[(atom_idx, charge_idx)] = float(probs[i])
        # No averaging actions to assign priors to
        
        return priors, value
    
    return policy_value_fn

# =========================================================
# Training and Optimization Functions
# =========================================================

def optimize_dcmnet_combination(molecular_data: Dict[str, Any],
                              esp_target: jnp.ndarray,
                              vdw_surface: jnp.ndarray,
                              model_charges: Dict[int, jnp.ndarray],
                              model_positions: Dict[int, jnp.ndarray],
                              verbose: bool = True,
                              n_simulations: int = 1000,
                              temperature: float = 0.5,
                              target_total_selected: Optional[int] = None,
                              target_span: int = 2,
                              neutrality_lambda: float = 0.0,
                              dq_max: float = 0.15,
                              dr_max: float = 0.15,
                              refine_steps: int = 0,
                              refine_lr: float = 1e-4,
                              alpha_e2e: float = 0.0,
                              nn_policy_only: bool = False,
                              excess_charge_penalty: float = 0.0,
                              log_interval_steps: int = 1,
                              log_interval_seconds: float = 5.0,
                              accept_only_better: bool = False,
                              mcts_c_puct: Optional[float] = None,
                              mcts_root_noise_frac: Optional[float] = None,
                              mcts_dirichlet_alpha: Optional[float] = None,
                              selection_metric: str = "loss",
                              run_dir: Optional[str] = None) -> Tuple[np.ndarray, float, Dict[int, Tuple[np.ndarray, float]]]:
    """
    Use MCTS to find optimal combination of individual charges from DCMNET models.
    
    Args:
        molecular_data: Molecular data for ESP calculation
        esp_target: Target ESP values
        vdw_surface: VdW surface points
        model_charges: Dict mapping model_id -> charges array of shape (n_atoms, n_charges_per_model)
        model_positions: Dict mapping model_id -> positions array of shape (n_atoms, n_charges_per_model, 3)
        n_simulations: Number of MCTS simulations
        temperature: Temperature for action selection
        
    Returns:
        Tuple of (best_charge_selection_matrix, best_esp_loss, best_by_total)
    """
    
    # Global state for Ctrl+C handling
    global_best_result = None
    interrupted = False
    
    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        if verbose:
            print(f"\n[INTERRUPT] Ctrl+C detected. Saving best result found so far...")
    
    # Set up signal handler for graceful interruption
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    # Create base environment
    base_env = DCMNETSelectionEnv(molecular_data, esp_target, vdw_surface, model_charges, model_positions)
    base_env.neutrality_lambda = float(neutrality_lambda)
    base_env.excess_charge_penalty = float(excess_charge_penalty)
    base_env.dq_max = float(dq_max)
    base_env.dr_max = float(dr_max / CONVERSION_FACTOR)
    if verbose:
        if nn_policy_only:
            print("[OPT] NN policy-only mode: priors shaped by NN; δq/δr disabled")
        # derive initial esp_pred deterministically from current selection
        try:
            s0 = np.asarray(base_env.selected_charges, dtype=float).reshape(-1)
            v0 = base_env.candidate_values.reshape(-1).astype(float)
            esp_pred0 = base_env.K_surface_to_candidates @ (v0 * s0)
            rmses = float(np.sqrt(np.mean((np.asarray(base_env.esp_target, dtype=float) - esp_pred0) ** 2)))
            print(f"[OPT] initial (base) rmses={rmses}")
        except Exception:
            pass
        base_init_loss = float(base_env.get_esp_loss())
        base_total_sel = int(np.sum(base_env.selected_charges))
        if np.isfinite(base_init_loss):
            jax.debug.print("[OPT] initial (base) total_selected={t} loss={L}",
                            t=jnp.array(base_total_sel), L=jnp.array(base_init_loss, dtype=jnp.float32))
        else:
            print(f"[OPT] initial (base) total_selected={base_total_sel} loss={base_init_loss}")
    
    # Build NN-based policy-value function from this file's simple MLP
    # charges flattened + step + esp feature + grad stats (2)
    input_dim = int(base_env.selected_charges.size + 2 + 2)
    # Number of toggle actions (no averaging actions)
    n_toggle_actions = int(base_env.n_atoms * base_env.total_charges_per_atom)
    n_actions = n_toggle_actions
    L = int(base_env.n_atoms * base_env.total_charges_per_atom)
    # Try to load existing NN parameters if run_dir provided
    nn_params = None
    if run_dir is not None:
        try:
            import os
            os.makedirs(run_dir, exist_ok=True)
            nn_path = os.path.join(run_dir, "nn_params.npz")
            if os.path.exists(nn_path):
                if verbose:
                    print(f"[NN] Loading existing parameters from {nn_path}")
                nn_params = dict(np.load(nn_path))
        except Exception as e:
            if verbose:
                print(f"Warning: failed to load NN parameters: {e}")
    
    # Initialize NN parameters if not loaded
    if nn_params is None:
        nn_params = _init_selection_net_params(input_dim=input_dim, hidden_dim=64, n_actions=n_actions,
                                               l_size=L, dr_size=3*L, n_atoms=base_env.n_atoms, seed=0)
        if verbose and run_dir is not None:
            print(f"[NN] Initialized new parameters")
    # Simple in-memory replay buffer
    replay_X: List[np.ndarray] = []
    replay_idx: List[np.ndarray] = []
    replay_pi: List[np.ndarray] = []
    replay_v: List[float] = []

    def nn_policy_value_fn(env):
        legal_actions = env.legal_actions()
        # Build feature vector (same as make_dcmnet_features but in NumPy)
        charge_features = env.selected_charges.flatten().astype(np.float32)
        step_feature = np.array([env.step_count / env.max_steps], dtype=np.float32)
        # Diagnostics: cheap ESP estimate proxy and grad stats
        esp_loss_estimate = env.get_esp_loss()
        esp_feature = np.array([min(esp_loss_estimate / 1000.0, 1.0)], dtype=np.float32)
        grad_mean, grad_std = env.last_grad_stats if hasattr(env, 'last_grad_stats') else (0.0, 0.0)
        grad_feats = np.array([grad_mean, grad_std], dtype=np.float32)
        x = np.concatenate([charge_features, step_feature, esp_feature, grad_feats])  # (input_dim,)

        logits, value, dq_pred, dr_pred, refine_logits = _selection_net_forward(nn_params, x)
        # Drive MCTS by actual loss at the leaf: negative loss as value
        try:
            value = -float(env.get_esp_loss())
        except Exception:
            value = float(value)

        # Map legal actions to indices in [0, n_actions)
        action_to_index = {}
        # Toggle actions occupy first range; index = atom_idx * C + charge_idx
        C = int(env.total_charges_per_atom)
        for atom_idx in range(env.n_atoms):
            for charge_idx in range(C):
                action_to_index[(atom_idx, charge_idx)] = atom_idx * C + charge_idx
        # Swap-model actions: append after toggle actions
        swap_base = n_toggle_actions
        if getattr(env, 'swap_per_atom_models', False):
            offset = 0
            for atom_idx in range(env.n_atoms):
                for mid in env.model_ids:
                    action_to_index[("swap_model", atom_idx, int(mid))] = swap_base + offset
                    offset += 1

        if not legal_actions:
            return {}, float(value)

        # In swap-per-atom mode, NN logits don't have a dedicated head; use uniform priors
        if getattr(env, 'swap_per_atom_models', False):
            probs = np.full((len(legal_actions),), 1.0 / max(1, len(legal_actions)), dtype=np.float32)
        else:
            idxs = [action_to_index[a] for a in legal_actions]
            legal_logits = logits[idxs]
            exp_logits = np.exp(legal_logits - np.max(legal_logits))
            probs = exp_logits / np.sum(exp_logits)
        priors = {a: float(p) for a, p in zip(legal_actions, probs)}
        # Stash latest deltas and refine predictions into env
        if nn_policy_only:
            env.nn_dq = np.zeros_like(dq_pred, dtype=np.float32)
            env.nn_dr = np.zeros_like(dr_pred, dtype=np.float32)
            env.nn_refine_atoms = np.zeros_like(refine_logits, dtype=np.float32)
        else:
            env.nn_dq = dq_pred.astype(np.float32)
            env.nn_dr = dr_pred.astype(np.float32)
            env.nn_refine_atoms = refine_logits.astype(np.float32)
        return priors, float(value)
    
    # Initialize MCTS
    mcts = DCMNET_MCTS(
        policy_value_fn=nn_policy_value_fn,
        c_puct=float(mcts_c_puct) if mcts_c_puct is not None else 1.5,
        dirichlet_alpha=float(mcts_dirichlet_alpha) if mcts_dirichlet_alpha is not None else 0.3,
        root_noise_frac=float(mcts_root_noise_frac) if mcts_root_noise_frac is not None else 0.25
    )

    # Helper to run a single constrained optimization
    def run_one(env: DCMNETSelectionEnv) -> Tuple[np.ndarray, float]:
        nonlocal global_best_result, interrupted
        current_env = env
        safety_steps = current_env.n_atoms * current_env.total_charges_per_atom
        steps_taken = 0
        import time as _time
        t_start = _time.perf_counter()
        t_last = t_start
        accepted_moves = 0
        best_loss_so_far = float('inf')
        best_selection_so_far = None
        if verbose:
            init_sel = int(np.sum(current_env.selected_charges))
            init_loss = float(current_env.get_esp_loss())
            if np.isfinite(init_loss):
                jax.debug.print("[OPT] initial total_selected={t} loss={L}",
                                t=jnp.array(init_sel), L=jnp.array(init_loss, dtype=jnp.float32))
            else:
                print(f"[OPT] initial total_selected={init_sel} loss={init_loss}")
        while not current_env.is_terminal() and steps_taken < safety_steps and not interrupted:
            if verbose and ((steps_taken % max(1, int(log_interval_steps)) == 0) or ((_time.perf_counter() - t_last) >= float(log_interval_seconds))):
                cps = np.sum(current_env.selected_charges, axis=1)
                covered = int(np.sum(cps > 0))
                total_sel = int(np.sum(current_env.selected_charges))

                jax.debug.print("[OPT] step={s} covered={c}/{n} total_selected={t}",
                                    s=jnp.array(steps_taken), c=jnp.array(covered),
                                    n=jnp.array(current_env.n_atoms), t=jnp.array(total_sel))
                # Also print RMSE occasionally for quick feedback
                try:
                    cur_loss = float(current_env.get_esp_loss())
                    if np.isfinite(cur_loss):
                        cur_rmse = float(np.sqrt(max(cur_loss, 0.0)))
                        jax.debug.print("[OPT] step={s} RMSE={r}", s=jnp.array(steps_taken), r=jnp.array(cur_rmse, dtype=jnp.float32))
                    # Timing info
                    now = _time.perf_counter()
                    dt = now - t_last
                    total_dt = now - t_start
                    if dt > 0:
                        sims_per_sec = float(n_simulations) / dt
                        jax.debug.print("[OPT] step={s} dt={dt}s sims/s={sps} totalT={tot}s",
                                        s=jnp.array(steps_taken), dt=jnp.array(dt, dtype=jnp.float32),
                                        sps=jnp.array(sims_per_sec, dtype=jnp.float32), tot=jnp.array(total_dt, dtype=jnp.float32))
                    t_last = now
                except Exception:
                    pass

            loss_before = float(current_env.get_esp_loss())
            best_action = mcts.search(current_env, n_simulations=n_simulations, temperature=temperature)
            proposed_env = current_env.step(best_action)
            if accept_only_better:
                try:
                    loss_after = float(proposed_env.get_esp_loss())
                except Exception:
                    loss_after = float('inf')
                if np.isfinite(loss_after) and loss_after <= loss_before:
                    current_env = proposed_env
                    accepted_moves += 1
                else:
                    # reject move, keep current_env; still count the step
                    pass
            else:
                current_env = proposed_env
                accepted_moves += 1
            
            # Track best result for early stopping
            current_loss = float(current_env.get_esp_loss())
            if np.isfinite(current_loss) and current_loss < best_loss_so_far:
                best_loss_so_far = current_loss
                best_selection_so_far = current_env.selected_charges.copy()
            # NN-guided refinement: apply gradient steps only to atoms selected by NN
            if hasattr(current_env, 'nn_refine_atoms') and np.any(current_env.nn_refine_atoms > 0):
                try:
                    # Select atoms to refine based on NN predictions (top-k or threshold)
                    refine_threshold = np.percentile(current_env.nn_refine_atoms, 70)  # Top 30%
                    atoms_to_refine = np.where(current_env.nn_refine_atoms > refine_threshold)[0]
                    
                    if len(atoms_to_refine) > 0:
                        s_flat = np.asarray(current_env.selected_charges, dtype=float).reshape(-1)
                        v_flat = current_env.candidate_values.reshape(-1).astype(float)
                        q_flat = v_flat * s_flat
                        K = current_env.K_surface_to_candidates
                        y = np.asarray(current_env.esp_target, dtype=float)
                        grad = 2.0 * (K.T @ (K @ q_flat - y))
                        
                        # Apply gradient steps only to selected atoms
                        for atom_idx in atoms_to_refine:
                            for charge_idx in range(current_env.total_charges_per_atom):
                                flat_idx = atom_idx * current_env.total_charges_per_atom + charge_idx
                                if s_flat[flat_idx] > 0.5:  # Only refine selected charges
                                    q_new = q_flat[flat_idx] - float(refine_lr) * grad[flat_idx]
                                    current_env.overridden_charge_values[(atom_idx, charge_idx)] = float(q_new)
                        
                        # Update gradient stats for selected atoms only
                        grad_sel = grad[s_flat > 0.5]
                        if grad_sel.size > 0:
                            current_env.last_grad_stats = (float(np.mean(np.abs(grad_sel))), float(np.std(grad_sel)))
                        
                        if verbose:
                            print(f"[REFINE] Applied gradient steps to {len(atoms_to_refine)} atoms: {atoms_to_refine}")
                except Exception as e:
                    if verbose:
                        print(f"Warning: NN-guided refinement skipped due to: {e}")
            steps_taken += 1
            
            # Check for interruption
            if interrupted:
                if verbose:
                    print(f"[INTERRUPT] Stopping early at step {steps_taken}")
                break
        
        # Return best result found (either final or best during search)
        if best_selection_so_far is not None and best_loss_so_far < float(current_env.get_esp_loss()):
            if verbose:
                print(f"[INTERRUPT] Using best result from step {steps_taken} (loss={best_loss_so_far:.6f})")
            return best_selection_so_far, best_loss_so_far
        
        # Optionally, store a replay sample from the last root visits (if available)
        try:
            if mcts.last_root is not None and len(mcts.last_root.P) > 0:
                legal_actions = list(mcts.last_root.P.keys())
                # Build feature vector x for this state
                charge_features = current_env.selected_charges.flatten().astype(np.float32)
                step_feature = np.array([current_env.step_count / current_env.max_steps], dtype=np.float32)
                esp_loss_estimate = current_env.get_esp_loss()
                esp_feature = np.array([min(esp_loss_estimate / 1000.0, 1.0)], dtype=np.float32)
                grad_mean, grad_std = current_env.last_grad_stats if hasattr(current_env, 'last_grad_stats') else (0.0, 0.0)
                grad_feats = np.array([grad_mean, grad_std], dtype=np.float32)
                x_np = np.concatenate([charge_features, step_feature, esp_feature, grad_feats])
                # Targets: softmax of visit counts over legal actions
                # Recompute mapping indices consistent with nn head
                C = int(current_env.total_charges_per_atom)
                def act_to_idx(a):
                    if isinstance(a, tuple) and len(a) == 2 and not isinstance(a[0], str):
                        atom_idx, charge_idx = a
                        return int(atom_idx) * C + int(charge_idx)
                    elif isinstance(a, tuple) and len(a) == 3 and a[0] == "swap_model":
                        atom_idx, model_id = a[1], a[2]
                        base = int(current_env.n_atoms * C)
                        offset = 0
                        for aidx in range(current_env.n_atoms):
                            for mid in current_env.model_ids:
                                if aidx == int(atom_idx) and mid == int(model_id):
                                    return base + offset
                                offset += 1
                        return base  # fallback
                    else:
                        raise ValueError(f"Unknown action type: {a}")
                idxs = np.array([act_to_idx(a) for a in legal_actions], dtype=np.int32)
                visits = np.array([mcts.last_root.children[a].N if a in mcts.last_root.children else 0 for a in legal_actions], dtype=np.float32)
                if visits.sum() > 0:
                    target_pi = visits / visits.sum()
                else:
                    target_pi = np.ones_like(visits) / max(1, len(visits))
                # Value target is negative loss (maximize value)
                target_value = float(-current_env.get_esp_loss())
                # Capture arrays used for end-to-end loss training
                s_flat = np.asarray(current_env.selected_charges, dtype=np.float32).reshape(-1)
                v_flat = current_env.candidate_values.reshape(-1).astype(np.float32)
                lam = float(current_env.neutrality_lambda)
                replay_X.append(x_np)
                replay_idx.append(idxs)
                replay_pi.append(target_pi)
                replay_v.append((target_value, s_flat, v_flat, lam))
        except Exception as e:
            if verbose:
                print(f"Warning: NN train step skipped due to: {e}")
        if verbose:
            total_sel = int(np.sum(current_env.selected_charges))
            final_loss = float(current_env.get_esp_loss())
            # Avoid converting inf to jax arrays in debug print
            if np.isfinite(final_loss):
                jax.debug.print("[OPT] finished steps={s} total_selected={t} loss={L}",
                                s=jnp.array(steps_taken), t=jnp.array(total_sel), L=jnp.array(final_loss, dtype=jnp.float32))
                try:
                    final_rmse = float(np.sqrt(max(final_loss, 0.0)))
                    jax.debug.print("[OPT] finished RMSE={r}", r=jnp.array(final_rmse, dtype=jnp.float32))
                    total_dt = _time.perf_counter() - t_start
                    jax.debug.print("[OPT] total time={t}s avg sims/s={sps}",
                                    t=jnp.array(total_dt, dtype=jnp.float32),
                                    sps=jnp.array((steps_taken * float(n_simulations)) / max(total_dt, 1e-6), dtype=jnp.float32))
                    # Report acceptance ratio
                    jax.debug.print("[OPT] accepted={a}/{s}", a=jnp.array(accepted_moves), s=jnp.array(steps_taken))
                    # Show last root NN priors vs visit counts top-5
                    if mcts.last_root is not None and len(mcts.last_root.P) > 0:
                        actions = list(mcts.last_root.P.keys())
                        priors = np.array([mcts.last_root.P[a] for a in actions], dtype=float)
                        visits = np.array([mcts.last_root.children[a].N if a in mcts.last_root.children else 0 for a in actions], dtype=float)
                        # Top-5 by visits
                        top_vis_idx = np.argsort(-visits)[:5]
                        top_prior_idx = np.argsort(-priors)[:5]
                        print("[POLICY] Top-5 by visits:")
                        for k in top_vis_idx:
                            print(f"  a={actions[k]} visits={int(visits[k])} prior={priors[k]:.3f}")
                        print("[POLICY] Top-5 by priors:")
                        for k in top_prior_idx:
                            print(f"  a={actions[k]} prior={priors[k]:.3f} visits={int(visits[k])}")
                except Exception:
                    pass
            else:
                print(f"[OPT] finished steps={steps_taken} total_selected={total_sel} loss={final_loss}")
        return current_env.selected_charges, current_env.get_esp_loss()

    # If no target provided, just run once
    if target_total_selected is None:
        sel, loss = run_one(base_env)
        return sel, loss, {int(np.sum(sel)): (sel, loss)}

    # Try target ± span and pick best by lowest loss
    candidates: List[Tuple[np.ndarray, float]] = []
    for offset in range(-int(target_span), int(target_span) + 1):
        if interrupted:
            if verbose:
                print(f"[INTERRUPT] Stopping target trials early (completed {len(candidates)}/{2*target_span+1})")
            break
            
        target = int(max(0, target_total_selected + offset))
        env = base_env.clone()
        env.enforce_target = True
        env.target_total_selected = target
        # Also restrict the search to remain within target ± span
        env.total_min_bound = max(1, target_total_selected - target_span)
        env.total_max_bound = target_total_selected + target_span
        env.excess_charge_penalty = float(excess_charge_penalty)
        if verbose:
            jax.debug.print("[OPT] trial target={t}", t=jnp.array(target))

        sel, loss = run_one(env)
        candidates.append((sel, loss))
    # Aggregate best-by-total-selected
    best_by_total: Dict[int, Tuple[np.ndarray, float]] = {}
    for sel, loss in candidates:
        total = int(np.sum(sel))
        if total not in best_by_total or float(loss) < float(best_by_total[total][1]):
            best_by_total[total] = (sel, loss)
    
    # Filter to only include models within target ± span range
    if target_total_selected is not None:
        min_total = max(1, target_total_selected - target_span)
        max_total = target_total_selected + target_span
        filtered_best_by_total = {k: v for k, v in best_by_total.items() 
                                 if min_total <= k <= max_total}
        if not filtered_best_by_total:
            # Fallback: if no models in range, use the closest one
            closest_total = min(best_by_total.keys(), 
                               key=lambda x: abs(x - target_total_selected))
            filtered_best_by_total = {closest_total: best_by_total[closest_total]}
        best_by_total = filtered_best_by_total
    
    # Pick globally best by requested metric
    metric = (selection_metric or "loss").lower()
    if metric == "rmse_per_charge":
        def score(item):
            sel_mat, loss_val = item
            total = max(1, int(np.sum(sel_mat)))
            rmse = float(np.sqrt(max(float(loss_val), 0.0)))
            return rmse / float(total)
    else:
        # default: raw loss
        def score(item):
            return float(item[1])

    best_sel, best_loss = min(best_by_total.values(), key=score)
    if verbose:
        try:
            print("[OPT] Best per total-selected (charges):")
            rows = sorted([(k, float(v[1])) for k, v in best_by_total.items()], key=lambda t: t[0])
            for total, loss_val in rows:
                rmse = float(np.sqrt(max(loss_val, 0.0)))
                rmse_per = rmse / max(1, int(total))
                print(f"  total={total}: loss={loss_val:.6f} rmse={rmse:.6f} rmse/charge={rmse_per:.6f}")
            print(f"[OPT] Selection metric: {metric}")
        except Exception:
            pass
    # Train NN for a few epochs on replay buffer to improve next runs
    try:
        if len(replay_X) > 0:
            epochs = 3
            batch_size = min(16, len(replay_X))
            rng = np.random.default_rng(0)
            for _ in range(epochs):
                indices = rng.permutation(len(replay_X))
                for i in range(0, len(indices), batch_size):
                    batch = indices[i:i+batch_size]
                    for j in batch:
                        tval, s_flat, v_flat, lam = replay_v[j]
                        _train_selection_net_step(
                            nn_params,
                            replay_X[j],
                            replay_idx[j],
                            replay_pi[j],
                            tval,
                            K=jnp.asarray(base_env.K_surface_to_candidates, dtype=jnp.float32),
                            J=jnp.asarray(base_env.J_surface_to_candidates, dtype=jnp.float32),
                            s_flat=s_flat,
                            v_flat=v_flat,
                            lam=lam,
                            dq_max=0.05,
                            dr_max=(0.1/CONVERSION_FACTOR),
                            lr=5e-4,
                            alpha_e2e=float(alpha_e2e),
                        )
    except Exception as e:
        if verbose:
            print(f"Warning: NN replay training skipped due to: {e}")
    if verbose:
        if np.isfinite(float(best_loss)):
            jax.debug.print("[OPT] best among targets loss={L}", L=jnp.array(float(best_loss), dtype=jnp.float32))
        else:
            print(f"[OPT] best among targets loss={best_loss}")
        
        if interrupted:
            print(f"[INTERRUPT] Optimization interrupted but completed with {len(candidates)} trials")

    # Save NN weights if run_dir provided
    if run_dir is not None:
        try:
            import numpy as _np
            _np.savez(os.path.join(run_dir, "nn_params.npz"), **nn_params)
        except Exception as e:
            if verbose:
                print(f"Warning: failed to save nn_params: {e}")

    # Restore original signal handler
    signal.signal(signal.SIGINT, original_handler)

    return best_sel, best_loss, best_by_total

