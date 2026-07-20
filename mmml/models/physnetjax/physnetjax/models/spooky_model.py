"""
Energy and Forces Neural Network Model implementation.

This module implements a neural network model for predicting molecular energies 
and forces using message passing and equivariant transformations.

The spooky model is trained on positions R, atomic numbers Z, and forces F, energies E,
and Charge Q and Spin (Multiplicity) S.
"""

from typing import Dict, List, Optional, Tuple

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special
from jax import Array
# from jax.experimental import mesh_utils
# from jax.sharding import Mesh
# from jax.sharding import NamedSharding
# from jax.sharding import PartitionSpec as P

from mmml.models.physnetjax.physnetjax.models.euclidean_fast_attention import fast_attention as efa
from mmml.models.physnetjax.physnetjax.models.mpnn_kernels import (
    calc_electrostatics_switches,
    pair_displacements,
    pair_electrostatics_energy,
    radial_spherical_basis,
)
from mmml.models.physnetjax.physnetjax.models.zbl import (
    ZBLRepulsion,
    geometric_pair_distances,
)

EFA = efa.EuclideanFastAttention
import ase.data

# Constants
DTYPE = jnp.float32
HARTREE_TO_KCAL_MOL = 627.509  # Conversion factor for energy units


class SpookyPhysNet(nn.Module):
    """SpookyNet-style PhysNet with charge and spin conditioning inputs.

    Trained on positions R, atomic numbers Z, forces F, energies E,
    charge Q, and spin multiplicity S.
    """

    features: int = 64
    max_degree: int = 1
    num_iterations: int = 2
    num_basis_functions: int = 32
    cutoff: float = 6.0
    max_atomic_number: int = 87
    charges: bool = False
    max_padded_atoms: int = 35
    total_charge: float = 0
    n_refinement_blocks: int = 2
    zbl: bool = True
    trainable_zbl: bool = False
    zbl_cuton: float | None = 0.1
    zbl_cutoff: float = 0.6
    debug: bool | List[str] = False
    efa: bool = False
    use_energy_bias: bool = False
    use_pbc: bool = False
    electrostatics_damping_sigma: float = 4.0
    # Electrostatics/short-vs-long-range switch distances (Angstrom), used by
    # _calc_switches. Defaults reproduce the values every checkpoint trained
    # before this field existed was implicitly hardcoded to -- changing them
    # changes electrostatics behavior and is NOT backward compatible with
    # checkpoints trained under the old defaults. Previously these were
    # decoupled from `cutoff` with no code anywhere checking the two stayed
    # consistent (see mmml/models/physnetjax/physnetjax/training/
    # far_field_augment.py's SAFE_SEPARATION_ANGSTROM, which depends on
    # electrostatics_off_end being an exact hard-zero point and silently
    # went stale if this ever changed without it knowing).
    switch_start: float = 1.0
    switch_end: float = 10.0
    electrostatics_off_start: float = 8.0
    electrostatics_off_end: float = 10.0

    @property
    def natoms(self) -> int:
        return self.max_padded_atoms

    @natoms.setter
    def natoms(self, value: int) -> None:
        object.__setattr__(self, "max_padded_atoms", value)

    @property
    def n_res(self) -> int:
        return self.n_refinement_blocks

    @n_res.setter
    def n_res(self, value: int) -> None:
        object.__setattr__(self, "n_refinement_blocks", value)

    def setup(self) -> None:
        """
        Initialize model components.
        
        Sets up the model architecture including ZBL repulsion and
        Euclidean Fast Attention (EFA) if enabled.
        """
        if self.zbl:
            self.repulsion = ZBLRepulsion(
                cutoff=self.zbl_cutoff,
                cuton=self.zbl_cuton,
                trainable=self.trainable_zbl,
            )
        self.efa_final = None
        if self.efa:
            b_max = 4 * jnp.pi
            # We now initialize an EFA module.
            self.efa_final = EFA(
                lebedev_num=194,
                parametrized=False,
                epe_max_frequency=b_max,
                epe_max_length=20.0,  # maximum distance in Angstroms for the EPE
                tensor_integration=True,
                ti_degree_scaling_constants=[
                    0.5**i for i in range(self.max_degree + 1)
                ],
            )

    def return_attributes(self) -> Dict:
        """
        Return model attributes for checkpointing.
        
        Returns
        -------
        Dict
            Dictionary containing all model hyperparameters and configuration
        """
        return {
            "features": self.features,
            "max_degree": self.max_degree,
            "num_iterations": self.num_iterations,
            "num_basis_functions": self.num_basis_functions,
            "cutoff": self.cutoff,
            "max_atomic_number": self.max_atomic_number,
            "charges": self.charges,
            "natoms": self.max_padded_atoms,
            "max_padded_atoms": self.max_padded_atoms,
            "total_charge": self.total_charge,
            "n_res": self.n_refinement_blocks,
            "n_refinement_blocks": self.n_refinement_blocks,
            "zbl": self.zbl,
            "trainable_zbl": self.trainable_zbl,
            "zbl_cuton": self.zbl_cuton,
            "zbl_cutoff": self.zbl_cutoff,
            "debug": self.debug,
            "efa": self.efa,
            "use_energy_bias": self.use_energy_bias,
            "use_pbc": self.use_pbc,
            "electrostatics_damping_sigma": self.electrostatics_damping_sigma,
            "switch_start": self.switch_start,
            "switch_end": self.switch_end,
            "electrostatics_off_start": self.electrostatics_off_start,
            "electrostatics_off_end": self.electrostatics_off_end,
        }

    def energy(
        self,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        spins: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
        batch_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
        cell: Optional[jnp.ndarray] = None,
        mol_id: jnp.ndarray | None = None,
        cgenff_type_idx: jnp.ndarray | None = None,
        cgenff_master_sigmas: jnp.ndarray | None = None,
        cgenff_master_epsilons: jnp.ndarray | None = None,
        edge_mask: jnp.ndarray | None = None,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """
        Calculate molecular energy and related properties.

        Computes the total energy of molecular systems including atomic energies,
        electrostatic interactions, ZBL repulsion, and optionally CGenFF LJ vdW.
        """
        basis, displacements = self._calculate_geometric_features(
            positions,
            dst_idx,
            src_idx,
            cell=cell,
            batch_mask=batch_mask,
            edge_mask=edge_mask,
        )

        graph_mask = jnp.ones(batch_size)

        x = self._process_atomic_features(
            atomic_numbers,
            charges,
            spins,
            basis,
            dst_idx,
            src_idx,
            positions,
            batch_segments,
            graph_mask,
            atom_mask,
        )

        return self._calculate(
            x,
            atomic_numbers,
            displacements,
            dst_idx,
            src_idx,
            atom_mask,
            batch_mask,
            batch_segments,
            batch_size,
            mol_id=mol_id,
            cgenff_type_idx=cgenff_type_idx,
            cgenff_master_sigmas=cgenff_master_sigmas,
            cgenff_master_epsilons=cgenff_master_epsilons,
        )

    def _calculate_geometric_features(
        self,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        cell: Optional[jnp.ndarray] = None,
        batch_mask: Optional[jnp.ndarray] = None,
        edge_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate geometric features including displacements and basis functions.

        ``batch_mask`` (per-edge 0/1) zeroes the message-passing basis for
        invalid edges — including real↔padding pairs. Without this, padding
        atoms at the origin corrupt features of any real atom within
        ``cutoff``, breaking translation invariance.

        ``edge_mask`` (per-edge 0/1) further zeroes the basis for masked edges,
        which is how the neural graph is restricted to intra-monomer edges when
        computing the monomer reference for the interaction-energy penalty. It
        does not touch the pairwise prior terms (those are gated separately by
        ``batch_mask``).
        
        Parameters
        ----------
        positions : jnp.ndarray
            Atomic positions
        dst_idx : jnp.ndarray
            Destination indices for message passing
        src_idx : jnp.ndarray
            Source indices for message passing
        cell : Optional[jnp.ndarray]
            If provided, apply minimum-image convention to displacements (PBC).
        batch_mask : Optional[jnp.ndarray]
            Per-edge validity mask (1 for real↔real, 0 for padding edges)
        edge_mask : Optional[jnp.ndarray]
            Optional extra per-edge mask (e.g. intra-monomer restriction)
            
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Tuple of (basis functions, displacements)
        """
        displacements = pair_displacements(
            positions,
            dst_idx,
            src_idx,
            cell=cell,
            use_pbc=bool(self.use_pbc),
        )
        return radial_spherical_basis(
            displacements,
            num_basis_functions=self.num_basis_functions,
            max_degree=self.max_degree,
            cutoff=self.cutoff,
            batch_mask=batch_mask,
            edge_mask=edge_mask,
        )

    def _process_atomic_features(
        self,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        spins: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        positions: jnp.ndarray,
        batch_segments: jnp.ndarray,
        graph_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process atomic features through message passing and refinement.

        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        positions : jnp.ndarray
            Atomic positions
        batch_segments : jnp.ndarray
            Batch segment indices
        graph_mask : jnp.ndarray
            Graph mask
        atom_mask : jnp.ndarray
            Per-atom validity mask (1 for real atoms, 0 for padding). Threaded
            through to EFA (see _message_passing_iteration) — unlike the
            regular message-passing/basis-function path (which has a smooth
            distance cutoff baked into e3x.nn.smooth_cutoff), EFA's attention
            has no distance decay of its own, so padded "ghost" atoms must be
            masked explicitly or their (not-guaranteed-zero) embedding leaks
            into the attention sum regardless of how far away they are.

        Returns
        -------
        jnp.ndarray
            Processed atomic features
        """
        embed = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            dtype=DTYPE,
        )
        x = embed(atomic_numbers)

        # Project per-atom scalar charge/spin conditioning to e3x feature format
        # before adding to atomic embeddings.
        n_atoms = atomic_numbers.shape[0]
        charge_inputs = jnp.asarray(charges, dtype=DTYPE).reshape(n_atoms, -1)
        spin_inputs = jnp.asarray(spins, dtype=DTYPE).reshape(n_atoms, -1)

        charge_proj = nn.Dense(
            self.features,
            dtype=DTYPE,
            name="charge_feature_projection",
        )(charge_inputs)
        spin_proj = nn.Dense(
            self.features,
            dtype=DTYPE,
            name="spin_feature_projection",
        )(spin_inputs)

        cond_features = charge_proj + spin_proj  # (n_atoms, features)
        cond_e3x = jnp.expand_dims(cond_features, axis=(1, 2))  # (n_atoms, 1, 1, features)
        cond_e3x = jnp.pad(
            cond_e3x,
            ((0, 0), (0, 0), (0, x.shape[2] - 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )  # (n_atoms, 1, max_degree+1, features)
        x = e3x.nn.add(x, cond_e3x)
        
        for i in range(self.num_iterations):
            x = self._message_passing_iteration(
                x, basis, dst_idx, src_idx, i, positions, batch_segments, graph_mask, atom_mask
            )
            x = self._refinement_iteration(x)

        basis = e3x.nn.change_max_degree_or_type(
            basis, max_degree=0, include_pseudotensors=False
        )
        x = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )
        if self.n_refinement_blocks <= -1:
            for i in range(self.num_iterations):
                x = self._attention(
                    x, basis, dst_idx, src_idx, num_heads=self.features // 8
                )
                x = self._refinement_iteration(x)
        return x

    def _attention(self, x, basis, dst_idx, src_idx, num_heads=2):
        """
        Apply self-attention mechanism.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        num_heads : int, optional
            Number of attention heads, by default 2
            
        Returns
        -------
        jnp.ndarray
            Attention output
        """
        return e3x.nn.modules.SelfAttention(
            max_degree=0,
            num_heads=num_heads,
            include_pseudotensors=False,
        )(x, basis, dst_idx=dst_idx, src_idx=src_idx)

    def _multiheadattention(self, x, y, basis, dst_idx, src_idx, num_heads=2):
        """
        Apply multi-head attention mechanism.
        
        Parameters
        ----------
        x : jnp.ndarray
            Query features
        y : jnp.ndarray
            Key/value features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        num_heads : int, optional
            Number of attention heads, by default 2
            
        Returns
        -------
        jnp.ndarray
            Multi-head attention output
        """
        return e3x.nn.modules.MultiHeadAttention(
            max_degree=self.max_degree,
            num_heads=num_heads,
            include_pseudotensors=False,
        )(x, y, basis, dst_idx=dst_idx, src_idx=src_idx)

    def _message_passing_iteration(
        self,
        x: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        iteration: int,
        positions: jnp.ndarray,
        batch_segments: jnp.ndarray,
        graph_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Perform one iteration of message passing.

        Parameters
        ----------
        x : jnp.ndarray
            Current features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        iteration : int
            Current iteration number
        positions : jnp.ndarray
            Atomic positions
        batch_segments : jnp.ndarray
            Batch segment indices
        graph_mask : jnp.ndarray
            Graph mask
        atom_mask : jnp.ndarray
            Per-atom validity mask, see _process_atomic_features docstring.

        Returns
        -------
        jnp.ndarray
            Updated features after message passing
        """
        # if it is the last iteration
        if iteration == self.num_iterations - 1:
            x = e3x.nn.MessagePass(
                max_degree=0,
                include_pseudotensors=False,
                # dense_kernel_init=jax.nn.initializers.he_uniform(),
                # dense_bias_init=jax.nn.initializers.zeros,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
            return x

        x = e3x.nn.MessagePass(
            include_pseudotensors=False,
            # dense_kernel_init=jax.nn.initializers.he_normal(),
            # dense_bias_init=jax.nn.initializers.zeros,
        )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
        if self.efa:
            # EFA's attention has no distance cutoff/decay of its own (unlike
            # the message-passing basis functions above, which use
            # e3x.nn.smooth_cutoff) — padded "ghost" atoms share
            # batch_segments=0 with real atoms in single-structure inference,
            # and their embedding (atomic number 0, row 0 of a trained table)
            # isn't guaranteed to be the zero vector. Mask explicitly on both
            # sides of the call so padding provably contributes nothing,
            # regardless of how the embedding for the padding species turned
            # out during training.
            mask4 = atom_mask[..., None, None, None]
            x1 = self.efa_final(x * mask4, positions, batch_segments, graph_mask)
            x = e3x.nn.add(x, x1 * mask4)
        return x

    def _refinement_iteration(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform refinement iterations with residual connections.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input features
            
        Returns
        -------
        jnp.ndarray
            Refined features
        """
        e3x.nn.silu(x)
        for _ in range(abs(self.n_refinement_blocks)):
            y = e3x.nn.silu(x)
            y = e3x.nn.add(x, y)
            y = e3x.nn.Dense(
                self.features,
            )(y)
            x = e3x.nn.add(x, y)
        y = e3x.nn.Dense(
            self.features,
        )(y)
        y = e3x.nn.silu(y)

        return y

    learn_cgenff_vdw_scale: bool = True
    predict_atomic_vdw_scale: bool = True

    # Learned per-element-pair shrinkage ("trust map") for the neural interaction energy.
    # When enabled, a (n_el, n_el) log-lambda matrix over these elements is created as a
    # parameter and surfaced in the output; the training loss turns it into an
    # evidence-balanced per-chemistry shrinkage and can dump it as a data-provenance
    # fingerprint. Off by default (no parameter, no behavioural change).
    interaction_trust_map: bool = False
    trust_map_elements: tuple = (1, 6, 7, 8, 16, 17)  # H, C, N, O, S, Cl

    def _calculate_atomic_vdw_scales(
        self, x: jnp.ndarray, atomic_numbers: jnp.ndarray, atom_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Predict dynamic atom-specific vdW scale factor gamma_i from atomic features x_i.

        Returns values smoothly bounded around 1.0 e.g. in range [0.5, 1.5].
        """
        scale_raw = nn.Dense(
            1, use_bias=True, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(x)
        atomic_vdw_scale = 1.0 + 0.5 * jnp.tanh(scale_raw)
        atomic_vdw_scale *= atom_mask[..., None, None, None]
        return atomic_vdw_scale.reshape(-1)

    def _calculate_cgenff_vdw(
        self,
        displacements: jnp.ndarray,
        off_dist: jnp.ndarray,
        cgenff_type_idx: jnp.ndarray,
        cgenff_master_sigmas: jnp.ndarray,
        cgenff_master_epsilons: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        atomic_vdw_scales: jnp.ndarray | None,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        mol_id: jnp.ndarray | None,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate inter-monomer CGenFF 6-12 Lennard-Jones vdW interactions in JAX with dynamic atom-specific scaling.

        Conversion: 1 kcal/mol = 0.0433641153 eV.
        Lorentz-Berthelot combination rules:
            sig_ij = 0.5 * (sig_i + sig_j)
            eps_ij = sqrt(eps_i * eps_j)
        """
        # Use the actual pair separation in Å.  The ``r`` produced by
        # ``_calc_switches`` is a damped Coulomb kernel (approximately 1/r),
        # and using it here forced nearly every active LJ pair onto the
        # 0.8*sigma clamp.
        pair_distances = jnp.sqrt(
            jnp.maximum(jnp.sum(displacements**2, axis=-1), 1.0e-12)
        )

        sig_dst = jnp.take(cgenff_master_sigmas, jnp.take(cgenff_type_idx, dst_idx, fill_value=0), fill_value=3.5)
        sig_src = jnp.take(cgenff_master_sigmas, jnp.take(cgenff_type_idx, src_idx, fill_value=0), fill_value=3.5)
        eps_dst = jnp.take(cgenff_master_epsilons, jnp.take(cgenff_type_idx, dst_idx, fill_value=0), fill_value=0.05)
        eps_src = jnp.take(cgenff_master_epsilons, jnp.take(cgenff_type_idx, src_idx, fill_value=0), fill_value=0.05)

        sig_ij = 0.5 * (sig_dst + sig_src)
        eps_ij = jnp.sqrt(eps_dst * eps_src)

        # Apply dynamic atom-specific vdW scale factors gamma_i predicted by neural network
        if atomic_vdw_scales is not None:
            scale_dst = jnp.take(atomic_vdw_scales, dst_idx)
            scale_src = jnp.take(atomic_vdw_scales, src_idx)
            eps_scale = jnp.sqrt(jnp.maximum(scale_dst * scale_src, 1e-4))
            eps_ij = eps_ij * eps_scale

        # Optional learnable global and per-element LJ parameter scaling
        if self.learn_cgenff_vdw_scale:
            global_scale = self.param(
                "global_vdw_scale",
                lambda rng, shape: jnp.ones(shape, dtype=pair_distances.dtype),
                (1,),
            )
            element_scale = self.param(
                "element_vdw_scale",
                lambda rng, shape: jnp.ones(shape, dtype=pair_distances.dtype),
                (self.max_atomic_number + 1,),
            )
            elem_dst = jnp.take(element_scale, jnp.take(atomic_numbers, dst_idx))
            elem_src = jnp.take(element_scale, jnp.take(atomic_numbers, src_idx))
            elem_scale = global_scale * jnp.sqrt(jnp.maximum(elem_dst * elem_src, 1e-4))
            eps_ij = eps_ij * elem_scale

        if mol_id is not None:
            inter_monomer_mask = (jnp.take(mol_id, dst_idx) != jnp.take(mol_id, src_idx)).astype(pair_distances.dtype)
        else:
            inter_monomer_mask = 1.0

        # Soft-core distance clamping to cap unphysical LJ 6-12 repulsion spikes at r < 0.8 * sig_ij
        r_safe = jnp.maximum(pair_distances, 0.8 * sig_ij)
        sr6 = (sig_ij / r_safe) ** 6
        sr12 = sr6 ** 2
        
        KCAL_TO_EV = jnp.asarray(0.0433641153, dtype=pair_distances.dtype)
        vdw_pair = 4.0 * eps_ij * (sr12 - sr6) * KCAL_TO_EV * off_dist * batch_mask * inter_monomer_mask
        vdw_pair = 0.5 * vdw_pair

        num_atoms_actual = cgenff_type_idx.shape[0]
        atomic_vdw = jax.ops.segment_sum(vdw_pair, segment_ids=dst_idx, num_segments=num_atoms_actual)[:num_atoms_actual]
        batch_vdw = jax.ops.segment_sum(atomic_vdw, segment_ids=batch_segments, num_segments=batch_size)

        atomic_vdw = atomic_vdw[..., None, None, None]
        batch_vdw = batch_vdw[..., None, None, None]
        return atomic_vdw, batch_vdw

    def _calculate(
        self,
        x: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        displacements: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
        mol_id: jnp.ndarray | None = None,
        cgenff_type_idx: jnp.ndarray | None = None,
        cgenff_master_sigmas: jnp.ndarray | None = None,
        cgenff_master_epsilons: jnp.ndarray | None = None,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        r, off_dist, eshift = self._calc_switches(displacements, batch_mask)
        zbl_distances = geometric_pair_distances(displacements, batch_mask)

        atomic_energies = self._calculate_atomic_energies(x, atomic_numbers, atom_mask)

        if mol_id is not None:
            inter_monomer_mask = (jnp.take(mol_id, dst_idx) != jnp.take(mol_id, src_idx)).astype(r.dtype)
            elec_batch_mask = batch_mask * inter_monomer_mask
        else:
            elec_batch_mask = batch_mask

        if self.charges:
            atomic_charges = self._calculate_atomic_charges(
                x, atomic_numbers, atom_mask
            )
            electrostatics, batch_electrostatics = self._calculate_electrostatics(
                atomic_charges,
                r,
                off_dist,
                eshift,
                dst_idx,
                src_idx,
                atom_mask,
                elec_batch_mask,
                batch_segments,
                batch_size,
            )
        else:
            atomic_charges = None
            electrostatics = 0.0
            batch_electrostatics = None

        if (cgenff_type_idx is not None and 
            cgenff_master_sigmas is not None and 
            cgenff_master_epsilons is not None):
            if self.predict_atomic_vdw_scale:
                atomic_vdw_scales = self._calculate_atomic_vdw_scales(x, atomic_numbers, atom_mask)
            else:
                atomic_vdw_scales = None

            atomic_vdw, batch_vdw = self._calculate_cgenff_vdw(
                displacements,
                off_dist,
                cgenff_type_idx,
                cgenff_master_sigmas,
                cgenff_master_epsilons,
                atomic_numbers,
                atomic_vdw_scales,
                dst_idx,
                src_idx,
                mol_id,
                batch_mask,
                batch_segments,
                batch_size,
            )
        else:
            atomic_vdw = 0.0
            batch_vdw = None

        if self.zbl:
            repulsion = self._calculate_repulsion(
                atomic_numbers,
                zbl_distances,
                None,
                eshift,
                dst_idx,
                src_idx,
                atom_mask,
                batch_mask,
                batch_segments,
                batch_size,
            )
        else:
            repulsion = 0.0

        energy = jax.ops.segment_sum(
            atomic_energies + electrostatics + repulsion + atomic_vdw,
            segment_ids=batch_segments,
            num_segments=batch_size,
        )
        energy = energy.reshape((batch_size, -1)).sum(axis=1, keepdims=True)

        return -1 * jnp.sum(energy), (
            energy,
            atomic_charges,
            batch_electrostatics,
            batch_vdw,
            repulsion,
            x,
        )

    def _calculate_atomic_charges(
        self, x: jnp.ndarray, atomic_numbers: jnp.ndarray, atom_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate atomic charges from atomic features.
        
        Parameters
        ----------
        x : jnp.ndarray
            Atomic features
        atomic_numbers : jnp.ndarray
            Atomic numbers
        atom_mask : jnp.ndarray
            Atom mask
            
        Returns
        -------
        jnp.ndarray
            Predicted atomic charges
        """
        x = e3x.nn.Dense(1, use_bias=False)(x)

        charge_bias = self.param(
            "charge_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_charges = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.normal(stddev=0.01), dtype=DTYPE
        )(x)
        atomic_charges += jnp.take(charge_bias, atomic_numbers, axis=0)[
            ..., None, None, None
        ]
        atomic_charges *= atom_mask[..., None, None, None]
        return atomic_charges

    def _calculate_repulsion(
        self,
        atomic_numbers: jnp.ndarray,
        distances: jnp.ndarray,
        off_dist: jnp.ndarray,
        eshift: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """
        Calculate repulsion energies between atoms.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        distances : jnp.ndarray
            Interatomic distances
        off_dist : jnp.ndarray
            Distance cutoff factors
        eshift : jnp.ndarray
            Energy shift factors
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        atom_mask : jnp.ndarray
            Atom mask
        batch_mask : jnp.ndarray
            Batch mask
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Batch size
            
        Returns
        -------
        jnp.ndarray
            Repulsion energies per atom
        """
        # add the learnable parameters to the model
        repulsion_energy = self.repulsion(
            atomic_numbers,
            distances,
            off_dist,
            eshift,
            dst_idx,
            src_idx,
            atom_mask,
            batch_mask,
            batch_segments,
            batch_size,
        )
        return repulsion_energy

    def _calculate_atomic_energies(
        self, x: jnp.ndarray, atomic_numbers: jnp.ndarray, atom_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate atomic energies from atomic features.
        
        Parameters
        ----------
        x : jnp.ndarray
            Atomic features
        atomic_numbers : jnp.ndarray
            Atomic numbers
        atom_mask : jnp.ndarray
            Atom mask
            
        Returns
        -------
        jnp.ndarray
            Predicted atomic energies
        """
        x = e3x.nn.Dense(1, use_bias=False)(x)
        
        # Optionally add per-element energy bias (learnable atomic reference energies)
        if self.use_energy_bias:
            energy_bias = self.param(
                "energy_bias",
                lambda rng, shape: jnp.zeros(shape),
                (self.max_atomic_number + 1),
            )
        
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(x)
        
        if self.use_energy_bias:
            atomic_energies += jnp.take(
                jnp.asarray(energy_bias), atomic_numbers
            )[..., None, None, None]
        
        atomic_energies *= atom_mask[..., None, None, None]

        return atomic_energies

    def _calc_switches(self, displacements: jnp.ndarray, batch_mask: jnp.ndarray):
        """
        Calculate switching functions for smooth interactions.
        
        Parameters
        ----------
        displacements : jnp.ndarray
            Interatomic displacements
        batch_mask : jnp.ndarray
            Batch mask
            
        Returns
        -------
        tuple
            Tuple of (r, off_dist, eshift) switching factors
        """
        return calc_electrostatics_switches(
            displacements,
            batch_mask,
            switch_start=self.switch_start,
            switch_end=self.switch_end,
            electrostatics_off_start=self.electrostatics_off_start,
            electrostatics_off_end=self.electrostatics_off_end,
            electrostatics_damping_sigma=self.electrostatics_damping_sigma,
        )

    def _calculate_electrostatics(
        self,
        atomic_charges: jnp.ndarray,
        r: jnp.ndarray,
        off_dist: jnp.ndarray,
        eshift: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.array]:
        """
        Calculate electrostatic interactions between atoms.

        Uses a smoothly switched combination of short-range and long-range electrostatics
        to avoid numerical instabilities at zero distance while maintaining accuracy.

        Parameters
        ----------
        atomic_charges : jnp.ndarray
            Predicted atomic charges
        r : jnp.ndarray
            Distance factors
        off_dist : jnp.ndarray
            Distance cutoff factors
        eshift : jnp.ndarray
            Energy shift factors
        dst_idx : jnp.ndarray
            Destination indices for pair interactions
        src_idx : jnp.ndarray
            Source indices for pair interactions
        atom_mask : jnp.ndarray
            Atom mask
        batch_mask : jnp.ndarray
            Batch mask
        batch_segments : jnp.ndarray
            Batch assignment for each atom
        batch_size : int
            Number of molecules in batch

        Returns
        -------
        Tuple[jnp.ndarray, jnp.array]
            Tuple of (atomic electrostatic energies, batch electrostatic energies)
        """
        del atom_mask  # retained for call-site compatibility
        return pair_electrostatics_energy(
            atomic_charges,
            r,
            off_dist,
            eshift,
            dst_idx,
            src_idx,
            batch_mask,
            batch_segments,
            batch_size,
        )

    def _calculate_dipole(
        self,
        positions: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """
        Calculate dipoles for a batch of molecules.

        Computes molecular dipole moments from atomic charges and positions
        relative to the center of mass of each molecule.

        Parameters
        ----------
        positions : jnp.ndarray
            Atomic positions
        atomic_numbers : jnp.ndarray
            Atomic numbers
        charges : jnp.ndarray
            Atomic charges
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Number of molecules in the batch

        Returns
        -------
        jnp.ndarray
            Calculated dipoles for each molecule in the batch
        """
        charges = charges.squeeze()
        positions = positions.squeeze()
        atomic_numbers = atomic_numbers.squeeze()
        
        # Get atomic masses
        masses = jnp.take(ase.data.atomic_masses, atomic_numbers)
        
        # Calculate COM for each molecule: COM = Σ(m_i * r_i) / Σ(m_i)
        # Use segment_sum to handle batches
        mass_weighted_pos = positions * masses[..., None]  # (natoms, 3)
        total_mass_weighted_pos = jax.ops.segment_sum(
            mass_weighted_pos, 
            segment_ids=batch_segments, 
            num_segments=batch_size
        )  # (batch_size, 3)
        total_mass = jax.ops.segment_sum(
            masses, 
            segment_ids=batch_segments, 
            num_segments=batch_size
        )  # (batch_size,)
        
        com = total_mass_weighted_pos / total_mass[..., None]  # (batch_size, 3)
        
        # Get COM for each atom (broadcast back)
        com_per_atom = jnp.take(com, batch_segments, axis=0)  # (natoms, 3)
        
        # Positions relative to COM
        pos_com = positions - com_per_atom  # (natoms, 3)
        
        # Dipole = Σ(q_i * (r_i - r_COM)) per molecule
        dipoles = jax.ops.segment_sum(
            pos_com * charges[..., None],
            segment_ids=batch_segments,
            num_segments=batch_size,
        )  # (batch_size, 3)
        
        return dipoles

    @nn.compact
    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        spins: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: Optional[jnp.ndarray] = None,
        batch_size: Optional[int] = None,
        batch_mask: Optional[jnp.ndarray] = None,
        atom_mask: Optional[jnp.ndarray] = None,
        cell: Optional[jnp.ndarray] = None,
        compute_forces: bool = True,
        mol_id: jnp.ndarray | None = None,
        cgenff_type_idx: jnp.ndarray | None = None,
        cgenff_master_sigmas: jnp.ndarray | None = None,
        cgenff_master_epsilons: jnp.ndarray | None = None,
        edge_mask: jnp.ndarray | None = None,
    ) -> Dict[str, Optional[jnp.ndarray]]:
        """
        Forward pass of the model.

        Computes energies, forces, and optionally charges and dipoles
        for molecular systems.

        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Array of atomic numbers
        positions : jnp.ndarray
            Array of atomic positions
        dst_idx : jnp.ndarray
            Destination indices for message passing
        src_idx : jnp.ndarray
            Source indices for message passing
        batch_segments : Optional[jnp.ndarray], optional
            Optional batch segment indices, by default None
        batch_size : Optional[int], optional
            Optional batch size, by default None
        batch_mask : Optional[jnp.ndarray], optional
            Optional batch mask, by default None
        atom_mask : Optional[jnp.ndarray], optional
            Optional atom mask, by default None

        Returns
        -------
        Dict[str, Optional[jnp.ndarray]]
            Dictionary containing:
            - energy: Predicted energies
            - forces: Predicted forces
            - charges: Predicted charges (if enabled)
            - electrostatics: Electrostatic energies (if charges enabled)
            - repulsion: Repulsion energies (if ZBL enabled)
            - dipoles: Predicted dipoles (if charges enabled)
            - sum_charges: Sum of charges per molecule (if charges enabled)
            - state: Final atomic features
        """
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
            batch_mask = jnp.ones_like(dst_idx)
            atom_mask = jnp.ones_like(atomic_numbers)

        # import lovely_jax as lj
        #
        # lj.monkey_patch()
        #
        # jax.debug.print("atomic_numbers {x}", x=atomic_numbers[::])
        # jax.debug.print("positions {x}", x=positions[::])
        # jax.debug.print("dst_idx {x}", x=dst_idx[::])
        # jax.debug.print("src_idx {x}", x=src_idx[::])
        # jax.debug.print("batch_segments {x}", x=batch_segments[::])
        # # jax.debug.print("batch_size {x}", x=batch_size[::1])
        # jax.debug.print("batch_mask {x}", x=batch_mask[::])
        # jax.debug.print("atom_mask {x}", x=atom_mask[::])

        if not compute_forces:
            _, (energy, charges, electrostatics, cgenff_vdw, repulsion, state) = self.energy(
                atomic_numbers,
                charges,
                spins,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                cell,
                mol_id=mol_id,
                cgenff_type_idx=cgenff_type_idx,
                cgenff_master_sigmas=cgenff_master_sigmas,
                cgenff_master_epsilons=cgenff_master_epsilons,
                edge_mask=edge_mask,
            )
            forces = None
        else:
            energy_and_forces = jax.value_and_grad(self.energy, argnums=3, has_aux=True)

            (_, (energy, charges, electrostatics, cgenff_vdw, repulsion, state)), gradient = energy_and_forces(
                atomic_numbers,
                charges,
                spins,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                cell,
                mol_id,
                cgenff_type_idx,
                cgenff_master_sigmas,
                cgenff_master_epsilons,
                edge_mask,
            )
            forces = gradient
            forces *= atom_mask[..., None]

        dipoles = (
            self._calculate_dipole(
                positions,
                atomic_numbers,
                charges,
                batch_segments,
                batch_size,
            )
            if self.charges
            else None
        )
        sum_charges = (
            jax.ops.segment_sum(
                charges,
                segment_ids=batch_segments,
                num_segments=batch_size,
            )
            if self.charges
            else None
        )

        # Per-element-pair "trust map": a learned log-shrinkage matrix over the neural
        # interaction energy (used only by the training loss, never by the forward
        # energy). Living here means it is checkpointed and restarted like any other
        # parameter; it is surfaced in the output so the loss can read it without
        # reaching into the params tree. See --interaction-trust-map in the trainer.
        neural_interaction_log_lambda = None
        if self.interaction_trust_map:
            n_el = len(self.trust_map_elements)
            neural_interaction_log_lambda = self.param(
                "neural_interaction_log_lambda",
                lambda rng, shape: jnp.zeros(shape, dtype=DTYPE),
                (n_el, n_el),
            )

        # Prepare output dictionary
        output = {
            "energy": energy,
            "forces": forces,
            "charges": charges,
            "electrostatics": electrostatics,
            "cgenff_vdw": cgenff_vdw,
            "repulsion": repulsion,
            "dipoles": dipoles,
            "sum_charges": sum_charges,
            "neural_interaction_log_lambda": neural_interaction_log_lambda,
            "state": state,
        }

        return output


EF = SpookyPhysNet  # deprecated alias; prefer SpookyPhysNet

__all__ = ["SpookyPhysNet", "EF"]
