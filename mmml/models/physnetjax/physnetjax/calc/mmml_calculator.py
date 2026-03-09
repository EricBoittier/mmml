R = coor.get_positions().to_numpy()

# System constants
ATOMS_PER_MONOMER: int = 5  # Number of atoms in each monomer
MAX_ATOMS_PER_SYSTEM: int = 10  # Maximum atoms in monomer/dimer system
SPATIAL_DIMS: int = 3  # Number of spatial dimensions (x, y, z)



def get_MM_energy_forces_fns(R, ATOMS_PER_MONOMER=5, N_MONOMERS=2):
    """Creates functions for calculating MM energies and forces with switching.
    
    Returns:
        Tuple[Callable, Callable]: Functions for energy and force calculations
    """
    # CG321EP = -0.0560
    # CG321RM = 2.0100
    # CLGA1EP = -0.3430
    # CLGA1RM = 1.9100
    # HGA2EP =  -0.0200  
    # HGA2RM = 1.3400 
    # params =  [75, 76, 77]
    # params.sort()
    # at_ep = {75: CG321EP, 76:CLGA1EP , 77: HGA2EP}
    # at_rm = {75: CG321RM, 76: CLGA1RM , 77: HGA2RM}
    # at_q = {75: -0.018, 76: -0.081 , 77: 0.09}
    # at_flat_rm = np.zeros(100)
    # at_flat_rm[75] = CG321RM
    # at_flat_rm[76] = CLGA1RM
    # at_flat_rm[77] = HGA2RM
    # at_flat_ep = np.zeros(100)
    # at_flat_ep[75] = CG321EP
    # at_flat_ep[76] = CLGA1EP
    # at_flat_ep[77] = HGA2EP
    # at_flat_q = np.zeros(100)
    # at_flat_q[75] =  -0.018
    # at_flat_q[76] =  -0.081
    # at_flat_q[77] =  0.09

    ############################################################################################


    read.rtf('/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf')
    bl =settings.set_bomb_level(-2)
    wl =settings.set_warn_level(-2)
    read.prm('/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm')

    # pycharmm.lingo.charmm_script(script1)
    
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script('bomlev 0')
    
    cgenff_params = open("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf").readlines()

    atc = pycharmm.param.get_atc()

    
    cgenff_params_dict_q = {}
    atom_name_to_param = {k: [] for k in atc}
    
    for _ in cgenff_params:
        if _.startswith("ATOM"):
            _, atomname, at, q = _.split()[:4]
            try:
                cgenff_params_dict_q[at] = float(q)
            except:
                cgenff_params_dict_q[at] = float(q.split("!")[0])
            atom_name_to_param[atomname] = at
    
    cgenff_params = open("/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm").readlines()
    cgenff_params_dict = {}
    
    for _ in cgenff_params:
        if len(_) > 5 and len(_.split()) > 4 and _.split()[1] == "0.0":
            res, _, ep, sig = _.split()[:4]
            if res in atc:
                cgenff_params_dict[res] = (float(ep), float(sig))

    for i, _ in enumerate(atc):
        print(i, _)

    params = list(range(len(atc)))
    print(params)
    
    atc_epsilons = [cgenff_params_dict[_][0] if _ in cgenff_params_dict.keys() else 0.0 for _ in atc ]
    atc_rmins = [cgenff_params_dict[_][1] if _ in cgenff_params_dict.keys() else 0.0 for _ in atc ]
    atc_qs = [cgenff_params_dict_q[_] if _ in cgenff_params_dict_q.keys() else 0.0 for _ in atc  ]
    at_ep = -1 * abs( np.array(atc_epsilons))
    at_rm = np.array(atc_rmins)
    at_q = np.array(atc_qs)

    print(at_ep[::10], np.array(at_ep).shape)
    print(at_rm[::10], np.array(at_rm).shape)
    print(at_q[::10], np.array(at_q).shape)
    # at_ep = {_: cgenff_params_dict[_][0] for _ in atc if _ in cgenff_params_dict.keys()]
    # atc_rmins = [cgenff_params_dict[_][1] for _ in atc if _ in cgenff_params_dict.keys()]
    # atc_qs = [cgenff_params_dict_q[_] for _ in atc if _ in cgenff_params_dict_q.keys()]
    ############################################################################################

    at_flat_q = np.array(atc_qs)
    at_flat_ep =  np.array(atc_epsilons)
    at_flat_rm =  np.array(atc_rmins)
    
    pair_idxs_product = jnp.array([(a,b) for a,b in list(product(np.arange(ATOMS_PER_MONOMER), repeat=2))])
    dimer_perms = jnp.array(dimer_permutations(N_MONOMERS))
    
    pair_idxs_np = dimer_perms * ATOMS_PER_MONOMER
    pair_idx_atom_atom = pair_idxs_np[:, None, :] + pair_idxs_product[None,...]
    pair_idx_atom_atom = pair_idx_atom_atom.reshape(-1, 2)
    
    displacements = R[pair_idx_atom_atom[:,0]] - R[pair_idx_atom_atom[:,1]]
    distances = jnp.linalg.norm(displacements, axis=1)
    at_perms = [_ for _ in list(product(params, repeat=2)) if _[0] <= _[1]]
    print("at_perms", at_perms)
    charges = np.array(psf.get_charges())[:N_MONOMERS*ATOMS_PER_MONOMER]
    masses = np.array(psf.get_amass())[:N_MONOMERS*ATOMS_PER_MONOMER]
    at_codes = np.array(psf.get_iac())[:N_MONOMERS*ATOMS_PER_MONOMER]
    atomtype_codes = np.array(psf.get_atype())[:N_MONOMERS*ATOMS_PER_MONOMER]

    print("at_codes", at_codes)
    print(list(set(at_codes)))
    print([len(_) for _ in [at_ep, at_rm, at_q] ])
    print(at_ep[list(set(at_codes))])
    print(at_rm[list(set(at_codes))])
    print(at_q[list(set(at_codes))])
    
    at_perms_ep = [ (at_ep[a] * at_ep[b])**0.5 for a,b in at_perms]
    at_perms_rm = [ (at_rm[a] + at_rm[b]) for a,b in at_perms]
    at_perms_qq = [ (at_q[a] * at_q[b]) for a,b in at_perms]

    rmins_per_system = jnp.take(at_flat_rm, at_codes) #jnp.array([ NBL["pair_rm"][k] for k in atom_keys ])
    epsilons_per_system = jnp.take(at_flat_ep, at_codes) #jnp.array([ NBL["pair_ep"][k] for k in atom_keys ])

    rs = distances
    q_per_system = jnp.take(at_flat_q, at_codes)


    q_a = jnp.take(q_per_system, pair_idx_atom_atom[:, 0])
    q_b = jnp.take(q_per_system, pair_idx_atom_atom[:, 1])
    
    rm_a = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 0])
    rm_b = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 1])
    
    ep_a = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 0])
    ep_b = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 1])

    pair_qq = q_a * q_b
    pair_rm = (rm_a + rm_b)
    pair_ep = (ep_a * ep_b)**0.5

    print("q", pair_qq)
    print("rm", pair_rm)
    print("ep", pair_ep)

    
    def lennard_jones(r, sig, ep):
        """
        rmin = 2^(1/6) * sigma
            https://de.wikipedia.org/wiki/Lennard-Jones-Potential
        Lennard-Jones potential for a pair of atoms
        """
        a = 6
        b = 2
        # sig = sig / (2 ** (1 / 6))
        r6 = (sig / r) ** a
        return ep * (r6 ** b - 2 * r6)
    
    coulombs_constant = 3.32063711e2 #Coulomb's constant kappa = 1/(4*pi*e0) in kcal-Angstroms/e^2.
    def coulomb(r, qq, constant = coulombs_constant):
        return constant * qq/r
    

    
    @jax.jit
    def apply_switching_function(
        positions: Array,  # Shape: (n_atoms, 3)
        pair_energies: Array,  # Shape: (n_pairs,)
        ml_cutoff_distance: float = 2.0,
        mm_switch_on: float = 5.0,
        mm_cutoff: float = 1.0,
        buffer_distance: float = 0.001,
    ) -> Array:
        """Applies smooth switching function to MM energies based on distances.
        
        Args:
            positions: Atomic positions
            pair_energies: Per-pair MM energies to be scaled
            ml_cutoff_distance: Distance where ML potential is cut off
            mm_switch_on: Distance where MM potential starts switching on
            mm_cutoff: Final cutoff for MM potential
            buffer_distance: Small buffer to avoid discontinuities
            
        Returns:
            Array: Scaled MM energies after applying switching function
        """
        # Calculate pairwise distances
        pair_positions = positions[pair_idx_atom_atom[:,0]] - positions[pair_idx_atom_atom[:,1]]
        distances = jnp.linalg.norm(pair_positions, axis=1)
        
        # Calculate switching functions
        ml_cutoff = mm_switch_on - ml_cutoff_distance
        switch_on = smooth_switch(distances, x0=ml_cutoff, x1=mm_switch_on)
        switch_off = 1 - smooth_switch(distances - mm_cutoff - mm_switch_on, 
                                     x0=ml_cutoff, 
                                     x1=mm_switch_on)
        cutoff = 1 - smooth_cutoff(distances, cutoff=2)
        
        # Combine switching functions and apply to energies
        switching_factor = switch_on * switch_off * cutoff
        scaled_energies = pair_energies * switching_factor
        
        return scaled_energies.sum()

    @jax.jit
    def calculate_mm_energy(positions: Array) -> Array:
        """Calculates MM energies including both VDW and electrostatic terms.
        
        Args:
            positions: Atomic positions (Shape: (n_atoms, 3))
            
        Returns:
            Array: Total MM energy
        """
        # Calculate pairwise distances
        displacements = positions[pair_idx_atom_atom[:,0]] - positions[pair_idx_atom_atom[:,1]]
        distances = jnp.linalg.norm(displacements, axis=1)
        
        # Only include interactions between unique pairs
        pair_mask = (pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])

        # Calculate VDW (Lennard-Jones) energies
        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
        vdw_total = vdw_energies.sum()
        
        # Calculate electrostatic energies
        electrostatic_energies = coulomb(distances, pair_qq) * pair_mask    
        electrostatic_total = electrostatic_energies.sum()
              
        return vdw_total + electrostatic_total

    @jax.jit
    def calculate_mm_pair_energies(positions: Array) -> Array:
        """Calculates per-pair MM energies for switching calculations.
        
        Args:
            positions: Atomic positions (Shape: (n_atoms, 3))
            
        Returns:
            Array: Per-pair energies (Shape: (n_pairs,))
        """
        displacements = positions[pair_idx_atom_atom[:,0]] - positions[pair_idx_atom_atom[:,1]]
        distances = jnp.linalg.norm(displacements, axis=1)
        pair_mask = (pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])
        
        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
        electrostatic_energies = coulomb(distances, pair_qq) * pair_mask    
              
        return vdw_energies + electrostatic_energies
    
    # Calculate gradients
    mm_energy_grad = jax.grad(calculate_mm_energy)
    switching_grad = jax.grad(apply_switching_function)

    @jax.jit 
    def calculate_mm_energy_and_forces(
        positions: Array,  # Shape: (n_atoms, 3)
    ) -> Tuple[Array, Array]:
        """Calculates MM energy and forces with switching.
        
        Args:
            positions: Atomic positions
            
        Returns:
            Tuple[Array, Array]: (Total energy, Forces per atom)
        """
        # Calculate base MM energies
        # mm_energy = calculate_mm_energy(positions)
        pair_energies = calculate_mm_pair_energies(positions)
        # Apply switching function
        switched_energy = apply_switching_function(positions, pair_energies)
        
        # Calculate forces with switching
        mm_forces = mm_energy_grad(positions)
        switching_forces = switching_grad(positions, pair_energies)
        total_forces = mm_forces * switching_forces

        return switched_energy, total_forces

    return calculate_mm_energy_and_forces


def prepare_batches_md(
    data,
    batch_size: int,
    data_keys = None,
    num_atoms: int = 60,
    dst_idx = None,
    src_idx= None,
    include_id: bool = False,
    debug_mode: bool = False,
) :
    """
    Efficiently prepare batches for training.

    Args:
        key: JAX random key for shuffling.
        data (dict): Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size (int): Size of each batch.
        data_keys (list, optional): List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms (int, optional): Number of atoms per example. Default is 60.
        dst_idx (jax.numpy.ndarray, optional): Precomputed destination indices for atom pairs.
        src_idx (jax.numpy.ndarray, optional): Precomputed source indices for atom pairs.
        include_id (bool, optional): Whether to include 'id' key if present in data.
        debug_mode (bool, optional): If True, run assertions and extra checks.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """

    # -------------------------------------------------------------------------
    # Validation and Setup
    # -------------------------------------------------------------------------

    # Check for mandatory keys
    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    # Default to all keys in data if none provided
    if data_keys is None:
        data_keys = list(data.keys())

    # Verify data sizes
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    # -------------------------------------------------------------------------
    # Compute Random Permutation for Batches
    # -------------------------------------------------------------------------
    # perms = jax.random.permutation(key, data_size)
    perms = jnp.arange(0, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # -------------------------------------------------------------------------
    # Precompute Batch Segments and Indices
    # -------------------------------------------------------------------------
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    # Compute pairwise indices only if not provided
    # E3x: e3x.ops.sparse_pairwise_indices(num_atoms) -> returns (dst_idx, src_idx)
    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Adjust indices for batching
    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    # Centralize reshape logic
    # For keys not listed here, we default to their original shape after indexing.
    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size,3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []

    # -------------------------------------------------------------------------
    # Batch Preparation Loop
    # -------------------------------------------------------------------------
    for perm in perms:
        # Build the batch dictionary
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][jnp.array(perm)]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                batch[k] = v

        # Optionally include 'id' if requested and present
        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][jnp.array(perm)]

        # Compute good_indices (mask for valid atom pairs)
        # Vectorized approach: We know N is shape (batch_size,)
        # Expand N to compare with dst_idx/src_idx
        # dst_idx[i], src_idx[i] range over atom pairs within the ith example
        # Condition: (dst_idx[i] < N[i]+i*num_atoms) & (src_idx[i] < N[i]+i*num_atoms)
        # We'll compute this for all i and concatenate.
        N = batch["N"]
        # Expand N and offsets for comparison
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

        # Add metadata to the batch
        atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
        batch.update(
            {
                "dst_idx": dst_idx.flatten(),
                "src_idx": src_idx.flatten(),
                "batch_mask": good_indices,
                "batch_segments": batch_segments,
                "atom_mask": atom_mask,
            }
        )

        # Debug checks
        if debug_mode:
            # Check expected shapes
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            assert batch["F"].shape == (
                batch_size * num_atoms,
                3,
            ), f"F shape mismatch: {batch['F'].shape}"
            assert batch["E"].shape == (
                batch_size,
                1,
            ), f"E shape mismatch: {batch['E'].shape}"
            assert batch["Z"].shape == (
                batch_size * num_atoms,
            ), f"Z shape mismatch: {batch['Z'].shape}"
            assert batch["N"].shape == (
                batch_size,
            ), f"N shape mismatch: {batch['N'].shape}"
            # Optional: print or log if needed

        output.append(batch)

    return output



# switch_MM_grad = jax.grad(switch_MM)


class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in kcal/mol
    forces: Array  # Shape: (n_atoms, 3), forces in kcal/mol/Å
    dH: Array # Shape: (,), total interaction energy in kcal/mol
    internal_E: Array # Shape: (,) total internal energy in kcal/mol
    internal_F: Array
    mm_E: Array
    mm_F: Array
    ml_2b_E: Array
    ml_2b_F: Array

def get_spherical_cutoff_calculator(
    atomic_numbers: Array,  # Shape: (n_atoms,)
    atomic_positions: Array,  # Shape: (n_atoms, 3)
    n_monomers: int,
    restart_path: str = "/path/to/default",
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    backprop: bool = False,
    debug: bool = False


) -> Any:  # Returns ASE calculator
    """Creates a calculator that combines ML and MM potentials with spherical cutoffs.
    
    This calculator handles:
    1. ML predictions for close-range interactions
    2. MM calculations for long-range interactions
    3. Smooth switching between the two regimes
    
    Args:
        atomic_numbers: Array of atomic numbers for each atom
        atomic_positions: Initial positions of atoms in Angstroms
        restart_path: Path to model checkpoint for ML component
        
    Returns:
        ASE-compatible calculator that computes energies and forces
    """

    all_dimer_idxs = []
    for a, b in dimer_permutations(n_monomers):
        all_dimer_idxs.append(indices_of_pairs(a + 1, b + 1))

    all_monomer_idxs = []
    for a in range(1, n_monomers + 1):
        all_monomer_idxs.append(indices_of_monomer(a))
        
    unique_res_ids = []
    collect_monomers = []
    dimer_perms = dimer_permutations(n_monomers)
    for i, _ in enumerate(dimer_perms):
        a,b = _
        if a not in unique_res_ids and b not in unique_res_ids:
            unique_res_ids.append(a)
            unique_res_ids.append(b)
            collect_monomers.append(1)
            print(a,b)
        else:
            collect_monomers.append(0)

    N_MONOMERS = n_monomers
    # Batch processing constants
    BATCH_SIZE: int = N_MONOMERS + len(dimer_perms) #MAX_ATOMS_PER_SYSTEM * N_MONOMERS + (MAX_ATOMS_PER_SYSTEM) * len(dimer_perms) # Number of systems per batch
    print(BATCH_SIZE)
    restart_path = Path("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-7c36e6f9-6f10-4d21-bf6d-693df9b8cd40")
    
    """Initialize monomer and dimer models from restart"""
    restart = get_last(restart_path)
    
    # Setup monomer model
    params, MODEL = get_params_model(restart)
    MODEL.natoms = 10
    # MODEL.charges = False

    def calc_dimer_energy_forces(R, Z, i, ml_e, ml_f):
        a,b = dimer_perms[i]
        a,b = all_monomer_idxs[a], all_monomer_idxs[b]
        idxs = np.array([a, b], dtype=int).flatten()
        # print(idxs)
        _R = R[idxs]
        # print(_R)
        final_energy = ml_e
        val_ml_s = switch_ML(_R, final_energy)  # ML switching value
        grad_ml_s = switch_ML_grad(_R, final_energy)  # ML switching gradient
        # Combine forces with switching functions
        ml_forces_out = ml_f * -grad_ml_s #ml_f * val_ml_s + grad_ml_s * final_energy 
        # final_forces = ml_f + grad_ml_s
        # Combine all force contributions for final forces
        # final_forces = ml_f + grad_ml_s #ml_forces_out #+ mm_forces_out #+ ase_dimers_1body_forces
        
        outdict = {
            "energy": val_ml_s,
            "forces": ml_forces_out,
        }
        return outdict

    MM_energy_and_gradient = get_MM_energy_forces_fns(atomic_positions, ATOMS_PER_MONOMER, n_monomers)


    def get_energy_fn(
        atomic_numbers: Array,  # Shape: (n_atoms,)
        positions: Array,  # Shape: (n_atoms, 3)
        BATCH_SIZE,
    ) -> Tuple[Any, Dict[str, Array]]:
        """Prepares the ML model and batching for energy calculations.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Atomic positions in Angstroms
            
        Returns:
            Tuple of (model_apply_fn, batched_inputs)
        """
        batch_data: Dict[str, Array] = {}
        
        # Prepare monomer data
        n_monomers = len(all_monomer_idxs)
        # Position of the atoms in the monomer
        monomer_positions = jnp.zeros((n_monomers, MAX_ATOMS_PER_SYSTEM, SPATIAL_DIMS))
        monomer_positions = monomer_positions.at[:, :ATOMS_PER_MONOMER].set(
            positions[jnp.array(all_monomer_idxs)]
        )
        # Atomic numbers of the atoms in the monomer
        monomer_atomic = jnp.zeros((n_monomers, MAX_ATOMS_PER_SYSTEM), dtype=jnp.int32)
        monomer_atomic = monomer_atomic.at[:, :ATOMS_PER_MONOMER].set(
            atomic_numbers[jnp.array(all_monomer_idxs)]
        )
        
        # Prepare dimer data
        n_dimers = len(all_dimer_idxs)
        # Position of the atoms in the dimer
        dimer_positions = jnp.zeros((n_dimers, MAX_ATOMS_PER_SYSTEM, SPATIAL_DIMS))
        dimer_positions = dimer_positions.at[:].set(
            positions[jnp.array(all_dimer_idxs)]
        )
        # Atomic numbers of the atoms in the dimer
        dimer_atomic = jnp.zeros((n_dimers, MAX_ATOMS_PER_SYSTEM), dtype=jnp.int32)
        dimer_atomic = dimer_atomic.at[:].set(
            atomic_numbers[jnp.array(all_dimer_idxs)]
        )
        
        # Combine monomer and dimer data
        batch_data["R"] = jnp.concatenate([monomer_positions, dimer_positions])
        batch_data["Z"] = jnp.concatenate([monomer_atomic, dimer_atomic])
        batch_data["N"] = jnp.concatenate([
            jnp.full((n_monomers,), ATOMS_PER_MONOMER),
            jnp.full((n_dimers,), MAX_ATOMS_PER_SYSTEM)
        ])
        
        batches = prepare_batches_md(batch_data, batch_size=BATCH_SIZE, num_atoms=MAX_ATOMS_PER_SYSTEM)[0]
        
        @jax.jit
        def apply_model(
            atomic_numbers: Array,  # Shape: (batch_size * num_atoms,)
            positions: Array,  # Shape: (batch_size * num_atoms, 3)
        ) -> Dict[str, Array]:
            """Applies the ML model to batched inputs.
            
            Args:
                atomic_numbers: Batched atomic numbers
                positions: Batched atomic positions
                
            Returns:
                Dictionary containing 'energy' and 'forces'
            """
            return MODEL.apply(
                params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                dst_idx=batches["dst_idx"],
                src_idx=batches["src_idx"],
                batch_segments=batches["batch_segments"],
                batch_size=BATCH_SIZE,
                batch_mask=batches["batch_mask"],
                atom_mask=batches["atom_mask"]
            )
    
        return apply_model, batches

    @jax.jit
    def spherical_cutoff_calculator(
        positions: Array,  # Shape: (n_atoms, 3)
        atomic_numbers: Array,  # Shape: (n_atoms,)
    ) -> ModelOutput:
        """Calculates energy and forces using combined ML/MM potential.
        
        Handles:
        1. ML predictions for each monomer and dimer
        2. MM long-range interactions
        3. Smooth switching between regimes
        
        Args:
            positions: Atomic positions in Angstroms
            atomic_numbers: Atomic numbers of each atom
            
        Returns:
            ModelOutput containing total energy and forces
        """
        # n_monomers = 20
        n_dimers = len(dimer_permutations(n_monomers))
        output_list: List[Dict[str, Array]] = []
        out_E = 0
        out_F = 0
        dH = 0
        internal_E = 0
        internal_F = 0
        ml_2b_E = 0
        ml_2b_F = 0
        
        if doML:
            # print("doML")
            apply_model, batches = get_energy_fn(atomic_numbers, positions, BATCH_SIZE)
            
            output = apply_model(batches["Z"], batches["R"])

            # convert to kcal/mol bc electrostatics use CHARMM 
            f = output["forces"] / (ase.units.kcal/ase.units.mol)
            e = output["energy"] / (ase.units.kcal/ase.units.mol)
           
            # energies from a batch of monomers and dimers
            ml_monomer_energy = jnp.array(e[:n_monomers]).flatten()

            # forces from a batch of monomers and dimers
            monomer_idx_max = MAX_ATOMS_PER_SYSTEM * n_monomers
            dimer_idx_max = MAX_ATOMS_PER_SYSTEM * n_dimers + monomer_idx_max


            ml_monomer_forces = f[:monomer_idx_max]
            ml_dimer_forces = f[monomer_idx_max:dimer_idx_max]


            monomer_segment_idxs = jnp.concatenate([
                jnp.arange(ATOMS_PER_MONOMER) + i * ATOMS_PER_MONOMER 
                for i in range(n_monomers)
            ])
 
            
            # Ensure monomer forces are properly shaped and masked
            monomer_forces = ml_monomer_forces.reshape(n_monomers, MAX_ATOMS_PER_SYSTEM, 3)
            atom_mask = jnp.arange(MAX_ATOMS_PER_SYSTEM)[None, :] < ATOMS_PER_MONOMER
 
            # Apply mask and reshape
            monomer_forces = jnp.where(
                atom_mask[..., None],
                monomer_forces,
                0.0
            )
            
            # Sum forces for valid atoms only
            monomer_forces = jax.ops.segment_sum(
                monomer_forces[:, :ATOMS_PER_MONOMER].reshape(-1, 3),
                monomer_segment_idxs,
                num_segments=n_monomers * ATOMS_PER_MONOMER
            )

            
            out_F +=monomer_forces
            out_E += ml_monomer_energy.sum()
            internal_E += ml_monomer_energy.sum()
            internal_F += monomer_forces
            
            if debug:
                print("doML")
                # print("monomer_segment_idxs", monomer_segment_idxs)
                # jax.debug.print("monomer_segment_idxs\n{x}", x=monomer_segment_idxs)
                # print("atom_mask", atom_mask)
                # jax.debug.print("atom_mask\n{x}", x=atom_mask)
                # print("monomer_forces", monomer_forces)
                # jax.debug.print("monomer_forces\n{x}", x=monomer_forces)
                # print("f", f.shape)
                # print("e", e.shape)
                # print("n_monomers", n_monomers)
                # print("N_ATOMS_MONOMER", ATOMS_PER_MONOMER)
                # print("ml_monomer_energy", ml_monomer_energy.shape)
                # print("ml_dimer_energy", ml_dimer_energy.shape)
                # print("monomer_segment_idxs", monomer_segment_idxs.shape)
                # jax.debug.print("monomer_segment_idxs\n{x}", x=monomer_segment_idxs)
                # print("ml_monomer_forces", ml_monomer_forces.shape)
                # print("ml_dimer_forces", ml_dimer_forces.shape)
                # jax.debug.print("ml_monomer_forces\n{x}", x=ml_monomer_forces)
                # jax.debug.print("ml_dimer_forces\n{x}", x=ml_dimer_forces)
                # print("monomer_idx_max", monomer_idx_max)
                # print("dimer_idx_max", dimer_idx_max)
                # print("ml_monomer_forces_sum", out_F.shape)
                # jax.debug.print("out_F\n{x}", x=out_F)


            if doML_dimer:
                
                ml_dimer_energy = jnp.array(e[n_monomers:]).flatten() # shape (n_dimers)
                # Create segment indices for dimers
                dimer_pairs = jnp.array(dimer_perms)
                # Calculate base indices for each monomer in the dimers
                first_monomer_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 0:1]  # Shape: (n_dimers, 1)
                second_monomer_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 1:2]  # Shape: (n_dimers, 1)
                # Create atom offsets for each monomer
                atom_offsets = jnp.arange(ATOMS_PER_MONOMER)  # Shape: (ATOMS_PER_MONOMER,)
                # Combine indices for both monomers in each dimer
                force_segments = jnp.concatenate([
                    first_monomer_indices + atom_offsets[None, :],   # Add offsets to first monomer
                    second_monomer_indices + atom_offsets[None, :]   # Add offsets to second monomer
                ], axis=1)  # Shape: (n_dimers, 2*ATOMS_PER_MONOMER)
                # Flatten the segments
                force_segments = force_segments.reshape(-1)  # Shape: (n_dimers * 2*ATOMS_PER_MONOMER)
                # # Create validity mask for the segments
                # valid_segments = (force_segments >= 0) & (force_segments < n_monomers * ATOMS_PER_MONOMER)
                # # Zero out invalid segments
                # force_segments = jnp.where(valid_segments, force_segments, 0)
                
                monomer_contrib_to_dimer_energy = ml_monomer_energy[dimer_pairs[:, 0]] + ml_monomer_energy[dimer_pairs[:, 1]]
                dimer_int_energies = ml_dimer_energy - monomer_contrib_to_dimer_energy

                # Calculate interaction forces
                dimer_int_forces = ml_dimer_forces.reshape(n_dimers, MAX_ATOMS_PER_SYSTEM, 3) 
                
                summed_dimer_int_forces = jax.ops.segment_sum(
                    dimer_int_forces.reshape(-1, 3),
                    force_segments,
                    num_segments=n_monomers * ATOMS_PER_MONOMER
                ) 
                
                switched_energy = jax.vmap(lambda x, f: switch_ML(x.reshape(MAX_ATOMS_PER_SYSTEM, 3), f))(
                    positions[jnp.array(all_dimer_idxs)],
                    dimer_int_energies
                )
                summed_switched_dimer_int_energies = switched_energy.sum()

                
                switched_energy_grad = jax.vmap(lambda x, f: switch_ML_grad(x.reshape(MAX_ATOMS_PER_SYSTEM, 3), f))(
                    positions[jnp.array(all_dimer_idxs)],
                    dimer_int_energies
                )
                # Perform segmented sum with validated indices and forces
                summed_switched_dimer_int_grad = jax.ops.segment_sum(
                    switched_energy_grad.reshape(-1, 3),
                    force_segments,
                    num_segments=n_monomers * ATOMS_PER_MONOMER
                ) 
                # Create atom existence mask
                atom_mask = jnp.arange(MAX_ATOMS_PER_SYSTEM)[None, :] < ATOMS_PER_MONOMER
                
                original_dimer_int_energies = dimer_int_energies.sum()

                # combine with product rule
                # d(f1*f2)/dx = f1*df2/dx + f2*df1/dx
                dudx_v = original_dimer_int_energies*summed_switched_dimer_int_grad 
                dvdx_u = summed_dimer_int_forces /summed_switched_dimer_int_energies
                combined_forces =  -1 * (dudx_v + dvdx_u)
               
                out_E += summed_switched_dimer_int_energies.sum()
                out_F += combined_forces
                
                dH += summed_switched_dimer_int_energies 
                ml_2b_E += summed_switched_dimer_int_energies.sum()
                ml_2b_F += combined_forces
                
                if debug:
                    print("doML_dimer")
                    print("switched_forces", summed_switched_dimer_int_grad.shape) 
                    jax.debug.print("switched_forces\n{x}", x=summed_switched_dimer_int_grad)
                    print("switched_energies", summed_switched_dimer_int_energies.sum().shape)
                    jax.debug.print("switched_energies\n{x}", x=summed_switched_dimer_int_energies.sum())
                    print("dudx_v", dudx_v.shape)
                    jax.debug.print("dudx_v\n{x}", x=dudx_v)
                    print("dvdx_u", dvdx_u.shape)
                    print("dvdx_u", dvdx_u.shape)
                    jax.debug.print("dvdx_u\n{x}", x=dvdx_u)
                    # jax.debug.print("ml_monomer_forces\n{x}", x=ml_monomer_forces)
                    # print("force_segments", force_segments.shape)
                    # jax.debug.print("force_segments\n{x}", x=force_segments)
                    # print("dimer_int_energies", dimer_int_energies.shape)
                    # jax.debug.print("dimer_int_energies\n{x}", x=dimer_int_energies)

        if doMM:
            # print("doMM")
            # MM energy and forces
            mm_E, mm_grad = MM_energy_and_gradient(positions)
            out_E += mm_E
            out_F += mm_grad
            dH += mm_E
            
            if debug:
                print("doMM")
                print("mm_E", mm_E.shape)
                jax.debug.print("mm_E\n{x}", x=mm_E)
                print("mm_grad", mm_grad.shape)
                jax.debug.print("mm_grad\n{x}", x=mm_grad)


        return ModelOutput(energy=out_E.sum() * (ase.units.kcal/ase.units.mol),
                           forces=out_F  * (ase.units.kcal/ase.units.mol), 
                           dH=dH * (ase.units.kcal/ase.units.mol), 
                           ml_2b_E=ml_2b_E * (ase.units.kcal/ase.units.mol),
                           ml_2b_F=ml_2b_F * (ase.units.kcal/ase.units.mol),
                           internal_E=internal_E * (ase.units.kcal/ase.units.mol),   
                           internal_F=internal_F * (ase.units.kcal/ase.units.mol),
                           mm_E = mm_E * (ase.units.kcal/ase.units.mol),
                           mm_F = mm_grad * (ase.units.kcal/ase.units.mol))

    def just_E(R, Z):
        return spherical_cutoff_calculator(R, Z).energy

    just_E_grad = jax.grad(just_E)

    class AseDimerCalculator(ase_calc.Calculator):
        implemented_properties = ["energy", "forces", "out"]

        def calculate(
            self, atoms, properties, system_changes=ase.calculators.calculator.all_changes
        ):
            ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
            R = atoms.get_positions()
            Z = atoms.get_atomic_numbers()
            out = spherical_cutoff_calculator(R, Z)
            if backprop:
                E = out.energy
                F = -just_E_grad(R, Z)
            else:
                E = out.energy
                F = out.forces
                self.results["out"] = out
            self.results["energy"] = E 
            self.results["forces"] = F
            

    return AseDimerCalculator(), spherical_cutoff_calculator
import matplotlib.pyplot as plt

# def validate_forces(forces):
#     """Validate forces and replace NaNs with zeros."""
#     is_valid = jnp.isfinite(forces).all()
#     if not is_valid:
#         print("Warning: Found invalid forces")
#         forces = jnp.nan_to_num(forces, 0.0)
#     return forces

# # Use in spherical_cutoff_calculator
# out_F = validate_forces(out_F)

Eref = np.zeros([20], dtype=float)
Eref[1] = -0.498232909223
Eref[6] = -37.731440432799
Eref[8] = -74.878159582108
Eref[17] = -459.549260062932