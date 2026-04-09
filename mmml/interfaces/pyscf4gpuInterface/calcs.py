import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pyscf
from pyscf.hessian import thermo
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.lib.cupy_helper import dist_matrix
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import ir, shielding, polarizability

import cupy

from mmml.interfaces.pyscf4gpuInterface.enums import *
from mmml.interfaces.pyscf4gpuInterface.helperfunctions import *
from mmml.interfaces.pyscf4gpuInterface.esp_helpers import balance_array


def _RZ_to_atom(R, Z):
    """Convert R (n_atoms, 3) in Angstrom and Z (n_atoms) to PySCF atom list [(symbol, (x,y,z)), ...].

    Z may be atomic numbers (integer-like) or chemical symbols (str / bytes), as in ASE NPZ output.
    """
    from ase.data import chemical_symbols

    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z)

    atoms = []
    for z, row in zip(np.ravel(Z), R):
        if isinstance(z, (bytes, np.bytes_)):
            sym = z.decode("ascii", errors="replace").strip()
        else:
            try:
                zi = int(z)
            except (TypeError, ValueError):
                sym = str(z).strip()
            else:
                if zi < 0 or zi >= len(chemical_symbols):
                    raise ValueError(f"Invalid atomic number: {zi}")
                sym = chemical_symbols[zi]
        atoms.append((sym, tuple(row)))
    return atoms


def setup_mol(atoms, basis, xc, spin, charge, log_file='./pyscf.log', 
    verbose=6, 
    lebedev_grids=(99,590),
    scf_tol=1e-10,
    scf_max_cycle=50,
    cpscf_tol=1e-3,
    conv_tol=1e-10,
    conv_tol_cpscf=1e-3,
    ):
    _mol_kw = dict(
        basis=basis,
        spin=spin,
        charge=charge,
        output=log_file,
        verbose=verbose,
    )
    if isinstance(atoms, str):
        path = Path(atoms).expanduser()
        if path.is_file():
            import ase.io

            ase_atoms = ase.io.read(str(path))
            atom_spec = _RZ_to_atom(
                ase_atoms.get_positions(),
                np.asarray(ase_atoms.get_atomic_numbers()),
            )
            mol = pyscf.M(atom=atom_spec, unit="Ang", **_mol_kw)
        else:
            suff = path.suffix.lower()
            if suff in (".xyz", ".mol", ".gjf", ".com", ".cif", ".extxyz"):
                raise FileNotFoundError(
                    f"--mol file not found: {path.resolve()}. "
                    "Pass a path relative to the job working directory, or an absolute path."
                )
            mol = pyscf.M(atom=atoms, **_mol_kw)
    else:
        mol = atoms
        
    mf_GPU = rks.RKS(                      # restricted Kohn-Sham DFT
        mol,                               # pyscf.gto.object
        xc=xc                         # xc funtionals, such as pbe0, wb97m-v, tpss,
        ).density_fit()                    # density fitting

    mf_GPU.grids.atom_grid = lebedev_grids      # (99,590) lebedev grids, (75,302) is often enough
    mf_GPU.conv_tol = scf_tol                   # controls SCF convergence tolerance
    mf_GPU.max_cycle = scf_max_cycle            # controls max iterations of SCF
    mf_GPU.conv_tol_cpscf = conv_tol_cpscf      # controls max iterations of CPSCF (for hessian)

    return mf_GPU, mol



def compute_dft(args, calcs, extra=None):

    engine, mol = setup_mol(args.mol, args.basis, args.xc, args.spin, args.charge)

    print(mol)
    from mmml.interfaces.pyscf4gpuInterface.helperfunctions import print_basis
    print_basis(mol)

    opt_callback = None

    if CALCS.OPTIMIZE in calcs:
        print("-"*100)
        print("Optimizing geometry")
        print("-"*100)
        from pyscf.geomopt.geometric_solver import optimize
        import time

        
        gradients = []
        energies = []
        coords = []
        def callback(envs):
            print(list(envs.keys()))
            gradients.append(envs['gradients'])
            energies.append(envs['energy'])
            coords.append(envs['coords'])

        start_time = time.time()
        mol = optimize(engine, maxsteps=20, callback=callback)
        print("Optimized coordinate:")
        print(mol.atom_coords())
        print('Geometry optimization took', time.time() - start_time, 's')
        opt_callback = {
            'gradients': gradients,
            'energies': energies,
            'coords': coords
        }

    output = {"mol": mol, "calcs": calcs, "opt_callback": opt_callback}

    if CALCS.ENERGY in calcs:
        print("-"*100)
        print("Computing Energy")
        print("-"*100)
        # Compute Energy
        e_dft = engine.kernel()
        print(f"total energy = {e_dft}")       
        output['energy'] = e_dft

    if CALCS.DENS_ESP in calcs:
        print("-"*100)
        print("Computing Density ESP")
        print("-"*100)
        print('------------------ Density ----------------------------')
        dm = engine.make_rdm1()
        grids = engine.grids
        grid_coords = grids.coords.get()
        density = engine._numint.get_rho(mol, dm, grids).get()
        print('------------------ Selecting points ----------------------------')
        grid_indices = np.where(np.less(density, 0.001) & np.greater_equal(density, 0.0001))[0]
        print(grid_indices)
        # grid_positions_a = grid_coords[cupy.where(density < 0.001)[0]]
        grid_positions_a = grid_coords[grid_indices]
        print(grid_positions_a.shape)
        print(grid_positions_a.min(), grid_positions_a.max())
        print('------------------ ESP ----------------------------')
        dm = engine.make_rdm1()  # compute one-electron density matrix
        coords = grid_positions_a  # in Bohr (PySCF grids convention)
        print(coords.shape)
        # Use CPU int1e_rinv for correct esp/grid alignment (gpu4pyscf int3c2e bug)
        dm_np = dm.get() if hasattr(dm, "get") else np.asarray(dm)
        res = _compute_esp_grid_cpu(mol, dm_np, coords)
        output['esp'] = res
        output['esp_grid'] = np.asarray(coords)
        output['R'] = mol.atom_coords(unit="ANG")
        output['Z'] = mol.atom_charges()
        output['D'] = engine.dip_moment(unit="DEBYE", dm=dm, verbose=0)
        output['Q'] = engine.quad_moment(unit="DEBYE-ANG", dm=dm)
        output['density'] = density
        output['density_grid'] = grid_coords


    if CALCS.GRADIENT in calcs:
        print("-"*100)
        print("Computing Gradient")
        print("-"*100)
        # Compute Gradient
        g = engine.nuc_grad_method()
        g_dft = g.kernel()
        output['gradient'] = g_dft

    if CALCS.HESSIAN in calcs:
        print("-"*100)
        print("Computing Hessian")
        print("-"*100)
        # Compute Hessian
        h = engine.Hessian()
        h.auxbasis_response = 2                # 0: no aux contribution, 1: some contributions, 2: all
        engine.cphf_grids.atom_grid = (50,194) # customize grids for solving CPSCF equation, SG1 by default
        h_dft = h.kernel()
        output['hessian'] = h_dft

    if CALCS.HARMONIC in calcs:
        print("-"*100)
        print("Computing Harmonic Analysis")
        print("-"*100)
        # harmonic analysis
        harmonic_results = thermo.harmonic_analysis(mol, h_dft)
        thermo.dump_normal_mode(mol, harmonic_results)
        output['harmonic'] = harmonic_results

    if CALCS.IR in calcs:
        assert CALCS.HESSIAN in calcs, "Hessian must be computed for IR"
        print("-"*100)
        print("Computing IR")
        print("-"*100)
        freq, intensity = ir.eval_ir_freq_intensity(engine, h)
        output['freq'] = freq
        output['intensity'] = intensity

    if CALCS.IR_EFIELD in calcs:
        print("-"*100)
        print("Computing IR under external electric field (E-field scan)")
        print("-"*100)
        from mmml.interfaces.pyscf4gpuInterface.efield import (
            efield_ir_scan,
            efield_response_finite_difference,
            parse_efield_points,
        )

        spec = getattr(args, "efield_points", None) or "0,0,0"
        efields = parse_efield_points(spec)
        fd_axis = int(getattr(args, "efield_fd_axis", 2))
        print(f"  Field points (a.u.): {efields.shape[0]} rows; FD axis index = {fd_axis}")
        inc_nuc = getattr(args, "efield_include_nuclear_energy", True)
        scan = efield_ir_scan(
            mol, efields, xc=args.xc, include_nuclear_field_energy=inc_nuc
        )
        scan["efield_response_fd"] = efield_response_finite_difference(
            scan, axis=fd_axis
        )
        output["efield_ir_scan"] = scan
        output["efield_Ef"] = scan["Ef"]
        output["efield_energy"] = scan["energy"]
        output["efield_D_au"] = scan["D_au"]
        if scan.get("polarizability") is not None:
            output["efield_polarizability"] = scan["polarizability"]
        output["efield_response_fd"] = scan["efield_response_fd"]

    if CALCS.EFIELD_SCF in calcs:
        print("-" * 100)
        print("Computing SCF in uniform E-field (energy, dipole, forces — no Hessian/IR)")
        print("-" * 100)
        from mmml.interfaces.pyscf4gpuInterface.efield import (
            efield_response_finite_difference,
            efield_scf_scan,
            parse_efield_points,
        )

        spec = getattr(args, "efield_points", None) or "0,0,0"
        efields = parse_efield_points(spec)
        fd_axis = int(getattr(args, "efield_fd_axis", 2))
        dip_u = getattr(args, "efield_dipole_unit", "DEBYE")
        do_forces = not getattr(args, "efield_scf_no_forces", False)
        print(
            f"  Field points (a.u.): {efields.shape[0]}; dipole unit={dip_u}; "
            f"forces={'on' if do_forces else 'off'}"
        )
        scan = efield_scf_scan(
            mol,
            efields,
            xc=args.xc,
            dipole_unit=dip_u,
            forces=do_forces,
            include_nuclear_field_energy=getattr(
                args, "efield_include_nuclear_energy", True
            ),
        )
        scan["efield_response_fd"] = efield_response_finite_difference(
            scan, axis=fd_axis
        )
        output["efield_scf_scan"] = scan
        output["efield_scf_Ef"] = scan["Ef"]
        output["efield_scf_energy"] = scan["energy"]
        output["efield_scf_D_au"] = scan["D_au"]
        D_rows = [
            np.asarray(s["D"], dtype=np.float64).ravel()[:3] for s in scan["summaries"]
        ]
        output["efield_scf_D"] = np.stack(D_rows, axis=0)
        if do_forces and "F" in scan:
            output["efield_scf_F"] = scan["F"]
        output["efield_scf_response_fd"] = scan["efield_response_fd"]

    if CALCS.SHIELDING in calcs:
        assert CALCS.ENERGY in calcs, "Energy must be computed for shielding"
        print("-"*100)
        print("Computing Shielding")
        print("-"*100)
        msc_d, msc_p = shielding.eval_shielding(engine)
        msc = (msc_d + msc_p).get()
        output['shielding'] = msc

    if CALCS.POLARIZABILITY in calcs:
        assert CALCS.ENERGY in calcs, "Energy must be computed for polarizability"
        print("-"*100)
        print("Computing Polarizability")
        print("-"*100)
        polar = polarizability.eval_polarizability(engine)
        output['polarizability'] = polar

    if CALCS.THERMO in calcs:
        assert CALCS.HARMONIC in calcs, "Harmonic must be computed for thermodynamics"
        print("-"*100)
        print("Computing Thermodynamics")
        print("-"*100)
        results = thermo.thermo(
            engine,                            # GPU4PySCF object
            harmonic_results['freq_au'],
            298.15,                            # room temperature
            101325)                            # standard atmosphere

        thermo.dump_thermo(mol, results)
        output['thermo'] = results


    if CALCS.INTERACTION in calcs:
        print("-"*100)
        print("Computing Interaction Energy")
        print("-"*100)
        # Compute Interaction Energy
        e_interaction = compute_interaction_energy(mol, extra)
        output['interaction'] = e_interaction

    # Always add R, Z, N for ML compatibility (from final mol, e.g. after optimization)
    mol_final = output.get('mol', mol)
    R = np.asarray(mol_final.atom_coords(unit="ANG"))
    Z = np.asarray(mol_final.atom_charges())
    output['R'] = R
    output['Z'] = Z
    output['N'] = np.array(len(Z))

    return output


def _compute_esp_grid_cpu(mol, dm, grid_bohr: np.ndarray) -> np.ndarray:
    """
    Compute ESP at grid points using CPU int1e_rinv.
    Guaranteed correct esp[i] <-> grid[i] alignment (avoids gpu4pyscf int3c2e ordering bug).
    Returns ESP in Hartree/e.
    """
    dm_np = np.asarray(dm) if hasattr(dm, "get") else np.asarray(dm)
    coords = mol.atom_coords(unit="Bohr")
    charges = mol.atom_charges()
    esp = np.zeros(len(grid_bohr), dtype=np.float64)
    for i, r in enumerate(grid_bohr):
        dr = coords - r[None, :]
        dist = np.linalg.norm(dr, axis=1) + 1e-12
        v_nuc = np.sum(charges / dist)
        with mol.with_rinv_origin(r):
            v_mat = mol.intor("int1e_rinv")
        v_elec = np.einsum("ij,ij", dm_np, v_mat)
        esp[i] = v_nuc - v_elec
    return esp


def compute_dft_single(
    R: np.ndarray,
    Z: np.ndarray,
    basis: str = "def2-SVP",
    xc: str = "PBE0",
    spin: int = 0,
    charge: int = 0,
    energy: bool = True,
    gradient: bool = True,
    dipole: bool = True,
    dens_esp: bool = False,
    compute_polarizability: bool = False,
    esp_cpu_fallback: bool = False,
    verbose: int = 0,
    efield: np.ndarray | None = None,
    efield_include_nuclear_energy: bool = True,
) -> dict:
    """
    Run DFT for a single geometry (R, Z). Used for batch evaluation.

    Returns dict with energy, gradient, D (dipole), esp, esp_grid (if dens_esp), R, Z, N.
    All in same process/GPU context for efficient batch loops.

    efield
        If set, shape (3,) electric field in **atomic units** (uniform field in the
        Hamiltonian). SCF uses hcore + E·μ; dipole/gradient/ESP use the converged
        field-polarized state.
    efield_include_nuclear_energy
        If True (default, with ``efield``), add nuclear-field energy to ``energy`` after SCF
        (:func:`mmml.interfaces.pyscf4gpuInterface.efield.nuclear_field_energy_correction_hartree`).
    """
    atom = _RZ_to_atom(R, Z)
    mol = pyscf.M(
        atom=atom,
        basis=basis,
        spin=spin,
        charge=charge,
        unit="Angstrom",
        verbose=verbose,
    )
    engine, mol = setup_mol(mol, basis, xc, spin, charge, verbose=verbose)

    efield = None if efield is None else np.asarray(efield, dtype=np.float64).reshape(3)

    # SCF in uniform electric field (modified core Hamiltonian)
    if efield is not None:
        from mmml.interfaces.pyscf4gpuInterface.efield import run_scf_uniform_efield

        mf, e_tot = run_scf_uniform_efield(
            efield,
            mol,
            xc=xc,
            include_nuclear_field_energy=efield_include_nuclear_energy,
        )
        out: dict = {}
        if energy:
            out["energy"] = np.array(e_tot)
        out["R"] = np.asarray(R)
        out["Z"] = np.asarray(Z)
        out["N"] = np.array(len(Z))
        out["Ef"] = efield.copy()

        if gradient:
            g = mf.nuc_grad_method()
            g_dft = g.kernel()
            out["gradient"] = _to_numpy(g_dft)

        if dipole:
            dm = mf.make_rdm1()
            out["D"] = _to_numpy(mf.dip_moment(unit="DEBYE", dm=dm, verbose=0))

        if dens_esp:
            dm = mf.make_rdm1()
            grids = mf.grids
            grid_coords = grids.coords.get()
            density = mf._numint.get_rho(mol, dm, grids).get()
            grid_indices = np.where(
                np.less(density, 0.001) & np.greater_equal(density, 0.0001)
            )[0]
            grid_positions_a = grid_coords[grid_indices]
            coords = grid_positions_a

            if esp_cpu_fallback:
                dm_np = dm.get() if hasattr(dm, "get") else np.asarray(dm)
                res = _compute_esp_grid_cpu(mol, dm_np, coords)
            else:
                v_elec = int1e_grids(mol, coords, dm=dm)
                v_elec = v_elec.get() if hasattr(v_elec, "get") else np.asarray(v_elec)
                charges = mol.atom_charges()
                atom_coords = mol.atom_coords(unit="Bohr")
                v_nuc = np.array(
                    [
                        np.sum(charges / (np.linalg.norm(atom_coords - r, axis=1) + 1e-12))
                        for r in coords
                    ],
                    dtype=np.float64,
                )
                res = v_nuc - v_elec

            out["esp"] = np.asarray(res)
            out["esp_grid"] = np.asarray(coords)
            out["density"] = density
            out["density_grid"] = grid_coords

        if compute_polarizability:
            out["polarizability"] = _to_numpy(
                polarizability.eval_polarizability(mf)
            )

        return out

    # Zero-field SCF
    run_scf = energy or gradient or dipole or dens_esp
    if run_scf:
        e = engine.kernel()
        if energy:
            out = {"energy": np.array(e)}
        else:
            out = {}
    else:
        out = {}
    out["R"] = np.asarray(R)
    out["Z"] = np.asarray(Z)
    out["N"] = np.array(len(Z))

    if gradient:
        g = engine.nuc_grad_method()
        g_dft = g.kernel()
        out["gradient"] = _to_numpy(g_dft)

    if dipole:
        dm = engine.make_rdm1()
        out["D"] = _to_numpy(engine.dip_moment(unit="DEBYE", dm=dm, verbose=0))

    if dens_esp:
        dm = engine.make_rdm1()
        grids = engine.grids
        grid_coords = grids.coords.get()
        density = engine._numint.get_rho(mol, dm, grids).get()
        grid_indices = np.where(
            np.less(density, 0.001) & np.greater_equal(density, 0.0001)
        )[0]
        grid_positions_a = grid_coords[grid_indices]
        coords = grid_positions_a  # in Bohr (PySCF grids convention)

        if esp_cpu_fallback:
            # CPU int1e_rinv: guaranteed correct esp[i] <-> grid[i] alignment.
            dm_np = dm.get() if hasattr(dm, "get") else np.asarray(dm)
            res = _compute_esp_grid_cpu(mol, dm_np, coords)
        else:
            # GPU path: int1e_grids (direct grid evaluation, correct ordering)
            # + V_nuc. Avoids get_j_int3c2e_pass1 aux basis ordering bug.
            v_elec = int1e_grids(mol, coords, dm=dm)
            v_elec = v_elec.get() if hasattr(v_elec, "get") else np.asarray(v_elec)
            charges = mol.atom_charges()
            atom_coords = mol.atom_coords(unit="Bohr")
            v_nuc = np.array(
                [
                    np.sum(charges / (np.linalg.norm(atom_coords - r, axis=1) + 1e-12))
                    for r in coords
                ],
                dtype=np.float64,
            )
            res = v_nuc - v_elec

        out["esp"] = np.asarray(res)
        out["esp_grid"] = np.asarray(coords)
        out["density"] = density
        out["density_grid"] = grid_coords

    if compute_polarizability:
        out["polarizability"] = _to_numpy(
            polarizability.eval_polarizability(engine)
        )

    return out


def compute_dft_batch(
    R_batch: np.ndarray,
    Z: np.ndarray,
    basis: str = "def2-SVP",
    xc: str = "PBE0",
    spin: int = 0,
    charge: int = 0,
    energy: bool = True,
    gradient: bool = True,
    dipole: bool = True,
    dens_esp: bool = False,
    compute_polarizability: bool = False,
    esp_cpu_fallback: bool = False,
    verbose: int = 0,
    efield: np.ndarray | None = None,
    efield_include_nuclear_energy: bool = True,
) -> dict:
    """
    Run DFT for multiple geometries in one process (same GPU context).
    Loops over R_batch, aggregates E, F, Dxyz, esp into batch arrays.
    """
    n = R_batch.shape[0]
    Z = np.asarray(Z)
    if R_batch.ndim == 2:
        R_batch = R_batch[np.newaxis, ...]

    efield_arr: np.ndarray | None = None
    if efield is not None:
        efield_arr = np.asarray(efield, dtype=np.float64)
        if efield_arr.shape == (3,):
            efield_arr = np.broadcast_to(efield_arr, (n, 3)).copy()
        elif efield_arr.shape != (n, 3):
            raise ValueError(
                f"efield must be (3,) or (n,3) with n={n}, got {efield_arr.shape}"
            )

    energies = []
    gradients = []
    dipoles = []
    polarizabilities = []
    esps = []
    esp_grids = []
    efs = []

    for i in tqdm(range(n), desc="pyscf-dft", unit="geom"):
        ef_i = None if efield_arr is None else efield_arr[i]
        out = compute_dft_single(
            R_batch[i],
            Z,
            basis=basis,
            xc=xc,
            spin=spin,
            charge=charge,
            energy=energy,
            gradient=gradient,
            dipole=dipole,
            dens_esp=dens_esp,
            compute_polarizability=compute_polarizability,
            esp_cpu_fallback=esp_cpu_fallback,
            verbose=verbose,
            efield=ef_i,
            efield_include_nuclear_energy=efield_include_nuclear_energy,
        )
        if energy:
            energies.append(out["energy"])
        if gradient:
            gradients.append(out["gradient"])
        if dipole:
            dipoles.append(out["D"])
        if compute_polarizability:
            polarizabilities.append(out["polarizability"])
        if dens_esp:
            esps.append(out["esp"])
            esp_grids.append(out["esp_grid"])
        if "Ef" in out:
            efs.append(out["Ef"])

    result = {
        "R": R_batch,
        "Z": np.asarray(Z),
        "N": np.array(len(Z)),
    }
    if efs:
        result["Ef"] = np.stack(efs, axis=0)
    if energy:
        result["E"] = np.stack(energies)
    if gradient:
        result["F"] = -np.stack(gradients)  # forces = -gradient
    if dipole:
        result["Dxyz"] = np.stack(dipoles)
    if compute_polarizability:
        result["polarizability"] = np.stack(polarizabilities)
    if dens_esp:
        # ESP grid size can vary per geometry; pad to max length
        max_n = max(e.size for e in esps)
        esp_padded = np.zeros((n, max_n), dtype=np.float64)
        esp_grid_padded = np.full((n, max_n, 3), 1e6, dtype=np.float64)
        for i, (e, g) in enumerate(zip(esps, esp_grids)):
            esp_padded[i, : e.size] = e
            esp_grid_padded[i, : g.shape[0], :] = g
        result["esp"] = esp_padded
        result["esp_grid"] = esp_grid_padded

    return result


def compute_interaction_energy(monomer_a, monomer_b, basis='cc-pVDZ', xc='PBE0'):
    # Convert string geometries to PySCF atom format
    def parse_xyz(xyz_str):
        atoms = []
        for line in xyz_str.strip().split('\n'):
            if not line.strip(): continue
            symbol, *coords = line.split()
            coords = tuple(float(x) for x in coords)
            atoms.append((symbol, coords))
        return atoms

    atom_A = parse_xyz(monomer_a)
    atom_B = parse_xyz(monomer_b)
    atom_AB = atom_A + atom_B

    # Build molecular objects
    mol_A = pyscf.M(atom=atom_A, basis=basis).build()
    mol_B = pyscf.M(atom=atom_B, basis=basis).build()
    mol_AB = pyscf.M(atom=atom_AB, basis=basis).build()

    # Create ghost-atom systems for BSSE correction
    mol_A_ghost = mol_A.copy()
    ghost_atoms_B = mol_B.atom
    mol_A_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_B])
    mol_A_ghost.build()

    mol_B_ghost = mol_B.copy()
    ghost_atoms_A = mol_A.atom
    mol_B_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_A])
    mol_B_ghost.build()

    regular_calcs = [CALCS.ENERGY, CALCS.GRADIENT, CALCS.HESSIAN, CALCS.HARMONIC, CALCS.THERMO]
    ghost_calcs = [CALCS.ENERGY]
    
    output_AB = compute_dft(mol_AB, regular_calcs)
    E_AB = output_AB["energy"]

    output_A = compute_dft(mol_A, regular_calcs)
    E_A = output_A["energy"]

    output_B = compute_dft(mol_B, regular_calcs)
    E_B = output_B["energy"]    

    output_A_ghost = compute_dft(mol_A_ghost, ghost_calcs)
    E_A_ghost = output_A_ghost["energy"]    

    output_B_ghost = compute_dft(mol_B_ghost, ghost_calcs)
    E_B_ghost = output_B_ghost["energy"]
    

    # Calculate interaction energies
    IE_no_bsse = E_AB - (E_A + E_B)
    IE_energy_bsse = E_AB - (E_A_ghost + E_B_ghost)

    intE = {
        'E_AB': E_AB,
        'E_A': E_A,
        'E_B': E_B,
        'E_A_ghost': E_A_ghost,
        'E_B_ghost': E_B_ghost,
        'IE_no_bsse': IE_no_bsse,
        'IE_energy_bsse': IE_energy_bsse,
    }
    output = {
        'results_AB': output_AB,
        'results_A': output_A,
        'results_B': output_B,
        'results_A_ghost': output_A_ghost,
        'results_B_ghost': output_B_ghost,
        'intE_results': intE
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    # molecule
    parser.add_argument("--mol", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--log_file", type=str, default="pyscf.log")
    parser.add_argument("--monomer_a", type=str, default="")
    parser.add_argument("--monomer_b", type=str, default="")
    parser.add_argument("--basis", type=str, default="def2-SVP")
    parser.add_argument("--xc", type=str, default="PBE0")
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    # flags to do certain calcs
    parser.add_argument("--energy", default=False, action="store_true")
    parser.add_argument("--optimize", default=False, action="store_true")
    parser.add_argument("--gradient", default=False, action="store_true")
    parser.add_argument("--hessian", default=False, action="store_true")
    parser.add_argument("--harmonic", default=False, action="store_true")
    parser.add_argument("--thermo", default=False, action="store_true")
    parser.add_argument("--interaction", default=False, action="store_true")
    parser.add_argument("--dens_esp", default=False, action="store_true")
    parser.add_argument("--ir", default=False, action="store_true")
    parser.add_argument("--shielding", default=False, action="store_true")
    parser.add_argument("--polarizability", default=False, action="store_true")
    parser.add_argument(
        "--ir-efield",
        default=False,
        action="store_true",
        help="IR + Hessian pipeline in a uniform E-field; scan fields from --efield-points",
    )
    parser.add_argument(
        "--efield-points",
        type=str,
        default="0,0,0",
        help="Semicolon-separated Ex,Ey,Ez in a.u., e.g. '0,0,0;0,0,0.001;0,0,-0.001'",
    )
    parser.add_argument(
        "--efield-fd-axis",
        type=int,
        default=2,
        help="Cartesian axis (0=x,1=y,2=z) for finite-difference dμ/dE from the scan",
    )
    parser.add_argument(
        "--efield-scf",
        default=False,
        action="store_true",
        help="SCF only in uniform E-field: energy, dipole, forces (use --efield-points); no IR/Hessian",
    )
    parser.add_argument(
        "--efield-scf-no-forces",
        default=False,
        action="store_true",
        help="With --efield-scf, skip nuclear gradient (energy + dipole only)",
    )
    parser.add_argument(
        "--efield-dipole-unit",
        type=str,
        default="DEBYE",
        help="Dipole unit for --efield-scf (e.g. DEBYE, AU)",
    )
    parser.add_argument(
        "--no-efield-include-nuclear-energy",
        dest="efield_include_nuclear_energy",
        action="store_false",
        help=(
            "After SCF in a uniform field, omit nuclear-field energy (use mf.kernel energy only)."
        ),
    )
    parser.set_defaults(efield_include_nuclear_energy=True)
    parser.add_argument("--save_option", type=str, default="hdf5")
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    return args


def process_calcs(args):
    calcs = []
    extra = None

    if args.optimize:
        calcs.append(CALCS.OPTIMIZE)

    if args.energy:
        calcs.append(CALCS.ENERGY)
    
    if args.gradient:
        calcs.append(CALCS.GRADIENT)
    
    if args.hessian:
        calcs.append(CALCS.HESSIAN)
    
    if args.harmonic:
        calcs.append(CALCS.HARMONIC)
    
    if args.thermo:
        calcs.append(CALCS.THERMO)

    if args.dens_esp:
        calcs.append(CALCS.DENS_ESP)

    if args.ir:
        calcs.append(CALCS.IR)

    if args.shielding:
        calcs.append(CALCS.SHIELDING)

    if args.polarizability:
        calcs.append(CALCS.POLARIZABILITY)

    if getattr(args, "ir_efield", False):
        calcs.append(CALCS.IR_EFIELD)

    if getattr(args, "efield_scf", False):
        calcs.append(CALCS.EFIELD_SCF)

    if args.interaction:
        calcs.append(CALCS.INTERACTION)
        extra = (args.monomer_a, args.monomer_b)

    return calcs, extra


def compute_mp2(mol_str: str, basis: str = "def2-SVP", spin: int = 0, charge: int = 0,
                energy: bool = True, gradient: bool = False, log_file: str = "pyscf.log") -> dict:
    """
    Run DF-MP2 calculation via gpu4pyscf (post-HF, not DFT).

    Returns dict with R, Z, N, energy, gradient (if requested), etc.
    """
    import pyscf
    from gpu4pyscf.scf import RHF
    from gpu4pyscf.mp import dfmp2

    mol = pyscf.M(atom=mol_str, basis=basis, spin=spin, charge=charge, output=log_file, verbose=6)
    mf = RHF(mol).density_fit()
    e_hf = mf.kernel()

    pt = dfmp2.DFMP2(mf)
    e_corr, _ = pt.kernel()
    e_mp2 = e_hf + e_corr

    output = {
        "mol": mol,
        "energy": np.array(e_mp2),
        "energy_hf": np.array(e_hf),
        "energy_corr": np.array(e_corr),
        "R": np.asarray(mol.atom_coords(unit="ANG")),
        "Z": np.asarray(mol.atom_charges()),
        "N": np.array(len(mol.atom_charges())),
    }

    if gradient:
        g = pt.nuc_grad_method()
        g_mp2 = g.kernel()
        output["gradient"] = g_mp2.get() if hasattr(g_mp2, "get") else np.asarray(g_mp2)

    return output


def get_dummy_args(mol: str, calcs: list[CALCS]):
    # instead of parsing the args, trick python into thinking we have parsed the args
    class Args:
        def __init__(self):
            self.mol = mol
            self.output = "output"
            self.log_file = "pyscf.log"
            self.monomer_a = ""
            self.monomer_b = ""
            self.basis = "def2-SVP"
            self.xc = "PBE0"
            self.spin = 0
            self.charge = 0 
            self.energy = CALCS.ENERGY in calcs
            self.optimize = CALCS.OPTIMIZE in calcs
            self.gradient = CALCS.GRADIENT in calcs
            self.hessian = CALCS.HESSIAN in calcs
            self.harmonic = CALCS.HARMONIC in calcs
            self.thermo = CALCS.THERMO in calcs
            self.dens_esp = CALCS.DENS_ESP in calcs
            self.ir = CALCS.IR in calcs
            self.shielding = CALCS.SHIELDING in calcs
            self.polarizability = CALCS.POLARIZABILITY in calcs
            self.ir_efield = CALCS.IR_EFIELD in calcs
            self.efield_scf = CALCS.EFIELD_SCF in calcs
            self.efield_points = "0,0,0"
            self.efield_fd_axis = 2
            self.efield_scf_no_forces = False
            self.efield_dipole_unit = "DEBYE"
            self.efield_include_nuclear_energy = True
            self.interaction = CALCS.INTERACTION in calcs
            self.save_option = "pkl"

    return Args()
            



def _to_numpy(value):
    """Best-effort conversion of arrays to numpy, pass through scalars/lists, drop unsupported."""
    try:
        import cupy as _cp
    except Exception:
        _cp = None

    if isinstance(value, np.ndarray):
        return value
    if _cp is not None and isinstance(value, _cp.ndarray):
        return value.get()
    if isinstance(value, (int, float, np.number)):
        return np.array(value)
    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, np.number)) for x in value):
        return np.asarray(value)
    return None


def build_ml_dict(data: dict) -> dict:
    """Build ML-style dict with keys R, Z, N, E, F, Dxyz, etc. from compute_dft output.

    Uses (1, ...) leading dimension for single-structure compatibility with batch schema.
    Units: R [Angstrom], E [Hartree], F [Hartree/Bohr], Dxyz [Debye].
    """
    out = {}
    for k, v in data.items():
        arr = _to_numpy(v)
        if arr is not None:
            out[k] = arr

    ml = {}
    if "R" in out:
        R = out["R"]
        ml["R"] = R[np.newaxis, ...] if R.ndim == 2 else R
    if "Z" in out:
        Z = out["Z"]
        ml["Z"] = Z[np.newaxis, ...] if Z.ndim == 1 else Z
    if "N" in out:
        N = out["N"]
        ml["N"] = np.atleast_1d(N)
    if "energy" in out:
        ml["E"] = np.atleast_1d(out["energy"])
    if "gradient" in out:
        # F = -gradient (forces in Hartree/Bohr)
        g = out["gradient"]
        ml["F"] = (-g)[np.newaxis, ...] if g.ndim == 2 else np.atleast_2d(-g)
    if "D" in out:
        D = out["D"]
        ml["Dxyz"] = D[np.newaxis, ...] if D.ndim == 1 else D
    if "Q" in out:
        ml["Q"] = out["Q"][np.newaxis, ...] if out["Q"].ndim == 2 else out["Q"]
    if "esp" in out:
        ml["esp"] = out["esp"][np.newaxis, ...] if out["esp"].ndim == 1 else out["esp"]
    if "esp_grid" in out:
        g = out["esp_grid"]
        ml["esp_grid"] = g[np.newaxis, ...] if g.ndim == 2 else g
    if "vdw_surface" in out:
        ml["vdw_surface"] = out["vdw_surface"]
    if "shielding" in out:
        ml["shielding"] = out["shielding"]
    if "polarizability" in out:
        ml["polarizability"] = out["polarizability"]
    if "hessian" in out:
        ml["hessian"] = out["hessian"]
    if "freq" in out:
        ml["freq"] = out["freq"]
    if "intensity" in out:
        ml["intensity"] = out["intensity"]
    if "efield_Ef" in out:
        ml["efield_Ef"] = out["efield_Ef"]
    if "efield_energy" in out:
        ml["efield_energy"] = out["efield_energy"]
    if "efield_D_au" in out:
        ml["efield_D_au"] = out["efield_D_au"]
    if "efield_scf_energy" in out:
        ml["efield_scf_energy"] = out["efield_scf_energy"]
    if "efield_scf_D" in out:
        ml["efield_scf_D"] = out["efield_scf_D"]
    if "efield_scf_D_au" in out:
        ml["efield_scf_D_au"] = out["efield_scf_D_au"]
    if "efield_scf_F" in out:
        ml["efield_scf_F"] = out["efield_scf_F"]
    if "efield_scf_Ef" in out:
        ml["efield_scf_Ef"] = out["efield_scf_Ef"]
    if "Ef" in out:
        ml["Ef"] = out["Ef"]
    if "density" in out:
        ml["density"] = out["density"]
    if "density_grid" in out:
        dg = out["density_grid"]
        arr = _to_numpy(dg)
        if arr is not None:
            ml["density_grid"] = arr

    return {k: v for k, v in ml.items() if v is not None}


def _arrays_only(data: dict) -> dict:
    """Extract all array-like values from data (top-level only)."""
    result = {}
    for k, v in data.items():
        if k == "mol":
            continue
        arr = _to_numpy(v)
        if arr is not None:
            result[k] = arr
    return result


def save_pyscf_results(base_path: str, data: dict) -> None:
    """
    Save pyscf-dft results as both NPZ (ML-style keys) and H5 (all arrays).

    Writes:
    - {base_path}.npz: ML dict with R, Z, N, E, F, Dxyz, esp, etc.
    - {base_path}.h5: All array-like data (original keys)
    """
    import os as _os

    base = _os.path.splitext(base_path)[0]
    if base.endswith(".h5") or base.endswith(".hdf5"):
        base = _os.path.splitext(base)[0]
    if _os.path.dirname(base) != "":
        _os.makedirs(_os.path.dirname(base), exist_ok=True)

    ml_dict = build_ml_dict(data)
    if ml_dict:
        np.savez_compressed(f"{base}.npz", **ml_dict)

    arrays = _arrays_only(data)
    if arrays:
        import h5py as _h5py

        def _write(parent, d):
            for k, v in d.items():
                arr = _to_numpy(v)
                if arr is not None:
                    parent.create_dataset(str(k), data=arr)
                elif isinstance(v, dict):
                    grp = parent.create_group(str(k))
                    _write(grp, v)

        with _h5py.File(f"{base}.h5", "w") as h5f:
            for k, v in data.items():
                if k == "mol":
                    continue
                arr = _to_numpy(v)
                if arr is not None:
                    h5f.create_dataset(str(k), data=arr)
                elif isinstance(v, dict):
                    grp = h5f.create_group(str(k))
                    _write(grp, v)


def save_output(output_path: str, data: dict, save_option: str = "pkl") -> None:
    """
    Save a dictionary of results. Intended for dictionaries whose values are primarily
    numpy/cupy arrays.

    Supported formats:
    - "pkl": Full pickle of the dict (allows arbitrary Python objects)
    - "npz": Only array-like entries are saved as separate arrays in a .npz
    - "hdf5": Only array-like entries saved under datasets by key

    Parquet/Feather are not supported here because they require tabular structures.
    """
    import os as _os
    import pickle as _pickle

    if _os.path.dirname(output_path) != "":
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)

    if save_option == "pkl":
        with open(output_path, "wb") as f:
            _pickle.dump(data, f, protocol=_pickle.HIGHEST_PROTOCOL)
        return

    arrays_only: dict[str, np.ndarray] = {}
    for k, v in data.items():
        arr = _to_numpy(v)
        if arr is not None:
            arrays_only[k] = arr

    if save_option == "npz":
        if len(arrays_only) == 0:
            with open(output_path, "wb") as f:
                _pickle.dump(data, f, protocol=_pickle.HIGHEST_PROTOCOL)
            return
        np.savez_compressed(output_path, **arrays_only)
        return

    if save_option == "hdf5":
        import h5py as _h5py
        with _h5py.File(output_path, "w") as h5f:
            for k, arr in arrays_only.items():
                h5f.create_dataset(str(k), data=arr)
        return

    if save_option in {"parquet", "feather"}:
        raise ValueError(
            "Parquet/Feather are not supported for arbitrary dicts of arrays. Use 'npz' or 'hdf5', or 'pkl' for full objects."
        )

    raise ValueError(f"Invalid save option: {save_option}")


if __name__ == "__main__":
    args = parse_args()
    calcs, extra = process_calcs(args)
    print(calcs, extra)
    output = compute_dft(args, calcs, extra)
    print(output)
    save_output(args.output, output, args.save_option)
