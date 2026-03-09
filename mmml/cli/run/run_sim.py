"""
Runs an MD simulation.

Requires the box size and system (as a pdb file) to be specified.
Temperature, pressure, and density can be specified.
If no solvent is specified, the system will be vacuum.
If no temperature is specified, the system will be at room temperature.
If no pressure is specified, the system will be at atmospheric pressure.
If no density is specified, the system will be at the density of the box.
If no side length is specified, the system will be at the side length of the box.
If no residue is specified, the system will be at the residue of the box.

Args:
    --pdbfile
    simulation conditions

Run from a notebook:
    from pathlib import Path
    import argparse
    from mmml.cli.run_sim import run

    args = argparse.Namespace(
        pdbfile=Path("pdb/init-packmol.pdb"),
        checkpoint=Path("ACO-..."),
        n_monomers=50,
        n_atoms_monomer=10,
        cell=40.0,
        # optional: temperature=200.0, timestep=0.3, ensemble="nve", ...
    )
    run(args)
"""


#!/usr/bin/env python3
from __future__ import annotations
"""PDB file processing and molecular dynamics simulation demo.

This demo loads PDB files and runs molecular dynamics simulations using
the hybrid MM/ML calculator with PyCHARMM integration.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mmml.cli.base import (
    load_model_parameters,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB file processing and MD simulation demo"
    )
    parser.add_argument(
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to the PDB file to load for pycharmm [requires correct atom names and types].",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the force field (default: False).",
    )

    # ========================================================================
    # MD SIMULATION ARGUMENTS
    # ========================================================================
    parser.add_argument(
        "--energy-catch",
        type=float,
        default=0.5,
        help="Energy catch factor for the simulation (default: 0.05).",
    )

    parser.add_argument(
        "--cell",
        type=float,
        default=None,
        help="Cubic cell side length in Å for periodic boundary conditions (default: None = no PBC).",
    )
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        help="Radius of flat bottom potential to constrain system COM to center (default: None = disabled). "
        "When set, V=0 for |d|<=R and V=k*(|d|-R)^2 for |d|>R. With --cell, center=box center; else origin.",
    )
    parser.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for flat bottom potential when outside radius (default: 1.0).",
    )

    parser.add_argument(
        "--n-monomers",
        type=int,
        required=True,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--n-atoms-monomer",
        type=int,
        required=True,
        help=(
            "Number of atoms per monomer. Defaults to total_atoms/n_monomers "
            "derived from the dataset."
        ),
    )
    parser.add_argument(
        "--ml-cutoff",
        type=float,
        default=2.0,
        help="ML cutoff distance passed to the calculator factory (default: 2.0 Å).",
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=5.0,
        help="MM switch-on distance for the hybrid calculator (default: 5.0 Å).",
    )
    parser.add_argument(
        "--mm-cutoff",
        type=float,
        default=1.0,
        help="MM cutoff width for the hybrid calculator (default: 1.0 Å).",
    )
    parser.add_argument(
        "--include-mm",
        action="store_true",
        help="Keep MM contributions enabled when evaluating the hybrid calculator.",
    )
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="If set, skip the ML dimer correction in the hybrid calculator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=210.0,
        help="Temperature for MD simulation in Kelvin (default: 300.0).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
        help="Timestep for MD simulation in fs (default: 0.5).",
    )
    parser.add_argument(
        "--nsteps_jaxmd",
        type=int,
        default=100_000,
        help="Number of MD steps to run in JAX-MD (default: 100000).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="md_simulation",
        help="Prefix for output files (default: md_simulation).",
    )
    parser.add_argument(
        "--nsteps_ase",
        type=int,
        default=10000,
        help="Number of steps to run in ASE (default: 10000).",
    )
    parser.add_argument(
        "--optimize-monomers",
        action="store_true",
        help="If set, run monomer-wise optimization with simple_physnet before hybrid BFGS. "
        "Default: skip (use hybrid BFGS from CHARMM structure). Monomer optimization ignores "
        "inter-monomer interactions and can produce overlapping geometries.",
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="nvt",
        help="Ensemble to run the simulation in (default: nvt).",
    )

    # Nose-Hoover chain (NHC) thermostat parameters for NVT ensemble
    parser.add_argument(
        "--nhc-chain-length",
        type=int,
        default=3,
        help="Number of chains in the Nose-Hoover chain thermostat (default: 3).",
    )
    parser.add_argument(
        "--nhc-chain-steps",
        type=int,
        default=2,
        help="Number of steps per chain in the Nose-Hoover chain thermostat (default: 2).",
    )
    parser.add_argument(
        "--nhc-sy-steps",
        type=int,
        default=3,
        help="Number of Suzuki-Yoshida steps in the Nose-Hoover chain thermostat (default: 3).",
    )
    parser.add_argument(
        "--nhc-tau",
        type=float,
        default=100.0,
        help="Thermostat coupling time multiplier (tau = nhc_tau * dt) (default: 100).",
    )

    # NPT barostat parameters
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.01325,
        help="Target pressure in bar for NPT ensemble (default: 1.01325).",
    )
    parser.add_argument(
        "--nhc-barostat-tau",
        type=float,
        default=10000.0,
        help="Barostat coupling time multiplier for NPT (tau = nhc_barostat_tau * dt) (default: 10000).",
    )
    parser.add_argument(
        "--npt-diagnose",
        action="store_true",
        help="Run NPT diagnostic tests before simulation (energy, stress, shift, step).",
    )

    parser.add_argument(
        "--heating_interval",
        type=int,
        default=500,
        help="Interval to heat the system in ASE (default: 500).",
    )
    parser.add_argument(
        "--write_interval",
        type=int,
        default=100,
        help="Interval to write the trajectory in ASE (default: 100).",
    )

    parser.add_argument(
        "--charmm_heat",
        action="store_true",
        help="Run CHARMM heat (default: False).",
    )
    parser.add_argument(
        "--charmm_equilibration",
        action="store_true",
        help="Run CHARMM equilibration (default: False).",
    )
    parser.add_argument(
        "--charmm_production",
        action="store_true",
        help="Run CHARMM production (default: False).",
    )

    parser.add_argument(
        "--use_physnet_calculator_for_full_system",
        action="store_true",
        help="Use the physnet calculator for the full system (default: False).",
    )

    return parser.parse_args()


def default_nhc_kwargs(tau, overrides=None):
    """Build Nose-Hoover chain kwargs dict with sensible defaults.

    Args:
        tau: Thermostat coupling timescale (typically ``nhc_tau * dt``).
        overrides: Optional dict to override individual defaults.

    Returns:
        Dict with keys ``chain_length``, ``chain_steps``, ``sy_steps``, ``tau``.
    """
    default_kwargs = {
        'chain_length': 3,
        'chain_steps': 2,
        'sy_steps': 3,
        'tau': tau,
    }
    if overrides is None:
        return default_kwargs
    return {k: overrides.get(k, default_kwargs[k]) for k in default_kwargs}


def _run_npt_diagnostics(
    *,
    state,
    npt_energy_fn,
    jax_md_force_fn,
    apply_fn,
    shift,
    space,
    simulate,
    quantity,
    npt_pair_idx,
    npt_pair_mask,
    npt_pressure,
    dt,
    kT,
    grad,
):
    """Run NPT diagnostic tests to locate instabilities. Call with --npt-diagnose."""
    import jax.numpy as jnp
    import numpy as np

    neighbor = (npt_pair_idx, npt_pair_mask)
    box_curr = simulate.npt_box(state)
    R = state.position
    P = state.momentum
    M = state.mass
    N, dim = R.shape

    print("\n" + "=" * 60)
    print("NPT DIAGNOSTIC TESTS (--npt-diagnose)")
    print("=" * 60)

    # 1. Energy and forces sanity
    print("\n[1] Energy and forces at initial state")
    E0 = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor))
    real_pos = space.transform(box_curr, R)
    F_calc = jax_md_force_fn(real_pos, mm_pair_idx=npt_pair_idx, mm_pair_mask=npt_pair_mask, box=box_curr)
    F_grad = -grad(lambda r: npt_energy_fn(r, box=box_curr, neighbor=neighbor))(R)
    print(f"    E = {E0:.6f} eV")
    print(f"    max|F_calc| = {float(np.max(np.abs(F_calc))):.6f}")
    print(f"    max|F_grad| = {float(np.max(np.abs(F_grad))):.6f}")
    print(f"    F_calc finite: {np.all(np.isfinite(F_calc))}, F_grad finite: {np.all(np.isfinite(F_grad))}")

    # 2. Perturbation / stress (dUdV)
    print("\n[2] Stress (dU/dV) via perturbation")
    eps_vals = [0.0, 1e-6, 1e-5, 1e-4]
    for eps in eps_vals:
        pert = 1.0 + eps
        E_pert = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=pert))
        print(f"    eps={eps:.0e}: E = {E_pert:.6f} eV")
    dE = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=1.0 + 1e-5)) - E0
    vol = float(quantity.volume(dim, box_curr))
    dUdV_fd = dE / (vol * 1e-5)  # finite-diff approx
    print(f"    dUdV (finite diff) ≈ {dUdV_fd:.4f} eV/Å³")
    print(f"    volume = {vol:.2f} Å³")

    # 3. Shift function with fractional R and Cartesian dR
    print("\n[3] Shift function (frac R + Cartesian dR)")
    dR_cart = dt * (P / M)  # small Cartesian displacement
    R_shifted = shift(R, dR_cart, box=box_curr)
    in_cube = np.all((R_shifted >= 0) & (R_shifted < 1.001))
    print(f"    R_shifted in [0,1)^3: {in_cube}")
    print(f"    R_shifted finite: {np.all(np.isfinite(R_shifted))}")
    print(f"    R_shifted sample [0]: {np.asarray(R_shifted[0])}")

    # 4. exp_iL1-like displacement (barostat scaling term)
    print("\n[4] Barostat scaling term R*(exp(x)-1) + dt*V*... (x=V_b*dt)")
    V_b = 0.0  # box velocity at start
    x = V_b * dt
    scale = np.exp(x) - 1
    term1 = R * scale  # fractional * scalar
    term2 = dt * (P / M) * np.exp(x / 2)  # velocity term
    dR_mixed = term1 + term2
    print(f"    x={x}, exp(x)-1={scale}")
    print(f"    max|term1|={float(np.max(np.abs(term1))):.6e}, max|term2|={float(np.max(np.abs(term2))):.6e}")
    R_after_scale = shift(R, dR_mixed, box=box_curr)
    print(f"    R_after_scale finite: {np.all(np.isfinite(R_after_scale))}")
    print(f"    R_after_scale in [0,1): {np.all((R_after_scale >= 0) & (R_after_scale < 1.001))}")

    # 5. Box and volume
    print("\n[5] Box and volume")
    print(f"    box shape: {np.asarray(box_curr).shape}")
    print(f"    box diag: {np.diagonal(np.asarray(box_curr))}")
    print(f"    box_position (log V/V0): {float(state.box_position)}")
    print(f"    box_momentum: {float(state.box_momentum)}")

    # 6. State components
    print("\n[6] State components")
    print(f"    position finite: {np.all(np.isfinite(R))}")
    print(f"    momentum finite: {np.all(np.isfinite(P))}")
    print(f"    force finite: {np.all(np.isfinite(state.force))}")
    print(f"    mass shape: {M.shape}, all positive: {np.all(M > 0)}")

    # 7. First step (apply_fn) and NaN location
    print("\n[7] First NPT step (apply_fn)")
    neighbor = (npt_pair_idx, npt_pair_mask)
    try:
        state_one = apply_fn(state, neighbor=neighbor, pressure=npt_pressure)
        pos_ok = np.all(np.isfinite(np.asarray(state_one.position)))
        mom_ok = np.all(np.isfinite(np.asarray(state_one.momentum)))
        box_ok = np.all(np.isfinite(np.asarray(simulate.npt_box(state_one))))
        print(f"    Step completed. position OK: {pos_ok}, momentum OK: {mom_ok}, box OK: {box_ok}")
        if not pos_ok:
            nan_count = np.sum(~np.isfinite(np.asarray(state_one.position)))
            print(f"    NaN in position: {nan_count} elements")
            first_nan = np.where(~np.isfinite(np.asarray(state_one.position)))
            if len(first_nan[0]) > 0:
                print(f"    First NaN at index: ({first_nan[0][0]}, {first_nan[1][0]})")
    except Exception as e:
        print(f"    apply_fn raised: {type(e).__name__}: {e}")

    print("\n" + "=" * 60 + "\n")


def run(args: argparse.Namespace) -> int:
    """Run MD simulation with the given arguments (CLI or notebook)."""
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    # Setup imports
    Atoms = setup_ase_imports()
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    
    # Additional imports for this demo
    try:
        import pycharmm
        import pycharmm.psf as psf
        import ase
        import ase.calculators.calculator as ase_calc
        import ase.io as ase_io
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase.optimize as ase_opt
        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
        from mmml.interfaces.pycharmmInterface.setupBox import setup_box_generic
        import pandas as pd
        from mmml.interfaces.pycharmmInterface.import_pycharmm import minimize
        from mmml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np
        import jax_md
        from jax_md import space, quantity, simulate, partition, units, units
        from ase.units import _amu

        import jax.numpy as jnp
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import wrap_groups

        import jax, e3x
        from jax import jit, grad, lax, ops, random
        import jax.numpy as jnp
        from ase.io import Trajectory
        from mmml.utils.hdf5_reporter import make_jaxmd_reporter
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    pdbfilename = str(args.pdbfile)
    
    # Setup box and load PDB
    setup_box_generic(pdbfilename, side_length=1000)
    pdb_ase_atoms = ase_io.read(pdbfilename)

    print(f"Loaded PDB file: {pdb_ase_atoms}")

    # --- Normalise n_atoms_monomer: int or list ---------------------------------
    raw_n_atoms_monomer = args.n_atoms_monomer
    if isinstance(raw_n_atoms_monomer, (list, tuple, np.ndarray)):
        atoms_per_monomer_list = [int(x) for x in raw_n_atoms_monomer]
        n_monomers = len(atoms_per_monomer_list)
        total_atoms = sum(atoms_per_monomer_list)
        n_atoms_first = atoms_per_monomer_list[0]  # for single-monomer test calc
    else:
        n_atoms_first = int(raw_n_atoms_monomer)
        n_monomers = args.n_monomers
        atoms_per_monomer_list = [n_atoms_first] * n_monomers
        total_atoms = n_atoms_first * n_monomers
    # Build monomer offsets for slicing
    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    for _mi, _na in enumerate(atoms_per_monomer_list):
        monomer_offsets[_mi + 1] = monomer_offsets[_mi] + _na
    print(f"[run_sim] atoms_per_monomer_list={atoms_per_monomer_list}, "
          f"n_monomers={n_monomers}, total_atoms={total_atoms}")
    
    # ========================================================================
    # MASS SETUP FOR JAX-MD SIMULATION
    # ========================================================================
    raw_masses = pdb_ase_atoms.get_masses()
    print(f"Raw masses from ASE: {raw_masses}")
    psf_masses = psf.get_amass()
    print(f"PSF masses: {psf_masses}")
    print(f"PSF masses sum: {sum(psf_masses)}")
    print("Setting the elements and masses from the psf")
    pdb_ase_atoms.set_masses(psf_masses)
    psf_masses_arr = np.array(psf_masses)[:, np.newaxis]
    correct_atomic_numbers_from_mass = np.argmin(
        np.abs(ase.data.atomic_masses_common[np.newaxis, :] - psf_masses_arr), axis=1)
    pdb_ase_atoms.set_atomic_numbers(correct_atomic_numbers_from_mass)
    print(f"PDB ASE atoms: {pdb_ase_atoms}")

    # Actual masses for COM (Si_mass was misnamed - it held atomic numbers before)
    masses_jax = jnp.array(psf_masses[:total_atoms], dtype=jnp.float32)
    Si_mass = masses_jax  # keep name for compatibility with JAX-MD closure
    Si_mass_sum = Si_mass.sum()
    print(f"Masses (amu) for JAX-MD: sum={float(Si_mass_sum):.2f}")
    
    Si_mass_expanded = jnp.repeat(Si_mass[:, None], 3, axis=1)
    print(f"Masses expanded shape: {Si_mass_expanded.shape}")

    print(f"PyCHARMM coordinates: {coor.get_positions()}")
    print(f"Ase coordinates: {pdb_ase_atoms.get_positions()}")

    print(coor.get_positions())
    # Use the first monomer for the quick single-monomer test calculator
    ase_monomer = pdb_ase_atoms[0:n_atoms_first]
    params, model = load_model_parameters(epoch_dir, n_atoms_first)
    simple_physnet_calculator = get_ase_calc(params, model, ase_monomer)
    print(f"Simple physnet calculator: {simple_physnet_calculator}")
    ase_monomer.calc = simple_physnet_calculator
    print(f"ASE monomer energy: {ase_monomer.get_potential_energy()}")




    # Load model parameters for the full system
    natoms = len(pdb_ase_atoms)
    assert total_atoms == natoms, (
        f"total_atoms ({total_atoms}) != natoms from PDB ({natoms}). "
        f"atoms_per_monomer_list={atoms_per_monomer_list}"
    )
    params, model = load_model_parameters(epoch_dir, natoms)
    model.natoms = natoms
    print(f"Model loaded: {model}")
    # Get atomic numbers and positions
    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()
    atoms = pdb_ase_atoms
    if args.cell is not None:
        print("Setting cell")
        from ase.cell import Cell
        print("Creating cell")
        cell = Cell.fromcellpar([float(args.cell), float(args.cell), float(args.cell), 90., 90., 90.])
        atoms.set_cell(cell)
        # Enable periodic boundary conditions
        atoms.set_pbc(True)
        print(f"Cell: {cell}")
        print(f"PBC enabled: {atoms.pbc}")
        print(f"Cell shape: {cell.shape}")
        print(f"Cell type: {type(cell)}")
        print(f"Cell dtype: {cell.dtype}")
        print(f"Cell size: {cell.size}")
        print(f"Cell dtype: {cell.dtype}")
        print(f"Cell ndim: {cell.ndim}")
        print(f"Cell dtype: {cell.dtype}")
    else:
        cell = None
        print("No cell provided")


    # Unified calculator supports both uniform and heterogeneous monomer sizes
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer_list,
        N_MONOMERS=n_monomers,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        debug=args.debug,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=total_atoms,
        ml_energy_conversion_factor=1,
        ml_force_conversion_factor=1,
        cell=args.cell,
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        flat_bottom_force_const=getattr(args, "flat_bottom_k", 1.0),
        ensemble=getattr(args, "ensemble", "nve"),
    )
    

    CUTOFF_PARAMS = CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
        )

    print(f"Cutoff parameters: {CUTOFF_PARAMS}")





    # Create hybrid calculator (MIC-only: factory uses pbc_cell for PBC, no pbc_map/transform)
    calc_result = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=n_monomers,
        cutoff_params=CUTOFF_PARAMS,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        # Final energies/forces are already in eV; MM terms are converted internally.
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        do_pbc_map=getattr(calculator_factory, "do_pbc_map", args.cell is not None),
        pbc_map=getattr(calculator_factory, "pbc_map", None),
    )
    if len(calc_result) == 3:
        hybrid_calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        hybrid_calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None

    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms

    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc

    # After: atoms.calc = hybrid_calc
    print(f"PBC status: cell={args.cell}, atoms.pbc={atoms.pbc}, "
        f"calc.do_pbc_map={getattr(hybrid_calc, 'do_pbc_map', 'N/A')}")



    # Test invariance of energy under translation of monomer 0 by a lattice vector (PBC)
    if args.cell is not None:
        E0 = atoms.get_potential_energy()
        a = np.array([float(args.cell), 0.0, 0.0])  # first lattice vector for cubic cell
        g0 = np.arange(int(monomer_offsets[1]))  # first monomer's atoms
        R_shift = R.copy()
        R_shift[g0] += a
        atoms.set_positions(R_shift)
        E1 = atoms.get_potential_energy()
        print(f"Energy invariance test: E0={E0}, E1={E1}, difference={E1-E0}")
        assert np.isclose(E0, E1, rtol=1e-2, atol=0.01), (
            f"Energy invariance test failed: |E1-E0|={abs(E1-E0):.2e} eV"
        )
        atoms.set_positions(R)
    
    ##### add an option to just use the physnet calculator for the full system
    if getattr(args, "use_physnet_calculator_for_full_system", False):
        atoms.calc = simple_physnet_calculator
        print("Using physnet calculator for the full system")


    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    print(f"Initial forces: {hybrid_forces}")
    

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
        pycharmm,
        safe_energy_show,
    )
    reset_block()
    nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
nbonds atom cutnb 10.0  ctofnb 9.0 ctonnb 8.0 -
vswitch NBXMOD 5 -
inbfrq -1 imgfrq -1

shake bonh para sele all end

"""
    pycharmm.lingo.charmm_script(nbonds)
    safe_energy_show()
    pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
    pycharmm.lingo.charmm_script("ENER")
    safe_energy_show()
    # Sync ASE atoms from PyCHARMM so BFGS/ASE MD start from CHARMM-minimized structure
    atoms.set_positions(coor.get_positions())

    def run_heat(): 
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import heat
        pycharmm.lingo.charmm_script(heat)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    def run_equilibration():
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import equi
        pycharmm.lingo.charmm_script(equi)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    def run_production():
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import production
        pycharmm.lingo.charmm_script(production)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    if getattr(args, "charmm_heat", False):
        atoms = run_heat()
    if getattr(args, "charmm_equilibration", False):
        atoms = run_equilibration()
    if getattr(args, "charmm_production", False):
        atoms = run_production()
    
    # Minimize structure if requested
    # if args.minimize_first:
    def wrap_positions_for_pbc(positions, masses=None):
        """Wrap positions into cell. Uses pbc_map when available; otherwise wrap by monomer (MIC-only).
        Uses mass-weighted center of mass when masses provided."""
        if args.cell is None:
            return positions
        pbc_map_fn = getattr(hybrid_calc, "pbc_map", None)
        if pbc_map_fn is not None and getattr(hybrid_calc, "do_pbc_map", False):
            R_mapped = pbc_map_fn(jnp.asarray(positions))
            return np.asarray(jax.device_get(R_mapped))
        # MIC-only: wrap by monomer into primary cell (COM-based)
        cell_matrix = np.diag([float(args.cell)] * 3) if np.isscalar(args.cell) else np.asarray(args.cell, dtype=np.float64)
        if cell_matrix.ndim == 1 and cell_matrix.shape[0] == 3:
            cell_matrix = np.diag(cell_matrix)
        return _wrap_groups_np(
            np.asarray(positions, dtype=np.float64), cell_matrix, monomer_offsets, masses=masses
        )

    def minimize_structure(atoms, run_index=0, nsteps=60, fmax=0.0006, charmm=False, ase=True):

        if charmm:
            pycharmm.minimize.run_abnr(nstep=10000, tolenr=1e-6, tolgrd=1e-6)
            pycharmm.lingo.charmm_script("ENER")
            safe_energy_show()
            atoms.set_positions(coor.get_positions())
            atoms = optimize_as_monomers(atoms, run_index=run_index, nsteps=100, fmax=0.0006)

        if ase:
            traj_path = Path(f'bfgs_{run_index}_{args.output_prefix}_minimized.traj')
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj = ase_io.Trajectory(str(traj_path), 'w')
            print("Minimizing structure with hybrid calculator")
            print(f"Running BFGS for {nsteps} steps")
            print(f"Running BFGS with fmax: {fmax}")
            _ = ase_opt.BFGS(atoms, trajectory=traj).run(fmax=fmax, steps=nsteps)
            # Sync with PyCHARMM
            xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
            coor.set_positions(xyz)
            traj.close()
            return atoms
        
        return atoms
        
    def optimize_as_monomers(atoms, run_index=0, nsteps=60, fmax=0.0006):
        optimized_atoms_positions = np.zeros_like(atoms.get_positions())
        # Use ASE and the original physnet calculator to optimize, one monomer at a time
        for i in range(n_monomers):
            off = int(monomer_offsets[i])
            n_i = atoms_per_monomer_list[i]
            monomer_atoms = atoms[off:off + n_i]
            monomer_atoms.calc = simple_physnet_calculator
            _ = ase_opt.BFGS(monomer_atoms).run(fmax=fmax, steps=nsteps)
            optimized_atoms_positions[off:off + n_i] = monomer_atoms.get_positions()

        atoms.set_positions(optimized_atoms_positions)
        # Wrap positions into cell after monomer optimization (avoids unwrapped coords for PBC)
        if args.cell is not None:
            wrapped = wrap_positions_for_pbc(atoms.get_positions(), masses=atoms.get_masses())
            atoms.set_positions(wrapped)
            xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
        else:
            xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)

        return atoms


        
    def run_ase_md(atoms, run_index=0, temperature=args.temperature):
        
        if run_index == 0 and getattr(args, "optimize_monomers", False):
            atoms = optimize_as_monomers(atoms, run_index=run_index, nsteps=100, fmax=0.0006)

        
        print(f"Pre-BFGS energy: {atoms.get_potential_energy():.6f} eV")

        atoms = minimize_structure(atoms, run_index=run_index,
         nsteps=100 if run_index == 0 else 10, fmax=0.0006 if run_index == 0 else 0.001)
        # Wrap positions into cell after BFGS (avoids unwrapped coords for PBC)
        if args.cell is not None:
            print(f"Wrapping positions into cell: {args.cell} Å")
            wrapped = wrap_positions_for_pbc(atoms.get_positions(), masses=atoms.get_masses())
            atoms.set_positions(wrapped)
            xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
            coor.set_positions(xyz)

        # Setup MD simulation
        
        timestep_fs = args.timestep
        num_steps = args.nsteps_ase
        ase_atoms = atoms
        
        # Draw initial momenta
        if run_index == 0:
            MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
            Stationary(ase_atoms)  # Remove center of mass translation
            ZeroRotation(ase_atoms)  # Remove rotations

        dt = timestep_fs*ase.units.fs
        print(f"Running ASE MD with timestep: {dt} (ase units)")
        # Initialize Velocity Verlet integrator
        integrator = VelocityVerlet(ase_atoms, timestep=dt)

        # Open trajectory file
        traj_filename = f'{run_index}_{args.output_prefix}_{temperature}K_{num_steps}steps_P{dt}.traj'
        Path(traj_filename).parent.mkdir(parents=True, exist_ok=True)
        traj = ase_io.Trajectory(traj_filename, 'w')

        # Run molecular dynamics
        frames = np.zeros((num_steps, len(ase_atoms), 3))
        potential_energy = np.zeros((num_steps,))
        kinetic_energy = np.zeros((num_steps,))
        total_energy = np.zeros((num_steps,))

        breakcount = 0
        ase_loop_start = time.perf_counter()
        for i in range(num_steps):
            # Run 1 time step
            integrator.run(1)
            # Do NOT wrap positions every timestep. The calculator uses MIC internally for
            # energy/forces (no coordinate transform). Integration uses unwrapped coordinates.
            # Wrap only for trajectory output if needed (see traj.write).
            # Save current frame and keep track of energies
            frames[i] = ase_atoms.get_positions()
            potential_energy[i] = ase_atoms.get_potential_energy()
            kinetic_energy[i] = ase_atoms.get_kinetic_energy()
            total_energy[i] = ase_atoms.get_total_energy()
            
            # Check for energy spikes and re-minimize if needed
            if i > 10 and (kinetic_energy[i] > 300 or potential_energy[i] > 0):
                print(f"Energy spike detected at step {i}, re-minimizing...")
                print(f"Kinetic energy: {kinetic_energy[i]:.6f} eV")
                print(f"Potential energy: {potential_energy[i]:.6f} eV")
                print(f"Total energy: {total_energy[i]:.6f} eV")
                print(f"Step: {i}")
                print(f"Breakcount: {breakcount}")
                print(f"Run index: {run_index}")
                print(f"Temperature: {temperature} K")
                print(f"Timestep: {timestep_fs} fs")
                print(f"Number of steps: {num_steps}")
                print(f"Number of atoms: {len(ase_atoms)}")
                print(f"Number of monomers: {n_monomers}")
                print(f"Atoms per monomer: {atoms_per_monomer_list}")
                print(f"ML cutoff: {args.ml_cutoff} Å")
                print(f"MM switch on: {args.mm_switch_on} Å")
                print(f"MM cutoff: {args.mm_cutoff} Å")
                print(f"Include MM: {args.include_mm}")
                print(f"Skip ML dimers: {args.skip_ml_dimers}")
                print(f"Debug: {args.debug}")
                
                minimize_structure(ase_atoms, run_index=f"{run_index}_{breakcount}_{i}_", nsteps=20 if run_index == 0 else 10, 
                fmax=0.0006 if run_index == 0 else 0.001, charmm=True)
                # assign new velocities
                # MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                cur_eng = ase_atoms.get_potential_energy()
                print(f"Current energy: {cur_eng:.6f} eV")
                Stationary(ase_atoms)
                ZeroRotation(ase_atoms)
                breakcount += 1
            if breakcount > 1:
                print("Maximum number of breaks reached")
                break
            # Occasionally print progress and adjust temperature
            if (i != 0) and (i % args.write_interval == 0):
                if args.cell is not None:
                    traj.write(ase_atoms)

                else:
                    traj.write(ase_atoms)
            if args.ensemble == "nvt":
                if  (i % args.heating_interval == 0):
                    Stationary(ase_atoms)
                    ZeroRotation(ase_atoms)
                    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                    print(f"Temperature adjusted to: {temperature} K")
            if i % 100 == 0:
                elapsed_s = time.perf_counter() - ase_loop_start
                completed_steps = i + 1
                simulated_ns = completed_steps * timestep_fs * 1e-6
                if simulated_ns > 0 and elapsed_s > 0:
                    avg_speed_ns_per_day = simulated_ns * 86400.0 / elapsed_s
                    time_per_ns_s = elapsed_s / simulated_ns
                    perf_msg = (
                        f"avg_speed {avg_speed_ns_per_day:8.4f} ns/day "
                        f"time_per_ns {time_per_ns_s:10.2f} s/ns"
                    )
                else:
                    perf_msg = "avg_speed n/a time_per_ns n/a"
                print(
                    f"step {i:5d} epot {potential_energy[i]: 5.3f} "
                    f"ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f} | "
                    f"{perf_msg}"
                )

        
        # Close trajectory file
        traj.close()
        print(f"Trajectory saved to: {traj_filename}")
        print("ASE MD simulation complete!")


    def set_up_nhc_sim_routine(atoms, T=args.temperature, dt=5e-3, steps_per_recording=250):
        @jax.jit
        def evaluate_energies_and_forces(
            atomic_numbers,
            positions,
            dst_idx,
            src_idx,
            mm_pair_idx=None,
            mm_pair_mask=None,
            box=None,
        ):
            return spherical_cutoff_calculator(
                atomic_numbers=atomic_numbers,
                positions=positions,
                n_monomers=n_monomers,
                cutoff_params=CUTOFF_PARAMS,
                doML=True,
                doMM=args.include_mm,
                doML_dimer=not args.skip_ml_dimers,
                debug=args.debug,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
            )


        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        R = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)

        @jit
        def jax_md_energy_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
            position = jnp.asarray(position, dtype=jnp.float32)
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=position,
                dst_idx=dst_idx,
                src_idx=src_idx,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
            )
            return result.energy.reshape(-1)[0]

        @jit
        def jax_md_force_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
            """Return forces from calculator (no autodiff). jax.grad(energy_fn) produces NaN."""
            position = jnp.asarray(position, dtype=jnp.float32)
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=position,
                dst_idx=dst_idx,
                src_idx=src_idx,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
            )
            return result.forces

        # evaluate_energies_and_forces (initial call - get update_fn if available)
        use_pbc = args.cell is not None
        is_npt = args.ensemble == "npt" and use_pbc
        update_fn = get_update_fn(R, CUTOFF_PARAMS) if get_update_fn else None
        pair_idx, pair_mask = None, None
        box_init = jnp.array([float(args.cell)]) if args.cell else None
        if update_fn is not None and use_pbc:
            if getattr(args, "debug", False):
                print("[nbr] Initial neighbor list update (PBC)")
            if is_npt:
                # NPT: neighbor list uses fractional_coordinates; pass frac pos and box [L,L,L]
                L = float(args.cell)
                R_frac = np.asarray(R) / L
                box_nl = np.array([L, L, L], dtype=np.float64)
                pair_idx, pair_mask = update_fn(R_frac, box=box_nl)
            else:
                # NVT/NVE: fixed box, do not pass box to update
                pair_idx, pair_mask = update_fn(np.asarray(R))
        result = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=R,
            dst_idx=dst_idx,
            src_idx=src_idx,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
            box=box_init,
        )
        print(f"Result: {result}")
        init_energy = result.energy.reshape(-1)[0]
        init_forces = result.forces.reshape(-1, 3)
        print(f"Initial energy: {init_energy:.6f} eV")
        print(f"Initial forces: {init_forces}")

        # MIC-only PBC: calculator uses minimum-image convention, no coordinate transform.
        pbc_map_fn = getattr(atoms.calc, "pbc_map", None) if atoms.calc else None
        if use_pbc:
            print(f"JAX-MD BOXSIZE: {float(args.cell)} Å, PBC: True (MIC-only)")
        else:
            print(f"JAX-MD: free space (no PBC), pbc_map: False")

        # Energy and force: use calculator's explicit forces (jax.grad through calculator gives NaN).
        # MIC-only PBC: no coordinate transform; calculator uses MIC internally.
        if use_pbc and pbc_map_fn is not None:
            @jax.custom_vjp
            def wrapped_energy_fn(position, **kwargs):
                pos = jnp.array(position)
                return jax_md_energy_fn(pbc_map_fn(pos), **kwargs)

            def wrapped_energy_fn_fwd(position, **kwargs):
                pos = jnp.array(position)
                R_mapped = pbc_map_fn(pos)
                E = jax_md_energy_fn(R_mapped, **kwargs)
                return E, (pos, R_mapped)

            def wrapped_energy_fn_bwd(res, g, **kwargs):
                pos, R_mapped = res
                result = evaluate_energies_and_forces(
                    atomic_numbers=atomic_numbers,
                    positions=R_mapped,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                )
                F_mapped = result.forces
                F_orig = pbc_map_fn.transform_forces(pos, F_mapped)
                return (F_orig,)

            wrapped_energy_fn.defvjp(wrapped_energy_fn_fwd, wrapped_energy_fn_bwd)
            wrapped_energy_fn = jit(wrapped_energy_fn)

            @jit
            def wrapped_force_fn(position, **kwargs):
                pos = jnp.array(position)
                R_mapped = pbc_map_fn(pos)
                F_mapped = jax_md_force_fn(R_mapped, **kwargs)
                return pbc_map_fn.transform_forces(pos, F_mapped)
        else:
            @jit
            def wrapped_energy_fn(position, **kwargs):
                return jax_md_energy_fn(jnp.array(position), **kwargs)

            wrapped_force_fn = jax_md_force_fn

        # Shift and displacement: NPT uses periodic_general with fractional coords; NVT/NVE use free or periodic
        is_npt = args.ensemble == "npt" and use_pbc
        if is_npt:
            # NPT: box as 3x3 diagonal (matches jax_md metal/Si example); volume = L^3
            L_npt = float(args.cell)
            box_npt = jnp.eye(3, dtype=jnp.float32) * L_npt
            displacement, shift = space.periodic_general(box=box_npt, fractional_coordinates=True)
        else:
            _displacement, _shift_free = space.free()
            shift = _shift_free
            displacement = _displacement

        unwrapped_init_fn, unwrapped_step_fn = jax_md.minimize.fire_descent(
            wrapped_force_fn, shift, dt_start=0.001, dt_max=0.001
        )
        unwrapped_step_fn = jit(unwrapped_step_fn)


        @jit
        def sim(state, neighbor=None, pressure=None):
            """Step function: for NPT pass neighbor and pressure; for NVT/NVE no kwargs."""
            def step_nve(i, s):
                return apply_fn(s)

            def step_npt(i, s):
                return apply_fn(s, neighbor=neighbor, pressure=pressure)

            step_fn = step_npt if (neighbor is not None and pressure is not None) else step_nve
            return lax.fori_loop(0, steps_per_recording, step_fn, state)




        # ========================================================================
        # SIMULATION PARAMETERS (metal units: eV, Å, ps, amu)
        # ========================================================================
        unit = units.metal_unit_system()
        # dt must be in ps: args.timestep is fs, 1 fs = 0.001 ps
        dt_fs = args.timestep
        dt = dt_fs * 0.001
        kT = T * unit['temperature']
        steps_per_recording = 1000
        rng_key = jax.random.PRNGKey(0)
        print(f"JAX-MD {args.ensemble.upper()}: dt={dt} ps ({dt_fs} fs), kT={kT} ({T} K)")

        # Select integrator based on ensemble
        if args.ensemble == "npt" and use_pbc:
            if update_fn is None:
                raise ValueError(
                    "NPT requires jax_md neighbor list (cell list cannot handle dynamic box). "
                    "Ensure jax_md is installed and pbc_cell is set."
                )
            pressure = getattr(args, 'pressure', 1.01325) * unit['pressure']
            # Barostat tau: 10000*dt (2.5 ps at 0.25 fs) avoids NaN from aggressive box scaling
            barostat_tau = getattr(args, 'nhc_barostat_tau', 10000.0) * dt
            nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
            nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
            nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
            nhc_tau = getattr(args, 'nhc_tau', 100.0)
            nhc_kwargs = {
                'chain_length': nhc_chain_length,
                'chain_steps': nhc_chain_steps,
                'sy_steps': nhc_sy_steps,
            }

            def _npt_energy_fn_raw(frac_pos, box=None, neighbor=None, perturbation=None, **kwargs):
                """Energy in fractional coords: transform to real, then evaluate.
                Supports perturbation=(1+eps) for NPT barostat stress (dU/dV)."""
                box_eff = jnp.asarray(box, dtype=jnp.float32)
                if perturbation is not None:
                    # Isotropic: V' = V * perturbation, so L' = L * perturbation^(1/3)
                    scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                    box_eff = box_eff * scale
                real_pos = space.transform(box_eff, frac_pos)
                pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
                result = evaluate_energies_and_forces(
                    atomic_numbers=atomic_numbers,
                    positions=real_pos,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                    mm_pair_idx=pair_idx,
                    mm_pair_mask=pair_mask,
                    box=box_eff,
                )
                return result.energy.reshape(-1)[0]

            @jax.custom_vjp
            def npt_energy_fn(frac_pos, box=None, neighbor=None, perturbation=None, kT=None, mass=None):
                """NPT energy with custom VJP: use explicit calculator forces (jax.grad gives NaN).
                All kwargs as explicit params so JAX resolve_kwargs can bind them to positions."""
                return _npt_energy_fn_raw(
                    frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
                )

            def npt_energy_fn_fwd(frac_pos, box, neighbor, perturbation, kT, mass):
                E = _npt_energy_fn_raw(
                    frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
                )
                return E, (frac_pos, box, neighbor, perturbation)

            def npt_energy_fn_bwd(res, g):
                frac_pos, box, neighbor, perturbation = res
                box_eff = jnp.asarray(box, dtype=jnp.float32)
                if perturbation is not None:
                    scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                    box_eff = box_eff * scale
                real_pos = space.transform(box_eff, frac_pos)
                pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
                F = jax_md_force_fn(
                    real_pos,
                    mm_pair_idx=pair_idx,
                    mm_pair_mask=pair_mask,
                    box=box_eff,
                )
                # grad(E) = -F; quantity.force = -grad, so we supply -F as grad
                grad_frac = -F * g
                return (grad_frac, None, None, None, None, None)

            npt_energy_fn.defvjp(npt_energy_fn_fwd, npt_energy_fn_bwd)
            npt_energy_fn = jit(npt_energy_fn)
            init_fn, apply_fn = simulate.npt_nose_hoover(
                npt_energy_fn,
                shift,
                dt=dt,
                pressure=pressure,
                kT=kT,
                barostat_kwargs=default_nhc_kwargs(jnp.array(barostat_tau), nhc_kwargs),
                thermostat_kwargs=default_nhc_kwargs(jnp.array(nhc_tau * dt), nhc_kwargs),
            )
            print(f"NPT Nose-Hoover: pressure={getattr(args, 'pressure', 1.01325)} bar, barostat_tau={barostat_tau:.6f} ps, "
                  f"thermostat tau={nhc_tau * dt:.6f} ps")
        elif args.ensemble == "nvt":
            nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
            nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
            nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
            nhc_tau = getattr(args, 'nhc_tau', 100.0)
            nhc_kwargs = {
                'chain_length': nhc_chain_length,
                'chain_steps': nhc_chain_steps,
                'sy_steps': nhc_sy_steps,
            }
            init_fn, apply_fn = simulate.nvt_nose_hoover(
                wrapped_force_fn, shift, dt=dt, kT=kT,
                thermostat_kwargs=default_nhc_kwargs(
                    jnp.array(nhc_tau * dt), nhc_kwargs
                ),
            )
            print(f"NVT Nose-Hoover chain: chain_length={nhc_chain_length}, "
                  f"chain_steps={nhc_chain_steps}, sy_steps={nhc_sy_steps}, "
                  f"tau={nhc_tau * dt:.6f} ps")
        else:  # nve
            init_fn, apply_fn = simulate.nve(wrapped_force_fn, shift, dt)
        apply_fn = jit(apply_fn)

        def run_sim(
            key,
            total_steps=args.nsteps_jaxmd,
            steps_per_recording=steps_per_recording,
            R=R,
            skip_minimization=False,
        ):
            total_records = total_steps // steps_per_recording

            # When ASE already ran (nsteps_ase > 0), R is ASE-minimized; skip JAX-MD minimization
            if skip_minimization:
                minimized_pos = jnp.asarray(R, dtype=jnp.float32)
                print("Skipping JAX-MD minimization (using ASE positions)")
            else:
                # Translate to center of mass before minimization (use actual masses)
                com = jnp.sum(Si_mass[:, None] * R, axis=0) / Si_mass.sum()
                initial_pos = jnp.asarray(R - com, dtype=jnp.float32)
                # Sanity check: ensure energy/gradient are finite at start; else use R directly
                try:
                    _e0 = float(wrapped_energy_fn(initial_pos))
                    _f0 = wrapped_force_fn(initial_pos)
                    if not (np.isfinite(_e0) and np.all(np.isfinite(np.asarray(_f0)))):
                        initial_pos = jnp.asarray(R, dtype=jnp.float32)
                        print("Non-finite energy/forces at COM-centered pos; using R directly for minimization")
                except Exception:
                    initial_pos = jnp.asarray(R, dtype=jnp.float32)
                    print("Fallback: using R directly for minimization")
                fire_state = unwrapped_init_fn(initial_pos, mass=Si_mass)
                fire_positions = []

                # FIRE minimization with step rejection (reject steps that produce NaN)
                print("*" * 10 + "\nMinimization\n" + "*" * 10)
                NMIN = 1000
                for i in range(NMIN):
                    fire_positions.append(fire_state.position)
                    new_state = unwrapped_step_fn(fire_state)
                    # Reject step if it produces NaN/Inf positions
                    if not jnp.all(jnp.isfinite(new_state.position)):
                        print("FIRE step produced NaN/Inf positions; rejecting and stopping")
                        break
                    # Check energy/forces at new position before accepting
                    energy = float(wrapped_energy_fn(new_state.position))
                    max_force = float(jnp.abs(wrapped_force_fn(new_state.position)).max())
                    if not (np.isfinite(energy) and np.isfinite(max_force)):
                        print("FIRE step led to NaN/Inf energy or forces; rejecting and stopping")
                        break
                    fire_state = new_state

                    if i % (NMIN // 10) == 0:
                        print(f"{i}/{NMIN}: E={energy:.6f} eV, max|F|={max_force:.6f}")
                # fire_state always holds last valid position (we reject bad steps)
                minimized_pos = fire_state.position
                if jnp.any(~jnp.isfinite(minimized_pos)) and fire_positions:
                    minimized_pos = fire_positions[-1]
                    print("Using last valid position from first minimization")
            # save pdb
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            min_pdb_path = Path(f"{args.output_prefix}_minimized_{current_time}.pdb")
            min_pdb_path.parent.mkdir(parents=True, exist_ok=True)
            ase_io.write(str(min_pdb_path), atoms)

            # ========================================================================
            # PBC MINIMIZATION (only when PBC enabled, i.e. cell is set)
            # ========================================================================
            pbc_fire_positions = []
            if not use_pbc:
                md_pos = minimized_pos
                print("No cell: skipping PBC minimization, using first-min result")
                atoms.set_positions(np.asarray(md_pos))
            else:
                print("*" * 10 + "\nPBC Minimization\n" + "*" * 10)
                pbc_unwrapped_init_fn, pbc_unwrapped_step_fn = jax_md.minimize.fire_descent(
                    wrapped_force_fn, shift, dt_start=0.001, dt_max=0.001
                )
                pbc_unwrapped_step_fn = jit(pbc_unwrapped_step_fn)
                # Start from wrapped positions so we're in the cell (first min can drift)
                if pbc_map_fn is not None:
                    pbc_start_pos = pbc_map_fn(minimized_pos)
                else:
                    # MIC-only: wrap by monomer into cell
                    _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
                    _monomer_groups = [
                        jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                        for m in range(n_monomers)
                    ]
                    pbc_start_pos = wrap_groups(
                        jnp.asarray(minimized_pos), _monomer_groups, _cell_jax, mass=Si_mass
                    )
                pbc_fire_state = pbc_unwrapped_init_fn(pbc_start_pos, mass=Si_mass)

                # Run PBC minimization (track best; stop early if forces increase - FIRE+unwrapped can wander)
                # Skip when first minimization already failed (minimized_pos invalid)
                NMIN_PBC = 1000
                if jnp.any(~jnp.isfinite(pbc_start_pos)):
                    print("Skipping PBC minimization (no valid start position)")
                    md_pos = minimized_pos
                else:
                    max_force_start = float(jnp.abs(wrapped_force_fn(pbc_start_pos)).max())
                    best_pbc_pos = pbc_start_pos
                    best_pbc_max_f = max_force_start
                    worsen_count = 0
                    prev_max_f = max_force_start
                    for i in range(NMIN_PBC):
                        pbc_fire_positions.append(pbc_fire_state.position)
                        new_pbc_state = pbc_unwrapped_step_fn(pbc_fire_state)
                        # Reject step if it produces NaN
                        if not jnp.all(jnp.isfinite(new_pbc_state.position)):
                            print("PBC FIRE step produced NaN; using first-min result")
                            break
                        energy = float(wrapped_energy_fn(new_pbc_state.position))
                        max_force = float(jnp.abs(wrapped_force_fn(new_pbc_state.position)).max())
                        if not (np.isfinite(energy) and np.isfinite(max_force)):
                            print("PBC minimization hit NaN energy/forces; using first-min result")
                            break
                        pbc_fire_state = new_pbc_state
                        if max_force < best_pbc_max_f:
                            best_pbc_max_f = max_force
                            best_pbc_pos = pbc_fire_state.position
                            worsen_count = 0
                        else:
                            worsen_count = worsen_count + 1 if max_force > prev_max_f else 0
                        prev_max_f = max_force
                        if i % (NMIN_PBC // 10) == 0:
                            print(f"{i}/{NMIN_PBC}: E={energy:.6f} eV, max|F|={max_force:.6f}")
                        if worsen_count >= 10:
                            print(f"PBC minimization: max|F| increased for 10 steps; stopping early at step {i} (best max|F|={best_pbc_max_f:.4f})")
                            break

                    # Use first-min result if PBC minimization worsened structure (max_force increased)
                    if best_pbc_max_f > max_force_start * 1.1:
                        md_pos = pbc_map_fn(minimized_pos) if pbc_map_fn else minimized_pos
                        print(f"PBC minimization increased max|F| ({max_force_start:.4f} -> {best_pbc_max_f:.4f}); using first-min wrapped structure")
                    else:
                        md_pos = best_pbc_pos

                # Save PBC minimized structure
                atoms.set_positions(np.asarray(md_pos))
                pbc_current_time = datetime.now().strftime("%H:%M:%S")
                pbc_pdb_path = Path(f"{args.output_prefix}_pbc_minimized_{pbc_current_time}.pdb")
                pbc_pdb_path.parent.mkdir(parents=True, exist_ok=True)
                ase_io.write(str(pbc_pdb_path), atoms)
                print(f"PBC minimization complete. Final energy: {float(wrapped_energy_fn(md_pos)):.6f} eV")

            # Use last valid positions if minimization produced NaN
            if jnp.any(~jnp.isfinite(md_pos)) and pbc_fire_positions:
                md_pos = pbc_fire_positions[-1]
                print("Warning: NaN in PBC minimization, using last valid position from PBC")
            if jnp.any(~jnp.isfinite(md_pos)) and fire_positions:
                md_pos = pbc_map_fn(fire_positions[-1]) if (use_pbc and pbc_map_fn) else fire_positions[-1]
                print("Warning: Using last valid position from first minimization")
            if jnp.any(~jnp.isfinite(md_pos)):
                print(f"Error: No valid positions for {args.ensemble.upper()}; skipping JAX-MD simulation")
                return 0, jnp.array([]).reshape(0, len(md_pos), 3), None

            
            if args.ensemble == "npt" and use_pbc:
                # NPT: positions in fractional coords; wrap md_pos into cell first, then convert to fractional
                box_curr = box_npt
                _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
                _monomer_groups = [
                    jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                    for m in range(n_monomers)
                ]
                md_pos_wrapped = wrap_groups(
                    jnp.asarray(md_pos), _monomer_groups, _cell_jax, mass=Si_mass
                )
                md_pos_frac = md_pos_wrapped / float(args.cell)  # cubic: frac = R / L
                # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
                box_nl = np.array([float(args.cell)] * 3, dtype=np.float64)
                pair_idx, pair_mask = update_fn(np.asarray(md_pos_frac), box=box_nl)
                state = init_fn(
                    key, md_pos_frac, box=box_curr,
                    neighbor=(pair_idx, pair_mask), kT=kT, mass=Si_mass
                )
                npt_pair_idx, npt_pair_mask = pair_idx, pair_mask
                npt_pressure = getattr(args, 'pressure', 1.01325) * unit['pressure']
            elif args.ensemble == "nvt":
                state = init_fn(key, md_pos, mass=Si_mass)
                npt_pair_idx, npt_pair_mask = None, None
                npt_pressure = None
            else:
                state = init_fn(key, md_pos, kT, mass=Si_mass)
                npt_pair_idx, npt_pair_mask = None, None
                npt_pressure = None
            print(f"Momentum initialized for {T} K")
            nhc_positions = []
            nhc_boxes = []  # NPT: box at each record step (for frac→real when saving)

            # get energy of initial state
            if is_npt and npt_pair_idx is not None:
                box_curr = simulate.npt_box(state)
                energy_initial = float(npt_energy_fn(state.position, box=box_curr, neighbor=(npt_pair_idx, npt_pair_mask)))
            else:
                energy_initial = float(wrapped_energy_fn(state.position))
            print(f"Initial energy: {energy_initial:.6f} eV")
            # Debug: forces from calculator (used by NVE; jax.grad gives NaN)
            if is_npt and npt_pair_idx is not None:
                box_curr = simulate.npt_box(state)
                real_pos = space.transform(box_curr, state.position)
                forces_jax = jax_md_force_fn(
                    real_pos,
                    mm_pair_idx=npt_pair_idx,
                    mm_pair_mask=npt_pair_mask,
                    box=box_curr,
                )
            else:
                forces_jax = wrapped_force_fn(state.position)
            print(f"JAX-MD initial forces (from calculator):\n{forces_jax}")
            # velocity = momentum / mass; position update = R + dt * v (half-step in VV)
            vel = state.momentum / state.mass
            disp_first = dt * vel
            print(f"JAX-MD velocity (p/m) sample [0]: {vel[0]}")
            print(f"JAX-MD first-step displacement dt*v [0]: {disp_first[0]}, max|disp|: {float(jnp.max(jnp.abs(disp_first))):.6f}")

            # ========================================================================
            # NPT DIAGNOSTIC TESTS (--npt-diagnose)
            # ========================================================================
            if is_npt and npt_pair_idx is not None and getattr(args, "npt_diagnose", False):
                _run_npt_diagnostics(
                    state=state,
                    npt_energy_fn=npt_energy_fn,
                    jax_md_force_fn=jax_md_force_fn,
                    apply_fn=apply_fn,
                    shift=shift,
                    space=space,
                    simulate=simulate,
                    quantity=quantity,
                    npt_pair_idx=npt_pair_idx,
                    npt_pair_mask=npt_pair_mask,
                    npt_pressure=npt_pressure,
                    dt=dt,
                    kT=kT,
                    grad=grad,
                )

            # Single-step diagnostic: catch NaN on first step (common with wrong mass/units)
            if is_npt and npt_pair_idx is not None:
                state_one = apply_fn(state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure)
            else:
                state_one = apply_fn(state)
            if not jnp.all(jnp.isfinite(state_one.position)):
                print("ERROR: First step produced NaN positions. Skipping JAX-MD.")
                print("  Check: mass in amu, dt in ps, energy_fn returns eV.")
                print(f"  mass shape: {state.mass.shape}, min/max: {float(jnp.min(state.mass)):.4f}/{float(jnp.max(state.mass)):.4f}")
                pos_out = space.transform(simulate.npt_box(state), state.position) if is_npt else state.position
                box_out = [np.asarray(jax.device_get(simulate.npt_box(state)))] if is_npt else None
                return 0, np.stack([np.asarray(jax.device_get(pos_out))]), box_out
            if is_npt and npt_pair_idx is not None:
                box_one = simulate.npt_box(state_one)
                e1 = float(npt_energy_fn(state_one.position, box=box_one, neighbor=(npt_pair_idx, npt_pair_mask)))
            else:
                e1 = float(wrapped_energy_fn(state_one.position))
            print(f"First step OK: E_pot={e1:.6f} eV")

            print("*" * 10 + f"\n{args.ensemble.upper()}\n" + "*" * 10)
            print("\t\tTime (ps)\tSteps\tE_pot (eV)\tE_tot (eV)\tT (K)\tt/ns (s)\tavg(ns/day)")

            # ========================================================================
            # HDF5 REPORTER SETUP
            # ========================================================================
            hdf5_path = Path(f"{args.output_prefix}_{args.ensemble}_trajectory.h5")
            hdf5_path.parent.mkdir(parents=True, exist_ok=True)
            hdf5_reporter = make_jaxmd_reporter(
                str(hdf5_path),
                n_atoms=len(atoms),
                buffer_size=min(100, total_records),
                include_positions=True,
                include_velocities=True,
                scalar_quantities=["total_energy", "time_ps"],
                attrs={
                    "ensemble": args.ensemble,
                    "temperature_target": T,
                    "dt_ps": dt,
                    "steps_per_recording": steps_per_recording,
                    "n_atoms": len(atoms),
                    "atomic_numbers": atoms.get_atomic_numbers(),
                },
            )

            # ========================================================================
            # PBC WRAPPING SETUP
            # ========================================================================
            if use_pbc:
                _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
                _monomer_groups = [
                    jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
                    for m in range(n_monomers)
                ]
                print(f"PBC wrapping enabled: {n_monomers} monomer groups, "
                      f"wrapping every {steps_per_recording} steps")

            # ========================================================================
            # MAIN SIMULATION LOOP
            # ========================================================================
            jaxmd_loop_start = time.perf_counter()

            for i in range(total_records):
                if is_npt and update_fn is not None:
                    box_curr = simulate.npt_box(state)
                    # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
                    box_nl = np.asarray(box_curr)
                    if box_nl.shape == (1,) or box_nl.ndim == 0:
                        L = float(box_nl.reshape(-1)[0])
                        box_nl = np.array([L, L, L], dtype=np.float64)
                    if getattr(args, "debug", False) and (i < 3 or i % 50 == 0):
                        print(f"[nbr] NPT record {i}: updating neighbor list, box L={float(box_nl[0]):.4f}")
                    npt_pair_idx, npt_pair_mask = update_fn(
                        np.asarray(state.position), box=box_nl
                    )
                    state = sim(state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure)
                else:
                    state = sim(state)

                if use_pbc:
                    if is_npt:
                        # NPT: wrap fractional coords to [0,1)
                        box_curr = simulate.npt_box(state)
                        frac_pos = state.position
                        wrapped_frac = frac_pos - jnp.floor(frac_pos)
                        state = state.set(position=wrapped_frac)
                    else:
                        wrapped_pos = wrap_groups(
                            state.position, _monomer_groups, _cell_jax, mass=Si_mass
                        )
                        state = state.set(position=wrapped_pos)

                # Store current position (NPT: fractional + box for correct real coords at save)
                if is_npt:
                    box_curr = simulate.npt_box(state)
                    nhc_positions.append(state.position)
                    nhc_boxes.append(box_curr)
                else:
                    nhc_positions.append(state.position)
                    
                # Print progress every 10 steps
                if i % 10 == 0:
                    steps = (i + 1) * steps_per_recording
                    time_ps = steps * dt
                    T_curr = jax_md.quantity.temperature(
                        momentum=state.momentum,
                        mass=state.mass
                    ) / unit['temperature']
                    temp = float(T_curr)
                    if is_npt and npt_pair_idx is not None:
                        box_curr = simulate.npt_box(state)
                        e_pot = float(npt_energy_fn(state.position, box=box_curr, neighbor=(npt_pair_idx, npt_pair_mask)))
                    else:
                        e_pot = float(wrapped_energy_fn(state.position))
                    e_kin = float(jax_md.quantity.kinetic_energy(
                        momentum=state.momentum,
                        mass=state.mass
                    ))
                    e_tot = e_pot + e_kin
                    elapsed_s = time.perf_counter() - jaxmd_loop_start
                    simulated_ns = steps * dt_fs * 1e-6
                    if simulated_ns > 0 and elapsed_s > 0:
                        avg_speed_ns_per_day = simulated_ns * 86400.0 / elapsed_s
                        time_per_ns_s = elapsed_s / simulated_ns
                    else:
                        avg_speed_ns_per_day = float("nan")
                        time_per_ns_s = float("nan")
                    print(
                        f"{time_ps:10.4f}\t{steps:6d}\t{e_pot:10.4f}\t{e_tot:10.4f}\t{temp:10.2f}\t"
                        f"{time_per_ns_s:10.2f}\t{avg_speed_ns_per_day:10.4f}"
                    )

                    # Record to HDF5 (NPT: save real positions via transform)
                    pos_for_h5 = state.position
                    if is_npt:
                        box_curr = simulate.npt_box(state)
                        pos_for_h5 = space.transform(box_curr, state.position)
                    hdf5_reporter.report(
                        potential_energy=e_pot,
                        kinetic_energy=e_kin,
                        temperature=temp,
                        invariant=e_tot,
                        total_energy=e_tot,
                        time_ps=time_ps,
                        positions=pos_for_h5,
                        velocities=state.momentum / state.mass,
                    )

                    # Stop on numerical instability (NaN, Inf, or energy blow-up to 0)
                    if not np.isfinite(e_pot) or not np.isfinite(temp):
                        print(f"Numerical instability at step {steps}; stopping.")
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                            if is_npt:
                                nhc_boxes = nhc_boxes[:-1]
                            state = type(state)(
                                position=nhc_positions[-1],
                                momentum=state.momentum,
                                mass=state.mass
                            )
                        break
                    if e_pot >= 0 and energy_initial < 0:
                        print(f"Energy blow-up at step {steps} (E_pot={e_pot:.4f}); stopping.")
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                            if is_npt:
                                nhc_boxes = nhc_boxes[:-1]
                            state = type(state)(
                                position=nhc_positions[-1],
                                momentum=state.momentum,
                                mass=state.mass
                            )
                        break

            hdf5_reporter.close()
            print(f"HDF5 trajectory saved to: {hdf5_path}")

            steps_completed = i * steps_per_recording
            print(f"\nSimulated {steps_completed} steps ({steps_completed * dt:.2f} ps)")


            nhc_positions_out = []
            nhc_boxes_out = []  # NPT: real-space box per frame for trajectory cell
            for idx, R in enumerate(nhc_positions):
                if is_npt:
                    # NPT: convert fractional to real using box at this step
                    box_i = nhc_boxes[idx]
                    R = space.transform(box_i, R)
                    nhc_boxes_out.append(np.asarray(jax.device_get(box_i)))
                elif use_pbc and pbc_map_fn is not None:
                    R = pbc_map_fn(R)
                nhc_positions_out.append(np.asarray(jax.device_get(R)))
            return steps_completed, np.stack(nhc_positions_out), nhc_boxes_out if is_npt else None

        return run_sim


    def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="traj", boxes=None, save_energy_forces=True):
        """Save trajectory in real (Cartesian) space. For NPT, pass boxes to set cell per frame.
        When save_energy_forces=True, recalculates and stores energy and forces for each frame."""
        trajectory = Trajectory(f"{filename}.{format}", "a")
        out_positions = np.asarray(out_positions).reshape(-1, len(atoms), 3)
        for i, R in enumerate(out_positions):
            atoms.set_positions(np.asarray(R))
            if boxes is not None and i < len(boxes):
                # NPT: set cell to match box at this frame (positions are in real space)
                box = np.asarray(boxes[i])
                if box.ndim == 2:
                    atoms.set_cell(box)
                elif box.size >= 3:
                    atoms.set_cell(np.diag(np.asarray(box).reshape(3)))
            if save_energy_forces and atoms.calc is not None:
                _ = atoms.get_potential_energy()
                _ = atoms.get_forces()
            trajectory.write(atoms)
        trajectory.close()


    def run_sim_loop(run_sim, sim_key, nsim=1, skip_minimization=False):
        """
        Run the simulation for the given indices and save the trajectory.
        Uses current atoms positions (after ASE MD if run) as initial positions.
        """
        out_positions = []
        out_boxes = []  # NPT: boxes per frame for trajectory cell
        max_is = []
        pos = np.asarray(atoms.get_positions(), dtype=np.float32)
        for i in range(nsim):
            mi, pos, boxes = run_sim(sim_key, R=pos, skip_minimization=skip_minimization)
            out_positions.append(pos)
            out_boxes.append(boxes)
            max_is.append(mi)

        return out_positions, out_boxes, max_is


    sim_key, data_key = jax.random.split(jax.random.PRNGKey(42), 2)
    temperature = args.temperature
    if args.nsteps_ase > 0:
        for i in range(1):
            run_ase_md(atoms, run_index=i, temperature=temperature)


    if args.nsteps_jaxmd > 0:
        for j in range(1):
            sim_key, data_key = jax.random.split(data_key, 2)
            s = set_up_nhc_sim_routine(atoms, T=temperature)
            # Skip JAX-MD minimization when ASE already ran (positions are ASE-minimized)
            skip_jaxmd_min = args.nsteps_ase > 0
            out_positions, out_boxes, _ = run_sim_loop(s, sim_key, skip_minimization=skip_jaxmd_min)

            print(f"Out positions: {out_positions}")

            for i in range(len(out_positions)):
                traj_filename = f'{args.output_prefix}_md_trajectory_{j}_{i}'
                Path(traj_filename).parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving trajectory to: {traj_filename}.traj")
                save_trajectory(
                    out_positions[i], atoms, filename=traj_filename,
                    boxes=out_boxes[i] if out_boxes and out_boxes[i] is not None else None,
                )

            # atoms = minimize_structure(atoms)

    print("Trajectories saved!")


    print("JAX-MD simulation complete!")
    return atoms




def main() -> int:
    """CLI entry point: parse args and run simulation."""
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
