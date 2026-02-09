"""
ASE Molecular Dynamics with the Electric Field calculator.

Runs NVT (Langevin) or NVE (VelocityVerlet) MD using AseCalculatorEF.

Usage:
    python ase_md.py --params params.json --data data-full.npz --steps 1000 --dt 0.5
    python ase_md.py --params params.json --data data-full.npz --thermostat langevin --temperature 300
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import argparse
import sys
from types import SimpleNamespace
import numpy as np
import ase
import ase.io as ase_io
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ase_calc_EF import AseCalculatorEF


def get_args(**kwargs):
    """
    Get configuration arguments. Works both from command line and notebooks.
    
    In notebooks, you can override defaults by passing keyword arguments:
        args = get_args(params="params.json", steps=1000, temperature=300)
    
    From command line, use argparse flags as before.
    """
    # Default values
    defaults = {
        "params": "params.json",
        "config": None,
        "data": "data-full.npz",
        "xyz": None,
        "index": 0,
        "electric_field": None,
        "thermostat": "langevin",
        "temperature": 300.0,
        "friction": 0.01,
        "dt": 0.5,
        "steps": 1000,
        "log_interval": 10,
        "traj_interval": 10,
        "output": "md_trajectory.traj",
        "seed": 42,
        "save_charges": False,
        "optimize": False,
        "optimizer": "fire",
        "fmax": 0.05,
        "opt_steps": 2000,
        "maxstep": 0.04,
    }
    
    # Check if we're in a notebook/IPython environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    # If kwargs are provided, always use notebook mode
    if kwargs:
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)
    
    # Check if any command line arguments look like our flags (start with --)
    has_flag_args = any(arg.startswith('--') for arg in sys.argv[1:])
    
    # If command line arguments are provided AND we're not in a notebook, use argparse
    if has_flag_args and not in_notebook:
        parser = argparse.ArgumentParser(description="Run MD with AseCalculatorEF")
        parser.add_argument("--params", type=str, default=defaults["params"],
                           help="Path to parameters JSON file")
        parser.add_argument("--config", type=str, default=defaults["config"],
                           help="Path to config JSON file (auto-detected from params UUID)")
        parser.add_argument("--data", type=str, default=defaults["data"],
                           help="Path to dataset NPZ file (to extract initial geometry)")
        parser.add_argument("--xyz", type=str, default=defaults["xyz"],
                           help="Path to XYZ file for initial geometry (overrides --data)")
        parser.add_argument("--index", type=int, default=defaults["index"],
                           help="Index of structure in dataset to use as starting geometry")
        parser.add_argument("--electric-field", type=float, nargs=3, default=defaults["electric_field"],
                           help="Electric field vector (Ef_x, Ef_y, Ef_z). "
                                "If omitted, uses dataset value; pass '0 0 0' for zero field.")
        parser.add_argument("--thermostat", type=str, default=defaults["thermostat"],
                           choices=["langevin", "nve"],
                           help="Thermostat type: 'langevin' (NVT) or 'nve' (NVE)")
        parser.add_argument("--temperature", type=float, default=defaults["temperature"],
                           help="Temperature in Kelvin (for Langevin thermostat)")
        parser.add_argument("--friction", type=float, default=defaults["friction"],
                           help="Friction coefficient for Langevin thermostat (1/fs)")
        parser.add_argument("--dt", type=float, default=defaults["dt"],
                           help="Time step in femtoseconds")
        parser.add_argument("--steps", type=int, default=defaults["steps"],
                           help="Number of MD steps")
        parser.add_argument("--log-interval", type=int, default=defaults["log_interval"],
                           help="Log every N steps")
        parser.add_argument("--traj-interval", type=int, default=defaults["traj_interval"],
                           help="Save trajectory every N steps")
        parser.add_argument("--output", type=str, default=defaults["output"],
                           help="Output trajectory file (ASE .traj format)")
        parser.add_argument("--seed", type=int, default=defaults["seed"],
                           help="Random seed for initial velocities")
        parser.add_argument("--save-charges", action="store_true",
                           help="Save ML atomic charges per frame (slower, for VCD)")
        parser.add_argument("--optimize", action="store_true",
                           help="Geometry-optimise before starting MD")
        parser.add_argument("--optimizer", choices=["bfgs", "fire"], default=defaults["optimizer"],
                           help="Optimiser: 'bfgs' or 'fire' (default fire, more robust for ML)")
        parser.add_argument("--fmax", type=float, default=defaults["fmax"],
                           help="Convergence criterion: max force (eV/Å)")
        parser.add_argument("--opt-steps", type=int, default=defaults["opt_steps"],
                           help="Max optimisation steps")
        parser.add_argument("--maxstep", type=float, default=defaults["maxstep"],
                           help="Max step size in Å (default 0.04; ASE default 0.2)")
        
        args = parser.parse_args()
        return SimpleNamespace(
            params=args.params,
            config=args.config,
            data=args.data,
            xyz=args.xyz,
            index=args.index,
            electric_field=args.electric_field,
            thermostat=args.thermostat,
            temperature=args.temperature,
            friction=args.friction,
            dt=args.dt,
            steps=args.steps,
            log_interval=args.log_interval,
            traj_interval=args.traj_interval,
            output=args.output,
            seed=args.seed,
            save_charges=args.save_charges,
            optimize=args.optimize,
            optimizer=args.optimizer,
            fmax=args.fmax,
            opt_steps=args.opt_steps,
            maxstep=args.maxstep,
        )
    
    # Otherwise, use notebook mode (defaults only)
    return SimpleNamespace(**defaults)


def run_md(args):
    """Run molecular dynamics simulation."""
    print("=" * 60)
    print("ASE Molecular Dynamics with Electric Field Model")
    print("=" * 60)

    # --- Load or build initial geometry ---
    if args.xyz is not None:
        print(f"\nLoading initial geometry from {args.xyz}...")
        atoms = ase_io.read(args.xyz)
    else:
        print(f"\nLoading initial geometry from {args.data} (index={args.index})...")
        dataset = np.load(args.data, allow_pickle=True)
        Z = dataset["Z"][args.index]
        R = dataset["R"][args.index]
        if R.ndim == 3 and R.shape[0] == 1:
            R = R.squeeze(0)
        atoms = ase.Atoms(numbers=Z, positions=R)

    # Set electric field
<<<<<<< HEAD
    Ef = np.array(args.electric_field, dtype=np.float64)
    # If electric field is all zeros and dataset has Ef, use dataset value
    if np.allclose(Ef, 0.0) and args.xyz is None:
        dataset = np.load(args.data, allow_pickle=True)
        if "Ef" in dataset.files:
            Ef = np.array(dataset["Ef"][args.index], dtype=np.float64)
            print(f"  Using electric field from dataset: {Ef}")
=======
    if args.electric_field is not None:
        Ef = np.array(args.electric_field, dtype=np.float64)
        print(f"  Electric field (CLI): {Ef}")
    elif args.xyz is None:
        dataset = np.load(args.data, allow_pickle=True)
        if "Ef" in dataset.files:
            Ef = np.array(dataset["Ef"][args.index], dtype=np.float64)
            print(f"  Electric field (dataset): {Ef}")
        else:
            Ef = np.zeros(3, dtype=np.float64)
            print(f"  Electric field (default): {Ef}")
    else:
        Ef = np.zeros(3, dtype=np.float64)
        print(f"  Electric field (default): {Ef}")
>>>>>>> bdfda8dbbf49b1f2d64d87d88d2231ee12619451
    atoms.info['electric_field'] = Ef

    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Atomic numbers: {atoms.get_atomic_numbers()}")
    print(f"  Electric field: {Ef}")

    # --- Create calculator ---
    print(f"\nInitializing calculator from {args.params}...")
    calc = AseCalculatorEF(
        params_path=args.params,
        config_path=args.config,
        electric_field=Ef,
    )
    atoms.calc = calc

    # --- Initial energy/forces ---
    print("\nComputing initial properties...")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"  Initial energy: {energy:.6f} eV ({energy * 23.06035:.4f} kcal/mol)")
    print(f"  Max force: {np.max(np.abs(forces)):.6f} eV/A")

<<<<<<< HEAD
=======
    # --- Optional geometry optimisation before MD ---
    if args.optimize:
        from ase.optimize import BFGS, FIRE
        opt_cls = FIRE if args.optimizer == 'fire' else BFGS
        print(f"\nOptimising geometry ({args.optimizer.upper()}, fmax={args.fmax} eV/Å, "
              f"max {args.opt_steps} steps, maxstep={args.maxstep} Å) ...")
        opt_traj = str(Path(args.output).with_suffix('.opt.traj'))
        opt = opt_cls(atoms, trajectory=opt_traj, logfile='-',
                      maxstep=args.maxstep)
        opt.run(fmax=args.fmax, steps=args.opt_steps)
        fmax_final = np.max(np.abs(atoms.get_forces()))
        if fmax_final <= args.fmax:
            print(f"  Converged in {opt.nsteps} steps.")
        else:
            print(f"  WARNING: not converged after {opt.nsteps} steps  "
                  f"(max |F| = {fmax_final:.6f} eV/Å)")
            print(f"  Consider increasing --opt-steps or relaxing --fmax.")
        energy = atoms.get_potential_energy()
        print(f"  Optimised energy: {energy:.6f} eV")
        print(f"  Optimised max|F|: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
        print(f"  Opt trajectory  : {opt_traj}")

>>>>>>> bdfda8dbbf49b1f2d64d87d88d2231ee12619451
    # --- Initialize velocities ---
    print(f"\nInitializing velocities at T={args.temperature} K (seed={args.seed})...")
    np.random.seed(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    # --- Set up integrator ---
    dt = args.dt * units.fs

    if args.thermostat == "langevin":
        print(f"\nUsing Langevin thermostat: T={args.temperature} K, friction={args.friction} 1/fs, dt={args.dt} fs")
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=args.temperature,
            friction=args.friction / units.fs,
        )
    else:
        print(f"\nUsing NVE (VelocityVerlet): dt={args.dt} fs")
        dyn = VelocityVerlet(atoms, timestep=dt)

    # --- Prepare ASE trajectory output ---
    output_path = Path(args.output)
    traj = Trajectory(str(output_path), 'w', atoms)

    # --- MD loop ---
    print(f"\nRunning {args.steps} MD steps...")
    print(f"{'Step':>8s} {'Time(fs)':>10s} {'E_pot(eV)':>12s} {'E_kin(eV)':>12s} {'E_tot(eV)':>12s} {'T(K)':>8s} {'MaxF(eV/A)':>12s}")
    print("-" * 80)

    def print_status():
        """Print MD status at current step."""
        step = dyn.nsteps
        time_fs = step * args.dt
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        e_tot = e_pot + e_kin
        temp = e_kin / (1.5 * units.kB * len(atoms))
        max_force = np.max(np.abs(atoms.get_forces()))
        print(f"{step:8d} {time_fs:10.2f} {e_pot:12.6f} {e_kin:12.6f} {e_tot:12.6f} {temp:8.1f} {max_force:12.6f}")

    # --- Property-saving callback (dipole always, charges optional) ---
    def save_properties():
        """Copy model predictions into atoms.info so they persist in .traj."""
        results = getattr(atoms.calc, 'results', {})
        if 'dipole' in results:
            atoms.info['ml_dipole'] = np.array(results['dipole'])
        if args.save_charges:
            try:
                q, mu_at = calc.get_atomic_charges(atoms)
                atoms.arrays['ml_charges'] = q
                atoms.arrays['ml_atomic_dipoles'] = mu_at
            except Exception:
                pass

    # Log and save initial state
    print_status()
    save_properties()
    traj.write()

    # Attach callbacks — save_properties BEFORE traj.write
    dyn.attach(print_status, interval=args.log_interval)
    dyn.attach(save_properties, interval=args.traj_interval)
    dyn.attach(traj.write, interval=args.traj_interval)

    # Run MD
    dyn.run(args.steps)

    # Final status
    print("-" * 80)
    print_status()

    # Close trajectory file
    traj.close()
    n_frames = len(Trajectory(str(output_path)))
    print(f"\nTrajectory saved: {output_path} ({n_frames} frames)")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("MD Simulation Complete!")
    print(f"{'=' * 60}")
    print(f"  Total steps: {args.steps}")
    print(f"  Total time: {args.steps * args.dt:.1f} fs")
    print(f"  Frames saved: {n_frames}")
    print(f"  Output: {output_path}")

    final_energy = atoms.get_potential_energy()
    final_temp = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
    print(f"  Final energy: {final_energy:.6f} eV")
    print(f"  Final temperature: {final_temp:.1f} K")


if __name__ == "__main__":
    args = get_args()
    run_md(args)
