import os
import json
from openmm.app.internal.unitcell import computePeriodicBoxVectors
from pathlib import Path
import argparse
import numpy as np
from datetime import datetime
from openmm.app import *
from openmm import *
from openmm.unit import *


global dcd_files 
dcd_files = []
global report_files
report_files = []

def setup_simulation(
    psf_file,
    pdb_file,
    rtf_file,
    prm_file,
    working_dir,
    temperatures,
    pressures,
    simulation_schedule,
    integrator_type,
    steps,
):
    # Create necessary directories
    os.makedirs(os.path.join(working_dir, "pdb"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "dcd"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "res"), exist_ok=True)

    # Define box size
    box_length = 3.5 * nanometer
    alpha, beta, gamma = 90.0 * degree, 90.0 * degree, 90.0 * degree

    # Compute periodic box vectors
    a, b, c = box_length, box_length, box_length
    box_vectors = computePeriodicBoxVectors(a, b, c, alpha, beta, gamma)

    # Load CHARMM files
    psf = CharmmPsfFile(psf_file)
    psf.setBox(a, b, c, alpha, beta, gamma)
    pdb = PDBFile(pdb_file)
    pdb.topology.setPeriodicBoxVectors(box_vectors)
    params = CharmmParameterSet(rtf_file, prm_file)

    # Create the system
    system = psf.createSystem(
        params, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometer
    )
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # Choose the integrator
    if integrator_type == "Langevin":
        integrator = LangevinIntegrator(
            temperatures[0] * kelvin, 1 / picosecond, 0.5 * femtoseconds
        )
    elif integrator_type == "Verlet":
        integrator = VerletIntegrator(0.5 * femtoseconds)
    elif integrator_type == "Nose-Hoover":
        integrator = NoseHooverIntegrator(
            temperatures[0] * kelvin, 1 / picosecond, 0.5 * femtoseconds
        )
    else:
        raise ValueError(f"Unsupported integrator type: {integrator_type}")

    # Choose the simulation platform
    platform = Platform.getPlatformByName("CUDA")

    # Create the simulation
    simulation = Simulation(psf.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    # Run the specified simulation schedule
    for i, task in enumerate(simulation_schedule):
        sim_type = task.get("type")
        temperature = temperatures[i]
        pressure = (
            pressures[i] if i < len(pressures) else 1.0
        )  # Default pressure if not enough values

        print(f"Running {sim_type} simulation...")
        print(f"Temperature: {temperature} K")
        print(f"Pressure: {pressure} atm")
        print(f"Integrator: {integrator_type}")

        # setup_reporters(simulation, working_dir, f"{sim_type}_{i}")

        if sim_type == "minimization":
            minimize_energy(simulation, working_dir)
        elif sim_type == "equilibration":
            equilibrate(
                simulation, integrator, temperature, 
                working_dir, integrator_type, steps
            )
        elif sim_type == "NPT":
            run_npt(simulation, integrator, pressure,
             working_dir, integrator_type, steps)
        elif sim_type == "NVE":
            run_nve(simulation, integrator,
             working_dir, steps)


def minimize_energy(simulation, working_dir):
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    save_state(simulation, os.path.join(working_dir, "res", "minimized.res"))


def equilibrate(
    simulation, integrator, temperature, working_dir, integrator_type, steps=10**6
):
    print("Equilibrating...")
    nsteps_equil = steps
    temp_start, temp_final = 150, temperature
    for temp in np.linspace(temp_start, temp_final, num=steps // 100):
        simulation.step(steps // 100)
    print("Equilibration complete.")
    save_state(simulation, os.path.join(working_dir, "res", "equilibrated.res"))


def run_npt(
    simulation, integrator, pressure, working_dir, integrator_type, steps=10**6
):
    print("Running NPT simulation...")
    system = simulation.system
    system.addForce(MonteCarloBarostat(pressure * atmosphere, 298 * kelvin, 25))
    if integrator_type == "Langevin":
        integrator.setTemperature(298 * kelvin)
    nsteps_prod = steps
    setup_reporters(simulation, working_dir, "npt")
    simulation.step(nsteps_prod)
    print("NPT simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", "npt_final.res"))


def run_nve(simulation, integrator, working_dir, steps=10**6):
    print("Running NVE simulation...")
    nsteps_prod = steps
    setup_reporters(simulation, working_dir, "nve")
    simulation.step(nsteps_prod)
    print("NVE simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", "nve_final.res"))


def setup_reporters(simulation, working_dir, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dcd_path = os.path.join(working_dir, "dcd", f"{prefix}_{timestamp}.dcd")
    report_path = os.path.join(working_dir, "res", f"{prefix}_{timestamp}.log")
    
    dcd_files.append(dcd_path)
    report_files.append(report_path)
    
    simulation.reporters.append(DCDReporter(dcd_path, 1000))
    simulation.reporters.append(
        StateDataReporter(
            report_path,
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
            speed=True,
            totalEnergy=True,
        )
    )


def save_state(simulation, filename):
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(filename, "w") as f:
        f.write(state.getPositions(asNumpy=True).__str__())


def parse_args():
    # current directory of current file
    
    from pathlib import Path
    cwd = Path(__file__).parents[1]
    CGENFF_RTF = cwd / "pycharmm_interface" / "top_all36_cgenff.rtf"
    CGENFF_PRM = cwd / "pycharmm_interface" / "par_all36_cgenff.prm"
    CGENFF_RTF = str(CGENFF_RTF)
    CGENFF_PRM = str(CGENFF_PRM)

    parser = argparse.ArgumentParser(
        description="Run OpenMM simulations with specified parameters."
    )
    parser.add_argument("--psf_file", required=True, help="Path to the PSF file.")
    parser.add_argument("--pdb_file", required=True, help="Path to the PDB file.")
    parser.add_argument("--rtf_file", required=False, default=CGENFF_RTF, help="Path to the RTF file.")
    parser.add_argument("--prm_file", required=False, default=CGENFF_PRM, help="Path to the PRM file.")
    parser.add_argument(
        "--working_dir", required=True, help="Working directory for output files."
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        required=True,
        help="List of temperatures in Kelvin for each simulation step.",
    )
    parser.add_argument(
        "--pressures",
        type=float,
        nargs="*",
        default=[1.0],
        help="List of pressures in atmospheres for each simulation step (default: 1.0).",
    )
    parser.add_argument(
        "--simulation_schedule",
        nargs="+",
        required=True,
        help="List of simulation types to run (e.g., minimization equilibration NPT NVE).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=10**6,
        help="Number of steps to run for each simulation.",
    )
    parser.add_argument(
        "--integrator",
        choices=["Langevin", "Verlet", "Nose-Hoover"],
        default="Langevin",
        help="Integrator type to use (default: Langevin).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = setup_simulation(
        psf_file=args.psf_file,
        pdb_file=args.pdb_file,
        rtf_file=args.rtf_file,
        prm_file=args.prm_file,
        working_dir=args.working_dir,
        temperatures=args.temperatures,
        pressures=args.pressures,
        simulation_schedule=[
            {"type": sim_type} for sim_type in args.simulation_schedule
        ],
        integrator_type=args.integrator,
        steps=args.steps,
    )

    # write variables to manifest file...
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    os.makedirs(Path(args.working_dir / "omm"), exist_ok=True)
    with open(Path(args.working_dir / "omm" / f"openmm-{current_time}.json"), "w") as f:
        json.dump({
            "dcd_files": dcd_files,
            "report_files": report_files,
            "args": args,
        }, f)


# example command:
# python openmm-test1.py --psf_file /pchem-data/meuwly/boittier/home/project-mmml/proh/proh-262.psf --pdb_file /pchem-data/meuwly/boittier/home/project-mmml/proh/mini.pdb --rtf_file /pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf --prm_file /pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm --working_dir /pchem-data/meuwly/boittier/home/project-mmml/proh/openmm-test1 --temperatures 100 200 300 --pressures 1.0 2.0 3.0 --simulation_schedule minimization equilibration NPT NVE --integrator Langevin
