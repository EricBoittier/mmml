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
    tag,
    timestep=0.5
):
    """runs the simulations with the given parameters
    
    Args:
        psf_file, str: path to the psf file
        pdb_file, str: path to the pdb file
        rtf_file, str: path to the rtf file
        prm_file, str: path to the prm file
        working_dir, str: path to the working directory
        temperatures, list of floats: list of temperatures
        pressures, list of floats: list of pressures
        simulation_schedule, list of dicts: list of simulation types
        integrator_type, str: type of integrator
        steps, int: number of steps
        tag, str: tag for output files
        timestep, float: timestep for the simulation (in femtoseconds)
    """
    # Create necessary directories
    os.makedirs(os.path.join(working_dir, "pdb"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "dcd"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "res"), exist_ok=True)

    # set timestep
    timestep = timestep * femtoseconds

    # Define box size
    box_length = 3.5 * nanometer
    alpha, beta, gamma = 90.0 * degree, 90.0 * degree, 90.0 * degree

    # Compute periodic box vectors
    a, b, c = box_length, box_length, box_length
    box_vectors = computePeriodicBoxVectors(a, b, c, alpha, beta, gamma)
    print(psf_file)
    # Load CHARMM files
    psf = CharmmPsfFile(psf_file)
    print(psf)
    psf.setBox(a, b, c, alpha, beta, gamma)
    pdb = PDBFile(pdb_file)
    pdb.topology.setPeriodicBoxVectors(box_vectors)
    params = CharmmParameterSet(rtf_file, prm_file)
    params2 = ForceField("charmm36/tip3p-pme-f.xml")
    
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
            minimize_energy(simulation, working_dir, tag)
        elif sim_type == "equilibration":
            equilibrate(
                simulation, integrator, temperature, 
                working_dir, integrator_type, steps, tag
            )
        elif sim_type == "NPT":
            run_npt(simulation, integrator, pressure, temperature,
             working_dir, integrator_type, steps, tag)
        elif sim_type == "NVE":
            run_nve(simulation, integrator,
             working_dir, temperature, steps, tag)


def minimize_energy(simulation, working_dir, tag=""):
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    save_state(simulation, os.path.join(working_dir, "res", f"minimized_{tag}.res"))


def equilibrate(
    simulation, integrator, temperature, working_dir, integrator_type, steps=10**6, tag=""
):
    print("Equilibrating...")
    temp_start, temp_final = 100, temperature
    for temp in np.linspace(temp_start, temp_final, num=steps // 100):
        simulation.step(steps // 100)
    print(f"Equilibration complete. ({steps} steps)")
    save_state(simulation, os.path.join(working_dir, "res", f"equilibrated_{tag}.res"))


def run_npt(
    simulation, integrator, pressure, temperature, working_dir, integrator_type, steps=10**6,
    tag=""
):
    print("Running NPT simulation...")
    system = simulation.system
    system.addForce(MonteCarloBarostat(pressure * atmosphere, temperature * kelvin, 25))
    if integrator_type == "Langevin":
        integrator.setTemperature(temperature * kelvin)
    nsteps_prod = steps
    setup_reporters(simulation, working_dir, "npt", tag)
    simulation.step(nsteps_prod)
    print("NPT simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", f"npt_final_{tag}.res"))


def run_nve(simulation, integrator, working_dir, temperature, steps=10**6, tag=""):
    print("Running NVE simulation...")
    nsteps_prod = steps
    setup_reporters(simulation, working_dir, "nve", tag)
    simulation.step(nsteps_prod)
    print("NVE simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", f"nve_final_{tag}.res"))


def setup_reporters(simulation, working_dir, prefix, tag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dcd_path = os.path.join(working_dir, "dcd", f"{prefix}_{timestamp}{tag}.dcd")
    report_path = os.path.join(working_dir, "res", f"{prefix}_{timestamp}{tag}.log")
    
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

    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep for the simulation (in femtoseconds) (default: 0.5).",
    )

    parser.add_argument(
        "--tag",
        default="",
        help="tag for output files (default: empty).",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    """
    example command:
    python openmm-sampling.py --psf_file /pchem-data/meuwly/boittier/home/project-mmml/proh/proh-262.psf --pdb_file /pchem-data/meuwly/boittier/home/project-mmml/proh/mini.pdb --rtf_file /pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf --prm_file /pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm --working_dir /pchem-data/meuwly/boittier/home/project-mmml/proh/openmm-test1 --temperatures 100 200 300 --pressures 1.0 2.0 3.0 --simulation_schedule minimization equilibration NPT NVE --integrator Langevin
    """
    
    # parse command line arguments
    args = parse_args()

    # setup simulation
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
        tag=args.tag,
    )

    # write variables to manifest file...
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    os.makedirs(Path(args.working_dir) / "omm", exist_ok=True)
    jsonout = Path(args.working_dir) / "omm" / f"openmm.json"
    args_dict = vars(args)
    args_dict["dcd_files"] = dcd_files
    args_dict["report_files"] = report_files
    # copy the last dcd file to the working directory as the tag
    if args.tag:
        import shutil
        shutil.copy(dcd_files[-1], Path(args.working_dir) / "dcd" / f"{args.tag}.dcd")
        args_dict["dcd_files"].append(str(Path(args.working_dir) / "dcd" / f"{args.tag}.dcd"))
        shutil.copy(report_files[-1], Path(args.working_dir) / "res" / f"{args.tag}.log")
        args_dict["report_files"].append(str(Path(args.working_dir) / "res" / f"{args.tag}.log"))

    for k, v in args_dict.items():
        print(k, v)
    
    with open(jsonout, "w") as f:
        json.dump(args_dict, f)

