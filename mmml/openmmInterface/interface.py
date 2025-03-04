from openmm.app import *
from openmm.app import internal as internal
from openmm.app.internal.unitcell import computePeriodicBoxVectors
from openmm import *
from openmm.unit import *
import numpy as np
import os

#import sys
#sys.path.append("/pchem-data/meuwly/boittier/home/openmm-torch")

from openmmml import MLPotential


# Input files
pdbid = "proh"
psf_file = f"/pchem-data/meuwly/boittier/home/project-mmml/proh/proh-262.psf"
pdb_file = f"/pchem-data/meuwly/boittier/home/project-mmml/proh/mini.pdb"
rtf_file = "/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf"
prm_file = "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm"


def setup_simulation(psf_file, pdb_file, rtf_file, prm_file, working_dir, temperature, simulation_type):
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
    system = psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer)
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # Add a Langevin thermostat
    integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.5*femtoseconds)

    # Choose the simulation platform
    platform = Platform.getPlatformByName("CUDA")

    # Create the simulation
    simulation = Simulation(psf.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    # Run the specified simulation type
    if "minimization" in simulation_type:
        minimize_energy(simulation, working_dir)

    if "equilibration" in simulation_type:
        equilibrate(simulation, integrator, temperature, working_dir)

    if "NPT" in simulation_type:
        run_npt(simulation, integrator, working_dir)

    if "NVE" in simulation_type:
        run_nve(simulation, integrator, working_dir)

def minimize_energy(simulation, working_dir):
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    save_state(simulation, os.path.join(working_dir, "res", "minimized.res"))

def equilibrate(simulation, integrator, temperature, working_dir):
    print("Equilibrating...")
    nsteps_equil = 10000
    temp_start, temp_final = 100, temperature
    for temp in np.linspace(temp_start, temp_final, num=10):
        integrator.setTemperature(temp * kelvin)
        simulation.step(nsteps_equil // 10)
    print("Equilibration complete.")
    save_state(simulation, os.path.join(working_dir, "res", "equilibrated.res"))

def run_npt(simulation, integrator, working_dir):
    print("Running NPT simulation...")
    system = simulation.system
    system.addForce(MonteCarloBarostat(1 * atmosphere, 298 * kelvin, 25))
    integrator.setTemperature(298*kelvin)
    nsteps_prod = 100000
    setup_reporters(simulation, working_dir, "npt")
    simulation.step(nsteps_prod)
    print("NPT simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", "npt_final.res"))

def run_nve(simulation, integrator, working_dir):
    print("Running NVE simulation...")
    nsteps_prod = 100000
    setup_reporters(simulation, working_dir, "nve")
    simulation.step(nsteps_prod)
    print("NVE simulation complete.")
    save_state(simulation, os.path.join(working_dir, "res", "nve_final.res"))

def setup_reporters(simulation, working_dir, prefix):
    simulation.reporters.append(DCDReporter(os.path.join(working_dir, "dcd", f"{prefix}.dcd"), 1000))
    simulation.reporters.append(StateDataReporter(os.path.join(working_dir, "res", f"{prefix}.log"), 1000, step=True, potentialEnergy=True, temperature=True))

def save_state(simulation, filename):
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(filename, "w") as f:
        f.write(state.getPositions(asNumpy=True).__str__())

# Example usage
setup_simulation(
    psf_file="/path/to/proh-262.psf",
    pdb_file="/path/to/mini.pdb",
    rtf_file="/path/to/top_all36_cgenff.rtf",
    prm_file="/path/to/par_all36_cgenff.prm",
    working_dir="/path/to/working/directory",
    temperature=298,
    simulation_type=["minimization", "equilibration", "NPT"]
)
