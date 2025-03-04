from openmm.app import *
from openmm.app import internal as internal
from openmm.app.internal.unitcell import computePeriodicBoxVectors
from openmm import *
from openmm.unit import *
import numpy as np

#import sys
#sys.path.append("/pchem-data/meuwly/boittier/home/openmm-torch")

from openmmml import MLPotential


# Input files
pdbid = "proh"
psf_file = f"/pchem-data/meuwly/boittier/home/project-mmml/proh/proh-262.psf"
pdb_file = f"/pchem-data/meuwly/boittier/home/project-mmml/proh/mini.pdb"
rtf_file = "/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf"
prm_file = "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm"


# Define box size (adjust these values based on your system)
box_length = 3.5 * nanometer  # Same in all directions (cubic box)
alpha, beta, gamma = 90.0 * degree, 90.0 * degree, 90.0 * degree  # Cubic angles

# Compute periodic box vectors
a, b, c = box_length, box_length, box_length
box_vectors = computePeriodicBoxVectors(a, b, c, alpha, beta, gamma)

# Load CHARMM files
psf = CharmmPsfFile(psf_file)
psf.setBox(a,b,c, alpha, beta, gamma)

pdb = PDBFile(pdb_file)
# Apply periodic box vectors to the PDB structure (if needed)
pdb.topology.setPeriodicBoxVectors(box_vectors)

params = CharmmParameterSet(rtf_file, prm_file)

# Create the system with PME
#system = psf.createSystem(params, nonbondedMethod=PME,
#                          nonbondedCutoff=1.0*nanometer, constraints=HBonds)
#system.setDefaultPeriodicBoxVectors(*box_vectors)

system = psf.createSystem(params, nonbondedMethod=PME,
                           nonbondedCutoff=1.0*nanometer) #, constraints=HBonds)
system.setDefaultPeriodicBoxVectors(*box_vectors)


# Add a Langevin thermostat (needed for temperature control)
integrator = LangevinIntegrator(100*kelvin, 1/picosecond, 0.5*femtoseconds)  # Start at 100K

# Choose the simulation platform
platform = Platform.getPlatformByName("CUDA")  # Change to "CPU" if no GPU

# Create the simulation
simulation = Simulation(psf.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)


# Openmm ML
#chains = list(psf.topology.chains())
#ml_atoms = [atom.index for i, atom in enumerate(chains[0].atoms()) if i < 5]
#potential = MLPotential('ani2x')
#ml_system = potential.createMixedSystem(topology, system, ml_atoms)


# Minimization
print("Minimizing energy...")
simulation.minimizeEnergy()

# Equilibration: Ramp temperature from 100K to 298K
nsteps_equil = 10000  # Adjust as needed
temp_start, temp_final = 100, 298
for temp in np.linspace(temp_start, temp_final, num=10):
    integrator.setTemperature(temp * kelvin)
    simulation.step(nsteps_equil // 10)
print("Equilibration complete.")

# Save restart files
state = simulation.context.getState(getPositions=True, getVelocities=True)
with open(f"res/{pdbid}.res", "w") as f:
    f.write(state.getPositions(asNumpy=True).__str__())



# Add barostat for pressure control (NPT ensemble)
system.addForce(MonteCarloBarostat(1 * atmosphere, 298 * kelvin, 25))  # Adjust frequency as needed
# Production Run
integrator.setTemperature(298*kelvin)
nsteps_prod = 100000  # Adjust as needed

# Set up output files
simulation.reporters.append(DCDReporter(f"dcd/{pdbid}-openmm.dcd", 1000))  # Save every 1000 steps
simulation.reporters.append(StateDataReporter(f"res/{pdbid}-openmm.log", 1000, step=True, 
                                              potentialEnergy=True, temperature=True))

print("Starting production run...")
simulation.step(nsteps_prod)
print("Production run complete.")

# Save final state
state = simulation.context.getState(getPositions=True, getVelocities=True)
with open(f"res/{pdbid}_final.res", "w") as f:
    f.write(state.getPositions(asNumpy=True).__str__())
