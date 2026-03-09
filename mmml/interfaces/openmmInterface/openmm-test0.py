from openmm.app import *
from openmm import *
from openmm.unit import *

# Load CHARMM topology (PSF) and structure (PDB)
psf = CharmmPsfFile("/pchem-data/meuwly/boittier/home/project-mmml/proh/proh-262.psf")
pdb = PDBFile("/pchem-data/meuwly/boittier/home/project-mmml/proh/mini.pdb")

# Load CHARMM force field parameters
forcefield = ForceField("charmm36.xml", "charmm36_cgenff.xml")

# Load CHARMM-specific parameter files
params = CharmmParameterSet(
    "/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf",
    "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm",
)

# Create system with long-range PME electrostatics
system = psf.createSystem(
    params, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometer, constraints=HBonds
)

# Define an integrator (Langevin dynamics)
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)

# Set up the simulation
platform = Platform.getPlatformByName("CUDA")  # Change to "CPU" if no GPU
simulation = Simulation(psf.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Minimize energy
print("Minimizing energy...")
simulation.minimizeEnergy()

# Set up output reporters
simulation.reporters.append(
    DCDReporter("trajectory.dcd", 1000)
)  # Save trajectory every 1000 steps
simulation.reporters.append(
    StateDataReporter(
        "output.log", 1000, step=True, potentialEnergy=True, temperature=True
    )
)

# Run simulation (100,000 steps ~200 ps)
print("Running simulation...")
simulation.step(1000)
print("Simulation complete.")
