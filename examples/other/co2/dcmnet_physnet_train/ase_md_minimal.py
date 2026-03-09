# Minimal ASE MD example - copy these lines into your notebook

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.io import Trajectory

# Setup
atoms.calc = calculator  # Your calculator here
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

# Choose integrator (NVE or NVT)
dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)  # NVE
# OR for NVT:
# dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01)

# Optional: Save trajectory
# traj = Trajectory('output.traj', 'w', atoms)
# dyn.attach(traj.write, interval=10)

# Run MD
dyn.run(10000)

# Optional: Close trajectory
# traj.close()

