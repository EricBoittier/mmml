from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import *

print sys.argv[0], sys.argv[1], sys.argv[2]

pdb = PDBFile(sys.argv[1])

f = open(sys.argv[2], "r").read()
system = XmlSerializer.deserialize(f)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
context = Context(system,integrator)
context.setPositions(pdb.positions)
state = context.getState(getEnergy=True)
potential = state.getPotentialEnergy()

print potential / 4.1842, sys.argv[2]

"""
forcefield = ForceField('amber99sb.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, implicitSolvent=OBC2, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.6*nanometer, soluteDielectric=1.0, solventDielectric=80)

systemfile = open("omm_gbsatest-amberobc2.xml","w")
systemstring = XmlSerializer.serialize(system)
systemfile.write(systemstring)
systemfile.close()
"""
