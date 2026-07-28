#!/usr/bin/env python3

#imports
import argparse
from ase import Atoms
import numpy as np
from ase.visualize import view
from ase.io import read, write
from ase.optimize import *
from ase import units
from KerNNCalculator.KerNNCalculator import *
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

'''
script to perform a geometry optimization using an NN model.
Usage python optimize_coarse.py -i structure.xyz --fmax 0.0001
where the fmax value can be chosen. The optimized structure is
saved in the xyz file as "opt_structure.xyz".
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
optional.add_argument("--fmax",  type=float, help="maximal force", default=0.000002)
from os.path import splitext

args = parser.parse_args()
filename, extension = splitext(args.input)
print("input ", args.input)

#read input file (molecular structure to predict) and create
#an atoms object
atoms = read(args.input)

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = KerNNCalculator(
    model_path="../training/models_ntrain3200_seed42/model_20240517_152140_999_seed42_ema", #load the model you want to used
    atoms=atoms,
    n_input=6,
    n_hidden=20,
    n_out=1)
#attach the calculator object (used to describe the atomic interaction) to the atoms object
atoms.set_calculator(calc)

# chose optimization algorithm (MDMin, BFGS, FIRE)
algorithm = BFGS
dyn = algorithm(atoms)

#dyn = algorithm(atoms, trajectory = 'opt_'+ filename +'.traj') #if you want to save optimization process
#in trajectory file

#run optimization until fmax is reached
dyn.run(args.fmax)

# save final structure in xyz format
write('opt_'+ args.input, atoms)

#print forces (which should be small now) to double check
print(atoms.get_forces(atoms))
