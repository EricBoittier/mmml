#!/usr/bin/env python3


import argparse
import numpy as np
from ase.io import read, write
from KerNNCalculator.KerNNCalculator import *
import argparse
from os.path import splitext
from ase.vibrations import Vibrations

'''
Usage: python ase_vibrations.py -i structure.xyz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input traj",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)


args = parser.parse_args()
filename, extension = splitext(args.input)

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
#setup calculator (which will be used to describe the atomic interactions)
atoms.set_calculator(calc)

vib = Vibrations(atoms, delta=0.01, nfree=2)
vib.run()

#print frequencies
vib.summary()

#delete written files
vib.clean()









