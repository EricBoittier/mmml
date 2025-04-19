#!/bin/bash
module load charmm
module load cudnn
module load openmm 
python ~/mmml/mmml/openmm_interface/openmm-test1.py --temperatures 100 200 300 --pressures 1.0 2.0 3.0 --simulation_schedule minimization equilibration NPT --integrator Langevin