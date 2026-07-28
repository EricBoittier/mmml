import torch
import torch.nn as nn
import numpy as np
import ase
from ase.io import read

# utils imports
from utils.plot_corr import plot_corr
from utils.neuralnets.FFNet import FFNet
from utils.distances import get_bond_length_ABCC
from utils.kernels import get_1D_kernels_k33
'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class KerNNCalculator:

    #most parameters are just passed to the neural network
    def __init__(self,
                 model_path,                     #ckpt file from which to restore the model (can also be a list for ensembles)
                 atoms,                          #ASE atoms object
                 n_input,                        #len of descriptor
                 n_hidden,                       #number of hidden nodes
                 n_out):                      #number of outputs 

        #save checkpoint
        self.model_path = model_path
        

        #create neural network
        self.n_atoms = len(atoms)
        # build/define neural network architecture
        # build/define neural network architecture
        self.model = FFNet(n_input,n_hidden,n_out)
                              
        #load parameters
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.double()

        #calculate properties once to initialize everything
        self._calculate_all_properties(atoms)
        
    def _calculate_all_properties(self, atoms):

        
        #standardize input features
        meanE = -15.92673969
        stdE  =   0.19340856
        
        minr = torch.from_numpy(np.array([1.20929146, 1.10203063, 1.10526860, 2.02403641, 2.01487780, 1.88028228]))
        meanK = torch.from_numpy(np.array([0.01513197, 0.02114981, 0.02089267, 0.00188985, 0.00190670, 0.00253538]))
        stdK  = torch.from_numpy(np.array([4.34973917e-04, 2.66251061e-03, 2.04474782e-03, 9.74953655e-05,
        9.38079320e-05, 1.50988708e-04]))


        pos = torch.tensor(atoms.get_positions(), requires_grad=True)#.double()
        self.nintdist = int(self.n_atoms *(self.n_atoms -1) /2)
        R = get_bond_length_ABCC(pos, self.nintdist)
        
        k = (get_1D_kernels_k33(R, minr) - meanK) / stdK


        #calculate energy and forces
        self._last_energy = self.model(k) * stdE + meanE
        
        self._last_forces = -torch.autograd.grad(torch.sum(self._last_energy), pos, create_graph=True)[0]


        #store copy of atoms
        self._last_atoms = atoms.copy()
        

    
        

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy.detach().numpy()

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces.detach().numpy() #the [0] is to prevent some "problems"
        
        
    def calculation_required(self, atoms, quantities=None):
        return atoms != self.last_atoms


        

    @property
    def sess(self):
        return self._sess

    @property
    def last_atoms(self):
        return self._last_atoms

    @property
    def last_energy(self):
        return self._last_energy[0]

    @property
    def last_forces(self):
        return self._last_forces
 
    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces


