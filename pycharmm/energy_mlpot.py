# MLpot: Custom 
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" A class to set PhysNet model to run during an energy calculation
"""

import ctypes

import numpy as np

import pycharmm


class MLpot():
    """
    Custom Machine Learning potential
    """

    def __init__(
        self,
        # Potential energy model
        ml_model,
        # ML atomic numbers as trained
        ml_Z,
        # ML atom PyCHARMM selection object
        ml_selection,
        # ML atoms total charge
        ml_charge=None,
        # Fluctuating ML atom charges for ML/MM electrostatic interaction
        ml_fq=True,
        # ML-MM cutoff radii ctonnb and ctofnb. If None, CHARMM parameter taken
        mlmm_ctonnb=None,
        mlmm_ctofnb=None,
        # Additional model keyword arguments
        **kwargs
    ):

        # ML&MM - atom number
        self.Natoms = pycharmm.coor.get_natom()

        # ML atoms - atom indices
        ml_indices = ml_selection.get_atom_indexes()
        ml_argsort = np.argsort(np.array(ml_indices, dtype=int))
        self.ml_indices = np.array(ml_indices, dtype=int)[ml_argsort]

        # ML atoms - atom number
        self.ml_Natoms = len(ml_indices)

        # ML - Delete bonds, angles, dihedrals, improper - if active
        pycharmm.psf.delete_bonds(ml_selection, ml_selection, psort=True)
        pycharmm.psf.delete_angles(ml_selection, ml_selection, psort=True)
        pycharmm.psf.delete_dihedrals(ml_selection, ml_selection, psort=True)
        pycharmm.psf.delete_impropers(ml_selection, ml_selection, psort=True)
        pycharmm.psf.delete_cmaps(ml_selection, ml_selection, psort=True)

        # ML&MM - psf charges - set ML charges zero when charges and ML-MM
        # interaction are handled by the ML potential
        self.ml_fq = ml_fq
        self.mlmm_charges = np.array(pycharmm.param.get_charge())
        if ml_fq:
            self.mlmm_charges[self.ml_indices] = 0.0
            _ = pycharmm.psf.set_charge(self.mlmm_charges)

        # ML - set non-bond exclusion list for ML atom pairs
        self.ml_iblo = np.zeros(self.Natoms, dtype=int)
        self.ml_inb = []
        for ii, idx in enumerate(ml_indices):
            self.ml_iblo[idx:] += self.ml_Natoms - ii - 1
            for jdx in self.ml_indices[(ii + 1):]:
                self.ml_inb.append(jdx + 1)  # + 1 as CHARMM start at index 1
        self.ml_nnb = len(self.ml_inb)

        pycharmm.psf.set_iblo_inb(self.ml_iblo, self.ml_inb)

        pycharmm.nbonds.update_bnbnd()  # Already executed in set_iblo_inb()
        pycharmm.image.update_bimag()

        ###################################################
        # START - Potential model dependent part
        ###################################################

        # Assign Potential model
        if ml_model is None:
            raise SyntaxError("Potential model is not defined (None)!")
        elif not getattr(ml_model, "get_pycharmm_calculator"):
            raise SyntaxError(
                "Potential model does not has callable function "
                + "'get_pycharmm_calculator'!")
        else:
            self.ml_model = ml_model

        # ML atoms - atomic numbers
        if ml_Z is None:
            raise SyntaxError("ML atom number are not defined (None)!")
        else:
            self.ml_Z = np.array(ml_Z, dtype=int)[ml_argsort]

        # System charge
        if ml_charge is None:
            self.ml_charge = 0
        else:
            self.ml_charge = ml_charge

        # ML/MM electrostatic interaction cutoffs
        if mlmm_ctonnb is None:
            self.mlmm_ctonnb = pycharmm.nbonds.get_ctonnb()
        else:
            self.mlmm_ctonnb = mlmm_ctonnb
        if mlmm_ctofnb is None:
            self.mlmm_ctofnb = pycharmm.nbonds.get_ctofnb()
        else:
            self.mlmm_ctofnb = mlmm_ctofnb

        # Assign model potential calculator
        self.calculator = self.ml_model.get_pycharmm_calculator(
            ml_atom_indices=self.ml_indices,
            ml_atomic_numbers=self.ml_Z,
            ml_charge=self.ml_charge,
            ml_fluctuating_charges=self.ml_fq,
            mlmm_atomic_charges=self.mlmm_charges,
            mlmm_cutoff=self.mlmm_ctofnb,
            mlmm_cuton=self.mlmm_ctonnb,
            **kwargs,
        )

        # Initialize custom energy function
        self.func_type = ctypes.CFUNCTYPE(
            ctypes.c_double,                    # User energy - E(user)
            ctypes.c_int,                       # Atom number central cell
                                                # (Natom)
            ctypes.c_int,                       # Number of central and image
                                                # cells (Ntrans)
            ctypes.c_int,                       # Atom number central + image
                                                # cells (Natim)
            ctypes.POINTER(ctypes.c_int),       # Central and image to central
                                                # atom index list
                                                # (range(Natom) + IMATTR)
            ctypes.POINTER(ctypes.c_double),    # Atom position x
            ctypes.POINTER(ctypes.c_double),    # Atom position y
            ctypes.POINTER(ctypes.c_double),    # Atom position y
            ctypes.POINTER(ctypes.c_double),    # Atom potential der. (dE/dx)
            ctypes.POINTER(ctypes.c_double),    # Atom potential der. (dE/dy)
            ctypes.POINTER(ctypes.c_double),    # Atom potential der. (dE/dz)
            ctypes.c_int,                       # Number of ML-ML atom pairs
                                                # (Nmlp)
            ctypes.c_int,                       # Number of ML-MM atom pairs
                                                # (Nmlmmp)
            ctypes.POINTER(ctypes.c_int),       # ML-ML pair atom i (idxi)
            ctypes.POINTER(ctypes.c_int),       # ML-ML pair atom j (idxj)
            ctypes.POINTER(ctypes.c_int),       # Image to central ML atom
                                                # pointer (idxjp)
            ctypes.POINTER(ctypes.c_int),       # ML-MM pair ML atom u (idxu)
            ctypes.POINTER(ctypes.c_int),       # ML-MM pair MM atom v (idxv)
            ctypes.POINTER(ctypes.c_int),       # Image to central MM atom
                                                # pointer (idxup)
            ctypes.POINTER(ctypes.c_int),       # Image to central MM atom
                                                # pointer (idxvp)
            )

        self.energy_func = self.func_type(self.calculator.calculate_charmm)

        ###################################################
        # END - Potential model dependent part
        ###################################################

        pycharmm.lib.charmm.mlpot_set_func(self.energy_func)

        mlidx = (ctypes.c_int * self.ml_Natoms)(*(self.ml_indices + 1))
        mlidz = (ctypes.c_int * self.ml_Natoms)(*ml_Z)
        Nml = (ctypes.c_int * 1)(self.ml_Natoms)
        pycharmm.lib.charmm.mlpot_set_properties(
            Nml, mlidx, mlidz)

        self.is_set = True

        return

    def __del__(self):
        """
        Class destructor
        """
        self.unset_mlpot()

    def unset_mlpot(self):
        """
        Just store the function and do not run it during energy calculations
        """
        pycharmm.lib.charmm.mlpot_unset()
        self.is_set = False
