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
        # Keep PSF bonds/angles; use CHARMM BLOCK to disable internal MM on ML atoms.
        preserve_psf_internals=True,
        # ML exclusions + ``upinb`` were already installed (PBC registration path).
        skip_iblo_inb_update=False,
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

        # Legacy path: strip ML-region connectivity from the PSF. Prefer BLOCK +
        # preserve_psf_internals=True (mmml default) so VMD/topology stay intact.
        if not preserve_psf_internals:
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

        if not skip_iblo_inb_update:
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

        mlidx = (ctypes.c_int * self.ml_Natoms)()
        mlidx[:] = self.ml_indices + 1
        mlidz = (ctypes.c_int * self.ml_Natoms)()
        mlidz[:] = ml_Z
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

    def reattach_mlpot(self, *, force: bool = False):
        """Re-enable MLpot after :meth:`unset_mlpot` without rebuilding exclusion lists.

        Re-running :func:`pycharmm.psf.set_iblo_inb` / :func:`pycharmm.nbonds.update_bnbnd`
        after long MD can segfault in CHARMM ``upinb``; reuse the existing lists instead.

        When ``force=True``, always re-register the callback even if Python ``is_set``
        is still True (Fortran ``mlpot_is_set`` may have been cleared by ``mlpot_unset``).
        """
        if self.is_set and not force:
            return
        pycharmm.lib.charmm.mlpot_set_func(self.energy_func)
        mlidx = (ctypes.c_int * self.ml_Natoms)()
        mlidx[:] = self.ml_indices + 1
        mlidz = (ctypes.c_int * self.ml_Natoms)()
        mlidz[:] = self.ml_Z
        nml = (ctypes.c_int * 1)(self.ml_Natoms)
        pycharmm.lib.charmm.mlpot_set_properties(nml, mlidx, mlidz)
        self.is_set = True


def get_mlpot_pair_counts():
    """Return ``(n_mlml, n_mlmm)`` from the last ``mlpot_update`` call."""
    try:
        getter = pycharmm.lib.charmm.mlpot_get_pair_counts
    except AttributeError:
        return None
    out_nmlp = (ctypes.c_int * 1)()
    out_nmlmmp = (ctypes.c_int * 1)()
    status = getter(out_nmlp, out_nmlmmp)
    if not bool(status):
        raise RuntimeError("mlpot_get_pair_counts failed")
    return int(out_nmlp[0]), int(out_nmlmmp[0])


def export_mlpot_mlmm_pairs(*, max_pairs: int | None = None):
    """Export Fortran ``idxu/idxv`` (0-based) after ``mlpot_update``."""
    try:
        exporter = pycharmm.lib.charmm.mlpot_export_mlmm_pairs
    except AttributeError:
        return None
    _nmlp, nmlmmp = get_mlpot_pair_counts() or (0, 0)
    cap = int(max_pairs) if max_pairs is not None else int(nmlmmp)
    if cap <= 0:
        return [], []
    out_u = (ctypes.c_int * cap)()
    out_v = (ctypes.c_int * cap)()
    out_count = (ctypes.c_int * 1)()
    status = exporter(out_u, out_v, ctypes.c_int(cap), out_count)
    if not bool(status):
        raise RuntimeError("mlpot_export_mlmm_pairs failed")
    n = int(out_count[0])
    return [int(out_u[k]) for k in range(n)], [int(out_v[k]) for k in range(n)]


def export_mlpot_mlml_pairs(*, max_pairs: int | None = None):
    """Export Fortran ``idxi/idxj`` (0-based) after ``mlpot_update``."""
    try:
        exporter = pycharmm.lib.charmm.mlpot_export_mlml_pairs
    except AttributeError:
        return None
    nmlp, _nmlmmp = get_mlpot_pair_counts() or (0, 0)
    cap = int(max_pairs) if max_pairs is not None else int(nmlp)
    if cap <= 0:
        return [], []
    out_i = (ctypes.c_int * cap)()
    out_j = (ctypes.c_int * cap)()
    out_count = (ctypes.c_int * 1)()
    status = exporter(out_i, out_j, ctypes.c_int(cap), out_count)
    if not bool(status):
        raise RuntimeError("mlpot_export_mlml_pairs failed")
    n = int(out_count[0])
    return [int(out_i[k]) for k in range(n)], [int(out_j[k]) for k in range(n)]

