from typing import List, Optional

import e3x
import numpy as np

__all__ = ["PyCharmm_Calculator"]

CHARMM_calculator_units = {
    "positions": "Ang",
    "energy": "kcal/mol",
    "atomic_energies": "kcal/mol",
    "forces": "kcal/mol/Ang",
    "hessian": "kcal/mol/Ang/Ang",
    "charge": "e",
    "atomic_charges": "e",
    "dipole": "e*Ang",
    "atomic_dipoles": "e*Ang",
}


class PyCharmm_Calculator:
    """
    Calculator for the interface between PyCHARMM and the model.

    Parameters
    ----------
    model_calculator: torch.nn.Module
        Asparagus model calculator object with already loaded parameter set
    ml_atom_indices: list(int)
        List of atom indices referring to the ML treated atoms in the total
        system loaded in CHARMM
    ml_atomic_numbers: list(int)
        Respective atomic numbers of the ML atom selection
    ml_charge: float
        Total charge of the partial ML atom selection
    ml_fluctuating_charges: bool
        If True, electrostatic interaction contribution between the MM atom
        charges and the model predicted ML atom charges. Else, the ML atom
        charges are considered fixed as defined by the CHARMM psf file.
    mlmm_atomic_charges: list(float)
        List of all atomic charges of the system loaded to CHARMM.
        If 'ml_fluctuating_charges' is True, the atomic charges of the ML
        atoms are ignored (usually set to zero anyways) and their atomic
        charge prediction is used.
    mlmm_cutoff: float
        Interaction cutoff distance for ML/MM electrostatic interactions
    mlmm_cuton: float
        Lower atom pair distance to start interaction switch-off for ML/MM
        electrostatic interactions
    mlmm_lambda: float, optional, default None
        ML/MM electrostatic interactions scaling factor. If None, no scaling
        is applied.
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        model_calculator,
        ml_atom_indices: Optional[List[int]] = None,
        ml_atomic_numbers: Optional[List[int]] = None,
        ml_charge: Optional[float] = None,
        ml_fluctuating_charges: Optional[bool] = False,
        mlmm_atomic_charges: Optional[List[float]] = None,
        mlmm_cutoff: Optional[float] = 12.0,
        mlmm_cuton: Optional[float] = 10.0,
        mlmm_lambda: Optional[float] = 1.0,
        ml_graph_cutoff: Optional[float] = 6.0,
        use_e3x_pair_list: bool = True,
        **kwargs,
    ):
        self.dtype = np.float64
        self.ml_graph_cutoff = ml_graph_cutoff
        self.use_e3x_pair_list = bool(use_e3x_pair_list)
        self.ml_num_atoms = len(ml_atom_indices) if ml_atom_indices is not None else 0
        self.ml_atom_indices = ml_atom_indices
        self.ml_atomic_numbers = ml_atomic_numbers
        self.ml_charge = ml_charge
        self.ml_fluctuating_charges = ml_fluctuating_charges
        self.mlmm_atomic_charges = mlmm_atomic_charges
        self.mlmm_cutoff = mlmm_cutoff
        self.mlmm_cuton = mlmm_cuton
        self.mlmm_lambda = mlmm_lambda
        self.model_calculator = model_calculator
        self.model_ensemble = False
        self.model_calculator_list = None
        self.model_calculator_num = 1
        # self.model2charmm_unit_conversion = {
        #     "energy": 1.0,
        #     "forces": 1.0,
        #     "charge": 1.0,
        # }
        self.implemented_properties = ["energy", "forces"]
        self.electrostatics_calc = None
        self.results = {}

        self.ml_idxp = None
        self.ml_idxjp = None

        self.ml_idxp = None
        self.ml_idxjp = None
        self.last_full_positions: Optional[np.ndarray] = None

    def _e3x_ml_pairs(self, ml_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same pair list as ``get_ase_calc`` (PhysNet applies cutoff in the model)."""
        n = int(self.ml_num_atoms)
        dst, src = e3x.ops.sparse_pairwise_indices(n)
        return np.asarray(dst, dtype=np.int32), np.asarray(src, dtype=np.int32)

    def calculate_charmm(
        self,
        Natom: int,
        Ntrans: int,
        Natim: int,
        idxp: List[float],
        x: List[float],
        y: List[float],
        z: List[float],
        dx: List[float],
        dy: List[float],
        dz: List[float],
        Nmlp: int,
        Nmlmmp: int,
        idxi: List[int],
        idxj: List[int],
        idxjp: List[int],
        idxu: List[int],
        idxv: List[int],
        idxup: List[int],
        idxvp: List[int],
    ) -> float:
        """
        This function matches the signature of the corresponding MLPot class in
        PyCHARMM.

        Parameters
        ----------
        Natom: int
            Number of atoms in primary cell
        Ntrans: int
            Number of unit cells (primary + images)
        Natim: int
            Number of atoms in primary and image unit cells
        idxp: list(int)
            List of primary and primary to image atom index pointer
        x: list(float)
            List of x coordinates
        y: list(float)
            List of y coordinates
        z: list(float)
            List of z coordinates
        dx: list(float)
            List of x derivatives
        dy: list(float)
            List of y derivatives
        dz: list(float)
            List of z derivatives
        Nmlp: int
            Number of ML atom pairs in the system
        Nmlmmp: int
            Number of ML/MM atom pairs in the system
        idxi: list(int)
            List of ML atom indices for ML potential
        idxj: list(int)
            List of ML atom indices for ML potential
        idxjp: list(int)
            List of image to primary ML atom index pointer
        idxu: list(int)
            List of ML atom indices for ML-MM embedding potential
        idxv: list(int)
            List of MM atom indices for ML-MM embedding potential
        idxup: list(int)
            List of image to primary ML atom index pointer
        idxvp: list(int)
            List of image to primary MM atom index pointer

        Return
        ------
        float
            ML potential
        """

        # Assign all positions
        if Ntrans:
            mlmm_R = np.array([x[:Natim], y[:Natim], z[:Natim]]).T
            idxp[:Natim]
        else:
            mlmm_R = np.array([x[:Natom], y[:Natom], z[:Natom]]).T
            idxp[:Natom]

        self.last_full_positions = np.asarray(mlmm_R, dtype=self.dtype, copy=True)

        ml_idx = np.asarray(self.ml_atom_indices, dtype=int)
        ml_R = mlmm_R[ml_idx]

        if self.use_e3x_pair_list or int(Nmlp) <= 0:
            ml_idxi, ml_idxj = self._e3x_ml_pairs(ml_R)
            ml_idxjp = np.arange(ml_idxi.shape[0], dtype=np.int32)
        else:
            ml_idxi = np.array(idxi[:Nmlp], dtype=np.int32)
            ml_idxj = np.array(idxj[:Nmlp], dtype=np.int32)
            ml_idxjp = np.array(idxjp[:Nmlp], dtype=np.int32)
        ml_sysi = np.zeros(self.ml_num_atoms, dtype=np.int32)
        # ML-MM pair indices and pointer
        np.array(idxu[:Nmlmmp], dtype=np.int32)
        np.array(idxv[:Nmlmmp], dtype=np.int32)
        np.array(idxup[:Nmlmmp], dtype=np.int32)
        np.array(idxvp[:Nmlmmp], dtype=np.int32)
        np.array(idxvp[:Nmlmmp], dtype=np.int64)

        # Create batch for evaluating the model
        atoms_batch = {}
        atoms_batch["N"] = self.ml_num_atoms
        atoms_batch["atomic_numbers"] = self.ml_atomic_numbers
        atoms_batch["positions"] = ml_R
        atoms_batch["charge"] = self.ml_charge
        atoms_batch["dst_idx"] = ml_idxi
        atoms_batch["src_idx"] = ml_idxj
        atoms_batch["sys_i"] = ml_sysi

        # PBC options
        atoms_batch["pbc_offset_ij"] = None
        atoms_batch["pbc_offset_uv"] = None
        atoms_batch["pbc_atoms"] = self.ml_atom_indices
        atoms_batch["pbc_idx"] = self.ml_idxp
        atoms_batch["pbc_idx_j"] = ml_idxjp

        # Compute model properties
        results = {}
        results = self.model_calculator(atoms_batch)

        # Unit conversion
        self.results = {}
        if results["energy"] == np.inf:
            return 0
        for prop in self.implemented_properties:
            self.results[prop] = results[prop]
        # Add forces to CHARMM derivative arrays (one accumulation per ML atom).
        ml_F = np.asarray(self.results["forces"], dtype=self.dtype).reshape(
            self.ml_num_atoms, 3
        )
        for ai, force in zip(self.ml_atom_indices, ml_F):
            dx[ai] -= force[0]
            dy[ai] -= force[1]
            dz[ai] -= force[2]

        return self.results["energy"]
