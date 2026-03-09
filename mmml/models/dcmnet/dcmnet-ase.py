"""
DCMNet calculator for ASE.

This calculator is used to calculate the distributed charges (and dipole moments) 
of a molecule using the DCMNet model.

TODO: energy and forces of a molecule using the DCMNet model.

"""

import ase

class DCMNetCalculator(ase.calculators.Calculator):
    """
    DCMNet calculator for ASE.
    """
    def __init__(self, model_params, cutoff=4.0):
        super().__init__()
        self.model_params = model_params
        self.cutoff = cutoff

    def calculate(self, atoms):
        pass