from enum import Enum as ENUM

class THEORY(ENUM):
    DFT = 0
    HF = 1
    MP2 = 2
    CCSD = 3
    MP3 = 4

class BASIS(ENUM):
    def2_tzvpp = 0
    def2_tzvp = 1


class CALCS(ENUM):
    ENERGY = 0
    GRADIENT = 1
    HESSIAN = 2
    HARMONIC = 3
    THERMO = 4
    INTERACTION = 5
    OPTIMIZE = 6
    DENS_ESP = 7
    IR = 8
    SHIELDING = 9
    POLARIZABILITY = 10
    IR_EFIELD = 11  # IR under external E-field + optional E-field scan (responsive)
    EFIELD_SCF = 12  # SCF in uniform E-field: energy, forces, dipole only (no Hessian/IR)

all_theory = [THEORY.DFT, THEORY.HF, THEORY.MP2, THEORY.CCSD, THEORY.MP3]
all_basis = [BASIS.def2_tzvpp, BASIS.def2_tzvp]
all_calcs = [
    CALCS.ENERGY, 
    CALCS.GRADIENT,
    CALCS.HESSIAN, 
    CALCS.HARMONIC, 
    CALCS.THERMO, 
    CALCS.INTERACTION, 
    CALCS.OPTIMIZE, 
    CALCS.DENS_ESP,
    CALCS.IR,
    CALCS.SHIELDING, 
    CALCS.POLARIZABILITY,
    CALCS.IR_EFIELD,
    CALCS.EFIELD_SCF,
]