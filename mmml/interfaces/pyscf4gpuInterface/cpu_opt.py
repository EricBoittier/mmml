from pyscf.geomopt.geometric_solver import optimize
import ase

def rotate_dimers(atoms, n_atoms_a, n_atoms_b):
    """
    Rotate a dimer.
    atoms is the atoms object.
    N_atoms_a is the number of atoms in the first monomer.
    N_atoms_b is the number of atoms in the second monomer.
    """
    s = """$freeze
rotation 1-{}
$scan
rotation {}->{} 0.0 1.0
""".format(n_atoms_a, n_atoms_a, n_atoms_a+n_atoms_b)
    return s

def write_constraints(atoms, s, filename):
    """
    Write the constraints to a file.
    atoms is the atoms object.
    s is the string of constraints.
    filename is the path to the constraints file.
    """
    with open(filename,"w") as f:
        f.write(s)
    return s

def optimize_cpu(atoms, filename):
    """
    Optimize a molecule using the CPU.
    atoms is the atoms object.
    filename is the path to the constraints file.
    """
    write_constraints(atoms, filename)
    params = {"constraints": filename,
            "maxsteps": 10}
    mol_eq = optimize(atoms.calc.mf, **params)
    return mol_eq
