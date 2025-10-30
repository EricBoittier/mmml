import numpy as np
from ase import Atoms
from ase.visualize import view


def plot_3d_molecule(batch, batch_size):
    # Infer number of atoms from batch shape
    num_atoms = len(batch["Z"]) // batch_size
    
    i = 0
    b1_ = batch["Z"].reshape(batch_size, num_atoms)[i]
    c1_ = batch["mono"].reshape(batch_size, num_atoms)[i]
    nonzero = np.nonzero(c1_)
    i = 0
    xyz = batch["R"].reshape(batch_size, num_atoms, 3)[i][nonzero]
    elem = batch["Z"].reshape(batch_size, num_atoms)[i][nonzero]

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    return V1, mol


def plot_3d_models(mono, dc, dcq, batch, batch_size):
    n_dcm = mono.shape[-1]
    i = 0
    b1_ = batch["Z"].reshape(batch_size, NATOMS)[i]
    c1_ = batch["mono"].reshape(batch_size, NATOMS)[i]
    nonzero = np.nonzero(b1_)
    print(nonzero)
    i = 0
    xyz = batch["R"].reshape(batch_size, NATOMS, 3)[i][nonzero]
    elem = batch["Z"].reshape(batch_size, NATOMS)[i][nonzero]

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    idx = len(nonzero[0]) * n_dcm
    print(idx)
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in dcq[i][:idx]], 
                  dc[:idx])
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    print(combined)
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3, mol, dcmol, combined
