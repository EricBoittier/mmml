import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromXYZBlock
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Draw
import ase
import pandas as pd

df = pd.read_csv("/pchem-data/meuwly/boittier/home/jaxeq/data/qm9.csv")

from rdkit.Chem import MolFromSmiles


def getXYZblock(batch):
    end = len(batch["atomic_numbers"].nonzero()[0])
    atoms = ase.Atoms(
        numbers=batch["atomic_numbers"][:end], positions=batch["positions"][:end, :]
    )
    xyzBlock = f"{end}\n"
    for s, xyz in zip(atoms.symbols, atoms.positions):
        xyzBlock += f"\n{s} {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}"
    return xyzBlock


def get_rdkit(batch):
    raw_mol = MolFromXYZBlock(getXYZblock(batch))
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol, charge=0)
    # DetermineBondOrders
    rdDetermineBonds.DetermineBondOrders(conn_mol, charge=0)
    Chem.SanitizeMol(conn_mol)
    Chem.Kekulize(conn_mol)
    bond_moll = Chem.Mol(conn_mol)
    smi = Chem.MolToSmiles(Chem.RemoveHs(bond_moll))
    print(smi)
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol)
    return img


def get_mol_from_id(batch):
    mols = []
    for id in batch["id"]:
        qm9_id = int(id.split("_")[1]) - 1
        mol = MolFromSmiles(df.iloc[qm9_id]["smiles"])
        mols.append(mol)
    return mols
