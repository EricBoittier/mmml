"""Utilities for working with RDKit molecules in DCMNet workflows.

The production implementation depends on optional libraries (RDKit, ASE) and a
QM9 CSV dataset.  During documentation builds those resources might be
unavailable, so the helper functions load dependencies lazily and raise clear
errors if used without the required components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import os

try:  # Optional heavy dependencies
    import rdkit  # type: ignore
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Draw, MolFromSmiles, MolFromXYZBlock, rdDetermineBonds  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised during docs/tests
    rdkit = None  # type: ignore[assignment]
    Chem = None  # type: ignore[assignment]

try:
    import ase  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ase = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


_QM9_CSV = Path(os.environ.get("MMML_QM9_CSV", "/pchem-data/meuwly/boittier/home/jaxeq/data/qm9.csv"))
_QM9_DF = None


def _require_dependencies() -> None:
    if rdkit is None or Chem is None:
        raise ModuleNotFoundError("rdkit is required for rdkit_utils helpers")
    if ase is None:
        raise ModuleNotFoundError("ASE is required for rdkit_utils helpers")
    if pd is None:
        raise ModuleNotFoundError("pandas is required for rdkit_utils helpers")


def _load_qm9_dataframe():
    global _QM9_DF
    if _QM9_DF is None:
        _require_dependencies()
        if not _QM9_CSV.exists():
            raise FileNotFoundError(
                "QM9 CSV file not found. Set MMML_QM9_CSV to point to the dataset."
            )
        _QM9_DF = pd.read_csv(_QM9_CSV)
    return _QM9_DF


def getXYZblock(batch: dict[str, Any]) -> str:
    _require_dependencies()
    end = len(batch["atomic_numbers"].nonzero()[0])
    atoms = ase.Atoms(
        numbers=batch["atomic_numbers"][:end], positions=batch["positions"][:end, :]
    )
    xyzBlock = f"{end}\n"
    for s, xyz in zip(atoms.symbols, atoms.positions):
        xyzBlock += f"\n{s} {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}"
    return xyzBlock


def get_rdkit(batch: dict[str, Any]):
    _require_dependencies()
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


def get_mol_from_id(batch: dict[str, Iterable[str]]):
    df = _load_qm9_dataframe()
    mols = []
    for id in batch["id"]:
        qm9_id = int(id.split("_")[1]) - 1
        mol = MolFromSmiles(df.iloc[qm9_id]["smiles"])
        mols.append(mol)
    return mols
