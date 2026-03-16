#!/usr/bin/env python
# coding: utf-8
"""
Absolute binding free energy (ABFE) calculations using OpenFE.

Performs ABFE of toluene binding to T4 Lysozyme via a thermodynamic cycle:
interactions are decoupled in solvent (ΔG_solvent) and in the complex (ΔG_complex)
to obtain ΔG_ABFE. Coulombic interactions are annihilated; LJ interactions
are decoupled (intermolecular off, intramolecular retained).
"""

import json
import pathlib

import openfe
from gufe.protocols import execute_DAG
from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
from openff.units import unit
import gufe
from rdkit import Chem

# --- 1. Load ligand ---
supp = Chem.SDMolSupplier("../cookbook/assets/toluene.sdf", removeHs=False)
ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]

# --- 2. Charge ligand ---
charge_settings = OpenFFPartialChargeSettings(
    partial_charge_method="am1bcc", off_toolkit_backend="ambertools"
)
ligands = bulk_assign_partial_charges(
    molecules=ligands,
    overwrite=False,
    method=charge_settings.partial_charge_method,
    toolkit_backend=charge_settings.off_toolkit_backend,
    generate_n_conformers=charge_settings.number_of_conformers,
    nagl_model=charge_settings.nagl_model,
    processors=1,
)

# --- 3. Create ChemicalSystems ---
solvent = openfe.SolventComponent()
protein = openfe.ProteinComponent.from_pdb_file("../cookbook/assets/t4_lysozyme.pdb")

systemA = openfe.ChemicalSystem(
    {"ligand": ligands[0], "protein": protein, "solvent": solvent},
    name=ligands[0].name,
)
systemB = openfe.ChemicalSystem({"protein": protein, "solvent": solvent})

# --- 4. Configure protocol ---
settings = AbsoluteBindingProtocol.default_settings()
settings.protocol_repeats = 1
settings.restraint_settings.host_min_distance = 0.5 * unit.nanometer
settings.restraint_settings.host_max_distance = 1.5 * unit.nanometer
settings.engine_settings.compute_platform = "CUDA"

protocol = AbsoluteBindingProtocol(settings=settings)

# --- 5. Create transformation and DAG ---
transformation = openfe.Transformation(
    stateA=systemA,
    stateB=systemB,
    mapping=None,
    protocol=protocol,
    name=systemA.name,
)

transformation_dir = pathlib.Path("abfe_json")
transformation_dir.mkdir(exist_ok=True)
transformation.dump(transformation_dir / f"{transformation.name}.json")

dag = protocol.create(stateA=systemA, stateB=systemB, mapping=None)

# --- 6. Execute simulation ---
results_dir = pathlib.Path("abfe_results")
results_dir.mkdir(exist_ok=True)
dag_results = execute_DAG(
    dag, scratch_basedir=results_dir, shared_basedir=results_dir, n_retries=0
)

# --- 7. Analyze and save results ---
protocol_results = protocol.gather([dag_results])
print(f"ABFE dG: {protocol_results.get_estimate()}, err {protocol_results.get_uncertainty()}")

outdict = {
    "estimate": protocol_results.get_estimate(),
    "uncertainty": protocol_results.get_uncertainty(),
    "protocol_result": protocol_results.to_dict(),
    "unit_results": {
        u.key: u.to_keyed_dict() for u in dag_results.protocol_unit_results
    },
}

results_file = results_dir / "toluene_results.json"
with open(results_file, "w") as f:
    json.dump(outdict, f, cls=gufe.tokenization.JSON_HANDLER.encoder)
