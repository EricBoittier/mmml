#!/usr/bin/env python
# coding: utf-8
"""
Absolute binding free energy (ABFE) calculations using OpenFE.

Performs ABFE via a thermodynamic cycle: interactions are decoupled in solvent
(ΔG_solvent) and in the complex (ΔG_complex) to obtain ΔG_ABFE. Coulombic
interactions are annihilated; LJ interactions are decoupled (intermolecular off,
intramolecular retained).

Can be used as a module or from the command line.
"""

import argparse
import json
import pathlib
from typing import Any

import gufe
import openfe
from gufe.protocols import execute_DAG
from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
from openff.units import unit
from rdkit import Chem


def run_abfe(
    config: dict[str, Any],
    *,
    output_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """
    Run an absolute binding free energy calculation.

    Args:
        config: Dict with required keys:
            - sdf: Path to ligand SDF file
            - pdb: Path to protein PDB file
            Optional keys:
            - protocol_repeats: int (default 1)
            - compute_platform: str (default "CUDA")
            - host_min_distance: float in nm (default 0.5)
            - host_max_distance: float in nm (default 1.5)
        output_dir: Base directory for results (default: "abfe_results").

    Returns:
        Dict with keys: estimate, uncertainty, protocol_result, unit_results.
    """
    output_dir = pathlib.Path(output_dir or "abfe_results")
    transformation_dir = output_dir / "transformation_json"
    results_dir = output_dir / "results"
    transformation_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = pathlib.Path(config["sdf"])
    pdb_path = pathlib.Path(config["pdb"])

    # Load ligand
    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    if not ligands:
        raise ValueError(f"No molecules loaded from {sdf_path}")

    # Charge ligand
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

    # Create ChemicalSystems
    solvent = openfe.SolventComponent()
    protein = openfe.ProteinComponent.from_pdb_file(str(pdb_path))

    systemA = openfe.ChemicalSystem(
        {"ligand": ligands[0], "protein": protein, "solvent": solvent},
        name=ligands[0].name,
    )
    systemB = openfe.ChemicalSystem({"protein": protein, "solvent": solvent})

    # Configure protocol
    settings = AbsoluteBindingProtocol.default_settings()
    settings.protocol_repeats = config.get("protocol_repeats", 1)
    settings.restraint_settings.host_min_distance = (
        config.get("host_min_distance", 0.5) * unit.nanometer
    )
    settings.restraint_settings.host_max_distance = (
        config.get("host_max_distance", 1.5) * unit.nanometer
    )
    settings.engine_settings.compute_platform = config.get(
        "compute_platform", "CUDA"
    )

    protocol = AbsoluteBindingProtocol(settings=settings)

    # Create transformation and DAG
    transformation = openfe.Transformation(
        stateA=systemA,
        stateB=systemB,
        mapping=None,
        protocol=protocol,
        name=systemA.name,
    )
    transformation.dump(transformation_dir / f"{transformation.name}.json")

    dag = protocol.create(stateA=systemA, stateB=systemB, mapping=None)

    # Execute
    dag_results = execute_DAG(
        dag,
        scratch_basedir=results_dir,
        shared_basedir=results_dir,
        n_retries=config.get("n_retries", 0),
    )

    # Analyze
    protocol_results = protocol.gather([dag_results])
    estimate = protocol_results.get_estimate()
    uncertainty = protocol_results.get_uncertainty()

    outdict = {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "protocol_result": protocol_results.to_dict(),
        "unit_results": {
            u.key: u.to_keyed_dict() for u in dag_results.protocol_unit_results
        },
    }

    results_file = results_dir / f"{systemA.name}_results.json"
    with open(results_file, "w") as f:
        json.dump(outdict, f, cls=gufe.tokenization.JSON_HANDLER.encoder)

    return outdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run absolute binding free energy calculation with OpenFE"
    )
    parser.add_argument(
        "--sdf",
        required=True,
        help="Path to ligand SDF file",
    )
    parser.add_argument(
        "--pdb",
        required=True,
        help="Path to protein PDB file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="abfe_results",
        help="Output directory (default: abfe_results)",
    )
    parser.add_argument(
        "--protocol-repeats",
        type=int,
        default=1,
        help="Number of protocol repeats (default: 1)",
    )
    parser.add_argument(
        "--compute-platform",
        default="CUDA",
        choices=["CUDA", "OpenCL", "CPU"],
        help="OpenMM compute platform (default: CUDA)",
    )
    parser.add_argument(
        "--host-min-distance",
        type=float,
        default=0.5,
        help="Boresch restraint min distance in nm (default: 0.5)",
    )
    parser.add_argument(
        "--host-max-distance",
        type=float,
        default=1.5,
        help="Boresch restraint max distance in nm (default: 1.5)",
    )

    args = parser.parse_args()

    config = {
        "sdf": args.sdf,
        "pdb": args.pdb,
        "protocol_repeats": args.protocol_repeats,
        "compute_platform": args.compute_platform,
        "host_min_distance": args.host_min_distance,
        "host_max_distance": args.host_max_distance,
    }

    result = run_abfe(config, output_dir=args.output_dir)
    print(
        f"ABFE dG: {result['estimate']}, err {result['uncertainty']}"
    )


if __name__ == "__main__":
    main()


"""

python abfe_script.py --sdf path/to/ligand.sdf --pdb path/to/protein.pdb
python abfe_script.py --sdf ligand.sdf --pdb protein.pdb -o my_results --protocol-repeats 3


from mmml.interfaces.openfeInterface.abfe_script import run_abfe

config = {"sdf": "toluene.sdf", "pdb": "t4_lysozyme.pdb"}
result = run_abfe(config, output_dir="abfe_results")
print(result["estimate"], result["uncertainty"])
"""