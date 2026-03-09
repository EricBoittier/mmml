import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

from physnetjax.directories import PYCHARMM_DIR

if PYCHARMM_DIR is None:
    raise ValueError(
        "PYCHARMM_DIR not set. "
        "Please add an entry to the paths.toml file with the path to the CHARMM installation directory."
    )
else:
    os.environ["CHARMM_HOME"] = str(PYCHARMM_DIR)
    os.environ["CHARMM_LIB_DIR"] = str(PYCHARMM_DIR / "build" / "cmake")

import ase
import ase.io as io
import ase.units as units
import jax
import numpy as np
import pandas as pd
import pycharmm
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.lingo as stream
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as write

from physnetjax.calc.helper_mlp import get_ase_calc, get_pyc
from physnetjax.restart.restart import get_params_model_with_ase

# Environment settings


def print_device_info():
    """Print JAX device information."""
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())


def initialize_model(pkl_path, model_path, atoms):
    """Initialize the model with parameters and model."""
    print(atoms, len(atoms))
    params, model = get_params_model_with_ase(pkl_path, model_path, atoms)

    return params, model


def initialize_system():
    """Initialize the system with PDB file and model parameters."""
    # Read topology and parameter files
    read.rtf("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_prot.rtf")
    read.prm(
        "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36m_prot.prm", flex=True
    )
    pycharmm.lingo.charmm_script(
        "stream /pchem-data/meuwly/boittier/home/charmm/toppar/toppar_water_ions.str"
    )


def setup_coordinates(pdb_file, psf_file, atoms):
    """Setup system coordinates and parameters."""
    settings.set_bomb_level(-2)
    settings.set_warn_level(-1)
    read.pdb(pdb_file, resid=True)
    read.psf_card(psf_file)
    if atoms is None:
        atoms = ase.io.read(pdb_file)

    import pycharmm.psf as psf

    Z = [str(_)[:1] for _ in psf.get_atype()]
    atoms = ase.Atoms(Z, atoms.get_positions())
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    coor.show()
    return atoms


def setup_coords_seq(seq):
    """Setup system coordinates and parameters."""
    settings.set_bomb_level(-2)
    settings.set_warn_level(-1)
    read.sequence_string(seq)
    stream.charmm_script("GENERATE PEPT FIRST ACE LAST CT3 SETUP")
    stream.charmm_script("ic param")
    stream.charmm_script("ic seed 1 CAY 1 CY 1 N  ")
    stream.charmm_script("ic build")
    # minmize the system
    minimize.run_sd(**{"nstep": 100, "tolenr": 1e-5, "tolgrd": 1e-5})
    coor.show()
    OUTPUT_PDB = "output.pdb"
    write.coor_pdb(OUTPUT_PDB)
    atoms = ase.io.read(OUTPUT_PDB)

    import pycharmm.psf as psf

    Z = [str(_)[:1] for _ in psf.get_atype()]
    positions = atoms.get_positions()
    atoms = ase.Atoms(Z, positions)

    atoms = ase.Atoms(Z, atoms.get_positions())
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    coor.show()
    minimize.run_sd(**{"nstep": 10, "tolenr": 1e-5, "tolgrd": 1e-5})
    print(atoms)
    return atoms


##########################


def setup_calculator(atoms, params, model, eatol=10, fatol=10):
    """Setup the calculator and verify energies."""
    Z = atoms.get_atomic_numbers()
    # hack to make sure the pdb atom names are not mistaken for atomic numbers
    # todo: fix this in the future
    Z = [_ if _ < 9 else 6 for _ in Z]
    R = atoms.get_positions()
    atoms = ase.Atoms(Z, R)

    conversion = {
        "energy": 1 / (units.kcal / units.mol),
        "forces": -1 / (units.kcal / units.mol),
        "dipole": 1,
    }

    calculator = get_ase_calc(params, model, atoms, conversion=conversion)
    atoms.calc = calculator

    ml_selection = pycharmm.SelectAtoms(seg_id="PEPT")
    print(list(ml_selection))
    energy.show()
    U = atoms.get_potential_energy()
    conversion = {
        "energy": 1 / (units.kcal / units.mol),
        "forces": 1,  # / (units.kcal / units.mol),
        "dipole": 1,
    }

    F = atoms.get_forces()
    model_instance = get_pyc(params, model, atoms, conversion=conversion)
    Z = np.array(Z)

    # Initialize PhysNet calculator
    mlp = pycharmm.MLpot(
        model_instance,
        Z,
        ml_selection,
        ml_fq=False,
    )

    energy_verified, U2 = verify_energy(U, atol=eatol)
    forces_verified, F2 = verify_forces(F, atol=fatol)
    if energy_verified and forces_verified:
        return True, mlp
    raise ValueError(
        "Error in setting up calculator. CHARMM energies do not match calculators'.\n"
        + "CHARMM: {}, Calculator: {}\n".format(U, U2)
        + "CHARMM: {}, Calculator: {}\n".format(F, F2)
        + "Energy absolute tolerance: {}, Forces absolute tolerance: {}".format(
            eatol, fatol
        )
    )


def verify_energy(potential_energy, atol=1e-4):
    """Verify that energies match within tolerance."""
    energy.show()
    user_energy = energy.get_energy()["USER"]
    print(user_energy, potential_energy)
    assert np.isclose(float(potential_energy.squeeze()), float(user_energy), atol=atol)
    print(f"Success! energies are close, within {atol} kcal/mol")
    return True, user_energy


def verify_forces(forces_expected, atol=2):
    """Verify that forces match within tolerance."""
    forces = coor.get_forces().values
    print(forces)
    print(forces_expected)
    assert np.allclose(forces_expected, forces, atol=atol)
    print(f"Success! forces are close, within {atol} kcal/mol/Ang")
    return True, forces


def run_minimization(output_pdb, abnr=False):
    """Run energy minimization and save results."""
    minimize.run_sd(**{"nstep": 100, "tolenr": 1e-5, "tolgrd": 1e-5})
    energy.show()
    if abnr:
        minimize.run_abnr(**{"nstep": 10, "tolenr": 1e-5, "tolgrd": 1e-5})
        energy.show()
    stream.charmm_script("print coor")
    write.coor_pdb(output_pdb)


def get_base_dynamics_dict():
    """Return the base dictionary for dynamics simulations."""
    return {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "nsavv": 0,
        "inbfrq": 10,
        "ihbfrq": 0,
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "TEMINC": 10,
        "TWINDH": 10,
        "TWINDL": -10,
        "iasors": 1,
        "iasvel": 1,
        "ichecw": 0,
    }


def setup_charmm_files(prefix, phase, restart=False):
    """Setup CHARMM files for different simulation phases."""
    files = {}
    if phase == "heating":
        files["res"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.res", file_unit=2, formatted=True, read_only=False
        )
        files["dcd"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.dcd", file_unit=1, formatted=False, read_only=False
        )
    else:
        if restart:
            files["str"] = pycharmm.CharmmFile(
                file_name=restart,
                file_unit=3,
                formatted=True,
                read_only=False,
            )
        files["res"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.{phase}.res",
            file_unit=2,
            formatted=True,
            read_only=False,
        )
        files["dcd"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.{phase}.dcd",
            file_unit=1,
            formatted=False,
            read_only=False,
        )

    return files


def change_integrator(dynamics_dict, integrator):
    if integrator == "langevin":
        dynamics_dict.update(
            {
                "leap": False,
                "verlet": False,
                "cpt": False,
                "new": False,
                "langevin": True,
                "iasors": 0,
                "iasvel": 1,
                "ichecw": 0,
                "twindh": 0,
                "twindl": 0,
            }
        )
    elif integrator == "verlet":
        dynamics_dict.update(
            {
                "leap": False,
                "verlet": True,
                "cpt": False,
                "new": False,
                "langevin": False,
                "iasors": 0,
                "iasvel": 0,
                "ichecw": 1,
            }
        )
    else:
        raise ValueError(f"Integrator {integrator} not supported.")
    return dynamics_dict


def run_heating(
    timestep=0.001,
    tottime=5.0,
    savetime=0.10,
    initial_temp=10,
    final_temp=300,
    prefix="restart",
    nprint=100,
    integrator="verlet",
):
    """
    Run the heating phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 1.0 fs)
        tottime (float): Total simulation time in ps (default: 5.0 = 10 ps)
        savetime (float): Save frequency in ps (default: 0.10 = 100 fs)
        initial_temp (float): Initial temperature in K (default: 10)
        final_temp (float): Final temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "restart")
        nprint (int): Print frequency (default: 10)
    """
    files = setup_charmm_files(prefix, "heating")
    nstep = int(tottime / timestep)
    nsavc = int(savetime / timestep)

    print(f"nstep: {nstep}, nsavc: {nsavc}")

    energy.show()

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": True,
            "nstep": nstep,
            "nsavc": nsavc,
            "ilbfrq": 50,
            "imgfrq": 0,
            "ixtfrq": 0,
            "iunrea": -1,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": nprint,
            "iprfrq": 1000,
            "isvfrq": 1000,
            "ntrfrq": 0,
            "ihtfrq": 50,
            "ieqfrq": 100,
            "firstt": initial_temp,
            "finalt": final_temp,
            "echeck": 10000,
            "tbath": final_temp,
            "twindh": 10,
            "twindl": -10,
        }
    )
    dynamics_dict = change_integrator(dynamics_dict, integrator)
    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    write.coor_pdb(f"{prefix}.pdb")

    for file in files.values():
        file.close()

    return files


def run_equilibration(
    timestep=0.001,
    tottime=5.0,
    savetime=0.01,
    temp=300,
    prefix="",
    integrator="verlet",
    restart=False,
):
    """
    Run the equilibration phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 0.2 fs)
        tottime (float): Total simulation time in ps (default: 5.0 = 50 ps)
        savetime (float): Save frequency in ps (default: 0.01 = 10 fs)
        temp (float): Temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "mm")
    """
    files = setup_charmm_files(prefix, "equi", restart=restart)
    nstep = int(tottime / timestep)
    nsavc = int(savetime / timestep)
    print(f"nstep: {nstep}, nsavc: {nsavc}")

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": False,
            "restart": True,
            "nstep": nstep,
            "nsavc": nsavc,
            "imgfrq": 10,
            "iunrea": files["str"].file_unit,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": 100,
            "iprfrq": 100,
            "ieqfrq": 100,
            "firstt": temp,
            "finalt": temp,
            "tbath": temp,
            "echeck": 10000,
        }
    )
    dynamics_dict = change_integrator(dynamics_dict, integrator)

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)

    # pycharmm.lingo.charmm_script(adaptive_umbrella_script)

    dyn_equi.run()

    for file in files.values():
        file.close()

    write.coor_pdb(f"{prefix}.equi.pdb")
    write.coor_card(f"{prefix}.equi.cor")
    write.psf_card(f"{prefix}.equi.psf")

    return files


def run_production(
    timestep=0.001,
    integrator="verlet",
    tottime=5.0,
    savetime=0.01,
    temp=300,
    prefix="mm",
    restart=False,
):
    """
    Run the production phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 0.2 fs)
        nsteps (int): Number of simulation steps (default: 1000000)
        temp (float): Temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "mm")
    """
    files = setup_charmm_files(prefix, "dyna", restart=restart)
    nsteps = int(tottime / timestep)
    nsavc = int(savetime / timestep)

    print(f"nsteps: {nsteps}")

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": False,
            "restart": True,
            "nstep": nsteps,
            "nsavc": nsavc,
            "imgfrq": 10,
            "iunrea": files["str"].file_unit,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": 10,
            "iprfrq": 100,
            "ieqfrq": 0,
            "firstt": temp,
            "finalt": temp,
        }
    )
    dynamics_dict = change_integrator(dynamics_dict, integrator)
    dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_prod.run()

    for file in files.values():
        file.close()

    write.coor_pdb(f"{prefix}.dyna.pdb")
    write.coor_card(f"{prefix}.dyna.cor")
    write.psf_card(f"{prefix}.dyna.psf")

    return files


def _setup_sim(
    pdb_file: str | Path | None = None,
    pkl_path: str | Path | None = None,
    model_path: str | Path | None = None,
    psf_file: str | Path | None = None,
    seq=False,
    atoms=None,
):
    output_pdb = Path(pdb_file).stem + "_min.pdb" if pdb_file is None else "output.pdb"
    if output_pdb is None:
        raise ValueError("PDB file not provided.")

    # Initialize and setup
    initialize_system()
    if not seq:
        atoms = setup_coordinates(pdb_file, psf_file, atoms)
    if atoms is None:
        atoms = setup_coords_seq("ALA ALA")
    print(atoms, len(atoms))
    params, model = initialize_model(pkl_path, model_path, atoms)
    # Setup calculator and run minimization
    calc_setup, _ = setup_calculator(atoms, params, model)
    if calc_setup:
        print("Calculator setup successful.")
    else:
        print("Error in setting up calculator.")

    run_minimization(output_pdb)
    timestep = 0.0005
    files = run_heating(
        integrator="verlet",
        final_temp=300.0,
        timestep=timestep,
        tottime=0.1,
    )
    files = run_equilibration(
        integrator="verlet",
        prefix="equi",
        temp=300.0,
        tottime=1000,
        timestep=timestep,
        restart=files["res"].file_name,
    )
    files = run_production(
        integrator="verlet",
        prefix="dyna",
        tottime=1000,
        temp=400.0,
        timestep=timestep,
        restart=files["res"].file_name,
    )
    return files


def setup_sim(
    pdb_file: str | Path | None = None,
    model_path: str | Path | None = None,
    pkl_path: str | Path | None = None,
    psf_file: str | Path | None = None,
    basepath: str | Path | None = None,
    atoms=None,
):

    if isinstance(pdb_file, Path):
        pdb_file = str(pdb_file) if basepath is None else str(basepath / pdb_file)
    if isinstance(model_path, Path):
        model_path = str(model_path) if basepath is None else str(basepath / model_path)
    if isinstance(pkl_path, Path):
        pkl_path = str(pkl_path) if basepath is None else str(basepath / pkl_path)
    if isinstance(psf_file, Path):
        psf_file = str(psf_file) if basepath is None else str(basepath / psf_file)

    return _setup_sim(pdb_file, model_path, pkl_path, psf_file, atoms)


def main():
    """Main function to run the simulation."""
    print_device_info()
    base_path = Path("/pchem-data/meuwly/boittier/home/pycharmm_test")
    # File paths
    pdb_file = base_path / "md" / "adp.pdb"
    psf_file = base_path / "md" / "adp.psf"
    restart_name = "cf3all-ecbb2297-d619-4bcf-9607-df23dfbce0dc"
    # "cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a"
    pkl_path = base_path / "ckpts" / restart_name / "params.pkl"
    model_path = base_path / "ckpts" / restart_name / "model_kwargs.pkl"

    swap_atoms = None

    setup_sim(pdb_file, pkl_path, model_path, psf_file, atoms=swap_atoms)


if __name__ == "__main__":
    main()
