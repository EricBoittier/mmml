from itertools import combinations, permutations, product
from typing import Dict, Tuple, List, Any, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
import os

os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
from jax import jit
import jax.numpy as jnp
import ase.calculators.calculator as ase_calc

# from jax import config
# config.update('jax_enable_x64', True)

# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

import sys
import e3x
import jax
import numpy as np
import optax
import orbax
from pathlib import Path
import pandas as pd

# Add custom path
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test")
import physnetjax

sys.path.append("/pchem-data/meuwly/boittier/home/dcm-lj-data")
from pycharmm_lingo_scripts import script1, script2, script3, load_dcm

from physnetjax.data.data import prepare_datasets
from physnetjax.training.loss import dipole_calc
from physnetjax.models.model import EF
from physnetjax.training.training import train_model  # from model import dipole_calc
from physnetjax.data.batches import (
    _prepare_batches as prepare_batches,
)  # prepare_batches, prepare_datasets

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

from pathlib import Path

from physnetjax.calc.helper_mlp import get_ase_calc


def parse_non_int(s):
    return "".join([_ for _ in s if _.isalpha()]).lower().capitalize()


read_parameter_card = """
read parameter card
* methanol
*
NONBONDED
CG321    0.0       {CG321EP:.4f}     {CG321RM:.4f}   0.0 -0.01 1.9 ! alkane (CT2), 4/98, yin, adm jr, also used by viv
CLGA1    0.0       {CLGA1EP:.4f}    {CLGA1RM:.4f} ! CLET, DCLE, chloroethane, 1,1-dichloroethane
HGA2     0.0       {HGA2EP:.4f}    {HGA2RM:.4f} ! alkane, yin and mackerell, 4/98
END
"""
# HGA2     0.0       -0.0200     1.3400 ! alkane, yin and mackerell, 4/98


NATOMS = 10

model = EF(
    # attributes
    features=128,
    max_degree=0,
    num_iterations=5,
    num_basis_functions=64,
    cutoff=10.0,
    max_atomic_number=18,
    charges=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=3,
    zbl=False,
    debug=False,
)


import pycharmm

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.select as select
import pycharmm.shake as shake

from pycharmm.lib import charmm as libcharmm


import ase
from ase.io import read as read_ase
from ase import visualize
from ase.visualize import view


from scipy.optimize import minimize

ev2kcalmol = 1 / (ase.units.kcal / ase.units.mol)

CG321EP = -0.0560
CG321RM = 2.0100
CLGA1EP = -0.3430
CLGA1RM = 1.9100
HGA2EP = -0.0200
HGA2RM = 1.3400


def set_pycharmm_xyz(atom_positions):
    xyz = pd.DataFrame(atom_positions, columns=["x", "y", "z"])
    coor.set_positions(xyz)


def capture_neighbour_list():
    # Print something
    distance_command = """
    open unit 1 write form name total.dmat
    
    COOR DMAT SINGLE UNIT 1 SELE ALL END SELE ALL END
    
    close unit 1"""
    _ = pycharmm.lingo.charmm_script(distance_command)

    with open("total.dmat") as f:
        output_dmat = f.read()

    atom_number_type_dict = {}
    atom_number_resid_dict = {}

    pair_distance_dict = {}
    pair_resid_dict = {}

    for _ in output_dmat.split("\n"):
        if _.startswith("*** "):
            _, n, resid, resname, at, _ = _.split()

            n = int(n.split("=")[0]) - 1
            atom_number_type_dict[n] = at
            atom_number_resid_dict[n] = int(resid) - 1

    for _ in output_dmat.split("\n"):
        if _.startswith("  "):
            a, b, dist = _.split()
            a = int(a) - 1
            b = int(b) - 1
            dist = float(dist)
            if atom_number_resid_dict[a] < atom_number_resid_dict[b]:
                pair_distance_dict[(a, b)] = dist
                pair_resid_dict[(a, b)] = (
                    atom_number_resid_dict[a],
                    atom_number_resid_dict[b],
                )

    return {
        "atom_number_type_dict": atom_number_type_dict,
        "atom_number_resid_dict": atom_number_resid_dict,
        "pair_distance_dict": pair_distance_dict,
        "pair_resid_dict": pair_resid_dict,
    }


def get_forces_pycharmm():
    positions = coor.get_positions()
    force_command = """coor force sele all end"""
    _ = pycharmm.lingo.charmm_script(force_command)
    forces = coor.get_positions()
    coor.set_positions(positions)
    return forces


def view_atoms(atoms):
    return view(atoms, viewer="x3d")


from itertools import combinations


def dimer_permutations(n_mol):
    dimer_permutations = list(combinations(range(n_mol), 2))
    return dimer_permutations


def calc_pycharmm_dimers(n_mol=20, n_atoms=5, forces=False):
    RANGE = len(dimer_permutations(n_mol))

    ele_energies = np.zeros(RANGE)
    evdw_energies = np.zeros(RANGE)
    mm_forces = np.zeros((RANGE, n_atoms * n_mol, 3))

    for i, (a, b) in enumerate(dimer_permutations(20)):
        reset_block_no_internal()
        a += 1
        b += 1
        block = f"""BLOCK
CALL 1 SELE .NOT. (RESID {a} .OR. RESID {b}) END
CALL 2 SELE (RESID {a} .OR. RESID {b}) END
COEFF 1 1 0.0
COEFF 2 2 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0
COEFF 1 2 0.0
END
        """
        _ = pycharmm.lingo.charmm_script(block)
        # print(_)
        energy.show()
        if forces:
            f = get_forces_pycharmm().to_numpy()
            mm_forces[i] = f

        evdw = energy.get_vdw()
        evdw_energies[i] = evdw
        e = energy.get_elec()
        ele_energies[i] = e

    return {
        "ele_energies": ele_energies,
        "evdw_energies": evdw_energies,
        "mm_forces": mm_forces,
    }


def reset_block():
    block = f"""BLOCK 
        CALL 1 SELE ALL END
          COEFF 1 1 1.0 
        END
        """
    _ = pycharmm.lingo.charmm_script(block)


def reset_block_no_internal():
    block = f"""BLOCK 
        CALL 1 SELE ALL END
          COEFF 1 1 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 
        END
        """
    _ = pycharmm.lingo.charmm_script(block)


reset_block_no_internal()


import MDAnalysis as mda


def load_pdb_data(pdb_file):
    loaded_pdb = mda.coordinates.PDB.PDBReader(pdb_file)
    loaded_pdb = mda.topology.PDBParser.PDBParser(pdb_file)
    atypes = psf.get_atype()
    atc = pycharmm.param.get_atc()
    residues = psf.get_res()
    psf.get_natom()
    # nl_info = capture_neighbour_list()

    # TODO: this assumes a pure system, need to update
    atoms_per_res = int(len(atypes) / len(residues))
    n_res = len(residues)
    resids = np.array([[i] * atoms_per_res for i in range(n_res)]).flatten()
    u = mda.Universe(pdb_file)
    atom_names = [s for s in list(u.atoms.names)]
    atom_positions = list(u.atoms.positions)
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[parse_non_int(s)] for s in atom_names]
    )
    mda_resids = [s for s in list(u.atoms.resids)]
    mda_res_at_dict = {
        (a - 1, b): i for i, (a, b) in enumerate(zip(mda_resids, atom_names))
    }
    charmm_res_at_dict = {(a, b): i for i, (a, b) in enumerate(zip(resids, atypes))}
    an_charmm_res_at_dict = {v: k for k, v in charmm_res_at_dict.items()}
    an_mda_res_at_dict = {v: k for k, v in mda_res_at_dict.items()}
    atom_positions = np.array(atom_positions)
    reorder = np.array(
        [charmm_res_at_dict[an_mda_res_at_dict[i]] for i in range(len(atom_positions))]
    )
    atom_positions = atom_positions[reorder]
    atomic_numbers = atomic_numbers[reorder]

    return {
        "atom_names": atom_names,
        "atom_positions": atom_positions,
        "atomic_numbers": atomic_numbers,
        "mda_resids": mda_resids,
        "mda_res_at_dict": mda_res_at_dict,
        "charmm_res_at_dict": charmm_res_at_dict,
        "an_charmm_res_at_dict": an_charmm_res_at_dict,
        "an_mda_res_at_dict": an_mda_res_at_dict,
        "atom_positions": atom_positions,
        "reorder": reorder,
        "atom_positions": atom_positions,
        "atomic_numbers": atomic_numbers,
    }


def get_data_mda(fn):
    pdb_file = data_path / "dcmk" / fn
    pdb_data_mda = load_pdb_data(pdb_file)
    return pdb_data_mda


epsilon = 10 ** (-6)
from e3x.nn import smooth_switch, smooth_cutoff


def combine_with_sigmoid(
    r,
    mm_energy,
    ml_energy,
    dif=10 ** (-6),
    MM_CUTON=5.0,
    MM_CUTOFF=10.0,
    BUFFER=0.1,
    debug=False,
):
    ML_CUTOFF = MM_CUTON - dif
    charmm_on_scale = smooth_switch(r, x0=ML_CUTOFF, x1=MM_CUTON)
    charmm_off_scale = smooth_cutoff(r - MM_CUTON, cutoff=MM_CUTOFF - MM_CUTON)
    # remove any sigularities
    charmm_off_scale = jax.numpy.nan_to_num(charmm_off_scale, posinf=1)

    ml_scale = 1 - abs(smooth_switch(r, x0=ML_CUTOFF - BUFFER, x1=ML_CUTOFF))
    ml_contrib = ml_scale * ml_energy

    mm_contrib = charmm_on_scale * mm_energy
    mm_contrib = mm_contrib * charmm_off_scale

    return mm_contrib, ml_contrib, charmm_off_scale, charmm_on_scale, ml_scale


def indices_of_pairs(a, b, n_atoms=5, n_mol=20):
    assert a < b, "by convention, res a must have a smaller index than res b"
    assert a >= 1, "res indices can't start from 1"
    assert b >= 1, "res indices can't start from 1"
    assert a != b, "pairs can't contain same residue"
    return np.concatenate(
        [
            np.arange(0, n_atoms, 1) + (a - 1) * n_atoms,
            np.arange(0, n_atoms, 1) + (b - 1) * n_atoms,
        ]
    )


def indices_of_monomer(a, n_atoms=5, n_mol=20):
    assert a < (n_mol + 1), "monomer index outside total n molecules"
    return np.arange(0, n_atoms, 1) + (a - 1) * n_atoms


def calc_physnet_via_idx_list(all_coordinates, all_idxs, calculator):
    RANGE = len(all_idxs)
    ml_energies = np.zeros(RANGE)
    ml_forces = np.zeros((RANGE, len(all_idxs[0]), 3))

    for i, idxs in enumerate(all_idxs):
        # set positions
        calculator.set_positions(all_coordinates[idxs])

        ml_energies[i] = calculator.get_potential_energy()
        ml_forces[i] = calculator.get_forces()

    return {"ml_energies": ml_energies, "ml_forces": ml_forces}


def get_dimer_distances(dimer_idxs, all_monomer_idxs, R):
    out_dists = np.zeros(len(dimer_idxs))
    for i, (a, b) in enumerate(dimer_idxs):
        a = all_monomer_idxs[a][0]  # just distance to first atom in the molecule...
        b = all_monomer_idxs[b][0]  # TODO: generalize...
        out_dists[i] = np.linalg.norm(R[a] - R[b])

    return out_dists


def setup_ase_atoms(atomic_numbers, positions, n_atoms):
    """Create and setup ASE Atoms object with centered positions"""
    Z = [_ for i, _ in enumerate(atomic_numbers) if i < n_atoms]
    R = np.array([_ for i, _ in enumerate(positions) if i < n_atoms])
    atoms = ase.Atoms(Z, R)
    # translate to center of mass
    # atoms.set_positions(R - R.T.mean(axis=1))
    return atoms


def create_physnet_calculator(params, model, atoms, ev2kcalmol):
    """Create PhysNet calculator with specified parameters"""
    calc = get_ase_calc(
        params,
        model,
        atoms,
        conversion={"energy": ev2kcalmol, "dipole": 1, "forces": ev2kcalmol},
    )
    atoms.calc = calc
    return atoms


def initialize_models(restart_path, N_ATOMS_MONOMER):
    """Initialize monomer and dimer models from restart"""
    restart = get_last(restart_path)

    # Setup monomer model
    params, monomer_model = get_params_model(restart)
    monomer_model.natoms = N_ATOMS_MONOMER

    # Setup dimer model
    params, dimer_model = get_params_model(restart)
    dimer_model.natoms = N_ATOMS_MONOMER * 2

    return params, monomer_model, dimer_model


def get_rmse_mae(energy, ref_energy):
    rmse = np.sqrt(np.mean((energy - ref_energy) ** 2))
    mae = np.mean(np.abs(energy - ref_energy))
    return rmse, mae


def print_energy_comparison(mmml_energy, charmm, ref_energy):
    """Print comparison of energies with reference data"""

    print("comb")
    print(mmml_energy, ref_energy, abs(mmml_energy - ref_energy))

    print("charmm")
    print(charmm, ref_energy, abs(charmm - ref_energy))


def calculate_E_pair(dimer_results, monomer_results, dimer_idxs, result):
    """Calculate and combine ML and MM energies"""
    summed_ml_intE = dimer_results["ml_energies"] - monomer_results["ml_energies"][
        np.array(dimer_idxs)
    ].sum(axis=1)
    summed_mm_intE = result["ele_energies"] + result["evdw_energies"]
    return summed_ml_intE, summed_mm_intE


def calculate_F_pair(dimer_results, monomer_results, dimer_idxs, result):
    """Calculate and combine ML and MM forces"""
    mono = monomer_results["ml_forces"][np.array(dimer_idxs)]
    print(mono.shape)
    a, b, c, d = mono.shape
    mono = mono.reshape(a, b * c, d)
    summed_ml_intF = dimer_results["ml_forces"] - mono
    summed_mm_intF = result["mm_forces"]
    return summed_ml_intF, summed_mm_intF


def get_fnkey(fn):
    fnkey = str(fn).split("/")[-1].split(".")[0].upper()
    fnkey = "_".join(fnkey.split("_")[:3])
    return fnkey


def calc_energies_forces(
    fn, DO_ML=True, DO_MM=True, MM_CUTON=6.0, MM_CUTOFF=10.0, BUFFER=0.1
):
    pdb_data_mda = get_data_mda(fn)
    atomic_numbers, atom_positions = (
        pdb_data_mda["atomic_numbers"],
        pdb_data_mda["atom_positions"],
    )
    set_pycharmm_xyz(atom_positions)
    energy.show()

    ase_atom_full_system = ase.Atoms(atomic_numbers, atom_positions)

    result = None
    summed_2body = None
    mmml_energy = None
    charmm = None

    if DO_MM:
        # Calculate CHARMM energies and forces first
        result = calc_pycharmm_dimers(forces=True)
        summed_2body = result["mm_forces"].sum(axis=0)
        mm_forces = result["mm_forces"]

    all_coordinates = ase_atom_full_system.get_positions()
    dimer_idxs = dimer_permutations(20)

    dimer_pair_c_c_distances = get_dimer_distances(
        dimer_idxs, all_monomer_idxs, all_coordinates
    )

    if DO_ML:
        dimer_results = calc_physnet_via_idx_list(
            all_coordinates, all_dimer_idxs, ase_atoms_dimer
        )

        monomer_results = calc_physnet_via_idx_list(
            all_coordinates, all_monomer_idxs, ase_atoms_monomer
        )

        # Calculate ML and MM energies
        summed_ml_intE, summed_mm_intE = calculate_E_pair(
            dimer_results, monomer_results, dimer_idxs, result
        )

        summed_ml_intF, summed_mm_intF = calculate_F_pair(
            dimer_results, monomer_results, dimer_idxs, result
        )

    if DO_MM and DO_ML:
        combined_with_switches = combine_with_sigmoid(
            dimer_pair_c_c_distances,
            summed_mm_intE,
            summed_ml_intE,
            MM_CUTON=MM_CUTON,
            MM_CUTOFF=MM_CUTOFF,
            BUFFER=BUFFER,
        )
        (
            mm_contrib,
            ml_contrib,
            charmm_off_scale,
            charmm_on_scale,
            ml_scale,
        ) = combined_with_switches

        mmml_energy = float(ml_contrib.sum() + mm_contrib.sum())
        charmm = float(summed_mm_intE.sum())
    else:
        mmml_energy = float(summed_ml_intE.sum())
        charmm = float(summed_mm_intE.sum())

    print(summed_ml_intE.shape, summed_mm_intE.shape)

    mm_forces = summed_mm_intF
    ml_forces = summed_ml_intF

    indices = np.array(all_dimer_idxs).flatten()[:, None].repeat(3, axis=1) + np.array(
        [0, mm_forces.shape[1], 2 * mm_forces.shape[1]]
    )
    flattened_ml_dimers = ml_forces.reshape(-1, 3).flatten()
    # indices = np.repeat(np.array(all_dimer_idxs).flatten(), 3)
    mmml_forces = jax.ops.segment_sum(flattened_ml_dimers, indices.flatten()).reshape(
        mm_forces.shape[1], 3
    )

    # mmml_forces = (mm_forces, ml_forces)

    output_dict = {
        "mmml_energy": mmml_energy,
        "charmm": charmm,
        "mm_forces": mm_forces,
        "ml_forces": ml_forces,
        "mmml_forces": mmml_forces,
    }

    return output_dict


def compare_energies(
    fn, df, DO_ML=True, DO_MM=True, MM_CUTON=6.0, MM_CUTOFF=10.0, BUFFER=0.1
):
    energy_forces_dict = calc_energies_forces(
        fn,
        DO_ML=DO_ML,
        DO_MM=DO_MM,
        MM_CUTON=MM_CUTON,
        MM_CUTOFF=MM_CUTOFF,
        BUFFER=BUFFER,
    )
    mmml_energy = energy_forces_dict["mmml_energy"]
    charmm = energy_forces_dict["charmm"]
    mm_forces = energy_forces_dict["mm_forces"]
    ml_forces = energy_forces_dict["ml_forces"]
    mmml_forces = energy_forces_dict["mmml_forces"]

    # print(fn)
    fnkey = get_fnkey(fn)
    # print(fnkey)
    # print(df)
    if fnkey in df["key"].values:
        df = df[df["key"] == fnkey]
        # print(df)
        ref_energy = df.iloc[0]["Formation Energy (kcal/mol)"]
        if DO_MM:
            err_mmml = mmml_energy - ref_energy
            err_charmm = charmm - ref_energy
        else:
            err_mmml = mmml_energy - ref_energy
            err_charmm = None
    else:
        ref_energy = None
        err_mmml = None
        err_charmm = None

    results_dict = {
        "ref_energy": ref_energy,
        "mmml_energy": mmml_energy,
        "charmm": charmm,
        "err_mmml": err_mmml,
        "err_charmm": err_charmm,
        "mm_forces": mm_forces,
        "ml_forces": ml_forces,
        "mmml_forces": mmml_forces,
    }
    return results_dict


def set_param_card(
    CG321EP=CG321EP,
    CG321RM=CG321RM,
    CLGA1EP=CLGA1EP,
    CLGA1RM=CLGA1RM,
    HGA2EP=HGA2EP,
    HGA2RM=HGA2RM,
):
    cmd = "PRNLev 5\nWRNLev 5"
    param_card = read_parameter_card.format(
        CG321EP=CG321EP,
        CG321RM=CG321RM,
        CLGA1EP=CLGA1EP,
        CLGA1RM=CLGA1RM,
        HGA2EP=HGA2EP,
        HGA2RM=HGA2RM,
    )
    print(param_card)
    pycharmm.lingo.charmm_script(param_card)
    cmd = "PRNLev 0\nWRNLev 0"
    pycharmm.lingo.charmm_script(cmd)


def get_loss_terms(
    fns,
    MM_CUTON=6.0,
    MM_CUTOFF=10.0,
    BUFFER=0.01,
    MM_lambda=1.0,
    ML_lambda=0.0,
    DO_MM=True,
    DO_ML=True,
):
    import time

    start = time.time()
    err_mmml_list = []
    err_charmm_list = []
    for fn in fns:
        results_dict = compare_energies(
            fn,
            df,
            DO_MM=DO_MM,
            DO_ML=DO_ML,
            MM_CUTON=MM_CUTON,
            MM_CUTOFF=MM_CUTOFF,
            BUFFER=BUFFER,
        )
        err_mmml_list.append(results_dict["err_mmml"])
        err_charmm_list.append(results_dict["err_charmm"])
        print(
            "{} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                fn.stem,
                results_dict["ref_energy"],
                results_dict["mmml_energy"],
                results_dict["charmm"],
                results_dict["err_mmml"],
                results_dict["err_charmm"],
            )
        )

    end = time.time()
    print("Finished")
    print("Time taken", end - start)
    print("--------------------------------")
    err_mmml_list = np.array(err_mmml_list)
    err_charmm_list = np.array(err_charmm_list)

    print("RMSE MMML", np.sqrt(np.mean(err_mmml_list**2)))
    print("MAE MMML", np.mean(np.abs(err_mmml_list)))
    print("RMSE Charmm", np.sqrt(np.mean(err_charmm_list**2)))
    print("MAE Charmm", np.mean(np.abs(err_charmm_list)))

    loss = MM_lambda * np.mean(err_mmml_list**2) + ML_lambda * np.mean(
        err_charmm_list**2
    )
    return loss, err_mmml_list, err_charmm_list


def get_loss_fn(
    train_filenames,
    DO_ML=True,
    DO_MM=True,
    NTRAIN=20,
    MM_CUTON=6.0,
    MM_lambda=1.0,
    ML_lambda=0.0,
):
    def loss_fn(x0):
        print("Starting")
        # random_indices = np.random.randint(0, len(train_filenames),6)
        fns = [train_filenames[i] for i in range(NTRAIN)]
        CG321EP, CG321RM, CLGA1EP, CLGA1RM = x0[:4]
        set_param_card(CG321EP, CG321RM, CLGA1EP, CLGA1RM)
        loss, _, _ = get_loss_terms(
            fns,
            MM_CUTON=MM_CUTON,
            MM_lambda=MM_lambda,
            ML_lambda=ML_lambda,
            DO_MM=DO_MM,
            DO_ML=DO_ML,
        )
        print("Loss", loss)
        return loss

    return loss_fn


def ep_scale_loss(x0):
    print("Starting")
    random_indices = np.random.randint(0, len(train_filenames), 4)
    fns = [train_filenames[i] for i in random_indices]
    ep_scale = float(x0)
    set_param_card(CG321EP * ep_scale, CG321RM, CLGA1EP * ep_scale, CLGA1RM)
    loss, _, _ = get_loss_terms(fns)
    print("Loss", loss)
    return loss


def create_initial_simplex(x0, delta=0.0001):
    initial_simplex = np.zeros((len(x0) + 1, len(x0)))
    initial_simplex[0] = x0  # First point is x0
    for i in range(len(x0)):
        initial_simplex[i + 1] = x0.copy()
        initial_simplex[i + 1, i] += delta  # Add small step in dimension i
    return initial_simplex


def optimize_params_simplex(
    x0, bounds, loss, method="Nelder-Mead", maxiter=100, xatol=0.0001, fatol=0.0001
):
    initial_simplex = create_initial_simplex(x0)
    res = minimize(
        loss,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "xatol": 0.0001,  # Absolute tolerance on x
            "fatol": 0.0001,  # Absolute tolerance on function value
            "initial_simplex": initial_simplex,
            "maxiter": 100,
        },
    )  # Initial simplex with steps of 0.0001

    print(res)
    return res


def get_bounds(x0, scale=0.1):
    b = [
        (
            (x0[i] * (1 - scale), x0[i] * (1 + scale))
            if x0[i] > 0
            else (x0[i] * (1 + scale), x0[i] * (1 - scale))
        )
        for i in range(len(x0))
    ]
    return b


from physnetjax.restart.restart import get_last, get_files, get_params_model
from physnetjax.analysis.analysis import plot_stats


def get_block(a, b):
    block = f"""BLOCK
CALL 1 SELE .NOT. (RESID {a} .OR. RESID {b}) END
CALL 2 SELE (RESID {a} .OR. RESID {b}) END
COEFF 1 1 0.0
COEFF 2 2 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0
COEFF 1 2 0.0
END
"""
    return block


@jit
def switch_MM(
    X,
    mm_energy,
    dif=10 ** (-6),
    MM_CUTON=6.0,
    MM_CUTOFF=10.0,
    BUFFER=0.1,
    debug=False,
):
    r = jnp.linalg.norm(X[:5].T.mean(axis=1) - X[5:10].T.mean(axis=1))
    ML_CUTOFF = MM_CUTON - dif
    charmm_on_scale = smooth_switch(r, x0=ML_CUTOFF, x1=MM_CUTON)
    charmm_off_scale = smooth_cutoff(r - MM_CUTON, cutoff=MM_CUTOFF - MM_CUTON)
    # remove any sigularities
    charmm_off_scale = jax.numpy.nan_to_num(charmm_off_scale, posinf=1)
    mm_contrib = charmm_on_scale * mm_energy * charmm_off_scale
    return mm_contrib


@jit
def switch_ML(
    X,
    ml_energy,
    dif=10 ** (-6),
    MM_CUTON=6.0,
    MM_CUTOFF=10.0,
    BUFFER=0.1,
    debug=False,
):
    # Calculate center-of-mass distance between monomers
    r = jnp.linalg.norm(X[:5].T.mean(axis=1) - X[5:10].T.mean(axis=1))

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    r = r + eps

    ML_CUTOFF = MM_CUTON - dif
    ml_scale = 1 - jnp.abs(smooth_switch(r, x0=ML_CUTOFF - BUFFER, x1=ML_CUTOFF))

    # Ensure scale is between 0 and 1
    ml_scale = jnp.clip(ml_scale, 0.0, 1.0)

    ml_contrib = ml_scale * ml_energy
    return ml_contrib


switch_ML_grad = jax.grad(switch_ML)
switch_MM_grad = jax.grad(switch_MM)


@jit
def combine_with_sigmoid_E(
    X,
    mm_energy,
    ml_energy,
    dif=10 ** (-6),
    MM_CUTON=6.0,
    MM_CUTOFF=10.0,
    BUFFER=0.1,
    debug=False,
):
    ml_contrib = switch_ML(X, ml_energy)
    mm_contrib = switch_ML(X, mm_energy)
    return mm_contrib + ml_contrib


# DATA
###################################################################
data_path = Path("/pchem-data/meuwly/boittier/home/dcm-lj-data")
df = pd.read_csv(data_path / "formation_energies_kcal_mol.csv", sep="\t")
df["key"] = df["Cluster"].apply(lambda x: "_".join(x.split("_")[:3]).upper())

print(df)


R = coor.get_positions().to_numpy()

# System constants
ATOMS_PER_MONOMER: int = 5  # Number of atoms in each monomer
MAX_ATOMS_PER_SYSTEM: int = 10  # Maximum atoms in monomer/dimer system
SPATIAL_DIMS: int = 3  # Number of spatial dimensions (x, y, z)

# Batch processing constants
BATCH_SIZE: int = 210  # Number of systems per batch


def get_MM_energy_forces_fns(R):
    """Creates functions for calculating MM energies and forces with switching.

    Returns:
        Tuple[Callable, Callable]: Functions for energy and force calculations
    """
    CG321EP = -0.0560
    CG321RM = 2.0100
    CLGA1EP = -0.3430
    CLGA1RM = 1.9100
    HGA2EP = -0.0200
    HGA2RM = 1.3400

    params = [75, 76, 77]
    params.sort()
    at_ep = {75: CG321EP, 76: CLGA1EP, 77: HGA2EP}
    at_rm = {75: CG321RM, 76: CLGA1RM, 77: HGA2RM}
    at_q = {75: -0.018, 76: -0.081, 77: 0.09}

    at_flat_rm = np.zeros(100)
    at_flat_rm[75] = CG321RM
    at_flat_rm[76] = CLGA1RM
    at_flat_rm[77] = HGA2RM

    at_flat_ep = np.zeros(100)
    at_flat_ep[75] = CG321EP
    at_flat_ep[76] = CLGA1EP
    at_flat_ep[77] = HGA2EP

    at_flat_q = np.zeros(100)
    at_flat_q[75] = -0.018
    at_flat_q[76] = -0.081
    at_flat_q[77] = 0.09

    pair_idxs_product = jnp.array(
        [(a, b) for a, b in list(product(np.arange(5), repeat=2))]
    )
    dimer_perms = jnp.array(dimer_permutations(20))

    pair_idxs_np = dimer_perms * 5
    pair_idx_atom_atom = pair_idxs_np[:, None, :] + pair_idxs_product[None, ...]
    pair_idx_atom_atom = pair_idx_atom_atom.reshape(-1, 2)

    displacements = R[pair_idx_atom_atom[:, 0]] - R[pair_idx_atom_atom[:, 1]]
    distances = jnp.linalg.norm(displacements, axis=1)
    at_perms = [_ for _ in list(product(params, repeat=2)) if _[0] <= _[1]]

    charges = np.array(psf.get_charges())
    masses = np.array(psf.get_amass())
    at_codes = np.array(psf.get_iac())
    atomtype_codes = np.array(psf.get_atype())

    at_perms_ep = [(at_ep[a] * at_ep[b]) ** 0.5 for a, b in at_perms]
    at_perms_rm = [(at_rm[a] + at_rm[b]) for a, b in at_perms]
    at_perms_qq = [(at_q[a] * at_q[b]) for a, b in at_perms]
    at_perms_ep, at_perms_rm

    rmins_per_system = jnp.take(
        at_flat_rm, at_codes
    )  # jnp.array([ NBL["pair_rm"][k] for k in atom_keys ])
    epsilons_per_system = jnp.take(
        at_flat_ep, at_codes
    )  # jnp.array([ NBL["pair_ep"][k] for k in atom_keys ])

    rs = distances
    q_per_system = jnp.take(at_flat_q, at_codes)

    q_a = jnp.take(q_per_system, pair_idx_atom_atom[:, 0])
    q_b = jnp.take(q_per_system, pair_idx_atom_atom[:, 1])

    rm_a = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 0])
    rm_b = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 1])

    ep_a = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 0])
    ep_b = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 1])

    pair_qq = q_a * q_b
    pair_rm = rm_a + rm_b
    pair_ep = (ep_a * ep_b) ** 0.5

    def lennard_jones(r, sig, ep):
        """
        rmin = 2^(1/6) * sigma
            https://de.wikipedia.org/wiki/Lennard-Jones-Potential
        Lennard-Jones potential for a pair of atoms
        """
        a = 6
        b = 2
        # sig = sig / (2 ** (1 / 6))
        r6 = (sig / r) ** a
        return ep * (r6**b - 2 * r6)

    coulombs_constant = (
        3.32063711e2  # Coulomb's constant kappa = 1/(4*pi*e0) in kcal-Angstroms/e^2.
    )

    def coulomb(r, qq, constant=coulombs_constant):
        return constant * qq / r

    @jax.jit
    def apply_switching_function(
        positions: Array,  # Shape: (n_atoms, 3)
        pair_energies: Array,  # Shape: (n_pairs,)
        ml_cutoff_distance: float = 2.0,
        mm_switch_on: float = 5.0,
        mm_cutoff: float = 1.0,
        buffer_distance: float = 0.001,
    ) -> Array:
        """Applies smooth switching function to MM energies based on distances.

        Args:
            positions: Atomic positions
            pair_energies: Per-pair MM energies to be scaled
            ml_cutoff_distance: Distance where ML potential is cut off
            mm_switch_on: Distance where MM potential starts switching on
            mm_cutoff: Final cutoff for MM potential
            buffer_distance: Small buffer to avoid discontinuities

        Returns:
            Array: Scaled MM energies after applying switching function
        """
        # Calculate pairwise distances
        pair_positions = (
            positions[pair_idx_atom_atom[:, 0]] - positions[pair_idx_atom_atom[:, 1]]
        )
        distances = jnp.linalg.norm(pair_positions, axis=1)

        # Calculate switching functions
        ml_cutoff = mm_switch_on - ml_cutoff_distance
        switch_on = smooth_switch(distances, x0=ml_cutoff, x1=mm_switch_on)
        switch_off = 1 - smooth_switch(
            distances - mm_cutoff - mm_switch_on, x0=ml_cutoff, x1=mm_switch_on
        )
        cutoff = 1 - smooth_cutoff(distances, cutoff=2)

        # Combine switching functions and apply to energies
        switching_factor = switch_on * switch_off * cutoff
        scaled_energies = pair_energies * switching_factor

        return scaled_energies.sum()

    @jax.jit
    def calculate_mm_energy(positions: Array) -> Array:
        """Calculates MM energies including both VDW and electrostatic terms.

        Args:
            positions: Atomic positions (Shape: (n_atoms, 3))

        Returns:
            Array: Total MM energy
        """
        # Calculate pairwise distances
        displacements = (
            positions[pair_idx_atom_atom[:, 0]] - positions[pair_idx_atom_atom[:, 1]]
        )
        distances = jnp.linalg.norm(displacements, axis=1)

        # Only include interactions between unique pairs
        pair_mask = pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1]

        # Calculate VDW (Lennard-Jones) energies
        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
        vdw_total = vdw_energies.sum()

        # Calculate electrostatic energies
        electrostatic_energies = coulomb(distances, pair_qq) * pair_mask
        electrostatic_total = electrostatic_energies.sum()

        return vdw_total + electrostatic_total

    @jax.jit
    def calculate_mm_pair_energies(positions: Array) -> Array:
        """Calculates per-pair MM energies for switching calculations.

        Args:
            positions: Atomic positions (Shape: (n_atoms, 3))

        Returns:
            Array: Per-pair energies (Shape: (n_pairs,))
        """
        displacements = (
            positions[pair_idx_atom_atom[:, 0]] - positions[pair_idx_atom_atom[:, 1]]
        )
        distances = jnp.linalg.norm(displacements, axis=1)
        pair_mask = pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1]

        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
        electrostatic_energies = coulomb(distances, pair_qq) * pair_mask

        return vdw_energies + electrostatic_energies

    # Calculate gradients
    mm_energy_grad = jax.grad(calculate_mm_energy)
    switching_grad = jax.grad(apply_switching_function)

    @jax.jit
    def calculate_mm_energy_and_forces(
        positions: Array,  # Shape: (n_atoms, 3)
    ) -> Tuple[Array, Array]:
        """Calculates MM energy and forces with switching.

        Args:
            positions: Atomic positions

        Returns:
            Tuple[Array, Array]: (Total energy, Forces per atom)
        """
        # Calculate base MM energies
        mm_energy = calculate_mm_energy(positions)
        pair_energies = calculate_mm_pair_energies(positions)

        # Apply switching function
        switched_energy = apply_switching_function(positions, pair_energies)

        # Calculate forces with switching
        mm_forces = mm_energy_grad(positions)
        switching_forces = switching_grad(positions, pair_energies)
        total_forces = mm_forces * switching_forces

        return switched_energy, total_forces

    return calculate_mm_energy_and_forces


def prepare_batches_md(
    data,
    batch_size: int,
    data_keys=None,
    num_atoms: int = 60,
    dst_idx=None,
    src_idx=None,
    include_id: bool = False,
    debug_mode: bool = False,
):
    """
    Efficiently prepare batches for training.

    Args:
        key: JAX random key for shuffling.
        data (dict): Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size (int): Size of each batch.
        data_keys (list, optional): List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms (int, optional): Number of atoms per example. Default is 60.
        dst_idx (jax.numpy.ndarray, optional): Precomputed destination indices for atom pairs.
        src_idx (jax.numpy.ndarray, optional): Precomputed source indices for atom pairs.
        include_id (bool, optional): Whether to include 'id' key if present in data.
        debug_mode (bool, optional): If True, run assertions and extra checks.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """

    # -------------------------------------------------------------------------
    # Validation and Setup
    # -------------------------------------------------------------------------

    # Check for mandatory keys
    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    # Default to all keys in data if none provided
    if data_keys is None:
        data_keys = list(data.keys())

    # Verify data sizes
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    # -------------------------------------------------------------------------
    # Compute Random Permutation for Batches
    # -------------------------------------------------------------------------
    # perms = jax.random.permutation(key, data_size)
    perms = jnp.arange(0, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # -------------------------------------------------------------------------
    # Precompute Batch Segments and Indices
    # -------------------------------------------------------------------------
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    # Compute pairwise indices only if not provided
    # E3x: e3x.ops.sparse_pairwise_indices(num_atoms) -> returns (dst_idx, src_idx)
    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Adjust indices for batching
    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    # Centralize reshape logic
    # For keys not listed here, we default to their original shape after indexing.
    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size, 3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []

    # -------------------------------------------------------------------------
    # Batch Preparation Loop
    # -------------------------------------------------------------------------
    for perm in perms:
        # Build the batch dictionary
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][jnp.array(perm)]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                batch[k] = v

        # Optionally include 'id' if requested and present
        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][jnp.array(perm)]

        # Compute good_indices (mask for valid atom pairs)
        # Vectorized approach: We know N is shape (batch_size,)
        # Expand N to compare with dst_idx/src_idx
        # dst_idx[i], src_idx[i] range over atom pairs within the ith example
        # Condition: (dst_idx[i] < N[i]+i*num_atoms) & (src_idx[i] < N[i]+i*num_atoms)
        # We'll compute this for all i and concatenate.
        N = batch["N"]
        # Expand N and offsets for comparison
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

        # Add metadata to the batch
        atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
        batch.update(
            {
                "dst_idx": dst_idx.flatten(),
                "src_idx": src_idx.flatten(),
                "batch_mask": good_indices,
                "batch_segments": batch_segments,
                "atom_mask": atom_mask,
            }
        )

        # Debug checks
        if debug_mode:
            # Check expected shapes
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            assert batch["F"].shape == (
                batch_size * num_atoms,
                3,
            ), f"F shape mismatch: {batch['F'].shape}"
            assert batch["E"].shape == (
                batch_size,
                1,
            ), f"E shape mismatch: {batch['E'].shape}"
            assert batch["Z"].shape == (
                batch_size * num_atoms,
            ), f"Z shape mismatch: {batch['Z'].shape}"
            assert batch["N"].shape == (
                batch_size,
            ), f"N shape mismatch: {batch['N'].shape}"
            # Optional: print or log if needed

        output.append(batch)

    return output


# switch_MM_grad = jax.grad(switch_MM)


class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in kcal/mol
    forces: Array  # Shape: (n_atoms, 3), forces in kcal/mol/Å


def get_spherical_cutoff_calculator(
    atomic_numbers: Array,  # Shape: (n_atoms,)
    atomic_positions: Array,  # Shape: (n_atoms, 3)
    n_monomers: int,
    restart_path: str = "/path/to/default",
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    backprop: bool = False,
    debug: bool = False,
) -> Any:  # Returns ASE calculator
    """Creates a calculator that combines ML and MM potentials with spherical cutoffs.

    This calculator handles:
    1. ML predictions for close-range interactions
    2. MM calculations for long-range interactions
    3. Smooth switching between the two regimes

    Args:
        atomic_numbers: Array of atomic numbers for each atom
        atomic_positions: Initial positions of atoms in Angstroms
        restart_path: Path to model checkpoint for ML component

    Returns:
        ASE-compatible calculator that computes energies and forces
    """

    all_dimer_idxs = []
    for a, b in dimer_permutations(n_monomers):
        all_dimer_idxs.append(indices_of_pairs(a + 1, b + 1))

    all_monomer_idxs = []
    for a in range(1, n_monomers + 1):
        all_monomer_idxs.append(indices_of_monomer(a))

    unique_res_ids = []
    collect_monomers = []
    dimer_perms = dimer_permutations(n_monomers)
    for i, _ in enumerate(dimer_perms):
        a, b = _
        if a not in unique_res_ids and b not in unique_res_ids:
            unique_res_ids.append(a)
            unique_res_ids.append(b)
            collect_monomers.append(1)
            print(a, b)
        else:
            collect_monomers.append(0)

    restart_path = Path(
        "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-7c36e6f9-6f10-4d21-bf6d-693df9b8cd40"
    )

    """Initialize monomer and dimer models from restart"""
    restart = get_last(restart_path)

    # Setup monomer model
    params, MODEL = get_params_model(restart)
    MODEL.natoms = 10
    # MODEL.charges = False

    def calc_dimer_energy_forces(R, Z, i, ml_e, ml_f):
        a, b = dimer_perms[i]
        a, b = all_monomer_idxs[a], all_monomer_idxs[b]
        idxs = np.array([a, b], dtype=int).flatten()
        # print(idxs)
        _R = R[idxs]
        # print(_R)
        final_energy = ml_e
        val_ml_s = switch_ML(_R, final_energy)  # ML switching value
        grad_ml_s = switch_ML_grad(_R, final_energy)  # ML switching gradient
        # Combine forces with switching functions
        ml_forces_out = ml_f * -grad_ml_s  # ml_f * val_ml_s + grad_ml_s * final_energy
        # final_forces = ml_f + grad_ml_s
        # Combine all force contributions for final forces
        # final_forces = ml_f + grad_ml_s #ml_forces_out #+ mm_forces_out #+ ase_dimers_1body_forces

        outdict = {
            "energy": val_ml_s,
            "forces": ml_forces_out,
        }
        return outdict

    MM_energy_and_gradient = get_MM_energy_forces_fns(atomic_positions)

    def get_energy_fn(
        atomic_numbers: Array,  # Shape: (n_atoms,)
        positions: Array,  # Shape: (n_atoms, 3)
    ) -> Tuple[Any, Dict[str, Array]]:
        """Prepares the ML model and batching for energy calculations.

        Args:
            atomic_numbers: Array of atomic numbers
            positions: Atomic positions in Angstroms

        Returns:
            Tuple of (model_apply_fn, batched_inputs)
        """
        batch_data: Dict[str, Array] = {}

        # Prepare monomer data
        n_monomers = len(all_monomer_idxs)
        # Position of the atoms in the monomer
        monomer_positions = jnp.zeros((n_monomers, MAX_ATOMS_PER_SYSTEM, SPATIAL_DIMS))
        monomer_positions = monomer_positions.at[:, :ATOMS_PER_MONOMER].set(
            positions[jnp.array(all_monomer_idxs)]
        )
        # Atomic numbers of the atoms in the monomer
        monomer_atomic = jnp.zeros((n_monomers, MAX_ATOMS_PER_SYSTEM), dtype=jnp.int32)
        monomer_atomic = monomer_atomic.at[:, :ATOMS_PER_MONOMER].set(
            atomic_numbers[jnp.array(all_monomer_idxs)]
        )

        # Prepare dimer data
        n_dimers = len(all_dimer_idxs)
        # Position of the atoms in the dimer
        dimer_positions = jnp.zeros((n_dimers, MAX_ATOMS_PER_SYSTEM, SPATIAL_DIMS))
        dimer_positions = dimer_positions.at[:].set(
            positions[jnp.array(all_dimer_idxs)]
        )
        # Atomic numbers of the atoms in the dimer
        dimer_atomic = jnp.zeros((n_dimers, MAX_ATOMS_PER_SYSTEM), dtype=jnp.int32)
        dimer_atomic = dimer_atomic.at[:].set(atomic_numbers[jnp.array(all_dimer_idxs)])

        # Combine monomer and dimer data
        batch_data["R"] = jnp.concatenate([monomer_positions, dimer_positions])
        batch_data["Z"] = jnp.concatenate([monomer_atomic, dimer_atomic])
        batch_data["N"] = jnp.concatenate(
            [
                jnp.full((n_monomers,), ATOMS_PER_MONOMER),
                jnp.full((n_dimers,), MAX_ATOMS_PER_SYSTEM),
            ]
        )

        batches = prepare_batches_md(
            batch_data, batch_size=BATCH_SIZE, num_atoms=MAX_ATOMS_PER_SYSTEM
        )[0]

        @jax.jit
        def apply_model(
            atomic_numbers: Array,  # Shape: (batch_size * num_atoms,)
            positions: Array,  # Shape: (batch_size * num_atoms, 3)
        ) -> Dict[str, Array]:
            """Applies the ML model to batched inputs.

            Args:
                atomic_numbers: Batched atomic numbers
                positions: Batched atomic positions

            Returns:
                Dictionary containing 'energy' and 'forces'
            """
            return MODEL.apply(
                params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                dst_idx=batches["dst_idx"],
                src_idx=batches["src_idx"],
                batch_segments=batches["batch_segments"],
                batch_size=BATCH_SIZE,
                batch_mask=batches["batch_mask"],
                atom_mask=batches["atom_mask"],
            )

        return apply_model, batches

    @jax.jit
    def spherical_cutoff_calculator(
        positions: Array,  # Shape: (n_atoms, 3)
        atomic_numbers: Array,  # Shape: (n_atoms,)
    ) -> ModelOutput:
        """Calculates energy and forces using combined ML/MM potential.

        Handles:
        1. ML predictions for each monomer and dimer
        2. MM long-range interactions
        3. Smooth switching between regimes

        Args:
            positions: Atomic positions in Angstroms
            atomic_numbers: Atomic numbers of each atom

        Returns:
            ModelOutput containing total energy and forces
        """
        n_monomers = 20
        n_dimers = 190
        output_list: List[Dict[str, Array]] = []
        out_E = 0
        out_F = 0

        if doML:
            # print("doML")
            apply_model, batches = get_energy_fn(atomic_numbers, positions)

            output = apply_model(batches["Z"], batches["R"])

            f = output["forces"] / (ase.units.kcal / ase.units.mol)
            e = output["energy"] / (ase.units.kcal / ase.units.mol)

            # energies from a batch of monomers and dimers
            ml_monomer_energy = jnp.array(e[:n_monomers]).flatten()

            # forces from a batch of monomers and dimers
            monomer_idx_max = MAX_ATOMS_PER_SYSTEM * n_monomers
            dimer_idx_max = MAX_ATOMS_PER_SYSTEM * n_dimers + monomer_idx_max

            ml_monomer_forces = f[:monomer_idx_max]
            ml_dimer_forces = f[monomer_idx_max:dimer_idx_max]

            monomer_segment_idxs = jnp.concatenate(
                [
                    jnp.arange(ATOMS_PER_MONOMER) + i * ATOMS_PER_MONOMER
                    for i in range(n_monomers)
                ]
            )

            # Ensure monomer forces are properly shaped and masked
            monomer_forces = ml_monomer_forces.reshape(
                n_monomers, MAX_ATOMS_PER_SYSTEM, 3
            )
            atom_mask = jnp.arange(MAX_ATOMS_PER_SYSTEM)[None, :] < ATOMS_PER_MONOMER

            # Apply mask and reshape
            monomer_forces = jnp.where(atom_mask[..., None], monomer_forces, 0.0)

            # Sum forces for valid atoms only
            out_F += jax.ops.segment_sum(
                monomer_forces[:, :ATOMS_PER_MONOMER].reshape(-1, 3),
                monomer_segment_idxs,
                num_segments=n_monomers * ATOMS_PER_MONOMER,
            )

            out_E += ml_monomer_energy.sum()

            if debug:
                print("doML")
                print("monomer_segment_idxs", monomer_segment_idxs)
                jax.debug.print("monomer_segment_idxs\n{x}", x=monomer_segment_idxs)
                print("atom_mask", atom_mask)
                jax.debug.print("atom_mask\n{x}", x=atom_mask)
                print("monomer_forces", monomer_forces)
                jax.debug.print("monomer_forces\n{x}", x=monomer_forces)
                print("f", f.shape)
                print("e", e.shape)
                print("n_monomers", n_monomers)
                print("N_ATOMS_MONOMER", ATOMS_PER_MONOMER)
                print("ml_monomer_energy", ml_monomer_energy.shape)
                print("ml_dimer_energy", ml_dimer_energy.shape)
                print("monomer_segment_idxs", monomer_segment_idxs.shape)
                jax.debug.print("monomer_segment_idxs\n{x}", x=monomer_segment_idxs)
                print("ml_monomer_forces", ml_monomer_forces.shape)
                print("ml_dimer_forces", ml_dimer_forces.shape)
                jax.debug.print("ml_monomer_forces\n{x}", x=ml_monomer_forces)
                jax.debug.print("ml_dimer_forces\n{x}", x=ml_dimer_forces)
                print("monomer_idx_max", monomer_idx_max)
                print("dimer_idx_max", dimer_idx_max)
                print("ml_monomer_forces_sum", out_F.shape)
                jax.debug.print("out_F\n{x}", x=out_F)

            if doML_dimer:

                ml_dimer_energy = jnp.array(
                    e[n_monomers:]
                ).flatten()  # shape (n_dimers)
                # Create segment indices for dimers
                dimer_pairs = jnp.array(dimer_perms)
                # Calculate base indices for each monomer in the dimers
                first_monomer_indices = (
                    ATOMS_PER_MONOMER * dimer_pairs[:, 0:1]
                )  # Shape: (n_dimers, 1)
                second_monomer_indices = (
                    ATOMS_PER_MONOMER * dimer_pairs[:, 1:2]
                )  # Shape: (n_dimers, 1)
                # Create atom offsets for each monomer
                atom_offsets = jnp.arange(
                    ATOMS_PER_MONOMER
                )  # Shape: (ATOMS_PER_MONOMER,)

                monomer_contrib_to_dimer_energy = (
                    ml_monomer_energy[dimer_pairs[:, 0]]
                    + ml_monomer_energy[dimer_pairs[:, 1]]
                )
                dimer_int_energies = ml_dimer_energy - monomer_contrib_to_dimer_energy

                # Combine indices for both monomers in each dimer
                force_segments = jnp.concatenate(
                    [
                        first_monomer_indices
                        + atom_offsets[None, :],  # Add offsets to first monomer
                        second_monomer_indices
                        + atom_offsets[None, :],  # Add offsets to second monomer
                    ],
                    axis=1,
                )  # Shape: (n_dimers, 2*ATOMS_PER_MONOMER)
                # Flatten the segments
                force_segments = force_segments.reshape(
                    -1
                )  # Shape: (n_dimers * 2*ATOMS_PER_MONOMER)
                # Create validity mask for the segments
                valid_segments = (force_segments >= 0) & (
                    force_segments < n_monomers * ATOMS_PER_MONOMER
                )
                # Zero out invalid segments
                force_segments = jnp.where(valid_segments, force_segments, 0)

                # Calculate interaction forces
                dimer_int_forces = ml_dimer_forces.reshape(
                    n_dimers, MAX_ATOMS_PER_SYSTEM, 3
                )

                switched_energy = jax.vmap(
                    lambda x, f: switch_ML(x.reshape(MAX_ATOMS_PER_SYSTEM, 3), f)
                )(positions[jnp.array(all_dimer_idxs)], dimer_int_energies)
                switched_energy_grad = jax.vmap(
                    lambda x, f: switch_ML_grad(x.reshape(MAX_ATOMS_PER_SYSTEM, 3), f)
                )(positions[jnp.array(all_dimer_idxs)], dimer_int_energies)

                # Create atom existence mask
                atom_mask = (
                    jnp.arange(MAX_ATOMS_PER_SYSTEM)[None, :] < ATOMS_PER_MONOMER
                )

                original_dimer_int_energies = dimer_int_energies.sum()
                summed_switched_dimer_int_energies = switched_energy.sum()

                # Perform segmented sum with validated indices and forces
                summed_switched_dimer_int_forces = jax.ops.segment_sum(
                    switched_energy_grad.reshape(-1, 3),
                    force_segments,
                    num_segments=n_monomers * ATOMS_PER_MONOMER,
                )
                summed_dimer_int_forces = jax.ops.segment_sum(
                    dimer_int_forces.reshape(-1, 3),
                    force_segments,
                    num_segments=n_monomers * ATOMS_PER_MONOMER,
                )
                # combine with product rule
                # d(f1*f2)/dx = f1*df2/dx + f2*df1/dx
                dudx_v = original_dimer_int_energies * summed_switched_dimer_int_forces
                dvdx_u = summed_dimer_int_forces * summed_switched_dimer_int_energies
                combined_forces = dudx_v + dvdx_u

                out_E += summed_switched_dimer_int_energies.sum()
                out_F += combined_forces

                if debug:
                    print("doML_dimer")
                    print("switched_forces", switched_forces.shape)
                    jax.debug.print("switched_forces\n{x}", x=switched_forces)
                    print("switched_energies", dimer_int_energies.shape)
                    jax.debug.print("switched_energies\n{x}", x=dimer_int_energies)
                    print("dimer_int_forces", dimer_int_forces.shape)
                    jax.debug.print("dimer_int_forces\n{x}", x=dimer_int_forces)
                    print("monomer_forces_reshaped", monomer_forces_reshaped.shape)
                    print("ml_monomer_forces", ml_monomer_forces.shape)
                    jax.debug.print(
                        "monomer_forces_reshaped\n{x}", x=monomer_forces_reshaped
                    )
                    jax.debug.print("ml_monomer_forces\n{x}", x=ml_monomer_forces)
                    print("force_segments", force_segments.shape)
                    jax.debug.print("force_segments\n{x}", x=force_segments)
                    print("dimer_int_energies", dimer_int_energies.shape)
                    jax.debug.print("dimer_int_energies\n{x}", x=dimer_int_energies)

        if doMM:
            # print("doMM")
            # MM energy and forces
            mm_E, mm_grad = MM_energy_and_gradient(positions)
            out_E += mm_E
            out_F += mm_grad
            if debug:
                print("doMM")
                print("mm_E", mm_E.shape)
                jax.debug.print("mm_E\n{x}", x=mm_E)
                print("mm_grad", mm_grad.shape)
                jax.debug.print("mm_grad\n{x}", x=mm_grad)

        return ModelOutput(energy=out_E.sum(), forces=-out_F)

    def just_E(R, Z):
        return spherical_cutoff_calculator(R, Z).energy

    just_E_grad = jax.grad(just_E)

    class AseDimerCalculator(ase_calc.Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms,
            properties,
            system_changes=ase.calculators.calculator.all_changes,
        ):
            ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
            R = atoms.get_positions()
            Z = atoms.get_atomic_numbers()

            if backprop:
                E = just_E(R, Z)
                F = -just_E_grad(R, Z)
            else:
                out = spherical_cutoff_calculator(R, Z)
                E = out.energy
                F = out.forces

            self.results["energy"] = E * (ase.units.kcal / ase.units.mol)
            self.results["forces"] = F * (ase.units.kcal / ase.units.mol)

    return AseDimerCalculator()


# def validate_forces(forces):
#     """Validate forces and replace NaNs with zeros."""
#     is_valid = jnp.isfinite(forces).all()
#     if not is_valid:
#         print("Warning: Found invalid forces")
#         forces = jnp.nan_to_num(forces, 0.0)
#     return forces

# # Use in spherical_cutoff_calculator
# out_F = validate_forces(out_F)
