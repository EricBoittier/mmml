from itertools import combinations, permutations, product
from typing import Dict, Tuple, List, Any, NamedTuple

import jax
# If you want to perform simulations in float64 you have to call this before any JAX compuation
# jax.config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
from jax import Array
import os

os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

import mmml.physnetjax.physnetjax

from mmml.physnetjax.physnetjax.data.data import prepare_datasets
from mmml.physnetjax.physnetjax.training.loss import dipole_calc
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.training import train_model  # from model import dipole_calc
from mmml.physnetjax.physnetjax.data.batches import (
    _prepare_batches as prepare_batches,
)  # prepare_batches, prepare_datasets

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

from pathlib import Path

from physnetjax.calc.helper_mlp import get_ase_calc


from physnetjax.data.read_ase import save_traj_to_npz

import ase
from ase.visualize import view

def view_atoms(a):
    return view(a, viewer="x3d")

from ase.io import read

# import pycharmm

# import pycharmm
# import pycharmm.generate as gen
# import pycharmm.ic as ic
# import pycharmm.coor as coor
# import pycharmm.energy as energy
# import pycharmm.dynamics as dyn
# import pycharmm.nbonds as nbonds
# import pycharmm.minimize as minimize
# import pycharmm.crystal as crystal
# import pycharmm.image as image
# import pycharmm.psf as psf
# import pycharmm.read as read
# import pycharmm.write as write
# import pycharmm.settings as settings
# import pycharmm.cons_harm as cons_harm
# import pycharmm.cons_fix as cons_fix
# import pycharmm.select as select
# import pycharmm.shake as shake

# from pycharmm.lib import charmm as libcharmm




# # # Read in the topology (rtf) and parameter file (prm) for proteins
# # # equivalent to the CHARMM scripting command: read rtf card name toppar/top_all36_prot.rtf
# # read.rtf('../toppar/top_all36_prot.rtf')
# # # equivalent to the CHARMM scripting command: read param card flexible name toppar/par_all36m_prot.prm
# # read.prm('../toppar/par_all36m_prot.prm', flex=True)

# # stream in the water/ions parameter using the pycharmm.lingo module
# # # equivalent to the CHARMM scripting command: stream toppar/toppar_water_ions.str
# # pycharmm.lingo.charmm_script('stream ../toppar/toppar_water_ions.str')
# # # end toppar/toppar_water_ions.str



# def generate_residue(resid) -> None:
#     """Generates a residue from the RTF file"""
#     print("*"*5, "Generating residue", "*"*5)
#     s="""DELETE ATOM SELE ALL END"""
#     pycharmm.lingo.charmm_script(s)
#     read.rtf('/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf')
#     bl =settings.set_bomb_level(-2)
#     wl =settings.set_warn_level(-2)
#     read.prm('/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm')
#     settings.set_bomb_level(bl)
#     settings.set_warn_level(wl)
#     pycharmm.lingo.charmm_script('bomlev 0')
#     read.sequence_string(resid)
#     gen.new_segment(seg_name=resid,
#                     setup_ic=True)
#     ic.prm_fill(replace_all=True)


# generate_residue('ACEH ACEH')

# # read.sequence_string('ACEH ACEH')
# # # equivalent to the CHARMM scripting command: generate ADP first ACE last CT3 setup
# # gen.new_segment(seg_name='ACEH', first_patch="ACEH", last_patch="ACEH", setup_ic=True)
# # # equivalent to the CHARMM scripting command: ic param
# # ic.prm_fill(replace_all=False)
# # ic.build()
# # coor.get_natom()


# # In[4]:


# at_codes = np.array(psf.get_iac())
# at_codes


# # In[5]:


# np.array(psf.get_atype())


# # In[6]:


# # atypes = psf.get_atype()
# # atc = pycharmm.param.get_atc()
# # residues = psf.get_res()
# # psf.get_natom()
# # coor.get_positions()

# # cgenff_params = open("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf").readlines()
# # cgenff_params_dict_q = {}
# # atom_name_to_param = {k: [] for k in atc}

# # for _ in cgenff_params:
# #     if _.startswith("ATOM"):
# #         _, atomname, at, q = _.split()[:4]
# #         try:
# #             cgenff_params_dict_q[at] = float(q)
# #         except:
# #             cgenff_params_dict_q[at] = float(q.split("!")[0])
# #         atom_name_to_param[atomname] = at

# # cgenff_params = open("/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm").readlines()
# # cgenff_params_dict = {}

# # for _ in cgenff_params:
# #     if len(_) > 5 and len(_.split()) > 4 and _.split()[1] == "0.0":
# #         res, _, ep, sig = _.split()[:4]
# #         if res in atc:
# #             cgenff_params_dict[res] = (float(ep), float(sig))

# # atc_epsilons = [cgenff_params_dict[_][0] for _ in atc]
# # atc_rmins = [cgenff_params_dict[_][1] for _ in atc]


# # In[7]:


# # atc


# # In[8]:


# # atc_epsilons


# # In[9]:


# # cgenff_params_dict, cgenff_params_dict_q


# # In[10]:


# dimers_path = Path("/pchem-data/meuwly/boittier/home/asecalcs/aceh/dimers")
# dimers = list(dimers_path.glob("*.traj"))


# # In[11]:


# atoms = ase.io.read(dimers[-1])

# e = atoms.get_potential_energy()


# for _ in dimers:
#     fn = str(_.stem) + ".npz"
#     print(fn)
#     save_traj_to_npz(_, fn)


# testData2 = np.load("sim_t_298.15_k_rho_1044.6_kgperm3_p101.325_kPa.npz", allow_pickle=True)
# testData1 = np.load("sim_t_293.15_k_rho_1044.3_kgperm3_pNone_kPa.npz", allow_pickle=True)

# from physnetjax.data.datasets import process_in_memory

# # process_in_memory(test, max_atoms=100)


# # In[17]:


# NATOMS = 16

# model = EF(
#     # attributes
#     features=64,
#     max_degree=1,
#     num_iterations=3,
#     num_basis_functions=100,
#     cutoff=6.0,
#     max_atomic_number=18,
#     charges=False,
#     natoms=NATOMS,
#     total_charge=0,
#     n_res=4,
#     zbl=False,
#     debug=False,
# )


# # In[18]:


# energy.show()


# # In[19]:


# view_atoms(atoms)


# # In[20]:


# atoms.has("energy")

# # -0.498232909223 * ase.units.Hartree


# # In[24]:


# testData = np.load("sim_t_298.15_k_rho_1044.6_kgperm3_p101.325_kPa.npz", allow_pickle=True)
# testData = np.load("sim_t_298.15_k_rho_1044.6_kgperm3_p101.325_kPa.npz", allow_pickle=True)


# # In[25]:


# Eref = np.zeros([20], dtype=float)
# Eref[1] = -0.498232909223
# Eref[6] = -37.731440432799
# Eref[8] = -74.878159582108
# Eref[17] = -459.549260062932
# # Eref[testData["Z"]].sum() * ase.units.Hartree


# # In[26]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# shiftE = testData["E"] - Eref[testData["Z"]].sum() * ase.units.Hartree
# sns.histplot(shiftE)
# plt.show()
# sns.histplot(testData["F"].flatten())
# xyz = pd.DataFrame(test["R"][0], columns=["x", "y", "z"])
# coor.set_positions(xyz)


# def clean_data(testData):
#     testDataDict = dict(testData)
#     testDataDict["E"] = testData["E"] - Eref[testData["Z"]].sum() * ase.units.Hartree
#     testDataDict["Z"] = np.array([testData["Z"] for _ in range(testData["R"].shape[0])])
#     data_ = {k:v for k,v in testDataDict.items() if k in ["R", "Z", "N", "E", "F"]}
#     testDataDict.keys(), data_.keys()
#     for _ in data_.keys():
#         print(_, data_[_].shape)
#     return data_


# # In[29]:


# trainingData = clean_data(testData)
# testData = clean_data(testData2)


# # In[ ]:





# # In[30]:


# ntest = 5000
# test_data = {k: v[ntest:] for k, v in testData.items()}
# valid_data = {k: v[:ntest] for k, v in testData.items()}


# # In[31]:


# test_data["Z"]


# # In[32]:


# restart = None
# restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-195504b5-9caf-466e-b3a8-4164c5bd514d"
# RESTART_ML = False
# TRAIN_ML = True
# ANALYSE_ML = False

# if RESTART_ML:
#     restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-d17aaa54-65e1-415e-94ae-980521fcd2b1"


# if TRAIN_ML:
#     # train_data, valid_data = prepare_datasets(
#     #     data_key,
#     #     8000,
#     #      2000,
#     #     ["sim_t_298.15_k_rho_1044.6_kgperm3_p101.325_kPa.npz"],
#     #     clip_esp=False,
#     #     natoms=16,
#     #     clean=False,
#     #     subtract_atom_energies=False,
#     #     verbose=True,
#     # )

#     # ntest = len(valid_data["E"]) // 2


#     DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
#     batch_size = 64

#     # test_batches = prepare_batches(data_key, test_data, batch_size,
#     #                               num_atoms=NATOMS,
#     #                               data_keys=DEFAULT_DATA_KEYS)


#     params = train_model(
#         train_key,
#         model,
#         trainingData,
#         valid_data,
#         num_epochs=int(1e4),
#         learning_rate=0.002,
#         energy_weight=1,
#         dipole_weight=1,
#         charges_weight=1,
#         forces_weight=50,
#         schedule_fn="constant",
#         optimizer="amsgrad",
#         batch_size=batch_size,
#         num_atoms=NATOMS,
#         data_keys=DEFAULT_DATA_KEYS,
#         # restart=restart,
#         name="dichloromethane",
#         print_freq=1,
#         objective="valid_loss",
#         best=1e6,
#         batch_method="default",
#     )

# if ANALYSE_ML:
#     output = plot_stats(test_batches, model, params, _set="Test",
#                    do_kde=True, batch_size=batch_size)


# # In[36]:


# from physnetjax.restart.restart import get_last, get_files, get_params_model
# from physnetjax.analysis.analysis import plot_stats
# # restart = get_last()
# restart = get_last("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-a8610063-0ef3-4e8c-8750-147d7696a58a")

# params, dimer_model = get_params_model(restart)
# dimer_model.natoms = 16


# # In[37]:


# from physnetjax.calc.helper_mlp import get_ase_calc
# z = testData["Z"][0]
# z.shape


# # In[38]:


# atoms = ase.Atoms(z, test_data["R"][0])
# atoms


# # In[39]:


# view_atoms(atoms)


# # In[42]:


# ase_calc = get_ase_calc(params, dimer_model, atoms)


# # In[43]:


# atoms.calc = ase_calc


# # In[ ]:





# # In[44]:


# def get_EF(i):
#     atoms.set_positions(test_data["R"][i])
#     E = atoms.get_potential_energy()
#     F = atoms.get_forces()
#     outdict = {"Epred": E}
#     for j, f in enumerate(F.flatten()):
#         _ = test_data["Z"][i][j//3]
#         outdict[f"Fpred_{_}_{j//3}_{j%3}"] = f
#     return  outdict


# # In[45]:


# testData["E"].shape


# # In[46]:


# # predE = [get_E(i) for i in range(len(test_data["R"]))]
# preds = [get_EF(i) for i in range(len(test_data["R"]))]


# # In[47]:




# epredkey = "E" #"$E_{\rm pred}$"
# ekey = "Epred" #"$E$"
# dfE = pd.DataFrame(preds)
# dfE["E"] = test_data["E"]
# conv = 1/(ase.units.kcal/ase.units.mol)
# for _ in dfE.columns:
#     dfE[_] = np.array(dfE[_], dtype=np.float64) * conv

# predF = dfE[dfE.columns[1:-1]].to_numpy().reshape(test_data["F"].shape)
# testF = test_data["F"] * conv

# dfF = pd.DataFrame({"F": testF.flatten() , "Fpred": predF.flatten() })
# fig, ax = plt.subplots(1,2, figsize=(10,17))
# reg_plot(ax[0], dfE, "E", "Epred")
# reg_plot(ax[1], dfF,  "F", "Fpred")
# ax[0].set_title("Energy [kcal/mol]")
# ax[1].set_title("Forces [(kcal/mol)/$\\mathrm{\\AA}$]")
# plt.subplots_adjust(wspace=0.25)


# (dfE["E"] - dfE["Epred"]).mean()


# from jax_md import partition
# from jax_md import space
# import jax.numpy as np
# from jax import random
# from jax import jit
# from jax import lax
# from jax import ops

# import time

# from jax_md import space, smap, energy, minimize, quantity, simulate

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

# import jax
# import jax.numpy as jnp
# import jax_md
# import numpy as np

# from ase.io import read as ase_read
# from jax_md import units
# from typing import Dict

# # from so3lr import to_jax_md
# # from so3lr import So3lrPotential

# import time
# from jax_md import minimize


# # In[55]:


# @jax.jit
# def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
#     return model.apply(
#         params,
#         atomic_numbers=atomic_numbers,
#         positions=positions,
#         dst_idx=dst_idx,
#         src_idx=src_idx,
#     )
# TESTIDX = 0
# dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
# atomic_numbers = test_data["Z"][TESTIDX]
# position = R = test_data["R"][TESTIDX]
# @jit
# def jax_md_energy_fn(position, **kwargs):
#   l_nbrs = nbrs.update(jnp.array(position))
#   _ = evaluate_energies_and_forces(
#         atomic_numbers=atomic_numbers,
#         positions=position,
#         dst_idx=dst_idx,
#         src_idx=src_idx,
#     )
#   return _["energy"].reshape(-1)[0]
# jax_md_grad_fn = jax.grad(jax_md_energy_fn)
# BOXSIZE = 100
# # displacement, shift = space.periodic(BOXSIZE, wrapped=False)
# displacement, shift = space.free()
# neighbor_fn = partition.neighbor_list(displacement, None, 30/2, format=partition.Sparse)
# nbrs = neighbor_fn.allocate(R)
# unwrapped_init_fn, unwrapped_step_fn = minimize.fire_descent(jax_md_energy_fn, shift, dt_start=0.001, dt_max=0.001)
# unwrapped_step_fn = jit(unwrapped_step_fn)
# jax_md_energy_fn(R), jax_md_grad_fn(R)



# @jit
# def sim(state, nbrs):
#   def step(i, state_nbrs):
#     state, nbrs = state_nbrs
#     nbrs = nbrs.update(state.position)
#     return apply_fn(state, neighbor=nbrs), nbrs
#   return lax.fori_loop(0, steps_per_recording, step, (state, nbrs))


# # In[56]:


# TESTIDX = 1
# data_key
# test_data["E"].shape
# Ecatch = test_data["E"].min() * 1.05
# steps_per_recording = 25

# K_B = 8.617e-5
# dt = 1e-4
# T = 300
# kT = K_B * T
# init_fn, apply_fn = simulate.nvt_nose_hoover(jax_md_energy_fn, shift, dt, kT)
# apply_fn = jit(apply_fn)

# def run_sim(TESTIDX, Ecatch, nbrs, TFACT = 5, total_steps = 100000, steps_per_recording = 25):

#     key = data_key * TESTIDX

#     # Define the simulation.

#     total_records = total_steps // steps_per_recording

#     Si_mass = 2.91086E-3

#     try:
#         del fire_state
#     except NameError:
#         pass

#     fire_state = unwrapped_init_fn(R - R.T.mean(axis=1).T)
#     fire_positions = []

#     N = 1000
#     print("*"*10)
#     print("Minimization")
#     print("*"*10)
#     for i in range(N):
#       fire_positions += [fire_state.position]
#       fire_state = jit(unwrapped_step_fn)(fire_state)
#       if (i) % int(N//10) == 0:
#           print(i, "/",
#                 N,
#                 float(jax_md_energy_fn(fire_state.position)),
#                 float(np.abs(np.array(jax_md_grad_fn(fire_state.position))).max()))

#     state = init_fn(key, fire_state.position, Si_mass, neighbor=nbrs)
#     nhc_positions = []


#     print("*"*10)
#     print("NVT")
#     print("*"*10)
#     # Run the simulation.
#     print('\t\tEnergy (eV)\tTemperature (K)')
#     for i in range(total_records):
#       state, nbrs = sim(state, nbrs)
#       nhc_positions += [state.position]
#       if i % 100 == 0:
#         iT = float(quantity.temperature(momentum=state.momentum, mass=Si_mass) / K_B)
#         iE = float(jax_md_energy_fn(state.position, neighbor=nbrs))
#         print(i*steps_per_recording*dt, "ps", 100*i/total_records, "% ={ " , '{:.02f}\t\t\t{:.02f}'.format(
#             iE , iT))
#       if iT > T*TFACT:
#           print("ERROR! bailing!")
#           print("T", iT, T*TFACT, "E",  iE, Ecatch)
#           break
#       if iE < Ecatch:
#           print("ERROR! bailing!")
#           print("T", iT, T*TFACT, "E",  iE, Ecatch)
#           break

#     print(f"Simulated (NVT, NHC) {i} steps at dt {dt * 1000} (fs)")
#     nhc_positions = np.stack(nhc_positions)

#     # show_atoms(fire_positions[-1:], atoms=atoms)
#     # show_atoms(nhc_positions[:], atoms=atoms)
#     # show_atoms(nhc_positions[:-50][-2:] - nhc_positions[:-50][-2:].T.mean(axis=1).mean(axis=1), atoms=atoms)
#     return i*steps_per_recording, nhc_positions



# out_positions = []
# max_is = []
# for i in range(len(test_data["Z"])//40):
#     print("test data", i)
#     mi, pos = run_sim(i, Ecatch, nbrs)[:5]
#     out_positions.append(pos)
#     max_is.append(mi)
# # np.array(out_positions)
# problem_structures = np.concatenate(out_positions)[10::111]
# trajectory = Trajectory("test.traj", "a")
# for R in problem_structures:
#     atoms.set_positions(R)
#     trajectory.write( atoms)
# trajectory.close()



# import ase.io as ase_io
# from ase.io import Trajectory

# from ase.calculators.calculator import Calculator, all_changes
# from ase.units import Ha, Bohr, Debye
# # from pyscf.prop.polarizability.uhf import polarizability, Polarizability
# from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
# import jsonpickle

# import sys
# # "./lig_3_1_0_1000_1.traj" $x --output "lig_3_1_0_1000_1"$x"_.traj"
# # Simulate passing command-line arguments
# fn = "test.traj"
# fn2 = "test_out.traj"
# sys.argv = [ "file.py", f"./{fn}", 0, "--output", f"./{fn2}"]

# import pyscf
# import time
# import argparse
# from pyscf import lib
# from gpu4pyscf import dft
# import numpy as np
# from ase import Atoms
# from ase.optimize import LBFGS
# # from pyscf4gpuInterface.aseInterface import PYSCF, parameters
# import argparse
# parser = argparse.ArgumentParser(description='Run PySCF calculations on structures from an ASE trajectory')
# parser.add_argument('trajectory', help='Path to ASE trajectory file')
# parser.add_argument("index", type=int, default=0)
# parser.add_argument('--output', default='output.traj',
#                    help='Output trajectory file (default: output.traj)')
# parser.add_argument('--method', choices=['dft', 'hf'], default='dft',
#                    help='Calculation method (default: dft)')
# parser.add_argument('--basis', default='cc-pVTZ',
#                    help='Basis set (default: cc-pVTZ)')
# parser.add_argument('--xc', default='wB97m-v',
#                    help='Exchange-correlation functional for DFT (default: wB97m-v)')
# parser.add_argument('--auxbasis', default='def2-tzvp-jkfit',
#                    help='Auxiliary basis set for density fitting (default: def2-tzvp-jkfit)')
# parser.add_argument('--solvent', choices=['cosmo', 'pcm'],
#                    help='Solvent model (optional)')
# parser.add_argument('--verbose', type=int, default=1,
#                    help='Verbosity level (default: 1)')

# args = parser.parse_args()

# sys.path.append("/pchem-data/meuwly/boittier/home/mmml")
# import mmml
# from mmml.pyscf4gpuInterface.aseInterface import *

# # Read trajectory
# from ase.io import read, write
# atoms_list = read(args.trajectory, f'{args.index}')
# if not isinstance(atoms_list, list):
#     atoms_list = [atoms_list]

# # Process each structure and write to trajectory
# print(f"\nProcessing {len(atoms_list)} structures using {args.method.upper()}/{args.basis}")

# for i, atoms in enumerate(atoms_list):
#     # Create fresh calculator for each structure
#     if args.method == 'dft':
#         mol = pyscf.M(atom=atoms_from_ase(atoms),
#                      basis=args.basis,
#                      spin=0,
#                      charge=0)
#         mf = dft.RKS(mol, xc=args.xc)
#     else:
#         mol = pyscf.M(atom=atoms_from_ase(atoms),
#                      basis=args.basis,
#                      spin=0,
#                      charge=0)
#         mf = mol.HF()
#         mf.verbose = args.verbose

#     # Set up parameters
#     p = parameters()
#     p.mode = args.method
#     p.basis = args.basis
#     p.verbose = args.verbose
#     if args.method == 'dft':
#         p.xc = args.xc
#         p.auxbasis = args.auxbasis
#         p.solvent = args.solvent

#     calc = PYSCF(mf=mf, p=p)
#     atoms.calc = calc

#     # Run calculation
#     energy = atoms.get_potential_energy()
#     do_forces = True
#     if do_forces:
#         forces = atoms.get_forces()
#     print(f"\nStructure {i+1}:")
#     print(f"Energy: {energy:.6f} eV")
#     if do_forces:
#         print(f"Forces (eV/Å):\n{forces}")

#     # Write structure to trajectory
#     if i == 0:
#         print(f"Writing structure {i+1} to trajectory")
#         write(args.output, atoms, format='traj')
#     else:
#         print(f"Appending structure {i+1} to trajectory")
#         write(args.output, atoms, format='traj', append=True)

#     # Force garbage collection
#     import gc
#     gc.collect()

# print(f"\nCalculation results written to: {args.output}")
# view_atoms(atoms)
# show_atoms(out_positions[1], atoms=atoms)

