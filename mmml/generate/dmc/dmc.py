#!/usr/bin/env python3
"""Diffusion Monte Carlo driver using PhysNetJax energies.

Originally adapted from the TensorFlow-based implementation by Silvan Kaeser.
This version evaluates walker energies with the PhysNetJax model to ensure
consistency with the rest of the MMML tooling.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from e3x import ops as e3x_ops

try:
    from ase.io import read as ase_read
    from ase.io.trajectory import Trajectory
    from ase.optimize import BFGS
except ModuleNotFoundError as exc:  # pragma: no cover - script requires ASE
    sys.exit("ASE is required to read/write geometries: " + str(exc))

try:
    from mmml.cli.base import load_model_parameters, resolve_checkpoint_paths
    from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mmml.cli.base import load_model_parameters, resolve_checkpoint_paths
    from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--natm", type=int, help="number of atoms")
parser.add_argument("--nwalker", type=int, help="number of walkers in simulation")
parser.add_argument("--stepsize", type=float, help="The stepsize in imaginary time (atomic unit)")
parser.add_argument("--nstep", type=int, help="Total number of steps")
parser.add_argument("--eqstep", type=int, help="Number of steps for equilibration")
parser.add_argument("--alpha", type=float, help="Feed-back parameter, usually propotional to 1/stepsize")
parser.add_argument("--fbohr", type=int, default=0,
                    help="1 if the geometry given in the input is in bohr, 0 if angstrom")
parser.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to the PhysNetJax checkpoint directory")
parser.add_argument("--max-batch", type=int, default=512,
                    help="Maximum number of walker geometries evaluated per JAX batch")
parser.add_argument("--minimize-fmax", type=float, default=1e-3,
                    help="Force convergence criterion for ASE geometry minimisation (eV/Å)")
parser.add_argument("--minimize-steps", type=int, default=200,
                    help="Maximum ASE optimisation steps for the reference geometry")
parser.add_argument("--random-sigma", type=float, default=0.02,
                    help="Standard deviation (Å) of the random distortion applied to x0")
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input", type=str,
                      help="input file specifying the minimum and staring geometry (similar to xyz format)",
                      required=True)
args = parser.parse_args()

input_path = Path(args.input)
filename = input_path.with_suffix("").name
print("input: ", args.input)

np.random.seed(int(time.time()))

potfile = open(filename + ".pot", "w")
logfile = open(filename + ".log", "w")
errorfile = open("defective_" + filename + ".xyz", "w")
trajfile = Trajectory("configs_" + filename + ".traj", mode="w")

##############################################
# initialize/prepare all values/objects
##############################################
# Define constants
emass = 1822.88848
auang = 0.5291772083
aucm = 219474.6313710
EV_TO_HARTREE = 0.0367493

# read data from input file using ASE (supports extended XYZ headers)
structures = ase_read(str(input_path), index=":")
if not isinstance(structures, list):
    structures = [structures]
if len(structures) == 0:
    sys.exit(f"No structures found in input file {input_path}")

first_frame = structures[0]
if len(first_frame) != args.natm:
    sys.exit(
        "Mismatch between --natm and atoms in input: "
        f"expected {args.natm}, found {len(first_frame)}"
    )

atom_type = np.asarray(first_frame.get_chemical_symbols(), dtype=str)


def minimise_structure_with_model(atoms, params, model):
    atoms_min = atoms.copy()
    calc = get_ase_calc(params, model, atoms_min)
    atoms_min.calc = calc
    dyn = BFGS(atoms_min, logfile=None)
    try:
        dyn.run(fmax=args.minimize_fmax, steps=args.minimize_steps)
    finally:
        atoms_min.calc = None
    return atoms_min


mass = []
nucl_charge = []
for i in range(args.natm):
    if atom_type[i] == "H":
        mass.append(1.008)
        nucl_charge.append(1)
    elif atom_type[i] == "C":
        mass.append(12.011)
        nucl_charge.append(6)
    elif atom_type[i] == "O":
        mass.append(15.999)
        nucl_charge.append(8)
    else:
        print("UNKNOWN LABEL/atom type", atom_type[i])
        sys.exit(1)

mass = np.array(mass)
mass = np.sqrt(np.array(mass * emass))

max_batch = max(1, args.max_batch)

natm = args.natm
nwalker = args.nwalker
stepsize = args.stepsize
nstep = args.nstep
eqstep = args.eqstep
alpha = args.alpha
fbohr = args.fbohr

energy_chunk_size = max(1, min(max_batch, nwalker))

# Placeholders initialised when the PhysNetJax model is loaded
single_energy_fn = None
batched_energy_fn = None


def log_begin(name: str) -> None:
    logfile.write("                  DMC for " + name + "\n\n")
    logfile.write("DMC Simulation started at " + str(datetime.now()) + "\n")
    logfile.write("Number of random walkers: " + str(nwalker) + "\n")
    logfile.write("Number of total steps: " + str(nstep) + "\n")
    logfile.write("Number of steps before averaging: " + str(eqstep) + "\n")
    logfile.write("Stepsize: " + str(stepsize) + "\n")
    logfile.write("Alpha: " + str(alpha) + "\n\n")


def log_end(name: str) -> None:
    logfile.write("DMC Simulation terminated at " + str(datetime.now()) + "\n")
    logfile.write("DMC calculation terminated successfully\n")


def record_error(refx, mass_arr, symb, errq, v, idx):
    auang_local = 0.5291772083
    aucm_local = 219474.6313710

    if len(idx[0]) == 1:
        natm_local = int(len(refx) / 3)
        errx = errq[0] * auang_local
        errx = errx.reshape(natm_local, 3)
        errorfile.write(str(int(natm_local)) + "\n")
        errorfile.write(str(v[idx[0]] * aucm_local) + "\n")
        for i in range(int(natm_local)):
            errorfile.write(
                str(symb[i])
                + "  "
                + str(errx[i, 0])
                + "  "
                + str(errx[i, 1])
                + "  "
                + str(errx[i, 2])
                + "\n"
            )

    else:
        natm_local = int(len(refx) / 3)
        errx = errq[0] * auang_local
        errx = errx.reshape(len(idx[0]), natm_local, 3)

        for j in range(len(errx)):
            errorfile.write(str(int(natm_local)) + "\n")
            errorfile.write(str(v[idx[0][j]] * aucm_local) + "\n")
            for i in range(int(natm_local)):
                errorfile.write(
                    str(symb[i])
                    + "  "
                    + str(errx[j, i, 0])
                    + "  "
                    + str(errx[j, i, 1])
                    + "  "
                    + str(errx[j, i, 2])
                    + "\n"
                )


def ini_dmc():
    deltax_local = np.sqrt(stepsize) / mass

    psips_f[:] = 1
    psips_f[0] = nwalker
    psips_f[nwalker + 1:] = 0

    psips[:, :, 0] = x0[:]

    v_ref_local = v0
    v_ave_local = 0
    v_ref_local = v_ref_local - vmin

    potfile.write(
        "0  "
        + str(psips_f[0])
        + "  "
        + str(v_ref_local)
        + "  "
        + str(v_ref_local * aucm)
        + "\n"
    )
    return deltax_local, psips, psips_f, v_ave_local, v_ref_local


def walk(psips_arr, dx):
    dim = len(psips_arr[0, :, 0])
    for i in range(dim):
        x = np.random.normal(size=(len(psips_arr[:, 0, 0])))
        psips_arr[:, i, 1] = psips_arr[:, i, 0] + x * dx[math.ceil((i + 1) / 3.0) - 1]
    return psips_arr


def gbranch(refx, mass_arr, symb, vmin_local, psips_arr, psips_f_arr, v_ref_local, v_tot, nalive):
    birth_flag = 0
    error_checker = 0
    v_psip = get_batch_energy(psips_arr[:nalive, :], nalive)
    v_psip = v_psip - vmin_local

    if np.any(v_psip < -1e-5):
        error_checker = 1
        idx_err = np.where(v_psip < -1e-5)
        record_error(refx, mass_arr, symb, psips_arr[idx_err, :], v_psip, idx_err)
        print("defective geometry is written to file")
        psips_f_arr[idx_err[0] + 1] = 0

    prob = np.exp((v_ref_local - v_psip) * stepsize)
    sigma = np.random.uniform(size=nalive)

    if np.any((1.0 - prob) > sigma):
        idx_die = np.array(np.where((1.0 - prob) > sigma)) + 1
        psips_f_arr[idx_die] = 0
        v_psip[idx_die - 1] = 0.0

    v_tot = np.sum(v_psip)

    if np.any(prob > 1):
        idx_prob = np.array(np.where(prob > 1)).reshape(-1)

        for i in idx_prob:
            if error_checker == 0:
                probtmp = prob[i] - 1.0
                n_birth = int(probtmp)
                sigma_local = np.random.uniform()

                if (probtmp - n_birth) > sigma_local:
                    n_birth += 1
                if n_birth > 2:
                    birth_flag += 1

                while n_birth > 0:
                    nalive += 1
                    n_birth -= 1
                    psips_arr[nalive - 1, :] = psips_arr[i, :]
                    psips_f_arr[nalive] = 1
                    v_tot = v_tot + v_psip[i]

            else:
                if np.any(i == idx_err[0]):
                    pass
                else:
                    probtmp = prob[i] - 1.0
                    n_birth = int(probtmp)
                    sigma_local = np.random.uniform()

                    if (probtmp - n_birth) > sigma_local:
                        n_birth += 1
                    if n_birth > 2:
                        birth_flag += 1

                    while n_birth > 0:
                        nalive += 1
                        n_birth -= 1
                        psips_arr[nalive - 1, :] = psips_arr[i, :]
                        psips_f_arr[nalive] = 1
                        v_tot = v_tot + v_psip[i]

    error_checker = 0
    return psips_arr, psips_f_arr, v_tot, nalive


def branch(refx, mass_arr, symb, vmin_local, psips_arr, psips_f_arr, v_ref_local):
    nalive = psips_f_arr[0]
    v_tot = 0.0

    psips_arr[:, :, 1], psips_f_arr, v_tot, nalive = gbranch(
        refx, mass_arr, symb, vmin_local, psips_arr[:, :, 1], psips_f_arr, v_ref_local, v_tot, nalive
    )

    count_alive = 0
    psips_arr[:, :, 0] = 0.0

    for i in range(nalive):
        if psips_f_arr[i + 1] == 1:
            count_alive += 1
            psips_arr[count_alive - 1, :, 0] = psips_arr[i, :, 1]
            psips_f_arr[count_alive] = 1
    psips_f_arr[0] = count_alive
    psips_arr[:, :, 1] = 0.0
    psips_f_arr[count_alive + 1:] = 0

    v_ref_local = v_tot / psips_f_arr[0] + alpha * (1.0 - 3.0 * psips_f_arr[0] / (len(psips_f_arr) - 1))

    return psips_arr, psips_f_arr, v_ref_local


def get_batch_energy(coor: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size == 0:
        return np.array([], dtype=np.float64)

    if batched_energy_fn is None:
        raise RuntimeError("Energy function not initialised; load checkpoint first.")

    walker_coords = coor.reshape(batch_size, natm, 3) * auang
    energies_hartree: list[np.ndarray] = []

    for start in range(0, batch_size, energy_chunk_size):
        stop = min(start + energy_chunk_size, batch_size)
        chunk = walker_coords[start:stop]

        if chunk.shape[0] < energy_chunk_size:
            pad_count = energy_chunk_size - chunk.shape[0]
            if chunk.shape[0] == 0:
                pad_source = np.zeros((1, natm, 3), dtype=chunk.dtype)
            else:
                pad_source = chunk[-1:, ...]
            pad = np.repeat(pad_source, pad_count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        chunk_ev = jax.device_get(batched_energy_fn(jnp.asarray(chunk, dtype=jnp.float32)))
        energies_hartree.append(chunk_ev[: stop - start] * EV_TO_HARTREE)

    return np.concatenate(energies_hartree)


devices = jax.devices()
if devices:
    first_device = devices[0]
    device_desc = f"{first_device.platform.upper()}:{first_device.device_kind}"
else:
    device_desc = "CPU"
print(f"\n===========\nrunning on {device_desc}")
print("nwalkers:", nwalker, "\n===========\n")

atomic_numbers = np.asarray(nucl_charge, dtype=np.int32)
atomic_numbers_jnp = jnp.asarray(atomic_numbers)
pair_dst, pair_src = e3x_ops.sparse_pairwise_indices(natm)
pair_dst_jnp = jnp.asarray(pair_dst, dtype=jnp.int32)
pair_src_jnp = jnp.asarray(pair_src, dtype=jnp.int32)

base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
params, model = load_model_parameters(epoch_dir, natoms=natm)


template_atoms = first_frame.copy()
template_atoms.calc = None

try:
    xmin_atoms = minimise_structure_with_model(first_frame, params, model)
    xmin = np.asarray(xmin_atoms.get_positions(), dtype=float)
except Exception as exc:  # pragma: no cover - fallback if minimisation fails
    print(f"Warning: ASE minimisation failed ({exc}); using input geometry.", file=sys.stderr)
    xmin = np.asarray(first_frame.get_positions(), dtype=float)

x0 = xmin + np.random.normal(scale=args.random_sigma, size=xmin.shape)
template_atoms.set_positions(xmin)

def _reshape_to_angstrom(coords_flat: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(coords_flat.reshape(natm, 3) * auang, dtype=jnp.float32)


@jax.jit
def single_energy_fn_impl(positions_angstrom: jnp.ndarray) -> jnp.ndarray:
    output = model.apply(
        params,
        atomic_numbers=atomic_numbers_jnp,
        positions=positions_angstrom,
        dst_idx=pair_dst_jnp,
        src_idx=pair_src_jnp,
    )
    return output["energy"].squeeze()


def initialise_energy_functions():
    single = single_energy_fn_impl
    batched = jax.jit(jax.vmap(single))
    _ = batched(jnp.zeros((energy_chunk_size, natm, 3), dtype=jnp.float32))
    return single, batched


single_energy_fn, batched_energy_fn = initialise_energy_functions()

xmin = xmin.reshape(-1)
x0 = x0.reshape(-1)

dim = natm * 3
psips_f = np.zeros([3 * nwalker + 1], dtype=int)
deltax = np.zeros([natm], dtype=float)
psips = np.zeros([3 * nwalker, dim, 2], dtype=float)
symb = atomic_numbers

if fbohr == 0:
    x0 = x0 / auang
    xmin = xmin / auang

vmin = float(jax.device_get(single_energy_fn(_reshape_to_angstrom(xmin))) * EV_TO_HARTREE)
v0 = float(jax.device_get(single_energy_fn(_reshape_to_angstrom(x0))) * EV_TO_HARTREE)

log_begin(filename)

deltax, psips, psips_f, v_ave, v_ref = ini_dmc()

for i in range(nstep):
    start_time = time.time()
    psips[:psips_f[0], :, :] = walk(psips[:psips_f[0], :, :], deltax)

    psips, psips_f, v_ref = branch(x0, mass, symb, vmin, psips, psips_f, v_ref)
    potfile.write(
        str(i + 1)
        + "   "
        + str(psips_f[0])
        + "   "
        + str(v_ref)
        + "   "
        + str(v_ref * aucm)
        + "\n"
    )

    if i > eqstep:
        v_ave += v_ref

    if i > nstep - 10:
        for j in range(psips_f[0]):
            snapshot = template_atoms.copy()
            walker_coords = psips[j, :, 0].reshape(natm, 3) * auang
            snapshot.set_positions(walker_coords)
            trajfile.write(snapshot)
    if i % 10 == 0:
        print("step:  ", i, "time/step:  ", time.time() - start_time, "nalive:   ", psips_f[0])

v_ave = v_ave / (nstep - eqstep)
logfile.write(
    "AVERAGE ENERGY OF TRAJ   " + "   " + str(v_ave) + " hartree   " + str(v_ave * aucm) + " cm**-1\n"
)

log_end(filename)
potfile.close()
logfile.close()
errorfile.close()
trajfile.close()
