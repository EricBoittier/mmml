#!/usr/bin/env python3
"""Diffusion Monte Carlo driver using PhysNetJax energies.

Originally adapted from the TensorFlow-based implementation by Silvan Kaeser.
This version evaluates walker energies with the PhysNetJax model (batched via
``jax.vmap``) to stay consistent with the rest of the MMML tooling.

CLI::

    mmml dmc \\
      --natm 20 --nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 \\
      --alpha 1200.0 --checkpoint path/to/epoch-NNNNNN \\
      --input mmml/generate/dmc/examples/acetone_dmc.extxyz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# Physical constants
EMASS = 1822.88848
AUANG = 0.5291772083
AUCM = 219474.6313710
EV_TO_HARTREE = 0.0367493

_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Cl": 35.45,
}
_ATOMIC_Z: dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8, "Cl": 17}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml dmc",
        description=(
            "Diffusion Monte Carlo with PhysNetJax energies "
            "(batched walker evaluation via jax.vmap)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
        epilog=(
            "Example (acetone dimer smoke):\n"
            "  mmml dmc --natm 20 --nwalker 64 --stepsize 5e-4 --nstep 200 "
            "--eqstep 50 --alpha 1200.0 \\\n"
            "    --checkpoint \"$MMML_CKPT\" \\\n"
            "    --input mmml/generate/dmc/examples/acetone_dmc.extxyz \\\n"
            "    --output-dir runs/dmc_acetone_smoke\n"
            "\n"
            "Docs: docs/dmc.md  |  mmml dmc --help"
        ),
    )
    parser.add_argument(
        "--natm",
        type=int,
        required=True,
        help="Number of atoms per configuration (must match the input frame).",
    )
    parser.add_argument(
        "--nwalker",
        type=int,
        required=True,
        help="Number of walkers in the simulation.",
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        required=True,
        help="Imaginary-time stepsize (atomic units).",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        required=True,
        help="Total number of diffusion steps.",
    )
    parser.add_argument(
        "--eqstep",
        type=int,
        required=True,
        help="Equilibration steps discarded before energy averaging.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Feedback parameter (typically proportional to 1/stepsize).",
    )
    parser.add_argument(
        "--fbohr",
        type=int,
        default=0,
        choices=(0, 1),
        help="1 if input geometry is already in Bohr; 0 if Angstrom (default).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PhysNetJax or KerNN checkpoint (JSON / Orbax).",
    )
    parser.add_argument(
        "--model",
        choices=("physnet", "kernnn"),
        default=None,
        help="Energy backend (default: auto-detect KerNN JSON vs PhysNet).",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=512,
        help="Maximum walker geometries evaluated per JAX energy batch (default: 512).",
    )
    parser.add_argument(
        "--minimize-fmax",
        type=float,
        default=1e-3,
        help="ASE BFGS force convergence criterion in eV/Å (default: 1e-3).",
    )
    parser.add_argument(
        "--minimize-steps",
        type=int,
        default=200,
        help="Maximum ASE BFGS steps for the reference geometry (default: 200).",
    )
    parser.add_argument(
        "--random-sigma",
        type=float,
        default=0.02,
        help="Gaussian noise (Å) applied to the minimised geometry for x0 (default: 0.02).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: wall-clock time).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Geometry file (XYZ/EXTXYZ/anything ASE can read).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .pot/.log/.traj outputs (default: current working directory).",
    )
    return parser


def _masses_and_charges(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass: list[float] = []
    nucl_charge: list[int] = []
    for symbol in symbols:
        if symbol not in _ATOMIC_MASS:
            raise SystemExit(f"UNKNOWN LABEL/atom type {symbol}")
        mass.append(_ATOMIC_MASS[symbol])
        nucl_charge.append(_ATOMIC_Z[symbol])
    mass_arr = np.sqrt(np.asarray(mass, dtype=float) * EMASS)
    return mass_arr, np.asarray(nucl_charge, dtype=np.int32)


def _minimise_structure_with_model(
    atoms,
    params,
    model,
    *,
    minimize_fmax: float,
    minimize_steps: int,
    backend: str = "physnet",
    kernnn_checkpoint: Path | None = None,
):
    from ase.optimize import BFGS

    atoms_min = atoms.copy()
    if backend == "kernnn":
        from mmml.models.kernnn import KerNNCalculator

        if kernnn_checkpoint is None:
            raise ValueError("KerNN minimize requires kernnn_checkpoint")
        calc = KerNNCalculator(kernnn_checkpoint)
    else:
        from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

        calc = get_ase_calc(params, model, atoms_min)
    atoms_min.calc = calc
    dyn = BFGS(atoms_min, logfile=None)
    try:
        dyn.run(fmax=minimize_fmax, steps=minimize_steps)
    finally:
        atoms_min.calc = None
    return atoms_min


def _record_error(errorfile, refx, symb, errq, v, idx) -> None:
    if len(idx[0]) == 1:
        natm_local = int(len(refx) / 3)
        errx = errq[0] * AUANG
        errx = errx.reshape(natm_local, 3)
        errorfile.write(str(int(natm_local)) + "\n")
        errorfile.write(str(v[idx[0]] * AUCM) + "\n")
        for i in range(int(natm_local)):
            errorfile.write(
                f"{symb[i]}  {errx[i, 0]}  {errx[i, 1]}  {errx[i, 2]}\n"
            )
        return

    natm_local = int(len(refx) / 3)
    errx = errq[0] * AUANG
    errx = errx.reshape(len(idx[0]), natm_local, 3)
    for j in range(len(errx)):
        errorfile.write(str(int(natm_local)) + "\n")
        errorfile.write(str(v[idx[0][j]] * AUCM) + "\n")
        for i in range(int(natm_local)):
            errorfile.write(
                f"{symb[i]}  {errx[j, i, 0]}  {errx[j, i, 1]}  {errx[j, i, 2]}\n"
            )


def _walk(psips_arr: np.ndarray, dx: np.ndarray) -> np.ndarray:
    dim = len(psips_arr[0, :, 0])
    for i in range(dim):
        x = np.random.normal(size=(len(psips_arr[:, 0, 0])))
        psips_arr[:, i, 1] = psips_arr[:, i, 0] + x * dx[math.ceil((i + 1) / 3.0) - 1]
    return psips_arr


def run_dmc(args: argparse.Namespace) -> int:
    """Execute a DMC run with batched PhysNetJax energy evaluation."""
    import jax
    import jax.numpy as jnp
    from ase.io import read as ase_read
    from ase.io.trajectory import Trajectory
    from e3x import ops as e3x_ops

    from mmml.cli.base import load_model_parameters, resolve_checkpoint_paths

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    filename = input_path.with_suffix("").name
    out_dir = Path(args.output_dir) if args.output_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("input: ", args.input)

    seed = int(time.time()) if args.seed is None else int(args.seed)
    np.random.seed(seed)

    potfile = open(out_dir / f"{filename}.pot", "w")
    logfile = open(out_dir / f"{filename}.log", "w")
    errorfile = open(out_dir / f"defective_{filename}.xyz", "w")
    trajfile = Trajectory(str(out_dir / f"configs_{filename}.traj"), mode="w")

    try:
        structures = ase_read(str(input_path), index=":")
        if not isinstance(structures, list):
            structures = [structures]
        if len(structures) == 0:
            print(f"No structures found in input file {input_path}", file=sys.stderr)
            return 1

        first_frame = structures[0]
        natm = int(args.natm)
        if len(first_frame) != natm:
            print(
                f"Mismatch between --natm and atoms in input: expected {natm}, "
                f"found {len(first_frame)}",
                file=sys.stderr,
            )
            return 1

        atom_type = np.asarray(first_frame.get_chemical_symbols(), dtype=str)
        mass, atomic_numbers = _masses_and_charges(atom_type)

        nwalker = int(args.nwalker)
        stepsize = float(args.stepsize)
        nstep = int(args.nstep)
        eqstep = int(args.eqstep)
        alpha = float(args.alpha)
        fbohr = int(args.fbohr)
        max_batch = max(1, int(args.max_batch))
        energy_chunk_size = max(1, min(max_batch, nwalker))

        def log_begin(name: str) -> None:
            logfile.write("                  DMC for " + name + "\n\n")
            logfile.write("DMC Simulation started at " + str(datetime.now()) + "\n")
            logfile.write("Number of random walkers: " + str(nwalker) + "\n")
            logfile.write("Number of total steps: " + str(nstep) + "\n")
            logfile.write("Number of steps before averaging: " + str(eqstep) + "\n")
            logfile.write("Stepsize: " + str(stepsize) + "\n")
            logfile.write("Alpha: " + str(alpha) + "\n")
            logfile.write("RNG seed: " + str(seed) + "\n\n")

        def log_end() -> None:
            logfile.write("DMC Simulation terminated at " + str(datetime.now()) + "\n")
            logfile.write("DMC calculation terminated successfully\n")

        devices = jax.devices()
        if devices:
            first_device = devices[0]
            device_desc = f"{first_device.platform.upper()}:{first_device.device_kind}"
        else:
            device_desc = "CPU"
        print(f"\n===========\nrunning on {device_desc}")
        print("nwalkers:", nwalker, "\n===========\n")

        atomic_numbers_jnp = jnp.asarray(atomic_numbers)
        pair_dst, pair_src = e3x_ops.sparse_pairwise_indices(natm)
        pair_dst_jnp = jnp.asarray(pair_dst, dtype=jnp.int32)
        pair_src_jnp = jnp.asarray(pair_src, dtype=jnp.int32)

        from mmml.models.kernnn import (
            KerNNApplyAdapter,
            is_kernnn_checkpoint,
            load_checkpoint,
        )

        model_name = getattr(args, "model", None)
        use_kernnn = (
            str(model_name or "").lower() == "kernnn"
            or (not model_name and is_kernnn_checkpoint(args.checkpoint))
        )

        kernnn_ckpt_path = None
        if use_kernnn:
            kernnn_ckpt_path = Path(args.checkpoint).expanduser()
            if kernnn_ckpt_path.is_dir():
                for name in ("best.json", "params.json"):
                    cand = kernnn_ckpt_path / name
                    if cand.is_file():
                        kernnn_ckpt_path = cand
                        break
            kn_params, kn_config, kn_stats, _ = load_checkpoint(kernnn_ckpt_path)
            if natm != 4:
                raise SystemExit(
                    f"KerNN DMC currently supports 4-atom ABCC systems; got --natm {natm}"
                )
            params = kn_params
            model = KerNNApplyAdapter(stats=kn_stats, config=kn_config, n_atoms=natm)
            backend = "kernnn"
        else:
            _base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
            params, model = load_model_parameters(epoch_dir, natoms=natm)
            backend = "physnet"

        template_atoms = first_frame.copy()
        template_atoms.calc = None

        try:
            xmin_atoms = _minimise_structure_with_model(
                first_frame,
                params,
                model,
                minimize_fmax=args.minimize_fmax,
                minimize_steps=args.minimize_steps,
                backend=backend,
                kernnn_checkpoint=kernnn_ckpt_path,
            )
            xmin = np.asarray(xmin_atoms.get_positions(), dtype=float)
        except Exception as exc:  # pragma: no cover - fallback if minimisation fails
            print(
                f"Warning: ASE minimisation failed ({exc}); using input geometry.",
                file=sys.stderr,
            )
            xmin = np.asarray(first_frame.get_positions(), dtype=float)

        x0 = xmin + np.random.normal(scale=args.random_sigma, size=xmin.shape)
        template_atoms.set_positions(xmin)

        def _reshape_to_angstrom(coords_flat: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(coords_flat.reshape(natm, 3) * AUANG, dtype=jnp.float32)

        if backend == "kernnn":

            @jax.jit
            def single_energy_fn(positions_angstrom: jnp.ndarray) -> jnp.ndarray:
                output = model.apply(
                    params,
                    positions=positions_angstrom,
                    compute_forces=False,
                )
                return jnp.asarray(output["energy"]).reshape(())

        else:

            @jax.jit
            def single_energy_fn(positions_angstrom: jnp.ndarray) -> jnp.ndarray:
                output = model.apply(
                    params,
                    atomic_numbers=atomic_numbers_jnp,
                    positions=positions_angstrom,
                    dst_idx=pair_dst_jnp,
                    src_idx=pair_src_jnp,
                )
                return output["energy"].squeeze()

        # Parallelised walker energies: vmap over geometries, then JIT.
        batched_energy_fn: Callable = jax.jit(jax.vmap(single_energy_fn))
        _ = batched_energy_fn(
            jnp.zeros((energy_chunk_size, natm, 3), dtype=jnp.float32)
        )

        def get_batch_energy(coor: np.ndarray, batch_size: int) -> np.ndarray:
            if batch_size == 0:
                return np.array([], dtype=np.float64)

            walker_coords = coor.reshape(batch_size, natm, 3) * AUANG
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

                chunk_ev = jax.device_get(
                    batched_energy_fn(jnp.asarray(chunk, dtype=jnp.float32))
                )
                energies_hartree.append(chunk_ev[: stop - start] * EV_TO_HARTREE)

            return np.concatenate(energies_hartree)

        def gbranch(
            refx,
            symb,
            vmin_local,
            psips_arr,
            psips_f_arr,
            v_ref_local,
            v_tot,
            nalive,
        ):
            birth_flag = 0
            error_checker = 0
            v_psip = get_batch_energy(psips_arr[:nalive, :], nalive)
            v_psip = v_psip - vmin_local

            if np.any(v_psip < -1e-5):
                error_checker = 1
                idx_err = np.where(v_psip < -1e-5)
                _record_error(
                    errorfile, refx, symb, psips_arr[idx_err, :], v_psip, idx_err
                )
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
                    if error_checker != 0 and np.any(i == idx_err[0]):
                        continue
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

            return psips_arr, psips_f_arr, v_tot, nalive

        def branch(refx, symb, vmin_local, psips_arr, psips_f_arr, v_ref_local):
            nalive = psips_f_arr[0]
            v_tot = 0.0

            psips_arr[:, :, 1], psips_f_arr, v_tot, nalive = gbranch(
                refx,
                symb,
                vmin_local,
                psips_arr[:, :, 1],
                psips_f_arr,
                v_ref_local,
                v_tot,
                nalive,
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
            psips_f_arr[count_alive + 1 :] = 0

            v_ref_local = v_tot / psips_f_arr[0] + alpha * (
                1.0 - 3.0 * psips_f_arr[0] / (len(psips_f_arr) - 1)
            )
            return psips_arr, psips_f_arr, v_ref_local

        xmin = xmin.reshape(-1)
        x0 = x0.reshape(-1)
        dim = natm * 3
        psips_f = np.zeros([3 * nwalker + 1], dtype=int)
        psips = np.zeros([3 * nwalker, dim, 2], dtype=float)
        symb = atomic_numbers

        if fbohr == 0:
            x0 = x0 / AUANG
            xmin = xmin / AUANG

        vmin = float(
            jax.device_get(single_energy_fn(_reshape_to_angstrom(xmin))) * EV_TO_HARTREE
        )
        v0 = float(
            jax.device_get(single_energy_fn(_reshape_to_angstrom(x0))) * EV_TO_HARTREE
        )

        def ini_dmc():
            deltax_local = np.sqrt(stepsize) / mass
            psips_f[:] = 1
            psips_f[0] = nwalker
            psips_f[nwalker + 1 :] = 0
            psips[:, :, 0] = x0[:]
            v_ref_local = v0 - vmin
            v_ave_local = 0.0
            potfile.write(
                "0  "
                + str(psips_f[0])
                + "  "
                + str(v_ref_local)
                + "  "
                + str(v_ref_local * AUCM)
                + "\n"
            )
            return deltax_local, psips, psips_f, v_ave_local, v_ref_local

        log_begin(filename)
        deltax, psips, psips_f, v_ave, v_ref = ini_dmc()

        for i in range(nstep):
            start_time = time.time()
            psips[: psips_f[0], :, :] = _walk(psips[: psips_f[0], :, :], deltax)
            psips, psips_f, v_ref = branch(x0, symb, vmin, psips, psips_f, v_ref)
            potfile.write(
                str(i + 1)
                + "   "
                + str(psips_f[0])
                + "   "
                + str(v_ref)
                + "   "
                + str(v_ref * AUCM)
                + "\n"
            )

            if i > eqstep:
                v_ave += v_ref

            if i > nstep - 10:
                for j in range(psips_f[0]):
                    snapshot = template_atoms.copy()
                    walker_coords = psips[j, :, 0].reshape(natm, 3) * AUANG
                    snapshot.set_positions(walker_coords)
                    trajfile.write(snapshot)
            if i % 10 == 0:
                print(
                    "step:  ",
                    i,
                    "time/step:  ",
                    time.time() - start_time,
                    "nalive:   ",
                    psips_f[0],
                )

        if nstep > eqstep:
            v_ave = v_ave / (nstep - eqstep)
        else:
            v_ave = 0.0
        logfile.write(
            "AVERAGE ENERGY OF TRAJ   "
            + "   "
            + str(v_ave)
            + " hartree   "
            + str(v_ave * AUCM)
            + " cm**-1\n"
        )
        log_end()
    finally:
        potfile.close()
        logfile.close()
        errorfile.close()
        trajfile.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``mmml dmc``."""
    try:
        import ase  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        print("ASE is required to read/write geometries: " + str(exc), file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_dmc(args)


if __name__ == "__main__":
    raise SystemExit(main())
