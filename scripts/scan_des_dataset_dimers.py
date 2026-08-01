#!/usr/bin/env python3
"""Rigid separation scans from real DES frames with original model parameters.

For each requested CGenFF residue pair, select a typeable frame directly from
``qcell_dimers.h5``.  The monomer geometries and relative orientation are held
fixed; only monomer B is translated along the frame's original COM axis.
Curves contain the original DES PhysNet interaction energy and stock (unit-
scale) CGenFF Coulomb/LJ interaction energy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_masses
from ase import Atoms

from mmml.data.cgenff_dataset import (
    assign_frame_cgenff,
    compute_inter_monomer_mm,
    load_reference,
)


def _com(z: np.ndarray, r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.average(r[idx], axis=0, weights=atomic_masses[z[idx]])


def translated_geometry(z, r, mol_id, separation):
    a = np.flatnonzero(mol_id == 0)
    b = np.flatnonzero(mol_id == 1)
    ca, cb = _com(z, r, a), _com(z, r, b)
    axis = cb - ca
    axis /= np.linalg.norm(axis)
    out = np.array(r, copy=True)
    out[b] += ca + float(separation) * axis - cb
    return out


def _physnet_energy_fn(checkpoint: Path, pad: int):
    import jax
    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint

    _, params, model = _load_physnet_checkpoint(checkpoint, pad)

    @jax.jit
    def evaluate(r, z):
        mask = (z > 0).astype(jnp.float32)
        ids = jnp.arange(pad)
        dst, src = jnp.meshgrid(ids, ids, indexing="ij")
        dst, src = dst.reshape(-1), src.reshape(-1)
        pair_mask = ((dst != src) & (mask[dst] > 0) & (mask[src] > 0)).astype(jnp.float32)
        out = model.apply(
            params,
            atomic_numbers=z,
            positions=r,
            dst_idx=dst,
            src_idx=src,
            batch_segments=jnp.zeros(pad, dtype=jnp.int32),
            batch_size=1,
            batch_mask=pair_mask,
            atom_mask=mask,
        )
        return jnp.asarray(out["energy"]).reshape(-1)[0]

    def wrapped(z: np.ndarray, r: np.ndarray) -> float:
        zp = np.zeros(pad, dtype=np.int32)
        rp = np.zeros((pad, 3), dtype=np.float64)
        zp[: len(z)] = z
        rp[: len(z)] = r
        return float(evaluate(jnp.asarray(rp), jnp.asarray(zp)))

    return wrapped


def _spooky_energy_fn(checkpoint: Path, ref):
    from mmml.models.spookynet_calc import SpookyNetCalculator

    calc = SpookyNetCalculator(checkpoint=checkpoint, mbd_checkpoint=False)

    def wrapped(z, r, type_idx, mol_id):
        atoms = Atoms(numbers=z, positions=r)
        atoms.arrays["cgenff_type_idx"] = np.asarray(type_idx, dtype=np.int32)
        atoms.arrays["mol_id"] = np.asarray(mol_id, dtype=np.int32)
        atoms.info["cgenff_master_sigmas"] = np.asarray(ref.sigmas)
        atoms.info["cgenff_master_epsilons"] = np.asarray(ref.epsilons)
        atoms.calc = calc
        atoms.get_potential_energy()
        return {
            key: float(calc.results.get(key, 0.0))
            for key in (
                "energy", "neural_energy", "electrostatics_energy",
                "cgenff_vdw_energy", "zbl_repulsion_energy",
            )
        }

    return wrapped, calc


def _mm_components(ref, r, a, ta, qa, b, tb, qb):
    dr = r[a, None, :] - r[None, b, :]
    distances = np.linalg.norm(dr, axis=-1)
    sigma = 0.5 * (ref.sigmas[ta, None] + ref.sigmas[None, tb])
    epsilon = np.sqrt(ref.epsilons[ta, None] * ref.epsilons[None, tb])
    coul = np.sum(332.06371 * qa[:, None] * qb[None, :] / np.maximum(distances, 1e-6))
    r_vdw = np.maximum(distances, 0.8 * sigma)
    sr6 = (sigma / r_vdw) ** 6
    lj = np.sum(4 * epsilon * (sr6**2 - sr6))
    return float(coul), float(lj)


def _select_frames(path: Path, wanted: set[tuple[str, str]], ref, group_names=()):
    selected = {}
    with h5py.File(path, "r") as fh:
        names = list(group_names) or sorted(k for k in fh if k != "metadata")
        for name in names:
            g = fh[name]
            z = np.asarray(g["atomic_numbers"][()], dtype=np.int32).reshape(-1)
            r = np.asarray(g["positions"][()], dtype=np.float64).reshape(-1, 3)
            assignment, _ = assign_frame_cgenff(z, r, ref, compute_mm=False)
            if assignment is None:
                continue
            pair = tuple(sorted(assignment.res_names))
            if pair in wanted and pair not in selected:
                selected[pair] = (name, z, r, assignment)
                print(f"selected {pair}: {name}")
                if len(selected) == len(wanted):
                    break
    missing = wanted - set(selected)
    if missing:
        raise RuntimeError(f"no typeable DES frame found for: {sorted(missing)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--spooky-checkpoint", type=Path)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--pairs", default="TIP3+TIP3,TIP3+MEOH,TIP3+BENZ,BENZ+BENZ,ACO+ACO,DCM+DCM")
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--points", type=int, default=65)
    p.add_argument("--groups", default="", help="comma-separated HDF5 groups to use directly")
    args = p.parse_args(argv)

    wanted = {tuple(sorted(x.strip().split("+"))) for x in args.pairs.split(",") if x.strip()}
    ref = load_reference()
    groups = tuple(x.strip() for x in args.groups.split(",") if x.strip())
    selected = _select_frames(args.h5.expanduser(), wanted, ref, groups)
    physnet = _physnet_energy_fn(args.checkpoint.expanduser(), pad=34)
    spooky = spooky_calc = None
    if args.spooky_checkpoint is not None:
        spooky, spooky_calc = _spooky_energy_fn(args.spooky_checkpoint.expanduser(), ref)
    rs = np.linspace(args.r_min, args.r_max, args.points)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    rows = []
    for ax, pair in zip(axes.ravel(), sorted(wanted), strict=True):
        name, z, r0, assignment = selected[pair]
        a = np.flatnonzero(assignment.mol_id == 0)
        b = np.flatnonzero(assignment.mol_id == 1)
        original_r = float(np.linalg.norm(_com(z, r0, a) - _com(z, r0, b)))
        ea = physnet(z[a], r0[a])
        eb = physnet(z[b], r0[b])
        spooky_a = spooky_b = None
        if spooky is not None:
            spooky_a = spooky(z[a], r0[a], assignment.cgenff_type_idx[a], np.zeros(len(a), dtype=np.int32))
            spooky_b = spooky(z[b], r0[b], assignment.cgenff_type_idx[b], np.zeros(len(b), dtype=np.int32))
        ml, mm, spooky_values, contacts = [], [], [], []
        for separation in rs:
            r = translated_geometry(z, r0, assignment.mol_id, separation)
            contact = float(np.min(np.linalg.norm(r[a, None, :] - r[None, b, :], axis=-1)))
            e_ml = physnet(z, r) - ea - eb
            e_mm, _ = compute_inter_monomer_mm(
                ref, r, a, assignment.cgenff_type_idx[a], assignment.cgenff_charge[a],
                b, assignment.cgenff_type_idx[b], assignment.cgenff_charge[b],
            )
            ml.append(e_ml * 23.0605)
            mm.append(e_mm * 23.0605)
            if spooky is not None:
                e_spooky = spooky(z, r, assignment.cgenff_type_idx, assignment.mol_id)
                spooky_interactions = {
                    key: (e_spooky[key] - spooky_a[key] - spooky_b[key]) * 23.0605
                    for key in e_spooky
                }
                spooky_values.append(spooky_interactions["energy"])
            else:
                spooky_values.append(np.nan)
            contacts.append(contact)
            e_coul, e_lj = _mm_components(
                ref, r, a, assignment.cgenff_type_idx[a], assignment.cgenff_charge[a],
                b, assignment.cgenff_type_idx[b], assignment.cgenff_charge[b],
            )
            if spooky is None:
                spooky_interactions = {key: np.nan for key in (
                    "energy", "neural_energy", "electrostatics_energy",
                    "cgenff_vdw_energy", "zbl_repulsion_energy",
                )}
            rows.append(("+".join(pair), name, original_r, separation, contact,
                         float(np.sum(assignment.cgenff_charge[a])),
                         float(np.sum(assignment.cgenff_charge[b])),
                         ml[-1], mm[-1], e_coul, e_lj,
                         spooky_interactions["energy"],
                         spooky_interactions["neural_energy"],
                         spooky_interactions["electrostatics_energy"],
                         spooky_interactions["cgenff_vdw_energy"],
                         spooky_interactions["zbl_repulsion_energy"]))
        physical = np.asarray(contacts) >= 1.2
        ml_plot = np.where(physical, ml, np.nan)
        mm_plot = np.where(physical, mm, np.nan)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvline(original_r, color="0.45", lw=0.8, ls=":", label="dataset frame")
        ax.plot(rs, ml_plot, lw=2, label="original DES PhysNet")
        ax.plot(rs, mm_plot, lw=1.5, ls="--", label="stock CGenFF")
        if spooky is not None:
            ax.plot(rs, np.where(physical, spooky_values, np.nan), lw=1.7,
                    ls="-.", label="SO3LR Spooky epoch 10")
        ax.set_title(" + ".join(pair))
        ax.set_xlabel("COM separation (Å)")
        ax.set_ylabel("interaction energy (kcal/mol)")
        visible = np.concatenate([np.asarray(ml)[physical], np.asarray(mm)[physical]])
        ax.set_ylim(max(-20, np.min(visible) * 1.2), min(50, max(10, np.percentile(visible, 90))))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(args.output_dir / "dataset_dimer_scans.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "dataset_dimer_scans.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("residue_pair", "h5_group", "original_com_A", "scan_com_A", "min_contact_A",
                         "monomer_a_charge", "monomer_b_charge",
                         "physnet_interaction_kcal", "cgenff_interaction_kcal",
                         "cgenff_coulomb_kcal", "cgenff_lj_kcal", "spooky_interaction_kcal",
                         "spooky_neural_kcal", "spooky_electrostatics_kcal",
                         "spooky_cgenff_vdw_kcal", "spooky_zbl_kcal"))
        writer.writerows(rows)
    if spooky_calc is not None:
        spooky_calc.write_energy_function_report(args.output_dir / "spooky_energy_function.json")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
