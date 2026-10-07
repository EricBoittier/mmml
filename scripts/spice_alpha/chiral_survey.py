#!/usr/bin/env python
"""Find chiral molecules in a SPICE-α split from their 3D geometries.

Rebuilds the ragged-frame -> (HDF5 group = SMILES key, conformer) map with the
same filters as ``convert_hdf5_ragged`` (checked against the shard atomic
numbers), then for one geometry per molecule perceives bonds with RDKit
(rdDetermineBonds, neutral) and assigns tetrahedral stereocentres from 3D.

Writes OUT_DIR/frame_map.npz (group index + conformer per ragged frame,
group names) and OUT_DIR/chiral_<split>.json (one record per molecule).

Usage:
  python scripts/spice_alpha/chiral_survey.py H5_DIR RAGGED_DIR SPLIT_NPZ OUT_DIR [--split test]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def kept_confs(group) -> np.ndarray:
    """Conformer indices the ragged converter keeps for one HDF5 group."""
    from karml.data.spice_alpha import _optional_charge
    from karml.data.spice_alpha_ragged import polar_e_angstrom2_per_volt_to_bohr3

    if "conformations" not in group or "atomic_numbers" not in group:
        return np.zeros(0, int)
    if "dft_total_energy" not in group or "dft_total_gradient" not in group:
        return np.zeros(0, int)
    R = group["conformations"][()]
    n_conf = R.shape[0]
    E = np.asarray(group["dft_total_energy"][()], float).reshape(-1)
    G = np.asarray(group["dft_total_gradient"][()], float)
    ok = np.isfinite(E) & np.isfinite(G).reshape(n_conf, -1).all(1)
    Q = _optional_charge(group, n_conf)
    if Q is not None:
        ok &= ~(np.abs(Q) > 0.5)  # NaN (MBIS failed, e.g. iodine) is kept, as in the converter
    if "polarizability" not in group:
        return np.zeros(0, int)
    P = polar_e_angstrom2_per_volt_to_bohr3(np.asarray(group["polarizability"][()], float))
    ok &= np.isfinite(P).reshape(n_conf, -1).all(1) & (np.abs(P).reshape(n_conf, -1).max(1) <= 1.0e4)
    D = np.asarray(group["scf_dipole"][()], float) if "scf_dipole" in group else np.full((n_conf, 3), np.nan)
    ok &= np.isfinite(D).all(1)
    return np.nonzero(ok)[0]


def map_file(h5_path: str):
    import h5py
    names, gidx, conf = [], [], []
    with h5py.File(h5_path, "r") as h5:
        for name in h5.keys():
            k = kept_confs(h5[name])
            if len(k):
                gidx.extend([len(names)] * len(k)); conf.extend(k.tolist()); names.append(str(name))
    return names, np.asarray(gidx, np.int32), np.asarray(conf, np.int32)


def stereo(Z, R):
    """(centres [(atom, 'R'/'S'/'?')], n_fragments, bond perception mode)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    RDLogger.DisableLog("rdApp.*")
    from ase.data import chemical_symbols
    xyz = f"{len(Z)}\n\n" + "\n".join(f"{chemical_symbols[int(z)]} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(Z, R))
    mol = Chem.MolFromXYZBlock(xyz)
    mode = "orders"
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(mol)
        mode = "connectivity"
    Chem.AssignStereochemistryFrom3D(mol)
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    nfrag = len(Chem.GetMolFrags(mol))
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol)) if mode == "orders" else ""
    return [(int(a), c) for a, c in centres], nfrag, mode, smi


def survey_one(args):
    f, Z, R = args
    try:
        c, nfrag, mode, smi = stereo(Z, R)
        return f, c, nfrag, mode, smi, ""
    except Exception as exc:  # report, do not abort the survey
        return f, [], 0, "failed", "", str(exc)[:120]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5_dir", type=Path)
    ap.add_argument("ragged_dir", type=Path)
    ap.add_argument("split_npz", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from karml.data.spice_alpha_ragged import SUBSET_IDS, load_ragged

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.ragged_dir.glob("*.npz"))
    data = load_ragged(shards)
    off = data["offsets"]
    # Frame map, shard by shard in load order.
    with ProcessPoolExecutor(len(shards)) as ex:
        maps = list(ex.map(map_file, [str(args.h5_dir / (p.stem + ".hdf5")) for p in shards]))
    names, gidx, conf, base = [], [], [], 0
    for p, (nm, g, c) in zip(shards, maps):
        n_frames = len(np.load(p)["N"])
        assert len(g) == n_frames, f"{p.name}: map has {len(g)} frames, shard has {n_frames}"
        gidx.append(g + len(names)); conf.append(c); names.extend(nm); base += n_frames
    gidx, conf = np.concatenate(gidx), np.concatenate(conf)
    assert len(gidx) == len(data["N"])
    np.savez_compressed(args.out_dir / "frame_map.npz", group=gidx, conf=conf, names=np.asarray(names))
    # Spot-check atomic numbers against the HDF5 groups.
    import h5py
    rng = np.random.default_rng(0)
    file_of = np.repeat(np.arange(len(shards)), [len(np.load(p)["N"]) for p in shards])
    for f in rng.choice(len(gidx), 200, replace=False):
        with h5py.File(args.h5_dir / (shards[file_of[f]].stem + ".hdf5"), "r") as h5:
            g = h5[names[gidx[f]]]
            assert np.array_equal(g["atomic_numbers"][()], data["Z"][off[f]:off[f + 1]]), f
            assert np.allclose(g["conformations"][conf[f]], data["R"][off[f]:off[f + 1]], atol=1e-4), f
    print("frame map verified on 200 random frames", flush=True)

    idx = np.sort(np.load(args.split_npz)[args.split])
    mols = data["mol"][idx]
    first = {}
    count = {}
    for f, m in zip(idx, mols):
        first.setdefault(int(m), int(f)); count[int(m)] = count.get(int(m), 0) + 1
    jobs = [(f, data["Z"][off[f]:off[f + 1]], data["R"][off[f]:off[f + 1]]) for f in first.values()]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(survey_one, jobs, chunksize=64))
    names_sub = {v: k for k, v in SUBSET_IDS.items()}
    out = []
    for (f, c, nfrag, mode, smi, err), m in zip(res, first.keys()):
        out.append({"mol": m, "frame": f, "group": names[gidx[f]], "subset": names_sub[int(data["subset"][f])],
                    "n_atoms": int(off[f + 1] - off[f]), "n_frames": count[m], "centres": c,
                    "n_fragments": nfrag, "bonds": mode, "smiles": smi, "error": err})
    (args.out_dir / f"chiral_{args.split}.json").write_text(json.dumps(out))
    n_chiral = sum(1 for r in out if r["centres"])
    print(f"{len(out)} molecules, {n_chiral} with >=1 stereocentre, "
          f"{sum(r['bonds'] == 'failed' for r in out)} failed", flush=True)


if __name__ == "__main__":
    main()
