"""ML/MM fine-tune cache from OMol25 non-covalent multi-molecule structures (4 Nov 2026).

Selection (from fragment_census_train.npz): subset in SUBSETS, n_frag >= 2, closest inter-fragment heavy-atom contact
>= DMIN (non-covalent; excludes bond-breaking reactivity/TS fragments). Per atom: mol_id (fragment index within the
structure, covalent radii x1.2 -- same rule as the census) and cgenff_type_idx = heuristic PSEUDO-TYPE (element, H split
into polar / non-polar by its bonded partner). LJ tables: representative CGenFF / CHARMM36 values per pseudo-type
(APPROXIMATE, placeholder until the sciCORE FF-typer types arrive). Energy: same convention as the pilot
(E_DFT - sum E_iso - pilot linear ref incl. b_Q*Q); train with --energy-add-charge-ref 5.661417917773716.
    python build_mlmm_cache.py --census <npz> --shards <npz_dir> --linear-ref-from <meta.json> --n N --out <cache>
"""
import argparse, glob, json, os
import numpy as np
from fragment_census import frag_one

SUBSETS = ("biomolecules", "elytes", "ani2x", "spice", "metal_complexes", "orbnet_denali")
# pseudo-type: (name, Rmin/2 [A], eps [kcal/mol, >=0])  -- representative CGenFF / CHARMM36 (approximate)
PT = [("H_nonpolar", 1.34, 0.035),   # HGA2/HGA3/HGR61
      ("H_polar", 0.2245, 0.046),    # HGP1 / HT (TIP3P)
      ("C", 2.00, 0.060),            # CG321 2.01/0.056, CG2R61 1.9925/0.070
      ("N", 1.85, 0.200),            # NG321 / NG2S1
      ("O", 1.75, 0.150),            # OG311 1.765/0.192, OG2D1 1.70/0.12, OT 1.7682/0.1521
      ("F", 1.63, 0.135), ("P", 2.15, 0.585), ("S", 2.00, 0.450), ("Cl", 1.91, 0.343), ("Br", 1.97, 0.480),
      ("I", 2.19, 0.550), ("Li", 1.2975, 0.00233), ("Na", 1.41075, 0.0469), ("K", 1.76375, 0.0870),
      ("Mg", 1.185, 0.0150), ("Ca", 1.367, 0.120), ("Zn", 1.09, 0.250), ("other", 1.80, 0.100)]
ZMAP = {6: 2, 7: 3, 8: 4, 9: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10, 3: 11, 11: 12, 19: 13, 12: 14, 20: 15, 30: 16}
SIG = np.array([2 * r / 2 ** (1 / 6) for _, r, _ in PT]); EPS = np.array([e for *_, e in PT])

def pseudo_types(Z, R):
    t = np.array([ZMAP.get(int(z), 17) for z in Z], np.int32)
    h = np.where(Z == 1)[0]
    if len(h):
        heavy = np.where(Z > 1)[0]
        if len(heavy):
            d = np.linalg.norm(R[h][:, None] - R[heavy][None], axis=-1); nb = Z[heavy][d.argmin(1)]
            t[h] = np.where(np.isin(nb, (7, 8, 16)), 1, 0)
        else:
            t[h] = 0
    return t

ALLOWED = None          # set of Z allowed anywhere (None = all)
FREE_IONS = {11, 17}    # Na+, Cl-: allowed only as single-atom fragments (typer vocab: SOD, CLA)


def _one_shard(job):
    path, rows, tab, bq, allowed = job
    with np.load(path, allow_pickle=False) as d:          # read each array ONCE (npz re-reads on every key access)
        off = d["mol_offsets"]; Za, Ra, Fa = d["Z"], d["R"], d["F"]
        Ea, Qa, Sa = d["E"].reshape(-1), d["Q"].reshape(-1), d["S"].reshape(-1)
    out = {k: [] for k in ("R", "Z", "F", "E", "N", "Q", "S", "D", "mol_id", "cgenff_type_idx")}
    for r in rows:
        Z = Za[off[r]:off[r + 1]]; R = Ra[off[r]:off[r + 1]]
        nc, lab, _ = frag_one(Z, R)
        if nc < 2:
            continue
        if allowed is not None:
            single = np.bincount(lab)[lab] == 1
            if not np.all(np.isin(Z, list(allowed)) | (np.isin(Z, list(FREE_IONS)) & single)):
                continue
        q = float(Qa[r])
        out["R"].append(R); out["Z"].append(Z); out["F"].append(Fa[off[r]:off[r + 1]])
        out["mol_id"].append(lab.astype(np.int32)); out["cgenff_type_idx"].append(pseudo_types(Z, R))
        out["E"].append(float(Ea[r]) - tab[Z].sum() - bq * q); out["N"].append(len(Z))
        out["Q"].append(q); out["S"].append(float(Sa[r])); out["D"].append(np.zeros(3))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--census", required=True); ap.add_argument("--shards", required=True)
    ap.add_argument("--linear-ref-from", required=True); ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--dmin", type=float, default=2.0); ap.add_argument("--max-atoms", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allowed-elements", default="", help="comma list of Z allowed anywhere (plus free Na+/Cl-), e.g. 1,6,7,8")
    ap.add_argument("--n-draw", type=int, default=0, help="draw this many candidates before the element filter, then keep --n")
    a = ap.parse_args()
    with np.load(a.census, allow_pickle=False) as cz:   # read each array ONCE (NpzFile re-reads on every key access)
        c = {k: cz[k] for k in ("subset", "n_frag", "dmin_inter", "n_atoms", "shard", "row")}
    sel = np.isin(c["subset"], SUBSETS) & (c["n_frag"] >= 2) & (c["dmin_inter"] >= a.dmin) & (c["n_atoms"] <= a.max_atoms)
    idx = np.where(sel)[0]; print("eligible", len(idx), {s: int((c["subset"][idx] == s).sum()) for s in SUBSETS}, flush=True)
    rng = np.random.default_rng(a.seed); idx = np.sort(rng.choice(idx, size=min(max(a.n, a.n_draw), len(idx)), replace=False))
    allowed = set(int(x) for x in a.allowed_elements.split(",")) if a.allowed_elements else None
    lr = json.load(open(a.linear_ref_from))["linear_ref"]; tab = np.zeros(119); tab[np.asarray(lr["Z"])] = lr["b_eV"]; bq = lr["b_Q_eV_per_e"]
    parts = {k: [] for k in ("R", "Z", "F", "E", "N", "Q", "S", "D", "mol_id", "cgenff_type_idx")}
    by_shard = {}
    for sh, r in zip(c["shard"][idx].tolist(), c["row"][idx].tolist()):
        by_shard.setdefault(sh, []).append(int(r))
    from multiprocessing import Pool
    jobs = [(os.path.join(a.shards, sh), rows, tab, bq, allowed) for sh, rows in sorted(by_shard.items())]
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap(_one_shard, jobs)):
            for key in parts:
                parts[key].extend(res[key])
            if k % 10 == 0:
                print(f"shard {k + 1}/{len(jobs)}: {len(parts['N'])} structures", flush=True)
    if len(parts["N"]) > a.n:
        keep = np.sort(np.random.default_rng(a.seed + 1).choice(len(parts["N"]), size=a.n, replace=False))
        parts = {k: [v[i] for i in keep] for k, v in parts.items()}
    print(f"kept {len(parts['N'])} structures after the element filter", flush=True)
    data = {"R": np.concatenate(parts["R"]).astype(np.float64), "Z": np.concatenate(parts["Z"]).astype(np.int32),
            "F": np.concatenate(parts["F"]).astype(np.float64), "E": np.array(parts["E"]).reshape(-1, 1),
            "N": np.array(parts["N"], np.int32).reshape(-1, 1), "Q": np.array(parts["Q"]).reshape(-1, 1),
            "S": np.array(parts["S"]).reshape(-1, 1), "D": np.array(parts["D"]).reshape(-1, 3),
            "mol_id": np.concatenate(parts["mol_id"]), "cgenff_type_idx": np.concatenate(parts["cgenff_type_idx"]),
            "cgenff_master_sigmas": SIG.astype(np.float64), "cgenff_master_epsilons": EPS.astype(np.float64)}
    data["mol_offsets"] = np.concatenate([[0], np.cumsum(data["N"].ravel())]).astype(np.int64)
    n, na = data["N"].shape[0], int(data["mol_offsets"][-1])
    data["metadata_n_structures"] = np.asarray(n, np.int64); data["metadata_n_atoms_total"] = np.asarray(na, np.int64)
    data["metadata_max_atoms"] = np.asarray(int(data["N"].max()), np.int32)
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils
    out = os.path.abspath(a.out); ocp.PyTreeCheckpointer().save(out, data, save_args=orbax_utils.save_args_from_target(data))
    meta = {"n_structures": n, "n_atoms_total": na, "max_atoms": int(data["N"].max()), "selection": {"subsets": SUBSETS, "dmin": a.dmin, "seed": a.seed},
            "linear_ref": dict(lr, applied_from=os.path.abspath(a.linear_ref_from)), "pseudo_types": [p[0] for p in PT],
            "lj": {"sigma_A": SIG.tolist(), "eps_kcal": EPS.tolist(), "note": "approximate CGenFF/CHARMM36 per pseudo-type"},
            "dipole_key": "", "dipole_is_zero": True, "E_std": float(data["E"].std())}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n_structures", "n_atoms_total", "E_std")}), "->", out)

if __name__ == "__main__":
    main()
