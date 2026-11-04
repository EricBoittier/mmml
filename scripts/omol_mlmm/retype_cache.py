"""Write caches/omol_mlmm_noncov300k_cgenff: the 300k ML/MM cache with REAL CGenFF/CHARMM types from the environment
typer (types300k.npz) and LJ tables from the CHARMM parameter files. Untyped atoms -> element-level fallback type
(approximate CGenFF per-element LJ, named X_<el>), counted in metadata."""
import json, os, sys, numpy as np, orbax.checkpoint as ocp
from flax.training import orbax_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ljtab, build_mlmm_cache as B
U = "/mmhome/boittier/home/metatomic-runs/training/omol_universal"
src, dst = f"{U}/caches/omol_mlmm_noncov300k", f"{U}/caches/omol_mlmm_noncov300k_cgenff"
d = ocp.PyTreeCheckpointer().restore(src); d = {k: np.asarray(v) for k, v in d.items()}
ty = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/types300k.npz", allow_pickle=False)
names = [str(x) for x in ty["type_names"]]; tidx = ty["type_idx"].astype(np.int64); level = ty["level"]
assert len(tidx) == d["Z"].shape[0], (len(tidx), d["Z"].shape)
lj = ljtab.lj_table()
Z = d["Z"].astype(int); el_pt = {z: i for i, z in [(0, 1)]}  # unused placeholder
# final type list: real types with LJ, then per-element fallbacks
real = [t for t in names if t and t in lj]; fall_el = sorted(set(Z[(level < 0) | np.isin(np.array(names)[tidx], [t for t in names if not t or t not in lj])].tolist()))
final = real + [f"X_{z}" for z in fall_el]; pos = {t: i for i, t in enumerate(final)}
def pt_lj(z):   # element-level approximate CGenFF LJ (same table as the pseudo-types), H -> non-polar H
    i = 0 if z == 1 else B.ZMAP.get(int(z), 17); return B.PT[i][2], B.PT[i][1]
cg = np.empty(len(tidx), np.int32); nm = np.array(names)[tidx]
ok = np.array([t in pos for t in nm]); cg[ok] = [pos[t] for t in nm[ok]]
cg[~ok] = [pos[f"X_{z}"] for z in Z[~ok]]
eps = np.array([lj[t][0] if t in lj else pt_lj(int(t[2:]))[0] for t in final]); rmin = np.array([lj[t][1] if t in lj else pt_lj(int(t[2:]))[1] for t in final])
d["cgenff_type_idx"] = cg; d["cgenff_master_epsilons"] = eps; d["cgenff_master_sigmas"] = 2 * rmin / 2 ** (1 / 6)
ocp.PyTreeCheckpointer().save(dst, d, save_args=orbax_utils.save_args_from_target(d), force=True)
meta = json.load(open(f"{src}/metadata.json")); cnt = np.bincount(cg, minlength=len(final))
meta.update(types_source="CGenFF environment typer (mlmm_ft/envtyper, CHARMM RTF libraries; LORO 93.8 % exact, 95.7 % LJ-equivalent)",
            n_types=len(final), type_names=final, type_atoms=cnt.tolist(), fallback_atoms=int((~ok).sum()), fallback_frac=float((~ok).mean()),
            level_frac={"r2": float(np.mean(level == 2)), "r1": float(np.mean(level == 1)), "r0": float(np.mean(level == 0)),
                        "fallback_library": float(np.mean(level >= 10)), "untyped": float(np.mean(level < 0))})
json.dump(meta, open(f"{dst}/metadata.json", "w"), indent=1)
print(f"{len(final)} types ({len(real)} CHARMM + {len(fall_el)} element fallbacks); fallback atoms {(~ok).sum()} ({(~ok).mean()*100:.2f} %) -> {dst}")
