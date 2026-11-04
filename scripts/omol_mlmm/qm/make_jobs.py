"""jobs.json: all new auto residues + existing CHARMM residues seen in OMol25 (validation, with their RTF charges)."""
import json, sys, os, numpy as np
sys.path.insert(0, "../../envtyper"); sys.path.insert(0, ".."); import cgenff_envtyper as E
from frag_mols import formula
D = json.load(open("../omol_auto.json")); F = {m["key"]: m for m in json.load(open("../frag_mols_4M_full.json"))["mols"]}
mass, res = E.parse_rtfs(); byname = {r["name"]: r for r in res}
jobs = []
for o in D["molecules"]:
    if not o["existing"]:
        jobs.append(dict(name=o["resi"], Z=o["Z"], R=o["R"], Q=o["Q"])); continue
# validation: existing residues -> find their OMol25 geometry by graph key, map RTF atoms onto it by r2 signature
keys = {}
for r in res:
    if not r["atoms"]: continue
    Z, adj, st = E.residue_graph(r, mass)
    if any(st) or 0 in Z or len(Z) > 32: continue
    sig = E.signatures(Z, adj); k = formula(Z) + "|" + ";".join(sorted(sig[2]))
    if k in F and r["file"] in E.PRIMARY and k not in keys: keys[k] = (r, sig)
val = []
for k, (r, sig) in keys.items():
    m = F[k]; Z = np.array(m["ex"]["Z"]); R = np.array(m["ex"]["R"]); s2 = E.signatures(Z, E.graph_from_xyz(Z, R))[2]
    ref = {}
    for a, s in zip(r["atoms"], sig[2]): ref.setdefault(s, []).append(r["charges"][a])
    if sorted(s2) != sorted(sig[2]): continue
    qref = [float(np.mean(ref[s])) for s in s2]
    jobs.append(dict(name="VAL_" + r["name"], Z=Z.tolist(), R=R.tolist(), Q=int(round(sum(r["charges"].values()))), q_cgenff=qref, n=m["n"]))
    val.append(r["name"])
json.dump(jobs, open("jobs.json", "w")); print(len(jobs), "jobs;", len(val), "validation:", val[:40])
