"""Auto residues for frequent OMol25 small molecules.  For each graph seen >= MIN times (non-covalently separated):
  * matched to an existing CHARMM RESI by the same graph key (formula + sorted r2 signatures) -> reuse, no new residue;
  * net charge: mode over single-molecule structures, else RDKit DetermineBonds over 0,-1,+1,-2,+2 (closed shell only);
  * types: CGenFF environment typer v3 (so CHARMM typing via the RESI == the training labels by construction);
    untyped atoms -> provisional new types XN<el><k>, one per distinct (element, degree, nH, ring, neighbour elements);
  * charges: EnvCharger transfer from CHARMM RTFs, neutralised to the net charge (LORO MAE 0.028 e).
    python build_residues.py frag_mols_4M.json MIN out_prefix"""
import json, sys, os, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envtyper"))
import cgenff_envtyper as E, envcharge as EC
from frag_mols import formula, EL
from rdkit import Chem, RDLogger; from rdkit.Chem import rdDetermineBonds, Descriptors
RDLogger.DisableLog("rdApp.*")
def hill(Zl):   # display formula (frag_mols.formula keeps an empty 'H' for keys; do not change keys)
    c = collections.Counter(EL.get(int(x), str(x)) for x in Zl); ks = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in ks if c[k])
src, MIN, pre = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D = json.load(open(src)); scale = 1 / D["fraction"]
mass, res = E.parse_rtfs(); prim = [r for r in res if r["file"] in E.PRIMARY]
CH = EC.EnvCharger(prim, mass)
known = {}
for r in res:
    if not r["atoms"]: continue
    Z, adj, stubs = E.residue_graph(r, mass)
    if any(stubs) or 0 in Z: continue
    sig = E.signatures(Z, adj); known.setdefault(formula(Z) + "|" + ";".join(sorted(sig[2])), f"{r['name']} ({r['file'].replace('top_all36_','').replace('.rtf','')})")
def rdmol(Zl, R, q):
    m = Chem.RWMol()
    for z in Zl: a = Chem.Atom(int(z)); a.SetNoImplicit(True); m.AddAtom(a)
    if len(Zl) == 1:
        m.GetAtomWithIdx(0).SetFormalCharge(q); Chem.SanitizeMol(m)
        return m if Descriptors.NumRadicalElectrons(m) == 0 else None
    conf = Chem.Conformer(len(Zl))
    for i, x in enumerate(R): conf.SetAtomPosition(i, [float(v) for v in x])
    m.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineBonds(m, charge=q); Chem.SanitizeMol(m)
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()): return None
        return m
    except Exception: return None
out = []; newtypes = {}; skipped = collections.Counter(); hyper_used = []
for M in D["mols"]:
    if M["n"] < MIN: break
    key = M["key"].split("|", 1)[1]; zf = hill(M["ex"]["Z"])
    ex = M["ex"]; Zl = ex["Z"]; R = np.array(ex["R"])     # same occurrence the types were computed from (atom order!)
    nel = sum(Zl)
    adj0 = E.graph_from_xyz(np.array(Zl), R)
    hyper = len(Zl) > 1 and any(Zl[i] in (5, 15, 16, 17, 33) and len(adj0[i]) >= 4 and all(Zl[j] in (7, 8, 9) for j in adj0[i]) for i in range(len(Zl))) \
            and all(len(adj0[i]) >= 1 for i in range(len(Zl)))
    if M["Q"]:
        cands = [int(max(M["Q"], key=lambda k: M["Q"][k]))]
    elif hyper and nel % 2 == 1: cands = [-1]          # PF6-, BF4-, ClO4-, FSI-, TFSI-, OTf-: hypervalent centre with F/O/N ligands
    elif nel % 2 == 1 and not (len(Zl) == 1 and Zl[0] in (9, 17, 35, 53)):
        skipped["charge/spin ambiguous (never isolated, odd electrons: radical or ion)"] += 1; continue
    else: cands = [0, -1, 1, -2, 2]
    cands = [q for q in cands if (nel - q) % 2 == 0]
    mol = None
    for q in cands:
        mol = rdmol(Zl, R, q)
        if mol is not None: break
    if mol is None and hyper and (nel - cands[0]) % 2 == 0:   # RDKit cannot draw octet Lewis structures for these: connectivity only
        q = cands[0]; mol = Chem.RWMol()
        for z in Zl: a = Chem.Atom(int(z)); a.SetNoImplicit(True); mol.AddAtom(a)
        for i in range(len(Zl)):
            for j in adj0[i]:
                if i < j: mol.AddBond(i, j, Chem.BondType.SINGLE)
        mol = mol.GetMol(); mol.UpdatePropertyCache(strict=False); hyper_used.append(hill(Zl))
    if mol is None: skipped["no closed-shell Lewis structure (radical / bonded contact / TS)"] += 1; continue
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False)) if hill(Zl) in hyper_used else Chem.MolToSmiles(Chem.RemoveHs(mol))
    if key in known or M["key"] in known:
        out.append(dict(smiles=smi, formula=zf, n=int(M["n"] * scale), Q=q, existing=known.get(M["key"]), subsets=M["subsets"])); continue
    adj = [[] for _ in Zl]
    for b in mol.GetBonds(): i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[i].append(j); adj[j].append(i)
    Za = np.array(Zl); sig = E.signatures(Za, adj); types = list(M["types"]); levels = list(M["levels"])
    if len(types) != len(Zl): skipped["atom order mismatch"] += 1; continue
    for i, t in enumerate(types):
        if not t:
            nb = "".join(sorted(EL.get(int(Za[j]), "?") for j in adj[i]))
            k = (EL.get(int(Za[i]), str(Za[i])), len(adj[i]), sig[7][i], sig[0][i].split(".")[3] != "0", nb)
            if k not in newtypes: newtypes[k] = f"XN{k[0].upper()}{sum(1 for x in newtypes if x[0] == k[0]) + 1}"
            types[i] = newtypes[k]
    qq, qlev, resid = CH.predict(sig, types, q)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]
    out.append(dict(smiles=smi, formula=zf, n=int(M["n"] * scale), Q=q, existing=None, subsets=M["subsets"], Z=Zl, types=types, levels=levels,
                    charges=[round(float(x), 4) for x in qq], charge_levels=qlev.tolist(), resid=float(resid), bonds=bonds, R=ex["R"]))
new = [o for o in out if o["existing"] is None]
for k, o in enumerate(new): o["resi"] = f"OM{k:02d}" if k < 100 else f"O{k:03d}"
# RTF
L = ["* OMol25 auto residues (envtyper v3 types, EnvCharger charges); generated by mlmm_ft/residues/build_residues.py",
     "* provisional types XN* need LJ fitting (OMol25 dimers) before production use", "*", "36 1", ""]
for k, ((el, deg, nh, ring, nb), name) in enumerate(newtypes.items()):
    L.append(f"MASS  -1  {name:6s} {Chem.GetPeriodicTable().GetAtomicWeight(el.capitalize()):9.5f} {el.capitalize()} ! provisional: {el} deg{deg} nH{nh} {'ring' if ring else 'acyclic'} nbrs {nb}")
L.append("")
for o in new:
    cnt = collections.Counter(); names = []
    for z in o["Z"]: e = EL.get(int(z), "X").upper(); cnt[e] += 1; names.append(f"{e}{cnt[e]}"[:4])
    L.append(f"RESI {o['resi']:6s} {o['Q']:6.2f} ! {o['formula']} {o['smiles']}  (~{o['n']} occurrences in OMol25 4M)")
    L.append("GROUP")
    for nm, t, q, lv in zip(names, o["types"], o["charges"], o["levels"]):
        L.append(f"ATOM {nm:5s} {t:7s} {q:8.4f}" + ("  ! provisional type" if t.startswith("XN") else "  ! low-confidence type" if lv in (0, 5, 6) else ""))
    for i, j, bo in o["bonds"]:
        L.append(f"{'DOUBLE' if bo == 2 else 'TRIPLE' if bo == 3 else 'BOND  '} {names[i]:5s} {names[j]:5s}")
    L.append("")
L.append("END")
open(pre + ".rtf", "w").write("\n".join(L))
json.dump(dict(min_occ=MIN, scale=scale, molecules=out, newtypes=[dict(name=v, element=k[0], degree=k[1], nH=k[2], ring=k[3], neighbours=k[4]) for k, v in newtypes.items()],
               skipped=dict(skipped)), open(pre + ".json", "w"), default=int)
ex = [o for o in out if o["existing"]]
print(f"{len(out)+sum(skipped.values())} graphs >= {MIN} (sample counts); existing CHARMM RESI {len(ex)}, new residues {len(new)}, skipped {dict(skipped)}; provisional new types {len(newtypes)}")
print("existing:", [(o["formula"], o["existing"]) for o in ex[:25]])
print("new types:", list(newtypes.values())); print("hypervalent (connectivity-only):", hyper_used)
for o in new[:30]: print(f"  {o['resi']} n~{o['n']:7d} Q{o['Q']:+d} {o['smiles']:28s} lowconf {sum(l in (0,5,6) for l in o['levels'])} prov {sum(t.startswith('XN') for t in o['types'])} |resid| {abs(o['resid']):.2f}")
