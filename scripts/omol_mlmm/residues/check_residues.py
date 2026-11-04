"""Self-consistency: type each auto residue's OMol25 example geometry with typer v3 + the auto RTF; compare with RESI types."""
import json, sys, os, numpy as np
sys.path.insert(0, "../envtyper"); import cgenff_envtyper as E
T = E.CGenFFTyper(extra=os.path.abspath("omol_auto.rtf")); D = json.load(open("omol_auto.json"))
tot = ok = 0; bad = []
for o in D["molecules"]:
    if o["existing"]: continue
    Z = np.array(o["Z"]); adj = E.graph_from_xyz(Z, np.array(o["R"])); a = T.assign(E.signatures(Z, adj))
    p = [x[0] or "" for x in a]; tot += len(p); k = sum(x == y for x, y in zip(p, o["types"])); ok += k
    if k < len(p): bad.append((o["resi"], o["smiles"], [(x, y) for x, y in zip(p, o["types"]) if x != y][:4]))
print(f"auto residues: {ok}/{tot} atoms re-typed identically ({100*ok/tot:.2f} %); residues with any mismatch {len(bad)}"); print(bad[:10])
json.dump(dict(atoms=tot, identical=ok, mismatched_residues=bad), open("check_residues.json", "w"))
