import json, sys
from rdkit import Chem; from rdkit.Chem import Draw, AllChem
D = json.load(open("omol_auto.json")); new = [o for o in D["molecules"] if not o["existing"]]
ms, leg, hl = [], [], []
for o in new[:60]:
    m = Chem.MolFromSmiles(o["smiles"])
    if m is None: continue
    # highlight heavy atoms whose residue type is provisional (red) -- map by element order is unsafe, so mark by SMILES element match count only
    prov = {i for i, a in enumerate(m.GetAtoms())} if all(t.startswith("XN") for t, z in zip(o["types"], o["Z"]) if z > 1) and any(t.startswith("XN") for t in o["types"]) else set()
    ms.append(m); hl.append(sorted(prov)); leg.append(f"{o['resi']}  {o['formula']}  ~{o['n']:,}" + ("  (not MM-valid)" if o.get("mm_valid") is False else ""))
img = Draw.MolsToGridImage(ms, molsPerRow=6, subImgSize=(260, 200), legends=leg, highlightAtomLists=hl, useSVG=False)
img.save(sys.argv[1]); print(len(ms), "drawn")
