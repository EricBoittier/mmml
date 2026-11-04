"""Re-type auto residues with the current typer (+ auto library for XN atoms); patch omol_auto.json/rtf ATOM types."""
import json, os, re, sys, numpy as np
sys.path.insert(0, "../envtyper"); import cgenff_envtyper as E
T = E.CGenFFTyper(extra=os.path.abspath("omol_auto.rtf")); D = json.load(open("omol_auto.json")); ch = {}
for o in D["molecules"]:
    if o["existing"]: continue
    Z = np.array(o["Z"]); a = T.assign(E.signatures(Z, E.graph_from_xyz(Z, np.array(o["R"]))))
    new = [t or old for (t, _, _), old in zip(a, o["types"])]
    diff = [(i, o["types"][i], new[i]) for i in range(len(new)) if new[i] != o["types"][i]]
    if diff: ch[o["resi"]] = diff; o["types"] = new
json.dump(D, open("omol_auto.json", "w"), default=int)
out = []; cur = None; k = 0
for l in open("omol_auto.rtf").read().split("\n"):
    m = re.match(r"RESI (\S+)", l)
    if m: cur = m.group(1); k = 0
    elif l.startswith("ATOM") and cur:
        if cur in ch:
            for i, a, b in ch[cur]:
                if i == k: t = l.split(); l = l.replace(f" {a} ", f" {b} ".ljust(len(a) + 2), 1) + "  ! v3.1 polar H"
        k += 1
    out.append(l)
open("omol_auto.rtf", "w").write("\n".join(out))
print(len(ch), "residues changed:", {k: v for k, v in list(ch.items())[:12]})
