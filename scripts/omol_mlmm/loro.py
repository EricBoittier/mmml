"""Leave-one-residue-out validation of the environment typer on the CGenFF RTF (primary table only)."""
import collections, json, numpy as np, cgenff_envtyper as E
mass, res = E.parse_rtfs(); prim = [r for r in res if r["file"] in E.PRIMARY]
T = E.EnvTyper(prim, mass)
tot = collections.Counter(); ok = collections.Counter(); el_tot = collections.Counter(); el_ok = collections.Counter(); fails = collections.Counter()
for key, (sig, tys) in list(T.res_sigs.items()):
    T.remove(key); pred = T.assign(sig); T.add(key)
    Z = [mass.get(t, 0) for t in tys]
    for (p, lev, pur), t, z in zip(pred, tys, Z):
        tot[lev] += 1; ok[lev] += p == t; el_tot[z] += 1; el_ok[z] += p == t
        if p != t: fails[(t, p)] += 1
n = sum(tot.values()); acc = sum(ok.values()) / n
out = {"atoms": n, "acc": acc, "by_level": {str(k): [tot[k], ok[k] / max(tot[k], 1)] for k in sorted(tot)},
       "by_element": {str(z): [el_tot[z], el_ok[z] / el_tot[z]] for z in sorted(el_tot)}, "top_confusions": [[a, b, c] for (a, b), c in fails.most_common(15)]}
json.dump(out, open("loro_results.json", "w"), indent=1); print(json.dumps(out, indent=0)[:1800])
