"""Write fitted charges into ../omol_auto.rtf (backup kept) and ../omol_auto.json (charges_fit, charge_method)."""
import json, re, shutil, sys, numpy as np
sys.argv = ["x", "out2"]; import fit_hybrid as F
hp = json.load(open("hybrid_cv.json"))["final_hp"]; WH = hp["w"]; WN = [0.1, 0.01, 1e-5, 1e-5]
D = json.load(open("../omol_auto.json")); new = {o["resi"]: o for o in D["molecules"] if not o["existing"]}
fit = {}; meth = {}
for n, o in new.items():
    p = F.prep(n)
    if p is None: fit[n] = o["charges"]; meth[n] = "env-transfer (no ESP)"; continue
    prov = any(t.startswith("XN") for t in o["types"]); hyper = n in ("OM51", "OM60") or any(z in (15, 16, 17) and sum(1 for b in o["bonds"] if n_ in b[:2]) >= 4 for n_, z in [])
    back = (p["g"] >= 2).mean() > 0.3; novel = prov or abs(o["resid"]) > 0.2 or back or len(o["Z"]) == 1
    q = F.solve(p, 1.0, WN) if novel else F.solve(p, hp["lam"], WH)
    if o["formula"] in ("ClO4", "F6P", "O4P", "F2NO4S2"):     # hypervalent anions: analogues meaningless, shell-degenerate ESP -> plain RESP
        q = np.array(json.load(open(f"out2/{n}.json"))["q"]); novel = "resp"
    q = np.round(q, 4); q[np.argmax(np.abs(q))] += round(o["Q"] - q.sum(), 4); fit[n] = q.tolist()
    meth[n] = "RESP (HF/6-31+G*, hyperbolic restraint)" if novel == "resp" else "ESP (HF/6-31G*, RESP-type) restrained to r2/r1 analogues" if novel else "env-transfer + ESP restraint (CV hybrid)"
L = open("../omol_auto.rtf").read().split("\n"); out = []; cur = None; k = 0
for l in L:
    m = re.match(r"RESI (\S+)", l)
    if m: cur = m.group(1); k = 0; l = l + f"  charges: {meth.get(cur, '')}" if cur in meth else l
    elif l.startswith("ATOM") and cur in fit:
        t = l.split(); rest = l.split("!", 1)[1] if "!" in l else ""
        l = f"ATOM {t[1]:5s} {t[2]:7s} {fit[cur][k]:8.4f}" + (f"  !{rest}" if rest else ""); k += 1
    out.append(l)
open("../omol_auto.rtf", "w").write("\n".join(out))
for o in D["molecules"]:
    if o.get("resi") in fit: o.setdefault("charges_env", o["charges"]); o["charges"] = fit[o["resi"]]; o["charge_method"] = meth[o["resi"]]
json.dump(D, open("../omol_auto.json", "w"), default=int)
import collections; print(collections.Counter(meth.values())); print("PF6", fit.get("OM60")); print("EC", fit.get("OM06"))
