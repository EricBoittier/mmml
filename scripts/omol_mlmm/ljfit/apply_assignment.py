"""Replace provisional XN* types by the assigned existing CHARMM/CGenFF types (lj_fit.json) in ../omol_auto.rtf/json.
Keeps the provisional version as ../omol_auto.provisional.rtf; MASS XN lines are dropped (only CHARMM types remain)."""
import json, re, shutil
import sys; sys.path.insert(0, "../../envtyper"); sys.path.insert(0, "../../scripts"); import ljtab, cgenff_envtyper as E, build_mlmm_cache as B
F = json.load(open("lj_fit.json")); A = dict(F["assignment"]); info = F["types"]; LJ = ljtab.lj_table(); mass, _ = E.parse_rtfs()
INVALID = {"XNC4": 6, "XNC5": 6, "XNN1": 7, "XNO1": 8, "XNP3": 15, "XNP4": 15, "XNP5": 15, "XNP6": 15, "XNP7": 15, "XNP8": 15, "XNS3": 16}
for t, z in INVALID.items():      # nearest existing type (same element) to the element-default LJ
    e0, r0 = (lambda i: (B.PT[i][2], B.PT[i][1]))(B.ZMAP.get(z, 17))
    A[t] = min((c for c in LJ if mass.get(c) == z), key=lambda c: abs(LJ[c][1] - r0) + abs(LJ[c][0] - e0))
INV_RESI = set()
shutil.copy("../omol_auto.rtf", "../omol_auto.provisional.rtf"); shutil.copy("../omol_auto.json", "../omol_auto.provisional.json")
out = []; cur_resi = None
for l in open("../omol_auto.rtf").read().split("\n"):
    m = re.match(r"MASS\s+-1\s+(XN\S+)", l)
    if m: continue
    mr = re.match(r"RESI (\S+)", l)
    if mr: cur_resi = mr.group(1)
    if l.startswith("ATOM") and l.split()[2] in INVALID: INV_RESI.add(cur_resi)
    if l.startswith("ATOM"):
        t = l.split()
        if t[2] in A and A[t[2]]:
            l = f"ATOM {t[1]:5s} {A[t[2]]:7s} {float(t[3]):8.4f}  ! assigned from dimer fit (was {t[2]})"
    out.append(l)
out = [l + "  ! NOT MM-VALID: reactive gas-phase species (types assigned for labelling only)" if (re.match(r"RESI (\S+)", l) and re.match(r"RESI (\S+)", l).group(1) in INV_RESI) else l for l in out]
hdr = "* XN provisional types replaced by existing CHARMM/CGenFF types chosen by fit to wB97M-V/def2-TZVPD dimer scans (ljfit/)"
out.insert(1, hdr); open("../omol_auto.rtf", "w").write("\n".join(out))
D = json.load(open("../omol_auto.json"))
for o in D["molecules"]:
    if not o["existing"]:
        if any(t in INVALID for t in o["types"]): o["mm_valid"] = False
        o["types_provisional"] = o["types"]; o["types"] = [A.get(t) or t for t in o["types"]]
D["newtypes_assigned"] = A; json.dump(D, open("../omol_auto.json", "w"), default=int)
left = sum(1 for o in D["molecules"] if not o["existing"] for t in o["types"] if t.startswith("XN"))
print("assigned", {k: v for k, v in A.items()}, "remaining XN atoms", left)
