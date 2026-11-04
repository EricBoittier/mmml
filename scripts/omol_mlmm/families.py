"""Chemical families of the remaining typing gaps (top-300 environments per class) -> gap_families.json"""
import json, re, sys, collections
d = json.load(open(sys.argv[1])); A = d["atoms"]
fam = [("free halide / X- anion", lambda c, n: re.match(r"(9|17|35|53)\.0\.", c)),
 ("small molecules (H2, CO, H2S, HCl, H3O+)", lambda c, n: (c in ("1.1.1.0", "6.1.0.0", "8.1.0.0") and n in ("1.1.1.0", "8.1.0.0", "6.1.0.0")) or c.startswith(("16.2.2.0", "17.1.1.0", "8.3.3.0", "6.4.4.0")) or (c == "1.1.0.0" and n.startswith(("16.2.2", "17.1.1", "8.3.3", "6.4.4")))),
 ("PF6- / BF4- / ClO4- type anions", lambda c, n: c.startswith(("15.6.", "5.4.0.0", "17.4.", "33.6")) or re.search(r"(^|,)(15\.6|5\.4\.0\.0|17\.4|33\.6)", n)),
 ("sulfonyl imides FSI / TFSI", lambda c, n: (".3h" in c and ("16." in c or "16." in n)) or (c.startswith("7.2.0") and n.count("16.") >= 2)),
 ("cyclic sulfate / sultone / sulfolane", lambda c, n: "16.4.0.5h" in c + n or "16.4.0.6h" in c + n),
 ("cyclic carbonates EC / FEC / VC", lambda c, n: "5h2" in c and ("8.1.0.0" in n or "8.2.0.5h2" in n)),
 ("phosphorus species", lambda c, n: c.startswith("15.") or n.startswith("15.") or ",15." in n),
 ("open-shell / under-coordinated (radicals, arynes, carbanions)", lambda c, n: re.match(r"6\.(1|2)\.0\.6h|6\.3\.3\.0|7\.2\.2\.0|7\.1\.|16\.1\.1|8\.1\.1|6\.2\.[12]\.0|6\.3\.[12]\.0\|1", c)),
 ("organohalogen", lambda c, n: re.search(r"(^|,)(9|17|35|53)\.1", n) or re.match(r"(9|17|35|53)\.1", c))]
res = {}
for k in ("untyped", "r0"):
    C = collections.Counter(); cov = 0
    for e in d[k]:
        c, n = e["env"].split("|"); cov += e["atoms"]
        C[next((nm for nm, f in fam if f(c, n)), "other (long tail)")] += e["atoms"]
    tot = d[k + "_total"]; res[k] = dict(total=tot, pct_atoms=100 * tot / A, top300_cover=100 * cov / tot, fam=dict(C.most_common()))
    print(k, round(100 * tot / A, 2), {a: round(100 * b / tot, 1) for a, b in C.most_common()})
json.dump(res, open("gap_families.json", "w"), indent=1)
