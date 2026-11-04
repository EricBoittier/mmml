"""CGenFF atom typing from local connectivity environments learned from the CHARMM topology libraries (v3.1, 4 Nov 2026).

Topology files: $MMML_TOPPAR, else <repo>/setup/charmm/toppar. OMol25 auto-residue library: mmml/fftype/data/omol_auto.rtf
(pass CGenFFTyper(extra=OMOL_AUTO_RTF)).

Every atom is described by an invariant (element, degree, n_H, smallest ring size<=8 or 0). Signatures:
  r0 = invariant; r1 = r0 + sorted neighbour invariants; r2 = r1 + per neighbour its sorted other-neighbour invariants.
A table maps each signature to a Counter of CGenFF/CHARMM types seen in the RTF residues. Typing uses the largest radius
present (r2 > r1 > r0) and the majority type; purity = majority share. Inter-residue bonds (+X/-X) count as stub
neighbours (element from the name). Connectivity only: no bond orders or formal charges needed.
"""
import collections, os, re
from pathlib import Path
import numpy as np

_REPO_TOP = Path(__file__).resolve().parents[2] / "setup" / "charmm" / "toppar"
TOP = Path(os.environ.get("MMML_TOPPAR", _REPO_TOP if _REPO_TOP.exists() else "/mmhome/boittier/home/mmml/setup/charmm/toppar"))
RTFS = ["top_all36_cgenff.rtf", "top_all36_prot.rtf", "top_all36_na.rtf", "top_all36_lipid.rtf", "top_all36_carb.rtf",
        "top_all35_ethers.rtf", "toppar_water_ions.str"]
SYM = {"H": 1, "LI": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16,
       "CL": 17, "K": 19, "CA": 20, "ZN": 30, "BR": 35, "I": 53, "RB": 37, "CS": 55, "BA": 56, "CD": 48, "SE": 34}

def parse_rtfs(files=RTFS):
    mass = {}; residues = []
    for fn in files:
        cur = None; on = True
        for raw in open(TOP / fn, errors="ignore"):
            l = raw.split("!")[0].strip()
            if not l: continue
            t = l.split(); k = t[0].upper()
            if k == "MASS" and len(t) >= 4:
                el = t[4].upper() if len(t) >= 5 else re.sub(r"[^A-Z]", "", t[2].upper())[:2]
                mass[t[2]] = SYM.get(el, SYM.get(el[:1], 0))
            elif k in ("RESI", "PRES"):
                cur = None
                if k == "RESI":
                    cur = dict(name=t[1], file=fn, atoms=[], types={}, charges={}, bonds=[]); residues.append(cur)
            elif cur is None: continue
            elif k == "ATOM" and len(t) >= 4:
                cur["atoms"].append(t[1]); cur["types"][t[1]] = t[2]; cur["charges"][t[1]] = float(t[3])
            elif k in ("BOND", "DOUBLE", "DOUB", "TRIPLE", "TRIP"):
                for a, b in zip(t[1::2], t[2::2]): cur["bonds"].append((a, b))
            elif k in ("END",): cur = None
    # CHARMM ion types: element not parseable from their MASS lines -> pin explicitly
    for t, z in dict(SOD=11, POT=19, CLA=17, LIT=3, MG=12, CAL=20, ZN2=30, CES=55, RUB=37, BAR=56, CAD=48, CD2=48).items():
        if t in mass: mass[t] = z
    return mass, residues

def residue_graph(res, mass):
    """atoms (types, Z) + adjacency incl. stub neighbours for +X/-X/#X inter-residue bonds."""
    names = res["atoms"]; idx = {n: i for i, n in enumerate(names)}
    Z = [mass.get(res["types"][n], 0) for n in names]; adj = [[] for _ in names]; stubs = [[] for _ in names]
    for a, b in res["bonds"]:
        ia, ib = idx.get(a), idx.get(b)
        if ia is not None and ib is not None:
            adj[ia].append(ib); adj[ib].append(ia)
        elif ia is not None or ib is not None:
            i, other = (ia, b) if ia is not None else (ib, a)
            el = re.sub(r"[^A-Z]", "", other.upper())[:1]; stubs[i].append(SYM.get(el, 6))
    return np.array(Z), adj, stubs

def ring_info(n, adj):
    """smallest ring (<=8) per atom via BFS: (size, member set) -- members from the two BFS branches."""
    size = np.zeros(n, int); members = [()] * n
    for s in range(n):
        best = None; dist = {s: 0}; par = {s: -1}; br = {s: s}; q = [s]   # br = first hop from s (cycle must pass through s)
        for u in q:
            if dist[u] > 4: break
            for v in adj[u]:
                if v == par[u]: continue
                if v in dist:
                    if br[u] == br[v]: continue
                    L = dist[u] + dist[v] + 1
                    if 3 <= L <= 8 and (best is None or L < best[0]):
                        path = set()
                        for w in (u, v):
                            while w != -1: path.add(w); w = par[w]
                        best = (L, tuple(sorted(path)))
                else:
                    dist[v] = dist[u] + 1; par[v] = u; br[v] = v if u == s else br[u]; q.append(v)
        if best: size[s], members[s] = best
    return size, members

def signatures(Z, adj, stubs=None):
    n = len(Z); stubs = stubs or [[] for _ in range(n)]; rs, mem = ring_info(n, adj)
    deg = [len(adj[i]) + len(stubs[i]) for i in range(n)]
    nH = [sum(Z[j] == 1 for j in adj[i]) + sum(z == 1 for z in stubs[i]) for i in range(n)]
    # ring descriptors: heteroatoms in the smallest ring; 'u' if every ring atom is unsaturated (C deg<=3, N deg<=2..3)
    def rdesc(i):
        if not rs[i]: return "0"
        m = mem[i]; het = sum(Z[k] not in (1, 6) for k in m)
        unsat = all((Z[k] == 6 and deg[k] <= 3) or (Z[k] == 7 and deg[k] <= 3) or (Z[k] in (8, 16) and deg[k] <= 2) for k in m)
        if rs[i] >= 7 and not unsat: return "0"      # CGenFF has no saturated 7/8-ring types: acyclic types apply
        return f"{rs[i]}h{het}{'u' if unsat else 's'}"
    inv = [f"{Z[i]}.{deg[i]}.{nH[i]}.{rdesc(i)}" for i in range(n)]
    stub_inv = [[f"{z}.?" for z in stubs[i]] for i in range(n)]
    r0 = inv
    r1 = [inv[i] + "|" + ",".join(sorted([inv[j] for j in adj[i]] + stub_inv[i])) for i in range(n)]
    r2 = []
    for i in range(n):
        shells = sorted(inv[j] + "(" + ",".join(sorted([inv[k] for k in adj[j] if k != i] + stub_inv[j])) + ")" for j in adj[i])
        r2.append(r1[i] + "|" + ";".join(shells + sorted(stub_inv[i])))
    # back-off keys: r1b = centre + neighbours as Z.deg; r1c = centre + neighbour elements; r0c = centre without ring
    # descriptor; parent = bonded heavy atom for H (CGenFF H types follow the parent type), pnH = parent's H count
    cinv = [[f"{Z[j]}.{deg[j]}" for j in adj[i]] + [f"{z}.?" for z in stubs[i]] for i in range(n)]
    r1b = [inv[i] + "|" + ",".join(sorted(cinv[i])) for i in range(n)]
    r1c = [inv[i] + "|" + ",".join(sorted(x.split(".")[0] for x in cinv[i])) for i in range(n)]
    r0c = [f"{Z[i]}.{deg[i]}.{nH[i]}" for i in range(n)]
    parent = [adj[i][0] if Z[i] == 1 and len(adj[i]) == 1 else -1 for i in range(n)]
    return r0, r1, r2, r1b, r1c, r0c, parent, nH

class EnvTyper:
    def __init__(self, residues=None, mass=None, exclude=()):
        if residues is None: mass, residues = parse_rtfs()
        self.mass = mass; self.tab = [collections.defaultdict(collections.Counter) for _ in range(3)]
        self.xtab = {k: collections.defaultdict(collections.Counter) for k in (3, 4, 5)}   # sig index -> table (r1b, r1c, r0c)
        self.htab = collections.defaultdict(collections.Counter)                          # (parent type, parent nH) -> H type
        self.res_sigs = {}
        for r in residues:
            if not r["atoms"]: continue
            Z, adj, stubs = residue_graph(r, mass); sig = signatures(Z, adj, stubs)
            tys = [r["types"][a] for a in r["atoms"]]; self.res_sigs[(r["file"], r["name"])] = (sig, tys)
            if (r["file"], r["name"]) in exclude: continue
            self._update(sig, tys, +1)
    LEVELS = {2: "r2", 1: "r1", 3: "H<-parent", 4: "r1b (nbr Z.deg)", 5: "r1c (nbr Z)", 0: "r0", 6: "r0c (no ring)"}
    def _update(self, sig, tys, w):
        def bump(c, key, t):
            c[key][t] += w
            if c[key][t] <= 0: del c[key][t]
            if not c[key]: del c[key]
        for lev in range(3):
            for s, t in zip(sig[lev], tys): bump(self.tab[lev], s, t)
        for k in (3, 4, 5):
            for s, t in zip(sig[k], tys): bump(self.xtab[k], s, t)
        par, nH = sig[6], sig[7]
        for i, t in enumerate(tys):
            if par[i] >= 0: bump(self.htab, (tys[par[i]], nH[par[i]]), t)
    def assign(self, sig):
        """per atom (type, level, purity); levels in precedence order 2, 1, 3 (H from parent type), 4, 5, 0, 6; -1 untyped."""
        n = len(sig[0]); out = [None] * n
        def look(tab, key):
            c = tab.get(key)
            if c: t, k = c.most_common(1)[0]; return t, k / sum(c.values())
        for i in range(n):
            for lev in (2, 1):
                r = look(self.tab[lev], sig[lev][i])
                if r: out[i] = (r[0], lev, r[1]); break
        par, nH = sig[6], sig[7]
        def heavy_rest(i):
            for lev, tab, key in ((4, self.xtab[3], sig[3][i]), (5, self.xtab[4], sig[4][i]), (0, self.tab[0], sig[0][i]), (6, self.xtab[5], sig[5][i])):
                r = look(tab, key)
                if r: return (r[0], lev, r[1])
            return (None, -1, 0.0)
        for i in range(n):                       # heavy atoms (and lone H) first, so H can use the parent's type
            if out[i] is None and par[i] < 0: out[i] = heavy_rest(i)
        for i in range(n):
            if out[i] is None:
                pt = out[par[i]][0] if out[par[i]] is not None else None
                r = look(self.htab, (pt, nH[par[i]])) if pt else None
                out[i] = (r[0], 3, r[1]) if r else heavy_rest(i)
        return out
    def remove(self, key):
        sig, tys = self.res_sigs[key]; self._update(sig, tys, -1)
    def add(self, key):
        sig, tys = self.res_sigs[key]; self._update(sig, tys, +1)

COV = {1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 11: 1.66, 12: 1.41, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
       19: 2.03, 20: 1.76, 30: 1.22, 35: 1.20, 53: 1.39}
METALS = {3, 11, 12, 19, 20, 30}
def graph_from_xyz(Z, R, scale=1.2):
    """covalent graph from coordinates (same rule as the fragment census); metal-ligand contacts are not bonds."""
    from scipy.spatial import cKDTree
    n = len(Z); adj = [[] for _ in range(n)]
    for i, j in cKDTree(R).query_pairs(2.6):
        if Z[i] in METALS or Z[j] in METALS: continue
        if np.linalg.norm(R[i] - R[j]) < scale * (COV.get(int(Z[i]), 0.8) + COV.get(int(Z[j]), 0.8)):
            adj[i].append(j); adj[j].append(i)
    return adj

PRIMARY = ("top_all36_cgenff.rtf", "toppar_water_ions.str")
class CGenFFTyper:
    """CGenFF + water/ions first; other CHARMM libraries only where CGenFF has no match at any radius."""
    def __init__(self, extra=None):
        """extra: path of an auto-residue RTF (OMol25 residues + provisional XN* types), consulted only for atoms that neither
        CGenFF nor the other CHARMM libraries can type (level 20+lev)."""
        self.mass, res = parse_rtfs()
        self.prim = EnvTyper([r for r in res if r["file"] in PRIMARY], self.mass)
        self.fall = EnvTyper([r for r in res if r["file"] not in PRIMARY], self.mass)
        self.extra = None
        if extra:
            m2, r2 = parse_rtfs([extra]); m2 = {**self.mass, **m2}; self.extra = EnvTyper(r2, m2)
    def assign(self, sig):
        a = self.prim.assign(sig); b = None
        out = []
        for i, (t, lev, pur) in enumerate(a):
            if t is None:
                b = b or self.fall.assign(sig); t2, l2, p2 = b[i]
                if t2 is None and self.extra is not None:
                    c = getattr(self, "_c", None)
                    if c is None or c[0] is not sig: self._c = c = (sig, self.extra.assign(sig))
                    t3, l3, p3 = c[1][i]
                    if t3 is not None: out.append((t3, 20 + l3, p3)); continue
                out.append((t2, -2 if t2 is None else 10 + l2, p2))    # 10+lev marks a fallback-library type
            else: out.append((t, lev, pur))
        # v3.1: H on a polar heavy atom that was only typed by back-off must be a polar H (CHARMM has no H on halogens;
        # the element-level majority HGA3/HGR6x is an aliphatic H with Rmin/2 1.34 A -> wrong H-bond repulsion)
        for i, (t, lev, pur) in enumerate(out):
            p = sig[6][i]
            if p >= 0 and lev not in (1, 2) and t and t[:3] in ("HGA", "HGR", "HCA"):
                zp = int(sig[0][p].split(".")[0])
                if zp in POLAR_H: out[i] = (POLAR_H[zp], 7, 1.0)
        return out
POLAR_H = {7: "HGP1", 8: "HGP1", 9: "HGP1", 17: "HGP1", 35: "HGP1", 53: "HGP1", 16: "HGP3"}

OMOL_AUTO_RTF = str(Path(__file__).resolve().parent / "data" / "omol_auto.rtf")
