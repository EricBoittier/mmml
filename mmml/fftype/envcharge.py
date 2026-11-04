"""CGenFF-style charge transfer from the CHARMM RTF libraries by connectivity environment (same signatures as the typer).
q_i = mean RTF charge of the most specific signature seen (r2 > r1 > r1b > r1c > r0 > r0c), H atoms additionally via
(parent type, parent nH) after r1. The residual to the integer net charge is spread over atoms with weight by level
(well-matched atoms move least). This mirrors how CGenFF itself transfers charges (by analogy to parametrised fragments),
NOT a QM fit.  LORO (leave one RESI out) gives the transfer error."""
import collections, json, sys, numpy as np, cgenff_envtyper as E
W = {2: 1.0, 1: 2.0, 3: 2.0, 4: 4.0, 5: 6.0, 0: 8.0, 6: 10.0, -1: 10.0}
class EnvCharger:
    def __init__(self, residues, mass):
        self.mass = mass; self.tab = {k: collections.defaultdict(list) for k in (0, 1, 2, 3, 4, 5)}; self.htab = collections.defaultdict(list)
        self.res = {}
        for r in residues:
            if not r["atoms"]: continue
            Z, adj, stubs = E.residue_graph(r, mass); sig = E.signatures(Z, adj, stubs)
            tys = [r["types"][a] for a in r["atoms"]]; q = [r["charges"][a] for a in r["atoms"]]
            self.res[(r["file"], r["name"])] = (sig, tys, q, any(stubs))
            self._upd(sig, tys, q, +1)
    def _upd(self, sig, tys, q, w):
        for k in self.tab:
            for s, x in zip(sig[k], q):
                if w > 0: self.tab[k][s].append(x)
                else: self.tab[k][s].remove(x); (not self.tab[k][s]) and self.tab[k].pop(s)
        for i, x in enumerate(q):
            p = sig[6][i]
            if p >= 0:
                key = (tys[p], sig[7][p])
                if w > 0: self.htab[key].append(x)
                else: self.htab[key].remove(x); (not self.htab[key]) and self.htab.pop(key)
    def predict(self, sig, types, net):
        n = len(sig[0]); q = np.zeros(n); lev = np.full(n, -1)
        order = ((2, 2), (1, 1), (3, 4), (4, 5), (0, 0), (5, 6))      # (sig index, level code)
        for i in range(n):
            for k, code in order[:2]:
                v = self.tab[k].get(sig[k][i])
                if v: q[i] = np.mean(v); lev[i] = code; break
            else:
                p = sig[6][i]; v = self.htab.get((types[p], sig[7][p])) if p >= 0 and types[p] else None
                if v: q[i] = np.mean(v); lev[i] = 3; continue
                for k, code in order[2:]:
                    v = self.tab[k].get(sig[k][i])
                    if v: q[i] = np.mean(v); lev[i] = code; break
        w = np.array([W[int(l)] for l in lev]); resid = net - q.sum(); q += resid * w / w.sum()
        return q, lev, resid
if __name__ == "__main__":
    mass, res = E.parse_rtfs(); prim = [r for r in res if r["file"] in E.PRIMARY]
    C = EnvCharger(prim, mass); TY = E.EnvTyper(prim, mass)
    # baseline: mean charge per CGenFF type (what you get from atom types alone)
    tq = collections.defaultdict(list)
    for sig, tys, q, st in C.res.values():
        for t, x in zip(tys, q): tq[t].append(x)
    err = []; errt = []; errlev = collections.defaultdict(list); resid = []; dip = []
    for key, (sig, tys, q, st) in list(C.res.items()):
        if st or len(q) < 2: continue                    # skip residues with inter-residue stubs (not complete molecules)
        C._upd(sig, tys, q, -1); TY.remove(key)
        ty = [t or "" for t, _, _ in TY.assign(sig)]
        p, lev, r = C.predict(sig, ty, round(sum(q)))
        C._upd(sig, tys, q, +1); TY.add(key)
        q = np.array(q); err += list(np.abs(p - q)); resid.append(abs(r))
        for l, e in zip(lev, np.abs(p - q)): errlev[int(l)].append(e)
        bt = np.array([np.mean(tq[t]) for t in tys]); bt += (round(q.sum()) - bt.sum()) / len(bt); errt += list(np.abs(bt - q))
    out = dict(n_atoms=len(err), mae=float(np.mean(err)), rmse=float(np.sqrt(np.mean(np.square(err)))), p95=float(np.percentile(err, 95)),
               mae_type_mean_baseline=float(np.mean(errt)), residual_before_neutralise_mean=float(np.mean(resid)),
               by_level={str(k): [len(v), float(np.mean(v))] for k, v in sorted(errlev.items())})
    json.dump(out, open("envcharge_loro.json", "w"), indent=1); print(json.dumps(out, indent=1))
