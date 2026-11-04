"""Type every atom of an exported structure set with the CGenFF environment typer; parallel over chunks.
    python type_cache.py <structures.npz> <out.npz> [n_max] [workers]"""
import sys, time, json, numpy as np
from multiprocessing import Pool
import cgenff_envtyper as E
TY = None
def init():
    global TY; TY = E.CGenFFTyper()
def chunk(args):
    Zs, Rs, off = args; T = []; L = []; P = []
    for k in range(len(off) - 1):
        Z = Zs[off[k]:off[k + 1]].astype(int); R = Rs[off[k]:off[k + 1]].astype(float)
        adj = E.graph_from_xyz(Z, R); a = TY.assign(E.signatures(Z, adj))
        T += [t or "" for t, _, _ in a]; L += [l for _, l, _ in a]; P += [p for _, _, p in a]
    return T, L, P
if __name__ == "__main__":
    S = np.load(sys.argv[1], allow_pickle=False); Zs, Rs, off = S["Z"], S["R"], S["offsets"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) > 0 else len(off) - 1; W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    cs = 500; jobs = [(Zs[off[i]:off[min(i + cs, n)]], Rs[off[i]:off[min(i + cs, n)]], off[i:min(i + cs, n) + 1] - off[i]) for i in range(0, n, cs)]
    t0 = time.time(); T, L, P = [], [], []
    with Pool(W, initializer=init) as pool:
        for k, (t, l, p) in enumerate(pool.imap(chunk, jobs)):
            T += t; L += l; P += p
            if k % 50 == 0: print(f"chunk {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    names = sorted(set(T)); idx = {t: i for i, t in enumerate(names)}
    np.savez(sys.argv[2], type_idx=np.array([idx[t] for t in T], np.int16), type_names=np.array(names), level=np.array(L, np.int8),
             purity=np.array(P, np.float16), offsets=off[:n + 1])
    L = np.array(L); print(f"typed {n} structures / {len(T)} atoms in {dt:.1f}s ({1e3 * dt / n:.2f} ms/structure, {W} workers); "
          f"level r2 {np.mean(L == 2):.3f} r1 {np.mean(L == 1):.3f} r0 {np.mean(L == 0):.3f} fallback-lib {np.mean(L >= 10):.4f} untyped {np.mean(L < 0):.4f}; {len(names)} types")
