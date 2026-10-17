#!/usr/bin/env python
"""Structure at the model's own density: O–O g(r), coordination to 3.5 Å and X-ray S(Q) from the last part of an NpT
trajectory (h5 with positions and density_g_cm3 per frame; the cubic box side is recovered from the density).
Compares with the fixed-volume (experimental density) numbers and with the digitized X-ray curve of the composition.

  python wm_npt_structure.py OUT --runs TAG=NPT_RUN_DIR ... --box TAG=BOX.json ... [--last-ps 25] [--tag _20261016]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import h5py
sys.path.insert(0, str(Path.home() / "handoff/handoff_large_20261014/scripts")); sys.path.insert(0, str(Path.home() / "mmml_dev"))
import sq_xray as sx                                        # noqa: E402

DG = Path.home() / "handoff/handoff_large_20261014/refdata/digitized_csv"
AMU = 1.66053906660e-24
XS = {0.0: "jml_fig2a_x0", 0.2: "jml_fig2b_x20", 0.4: "jml_fig2c_x40", 0.6: "jml_fig2d_x60", 0.7: "jml_fig3c_x70", 0.8: "jml_fig3b_x80", 1.0: "jml_fig3a_x100"}


def csv(path):
    rows = [ln for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return np.loadtxt(rows[1:], delimiter=",")


def g_oo(frames, Ls, nO_idx, edges):
    hist = np.zeros(len(edges) - 1); n = len(nO_idx); norm = 0.0
    for P, L in zip(frames, Ls):
        O = P[nO_idx]; d = O[:, None] - O[None]; d -= L * np.round(d / L); r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(n, 1)]
        hist += np.histogram(r, edges)[0]; norm += n * (n / L ** 3)
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return 0.5 * (edges[1:] + edges[:-1]), 2 * hist / (norm * shell)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--box", nargs="+", required=True)
    ap.add_argument("--last-ps", type=float, default=25.0); ap.add_argument("--tag", default="_20261016"); ap.add_argument("--per-bin", type=int, default=100)
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    boxes = {k: json.load(open(v)) for k, v in (s.split("=", 1) for s in a.box)}
    res = {}
    for s in a.runs:
        tag, d = s.split("=", 1); box = boxes[tag]; nw, nm = box["n_water"], box["n_meoh"]
        with h5py.File(glob.glob(f"{d}/*.h5")[0], "r") as f:
            P = f["positions"][:]; t = f["time_ps"][:]; rho = f["density_g_cm3"][:]; T = f["temperature"][:]; Z = f.attrs["atomic_numbers"][:]
        mass_g = (nw * 18.015 + nm * 32.042) * AMU; L = (mass_g / (rho * 1e-24)) ** (1 / 3)      # Å per frame
        sel = t >= t[-1] - a.last_ps; Ps, Ls = P[sel], L[sel]
        nO = np.where(Z == 8)[0][:nw] if nw else np.where(Z == 8)[0]
        r, g = g_oo(Ps[:: max(1, len(Ps) // 40)], Ls[:: max(1, len(Ls) // 40)], nO, np.arange(2.0, 8.001, 0.02))
        m = r > 2.4; ip = np.argmax(g[m]); rp, gp = r[m][ip], g[m][ip]; after = (r > rp) & (r < 5.0); im = np.argmin(g[after])
        rho_w = nw / Ls.mean() ** 3; dr = r[1] - r[0]; cn = float(4 * np.pi * rho_w * np.sum(g[r <= 3.5] * r[r <= 3.5] ** 2 * dr))
        el = np.array([{1: "H", 6: "C", 8: "O"}[int(z)] for z in Z])
        fr = Ps[:: max(1, len(Ps) // 20)]; Lm = float(Ls.mean()); fr = fr - Lm * np.floor(fr / Lm)
        Q, S, _, _ = sx.structure_factor(fr, Lm, el, per_bin=a.per_bin)
        x = round(nm / (nw + nm), 3); exp = None
        for xk, pre in XS.items():
            if abs(xk - x) < 0.06: exp = csv(DG / f"{pre}__exp.csv")
        rms = None
        if exp is not None:
            qm = (exp[:, 0] >= 0.5) & (exp[:, 0] <= 12); rms = float(np.sqrt(np.mean((np.interp(exp[qm, 0], Q, S) - exp[qm, 1]) ** 2)))
        k = (Q >= 1.5) & (Q <= 2.6)
        res[tag] = dict(run=d, x=x, frames=int(sel.sum()), T_mean=float(T[sel].mean()), rho_mean=float(rho[sel].mean()), rho_std=float(rho[sel].std()), L_mean=Lm,
                        gOO_peak=float(gp), rOO_peak=float(rp), gOO_min=float(g[after][im]), rOO_min=float(r[after][im]), cn_3p5=cn,
                        sq_rms_0p5_12=rms, sq_peak1=[float(Q[k][np.argmax(S[k])]), float(S[k].max())])
        np.savez(out / f"npt_structure_{tag}{a.tag}.npz", r=r, g=g, Q=Q, S=S)
        print(tag, json.dumps({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in res[tag].items() if k2 != "run"}), flush=True)
    (out / f"npt_structure{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
