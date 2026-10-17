#!/usr/bin/env python
"""Pure-water ML/MM runs against experiment as a function of the Thole polarization strength.

For each NVE production directory (h5 with positions every 0.5 ps): self-diffusion from the MSD slope (5–30 ps lags,
or 2–8 ps when the run is shorter than 40 ps), the O–O radial distribution (first maximum, first minimum, coordination
to 3.5 Å), and the X-ray S(Q) (sq_xray.structure_factor of the handoff) with its RMS against the digitized curve of
Galicia-Andrés 2015 at x = 0. Figure: g_OO against Dixit's pure-water circles, S(Q) against the X-ray curve, and the three
scalars against the polarization scale.

  python wm_water_scan.py OUT --runs s0.00=RUN_DIR s0.50=RUN_DIR ... [--last-ps 50] [--tag _20261016]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.home() / "handoff/handoff_large_20261014/scripts")); sys.path.insert(0, str(Path.home() / "mmml_dev"))
import sq_xray as sx                                        # noqa: E402
from mmml.utils.plotting.styles import apply_plot_style     # noqa: E402

DG = Path.home() / "handoff/handoff_large_20261014/refdata/digitized_csv"
EXP = {"D_water_1e9": 2.30, "rOO_peak": 2.80, "gOO_peak": 2.57, "cn_3p5": 4.7}   # Skinner 2013 (X-ray g_OO: 2.80 Å, 2.57), Soper n_OO(3.5) ≈ 4.7; D 2.30 at 298 K


def csv(path):
    rows = [ln for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return np.loadtxt(rows[1:], delimiter=",")


def load_run(d, last_ps):
    h = glob.glob(f"{d}/*.h5")[0]
    with h5py.File(h, "r") as f:
        P = f["positions"][:]; t = f["time_ps"][:]; T = f["temperature"][:]; Et = f["total_energy"][:]; Z = f.attrs["atomic_numbers"][:]
    pdb = glob.glob(f"{d}/*_minimized.pdb")[0]; cr = [ln for ln in open(pdb) if ln.startswith("CRYST1")]
    L = float(cr[0].split()[1]) if cr else None
    if os.environ.get("BOX_JSON"): L = json.load(open(os.environ["BOX_JSON"]))["box_A"]
    return P, t, T, Et, Z, L


def msd_D(O, t, L):
    dU = np.diff(O, axis=0); dU -= L * np.round(dU / L); U = np.concatenate([O[:1], O[:1] + np.cumsum(dU, 0)])
    dt = t[1] - t[0]; tmax = t[-1] - t[0]
    lo, hi = (5.0, 30.0) if tmax >= 40 else (2.0, min(8.0, tmax / 2.5))
    lag = np.arange(1, int(hi / dt) + 1); tl = lag * dt; m = tl >= lo
    msd = np.array([np.mean(np.sum((U[l:] - U[:-l]) ** 2, -1)) for l in lag])
    return float(np.polyfit(tl[m], msd[m], 1)[0] / 6 * 10), (lo, hi)      # 1e-9 m2/s


def g_oo(O, L, edges):
    """Intermolecular O–O g(r) averaged over frames (MIC, cubic box)."""
    n = O.shape[1]; rho = n / L ** 3; hist = np.zeros(len(edges) - 1)
    for P in O:
        d = P[:, None] - P[None]; d -= L * np.round(d / L); r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(n, 1)]
        hist += np.histogram(r, edges)[0]
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = 2 * hist / len(O) / (n * rho * shell); r = 0.5 * (edges[1:] + edges[:-1])
    return r, g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--last-ps", type=float, default=50.0); ap.add_argument("--tag", default="_20261016"); ap.add_argument("--per-bin", type=int, default=120)
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    exp_sq = csv(DG / "jml_fig2a_x0__exp.csv"); exp_g = csv(DG / "dixit_fig3__OwOw_pure_water.csv")
    res = {}; curves = {}
    for s in a.runs:
        tag, d = s.split("=", 1); P, t, T, Et, Z, L = load_run(d, a.last_ps)
        O = P[:, Z == 8]; nw = O.shape[1]
        sel = t >= t[-1] - a.last_ps; Os = O[sel]
        D, lags = msd_D(O, t, L)
        edges = np.arange(2.0, 8.001, 0.02); r, g = g_oo(Os[:: max(1, len(Os) // 50)], L, edges)
        m = r > 2.4; ip = np.argmax(g[m]); rp = r[m][ip]; gp = g[m][ip]
        after = (r > rp) & (r < 5.0); im = np.argmin(g[after]); rm = r[after][im]; gm = g[after][im]
        rho = nw / L ** 3; dr = r[1] - r[0]; cn = float(4 * np.pi * rho * np.sum(g[r <= 3.5] * r[r <= 3.5] ** 2 * dr))
        frames = P[sel][:: max(1, sel.sum() // 25)]; el = np.array(["O" if z == 8 else "H" for z in Z])
        Q, S, _, _ = sx.structure_factor(frames, L, el, per_bin=a.per_bin)
        qm = (exp_sq[:, 0] >= 0.5) & (exp_sq[:, 0] <= 12); rms = float(np.sqrt(np.mean((np.interp(exp_sq[qm, 0], Q, S) - exp_sq[qm, 1]) ** 2)))
        k = (Q >= 1.5) & (Q <= 2.6); q1, s1 = float(Q[k][np.argmax(S[k])]), float(S[k].max())
        res[tag] = dict(run=d, n_water=int(nw), L=L, frames_total=int(len(t)), t_end_ps=float(t[-1]), T_mean=float(T[sel].mean()), Etot_drift_eV=float(Et[-1] - Et[0]),
                        D_water_1e9=D, msd_lags_ps=lags, gOO_peak=float(gp), rOO_peak=float(rp), gOO_min=float(gm), rOO_min=float(rm), cn_3p5=cn,
                        sq_rms_0p5_12=rms, sq_peak1=[q1, s1])
        curves[tag] = dict(r=r, g=g, Q=Q, S=S); print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[tag].items() if k != "run"}), flush=True)
    (out / f"water_scan{a.tag}.json").write_text(json.dumps(res, indent=1)); np.savez(out / f"water_scan_curves{a.tag}.npz", **{f"{k}__{q}": v for k, c in curves.items() for q, v in c.items()})
    apply_plot_style("icml")
    tags = list(res); cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(tags)))
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.3), gridspec_kw=dict(wspace=0.5))
    ax[0].plot(exp_g[:, 0], exp_g[:, 1], "o", ms=2.2, mfc="none", mec="k", mew=0.6, label="pure water (Dixit 2002)")
    ax[1].plot(exp_sq[:, 0], exp_sq[:, 1], "k-", lw=1.6, label="X-ray (Galicia-Andrés 2015)")
    for c, tg in zip(cols, tags):
        ax[0].plot(curves[tg]["r"], curves[tg]["g"], color=c, lw=0.9, label=tg); ax[1].plot(curves[tg]["Q"], curves[tg]["S"], color=c, lw=0.9)
    ax[0].set_xlim(2.3, 7); ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("g$_{OO}$(r)"); ax[0].legend(frameon=False, fontsize=5.5)
    ax[1].set_xlim(0.5, 10); ax[1].set_xlabel("Q (Å$^{-1}$)"); ax[1].set_ylabel("S(Q)"); ax[1].legend(frameon=False, fontsize=5.5)
    def scale(tg):
        try: return float(tg.lstrip("s"))
        except ValueError: return None
    xs = [scale(tg) for tg in tags]; have = [x is not None for x in xs]
    if any(have):
        xx = np.array([x for x in xs if x is not None]); pick = lambda key: np.array([res[tg][key] for tg, h in zip(tags, have) if h])
        ax[2].plot(xx, pick("D_water_1e9") / EXP["D_water_1e9"], "o-", color="#1b6f8a", ms=3, label="D / D$_{exp}$")
        ax[2].plot(xx, pick("cn_3p5") / EXP["cn_3p5"], "s-", color="#b85c2a", ms=3, label="n$_{OO}$(3.5 Å) / exp")
        ax[2].plot(xx, pick("rOO_peak") / EXP["rOO_peak"], "^-", color="#6a994e", ms=3, label="r$_{OO}$ peak / exp")
        ax[2].plot(xx, pick("sq_rms_0p5_12"), "d--", color="#777", ms=3, label="S(Q) RMS")
        for tg, h in zip(tags, have):
            if not h: ax[2].axhline(res[tg]["D_water_1e9"] / EXP["D_water_1e9"], color="#1b6f8a", lw=0.5, ls=":")
        ax[2].axhline(1, color="k", lw=0.5); ax[2].set_xlabel("polarizability scale (v1 = 1)"); ax[2].set_ylabel("ratio to experiment"); ax[2].legend(frameon=False, fontsize=5.5)
    for a_ in ax: [a_.spines[s].set_visible(False) for s in ("top", "right")]
    fig.suptitle("Pure water, 30 Å ML/MM hybrid, 300 K: structure and diffusion against the Thole polarization strength", fontsize=7.5)
    fig.savefig(out / f"fig_water_scan{a.tag}.png", dpi=300, bbox_inches="tight"); fig.savefig(out / f"fig_water_scan{a.tag}.pdf", bbox_inches="tight")
    print("wrote", out / f"fig_water_scan{a.tag}.png")


if __name__ == "__main__":
    main()
