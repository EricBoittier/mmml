#!/usr/bin/env python
"""rw_neutral against the ORCA wB97M-D3(BJ)/def2-TZVPPD labels of the blow-up contact clusters (labels/clusters_tzvppd).

Forces atom by atom (DFT gradient -> eV/A), with the error on the contact O-H-O triple separately; energies as relative
energies inside each composition class (clusters with the same atom list), since absolute energies of different
clusters are not comparable.  Figure: force parity, per-cluster force MAE against the contact distance, contact-H force
error against the trajectory frame.   python wm_cluster_dft.py [--tag _20261017] [--labels labels/clusters_tzvppd]
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / "mmml_dev"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmml.utils.plotting.styles import apply_plot_style       # noqa: E402
from wm_blowup import Model, CK, PARAMS                         # noqa: E402

HA = 27.211386; BOHR = 0.529177210903; ZOF = {"H": 1, "C": 6, "O": 8}
NAME = re.compile(r"cluster_f(\d+)_(\w+?)_(\w+?)_([\d.]+)A")


class ForceModel(Model):
    def forces(self, Z, R):
        import jax
        key = ("F",) + tuple(Z)
        if key not in self.cache:
            s = self.System(self.model, self.params, self.cfg, np.asarray(Z), np.array([self.L] * 3), 1, self.jnp.float32, 0.8,
                            float(self.cfg["cutoff"]) - 0.5, n_edges_hint=max(len(Z) * len(Z), 64))
            e = lambda p: s.energy(p, s.neighbors(p))[0]
            self.cache[key] = jax.jit(lambda p: (e(p), -jax.grad(e)(p)))
        E, F = self.cache[key](self.jnp.asarray(R[None] + self.L / 2, self.jnp.float32))
        return float(E), np.asarray(F)[0]


def contact(Z, R):
    """Closest intermolecular O...H (bond graph < 1.25 A): (iO_donor, iH, iO_acceptor, d)."""
    d = np.linalg.norm(R[:, None] - R[None], axis=-1); n = len(Z); bonded = (d < 1.25) & ~np.eye(n, dtype=bool)
    best = None
    for h in np.where(Z == 1)[0]:
        od = [o for o in np.where(Z == 8)[0] if bonded[h, o]]
        if not od: continue
        for o in np.where(Z == 8)[0]:
            if o == od[0] or bonded[h, o]: continue
            if best is None or d[h, o] < best[3]: best = (int(od[0]), int(h), int(o), float(d[h, o]))
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", default="labels/clusters_tzvppd"); ap.add_argument("--tag", default="_20261017"); ap.add_argument("--out", default="blowup")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    model = ForceModel(); rows = []; allF = []
    for f in sorted(glob.glob(f"{a.labels}/*.json")):
        r = json.load(open(f))
        if not r.get("ok"): continue
        m = NAME.match(r["name"]); Z = np.array([ZOF[s] for s in r["symbols"]]); R = np.array(r["positions_A"], float)
        E, F = model.forces(Z, R); Fd = -np.array(r["gradient_Eh_bohr"], float) * HA / BOHR
        c = contact(Z, R); dF = F - Fd; mae = float(np.abs(dF).mean()); mx = float(np.abs(dF).max())
        trip = [c[0], c[1], c[2]] if c else []; eh = float(np.abs(dF[c[1]]).mean()) if c else np.nan; etrip = float(np.abs(dF[trip]).mean()) if c else np.nan
        rows.append(dict(name=r["name"], frame=int(m.group(1)), pair=f"{m.group(2)}-{m.group(3)}", d_name=float(m.group(4)), n=int(len(Z)), comp="".join(sorted(r["symbols"])),
                         E_model_eV=E, E_dft_eV=r["energy_Eh"] * HA, F_mae=mae, F_max=mx, F_rms_dft=float(np.sqrt((Fd ** 2).mean())), contact=trip, d_contact=(c[3] if c else np.nan),
                         F_err_H=eh, F_err_triple=etrip))
        allF.append((F.ravel(), Fd.ravel())); print(r["name"], f"MAE {mae:.3f} max {mx:.2f} H {eh:.3f} eV/A", flush=True)
    Fm = np.concatenate([x[0] for x in allF]); Fd = np.concatenate([x[1] for x in allF])
    # relative energies inside composition classes
    comps = {}
    for i, r in enumerate(rows): comps.setdefault(r["comp"], []).append(i)
    dE = []
    for k, idx in comps.items():
        if len(idx) < 2: continue
        em = np.array([rows[i]["E_model_eV"] for i in idx]); ed = np.array([rows[i]["E_dft_eV"] for i in idx])
        dm = em[:, None] - em[None]; dd = ed[:, None] - ed[None]; iu = np.triu_indices(len(idx), 1); dE += list((dm - dd)[iu])
    dE = np.array(dE)
    summ = dict(n_clusters=len(rows), F_mae_all=float(np.abs(Fm - Fd).mean()), F_rmse_all=float(np.sqrt(((Fm - Fd) ** 2).mean())), F_rms_dft=float(np.sqrt((Fd ** 2).mean())),
                F_err_H_mean=float(np.nanmean([r["F_err_H"] for r in rows])), F_err_triple_mean=float(np.nanmean([r["F_err_triple"] for r in rows])),
                dE_pairs=int(len(dE)), dE_mae=float(np.abs(dE).mean()) if len(dE) else None, dE_rmse=float(np.sqrt((dE ** 2).mean())) if len(dE) else None,
                by_pair={p: dict(n=sum(r["pair"] == p for r in rows), F_mae=float(np.mean([r["F_mae"] for r in rows if r["pair"] == p])), F_err_H=float(np.nanmean([r["F_err_H"] for r in rows if r["pair"] == p])))
                         for p in sorted({r["pair"] for r in rows})})
    json.dump(dict(summary=summ, clusters=rows), open(out / f"clusters_vs_dft{a.tag}.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    apply_plot_style("icml"); fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4), gridspec_kw=dict(wspace=0.45))
    ax[0].plot(Fd, Fm, ".", ms=1.5, color="#1b6f8a", alpha=0.5); lim = np.percentile(np.abs(Fd), 99.5); ax[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim); ax[0].set_xlabel("DFT force component (eV/Å)"); ax[0].set_ylabel("rw_neutral (eV/Å)")
    ax[0].set_title(f"{len(rows)} clusters, MAE {summ['F_mae_all']:.3f} eV/Å", fontsize=7.5)
    pairs = sorted({r["pair"] for r in rows}); cols = dict(zip(pairs, ["#1b6f8a", "#b85c2a", "#6a994e", "#8a1b6f"]))
    for p in pairs:
        rs = [r for r in rows if r["pair"] == p]
        ax[1].plot([r["d_contact"] for r in rs], [r["F_mae"] for r in rs], "o", ms=3, color=cols[p], label=p, mfc="none")
        ax[2].plot([r["frame"] for r in rs], [r["F_err_H"] for r in rs], "o", ms=3, color=cols[p], mfc="none")
    ax[1].set_xlabel("contact O···H (Å)"); ax[1].set_ylabel("force MAE per cluster (eV/Å)"); ax[1].legend(frameon=False, fontsize=6)
    ax[2].set_xlabel("trajectory frame (2 fs each ×10)"); ax[2].set_ylabel("|ΔF| on the contact H (eV/Å)")
    for a_ in ax: [a_.spines[s].set_visible(False) for s in ("top", "right")]
    fig.suptitle("Blow-up contact clusters: rw_neutral forces against ωB97M-D3(BJ)/def2-TZVPPD", fontsize=8)
    fig.savefig(out / f"fig_clusters_dft{a.tag}.png", dpi=300, bbox_inches="tight"); fig.savefig(out / f"fig_clusters_dft{a.tag}.pdf", bbox_inches="tight")
    print("wrote", out / f"fig_clusters_dft{a.tag}.png")


if __name__ == "__main__":
    main()
