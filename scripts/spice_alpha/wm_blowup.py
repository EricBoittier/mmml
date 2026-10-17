#!/usr/bin/env python
"""Diagnostics of the EFieldPhysNet (rw_neutral) NVE blow-up of the 30 Å water–methanol box (x = 0.5) and the
proton-transfer scan that exposes the model gap.

  (1) T(t), E_tot(t) of the three periodic EFieldPhysNet NVE runs (x = 0, 0.5, 1; 2 fs records);
  (2) closest intermolecular O···H contact per pair type vs time (positions every 20 fs), and the distribution of the
      per-frame minimum in the stable runs;
  (3) POV-Ray render of the environment of the shortest contact in the last frame before the collapse;
  (4) rw_neutral energy along a proton-transfer coordinate for H-bonded dimers taken from the trajectory
      (water→methanol, methanol→water, water→water, methanol→methanol) at compressed O···O distances, plus the O···O
      compression curve; geometries written as xyz for QM labelling;
  (5) candidate training clusters: the molecules within --rcl Å of every O···H contact < --rcut Å in the frames before
      the collapse.

  python wm_blowup.py OUT --traj x0.000=...npz x0.500=...npz x1.000=...npz [--tag _20261016]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / "mmml_dev"))
from render_pov import ELEM, render, v                      # noqa: E402
from embedding_figs import sph, cyl, dashed, camera         # noqa: E402
from ase.data import covalent_radii                          # noqa: E402
from mmml.utils.plotting.styles import apply_plot_style      # noqa: E402

KB = 8.617333262e-5
CK = Path.home() / "mmml_new/ckpts/spicea_full_rw_neutral"; PARAMS = CK / "analysis/final/params-best.json"
SYM = {1: "H", 6: "C", 8: "O"}


def load(traj):
    t = np.load(traj); pos = t["pos"]; pos = pos[:, 0] if pos.ndim == 4 else pos
    Z = t["Z"].astype(int); L = float(np.ravel(t["L"])[0]); dt = float(t["dt_fs"]); pe = int(t["pos_every"]); se = int(t["save_every"])
    ek = t["ekin"]; ep = t["epot"]; ek = ek[:, 0] if ek.ndim == 2 else ek; ep = ep[:, 0] if ep.ndim == 2 else ep
    nw = 0; i = 0
    while i + 2 < len(Z) and Z[i] == 8 and Z[i + 1] == 1 and Z[i + 2] == 1: nw += 1; i += 3
    nm = (len(Z) - 3 * nw) // 6
    mol = np.concatenate([np.repeat(np.arange(nw), 3), nw + np.repeat(np.arange(nm), 6)])
    kind = np.concatenate([np.zeros(nw, int), np.ones(nm, int)])        # molecule kind 0 water, 1 methanol
    return dict(pos=pos, Z=Z, L=L, t_pos=np.arange(len(pos)) * pe * dt / 1000, t_e=np.arange(len(ek)) * se * dt / 1000,
                T=2 * ek / (3 * len(Z) * KB), Etot=ek + ep, nw=nw, nm=nm, mol=mol, kind=kind)


def mic(d, L):
    return d - L * np.round(d / L)


def min_contacts(D):
    """Per frame and pair type, the shortest intermolecular O···H distance. Types: Ow-Hw, Ow-Hm(O-H), Om-Hw, Om-Hm."""
    Z, mol, kind, L = D["Z"], D["mol"], D["kind"], D["L"]
    O = np.where(Z == 8)[0]; H = np.where(Z == 1)[0]
    # methanol hydroxyl H: in the 6-atom block C O H H H H the hydroxyl H is the one bonded to O (find by distance in frame 0)
    P0 = D["pos"][0]
    hyd = np.zeros(len(Z), bool)
    for m in range(D["nw"], D["nw"] + D["nm"]):
        at = np.where(mol == m)[0]; o = at[Z[at] == 8][0]; hs = at[Z[at] == 1]
        r = np.linalg.norm(mic(P0[hs] - P0[o], L), axis=-1); hyd[hs[np.argmin(r)]] = True
    hyd[Z == 1][: 2 * D["nw"]] = True
    wH = np.array([h for h in H if kind[mol[h]] == 0]); mH = np.array([h for h in H if kind[mol[h]] == 1 and hyd[h]])
    mC = np.array([h for h in H if kind[mol[h]] == 1 and not hyd[h]])
    wO = O[kind[mol[O]] == 0]; mO = O[kind[mol[O]] == 1]
    types = {"Ow···Hw": (wO, wH), "Ow···H(O)m": (wO, mH), "Om···Hw": (mO, wH), "Om···H(O)m": (mO, mH), "O···H(C)m": (O, mC)}
    out = {k: np.full(len(D["pos"]), np.nan) for k in types}; who = {}
    for f, P in enumerate(D["pos"]):
        for k, (a, b) in types.items():
            if len(a) == 0 or len(b) == 0: continue
            d = np.linalg.norm(mic(P[a][:, None] - P[b][None], L), axis=-1)
            d[mol[a][:, None] == mol[b][None]] = 99.0
            j = np.argmin(d); out[k][f] = d.ravel()[j]; ia, ib = np.unravel_index(j, d.shape); who[(k, f)] = (int(a[ia]), int(b[ib]))
    return out, who


def environment(D, f, i, j, rcl=4.5):
    """Molecules with any atom within rcl of atoms i or j in frame f, positions unwrapped around atom i."""
    P = D["pos"][f]; L = D["L"]; mol = D["mol"]
    ref = P[i]; Q = ref + mic(P - ref, L)
    near = np.linalg.norm(Q - Q[i], axis=-1) < rcl; near |= np.linalg.norm(Q - Q[j], axis=-1) < rcl
    mols = np.unique(mol[near]); sel = np.isin(mol, mols)
    return Q[sel], D["Z"][sel], mol[sel], np.where(sel)[0]


def pov_contact(out, name, Q, Z, mol, idx, i, j, dist, tag):
    """Render: the two molecules of the contact as CPK, the rest as thin sticks; dashed line for the contact."""
    c = 0.5 * (Q[idx == i][0] + Q[idx == j][0]); R = Q - c
    # orient: contact along x, in the screen plane
    a = R[idx == i][0]; b = R[idx == j][0]; x = (b - a) / np.linalg.norm(b - a)
    rnd = np.array([0.0, 0.0, 1.0]); z = rnd - np.dot(rnd, x) * x
    if np.linalg.norm(z) < 1e-3: rnd = np.array([0, 1.0, 0]); z = rnd - np.dot(rnd, x) * x
    z /= np.linalg.norm(z); y = np.cross(z, x); M = np.stack([x, y, z]); R = R @ M.T
    mi, mj = mol[idx == i][0], mol[idx == j][0]; core = np.isin(mol, [mi, mj])
    s = camera(np.array([5.2, 4.2, 5]), 1000, 800)
    from render_pov import bonds
    for grp, scale, rad, t in ((core, 0.30, 0.11, 0.0), (~core, 0.16, 0.06, 0.35)):
        Zg, Rg = Z[grp], R[grp]
        if len(Zg) == 0: continue
        for zz, rr in zip(Zg, Rg):
            col = ELEM.get(int(zz), (0.7, 0.7, 0.7)) if t == 0 else (0.62, 0.66, 0.70)
            s += sph(rr, scale * (0.55 + covalent_radii[zz]), col, t)
        for p, q in bonds(Zg, Rg):
            s += cyl(Rg[p], Rg[q], rad, (0.62, 0.62, 0.60) if t == 0 else (0.62, 0.66, 0.70), t)
    s += dashed(R[idx == i][0], R[idx == j][0], (0.95, 0.55, 0.1), r=0.06, n=9)
    png = render(out, name, s, w=1000, h=800)
    return png


def dimer_from_traj(D, donor_kind, acceptor_kind, f=0, dmax=2.1):
    """Pick the H-bonded pair donor-O–H···acceptor-O with the most linear, shortest contact in frame f. Returns atom lists."""
    P = D["pos"][f]; Z, mol, kind, L = D["Z"], D["mol"], D["kind"], D["L"]
    best = None
    for m in np.where(kind == donor_kind)[0]:
        at = np.where(mol == m)[0]; o = at[Z[at] == 8][0]
        hs = at[Z[at] == 1]; hs = hs[np.linalg.norm(mic(P[hs] - P[o], L), axis=-1) < 1.2]
        for h in hs:
            accO = np.where((Z == 8) & (kind[mol] == acceptor_kind) & (mol != m))[0]
            d = np.linalg.norm(mic(P[accO] - P[h], L), axis=-1); k = np.argmin(d)
            if d[k] < dmax:
                oo = np.linalg.norm(mic(P[accO[k]] - P[o], L)); ang = d[k] + np.linalg.norm(mic(P[h] - P[o], L)) - oo
                score = d[k] + 2 * ang
                if best is None or score < best[0]: best = (score, m, int(mol[accO[k]]), int(o), int(h), int(accO[k]))
    assert best is not None
    _, md, ma, o, h, oa = best
    atd = np.where(mol == md)[0]; ata = np.where(mol == ma)[0]
    Q = P[o] + mic(P - P[o], L)
    return dict(Zd=Z[atd], Rd=Q[atd], Za=Z[ata], Ra=Q[ata], o=o - atd[0], h=h - atd[0], oa=oa - ata[0])


class Model:
    def __init__(self, L=30.0):
        import jax, jax.numpy as jnp
        from les_water_eval import build
        from pbc_water_md import System
        jax.config.update("jax_default_matmul_precision", "highest")
        self.jnp = jnp; self.L = L
        self.model, params, self.cfg = build(CK, PARAMS)
        self.params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), params)
        self.System = System; self.cache = {}

    def energies(self, Z, Rs):
        """Energies (eV) of geometries Rs (n, N, 3) with atomic numbers Z, one isolated cluster in a 30 Å box."""
        key = tuple(Z)
        if key not in self.cache:
            s = self.System(self.model, self.params, self.cfg, np.asarray(Z), np.array([self.L] * 3), 1, self.jnp.float32, 0.8,
                            float(self.cfg["cutoff"]) - 0.5, n_edges_hint=max(len(Z) * len(Z), 64))
            import jax
            self.cache[key] = (s, jax.jit(lambda p: s.energy(p, s.neighbors(p))[0]))
        s, fn = self.cache[key]
        return np.array([float(fn(self.jnp.asarray(R[None] + self.L / 2, self.jnp.float32))) for R in Rs])


def scan_dimer(dim, model, name, out, oo_list=(2.5, 2.6, 2.7), rs=np.arange(0.90, 1.75, 0.05), oo_compress=np.arange(2.3, 3.41, 0.1)):
    """Proton-transfer coordinate: donor O–H distance r along the O···O line at fixed O···O; plus O···O compression with
    the donor O–H fixed. Returns dict of curves (eV, relative to the relaxed H-bond geometry as found in the trajectory)."""
    Zd, Rd, Za, Ra = dim["Zd"], dim["Rd"].copy(), dim["Za"], dim["Ra"].copy(); o, h, oa = dim["o"], dim["h"], dim["oa"]
    Z = np.concatenate([Zd, Za]); R0 = np.concatenate([Rd, Ra])
    e0 = model.energies(Z, R0[None])[0]
    axis = Ra[oa] - Rd[o]; d0 = np.linalg.norm(axis); u = axis / d0
    curves = {"r_OH_ref": float(np.linalg.norm(Rd[h] - Rd[o])), "d_OO_ref": float(d0), "e_ref_eV": float(e0)}
    geoms = []
    # compression: translate the acceptor molecule along the axis
    Rs = []
    for d in oo_compress:
        Ra2 = Ra + (d - d0) * u; Rs.append(np.concatenate([Rd, Ra2])); geoms.append((f"{name}_compress_dOO{d:.2f}", Rs[-1]))
    curves["compress_dOO"] = [float(x) for x in oo_compress]; curves["compress_E"] = (model.energies(Z, np.array(Rs)) - e0).tolist()
    # proton transfer at fixed O···O: H on the O···O line at distance r from the donor O
    for d in oo_list:
        Ra2 = Ra + (d - d0) * u; Rs = []
        for r in rs:
            Rd2 = Rd.copy(); Rd2[h] = Rd[o] + r * u; Rs.append(np.concatenate([Rd2, Ra2])); geoms.append((f"{name}_dOO{d:.2f}_rOH{r:.2f}", Rs[-1]))
        curves[f"pt_dOO{d:.2f}_r"] = [float(x) for x in rs]; curves[f"pt_dOO{d:.2f}_E"] = (model.energies(Z, np.array(Rs)) - e0).tolist()
    # write xyz
    xyzdir = out / "xyz_scan"; xyzdir.mkdir(exist_ok=True)
    for nm, R in geoms:
        (xyzdir / f"{nm}.xyz").write_text(f"{len(Z)}\n{nm} charge 0 mult 1\n" + "".join(f"{SYM[int(z)]} {x:.6f} {y:.6f} {w:.6f}\n" for z, (x, y, w) in zip(Z, R)))
    return curves, len(geoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("--traj", nargs="+", required=True); ap.add_argument("--tag", default="_20261016")
    ap.add_argument("--rcut", type=float, default=1.55); ap.add_argument("--rcl", type=float, default=4.0); ap.add_argument("--no-scan", action="store_true")
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    D = {k: load(v) for k, v in (s.split("=", 1) for s in a.traj)}
    res = {}
    # (1)+(2)
    contacts = {}
    for k, d in D.items():
        c, who = min_contacts(d); contacts[k] = (c, who)
        bad = np.where(d["T"] > 400)[0]; t_bad = float(d["t_e"][bad[0]]) if len(bad) else None
        res[k] = dict(n_atoms=int(len(d["Z"])), nw=d["nw"], nm=d["nm"], L=d["L"], t_blowup_ps=t_bad,
                      T_mean_before=float(d["T"][: (bad[0] if len(bad) else len(d["T"]))].mean()),
                      min_contact_A={kk: float(np.nanmin(vv[: (np.searchsorted(d["t_pos"], t_bad) - 1) if t_bad else len(vv)])) for kk, vv in c.items() if np.isfinite(vv).any()},
                      p01_contact_A={kk: float(np.nanpercentile(vv[: (np.searchsorted(d["t_pos"], t_bad) - 1) if t_bad else len(vv)], 1)) for kk, vv in c.items() if np.isfinite(vv).any()})
        print(k, json.dumps(res[k]), flush=True)
    apply_plot_style("icml")
    cols = {"x0.000": "#1b6f8a", "x0.500": "#b85c2a", "x1.000": "#6a994e"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.35))
    for k, d in D.items():
        m = d["T"] < 1e4
        ax[0].plot(d["t_e"][m], d["T"][m], color=cols[k], lw=0.8, label=k.replace("x", "x = "))
        ax[1].plot(d["t_e"][m], (d["Etot"] - d["Etot"][0])[m] * 1000 / (d["nw"] + d["nm"]), color=cols[k], lw=0.8)
    ax[0].set_ylim(280, 340); ax[0].set_ylabel("T (K)"); ax[0].set_xlabel("NVE time (ps)"); ax[0].legend(frameon=False, fontsize=6)
    ax[1].set_ylabel("E$_{tot}$ − E$_{tot}$(0) (meV per molecule)"); ax[1].set_xlabel("NVE time (ps)"); ax[1].set_ylim(-3, 3)
    c05, _ = contacts["x0.500"]; d05 = D["x0.500"]; tb = res["x0.500"]["t_blowup_ps"]
    ls = {"Ow···Hw": "-", "Ow···H(O)m": "-", "Om···Hw": "-", "Om···H(O)m": "-", "O···H(C)m": ":"}
    cc = {"Ow···Hw": "#1b6f8a", "Ow···H(O)m": "#b85c2a", "Om···Hw": "#d4a017", "Om···H(O)m": "#6a994e", "O···H(C)m": "#888"}
    for kk, vv in c05.items():
        if np.isfinite(vv).any(): ax[2].plot(d05["t_pos"], vv, color=cc[kk], ls=ls[kk], lw=0.8, label=kk)
    for k2 in ("x0.000", "x1.000"):
        c2, _ = contacts[k2]; kk = "Ow···Hw" if k2 == "x0.000" else "Om···H(O)m"
        ax[2].axhline(np.nanmin(c2[kk]), color=cols[k2], lw=0.6, ls="--")
    if tb: ax[2].axvline(tb, color="k", lw=0.6, ls=":")
    ax[2].set_xlim(0, (tb or 20) + 0.5); ax[2].set_ylim(1.2, 2.0); ax[2].set_xlabel("NVE time (ps), x = 0.5"); ax[2].set_ylabel("shortest intermolecular O···H (Å)")
    ax[2].legend(frameon=False, fontsize=5, ncol=2, loc="upper right")
    for a_ in ax: [a_.spines[s].set_visible(False) for s in ("top", "right")]
    for i, lab in enumerate("abc"): ax[i].text(-0.18, 1.04, lab, transform=ax[i].transAxes, fontweight="bold", fontsize=9)
    fig.savefig(out / f"fig_blowup_traj{a.tag}.png", dpi=300, bbox_inches="tight"); fig.savefig(out / f"fig_blowup_traj{a.tag}.pdf", bbox_inches="tight"); plt.close(fig)
    # (3) render the shortest heteromolecular contact in the last frame before the collapse
    fr = np.searchsorted(d05["t_pos"], tb) - 2 if tb else len(d05["pos"]) - 1   # last frame before the collapse (positions every 20 fs)
    c, who = contacts["x0.500"]
    best = min(((c[kk][fr], kk) for kk in ("Ow···H(O)m", "Om···Hw", "Ow···Hw", "Om···H(O)m") if np.isfinite(c[kk][fr])))
    i, j = who[(best[1], fr)]
    Q, Zs, mols, idx = environment(d05, fr, i, j, 4.5)
    png = pov_contact(out, f"pov_blowup_contact{a.tag}", Q, Zs, mols, idx, i, j, best[0], a.tag)
    res["render"] = dict(frame_ps=float(d05["t_pos"][fr]), pair=best[1], distance_A=float(best[0]), atoms=[i, j], molecules=[int(d05["mol"][i]), int(d05["mol"][j])], n_molecules_shown=int(len(np.unique(mols))))
    print("render", res["render"], flush=True)
    # (5) candidate clusters
    cl = out / "xyz_clusters"; cl.mkdir(exist_ok=True); ncl = 0; sizes = []
    if tb:
        f0 = max(0, fr - 25)
        for f in range(f0, fr + 1):
            for kk in ("Ow···H(O)m", "Om···Hw", "Ow···Hw", "Om···H(O)m"):
                if np.isfinite(c[kk][f]) and c[kk][f] < a.rcut:
                    i2, j2 = who[(kk, f)]; Q2, Z2, m2, _ = environment(d05, f, i2, j2, a.rcl)
                    nm = f"cluster_f{f:04d}_{kk.replace('···', '_').replace('(', '').replace(')', '')}_{c[kk][f]:.2f}A"
                    (cl / f"{nm}.xyz").write_text(f"{len(Z2)}\n{nm} t={d05['t_pos'][f]:.2f}ps charge 0 mult 1\n" + "".join(f"{SYM[int(z)]} {x:.6f} {y:.6f} {w:.6f}\n" for z, (x, y, w) in zip(Z2, Q2 - Q2.mean(0))))
                    ncl += 1; sizes.append(len(Z2))
    res["clusters"] = dict(n=ncl, atoms_min=int(min(sizes)) if sizes else 0, atoms_max=int(max(sizes)) if sizes else 0, rcut=a.rcut, rcl=a.rcl)
    print("clusters", res["clusters"], flush=True)
    # (4) proton-transfer scans
    if not a.no_scan:
        model = Model(); scans = {}; ngeom = 0
        for name, (dk, ak) in {"water→methanol": (0, 1), "methanol→water": (1, 0), "water→water": (0, 0), "methanol→methanol": (1, 1)}.items():
            dim = dimer_from_traj(d05, dk, ak); cur, n = scan_dimer(dim, model, name.replace("→", "_to_"), out); scans[name] = cur; ngeom += n
            print(name, "ref r_OH %.3f d_OO %.3f" % (cur["r_OH_ref"], cur["d_OO_ref"]), "E(pt, dOO 2.5) min %.3f eV at r %.2f" % (min(cur["pt_dOO2.50_E"]), cur["pt_dOO2.50_r"][int(np.argmin(cur["pt_dOO2.50_E"]))]), flush=True)
        res["scans"] = scans; res["n_scan_geometries"] = ngeom
        fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True, gridspec_kw=dict(wspace=0.12))
        for k, (name, cur) in enumerate(scans.items()):
            for d, col in zip((2.5, 2.6, 2.7), ("#b85c2a", "#1b6f8a", "#6a994e")):
                ax[k].plot(cur[f"pt_dOO{d:.2f}_r"], cur[f"pt_dOO{d:.2f}_E"], color=col, lw=1.0, label=f"O···O {d:.1f} Å")
            ax[k].axvline(cur["r_OH_ref"], color="#999", lw=0.5, ls=":"); ax[k].set_title(name, fontsize=8); ax[k].set_xlabel("r(O$_d$–H) (Å)")
            [ax[k].spines[s].set_visible(False) for s in ("top", "right")]
        ax[0].set_ylabel("E − E(H-bond) (eV)"); ax[0].legend(frameon=False, fontsize=6); ax[0].set_ylim(-1.0, 2.0)
        fig.savefig(out / f"fig_blowup_scan{a.tag}.png", dpi=300, bbox_inches="tight"); fig.savefig(out / f"fig_blowup_scan{a.tag}.pdf", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        for name, cur in scans.items(): ax.plot(cur["compress_dOO"], cur["compress_E"], lw=1.0, label=name)
        ax.set_xlabel("O···O (Å), donor O–H fixed"); ax.set_ylabel("E − E(H-bond) (eV)"); ax.legend(frameon=False, fontsize=6); [ax.spines[s].set_visible(False) for s in ("top", "right")]
        fig.savefig(out / f"fig_blowup_compress{a.tag}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    (out / f"blowup{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
