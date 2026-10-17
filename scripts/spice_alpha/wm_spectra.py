#!/usr/bin/env python
"""IR, isotropic / anisotropic Raman and a VCD null test for the water–methanol boxes from both models.

  python wm_spectra.py OUT_DIR --hybrid TAG=RUN_DIR[:POL] ... --efield TAG=TRAJ.npz ... --box x=BOX.json ... [--stride 1] [--seg-ps 5]

hybrid runs (handoff run_md.sh with DENSE=4): *_dense_*.npz hold positions/velocities every 2 fs; the MM charges come
from the run's h5. Dipole (a) μ_MM = Σ q_i r_i (fixed MM charges), (b) μ_EF and α_EF from EFieldPhysNet (rw_neutral)
on the same unwrapped frames (periodic, minimum image, as pbc_water_md.System). VCD null test: electric and charges-only
magnetic dipole m = ½ Σ q_i r_i × v_i (MM charges and model charges), rotational-strength cross spectrum Re[F(μ̇)* F(ṁ)].
efield runs (pbc_water_md.py): <tag>_traj.npz with mu, alpha every 2 fs. Spectra by Welch (dx/dt, Hann, segments).
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / "mmml_dev"))
from pbc_water_analysis import welch, smooth, band_peaks, C_CM_PER_FS   # noqa: E402
from mmml.utils.plotting.styles import apply_plot_style               # noqa: E402

CK = Path.home() / "mmml_new/ckpts/spicea_full_rw_neutral"; PARAMS = CK / "analysis/final/params-best.json"
WIN = {"CO/CH3 rock": (900, 1200), "bend": (1400, 1900), "CH stretch": (2700, 3100), "OH stretch": (3100, 3800)}


def unwrap(P, L):
    """Make the trajectory continuous: minimum-image displacements between consecutive frames."""
    d = np.diff(P, axis=0); d -= L * np.round(d / L)
    return np.concatenate([P[:1], P[:1] + np.cumsum(d, axis=0)])


def cross_welch(x, y, tau_fs, seg):
    T = len(x); nseg = max(T // seg, 1); seg = min(seg, T)
    xs = x[:nseg * seg].reshape(nseg, seg, -1); ys = y[:nseg * seg].reshape(nseg, seg, -1)
    dx = np.gradient(xs, tau_fs, axis=1); dy = np.gradient(ys, tau_fs, axis=1)
    dx -= dx.mean(1, keepdims=True); dy -= dy.mean(1, keepdims=True)
    w = np.hanning(seg)[None, :, None]
    Fx = np.fft.rfft(dx * w, axis=1); Fy = np.fft.rfft(dy * w, axis=1)
    p = np.real(np.conj(Fx) * Fy).sum(-1)
    return np.fft.rfftfreq(seg, d=tau_fs) / C_CM_PER_FS, p.mean(0), p.std(0) / np.sqrt(nseg)


def psf_charges(psf):
    """Per-atom charges from a CHARMM PSF (ATOM section: id segid resid resname name type charge mass ...)."""
    q, n = [], None
    with open(psf) as fh:
        for line in fh:
            if "!NATOM" in line:
                n = int(line.split()[0]); continue
            if n is not None and len(q) < n and line.strip():
                q.append(float(line.split()[6]))
    assert n is not None and len(q) == n, f"PSF parse failed: {psf}"
    return np.asarray(q, np.float64)


def mol_index(nw, nm):
    """Atom -> molecule index for the campaign boxes: nw waters (3 atoms) first, then nm methanols (6 atoms)."""
    return np.concatenate([np.repeat(np.arange(nw), 3), nw + np.repeat(np.arange(nm), 6)])


def magnetic_dipole(P, V, q, mol, masses):
    """Charges-only magnetic dipole, m = ½ Σ_i q_i (r_i − R_mol(i)) × v_i, with each molecule's centre of mass as its own
    origin (distributed-origin gauge: the origin-dependent R_mol × μ̇_mol term, which is pure noise for an achiral liquid
    and dominates the box total, is removed). q may be (N,) or (T, N)."""
    nmol = mol.max() + 1; M = np.bincount(mol, weights=masses, minlength=nmol)
    out = np.zeros((len(P), 3))
    w = masses / M[mol]
    for t in range(len(P)):
        R = np.zeros((nmol, 3))
        for k in range(3): R[:, k] = np.bincount(mol, weights=w * P[t, :, k], minlength=nmol)
        qt = q[t] if q.ndim == 2 else q
        out[t] = 0.5 * np.einsum("i,ij->j", qt, np.cross(P[t] - R[mol], V[t]))
    return out


def load_hybrid(run_dir, stride):
    import h5py
    files = sorted(glob.glob(f"{run_dir}/*_dense_*.npz"))
    P, V = [], []
    for f in files:
        d = np.load(f); P.append(d["positions"]); V.append(d["velocities"]); dt_ps = float(d["dt_frame_ps"])
    P = np.concatenate(P)[::stride].astype(np.float64); V = np.concatenate(V)[::stride].astype(np.float64)
    h5 = glob.glob(f"{run_dir}/*.h5")[0]
    with h5py.File(h5) as h:
        q = np.asarray(h["charges"][0], np.float64); Z = np.asarray(h.attrs["atomic_numbers"], np.int32)
    return P, V, q, Z, dt_ps * stride * 1000.0


class EF:
    """EFieldPhysNet periodic evaluator on given frames (one box)."""
    def __init__(self, Z, L):
        import jax, jax.numpy as jnp
        from les_water_eval import build
        from pbc_water_md import System
        jax.config.update("jax_default_matmul_precision", "highest")
        self.jax, self.jnp = jax, jnp
        model, params, cfg = build(CK, PARAMS)
        self.params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), params)
        self.model = model
        self.sys = System(model, self.params, cfg, Z, np.asarray(L, np.float64), 1, jnp.float32, 0.8, float(cfg["cutoff"]) - 0.5, n_edges_hint=len(Z) * 120)
        s = self.sys
        def f(pos):
            pr = pos[None]
            nl = s.neighbors(pr)
            _, (e, mu, q) = s.energy(pr, nl)
            mu2, al = s.dipole_polar(pr, nl, model.apply, self.params)
            return e[0], mu[0], al[0], q
        self.f = jax.jit(f)

    def run(self, P):
        mu, al, q, e = [], [], [], []
        for k, pos in enumerate(P):
            ek, mk, ak, qk = self.f(self.jnp.asarray(pos, self.jnp.float32))
            mu.append(np.asarray(mk)); al.append(np.asarray(ak)); q.append(np.asarray(qk)); e.append(float(ek))
            if k % 500 == 0: print(f"    frame {k}/{len(P)}", flush=True)
        return np.array(mu, np.float64), np.array(al, np.float64), np.array(q, np.float64), np.array(e)


def null_stats(f, vcd, sem, nseg):
    """Is the rotational-strength cross spectrum consistent with zero? t = vcd / sem per bin (nseg − 1 dof), bins 100–4000 cm⁻¹."""
    sel = (f > 100) & (f < 4000); t = vcd[sel] / np.maximum(sem[sel], 1e-30)
    return {"nseg": int(nseg), "bins": int(sel.sum()), "max_abs_t": float(np.abs(t).max()), "frac_abs_t_gt3": float((np.abs(t) > 3).mean()),
            "mean_t": float(t.mean()), "sum_vcd_over_sum_abs": float(vcd[sel].sum() / np.abs(vcd[sel]).sum())}


def spectra_from(mu, al, tau, seg):
    freq, ir = welch(mu[:, None, :], tau, seg)
    out = {"freq_cm": freq, "ir": ir}
    if al is not None:
        abar = np.trace(al, axis1=-2, axis2=-1) / 3.0
        beta = al - abar[..., None, None] * np.eye(3)
        _, iso = welch(abar[:, None, None], tau, seg); _, aniso = welch(beta[:, None], tau, seg)
        out["raman_iso"], out["raman_aniso"] = 45.0 * iso, 7.0 * 1.5 * aniso
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--hybrid", nargs="*", default=[], help="TAG=RUN_DIR (dense npz + h5)")
    ap.add_argument("--efield", nargs="*", default=[], help="TAG=TRAJ.npz from pbc_water_md.py")
    ap.add_argument("--box", nargs="*", default=[], help="TAG=BOX.json (box length)")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--seg-ps", type=float, default=5.0); ap.add_argument("--no-ef", action="store_true")
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    boxes = {k: json.load(open(v)) for k, v in (s.split("=", 1) for s in args.box)}
    psfs = {k: Path(v).parent / "model.psf" for k, v in (s.split("=", 1) for s in args.box)}
    res, spec = {}, {}
    for s in args.hybrid:
        tag, run = s.split("=", 1); t0 = time.time()
        P, V, q, Z, tau = load_hybrid(run, args.stride); L = boxes[tag]["box_A"]
        P = unwrap(P, L)
        if not np.any(q):   # the molecule-pair fast path reports no MM charges in the h5: take the PSF (CGenFF / TIP3P) charges
            q = psf_charges(psfs[tag]); print(f"   h5 charges are zero; using {len(q)} PSF charges, sum {q.sum():.3f}", flush=True)
        assert len(q) == P.shape[1]
        seg = int(round(args.seg_ps * 1000 / tau))
        print(f"== hybrid {tag}: {len(P)} frames, {tau} fs, {len(Z)} atoms, L {L}", flush=True)
        mol = mol_index(boxes[tag]["n_water"], boxes[tag]["n_meoh"]); masses = np.array([{1: 1.008, 6: 12.011, 8: 15.999}[int(z)] for z in Z])
        seg_v = max(int(round(2.0 * 1000 / tau)), 16)              # 2 ps segments for the null test: 10 segments, 9 dof
        mu_mm = np.einsum("tni,n->ti", P, q)                    # (T,3) e·Å, molecules neutral
        m_mm = magnetic_dipole(P, V, q, mol, masses)             # e·Å²/fs (× 1/c implicit), molecule-centred origins
        sp = spectra_from(mu_mm, None, tau, seg); fv, vcd, vcd_err = cross_welch(mu_mm, m_mm, tau, seg_v); f = sp["freq_cm"]
        spec[f"{tag}|hybrid MM charges"] = {**sp, "vcd": vcd, "vcd_sem": vcd_err, "freq_vcd": fv}
        r = {"frames": int(len(P)), "tau_fs": tau, "ir_peaks": band_peaks(f, smooth(f, sp["ir"]), WIN), "vcd_null": null_stats(fv, vcd, vcd_err, len(P) // seg_v)}
        if not args.no_ef:
            cache = out / f"ef_on_hybrid_{tag}.npz"
            if cache.exists() and int(np.load(cache)["mu"].shape[0]) == len(P):
                c = np.load(cache); mu_ef, al_ef, q_ef, e_ef = c["mu"], c["alpha"], c["q"], c["e"]; print(f"   reusing {cache}", flush=True)
            else:
                ef = EF(Z, L); mu_ef, al_ef, q_ef, e_ef = ef.run(P)
            m_ef = magnetic_dipole(P, V, q_ef, mol, masses)
            sp2 = spectra_from(mu_ef, al_ef, tau, seg); _, vcd2, vcd2_err = cross_welch(mu_ef, m_ef, tau, seg_v)
            spec[f"{tag}|hybrid dyn., EF response"] = {**sp2, "vcd": vcd2, "vcd_sem": vcd2_err, "freq_vcd": fv}
            r.update(ef_ir_peaks=band_peaks(f, smooth(f, sp2["ir"]), WIN), ef_raman_iso_peaks=band_peaks(f, smooth(f, sp2["raman_iso"]), WIN),
                     ef_raman_aniso_peaks=band_peaks(f, smooth(f, sp2["raman_aniso"]), WIN), ef_alpha_iso_mean_A3=float(np.trace(al_ef, axis1=1, axis2=2).mean() / 3),
                     ef_epot_drift_eV=float(e_ef[-1] - e_ef[0]), vcd_null_ef=null_stats(fv, vcd2, vcd2_err, len(P) // seg_v))
            np.savez_compressed(out / f"ef_on_hybrid_{tag}.npz", mu=mu_ef, alpha=al_ef, q=q_ef, e=e_ef, mu_mm=mu_mm, tau_fs=tau)
        res[f"hybrid_{tag}"] = r; print(json.dumps(r, indent=None)[:600], f"{time.time()-t0:.0f}s", flush=True)
    for s in args.efield:
        tag, traj = s.split("=", 1); t = np.load(traj)
        tau = float(t["dt_fs"]) * int(t["save_every"]); seg = int(round(args.seg_ps * 1000 / tau))
        mu = t["mu"][:, 0].astype(np.float64); al = t["alpha"][:, 0].astype(np.float64) if "alpha" in t.files else None
        sp = spectra_from(mu, al, tau, seg); spec[f"{tag}|EFieldPhysNet"] = sp
        f = sp["freq_cm"]; r = {"frames": int(len(mu)), "tau_fs": tau, "ir_peaks": band_peaks(f, smooth(f, sp["ir"]), WIN)}
        if al is not None: r.update(raman_iso_peaks=band_peaks(f, smooth(f, sp["raman_iso"]), WIN), raman_aniso_peaks=band_peaks(f, smooth(f, sp["raman_aniso"]), WIN))
        res[f"efield_{tag}"] = r; print(f"== efield {tag}", json.dumps(r)[:400], flush=True)
    # explicit flat save
    flat = {}
    for k, d in spec.items():
        kk = k.replace("|", "__").replace(" ", "_").replace(".", "").replace(",", "")
        for q_, v in d.items(): flat[f"{kk}__{q_}"] = v
    np.savez_compressed(out / "wm_spectra.npz", **flat)
    (out / "wm_spectra.json").write_text(json.dumps(res, indent=1))
    # figure: rows = boxes, cols = IR, Raman iso, Raman aniso, VCD null
    apply_plot_style("icml")
    tags = sorted({k.split("|")[0] for k in spec})
    cols = [("ir", "IR"), ("raman_iso", "isotropic Raman"), ("raman_aniso", "anisotropic Raman"), ("vcd", "VCD (null test)")]
    style = {"hybrid MM charges": ("#777777", (0, (1.5, 1.5)), 0.9), "hybrid dyn., EF response": ("#0b6e66", "-", 0.9), "EFieldPhysNet": ("#c25d35", "-", 1.0)}
    fig, axes = plt.subplots(len(tags), 4, figsize=(7.2, 1.55 * len(tags) + 0.8), sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.12)); axes = np.atleast_2d(axes)
    for i, tag in enumerate(tags):
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            for name, d in spec.items():
                if not name.startswith(tag + "|") or key not in d: continue
                lab = name.split("|")[1]; col, ls, lw = style[lab]; f = d["freq_vcd"] if key == "vcd" else d["freq_cm"]; y = smooth(f, d[key]); sel = (f > 100) & (f < 4000)
                norm = np.abs(y[sel]).max() if key == "vcd" else y[sel].max()
                ax.plot(f[sel], y[sel] / (norm if norm > 0 else 1), color=col, ls=ls, lw=lw, label=lab)
                if key == "vcd" and "vcd_sem" in d: ax.fill_between(f[sel], -smooth(f, d["vcd_sem"])[sel] / norm, smooth(f, d["vcd_sem"])[sel] / norm, color=col, alpha=0.15, lw=0)
            if key == "vcd": ax.axhline(0, color="#aaa", lw=0.5)
            ax.set_xlim(100, 4000); ax.set_yticks([]); ax.tick_params(labelsize=6); [ax.spines[s].set_visible(False) for s in ("top", "right")]
            if i == 0: ax.set_title(title, fontsize=8)
            if j == 0: ax.set_ylabel(tag, fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("wavenumber (cm$^{-1}$)", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            if b not in l: h.append(a); l.append(b)
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Water–methanol 30 Å boxes, 300 K: spectra from 20 ps NVE at 2 fs (Welch, 5 ps segments, 10 cm⁻¹ smoothing; each curve scaled to its maximum)", fontsize=7.5)
    fig.savefig(out / "fig_wm_spectra.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "fig_wm_spectra.pdf", bbox_inches="tight")
    print("wrote", out / "fig_wm_spectra.png")


if __name__ == "__main__":
    main()
