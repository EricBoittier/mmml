#!/usr/bin/env python
"""POV-Ray renders for the SPICE-α pages: data-cleaning cases, a mirror (enantiomer) pair with
predicted and reference dipoles, and the zero-shot spectra systems.

Transparent PNGs (+UA), element colours (Jmol-like), bonds from covalent radii (single molecules only).
Arrow colours: reference (green), parity_rw prediction (amber), pre-fix q10 prediction (grey).

Usage: python scripts/spice_alpha/render_pov.py OUT_DIR
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.data import covalent_radii

C = Path.home() / "mmml_new/ckpts"
POV = str(Path.home() / "envs/povray/bin/povray")
REF, PRED, PREFIX, CLASH = "<0.11,0.69,0.48>", "<0.93,0.63,0.0>", "<0.45,0.45,0.45>", "<0.89,0.29,0.28>"
ELEM = {1: (0.95, 0.95, 0.95), 6: (0.35, 0.35, 0.37), 7: (0.25, 0.4, 0.9), 8: (0.9, 0.2, 0.2), 9: (0.5, 0.85, 0.3),
        15: (1.0, 0.5, 0.0), 16: (0.95, 0.85, 0.2), 17: (0.2, 0.8, 0.2), 35: (0.6, 0.2, 0.1), 53: (0.45, 0.1, 0.6)}
MIRROR = np.diag([1.0, 1.0, -1.0])   # the mirror used for the test (z -> -z)
FLIPX = np.diag([-1.0, 1.0, 1.0])    # display: show the mirror image as a left-right reflection


def v(p):
    return f"<{p[0]:.4f},{p[1]:.4f},{-p[2]:.4f}>"  # POV-Ray is left-handed: flip z


def arrow(a, b, col, r):
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-3:
        return ""
    head = min(3.0 * r, 0.45 * L)
    m = b - d / L * head
    f = f"pigment {{ rgb {col} }} finish {{ phong 0.4 ambient 0.25 }}"
    return f"cylinder {{ {v(a)}, {v(m)}, {r:.3f} {f} }}\ncone {{ {v(m)}, {2.3 * r:.3f}, {v(b)}, 0 {f} }}\n"


def bonds(Z, R):
    return [(i, j) for i in range(len(Z)) for j in range(i + 1, len(Z))
            if np.linalg.norm(R[i] - R[j]) < 1.15 * (covalent_radii[Z[i]] + covalent_radii[Z[j]])]


def scene(Z, R, extra="", pts=None, scale_atoms=0.3, w=900, h=700, pad=1.4):
    P = R if pts is None else np.vstack([R, pts])
    lo, hi = P.min(0) - pad, P.max(0) + pad
    aspect = w / h
    width = max(hi[0] - lo[0], (hi[1] - lo[1]) * aspect)
    c = 0.5 * (lo + hi)
    s = [f"""#version 3.7;
global_settings {{ assumed_gamma 1.0 }}
camera {{ perspective location <{c[0]:.3f},{c[1]:.3f},-400> right x*{aspect:.4f} up y
         angle {2 * np.degrees(np.arctan(width / 800)):.4f} look_at <{c[0]:.3f},{c[1]:.3f},0> }}
light_source {{ <{c[0] - 20:.1f},{c[1] + 30:.1f},-60> color rgb 1.0 }}
light_source {{ <{c[0] + 30:.1f},{c[1] - 10:.1f},-40> color rgb 0.45 shadowless }}
"""]
    for z, r in zip(Z, R):
        col = ELEM.get(int(z), (0.7, 0.7, 0.7))
        s.append(f"sphere {{ {v(r)}, {scale_atoms * (0.55 + covalent_radii[z]):.3f} pigment {{ rgb <{col[0]},{col[1]},{col[2]}> }} "
                 f"finish {{ phong 0.6 phong_size 60 ambient 0.22 diffuse 0.75 }} }}\n")
    for i, j in bonds(Z, R):
        s.append(f"cylinder {{ {v(R[i])}, {v(R[j])}, 0.10 pigment {{ rgb <0.62,0.62,0.6> }} finish {{ ambient 0.25 }} }}\n")
    s.append(extra)
    return "".join(s)


def render(out, name, text, w=900, h=700):
    pov = out / f"{name}.pov"
    pov.write_text(text)
    subprocess.run([POV, f"+I{pov}", f"+O{out / name}.png", f"+W{w}", f"+H{h}", "+A0.1", "+UA", "-D", "+Q9"],
                   check=True, capture_output=True)
    pov.unlink()
    return out / f"{name}.png"


def orient(R):
    """Centre and rotate onto principal axes (largest extent along x)."""
    c = R.mean(0)
    _, _, Vt = np.linalg.svd(R - c, full_matrices=False)
    if np.linalg.det(Vt) < 0:
        Vt[2] *= -1
    return (R - c) @ Vt.T, Vt, c


def load_frames(idx):
    sys.path.insert(0, str(Path.home() / "mmml_dev"))
    from mmml.data.spice_alpha_ragged import load_ragged
    d = load_ragged(sorted((Path.home() / "data/spicealpha/ragged").glob("*.npz")))
    off = d["offsets"]
    return {i: {"Z": d["Z"][off[i]:off[i + 1]].astype(int), "R": d["R"][off[i]:off[i + 1]].astype(float),
                "F": d["F"][off[i]:off[i + 1]].astype(float), "D": d["D"][i].astype(float), "Q": float(d["Q"][i]),
                "mol": int(d["mol"][i])} for i in idx}, d


def broken_frame(out, fr):
    """Frame 1,610,563 (two H 0.39 Å apart) next to a healthy conformer of the same molecule."""
    paths = []
    for tag, f in (("broken", fr[1610563]), ("healthy", fr[1610562])):
        R, Vt, _ = orient(f["R"])
        F = f["F"] @ Vt.T
        D = np.linalg.norm(R[:, None] - R[None], axis=-1) + np.eye(len(R)) * 9
        i, j = np.unravel_index(D.argmin(), D.shape)
        fs = 2.2 / 61.4  # Å per eV/Å, same scale in both renders (largest force 61 eV/Å -> 2.2 Å)
        extra = "".join(arrow(R[k], R[k] + fs * F[k], REF, 0.06) for k in range(len(R)) if np.linalg.norm(F[k]) > 2.0)
        if tag == "broken":
            m = 0.5 * (R[i] + R[j])
            extra += f"sphere {{ {v(m)}, 0.75 pigment {{ rgbt <{CLASH.strip('<>')},0.65> }} finish {{ ambient 0.4 }} no_shadow }}\n"
        paths.append(render(out, f"pov_clean_{tag}", scene(f["Z"], R, extra, pts=R + fs * F, scale_atoms=0.26)))
    return paths


def mirror_pair(out, frame):
    """A chiral test molecule and its mirror image, each with reference, parity_rw and q10 dipoles."""
    preds = {}
    for key, ck in (("parity_rw", "spicea_full_v1_q10_parity_rw"), ("q10", "spicea_full_v1_q10")):
        for mode in ("orig", "mirror"):
            z = np.load(C / ck / f"analysis/final_test/{mode}/preds_test.npz")
            k = int(np.flatnonzero(z["frame"] == frame)[0])
            preds[key, mode] = (z["dipole"][k], z["dipole_ref"][k])
    fr, _ = load_frames([frame])
    f = fr[frame]
    R, Vt, c = orient(f["R"])
    rots = {"orig": Vt.T, "mirror": MIRROR @ Vt.T @ FLIPX}       # row-vector maps into the display frame
    pos = {"orig": R, "mirror": (f["R"] - c) @ MIRROR @ MIRROR @ Vt.T @ FLIPX}  # = R with x flipped
    # rotate about x so the reference dipole lies in the image plane (no foreshortening);
    # a rotation about x commutes with the left-right flip, so the pair stays exact mirror images
    m0 = preds["parity_rw", "orig"][1] @ rots["orig"]
    th = np.arctan2(-m0[2], m0[1])
    Rx = np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]])  # row-vector rotation
    rots = {k: r @ Rx for k, r in rots.items()}
    pos = {k: p @ Rx for k, p in pos.items()}
    mags = [np.linalg.norm(preds[k][0]) for k in preds] + [np.linalg.norm(preds["parity_rw", "orig"][1])]
    sc = 3.0 / max(mags)  # Å per e·Å, shared by both renders
    paths, info = [], {}
    for mode in ("orig", "mirror"):
        Rm = pos[mode]
        vecs = [preds["parity_rw", mode][1] @ rots[mode], preds["parity_rw", mode][0] @ rots[mode],
                preds["q10", mode][0] @ rots[mode]]
        # three rows beside the molecule: reference, parity_rw, q10 (same scale)
        x0 = Rm[:, 0].max() + 2.5 if mode == "orig" else Rm[:, 0].min() - 2.5
        y0 = Rm[:, 1].mean() + 2.0
        extra, tips = "", []
        for row, (vec, col, r) in enumerate(zip(vecs, (REF, PRED, PREFIX), (0.16, 0.16, 0.16))):
            o = np.array([x0, y0 - 2.0 * row, Rm[:, 2].max() + 1.0])
            vv = vec.copy(); vv[2] = 0.0  # in-plane part (out-of-plane is ~0 for the reference by construction)
            extra += arrow(o - 0.5 * sc * vv, o + 0.5 * sc * vv, col, r)
            tips += [o - 0.5 * sc * vv, o + 0.5 * sc * vv]
        paths.append(render(out, f"pov_mirror_{mode}", scene(f["Z"], Rm, extra, pts=np.array(tips), scale_atoms=0.26),
                            w=1100, h=700))
        info[mode] = {k: v.tolist() for k, v in zip(("ref", "parity_rw", "q10"), vecs)}
    return paths, info


def spectra_systems(out):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    sys.path.insert(0, str(Path.home() / "mmml_dev/scripts/spice_alpha"))
    from nve_ir import SYSTEMS
    paths = []
    for name in ("h2o2", "cyclopentane", "cyclohexane", "thioamide"):
        nve = C / f"nve_ir/parity_rw/{name}_nve.npz"
        if nve.exists():  # geometry relaxed on the model PES
            z = np.load(nve)
            Z, R = z["Z"].astype(int), z["R_min"]
        else:
            mol = Chem.AddHs(Chem.MolFromSmiles(SYSTEMS[name]))
            AllChem.EmbedMolecule(mol, randomSeed=0)
            AllChem.MMFFOptimizeMolecule(mol)
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
            R = mol.GetConformer().GetPositions()
        R, Vt, c = orient(R)
        paths.append(render(out, f"pov_sys_{name}", scene(Z, R, scale_atoms=0.3, pad=1.0), w=700, h=560))
        if name == "thioamide":  # D enantiomer as the left-right reflection
            paths.append(render(out, "pov_sys_thioamide_D", scene(Z, R @ FLIPX, scale_atoms=0.3, pad=1.0), w=700, h=560))
    return paths


def charged_example(out):
    """An odd-electron (certainly charged) amino-acid–ligand frame that the original filter kept (Q = NaN)."""
    sys.path.insert(0, str(Path.home() / "mmml_dev"))
    from mmml.data.spice_alpha_ragged import SUBSET_IDS, load_ragged
    d = load_ragged(sorted((Path.home() / "data/spicealpha/ragged").glob("*.npz")))
    off = d["offsets"]
    zs = np.add.reduceat(d["Z"].astype(int), off[:-1])
    cand = np.flatnonzero(np.isnan(d["Q"]) & (zs % 2 == 1) & (d["subset"] == SUBSET_IDS["amino_acid_ligand"]))
    i = int(cand[len(cand) // 2])
    Z, R = d["Z"][off[i]:off[i + 1]].astype(int), d["R"][off[i]:off[i + 1]].astype(float)
    R, _, _ = orient(R)
    return render(out, "pov_clean_charged", scene(Z, R, scale_atoms=0.26)), {"frame": i, "n_atoms": len(Z),
                                                                             "electrons": int(zs[i])}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fr, _ = load_frames([1610563, 1610562])
    info = {"broken": [str(p) for p in broken_frame(out, fr)]}
    chi = json.load(open(C / "chirality/chiral_test.json"))
    rw = np.load(C / "spicea_full_v1_q10_parity_rw/analysis/final_test/orig/preds_test.npz")
    have = set(rw["frame"].tolist())
    pick = next(e for e in chi if e["subset"] == "dipeptides" and len([c for c in e["centres"] if c[1] in "RS"]) >= 2
                and e["frame"] in have)
    p, mu = mirror_pair(out, pick["frame"])
    info["mirror"] = {"frame": pick["frame"], "group": pick["group"], "smiles": pick["smiles"], "dipoles": mu}
    info["systems"] = [str(x) for x in spectra_systems(out)]
    pc, info["charged"] = charged_example(out)
    (out / "pov_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1)[:1500])


if __name__ == "__main__":
    main()


def alpha_axes(out, frame, name, compare="median", B3_=97.17):
    """Principal axes of the label α (vermillion; dark grey if negative) against the same molecule's median
    eigenvalues (green, along the same axes) or a model prediction (blue, its own axes)."""
    fr, d = load_frames([frame])
    f = fr[frame]
    P = d["polar"][frame].astype(float) / B3_
    lam, U = np.linalg.eigh(0.5 * (P + P.T))
    mol = d["mol"] == f["mol"]
    med = np.median(np.linalg.eigvalsh(d["polar"][mol].astype(float) / B3_), axis=0)
    R, Vt, c = orient(f["R"])
    U = Vt @ U  # axes in the display frame (columns)
    ext = 0.5 * np.ptp(R[:, :2], axis=0).max()
    other = None
    if compare != "median":  # (eigenvalues, axes) of a prediction
        lp, Up = compare
        other = (lp, Vt @ Up)
    s = 0.85 * ext / max(np.abs(lam).max(), np.abs(med).max(), 1e-6)
    zf = R[:, 2].max() + 1.0
    o = np.array([0.0, 0.0, zf])
    extra = ""
    for k in range(3):
        u = U[:, k]
        col = "<0.30,0.30,0.30>" if lam[k] <= 0 else "<0.84,0.37,0.0>"
        L = s * abs(lam[k])
        extra += f"cylinder {{ {v(o - L * u)}, {v(o + L * u)}, 0.16 pigment {{ rgb {col} }} finish {{ phong 0.4 ambient 0.3 }} }}\n"
        if other is None:
            Lm = s * med[k]
            extra += (f"cylinder {{ {v(o - Lm * u + np.array([0, 0, 0.35]))}, {v(o + Lm * u + np.array([0, 0, 0.35]))}, 0.08 "
                      f"pigment {{ rgb {REF} }} finish {{ ambient 0.35 }} }}\n")
    if other is not None:
        lp, Up = other
        for k in range(3):
            Lp = s * lp[k]
            u = Up[:, k]
            extra += (f"cylinder {{ {v(o - Lp * u + np.array([0, 0, 0.35]))}, {v(o + Lp * u + np.array([0, 0, 0.35]))}, 0.09 "
                      f"pigment {{ rgb <0.0,0.45,0.70> }} finish {{ ambient 0.35 }} }}\n")  # model: blue
    pts = np.array([o + s * np.abs(lam).max() * U[:, k] * sg for k in range(3) for sg in (-1, 1)])
    path = render(out, name, scene(f["Z"], R, extra, pts=pts, scale_atoms=0.24), w=900, h=700)
    return path, {"frame": frame, "label_eig": lam.round(3).tolist(), "median_eig": med.round(3).tolist(),
                  "n_atoms": len(f["Z"])}
