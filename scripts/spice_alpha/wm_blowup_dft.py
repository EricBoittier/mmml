#!/usr/bin/env python
"""rw_neutral dimer scans (blowup/blowup_*.json) against the ORCA wB97M-D3(BJ)/def2-TZVPPD labels (labels/scan_tzvppd)."""
import json, glob, sys, numpy as np
sys.path.insert(0, "/scicore/home/meuwly/boitti0000/mmml_dev")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mmml.utils.plotting.styles import apply_plot_style
HA = 27.211386; tag = sys.argv[1] if len(sys.argv) > 1 else "_20261016"
lab = {}
for f in glob.glob("labels/scan_tzvppd/*.json"):
    r = json.load(open(f))
    if r.get("ok"): lab[r["name"]] = r["energy_Eh"]
bu = json.load(open(f"blowup/blowup{tag}.json"))["scans"]
apply_plot_style("icml"); fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.9), sharex="row", gridspec_kw=dict(wspace=0.15, hspace=0.45)); summary = {}
for k, (name, cur) in enumerate(bu.items()):
    pre = name.replace("→", "_to_"); e_ref_dft = lab[f"{pre}_compress_dOO2.80"]
    i80 = int(np.argmin(np.abs(np.array(cur["compress_dOO"]) - 2.8))); e_ref_mod = cur["compress_E"][i80]
    dd = np.array(cur["compress_dOO"]); em = np.array(cur["compress_E"]) - e_ref_mod
    ed = (np.array([lab.get(f"{pre}_compress_dOO{d:.2f}", np.nan) for d in dd]) - e_ref_dft) * HA
    ax[0, k].plot(dd, ed, "k-", lw=1.4, label="ωB97M-D3(BJ)/def2-TZVPPD"); ax[0, k].plot(dd, em, "-", color="#b85c2a", lw=1.0, label="rw_neutral")
    ax[0, k].set_title(name, fontsize=8); ax[0, k].set_xlabel("O···O (Å)"); summary[name] = {"compress_dOO": dd.tolist(), "dft": ed.tolist(), "model": em.tolist()}
    for d, col in ((2.5, "#1b6f8a"), (2.7, "#6a994e")):
        rr = np.array(cur[f"pt_dOO{d:.2f}_r"]); em2 = np.array(cur[f"pt_dOO{d:.2f}_E"]) - e_ref_mod
        ed2 = (np.array([lab.get(f"{pre}_dOO{d:.2f}_rOH{r:.2f}", np.nan) for r in rr]) - e_ref_dft) * HA
        ax[1, k].plot(rr, ed2, "-", color=col, lw=1.4, label=f"DFT, O···O {d} Å"); ax[1, k].plot(rr, em2, "--", color=col, lw=1.0, label=f"model, O···O {d} Å")
        summary[name][f"pt{d}"] = {"r": rr.tolist(), "dft": ed2.tolist(), "model": em2.tolist()}
    ax[1, k].set_xlabel("r(O$_d$–H) (Å)")
    for a_ in (ax[0, k], ax[1, k]): [a_.spines[s].set_visible(False) for s in ("top", "right")]
ax[0, 0].set_ylabel("E − E(O···O 2.8 Å) (eV)"); ax[1, 0].set_ylabel("E − E(O···O 2.8 Å) (eV)"); ax[0, 0].legend(frameon=False, fontsize=5.5); ax[1, 0].legend(frameon=False, fontsize=5)
ax[0, 0].set_ylim(-0.3, 1.0); ax[1, 0].set_ylim(-0.3, 3.5)
fig.savefig(f"blowup/fig_blowup_dft{tag}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"blowup/fig_blowup_dft{tag}.pdf", bbox_inches="tight")
json.dump(summary, open(f"blowup/scan_vs_dft{tag}.json", "w"), indent=1)
allm = []; alld = []
for name, s in summary.items():
    for key in [f"pt{d}" for d in (2.5, 2.7)]:
        p = s[key]; m = np.array(p["model"]); d_ = np.array(p["dft"]); ok = np.isfinite(d_) & (np.array(p["r"]) <= 1.35); allm += list(m[ok]); alld += list(d_[ok])
    m = np.array(s["model"]); d_ = np.array(s["dft"]); ok = np.isfinite(d_); allm += list(m[ok]); alld += list(d_[ok])
allm, alld = np.array(allm), np.array(alld); print(f"{len(lab)} labels; model vs DFT over compression + PT to r=1.35 A: MAE {np.mean(np.abs(allm-alld)):.3f} eV, max {np.max(np.abs(allm-alld)):.3f} eV, n={len(allm)}")
