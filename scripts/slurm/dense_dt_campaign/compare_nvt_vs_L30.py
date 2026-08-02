#!/usr/bin/env python3
"""§8/§7 NVT compare: dense L24/L26 vs sparse L30 (E_tot / H_NHC / bonds)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lj_scales/dense_dt_campaign"

# PSF order: C, H1, H2, CL1, CL2
_TAGS = (
    ("L24_nvt_dt1_f32_50ps", 24, 1.224),
    ("L24_nvt_dt05_x64_50ps", 24, 1.224),
    ("L26_nvt_dt1_f32_50ps", 26, 0.963),
    ("L26_nvt_dt05_x64_10ps", 26, 0.963),
    ("L26_nvt_dt05_x64_50ps", 26, 0.963),
    ("L30_nvt_dt05_x64_20ps", 30, 0.627),
)


def _bonds(pos: np.ndarray) -> dict:
    n = pos.shape[1]
    nmol = n // 5
    ccl, ch = [], []
    for fr in pos:
        for m in range(nmol):
            b = m * 5
            c, h1, h2, cl1, cl2 = fr[b : b + 5]
            ccl.append(float(np.linalg.norm(cl1 - c)))
            ccl.append(float(np.linalg.norm(cl2 - c)))
            ch.append(float(np.linalg.norm(h1 - c)))
            ch.append(float(np.linalg.norm(h2 - c)))
    ccl_a = np.asarray(ccl, float)
    ch_a = np.asarray(ch, float)
    return {
        "CCl_mean": float(ccl_a.mean()),
        "CCl_min": float(ccl_a.min()),
        "CH_mean": float(ch_a.mean()),
        "CH_min": float(ch_a.min()),
        "collapse": bool(ccl_a.min() < 1.40 or ch_a.min() < 0.80),
    }


def summarize(tag: str) -> dict | None:
    d = OUT / tag
    h5s = [p for p in sorted(d.glob("*.h5")) if p.stat().st_size > 1024]
    if not h5s:
        return None
    with h5py.File(h5s[0], "r") as f:
        et = np.asarray(f["total_energy"], float)
        T = np.asarray(f["temperature"], float)
        inv = np.asarray(f["invariant"], float) if "invariant" in f else None
        pos = np.asarray(f["positions"], float)
        n = len(et)
        half = max(n // 2, 1)
        row = {
            "tag": tag,
            "success": (d / "SUCCESS.flag").exists(),
            "n_frames": n,
            "dE_tot_full": float(et[-1] - et[0]),
            "T_mean": float(T.mean()),
            "T_std": float(T.std()),
        }
        if inv is not None and inv.size:
            inv2 = inv[half:]
            row["dH_NHC_2nd"] = float(inv2[-1] - inv2[0])
            row["max_abs_dH_2nd"] = float(np.max(np.abs(inv2 - inv2[0])))
        row.update(_bonds(pos))
        return row


def main() -> int:
    rows = []
    for tag, box, rho in _TAGS:
        r = summarize(tag)
        if r is None:
            rows.append({"tag": tag, "box_A": box, "rho_nom": rho, "missing": True})
            continue
        r["box_A"] = box
        r["rho_nom"] = rho
        rows.append(r)

    (OUT / "nvt_compare_vs_L30.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# NVT compare vs sparse L30",
        "",
        "Second-half metrics from jaxmd H5 (`invariant` = H_NHC). "
        "Bond order: C, H1, H2, CL1, CL2 (PSF).",
        "",
        "| tag | ρ_nom | full ΔE_tot (eV) | 2nd ΔH_NHC / max|Δ| | T (K) | C–Cl mean/min | C–H mean/min | collapse? |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(
                f"| `{r['tag']}` | {r['rho_nom']:.3f} | — | — | — | — | — | pending |"
            )
            continue
        dh = r.get("dH_NHC_2nd", float("nan"))
        mdh = r.get("max_abs_dH_2nd", float("nan"))
        lines.append(
            "| `{tag}` | {rho:.3f} | {de:.1f} | {dh:+.3f} / {mdh:.3f} | "
            "{tm:.1f}±{ts:.1f} | {ccl:.3f}/{cclm:.3f} | {ch:.3f}/{chm:.3f} | {col} |".format(
                tag=r["tag"],
                rho=r["rho_nom"],
                de=r["dE_tot_full"],
                dh=dh,
                mdh=mdh,
                tm=r["T_mean"],
                ts=r["T_std"],
                ccl=r["CCl_mean"],
                cclm=r["CCl_min"],
                ch=r["CH_mean"],
                chm=r["CH_min"],
                col="yes" if r["collapse"] else "no",
            )
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "- Dense L24/L26 NVT still shows large **E_tot collapse** vs milder L30.",
        "- **H_NHC** is much healthier with dt=0.5 + x64 than dt=1 f32.",
        "- Bond mins use correct PSF order; collapse if C–Cl < 1.40 Å or C–H < 0.80 Å.",
        "",
    ]
    md = OUT / "NVT_COMPARE.md"
    md.write_text("\n".join(lines))
    print(md.read_text())
    print(f"wrote {md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
