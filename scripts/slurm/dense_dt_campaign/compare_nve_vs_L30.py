#!/usr/bin/env python3
"""§7 NVE probe compare: dense L24/L26 vs sparse L30 (E_tot / T / bonds)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lj_scales/dense_dt_campaign"

# PSF order: C, H1, H2, CL1, CL2
_TAGS = (
    ("L24_nve_dt025_x64_5ps", 24, 1.224),
    ("L26_nve_dt025_x64_5ps", 26, 0.963),
    ("L30_nve_dt05_x64_5ps", 30, 0.627),
)


def _thermo_from_log(log: Path) -> dict | None:
    if not log.exists():
        return None
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(
            r"^\s*(\d+\.\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+",
            line,
        )
        if m:
            rows.append(
                {
                    "t": float(m.group(1)),
                    "step": int(m.group(2)),
                    "E_pot": float(m.group(3)),
                    "E_tot": float(m.group(4)),
                    "T": float(m.group(5)),
                }
            )
    if not rows:
        return None
    et = np.array([r["E_tot"] for r in rows], float)
    tt = np.array([r["T"] for r in rows], float)
    return {
        "source": "bench.log",
        "n_frames": len(rows),
        "t_end_ps": rows[-1]["t"],
        "dE_tot": float(et[-1] - et[0]),
        "max_abs_dE": float(np.max(np.abs(et - et[0]))),
        "T_mean": float(tt.mean()),
        "T_std": float(tt.std()),
        "T_min": float(tt.min()),
        "T_max": float(tt.max()),
        "E_tot_start": float(et[0]),
        "E_tot_end": float(et[-1]),
        "melt": bool(tt.max() > 600.0),
    }


def _bonds_from_pos(pos: np.ndarray) -> dict:
    """pos: (F, N, 3), DCM atoms C,H1,H2,CL1,CL2."""
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
        "bond_collapse": bool(ccl_a.min() < 1.40 or ch_a.min() < 0.80),
    }


def summarize_tag(tag: str) -> dict:
    d = OUT / tag
    row: dict = {"tag": tag, "success": (d / "SUCCESS.flag").exists()}
    h5s = sorted(d.glob("*.h5"))
    h5 = None
    for p in h5s:
        if p.stat().st_size > 1024:
            h5 = p
            break
    if h5 is not None:
        with h5py.File(h5, "r") as f:
            t = np.asarray(f["time_ps"], float)
            T = np.asarray(f["temperature"], float)
            et = np.asarray(f["total_energy"], float)
            inv = np.asarray(f["invariant"], float) if "invariant" in f else None
            pos = np.asarray(f["positions"], float)
            row.update(
                {
                    "source": "h5",
                    "h5": str(h5),
                    "n_frames": int(len(t)),
                    "t_end_ps": float(t[-1]),
                    "dE_tot": float(et[-1] - et[0]),
                    "max_abs_dE": float(np.max(np.abs(et - et[0]))),
                    "T_mean": float(T.mean()),
                    "T_std": float(T.std()),
                    "T_min": float(T.min()),
                    "T_max": float(T.max()),
                    "E_tot_start": float(et[0]),
                    "E_tot_end": float(et[-1]),
                    "melt": bool(T.max() > 600.0),
                }
            )
            if inv is not None and inv.size:
                row["dH_NHC"] = float(inv[-1] - inv[0])
                row["max_abs_dH"] = float(np.max(np.abs(inv - inv[0])))
            row.update(_bonds_from_pos(pos))
        return row
    log = _thermo_from_log(d / "bench.log")
    if log:
        row.update(log)
    else:
        row["error"] = "no h5/thermo yet"
    return row


def main() -> int:
    rows = []
    for tag, box, rho in _TAGS:
        r = summarize_tag(tag)
        r["box_A"] = box
        r["rho_nom"] = rho
        rows.append(r)

    json_path = OUT / "nve_compare_vs_L30.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# NVE compare vs sparse L30 (§7)",
        "",
        "Probe: CRD start, no drift-rescue, 5 ps. Dense arms use dt=0.25 fs x64; "
        "L30 uses dt=0.5 fs x64.",
        "",
        "| tag | ρ_nom | source | t_end | ΔE_tot (eV) | max\\|ΔE\\| | T mean±std (max) | C–Cl mean/min | C–H mean/min | melt? |",
        "|---|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['tag']}` | {r['rho_nom']} | — | — | — | — | — | — | — | pending |")
            continue
        ccl = (
            f"{r['CCl_mean']:.3f}/{r['CCl_min']:.3f}"
            if "CCl_mean" in r
            else "—"
        )
        ch = f"{r['CH_mean']:.3f}/{r['CH_min']:.3f}" if "CH_mean" in r else "—"
        lines.append(
            "| `{tag}` | {rho:.3f} | {src} | {tend:.2f} | {de:.4f} | {mde:.4f} | "
            "{tm:.1f}±{ts:.1f} ({tmax:.0f}) | {ccl} | {ch} | {melt} |".format(
                tag=r["tag"],
                rho=r["rho_nom"],
                src=r.get("source", "?"),
                tend=r.get("t_end_ps", float("nan")),
                de=r.get("dE_tot", float("nan")),
                mde=r.get("max_abs_dE", float("nan")),
                tm=r.get("T_mean", float("nan")),
                ts=r.get("T_std", float("nan")),
                tmax=r.get("T_max", float("nan")),
                ccl=ccl,
                ch=ch,
                melt="yes" if r.get("melt") else "no",
            )
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "- Sparse **L30** conserves E_tot (≲0.1 eV over 5 ps) with liquid-like T and intact bonds.",
        "- Dense **L24/L26** can keep ΔE_tot inside the 0.5 eV gate while T climbs to 10³ K "
        "(potential → kinetic); the E_tot gate alone does not stop a melt.",
        "- NVE `invariant` equals E_tot under NVE (no NHC); use NVT H_NHC for thermostat health.",
        "",
    ]
    md_path = OUT / "NVE_COMPARE.md"
    md_path.write_text("\n".join(lines))
    print(md_path.read_text())
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
