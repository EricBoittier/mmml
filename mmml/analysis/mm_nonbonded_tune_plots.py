"""Figures for ``mmml tune-mm-nonbonded fit`` (house ``icml`` style, Okabe-Ito)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

TEACHER = "#0072B2"
STUDENT = "#D55E00"
CGENFF = "#009E73"
SAMPLER = "#56B4E9"
TUNED = "#CC79A7"
UNDERLAY = "#E69F00"
MINIMAL = "#000000"


def _variant_color(name: str) -> str:
    if name == "cgenff":
        return CGENFF
    if "underlay" in name:
        return UNDERLAY
    if name.startswith("tail-min"):
        return MINIMAL
    return TUNED


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def make_tune_figures(
    payload: dict[str, Any],
    ff,
    results: Sequence[dict[str, Any]],
    data,
    is_valid: np.ndarray,
    figs_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.models.mm_nonbonded_tune import TuneParams, predict_energy
    from mmml.utils.plotting.styles import apply_plot_style, legend_outside

    apply_plot_style("icml")
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    split = "valid" if is_valid.any() else "all"

    # 1. cohesion budget ---------------------------------------------------
    rows = [("teacher", None)]
    first = results[0]
    rows.append(("hybrid, CGenFF", first["cgenff"][split]["cohesion_budget"]))
    for r in results:
        rows.append((r["name"], r["tuned"][split]["cohesion_budget"]))
    teacher_total = first["cgenff"][split]["cohesion_budget"]["teacher_total"]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.0))
    labels = []
    for k, (name, b) in enumerate(rows):
        y = len(rows) - 1 - k
        labels.append((y, name))
        if b is None:
            ax.barh(y, teacher_total, color=TEACHER, label="teacher E_int (pet-omol-l)")
            ax.text(teacher_total, y, f"{teacher_total:.2f} ", va="center", ha="right",
                    color="0.2", fontsize=9)
            continue
        left = 0.0
        for key, color, lab in (
            ("ml_pairs", STUDENT, "student ML pairs"),
            ("mm_tail", CGENFF if "CGenFF" in name else _variant_color(name), None),
            ("underlay", "#F0E442", "MM underlay (ML region)"),
        ):
            v = float(b[key])
            if abs(v) < 1e-6:
                continue
            if lab is None:
                lab = "MM tail, CGenFF" if "CGenFF" in name else f"MM tail, {name}"
            ax.barh(y, v, left=left, color=color, label=lab)
            left += v
        ax.text(left, y, f"{b['hybrid_total']:.2f} ", va="center", ha="right", color="0.2",
                fontsize=9)
    ax.axvline(teacher_total, color=TEACHER, ls="--", lw=1)
    lo_x = min([teacher_total] + [b["hybrid_total"] for _, b in rows if b is not None])
    ax.set_xlim(1.2 * lo_x, 0.0)
    ax.set_yticks([y for y, _ in labels], [n for _, n in labels])
    ax.set_xlabel("interaction energy per molecule (kcal/mol)")
    ax.set_title(f"Cohesion budget, {split} frames (n={first['cgenff'][split]['n_frames']})")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    legend_outside(ax, handles=list(uniq.values()), labels=list(uniq.keys()))
    out.append(_save(fig, figs_dir / "lj_elec_tune_budget.png"))

    # 2. per-frame parity -------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ref = data.e_teacher / data.n_mol
    cg = TuneParams.cgenff(ff.n_types)
    series = [("CGenFF", cg, CGENFF, "o")]
    for r, m in zip(results, ("s", "^", "D", "v")):
        series.append((r["name"], r["_params_obj"], _variant_color(r["name"]), m))
    lo, hi = float(ref.min()), float(ref.max())
    for name, p, color, marker in series:
        pred = (data.e_ml + predict_energy(ff, p, data)) / data.n_mol
        for mask, fill, tag in ((~is_valid, True, "train"), (is_valid, False, "valid")):
            if not mask.any():
                continue
            rmse = float(np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2)))
            ax.scatter(ref[mask], pred[mask], s=14, marker=marker,
                       facecolors=color if fill else "none", edgecolors=color, lw=0.8,
                       label=f"{name}, {tag} (RMSE {rmse:.2f})")
            lo, hi = min(lo, float(pred[mask].min())), max(hi, float(pred[mask].max()))
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls=":")
    ax.set_xlabel("teacher E_int / N (kcal/mol)")
    ax.set_ylabel("hybrid E_int / N (kcal/mol)")
    ax.set_title("Liquid frames: hybrid vs teacher")
    legend_outside(ax)
    out.append(_save(fig, figs_dir / "lj_elec_tune_parity.png"))

    # 3. parameters with bootstrap spread ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [3, 3, 1.4]})
    T = ff.n_types
    x = np.arange(T)
    width = 0.8 / len(results)
    for j, r in enumerate(results):
        pb = r["params_bootstrap"]
        color = _variant_color(r["name"])
        for ax, key, title in ((axes[0], "eps_scale", "epsilon scale"),
                               (axes[1], "rmin_scale", "Rmin scale")):
            m = [pb[f"{key}[{t}]"]["mean"] for t in ff.type_names]
            s = [pb[f"{key}[{t}]"]["std"] for t in ff.type_names]
            ax.bar(x + (j - (len(results) - 1) / 2) * width, m, width, yerr=s, color=color,
                   capsize=2, label=r["name"])
            ax.set_xticks(x, ff.type_names, rotation=45)
            ax.set_title(title)
            ax.axhline(1.0, color=CGENFF, lw=1, ls="--")
        axes[2].bar(j, pb["charge_scale"]["mean"], 0.6, yerr=pb["charge_scale"]["std"],
                    color=color, capsize=2)
    axes[2].axhline(1.0, color=CGENFF, lw=1, ls="--", label="CGenFF")
    axes[2].set_xticks(range(len(results)), [r["name"] for r in results], rotation=45)
    axes[2].set_title("charge scale")
    axes[1].set_ylim(0.9, 1.1)
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    legend_outside(fig, handles=h0 + h2, labels=l0 + l2)
    out.append(_save(fig, figs_dir / "lj_elec_tune_params.png"))

    # 4. dimer 2-body check ----------------------------------------------
    dc = payload.get("dimer_check")
    if dc:
        r = np.asarray(dc["r_com"])
        bins = np.linspace(r.min(), r.max(), 10)
        mid = 0.5 * (bins[1:] + bins[:-1])
        idx = np.digitize(r, bins) - 1

        def binned(v):
            v = np.asarray(v)
            return np.array([v[idx == k].mean() if np.any(idx == k) else np.nan
                             for k in range(len(mid))])

        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.plot(mid, binned(dc["teacher"]), "o-", color=TEACHER, label="teacher 2-body")
        for name, m in dc["models"].items():
            color = _variant_color(name)
            ax.plot(mid, binned(m["pred"]), "s--", color=color,
                    label=f"MM {name} (RMSE {m['rmse']:.2f})")
        ax.set_xlabel("dimer COM distance (A)")
        ax.set_ylabel("mean dimer E_int (kcal/mol)")
        ax.set_title(f"Isolated dimers, MM-only region (n={dc['n']})")
        legend_outside(ax)
        out.append(_save(fig, figs_dir / "lj_elec_tune_dimers.png"))

    # 5. handoff grid ------------------------------------------------------
    grid = payload.get("handoff_grid") or []
    if grid:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        names = [g["name"].replace("handoff ", "") for g in grid]
        xs = np.arange(len(grid))
        cg_r = [(g["cgenff"]["valid"] or g["cgenff"]["all"])["E_int_rmse_kcal_per_mol"] for g in grid]
        tu_r = [(g["tuned"]["valid"] or g["tuned"]["all"])["E_int_rmse_kcal_per_mol"] for g in grid]
        ax.bar(xs - 0.2, cg_r, 0.4, color=CGENFF, label="CGenFF")
        ax.bar(xs + 0.2, tu_r, 0.4, color=MINIMAL, label=f"tuned ({grid[0]['variant']})")
        ax.set_xticks(xs, names)
        ax.set_xlabel("mm_switch_on : mm_switch_width (A)")
        ax.set_ylabel("E_int/N RMSE (kcal/mol)")
        ax.set_title("Handoff window (energy-only refit)")
        legend_outside(ax)
        out.append(_save(fig, figs_dir / "lj_elec_tune_handoff.png"))
    return out
