#!/usr/bin/env python3
"""1D DCM dimer profiles with transparent POV overlays (no text boxes).

Follows ``docs/plotting-style-guide.md`` (transparent glossy thumbs on the
quantitative axis). Also writes a contact-ok settings comparison across
handoff ablations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mmml.analysis.dimer_scans import DEFAULT_ORIENT_MIN_CONTACT_A  # noqa: E402
from mmml.utils.plotting.styles import (  # noqa: E402
    apply_plot_style,
    comparison_colors,
    legend_outside,
)
from scripts.render_povray_data_overlays import _overlay  # noqa: E402
from scripts.slurm.dense_dt_campaign.dimer_scan_contacts import (  # noqa: E402
    annotate_dmin,
    contact_filtered_metrics,
    dimer_positions,
    fibonacci_sphere,
    load_monomer,
    super_fibonacci,
)
from scripts.slurm.dense_dt_campaign.plot_dimer_profiles import (  # noqa: E402
    load_annotated,
    load_mean_curve,
)
from scripts.slurm.dense_dt_campaign.render_dimer_scan_povray import (  # noqa: E402
    HybridFrameEval,
    _charge_rgb,
    _dipole_overlay,
    _force_overlay,
    _load_charge_cmap,
    _nice_q_lim,
    _normalize_alpha,
    _pov_include_dirs,
    _r_tag,
    build_dimer,
    compute_shared_box_half,
    glossy_scene,
    render_pov,
    CHARGE_CMAP_NAME,
    DIPOLE_ARROW_LEN,
)

SCAN = ROOT / "artifacts/lj_scales/dense_dt_campaign/dimer_scans"
ABL = ROOT / "artifacts/lj_scales/dense_dt_campaign/overbind_ablation"
OUT = ROOT / "docs/images/dense-dt-campaign/dimer_scans"
DATA = ROOT / "artifacts/lj_scales/dataset_cgenff.npz"
CKPT = ROOT / "artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json"
SIDE = ROOT / (
    "artifacts/lj_scales/ckpts/"
    "hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json"
)
POVRAY = "/mmhome/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray"
FONT_SKIP = True  # overlays: never stamp text boxes


def _clean_png(png: Path) -> None:
    """Normalize alpha only — no legend / title box."""
    from PIL import Image

    image = Image.open(png).convert("RGBA")
    image = _normalize_alpha(image)
    image.save(png)


def render_clean_frame(
    atoms,
    forces,
    charges,
    *,
    out_png: Path,
    half: float,
    force_scale: float,
    force_cap: float,
    q_lim: float,
    cmap,
    mode: str,
    povray: str,
    include_dirs: list[Path],
    width: int = 560,
    height: int = 420,
    n_mono: int = 5,
) -> Path:
    xyz = atoms.positions - atoms.positions.mean(0)
    if mode == "charge":
        colors = [_charge_rgb(float(q), q_lim, cmap) for q in charges]
        base = glossy_scene(
            atoms, half=half, width=width, height=height, atom_colors=colors, draw_box=True
        )
        overlay, _ = _dipole_overlay(xyz, charges, n_mono)
    else:
        base = glossy_scene(
            atoms, half=half, width=width, height=height, draw_box=True
        )
        overlay = _force_overlay(xyz, forces, force_scale, max_len=force_cap)
        dip, _ = _dipole_overlay(xyz, charges, n_mono)
        overlay += dip
    ok = render_pov(
        base + "\n".join(overlay) + "\n",
        out_png,
        povray=povray,
        width=width,
        height=height,
        include_dirs=include_dirs,
    )
    if not ok:
        raise RuntimeError(f"POV failed for {out_png}")
    _clean_png(out_png)
    return out_png


def _ablation_contact_ok_table(R1: np.ndarray, min_contact: float) -> pd.DataFrame:
    """Soft-well medians for ablation CSVs (8×8) + baseline (8×12)."""
    from mmml.analysis.dimer_scans import intermolecular_min_distance

    rows = []

    def summarize(df: pd.DataFrame, tag: str, n_dir: int, n_ori: int) -> dict:
        dirs = fibonacci_sphere(n_dir)
        quats = super_fibonacci(n_ori)
        cache: dict[tuple[int, float], float] = {}

        def dmin(ray: int, r: float) -> float:
            key = (int(ray), float(r))
            if key not in cache:
                di, qi = divmod(int(ray), n_ori)
                Ra, Rb = dimer_positions(R1, dirs[di], quats[qi], float(r))
                cache[key] = intermolecular_min_distance(Ra, Rb)
            return cache[key]

        if "dmin_A" not in df.columns:
            df = df.copy()
            df["dmin_A"] = [dmin(int(r.ray), float(r.r_A)) for r in df.itertuples()]
        ok = df[df["dmin_A"] >= min_contact]
        soft = []
        for _, sub in ok.groupby("ray"):
            s = sub[sub["r_A"] >= 3.4]
            if len(s):
                soft.append(float(s["E_int_kcal"].min()))
        soft_a = np.asarray(soft, dtype=float)
        g = ok.groupby("r_A")["E_int_kcal"]
        counts, means = g.count(), g.mean()
        keep = counts >= max(8, int(np.ceil(0.1 * df["ray"].nunique())))
        return dict(
            tag=tag,
            soft_median=float(np.median(soft_a)) if soft_a.size else np.nan,
            soft_mean=float(soft_a.mean()) if soft_a.size else np.nan,
            soft_deepest=float(soft_a.min()) if soft_a.size else np.nan,
            mean_curve_min=float(means[keep].min()) if keep.any() else float(means.min()),
            r_mean=float(means[keep].idxmin()) if keep.any() else float(means.idxmin()),
            n_soft=int(soft_a.size),
        )

    base = pd.read_csv(SCAN / "orient_components.csv")
    if "dmin_A" not in base.columns:
        base = annotate_dmin(base, R1=R1, n_directions=8, n_orientations=12)
    rows.append(summarize(base, "baseline_on8", 8, 12))

    for csv in sorted(ABL.glob("*_components.csv")):
        tag = csv.name.replace("_components.csv", "")
        if tag.startswith("ft_"):
            continue  # short FT collapsed wells — separate story
        rows.append(summarize(pd.read_csv(csv), tag, 8, 8))
    return pd.DataFrame(rows)


def plot_settings_compare(table: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    colors = comparison_colors(apply_plot_style("icml"), n=4)
    # Order: baseline → earlier handoff → contact deploy
    order = [
        "baseline_on8",
        "es_off_on8",
        "handoff_on6_w1p5",
        "handoff_on5_w1p5",
        "handoff_on4p5_w1",
        "es_off_handoff_on5",
        "contact_on4_w1p5",
        "contact_on3p5_w1p5",
    ]
    t = table.set_index("tag").reindex([x for x in order if x in set(table.tag)])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(t))
    ax.axhspan(-5, -3, color="0.88", alpha=0.7, label="lit. DCM dimer")
    ax.bar(x - 0.18, t["soft_median"], width=0.36, color=colors[0], label="soft-well median")
    ax.bar(x + 0.18, t["mean_curve_min"], width=0.36, color=colors[1], label="mean-curve min")
    ax.scatter(x, t["soft_deepest"], color=colors[2], s=28, zorder=3, label="soft deepest")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title(
        f"Contact-ok DCM–DCM wells ($d_\\mathrm{{min}}\\geq"
        f"{DEFAULT_ORIENT_MIN_CONTACT_A:g}$ Å) vs handoff"
    )
    ax.set_ylim(-16, 2)
    ax.grid(axis="y", alpha=0.2)
    legend_outside(ax, side="right", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_profile_with_overlays(
    learned: pd.DataFrame,
    thumbs: list[tuple[float, Path, float]],
    *,
    out: Path,
    title: str,
) -> None:
    """``thumbs``: list of (r_A, png_path, E_int at marker)."""
    import matplotlib.pyplot as plt

    apply_plot_style("icml")
    colors = comparison_colors(apply_plot_style("icml"), n=3)
    r = learned["r_A"].to_numpy()
    e = learned["E_int_kcal_mean"].to_numpy()
    emin = learned["E_int_kcal_min"].to_numpy()
    emax = learned["E_int_kcal_max"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.fill_between(r, emin, emax, color=colors[0], alpha=0.14, label="orientation envelope")
    ax.plot(r, e, color=colors[0], lw=2.0, label="mean (contact-ok)")
    ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax.axhspan(-5, -3, color="0.88", alpha=0.55, label="lit. ~−3…−5")
    ax.set_xlabel(r"COM–COM distance $r$ (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal mol$^{-1}$)")
    ax.set_title(title)
    ax.set_xlim(float(r.min()), min(10.0, float(r.max())))
    # Cap the envelope so a few repulsive soft-contact orientations don't
    # squash the well; leave headroom for POV thumbs.
    m8 = r <= 8.0
    y_lo = min(-12.0, float(np.nanmin(e[m8])) - 1.5)
    y_hi_data = float(np.nanpercentile(np.clip(emax[m8], None, 15.0), 90))
    y_hi = max(6.0, y_hi_data) + 12.0
    ax.set_ylim(y_lo, y_hi)
    ax.grid(alpha=0.18)

    # Place thumbs above the curve in path order (style guide).
    y_thumb = y_hi - 4.0
    for r_i, png, e_i in thumbs:
        ax.scatter([r_i], [e_i], s=22, color="white", edgecolor="0.2", zorder=5)
        ax.plot([r_i, r_i], [e_i, y_thumb - 2.0], color="0.65", lw=0.7, zorder=1)
        _overlay(ax, png, (r_i, y_thumb), zoom=0.14)

    legend_outside(ax, side="right", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import shutil

    min_contact = DEFAULT_ORIENT_MIN_CONTACT_A
    OUT.mkdir(parents=True, exist_ok=True)
    R1, Z1 = load_monomer(DATA)
    n_mono = 5

    learned_df = load_annotated(SCAN / "orient_components.csv", R1)
    metrics = contact_filtered_metrics(learned_df, min_contact=min_contact)
    min_rays = int(metrics["min_rays_for_mean"])
    learned = load_mean_curve(learned_df, min_contact=min_contact, min_rays=min_rays)

    # Settings comparison (contact-ok)
    table = _ablation_contact_ok_table(R1, min_contact)
    plot_settings_compare(
        table, OUT.parent / "overbind_ablation" / "contact_ok_settings_compare.png"
    )
    (OUT.parent / "overbind_ablation" / "contact_ok_settings.json").write_text(
        json.dumps(
            {"min_contact_A": min_contact, "runs": table.to_dict(orient="records")},
            indent=2,
        )
        + "\n"
    )

    # Pick overlay geometries along the median soft-well ray (contact-ok).
    soft_wells = metrics.get("soft_wells") or []
    if not soft_wells:
        print("ERROR: no contact-ok soft wells")
        return 2
    soft_wells = sorted(soft_wells, key=lambda x: x["E_int_kcal"])
    mid = soft_wells[len(soft_wells) // 2]
    di, qi = int(mid["direction"]), int(mid["orientation"])
    dirs = fibonacci_sphere(int(learned_df["direction"].max()) + 1)
    quats = super_fibonacci(int(learned_df["orientation"].max()) + 1)
    dvec, quat = dirs[di], quats[qi]

    # r picks: near well, soft shoulder, handoff, asymptote — all contact-ok on this ray
    ray_df = learned_df[
        (learned_df["direction"] == di)
        & (learned_df["orientation"] == qi)
        & (learned_df["dmin_A"] >= min_contact)
    ].sort_values("r_A")
    targets = [float(mid["r_A"]), 4.5, 6.5, 8.5]
    picks = []
    for t in targets:
        i = int(np.argmin(np.abs(ray_df["r_A"].to_numpy() - t)))
        row = ray_df.iloc[i]
        picks.append((float(row.r_A), float(row.E_int_kcal), float(row.dmin_A)))
    # de-dup
    uniq = []
    seen = set()
    for p in picks:
        key = round(p[0], 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    picks = uniq

    print(f"Overlay ray d{di} q{qi}; picks: {picks}")
    print("Loading hybrid for forces/charges…", flush=True)
    ev = HybridFrameEval(
        checkpoint=CKPT,
        sidecar=SIDE,
        data=DATA,
        mm_switch_on=8.0,
        ml_switch_width=1.5,
        mm_switch_width=5.0,
        n_mono=n_mono,
    )
    atoms_list = [build_dimer(R1, Z1, dvec, quat, r) for r, _, _ in picks]
    pred = ev.evaluate([a.positions.copy() for a in atoms_list])
    f_ref = float(np.linalg.norm(pred["forces"], axis=-1).max())
    force_scale = 1.35 / max(f_ref, 1e-12)
    force_cap = 2.5
    q_lim = _nice_q_lim(float(np.max(np.abs(pred["charges"]))))
    cmap = _load_charge_cmap()
    frames = [
        dict(atoms=a, dmin=dmin, r=r)
        for a, (r, _, dmin) in zip(atoms_list, picks)
    ]
    # Fake pred dict shape for box helper
    box_half = compute_shared_box_half(
        [dict(atoms=a, r=r, dmin=d) for a, (r, _, d) in zip(atoms_list, picks)],
        pred,
        n_mono,
        force_scale=force_scale,
        force_cap=force_cap,
        dipole_len=DIPOLE_ARROW_LEN,
    )

    povray = POVRAY if Path(POVRAY).is_file() else (shutil.which("povray") or "")
    if not povray:
        print("ERROR: povray not found")
        return 2
    include_dirs = _pov_include_dirs(povray)
    thumb_dir = OUT / "povray" / "overlay_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    thumbs_fd: list[tuple[float, Path, float]] = []
    thumbs_q: list[tuple[float, Path, float]] = []
    for i, (r, e_i, dmin) in enumerate(picks):
        stem = f"overlay_d{di:02d}_q{qi:02d}_r{_r_tag(r)}"
        png_fd = thumb_dir / f"{stem}_forces_dipoles.png"
        png_q = thumb_dir / f"{stem}_by_charge.png"
        render_clean_frame(
            atoms_list[i],
            pred["forces"][i],
            pred["charges"][i],
            out_png=png_fd,
            half=box_half,
            force_scale=force_scale,
            force_cap=force_cap,
            q_lim=q_lim,
            cmap=cmap,
            mode="forces",
            povray=povray,
            include_dirs=include_dirs,
        )
        render_clean_frame(
            atoms_list[i],
            pred["forces"][i],
            pred["charges"][i],
            out_png=png_q,
            half=box_half,
            force_scale=force_scale,
            force_cap=force_cap,
            q_lim=q_lim,
            cmap=cmap,
            mode="charge",
            povray=povray,
            include_dirs=include_dirs,
        )
        # Marker energy: mean curve at nearest r (style guide: mark on the curve)
        j = int(np.argmin(np.abs(learned["r_A"].to_numpy() - r)))
        e_mark = float(learned["E_int_kcal_mean"].iloc[j])
        thumbs_fd.append((r, png_fd, e_mark))
        thumbs_q.append((r, png_q, e_mark))
        print(f"  {stem} dmin={dmin:.2f} E_mean={e_mark:.2f}")

    plot_profile_with_overlays(
        learned,
        thumbs_fd,
        out=OUT / "dcm_dimer_Eint_profile_povray.png",
        title=(
            f"DCM–DCM contact-ok profile with geometries "
            f"(ray d{di} q{qi}; $d_\\mathrm{{min}}\\geq{min_contact:g}$ Å)"
        ),
    )
    plot_profile_with_overlays(
        learned,
        thumbs_q,
        out=OUT / "dcm_dimer_Eint_profile_povray_by_charge.png",
        title=(
            f"DCM–DCM contact-ok profile — atoms by {CHARGE_CMAP_NAME} charge "
            f"(ray d{di} q{qi})"
        ),
    )

    # Zoom with overlays
    learned_zoom = learned[learned["r_A"] <= 7.5].reset_index(drop=True)
    thumbs_zoom = [(r, p, e) for r, p, e in thumbs_fd if r <= 7.5]
    plot_profile_with_overlays(
        learned_zoom,
        thumbs_zoom,
        out=OUT / "dcm_dimer_Eint_zoom_povray.png",
        title="Well region with POV geometries (contact-ok)",
    )

    # Refresh dimer README bullets for settings
    soft_med = metrics["median_soft_well_kcal"]
    readme = OUT / "README.md"
    extra = "\n".join(
        [
            "",
            "## POV overlays (no text boxes)",
            "",
            "| Figure | Content |",
            "|---|---|",
            "| `dcm_dimer_Eint_profile_povray.png` | 1D mean + force/dipole thumbs |",
            "| `dcm_dimer_Eint_profile_povray_by_charge.png` | same with charge coloring |",
            "| `dcm_dimer_Eint_zoom_povray.png` | well zoom + thumbs |",
            "| `../overbind_ablation/contact_ok_settings_compare.png` | handoff settings, contact-ok |",
            "",
            f"Soft-well median (contact-ok): **{soft_med:.1f} kcal/mol**. "
            "See ablation compare for on=5 / contact deploy.",
            "",
            "Regenerate overlays:",
            "```bash",
            "uv run python scripts/slurm/dense_dt_campaign/plot_dimer_profile_overlays.py",
            "```",
            "",
        ]
    )
    text = readme.read_text() if readme.exists() else ""
    if "POV overlays" not in text:
        readme.write_text(text.rstrip() + "\n" + extra)

    print("DONE overlays →", OUT)
    print(table.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
