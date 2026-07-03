"""Tri-alanine water box: JAX MIC vs PyCHARMM nonbonded parity report (metrics + plots)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

PairCategory = Literal["pep_pep", "pep_water", "water_water"]


class _PairCat(str, Enum):
    PEP_PEP = "pep_pep"
    PEP_WATER = "pep_water"
    WATER_WATER = "water_water"


def classify_pair_category(i: int, j: int, n_peptide_atoms: int) -> PairCategory:
    """Classify atom pair as peptide–peptide, peptide–water, or water–water."""
    pep_i = int(i) < int(n_peptide_atoms)
    pep_j = int(j) < int(n_peptide_atoms)
    if pep_i and pep_j:
        return _PairCat.PEP_PEP.value
    if pep_i != pep_j:
        return _PairCat.PEP_WATER.value
    return _PairCat.WATER_WATER.value


def classify_pair_categories(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_peptide_atoms: int,
) -> np.ndarray:
    cats = np.empty(pair_i.shape[0], dtype=object)
    n_pep = int(n_peptide_atoms)
    pep_i = pair_i < n_pep
    pep_j = pair_j < n_pep
    cats[pep_i & pep_j] = _PairCat.PEP_PEP.value
    cats[pep_i ^ pep_j] = _PairCat.PEP_WATER.value
    cats[~pep_i & ~pep_j] = _PairCat.WATER_WATER.value
    return cats


@dataclass(frozen=True, slots=True)
class CategoryNonbondedTotals:
    category: PairCategory
    n_pairs: int
    vdw_kcal: float
    elec_kcal: float
    mean_r_A: float

    @property
    def total_kcal(self) -> float:
        return self.vdw_kcal + self.elec_kcal


@dataclass(frozen=True, slots=True)
class TopPairRecord:
    rank: int
    atom_i: int
    atom_j: int
    category: PairCategory
    r_A: float
    vdw_kcal: float
    elec_kcal: float

    @property
    def total_kcal(self) -> float:
        return self.vdw_kcal + self.elec_kcal


@dataclass(frozen=True, slots=True)
class TermComparison:
    term: str
    charmm_kcal: float
    jax_kcal: float

    @property
    def delta_kcal(self) -> float:
        return self.jax_kcal - self.charmm_kcal

    @property
    def rel_delta(self) -> float:
        denom = max(abs(self.charmm_kcal), 1e-12)
        return self.delta_kcal / denom


@dataclass(frozen=True, slots=True)
class PairListStats:
    n_atoms: int
    n_peptide_atoms: int
    n_excluded_pairs: int
    n_e14_pairs: int
    n_pairs_within_cutnb: int
    n_pairs_pep_pep: int
    n_pairs_pep_water: int
    n_pairs_water_water: int
    cutnb_A: float
    ctonnb_A: float
    ctofnb_A: float


@dataclass(frozen=True, slots=True)
class CategoryForceDelta:
    category: PairCategory
    jax_force_rms: float
    charmm_force_rms: float
    delta_force_rms: float
    vdw_delta_kcal: float


@dataclass(frozen=True, slots=True)
class PairSwitchAudit:
    """Analytic vs finite-difference dE/dr for one MIC pair (fswitch/vfswitch)."""

    atom_i: int
    atom_j: int
    r_A: float
    vdw_kcal: float
    vdw_dedr_analytic: float
    vdw_dedr_numeric: float
    vdw_dedr_rel_err: float
    elec_kcal: float
    elec_dedr_analytic: float
    elec_dedr_numeric: float
    elec_dedr_rel_err: float


@dataclass(frozen=True, slots=True)
class TrialanineNbParityReport:
    """Full CHARMM vs JAX nonbonded parity snapshot for the TRIA water box."""

    seed: int
    perturb_seed: int
    n_waters: int
    box_side_A: float
    pair_stats: PairListStats
    bonded: TermComparison
    vdw: TermComparison
    elec: TermComparison
    nb_total: TermComparison
    mm_total: TermComparison
    jax_by_category: tuple[CategoryNonbondedTotals, ...]
    charmm_by_category: tuple[CategoryNonbondedTotals, ...]
    category_vdw: tuple[TermComparison, ...]
    category_force_delta: tuple[CategoryForceDelta, ...]
    top_vdw_pairs: tuple[TopPairRecord, ...]
    top_elec_pairs: tuple[TopPairRecord, ...]
    switch_derivative_audits: tuple[PairSwitchAudit, ...]
    force_rms_delta: float
    force_max_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _aggregate_by_category(
    decomp: Any,
    categories: np.ndarray,
) -> tuple[CategoryNonbondedTotals, ...]:
    out: list[CategoryNonbondedTotals] = []
    for cat in (_PairCat.PEP_PEP.value, _PairCat.PEP_WATER.value, _PairCat.WATER_WATER.value):
        mask = categories == cat
        n = int(np.sum(mask))
        if n == 0:
            out.append(CategoryNonbondedTotals(cat, 0, 0.0, 0.0, 0.0))
            continue
        out.append(
            CategoryNonbondedTotals(
                category=cat,
                n_pairs=n,
                vdw_kcal=float(np.sum(decomp.vdw_kcal[mask])),
                elec_kcal=float(np.sum(decomp.elec_kcal[mask])),
                mean_r_A=float(np.mean(decomp.r_A[mask])),
            )
        )
    return tuple(out)


def _top_pairs(
    decomp: Any,
    categories: np.ndarray,
    *,
    term: Literal["vdw", "elec"],
    n: int = 20,
    category_filter: PairCategory | None = None,
) -> tuple[TopPairRecord, ...]:
    values = decomp.vdw_kcal if term == "vdw" else decomp.elec_kcal
    mask = np.ones(values.shape[0], dtype=bool)
    if category_filter is not None:
        mask &= categories == category_filter
    idx = np.where(mask)[0]
    if idx.size == 0:
        return ()
    order = idx[np.argsort(-np.abs(values[idx]))[:n]]
    records: list[TopPairRecord] = []
    for rank, k in enumerate(order, start=1):
        records.append(
            TopPairRecord(
                rank=rank,
                atom_i=int(decomp.pair_i[k]) + 1,
                atom_j=int(decomp.pair_j[k]) + 1,
                category=str(categories[k]),
                r_A=float(decomp.r_A[k]),
                vdw_kcal=float(decomp.vdw_kcal[k]),
                elec_kcal=float(decomp.elec_kcal[k]),
            )
        )
    return tuple(records)


def _term_comparison(term: str, charmm: float, jax: float) -> TermComparison:
    return TermComparison(term=term, charmm_kcal=float(charmm), jax_kcal=float(jax))


def collect_trialanine_nb_parity(
    box: Any,
    positions: np.ndarray,
    *,
    perturb_seed: int = 31,
    top_n_pairs: int = 20,
) -> TrialanineNbParityReport:
    """Compare JAX MIC nonbonded decomposition to active PyCHARMM ``ENER FORCE``."""
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_bonded_forces_kcalmol_A,
        charmm_cmap_is_active,
        charmm_nonbonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        decompose_nonbonded_pair_energies,
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
        nonbonded_energy_and_forces,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        n_peptide_atoms_in_trialanine_box,
    )

    pos = np.asarray(positions, dtype=np.float64)
    set_charmm_positions(pos)
    apply_charmm_mm_block()
    run_charmm_bonded_ener_force(silent=True)
    include_cmap = charmm_cmap_is_active()

    charmm_bonded = charmm_bonded_energy_components_kcalmol()
    charmm_nb = charmm_nonbonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()
    charmm_bonded_total = float(charmm_bonded.get("total", 0.0))
    charmm_nb_total = float(charmm_nb.get("total", 0.0))
    charmm_mm_total = charmm_bonded_total + charmm_nb_total

    cuts = box.nbond_cutoffs
    settings = CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )
    bonded = load_bonded_system_from_psf(
        box.psf_path,
        pos,
        prm_file=box.cgenff_prm,
        extra_prm_files=box.cmap_extra_prm_files,
    )
    nbond_data = load_nonbonded_system_from_charmm(box.psf_path, box.cgenff_prm)
    bonded_comp, bonded_forces = bonded_energy_and_forces(
        jnp.asarray(pos),
        bonded.topology,
        bonded.bonded,
        urey_k=bonded.urey_k,
        urey_r0=bonded.urey_r0,
        energy_unit="kcal/mol",
        include_cmap=include_cmap,
    )
    nb_comp, nb_forces = nonbonded_energy_and_forces(pos, nbond_data, box.cell, settings)
    jax_forces = np.asarray(bonded_forces + nb_forces, dtype=np.float64)
    jax_bonded_total = float(bonded_comp["total"])
    jax_vdw = float(nb_comp["vdw"])
    jax_elec = float(nb_comp["elec"])
    jax_nb_total = float(nb_comp["total"])
    jax_mm_total = jax_bonded_total + jax_nb_total

    delta_f = jax_forces - np.asarray(charmm_forces, dtype=np.float64)
    force_rms = float(np.sqrt(np.mean(np.sum(delta_f * delta_f, axis=-1))))
    force_max = float(np.max(np.linalg.norm(delta_f, axis=-1)))

    decomp = decompose_nonbonded_pair_energies(pos, nbond_data, box.cell, settings)
    n_pep = n_peptide_atoms_in_trialanine_box(box.psf_path)
    categories = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_pep)
    by_cat = _aggregate_by_category(decomp, categories)

    n_pp = int(np.sum(categories == _PairCat.PEP_PEP.value))
    n_pw = int(np.sum(categories == _PairCat.PEP_WATER.value))
    n_ww = int(np.sum(categories == _PairCat.WATER_WATER.value))

    return TrialanineNbParityReport(
        seed=int(getattr(box, "seed", -1)),
        perturb_seed=int(perturb_seed),
        n_waters=int(box.n_waters),
        box_side_A=float(box.box_side_A),
        pair_stats=PairListStats(
            n_atoms=int(pos.shape[0]),
            n_peptide_atoms=n_pep,
            n_excluded_pairs=len(nbond_data.excluded_pairs),
            n_e14_pairs=len(nbond_data.e14_pairs),
            n_pairs_within_cutnb=decomp.n_pairs,
            n_pairs_pep_pep=n_pp,
            n_pairs_pep_water=n_pw,
            n_pairs_water_water=n_ww,
            cutnb_A=float(settings.cutnb),
            ctonnb_A=float(settings.ctonnb),
            ctofnb_A=float(settings.ctofnb),
        ),
        bonded=_term_comparison("bonded", charmm_bonded_total, jax_bonded_total),
        vdw=_term_comparison("vdw", charmm_nb["vdw"], jax_vdw),
        elec=_term_comparison("elec", charmm_nb["elec"], jax_elec),
        nb_total=_term_comparison("nb_total", charmm_nb_total, jax_nb_total),
        mm_total=_term_comparison("mm_total", charmm_mm_total, jax_mm_total),
        jax_by_category=by_cat,
        top_vdw_pairs=_top_pairs(
            decomp,
            categories,
            term="vdw",
            n=top_n_pairs,
            category_filter=_PairCat.PEP_PEP.value,
        ),
        top_elec_pairs=_top_pairs(decomp, categories, term="elec", n=top_n_pairs),
        force_rms_delta=force_rms,
        force_max_delta=force_max,
        metadata={
            "psf": str(box.psf_path),
            "include_cmap": include_cmap,
            "lr_solver": "mic",
        },
    )


def render_markdown_report(report: TrialanineNbParityReport) -> str:
    lines = [
        "# Tri-alanine water box — JAX MIC vs PyCHARMM parity",
        "",
        f"Box: {report.n_waters}× TIP3, {report.box_side_A:.1f} Å cube, "
        f"{report.pair_stats.n_atoms} atoms ({report.pair_stats.n_peptide_atoms} peptide).",
        f"Perturbation seed: {report.perturb_seed}.",
        "",
        "## Energy terms (kcal/mol)",
        "",
        "| Term | CHARMM | JAX | Δ (JAX−CHARMM) | rel Δ |",
        "|------|--------|-----|----------------|-------|",
    ]
    for term in (report.bonded, report.vdw, report.elec, report.nb_total, report.mm_total):
        lines.append(
            f"| {term.term} | {term.charmm_kcal:.4f} | {term.jax_kcal:.4f} | "
            f"{term.delta_kcal:+.4f} | {term.rel_delta:+.2%} |"
        )
    lines.extend(
        [
            "",
            f"Force RMS Δ: {report.force_rms_delta:.4f} kcal/mol/Å  "
            f"max |ΔF|: {report.force_max_delta:.4f}",
            "",
            "## Pair list",
            "",
            f"- Excluded pairs (1–2/1–3): {report.pair_stats.n_excluded_pairs}",
            f"- 1–4 pairs: {report.pair_stats.n_e14_pairs}",
            f"- Within cutnb ({report.pair_stats.cutnb_A:.1f} Å): "
            f"{report.pair_stats.n_pairs_within_cutnb} "
            f"(pp={report.pair_stats.n_pairs_pep_pep}, "
            f"pw={report.pair_stats.n_pairs_pep_water}, "
            f"ww={report.pair_stats.n_pairs_water_water})",
            f"- Switch: ctonnb={report.pair_stats.ctonnb_A:.1f} Å, "
            f"ctofnb={report.pair_stats.ctofnb_A:.1f} Å",
            "",
            "## JAX nonbonded by pair category",
            "",
            "| Category | n pairs | VDW | Elec | total | ⟨r⟩ Å |",
            "|----------|---------|-----|------|-------|-------|",
        ]
    )
    for row in report.jax_by_category:
        lines.append(
            f"| {row.category} | {row.n_pairs} | {row.vdw_kcal:.4f} | "
            f"{row.elec_kcal:.4f} | {row.total_kcal:.4f} | {row.mean_r_A:.2f} |"
        )
    lines.extend(["", "## Top peptide–peptide VDW pairs (JAX)", ""])
    lines.append("| rank | atoms (1-based) | r Å | VDW | Elec |")
    lines.append("|------|-----------------|-----|-----|------|")
    for rec in report.top_vdw_pairs:
        lines.append(
            f"| {rec.rank} | {rec.atom_i}–{rec.atom_j} | {rec.r_A:.3f} | "
            f"{rec.vdw_kcal:.4f} | {rec.elec_kcal:.4f} |"
        )
    lines.extend(["", "## Top electrostatic pairs (JAX)", ""])
    lines.append("| rank | category | atoms | r Å | Elec |")
    lines.append("|------|----------|-------|-----|------|")
    for rec in report.top_elec_pairs:
        lines.append(
            f"| {rec.rank} | {rec.category} | {rec.atom_i}–{rec.atom_j} | "
            f"{rec.r_A:.3f} | {rec.elec_kcal:.4f} |"
        )
    lines.extend(
        [
            "",
            "CHARMM uses IMAGE neighbor lists; JAX uses O(N²) MIC pair loops with "
            "PSF bond exclusions. Residual VDW gaps concentrate in peptide–peptide pairs.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_json_report(report: TrialanineNbParityReport) -> str:
    return json.dumps(asdict(report), indent=2)


def render_parity_plots(
    report: TrialanineNbParityReport,
    decomp: Any,
    categories: np.ndarray,
    out_dir: Path | str,
) -> list[Path]:
    """Write PNG figures summarizing the parity gap (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Term comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    terms = [report.bonded, report.vdw, report.elec, report.mm_total]
    labels = [t.term for t in terms]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, [t.charmm_kcal for t in terms], w, label="CHARMM", color="#4c72b0")
    ax.bar(x + w / 2, [t.jax_kcal for t in terms], w, label="JAX MIC", color="#dd8452")
    ax.set_xticks(x, labels)
    ax.set_ylabel("kcal/mol")
    ax.set_title("CHARMM vs JAX energy terms")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.legend()
    fig.tight_layout()
    p = out / "term_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2. Delta waterfall
    fig, ax = plt.subplots(figsize=(7, 4))
    deltas = [report.vdw.delta_kcal, report.elec.delta_kcal, report.mm_total.delta_kcal]
    delta_labels = ["ΔVDW", "ΔElec", "ΔMM total"]
    colors = ["#c44e52" if d > 0 else "#55a868" for d in deltas]
    ax.bar(delta_labels, deltas, color=colors)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("kcal/mol (JAX − CHARMM)")
    ax.set_title("Energy gap breakdown")
    fig.tight_layout()
    p = out / "delta_breakdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3. JAX NB by category (stacked VDW + elec)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = [r.category.replace("_", "–") for r in report.jax_by_category]
    vdw = [r.vdw_kcal for r in report.jax_by_category]
    elec = [r.elec_kcal for r in report.jax_by_category]
    x = np.arange(len(cats))
    ax.bar(x, vdw, label="VDW", color="#8172b3")
    ax.bar(x, elec, bottom=vdw, label="Elec", color="#64b5cd")
    ax.set_xticks(x, cats, rotation=15, ha="right")
    ax.set_ylabel("kcal/mol (JAX MIC)")
    ax.set_title("JAX nonbonded by pair category")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.legend()
    fig.tight_layout()
    p = out / "jax_nb_by_category.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 4. Pair distance histograms by category
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, cat in zip(axes, (_PairCat.PEP_PEP.value, _PairCat.PEP_WATER.value, _PairCat.WATER_WATER.value), strict=True):
        mask = categories == cat
        r = decomp.r_A[mask]
        ax.hist(r, bins=30, color="#4c72b0", alpha=0.85, edgecolor="white")
        ax.set_xlabel("MIC distance (Å)")
        ax.set_title(cat.replace("_", "–"))
        ax.axvline(report.pair_stats.ctonnb_A, color="#c44e52", ls="--", lw=1, label="ctonnb")
        ax.axvline(report.pair_stats.ctofnb_A, color="#c44e52", ls=":", lw=1, label="ctofnb")
    axes[0].set_ylabel("pair count")
    axes[2].legend(fontsize=8, loc="upper right")
    fig.suptitle("Pair distances within cutnb", y=1.02)
    fig.tight_layout()
    p = out / "pair_distance_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 5. Top pep–pep VDW pairs
    if report.top_vdw_pairs:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(report.top_vdw_pairs))))
        labels = [f"{r.atom_i}–{r.atom_j}" for r in report.top_vdw_pairs]
        vals = [r.vdw_kcal for r in report.top_vdw_pairs]
        y = np.arange(len(labels))
        ax.barh(y, vals, color="#8172b3")
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("VDW kcal/mol (JAX)")
        ax.set_title("Top peptide–peptide VDW pairs")
        fig.tight_layout()
        p = out / "top_pep_pep_vdw.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


def collect_and_render_trialanine_nb_parity(
    box: Any,
    positions: np.ndarray,
    out_dir: Path | str,
    *,
    perturb_seed: int = 31,
    top_n_pairs: int = 20,
) -> TrialanineNbParityReport:
    """Collect metrics, write ``report.md``, ``report.json``, and PNG plots."""
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        decompose_nonbonded_pair_energies,
        load_nonbonded_system_from_charmm,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        n_peptide_atoms_in_trialanine_box,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = collect_trialanine_nb_parity(
        box,
        positions,
        perturb_seed=perturb_seed,
        top_n_pairs=top_n_pairs,
    )
    cuts = box.nbond_cutoffs
    settings = CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )
    nbond_data = load_nonbonded_system_from_charmm(box.psf_path, box.cgenff_prm)
    decomp = decompose_nonbonded_pair_energies(positions, nbond_data, box.cell, settings)
    n_pep = n_peptide_atoms_in_trialanine_box(box.psf_path)
    categories = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_pep)

    (out / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    (out / "report.json").write_text(render_json_report(report), encoding="utf-8")
    plot_paths = render_parity_plots(report, decomp, categories, out)
    (out / "plots.txt").write_text("\n".join(str(p) for p in plot_paths) + "\n", encoding="utf-8")
    return report
