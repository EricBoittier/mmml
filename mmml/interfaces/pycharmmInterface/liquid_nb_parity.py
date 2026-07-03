"""Liquid PBC cluster: JAX MIC vs PyCHARMM parity (inter-monomer VDW focus)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.trialanine_nb_parity import (
    CategoryNonbondedTotals,
    PairSwitchAudit,
    TermComparison,
    TopPairRecord,
    _log,
    _term_comparison,
    _top_pairs,
    audit_switch_derivatives,
)

LiquidPairCategory = Literal["intra_monomer", "inter_monomer"]

LIQUID_PAIR_CATEGORIES: tuple[str, ...] = ("intra_monomer", "inter_monomer")


class _LiqCat(str, Enum):
    INTRA = "intra_monomer"
    INTER = "inter_monomer"


def classify_liquid_pair_category(
    i: int,
    j: int,
    monomer_id: np.ndarray,
) -> LiquidPairCategory:
    if int(monomer_id[int(i)]) == int(monomer_id[int(j)]):
        return _LiqCat.INTRA.value
    return _LiqCat.INTER.value


def classify_liquid_pair_categories(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    monomer_id: np.ndarray,
) -> np.ndarray:
    mid = np.asarray(monomer_id, dtype=np.int32)
    same = mid[np.asarray(pair_i, dtype=np.int32)] == mid[np.asarray(pair_j, dtype=np.int32)]
    cats = np.empty(pair_i.shape[0], dtype=object)
    cats[same] = _LiqCat.INTRA.value
    cats[~same] = _LiqCat.INTER.value
    return cats


def monomer_id_from_offsets(monomer_offsets: np.ndarray, natom: int) -> np.ndarray:
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    monomer_id = np.empty(int(natom), dtype=np.int32)
    for mi in range(int(offsets.shape[0]) - 1):
        monomer_id[int(offsets[mi]) : int(offsets[mi + 1])] = mi
    return monomer_id


@dataclass(frozen=True, slots=True)
class LiquidPairListStats:
    n_atoms: int
    n_monomers: int
    n_excluded_pairs: int
    n_e14_pairs: int
    n_pairs_within_cutnb: int
    n_pairs_intra: int
    n_pairs_inter: int
    cutnb_A: float
    ctonnb_A: float
    ctofnb_A: float


@dataclass(frozen=True, slots=True)
class InterMonomerVdwDiagnosis:
    """CHARMM-implied inter-monomer VDW without selective BLOCK (ww analog)."""

    jax_intra_vdw_kcal: float
    jax_inter_vdw_kcal: float
    charmm_total_vdw_kcal: float
    charmm_implied_inter_vdw_kcal: float
    inter_vdw_delta_kcal: float

    @property
    def inter_fraction_of_jax_vdw(self) -> float:
        total = self.jax_intra_vdw_kcal + self.jax_inter_vdw_kcal
        if abs(total) < 1e-12:
            return 0.0
        return self.jax_inter_vdw_kcal / total


@dataclass(frozen=True, slots=True)
class Tip3OoInterTotals:
    """Inter-monomer O–O pairs only (TIP3 OT–OT MIC pairs within cutnb)."""

    n_pairs: int
    vdw_kcal: float
    elec_kcal: float
    mean_r_A: float
    min_r_A: float
    fraction_of_inter_vdw: float


@dataclass(frozen=True, slots=True)
class LiquidNbParityReport:
    seed: int
    perturb_seed: int
    composition: str
    box_side_A: float
    pair_stats: LiquidPairListStats
    bonded: TermComparison
    vdw: TermComparison
    elec: TermComparison
    nb_total: TermComparison
    mm_total: TermComparison
    jax_by_category: tuple[CategoryNonbondedTotals, ...]
    inter_monomer_vdw: InterMonomerVdwDiagnosis
    tip3_oo_inter: Tip3OoInterTotals | None
    top_inter_vdw_pairs: tuple[TopPairRecord, ...]
    top_oo_vdw_pairs: tuple[TopPairRecord, ...]
    top_intra_vdw_pairs: tuple[TopPairRecord, ...]
    switch_derivative_audits: tuple[PairSwitchAudit, ...]
    force_rms_delta: float
    force_max_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


def oxygen_atom_mask_from_psf(psf_path: Path | str) -> np.ndarray:
    """True for CHARMM water oxygens (CGENFF ``OT`` / ``OW`` types)."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import parse_psf_ext

    psf_data = parse_psf_ext(psf_path)
    oxygen_types = frozenset({"OT", "OW", "OH"})
    return np.array(
        [str(t).strip().upper() in oxygen_types for t in psf_data.atom_types],
        dtype=bool,
    )


def aggregate_tip3_oo_inter_pairs(
    decomp: Any,
    categories: np.ndarray,
    oxygen_mask: np.ndarray,
    *,
    inter_vdw_kcal: float,
) -> Tip3OoInterTotals:
    oxy = np.asarray(oxygen_mask, dtype=bool)
    inter = categories == _LiqCat.INTER.value
    pi = np.asarray(decomp.pair_i, dtype=np.int32)
    pj = np.asarray(decomp.pair_j, dtype=np.int32)
    mask = inter & oxy[pi] & oxy[pj]
    n = int(np.sum(mask))
    if n == 0:
        return Tip3OoInterTotals(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r = np.asarray(decomp.r_A[mask], dtype=np.float64)
    vdw = float(np.sum(decomp.vdw_kcal[mask]))
    elec = float(np.sum(decomp.elec_kcal[mask]))
    frac = vdw / float(inter_vdw_kcal) if abs(inter_vdw_kcal) > 1e-12 else 0.0
    return Tip3OoInterTotals(
        n_pairs=n,
        vdw_kcal=vdw,
        elec_kcal=elec,
        mean_r_A=float(np.mean(r)),
        min_r_A=float(np.min(r)),
        fraction_of_inter_vdw=frac,
    )


def _top_oo_vdw_pairs(
    decomp: Any,
    categories: np.ndarray,
    oxygen_mask: np.ndarray,
    *,
    n: int = 10,
) -> tuple[TopPairRecord, ...]:
    oxy = np.asarray(oxygen_mask, dtype=bool)
    inter = categories == _LiqCat.INTER.value
    pi = np.asarray(decomp.pair_i, dtype=np.int32)
    pj = np.asarray(decomp.pair_j, dtype=np.int32)
    mask = inter & oxy[pi] & oxy[pj]
    if not np.any(mask):
        return ()
    sub_i = pi[mask]
    sub_j = pj[mask]
    sub_vdw = np.asarray(decomp.vdw_kcal[mask], dtype=np.float64)
    sub_elec = np.asarray(decomp.elec_kcal[mask], dtype=np.float64)
    sub_r = np.asarray(decomp.r_A[mask], dtype=np.float64)
    order = np.argsort(-np.abs(sub_vdw))[:n]
    records: list[TopPairRecord] = []
    for rank, k in enumerate(order, start=1):
        records.append(
            TopPairRecord(
                rank=rank,
                atom_i=int(sub_i[k]) + 1,
                atom_j=int(sub_j[k]) + 1,
                category=_LiqCat.INTER.value,
                r_A=float(sub_r[k]),
                vdw_kcal=float(sub_vdw[k]),
                elec_kcal=float(sub_elec[k]),
            )
        )
    return tuple(records)


def _aggregate_liquid_by_category(
    decomp: Any,
    categories: np.ndarray,
) -> tuple[CategoryNonbondedTotals, ...]:
    out: list[CategoryNonbondedTotals] = []
    for cat in LIQUID_PAIR_CATEGORIES:
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


def diagnose_inter_monomer_vdw(
    charmm_vdw_kcal: float,
    jax_by_category: Sequence[CategoryNonbondedTotals],
) -> InterMonomerVdwDiagnosis:
    jax_map = {row.category: row for row in jax_by_category}
    intra = float(jax_map.get(_LiqCat.INTRA.value, CategoryNonbondedTotals(_LiqCat.INTRA.value, 0, 0, 0, 0)).vdw_kcal)
    inter = float(jax_map.get(_LiqCat.INTER.value, CategoryNonbondedTotals(_LiqCat.INTER.value, 0, 0, 0, 0)).vdw_kcal)
    charmm_implied_inter = float(charmm_vdw_kcal) - intra
    return InterMonomerVdwDiagnosis(
        jax_intra_vdw_kcal=intra,
        jax_inter_vdw_kcal=inter,
        charmm_total_vdw_kcal=float(charmm_vdw_kcal),
        charmm_implied_inter_vdw_kcal=charmm_implied_inter,
        inter_vdw_delta_kcal=inter - charmm_implied_inter,
    )


def collect_liquid_nb_parity(
    box: Any,
    positions: np.ndarray,
    monomer_id: np.ndarray,
    *,
    perturb_seed: int = 31,
    top_n_pairs: int = 20,
    run_switch_audit: bool = True,
    switch_audit_top_k: int = 5,
    verbose: bool = True,
) -> LiquidNbParityReport:
    """Compare JAX MIC nonbonded decomposition to active PyCHARMM ``ENER FORCE``."""
    import jax.numpy as jnp

    _log("Collecting parity metrics (PyCHARMM ENER + JAX MIC)...", verbose=verbose)
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

    pos = np.asarray(positions, dtype=np.float64)
    monomer_id = np.asarray(monomer_id, dtype=np.int32)
    set_charmm_positions(pos)
    _log("Applying CHARMM MM block + ENER FORCE...", verbose=verbose)
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
        extra_prm_files=getattr(box, "cmap_extra_prm_files", ()),
    )
    nbond_data = load_nonbonded_system_from_charmm(box.psf_path, box.cgenff_prm)
    _log("JAX bonded + nonbonded evaluation...", verbose=verbose)
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
    categories = classify_liquid_pair_categories(decomp.pair_i, decomp.pair_j, monomer_id)
    by_cat = _aggregate_liquid_by_category(decomp, categories)
    inter_diag = diagnose_inter_monomer_vdw(charmm_nb["vdw"], by_cat)

    oxygen_mask = oxygen_atom_mask_from_psf(box.psf_path)
    tip3_oo: Tip3OoInterTotals | None = None
    top_oo: tuple[TopPairRecord, ...] = ()
    if int(np.sum(oxygen_mask)) >= 2:
        tip3_oo = aggregate_tip3_oo_inter_pairs(
            decomp,
            categories,
            oxygen_mask,
            inter_vdw_kcal=inter_diag.jax_inter_vdw_kcal,
        )
        top_oo = _top_oo_vdw_pairs(decomp, categories, oxygen_mask, n=min(top_n_pairs, 15))
        report_meta_oo = {
            "tip3_oo_vdw_kcal": tip3_oo.vdw_kcal,
            "tip3_oo_mean_r_A": tip3_oo.mean_r_A,
            "tip3_oo_n_pairs": tip3_oo.n_pairs,
        }
    else:
        report_meta_oo = {}

    switch_audits: tuple[PairSwitchAudit, ...] = ()
    if run_switch_audit:
        _log(
            f"Switching derivative audit (top {switch_audit_top_k} |VDW| pairs)...",
            verbose=verbose,
        )
        switch_audits = audit_switch_derivatives(
            pos, decomp, nbond_data, box.cell, settings, top_k=switch_audit_top_k
        )

    n_intra = int(np.sum(categories == _LiqCat.INTRA.value))
    n_inter = int(np.sum(categories == _LiqCat.INTER.value))

    report_meta: dict[str, Any] = {
        "psf": str(box.psf_path),
        "include_cmap": include_cmap,
        "lr_solver": "mic",
        "inter_monomer_vdw_delta_kcal": inter_diag.inter_vdw_delta_kcal,
        **report_meta_oo,
    }
    if switch_audits:
        report_meta["switch_vdw_max_dedr_rel_err"] = max(
            a.vdw_dedr_rel_err for a in switch_audits
        )

    return LiquidNbParityReport(
        seed=int(getattr(box, "seed", -1)),
        perturb_seed=int(perturb_seed),
        composition=str(getattr(box, "composition", "")),
        box_side_A=float(box.box_side_A),
        pair_stats=LiquidPairListStats(
            n_atoms=int(pos.shape[0]),
            n_monomers=int(getattr(box, "n_monomers", int(monomer_id.max()) + 1)),
            n_excluded_pairs=len(nbond_data.excluded_pairs),
            n_e14_pairs=len(nbond_data.e14_pairs),
            n_pairs_within_cutnb=decomp.n_pairs,
            n_pairs_intra=n_intra,
            n_pairs_inter=n_inter,
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
        inter_monomer_vdw=inter_diag,
        tip3_oo_inter=tip3_oo,
        top_inter_vdw_pairs=_top_pairs(
            decomp,
            categories,
            term="vdw",
            n=top_n_pairs,
            category_filter=_LiqCat.INTER.value,
        ),
        top_oo_vdw_pairs=top_oo,
        top_intra_vdw_pairs=_top_pairs(
            decomp,
            categories,
            term="vdw",
            n=min(top_n_pairs, 10),
            category_filter=_LiqCat.INTRA.value,
        ),
        switch_derivative_audits=switch_audits,
        force_rms_delta=force_rms,
        force_max_delta=force_max,
        metadata=report_meta,
    )


def render_liquid_markdown_report(report: LiquidNbParityReport) -> str:
    diag = report.inter_monomer_vdw
    lines = [
        f"# {report.composition} liquid box — JAX MIC vs PyCHARMM parity",
        "",
        f"Box: {report.composition}, {report.box_side_A:.1f} Å cube, "
        f"{report.pair_stats.n_atoms} atoms ({report.pair_stats.n_monomers} monomers).",
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
            "## Inter-monomer VDW (water–water analog)",
            "",
            "Without selective BLOCK: CHARMM implied inter = total VDW − JAX intra "
            "(intra is mostly scaled 1–4 within each monomer).",
            "",
            "| Quantity | kcal/mol |",
            "|----------|----------|",
            f"| JAX intra-monomer VDW | {diag.jax_intra_vdw_kcal:.4f} |",
            f"| JAX inter-monomer VDW | {diag.jax_inter_vdw_kcal:.4f} |",
            f"| CHARMM total VDW | {diag.charmm_total_vdw_kcal:.4f} |",
            f"| CHARMM implied inter VDW | {diag.charmm_implied_inter_vdw_kcal:.4f} |",
            f"| **Δ inter VDW (JAX−CHARMM)** | **{diag.inter_vdw_delta_kcal:+.4f}** |",
            f"| Inter share of JAX VDW | {diag.inter_fraction_of_jax_vdw:.1%} |",
            "",
            "## Pair list",
            "",
            f"- Excluded pairs (1–2/1–3): {report.pair_stats.n_excluded_pairs}",
            f"- 1–4 pairs: {report.pair_stats.n_e14_pairs}",
            f"- Within cutnb ({report.pair_stats.cutnb_A:.1f} Å): "
            f"{report.pair_stats.n_pairs_within_cutnb} "
            f"(intra={report.pair_stats.n_pairs_intra}, "
            f"inter={report.pair_stats.n_pairs_inter})",
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
    if report.tip3_oo_inter is not None and report.tip3_oo_inter.n_pairs > 0:
        oo = report.tip3_oo_inter
        lines.extend(
            [
                "",
                "## TIP3 O–O inter-monomer pairs (JAX)",
                "",
                "Oxygen–oxygen MIC pairs between different waters (``OT`` types only). "
                "For TIP3-only boxes this is the VDW-relevant subset of inter-monomer pairs.",
                "",
                "| Quantity | Value |",
                "|----------|-------|",
                f"| n O–O pairs | {oo.n_pairs} |",
                f"| VDW | {oo.vdw_kcal:.4f} kcal/mol |",
                f"| Elec | {oo.elec_kcal:.4f} kcal/mol |",
                f"| ⟨r⟩ | {oo.mean_r_A:.2f} Å |",
                f"| min r | {oo.min_r_A:.2f} Å |",
                f"| Share of inter VDW | {oo.fraction_of_inter_vdw:.1%} |",
            ]
        )
    if report.switch_derivative_audits:
        lines.extend(
            [
                "",
                "## Switching derivative audit (top |VDW| pairs)",
                "",
                "Central difference vs JAX autodiff dE/dr (fswitch/vfswitch).",
                "",
                "| atoms | r Å | VDW | dVDW/dr ana | dVDW/dr num | rel err | dElec/dr rel err |",
                "|-------|-----|-----|-------------|-------------|---------|----------------|",
            ]
        )
        for row in report.switch_derivative_audits:
            lines.append(
                f"| {row.atom_i}–{row.atom_j} | {row.r_A:.3f} | {row.vdw_kcal:.4f} | "
                f"{row.vdw_dedr_analytic:.4f} | {row.vdw_dedr_numeric:.4f} | "
                f"{row.vdw_dedr_rel_err:.2e} | {row.elec_dedr_rel_err:.2e} |"
            )
    if report.top_inter_vdw_pairs:
        lines.extend(
            [
                "",
                "## Top inter-monomer VDW pairs (JAX)",
                "",
                "| rank | atoms (1-based) | r Å | VDW | Elec |",
                "|------|-----------------|-----|-----|------|",
            ]
        )
        for row in report.top_inter_vdw_pairs[:10]:
            lines.append(
                f"| {row.rank} | {row.atom_i}–{row.atom_j} | {row.r_A:.3f} | "
                f"{row.vdw_kcal:.4f} | {row.elec_kcal:.4f} |"
            )
    if report.top_oo_vdw_pairs:
        lines.extend(
            [
                "",
                "## Top O–O inter-monomer VDW pairs (JAX)",
                "",
                "| rank | atoms (1-based) | r Å | VDW | Elec |",
                "|------|-----------------|-----|-----|------|",
            ]
        )
        for row in report.top_oo_vdw_pairs[:10]:
            lines.append(
                f"| {row.rank} | {row.atom_i}–{row.atom_j} | {row.r_A:.3f} | "
                f"{row.vdw_kcal:.4f} | {row.elec_kcal:.4f} |"
            )
    return "\n".join(lines) + "\n"


def render_liquid_json_report(report: LiquidNbParityReport) -> str:
    payload = asdict(report)
    payload["inter_monomer_vdw"] = asdict(report.inter_monomer_vdw)
    if report.tip3_oo_inter is not None:
        payload["tip3_oo_inter"] = asdict(report.tip3_oo_inter)
    return json.dumps(payload, indent=2, default=str) + "\n"


def render_liquid_parity_plots(
    report: LiquidNbParityReport,
    decomp: Any,
    categories: np.ndarray,
    out_dir: Path | str,
    *,
    oxygen_mask: np.ndarray | None = None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out = Path(out_dir)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    terms = ["bonded", "vdw", "elec", "mm_total"]
    charmm_vals = [
        report.bonded.charmm_kcal,
        report.vdw.charmm_kcal,
        report.elec.charmm_kcal,
        report.mm_total.charmm_kcal,
    ]
    jax_vals = [
        report.bonded.jax_kcal,
        report.vdw.jax_kcal,
        report.elec.jax_kcal,
        report.mm_total.jax_kcal,
    ]
    x = np.arange(len(terms))
    w = 0.35
    ax.bar(x - w / 2, charmm_vals, w, label="CHARMM")
    ax.bar(x + w / 2, jax_vals, w, label="JAX")
    ax.set_xticks(x, terms)
    ax.set_ylabel("kcal/mol")
    ax.set_title(f"{report.composition} — term comparison")
    ax.legend()
    fig.tight_layout()
    p = out / "term_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    cats = list(LIQUID_PAIR_CATEGORIES)
    vdw_vals = [
        next((r.vdw_kcal for r in report.jax_by_category if r.category == c), 0.0) for c in cats
    ]
    elec_vals = [
        next((r.elec_kcal for r in report.jax_by_category if r.category == c), 0.0) for c in cats
    ]
    x = np.arange(len(cats))
    ax.bar(x, vdw_vals, label="VDW")
    ax.bar(x, elec_vals, bottom=vdw_vals, label="Elec")
    ax.set_xticks(x, cats, rotation=15)
    ax.set_ylabel("kcal/mol")
    ax.set_title("JAX nonbonded by category")
    ax.legend()
    fig.tight_layout()
    p = out / "jax_nb_by_category.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    diag = report.inter_monomer_vdw
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["JAX inter", "CHARMM\nimplied inter"]
    vals = [diag.jax_inter_vdw_kcal, diag.charmm_implied_inter_vdw_kcal]
    colors = ["#c44e52", "#4c72b0"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylabel("VDW kcal/mol")
    ax.set_title(f"Inter-monomer VDW Δ = {diag.inter_vdw_delta_kcal:+.3f}")
    fig.tight_layout()
    p = out / "inter_monomer_vdw.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    if report.top_inter_vdw_pairs:
        fig, ax = plt.subplots(figsize=(7, 4))
        top = report.top_inter_vdw_pairs[:10]
        labels = [f"{r.atom_i}-{r.atom_j}" for r in top]
        ax.barh(labels[::-1], [r.vdw_kcal for r in top][::-1])
        ax.set_xlabel("VDW kcal/mol")
        ax.set_title("Top inter-monomer VDW pairs (JAX)")
        fig.tight_layout()
        p = out / "top_inter_monomer_vdw.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    for cat, suffix in ((_LiqCat.INTRA.value, "intra"), (_LiqCat.INTER.value, "inter")):
        mask = categories == cat
        if not np.any(mask):
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(decomp.r_A[mask], bins=30, alpha=0.8)
        ax.axvline(report.pair_stats.ctonnb_A, color="g", ls="--", label="ctonnb")
        ax.axvline(report.pair_stats.ctofnb_A, color="r", ls="--", label="ctofnb")
        ax.set_xlabel("MIC distance (Å)")
        ax.set_ylabel("pair count")
        ax.set_title(f"{cat} distance histogram")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / f"pair_distance_hist_{suffix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    if oxygen_mask is not None and report.tip3_oo_inter is not None:
        oxy = np.asarray(oxygen_mask, dtype=bool)
        inter = categories == _LiqCat.INTER.value
        pi = np.asarray(decomp.pair_i, dtype=np.int32)
        pj = np.asarray(decomp.pair_j, dtype=np.int32)
        oo_mask = inter & oxy[pi] & oxy[pj]
        if np.any(oo_mask):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(decomp.r_A[oo_mask], bins=30, alpha=0.8, color="#4c72b0")
            ax.axvline(report.pair_stats.ctonnb_A, color="g", ls="--", label="ctonnb")
            ax.axvline(report.pair_stats.ctofnb_A, color="r", ls="--", label="ctofnb")
            ax.set_xlabel("O–O MIC distance (Å)")
            ax.set_ylabel("pair count")
            ax.set_title(
                f"O–O inter VDW sum = {report.tip3_oo_inter.vdw_kcal:.3f} kcal/mol"
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = out / "oo_inter_distance_hist.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(p)

    return written


def collect_and_render_liquid_nb_parity(
    box: Any,
    positions: np.ndarray,
    monomer_id: np.ndarray,
    out_dir: Path | str,
    *,
    perturb_seed: int = 31,
    top_n_pairs: int = 20,
    run_switch_audit: bool = True,
    switch_audit_top_k: int = 5,
    verbose: bool = True,
) -> LiquidNbParityReport:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        decompose_nonbonded_pair_energies,
        load_nonbonded_system_from_charmm,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = collect_liquid_nb_parity(
        box,
        positions,
        monomer_id,
        perturb_seed=perturb_seed,
        top_n_pairs=top_n_pairs,
        run_switch_audit=run_switch_audit,
        switch_audit_top_k=switch_audit_top_k,
        verbose=verbose,
    )
    cuts = box.nbond_cutoffs
    settings = CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )
    nbond_data = load_nonbonded_system_from_charmm(box.psf_path, box.cgenff_prm)
    decomp = decompose_nonbonded_pair_energies(positions, nbond_data, box.cell, settings)
    categories = classify_liquid_pair_categories(decomp.pair_i, decomp.pair_j, monomer_id)

    _log("Writing report and plots...", verbose=verbose)
    (out / "report.md").write_text(render_liquid_markdown_report(report), encoding="utf-8")
    (out / "report.json").write_text(render_liquid_json_report(report), encoding="utf-8")
    oxy_mask = oxygen_atom_mask_from_psf(box.psf_path)
    plot_paths = render_liquid_parity_plots(
        report, decomp, categories, out, oxygen_mask=oxy_mask
    )
    (out / "plots.txt").write_text("\n".join(str(p) for p in plot_paths) + "\n", encoding="utf-8")
    return report
