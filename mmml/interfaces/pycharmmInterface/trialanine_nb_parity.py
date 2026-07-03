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


def _charmm_dict_to_category_totals(
    charmm_cats: dict[str, dict[str, float | np.ndarray]],
) -> tuple[CategoryNonbondedTotals, ...]:
    out: list[CategoryNonbondedTotals] = []
    for cat in (_PairCat.PEP_PEP.value, _PairCat.PEP_WATER.value, _PairCat.WATER_WATER.value):
        row = charmm_cats.get(cat)
        if row is None:
            out.append(CategoryNonbondedTotals(cat, 0, 0.0, 0.0, 0.0))
            continue
        out.append(
            CategoryNonbondedTotals(
                category=cat,
                n_pairs=0,
                vdw_kcal=float(row["vdw"]),
                elec_kcal=float(row["elec"]),
                mean_r_A=0.0,
            )
        )
    return tuple(out)


def _mic_unit_vector(
    positions: np.ndarray,
    i: int,
    j: int,
    cell: np.ndarray,
) -> np.ndarray:
    cell_mat = np.asarray(cell, dtype=np.float64)
    if cell_mat.shape == (3,):
        cell_mat = np.diag(cell_mat)
    inv = np.linalg.inv(cell_mat)
    dr = np.asarray(positions[j], dtype=np.float64) - np.asarray(positions[i], dtype=np.float64)
    frac = dr @ inv.T
    frac = frac - np.round(frac)
    disp = frac @ cell_mat
    norm = float(np.linalg.norm(disp))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return disp / norm


def _single_pair_nb_energies(
    positions: np.ndarray,
    pair_i: int,
    pair_j: int,
    nbond_data: Any,
    cell: np.ndarray,
    settings: Any,
) -> tuple[float, float]:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        decompose_nonbonded_pair_energies,
    )

    pi = np.asarray([pair_i], dtype=np.int32)
    pj = np.asarray([pair_j], dtype=np.int32)
    decomp = decompose_nonbonded_pair_energies(
        positions,
        nbond_data,
        cell,
        settings,
        pair_i=pi,
        pair_j=pj,
    )
    return float(decomp.vdw_kcal[0]), float(decomp.elec_kcal[0])


def _single_pair_analytic_dedr(
    positions: np.ndarray,
    pair_i: int,
    pair_j: int,
    nbond_data: Any,
    cell: np.ndarray,
    settings: Any,
    r_hat: np.ndarray,
) -> tuple[float, float]:
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        decompose_nonbonded_pair_energies,
    )

    pos = jnp.asarray(positions, dtype=jnp.float64)
    pi = np.asarray([pair_i], dtype=np.int32)
    pj = np.asarray([pair_j], dtype=np.int32)

    def _vdw_energy(p: jnp.ndarray) -> jnp.ndarray:
        d = decompose_nonbonded_pair_energies(
            p, nbond_data, cell, settings, pair_i=pi, pair_j=pj
        )
        return jnp.asarray(d.vdw_kcal[0], dtype=jnp.float64)

    def _elec_energy(p: jnp.ndarray) -> jnp.ndarray:
        d = decompose_nonbonded_pair_energies(
            p, nbond_data, cell, settings, pair_i=pi, pair_j=pj
        )
        return jnp.asarray(d.elec_kcal[0], dtype=jnp.float64)

    grad_vdw = np.asarray(jax.grad(_vdw_energy)(pos), dtype=np.float64)
    grad_elec = np.asarray(jax.grad(_elec_energy)(pos), dtype=np.float64)
    # dE/dr along i→j: ∂E/∂x_j · r̂  (equivalently −∂E/∂x_i · r̂).
    dedr_v = float(np.dot(grad_vdw[pair_j], r_hat))
    dedr_e = float(np.dot(grad_elec[pair_j], r_hat))
    return dedr_v, dedr_e


def _relative_deriv_error(analytic: float, numeric: float) -> float:
    denom = max(abs(analytic), abs(numeric), 1e-12)
    return abs(analytic - numeric) / denom


def audit_switch_derivatives(
    positions: np.ndarray,
    decomp: Any,
    nbond_data: Any,
    cell: np.ndarray,
    settings: Any,
    *,
    top_k: int = 10,
    dr: float = 1e-4,
) -> tuple[PairSwitchAudit, ...]:
    """Compare JAX autodiff dE/dr vs central difference for top |VDW| pairs."""
    order = np.argsort(-np.abs(decomp.vdw_kcal))[:top_k]
    pos = np.asarray(positions, dtype=np.float64)
    audits: list[PairSwitchAudit] = []
    for k in order:
        i = int(decomp.pair_i[k])
        j = int(decomp.pair_j[k])
        r_hat = _mic_unit_vector(pos, i, j, cell)
        if float(np.linalg.norm(r_hat)) < 1e-12:
            continue
        vdw0, elec0 = _single_pair_nb_energies(pos, i, j, nbond_data, cell, settings)
        pos_plus = pos.copy()
        pos_minus = pos.copy()
        pos_plus[i] -= 0.5 * dr * r_hat
        pos_plus[j] += 0.5 * dr * r_hat
        pos_minus[i] += 0.5 * dr * r_hat
        pos_minus[j] -= 0.5 * dr * r_hat
        vdw_p, elec_p = _single_pair_nb_energies(pos_plus, i, j, nbond_data, cell, settings)
        vdw_m, elec_m = _single_pair_nb_energies(pos_minus, i, j, nbond_data, cell, settings)
        dedr_v_num = (vdw_p - vdw_m) / dr
        dedr_e_num = (elec_p - elec_m) / dr
        dedr_v_ana, dedr_e_ana = _single_pair_analytic_dedr(
            pos, i, j, nbond_data, cell, settings, r_hat
        )
        audits.append(
            PairSwitchAudit(
                atom_i=i + 1,
                atom_j=j + 1,
                r_A=float(decomp.r_A[k]),
                vdw_kcal=vdw0,
                vdw_dedr_analytic=dedr_v_ana,
                vdw_dedr_numeric=dedr_v_num,
                vdw_dedr_rel_err=_relative_deriv_error(dedr_v_ana, dedr_v_num),
                elec_kcal=elec0,
                elec_dedr_analytic=dedr_e_ana,
                elec_dedr_numeric=dedr_e_num,
                elec_dedr_rel_err=_relative_deriv_error(dedr_e_ana, dedr_e_num),
            )
        )
    return tuple(audits)


def _log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _collect_charmm_category_breakdown(
    jax_by_category: tuple[CategoryNonbondedTotals, ...],
    jax_cat_forces: dict[str, np.ndarray],
    *,
    enabled: bool,
) -> tuple[
    tuple[CategoryNonbondedTotals, ...],
    tuple[TermComparison, ...],
    tuple[CategoryForceDelta, ...],
    dict[str, Any],
]:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_nonbonded_by_segment_category,
    )
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        selective_bonded_block_unsafe_under_mpi,
    )

    meta: dict[str, Any] = {}
    if not enabled:
        meta["charmm_category_skipped"] = "disabled (pass --category-block to enable)"
        return (), (), (), meta
    if selective_bonded_block_unsafe_under_mpi():
        meta["charmm_category_skipped"] = (
            "selective_BLOCK_unsafe_under_mpi "
            "(set MMML_ALLOW_SELECTIVE_BONDED_BLOCK=1 and use --category-block)"
        )
        return (), (), (), meta
    _log("Running CHARMM segment BLOCK category breakdown (3× ENER FORCE)...")
    try:
        charmm_raw = charmm_nonbonded_by_segment_category(restore_full_mm_block=True)
    except Exception as exc:
        meta["charmm_category_error"] = str(exc)
        return (), (), (), meta
    meta["charmm_category_method"] = "segment_BLOCK"
    charmm_by = _charmm_dict_to_category_totals(charmm_raw)
    jax_map = {row.category: row for row in jax_by_category}
    vdw_terms: list[TermComparison] = []
    force_rows: list[CategoryForceDelta] = []
    for cat in (_PairCat.PEP_PEP.value, _PairCat.PEP_WATER.value, _PairCat.WATER_WATER.value):
        ch_row = next((r for r in charmm_by if r.category == cat), None)
        jax_row = jax_map.get(cat)
        if ch_row is None or jax_row is None:
            continue
        vdw_terms.append(_term_comparison(f"vdw_{cat}", ch_row.vdw_kcal, jax_row.vdw_kcal))
        ch_f = np.asarray(charmm_raw[cat]["forces"], dtype=np.float64)
        jax_f = jax_cat_forces.get(cat, np.zeros_like(ch_f))
        force_rows.append(
            CategoryForceDelta(
                category=cat,
                jax_force_rms=_force_rms(jax_f),
                charmm_force_rms=_force_rms(ch_f),
                delta_force_rms=_force_delta_rms(jax_f, ch_f),
                vdw_delta_kcal=jax_row.vdw_kcal - ch_row.vdw_kcal,
            )
        )
    return charmm_by, tuple(vdw_terms), tuple(force_rows), meta


def _jax_category_forces(
    positions: np.ndarray,
    nbond_data: Any,
    cell: np.ndarray,
    settings: Any,
    decomp: Any,
    categories: np.ndarray,
) -> dict[str, np.ndarray]:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        nonbonded_energy_and_forces,
    )

    out: dict[str, np.ndarray] = {}
    for cat in (_PairCat.PEP_PEP.value, _PairCat.PEP_WATER.value, _PairCat.WATER_WATER.value):
        mask = categories == cat
        if not np.any(mask):
            out[cat] = np.zeros((positions.shape[0], 3), dtype=np.float64)
            continue
        _, forces = nonbonded_energy_and_forces(
            positions,
            nbond_data,
            cell,
            settings,
            pair_i=decomp.pair_i[mask],
            pair_j=decomp.pair_j[mask],
        )
        out[cat] = np.asarray(forces, dtype=np.float64)
    return out


def _force_rms(forces: np.ndarray) -> float:
    f = np.asarray(forces, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(f * f, axis=-1))))


def _force_delta_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def collect_trialanine_nb_parity(
    box: Any,
    positions: np.ndarray,
    *,
    perturb_seed: int = 31,
    top_n_pairs: int = 20,
    run_category_block: bool = False,
    run_switch_audit: bool = True,
    switch_audit_top_k: int = 5,
    verbose: bool = True,
) -> TrialanineNbParityReport:
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
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        n_peptide_atoms_in_trialanine_box,
    )

    pos = np.asarray(positions, dtype=np.float64)
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
        extra_prm_files=box.cmap_extra_prm_files,
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
    n_pep = n_peptide_atoms_in_trialanine_box(box.psf_path)
    categories = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_pep)
    by_cat = _aggregate_by_category(decomp, categories)
    _log("JAX masked-pair forces by category...", verbose=verbose)
    jax_cat_forces = _jax_category_forces(
        pos, nbond_data, box.cell, settings, decomp, categories
    )
    charmm_by_cat, category_vdw, category_force_delta, cat_meta = (
        _collect_charmm_category_breakdown(
            by_cat, jax_cat_forces, enabled=run_category_block
        )
    )
    switch_audits: tuple[PairSwitchAudit, ...] = ()
    if run_switch_audit:
        _log(
            f"Switching derivative audit (top {switch_audit_top_k} |VDW| pairs)...",
            verbose=verbose,
        )
        switch_audits = audit_switch_derivatives(
            pos, decomp, nbond_data, box.cell, settings, top_k=switch_audit_top_k
        )

    n_pp = int(np.sum(categories == _PairCat.PEP_PEP.value))
    n_pw = int(np.sum(categories == _PairCat.PEP_WATER.value))
    n_ww = int(np.sum(categories == _PairCat.WATER_WATER.value))

    report_meta = {
        "psf": str(box.psf_path),
        "include_cmap": include_cmap,
        "lr_solver": "mic",
        **cat_meta,
    }
    if switch_audits:
        report_meta["switch_vdw_max_dedr_rel_err"] = max(
            a.vdw_dedr_rel_err for a in switch_audits
        )
        report_meta["switch_elec_max_dedr_rel_err"] = max(
            a.elec_dedr_rel_err for a in switch_audits
        )

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
        charmm_by_category=charmm_by_cat,
        category_vdw=category_vdw,
        category_force_delta=category_force_delta,
        top_vdw_pairs=_top_pairs(
            decomp,
            categories,
            term="vdw",
            n=top_n_pairs,
            category_filter=_PairCat.PEP_PEP.value,
        ),
        top_elec_pairs=_top_pairs(decomp, categories, term="elec", n=top_n_pairs),
        switch_derivative_audits=switch_audits,
        force_rms_delta=force_rms,
        force_max_delta=force_max,
        metadata=report_meta,
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
    if report.charmm_by_category:
        lines.extend(
            [
                "",
                "## CHARMM nonbonded by segment BLOCK",
                "",
                "| Category | VDW | Elec | total |",
                "|----------|-----|------|-------|",
            ]
        )
        for row in report.charmm_by_category:
            lines.append(
                f"| {row.category} | {row.vdw_kcal:.4f} | "
                f"{row.elec_kcal:.4f} | {row.total_kcal:.4f} |"
            )
    if report.category_vdw:
        lines.extend(
            [
                "",
                "## VDW gap by category (JAX − CHARMM)",
                "",
                "| Category | CHARMM | JAX | Δ |",
                "|----------|--------|-----|---|",
            ]
        )
        for term in report.category_vdw:
            lines.append(
                f"| {term.term.replace('vdw_', '')} | {term.charmm_kcal:.4f} | "
                f"{term.jax_kcal:.4f} | {term.delta_kcal:+.4f} |"
            )
    if report.category_force_delta:
        lines.extend(
            [
                "",
                "## Force RMS by category (masked pair lists)",
                "",
                "| Category | JAX RMS | CHARMM RMS | Δ RMS | ΔVDW |",
                "|----------|---------|------------|-------|------|",
            ]
        )
        for row in report.category_force_delta:
            lines.append(
                f"| {row.category} | {row.jax_force_rms:.4f} | "
                f"{row.charmm_force_rms:.4f} | {row.delta_force_rms:.4f} | "
                f"{row.vdw_delta_kcal:+.4f} |"
            )
    if report.switch_derivative_audits:
        lines.extend(
            [
                "",
                "## Switching derivative audit (top |VDW| pairs)",
                "",
                "Central difference vs JAX autodiff dE/dr (fswitch/vfswitch). "
                "Large rel errors implicate switch implementation, not pair lists.",
                "",
                "| atoms | r Å | VDW | dVDW/dr ana | dVDW/dr num | rel err | "
                "dElec/dr rel err |",
                "|-------|-----|-----|-------------|-------------|---------|"
                "----------------|",
            ]
        )
        for a in report.switch_derivative_audits:
            lines.append(
                f"| {a.atom_i}–{a.atom_j} | {a.r_A:.3f} | {a.vdw_kcal:.4f} | "
                f"{a.vdw_dedr_analytic:.4f} | {a.vdw_dedr_numeric:.4f} | "
                f"{a.vdw_dedr_rel_err:.2e} | {a.elec_dedr_rel_err:.2e} |"
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

    # 3b. CHARMM vs JAX VDW by category
    if report.category_vdw:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        labels = [t.term.replace("vdw_", "").replace("_", "–") for t in report.category_vdw]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(
            x - w / 2,
            [t.charmm_kcal for t in report.category_vdw],
            w,
            label="CHARMM BLOCK",
            color="#4c72b0",
        )
        ax.bar(
            x + w / 2,
            [t.jax_kcal for t in report.category_vdw],
            w,
            label="JAX MIC",
            color="#dd8452",
        )
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("VDW kcal/mol")
        ax.set_title("VDW by pair category")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.legend()
        fig.tight_layout()
        p = out / "category_vdw_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3c. Force RMS delta by category
    if report.category_force_delta:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [r.category.replace("_", "–") for r in report.category_force_delta]
        vals = [r.delta_force_rms for r in report.category_force_delta]
        ax.bar(labels, vals, color="#c44e52")
        ax.set_ylabel("kcal/mol/Å")
        ax.set_title("Force RMS Δ (JAX masked − CHARMM BLOCK)")
        fig.tight_layout()
        p = out / "force_by_category.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3d. Switch derivative audit
    if report.switch_derivative_audits:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        audits = report.switch_derivative_audits
        vdw_ana = [a.vdw_dedr_analytic for a in audits]
        vdw_num = [a.vdw_dedr_numeric for a in audits]
        elec_ana = [a.elec_dedr_analytic for a in audits]
        elec_num = [a.elec_dedr_numeric for a in audits]
        lim_v = max(max(map(abs, vdw_ana + vdw_num), default=1.0), 1e-6)
        lim_e = max(max(map(abs, elec_ana + elec_num), default=1.0), 1e-6)
        axes[0].scatter(vdw_num, vdw_ana, c="#8172b3", s=36)
        axes[0].plot([-lim_v, lim_v], [-lim_v, lim_v], "k--", lw=0.8)
        axes[0].set_xlabel("dVDW/dr numeric")
        axes[0].set_ylabel("dVDW/dr autodiff")
        axes[0].set_title("VDW switch derivative")
        axes[1].scatter(elec_num, elec_ana, c="#64b5cd", s=36)
        axes[1].plot([-lim_e, lim_e], [-lim_e, lim_e], "k--", lw=0.8)
        axes[1].set_xlabel("dElec/dr numeric")
        axes[1].set_ylabel("dElec/dr autodiff")
        axes[1].set_title("Elec fswitch derivative")
        fig.suptitle("Switching derivative self-consistency (JAX)", y=1.02)
        fig.tight_layout()
        p = out / "switch_derivative_audit.png"
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
    run_category_block: bool = False,
    run_switch_audit: bool = True,
    switch_audit_top_k: int = 5,
    verbose: bool = True,
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
        run_category_block=run_category_block,
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
    n_pep = n_peptide_atoms_in_trialanine_box(box.psf_path)
    categories = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_pep)

    _log("Writing report and plots...", verbose=verbose)
    (out / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    (out / "report.json").write_text(render_json_report(report), encoding="utf-8")
    plot_paths = render_parity_plots(report, decomp, categories, out)
    (out / "plots.txt").write_text("\n".join(str(p) for p in plot_paths) + "\n", encoding="utf-8")
    return report
