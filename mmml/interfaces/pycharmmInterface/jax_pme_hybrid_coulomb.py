"""Hybrid ML/MM jax-pme: cross-monomer periodic Coulomb and r⁻⁶ LJ dispersion.

The hybrid ``build_mm_energy_forces_fn`` path evaluates MM on **cross-monomer**
pairs with COM switching (r⁻¹² repulsion when jax-pme supplies dispersion).
Full-box jax-pme includes intra-monomer terms that must not enter the MM term.

Each MM long-range contribution is::

    scale * (E_pme_full - E_intra)

where ``E_intra`` is the sum of jax-pme energies over each monomer slice in the
same periodic box.  COM switching uses the same sharpstep logic as the pair loop.

A finer SR/LR split (MIC cross pairs + ``E_pme - E_intra - E_mic``) is available
for Coulomb testing but is not used in production: CHARMM MIC 1/r does not match
the jax-pme screened real-space kernel within ``sr_cutoff_A``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    CHARMM_COULOMB_KCAL,
    box_length_from_cell,
    compute_jax_pme_coulomb,
    compute_jax_pme_lj_dispersion,
    compute_jax_pme_power_law,
)


@dataclass(frozen=True)
class HybridJaxPmeCorrectionResult:
    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray
    energy_intra_kcalmol: float
    energy_mic_cross_kcalmol: float
    switch_scale: float


# Backward-compatible alias
HybridJaxPmeCoulombResult = HybridJaxPmeCorrectionResult


@dataclass(frozen=True)
class HybridJaxPmeMmResult:
    """Combined Coulomb + r⁻⁶ dispersion hybrid corrections."""

    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray
    coulomb: HybridJaxPmeCorrectionResult
    dispersion: HybridJaxPmeCorrectionResult | None


def _mic_displacement_np(ri: np.ndarray, rj: np.ndarray, cell: np.ndarray) -> np.ndarray:
    cell_mat = np.asarray(cell, dtype=np.float64)
    if cell_mat.ndim == 1:
        cell_mat = np.diag(cell_mat)
    inv = np.linalg.inv(cell_mat)
    dr = np.asarray(rj, dtype=np.float64) - np.asarray(ri, dtype=np.float64)
    frac = dr @ inv.T
    frac -= np.round(frac)
    return frac @ cell_mat


def _mm_switch_scales(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    complementary_handoff: bool,
    pbc_cell: np.ndarray | None,
    mm_r_min: float | None,
) -> np.ndarray:
    """COM switching factors per monomer pair (same logic as ``mm_energy_forces``)."""
    from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep
    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

    n_monomers = int(len(monomer_offsets) - 1)
    coms = np.stack(
        [
            positions[int(monomer_offsets[k]) : int(monomer_offsets[k + 1])].mean(axis=0)
            for k in range(n_monomers)
        ],
        axis=0,
    )
    scales = np.ones(max(1, n_monomers * (n_monomers - 1) // 2), dtype=np.float64)
    idx = 0
    for mi in range(n_monomers):
        for mj in range(mi + 1, n_monomers):
            if pbc_cell is not None:
                d_vec = _mic_displacement_np(coms[mi], coms[mj], pbc_cell)
                r = float(np.linalg.norm(d_vec))
            else:
                r = float(np.linalg.norm(coms[mj] - coms[mi]))
            if mm_r_min is not None and r < float(mm_r_min):
                scale = 0.0
            elif complementary_handoff:
                handoff = float(
                    _sharpstep(r, mm_switch_on - ml_switch_width, mm_switch_on, gamma=GAMMA_ON)
                )
                taper = 1.0 - float(
                    _sharpstep(r, mm_switch_on, mm_switch_on + mm_switch_width, gamma=GAMMA_OFF)
                )
                scale = handoff * taper
            else:
                mm_on = float(
                    _sharpstep(r, mm_switch_on, mm_switch_on + mm_switch_width, gamma=GAMMA_ON)
                )
                mm_off = float(
                    _sharpstep(
                        r,
                        mm_switch_on + mm_switch_width,
                        mm_switch_on + 2.0 * mm_switch_width,
                        gamma=GAMMA_OFF,
                    )
                )
                scale = mm_on * (1.0 - mm_off)
            if idx < scales.shape[0]:
                scales[idx] = scale
            idx += 1
    return scales[:idx] if idx else np.array([0.0], dtype=np.float64)


def _mean_switch_scale(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    pbc_cell: np.ndarray | None,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    complementary_handoff: bool,
    mm_r_min: float | None,
) -> float:
    if pbc_cell is None:
        return 1.0
    scales = _mm_switch_scales(
        positions,
        monomer_offsets,
        ml_switch_width=ml_switch_width,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        complementary_handoff=complementary_handoff,
        pbc_cell=np.asarray(pbc_cell, dtype=np.float64),
        mm_r_min=mm_r_min,
    )
    return float(np.mean(scales)) if scales.size else 0.0


def _intra_monomer_jax_pme_power_law(
    positions_A: np.ndarray,
    coefficients: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
    exponent: int,
    prefactor: float | None,
) -> HybridJaxPmeCorrectionResult:
    """Sum jax-pme 1/r^p over each monomer slice (intra-monomer contribution)."""
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    n_monomers = int(len(offsets) - 1)
    energy = 0.0
    forces = np.zeros_like(pos)
    for m in range(n_monomers):
        sl = slice(int(offsets[m]), int(offsets[m + 1]))
        if sl.stop - sl.start == 0:
            continue
        sub = compute_jax_pme_power_law(
            pos[sl],
            coef[sl],
            box_length_A=float(box_length_A),
            method=method,
            sr_cutoff_A=sr_cutoff_A,
            exponent=int(exponent),
            prefactor=prefactor,
        )
        energy += float(sub.energy_kcalmol)
        forces[sl] += sub.forces_kcalmol_A
    return HybridJaxPmeCorrectionResult(
        energy_kcalmol=energy,
        forces_kcalmol_A=forces,
        energy_intra_kcalmol=energy,
        energy_mic_cross_kcalmol=0.0,
        switch_scale=1.0,
    )


def intra_monomer_jax_pme_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
) -> HybridJaxPmeCorrectionResult:
    """Sum jax-pme Coulomb over each monomer slice (intra-monomer contribution)."""
    from jaxpme import prefactors as jpref

    return _intra_monomer_jax_pme_power_law(
        positions_A,
        charges_e,
        monomer_offsets,
        box_length_A=box_length_A,
        method=method,
        sr_cutoff_A=sr_cutoff_A,
        exponent=1,
        prefactor=float(jpref.kcalmol_A),
    )


def intra_monomer_jax_pme_lj_dispersion(
    positions_A: np.ndarray,
    c6_sqrt: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
) -> HybridJaxPmeCorrectionResult:
    """Sum jax-pme r⁻⁶ dispersion over each monomer slice."""
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        DEFAULT_JAX_PME_LJ_PREFACTOR,
    )

    return _intra_monomer_jax_pme_power_law(
        positions_A,
        c6_sqrt,
        monomer_offsets,
        box_length_A=box_length_A,
        method=method,
        sr_cutoff_A=sr_cutoff_A,
        exponent=6,
        prefactor=DEFAULT_JAX_PME_LJ_PREFACTOR,
    )


def mic_cross_coulomb_unswitched(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_mask: np.ndarray,
    monomer_id: np.ndarray,
    *,
    pbc_cell: np.ndarray,
    lambda_monomer: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """MIC 1/r Coulomb on cross-monomer pairs (no COM switching), for SR subtraction."""
    pos = np.asarray(positions_A, dtype=np.float64)
    chg = np.asarray(charges_e, dtype=np.float64)
    pi = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pj = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    mask = np.asarray(pair_mask, dtype=np.float64).reshape(-1) > 0
    mid = np.asarray(monomer_id, dtype=np.int64)
    lam = np.asarray(lambda_monomer, dtype=np.float64) if lambda_monomer is not None else None

    forces = np.zeros_like(pos)
    energy = 0.0
    eps = 1e-10
    for k in range(pi.shape[0]):
        if not mask[k] or pi[k] >= pj[k]:
            continue
        if mid[pi[k]] == mid[pj[k]]:
            continue
        d_vec = _mic_displacement_np(pos[pi[k]], pos[pj[k]], pbc_cell)
        r = max(float(np.linalg.norm(d_vec)), eps)
        qq = chg[pi[k]] * chg[pj[k]]
        if lam is not None:
            qq *= float(lam[mid[pi[k]]] * lam[mid[pj[k]]])
        e_ij = CHARMM_COULOMB_KCAL * qq / r
        f_ij = CHARMM_COULOMB_KCAL * qq / (r**3) * d_vec
        energy += e_ij
        forces[pi[k]] -= f_ij
        forces[pj[k]] += f_ij
    return energy, forces


def _hybrid_jax_pme_power_law_correction(
    positions_A: np.ndarray,
    coefficients: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
    exponent: int,
    prefactor: float | None,
    switch_scale: float,
    full_compute,
    intra_compute,
) -> HybridJaxPmeCorrectionResult:
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    full = full_compute(pos, coef)
    intra = intra_compute(pos, coef)
    lr_e = float(full.energy_kcalmol) - intra.energy_kcalmol
    lr_f = np.asarray(full.forces_kcalmol_A - intra.forces_kcalmol_A, dtype=np.float64)
    return HybridJaxPmeCorrectionResult(
        energy_kcalmol=switch_scale * lr_e,
        forces_kcalmol_A=switch_scale * lr_f,
        energy_intra_kcalmol=intra.energy_kcalmol,
        energy_mic_cross_kcalmol=0.0,
        switch_scale=switch_scale,
    )


def hybrid_jax_pme_coulomb_correction(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
    pair_i: np.ndarray | None = None,
    pair_j: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    monomer_id: np.ndarray | None = None,
    lambda_monomer: np.ndarray | None = None,
    pbc_cell: np.ndarray | None = None,
    ml_switch_width: float = 1.0,
    mm_switch_on: float = 12.0,
    mm_switch_width: float = 1.0,
    complementary_handoff: bool = True,
    mm_r_min: float | None = None,
    subtract_pair_mic_sr: bool = False,
) -> HybridJaxPmeCorrectionResult:
    """Periodic Coulomb correction: jax-pme full box minus intra-monomer terms."""
    pos = np.asarray(positions_A, dtype=np.float64)
    chg = np.asarray(charges_e, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    switch_scale = _mean_switch_scale(
        pos,
        offsets,
        pbc_cell=pbc_cell,
        ml_switch_width=ml_switch_width,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        complementary_handoff=complementary_handoff,
        mm_r_min=mm_r_min,
    )

    full = compute_jax_pme_coulomb(
        pos,
        chg,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=sr_cutoff_A,
    )
    intra = intra_monomer_jax_pme_coulomb(
        pos,
        chg,
        offsets,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=sr_cutoff_A,
    )
    mic_e = 0.0
    mic_f = np.zeros_like(pos)
    if subtract_pair_mic_sr and (
        pair_i is not None
        and pair_j is not None
        and pair_mask is not None
        and monomer_id is not None
        and pbc_cell is not None
    ):
        mic_e, mic_f = mic_cross_coulomb_unswitched(
            pos,
            chg,
            pair_i,
            pair_j,
            pair_mask,
            monomer_id,
            pbc_cell=np.asarray(pbc_cell, dtype=np.float64),
            lambda_monomer=lambda_monomer,
        )

    lr_e = float(full.energy_kcalmol) - intra.energy_kcalmol - mic_e
    lr_f = np.asarray(full.forces_kcalmol_A - intra.forces_kcalmol_A - mic_f, dtype=np.float64)

    return HybridJaxPmeCorrectionResult(
        energy_kcalmol=switch_scale * lr_e,
        forces_kcalmol_A=switch_scale * lr_f,
        energy_intra_kcalmol=intra.energy_kcalmol,
        energy_mic_cross_kcalmol=mic_e,
        switch_scale=switch_scale,
    )


def hybrid_jax_pme_lj_dispersion_correction(
    positions_A: np.ndarray,
    c6_sqrt: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
    switch_scale: float,
) -> HybridJaxPmeCorrectionResult:
    """Periodic r⁻⁶ LJ tail: jax-pme full box minus intra-monomer dispersion."""
    pos = np.asarray(positions_A, dtype=np.float64)
    coef = np.asarray(c6_sqrt, dtype=np.float64).reshape(-1)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)

    full = compute_jax_pme_lj_dispersion(
        pos,
        coef,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=sr_cutoff_A,
    )
    intra = intra_monomer_jax_pme_lj_dispersion(
        pos,
        coef,
        offsets,
        box_length_A=float(box_length_A),
        method=method,
        sr_cutoff_A=sr_cutoff_A,
    )
    lr_e = float(full.energy_kcalmol) - intra.energy_kcalmol
    lr_f = np.asarray(full.forces_kcalmol_A - intra.forces_kcalmol_A, dtype=np.float64)
    return HybridJaxPmeCorrectionResult(
        energy_kcalmol=switch_scale * lr_e,
        forces_kcalmol_A=switch_scale * lr_f,
        energy_intra_kcalmol=intra.energy_kcalmol,
        energy_mic_cross_kcalmol=0.0,
        switch_scale=switch_scale,
    )


def hybrid_jax_pme_mm_lr_correction(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    box_length_A: float,
    method: str,
    sr_cutoff_A: float,
    c6_sqrt: np.ndarray | None = None,
    pair_i: np.ndarray | None = None,
    pair_j: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    monomer_id: np.ndarray | None = None,
    lambda_monomer: np.ndarray | None = None,
    pbc_cell: np.ndarray | None = None,
    ml_switch_width: float = 1.0,
    mm_switch_on: float = 12.0,
    mm_switch_width: float = 1.0,
    complementary_handoff: bool = True,
    mm_r_min: float | None = None,
    subtract_pair_mic_sr: bool = False,
) -> HybridJaxPmeMmResult:
    """Combined hybrid Coulomb + r⁻⁶ dispersion (each full − intra, same COM scale)."""
    coulomb = hybrid_jax_pme_coulomb_correction(
        positions_A,
        charges_e,
        monomer_offsets,
        box_length_A=box_length_A,
        method=method,
        sr_cutoff_A=sr_cutoff_A,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_mask=pair_mask,
        monomer_id=monomer_id,
        lambda_monomer=lambda_monomer,
        pbc_cell=pbc_cell,
        ml_switch_width=ml_switch_width,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        complementary_handoff=complementary_handoff,
        mm_r_min=mm_r_min,
        subtract_pair_mic_sr=subtract_pair_mic_sr,
    )
    dispersion: HybridJaxPmeCorrectionResult | None = None
    if c6_sqrt is not None:
        dispersion = hybrid_jax_pme_lj_dispersion_correction(
            positions_A,
            c6_sqrt,
            monomer_offsets,
            box_length_A=box_length_A,
            method=method,
            sr_cutoff_A=sr_cutoff_A,
            switch_scale=coulomb.switch_scale,
        )
    e = coulomb.energy_kcalmol
    f = np.asarray(coulomb.forces_kcalmol_A, dtype=np.float64)
    if dispersion is not None:
        e += dispersion.energy_kcalmol
        f = f + dispersion.forces_kcalmol_A
    return HybridJaxPmeMmResult(
        energy_kcalmol=e,
        forces_kcalmol_A=f,
        coulomb=coulomb,
        dispersion=dispersion,
    )


def box_length_from_positions_context(
    pbc_cell: np.ndarray | None,
    box_override: np.ndarray | None = None,
) -> float:
    if box_override is not None:
        return box_length_from_cell(np.asarray(box_override))
    if pbc_cell is None:
        raise ValueError("jax_pme requires pbc_cell or box_override")
    return box_length_from_cell(np.asarray(pbc_cell))
