"""Extract activation / Jacobian / loss-gradient representations.

Teacher energies and forces are used only for the surrogate loss-gradient
method.  Expensive reference labels are not accepted here; passing them is an
error so a wiring bug cannot leak ground truth into acquisition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mmml.acquisition.linear_student import LinearStudent
from mmml.acquisition.pca import PCAFit, fit_pca, project_jacobian_rows, transform_pca
from mmml.acquisition.pooling import pool_species_aware
from mmml.acquisition.splits import StructureRecord


@dataclass
class ExtractStats:
    kind: str
    n_structures: int
    n_features: int
    runtime_s: float
    teacher_calls: int
    peak_features: int
    notes: list[str] = field(default_factory=list)


def _forbid_reference_labels(records: Sequence[StructureRecord]) -> None:
    for rec in records:
        extra = rec.extra or {}
        if extra.get("reference_energy") is not None or extra.get("reference_forces") is not None:
            raise ValueError(
                f"reference labels leaked into acquisition for {rec.structure_id}"
            )


def extract_activations(
    student: LinearStudent,
    records: Sequence[StructureRecord],
    *,
    species: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[str], ExtractStats, list[np.ndarray]]:
    _forbid_reference_labels(records)
    t0 = time.perf_counter()
    atom_feats = []
    zs = []
    ids = []
    for rec in records:
        phi = student.atomic_features(rec.positions, rec.atomic_numbers)
        atom_feats.append(phi)
        zs.append(np.asarray(rec.atomic_numbers, dtype=np.int32))
        ids.append(rec.structure_id)
    pooled = pool_species_aware(atom_feats, zs, species=species, include_max=True)
    stats = ExtractStats(
        kind="activations",
        n_structures=len(records),
        n_features=int(pooled.vectors.shape[1]),
        runtime_s=time.perf_counter() - t0,
        teacher_calls=0,
        peak_features=int(pooled.vectors.shape[1]),
        notes=["species-aware mean+max+global-mean pooling of invariant atomic features"],
    )
    return pooled.vectors, ids, stats, atom_feats


def extract_energy_jacobians(
    student: LinearStudent,
    records: Sequence[StructureRecord],
) -> tuple[np.ndarray, list[str], ExtractStats]:
    _forbid_reference_labels(records)
    t0 = time.perf_counter()
    rows = [student.energy_jacobian(rec.positions, rec.atomic_numbers) for rec in records]
    X = np.stack(rows, axis=0) if rows else np.zeros((0, 0))
    stats = ExtractStats(
        kind="energy_jacobian",
        n_structures=len(records),
        n_features=int(X.shape[1]) if X.ndim == 2 and X.size else 0,
        runtime_s=time.perf_counter() - t0,
        teacher_calls=0,
        peak_features=int(X.shape[1]) if X.size else 0,
        notes=["∇_readout E; final linear energy head including per-element bias"],
    )
    return X, [r.structure_id for r in records], stats


def extract_force_information_blocks(
    student: LinearStudent,
    records: Sequence[StructureRecord],
    *,
    energy_weight: float,
    forces_weight: float,
) -> tuple[list[np.ndarray], list[str], ExtractStats]:
    _forbid_reference_labels(records)
    t0 = time.perf_counter()
    blocks = [
        student.information_block(
            rec.positions,
            rec.atomic_numbers,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            include_forces=True,
        )
        for rec in records
    ]
    p = int(blocks[0].shape[1]) if blocks else 0
    stats = ExtractStats(
        kind="force_jacobian",
        n_structures=len(records),
        n_features=p,
        runtime_s=time.perf_counter() - t0,
        teacher_calls=0,
        peak_features=p,
        notes=[
            "J_x stacks √α_E ∇E and √(α_F/N) ∇F rows; α from the fine-tuning loss",
        ],
    )
    return blocks, [r.structure_id for r in records], stats


def extract_loss_gradients(
    student: LinearStudent,
    teacher: LinearStudent,
    records: Sequence[StructureRecord],
    *,
    energy_weight: float,
    forces_weight: float,
) -> tuple[np.ndarray, list[str], ExtractStats]:
    _forbid_reference_labels(records)
    t0 = time.perf_counter()
    rows = []
    teacher_calls = 0
    for rec in records:
        e_t, f_t = teacher.energy_forces(rec.positions, rec.atomic_numbers)
        teacher_calls += 1
        g = student.loss_gradient(
            rec.positions,
            rec.atomic_numbers,
            energy_target=e_t,
            forces_target=f_t,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
        )
        rows.append(g)
    X = np.stack(rows, axis=0) if rows else np.zeros((0, 0))
    stats = ExtractStats(
        kind="loss_gradient",
        n_structures=len(records),
        n_features=int(X.shape[1]) if X.size else 0,
        runtime_s=time.perf_counter() - t0,
        teacher_calls=teacher_calls,
        peak_features=int(X.shape[1]) if X.size else 0,
        notes=[
            "g_x = ∇_w [α_E (E_S-E_T)^2 + α_F |F_S-F_T|^2] with training reductions",
            "measures student–teacher discrepancy; cannot find errors both models share",
            "must not use the student's own predictions as labels (those gradients vanish)",
        ],
    )
    return X, [r.structure_id for r in records], stats


def activation_energy_alignment(
    activations: np.ndarray,
    energy_jacobians: np.ndarray,
) -> dict[str, float]:
    """Cosine similarity between pooled activations and energy Jacobians.

    For a purely linear readout of the same features, this is 1.  The actual
    PhysNet architecture has a two-step readout (e3x Dense then Dense) plus
    optional energy_bias / ZBL / electrostatics, so the two representations
    need not be equivalent — that is reported, not assumed.
    """
    A = np.asarray(activations, dtype=np.float64)
    G = np.asarray(energy_jacobians, dtype=np.float64)
    if A.shape[0] != G.shape[0]:
        raise ValueError("row count mismatch")
    # Compare after centering each row's scale via cosine on the shared
    # overlapping dimensions if widths differ (species-pooled act vs readout jac).
    n = A.shape[0]
    cos = []
    rel = []
    for i in range(n):
        a = A[i]
        g = G[i]
        # If dims differ, cosine is undefined; report NaN for that pair and
        # also a Procrustes-free correlation via ranking of norms.
        if a.shape != g.shape:
            cos.append(float("nan"))
            rel.append(float("nan"))
            continue
        na = np.linalg.norm(a)
        ng = np.linalg.norm(g)
        if na < 1e-12 or ng < 1e-12:
            cos.append(float("nan"))
            rel.append(float("nan"))
            continue
        cos.append(float(np.dot(a, g) / (na * ng)))
        rel.append(float(np.linalg.norm(a - g) / max(ng, 1e-12)))
    cos_a = np.asarray(cos, dtype=np.float64)
    return {
        "n": float(n),
        "mean_cosine": float(np.nanmean(cos_a)) if cos_a.size else float("nan"),
        "min_cosine": float(np.nanmin(cos_a)) if cos_a.size else float("nan"),
        "mean_relative_l2": float(np.nanmean(rel)) if rel else float("nan"),
        "dims_equal": bool(A.shape[1] == G.shape[1]) if A.ndim == 2 and G.ndim == 2 else False,
        "activation_dim": int(A.shape[1]) if A.ndim == 2 else 0,
        "jacobian_dim": int(G.shape[1]) if G.ndim == 2 else 0,
    }


def linear_readout_pooled_equals_energy_grad(
    student: LinearStudent,
    records: Sequence[StructureRecord],
) -> dict[str, float]:
    """Pooled φ (sum over atoms, no species channels) vs ∇_w E for the linear student."""
    pooled = []
    grads = []
    for rec in records:
        phi = student.atomic_features(rec.positions, rec.atomic_numbers)
        # Energy jacobian w.r.t. w is sum_i φ_i; bias jacobian is per-element counts.
        pooled.append(phi.sum(axis=0))
        g = student.energy_jacobian(rec.positions, rec.atomic_numbers)
        grads.append(g[: phi.shape[1]])
    A = np.stack(pooled)
    G = np.stack(grads)
    return activation_energy_alignment(A, G)


def fit_embedding_pca(
    X: np.ndarray,
    ids: Sequence[str],
    *,
    n_components: int | None,
    variance_threshold: float | None,
    row_normalize: bool,
) -> tuple[PCAFit, np.ndarray]:
    from mmml.acquisition.selection import maybe_row_normalize

    Xn = maybe_row_normalize(X, row_normalize)
    fit = fit_pca(
        Xn,
        n_components=n_components,
        variance_threshold=variance_threshold,
        center=True,
        scale=True,
        fit_ids=ids,
        kind="embedding",
    )
    Z = transform_pca(Xn, fit, apply_center=True)
    return fit, Z


def fit_jacobian_basis(
    blocks: Sequence[np.ndarray],
    ids: Sequence[str],
    *,
    n_components: int | None,
    variance_threshold: float | None,
) -> tuple[PCAFit, list[np.ndarray]]:
    """Shared parameter-space PCA; project rows without centering."""
    if not blocks:
        raise ValueError("no Jacobian blocks")
    stacked = np.vstack([np.asarray(b, dtype=np.float64) for b in blocks if np.asarray(b).size])
    fit = fit_pca(
        stacked,
        n_components=n_components,
        variance_threshold=variance_threshold,
        center=True,  # basis estimated from centered rows
        scale=False,
        fit_ids=ids,
        kind="jacobian_basis",
    )
    projected = [project_jacobian_rows(np.asarray(b, dtype=np.float64), fit) for b in blocks]
    return fit, projected
