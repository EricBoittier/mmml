"""PCA for acquisition embeddings — the only dimensionality reduction used.

UMAP, t-SNE, and random projections are intentionally unsupported.

Fits must use acquisition-accessible rows only.  Held-out validation / test
trajectories must not influence centering, scaling, or components.  Callers
pass the fit-set IDs; :func:`assert_no_leakage` checks that held-out IDs are
absent from that set.

Two PCA uses are documented separately:

1. **Centered embedding PCA** (activations, energy-Jacobian vectors, loss
   gradients used with farthest-point sampling).  Rows are centered and
   optionally standard-scaled.  Distances are Euclidean in the resulting
   coordinates.

2. **Parameter-space Jacobian PCA** for information-gain compression.
   A shared basis is estimated from stacked Jacobian *rows*.  When the
   information matrix is formed, those rows are projected with
   ``J @ components.T`` **without subtracting the row mean** used to estimate
   the basis.  Centering would change the Gram matrix ``J^T J`` that D-optimal
   selection actually needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class PCAFit:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    """``(n_components, n_features)``, rows are principal axes."""
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    n_samples: int
    n_features: int
    n_components: int
    centered: bool
    scaled: bool
    fit_ids: tuple[str, ...]
    retained_variance: float
    solver: str = "full_svd"
    kind: str = "embedding"
    """``embedding`` (centered distances) or ``jacobian_basis`` (uncentered projection)."""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("mean", "scale", "components", "explained_variance", "explained_variance_ratio"):
            payload[key] = np.asarray(payload[key]).tolist()
        payload["fit_ids"] = list(self.fit_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PCAFit":
        kw = dict(payload)
        for key in ("mean", "scale", "components", "explained_variance", "explained_variance_ratio"):
            kw[key] = np.asarray(kw[key], dtype=np.float64)
        kw["fit_ids"] = tuple(kw.get("fit_ids") or ())
        return cls(**kw)


def _resolve_n_components(
    n_samples: int,
    n_features: int,
    *,
    n_components: int | None,
    variance_threshold: float | None,
    explained_ratio: np.ndarray,
) -> int:
    max_k = max(1, min(n_samples, n_features))
    if n_components is not None and variance_threshold is not None:
        raise ValueError("specify only one of n_components or variance_threshold")
    if variance_threshold is not None:
        if not 0.0 < float(variance_threshold) <= 1.0:
            raise ValueError("variance_threshold must be in (0, 1]")
        cum = np.cumsum(explained_ratio)
        k = int(np.searchsorted(cum, float(variance_threshold), side="left") + 1)
        return min(max(k, 1), max_k)
    if n_components is None:
        return max_k
    k = int(n_components)
    if k < 1:
        raise ValueError("n_components must be >= 1")
    return min(k, max_k)


def fit_pca(
    X: np.ndarray,
    *,
    n_components: int | None = None,
    variance_threshold: float | None = None,
    center: bool = True,
    scale: bool = True,
    fit_ids: Sequence[str] | None = None,
    kind: str = "embedding",
) -> PCAFit:
    """Fit PCA on ``X`` with shape ``(n_samples, n_features)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n_samples, n_features = X.shape
    if n_samples < 1 or n_features < 1:
        raise ValueError("X is empty")
    mean = X.mean(axis=0) if center else np.zeros(n_features, dtype=np.float64)
    xc = X - mean
    if scale:
        std = xc.std(axis=0, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)
    else:
        std = np.ones(n_features, dtype=np.float64)
    xs = xc / std
    # SVD of the (centered, scaled) matrix.
    _, s, vt = np.linalg.svd(xs, full_matrices=False)
    # Population-style explained variance matching sklearn's n_samples factor
    # for the *ratio*; we store both.
    n_sv = s.shape[0]
    denom = max(n_samples - 1, 1)
    exp_var = (s ** 2) / denom
    total = float(exp_var.sum()) if exp_var.size else 0.0
    ratio = exp_var / total if total > 0 else np.zeros_like(exp_var)
    k = _resolve_n_components(
        n_samples,
        n_sv,
        n_components=n_components,
        variance_threshold=variance_threshold,
        explained_ratio=ratio,
    )
    components = vt[:k]
    return PCAFit(
        mean=mean,
        scale=std,
        components=components,
        explained_variance=exp_var[:k],
        explained_variance_ratio=ratio[:k],
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_components=int(k),
        centered=bool(center),
        scaled=bool(scale),
        fit_ids=tuple(str(i) for i in (fit_ids or ())),
        retained_variance=float(ratio[:k].sum()) if ratio.size else 0.0,
        kind=str(kind),
    )


def transform_pca(X: np.ndarray, fit: PCAFit, *, apply_center: bool | None = None) -> np.ndarray:
    """Project rows of ``X`` into the fitted PCA basis.

    ``apply_center`` defaults to ``fit.centered``.  For Jacobian information
    matrices pass ``apply_center=False`` even if the basis was estimated from
    centered rows.
    """
    X = np.asarray(X, dtype=np.float64)
    use_center = fit.centered if apply_center is None else bool(apply_center)
    y = X - fit.mean if use_center else X
    y = y / fit.scale
    return y @ fit.components.T


def project_jacobian_rows(J: np.ndarray, fit: PCAFit) -> np.ndarray:
    """Project Jacobian rows without centering (information-matrix convention).

    ``J`` is ``(n_obs, n_params)``.  The result is ``(n_obs, n_components)``.
    """
    if fit.kind != "jacobian_basis":
        # Still allowed, but the caller should have fitted with kind=jacobian_basis.
        pass
    return transform_pca(J, fit, apply_center=False)


def assert_no_leakage(fit: PCAFit, held_out_ids: Sequence[str]) -> None:
    """Raise if any held-out ID participated in the PCA fit."""
    fit_set = set(fit.fit_ids)
    leaked = [i for i in held_out_ids if i in fit_set]
    if leaked:
        raise ValueError(
            "PCA leakage: held-out IDs were in the fit set: " + ", ".join(leaked[:8])
        )
    if held_out_ids and not fit.fit_ids:
        raise ValueError(
            "PCA fit recorded no fit_ids; cannot prove held-out trajectories were excluded"
        )
