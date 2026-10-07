"""Precomputed per-monomer latent-charge templates for liquid MD (Mode D).

``mm_charge_mode=latent_mean`` (see :mod:`mmml.models.mm_charge_mode`) needs a
*fixed* set of per-atom charges for one monomer, derived offline by averaging
``neutralize_per_monomer(q_ML)`` (the same quantity Mode B uses live) over many
dimer forwards of a trained checkpoint -- see
``scripts/compute_latent_monomer_charges.py``.  This module is the shared
save/load/tile contract between that script and the MD calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LatentChargeTemplate",
    "load_latent_charge_template",
    "save_latent_charge_template",
    "tile_latent_charge_template",
]

_NET_CHARGE_TOL = 1e-3


@dataclass(frozen=True)
class LatentChargeTemplate:
    """One monomer's worth of averaged latent MM charges.

    Attributes
    ----------
    atomic_numbers
        ``(n_atoms_per_monomer,)`` -- for a sanity check against the liquid
        box's actual monomer composition at load time.
    charges
        ``(n_atoms_per_monomer,)`` mean ``neutralize_per_monomer(q_ML)`` over
        the source dataset, in electron charge units (same convention as
        CGenFF/PSF charges).
    charges_std
        Per-atom standard deviation over the averaged samples (diagnostic
        only -- large values mean the latent charge is not well approximated
        by a single frozen template for this species).
    n_samples
        Number of dimer structures averaged over.
    resid
        Residue/monomer name the template was computed for.
    source_checkpoint, source_data
        Provenance strings, recorded for `hybrid_mm.json`-style traceability.
    """

    atomic_numbers: np.ndarray
    charges: np.ndarray
    charges_std: np.ndarray
    n_samples: int
    resid: str
    source_checkpoint: str
    source_data: str


def save_latent_charge_template(path: str | Path, template: LatentChargeTemplate) -> None:
    """Write a template to ``path`` (``.npz``)."""
    np.savez(
        Path(path),
        atomic_numbers=np.asarray(template.atomic_numbers, dtype=np.int32),
        charges=np.asarray(template.charges, dtype=np.float64),
        charges_std=np.asarray(template.charges_std, dtype=np.float64),
        n_samples=np.int64(template.n_samples),
        resid=str(template.resid),
        source_checkpoint=str(template.source_checkpoint),
        source_data=str(template.source_data),
    )


def load_latent_charge_template(path: str | Path) -> LatentChargeTemplate:
    """Read a template saved by :func:`save_latent_charge_template`.

    Raises if the charges are not net-neutral within ``1e-3 e`` -- a
    non-neutral template makes the liquid Ewald sum ill-defined (net charge
    per periodic image), and the whole point of averaging
    ``neutralize_per_monomer`` outputs is that the mean stays neutral too.
    """
    with np.load(Path(path), allow_pickle=False) as d:
        template = LatentChargeTemplate(
            atomic_numbers=np.asarray(d["atomic_numbers"], dtype=np.int32),
            charges=np.asarray(d["charges"], dtype=np.float64),
            charges_std=np.asarray(d["charges_std"], dtype=np.float64),
            n_samples=int(d["n_samples"]),
            resid=str(d["resid"]),
            source_checkpoint=str(d["source_checkpoint"]),
            source_data=str(d["source_data"]),
        )
    net = float(np.sum(template.charges))
    if abs(net) > _NET_CHARGE_TOL:
        raise ValueError(
            f"latent charge template {path} has net charge {net:.4f} e "
            f"(> {_NET_CHARGE_TOL} e tolerance) -- a non-neutral per-monomer "
            "template makes the tiled liquid box non-neutral, which breaks "
            "the Ewald sum's conditional convergence. Regenerate it with "
            "scripts/compute_latent_monomer_charges.py."
        )
    return template


def tile_latent_charge_template(
    template: LatentChargeTemplate | np.ndarray, n_monomers: int
) -> np.ndarray:
    """Tile one monomer's charge template across ``n_monomers`` copies.

    Assumes a homogeneous liquid: every monomer has the same size and the
    same atom ordering as the template (monomer-blocked layout, i.e. atom
    ``m * n_per_monomer + k`` is atom ``k`` of monomer ``m`` -- the layout
    ``monomer_offsets``/``atoms_per_monomer_list`` produce when all monomers
    are equal size). Heterogeneous mixtures are not supported by Mode D v1.
    """
    charges = (
        template.charges if isinstance(template, LatentChargeTemplate) else np.asarray(template)
    )
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    return np.tile(charges, int(n_monomers))
