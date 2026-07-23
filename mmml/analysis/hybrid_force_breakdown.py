"""Decompose hybrid ModelOutput forces into ML/MM/wall residual terms.

Units: energies in eV, forces in eV/Å.  ``ModelOutput.forces`` is the total
force on each atom (not −∇E stored under another name).  Term forces already
share that sign convention.

``wall_F`` is not stored on ``ModelOutput``; it is recovered as the residual
after subtracting ``internal_F``, ``ml_2b_F``, and ``mm_F`` from the total
(plus any flat-bottom / COM-restraint / MBD forces that are not exposed as
separate force arrays).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Residual |F| below this (eV/Å) is treated as numerical noise in reports.
FORCE_TERM_RESIDUAL_NOISE_EVA = 1.0e-6


@dataclass(frozen=True)
class ForceTermStats:
    """Per-term force magnitude statistics (eV/Å) and optional energy (eV)."""

    name: str
    max_abs_eVA: float
    mean_abs_eVA: float
    rms_eVA: float
    energy_eV: float | None = None
    # Fraction of total max|F| explained by this term's max|F| (not a partition).
    max_vs_total_max: float | None = None
    # Mean cosine of per-atom alignment with the total force (1 = parallel).
    mean_align_with_total: float | None = None


def _as_force_array(value: Any, n_atoms: int) -> np.ndarray | None:
    """Return ``(n_atoms, 3)`` forces or ``None`` when the term is inactive."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if abs(float(arr)) < FORCE_TERM_RESIDUAL_NOISE_EVA:
            return None
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] != 3 * n_atoms:
            return None
        arr = arr.reshape(n_atoms, 3)
    if arr.shape != (n_atoms, 3):
        return None
    if not np.any(np.isfinite(arr)):
        return None
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if float(np.max(np.abs(arr))) < FORCE_TERM_RESIDUAL_NOISE_EVA:
        return None
    return arr


def _scalar_energy(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    x = float(arr[0])
    return x if np.isfinite(x) else None


def force_magnitude_stats(
    forces: np.ndarray,
    *,
    name: str,
    energy_eV: float | None = None,
    total_forces: np.ndarray | None = None,
) -> ForceTermStats:
    """Compute max/mean/RMS of per-atom |F| for one term."""
    f = np.asarray(forces, dtype=float).reshape(-1, 3)
    mags = np.linalg.norm(f, axis=-1)
    max_abs = float(np.max(mags)) if mags.size else 0.0
    mean_abs = float(np.mean(mags)) if mags.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mags)))) if mags.size else 0.0
    max_vs_total = None
    mean_align = None
    if total_forces is not None:
        tot = np.asarray(total_forces, dtype=float).reshape(-1, 3)
        tot_mags = np.linalg.norm(tot, axis=-1)
        tot_max = float(np.max(tot_mags)) if tot_mags.size else 0.0
        if tot_max > FORCE_TERM_RESIDUAL_NOISE_EVA:
            max_vs_total = max_abs / tot_max
        # Per-atom cosine; skip near-zero total force atoms.
        dots = np.sum(f * tot, axis=-1)
        denom = mags * tot_mags
        mask = denom > FORCE_TERM_RESIDUAL_NOISE_EVA
        if np.any(mask):
            mean_align = float(np.mean(dots[mask] / denom[mask]))
    return ForceTermStats(
        name=name,
        max_abs_eVA=max_abs,
        mean_abs_eVA=mean_abs,
        rms_eVA=rms,
        energy_eV=energy_eV,
        max_vs_total_max=max_vs_total,
        mean_align_with_total=mean_align,
    )


def hybrid_force_term_breakdown(
    model_output: Any,
    *,
    atomic_numbers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a serializable force/energy term breakdown from ``ModelOutput``.

    Returns a dict with ``terms`` (list of :class:`ForceTermStats` as dicts),
    ``dominant_term`` (highest mean |F|), ``total``, and optional
    ``by_element`` max|F| for the total force.
    """
    total = np.asarray(getattr(model_output, "forces"), dtype=float).reshape(-1, 3)
    n_atoms = int(total.shape[0])
    total_stats = force_magnitude_stats(
        total,
        name="total",
        energy_eV=_scalar_energy(getattr(model_output, "energy", None)),
    )

    internal = _as_force_array(getattr(model_output, "internal_F", None), n_atoms)
    ml_2b = _as_force_array(getattr(model_output, "ml_2b_F", None), n_atoms)
    mm = _as_force_array(getattr(model_output, "mm_F", None), n_atoms)

    residual = total.copy()
    for part in (internal, ml_2b, mm):
        if part is not None:
            residual = residual - part
    residual_arr = _as_force_array(residual, n_atoms)

    term_specs: list[tuple[str, np.ndarray | None, float | None]] = [
        (
            "internal_ML",
            internal,
            _scalar_energy(getattr(model_output, "internal_E", None)),
        ),
        (
            "ml_dimer",
            ml_2b,
            _scalar_energy(getattr(model_output, "ml_2b_E", None)),
        ),
        (
            "mm",
            mm,
            _scalar_energy(getattr(model_output, "mm_E", None)),
        ),
        (
            "residual_wall_mbd_restraints",
            residual_arr,
            _scalar_energy(getattr(model_output, "wall_E", None)),
        ),
    ]

    terms: list[ForceTermStats] = []
    for name, forces, energy in term_specs:
        if forces is None:
            terms.append(
                ForceTermStats(
                    name=name,
                    max_abs_eVA=0.0,
                    mean_abs_eVA=0.0,
                    rms_eVA=0.0,
                    energy_eV=energy,
                    max_vs_total_max=0.0,
                    mean_align_with_total=None,
                )
            )
            continue
        terms.append(
            force_magnitude_stats(
                forces,
                name=name,
                energy_eV=energy,
                total_forces=total,
            )
        )

    active = [t for t in terms if t.mean_abs_eVA > FORCE_TERM_RESIDUAL_NOISE_EVA]
    dominant = max(active, key=lambda t: t.mean_abs_eVA).name if active else "total"

    by_element: dict[str, dict[str, float]] = {}
    if atomic_numbers is not None:
        z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        if z.shape[0] == n_atoms:
            tot_mags = np.linalg.norm(total, axis=-1)
            for zi in sorted(set(int(x) for x in z)):
                mask = z == zi
                m = tot_mags[mask]
                by_element[str(zi)] = {
                    "n_atoms": int(mask.sum()),
                    "max_abs_eVA": float(np.max(m)) if m.size else 0.0,
                    "mean_abs_eVA": float(np.mean(m)) if m.size else 0.0,
                }

    mm_vdw_E = _scalar_energy(getattr(model_output, "mm_vdw_E", None))
    mm_elec_E = _scalar_energy(getattr(model_output, "mm_elec_E", None))
    wall_E = _scalar_energy(getattr(model_output, "wall_E", None))
    mbd_E = _scalar_energy(getattr(model_output, "mbd_E", None))

    return {
        "units": {"energy": "eV", "force": "eV/Å"},
        "n_atoms": n_atoms,
        "total": asdict(total_stats),
        "terms": [asdict(t) for t in terms],
        "dominant_term_by_mean_abs_F": dominant,
        "energy_bookkeeping_eV": {
            "mm_vdw_E": mm_vdw_E,
            "mm_elec_E": mm_elec_E,
            "wall_E": wall_E,
            "mbd_E": mbd_E,
        },
        "by_element_total_F": by_element,
        "notes": [
            "internal_ML = monomer (1-body) PhysNet forces",
            "ml_dimer = switched ML 2-body forces",
            "mm = JAX MM (LJ + Coulomb/Ewald) forces",
            "residual_wall_mbd_restraints ≈ wall + MBD + COM/flat-bottom "
            "(wall_F is not a separate ModelOutput field)",
        ],
    }


def print_hybrid_force_term_breakdown(
    breakdown: Mapping[str, Any],
    *,
    title: str = "Hybrid force-term breakdown",
) -> None:
    """Emit a compact CLI table for :func:`hybrid_force_term_breakdown`."""
    from mmml.utils.rich_report import get_reporter

    reporter = get_reporter()
    rows = []
    for term in breakdown.get("terms", []):
        e = term.get("energy_eV")
        e_s = f"{e:.4f}" if e is not None and np.isfinite(e) else "—"
        align = term.get("mean_align_with_total")
        a_s = f"{align:.3f}" if align is not None and np.isfinite(align) else "—"
        rows.append(
            (
                term["name"],
                e_s,
                f"{term['max_abs_eVA']:.4f}",
                f"{term['mean_abs_eVA']:.4f}",
                f"{term['rms_eVA']:.4f}",
                a_s,
            )
        )
    tot = breakdown.get("total") or {}
    rows.append(
        (
            "total",
            f"{tot.get('energy_eV', float('nan')):.4f}"
            if tot.get("energy_eV") is not None
            else "—",
            f"{tot.get('max_abs_eVA', 0.0):.4f}",
            f"{tot.get('mean_abs_eVA', 0.0):.4f}",
            f"{tot.get('rms_eVA', 0.0):.4f}",
            "1.000",
        )
    )
    reporter.table(
        title,
        columns=["term", "E (eV)", "max|F|", "mean|F|", "RMS|F|", "align"],
        rows=rows,
    )
    book = breakdown.get("energy_bookkeeping_eV") or {}
    reporter.summary(
        "Energy bookkeeping (eV)",
        {
            "dominant_term (by mean|F|)": breakdown.get("dominant_term_by_mean_abs_F"),
            "mm_vdw_E": book.get("mm_vdw_E"),
            "mm_elec_E": book.get("mm_elec_E"),
            "wall_E": book.get("wall_E"),
            "mbd_E": book.get("mbd_E"),
        },
    )
    by_el = breakdown.get("by_element_total_F") or {}
    if by_el:
        reporter.table(
            "Total |F| by atomic number",
            columns=["Z", "n", "max|F|", "mean|F|"],
            rows=[
                (
                    z,
                    str(v["n_atoms"]),
                    f"{v['max_abs_eVA']:.4f}",
                    f"{v['mean_abs_eVA']:.4f}",
                )
                for z, v in by_el.items()
            ],
        )


def write_hybrid_force_term_breakdown_json(
    breakdown: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write breakdown dict as JSON; return the resolved path."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(breakdown, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
