"""Always-on geometry / topology diagnostics for prep-ladder checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | Path


def _resolve_checkpoint_psf(mlpot_ctx: Any | None) -> Path | None:
    if mlpot_ctx is None:
        return None
    topo = getattr(mlpot_ctx, "topology_psf_path", None)
    if topo is not None and Path(topo).is_file():
        return Path(topo).expanduser().resolve()
    return None


def _bond_stretch_summary(positions: np.ndarray, psf_path: Path) -> dict[str, float | int]:
    """Bond stretch stats from a cluster PSF + CGENFF parameters."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import load_cgenff_bonded_from_psf

    pos = np.asarray(positions, dtype=np.float64)
    system = load_cgenff_bonded_from_psf(psf_path, pos)
    bonds = np.asarray(system.topology.bonds, dtype=np.int32)
    r0 = np.asarray(system.bonded.bond_r0, dtype=np.float64)
    if bonds.size == 0:
        return {"n_bonds": 0, "n_stretched": 0, "max_stretch_A": 0.0, "mean_stretch_A": 0.0}
    lengths = np.linalg.norm(pos[bonds[:, 0]] - pos[bonds[:, 1]], axis=1)
    stretch = lengths - r0
    stretched = lengths > np.maximum(1.25 * r0, r0 + 0.45)
    return {
        "n_bonds": int(bonds.shape[0]),
        "n_stretched": int(np.count_nonzero(stretched)),
        "max_stretch_A": float(np.max(stretch)),
        "mean_stretch_A": float(np.mean(np.abs(stretch))),
    }


def print_topology_composition_note(
    mlpot_ctx: Any | None,
    *,
    context: str,
) -> None:
    """Print live CHARMM PSF composition vs stored cluster PSF (never gates)."""
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
            capture_topology_fingerprint_from_charmm,
            describe_fingerprint_diff,
            fingerprints_equivalent,
            resolve_topology_fingerprint,
        )
    except Exception as exc:
        print(f"{context} topology: (unavailable: {exc})", flush=True)
        return

    try:
        live = capture_topology_fingerprint_from_charmm()
    except Exception as exc:
        print(f"{context} topology: live CHARMM PSF unreadable ({exc})", flush=True)
        return

    stored = getattr(mlpot_ctx, "topology_fingerprint", None) if mlpot_ctx else None
    topo_path = _resolve_checkpoint_psf(mlpot_ctx)
    if stored is None and topo_path is not None:
        stored = resolve_topology_fingerprint(topo_path)

    if stored is None:
        print(
            f"{context} topology: live natom={live.natom} nres={live.nres} "
            "(no stored cluster PSF fingerprint to compare)",
            flush=True,
        )
        return

    if fingerprints_equivalent(stored, live):
        note = "matches stored cluster PSF fingerprint"
    else:
        note = describe_fingerprint_diff(stored, live)
    topo_txt = str(topo_path) if topo_path is not None else "(unknown path)"
    print(
        f"{context} topology: live natom={live.natom} nres={live.nres}; "
        f"cluster PSF {topo_txt}: {note}",
        flush=True,
    )


def print_geometry_checkpoint_diff(
    before: np.ndarray,
    after: np.ndarray,
    *,
    step_label: str,
    mlpot_ctx: Any | None = None,
    topology_psf: PathLike | None = None,
) -> None:
    """Print coordinate and bonded-geometry deltas (informational only)."""
    pos0 = np.asarray(before, dtype=np.float64)
    pos1 = np.asarray(after, dtype=np.float64)
    if pos0.shape != pos1.shape:
        print(
            f"{step_label} geometry diff: shape mismatch "
            f"{pos0.shape} -> {pos1.shape}",
            flush=True,
        )
        return

    delta = pos1 - pos0
    per_atom = np.linalg.norm(delta, axis=1)
    rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    max_disp = float(np.max(per_atom)) if per_atom.size else 0.0
    mean_disp = float(np.mean(per_atom)) if per_atom.size else 0.0
    print(
        f"{step_label} geometry diff: RMSD={rmsd:.4f} Å, "
        f"mean|Δ|={mean_disp:.4f} Å, max|Δ|={max_disp:.4f} Å",
        flush=True,
    )

    psf_path = (
        Path(topology_psf).expanduser().resolve()
        if topology_psf is not None
        else _resolve_checkpoint_psf(mlpot_ctx)
    )
    if psf_path is not None and psf_path.is_file():
        try:
            before_bonds = _bond_stretch_summary(pos0, psf_path)
            after_bonds = _bond_stretch_summary(pos1, psf_path)
            print(
                f"{step_label} bonded (PSF {psf_path.name}): "
                f"stretched {before_bonds['n_stretched']} -> {after_bonds['n_stretched']} "
                f"/ {after_bonds['n_bonds']} bonds; "
                f"max stretch {before_bonds['max_stretch_A']:.3f} -> "
                f"{after_bonds['max_stretch_A']:.3f} Å; "
                f"mean |ΔL| {before_bonds['mean_stretch_A']:.3f} -> "
                f"{after_bonds['mean_stretch_A']:.3f} Å",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{step_label} bonded: PSF stretch analysis failed ({exc})",
                flush=True,
            )
    else:
        print(
            f"{step_label} bonded: no cluster PSF for stretch analysis "
            "(load prep_ladder/*.psf with *.pdb in VMD)",
            flush=True,
        )

    print_topology_composition_note(mlpot_ctx, context=f"{step_label}")
