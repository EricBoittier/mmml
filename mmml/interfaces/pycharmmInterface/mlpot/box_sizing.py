"""Initial PBC box sizing and optional mini-stage box equilibration helpers."""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_length_from_geometry

AVOGADRO = 6.02214076e23

# Experimental bulk liquid densities (~298 K) for auto-sizing.
SOLVENT_BULK_PROPS: dict[str, dict[str, float]] = {
    "DCM": {"rho_g_cm3": 1.326, "mw_g_mol": 84.93},
    "ACO": {"rho_g_cm3": 0.784, "mw_g_mol": 58.08},
    "MEOH": {"rho_g_cm3": 0.792, "mw_g_mol": 32.04},
    "ETOH": {"rho_g_cm3": 0.789, "mw_g_mol": 46.07},
    "TIP3": {"rho_g_cm3": 1.000, "mw_g_mol": 18.015},
    "WAT": {"rho_g_cm3": 1.000, "mw_g_mol": 18.015},
}

BoxAutoMode = Literal["geometry", "density"]


def parse_composition_dict(spec: str | None) -> dict[str, int] | None:
    """Parse ``RES:N,RES:N`` into a residue count map."""
    if spec is None or not str(spec).strip():
        return None
    out: dict[str, int] = {}
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Invalid composition token: {tok!r}")
        residue, count_s = tok.split(":", 1)
        residue = residue.strip().upper()
        count = int(count_s.strip())
        if not residue or count <= 0:
            raise ValueError(f"Invalid composition token: {tok!r}")
        out[residue] = out.get(residue, 0) + int(count)
    return out or None


def resolve_box_auto_mode(args: argparse.Namespace) -> BoxAutoMode:
    """Return ``geometry`` (default) or ``density`` when ``--box-size`` is unset."""
    raw = getattr(args, "box_auto", None)
    if raw is None:
        if getattr(args, "target_density_g_cm3", None) is not None:
            return "density"
        if getattr(args, "bulk_density_fraction", None) is not None:
            return "density"
        return "geometry"
    mode = str(raw).strip().lower()
    if mode in ("geometry", "density"):
        return mode  # type: ignore[return-value]
    raise ValueError(f"--box-auto must be 'geometry' or 'density', got {raw!r}")


def total_mass_g_for_composition(composition: dict[str, int]) -> float:
    """Total cluster mass (g) from per-residue monomer counts."""
    mass_g = 0.0
    unknown: list[str] = []
    for residue, count in composition.items():
        key = str(residue).strip().upper()
        props = SOLVENT_BULK_PROPS.get(key)
        if props is None:
            unknown.append(key)
            continue
        mass_g += int(count) * float(props["mw_g_mol"]) / AVOGADRO
    if unknown:
        raise ValueError(
            f"Cannot size box from density: unknown residue(s) {unknown}. "
            f"Known: {sorted(SOLVENT_BULK_PROPS)}. Use --box-size or --box-auto geometry."
        )
    if mass_g <= 0.0:
        raise ValueError("composition mass is zero; cannot compute density-based box")
    return mass_g


def cubic_box_side_from_target_density(
    *,
    n_molecules: int,
    total_mass_g: float,
    target_density_g_cm3: float,
    min_side_A: float | None = None,
) -> float:
    """Cubic box side (Å) for ``n`` molecules at target mass density."""
    if int(n_molecules) <= 0:
        raise ValueError(f"n_molecules must be positive, got {n_molecules}")
    rho = float(target_density_g_cm3)
    if rho <= 0.0:
        raise ValueError(f"target density must be positive, got {rho}")
    vol_cm3 = float(total_mass_g) / rho
    vol_A3 = vol_cm3 * 1e24
    side = float(vol_A3 ** (1.0 / 3.0))
    if min_side_A is not None:
        side = max(side, float(min_side_A))
    return side


def resolve_target_density_g_cm3(
    args: argparse.Namespace,
    composition: dict[str, int] | None,
) -> float:
    """Target bulk density (g/cm³) from CLI or bulk-fraction scaling."""
    explicit = getattr(args, "target_density_g_cm3", None)
    fraction = getattr(args, "bulk_density_fraction", None)
    if explicit is not None and fraction is not None:
        raise ValueError(
            "Use only one of --target-density-g-cm3 or --bulk-density-fraction."
        )
    if explicit is not None:
        rho = float(explicit)
        if rho <= 0.0:
            raise ValueError(f"--target-density-g-cm3 must be positive, got {rho}")
        return rho
    if fraction is not None:
        if composition is None or len(composition) != 1:
            raise ValueError(
                "--bulk-density-fraction requires a single-species --composition "
                "(e.g. DCM:60)."
            )
        residue = next(iter(composition))
        key = str(residue).strip().upper()
        props = SOLVENT_BULK_PROPS.get(key)
        if props is None:
            raise ValueError(
                f"No bulk density table entry for {key!r}; "
                f"use --target-density-g-cm3 instead."
            )
        frac = float(fraction)
        if frac <= 0.0:
            raise ValueError(f"--bulk-density-fraction must be positive, got {frac}")
        return float(props["rho_g_cm3"]) * frac
    raise ValueError(
        "--box-auto density requires --target-density-g-cm3 or --bulk-density-fraction."
    )


def resolve_density_box_side(
    args: argparse.Namespace,
    positions: np.ndarray,
    *,
    composition: dict[str, int] | None = None,
    n_molecules: int | None = None,
    ml_cutoff: float = 12.0,
) -> float:
    """Cubic side (Å) from composition + target density, with geometry floor."""
    comp = composition
    if comp is None:
        comp = parse_composition_dict(getattr(args, "composition", None))
    if comp is None:
        raise ValueError(
            "--box-auto density requires --composition (e.g. DCM:60)."
        )
    n_mol = int(n_molecules) if n_molecules is not None else int(sum(comp.values()))
    rho = resolve_target_density_g_cm3(args, comp)
    mass_g = total_mass_g_for_composition(comp)
    geom_floor = cubic_box_length_from_geometry(
        positions,
        ml_cutoff=float(ml_cutoff),
    )
    return cubic_box_side_from_target_density(
        n_molecules=n_mol,
        total_mass_g=mass_g,
        target_density_g_cm3=rho,
        min_side_A=geom_floor,
    )


def resolve_density_packmol_cube_side(
    args: argparse.Namespace,
    *,
    composition: dict[str, int] | None = None,
    n_molecules: int | None = None,
) -> float:
    """Cubic side (Å) for Packmol ``inside cube`` from ``--box-auto density``.

    Uses composition + target density only (no post-placement geometry floor).
    """
    if resolve_box_auto_mode(args) != "density":
        raise ValueError(
            "resolve_density_packmol_cube_side requires --box-auto density."
        )
    comp = composition
    if comp is None:
        comp = parse_composition_dict(getattr(args, "composition", None))
    if comp is None:
        raise ValueError(
            "--box-auto density requires --composition (e.g. DCM:60) for Packmol cube sizing."
        )
    n_mol = int(n_molecules) if n_molecules is not None else int(sum(comp.values()))
    n_mol_cli = int(getattr(args, "n_molecules", 0) or 0)
    if n_mol_cli > 0:
        n_mol = n_mol_cli
    rho = resolve_target_density_g_cm3(args, comp)
    mass_g = total_mass_g_for_composition(comp)
    return cubic_box_side_from_target_density(
        n_molecules=n_mol,
        total_mass_g=mass_g,
        target_density_g_cm3=rho,
    )


def resolve_initial_pbc_box_side(
    args: argparse.Namespace,
    positions: np.ndarray,
    *,
    composition: dict[str, int] | None = None,
    n_molecules: int | None = None,
    ml_cutoff: float | None = None,
) -> tuple[float, str]:
    """Resolve cubic box side (Å) and a short source label."""
    if getattr(args, "box_size", None) is not None:
        return float(args.box_size), "explicit"
    mode = resolve_box_auto_mode(args)
    ml_cut = float(ml_cutoff if ml_cutoff is not None else getattr(args, "ml_cutoff", 12.0))
    if mode == "density":
        side = resolve_density_box_side(
            args,
            positions,
            composition=composition,
            n_molecules=n_molecules,
            ml_cutoff=ml_cut,
        )
        return side, "density"
    return cubic_box_length_from_geometry(positions, ml_cutoff=ml_cut), "geometry"


def resolve_suite_auto_box_side(
    args: argparse.Namespace,
    positions: np.ndarray,
    *,
    ml_cutoff: float,
) -> tuple[float, str]:
    """ASE/JAX-MD helper when ``--box-size`` and handoff cell are absent."""
    n_mol = int(getattr(args, "n_molecules", 0) or 0) or None
    comp = parse_composition_dict(getattr(args, "composition", None))
    if comp is not None and n_mol is None:
        n_mol = int(sum(comp.values()))
    side, source = resolve_initial_pbc_box_side(
        args,
        positions,
        composition=comp,
        n_molecules=n_mol,
        ml_cutoff=ml_cutoff,
    )
    return side, f"auto_{source}"


def should_run_mini_box_equil(
    args: argparse.Namespace,
    *,
    charmm_pbc: bool,
    pretreat_mm: bool,
    stages: list[str],
) -> bool:
    """True when a short CPT NPT leg should run at the start of mini."""
    ps = float(getattr(args, "mini_box_equil_ps", 0.0) or 0.0)
    if ps <= 0.0 or not charmm_pbc:
        return False
    if "mini" not in stages:
        return False
    if pretreat_mm and float(getattr(args, "charmm_mm_pretreat_ps_equi", 0.0) or 0.0) > 0.0:
        return False
    if getattr(args, "box_size", None) is not None and not bool(
        getattr(args, "mini_box_equil_allow_fixed_box", False)
    ):
        return False
    return True


def add_box_sizing_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for automatic box sizing and mini-stage box equilibration."""
    group = parser.add_argument_group("PBC box sizing")
    group.add_argument(
        "--box-auto",
        choices=("geometry", "density"),
        default=None,
        help=(
            "How to choose the cubic box when --box-size is unset: "
            "geometry=span+padding (default); density=from --composition and target ρ."
        ),
    )
    group.add_argument(
        "--target-density-g-cm3",
        type=float,
        default=None,
        metavar="RHO",
        help="Target mass density (g/cm³) for --box-auto density (requires --composition).",
    )
    group.add_argument(
        "--bulk-density-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Fraction of experimental bulk ρ for a single-species --composition "
            "(e.g. 0.85 for 85%% of liquid DCM density)."
        ),
    )
    group.add_argument(
        "--mc-density-equalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run default post-build MC cubic-volume equalization for PBC composition "
            "builds when a density target can be resolved (default: on)."
        ),
    )
    group.add_argument(
        "--mc-density-target-g-cm3",
        type=float,
        default=None,
        metavar="RHO",
        help=(
            "Target density for MC density equalization. Defaults to "
            "--target-density-g-cm3, --bulk-density-fraction, or known single-solvent bulk density."
        ),
    )
    group.add_argument(
        "--mc-density-steps",
        type=int,
        default=64,
        metavar="N",
        help="MC density equalization proposal count (default: 64).",
    )
    group.add_argument(
        "--mc-density-step-scale",
        type=float,
        default=0.04,
        metavar="LOGSCALE",
        help="Log box-side proposal noise scale for MC density equalization (default: 0.04).",
    )
    group.add_argument(
        "--mc-density-temperature",
        type=float,
        default=0.02,
        metavar="T",
        help="Dimensionless Metropolis temperature for density-error acceptance (default: 0.02).",
    )
    group.add_argument(
        "--mc-density-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for MC density equalization (default: --seed).",
    )
    group.add_argument(
        "--mc-density-min-scale",
        type=float,
        default=0.35,
        metavar="S",
        help="Minimum allowed final box side relative to the initial side (default: 0.35).",
    )
    group.add_argument(
        "--mc-density-max-scale",
        type=float,
        default=1.50,
        metavar="S",
        help="Maximum allowed final box side relative to the initial side (default: 1.50).",
    )
    group.add_argument(
        "--mini-box-equil-ps",
        type=float,
        default=0.0,
        metavar="PS",
        help=(
            "PyCHARMM mini: short CPT NPT equilibration (ps) after coordinate-only "
            "CHARMM MM mini and before MLpot SD. 0=off. Skipped when pretreat NPT equi runs."
        ),
    )
    group.add_argument(
        "--mini-box-equil-allow-fixed-box",
        action="store_true",
        help=(
            "Allow --mini-box-equil-ps CPT NPT even when --box-size is set "
            "(default: fixed --box-size uses Hoover NVT only during pretreat)."
        ),
    )
    group.add_argument(
        "--jaxmd-mini-box-equil-ps",
        type=float,
        default=0.0,
        metavar="PS",
        help=(
            "JAX-MD: short NPT prelude (ps) after PBC FIRE minimize to relax the cell "
            "before the main ensemble. 0=off."
        ),
    )
    group.add_argument(
        "--mini-lattice-abnr-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "PyCHARMM mini: CHARMM MINI ABNR LATTice steps to optimize the cubic unit cell "
            "after coordinate-only CHARMM MM mini and before MLpot SD. 0=off. "
            "Requires CRYSTAL/PBC."
        ),
    )
    group.add_argument(
        "--mini-lattice-abnr-nocoords",
        action="store_true",
        help=(
            "With --mini-lattice-abnr-steps: optimize only the unit cell (NOCOordinates); "
            "default optimizes coordinates and box together."
        ),
    )
    group.add_argument(
        "--mini-lattice-abnr-allow-fixed-box",
        action="store_true",
        help=(
            "Allow --mini-lattice-abnr-steps even when --box-size is set "
            "(default: fixed --box-size skips lattice minimization)."
        ),
    )
    group.add_argument(
        "--liquid-prep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Easy dense-liquid setup: same as --density-prep-mode resilient "
            "(looser Packmol, MC density equalization, stronger CHARMM/lattice mini, "
            "mini box equil, post-mini rescue ladder when GRMS is high). "
            "For full prep + dynamics recovery in one flag, prefer --cleanup."
        ),
    )
    group.add_argument(
        "--density-prep-mode",
        choices=("off", "resilient"),
        default="off",
        help=(
            "Condensed-phase box prep strategy. resilient: start Packmol slightly "
            "below target density, enable MC equalization, stronger CHARMM/lattice "
            "mini, and the post-mini density prep ladder when GRMS is high."
        ),
    )
    group.add_argument(
        "--density-prep-ladder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "After MLpot mini, run a multi-step density/box rescue ladder "
            "(repack, MC density, lattice ABNR, bonded MM, ASE BFGS/FIRE, MLpot SD) "
            "when GRMS exceeds --max-grms-before-dyn. Default on for "
            "--density-prep-mode resilient."
        ),
    )
    group.add_argument(
        "--density-prep-ladder-max-rounds",
        type=int,
        default=3,
        metavar="N",
        help="Maximum density prep ladder rounds (default: 3).",
    )
    group.add_argument(
        "--density-prep-lattice-abnr-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Lattice ABNR steps inside the density prep ladder (0=use "
            "--mini-lattice-abnr-steps or 100)."
        ),
    )
    group.add_argument(
        "--pre-mlpot-overlap-min-distance",
        type=float,
        default=None,
        metavar="ANG",
        help=(
            "Pre-MLpot geometry gate: minimum inter-monomer atom distance in Å "
            "(default: 1.0; independent of --dynamics-overlap-min-distance). "
            "Catches true cross-monomer clashes while allowing tight liquid "
            "contacts that hybrid mini relaxes."
        ),
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import add_cleanup_args
    from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
        add_recovery_artifact_args,
    )

    add_recovery_artifact_args(parser)
    add_cleanup_args(parser)
