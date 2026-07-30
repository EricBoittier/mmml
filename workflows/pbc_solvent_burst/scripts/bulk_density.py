"""Bulk liquid density helpers for PBC solvent burst matrix sizing."""

from __future__ import annotations

from typing import Any, Mapping

# Experimental bulk liquid ~298 K (g/cm³, g/mol). Used only for matrix N sizing.
BULK_SOLVENTS: dict[str, dict[str, float]] = {
    "DCM": {"rho_g_cm3": 1.326, "mw_g_mol": 84.93},
    "ACO": {"rho_g_cm3": 0.784, "mw_g_mol": 58.08},
    "TIP3": {"rho_g_cm3": 0.9970, "mw_g_mol": 18.01528},
    "MEOH": {"rho_g_cm3": 0.7866, "mw_g_mol": 32.04186},
    # Liquid CH4 near the NBP (~111.7 K); used for liquid-density matrix sizing.
    "METH": {"rho_g_cm3": 0.4226, "mw_g_mol": 16.0425},
}

AVOGADRO = 6.02214076e23


def ml_atoms_for_cell(solvent: str, n_monomers: int) -> int:
    """ML atom count for a burst matrix cell (CGenFF all-atom monomer)."""
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms

    return estimate_ml_atoms(int(n_monomers), solvent=solvent)


def volume_per_molecule_ang3(*, mw_g_mol: float, rho_g_cm3: float) -> float:
    """Molecular volume (Å³) from bulk density and molecular weight."""
    molar_vol_cm3 = float(mw_g_mol) / float(rho_g_cm3)
    return molar_vol_cm3 / AVOGADRO * 1e24


def n_monomers_at_bulk_density(
    solvent: str,
    box_side_A: float,
    fraction: float,
    *,
    min_n: int = 1,
    max_n: int | None = None,
) -> int:
    """Monomer count for ``fraction`` of bulk liquid density in a cubic box."""
    key = str(solvent).strip().upper()
    if key not in BULK_SOLVENTS:
        raise ValueError(
            f"Unknown solvent {solvent!r} for bulk-density sizing; "
            f"supported: {sorted(BULK_SOLVENTS)}"
        )
    props = BULK_SOLVENTS[key]
    vol = float(box_side_A) ** 3
    v_mol = volume_per_molecule_ang3(
        mw_g_mol=props["mw_g_mol"],
        rho_g_cm3=props["rho_g_cm3"],
    )
    n_bulk = vol / v_mol
    n = int(round(float(fraction) * n_bulk))
    n = max(int(min_n), n)
    if max_n is not None:
        n = min(n, int(max_n))
    return n


def normalize_mole_fractions(raw: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Canonical positive mole fractions for an ideal liquid mixture."""
    values = {str(key).strip().upper(): float(value) for key, value in raw.items()}
    if len(values) < 2:
        raise ValueError("A mixture requires at least two solvent components")
    unknown = sorted(set(values) - set(BULK_SOLVENTS))
    if unknown:
        raise ValueError(
            f"Unknown mixture solvents {unknown}; supported: {sorted(BULK_SOLVENTS)}"
        )
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("Mixture mole fractions must all be positive")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Mixture mole fractions must have a positive sum")
    return tuple((key, value / total) for key, value in sorted(values.items()))


def mixture_counts_at_bulk_density(
    mole_fractions: Mapping[str, float] | tuple[tuple[str, float], ...],
    box_side_A: float,
    fraction: float,
    *,
    min_n: int = 1,
    max_n: int | None = None,
) -> dict[str, int]:
    """Integer component counts using ideal volume mixing at 298 K."""
    normalized = normalize_mole_fractions(dict(mole_fractions))
    mean_volume = sum(
        x
        * volume_per_molecule_ang3(
            mw_g_mol=BULK_SOLVENTS[solvent]["mw_g_mol"],
            rho_g_cm3=BULK_SOLVENTS[solvent]["rho_g_cm3"],
        )
        for solvent, x in normalized
    )
    n_total = int(round(float(fraction) * float(box_side_A) ** 3 / mean_volume))
    n_total = max(int(min_n), n_total)
    if max_n is not None:
        n_total = min(int(max_n), n_total)

    # Largest-remainder allocation preserves the requested total and avoids
    # silently dropping a minority component at the practical matrix sizes.
    exact = [(solvent, x * n_total) for solvent, x in normalized]
    counts = {solvent: int(value) for solvent, value in exact}
    remaining = n_total - sum(counts.values())
    for solvent, _value in sorted(exact, key=lambda item: item[1] % 1.0, reverse=True):
        if remaining <= 0:
            break
        counts[solvent] += 1
        remaining -= 1
    if n_total >= len(counts) and any(count == 0 for count in counts.values()):
        raise ValueError("Mixture allocation dropped a component; increase bulk_density_n_min")
    return counts


def mixture_total_at_bulk_density(
    mole_fractions: Mapping[str, float] | tuple[tuple[str, float], ...],
    box_side_A: float,
    fraction: float,
    **kwargs: Any,
) -> int:
    return sum(
        mixture_counts_at_bulk_density(
            mole_fractions, box_side_A, fraction, **kwargs
        ).values()
    )


def effective_mass_density_g_cm3(*, solvent: str, n_monomers: int, box_side_A: float) -> float:
    key = str(solvent).strip().upper()
    mw = BULK_SOLVENTS[key]["mw_g_mol"]
    vol_cm3 = float(box_side_A) ** 3 * 1e-24
    mass_g = int(n_monomers) * mw / AVOGADRO
    return mass_g / vol_cm3


def matrix_uses_bulk_density(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("bulk_density_fractions"))


def matrix_density_fractions(cfg: dict[str, Any]) -> list[float]:
    raw = cfg.get("bulk_density_fractions")
    if not raw:
        return []
    return [float(x) for x in raw]


def matrix_cluster_sizes_for_cell(
    cfg: dict[str, Any],
    *,
    solvent: str,
    box_size: float,
) -> list[int]:
    """Return monomer counts for one solvent/box (explicit or bulk-derived)."""
    if matrix_uses_bulk_density(cfg):
        min_n = int(cfg.get("bulk_density_n_min", 1))
        max_raw = cfg.get("bulk_density_n_max")
        max_n = int(max_raw) if max_raw is not None else None
        seen: set[int] = set()
        sizes: list[int] = []
        for frac in matrix_density_fractions(cfg):
            n = n_monomers_at_bulk_density(
                solvent,
                box_size,
                frac,
                min_n=min_n,
                max_n=max_n,
            )
            if n in seen:
                continue
            seen.add(n)
            sizes.append(n)
        return sorted(sizes)
    return [int(n) for n in cfg.get("cluster_sizes", [])]


def bulk_reference_table(box_sizes: list[float]) -> str:
    """Human-readable N_bulk per solvent and box (for preflight / docs)."""
    solvents = tuple(BULK_SOLVENTS)
    header = f"{'L (Å)':>6}  {'V (Å³)':>10}"
    header += "".join(f"  {f'{sol} N_bulk':>12}" for sol in solvents)
    lines = [header]
    for L in box_sizes:
        vol = float(L) ** 3
        counts = [n_monomers_at_bulk_density(sol, L, 1.0) for sol in solvents]
        lines.append(
            f"{L:6.0f}  {vol:10.0f}"
            + "".join(f"  {count:12d}" for count in counts)
        )
    return "\n".join(lines)
