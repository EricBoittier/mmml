"""Bulk liquid density helpers for PBC solvent burst matrix sizing."""

from __future__ import annotations

from typing import Any

# Experimental bulk liquid ~298 K (g/cm³, g/mol). Used only for matrix N sizing.
BULK_SOLVENTS: dict[str, dict[str, float]] = {
    "DCM": {"rho_g_cm3": 1.326, "mw_g_mol": 84.93},
    "ACO": {"rho_g_cm3": 0.784, "mw_g_mol": 58.08},
}

AVOGADRO = 6.02214076e23


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
    lines = [
        f"{'L (Å)':>6}  {'V (Å³)':>10}  {'DCM N_bulk':>12}  {'ACO N_bulk':>12}",
    ]
    for L in box_sizes:
        vol = float(L) ** 3
        nd = n_monomers_at_bulk_density("DCM", L, 1.0)
        na = n_monomers_at_bulk_density("ACO", L, 1.0)
        lines.append(f"{L:6.0f}  {vol:10.0f}  {nd:12d}  {na:12d}")
    return "\n".join(lines)
