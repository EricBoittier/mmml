"""Compare ASE/PyXtal crystal builds against bundled literature CIFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pyxtal_placement import (
    MolecularCrystalBuildRequest,
    build_molecular_crystal_random,
    crystal_mass_density_g_cm3,
    have_pyxtal,
    scale_atoms_cell_to_density,
)
from mmml.paths import (
    default_benzene_crystal_cif,
    default_dcm_crystal_cif,
    default_dcm_molecule_xyz,
)


@dataclass(frozen=True)
class CrystalMetrics:
    """Unit-cell descriptors for side-by-side literature comparison."""

    label: str
    natoms: int
    lengths_a: tuple[float, float, float]
    angles_deg: tuple[float, float, float]
    volume_a3: float
    density_g_cm3: float
    space_group: int | None = None

    def as_dict(self) -> dict[str, Any]:
        a, b, c = self.lengths_a
        alpha, beta, gamma = self.angles_deg
        return {
            "label": self.label,
            "natoms": self.natoms,
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "volume_a3": self.volume_a3,
            "density_g_cm3": self.density_g_cm3,
            "space_group": self.space_group,
        }


def metrics_from_atoms(
    atoms: Any,
    *,
    label: str = "",
    space_group: int | None = None,
) -> CrystalMetrics:
    par = tuple(float(x) for x in atoms.cell.cellpar())
    return CrystalMetrics(
        label=label,
        natoms=int(len(atoms)),
        lengths_a=(par[0], par[1], par[2]),
        angles_deg=(par[3], par[4], par[5]),
        volume_a3=float(atoms.get_volume()),
        density_g_cm3=float(crystal_mass_density_g_cm3(atoms)),
        space_group=space_group,
    )


def metrics_from_cif(
    path: Path | str,
    *,
    label: str | None = None,
    space_group: int | None = None,
) -> CrystalMetrics:
    from ase.io import read

    p = Path(path)
    atoms = read(str(p))
    return metrics_from_atoms(
        atoms,
        label=label or p.name,
        space_group=space_group,
    )


def _pct_delta(built: float, reference: float) -> str:
    if reference == 0.0:
        return "—"
    return f"{100.0 * (built - reference) / reference:+.1f}%"


def comparison_table_markdown(
    literature: CrystalMetrics,
    built: CrystalMetrics | None,
    *,
    literature_citation: str,
    built_caption: str,
) -> str:
    """Markdown table: literature vs optional PyXtal build."""
    lines = [
        literature_citation,
        "",
        "| Quantity | Literature | PyXtal build | Δ (build − lit) |",
        "|----------|------------|--------------|-----------------|",
    ]
    if built is None:
        lines.extend(
            [
                f"| Space group | {literature.space_group or '—'} | — | — |",
                f"| N atoms | {literature.natoms} | — | — |",
                f"| *a* (Å) | {literature.lengths_a[0]:.3f} | — | — |",
                f"| *b* (Å) | {literature.lengths_a[1]:.3f} | — | — |",
                f"| *c* (Å) | {literature.lengths_a[2]:.3f} | — | — |",
                f"| α (°) | {literature.angles_deg[0]:.1f} | — | — |",
                f"| β (°) | {literature.angles_deg[1]:.1f} | — | — |",
                f"| γ (°) | {literature.angles_deg[2]:.1f} | — | — |",
                f"| Volume (Å³) | {literature.volume_a3:.1f} | — | — |",
                f"| ρ (g/cm³) | {literature.density_g_cm3:.3f} | — | — |",
                "",
                f"_{built_caption}_",
                "",
            ]
        )
        return "\n".join(lines)

    lit, bld = literature, built
    rows: list[tuple[str, str, str, str]] = [
        (
            "Space group",
            str(lit.space_group or "—"),
            str(bld.space_group or "—"),
            "—",
        ),
        (
            "N atoms",
            str(lit.natoms),
            str(bld.natoms),
            _pct_delta(float(bld.natoms), float(lit.natoms)),
        ),
    ]
    for axis, lv, bv in zip(("a", "b", "c"), lit.lengths_a, bld.lengths_a):
        rows.append((f"*{axis}* (Å)", f"{lv:.3f}", f"{bv:.3f}", _pct_delta(bv, lv)))
    for name, lv, bv in zip(
        ("α", "β", "γ"), lit.angles_deg, bld.angles_deg
    ):
        rows.append((f"{name} (°)", f"{lv:.1f}", f"{bv:.1f}", _pct_delta(bv, lv)))
    rows.append(
        (
            "Volume (Å³)",
            f"{lit.volume_a3:.1f}",
            f"{bld.volume_a3:.1f}",
            _pct_delta(bld.volume_a3, lit.volume_a3),
        )
    )
    rows.append(
        (
            "ρ (g/cm³)",
            f"{lit.density_g_cm3:.3f}",
            f"{bld.density_g_cm3:.3f}",
            _pct_delta(bld.density_g_cm3, lit.density_g_cm3),
        )
    )
    for qty, lval, bval, delta in rows:
        lines.append(f"| {qty} | {lval} | {bval} | {delta} |")
    lines.extend(["", f"_{built_caption}_", ""])
    return "\n".join(lines)


def _pyxtal_built_dcm(seed: int = 42) -> CrystalMetrics | None:
    if not have_pyxtal():
        return None
    lit = metrics_from_cif(
        default_dcm_crystal_cif(), space_group=60, label="literature"
    )
    result = build_molecular_crystal_random(
        MolecularCrystalBuildRequest(
            molecules=[str(default_dcm_molecule_xyz())],
            stoichiometry=[4],
            space_group=60,
            seed=seed,
            max_attempts=40,
        )
    )
    atoms = result.atoms.copy()
    scale_atoms_cell_to_density(atoms, lit.density_g_cm3)
    return metrics_from_atoms(atoms, label="pyxtal", space_group=60)


def _pyxtal_built_benzene(seed: int = 7) -> CrystalMetrics | None:
    if not have_pyxtal():
        return None
    lit = metrics_from_cif(
        default_benzene_crystal_cif(), space_group=14, label="literature"
    )
    result = build_molecular_crystal_random(
        MolecularCrystalBuildRequest(
            molecules=["benzene"],
            stoichiometry=[2],
            space_group=14,
            seed=seed,
            max_attempts=40,
        )
    )
    atoms = result.atoms.copy()
    scale_atoms_cell_to_density(atoms, lit.density_g_cm3)
    return metrics_from_atoms(atoms, label="pyxtal", space_group=14)


def literature_comparison_markdown(*, include_pyxtal: bool = True) -> str:
    """Full markdown section comparing bundled COD structures to PyXtal builds."""
    dcm_lit = metrics_from_cif(
        default_dcm_crystal_cif(), space_group=60, label="COD 2100015"
    )
    benz_lit = metrics_from_cif(
        default_benzene_crystal_cif(), space_group=14, label="COD 4501704"
    )
    dcm_built = _pyxtal_built_dcm() if include_pyxtal else None
    benz_built = _pyxtal_built_benzene() if include_pyxtal else None

    unavailable = (
        "_PyXtal column omitted — install `uv sync --extra chem` and regenerate._"
        if dcm_built is None
        else ""
    )

    parts = [
        "### Literature cross-check (auto-generated)",
        "",
        "Side-by-side metrics for bundled experimental CIFs vs a **single** PyXtal "
        "`from_random` trial (fixed seeds) with `--target-density-g-cm3` matched to "
        "the literature ρ. Unit-cell axes can differ in setting/orientation even when "
        "space group and density agree.",
        "",
        "Regenerate: `uv run python scripts/generate_crystal_lit_compare.py`",
        "",
        "#### DCM (CH₂Cl₂) — [COD 2100015](https://www.crystallography.net/2100015.html)",
        "",
        comparison_table_markdown(
            dcm_lit,
            dcm_built,
            literature_citation=(
                "Podsiadło *et al.*, *Acta Cryst.* B **2005**, 61, 595 "
                "([CCDC doi:10.5517/cc9lyjb](https://www.ccdc.cam.ac.uk/structures/search?id=doi:10.5517/cc9lyjb&sid=DataCite)); "
                "Pbcn, Z=4, 1.63 GPa / 293 K."
            ),
            built_caption=(
                "PyXtal: `-m default_dcm_molecule_xyz()`, `--spg 60 --z 4 --seed 42`, "
                "ρ scaled to literature."
            ),
        ),
        "#### Benzene (C₆H₆) — [COD 4501704](https://www.crystallography.net/cod/4501704.html)",
        "",
        comparison_table_markdown(
            benz_lit,
            benz_built,
            literature_citation=(
                "Katrusiak *et al.*, *Cryst. Growth Des.* **2010**, 10, 3461 "
                "([doi:10.1021/cg1002594](https://doi.org/10.1021/cg1002594)); "
                "P2₁/c, Z=2, ~0.97 GPa / 295 K."
            ),
            built_caption=(
                "PyXtal: `-m benzene` (not `c1ccccc1`), `--spg 14 --z 2 --seed 7`, "
                "ρ scaled to literature."
            ),
        ),
    ]
    if unavailable:
        parts.extend([unavailable, ""])
    return "\n".join(parts)
