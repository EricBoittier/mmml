"""Compare ASE/PyXtal crystal builds against bundled literature CIFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


# Reference PyXtal builds (seed 42 / 7, ρ scaled to literature). Used when PyXtal
# is not installed (e.g. docs CI) so comparison tables stay complete.
_FROZEN_DCM_PYXTAL = CrystalMetrics(
    label="pyxtal",
    natoms=20,
    lengths_a=(7.773, 6.651, 5.521),
    angles_deg=(90.0, 90.0, 90.0),
    volume_a3=285.462,
    density_g_cm3=1.976,
    space_group=60,
)
_FROZEN_BENZENE_PYXTAL = CrystalMetrics(
    label="pyxtal",
    natoms=24,
    lengths_a=(5.175, 11.337, 3.813),
    angles_deg=(90.0, 74.73, 90.0),
    volume_a3=215.800,
    density_g_cm3=1.202,
    space_group=14,
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
    charmm: CrystalMetrics | None = None,
    charmm_caption: str = "",
) -> str:
    """Markdown table: literature vs optional make-res+CIF and PyXtal builds."""
    if charmm is not None:
        header = (
            "| Quantity | Literature | make-res+CIF | Δ (CIF−lit) | "
            "PyXtal build | Δ (PyXtal−lit) |"
        )
        sep = "|----------|------------|--------------|-------------|--------------|----------------|"
    else:
        header = "| Quantity | Literature | PyXtal build | Δ (build − lit) |"
        sep = "|----------|------------|--------------|-----------------|"

    lines = [literature_citation, "", header, sep]
    if built is None and charmm is None:
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

    lit = literature
    bld = built
    ch = charmm

    def _fmt3(val: float | None) -> str:
        return f"{val:.3f}" if val is not None else "—"

    def _fmt1(val: float | None) -> str:
        return f"{val:.1f}" if val is not None else "—"

    def _delta(built_val: float | None, ref: float) -> str:
        return _pct_delta(built_val, ref) if built_val is not None else "—"

    sg_lit = str(lit.space_group or "—")
    sg_ch = str(ch.space_group or "—") if ch else "—"
    sg_bld = str(bld.space_group or "—") if bld else "—"
    if charmm is not None:
        lines.append(f"| Space group | {sg_lit} | {sg_ch} | — | {sg_bld} | — |")
        lines.append(
            f"| N atoms | {lit.natoms} | {ch.natoms} | "
            f"{_pct_delta(float(ch.natoms), float(lit.natoms))} | "
            f"{bld.natoms if bld else '—'} | "
            f"{_delta(float(bld.natoms) if bld else None, float(lit.natoms))} |"
        )
    else:
        lines.append(f"| Space group | {sg_lit} | {sg_bld} | — |")
        lines.append(
            f"| N atoms | {lit.natoms} | {bld.natoms if bld else '—'} | "
            f"{_delta(float(bld.natoms) if bld else None, float(lit.natoms))} |"
        )

    ch_lengths = ch.lengths_a if ch else (None, None, None)
    bld_lengths = bld.lengths_a if bld else (None, None, None)
    for axis, lv, cv, bv in zip(("a", "b", "c"), lit.lengths_a, ch_lengths, bld_lengths):
        if charmm is not None:
            lines.append(
                f"| *{axis}* (Å) | {lv:.3f} | {_fmt3(cv)} | {_delta(cv, lv)} | "
                f"{_fmt3(bv)} | {_delta(bv, lv)} |"
            )
        elif bld is not None:
            lines.append(
                f"| *{axis}* (Å) | {lv:.3f} | {bv:.3f} | {_pct_delta(bv, lv)} |"
            )

    ch_angles = ch.angles_deg if ch else (None, None, None)
    bld_angles = bld.angles_deg if bld else (None, None, None)
    for name, lv, cv, bv in zip(("α", "β", "γ"), lit.angles_deg, ch_angles, bld_angles):
        if charmm is not None:
            lines.append(
                f"| {name} (°) | {lv:.1f} | {_fmt1(cv)} | {_delta(cv, lv)} | "
                f"{_fmt1(bv)} | {_delta(bv, lv)} |"
            )
        elif bld is not None:
            lines.append(
                f"| {name} (°) | {lv:.1f} | {bv:.1f} | {_pct_delta(bv, lv)} |"
            )

    if charmm is not None:
        lines.append(
            f"| Volume (Å³) | {lit.volume_a3:.1f} | {ch.volume_a3:.1f} | "
            f"{_pct_delta(ch.volume_a3, lit.volume_a3)} | "
            f"{_fmt1(bld.volume_a3 if bld else None)} | "
            f"{_delta(bld.volume_a3 if bld else None, lit.volume_a3)} |"
        )
        lines.append(
            f"| ρ (g/cm³) | {lit.density_g_cm3:.3f} | {ch.density_g_cm3:.3f} | "
            f"{_pct_delta(ch.density_g_cm3, lit.density_g_cm3)} | "
            f"{_fmt3(bld.density_g_cm3 if bld else None)} | "
            f"{_delta(bld.density_g_cm3 if bld else None, lit.density_g_cm3)} |"
        )
    elif bld is not None:
        lines.append(
            f"| Volume (Å³) | {lit.volume_a3:.1f} | {bld.volume_a3:.1f} | "
            f"{_pct_delta(bld.volume_a3, lit.volume_a3)} |"
        )
        lines.append(
            f"| ρ (g/cm³) | {lit.density_g_cm3:.3f} | {bld.density_g_cm3:.3f} | "
            f"{_pct_delta(bld.density_g_cm3, lit.density_g_cm3)} |"
        )

    captions = [c for c in (charmm_caption, built_caption) if c]
    if captions:
        lines.extend(["", *(f"_{c}_" for c in captions), ""])
    return "\n".join(lines)


def _pyxtal_built_dcm(seed: int = 42) -> CrystalMetrics:
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


def _pyxtal_built_benzene(seed: int = 7) -> CrystalMetrics:
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


def literature_comparison_markdown(*, use_live_pyxtal: bool = False) -> str:
    """Full markdown section comparing bundled COD structures to CHARMM + PyXtal."""
    from mmml.interfaces.crystal_charmm import charmm_crystal_metrics_from_preset

    dcm_lit = metrics_from_cif(
        default_dcm_crystal_cif(), space_group=60, label="COD 2100015"
    )
    benz_lit = metrics_from_cif(
        default_benzene_crystal_cif(), space_group=14, label="COD 4501704"
    )
    dcm_charmm = charmm_crystal_metrics_from_preset("dcm")
    benz_charmm = charmm_crystal_metrics_from_preset("benz")
    if use_live_pyxtal and have_pyxtal():
        dcm_built = _pyxtal_built_dcm()
        benz_built = _pyxtal_built_benzene()
    else:
        dcm_built = _FROZEN_DCM_PYXTAL
        benz_built = _FROZEN_BENZENE_PYXTAL

    parts = [
        "### Literature cross-check (auto-generated)",
        "",
        "Bundled experimental CIFs vs **make-res+CIF** (exact literature unit cell, "
        "CHARMM atom names) and a single PyXtal `from_random` trial (fixed seeds) "
        "with ρ scaled to literature. PyXtal unit-cell axes can differ in "
        "setting/orientation even when space group and density agree.",
        "",
        "Regenerate: `uv run python scripts/generate_crystal_lit_compare.py`",
        "",
        "#### DCM (CH₂Cl₂) — [COD 2100015](https://www.crystallography.net/2100015.html)",
        "",
        comparison_table_markdown(
            dcm_lit,
            dcm_built,
            charmm=dcm_charmm,
            literature_citation=(
                "Podsiadło *et al.*, *Acta Cryst.* B **2005**, 61, 595 "
                "([CCDC doi:10.5517/cc9lyjb](https://www.ccdc.cam.ac.uk/structures/search?id=doi:10.5517/cc9lyjb&sid=DataCite)); "
                "Pbcn, Z=4, 1.63 GPa / 293 K."
            ),
            charmm_caption=(
                "make-res+CIF: `mmml build-crystal --literature dcm` (unit cell)."
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
            charmm=benz_charmm,
            literature_citation=(
                "Katrusiak *et al.*, *Cryst. Growth Des.* **2010**, 10, 3461 "
                "([doi:10.1021/cg1002594](https://doi.org/10.1021/cg1002594)); "
                "P2₁/c, Z=2, ~0.97 GPa / 295 K."
            ),
            charmm_caption=(
                "make-res+CIF: `mmml build-crystal --literature benz` (unit cell)."
            ),
            built_caption=(
                "PyXtal: `-m benzene` (not `c1ccccc1`), `--spg 14 --z 2 --seed 7`, "
                "ρ scaled to literature."
            ),
        ),
    ]
    return "\n".join(parts)
