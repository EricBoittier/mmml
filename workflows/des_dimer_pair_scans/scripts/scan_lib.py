"""Pair matrix and config helpers for des_dimer_pair_scans workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from mmml.interfaces.pycharmmInterface.cgenff_residues import parse_cgenff_residues
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition


@dataclass(frozen=True)
class Species:
    tag: str
    residue: str
    label: str


@dataclass(frozen=True)
class PairSpec:
    tag: str
    species_a: Species
    species_b: Species
    composition: str
    homo: bool

    @property
    def label(self) -> str:
        if self.homo:
            return f"{self.species_a.label} – {self.species_b.label}"
        return f"{self.species_a.label} + {self.species_b.label}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else workflow_root() / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {cfg_path}")
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def iter_species(cfg: dict[str, Any]) -> list[Species]:
    out: list[Species] = []
    seen: set[str] = set()
    for row in cfg.get("species") or []:
        tag = str(row["tag"]).strip()
        residue = str(row["residue"]).strip().upper()
        label = str(row.get("label") or tag).strip()
        if tag in seen:
            raise ValueError(f"duplicate species tag: {tag}")
        seen.add(tag)
        out.append(Species(tag=tag, residue=residue, label=label))
    if not out:
        raise ValueError("config.species is empty")
    return out


def composition_for_pair(a: Species, b: Species) -> str:
    if a.residue == b.residue:
        return f"{a.residue}:2"
    # Stable order: lexicographic by residue then tag.
    first, second = (a, b) if (a.residue, a.tag) <= (b.residue, b.tag) else (b, a)
    return f"{first.residue}:1,{second.residue}:1"


def pair_tag(a: Species, b: Species) -> str:
    t1, t2 = sorted([a.tag, b.tag])
    return f"{t1}__{t2}"


def iter_pairs(cfg: dict[str, Any]) -> Iterator[PairSpec]:
    species = iter_species(cfg)
    for i, a in enumerate(species):
        for b in species[i:]:
            homo = a.tag == b.tag
            yield PairSpec(
                tag=pair_tag(a, b),
                species_a=a,
                species_b=b,
                composition=composition_for_pair(a, b),
                homo=homo,
            )


def pair_from_tag(cfg: dict[str, Any], tag: str) -> PairSpec:
    for pair in iter_pairs(cfg):
        if pair.tag == tag:
            return pair
    raise KeyError(f"unknown pair tag: {tag}")


def output_dir(cfg: dict[str, Any], pair: PairSpec) -> Path:
    root = cfg.get("output_root", "artifacts/des_dimer_pair_scans")
    return repo_root() / root / pair.tag


def scan_grids(cfg: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np

    scan = cfg.get("scan") or {}
    d_min = float(scan.get("d_min", 3.0))
    d_max = float(scan.get("d_max", 10.0))
    steps = int(scan.get("steps", 12))
    grid = np.linspace(d_min, d_max, steps, dtype=np.float64)
    return grid, grid.copy()


def validate_species_residues(cfg: dict[str, Any]) -> list[str]:
    """Return CGENFF RESI names missing from the bundled RTF."""
    known = {r.name.upper() for r in parse_cgenff_residues()}
    missing = []
    for sp in iter_species(cfg):
        if sp.residue.upper() not in known:
            missing.append(sp.residue)
    return missing


def validate_compositions(cfg: dict[str, Any]) -> None:
    for pair in iter_pairs(cfg):
        parsed = parse_composition(pair.composition)
        n_monomers = sum(n for _, n in parsed)
        if n_monomers != 2:
            raise ValueError(f"{pair.tag}: expected 2 monomers, got {pair.composition!r}")
