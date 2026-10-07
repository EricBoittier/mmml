"""YAML ML/MM cut: which atoms PET sees, and the ghost hydrogen on each cut bond.

A cut file is separate from the system config. The system YAML points at it
with ``ml_cut: path/to/cut.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mmml.interfaces.calculators.link_atoms import LinkAtom

_SELECTOR_KEYS = frozenset({"resname", "segid", "resid", "names", "name"})


@dataclass(frozen=True, slots=True)
class AtomSelector:
    """Match PSF atoms by residue name, segment, resid, and atom name."""

    resname: str | None = None
    segid: str | None = None
    resid: int | None = None
    names: frozenset[str] | None = None
    name: str | None = None

    def matches(self, *, resname: str, segid: str, resid: int, atom_name: str) -> bool:
        if self.resname is not None and resname != self.resname:
            return False
        if self.segid is not None and segid != self.segid:
            return False
        if self.resid is not None and resid != self.resid:
            return False
        if self.name is not None and atom_name != self.name:
            return False
        if self.names is not None and atom_name not in self.names:
            return False
        return True

    def label(self) -> str:
        parts: list[str] = []
        if self.segid is not None:
            parts.append(f"segid={self.segid}")
        if self.resid is not None:
            parts.append(f"resid={self.resid}")
        if self.resname is not None:
            parts.append(f"resname={self.resname}")
        if self.name is not None:
            parts.append(f"name={self.name}")
        if self.names is not None:
            parts.append("names=" + ",".join(sorted(self.names)))
        return " ".join(parts) if parts else "(empty selector)"


@dataclass(frozen=True, slots=True)
class MlCutSpec:
    """Charge, spin, ML atom selectors, and cut bonds from one YAML file."""

    path: Path
    charge: int
    spin_multiplicity: int
    ml_atoms: tuple[AtomSelector, ...]
    links: tuple[tuple[AtomSelector, AtomSelector], ...]


def load_ml_cut(path: Path | str) -> MlCutSpec:
    """Read a cut file. Paths are used as given (run from the repository root)."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"ML cut file not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"ML cut file {source} is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"ML cut file {source} must be a mapping")
    unknown = sorted(set(payload) - {"charge", "spin_multiplicity", "ml_atoms", "links"})
    if unknown:
        raise ValueError(f"ML cut file {source} has unknown keys: {', '.join(unknown)}")
    if "charge" not in payload or "spin_multiplicity" not in payload:
        raise ValueError(f"ML cut file {source} needs charge and spin_multiplicity")
    ml_raw = payload.get("ml_atoms")
    if not isinstance(ml_raw, list) or not ml_raw:
        raise ValueError(f"ML cut file {source} needs a non-empty ml_atoms list")
    link_raw = payload.get("links") or []
    if not isinstance(link_raw, list):
        raise ValueError(f"ML cut file {source} links must be a list")
    return MlCutSpec(
        path=source,
        charge=int(payload["charge"]),
        spin_multiplicity=int(payload["spin_multiplicity"]),
        ml_atoms=tuple(_selector(item, role="ml_atoms", source=source) for item in ml_raw),
        links=tuple(_link(item, source=source) for item in link_raw),
    )


def partition_ml_cut(
    spec: MlCutSpec,
    atom_names: Sequence[str],
    resnames: Sequence[str],
    resids: Sequence[int],
    segids: Sequence[str],
) -> tuple[np.ndarray, tuple[LinkAtom, ...]]:
    """ML indices and one ghost hydrogen per cut bond."""
    n = len(atom_names)
    if not (n == len(resnames) == len(resids) == len(segids)):
        raise ValueError("atom columns have different lengths")
    names = [str(name).strip().upper() for name in atom_names]
    residues = [str(name).strip().upper() for name in resnames]
    ids = [int(resid) for resid in resids]
    segs = [str(seg).strip().upper() for seg in segids]
    columns = list(zip(names, residues, ids, segs))

    for selector in spec.ml_atoms:
        if not _matching_indices(columns, selector):
            raise ValueError(
                f"ML cut {spec.path.name}: ml_atoms matched no atoms ({selector.label()})"
            )
    ml = [
        index
        for index, (atom_name, resname, resid, segid) in enumerate(columns)
        if any(
            selector.matches(
                resname=resname,
                segid=segid,
                resid=resid,
                atom_name=atom_name,
            )
            for selector in spec.ml_atoms
        )
    ]
    ml_set = set(ml)

    links: list[LinkAtom] = []
    for qm_selector, mm_selector in spec.links:
        qm = _one_index(columns, qm_selector, spec=spec, role="link qm")
        mm = _one_index(columns, mm_selector, spec=spec, role="link mm")
        if qm not in ml_set:
            raise ValueError(
                f"ML cut {spec.path.name}: link QM atom {qm_selector.label()} "
                "is not in ml_atoms"
            )
        if mm in ml_set:
            raise ValueError(
                f"ML cut {spec.path.name}: link MM atom {mm_selector.label()} "
                "is also in ml_atoms"
            )
        links.append(LinkAtom(qm_index=qm, mm_index=mm))
    return np.asarray(ml, dtype=int), tuple(links)


def ml_cut_from_args(args: object | None) -> tuple[np.ndarray, tuple[LinkAtom, ...]] | None:
    """Partition stashed PSF columns when ``--ml-cut`` is set."""
    if args is None:
        return None
    raw = getattr(args, "ml_cut", None)
    if raw is None or not str(raw).strip():
        return None
    mm_region = str(getattr(args, "mm_region", None) or "").strip().lower()
    if mm_region not in {"", "none"}:
        raise ValueError(
            f"--ml-cut {raw} replaces --mm-region; drop mm_region ({mm_region})"
        )
    cached = getattr(args, "_ml_cut_partition", None)
    if cached is not None and cached[0] == str(raw):
        return cached[1]
    spec = load_ml_cut(raw)
    names = getattr(args, "_cluster_atom_names", None)
    resnames = getattr(args, "_cluster_atom_resnames", None)
    resids = getattr(args, "_cluster_atom_resids", None)
    segids = getattr(args, "_cluster_atom_segids", None)
    if not names or not resnames or not resids or not segids:
        raise RuntimeError(
            f"--ml-cut {spec.path} needs the atom names from the cluster build"
        )
    partition = partition_ml_cut(spec, names, resnames, resids, segids)
    setattr(args, "_ml_cut_partition", (str(raw), partition))
    setattr(args, "_ml_cut_spec", spec)
    return partition


def ml_cut_spec_from_args(args: object | None) -> MlCutSpec | None:
    """Loaded cut file, after :func:`ml_cut_from_args` or on its own."""
    if args is None:
        return None
    cached = getattr(args, "_ml_cut_spec", None)
    if isinstance(cached, MlCutSpec):
        return cached
    raw = getattr(args, "ml_cut", None)
    if raw is None or not str(raw).strip():
        return None
    spec = load_ml_cut(raw)
    setattr(args, "_ml_cut_spec", spec)
    return spec


def _selector(raw: object, *, role: str, source: Path) -> AtomSelector:
    if not isinstance(raw, Mapping):
        raise ValueError(f"ML cut {source}: {role} entry must be a mapping")
    unknown = sorted(set(raw) - _SELECTOR_KEYS)
    if unknown:
        raise ValueError(
            f"ML cut {source}: {role} has unknown keys: {', '.join(unknown)}"
        )
    resname = _upper(raw.get("resname"))
    segid = _upper(raw.get("segid"))
    name = _upper(raw.get("name"))
    resid_raw = raw.get("resid")
    resid = None if resid_raw is None else int(resid_raw)
    names_raw = raw.get("names")
    names: frozenset[str] | None
    if names_raw is None:
        names = None
    elif isinstance(names_raw, list) and names_raw:
        names = frozenset(_upper(item) or "" for item in names_raw)
        if "" in names:
            raise ValueError(f"ML cut {source}: {role} names must be non-empty")
    else:
        raise ValueError(f"ML cut {source}: {role} names must be a non-empty list")
    if name is not None and names is not None:
        raise ValueError(f"ML cut {source}: {role} uses name or names, not both")
    selector = AtomSelector(
        resname=resname,
        segid=segid,
        resid=resid,
        names=names,
        name=name,
    )
    if selector.label() == "(empty selector)":
        raise ValueError(f"ML cut {source}: {role} entry matches every atom")
    return selector


def _link(raw: object, *, source: Path) -> tuple[AtomSelector, AtomSelector]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"ML cut {source}: each link must be a mapping")
    unknown = sorted(set(raw) - {"qm", "mm"})
    if unknown:
        raise ValueError(f"ML cut {source}: link has unknown keys: {', '.join(unknown)}")
    if "qm" not in raw or "mm" not in raw:
        raise ValueError(f"ML cut {source}: each link needs qm and mm")
    return (
        _selector(raw["qm"], role="link qm", source=source),
        _selector(raw["mm"], role="link mm", source=source),
    )


def _upper(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _matching_indices(
    columns: Sequence[tuple[str, str, int, str]],
    selector: AtomSelector,
) -> list[int]:
    hits: list[int] = []
    for index, (atom_name, resname, resid, segid) in enumerate(columns):
        if selector.matches(
            resname=resname,
            segid=segid,
            resid=resid,
            atom_name=atom_name,
        ):
            hits.append(index)
    return hits


def _one_index(
    columns: Sequence[tuple[str, str, int, str]],
    selector: AtomSelector,
    *,
    spec: MlCutSpec,
    role: str,
) -> int:
    hits = _matching_indices(columns, selector)
    if len(hits) != 1:
        raise ValueError(
            f"ML cut {spec.path.name}: {role} {selector.label()} matched "
            f"{len(hits)} atoms"
        )
    return hits[0]
