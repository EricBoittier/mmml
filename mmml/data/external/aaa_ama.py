"""Inspect the public ``aaa.ama`` peptide ML dataset ([MMunibas/aaa.ama](https://github.com/MMunibas/aaa.ama)).

The ``aaa_model/dataset_aaa.npz`` file holds MD snapshots of a single capped
tri-alanine peptide (34 atoms, net charge +1 e).  It is the training set used
with the legacy PhysNet + PyCHARMM workflow in ``aaa_model/dyna.sol.py``
(peptide ``PEPT`` segment + TIP3 solvent — solvent is **not** in the NPZ).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

AAA_AMA_REPO = "https://github.com/MMunibas/aaa.ama"
AAA_DATASET_URL = (
    "https://github.com/MMunibas/aaa.ama/raw/main/aaa_model/dataset_aaa.npz"
)

_ELEMENT_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O"}


@dataclass(frozen=True, slots=True)
class ElementSpecies:
    """Per-element slice of one NPZ topology (used for grouped histograms)."""

    symbol: str
    atomic_number: int
    atom_indices: tuple[int, ...]
    n_atoms: int


@dataclass(frozen=True, slots=True)
class AaaAmaDatasetReport:
    """Summary of ``dataset_aaa.npz`` contents."""

    n_frames: int
    n_atoms: int
    net_charge: float
    molecule_label: str
    formula: str
    element_species: tuple[ElementSpecies, ...]
    energy_ev: dict[str, float]
    force_mag_ev_A: dict[str, float]
    z_frame0: tuple[int, ...]
    source_url: str = AAA_DATASET_URL

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def download_dataset_aaa(dest: Path | str) -> Path:
    """Download ``dataset_aaa.npz`` from GitHub."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(AAA_DATASET_URL, path)  # noqa: S310
    return path


def load_dataset_aaa(path: Path | str) -> dict[str, np.ndarray]:
    """Load NPZ arrays (``N``, ``Z``, ``R``, ``E``, ``F``, ``Q``, ``D``)."""
    npz = np.load(Path(path), allow_pickle=True)
    return {key: np.asarray(npz[key]) for key in npz.files}


def element_species_from_z(z: np.ndarray) -> tuple[ElementSpecies, ...]:
    """Group atom indices by element (histogram bins for 'per species')."""
    z0 = np.asarray(z[0] if z.ndim == 2 else z, dtype=int)
    out: list[ElementSpecies] = []
    for anum in sorted(int(x) for x in np.unique(z0)):
        idx = tuple(int(i) for i in np.flatnonzero(z0 == anum))
        sym = _ELEMENT_SYMBOLS.get(anum, f"Z{anum}")
        out.append(ElementSpecies(sym, anum, idx, len(idx)))
    return tuple(out)


def _formula_from_z(z: np.ndarray) -> str:
    from collections import Counter

    c = Counter(int(x) for x in np.asarray(z).ravel())
    order = [6, 1, 7, 8]  # Hill-like: C, H, then N, O
    parts: list[str] = []
    for anum in order:
        if anum in c:
            sym = _ELEMENT_SYMBOLS[anum]
            n = c[anum]
            parts.append(sym if n == 1 else f"{sym}{n}")
    for anum in sorted(c):
        if anum in order:
            continue
        sym = _ELEMENT_SYMBOLS.get(anum, f"Z{anum}")
        n = c[anum]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def identify_molecule(z: np.ndarray, *, net_charge: float) -> str:
    """Heuristic label from stoichiometry (matches capped tri-alanine peptide)."""
    formula = _formula_from_z(z)
  # C9H18N3O4, Q=+1 → ACE–ALA×3–CT3–like training peptide
    if formula == "C9H18N3O4" and abs(net_charge - 1.0) < 0.01:
        return "ACE–ALA×3–CT3 peptide (34 atoms, training topology)"
    return f"custom peptide ({formula}, Q={net_charge:+.0f})"


def inspect_dataset_aaa(data: dict[str, np.ndarray]) -> AaaAmaDatasetReport:
    """Build a JSON-serializable inspection report."""
    z = np.asarray(data["Z"], dtype=int)
    e = np.asarray(data["E"], dtype=float).ravel()
    f = np.asarray(data["F"], dtype=float)
    n_atoms = int(np.asarray(data["N"]).ravel()[0])
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if not np.all(z == z[0]):
        raise ValueError("aaa.ama NPZ: atom types differ across frames (unexpected)")
    q = float(np.asarray(data["Q"]).ravel()[0])
    f_mag = np.linalg.norm(f, axis=-1)

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    return AaaAmaDatasetReport(
        n_frames=int(z.shape[0]),
        n_atoms=n_atoms,
        net_charge=q,
        molecule_label=identify_molecule(z[0], net_charge=q),
        formula=_formula_from_z(z[0]),
        element_species=element_species_from_z(z),
        energy_ev=_stats(e),
        force_mag_ev_A=_stats(f_mag),
        z_frame0=tuple(int(x) for x in z[0]),
    )


def write_report_json(report: AaaAmaDatasetReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def per_element_force_magnitudes(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Flatten |F| per element symbol across all frames and atoms."""
    z = np.asarray(data["Z"], dtype=int)
    f = np.asarray(data["F"], dtype=float)
    f_mag = np.linalg.norm(f, axis=-1)
    report = inspect_dataset_aaa(data)
    out: dict[str, np.ndarray] = {}
    for sp in report.element_species:
        out[sp.symbol] = f_mag[:, sp.atom_indices].ravel()
    return out
