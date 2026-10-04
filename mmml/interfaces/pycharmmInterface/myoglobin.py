"""Sperm-whale MbCO from the CHARMM crystal CRD.

``--residue MBCO`` generates the protein, heme, CO, and crystal waters in the
CHARMM test structure ``mbco_au_q0.crd``. ``PRES PHEM`` bonds His93 NE2 to the
heme iron. Sulfate is left out: ``RESI SO4`` is not in the all36 protein
topology.

Six-coordinate Fe(II)–CO is a singlet (multiplicity 1). The heme formal
charge stays −2. ``--mm-region his93`` puts the heme, CO, and the His93
imidazole in PET and caps the CB–CG cut with one ghost hydrogen. The Fe–NE2
bond is a real CHARMM bond, not a link atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from mmml.interfaces.calculators.link_atoms import LinkAtom
from mmml.interfaces.pycharmmInterface.charmm_paths import mmml_repo_root
from mmml.interfaces.pycharmmInterface.cgenff_residues import parse_cgenff_residues
from mmml.interfaces.pycharmmInterface.heme_library import heme_toppar_paths

MBCO_RESIDUE_NAMES = frozenset({"MBCO", "MYOGLOBIN"})
DEFAULT_MBCO_CRD = Path("setup/charmm/test/data/mbco_au_q0.crd")

PHEM_SITES = "MB 93, HEM 1"
HIS93_SEGID = "MB"
HIS93_RESID = 93
# Six-coordinate Fe(II)–CO. The bare RESI HEME triplet does not apply.
MBCO_SPIN_MULTIPLICITY = 1
# Imidazole + CO are neutral. The two propionates stay on the heme.
HIS93_ML_CHARGE = -2

# PRES charges in top_all36_prot.rtf. They are not RESI records.
NTER_CHARGE = 1
CTER_CHARGE = -1

# His93 atoms on the MM side of the CB–CG cut. The imidazole is ML.
HIS_MM_NAMES = frozenset(
    {
        "N",
        "HN",
        "HT1",
        "HT2",
        "HT3",
        "CA",
        "HA",
        "HA1",
        "HA2",
        "C",
        "O",
        "OT1",
        "OT2",
        "CB",
        "HB",
        "HB1",
        "HB2",
        "HB3",
    }
)
HIS_IMIDAZOLE_NAMES = frozenset(
    {
        "CG",
        "ND1",
        "HD1",
        "CD2",
        "HD2",
        "CE1",
        "HE1",
        "NE2",
        "HE2",
    }
)
OMITTED_RESIDUES = frozenset({"SO4"})


def is_myoglobin_residue(name: str) -> bool:
    return str(name).strip().upper() in MBCO_RESIDUE_NAMES


def is_myoglobin_args(args: object | None) -> bool:
    """True for ``--residue MBCO`` with no composition override."""
    if args is None or getattr(args, "composition", None):
        return False
    return is_myoglobin_residue(str(getattr(args, "residue", "") or ""))


def default_mbco_crd_path(repo_root: Path | None = None) -> Path:
    return (repo_root or mmml_repo_root()) / DEFAULT_MBCO_CRD


@dataclass(frozen=True, slots=True)
class CrdAtom:
    resname: str
    name: str
    x: float
    y: float
    z: float
    segid: str
    resid: int

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class MbcoSegment:
    segid: str
    kind: str
    resnames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MbcoStructure:
    """Crystal MbCO with sulfate removed."""

    atoms: tuple[CrdAtom, ...]
    segments: tuple[MbcoSegment, ...]
    source: Path

    def formal_charge(self) -> int:
        charges = residue_formal_charges()
        total = 0
        for segment in self.segments:
            if segment.kind == "omit":
                continue
            if segment.kind == "protein":
                total += NTER_CHARGE + CTER_CHARGE
            for name in segment.resnames:
                try:
                    total += charges[name]
                except KeyError as exc:
                    raise ValueError(
                        f"no formal charge for {name} in segment {segment.segid}"
                    ) from exc
        return int(total)

    def positions(self) -> np.ndarray:
        return np.asarray([atom.xyz for atom in self.atoms], dtype=np.float64)


def parse_charmm_crd(path: Path | str) -> tuple[CrdAtom, ...]:
    """Read a CHARMM card coordinate file (title, count, then atom rows)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    body = [line for line in lines if line.strip() and not line.startswith("*")]
    if not body:
        raise ValueError(f"{path} has no coordinate rows")
    expected = int(body[0].split()[0])
    atoms: list[CrdAtom] = []
    for line in body[1:]:
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"{path} has a short coordinate row: {line!r}")
        atoms.append(
            CrdAtom(
                resname=parts[2].upper(),
                name=parts[3].upper(),
                x=float(parts[4]),
                y=float(parts[5]),
                z=float(parts[6]),
                segid=parts[7].upper(),
                resid=int(parts[8]),
            )
        )
    if len(atoms) != expected:
        raise ValueError(
            f"{path} declares {expected} atoms and has {len(atoms)} coordinate rows"
        )
    return tuple(atoms)


def load_mbco(path: Path | str | None = None) -> MbcoStructure:
    """Load MbCO and drop sulfate."""
    crd = Path(path) if path else default_mbco_crd_path()
    if not crd.is_file():
        raise FileNotFoundError(f"MbCO coordinate file not found: {crd}")
    atoms = tuple(
        atom for atom in parse_charmm_crd(crd) if atom.resname not in OMITTED_RESIDUES
    )
    return MbcoStructure(
        atoms=atoms,
        segments=tuple(_segments(atoms)),
        source=crd,
    )


def mbco_topology_residue_names(path: Path | str | None = None) -> tuple[str, ...]:
    """Unique residue names the MbCO build will generate, sulfate excluded."""
    structure = load_mbco(path)
    seen: list[str] = []
    for segment in structure.segments:
        if segment.kind == "omit":
            continue
        for name in segment.resnames:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


def his93_partition_columns(
    atom_names: Sequence[str],
    resnames: Sequence[str],
    resids: Sequence[int],
    segids: Sequence[str],
) -> tuple[np.ndarray, tuple[LinkAtom, ...]]:
    """ML indices and the His93 CB–CG ghost hydrogen.

    ML atoms are every heme atom, every CO atom, and the His93 imidazole.
    The link atom sits on CB (MM) → CG (ML).
    """
    n = len(atom_names)
    if not (n == len(resnames) == len(resids) == len(segids)):
        raise ValueError("MbCO atom columns have different lengths")
    names = [str(name).strip().upper() for name in atom_names]
    residues = [str(name).strip().upper() for name in resnames]
    ids = [int(resid) for resid in resids]
    segs = [str(seg).strip().upper() for seg in segids]
    ml: list[int] = []
    cb: int | None = None
    cg: int | None = None
    for i, (name, resname, resid, segid) in enumerate(zip(names, residues, ids, segs)):
        if resname == "HEME" or resname == "CO":
            ml.append(i)
            continue
        if segid == HIS93_SEGID and resid == HIS93_RESID and resname in {"HSD", "HSE", "HSP"}:
            if name in HIS_IMIDAZOLE_NAMES:
                ml.append(i)
                if name == "CG":
                    cg = i
            elif name in HIS_MM_NAMES:
                if name == "CB":
                    cb = i
            else:
                raise ValueError(
                    f"His93 atom {name} is neither the imidazole nor the MM backbone"
                )
    if cb is None or cg is None:
        raise ValueError("MbCO His93 is missing CB or CG for the link atom")
    if not any(residues[i] == "HEME" for i in ml):
        raise ValueError("MbCO ML region has no HEME atoms")
    return np.asarray(ml, dtype=int), (LinkAtom(qm_index=cg, mm_index=cb),)


def his93_cut_from_args(args: object) -> tuple[np.ndarray, tuple[LinkAtom, ...]]:
    """Partition stashed PSF columns. Cached on ``args`` for selection and PET."""
    cached = getattr(args, "_mbco_his93_cut", None)
    if cached is not None:
        return cached
    names = getattr(args, "_cluster_atom_names", None)
    resnames = getattr(args, "_cluster_atom_resnames", None)
    resids = getattr(args, "_cluster_atom_resids", None)
    segids = getattr(args, "_cluster_atom_segids", None)
    if not names or not resnames or not resids or not segids:
        raise RuntimeError(
            "--mm-region his93 needs the MbCO atom names from the cluster build"
        )
    cut = his93_partition_columns(names, resnames, resids, segids)
    setattr(args, "_mbco_his93_cut", cut)
    return cut


def myoglobin_electronic_state(args: object):
    """Charge and spin for PET on MbCO."""
    from mmml.interfaces.pycharmmInterface.heme_electronic import MetatomicElectronicState

    mm_region = str(getattr(args, "mm_region", None) or "none").strip().lower()
    if mm_region == "propionates":
        raise ValueError(
            "--mm-region propionates is the isolated-heme cut. "
            "MbCO uses --mm-region his93 or none."
        )
    if mm_region == "his93":
        charge: int = HIS93_ML_CHARGE
        reason = (
            "MbCO His93 imidazole, heme, and CO in PET; "
            "six-coordinate Fe(II)–CO is a singlet, heme formal charge −2"
        )
    elif mm_region in {"", "none"}:
        charge = load_mbco(getattr(args, "mbco_crd", None)).formal_charge()
        reason = (
            f"MbCO whole system formal charge {charge:+d}; "
            "six-coordinate Fe(II)–CO is a singlet"
        )
    else:
        raise ValueError(f"MBCO does not use --mm-region {mm_region!r}")
    spin = MBCO_SPIN_MULTIPLICITY
    explicit_charge = getattr(args, "charge", None)
    explicit_spin = getattr(args, "spin_multiplicity", None)
    if explicit_charge is not None or explicit_spin is not None:
        reason = f"CLI override; {reason}"
    if explicit_charge is not None:
        charge = int(explicit_charge)
    if explicit_spin is not None:
        spin = int(explicit_spin)
    return MetatomicElectronicState(int(charge), int(spin), None, None, reason)


def psf_per_atom_identity() -> tuple[list[str], list[str], list[int], list[str]]:
    """IUPAC name, residue name, resid, and segid for each PSF atom."""
    import pycharmm.psf as psf

    natom = int(psf.get_natom())
    names = [str(item).strip().upper() for item in psf.get_atype()]
    resnames = [str(item).strip().upper() for item in psf.get_res()]
    resids = [int(str(item).strip()) for item in psf.get_resid()]
    segids = [str(item).strip().upper() for item in psf.get_segid()]
    ibase = [int(item) for item in psf.get_ibase()]
    nictot = [int(item) for item in psf.get_nictot()]
    if len(names) != natom:
        raise RuntimeError(f"PSF atom names {len(names)} != natom {natom}")
    if len(ibase) != len(resnames) + 1:
        raise RuntimeError(
            f"PSF ibase length {len(ibase)} != nres+1 ({len(resnames) + 1})"
        )
    if len(nictot) != len(segids) + 1:
        raise RuntimeError(
            f"PSF nictot length {len(nictot)} != nseg+1 ({len(segids) + 1})"
        )
    residue_segids: list[str] = []
    for seg_i, segid in enumerate(segids):
        n_res = int(nictot[seg_i + 1]) - int(nictot[seg_i])
        residue_segids.extend([segid] * n_res)
    if len(residue_segids) != len(resnames):
        raise RuntimeError("PSF segment residue counts do not cover the residue list")
    atom_resnames: list[str] = []
    atom_resids: list[int] = []
    atom_segids: list[str] = []
    for res_i, resname in enumerate(resnames):
        start = int(ibase[res_i])
        stop = int(ibase[res_i + 1])
        count = stop - start
        if count <= 0:
            raise RuntimeError(f"PSF residue {res_i} has no atoms")
        atom_resnames.extend([resname] * count)
        atom_resids.extend([resids[res_i]] * count)
        atom_segids.extend([residue_segids[res_i]] * count)
    if len(atom_resnames) != natom:
        raise RuntimeError(
            f"PSF residue boundaries cover {len(atom_resnames)} atoms, natom={natom}"
        )
    return names, atom_resnames, atom_resids, atom_segids


def positions_from_crd(structure: MbcoStructure) -> np.ndarray:
    """PSF-order coordinates looked up by segid, resid, and IUPAC name."""
    names, resnames, resids, segids = psf_per_atom_identity()
    table = {
        (atom.segid, atom.resid, atom.name): atom.xyz for atom in structure.atoms
    }
    rows: list[tuple[float, float, float]] = []
    missing: list[tuple[str, int, str]] = []
    for segid, resid, name in zip(segids, resids, names):
        xyz = table.get((segid, int(resid), name))
        if xyz is None:
            missing.append((segid, int(resid), name))
        else:
            rows.append(xyz)
    if missing:
        sample = ", ".join(f"{seg} {resid} {name}" for seg, resid, name in missing[:6])
        raise ValueError(
            f"MbCO CRD is missing {len(missing)} PSF atoms (for example {sample})"
        )
    if len(rows) != len(structure.atoms):
        raise ValueError(
            f"PSF has {len(rows)} atoms and the CRD kept {len(structure.atoms)} "
            "(sulfate omitted)"
        )
    return np.asarray(rows, dtype=np.float64)


def build_myoglobin_in_charmm(
    path: Path | str | None = None,
    *,
    n_molecules: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Generate MbCO in the live CHARMM session and return ``(Z, positions, ...)``.

    Protein segment uses NTER/CTER. Heme and CO use ``first none last none``.
    Waters are generated with no angles and no dihedrals. ``PHEM`` is applied
    with angle and dihedral autogeneration off, then coordinates are assigned
    from the CRD.
    """
    if int(n_molecules) != 1:
        raise ValueError("MBCO is one crystal structure; --n-molecules must be 1")
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        ensure_charmm_session_ready,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    ensure_charmm_session_ready()
    import pycharmm.coor as coor
    import pycharmm.generate as gen
    import pycharmm.lingo as lingo
    import pycharmm.read as read
    import pandas as pd

    structure = load_mbco(path)
    lingo.charmm_script("DELETE ATOM SELE ALL END")
    prepare_charmm_vacuum()
    read_cgenff_toppar(enable_drude=False)
    for segment in structure.segments:
        if segment.kind == "omit":
            continue
        read.sequence_string(" ".join(segment.resnames))
        if segment.kind == "protein":
            gen.new_segment(seg_name=segment.segid, setup_ic=False)
        elif segment.kind in {"heme", "ligand"}:
            gen.new_segment(
                seg_name=segment.segid,
                first_patch="NONE",
                last_patch="NONE",
                setup_ic=False,
            )
        elif segment.kind == "water":
            gen.new_segment(
                seg_name=segment.segid,
                setup_ic=False,
                angle=False,
                dihedral=False,
            )
        else:
            raise ValueError(f"MbCO segment {segment.segid} kind {segment.kind} is unknown")
    gen.patch("PHEM", PHEM_SITES, setup=False, angle=False, dihedral=False)
    positions = positions_from_crd(structure)
    coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))
    atomic_numbers = np.asarray(get_Z_from_psf(), dtype=int)
    if len(atomic_numbers) != len(positions):
        raise RuntimeError(
            f"MbCO Z length {len(atomic_numbers)} != coordinates {len(positions)}"
        )
    print(
        f"MbCO: {len(positions)} atoms, formal charge {structure.formal_charge():+d}, "
        f"PHEM {PHEM_SITES}, sulfate omitted ({structure.source.name})",
        flush=True,
    )
    return atomic_numbers, positions, [len(positions)], ["MBCO"]


@lru_cache(maxsize=1)
def residue_formal_charges() -> dict[str, int]:
    """Integer RESI charges from the protein, heme, and water-ion libraries."""
    root = mmml_repo_root()
    rtf, _prm, stream = heme_toppar_paths(root)
    paths = [rtf, stream, root / "setup/charmm/toppar/toppar_water_ions.str"]
    charges: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for residue in parse_cgenff_residues(path):
            charges[residue.name.upper()] = _integer_charge(residue.charge)
    return charges


@lru_cache(maxsize=1)
def protein_rtf_residue_names() -> frozenset[str]:
    rtf, _prm, _stream = heme_toppar_paths()
    return frozenset(residue.name.upper() for residue in parse_cgenff_residues(rtf))


def _integer_charge(text: str) -> int:
    value = float(text)
    rounded = int(round(value))
    if abs(value - rounded) > 1.0e-4:
        raise ValueError(f"residue charge {text!r} is not an integer")
    return rounded


def _segments(atoms: Sequence[CrdAtom]) -> list[MbcoSegment]:
    grouped: list[tuple[str, list[str]]] = []
    current_key: tuple[str, int] | None = None
    for atom in atoms:
        key = (atom.segid, atom.resid)
        if key != current_key:
            if not grouped or grouped[-1][0] != atom.segid:
                grouped.append((atom.segid, []))
            grouped[-1][1].append(atom.resname)
            current_key = key
    protein = protein_rtf_residue_names()
    segments: list[MbcoSegment] = []
    for segid, resnames in grouped:
        unique = set(resnames)
        if unique <= OMITTED_RESIDUES:
            kind = "omit"
        elif unique == {"TIP3"}:
            kind = "water"
        elif unique == {"HEME"}:
            kind = "heme"
        elif unique == {"CO"}:
            kind = "ligand"
        elif unique <= protein:
            kind = "protein"
        else:
            unknown = sorted(unique - protein)
            raise ValueError(
                f"MbCO segment {segid} has residues that are not in the protein "
                f"topology: {', '.join(unknown)}"
            )
        segments.append(MbcoSegment(segid=segid, kind=kind, resnames=tuple(resnames)))
    return segments
