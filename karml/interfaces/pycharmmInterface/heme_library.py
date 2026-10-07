"""CHARMM protein heme library (``toppar_all36_prot_heme.str``).

``RESI HEME`` is not in CGenFF. It lives in the protein stream, which must be
read after ``top_all36_prot.rtf`` and ``par_all36m_prot.prm``. The protein
topology's default terminal patches are NTER/CTER; a heme-only segment uses
``first none last none``.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

from mmml.interfaces.pycharmmInterface.charmm_paths import mmml_repo_root
from mmml.interfaces.pycharmmInterface.cgenff_residues import parse_cgenff_residues

_ACTIVE: ContextVar[tuple[str, ...] | None] = ContextVar(
    "mmml_topology_residues", default=None
)

_HEME_STREAM = Path("setup/charmm/toppar/stream/prot/toppar_all36_prot_heme.str")
_WATER_IONS = Path("setup/charmm/toppar/toppar_water_ions.str")
_HEME_COORDS = (
    Path(__file__).resolve().parents[2] / "data" / "charmm" / "heme_mbco_coords.txt"
)
_PROT_RTF = Path("setup/charmm/toppar/top_all36_prot.rtf")
_PROT_PRM_CANDIDATES = (
    Path("setup/charmm/toppar/par_all36m_prot.prm"),
    Path("setup/charmm/toppar/par_all36_prot.prm"),
)


def heme_toppar_paths(repo_root: Path | None = None) -> tuple[Path, Path, Path]:
    """Return ``(protein rtf, protein prm, heme stream)``."""
    root = repo_root or mmml_repo_root()
    rtf = root / _PROT_RTF
    stream = root / _HEME_STREAM
    prm = next((root / rel for rel in _PROT_PRM_CANDIDATES if (root / rel).is_file()), None)
    missing = [str(p) for p in (rtf, stream) if not p.is_file()]
    if prm is None:
        missing.append(str(root / _PROT_PRM_CANDIDATES[0]))
    if missing or prm is None:
        raise FileNotFoundError(
            "CHARMM heme library is incomplete. Expected protein topology, "
            f"protein parameters, and {_HEME_STREAM.as_posix()}. Missing: {missing}"
        )
    return rtf, prm, stream


@lru_cache(maxsize=1)
def _default_heme_library_residue_names() -> frozenset[str]:
    try:
        _rtf, _prm, stream = heme_toppar_paths()
    except FileNotFoundError:
        return frozenset()
    return frozenset(r.name.upper() for r in parse_cgenff_residues(stream))


def heme_library_residue_names(repo_root: Path | None = None) -> frozenset[str]:
    """Uppercase ``RESI`` names in the heme stream (not ``PRES`` patches)."""
    if repo_root is None:
        return _default_heme_library_residue_names()
    try:
        _rtf, _prm, stream = heme_toppar_paths(repo_root)
    except FileNotFoundError:
        return frozenset()
    return frozenset(r.name.upper() for r in parse_cgenff_residues(stream))


def is_heme_library_residue(name: str, *, repo_root: Path | None = None) -> bool:
    key = str(name).strip().upper()
    return bool(key) and key in heme_library_residue_names(repo_root)


def active_topology_residues() -> tuple[str, ...] | None:
    return _ACTIVE.get()


@contextmanager
def topology_residue_context(residues: Sequence[str]) -> Iterator[None]:
    """Make ``read_cgenff_toppar`` load the heme library when every name is in it."""
    names = tuple(str(r).strip().upper() for r in residues if str(r).strip())
    token = _ACTIVE.set(names)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def residues_from_cluster_args(args: object) -> tuple[str, ...]:
    """Residue names a cluster build will generate (composition, else ``--residue``)."""
    composition = getattr(args, "composition", None)
    if composition:
        from mmml.interfaces.pycharmmInterface.mlpot.composition_spec import (
            parse_composition_entries,
        )

        return tuple(entry.residue for entry in parse_composition_entries(str(composition)))
    residue = getattr(args, "residue", None)
    if residue:
        from mmml.interfaces.pycharmmInterface.myoglobin import (
            is_myoglobin_residue,
            mbco_topology_residue_names,
        )

        key = str(residue).strip().upper()
        if is_myoglobin_residue(key):
            return mbco_topology_residue_names(getattr(args, "mbco_crd", None))
        return (key,)
    return ()


def topology_family(residues: Sequence[str] | None) -> str:
    """``cgenff`` or ``heme``. Mixed libraries raise ``ValueError``.

    Protein ions (SOD, POT, CLA, …) and TIP3 live in ``toppar_water_ions.str``
    and are read with the heme library. Standard amino acids from
    ``top_all36_prot.rtf`` can share that build (MbCO). CGenFF names cannot.
    """
    if not residues:
        return "cgenff"
    from mmml.interfaces.pycharmmInterface.heme_electronic import is_protein_ion
    from mmml.interfaces.pycharmmInterface.myoglobin import protein_rtf_residue_names

    names = [str(r).strip().upper() for r in residues if str(r).strip()]
    heme = [name for name in names if is_heme_library_residue(name)]
    if not heme:
        return "cgenff"
    protein = protein_rtf_residue_names()

    def _allowed(name: str) -> bool:
        return (
            is_heme_library_residue(name)
            or is_protein_ion(name)
            or name == "TIP3"
            or name in protein
        )

    stray = [name for name in names if not _allowed(name)]
    if stray:
        raise ValueError(
            "HEME library residues "
            f"({', '.join(sorted(set(heme)))}) use top_all36_prot.rtf and "
            "toppar_all36_prot_heme.str, not CGenFF. Protein residues, TIP3, "
            "and protein ions (SOD, POT, CLA, …) can share that build. "
            f"Other names in this build: {', '.join(stray)}."
        )
    return "heme"


_READ_RTF = re.compile(r"^\s*read\s+rtf\b", re.IGNORECASE)
_READ_PARA = re.compile(r"^\s*read\s+para", re.IGNORECASE)
_CARD_END = re.compile(r"^\s*end\b", re.IGNORECASE)


def heme_stream_cards(stream: Path) -> tuple[str, str]:
    """Split a CHARMM ``.str`` into the RTF card and the parameter card.

    ``toppar_all36_prot_heme.str`` is a script (``read rtf card append`` /
    ``read para card flex append``). Library builds do not link those script
    commands, so ``read.stream`` opens the file and leaves ``RESI HEME`` out
    of the topology. The cards themselves are what ``read.rtf`` / ``read.prm``
    accept.
    """
    lines = Path(stream).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    rtf = _card_body(lines, _READ_RTF)
    prm = _card_body(lines, _READ_PARA)
    if not rtf or not prm:
        raise ValueError(
            f"{stream} is missing a 'read rtf' or 'read para' card. "
            "Expected the CHARMM protein heme stream layout."
        )
    return rtf, prm


def _card_body(lines: list[str], header: re.Pattern[str]) -> str:
    start = next((i + 1 for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start:]:
        body.append(line)
        if _CARD_END.match(line):
            break
    return "".join(body)


@lru_cache(maxsize=1)
def heme_reference_coordinate_table() -> dict[str, tuple[float, float, float]]:
    """Crystal ``RESI HEME`` coordinates from CHARMM's myoglobin CO test CRD.

    ``toppar_all36_prot_heme.str`` stores IC bond lengths as zero, so
    ``ic.build`` leaves every atom undefined and the make-res recipe replaces
    them with a random cloud.
    """
    table: dict[str, tuple[float, float, float]] = {}
    for line in _HEME_COORDS.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        name, xs, ys, zs = text.split()
        table[name.upper()] = (float(xs), float(ys), float(zs))
    return table


def heme_reference_positions(atom_names: Sequence[str]):
    """Centered coordinates in PSF atom order, or ``None`` if a name is missing."""
    import numpy as np

    table = heme_reference_coordinate_table()
    rows: list[tuple[float, float, float]] = []
    for name in atom_names:
        xyz = table.get(str(name).strip().upper())
        if xyz is None:
            return None
        rows.append(xyz)
    coords = np.asarray(rows, dtype=float)
    coords -= coords.mean(axis=0)
    return coords


def _write_card(text: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="mmml_heme_")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def segment_terminal_patches() -> dict[str, str]:
    """Protein topology defaults to NTER/CTER; heme is not a peptide."""
    if topology_family(active_topology_residues()) == "heme":
        return {"first_patch": "NONE", "last_patch": "NONE"}
    return {}


def read_protein_heme_toppar() -> None:
    """Read protein RTF/PRM, then append the heme stream's RTF and parameter cards.

    The stream is not executed with ``read.stream``. Its ``read rtf`` / ``read
    para`` lines are CHARMM script, which this library build does not link, so
    a streamed file never registers ``RESI HEME`` and ``GENIC`` dies with
    ``Residue 'HEME' was not found``.
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        _rtf_path_for_append,
    )

    rtf, prm, stream = heme_toppar_paths()
    rtf_card, prm_card = heme_stream_cards(stream)
    cards: list[tuple[str, str]] = [(rtf_card, prm_card)]
    if _active_residues_include_ions():
        ion_stream = (mmml_repo_root() / _WATER_IONS)
        if not ion_stream.is_file():
            raise FileNotFoundError(
                f"TIP3 or protein ions were requested but {ion_stream} is missing"
            )
        cards.append(heme_stream_cards(ion_stream))
    written: list[str] = []
    try:
        with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
            read.rtf(str(rtf))
            read.prm(str(prm), flex=True)
            for rtf_text, prm_text in cards:
                rtf_raw = _write_card(rtf_text, ".rtf")
                prm_path = _write_card(prm_text, ".prm")
                rtf_append = _rtf_path_for_append(rtf_raw)
                written.extend((rtf_raw, prm_path, rtf_append))
                read.rtf(rtf_append, append=True)
                read.prm(prm_path, append=True, flex=True)
    finally:
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass


def _active_residues_include_ions() -> bool:
    from mmml.interfaces.pycharmmInterface.heme_electronic import is_protein_ion

    residues = active_topology_residues() or ()
    return any(
        is_protein_ion(name) or str(name).strip().upper() == "TIP3" for name in residues
    )
