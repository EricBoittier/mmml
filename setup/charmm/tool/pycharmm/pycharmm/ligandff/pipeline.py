"""
Ligand pipeline: SMILES/sdf -> CHARMM rtf + prm (+ pdb).

The public entry point :func:`build_ligand_ff` ties the stages together:
GAFF2 parameterization (:mod:`ligandff.gaff`), the Amber->CHARMM conversion
(:mod:`ligandff.amber_to_charmm` + :mod:`ligandff.charmm_writer`), and a
ParmEd-written pdb.

Python packages required: rdkit (SMILES input only), parmed, pandas, numpy.
All are imported lazily, so ``import ligandff`` succeeds without them; a missing
dependency is reported only when the code path that needs it runs.
"""

from __future__ import annotations

import shutil
from collections import namedtuple
from pathlib import Path

from .gaff import (
    TOOLS_PRMTOP,
    detect_formal_charge,
    estimated_parameters,
    sdf_to_prmtop,
    smiles_to_sdf,
    which_missing,
)

LigandFF = namedtuple(
    "LigandFF",
    [
        "rtf",
        "prm",
        "pdb",
        "prmtop",
        "inpcrd",
        "mol2",
        "frcmod",
        "sdf",
        "estimated_parameters",
    ],
)
LigandFF.__doc__ = """Paths to the files produced by a successful ligand build.

Returned by :func:`build_ligand_ff`. Every field is a
:class:`pathlib.Path` except ``pdb``, which may be ``None``.

Attributes
----------
rtf : pathlib.Path
    CHARMM residue topology file.
prm : pathlib.Path
    CHARMM parameter file.
pdb : pathlib.Path or None
    CHARMM-readable coordinate (pdb) file, or ``None`` when the build was
    run with ``make_pdb=False``.
prmtop : pathlib.Path
    Amber topology file (intermediate).
inpcrd : pathlib.Path
    Amber coordinate file (intermediate).
mol2 : pathlib.Path
    GAFF2 ``mol2`` with AM1-BCC charges (antechamber output).
frcmod : pathlib.Path
    Missing-parameter file produced by parmchk2.
sdf : pathlib.Path
    The 3D ``sdf`` used as input (generated from SMILES, or copied in).
estimated_parameters : tuple of str
    parmchk2 parameter lines that were guessed rather than taken from GAFF2 --
    by analogy (``penalty score= N``) or with no data (``Using the default
    value``). Empty means every parameter came from GAFF2 proper. A non-empty
    tuple is not an error, but those terms are the least trustworthy part of
    the force field and are worth inspecting before production use.
"""


def _prmtop_to_charmm(prmtop, rtf_path, prm_path):
    """Convert an Amber ligand prmtop into CHARMM rtf + prm files.

    Loads the prmtop with ParmEd, builds the force-field model with the parser
    (:func:`ligandff.amber_to_charmm.build_forcefield`, ligand path), and
    serializes it with :mod:`ligandff.charmm_writer`.

    Parameters
    ----------
    prmtop : str or path-like
        Amber topology file (as produced by tleap).
    rtf_path : str or path-like
        Path of the CHARMM residue topology file to write.
    prm_path : str or path-like
        Path of the CHARMM parameter file to write.

    Returns
    -------
    tuple of (str or path-like)
        ``(rtf_path, prm_path)``, for convenience.

    Raises
    ------
    ImportError
        If parmed, pandas or numpy is not installed.
    """
    try:
        import parmed as pmd

        from . import charmm_writer as writer
        from .amber_to_charmm import build_forcefield
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the Amber->CHARMM conversion requires parmed, pandas and numpy "
            "(`conda install -c conda-forge parmed pandas numpy`)"
        ) from exc

    ff = build_forcefield(pmd.load_file(str(prmtop)), ligand=True)
    Path(rtf_path).write_text(
        writer.write_rtf(ff, title=writer.LIGAND_RTF_TITLE, declarations=writer.LIGAND_RTF_DECL)
    )
    Path(prm_path).write_text(writer.write_prm(ff, title=writer.LIGAND_PRM_TITLE))
    return rtf_path, prm_path


def _write_charmm_pdb(prmtop, inpcrd, pdb_path, resname):
    """Write a CHARMM-readable pdb from the Amber prmtop+inpcrd, via ParmEd.

    Emits ATOM (not HETATM) records with ``segid=resname`` so CHARMM's
    ``read sequ pdb`` / ``read coor pdb`` accept it: ``charmm=True`` writes the
    segid column, ``use_hetatoms=False`` forces ATOM records, and
    ``renumber=True`` numbers atoms/residues from 1.

    Parameters
    ----------
    prmtop : str or path-like
        Amber topology file.
    inpcrd : str or path-like
        Amber coordinate file (supplies the coordinates written to the pdb).
    pdb_path : str or path-like
        Path of the pdb file to write.
    resname : str
        Residue name and segid to stamp on every residue.

    Returns
    -------
    pathlib.Path or str
        `pdb_path`, for convenience.

    Raises
    ------
    ImportError
        If parmed is not installed.
    """
    try:
        import parmed as pmd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing the CHARMM pdb requires parmed (`conda install -c conda-forge parmed`)"
        ) from exc

    struct = pmd.load_file(str(prmtop), str(inpcrd))
    for res in struct.residues:
        res.name = resname
        res.segid = resname
    pmd.formats.PDBFile.write(struct, str(pdb_path), charmm=True, use_hetatoms=False, renumber=True)
    return pdb_path


def build_ligand_ff(
    source,
    resname="LIG",
    net_charge=None,
    from_smiles=None,
    name=None,
    workdir=None,
    make_pdb=True,
):
    """Build CHARMM rtf + prm (+ pdb) for a ligand from a SMILES string or sdf.

    Parameters
    ----------
    source : str or path-like
        A SMILES string, or the path to an existing ``.sdf`` file.
    resname : str, optional
        CHARMM/Amber residue name for the ligand (antechamber ``-rn``).
        Default ``"LIG"``.
    net_charge : int or None, optional
        Formal net charge of the molecule (antechamber ``-nc``). Default
        ``None``, which detects the charge from the structure with RDKit
        (falling back to ``0`` if RDKit is unavailable). Passing a value that
        disagrees with the detected charge raises, rather than letting
        antechamber solve AM1-BCC charges for the wrong total charge.
    from_smiles : bool or None, optional
        Force interpretation of `source`. If ``None`` (default), `source` is
        treated as an sdf path when it ends in ``.sdf`` and as a SMILES string
        otherwise.
    name : str or None, optional
        Base name for the output files. Defaults to the sdf stem, or
        ``resname.lower()`` for SMILES input.
    workdir : str, path-like, or None, optional
        Directory in which intermediates and outputs are written. Created if
        needed; defaults to ``./<name>_ligandff``.
    make_pdb : bool, optional
        Also emit a CHARMM-readable pdb (written with ParmEd; no extra external
        tools). Default ``True``.

    Returns
    -------
    LigandFF
        Named tuple of paths: rtf, prm, pdb, prmtop, inpcrd, mol2, frcmod, sdf,
        plus ``estimated_parameters``. `pdb` is ``None`` if ``make_pdb=False``.

    Raises
    ------
    FileNotFoundError
        If a required AmberTools executable (antechamber, parmchk2, tleap) is
        not on PATH -- raised up front, before any work is done -- or if
        `source` names an sdf file that does not exist.
    ValueError
        If an explicit `net_charge` disagrees with the structure's detected
        formal charge.
    ImportError
        If an optional Python package needed by the chosen path is missing:
        rdkit (SMILES input), or parmed/pandas/numpy (conversion and pdb).
    RuntimeError
        If antechamber, parmchk2 or tleap fails during the run.

    Examples
    --------
    From a SMILES string:

    >>> ff = build_ligand_ff("CC(=O)Oc1ccccc1C(O)=O", resname="AIN")
    >>> ff.rtf, ff.prm  # doctest: +SKIP
    (PosixPath('.../ain_ligandff/ain.rtf'), PosixPath('.../ain_ligandff/ain.prm'))

    From an existing sdf, into a chosen directory, skipping the pdb:

    >>> ff = build_ligand_ff(
    ...     "mol.sdf", resname="LIG", workdir="build", make_pdb=False
    ... )  # doctest: +SKIP
    """
    # -- resolve SMILES vs sdf ------------------------------------------------
    # Anything named *.sdf is a file, full stop: treating a mistyped path as a
    # SMILES string would report "could not parse SMILES: my/typo.sdf".
    src = str(source)
    if from_smiles is None:
        from_smiles = not src.lower().endswith(".sdf")
    if not from_smiles and not Path(src).is_file():
        raise FileNotFoundError(f"no such sdf file: {src}")

    # -- reconcile the declared net charge with the structure ------------------
    detected = detect_formal_charge(src, from_smiles=from_smiles)
    if net_charge is None:
        net_charge = 0 if detected is None else detected
    elif detected is not None and detected != net_charge:
        raise ValueError(
            f"net_charge={net_charge} disagrees with the structure's formal "
            f"charge of {detected}; antechamber would solve AM1-BCC charges for "
            "the wrong total charge. Pass the correct net_charge, or leave it "
            "as None to use the detected value."
        )

    if name is None:
        name = resname.lower() if from_smiles else Path(src).stem

    # Resolve to an absolute path: every derived file path is built from
    # `workdir`, and the external tools run with cwd=workdir, so a relative
    # workdir would otherwise be applied twice (e.g. _exp/_exp/aspirin.sdf).
    workdir = (Path(workdir) if workdir else Path.cwd() / f"{name}_ligandff").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # -- preflight: required external tools -----------------------------------
    missing = which_missing(TOOLS_PRMTOP)
    if missing:
        raise FileNotFoundError(
            "missing required AmberTools executables on PATH: "
            + ", ".join(missing)
            + " (install AmberTools, e.g. `conda install -c conda-forge ambertools`)"
        )
    log_path = workdir / f"{name}.ligandff.log"
    with log_path.open("w") as log:
        # -- SMILES -> sdf (or use the sdf we were handed) --------------------
        if from_smiles:
            sdf = smiles_to_sdf(src, workdir / f"{name}.sdf")
        else:
            sdf = workdir / f"{name}.sdf"
            if Path(src).resolve() != sdf.resolve():
                shutil.copyfile(src, sdf)

        # -- sdf -> prmtop ----------------------------------------------------
        mol2, frcmod, prmtop, inpcrd = sdf_to_prmtop(sdf, name, resname, net_charge, workdir, log)

        # -- prmtop -> CHARMM rtf/prm -----------------------------------------
        rtf, prm = _prmtop_to_charmm(prmtop, workdir / f"{name}.rtf", workdir / f"{name}.prm")

        # -- prmtop+inpcrd -> CHARMM-readable pdb (ParmEd, no shell-out) ------
        pdb = None
        if make_pdb:
            pdb = _write_charmm_pdb(prmtop, inpcrd, workdir / f"{name}.pdb", resname)

        # -- which parameters did parmchk2 have to guess? ----------------------
        estimated = estimated_parameters(frcmod)
        if estimated:
            log.write(
                f"\n{len(estimated)} parameter(s) were estimated by parmchk2 "
                "rather than taken from GAFF2:\n"
            )
            log.writelines(f"  {line}\n" for line in estimated)

    return LigandFF(
        rtf=rtf,
        prm=prm,
        pdb=pdb,
        prmtop=prmtop,
        inpcrd=inpcrd,
        mol2=mol2,
        frcmod=frcmod,
        sdf=sdf,
        estimated_parameters=estimated,
    )


LigandSet = namedtuple("LigandSet", ["rtf", "prm", "ligands", "resnames", "estimated_parameters"])
LigandSet.__doc__ = """Several ligands sharing one CHARMM force field.

Returned by :func:`combine_ligand_ffs`, and read into CHARMM by
:func:`~ligandff.charmm_load.load_ligand_ffs`.

Attributes
----------
rtf : pathlib.Path
    The combined residue topology file, holding one ``RESI`` per ligand.
prm : pathlib.Path
    The combined parameter file.
ligands : tuple of LigandFF
    The individual builds, in the order they were merged. Each still carries its
    own pdb, which is what supplies coordinates for its segment.
resnames : tuple of str
    Each ligand's residue name, taken from its topology. Used as the default
    segment names when loading.
estimated_parameters : tuple of str
    Every parmchk2-estimated parameter across all the ligands (see
    :class:`LigandFF`).
"""


def combine_ligand_ffs(ligands, *, rtf, prm):
    """Merge built ligands into one rtf/prm so CHARMM can hold them together.

    CHARMM reads a topology and parameter set as one coherent whole, so several
    ligands cannot be loaded one after another: two GAFF ligands almost always
    share atom-type names (``_ca``, ``_ha``, ...) and re-reading them makes CHARMM
    report ``ATOM already exists with the same name`` and then abort. Merging
    first is the way to hold more than one ligand at once -- shared atom types
    and parameters collapse to a single entry, and each ligand keeps its own
    ``RESI``.

    Parameters
    ----------
    ligands : iterable of LigandFF
        Results of :func:`build_ligand_ff`. Each must have a distinct residue
        name, since a residue name identifies a topology in the rtf.
    rtf : str or path-like
        Path of the combined residue topology file to write.
    prm : str or path-like
        Path of the combined parameter file to write.

    Returns
    -------
    LigandSet
        The combined files plus the ligands they cover.

    Raises
    ------
    ValueError
        If two ligands share a residue name but differ in topology, or give
        conflicting values for the same parameter (which happens when parmchk2
        filled the same gap differently for two molecules -- those ligands cannot
        share one parameter table and must be run separately).
    ImportError
        If parmed is not installed.

    Examples
    --------
    >>> a = build_ligand_ff("CC(=O)Oc1ccccc1C(O)=O", resname="AIN")  # doctest: +SKIP
    >>> b = build_ligand_ff("CC(=O)Nc1ccc(O)cc1", resname="TYL")  # doctest: +SKIP
    >>> both = combine_ligand_ffs([a, b], rtf="ligs.rtf", prm="ligs.prm")  # doctest: +SKIP
    """
    try:
        import parmed as pmd

        from . import charmm_writer as writer
        from .amber_to_charmm import build_forcefield
        from .model import ForceField
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "combining ligand force fields requires parmed, pandas and numpy "
            "(`conda install -c conda-forge parmed pandas numpy`)"
        ) from exc

    ligands = tuple(ligands)
    if not ligands:
        raise ValueError("combine_ligand_ffs needs at least one ligand")

    combined = ForceField()
    resnames = []
    estimated = []
    for ligand in ligands:
        parm = pmd.load_file(str(ligand.prmtop))
        resnames.append(parm.residues[0].name)
        combined.extend(build_forcefield(parm, ligand=True))
        estimated.extend(ligand.estimated_parameters)

    Path(rtf).write_text(
        writer.write_rtf(
            combined, title=writer.LIGAND_RTF_TITLE, declarations=writer.LIGAND_RTF_DECL
        )
    )
    Path(prm).write_text(writer.write_prm(combined, title=writer.LIGAND_PRM_TITLE))
    return LigandSet(
        rtf=Path(rtf),
        prm=Path(prm),
        ligands=ligands,
        resnames=tuple(resnames),
        estimated_parameters=tuple(estimated),
    )
