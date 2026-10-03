"""
GAFF2 parameterization: SMILES/sdf -> Amber prmtop, via external tools.

The external-program stages of the ligand pipeline: RDKit (SMILES -> 3D sdf),
then antechamber (GAFF2 atom types + AM1-BCC charges), parmchk2 (missing
parameters), and tleap (Amber topology/coordinates). Kept separate from the
:mod:`ligandff.pipeline` orchestration and the Amber->CHARMM conversion.

External programs required at run time (NOT Python packages): antechamber,
parmchk2, tleap (AmberTools). RDKit is needed only for SMILES input and is
imported lazily.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# External executables needed to get from an sdf to an Amber prmtop.
TOOLS_PRMTOP = ("antechamber", "parmchk2", "tleap")


def which_missing(tools):
    """Return the executables in `tools` that are not found on PATH.

    Parameters
    ----------
    tools : iterable of str
        Executable names to look for (via :func:`shutil.which`).

    Returns
    -------
    list of str
        The subset of `tools` not present on PATH, in input order. Empty
        if every tool was found.
    """
    return [t for t in tools if shutil.which(t) is None]


def _run(cmd, cwd, log=None):
    """Run an external command and fail loudly if it returns non-zero.

    The command is run without a shell (``cmd`` is an argument list). Both
    streams are captured; on success they are optionally appended to `log`.

    Parameters
    ----------
    cmd : list of str
        Command and arguments, e.g. ``["antechamber", "-i", ...]``.
    cwd : str or path-like
        Working directory in which to run the command.
    log : file-like or None, optional
        If given, the command line and its combined stdout/stderr are written
        to this open text stream. Default ``None``.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process (with ``stdout``/``stderr`` captured).

    Raises
    ------
    RuntimeError
        If the command exits with a non-zero return code; the message
        includes the command line and its captured stdout and stderr.
    """
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if log is not None:
        log.write(f"$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def detect_formal_charge(source, *, from_smiles):
    """Return the total formal charge of a molecule, or None if undeterminable.

    Used to catch a ``net_charge`` that disagrees with the structure before
    antechamber solves AM1-BCC charges for the wrong total charge.

    Parameters
    ----------
    source : str or path-like
        A SMILES string (``from_smiles=True``) or a path to an sdf file.
    from_smiles : bool
        How to interpret `source`.

    Returns
    -------
    int or None
        The summed formal charge, or ``None`` if RDKit is unavailable or cannot
        read `source` (in which case the caller should not block on it).
    """
    try:
        from rdkit import Chem, RDLogger
    except ImportError:  # pragma: no cover - rdkit is an optional dependency
        return None

    RDLogger.DisableLog("rdApp.*")  # a parse failure here is not the user's error
    try:
        if from_smiles:
            mol = Chem.MolFromSmiles(str(source))
        else:
            mol = Chem.MolFromMolFile(str(source), removeHs=False)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol)


def estimated_parameters(frcmod):
    """Return the parmchk2 parameter lines that were guessed rather than measured.

    parmchk2 annotates every parameter it had to invent for a ligand -- either
    by analogy to another term (``penalty score= N``) or with no data at all
    (``Using the default value`` / ``ATTN, need revision``). Those terms are
    indistinguishable from real GAFF2 parameters once written to a CHARMM prm,
    so we surface them to the caller.

    Parameters
    ----------
    frcmod : str or path-like
        A parmchk2 frcmod file.

    Returns
    -------
    tuple of str
        The flagged parameter lines, stripped, in file order. Empty if every
        parameter came from GAFF2 proper (or the file cannot be read).
    """
    markers = ("penalty score", "using the default value", "attn")
    try:
        text = Path(frcmod).read_text()
    except OSError:  # pragma: no cover - caller already verified it exists
        return ()
    return tuple(
        line.strip()
        for line in text.splitlines()
        if any(m in line.lower() for m in markers) and line.strip()
    )


def _require_output(path, tool):
    """Raise RuntimeError if `tool` did not leave `path` behind.

    Parameters
    ----------
    path : pathlib.Path
        The file the tool was expected to write.
    tool : str
        Tool name, for the error message.

    Raises
    ------
    RuntimeError
        If `path` does not exist or is empty.
    """
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"{tool} reported success but did not write {path.name}; "
            f"see the run log in {path.parent} for its output"
        )


def smiles_to_sdf(smiles, sdf_path):
    """Build a 3D ``sdf`` from a SMILES string with RDKit.

    Adds explicit hydrogens and embeds a single 3D conformer with ETKDG.
    Mirrors ``reference/smile_2_sdf.py``.

    Parameters
    ----------
    smiles : str
        SMILES string for the molecule.
    sdf_path : str or path-like
        Path of the ``sdf`` file to write.

    Returns
    -------
    pathlib.Path or str
        `sdf_path`, for convenience.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If RDKit cannot parse `smiles`.
    RuntimeError
        If 3D conformer embedding fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "building a ligand from a SMILES string requires rdkit "
            "(`conda install -c conda-forge rdkit`)"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        raise RuntimeError(f"rdkit failed to embed 3D coordinates for {smiles!r}")
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return sdf_path


def sdf_to_prmtop(sdf, name, resname, net_charge, workdir, log):
    """Parameterize an ``sdf`` with GAFF2 and write an Amber prmtop/inpcrd.

    Runs antechamber (GAFF2 atom types + AM1-BCC charges), parmchk2 (missing
    parameters), then tleap (topology/coordinates). Mirrors ``reference/gaff2.py``,
    with the hard-coded ligand name, residue name and net charge lifted to
    arguments.

    Parameters
    ----------
    sdf : str or path-like
        Input 3D structure file.
    name : str
        Base name for the output files (``<name>.mol2``, ``<name>.prmtop`` ...).
    resname : str
        Residue name assigned by antechamber (``-rn``).
    net_charge : int
        Formal net charge passed to antechamber (``-nc``).
    workdir : pathlib.Path
        Directory in which the tools run and files are written.
    log : file-like or None
        Open text stream for command logging (see :func:`_run`).

    Returns
    -------
    tuple of pathlib.Path
        ``(mol2, frcmod, prmtop, inpcrd)``.

    Raises
    ------
    RuntimeError
        If antechamber, parmchk2 or tleap exits non-zero, or exits successfully
        without writing the output it was asked for.
    """
    mol2 = workdir / f"{name}.mol2"
    frcmod = workdir / f"{name}.frcmod"
    prmtop = workdir / f"{name}.prmtop"
    inpcrd = workdir / f"{name}.inpcrd"

    # Clear any outputs left by an earlier run in this workdir, so a stale file
    # can never be mistaken for one this run produced.
    for stale in (mol2, frcmod, prmtop, inpcrd):
        stale.unlink(missing_ok=True)

    _run(
        [
            "antechamber",
            "-i",
            str(sdf),
            "-fi",
            "sdf",
            "-o",
            str(mol2),
            "-fo",
            "mol2",
            "-at",
            "gaff2",
            "-c",
            "bcc",
            "-rn",
            resname,
            "-nc",
            str(net_charge),
        ],
        workdir,
        log,
    )
    _require_output(mol2, "antechamber")

    _run(
        ["parmchk2", "-i", str(mol2), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2"],
        workdir,
        log,
    )
    _require_output(frcmod, "parmchk2")

    leap_in = workdir / "ligand.in"
    leap_in.write_text(
        "source leaprc.gaff2\n"
        f"loadamberparams {frcmod.name}\n"
        f"mol = loadmol2 {mol2.name}\n"
        f"saveamberparm mol {prmtop.name} {inpcrd.name}\n"
        "quit\n"
    )
    _run(["tleap", "-f", leap_in.name], workdir, log)
    # tleap exits non-zero on hard failures, but can also report errors in
    # leap.log and leave no topology behind, so verify rather than assume.
    _require_output(prmtop, "tleap")
    _require_output(inpcrd, "tleap")

    return mol2, frcmod, prmtop, inpcrd
