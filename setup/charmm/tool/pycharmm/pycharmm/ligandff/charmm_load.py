"""
Read a built ligand force field into the running CHARMM.

:func:`build_ligand_ff` only writes files; this is the opt-in second step that
loads them, so building a force field never mutates CHARMM state as a side
effect. The two are meant to be used together::

    from pycharmm import ligandff

    lig = ligandff.build_ligand_ff("CC(=O)Oc1ccccc1C(O)=O", resname="AIN")
    ligandff.load_ligand_ff(lig, segid="AIN")

Notes
-----
The ``pycharmm`` submodules used here are imported *inside* the function on
purpose. This module is part of the ``pycharmm`` package, so importing
``pycharmm.read`` at module scope would be a circular import while
``pycharmm/__init__.py`` is still executing.
"""

from __future__ import annotations

import os
from pathlib import Path

# Deliberately conservative: CHARMM has held a file name in a fixed buffer of
# this size (`FNMAX` in source/io/mainio.F90), and a longer name was silently
# truncated -- the failed open is reported only as a level-0 warning, so reading
# would continue and build a force field with missing parameters. Newer builds
# size that buffer from the command line instead, but sticking to the smaller
# limit keeps this working against either, and the command line itself is read
# into a 200-character buffer (`mxcard` in source/util/parse.F90) regardless.
_CHARMM_FILENAME_MAX = 128


def _charmm_filename(path, what):
    """Return a spelling of `path` that CHARMM can open.

    Prefers the shortest of the absolute and cwd-relative forms, since CHARMM
    cannot hold a name longer than ``FNMAX`` (128) characters. Purely a string
    computation -- notably it does *not* change the working directory, so it has
    no effect a caller could observe.

    Parameters
    ----------
    path : str or path-like
        The file to read.
    what : str
        What the file is, for the error messages.

    Returns
    -------
    str
        A path no longer than CHARMM's limit.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If neither the absolute nor the relative form is short enough.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{what} not found: {resolved}. CHARMM reports a failed open only as "
            "a level-0 warning, so loading would continue and produce a force "
            "field with missing parameters."
        )

    candidates = [str(resolved)]
    try:
        candidates.append(os.path.relpath(resolved))
    except ValueError:  # pragma: no cover - only on Windows, across drives
        pass
    usable = [c for c in candidates if len(c) <= _CHARMM_FILENAME_MAX]
    if not usable:
        raise ValueError(
            f"{what} path is too long for CHARMM, which truncates a file name "
            f"beyond {_CHARMM_FILENAME_MAX} characters and then fails to open it "
            f"(shortest form tried: {min(candidates, key=len)!r}, "
            f"{len(min(candidates, key=len))} characters). Build the ligand in a "
            "shorter workdir, or run from a directory nearer the files."
        )
    return min(usable, key=len)


def load_ligand_ff(ligand, *, segid="LIG", append=None, coordinates=True, warn=True):
    """Read a built ligand's rtf, prm, and (optionally) coordinates into CHARMM.

    Parameters
    ----------
    ligand : LigandFF
        The result of :func:`build_ligand_ff`.
    segid : str, optional
        Segment name for the generated segment. Default ``"LIG"``. This is the
        CHARMM segment identifier and need not match the residue name.
    append : bool or None, optional
        Whether to add to the topology and parameters already in memory rather
        than replacing them. Default ``None``, which appends only if something
        is already loaded -- so a ligand joins a protein force field already in
        memory (what the ligand atom-type ``_`` prefix makes safe), while
        loading a ligand on its own still works. ``True`` forces appending, but
        CHARMM treats appending to an empty topology as a fatal error
        (``Cant append to zero RTF``), so pass it only when something is loaded.
    coordinates : bool, optional
        Also read the sequence from the ligand's pdb, generate the segment, and
        read its coordinates. Default ``True``. Requires that the ligand was
        built with ``make_pdb=True``.
    warn : bool, optional
        Pass CHARMM's ``warn`` option to the generate step. Default ``True``.

    Returns
    -------
    int or None
        The return code from the segment generation (``1`` on success), or
        ``None`` when ``coordinates=False`` and nothing was generated.

    Raises
    ------
    ValueError
        If ``coordinates=True`` but the ligand was built without a pdb, or a
        path is longer than CHARMM can hold in a file name.
    FileNotFoundError
        If any file the load needs is missing.

    Notes
    -----
    Appending works because a ligand's GAFF types are prefixed with ``_`` and so
    cannot collide with a protein force field's types. It does *not* extend to
    appending a **second ligand**: two GAFF ligands almost always share type
    names (``_ca``, ``_ha``, ...), and re-reading them makes CHARMM report
    ``ATOM already exists with the same name`` and then abort with ``Null
    nonbond group found``. For a system with several ligands, merge them into one
    force field with :func:`~ligandff.pipeline.combine_ligand_ffs` and read that
    with :func:`load_ligand_ffs`, rather than loading each in turn.

    Examples
    --------
    Build a ligand and add it to a protein force field already in memory:

    >>> lig = build_ligand_ff("CC(=O)Nc1ccc(O)cc1", resname="TYL")  # doctest: +SKIP
    >>> load_ligand_ff(lig, segid="TYL")  # doctest: +SKIP
    1
    """
    # Imported here, not at module scope: see the module docstring.
    from pycharmm import generate, param, read, rtf

    if coordinates and ligand.pdb is None:
        raise ValueError(
            "coordinates=True needs the ligand's pdb, but it was built with "
            "make_pdb=False; rebuild with make_pdb=True or pass "
            "coordinates=False to load only the rtf and prm"
        )

    # Append only where there is something to append to: CHARMM treats
    # `read rtf append` against an empty topology as fatal ("Cant append to zero
    # RTF"), and likewise for an empty parameter table, which would abort the
    # caller's session rather than raise.
    append_rtf = rtf.get_num_residues() > 0 if append is None else append
    append_prm = param.get_natc() > 0 if append is None else append

    read.rtf(_charmm_filename(ligand.rtf, "ligand rtf"), append=append_rtf)
    # flex: our parameter files use the flexible CHARMM parameter format.
    read.prm(_charmm_filename(ligand.prm, "ligand prm"), append=append_prm, flex=True)
    if param.get_natc() == 0:
        raise RuntimeError(
            f"reading {Path(ligand.prm).name} left the parameter table empty; "
            "the force field would have no parameters. Check the run log for "
            "CHARMM's complaint about the file."
        )

    if not coordinates:
        return None

    pdb = _charmm_filename(ligand.pdb, "ligand pdb")
    read.sequence_pdb(pdb)
    err = generate.new_segment(segid, setup_ic=True, warn=warn)
    # Deliberately no resid=True: the ligand pdb carries the residue name as its
    # segid and numbers from 1, so resid matching leaves every atom without
    # coordinates ("outside the specified sequence range"). Matching by atom
    # name, as here, is what a segment generated from this same pdb needs.
    read.pdb(pdb)
    return err


def load_ligand_ffs(ligand_set, *, segids=None, append=None, warn=True):
    """Read a combined ligand force field into CHARMM, one segment per ligand.

    The counterpart of :func:`load_ligand_ff` for a :class:`~ligandff.pipeline.LigandSet`
    from :func:`~ligandff.pipeline.combine_ligand_ffs`: the shared topology and
    parameters are read once, then each ligand contributes a segment built from
    its own pdb.

    Parameters
    ----------
    ligand_set : LigandSet
        The combined force field to read.
    segids : sequence of str or None, optional
        Segment name per ligand, in the same order. Default ``None`` uses each
        ligand's residue name.
    append : bool or None, optional
        As for :func:`load_ligand_ff`: ``None`` appends only if something is
        already loaded, so the set can join a protein force field in memory.
    warn : bool, optional
        Pass CHARMM's ``warn`` option to each generate step. Default ``True``.

    Returns
    -------
    list of int
        The generate return code for each ligand, in order.

    Raises
    ------
    ValueError
        If `segids` is given with a different length than the set's ligands, or
        any ligand was built without a pdb.
    FileNotFoundError
        If any file the load needs is missing.
    """
    from pycharmm import generate, param, read, rtf

    segids = list(ligand_set.resnames) if segids is None else list(segids)
    if len(segids) != len(ligand_set.ligands):
        raise ValueError(
            f"got {len(segids)} segids for {len(ligand_set.ligands)} ligands; "
            "pass one per ligand, in the same order, or leave segids as None"
        )
    missing = [i for i, lig in enumerate(ligand_set.ligands) if lig.pdb is None]
    if missing:
        raise ValueError(
            f"ligand(s) at position {missing} were built with make_pdb=False, so "
            "they have no coordinates to generate a segment from"
        )

    append_rtf = rtf.get_num_residues() > 0 if append is None else append
    append_prm = param.get_natc() > 0 if append is None else append
    read.rtf(_charmm_filename(ligand_set.rtf, "combined rtf"), append=append_rtf)
    read.prm(_charmm_filename(ligand_set.prm, "combined prm"), append=append_prm, flex=True)
    if param.get_natc() == 0:
        raise RuntimeError(
            f"reading {Path(ligand_set.prm).name} left the parameter table empty; "
            "check the run log for CHARMM's complaint about the file."
        )

    codes = []
    for ligand, segid in zip(ligand_set.ligands, segids):
        pdb = _charmm_filename(ligand.pdb, f"pdb for segment {segid}")
        read.sequence_pdb(pdb)
        codes.append(generate.new_segment(segid, setup_ic=True, warn=warn))
        read.pdb(pdb)
    return codes
