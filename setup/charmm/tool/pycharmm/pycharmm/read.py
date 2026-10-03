# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Get data into CHARMM from several file types.

Corresponds to CHARMM command `READ` in the IO module
See [READ documentation](https://academiccharmm.org/documentation/version/c47b1/io#Read)

Examples
========
>>> import pycharmm
>>> import pycharmm.read

Read a RTF file.
>>> read.rtf('data/top_all36_prot.rtf')

Read a paramter file
>>> read.prm('data/par_all36_prot.prm', flex=True)

Read a sequence from a PDB file
>>> read.sequence_pdb('...')

"""

import ctypes
from pycharmm.loader import lib
import pycharmm.script
from pycharmm.charmm_file import c_api_path_buffer
import pycharmm.atom_info as atom_info



def _resolve_read_path(filename: str) -> str:
    try:
        from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path
    except ImportError:
        return filename
    fortran_path, _alias = charmm_fortran_path(filename, for_write=False)
    return fortran_path


def _invalidate_cache():
    """Invalidate atom cache after PSF modifications."""
    atom_info.invalidate_atom_cache()


# read a topology file given a bytes filename
def rtf(filename, append=False, prnt=False, **kwargs):
    """Read a topology (RTF) file via the KEY_LIBRARY C API.

    ``prnt`` is accepted for call-site compatibility. Library builds do not
    link the ``read rtf`` script command.
    """
    del prnt, kwargs
    fortran_path = _resolve_read_path(filename)
    buf, fn_len = c_api_path_buffer(fortran_path)
    append = ctypes.c_int(1 if append else 0)
    status = int(
        lib.read_rtf_file(
            buf,
            ctypes.byref(fn_len),
            ctypes.byref(append),
        )
    )
    if status != 1:
        raise RuntimeError(
            f"read_rtf_file failed for {filename!r} "
            f"(staging={fortran_path!r}, status={status})"
        )
    _invalidate_cache()


def prm(filename, append=False, flex=False, prnt=False, **kwargs):
    """Read a parameter file via the KEY_LIBRARY C API.

    ``prnt`` is accepted for call-site compatibility. Library builds do not
    link the ``read param`` script command.
    """
    del prnt, kwargs
    fortran_path = _resolve_read_path(filename)
    buf, fn_len = c_api_path_buffer(fortran_path)
    append = ctypes.c_int(1 if append else 0)
    flex = ctypes.c_int(1 if flex else 0)
    status = int(
        lib.read_param_file(
            buf,
            ctypes.byref(fn_len),
            ctypes.byref(append),
            ctypes.byref(flex),
        )
    )
    if status != 1:
        raise RuntimeError(
            f"read_param_file failed for {filename!r} "
            f"(staging={fortran_path!r}, status={status})"
        )
    _invalidate_cache()


def psf_card(filename, append=False, xplor=False, **kwargs):
    """
    Read a PSF (Protein Structure File) card file from disk

    Parameters
    ----------
    filename : str
        Path to the PSF card file
    append : bool
        If True, append atoms to existing PSF instead of replacing.
        (default: False)
    xplor : bool
        If True, read XPLOR format PSF file. (default: False)
    **kwargs
        Additional keyword arguments passed to CommandScript.

    Examples
    --------
    >>> import pycharmm.read as read
    >>> read.psf_card('protein.psf')
    >>> read.psf_card('ligand.psf', append=True)
    >>> read.psf_card('xplor_format.psf', xplor=True)
    """
    cmd_kwargs = {}
    if append:
        cmd_kwargs['append'] = True
    if xplor:
        cmd_kwargs['xplor'] = True

    psf_script = pycharmm.script.CommandScript('read',
                                               psf='card',
                                               name=filename,
                                               **cmd_kwargs,
                                               **kwargs)
    result = psf_script.run()
    _invalidate_cache()
    return result


def pdb(filename, resid=False, comp=False, offset=0, model=0,
        official=False, append=False, initial=False, **kwargs):
    """
    Read a PDB file from disk given a path

    Parameters
    ----------
    filename : str
        Path to the PDB file
    resid : bool
        If True, map atoms using SEGID and RESID labels instead of residue
        numbers. Recommended when coordinates come from different PSF.
        (default: False)
    comp : bool
        If True, read coordinates into comparison set instead of main set.
        (default: False)
    offset : int
        Residue number offset for reading coordinates. Both positive and
        negative values are allowed. (default: 0)
    model : int
        NMR MODEL number to read from multi-model PDB file.
        0 reads the first model. (default: 0)
    official : bool
        If True, read official PDB format where chain ID (segid) is limited
        to one character. (default: False)
    append : bool
        If True, deselect all atoms up to the highest one with known
        position. Useful for multi-segment structures. (default: False)
    initial : bool
        If True, initialize/restart coordinates. (default: False)
    **kwargs
        Additional keyword arguments passed to CommandScript, including
        atom selection keywords.

    Examples
    --------
    >>> import pycharmm.read as read
    >>> read.pdb('structure.pdb')
    >>> read.pdb('nmr_ensemble.pdb', model=5)
    >>> read.pdb('official.pdb', official=True, resid=True)
    >>> read.pdb('segment2.pdb', append=True, comp=True)
    """
    # Build kwargs, only include non-default values
    cmd_kwargs = {}
    if resid:
        cmd_kwargs['resid'] = True
    if comp:
        cmd_kwargs['comp'] = True
    if offset != 0:
        cmd_kwargs['offset'] = offset
    if model != 0:
        cmd_kwargs['model'] = model
    if official:
        cmd_kwargs['official'] = True
    if append:
        cmd_kwargs['append'] = True
    if initial:
        cmd_kwargs['initial'] = True

    pdb_script = pycharmm.script.CommandScript('read',
                                               coor='pdb',
                                               name=filename,
                                               **cmd_kwargs,
                                               **kwargs)
    pdb_script.run()


def mmcif(filename, resid=False, comp=False, offset=0, model=0,
          append=False, initial=False, label=False, **kwargs):
    """Read coordinates from an mmCIF/PDBx file."""
    cmd_kwargs = {}
    if resid:
        cmd_kwargs['resid'] = True
    if comp:
        cmd_kwargs['comp'] = True
    if offset != 0:
        cmd_kwargs['offset'] = offset
    if model != 0:
        cmd_kwargs['model'] = model
    if append:
        cmd_kwargs['append'] = True
    if initial:
        cmd_kwargs['initial'] = True
    if label:
        cmd_kwargs['label'] = True

    mmcif_script = pycharmm.script.CommandScript('read',
                                                 coor='mmcif',
                                                 name=filename,
                                                 **cmd_kwargs,
                                                 **kwargs)
    mmcif_script.run()


def pdbx(filename, **kwargs):
    """Alias for :func:`mmcif`."""
    return mmcif(filename, **kwargs)


def stream(filename, **kwargs):
    """
    Read a stream file from disk given a path

    Parameters
    ----------
    filename : str

    Note
    ----
    Stream files can contain arbitrary CHARMM commands that may modify
    the PSF structure. The atom cache is invalidated after execution.
    """
    stream_command = 'stream '+str(filename)
    stream_script = pycharmm.script.CommandScript(stream_command,
                                               **kwargs)
    result = stream_script.run()
    _invalidate_cache()
    return result


def sequence_pdb(filename, chain=None, segi=None, nchain=0, hetatm=False,
                 noatom=False, seqres=False, firstresid=1, skip=None,
                 alias=None, **kwargs):
    """Read sequence from a PDB file

    Reads residue sequence from a PDB file. By default reads from ATOM records.
    Residue IDs (resid) are always read from the resSeq field of the PDB file.

    Parameters
    ----------
    filename : str
        Path to the PDB file
    chain : str, optional
        One-letter PDB chain ID (position 22 of ATOM/HETATM records) to read.
        If None, reads all chains. (default: None)
    segi : str, optional
        Segment ID to filter (columns 73-76 of PDB file).
        If None, reads all segments. (default: None)
    nchain : int
        Start reading from chain number N as defined by TER separator records.
        (default: 0 - start from first chain)
    hetatm : bool
        If True, include HETATM records in addition to ATOM records.
        (default: False)
    noatom : bool
        If True, do not read ATOM records (use with hetatm=True to read
        only HETATM). (default: False)
    seqres : bool
        If True, read sequence from SEQRES records instead of ATOM records.
        Useful when there are missing residues in ATOM records. (default: False)
    firstresid : int
        Starting residue ID for numbering when using seqres=True. (default: 1)
    skip : str or list of str, optional
        Residue name(s) to skip/ignore when reading sequence. (default: None)
    alias : dict, optional
        Residue name translation dictionary {old_name: new_name}.
        Each occurrence of old_name will be replaced by new_name. (default: None)
    **kwargs
        Additional keyword arguments passed to CommandScript.

    Notes
    -----
    After reading, the variables SQNRES and SQRESID are set to the number
    of residues read and the segment ID used, respectively.

    Examples
    --------
    >>> import pycharmm.read as read
    >>> read.sequence_pdb('protein.pdb')
    >>> read.sequence_pdb('protein.pdb', chain='A')
    >>> read.sequence_pdb('protein.pdb', hetatm=True, skip='HOH')
    >>> read.sequence_pdb('protein.pdb', seqres=True, firstresid=1)
    >>> read.sequence_pdb('protein.pdb', alias={'HSD': 'HIS', 'HSE': 'HIS'})
    """
    cmd_kwargs = {}
    if chain is not None:
        cmd_kwargs['chain'] = chain
    if segi is not None:
        cmd_kwargs['segi'] = segi
    if nchain != 0:
        cmd_kwargs['nchain'] = nchain
    if hetatm:
        cmd_kwargs['hetatm'] = True
    if noatom:
        cmd_kwargs['noatom'] = True
    if seqres:
        cmd_kwargs['seqres'] = True
    if firstresid != 1:
        cmd_kwargs['firstresid'] = firstresid
    if skip is not None:
        if isinstance(skip, str):
            cmd_kwargs['skip'] = skip
        else:
            # Multiple skip values - add each one
            for s in skip:
                cmd_kwargs[f'skip'] = s  # Note: may need special handling
    if alias is not None:
        for old_name, new_name in alias.items():
            cmd_kwargs[f'alias_{old_name}'] = new_name  # Note: may need special handling

    read_sequence_pdb = pycharmm.script.CommandScript('read',
                                                      sequence='pdb',
                                                      name=filename,
                                                      **cmd_kwargs,
                                                      **kwargs)
    read_sequence_pdb.run()


def sequence_mmcif(filename, chain=None, segi=None, nchain=0, hetatm=False,
                   noatom=False, seqres=False, firstresid=1, skip=None,
                   alias=None, label=False, **kwargs):
    """Read sequence from an mmCIF/PDBx file."""
    cmd_kwargs = {}
    if chain is not None:
        cmd_kwargs['chain'] = chain
    if segi is not None:
        cmd_kwargs['segi'] = segi
    if nchain != 0:
        cmd_kwargs['nchain'] = nchain
    if hetatm:
        cmd_kwargs['hetatm'] = True
    if noatom:
        cmd_kwargs['noatom'] = True
    if seqres:
        cmd_kwargs['seqres'] = True
    if firstresid != 1:
        cmd_kwargs['firstresid'] = firstresid
    if label:
        cmd_kwargs['label'] = True
    if skip is not None:
        if isinstance(skip, str):
            cmd_kwargs['skip'] = skip
        else:
            raise ValueError("skip accepts one residue name")
    if alias:
        cmd_kwargs['alias'] = ' ALIAS '.join(
            '{} {}'.format(old_name, new_name)
            for old_name, new_name in alias.items()
        )

    read_sequence_mmcif = pycharmm.script.CommandScript('read',
                                                        sequence='mmcif',
                                                        name=filename,
                                                        **cmd_kwargs,
                                                        **kwargs)
    read_sequence_mmcif.run()


def sequence_pdbx(filename, **kwargs):
    """Alias for :func:`sequence_mmcif`."""
    return sequence_mmcif(filename, **kwargs)


# read in a sequence from bytes
# eg b'AMN CBX'
def sequence_string(seq):
    """Create a sequence

    Parameters
    ----------
    seq : str
          a string of space delimited names (see example below)

    Returns
    -------
    int
       1 indicates success, any other value indicates failure

    Examples
    --------
    >>> import pycharmm
    >>> import pycharmm.read
    >>> read.sequence_pdb('AMN CBX')

    """
    seq = seq.encode()
    seq_len = ctypes.c_int(len(seq))
    seq_str = ctypes.create_string_buffer(seq)
    err_code = lib.read_sequence_string(seq_str,
                                               ctypes.byref(seq_len))
    return err_code


def coor_card(filename, resid=False, comp=False, offset=0,
              append=False, initial=False, ignore=False, **kwargs):
    """Read a CHARMM format coordinate (CRD) file

    Parameters
    ----------
    filename : str
        Path to the CHARMM coordinate card file
    resid : bool
        If True, map atoms using SEGID and RESID labels instead of residue
        numbers. Recommended when coordinates come from different PSF.
        (default: False)
    comp : bool
        If True, read coordinates into comparison set instead of main set.
        (default: False)
    offset : int
        Residue number offset for reading coordinates. Allows reading
        coordinates from a different PSF. (default: 0)
    append : bool
        If True, add offset pointing to residue beyond highest with known
        positions, and deselect atoms below that residue. Useful for
        multi-segment structures. (default: False)
    initial : bool
        If True, initialize/restart coordinates. (default: False)
    ignore : bool
        If True, bypass normal tests of residue name, number, and atom type.
        Coordinates are mapped sequentially - use with caution! (default: False)
    **kwargs
        Additional keyword arguments passed to CommandScript, including
        atom selection keywords.

    Examples
    --------
    >>> import pycharmm.read as read
    >>> read.coor_card('structure.crd')
    >>> read.coor_card('segment.crd', resid=True)
    >>> read.coor_card('second_chain.crd', append=True)
    """
    fortran_path = _resolve_read_path(filename)
    filename = fortran_path
    cmd_kwargs = {}
    if resid:
        cmd_kwargs['resid'] = True
    if comp:
        cmd_kwargs['comp'] = True
    if offset != 0:
        cmd_kwargs['offset'] = offset
    if append:
        cmd_kwargs['append'] = True
    if initial:
        cmd_kwargs['initial'] = True
    if ignore:
        cmd_kwargs['ignore'] = True

    read_script = pycharmm.script.CommandScript('read',
                                                coor='CARD',
                                                name=filename,
                                                **cmd_kwargs,
                                                **kwargs)
    read_script.run()


def sequence_coor(filename, resid=False, segi=None, **kwargs):
    """Read sequence from a CHARMM coordinate (CRD) file

    Reads residue sequence from a CHARMM format coordinate file. Residue
    numbers are ignored except that when a change occurs, a new residue
    is added.

    Parameters
    ----------
    filename : str
        Path to the CHARMM coordinate card file
    resid : bool
        If True, obtain residue IDs from the resid field of the coordinate
        file instead of sequential numbering. (default: False)
    segi : str, optional
        Segment ID to filter. If provided, only residues belonging to
        this segment ID will be read. (default: None)
    **kwargs
        Additional keyword arguments passed to CommandScript.

    Examples
    --------
    >>> import pycharmm.read as read
    >>> read.sequence_coor('structure.crd')
    >>> read.sequence_coor('structure.crd', resid=True)
    >>> read.sequence_coor('multi_seg.crd', segi='PROT')
    """
    cmd_kwargs = {}
    if resid:
        cmd_kwargs['resid'] = True
    if segi is not None:
        cmd_kwargs['segi'] = segi

    read_sequence_coor = pycharmm.script.CommandScript('read',
                                                       sequence='coor',
                                                       name=filename,
                                                       **cmd_kwargs,
                                                       **kwargs)
    read_sequence_coor.run()


# read a topology file given a bytes filename
# @functools.singledispatch
# def rtf(rtf_name, append=False):
#     """read a topology file from disk given a path

#     Parameters
#     ----------
#     rtf_name : str or bytes
#                path to a topology file to be read from disk
#     append : bool
#              append topology to existing data

#     Returns
#     -------
#     err_code : int
#                one indicates success, any other value indicates failure
#     """
#     rtf_name_len = ctypes.c_int(len(rtf_name))
#     rtf_name = ctypes.create_string_buffer(rtf_name)

#     if append:
#         append = 1
#     else:
#         append = 0

#     err_code = lib.read_rtf_file(rtf_name,
#                                         ctypes.byref(rtf_name_len),
#                                         ctypes.byref(ctypes.c_int(append)))
#     return err_code


# # read a topology file given a str filename
# @rtf.register(str)
# def _(rtf_name, append=False):
#     rtf_name = rtf_name.encode()
#     err_code = rtf(rtf_name, append)
#     return err_code


# # read a flexible parameter file given a bytes filename
# @functools.singledispatch
# def prm(prm_name, append=False, flex=False):
#     """read a parameter file from disk given a path

#     Parameters
#     ----------
#     prm_name : str or bytes
#                path to a parameter file to be read from disk
#     append : bool
#              append parameters to existing data
#     flex : bool
#            read a flexible format paramter file

#     Returns
#     -------
#     err_code : int
#                one indicates success, any other value indicates failure
#     """
#     prm_name_len = ctypes.c_int(len(prm_name))
#     prm_name = ctypes.create_string_buffer(prm_name)

#     if append:
#         append = 1
#     else:
#         append = 0

#     if flex:
#         flex = 1
#     else:
#         flex = 0

#     err_code = lib.read_param_file(prm_name,
#                                           ctypes.byref(prm_name_len),
#                                           ctypes.byref(ctypes.c_int(append)),
#                                           ctypes.byref(ctypes.c_int(flex)))
#     return err_code


# # read a flexible parameter file given a str filename
# @prm.register(str)
# def _(prm_name, append=False, flex=False):
#     prm_name = prm_name.encode()
#     err_code = prm(prm_name, append, flex)
#     return err_code


# @functools.singledispatch
# def psf_card(filename, append=False, xplor=False):
#     """read a psf card file from disk given a path

#     Parameters
#     ----------
#     filename : str or bytes
#                path to a psf card file to be read from disk
#     append : bool
#              append atoms to psf or replace
#     xplor : bool
#             read an XPLOR format file

#     Returns
#     -------
#     status : bool
#              True indicates success
#     """
#     fn_len = ctypes.c_int(len(filename))
#     c_filename = ctypes.create_string_buffer(filename)

#     c_append = ctypes.c_int(append)
#     c_xplor = ctypes.c_int(xplor)

#     status = lib.read_psf_card(c_filename,
#                                       ctypes.byref(fn_len),
#                                       ctypes.byref(c_append),
#                                       ctypes.byref(c_xplor))

#     status = bool(status)
#     return status


# # read a psf card file
# @psf_card.register(str)
# def _(filename, append=False, xplor=False):
#     filename = filename.encode()
#     status = psf_card(filename, append, xplor)
#     return status


# @functools.singledispatch
# def pdb(filename, resid=False):
#     """read a pdb file from disk given a path

#     Parameters
#     ----------
#     filename : str or bytes
#                path to a psf card file to be read from disk
#     resid : bool

#     Returns
#     -------
#     status : bool
#              True indicates success
#     """
#     fn_len = ctypes.c_int(len(filename))
#     c_filename = ctypes.create_string_buffer(filename)
#     c_resid = ctypes.c_int(resid)
#     status = lib.read_pdb(c_filename, ctypes.byref(fn_len),
#                                  ctypes.byref(c_resid))
#     status = bool(status)
#     return status


# @pdb.register(str)
# def _(filename, resid=False):
#     filename = filename.encode()
#     status = pdb(filename, resid)
#     return status
