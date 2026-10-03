# pyCHARMM: molecular dynamics in python with CHARMM
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

"""Functions to write coordinate and psf files to disk

Corresponds to CHARMM command `WRITe`

See CHARMM documentation [WRITe](<https://academiccharmm.org/documentation/version/c47b1/io#Write>)
for more information

Functions
=========
- `coor_pdb` -- write a coordinate file in pdb format to disk
- `coor_card` -- write a CHARMM coordinate file to disk
- `psf_card` -- write a psf file to disk

Examples
========
>>> import pycharmm.write as write

Write PSF to a file named protein.psf
>>> write.psf_card('protein.psf')

Write coordinates to a PDB file named out.pdb
>>> write.coor_pdb('out.pdb')

Write coordinates to a CHARMM coordinate file named out.crd
>>> write.coor_card('out.crd')
"""

import ctypes

import pycharmm.script
from pycharmm.charmm_file import c_api_path_buffer



def _resolve_write_path(filename: str):
    try:
        from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path
    except ImportError:
        return filename, None
    return charmm_fortran_path(filename, for_write=True)


def coor_pdb(filename, title='', comp=False, official=False, model=0,
             first=False, last=False, **kwargs):
    """Write coordinates to a PDB file via the KEY_LIBRARY C API.

    Extra script-only options are ignored. Library builds do not link ``write``.
    """
    del title, official, model, first, last, kwargs
    import pycharmm

    comparison = bool(comp)
    fortran_path, alias = _resolve_write_path(filename)
    try:
        selection = pycharmm.SelectAtoms().all_atoms()
        buf, fn_len = c_api_path_buffer(fortran_path)
        c_comp = ctypes.c_int(1 if comparison else 0)
        from pycharmm.loader import lib
        status = int(
            lib.write_coor_pdb(
                buf,
                ctypes.byref(fn_len),
                selection.as_ctypes(),
                ctypes.byref(c_comp),
            )
        )
        if status != 1:
            raise RuntimeError(
                f"write_coor_pdb failed for {filename!r} "
                f"(staging={fortran_path!r}, status={status})"
            )
    finally:
        if alias is not None:
            alias.finalize()


def coor_mmcif(filename, title='', comp=False, model=0, **kwargs):
    """Write coordinates to an mmCIF/PDBx file."""
    for option in ('first', 'last', 'label'):
        if option in kwargs:
            raise ValueError(f"Unsupported mmCIF write option: {option}")
    cmd_kwargs = {}
    if comp:
        cmd_kwargs['comp'] = True
    if model != 0:
        cmd_kwargs['model'] = model

    write_command = pycharmm.script.WriteScript(filename,
                                                title,
                                                coor='mmcif',
                                                **cmd_kwargs,
                                                **kwargs)
    write_command.run()


def coor_pdbx(filename, **kwargs):
    """Alias for :func:`coor_mmcif`."""
    return coor_mmcif(filename, **kwargs)


def coor_card(filename, title='', comp=False, offset=0, **kwargs):
    """Write a CHARMM card coordinate file.

    Prefers mmml's coordinate writer when that package is installed. Otherwise
    uses the script path with an uppercase ``CARD`` token.
    """
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            write_charmm_crd_from_charmm,
        )
    except ImportError:
        pass
    else:
        write_charmm_crd_from_charmm(filename, title=title or "COORD")
        return

    fortran_path, alias = _resolve_write_path(filename)
    try:
        cmd_kwargs = {}
        if comp:
            cmd_kwargs['comp'] = True
        if offset != 0:
            cmd_kwargs['offset'] = offset
        write_command = pycharmm.script.WriteScript(
            fortran_path,
            title,
            coor='CARD',
            **cmd_kwargs,
            **kwargs,
        )
        write_command.run()
    finally:
        if alias is not None:
            alias.finalize()


def psf_card(filename, title='', xplor=False, **kwargs):
    """Write a PSF card via the KEY_LIBRARY C API.

    ``title`` and ``xplor`` are accepted for call-site compatibility.
    """
    del title, xplor, kwargs
    from pycharmm.loader import lib

    fortran_path, alias = _resolve_write_path(filename)
    try:
        buf, fn_len = c_api_path_buffer(fortran_path)
        status = int(lib.write_psf_card(buf, ctypes.byref(fn_len)))
        if status != 1:
            raise RuntimeError(
                f"write_psf_card failed for {filename!r} "
                f"(staging={fortran_path!r}, status={status})"
            )
    finally:
        if alias is not None:
            alias.finalize()


