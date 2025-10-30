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

import pycharmm.script


def coor_pdb(filename, title='', **kwargs):
    """write a coordinate set to a pdb file

    Parameters
    ----------
    filename : str
        new file path to write
    title: str
        title to write at the beginning of the file
    **kwargs: dict
        extra settings to pass to the CHARMM command
    """
    write_command = pycharmm.script.WriteScript(filename,
                                                title,
                                                coor='pdb',
                                                **kwargs)
    write_command.run()


def coor_card(filename, title='', **kwargs):
    """write a coordinate set to a CHARMM card *.chr format file

    Parameters
    ----------
    filename: str 
        new file path to write
    title :  str 
        title to write at the beginning of the file
    **kwargs: dict 
        extra settings to pass to the CHARMM command
    """
    write_command = pycharmm.script.WriteScript(filename,
                                                title,
                                                coor='card',
                                                **kwargs)
    write_command.run()


def psf_card(filename, title='', **kwargs):
    """write psf details in card format to a file

    Parameters
    ----------
    filename: str
        new file path to write
    title: str
        title to write at the beginning of the file
    **kwargs: dict
        extra settings to pass to the CHARMM command
    """
    write_command = pycharmm.script.WriteScript(filename,
                                                title,
                                                psf='card',
                                                **kwargs)
    write_command.run()


# def coor_pdb(filename, selection=None, comparison=False):
#     """write a coordinate set to a pdb file

#     Parameters
#     ----------
#     filename : string
#                new file path to write
#     selection : SelectAtoms
#                 a selection of atom indexes to write
#     comparison : bool
#                  if True, write from comparison set instead of main set

#     Returns
#     -------
#     status : bool
#              true if successful
#     """
#     if selection is None:
#         selection = pycharmm.SelectAtoms().all_atoms()

#     fn = ctypes.c_char_p(filename.encode())
#     len_fn = ctypes.c_int(len(filename))
#     c_comp = ctypes.c_int(comparison)

#     status = lib.charmm.write_coor_pdb(fn, ctypes.byref(len_fn),
#                                        selection.as_ctypes(),
#                                        ctypes.byref(c_comp))
#     status = bool(status)
#     return status


# def psf_card(filename):
#     """write psf details in card format to a file

#     Parameters
#     ----------
#     filename : string
#                new file path to write

#     Returns
#     -------
#     status : bool
#              true if successful
#     """
#     fn = ctypes.c_char_p(filename.encode())
#     len_fn = ctypes.c_int(len(filename))
#     status = lib.charmm.write_psf_card(fn, ctypes.byref(len_fn))
#     status = bool(status)
#     return status
