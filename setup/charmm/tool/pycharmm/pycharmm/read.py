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
import pycharmm.lib as lib
import pycharmm.script


# read a topology file given a bytes filename
def rtf(filename, **kwargs):
    """Read a topology file

    Parameters
    ----------
    filename : str
    **kwargs
        additional keyword arguments
    """
    rtf_script = pycharmm.script.CommandScript('read',
                                               rtf='card',
                                               name=filename,
                                               **kwargs)
    rtf_script.run()


def prm(filename, **kwargs):
    """
    Read a parameter file

    Parameters
    ----------
    filename : str
    """
    prm_script = pycharmm.script.CommandScript('read',
                                               param='card',
                                               name=filename,
                                               **kwargs)
    prm_script.run()


def psf_card(filename, **kwargs):
    """
    Read a psf card file from disk given a path.

    Parameters
    ----------
    filename : str
    """
    psf_script = pycharmm.script.CommandScript('read',
                                               psf='card',
                                               name=filename,
                                               **kwargs)
    psf_script.run()


def pdb(filename, **kwargs):
    """
    Read a PDB file from disk given a path

    Parameters
    ----------
    filename : str

    """
    pdb_script = pycharmm.script.CommandScript('read',
                                               coor='pdb',
                                               name=filename,
                                               **kwargs)
    pdb_script.run()


def stream(filename, **kwargs):
    """
    Read a stream file from disk given a path

    Parameters
    ----------
    filename : str
    """
    stream_command = 'stream '+str(filename)
    stream_script = pycharmm.script.CommandScript(stream_command,
                                               **kwargs)
    stream_script.run()


def sequence_pdb(filename, **kwargs):
    """Read a PDB file from disk given a path

    Parameters
    ----------
    filename : str

    Returns
    -------
    bool
        True indicates successful reading
    """
    read_sequence_pdb = pycharmm.script.CommandScript('read',
                                                sequence='pdb',
                                                name=filename,
                                                **kwargs)
    read_sequence_pdb.run()


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
    err_code = lib.charmm.read_sequence_string(seq_str,
                                               ctypes.byref(seq_len))
    return err_code


def coor_card(filename, **kwargs):
    """Read a CHARMM format coordinate file

    Parameters
    ----------
    filename : str

    """
    read_script = pycharmm.script.CommandScript('read',
                                                coor='card',
                                                name=filename,
                                                **kwargs)
    read_script.run()


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

#     err_code = lib.charmm.read_rtf_file(rtf_name,
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

#     err_code = lib.charmm.read_param_file(prm_name,
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

#     status = lib.charmm.read_psf_card(c_filename,
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
#     status = lib.charmm.read_pdb(c_filename, ctypes.byref(fn_len),
#                                  ctypes.byref(c_resid))
#     status = bool(status)
#     return status


# @pdb.register(str)
# def _(filename, resid=False):
#     filename = filename.encode()
#     status = pdb(filename, resid)
#     return status
