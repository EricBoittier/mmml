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

"""Functions to construct and manipulate the PSF

The central data structure in CHARMM, the PSF holds lists giving every
bond, bond angle, torsion angle, and improper torsion angle
as well as information needed to generate the hydrogen bonds and
the non-bonded list.

Corresponds to CHARMM module `struct`
See [struct documentation](https://academiccharmm.org/documentation/version/c47b1/struct#Top)

Examples
========
>>> import pycharmm
>>> import pycharmm.generate as gen

Generate a segment called `ADP` and apply `ACE` and `CT3` patches on the N- and C- terminals
>>> gen.new_segment('ADP', 'ACE', 'CT3', setup_ic=True)

Generate a segment called `WT00` and *DO NOT* rewrite angles and dihedrals for molecules in the segment
>>> gen.new_segment('WT00', angle=False, dihedral=False)

"""

import ctypes

import pycharmm.lib as lib
import pycharmm.script


_options = [('new_seg', ctypes.c_char * 9),
            ('dup_seg', ctypes.c_char * 9),
            ('patf', ctypes.c_char * 9),
            ('patl', ctypes.c_char * 9),
            ('ldrude', ctypes.c_int),
            ('lsetic', ctypes.c_int),
            ('lwarn', ctypes.c_int),
            ('lshow', ctypes.c_int),
            ('langle', ctypes.c_int),
            ('lphi', ctypes.c_int),
            ('dmass', ctypes.c_double)]


class OPTIONS(ctypes.Structure):
    """A ctypes structure to hold settings to generate a new segment.

    Attributes are listed by name, type, charmm input script equivalent,
    default value and a short description.

    Attributes
    ----------
    new_seg : str
        name of the new segment
    dup_seg : str
        DUPL name of segment to clone
    patf : str
        FIRS DEFA patch residue name for terminating residue
    patl : str
        LAST DEFA patch residue name for terminating residue
    ldrude : int
        DRUD 0 create drude particles?
    lsetic : int
        SETU 0 append ic table from topo file to main ic table?
    lwarn : int
        WARN 0 list elts deleted due to nonexistent atoms?
    lshow : int
        SHOW 0
    langle : int
        ANGL/NOAN autot overide autogen option from topo file?
    lphi : int
        DIHE/NODI autod overide autogen option from topo file?
    dmass : float
        DMAS 0.0
    """
    _fields_ = _options


def new_segment(seg_name='', first_patch='', last_patch='', **kwargs):
    """Add the next segment to the PSF and name it `seg_name`

    This function will cause any internal coordinate table entries
    (IC) from the topology file to be appended to the main IC table.
    This function uses the sequence of residues specified in the last
    input_sequence function and the information stored in the residue
    topology file to add the next segment to the PSF.

    Each segment contains a list of all the bonds, angles,
    dihedral angles, and improper torsions needed to calculate the energy.

    It also assigns charges to all the atoms, sets up the
    nonbonded exclusions list, and specifies hydrogen bond donors and
    acceptors. Any internal coordinate which references atoms outside
    the range of the segment is deleted. This prevents any
    unexpected bonding of segments.

    Parameters
    ----------
    seg_name : str
               name for the new segment
    first_patch : str
        name of the patch applied to the N-terminal
    last_patch : str
        name of the patch applied to the C-terminal
    **kwargs : optional

    Returns
    -------
    int
               1 indicates success, any other value indicates failure
    """
    seg_name = seg_name.ljust(9)
    first_patch = first_patch.ljust(9)
    last_patch = last_patch.ljust(9)

    opts = OPTIONS(''.encode(), ''.encode(),
                   ''.encode(), ''.encode(),
                   0, 0, 0, 0, 0, 0,
                   0.0)

    # valid_opts = ['setup_ic', 'angle', 'dihedral', 'drude',
    #               'warn', 'show', 'mass']

    autot = lib.charmm.generate_get_autot()
    autod = lib.charmm.generate_get_autod()

    opts.new_seg = seg_name.encode()
    opts.dup_seg = '         '.encode()
    opts.patf = first_patch.encode()
    opts.patl = last_patch.encode()

    opts.langle = autot
    opts.lphi = autod

    types = dict(_options)
    for k, v in kwargs.items():
        if k == 'setup_ic':
            opts.lsetic = types['lsetic'](v)
        elif k == 'angle':
            opts.langle = types['langle'](v)
        elif k == 'dihedral':
            opts.lphi = types['lphi'](v)
        elif k == 'drude':
            opts.ldrude = types['ldrude'](v)
        elif k == 'warn':
            opts.lwarn = types['lwarn'](v)
        elif k == 'show':
            opts.lshow = types['lshow'](v)
        elif k == 'mass':
            opts.dmass = types['dmass'](v)

    err_code = lib.charmm.generate_segment(ctypes.byref(opts))
    return err_code


def setup_seg(seg):
    """Add the next segment to the PSF and name it `seg`

    This function will cause any internal coordinate table entries
    (IC) from the topology file to be appended to the main IC table.
    This function uses the sequence of residues specified in the last
    input_sequence function and the information stored in the residue
    topology file to add the next segment to the PSF. Each segment contains a
    list of all the bonds, angles, dihedral angles, and improper torsions
    needed to calculate the energy. It also assigns charges to all the
    atoms, sets up the nonbonded exclusions list, and specifies hydrogen
    bond donors and acceptors. Any internal coordinate which references
    atoms outside the range of the segment is deleted. This prevents any
    unexpected bonding of segments.

    Parameters
    ----------
    seg : str
          name for the new segment

    Returns
    -------
    int
               1 indicates success, any other value indicates failure
    """
    seg_len = len(seg)
    seg_str = ctypes.create_string_buffer(seg.encode())
    err_code = lib.charmm.generate_setup(seg_str,
                                         ctypes.byref(ctypes.c_int(seg_len)))
    return err_code


def patch(name, patch_sites, **kwargs):
    """Patch a segment in the PSF with `name`

    This function applies patches to the sequence.

    Parameters
    ----------
    name : str
           name for the patch to apply
    patch_sites : str
                  comma separated string of pairs
                  segid1 resid1 [, segid2 resid2 [, ... [,segid 9 resid 9]...]
    **kwargs : [sort=bool] [awtup=bool] [warn=bool]

    Returns
    -------
    bool
        True indicates success
    """
    patch_command = 'patch '+str(name)
    patch_command += ' '+str(patch_sites)
    patch_script = pycharmm.script.CommandScript(patch_command, **kwargs)
    return patch_script.run()


def rename(to_rename='', new_name='', selection=None):
    """Rename a segid, resid, resn atom in the PSF with `new_name`

    This function renames elements of the current psf, segid, resid, resn or atom.

    Parameters
    ----------
    to_rename : str
           name of the psf element to rename {SEGID} {RESID} {RESN} {ATOM}
    new_name : str
           new name for the element of the psf you wish to change
    selection : pycharmm.SelectAtoms
           selection of atoms to be renamed

    Returns
    -------
    bool
        True indicates success
    """
    if to_rename not in ['SEGID', 'RESID', 'RESN', 'ATOM']:
        message = 'invalid option to_rename = {}'
        raise ValueError(message.format(to_rename))

    if new_name == '':
        message = 'invalid option new_name = {}'
        raise ValueError(message.format(new_name))

    rename_command = f'rename {to_rename} {new_name}'
    rename_script = pycharmm.script.CommandScript(rename_command, selection=selection)
    return rename_script.run()


def join(segid_1, segid_2='', renumber=False):
    """Join two adjacent segments and optionally renumber them.

    This function joins two adjacent segments of the current psf.

    Parameters
    ----------
    segid_1 : str
           name of first segid in the psf involved in the join
    segid_2 : str
           name of second segid in the psf involved in the join
    renumber : bool

    Returns
    -------
    bool
        True indicates success
    """
    join_command = ' '.join(['join', segid_1, segid_2])
    join_script = pycharmm.script.CommandScript(join_command, renumber=renumber)
    return join_script.run()


def replica(selection=None,
            segid='', nreplica=1,
            setup=False, comp=False, reset=False):
    """Replica runs the CHARMM replica command:  replicate part of current PSF

    This function produces multiple (nreplica) copies of the selected part of
    the current psf.

    Parameters
    ----------
    selection : pycharmm.selectAtoms
           selection of atoms comprising atoms to be replicated
    segid : str
           base name for the replicated segments,
           segments will be names base_name1 ... base_nameN
           up to N = nreplica
    nreplica : int
           number of replica copies to make
    setup : bool
           if True setup ic tables for replicated segments
    comp : bool
           if True use comparison coordinate values for replicated segment atoms
    reset : bool
           if True the exclusions between replicated atoms is turned off

    Returns
    -------
    bool
        True indicates success
    """
    replica_command = ' '.join(['replica', segid])
    replica_script = pycharmm.script.CommandScript(replica_command,
                                                   selection=selection,
                                                   nreplica=nreplica,
                                                   setup=setup,
                                                   comp=comp,
                                                   reset=reset)
    return replica_script.run()
