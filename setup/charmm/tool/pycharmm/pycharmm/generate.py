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

Functions
=========
- `new_segment` -- Generate a new segment in the PSF
- `setup_seg` -- Setup a segment using simplified interface
- `patch` -- Apply a patch to modify the PSF
- `rename` -- Rename segid, resid, resname, or atom in the PSF
- `join` -- Join two adjacent segments
- `replica` -- Replicate part of current PSF
- `autogenerate` -- Control autogeneration of angles and dihedrals

Examples
========
>>> import pycharmm
>>> import pycharmm.generate as gen

Generate a segment called `ADP` with terminal patches
>>> gen.new_segment('ADP', 'ACE', 'CT3', setup_ic=True)

Generate a water segment (no angles/dihedrals)
>>> gen.new_segment('WT00', angle=False, dihedral=False)

Apply a disulfide bridge patch
>>> gen.patch('DISU', 'PROA 5, PROA 20', setup=True)

Rename a segment
>>> gen.rename('SEGID', 'PROA', selection=pycharmm.SelectAtoms(seg_id='PROT'))

Join two segments
>>> gen.join('SEG1', 'SEG2', renumber=True)

Enable autogeneration of angles
>>> gen.autogenerate(angles=True, dihedrals=False)

"""

import ctypes

# import pycharmm.loader as lib
from pycharmm.loader import lib
import pycharmm.script
import pycharmm.atom_info as atom_info


def _invalidate_cache():
    """Invalidate atom cache after PSF modifications."""
    atom_info.invalidate_atom_cache()


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

    autot = lib.generate_get_autot()
    autod = lib.generate_get_autod()

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

    err_code = lib.generate_segment(ctypes.byref(opts))
    _invalidate_cache()
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
    err_code = lib.generate_setup(seg_str,
                                         ctypes.byref(ctypes.c_int(seg_len)))
    _invalidate_cache()
    return err_code


def patch(name, patch_sites, setup=False, warn=False, sort=False,
          angle=None, dihedral=None, **kwargs):
    """Apply a patch to modify the PSF

    This function applies patches to modify residues in the current PSF.
    Patches can add or remove atoms, bonds, angles, dihedrals, and improper
    torsions. Common uses include disulfide bridges, protonation state changes,
    and terminal modifications.

    Parameters
    ----------
    name : str
        Name of the patch residue (PRES) to apply from the topology file.
    patch_sites : str
        Comma-separated string of segid/resid pairs specifying which
        residues to patch.
        Format: "segid1 resid1 [, segid2 resid2 [, ... [, segid9 resid9]]]"
    setup : bool
        If True, append IC table entries from the patch to the main IC table.
        (default: False)
    warn : bool
        If True, list elements deleted due to nonexistent atoms.
        (default: False)
    sort : bool
        If True, sort PSF arrays after patching. (default: False)
    angle : bool or None
        If True, enable angle autogeneration. If False, disable (NOANGLE).
        If None, use default from topology file. (default: None)
    dihedral : bool or None
        If True, enable dihedral autogeneration. If False, disable (NODIHEDRAL).
        If None, use default from topology file. (default: None)
    **kwargs : dict
        Additional settings to pass to the CHARMM command.

    Returns
    -------
    bool
        True indicates success.

    Notes
    -----
    The patch command modifies PSF, coordinates, comparison coordinates,
    harmonic constraints, fixed atom list, and internal coordinates.
    However, NBONDS, HBONDS, SHAKE, and DYNAMICS are NOT mapped.
    Atom numbers may change after patching.

    Examples
    --------
    >>> import pycharmm.generate as gen

    Apply disulfide bridge between CYS residues
    >>> gen.patch('DISU', 'PROA 5, PROA 20', setup=True)

    Apply terminal patch with warnings
    >>> gen.patch('ACE', 'PROT 1', warn=True)

    Apply patch with autogeneration control
    >>> gen.patch('MYMOD', 'SEG1 10', angle=True, dihedral=False)
    """
    cmd_kwargs = {}
    if setup:
        cmd_kwargs['setup'] = True
    if warn:
        cmd_kwargs['warn'] = True
    if sort:
        cmd_kwargs['sort'] = True
    if angle is True:
        cmd_kwargs['angle'] = True
    elif angle is False:
        cmd_kwargs['noangle'] = True
    if dihedral is True:
        cmd_kwargs['dihedral'] = True
    elif dihedral is False:
        cmd_kwargs['nodihedral'] = True

    patch_command = 'patch ' + str(name)
    patch_command += ' ' + str(patch_sites)
    patch_script = pycharmm.script.CommandScript(patch_command,
                                                 **cmd_kwargs,
                                                 **kwargs)
    result = patch_script.run()
    _invalidate_cache()
    return result


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
    result = join_script.run()
    _invalidate_cache()
    return result


def replica(selection=None,
            segid='', nreplica=1,
            setup=False, comp=False, reset=False):
    """Replicate part of current PSF

    This function produces multiple (nreplica) copies of the selected part of
    the current PSF.

    Parameters
    ----------
    selection : pycharmm.SelectAtoms
        Selection of atoms comprising atoms to be replicated.
    segid : str
        Base name for the replicated segments.
        Segments will be named base_name1 ... base_nameN up to N = nreplica.
    nreplica : int
        Number of replica copies to make. (default: 1)
    setup : bool
        If True, setup IC tables for replicated segments. (default: False)
    comp : bool
        If True, use comparison coordinate values for replicated segment atoms.
        (default: False)
    reset : bool
        If True, turn off exclusions between replicated atoms. (default: False)

    Returns
    -------
    bool
        True indicates success.

    Examples
    --------
    >>> import pycharmm
    >>> import pycharmm.generate as gen
    >>> sel = pycharmm.SelectAtoms(seg_id='PROT')
    >>> gen.replica(selection=sel, segid='REP', nreplica=3, setup=True)
    """
    replica_command = ' '.join(['replica', segid])
    replica_script = pycharmm.script.CommandScript(replica_command,
                                                   selection=selection,
                                                   nreplica=nreplica,
                                                   setup=setup,
                                                   comp=comp,
                                                   reset=reset)
    result = replica_script.run()
    _invalidate_cache()
    return result


def autogenerate(angles=None, dihedrals=None, patch_mode=None,
                 selection=None, on=False, off=False, **kwargs):
    """Control autogeneration of angles and/or dihedrals

    This function controls the automatic generation of angles and dihedrals
    based on bond connectivity. It can also set autogeneration flags for
    specific atoms.

    Parameters
    ----------
    angles : bool or None
        If True, regenerate all angles based on connectivity.
        If False, disable angle autogeneration (NOANGLE).
        If None, do not modify angle autogeneration. (default: None)
    dihedrals : bool or None
        If True, regenerate all dihedrals based on connectivity.
        If False, disable dihedral autogeneration (NODIHEDRAL).
        If None, do not modify dihedral autogeneration. (default: None)
    patch_mode : bool or None
        If True (PATCH), activate autogeneration for patches.
        If False (NOPATCH), suppress autogeneration when patching.
        If None, do not modify patch mode. (default: None)
    selection : pycharmm.SelectAtoms or None
        Atom selection for ON/OFF operations. Required if on or off is True.
        (default: None)
    on : bool
        If True, clear autogeneration flags (bits 32 and 64) for selected atoms,
        allowing autogeneration to modify angles/dihedrals for these atoms.
        (default: False)
    off : bool
        If True, set autogeneration flags (bits 32 and 64) for selected atoms,
        preventing autogeneration from modifying angles/dihedrals for these atoms.
        (default: False)
    **kwargs : dict
        Additional settings including:
        - drude: bool - Enable Drude particle autogeneration
        - nodrude: bool - Disable Drude particle autogeneration
        - pcheck: bool - Enable parameter checking
        - nopcheck: bool - Disable parameter checking

    Returns
    -------
    bool
        True indicates success.

    Notes
    -----
    The ANGLes and DIHEdrals options delete all current angles/dihedrals
    and regenerate lists based on connectivity. This may be needed after
    patching or other PSF modifications.

    Atom autogeneration flags are bit-coded:
    - Bit 1: Delete angles where atom is central (J position)
    - Bit 2: Delete angles where atom is non-central (I or K position)
    - Bit 4: Delete dihedrals where atom is central (J or K position)
    - Bit 8: Delete dihedrals where atom is non-central (I or L position)
    - Bit 16: Skip parameter checking for this atom
    - Bit 32: Prevent angle modification by autogen
    - Bit 64: Prevent dihedral modification by autogen

    Examples
    --------
    >>> import pycharmm
    >>> import pycharmm.generate as gen

    Regenerate all angles and dihedrals
    >>> gen.autogenerate(angles=True, dihedrals=True)

    Enable autogeneration for patches
    >>> gen.autogenerate(patch_mode=True)

    Disable autogeneration for specific atoms
    >>> sel = pycharmm.SelectAtoms(atom_type='LP*')
    >>> gen.autogenerate(selection=sel, off=True)
    """
    cmd_kwargs = {}
    cmd_parts = ['autogenerate']

    if on:
        cmd_parts.append('on')
    elif off:
        cmd_parts.append('off')
    else:
        if angles is True:
            cmd_parts.append('angles')
        elif angles is False:
            cmd_parts.append('noangles')
        if dihedrals is True:
            cmd_parts.append('dihedrals')
        elif dihedrals is False:
            cmd_parts.append('nodihedrals')
        if patch_mode is True:
            cmd_parts.append('patch')
        elif patch_mode is False:
            cmd_parts.append('nopatch')

    auto_command = ' '.join(cmd_parts)
    auto_script = pycharmm.script.CommandScript(auto_command,
                                                selection=selection,
                                                **cmd_kwargs,
                                                **kwargs)
    result = auto_script.run()
    # Autogenerate can modify PSF angles/dihedrals, invalidate cache
    _invalidate_cache()
    return result

