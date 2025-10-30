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

"""Get data from CHARMM associated with 'psf' files

Functions
=========
- `get_natom` -- get the current total number of atoms from CHARMM
- `get_nres` -- get the current total number of residues from CHARMM
- `get_nseg` -- get the current total number of segments from CHARMM
- `get_ngrp` -- get the current total number of groups from CHARMM
- `get_iac` -- export a copy of iac (param type codes)
- `get_amass` -- export a copy of amass (atom masses)
- `get_charges` -- export a copy of cg (atom charges)
- `get_ibase` -- export a copy of ibase (last atom of each residue)
- `get_atype` -- export a copy of atype (atom name array)
- `get_res` -- export a copy of res (residue name array)
- `get_resid` -- export a copy of resid (residue identifier array)
- `get_segid` -- export a copy of segid (segment identifier array)
- `get_nictot` -- export a copy of nictot (nres for each seg)
- `get_igpbs` -- export a copy of igpbs (pointer for 1st atom in each group)
- `get_igptyp` -- export a copy of gptyp (code type of each group)

Examples
========
>>> import pycharmm
>>> import pycharmm.psf as psf

Get number of atoms in the system
>>> n = psf.get_natom()

Delete all TIP3 water molecules
>>> water = pycharmm.SelectAtoms(res_name='TIP3')
>>> psf.delete_atoms(water)
"""

import ctypes

import numpy

import pycharmm
import pycharmm.lib as lib
import pycharmm.nbonds
import pycharmm.image
import pycharmm.write
import pycharmm.settings
import pycharmm.scalar
import pycharmm.lingo


def get_natom():
    """Get the current total number of atoms from CHARMM

    Returns
    -------
    n : integer
        current total number of atoms
    """
    n = lib.charmm.psf_get_natom()
    return n


def get_nres():
    """Get the current total number of residues from CHARMM

    Returns
    -------
    n : integer
        current total number of residues
    """
    n = lib.charmm.psf_get_nres()
    return n


def get_nseg():
    """Get the current total number of segments from CHARMM

    Returns
    -------
    n : integer
        current total number of segments
    """
    n = lib.charmm.psf_get_nseg()
    return n


def get_ngrp():
    """Get the current total number of groups from CHARMM

    Returns
    -------
    n : integer
        current total number of groups
    """
    n = lib.charmm.psf_get_ngrp()
    return n


def get_nbond():
    """Get the total number of bonds from CHARMM

    Returns
    -------
    n : integer
        current total number of bonds
    """

    n = lib.charmm.psf_get_nbond()
    return n


def get_iac():
    """Get an array of the param type code for each atom

    Returns
    -------
    iac : integer list
          param type code for each atom
    """
    n = get_natom()
    if n <= 0:
        return list()

    iac = (ctypes.c_int * n)(0)
    lib.charmm.psf_get_iac(iac)
    iac = [i - 1 for i in iac]
    return iac


def get_amass():
    """Get an array of the mass of each atom

    Returns
    -------
    amass : double list
            mass of each atom
    """
    n = get_natom()
    if n <= 0:
        return list()

    amass = (ctypes.c_double * n)(0)
    lib.charmm.psf_get_amass(amass)
    return list(amass)


def get_charges():
    """Get an array of atom charges

    Returns
    -------
    charges : double list
              charge of each atom
    """
    n = get_natom()
    if n <= 0:
        return list()

    charges = (ctypes.c_double * n)(0)
    lib.charmm.psf_get_charges(charges)
    return list(charges)


def get_ibase():
    """Get array of last atom index of each residue

    Returns
    -------
    ibase : integer list
            last atom index of each residue
    """
    n = get_nres()
    ibase = (ctypes.c_int * (n + 1))(0)
    lib.charmm.psf_get_ibase(ibase)
    return ibase


def get_atype():
    """Get list of IUPAC atom names

    Returns
    -------
    atype : string list
            IUPAC name for each atom
    """
    n = get_natom()
    if n <= 0:
        return list()

    atype_buffers = [ctypes.create_string_buffer(8) for _ in range(n)]
    atype_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                atype_buffers))

    lib.charmm.psf_get_atype(atype_pointers)
    atype = [b.value.decode(errors='ignore') for b in atype_buffers[0:n]]
    return atype


def get_res():
    """Get list of residue names

    Returns
    -------
    res : string list
          name for each residue
    """
    n = get_nres()
    if n <= 0:
        return list()

    res_buffers = [ctypes.create_string_buffer(8) for _ in range(n)]
    res_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof, res_buffers))

    lib.charmm.psf_get_res(res_pointers)
    # res = [b[:].decode().strip() for b in res_buffers[0:n]]
    res = [b.value.decode(errors='ignore') for b in res_buffers]
    return res


def get_res_idx(res_name):
    """Get the index of the first residue with name res_name

    Parameters
    ----------
    res_name : string
               name to look for in res list

    Returns
    -------
    i : integer
        if res_name found, index of residue in res

        if res_name not found, 0
    """
    res_names = get_res()
    try:
        i = res_names.index(res_name)
    except ValueError:
        i = 0

    return i


def get_resid():
    """Get list of residue IDs

    Returns
    -------
    resid : string list
           identifier for each residue
    """
    n = get_nres()
    if n <= 0:
        return list()

    resid_buffers = [ctypes.create_string_buffer(8) for _ in range(n)]
    resid_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                resid_buffers))

    lib.charmm.psf_get_resid(resid_pointers)
    resid = [b.value.decode(errors='ignore') for b in resid_buffers]
    return resid


def get_segid():
    """Get list of segment IDs

    Returns
    -------
    segid : string list
           identifier for each segment
    """
    n = get_nseg()
    if n <= 0:
        return list()

    segid_buffers = [ctypes.create_string_buffer(8) for _ in range(n)]
    segid_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                segid_buffers))

    lib.charmm.psf_get_segid(segid_pointers)
    segid = [b.value.decode(errors='ignore') for b in segid_buffers[0:n]]
    return segid


def get_nictot():
    """Get number of residues for each segment

    Returns
    -------
    nictot : integer list
             number of residues for each segment
    """
    n = get_nseg()
    nictot = (ctypes.c_int * (n + 1))(0)
    lib.charmm.psf_get_nictot(nictot)
    return nictot


def get_igpbs():
    """Get first atom in each group

    Returns
    -------
    igpbs : integer list
            first atom in each group
    """
    n = get_ngrp()
    igpbs = (ctypes.c_int * (n + 1))(0)
    lib.charmm.psf_get_igpbs(igpbs)
    return igpbs


def get_igptyp():
    """Get code type of each group

    Returns
    -------
    igptyp : integer list
             code type of each group
    """
    n = get_ngrp()
    if n <= 0:
        return list()

    igptyp = (ctypes.c_int * n)(0)
    lib.charmm.psf_get_igptyp(igptyp)
    return igptyp


def get_ib_jb():
    """Get index of 1st and 2nd atom of each bond

    Returns
    -------
    ib, jb: list of integers
            index of 1st (ib) and 2nd (jb) atoms that make bonds
    """
    n = get_nbond()
    if n <= 0:
        return list()

    # print(f'psf.py, get_ib_jb, n = {n}')
    ib = (ctypes.c_int * n)(0)
    jb = (ctypes.c_int * n)(0)

    lib.charmm.psf_get_ib(ib)
    lib.charmm.psf_get_jb(jb)
    return list(ib), list(jb)


def delete_atoms(selection=None, psort=False):
    """Delete a selection of atoms from the psf

    Parameters
    ----------
    selection : pycharmm.SelectAtoms
                selection[i] == True <=> atom i is selected
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    if not selection:
        selection = pycharmm.SelectAtoms().all_atoms()

    sel_c = selection.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_atoms(sel_c, ctypes.byref(isort))

    status = bool(status)
    return status


def delete_bonds(iselect, jselect, psort=False):
    """Delete bonds between two selections from the psf

    If atom i and atom j have a bond, and
    iselect(i) is 1 and jselect(j) is 1,
    then the bond is deleted

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_bonds(isel_c, jsel_c, ctypes.byref(isort))

    status = bool(status)
    return status


def delete_angles(iselect, jselect, psort=False):
    """Delete angles between two selections from the psf

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_angles(isel_c, jsel_c, ctypes.byref(isort))

    status = bool(status)
    return status


def delete_dihedrals(iselect, jselect, psort=False):
    """Delete dihedrals between two selections from the psf

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_dihedrals(isel_c, jsel_c,
                                             ctypes.byref(isort))

    status = bool(status)
    return status


def delete_impropers(iselect, jselect, psort=False):
    """Delete impropers between two selections from the psf

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_impropers(isel_c, jsel_c,
                                             ctypes.byref(isort))

    status = bool(status)
    return status


def delete_cmaps(iselect, jselect, psort=False):
    """Delete cmaps between two selections from the psf

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_cmaps(isel_c, jsel_c, ctypes.byref(isort))

    status = bool(status)
    return status


def delete_connectivity(iselect, jselect, psort=False):
    """Delete all connectivity between two selections from the psf

    includes bonds, angles, dihedrals, impropers, and cmaps

    Parameters
    ----------
    iselect : pycharmm.SelectAtoms
    jselect : pycharmm.SelectAtoms
    psort : boolean
            True <=> sort psf after deleted atoms are mapped out

    Returns
    -------
    status : bool
             true if successful
    """
    isel_c = iselect.as_ctypes()
    jsel_c = jselect.as_ctypes()
    isort = ctypes.c_int(psort)

    status = lib.charmm.psf_delete_conn(isel_c, jsel_c, ctypes.byref(isort))

    status = bool(status)
    return status


# Addition Kai Toepfer May 2022
def get_nnb():
    """Get the current total number of non-bonded exclusions

    Returns
    -------
    n : integer
        current number of non-bonded exclusions
    """
    n = lib.charmm.psf_get_nnb()
    return n


def set_charge(new_charges):
    """Set a new atom charge array

    Parameters
    ----------
    new_charges : int list(natom)
        list of new charges
    """
    n = get_natom()
    if n <= 0:
        return list()

    charges = (ctypes.c_double * n)(*new_charges)

    lib.charmm.psf_set_charges(charges)

    return


def get_iblo_inb():
    """Get non-bonded exclusion list

    Returns
    -------
    inb : int list
          non-bonded exclusion list
    """
    natom = get_natom()
    nnb = get_nnb()
    if natom <= 0 or nnb <= 0:
        return list()

    iblo = (ctypes.c_int * natom)()
    inb = (ctypes.c_int * nnb)()
    lib.charmm.psf_get_iblo_inb(iblo, inb)
    return list(iblo), list(inb)


def set_iblo_inb(new_inblo, new_inb):
    """Set non-bonded exclusion list

    Parameters
    ----------
    new_inblo : int list(natom)
                INBLO list
    new_inb : int list(nnb)
                INB list

    Returns
    -------
    nnb : int
          number of atom pairs in non-bonded exclusion list
    """
    natom = get_natom()
    if natom > 0:
        iblo = (ctypes.c_int * natom)(*new_inblo)
    else:
        iblo = list()

    nnb = len(new_inb)
    if nnb > 0:
        inb = (ctypes.c_int * nnb)(*new_inb)
    else:
        inb = list()

    #nnb = ctypes.c_int(nnb)  # Segmentation fault for single c_int parameter
    nnb = (ctypes.c_int * 1)(nnb)  # c_int array with one element: nnb
    lib.charmm.psf_set_iblo_inb(nnb, iblo, inb)

    nnb = get_nnb()
    pycharmm.nbonds.update_bnbnd()
    pycharmm.image.update_bimag()

    return nnb


def set_hmr(seg_ids=[], newpsf=''):
    """
    This function will create hydrogen mass repartitioning on
    requested segments for use in enhnaced sampling. Normally,
    all segments not comprising water or non-hydrogen
    containing ions would be passed as segments for HMR

    Parameters
    ----------
    seg_ids: list, default []
             a comma separated list of strings of segments requested for HMR
    newpsf: string, default ''
            a string containing the mass-adjusted psf.
    Returns
    -------
    status : bool
             true if successful
    """
    # Find hydrogen and heavy atoms in selected segids
    selected_segs = pycharmm.SelectAtoms()
    for seg_id in seg_ids:
        selected_segs |= pycharmm.SelectAtoms(seg_id=seg_id.upper())

    hydrogens = selected_segs & pycharmm.SelectAtoms(hydrogens=True)
    heavy_atoms = selected_segs & ~hydrogens
    masses = numpy.array(pycharmm.scalar.get_masses())
    ov = pycharmm.settings.set_verbosity(0)
    wl = pycharmm.settings.set_warn_level(-5)
    mass_incr = numpy.zeros(get_natom())
    for indx in numpy.arange(0, get_natom())[list(heavy_atoms)]:
        pycharmm.lingo.charmm_script(f'define ha_list select hydrogen .and. .bonded. bynu {indx+1}  end')
        nsel = pycharmm.lingo.get_energy_value('NSEL')
        if nsel > 0:
            mass_incr[indx] += nsel * 2

    pycharmm.settings.set_verbosity(ov)
    pycharmm.settings.set_warn_level(wl)
    masses[list(hydrogens)] += 2
    masses -= mass_incr
    pycharmm.scalar.set_masses(masses)
    if len(newpsf) > 0:
        pycharmm.write.psf_card(newpsf)

    return True
