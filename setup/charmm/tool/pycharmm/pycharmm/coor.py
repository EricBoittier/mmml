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
  
"""Functions for manipulating coordinates of atoms.  
  
Corresponds to CHARMM module `corman` for Coordinate manipulation and analyses.    
See [corman documentation](https://academiccharmm.org/documentation/version/c47b1/corman)  
  
"""  
  
import ctypes  
  
import pandas  
import numpy as np
  
import pycharmm  
import pycharmm.lib as lib  
  
def get_natom():
    """Returns the number of atoms currently in the simulation

    Returns
    -------
    natom : int
        number of atoms currently in the simulation
    """
    natom = lib.charmm.coor_get_natom()
    return natom


# --- ctypes <-> numpy plumbing -------------------------------------------------
#
# These accessors used to box 3*natom (or 4*natom) Python floats per call, via
# list comprehensions on the way out and .tolist() on the way in. That dominated
# the cost of a hybrid MM/ML step: reading forces cost several times more than the
# ENER FORCE evaluation that produced them, and the cost grew linearly with natom.
# Moving the same bytes through numpy keeps the public DataFrame API unchanged and
# is bitwise identical, but avoids the per-element Python objects.

def _out_buffers(natom, count):
    """Allocate `count` C double buffers of length natom."""
    return [(ctypes.c_double * natom)() for _ in range(count)]


def _as_np(cbuf):
    """Copy a C double buffer into a numpy array (no per-element Python floats)."""
    return np.array(np.ctypeslib.as_array(cbuf), dtype=np.float64, copy=True)


def _in_buffer(column, natom):
    """Build a C double buffer from a DataFrame column / array-like.

    Fills a fresh buffer rather than aliasing the input: pandas hands out read-only
    views, which ctypes.from_buffer rejects. The copy is done in C by numpy, so this
    still avoids materialising natom Python floats.
    """
    arr = np.asarray(column, dtype=np.float64)[0:natom]
    cbuf = (ctypes.c_double * natom)()
    np.ctypeslib.as_array(cbuf)[:] = arr
    return cbuf, arr


  
def set_positions(pos):  
    """Sets the positions of the atoms in the simulation  
  
    Parameters
    ----------
    pos : pandas.core.frame.DataFrame  
        a dataframe with columns named x, y and z  

    Returns
    -------
    natom : int  
        number of atoms in the simulation  
    """  
    natom = get_natom()
    c_x, _bx = _in_buffer(pos['x'], natom)
    c_y, _by = _in_buffer(pos['y'], natom)
    c_z, _bz = _in_buffer(pos['z'], natom)

    natom = lib.charmm.coor_set_positions(c_x, c_y, c_z)
    return natom
  

def get_positions():  
    """Gets the positions of the atoms in the simulation  
  
    Returns
    -------
    pos : pandas.core.frame.DataFrame  
        a dataframe with columns named *x*, *y* and *z*  
    """  
    natom = get_natom()
    c_x, c_y, c_z = _out_buffers(natom, 3)

    lib.charmm.coor_get_positions(c_x, c_y, c_z)

    pos = pandas.DataFrame({'x': _as_np(c_x),
                            'y': _as_np(c_y),
                            'z': _as_np(c_z)})
    return pos
  
  
def set_weights(weights):  
    """Sets the weights of the atoms in the simulation  
  
    Parameters
    ----------
    weights : list[float]  
        list of one weight for each atom 0:natom  
  
    Returns
    -------
    natom : int    
        number of atoms in the simulation  
    """  
    natom = get_natom()
    c_weights, _bw = _in_buffer(weights, natom)
    natom = lib.charmm.coor_set_weights(c_weights)
    return natom
  
  
def get_weights():  
    """Gets the weights of the atoms in the simulation  
  
    Returns
    -------
    weights : list[float]  
        a list of float atom weights 0:natom  
    """  
    natom = get_natom()
    (c_weights,) = _out_buffers(natom, 1)
    lib.charmm.coor_get_weights(c_weights)

    # Public API is a list; .tolist() converts in C rather than element by element.
    return _as_np(c_weights).tolist()
  
  
def copy_forces(mass=False, selection=None):  
    """Copy the forces to the comparison set  
  
    Parameters
    ----------
    mass : bool, default = False    
        Whether to weight the forces by mass  
    selection : pycharmm.SelectAtoms    
        only copy forces of selected atoms  
  
    Returns
    -------
    bool  
        true if successful  
    """  
    if not selection:  
        selection = pycharmm.SelectAtoms().all_atoms()  
  
    c_sel = selection.as_ctypes()  
    c_mass = ctypes.c_int(mass)  
  
    status = lib.charmm.coor_copy_forces(ctypes.byref(c_mass), c_sel)  
  
    status = bool(status)  
    return status  
  
  
def set_forces(forces):  
    """Sets the forces of the atoms  
  
    Parameters
    ----------
    forces : pandas.core.frame.DataFrame    
        a dataframe with columns named dx, dy and dz  

    Returns
    -------
    natom : int  
        number of atoms in the simulation  
    """  
    natom = get_natom()
    c_dx, _bx = _in_buffer(forces['dx'], natom)
    c_dy, _by = _in_buffer(forces['dy'], natom)
    c_dz, _bz = _in_buffer(forces['dz'], natom)

    natom = lib.charmm.coor_set_forces(c_dx, c_dy, c_dz)
    return natom
  
  
def get_forces():  
    """Gets the forces of the atoms  
  
    Returns
    -------
    forces : pandas.core.frame.DataFrame  
        a dataframe with columns named *dx*, *dy* and *dz*  
    """  
    natom = get_natom()
    c_dx, c_dy, c_dz = _out_buffers(natom, 3)

    lib.charmm.coor_get_forces(c_dx, c_dy, c_dz)

    forces = pandas.DataFrame({'dx': _as_np(c_dx),
                               'dy': _as_np(c_dy),
                               'dz': _as_np(c_dz)})
    return forces
  
  
def set_comparison(pos):  
    """Set the comparison set in CHARMM  
  
    Parameters
    ----------
    pos : pandas.core.frame.DataFrame   
        a dataframe with columns named *x*, *y*, *z*, and *w*  
      
    Returns
    -------
    natom : int  
        number of atoms in the simulation  
    """  
    natom = get_natom()
    x, _bx = _in_buffer(pos['x'], natom)
    y, _by = _in_buffer(pos['y'], natom)
    z, _bz = _in_buffer(pos['z'], natom)
    w, _bw = _in_buffer(pos['w'], natom)

    natom = lib.charmm.coor_set_comparison(x, y, z, w)
    return natom
  
  
def get_comparison():  
    """Gets the comparison of the atoms  
  
    Returns
    -------
    pandas.core.frame.DataFrame    
        dataframe with columns named *x*, *y*, *z*, and *w*  
    """  
    natom = get_natom()  
    x = (ctypes.c_double * natom)()  
    y = (ctypes.c_double * natom)()  
    z = (ctypes.c_double * natom)()  
    w = (ctypes.c_double * natom)()  
  
    lib.charmm.coor_get_comparison(x, y, z, w)

    pos = pandas.DataFrame({'x': _as_np(x),
                            'y': _as_np(y),
                            'z': _as_np(z),
                            'w': _as_np(w)})
    return pos
  
  
def set_comp2(pos):  
    """Set the comp2 set in CHARMM  
  
    Parameters
    ----------
    pos : pandas.core.frame.DataFrame  
        a dataframe with columns named *x*, *y*, *z*, and *w*  

    Returns
    -------
    natom : int  
        number of atoms in the simulation  
    """  
    natom = get_natom()
    x, _bx = _in_buffer(pos['x'], natom)
    y, _by = _in_buffer(pos['y'], natom)
    z, _bz = _in_buffer(pos['z'], natom)
    w, _bw = _in_buffer(pos['w'], natom)

    natom = lib.charmm.coor_set_comp2(x, y, z, w)
    return natom
  
  
def get_comp2():  
    """Gets the comp2 of the atoms  
  
    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*  
    """  
    natom = get_natom()  
    x = (ctypes.c_double * natom)()  
    y = (ctypes.c_double * natom)()  
    z = (ctypes.c_double * natom)()  
    w = (ctypes.c_double * natom)()  
  
    lib.charmm.coor_get_comp2(x, y, z, w)

    pos = pandas.DataFrame({'x': _as_np(x),
                            'y': _as_np(y),
                            'z': _as_np(z),
                            'w': _as_np(w)})
    return pos
  
  
def set_main(pos):  
    """Set the main set in CHARMM  
  
    Parameters
    ----------
    pos : pandas.core.frame.DataFrame    
        a dataframe with columns named *x*, *y*, *z*, and *w*  

    Returns
    -------
    natom : int  
        number of atoms in the simulation  
    """  
    natom = get_natom()
    x, _bx = _in_buffer(pos['x'], natom)
    y, _by = _in_buffer(pos['y'], natom)
    z, _bz = _in_buffer(pos['z'], natom)
    w, _bw = _in_buffer(pos['w'], natom)

    natom = lib.charmm.coor_set_main(x, y, z, w)
    return natom
  
  
def get_main():  
    """Gets the MAIN of the atoms  
  
    Returns
    -------
    pandas.core.frame.DataFrame  
        dataframe with columns named *x*, *y*, *z*, and *w*  
    """  
    natom = get_natom()  
    x = (ctypes.c_double * natom)()  
    y = (ctypes.c_double * natom)()  
    z = (ctypes.c_double * natom)()  
    w = (ctypes.c_double * natom)()  
  
    lib.charmm.coor_get_main(x, y, z, w)

    pos = pandas.DataFrame({'x': _as_np(x),
                            'y': _as_np(y),
                            'z': _as_np(z),
                            'w': _as_np(w)})
    return pos
  
  
# def orient(by_mass, by_rms, by_noro):  
#     """modifies coordinates of all atoms according to the passed flags  
  
#     The select set of atoms is first centered about the origin,  
#     and then rotated to either align with the axis,  
#     or the other coordinate set.  
#     The RMS keyword will use the other coordinate set as a rotation reference.  
#     The MASS keyword cause a mass weighting to be done. This will  
#     align the specified atoms along their moments of inertia. When the RMS  
#     keyword is not used, then the structure is rotated so that its principle  
#     geometric axis coincides with the X-axis and the next largest coincides  
#     with the Y-axis. This command is primarily used for preparing a  
#     structure for graphics and viewing. It can also be used for finding  
#     RMS differences, and in conjunction with the vibrational analysis.  
#         The NOROtation keyword will suppress rotations. In this case,  
#     only one coordinate set will be modified.  
  
#     Parameters
#     ----------
#     by_mass : bool  
#               if true, causes a mass weighting of coordinates  
#     by_rms : bool  
#              if true, will use the other coordinate set as a rotation reference  
#     by_noro : bool  
#               if true, supress rotations; only one coordinate set is modified  
  
#     Returns
#     -------
#     success : bool  
#               True indicates success, any other value indicates failure  
#     """  
#     by_mass_flag = ctypes.c_int(by_mass)  
#     by_rms_flag = ctypes.c_int(by_rms)  
#     by_noro_flag = ctypes.c_int(by_noro)  
  
#     success = lib.charmm.coor_orient(ctypes.byref(by_mass_flag),  
#                                      ctypes.byref(by_rms_flag),  
#                                      ctypes.byref(by_noro_flag))  
#     success = bool(success)  
#     return success  
  
  
def orient(**kwargs):  
    """Modifies coordinates of all atoms according to the passed flags  
  
    The select set of atoms is first centered about the origin,  
    and then rotated to either align with the axis,  
    or the other coordinate set.  
  
    The *RMS* keyword will use the other coordinate set as a rotation reference.  
  
    The *MASS* keyword cause a mass weighting to be done. This will  
    align the specified atoms along their moments of inertia. When the RMS  
    keyword is not used, then the structure is rotated so that its principle  
    geometric axis coincides with the X-axis and the next largest coincides  
    with the Y-axis. This command is primarily used for preparing a  
    structure for graphics and viewing. It can also be used for finding  
    RMS differences, and in conjunction with the vibrational analysis.  
  
    The *NORO*tation keyword will suppress rotations. In this case,  
    only one coordinate set will be modified.  
  
    Parameters
    ----------
    **kwargs : dictionary  
        key words for the coor orient CHARMM command  
    """  
    oscript = pycharmm.script.CommandScript('coor', orient=True, **kwargs)  
    oscript.run()  
  
  
def show_comp():  
    """Print the comparison set of the atoms  
  

    Returns
    -------
    bool  
        true if successful  
    """  
    status = lib.charmm.coor_print_comp()  
    status = bool(status)  
    return status  
  
  
def show():  
    """Print the main coordinate set of the atoms  
  
    Returns
    -------
    bool  
        true if successful  
    """  
    status = lib.charmm.coor_print()  
    bool(status)  
    return status  
  
  
def stat(selection=None, comp=False, mass=False):  
    """Computes *max*, *min*, *ave* for `x`, `y`, `z`, `w` over selection of main or comp sets  
  
    Parameters
    ----------
    selection : pycharmm.SelectAtoms    
        a selection of atom indexes to use for stats  
    comp : bool    
        if true, stats computed for comparison set selection  
    mass : bool    
        if true, will place the average values at the center of mass  

    Returns
    -------
    dict
        A python dictionary with keys   
        xmin, xmax, xave,   
        ymin, ymax, yave,   
        zmin, zmax, zave,   
        wmin, wmax, wave,   
        n_selected, n_misses  
    """  
    c_comp = ctypes.c_int(comp)  
    c_mass = ctypes.c_int(mass)  
  
    if selection is None:  
        selection = pycharmm.SelectAtoms().all_atoms()  
  
    c_sel = selection.as_ctypes()  
  
    n_stats = 13  
    max_label = 5  
  
    vals = (ctypes.c_double * n_stats)()  
  
    labels_bufs = [ctypes.create_string_buffer(max_label)  
                   for _ in range(n_stats)]  
    labels_ptrs = (ctypes.c_char_p * n_stats)(*map(ctypes.addressof,  
                                                   labels_bufs))  
  
    n_sel = ctypes.c_int(0)  
    n_misses = ctypes.c_int(0)  
  
    lib.charmm.coor_stat(c_sel,  
                         ctypes.byref(c_comp), ctypes.byref(c_mass),  
                         labels_ptrs, vals,  
                         ctypes.byref(n_sel), ctypes.byref(n_misses))  
  
    labels_str = [label.value.decode(errors='ignore')  
                  for label in labels_bufs[0:n_stats]]  
    labels = [label.strip().lower() for label in labels_str]  
    labels = [label for label in labels if label]  
  
    stats = dict(zip(labels, vals))  
    stats['n_selected'] = n_sel.value  
    stats['n_misses'] = n_misses.value  
    return stats  
  
# not compat with python 3.7  
# CoordSet = typing.Literal['main', 'comp', 'comp2']  
def qexclt(i, j):
    """

    Returns a integer label based on the presence of 
    atom with index `i` and `j` in the exclusion list.

    *For dev or internal use only*
    Parameters
    ----------
    i, j: int
          Atom indices

    Returns
    -------
    is14exclt: int
              A value of 1, 2 or 3 is returned 
              based on what exclusion list the input pair is in.
    """
    
    # Converting 0-indexed i, j to 1-indexed values
    # that is compatible with CHARMM arrays
    i += 1
    j += 1

    ctypes.c_int(i)
    ctypes.c_int(j)

    is14exclt = lib.charmm.qexclt(i, j)
    return is14exclt
  

def dist(selection1, selection2 = None, cutoff = None, resi = True, omit_14excl = True, omit_nonbonds = False, omit_excl = True ):
    """
    Performs equivalent of CHARMM's `COOR DIST` operation. It can find distances
    between atoms within a selection or find distances between atoms in two selections.
    Returns `None` if no contacts (within specified cutoff value) is found.

    Parameters
    ----------
    selection1: pycharmm.Selection
    selection2: pycharmm.Selection, default = None
                If nothing is eplicitly provided, selection1 is copied into it.
    cutoff: float
            Determine contacts within this value (in Angstroms)
    resi : bool, default = True
           Calculate residue-to-residue minimum distance between two selections.
    omit_14excl: bool, default True
                 Same as `NO14exclusions` in COOR DIST command
    omit_nonbonds: bool, default False
                   Same as `NONOnbonds` in COOR DIST command
    omit_excl: bool ,default True
               Same as `NOEXclusions` in COOR DIST command
           

    Return
    ------
    contacts : pandas.DataFrame
               If no contacts are found, returns `None`. Otherwise a dataframe whose
               columns are named - 'atomid_1','segid_1','resn_1','resid_1','type_1',
               'atom_2', 'segid_2', 'resn_2','resid_2','type_2', 'distance'
    """
    contactDict = {}
    
    from pycharmm import psf
    from pycharmm import atom_info as atom
    from copy import copy

    # handle faulty inputs
    if cutoff is None:
        raise ValueError("FATAL ERROR: cutoff value unspecified.")
        
    if selection2 is None:
        print("<COOR DIST> Copying selection1 into selection2 because none was provided.")
        selection2 = copy(selection1)
    
    # kinds of exclusions requested
    qexcl = [ omit_excl, omit_14excl, omit_nonbonds ]
    # counter for pairs respective exclusion lists
    numexcl = [0, 0, 0] 

    # fetch resids and segids in the system
    psf.get_resid()
    psf.get_segid()
    
    # number of atoms in the selections
    insel = sum(selection1.get_selection()) 
    jnsel = sum(selection2.get_selection())
    if insel == 0 or jnsel == 0:
        raise ValueError(f"FATAL ERROR: Selection with no atoms found. selection1 has {insel} atoms, selection2 has {jnsel} atoms.")
    else:
        print(f"<COOR DIST> {insel} atoms in selection1")
        print(f"<COOR DIST> {jnsel} atoms in selection2")

    # atom indices, resids, segids for the selections
    iatom = [i for i, qselected in enumerate(selection1.get_selection()) if qselected]
    iresi = atom.get_res_ids(iatom)
    isegi = atom.get_seg_ids(iatom) 
    
    jatom = [i for i, qselected in enumerate(selection2.get_selection()) if qselected]
    jresi = atom.get_res_ids(jatom)
    jsegi = atom.get_seg_ids(jatom)
    
    # coordinates
    coords = get_positions()
    ixyz = coords[pandas.Series(selection1.get_selection())]
    jxyz = coords[pandas.Series(selection2.get_selection())]
    
    if resi:
        # start identifying pairs of residues for minimum distance calculation
        idx = 0
        icount = 0
        while idx < (insel-1):
            icurr_resi = iresi[idx]
            icurr_segi = isegi[idx]
            #print(f"IDX = {idx:8d}: icurr_resi = {icurr_resi} and icurr_segi = {icurr_segi}")
        
            idx1 = idx + 1
            inatoms = 1
            while idx1 < insel:
                if iresi[idx1] != icurr_resi or isegi[idx1] != icurr_segi:
                    #print(f"{iresi[idx1]} != {icurr_resi} and {isegi[idx1]} != {icurr_segi}")
                    break
                else:
                    #print(f"Else resi = {iresi[idx1]} and segi = {isegi[idx1]}, leading to idx1 = {idx1 + 1}")
                    inatoms += 1
                    idx1 += 1
            icurr_indices = idx + np.arange(inatoms)
        
            jdx = 0
            while jdx < (jnsel-1):
                jcurr_resi = jresi[jdx]
                jcurr_segi = jsegi[jdx]
        
                jdx1 = jdx + 1
                jnatoms = 1
                while jdx1 < jnsel:
                    if jresi[jdx1] != jcurr_resi or jsegi[jdx1] != jcurr_segi:
                        break
                    else:
                        jnatoms += 1
                        jdx1 += 1
                jcurr_indices = jdx + np.arange(jnatoms)
        
                ## find the distances between atoms in icurr_resi and jcurr_resi
                ijDistMtx = np.ndarray((inatoms, jnatoms))
                for ii, idx2 in enumerate(icurr_indices):
                    for jj, jdx2 in enumerate(jcurr_indices):
                        dx = ixyz.iloc[idx2].x - jxyz.iloc[jdx2].x
                        dy = ixyz.iloc[idx2].y - jxyz.iloc[jdx2].y
                        dz = ixyz.iloc[idx2].z - jxyz.iloc[jdx2].z

                        if (abs(dx) > cutoff) or (abs(dy) > cutoff) or (abs(dz) > cutoff):
                            ijDistMtx[ii, jj] = 9999.0 # absurdly high
                        else:
                            ijDistMtx[ii, jj] = np.sqrt(dx*dx + dy*dy + dz*dz)
        
                
                # find minimum distance index
                imin = int(np.argmin(ijDistMtx)/jnatoms)
                jmin = np.argmin(ijDistMtx) % jnatoms
        
                if ijDistMtx[imin, jmin] <= cutoff:
                    iminAtomId = iatom[icurr_indices[imin]]
                    iminResId = iresi[icurr_indices[imin]].strip()
                    iminSegId = isegi[icurr_indices[imin]].strip()
                    iminResn = atom.get_res_names([iminAtomId])[0].strip()
                    iminAtype = atom.get_atom_types([iminAtomId])[0].strip()
                    
                    jminAtomId = jatom[jcurr_indices[jmin]]
                    jminResId = jresi[jcurr_indices[jmin]].strip()
                    jminSegId = jsegi[jcurr_indices[jmin]].strip()
                    jminResn = atom.get_res_names([jminAtomId])[0].strip()
                    jminAtype = atom.get_atom_types([jminAtomId])[0].strip()
        
                    contactDict[icount] = [iminAtomId, iminSegId, iminResn, iminResId, iminAtype,\
                                           jminAtomId, jminSegId, jminResn, jminResId, jminAtype,
                                           ijDistMtx[imin, jmin]]
                    
                    # print(f"{iminAtomId:6d} \
                    # {iminSegId:8s} \
                    # {iminResn:8s} \
                    # {iminResId:6s} \
                    # {iminAtype:8s} \
                    # -- \
                    # {jminAtomId:6d} \
                    # {jminSegId:8s} \
                    # {jminResn:8s} \
                    # {jminResId:8s} \
                    # {jminAtype:6s} \
                    # {ijDistMtx[imin, jmin]:12.4f}")
        
                    icount += 1
                
                jdx += jnatoms # while jdx < (jnsel-1)
            idx += inatoms  # while idx < (insel-1)

    else:
        icount = 0
        for ii in range(insel):
            idx = iatom[ii]
            if not pycharmm.select.is_initial(idx):
                continue
            for jj in range(jnsel):
                jdx = jatom[jj]
                if not pycharmm.select.is_initial(jdx):
                    continue

                dx = ixyz.iloc[ii].x - jxyz.iloc[jj].x
                dy = ixyz.iloc[ii].y - jxyz.iloc[jj].y
                dz = ixyz.iloc[ii].z - jxyz.iloc[jj].z


                if (abs(dx) > cutoff) or (abs(dy) > cutoff) or (abs(dz) > cutoff):
                    continue # omit because distance > cutoff
                
                else:
                    tmpDist2 = dx*dx + dy*dy + dz*dz

                    if tmpDist2 <= cutoff*cutoff:
                        iexcl = qexclt(idx, jdx)
                        numexcl[iexcl-1] += 1
                        if qexcl[iexcl-1]:
                            continue #  omit based on exclusion criteria requested

                        iAtomId = idx
                        iResId = iresi[ii].strip()
                        iSegId = isegi[ii].strip()
                        iResn = atom.get_res_names([iAtomId])[0].strip()
                        iAtype = atom.get_atom_types([iAtomId])[0].strip()
                    
                        jAtomId = jdx
                        jResId = jresi[jj].strip()
                        jSegId = jsegi[jj].strip()
                        jResn = atom.get_res_names([jAtomId])[0].strip()
                        jAtype = atom.get_atom_types([jAtomId])[0].strip()
        
                        contactDict[icount] = [iAtomId, iSegId, iResn, iResId, iAtype,\
                                               jAtomId, jSegId, jResn, jResId, jAtype,
                                               np.sqrt(tmpDist2)]

                        
                        #print(f"{iAtomId:6d} \
                        #{iSegId:8s} \
                        #{iResn:8s} \
                        #{iResId:6s} \
                        #{iAtype:8s} \
                        #-- \
                        #{jAtomId:6d} \
                        #{jSegId:8s} \
                        #{jResn:8s} \
                        #{jResId:8s} \
                        #{jAtype:6s} \
                        #{tmpDist:12.4f}")
                        icount += 1

        # print pair counts in different exclusion list
        # to be consistent with CHARMM output with COOR DIST
        print(f"<COOR DIST> Total Exclusion count ={numexcl[0]:10d}")
        print(f"<COOR DIST> Total 1-4 exclusions  ={numexcl[1]:10d}")
        print(f"<COOR DIST> Total non-exclusions  ={numexcl[2]:10d}")
    
    # END if RESI else 

    contacts = pandas.DataFrame.from_dict(contactDict, orient = 'index')
    contacts.columns = ['atomid_1','segid_1','resn_1','resid_1','type_1',\
                         'atom_2', 'segid_2', 'resn_2','resid_2','type_2',\
                         'distance']
    return contacts

  
class Coordinates:  
    __slots__ = 'which_set', 'coords', 'are_dirty'  
  
    def __init__(self, coord_set):  
        self.which_set = coord_set  
        self.coords = pandas.DataFrame(columns=['x', 'y', 'z', 'w'])  
        self.are_dirty = True  
        self.pull()  
  
    def pull(self):  
        if self.which_set == 'main':  
            self.coords = get_main()  
        elif self.which_set == 'comp':  
            self.coords = get_comparison()  
        elif self.which_set == 'comp2':  
            self.coords = get_comp2()  
        else:  
            msg = '{} is not a valid coordinate set'.format(self.which_set)  
            raise ValueError(msg)  
  
        self.are_dirty = False  
        return self  
  
    def push(self):  
        if self.which_set == 'main':  
            set_main(self.coords)  
        elif self.which_set == 'comp':  
            set_comparison(self.coords)  
        elif self.which_set == 'comp2':  
            set_comp2(self.coords)  
        else:  
            msg = '{} is not a valid coordinate set'.format(self.which_set)  
            raise ValueError(msg)  
  
        self.are_dirty = False  
        return self  
  
    def __len__(self):  
        return len(self.coords)  
