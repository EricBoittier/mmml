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

"""Functions for standard CDOCKER calcualtion

Created by Yujin Wu (wyujin@umich.edu)

See CHARMM documentation [openmm_dock](<https://academiccharmm.org/documentation/version/c47b1>)
and [GRID](<https://academiccharmm.org/documentation/version/c47b1/grid>)
for more information

Functions
=========
- `calc_dihedral` -- calculate dihedral angles for a given four position
- `rand_rot_trans` -- random translate and rotate of the ligand
- `FACTS_rescore` -- FACTS rescoring of docked poses
- `RCDOCKER_init_place` -- rigid CDOCKER original initial placement
- `RCDOCKER_fast_init_place` -- rigid CDOCKER fast initial placement
- `FCDOCKER_init_place` -- flexible CDOCKER initial placement
- `FCDOCKER_fast_init_place` -- flexible CDOCKER fast initial placement
- `FCDOCKER_mutation` -- mutation in the genetic algorithm in flexible CDOCKER
- `FCDOCKER_crossover` -- crossover in the genetic algorithm in flexible CDOCKER
- `FCDOCKER_calc_dihedral` -- calculate dihedral angles in flexible CDOCKER
- `default_ommd_sian` -- default simulated annealing algorithm
- `cluster_mmtsb` -- clustering with mmtsb cluster.pl
- `scan_cluster_radius` -- find best cluster radius
- `rcdocker_default_sort` -- default rigid cdocker sort method
- `top_N_cluster` -- find top N cluster result
- `Rigid_CDOCKER` -- standard rigid CDOCKER docking method
- `Flexible_CDOCKER` -- standard flexible CDOCKER docking method

Examples
========
Import `Rigid_CDOCKER` module to perform rigid CDOCKER using the standard protocol
>>> from pycharmm.cdocker import Rigid_CDOCKER

With default input, users only need to provide grid box information
>>> Rigid_CDOCKER(xcen = 1, ycen = 1, zcen = 1, maxlen = 10)

Import `Flexible_CDOCKER` to perform flexible CDOCKER using the standard protocol
>>> from pycharmm.cdocker import Flexible_CDOCKER

With default input, users only need to provide grid box information and flexible side chain selection
>>> import pandas as pd
>>> flex={'res_id': [84, 87, 99, 111, 118], 'seg_id': ['PROT', 'PROT', 'PROT', 'PROT', 'PROT']}
>>> flexchain = pd.DataFrame(data=flex)
>>> print(flexchain)
   res_id seg_id
0      84   PROT
1      87   PROT
2      99   PROT
3     111   PROT
4     118   PROT
>>> Flexible_CDOCKER(xcen = 1, ycen = 1, zcen = 1, maxlen = 10, flexchain = flexchain)

"""

## Import module
import pycharmm
import pycharmm.psf as psf
import pycharmm.coor as coor
import pycharmm.read as read
import pycharmm.grid as grid
import pycharmm.write as write
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.energy as energy
import pycharmm.cons_fix as cons_fix
import pycharmm.minimize as minimize
import pycharmm.settings as settings
from pycharmm.implicit_solvent import FACTS

import numpy as np
import pandas as pd
from os import listdir, system
from math import pi, exp, atan2
from random import random, choices
from scipy.spatial.transform import Rotation as R

## Helper function
def _mkdir_(fileName):
    """Bash command for removing folder and creating folder

    Parameters
    ----------
    fileName: str
               name for the new folder

    Returns
    -------
    bool: true for success
    """
    cmd = '''rm -rf %s''' % (str(fileName))
    system(cmd)
    cmd = '''mkdir -p %s''' % (str(fileName))
    system(cmd)
    return True

def _rm_(fileName):
    """Bash command for removing files

    Parameters
    ----------
    fileName: str
               name for files

    Returns
    -------
    bool: true for success
    """
    cmd = '''rm -rf %s''' % (str(fileName))
    system(cmd)
    return True

def _cp_(file1, file2):
    """Bash command for copying files

    Parameters
    ----------
    file1: str
            source file name
    file2: str
            destination file name

    Returns
    -------
    bool: true for success
    """
    cmd = '''cp -r %s %s''' % (str(file1), str(file2))
    system(cmd)
    return True

def _mv_(file1, file2):
    """Bash command for moving files

    Parameters
    ----------
    file1: str
            source file name
    file2: str
            destination file name

    Returns
    -------
    bool: true for success
    """
    cmd = '''mv %s %s''' % (str(file1), str(file2))
    system(cmd)
    return True

def _mutate_prob_(DEner):
    """Calculate mutation probability based on energy difference

    Parameters
    ----------
    DEner: float
            energy differnce

    Returns
    -------
    float: mutation probability
    """
    if DEner <= 0:
        return 1 - 0.5 * exp(DEner)
    else:
        return 0.5 * exp(-DEner)

## Flexible docking side dihedral angle and entropy calculation
protein_dihedral = pd.DataFrame()
tmp_aa = ['ARG', 'ARG', 'ARG', 'ARG', 'ASN',
   'ASN', 'ASP', 'ASP', 'CSY', 'GLN', 'GLN',
   'GLN', 'GLU', 'GLU', 'GLU', 'HSD', 'HSD',
   'ILE', 'ILE', 'LEU', 'LEU', 'LYS', 'LYS',
   'LYS', 'LYS', 'MET', 'MET', 'MET', 'PHE',
   'PHE', 'SER', 'THR', 'TRP', 'TRP', 'TYR',
   'TYR', 'VAL']
tmp_sc = ['N CA CB CG', 'CA CB CG CD', 'CB CG CD NE',
   'CG CD NE CZ', 'N CA CB CG', 'CA CB CG OD1', 'N CA CB CG',
   'CA CB CG OD1', 'N CA CB SG', 'N CA CB CG', 'CA CB CG CD',
   'CB CG CD OE1', 'N CA CB CG', 'CA CB CG CD', 'CB CG CD OE1',
   'N CA CB CG', 'CA CB CG ND1', 'N CA CB CG1', 'CA CB CG1 CD',
   'N CA CB CG', 'CA CB CG CD1', 'N CA CB CG', 'CA CB CG CD',
   'CB CG CD CE', 'CG CD CE NZ', 'N CA CB CG', 'CA CB CG SD',
   'CB CG SD CE', 'N CA CB CG', 'CA CB CG CD1', 'N CA CB OG',
   'N CA CB OG1', 'N CA CB CG', 'CA CB CG CD1', 'N CA CB CG',
   'CA CB CG CD1', 'N CA CB CG1']
protein_dihedral['Amino_acid'] = tmp_aa
protein_dihedral['Side_chain_atom'] = tmp_sc

def calc_dihedral(xyz):
    """Calculate dihedral angle entropy contribution

    Parameters
    ----------
    xyz: numpy 4 by 3 array
          4 points in xyz-coordinates

    Returns
    -------
    float
          dihedral angles in degree
    """
    v1 = xyz[1, :] - xyz[0, :]
    v2 = xyz[2, :] - xyz[1, :]
    v3 = xyz[3, :] - xyz[2, :]

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    m1 = np.cross(n1, v2)

    x = n1 @ n2
    y = m1 @ n2

    rad = atan2(y, x)
    return rad / pi * 180

## Random rotation & translation function
def rand_rot_trans(xyz, max_rot = pi, max_trans = 2):
    """Random rotation and translation of the ligand

    Parameters
    ----------
    xyz: np.array
          n by 3 array (i.e., xyz coordinates)
    max_rot: float
              maximum rotation angle (in radian)
    max_trans: float
                maximum translation

    Returns
    -------
    np.array
         new ligand xyz coordinates
    """
    ## Get random rotation angle and translation distance
    tmp = np.random.rand(2, 3) * 2 - 1
    rand_trans = tmp[0, :] * max_trans
    rot_vec = tmp[1, :] / np.linalg.norm(tmp[1, :])
    max_rot = max_rot * (random() * 2 - 1)
    matrix = R.from_rotvec(rot_vec * max_rot).as_matrix()

    ## Random rotation and translation
    ligCenter = (np.amin(xyz, axis = 0) + np.amax(xyz, axis = 0)) / 2
    origin_xyz = xyz - ligCenter
    tmp_xyz = origin_xyz @ matrix.T + ligCenter + rand_trans
    return tmp_xyz

## FACTS implicit solvent rescoring
def FACTS_rescore(fixAtomSel = None, steps = 1000, tolgrd = 0.001):
    """Default FACTS docking rescore method

    Parameters
    ----------
    fixAtomSel : pycharmm.SelectAtoms
                 fixed atoms that undergoes FACTS implicit solvent minimization
    steps : int
            number of minimization steps
    tolgrd : float
             minimization tolerance (exit thresold)

    Returns
    -------
    facts_ener : float
                FACTS implicit solvent energy
    """

    ## Non-bond option
    lingo.charmm_script("faster on")
    update_nonbond = pycharmm.UpdateNonBondedScript(
     nbxmod = 5, atom = True, cdiel = True, eps = 1, shift = True,
     vatom = True, vdistance = True, vswitch = True, cutnb = 14.0,
     ctofnb = 12.0, ctonnb = 10.0, e14fac = 1.0, wmin = 1.5).run()
    lingo.charmm_script("scalar wmain = radius")

    ## FACTS implicit solvent setup
    FACTS(tcps = 22, teps = 1, gamm = 0.015,
     tavw = True, conc = 0.1, temp = 298).run()

    ## Minimiziation
    if fixAtomSel != None: cons_fix.setup(fixAtomSel)
    minimize.run_abnr(nstep = steps, tolgrd = tolgrd)
    if fixAtomSel != None: cons_fix.turn_off()

    ## Get energy
    energy.show()
    facts_ener = energy.get_total()
    return facts_ener

## Grids for fast ligand initial placement in rigid receptor
def _fill_grid(atomCoor, size):
    """Filling in grids

    Parameters
    ----------
    atomCoor : np.array
               n by 3 array (i.e., xyz coordinates)
    size : int
           size of the grid boxes

    Returns
    -------
    numpy 3d array: grid for fast ligand initial placement
    """
    filled_grid = np.zeros([int(size) + 1, int(size) + 1, int(size) + 1])
    filled_grid[atomCoor[:, 0], atomCoor[:, 1], atomCoor[:, 2]] += 1
    atomCoor = atomCoor - 1
    filled_grid[atomCoor[:, 0], atomCoor[:, 1], atomCoor[:, 2]] += 1
    return filled_grid

def _prot_grid(protCoor, expand_limit = None):
    """Filling in protein grids for fast initial placement

    Parameters
    ----------
    protCoor: np.array
               n by 3 array (i.e., xyz coordinates)
    expand_limit: np.array, default = None
               m by 3 array of points (coordinates) that must be accounted for
               in building the grid. `m` is the number of points that must be
               accounted for.

    Returns
    -------
    float size: size of the grid boxes
    numpy 3d array: protein grid for fast ligand initial placement
    """
    minCoor = np.amin(protCoor, axis = 0)
    maxCoor = np.amax(protCoor, axis = 0)
    if expand_limit is None:
        size = np.ceil(np.max(np.ceil(maxCoor) - np.floor(minCoor))) + 2
    elif  np.any(expand_limit is not None):
        minCorner = np.amin([minCoor, np.amin(expand_limit, axis = 0)], axis = 0)
        maxCorner = np.amax([maxCoor, np.amax(expand_limit, axis = 0)], axis = 0)
        size = np.ceil(np.max(np.ceil(maxCorner) - np.floor(minCorner)))
        
    atomCoor = np.floor(protCoor + size / 2).astype(int)
    protGrid = _fill_grid(atomCoor, size)
    return size, protGrid

def _lig_grid(ligCoor, size, protCenter):
    """Filling in ligand grids for fast initial placement

    Parameters
    ----------
    ligCoor: np.array
              n by 3 array (i.e., xyz coordinates)
    size: int
           size of the grid boxes
    protCenter: np.array
                 center of the protein, [Xcenter, Ycenter, Zcenter]

    Returns
    -------
    numpy 3d array: ligand grid for fast ligand initial placement
    """
    atomCoor = np.floor(ligCoor + size / 2 - protCenter).astype(int)
    ligGrid = _fill_grid(atomCoor, size)
    return ligGrid

## Rigid CDOCKER function
## Original initial placement method
## Fast initial placement method

def RCDOCKER_init_place(ligPDB = './ligand.pdb', ligSeg = 'LIGA',
                        hardGridFile = 'grid-emax-3-mine--30-maxe-30.bin',
                        nativeGridFile = 'grid-emax-100-mine--100-maxe-100.bin',
                        confDir = './conformer', placementDir = './placement',
                        flag_center_ligand = True, flag_use_hbond = True,
                        flag_form = False, flag_rdie = True, dielec = 3,
                        xcen = 0, ycen = 0, zcen = 0, numPlace = 100, threshold = 2500):
    """Rigid CDOCKER original initial placement

    Parameters
    ----------
    ligPDB : str
             ligand pdb file name
    ligSeg : str
             ligand segment ID
    hardGridFile : str
                   hard grid file name
    nativeGridFile : str
                     native grid file name
    confDir : str
              conformer folder name
    placementDir : str
                   placement folder name
    flag_center_ligand : bool
                         whether ligand needed to be centered or not
    flag_use_hbond : bool
                     whether hydrogen bond is used in ligand initial placement
    flag_form: bool
                whether or not the grid file is formatted
    flag_rdie: bool
                true for rdie, false for cdie
    dielec: float
             dielectric constant
    xcen : float
           center of the docking pocket
    ycen : float
           center of the docking pocket
    zcen : float
           center of the docking pocket
    numPlace : int
               number of placement for each conformer
    threshold : float
                threshold for energy cutoff

    Returns
    -------
    int
        number of conformers
    numPlace :  int
        number of placement for each conformer
    """
    ## Prepare
    gridCenter = np.asarray([xcen, ycen, zcen])
    if psf.get_natom() > 0 : psf.delete_atoms()
    read.sequence_pdb(ligPDB)
    settings.set_bomb_level(-1)
    gen.new_segment(seg_name = ligSeg)
    settings.set_bomb_level(0)
    read.pdb(ligPDB, resid = True)
    ligand = pycharmm.SelectAtoms().by_seg_id(ligSeg)

    ## Update nonbond interactions
    if flag_rdie:
        update_nonbond = pycharmm.UpdateNonBondedScript(
          atom = True, switch = True, vswitch = True, vdwe = True,
          elec = True, rdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
          emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()
    else:
        update_nonbond = pycharmm.UpdateNonBondedScript(
          atom = True, switch = True, vswitch = True, vdwe = True,
          elec = True, cdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
          emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()

    ## Grid information
    hardSet = {'selection' : ligand, 'flag_form' : flag_form, 'gridFile' :
                hardGridFile, 'flag_grhb' : flag_use_hbond}
    nativeSet = {'selection' : ligand, 'flag_form' : flag_form, 'gridFile' :
                  nativeGridFile, 'flag_grhb': flag_use_hbond}

    ## Loop through conformers
    idxCopy = 1
    _mkdir_(placementDir)
    for conformer in listdir(confDir):
        read.pdb(confDir + conformer, resid = True)
        minimize.run_sd(nstep = 50, tolenr = 0.01)
        minimize.run_abnr(nstep = 50, tolenr = 0.005)
        totalener = energy.get_total() + threshold
        xyz = coor.get_positions().to_numpy()
        if flag_center_ligand :
            ligCenter = (np.amin(xyz, axis = 0) + np.amax(xyz, axis = 0)) / 2
            xyz = xyz - ligCenter + gridCenter
            new_xyz = pd.DataFrame(xyz, columns = ['x', 'y', 'z'])
            coor.set_positions(new_xyz)

        ## Now start placement and save result
        idxnum = 1
        xyz = coor.get_positions().to_numpy()
        while idxnum <= numPlace / 5:

            ## Find initial conformer place
            new_xyz = pd.DataFrame(rand_rot_trans(xyz), columns = ['x', 'y', 'z'])
            coor.set_positions(new_xyz)

            ## Minimize in Grids
            hardGrid = grid.CDOCKER()
            hardGrid.setVar(hardSet)
            hardGrid.read()
            minimize.run_sd(nstep = 100)
            minimize.run_abnr(nstep = 500, tolenr = 0.01)
            hardGrid.off()
            hardGrid.clear()

            nativeGrid = grid.CDOCKER()
            nativeGrid.setVar(nativeSet)
            nativeGrid.read()
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 100, tolenr = 0.01)

            ## Check energy
            energy.show()
            flag_mutate = False
            allEner = energy.get_energy()
            confEner = energy.get_total()
            if confEner <= totalener : flag_mutate = True
            if flag_use_hbond:
                dockhbond = energy.get_eterm(111)
                if dockhbond > 0 : flag_mutate = False
            nativeGrid.off()
            nativeGrid.clear()

            if flag_mutate :
                tmpidx = 1
                randomEner = confEner
                while tmpidx <= 5:
                    ## Random rotation and translation
                    mutate_xyz = pd.DataFrame(rand_rot_trans(new_xyz.to_numpy(), max_rot = pi / 6),
                                              columns = ['x', 'y', 'z'])
                    coor.set_positions(mutate_xyz)

                    ## Minimize in Grids
                    hardGrid = grid.CDOCKER()
                    hardGrid.setVar(hardSet)
                    hardGrid.read()
                    minimize.run_sd(nstep = 100)
                    minimize.run_abnr(nstep = 500, tolenr = 0.01)
                    hardGrid.off()
                    hardGrid.clear()

                    nativeGrid = grid.CDOCKER()
                    nativeGrid.setVar(nativeSet)
                    nativeGrid.read()
                    minimize.run_sd(nstep = 50)
                    minimize.run_abnr(nstep = 100, tolenr = 0.01)

                    ## Check energy difference
                    DEner = energy.get_total() - confEner
                    nativeGrid.off()
                    nativeGrid.clear()
                    prob = _mutate_prob_(DEner)
                    rand = random()

                    if rand <= prob:
                        print("Rigid CDOCKER ligand original placement", str(idxCopy))
                        write.coor_pdb(placementDir + str(idxCopy) + ".pdb", title = '''Title
                        * The conformer pdb is %s
                        * Placement ID is %d and mutate ID is %d'''
                        % (conformer, idxnum, tmpidx))
                        idxCopy += 1
                        tmpidx += 1

                idxnum += 1
    return len(listdir(confDir)), numPlace

def RCDOCKER_fast_init_place(receptorPDB = './protein.pdb', receptorPSF = './protein.psf',
               ligPDB = './ligand.pdb', ligSeg = 'LIGA', confDir = './conformer/',
               placementDir = './placement/',  exhaustiveness = 'high', flag_center_ligand = True,
               xcen = 0, ycen = 0, zcen = 0, maxlen = 10, numPlace = 100):
    """Rigid CDOCKER fast initial placement

    Parameters
    ----------
    receptorPDB: str
              protein pdb file name
    receptorPSF: str
              protein psf file name
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    confDir: str
              conformer folder name
    placementDir: str
                   placement folder name
    exhaustiveness: str
                     exhaustiveness for fast placement, high, medium, low
    flag_center_ligand: bool
                         whether or not ligand need to be centered
    xcen: float
           center of the docking pocket
    ycen: float
           center of the docking pocket
    zcen: float
           center of the docking pocket
    maxlen: float
           size of the grid box
    numPlace: int
               number of placement for each conformer

    Returns
    -------
    int
        number of conformers
    numPlace : int
        number of placement for each cnoformer
    """
    ## Protein grid information
    if psf.get_natom() > 0: psf.delete_atoms()
    read.psf_card(receptorPSF, append = True)
    read.pdb(receptorPDB, resid = True)
    xyz = coor.get_positions().to_numpy()
    protCenter = (np.amin(xyz, axis = 0) + np.amax(xyz, axis = 0)) / 2
    xyz = xyz - protCenter
    
    ### before making the protein grid,
    ### see if the xcen ycen, zcen, maxlen values for the ligand
    ### would require expanded coverage of the protein grid
    ### than the original code allowed (where grid size was decided
    ### purely on the basis of the minmax of the protein coordinates)
    gridCenter = np.asarray([xcen, ycen, zcen])
    ligBox = np.asarray([gridCenter + maxlen, gridCenter - maxlen])
    size, protGrid = _prot_grid(xyz, expand_limit = ligBox)


    ## Build ligand
    psf.delete_atoms()
    read.sequence_pdb(ligPDB)
    settings.set_bomb_level(-1)        ## for 3-mem ring
    gen.new_segment(seg_name = ligSeg)
    settings.set_bomb_level(0)
    read.pdb(ligPDB, resid = True)
    num_atom = psf.get_natom()

    ## Check cutoff, default is high
    if exhaustiveness == 'high':
        cutoff = int(num_atom * 0.1) + 1
    elif exhaustiveness == 'medium':
        cutoff = int(num_atom * 0.2) + 1
    elif exhaustiveness == 'low':
        cutoff = int(num_atom * 0.5) + 1
    else:
        cutoff = int(num_atom * 0.1) + 1

    ## Loop through conformers
    idxCopy = 1
    _mkdir_(placementDir)
    for conformer in listdir(confDir):
        read.pdb(confDir + conformer, resid = True)
        xyz = coor.get_positions().to_numpy()
        if flag_center_ligand:
            ligCenter = (np.amin(xyz, axis = 0) + np.amax(xyz, axis = 0)) / 2
            xyz = xyz - ligCenter + gridCenter
            new_xyz = pd.DataFrame(xyz, columns = ['x', 'y', 'z'])
            coor.set_positions(new_xyz)

        ## Now start placement and save result
        idxnum = 1
        while idxnum <= numPlace / 5:

            ## Find initial conformer place
            tmp_xyz = rand_rot_trans(xyz)
            ligGrid = _lig_grid(tmp_xyz, size, protCenter)
            conformer_score = np.sum(protGrid * ligGrid)
            if conformer_score <= cutoff:
                conformer_xyz = tmp_xyz
                tmpidx = 1

                ## Mutate conformer position, save pose based on probability
                while tmpidx <= 5:
                    tmp_xyz = rand_rot_trans(xyz, max_rot = pi / 6, max_trans = 1)
                    ligGrid = _lig_grid(tmp_xyz, size, protCenter)
                    score = conformer_score - np.sum(protGrid * ligGrid)
                    prob = exp(score)
                    rand = random()

                    if prob > rand:
                        new_xyz = pd.DataFrame(tmp_xyz, columns = ['x', 'y', 'z'])
                        coor.set_positions(new_xyz)
                        print("Ligand placement ", str(idxCopy))
                        write.coor_pdb(placementDir + str(idxCopy) + ".pdb", title = '''Title
                        * The conformer pdb is %s
                        * Placement ID is %d and mutate ID is %d and the score is %d
                        * The exp(score) is %8.6f and cutoff is %8.6f'''
                        % (conformer, idxnum, tmpidx, score, prob, rand))
                        tmpidx += 1
                        idxCopy += 1

                idxnum += 1
    return len(listdir(confDir)), numPlace

##  Flexible CDOCKER functions
## 	Initial placement
## 	Crossover
##	Next generation placement

def FCDOCKER_init_place(receptorCard= './flexchain.crd', receptorSel = None, flexSel = None,
                        ligSel = None, ligPDB = './ligand.pdb', ligSeg = "LIGA",
                        hardGridFile = 'grid-emax-3-mine--30-maxe-30.bin',
                        nativeGridFile = 'grid-emax-100-mine--100-maxe-100.bin',
                        placementDir = './placement/', num = 20, copy = 25,
                        threshold = 2500, flag_form = False):
    """Flexible CDOCKER initial placement

    Parameters
    ----------
    receptorCard: str
                   receptor coordinate card
    receptorSel: str
                  receptor selection
    flexSel: str
              flexible part selection
    ligSel: str
             ligand selection
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    hardGridFile: str
                   hard grid file name
    nativeGridFile: str
                     native grid file name
    placementDir: str
                   placement folder name
    num: int
           number of conformer
    copy: int
           number of copies for each conformer
    threshold: float
                threshold for energy cutoff
    flag_form: bool
                whether or not the grid file is formatted

    Returns
    -------
    int
        number of docking trials
    """
    ## Prepare
    idxNum = 1
    idxCopy = 1
    ligRotamerPDB = './ligand_rotamer.pdb'
    hardSet = {'selection': flexSel, 'flag_form' : flag_form, 'gridFile': hardGridFile}
    nativeSet = {'selection': flexSel, 'flag_form' : flag_form, 'gridFile': nativeGridFile}
    _mkdir_(placementDir)
    cmd = '''
          obrotamer %s | convpdb.pl -segnames -setseg %s > %s
          ''' % (ligPDB, ligSeg, ligRotamerPDB)

    ## Loop and save poses in the save dir
    while idxNum <= num:

        ## Read in random conformer and run brief minimization
        system(cmd)
        read.coor_card(receptorCard, selection = receptorSel)
        read.pdb(ligRotamerPDB, resid = True)
        minimize.run_sd(nstep = 50, tolenr = 0.01)
        minimize.run_abnr(nstep = 50, tolenr = 0.005)
        totalener = threshold + energy.get_total()

        ## Save conformer and side chain conformer
        allCoor = coor.get_positions()
        allCoor['atomIndex'] = np.arange(np.shape(list(ligSel))[0])
        ligand_xyz = allCoor.loc[np.asarray(list(ligSel)), ['x', 'y', 'z']]
        receptor_xyz = allCoor.loc[np.asarray(list(receptorSel)), ['x', 'y', 'z', 'atomIndex']]

        ## Random translation and rotation and minimize in grids.
        i = 1
        while i <= copy:

            ## Random rotation and translation
            new_xyz = pd.DataFrame(rand_rot_trans(ligand_xyz.to_numpy()), columns = ['x', 'y', 'z'])
            new_xyz['atomIndex'] = allCoor.loc[np.asarray(list(ligSel)), 'atomIndex'].to_numpy()
            new_xyz = pd.concat([receptor_xyz, new_xyz], ignore_index = True)
            new_xyz = new_xyz.sort_values(by = ['atomIndex'], ignore_index = True)[['x', 'y', 'z']]
            coor.set_positions(new_xyz)

            ## Minimize in Grids
            hardGrid = grid.CDOCKER()
            hardGrid.setVar(hardSet)
            hardGrid.read()
            minimize.run_sd(nstep = 100)
            minimize.run_abnr(nstep = 500, tolenr = 0.01)
            hardGrid.off()
            hardGrid.clear()

            nativeGrid = grid.CDOCKER()
            nativeGrid.setVar(nativeSet)
            nativeGrid.read()
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 100, tolenr = 0.01)
            nativeGrid.off()
            nativeGrid.clear()

            ## Check energy
            energy.show()
            if energy.get_total() < totalener:
                print("Flexible receptor + ligand placement ", str(idxCopy))
                write.coor_pdb(placementDir + str(idxCopy) + ".pdb", title = '''Title
                * The conformer ID is %d
                * The placement ID is %d
                * The score is %8.6f and cutoff is %8.6f '''
                % (idxNum, i, energy.get_total(), totalener))
                idxCopy += 1
                i += 1

        idxNum += 1
    return num * copy

def FCDOCKER_fast_init_place(receptorPDB='./protein.pdb', receptorPSF='./protein.psf',
                             ligPDB='./ligand.pdb', ligSeg='LIGA', placementDir='./placement/',
                             exhaustiveness='high', num = 20, copy = 25):
    """Rigid CDOCKER fast initial placement

    Parameters
    ----------
    receptorPDB: str
              protein pdb file name
    receptorPSF: str
              protein psf file name
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    placementDir: str
                   placement folder name
    exhaustiveness: str
                     exhaustiveness for fast placement, high, medium, low
    num: int
           number of conformer
    copy: int
           number of copies for each conformer
    Returns
    -------
    int
           number of docking trials
    """
    ## Protein grid information
    if psf.get_natom() > 0: psf.delete_atoms()
    read.psf_card(receptorPSF, append = True)
    read.pdb(receptorPDB, resid = True)
    xyz = coor.get_positions().to_numpy()
    protCenter = (np.amin(xyz, axis = 0) + np.max(xyz, axis = 0)) / 2
    xyz = xyz - protCenter
    size, protGrid = _prot_grid(xyz)

    ## Build ligand
    psf.delete_atoms()
    read.sequence_pdb(ligPDB)
    settings.set_bomb_level(-1)  ## for 3-mem ring
    gen.new_segment(seg_name = ligSeg)
    settings.set_bomb_level(0)
    read.pdb(ligPDB, resid = True)
    num_atom = psf.get_natom()

    ## Prepare
    idxNum = 1
    ligRotamerPDB = './ligand_rotamer.pdb'
    _mkdir_(placementDir)
    cmd = '''
          obrotamer %s | convpdb.pl -segnames -setseg %s > %s
          ''' % (ligPDB, ligSeg, ligRotamerPDB)

    ## Check cutoff, default is high
    if exhaustiveness == 'high':
        cutoff = int(num_atom * 0.1) + 1
    elif exhaustiveness == 'medium':
        cutoff = int(num_atom * 0.2) + 1
    elif exhaustiveness == 'low':
        cutoff = int(num_atom * 0.5) + 1
    else:
        cutoff = int(num_atom * 0.1) + 1

    ## Loop through conformers
    ommCopy = 1
    while idxNum <= num:

        ## Read in random conformer
        system(cmd)
        read.pdb(ligRotamerPDB, resid = True)
        xyz = coor.get_positions().to_numpy()

        ## Now start placement and save result
        idxCopy = 1
        while idxCopy <= copy / 5:

            ## Find initial conformer place
            tmp_xyz = rand_rot_trans(xyz)
            ligGrid = _lig_grid(tmp_xyz, size, protCenter)
            conformer_score = np.sum(protGrid * ligGrid)
            if conformer_score <= cutoff:
                conformer_xyz = tmp_xyz
                tmpidx = 1

                ## Mutate conformer position, save pose based on probability
                while tmpidx <= 5:
                    tmp_xyz = rand_rot_trans(xyz, max_rot = pi / 6, max_trans = 1)
                    ligGrid = _lig_grid(tmp_xyz, size, protCenter)
                    score = conformer_score - np.sum(protGrid * ligGrid)
                    prob = exp(score)
                    rand = random()

                    if prob > rand:
                        new_xyz = pd.DataFrame(tmp_xyz, columns = ['x', 'y', 'z'])
                        coor.set_positions(new_xyz)
                        print("Ligand placement ", str(ommCopy))
                        write.coor_pdb(placementDir + str(ommCopy) + ".pdb", title='''Title
                        * The conformer ID is %d
                        * Placement ID is %d and mutate ID is %d and the score is %d
                        * The exp(score) is %8.6f and cutoff is %8.6f'''
                        % (idxNum, ommCopy, tmpidx, score, prob, rand))
                        tmpidx += 1
                        ommCopy += 1

                idxCopy += 1
        idxNum += 1
    return num * copy

def FCDOCKER_mutation(final_result, pair_list, receptorSel = None, flexSel = None,
                      ligSel = None, ligPDB = './ligand.pdb', ligSeg = "LIGA",
                      saveLig = './ligand/', saveProt = './protein/',
                      crossoverLig = './r_ligand/', crossoverProt = './r_protein/',
                      hardGridFile = 'grid-emax-3-mine--30-maxe-30.bin',
                      nativeGridFile = 'grid-emax-100-mine--100-maxe-100.bin',
                      placementDir = './placement/', num= 20, copy = 25,
		      threshold = 100, flag_form = False):
    """Mutation in flexible CDOCKER genetic algorithm

    Parameters
    ----------
    final_result: np.array
                   analyzed clustering result
    pair_list: np.array
                pair list of the crossover result
    receptorSel: str
                  receptor selection
    flexSel: str
              flexible part selection
    ligSel: str
             ligand selection
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    saveLig: str
              ligand docked result saved folder name
    saveProt: str
               protein docked result saved folder name
    crossoverLig: str
                   crossover ligand folder before mutation
    crossoverProt: str
                    crossover protein folder before mutation
    hardGridFile: str
                   hard grid file name
    nativeGridFile: str
                     native grid file name
    placementDir: str
                   placement folder after mutation
    num: float
          number of conformer
    copy: float
           number of copies for each conformer
    threshold: float
                threshold for energy cutoff
    flag_form: bool
                whether or not the grid file is formatted

    Returns
    -------
    int
        number of docking trials

    """
    ## Prepare
    idxCopy = 1
    hardSet = {'selection': flexSel, 'flag_form' : flag_form, 'gridFile': hardGridFile}
    nativeSet = {'selection': flexSel, 'flag_form' : flag_form, 'gridFile': nativeGridFile}
    _mkdir_(placementDir)

    ## Get cutoff energy
    tmpener = []
    for parent in final_result[:, 0]:
        read.pdb(saveLig + str(parent) + ".pdb", resid = True)
        read.pdb(saveProt + str(parent) + ".pdb", resid = True)
        minimize.run_sd(nstep = 50, tolenr = 0.01)
        minimize.run_abnr(nstep = 50, tolenr = 0.005)
        tmpener.append(energy.get_total())
    totalEner = np.amax(np.asarray(tmpener))
    mutateEner = totalEner + threshold

    ## Loop and save poses in the save dir
    while idxCopy <= num * copy:

        ## Read in ligand - protein random crossover position
        ligandID = pair_list[idxCopy - 1, 0]
        proteinID = pair_list[idxCopy - 1, 1]
        read.pdb(crossoverLig + str(ligandID) + ".pdb", resid = True)
        read.pdb(crossoverProt + str(proteinID) + ".pdb", resid = True)
        minimize.run_sd(nstep = 50, tolenr = 0.01)
        minimize.run_abnr(nstep = 50, tolenr = 0.005)

        ## Compute energy difference
        ## Mutation probility cutoff based on energy difference
        DEner = energy.get_total() - totalEner
        prob = _mutate_prob_(DEner)
        rand = random()

        ## No mutation if random number smaller than the prob cutoff
        if rand <= prob:
            print("Flexible receptor + ligand placement ", str(idxCopy))
            write.coor_pdb(placementDir + str(idxCopy) + ".pdb", titel = '''Title
            * The ligand ID is %d
            * The protein ID is %d
            * The placement ID is %d
            * The energy difference is %8.6f
            * The mutate score is %8.6f and cutoff is %8.6f
            * No mutation applied'''
            % (ligandID, proteinID, idxCopy, DEner, rand, prob))
            idxCopy += 1

        ## Perform mutation if random number smaller than the prob cutoff
        else:

            ## Save conformer and side chain conformer
            randomEner = mutateEner
            allCoor = coor.get_positions()
            allCoor['atomIndex'] = np.arange(np.shape(list(ligSel))[0])
            ligand_xyz = allCoor.loc[np.asarray(list(ligSel)), ['x', 'y', 'z']]
            receptor_xyz = allCoor.loc[np.asarray(list(receptorSel)), ['x', 'y', 'z', 'atomIndex']]

            ## Random translation and rotation and minimize in grids.
            while randomEner >= mutateEner:

                ## Random rotation and translation
                new_receptor = receptor_xyz
                new_xyz = pd.DataFrame(rand_rot_trans(ligand_xyz.to_numpy(), max_rot = pi / 6), columns = ['x', 'y', 'z'])
                new_xyz['atomIndex'] = allCoor.loc[np.asarray(list(ligSel)), 'atomIndex'].to_numpy()
                new_xyz = pd.concat([new_receptor, new_xyz], ignore_index = True)
                new_xyz = new_xyz.sort_values(by = ['atomIndex'], ignore_index = True)[['x', 'y', 'z']]
                coor.set_positions(new_xyz)

                ## Minimize in Grids
                hardGrid = grid.CDOCKER()
                hardGrid.setVar(hardSet)
                hardGrid.read()
                minimize.run_sd(nstep = 100)
                minimize.run_abnr(nstep = 500, tolenr = 0.01)
                hardGrid.off()
                hardGrid.clear()

                nativeGrid = grid.CDOCKER()
                nativeGrid.setVar(nativeSet)
                nativeGrid.read()
                minimize.run_sd(nstep = 50)
                minimize.run_abnr(nstep = 100, tolenr = 0.01)
                nativeGrid.off()
                nativeGrid.clear()

                ## Check energy
                energy.show()
                randomEner = energy.get_total()

            ## Save mutated pose
            print("Flexible receptor + ligand placement ", str(idxCopy))
            write.coor_pdb(placementDir + str(idxCopy) + ".pdb", titel = '''Title
            * The ligand ID is %d
            * The protein ID is %d
            * The placement ID is %d
            * The energy difference is %8.6f
            * The mutate score is %8.6f and cutoff is %8.6f
            * After mutation
            * The total energy is %8.6f
            * The cutoff energy is %8.6f'''
            % (ligandID, proteinID, idxCopy, DEner, rand, prob, randomEner, mutateEner))
            idxCopy += 1

    return num * copy

def FCDOCKER_crossover(cluster_result, dock_result, num = 20, copy = 25):
    """Crossover in flexible CDOCKER genetic algorithm

    Parameters
    ----------
    cluster_result: np.array
                     clustering result using MMTSB
    dock_result: np.array
                  docking results with grid energy recorded
    num: float
          number of conformer
    copy: float
           number of copies for each conformer

    Returns
    -------
    final_result: np.array
          analyzed clustering result
    np.array
          pair list of the crossover result
    """
    ## Sort cluster result
    clusterIdx = np.unique(cluster_result[:, 1])
    final_result = np.ones((len(clusterIdx), 2))
    for cluster in clusterIdx:
        tmpEner = []
        pdbIdx = cluster_result[cluster_result[:, 1] == cluster][:, 0]
        for pdb in pdbIdx:
            tmpEner.append(dock_result[dock_result[:, 1] == pdb][:, 0])
        cResult= pd.DataFrame()
        cResult['PDB'] = np.asarray(pdbIdx)
        cResult['Energy'] = np.asarray(tmpEner)
        cResult = cResult.sort_values(by = ['Energy'], ignore_index = True)
        final_result[cluster - 1, 0] = cResult.iloc[0]['PDB']
        final_result[cluster - 1, 1] = len(tmpEner)
        final_result = final_result.astype(int)

    ## Perpare for crossover
    total = num * copy
    comb_pair = total // 4
    self_pair = total - comb_pair * 2
    idx = final_result[:, 0]
    count = final_result[:, 1]

    ## Random crossover
    tmp1 = choices(idx, weights = count, k = comb_pair)
    tmp2 = choices(idx, weights = count, k = comb_pair)
    tmp3 = choices(idx, weights = count, k = self_pair)

    ## Create protein-ligand list
    total_list = np.zeros((total, 2))
    for i in np.arange(comb_pair):
        j = 2 * i
        k = 2 * i + 1
        total_list[j, 0] = tmp1[i]
        total_list[k, 0] = tmp2[i]
        total_list[j, 1] = tmp2[i]
        total_list[k, 1] = tmp1[i]
        i += 1

    for i in np.arange(self_pair): total_list[comb_pair * 2 + i, :] = tmp3[i]
    return final_result, total_list.astype(int)

def FCDOCKER_calc_dihedral(flexchain, cluster_result, saveProt, protein_dihedral):
    """Calculate dihedral angle and corresponding entropy in flexible docking

    Parameters
    ----------
    flexchain: pd.DataFrame
                dataframe of the flexchain selection
    cluster_result: np.array
                     clustering result using MMTSB
    saveProt: str
               protein docked result saved folder name
    protein_dihedral: pd.DataFrame
                       protein dihedral angle look up table

    Returns
    -------
    entropy : np.array
            entropy for each cluster
    np.array
            cluster ID
    sc_entropy: np.array
            entropy for each cluster & each amino acid
    """
    ## Create entropy dataframe
    sc_entropy = np.zeros((len(np.unique(cluster_result[:, 1])), len(flexchain)))

    ## Loop through flexible chains
    for idx, row in flexchain.iterrows():
        tmp = pycharmm.SelectAtoms().by_res_and_type(row['seg_id'],
                                     str(row['res_id']), 'C')
        resname = tmp.get_res_names()[0]
        atoms = protein_dihedral.loc[protein_dihedral['Amino_acid'] == resname].to_numpy()
        if (len(atoms) > 0):
            tmp_result = np.zeros((len(cluster_result), len(atoms) + 1))
            print("Calculate conformational entropy contribution from amino acid ", resname)

            ## Loop through clustered pdbs
            for trial in np.arange(len(cluster_result)):
                pdbID = cluster_result[trial, 0]
                read.pdb(saveProt + str(pdbID) + ".pdb", resid = True)
                allCoor = coor.get_positions()

                for dihe_set in np.arange(len(atoms)):
                    dihe = pycharmm.SelectAtoms().by_res_and_type(row['seg_id'],
                                    str(row['res_id']), atoms[dihe_set, 1])
                    xyz = allCoor.loc[np.asarray(list(dihe))].to_numpy()
                    tmp_result[trial, dihe_set] = calc_dihedral(xyz)

            ## Post processing
            if resname != 'PHE' or resname != 'TYR':
                tmp_result[(tmp_result >= 0) & (tmp_result < 120)] = 1
                tmp_result[(tmp_result > -120) & (tmp_result < 0)] = 2
                tmp_result[(tmp_result <= -120) | (tmp_result >= 120)] = 3
            else:
                tmp_result[:, 0][(tmp_result[:, 0] >= 0) & (tmp_result[:, 0] < 120)] = 1
                tmp_result[:, 0][(tmp_result[:, 0] > -120) & (tmp_result[:, 0] < 0)] = 2
                tmp_result[:, 0][(tmp_result[:, 0] <= -120) | (tmp_result[:, 0] >= 120)] = 3

                tmp_result[:, 1][(tmp_result[:, 1] >= -45) & (tmp_result[:, 1] <= 45)] = 0
                tmp_result[:, 1][(tmp_result[:, 1] < -135) | (tmp_result[:, 1] > 135)] = 0
                tmp_result[:, 1][(tmp_result[:, 1] >= -135) & (tmp_result[:, 1] < -45)] = 1
                tmp_result[:, 1][(tmp_result[:, 1] > 45) & (tmp_result[:, 1] <= 135)] = 1
            tmp_result[:, dihe_set + 1] = cluster_result[:, 1]

            for trial in np.unique(cluster_result[:, 1]):
                tmp_total = []
                tmpEntropy = tmp_result[tmp_result[:, -1] == trial][:, 0:len(atoms)]
                tmp_total.append(len(tmpEntropy))
                state = np.unique(tmpEntropy, axis = 0)
                for tmpI in np.arange(len(state)):
                    tmp_total.append(np.sum((tmpEntropy == state[tmpI]).all(axis = 1)))

                tmp_total = np.asarray(tmp_total)
                tmp_total = tmp_total / tmp_total[0]
                result = np.sum(tmp_total * np.log(tmp_total))
                sc_entropy[trial - 1, idx] = result

    entropy = np.sum(sc_entropy, axis = 1) * 0.593
    return entropy, np.unique(cluster_result[:, 1]), sc_entropy

## Define default simulated annealing
def default_ommd_sian(ommd):
    """Default OpenMM docking simulated annealing

    Parameters
    ----------
    ommd : pycharmm.grid.OMMD
           OpenMM docking system

    Returns
    -------
    bool
         True for success
    """
    ## Stage 1
    ommdSet = {'soft': 1, 'hard': 0, 'emax': 0.6,
               'mine': -0.4, 'maxe': 0.4, 'eps': 3}
    ommd.setVar(ommdSet)
    ommd.change_softness()
    ommdSet = {'steps': 3000, 'heatFrq': 50, 'startTemp': 300,
               'endTemp': 700, 'incrTemp': 1}
    ommd.setVar(ommdSet)
    ommd.simulated_annealing()

    ## Stage 2
    ommdSet = {'soft': 1, 'hard': 0, 'emax': 0.6,
               'mine': -0.4, 'maxe': 0.4, 'eps': 3}
    ommd.setVar(ommdSet)
    ommd.change_softness()
    ommdSet = {'steps': 14000, 'heatFrq': 50, 'startTemp': 700,
               'endTemp': 300, 'incrTemp': -1}
    ommd.setVar(ommdSet)
    ommd.simulated_annealing()

    ## Stage 3
    ommdSet = {'soft': 0, 'hard': 1, 'emax': 3,
               'mine': -30, 'maxe': 30, 'eps': 3}
    ommd.setVar(ommdSet)
    ommd.change_softness()
    ommdSet = {'steps': 7000, 'heatFrq': 50, 'startTemp': 500,
               'endTemp': 300, 'incrTemp': -1}
    ommd.setVar(ommdSet)
    ommd.simulated_annealing()

    ## Stage 4
    ommdSet = {'soft': 0, 'hard': 1, 'emax': 30,
               'mine': -300, 'maxe': 300, 'eps': 3}
    ommd.setVar(ommdSet)
    ommd.change_softness()
    ommdSet = {'steps': 3000, 'heatFrq': 50, 'startTemp': 400,
               'endTemp': 50, 'incrTemp': -1}
    ommd.setVar(ommdSet)
    ommd.simulated_annealing()

    return True

## Clustering method
def cluster_mmtsb(radius, name):
    """Use MMTSB to perfrom clustering (cluster.pl)

    Parameters
    ----------
    radius : float
             clustering radius
    name : str
           name of the ligand pdbs

    Returns
    -------
    message : str
        clustering results
    """
    if type(name) != str: name = str(name)
    cluster_cmd = "cluster.pl -kclust -nolsqfit -radius " + str(radius) + " -selmode heavy " + name + "> cluster.log"
    status = system(cluster_cmd)

    if status == 0:
        message = "Clustering docked pose " + name + " with raidus " + str(radius) + " succeed."
    else:
        message = "Something wrong with the clustering method, please check."

    return message

def scan_cluster_radius(name):
    """Find the best clustering radius
    Default use in rigid CDOCKER

    Parameters
    ----------
    name: str
           name of the ligand pdbs

    Returns
    -------
    radius : float
          best clustering radius
    """
    ## Prepare file
    _rm_('tmpcluster')

    ## Scan clustering radius
    if type(name) != str: name = str(name)
    radii = np.arange(5, 21, 1) / 10
    sort_cluster_cmd = "grep @cluster cluster.log | sed 1d | sort -gk 4 | tail -n 1 | awk '{print $4}' >> tmpcluster"
    for radius in radii:
        cluster_mmtsb(radius, name)
        system(sort_cluster_cmd)

    ## Find the best clustering radius
    cluster_size = np.loadtxt('tmpcluster', dtype = int)
    radius = radii[cluster_size == np.amax(cluster_size)][0]

    return radius

def rcdocker_default_sort(radius, name):
    """Rigid CDOCKER default sorting methods

    Parameters
    ----------
    radius : float
             clustering radius
    name : str
           name of the ligand pdbs

    Returns
    -------
    np.array
          clustering result
    """
    if type(name) != str: name = str(name)
    cluster_mmtsb(radius, name)

    cmd = '''
    init=`grep @cluster cluster.log | sed 1d | sort -gk 4 | tail -n 1 | awk '{print $2}' | cut -b 3-`
    final=$[init+1]
    awk "/cluster t.$init /, /cluster t.$final / {print}" cluster.log | sed /cluster/d | awk '{print $2}' > tmp
    '''
    system(cmd)

    return np.loadtxt('tmp', dtype = str)

def sort_cluster(pdb_files, dock_result, sort_method):
    """Sort clustering results

    Parameters
    ----------
    pdb_files : np.array
                pdb names of docked poses
    dock_result : pd.DataFrame
                  docking results (i.e., energy)
    sort_method : str
                  sorting based on sort_method

    Returns
    -------
    tmpResult: pd.DataFrame
         sorted results
    """
    tmpEner = []
    cluster_size = np.size(pdb_files)
    if cluster_size > 1:
        for poseID in pdb_files:
            tmpEner.append(dock_result[dock_result['PDB_name'] == poseID][sort_method].values[0])

        tmpEner = np.asarray(tmpEner)
        minPose = pdb_files[tmpEner == np.amin(tmpEner)][0]
        tmpResult = dock_result[dock_result['PDB_name'] == minPose]
    else:
        print("Cluster only have one cluster member")
        tmpResult = dock_result[dock_result['PDB_name'] == pdb_files]

    tmp = pd.DataFrame(np.asarray([cluster_size]), columns = ['cluster_size'])
    tmp = tmp.astype(str)
    tmpResult = tmpResult.join(tmp.set_index(tmpResult.index))
    return tmpResult

def top_N_cluster(logFile = 'cluster.log', N = 10, total = 500):
    """Get top N largest clusters

    Parameters
    ----------
    logFile : str
              cluster log file
    N : int
        number of large cluster identified

    Returns
    -------
    np.array
          sorted top N cluster results
    """
    cutoff = int(np.ceil(total / 100) - 1)
    if cutoff <= 1: cutoff = 1
    cmd = '''
    idx=1
    rm -f cluster_list
    grep @cluster %s | sed 1d | awk '{if ($4 > %d) print}' | sort -nk 4 | tail -n %d | awk '{print $2}' | cut -b 3- > tmplist
    for init in `cat tmplist`; do
        final=$[init+1]
        awk "/cluster t.$init /, /cluster t.$final / {print}" %s | sed /cluster/d | awk '{print $2}' > tmp
        awk '{$2 = name; print}' name=$idx tmp >> cluster_list
        idx=$[idx+1]
        done
    ''' % (logFile, cutoff, N, logFile)
    system(cmd)
    try:
        cluster_list = np.loadtxt("cluster_list", dtype = int)
        return np.loadtxt("cluster_list", dtype = int)
    except IOError:
        print("All clusters have a cluster number of 1, please increase docking trials")
        return False

def _fcdocker_cluster_check_(logFile = 'cluster.log', total = 500):
    """Check number of clusters with cluster number greater than 1
    Parameters
    ----------
    logFile : str
              cluster log file

    Returns
    -------
    int
       number of clusters with cluster number greater than 1
    """
    cutoff = int(np.ceil(total / 100) - 1)
    if cutoff <= 1: cutoff = 1
    cmd = '''
    rm -f cluster_list
    grep @cluster %s | sed 1d | awk '{if ($4 > %d) print}' | wc -l > cluster_list
    ''' % (logFile, cutoff)
    system(cmd)
    return int(np.loadtxt("cluster_list", dtype = int))

################################################################
##
##		Begin of pyCHARMM Rigid CDOCKER
##
################################################################

def Rigid_CDOCKER(xcen = 0, ycen = 0, zcen = 0, maxlen = 10, dielec = 3,
                  rcta = 0, rctb = 0, hmax = 0, flag_grid = False,
                  flag_rdie = True, flag_form = False, flag_delete_grid = True,
                  probeFile = '"../Toppar/fftdock_c36prot_cgenff_probes.txt"',
                  softGridFile = 'grid-emax-0.6-mine--0.4-maxe-0.4.bin',
                  hardGridFile = 'grid-emax-3-mine--30-maxe-30.bin',
                  nativeGridFile = 'grid-emax-100-mine--100-maxe-100.bin',
                  receptorPDB = './protein.pdb', receptorPSF = './protein.psf',
                  ligPDB = './ligand.pdb', ligSeg = 'LIGA', confDir = './conformer/',
                  placementDir = './placement/',  exhaustiveness = 'high',
                  numPlace = 100, numCopy = 1000, flag_delete_conformer= True,
                  flag_delete_placement = True, flag_save_all = True,
                  flag_save_cluster= True, flag_save_top = True,
                  flag_suppress_print = True, flag_center_ligand = True,
                  flag_fast_grid = False, flag_use_hbond = False,
                  flag_fast_placement = True, threshold = 2500,
                  sort_energy = 'total_energy', saveDir = './dockresult/'):

    """Rigid CDOCKER standard docking method

    Parameters
    ----------
    xcen: float
           center of the docking pocket
    ycen: float
           center of the docking pocket
    zcen: float
           center of the docking pocket
    maxlen: float
             size of the grid box
    dielec: float
             dielectric constant
    rcta: float
           customizable grid left cutoff
    rctb: float
           customizable grid right cutoff
    hmax: float
           customizable grid well-depth
    flag_grid: bool
                whether or not grid need to be generated
    flag_rdie: bool
                true for rdie, false for cdie
    flag_form: bool
                whether or not grid form is formatted
    flag_delete_grid: bool
                       whether or not delete grid after calculation
    probeFile: str
                gpu probe file names
    softGridFile: str
                   soft grid file name
    hardGridFile: str
                   hard grid file name
    nativeGridFile: str
                     native grid file name
    receptorPDB: str
              protein pdb file name
    receptorPSF: str
              protein psf file name
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    confDir: str
              conformer folder name
    placementDir: str
                   placement folder name
    exhaustiveness: str
                     exhaustiveness for fast placement, high, medium, low
    numPlace: int
               number of placement for each conformer
    numCopy: int
              number of Copy for OpenMM docking simulated annealing
    flag_delete_conformer: bool
                            whether or not delete conformer after docking
    flag_delete_placement: bool
                            whether or not delete placement after docking
    flag_save_all: bool
                    whether or not save all docked pose
    flag_save_cluster: bool
                        whether or not save clustered results
    flag_save_top: bool
                    whether or not save top 10 lowest energy pose
    flag_suppress_print: bool
                          whether or not suppress printing
    flag_fast_grid: bool
                     whether or not just use grid minimize result i.e., skip all atom mini
    flag_center_ligand: bool
                         whether or not ligand need to be centered
    flag_use_hbond : bool
                     whether or not use hydrogen/covalent bond in RCDOCKER
    flag_fast_placement : bool
                          whether or not use fast placement
    threshold : float
                cutoff threshold for original RCDOCKER placement
                only meaningful when flag_fast_placement = True
    sort_energy: str
                  sorting method
    saveDir: str
              folder name for docked result

    Returns
    -------
    clusterResult : pd.DataFrame
         clustering result
    dockResult : pd.DataFrame
         docking result
    """
    ## Generate grids for docking if no grid files generated before
    if not flag_grid:
        if hmax > 0: hmax = 0
        if not flag_use_hbond: hmax = 0
        if psf.get_natom() > 0: psf.delete_atoms()
        read.psf_card(receptorPSF, append = True)
        read.pdb(receptorPDB, resid = True)

        genGrid = grid.Grid()
        gridSet = {'xCen': xcen, 'yCen': ycen, 'zCen': zcen,
                   'xMax': maxlen, 'yMax': maxlen, 'zMax': maxlen,
                   'dielec': dielec, 'flag_rdie': flag_rdie,
                   'rcta': rcta, 'rctb': rctb, 'hMax': hmax,
                   'emax': 0.6, 'maxe': 0.4, 'mine': -0.4, 'gridFile': softGridFile,
                   'probeFile': probeFile, 'flag_form' : flag_form, 'flag_grhb' : flag_use_hbond}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + softGridFile +  " is ", status)

        gridSet = {'emax': 3, 'maxe': 30, 'mine': -30, 'gridFile': hardGridFile}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + hardGridFile +  " is ", status)

        gridSet = {'emax': 100, 'maxe': 100, 'mine': -100, 'gridFile': nativeGridFile}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + nativeGridFile +  " is ", status)

        ## Update nonbond interactions
        if flag_rdie:
            update_nonbond = pycharmm.UpdateNonBondedScript(
              atom = True, switch = True, vswitch = True, vdwe = True,
              elec = True, rdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
              emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()
        else:
            update_nonbond = pycharmm.UpdateNonBondedScript(
              atom = True, switch = True, vswitch = True, vdwe = True,
              elec = True, cdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
              emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()

    ## Ligand initial placement
    if flag_suppress_print:
        settings.set_verbosity(1)
        settings.set_warn_level(1)
        settings.set_bomb_level(-1)

    if flag_fast_placement :
        numConf, numPlace = RCDOCKER_fast_init_place(receptorPDB = receptorPDB, receptorPSF = receptorPSF,
                            ligPDB = ligPDB, ligSeg = ligSeg, confDir = confDir,
                            placementDir = placementDir, exhaustiveness = exhaustiveness,
                            flag_center_ligand = flag_center_ligand, xcen = xcen,
                            ycen = ycen, zcen = zcen, maxlen = maxlen, numPlace = numPlace)
    else :
        numConf, numPlace = RCDOCKER_init_place(ligPDB = ligPDB, ligSeg = ligSeg,
                            hardGridFile = hardGridFile, nativeGridFile = nativeGridFile,
                            confDir = confDir, placementDir = placementDir,
                            flag_center_ligand = flag_center_ligand,
                            flag_use_hbond = flag_use_hbond, flag_form = flag_form,
                            flag_rdie = flag_rdie, dielec = dielec,
                            xcen = xcen, ycen = ycen, zcen = zcen,
                            numPlace = numPlace, threshold = threshold)

    settings.set_verbosity(5)
    settings.set_warn_level(5)
    settings.set_bomb_level(0)

    ## Update nonbond interactions
    if flag_rdie:
        update_nonbond = pycharmm.UpdateNonBondedScript(
          atom = True, switch = True, vswitch = True, vdwe = True,
          elec = True, rdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
          emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()
    else:
        update_nonbond = pycharmm.UpdateNonBondedScript(
          atom = True, switch = True, vswitch = True, vdwe = True,
          elec = True, cdie = True, cutnb = 12, ctofnb = 10, ctonnb = 8,
          emax = 10000, maxe = 10000, mine = -10000, epsilon = dielec).run()

    ## Get docking trials / number of OpenMM docking needed
    num = numConf * numPlace
    trial = num // numCopy
    remindar = num % numCopy
    dockResult = np.zeros([num, 2])

    ## Run OMMD docking
    idxBatch = 0
    gridVdw = []
    gridElec = []
    gridHbon = []
    totalEner = []
    allPose = []
    conformerPose = []
    placementPose = []
    placementID = 1
    conformerID = 1
    ligand = pycharmm.SelectAtoms(seg_id = ligSeg)

    if remindar > 0:
        ommd = grid.OMMD()
        ommdSet = {'softGridFile': softGridFile, 'hardGridFile': hardGridFile,
                   'flex_select': ligand, 'numCopy': remindar,
                   'flag_form' : flag_form, 'flag_grhb': flag_use_hbond}
        ommd.setVar(ommdSet)
        ommd.create()
        print("OMMD set up for batch ", idxBatch)

        if flag_suppress_print:
            settings.set_verbosity(1)
            settings.set_warn_level(1)
            settings.set_bomb_level(-1)

        idxCopy = 1
        while idxCopy <= remindar:
            read.pdb(placementDir + str(placementID) + '.pdb', resid = True)
            ommd.set_coor(idxCopy = idxCopy)
            print("Set OMMD ligand copy " + str(idxCopy))
            placementID += 1
            idxCopy += 1

        default_ommd_sian(ommd)

        nativeGrid = grid.CDOCKER()
        nativeSet = {'selection': ligand, 'flag_form' : flag_form, 'gridFile':
                      nativeGridFile, 'flag_grhb': flag_use_hbond}
        nativeGrid.setVar(nativeSet)
        nativeGrid.read()

        i = 1
        idxCopy = 1
        while idxCopy <= remindar:
            ommd.copy_coor(idxCopy = idxCopy)
            print("Minimize OMMD ligand copy " + str(idxCopy))
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 1000, tolenr = 1e-3)
            #allEner = energy.get_energy()
            dockPose = str(conformerID) + "_" + str(i) + ".pdb"
            tmpTotal = lingo.get_energy_value('ENER') #allEner[allEner['name'] == 'energy']['value'].values[0]
            tmpGrvdw = lingo.get_energy_value('GRVD') #allEner[allEner['name'] == 'grvdw']['value'].values[0]
            tmpGrelec = lingo.get_energy_value('GREL') #allEner[allEner['name'] == 'grelec']['value'].values[0]
            tmpGrhb = energy.get_eterm(111)
            write.coor_pdb(dockPose, title = '''Title
                       * The conformer ID is %d
                       * The dock pose ID is %d
                       * The total energy is %8.6f
                       * The total grid energy is %8.6f
                       * Grid vdw, elec and hbond energy is
                       * %8.6f, %8.6f and %8.6f'''
                       % (conformerID, i, tmpTotal, tmpGrvdw + tmpGrelec + tmpGrhb,
                       tmpGrvdw, tmpGrelec, tmpGrhb))
            totalEner.append(tmpTotal)
            gridVdw.append(tmpGrvdw)
            gridElec.append(tmpGrelec)
            gridHbon.append(tmpGrhb)
            conformerPose.append(conformerID)
            placementPose.append(i)
            allPose.append(dockPose)
            i += 1
            if i > numPlace:
                i = 1
                conformerID += 1
            idxCopy += 1

        nativeGrid.off()
        nativeGrid.clear()
        nativeSet.clear()
        ommd.clear()
        ommdSet.clear()
        settings.set_verbosity(5)
        settings.set_warn_level(5)
        settings.set_bomb_level(0)

    idxBatch += 1
    if idxBatch <= trial:
        ommd = grid.OMMD()
        ommdSet = {'softGridFile': softGridFile, 'hardGridFile': hardGridFile,
                   'flex_select': ligand, 'numCopy': numCopy,
                   'flag_form' : flag_form, 'flag_grhb': flag_use_hbond}
        ommd.setVar(ommdSet)
        ommd.create()

        if flag_suppress_print:
            settings.set_verbosity(1)
            settings.set_warn_level(1)
            settings.set_bomb_level(-1)

        while idxBatch <= trial:
            print("OMMD set up for batch ", idxBatch)

            idxCopy = 1
            while idxCopy <= numCopy:
                read.pdb(placementDir + str(placementID) + '.pdb', resid = True)
                ommd.set_coor(idxCopy = idxCopy)
                print("Set OMMD ligand copy " + str(idxCopy))
                placementID += 1
                idxCopy += 1

            default_ommd_sian(ommd)

            nativeGrid = grid.CDOCKER()
            nativeSet = {'selection': ligand, 'flag_form' : flag_form, 'gridFile':
                          nativeGridFile, 'flag_grhb': flag_use_hbond}
            nativeGrid.setVar(nativeSet)
            nativeGrid.read()

            i = 1
            idxCopy = 1
            while idxCopy <= numCopy:
                ommd.copy_coor(idxCopy = idxCopy)
                print("Minimize OMMD ligand copy " + str(idxCopy))
                minimize.run_sd(nstep = 50)
                minimize.run_abnr(nstep = 1000, tolenr = 1e-3)
                #allEner = energy.get_energy()
                dockPose = str(conformerID) + "_" + str(i) + ".pdb"
                tmpTotal = lingo.get_energy_value('ENER') #allEner[allEner['name'] == 'energy']['value'].values[0]
                tmpGrvdw = lingo.get_energy_value('GRVD') #allEner[allEner['name'] == 'grvdw']['value'].values[0]
                tmpGrelec = lingo.get_energy_value('GREL') #allEner[allEner['name'] == 'grelec']['value'].values[0]
                tmpGrhb = energy.get_eterm(111)
                write.coor_pdb(dockPose, title = '''Title
                           * The conformer ID is %d
                           * The dock pose ID is %d
                           * The total energy is %8.6f
                           * The total grid energy is %8.6f
                           * Grid vdw, elec and hbond energy is
                           * %8.6f, %8.6f and %8.6f'''
                           % (conformerID, i, tmpTotal, tmpGrvdw + tmpGrelec + tmpGrhb,
                           tmpGrvdw, tmpGrelec, tmpGrhb))
                totalEner.append(tmpTotal)
                gridVdw.append(tmpGrvdw)
                gridElec.append(tmpGrelec)
                gridHbon.append(tmpGrhb)
                conformerPose.append(conformerID)
                placementPose.append(i)
                allPose.append(dockPose)
                i += 1
                if i > numPlace:
                    i = 1
                    conformerID += 1
                idxCopy += 1

            nativeGrid.off()
            nativeGrid.clear()
            nativeSet.clear()
            idxBatch += 1

        ommd.clear()
        ommdSet.clear()
        settings.set_verbosity(5)
        settings.set_warn_level(5)
        settings.set_bomb_level(0)

    ## Save docking result for all docked pose
    dockResult = pd.DataFrame()
    dockResult["total_energy"] = totalEner
    dockResult["grid_total"] = np.asarray(gridVdw) + np.asarray(gridElec) + np.asarray(gridHbon)
    dockResult["grid_vdw"] = gridVdw
    dockResult["grid_elec"] = gridElec
    dockResult["grid_hbond"] = gridHbon
    dockResult["conformer_id"] = conformerPose
    dockResult["placement_id"] = placementPose
    dockResult["PDB_name"] = allPose

    ## Cluster and sort docking pose
    conformerID = 1
    clusterResult = pd.DataFrame(columns = list(dockResult.columns).append('cluster_size'))
    while conformerID <= numConf:
        conformerName = str(conformerID) + '_*'
        radius = scan_cluster_radius(name = conformerName)
        tmp = rcdocker_default_sort(radius = radius, name = conformerName)

        clusterResult = pd.concat([clusterResult, sort_cluster(pdb_files = tmp,
                       dock_result = dockResult, sort_method = "total_energy")],
                       ignore_index = True)

        conformerID += 1

    ## Sort dataframe and save grid docking results
    _mkdir_(saveDir)
    dockResult = dockResult.sort_values(by = ["total_energy"], ignore_index = True)
    dockResult.to_csv(saveDir + 'dockResult.tsv', sep = '	')
    clusterResult = clusterResult.sort_values(by = ["total_energy"], ignore_index = True)
    clusterResult.to_csv(saveDir + 'clusterResult.tsv', sep = '	')

    ## Explicit atom minimization
    if not flag_fast_grid:
        psf.delete_atoms()
        read.psf_card(receptorPSF, append = True)
        read.pdb(receptorPDB, resid = True)
        read.sequence_pdb(ligPDB)
        settings.set_bomb_level(-1)        ## for 3-mem ring
        gen.new_segment(seg_name = ligSeg)
        settings.set_bomb_level(0)
        read.pdb(ligPDB, resid = True)
        ligand = pycharmm.SelectAtoms().by_seg_id(ligSeg)
        cons_fix.setup(ligand.__invert__())

    ## Minimize cluster result
        vdw = []
        elec = []
        totalEner = []
        _rm_("cluster_*")
        pdbID = clusterResult['PDB_name'].tolist()
        grid_hbond = clusterResult['grid_hbond'].tolist()
        conformer_id = clusterResult["conformer_id"].tolist()
        placement_id = clusterResult["placement_id"].tolist()
        cluster_size = clusterResult["cluster_size"].tolist()
        for idx in np.arange(len(pdbID)):
            read.pdb(str(pdbID[idx]), resid = True)
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 1000, tolenr = 0.001)
            tmptotal = energy.get_total() + grid_hbond[idx]
            tmpelec = energy.get_elec()
            tmpvdw = energy.get_vdw()
            totalEner.append(tmptotal)
            elec.append(tmpelec)
            vdw.append(tmpvdw)

            write.coor_pdb("cluster_" + str(pdbID[idx]),
                           selection = ligand, title = '''Title
                           * The docked pose ID is %s
                           * The conformer ID is %d
                           * The placement ID is %d
                           * The total energy is %8.6f
                           * The vdw energy is %8.6f
                           * The elec energy is %8.6f
                           * The grid_hbond is %8.6f '''
                           % (pdbID[idx], conformer_id[idx], placement_id[idx],
                           tmptotal, tmpvdw, tmpelec, grid_hbond[idx]))

        explicitCluster = pd.DataFrame()
        explicitCluster["total_energy"] = totalEner
        explicitCluster["vdw"] = vdw
        explicitCluster["elec"] = elec
        explicitCluster["grid_hbond"] = grid_hbond
        explicitCluster["conformer_id"] = conformer_id
        explicitCluster["placement_id"] = placement_id
        explicitCluster["PDB_name"] = pdbID
        explicitCluster["cluster_size"] = cluster_size

	## Minimize top10 result
        vdw = []
        elec = []
        totalEner = []
        _rm_("top10_*")
        pdbID = dockResult['PDB_name'].tolist()[0:11]
        grid_hbond = dockResult['grid_hbond'].tolist()[0:11]
        conformer_id = dockResult["conformer_id"].tolist()[0:11]
        placement_id = dockResult["placement_id"].tolist()[0:11]
        for idx in np.arange(len(pdbID)):
            read.pdb(str(pdbID[idx]), resid = True)
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 1000, tolenr = 0.001)
            tmptotal = energy.get_total() + grid_hbond[idx]
            tmpelec = energy.get_elec()
            tmpvdw = energy.get_vdw()
            totalEner.append(tmptotal)
            elec.append(tmpelec)
            vdw.append(tmpvdw)

            write.coor_pdb("top10_" + str(pdbID[idx]),
                           selection = ligand, title = '''Title
                           * The docked pose ID is %s
                           * The conformer ID is %d
                           * The placement ID is %d
                           * The total energy is %8.6f
                           * The vdw energy is %8.6f
                           * The elec energy is %8.6f
                           * The grid_hbond is %8.6f '''
                           % (pdbID[idx], conformer_id[idx], placement_id[idx],
                           tmptotal, tmpvdw, tmpelec, grid_hbond[idx]))

        explicitTop10 = pd.DataFrame()
        explicitTop10["total_energy"] = totalEner
        explicitTop10["vdw"] = vdw
        explicitTop10["elec"] = elec
        explicitTop10["grid_hbond"] = grid_hbond
        explicitTop10["conformer_id"] = conformer_id
        explicitTop10["placement_id"] = placement_id
        explicitTop10["PDB_name"] = pdbID

    ## Sort dataframe and save explicit atom minimization results
    if not flag_fast_grid:
        explicitTop10 = explicitTop10.sort_values(by = [sort_energy],
                        ignore_index = True)
        explicitTop10.to_csv(saveDir + 'explicitTop10.tsv', sep = '	')
        explicitCluster = explicitCluster.sort_values(by = [sort_energy],
                          ignore_index = True)
        explicitCluster.to_csv(saveDir + 'explicitCluster.tsv', sep = '	')

    ## Clean and save results
    if flag_save_cluster:
        idx = 1
        _mkdir_(saveDir + 'cluster/')
        print("Save cluster result for each conformer")
        if flag_fast_grid:
            pdbList = clusterResult['PDB_name'].tolist()
            for pdb in pdbList:
                source = str(pdb)
                target = saveDir + 'cluster/top_' + str(idx) + '.pdb'
                _cp_(source, target)
                idx += 1
        else:
            pdbList = explicitCluster['PDB_name'].tolist()
            for pdb in pdbList:
                source = "cluster_" + str(pdb)
                target = saveDir + 'cluster/top_' + str(idx) + '.pdb'
                _cp_(source, target)
                idx += 1

    if flag_save_top:
        idx = 1
        _mkdir_(saveDir + 'top_ener/')
        print("Save top 10 lowest energy pose")
        if flag_fast_grid:
            pdbList = dockResult['PDB_name'].tolist()[0:11]
            topEnergy = dockResult["total_energy"].tolist()[0:10]
            if topEnergy[-1] - topEnergy[0] < 1:
                print("Please check dockResult.tsv, not all poses within 1 kcal/mol cutoff are saved")
            for pdb in pdbList[0:10]:
                source = str(pdb)
                target = saveDir + 'top_ener/top_' + str(idx) + '.pdb'
                _cp_(source, target)
                idx += 1
        else:
            pdbList = explicitTop10['PDB_name'].tolist()[0:11]
            topEnergy = explicitTop10[sort_energy].tolist()[0:10]
            if topEnergy[-1] - topEnergy[0] < 1:
                print("Please check explicitTop10.tsv, not all poses within 1 kcal/mol cutoff are saved")
            for pdb in pdbList[0:10]:
                source = "top10_" + str(pdb)
                target = saveDir + 'top_ener/top_' + str(idx) + '.pdb'
                _cp_(source, target)
                idx += 1

    if flag_save_all:
        _mkdir_(saveDir + 'allPose/')
        _mv_('[0-9]*', saveDir + 'allPose/')

    if not flag_fast_grid: _rm_("cluster_* top10_*")
    if flag_delete_conformer: _rm_(confDir)
    if flag_delete_placement: _rm_(placementDir)
    if flag_delete_grid: _rm_(nativeGridFile + ' ' + softGridFile + ' ' + hardGridFile)

    _rm_('[0-9]* tmp* cluster.log')
    return clusterResult, dockResult

################################################################
##
##		Begin of pyCHARMM Flexible CDOCKER
##
################################################################

def Flexible_CDOCKER(xcen = 0, ycen = 0, zcen = 0, maxlen = 10, num = 20, copy = 25,
                     generation = 2, threshold_init = 2500, threshold_mutate = 100,
                     flag_grid = False, flag_form = False, flag_delete_grid = True,
                     probeFile = '"../Toppar/fftdock_c36prot_cgenff_probes.txt"',
                     softGridFile = 'grid-emax-0.6-mine--0.4-maxe-0.4.bin',
                     hardGridFile = 'grid-emax-3-mine--30-maxe-30.bin',
                     nativeGridFile = 'grid-emax-100-mine--100-maxe-100.bin',
                     receptorPDB = './protein.pdb', receptorPSF = './protein.psf',
                     receptorCard = '"./flexchain.crd"', saveLig = './ligand/', saveProt = './protein/',
                     crossoverLig = './crossover_ligand/', crossoverProt = './crossover_protein/',
                     saveLigFinal = './ligand_final/', saveProtFinal = './protein_final/',
                     ligPDB = './ligand.pdb', ligSeg = 'LIGA', flexchain = None,
                     placementDir = './placement/',  flag_save_all = False,
                     flag_save_cluster= True, flag_save_placement = False,
                     flag_save_crossover = False, flag_suppress_print = True,
                     flag_center_ligand = True, flag_fast_grid = False,
                     flag_fast_placement = False, exhaustiveness = 'high', top_N_result = 10,
                     sort_energy = 'total_energy', saveDir = './dockresult/'):

    """Flexible CDOCKER standard docking method

    Parameters
    ----------
    xcen: float
           center of the docking pocket
    ycen: float
           center of the docking pocket
    zcen: float
           center of the docking pocket
    maxlen: float
             size of the grid box
    num: float
          number of conformer
    copy: float
           number of copies for each conformer
    threshold_init: float
                     energy threshold for initial placement
    threshold_mutate: float
                       energy threshold for mutation
    flag_grid: bool
                whether or not grid need to be generated
    flag_form: bool
                whether or not grid form is formatted
    flag_delete_grid: bool
                       whether or not delete grid after calculation
    probeFile: str
                gpu probe file names
    softGridFile: str
                   soft grid file name
    hardGridFile: str
                   hard grid file name
    nativeGridFile: str
                     native grid file name
    receptorPDB: str
              protein pdb file name
    receptorPSF: str
              protein psf file name
    receptorCard: str
                   receptor coordinate card name
    saveLig: str
              ligand docked result saved folder name
    saveProt: str
               protein docked result saved folder name
    crossoverLig: str
                   crossover ligand folder before mutation
    crossoverProt: str
                    crossover protein folder before mutation
    saveLigFinal: str
                   ligand final docked result saved folder name
    saveProtFinal: str
                    protein final docked result saved folder name
    ligPDB: str
             ligand pdb file name
    ligSeg: str
             ligand segment ID
    flexchain: pd.DataFrame
                dataframe of the flexchain selection
    placementDir: str
                   placement folder name
    flag_save_all: bool
                    whether or not save all docked pose
    flag_save_cluster: bool
                        whether or not save clustered results
    flag_save_placement: bool
                          whether or not save inital placement after docking
    flag_save_crossover: bool
                          whether or not save crossover after docking
    flag_suppress_print: bool
                          whether or not suppress printing
    flag_center_ligand: bool
                         whether or not ligand need to be centered
    flag_fast_grid: bool
                     whether or not just use grid minimize result i.e., skip all atom mini
    flag_fast_placement : bool
                          whether or not use fast initial placement
    exhaustiveness : str
                     exhaustiveness for fast placement, high, medium, low
    top_N_result: int
                   number of top N clusters, final generation uses top N + 5 clusters
    sort_energy: str
                  sorting method
    saveDir: str
              folder name for docked result

    Returns
    -------
    clusterResult: pd.DataFrame
          clustering result
    dockResult : pd.DataFrame
          docking result
    """
    ## Center ligand if flag_center_ligand = True
    if flag_center_ligand:
        read.sequence_pdb(ligPDB)
        settings.set_bomb_level(-1)        ## for 3-mem ring
        gen.new_segment(seg_name = ligSeg)
        settings.set_bomb_level(0)
        read.pdb(ligPDB, resid = True)
        ligand = pycharmm.SelectAtoms().by_seg_id(ligSeg)

        xyz = coor.get_positions().to_numpy()
        gridCenter = np.array([xcen, ycen, zcen])
        ligCenter = (np.amin(xyz, axis = 0) + np.amax(xyz, axis = 0)) / 2
        xyz = xyz - ligCenter + gridCenter
        new_xyz = pd.DataFrame(xyz, columns = ['x', 'y', 'z'])
        coor.set_positions(new_xyz)
        write.coor_pdb(ligPDB, selection = ligand, title = '''Title
                       * Move ligand to the center of binding pocket''')

    ## Generate grids for docking if no grid files generated before
    if not flag_grid:
        ## Read in protein and prepare flex side chain selection
        if psf.get_natom() > 0: psf.delete_atoms()
        read.psf_card(receptorPSF, append = True)
        read.pdb(receptorPDB, resid = True)

        for idx, row in flexchain.iterrows():
            resid = str(row['res_id'])
            segid = row['seg_id']
            tmpI = pycharmm.SelectAtoms().by_res_id(res_id = resid)
            tmpJ = pycharmm.SelectAtoms().by_seg_id(seg_id = segid)
            if idx == 0:
                flexSideChain = tmpI & tmpJ
            else:
                flexSideChain = flexSideChain | (tmpI & tmpJ)

        settings.set_bomb_level(-1)
        psf.delete_atoms(flexSideChain)
        settings.set_bomb_level(0)

	## Generate grids for docking
        genGrid = grid.Grid()
        gridSet = {'xCen': xcen, 'yCen': ycen, 'zCen': zcen,
                   'xMax': maxlen, 'yMax': maxlen, 'zMax': maxlen,
                   'emax': 0.6, 'maxe': 0.4, 'mine': -0.4, 'gridFile': softGridFile,
                   'flag_form' : flag_form, 'probeFile': probeFile}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + softGridFile +  " is ", status)

        gridSet = {'emax': 3, 'maxe': 30, 'mine': -30, 'gridFile': hardGridFile}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + hardGridFile +  " is ", status)

        gridSet = {'emax': 100, 'maxe': 100, 'mine': -100, 'gridFile': nativeGridFile}
        genGrid.setVar(gridSet)
        status = genGrid.generate()
        print("Grid generation for " + nativeGridFile +  " is ", status)

    ## Fast initial placement
    if flag_fast_placement :
        FCDOCKER_fast_init_place(receptorPDB = receptorPDB, receptorPSF = receptorPSF,
                                 ligPDB = ligPDB, ligSeg = ligSeg, placementDir = placementDir,
                                 exhaustiveness = exhaustiveness, num = num, copy = copy)

    ## Prepare protein explicit atoms
    if psf.get_natom() > 0: psf.delete_atoms()
    read.psf_card(receptorPSF, append = True)
    read.pdb(receptorPDB, resid = True)

    for idx, row in flexchain.iterrows():
        resid = str(row['res_id'])
        segid = row['seg_id']
        tmpI = pycharmm.SelectAtoms().by_res_id(res_id = resid)
        tmpJ = pycharmm.SelectAtoms().by_seg_id(seg_id = segid)
        if idx == 0:
            flexSideChain = tmpI & tmpJ
        else:
            flexSideChain = flexSideChain | (tmpI & tmpJ)
    settings.set_bomb_level(-1)
    psf.delete_atoms(flexSideChain.__invert__())
    settings.set_bomb_level(0)

    ## Read in ligand
    read.sequence_pdb(ligPDB)
    settings.set_bomb_level(-1)        ## for 3-mem ring
    gen.new_segment(seg_name = ligSeg)
    settings.set_bomb_level(0)
    read.pdb(ligPDB, resid = True)
    ligand = pycharmm.SelectAtoms().by_seg_id(ligSeg)

    ## Declare receptor(protein) backbone
    backboneAtom = ['N', 'CA', 'C', 'O', 'HA', 'HN', 'HA1', 'HA2',
                    'HT1', 'HT2', 'HT3', 'OT1', 'OT2']
    backboneAtom = np.asarray(backboneAtom)
    for idx in np.arange(len(backboneAtom)):
        atom = backboneAtom[idx]
        if idx == 0: backbone = pycharmm.SelectAtoms().by_atom_type(atom)
        backbone = backbone | pycharmm.SelectAtoms().by_atom_type(atom)

    ## Define fix part and flex part of the system
    fixsc = ligand.__invert__() & backbone
    flexsc = fixsc.__invert__()
    receptor = ligand.__invert__()
    cons_fix.setup(fixsc)
    write.coor_card(receptorCard, selection = receptor, title = '''Title
                    * Receptor Side Chain Initial Position''')

    ## Update nonbond interactions
    update_nonbond = pycharmm.UpdateNonBondedScript(
      atom = True,
      switch = True,
      vswitch = True,
      vdwe = True,
      elec = True,
      rdie = True,
      cutnb = 12,
      ctofnb = 10,
      ctonnb = 8,
      emax = 10000,
      maxe = 10000,
      mine = -10000,
      epsilon = 3).run()

    ## Coor stats
    allCoor = coor.get_positions()
    allCoor['atomIndex'] = np.arange(np.shape(list(ligand))[0])
    ligand_xyz = allCoor.loc[np.asarray(list(ligand)), ['x', 'y', 'z']]
    receptor_xyz = allCoor.loc[np.asarray(list(receptor)), ['x', 'y', 'z', 'atomIndex']]

    ligStat = coor.stat(selection = ligand)
    xcen = ligStat['xave']
    ycen = ligStat['yave']
    zcen = ligStat['zave']

    ## Set up the OpenMM system
    ommd = grid.OMMD()
    ommdSet = {'softGridFile': softGridFile, 'hardGridFile': hardGridFile,
               'fix_select': fixsc, 'flex_select': flexsc,
               'flag_form' : flag_form, 'numCopy': num * copy}
    ommd.setVar(ommdSet)
    ommd.create()
    print("OMMD set up for flexible docking")
    print("Flexible docking for the first generation")

    ## OpenMM docking for initial generation
    if flag_suppress_print:
        settings.set_verbosity(1)
        settings.set_warn_level(1)
        settings.set_bomb_level(-1)
    if not flag_fast_placement :
        FCDOCKER_init_place(receptorCard= receptorCard, receptorSel = receptor, flexSel = flexsc,
                            ligSel = ligand, ligPDB = ligPDB, ligSeg = ligSeg,
                            hardGridFile = hardGridFile, nativeGridFile = nativeGridFile,
                            placementDir = placementDir, num = num, copy = copy,
                            threshold = threshold_init, flag_form = flag_form)

    idxCopy = 1
    while idxCopy <= num * copy:
        read.pdb(placementDir + str(idxCopy) + '.pdb', resid = True)
        ommd.set_coor(idxCopy = idxCopy)
        print("Set OMMD ligand copy " + str(idxCopy))
        idxCopy += 1

    default_ommd_sian(ommd)
    _rm_('[0-9]*')

    hardSet = {'selection': flexsc, 'flag_form' : flag_form, 'gridFile': hardGridFile}
    hardGrid = grid.CDOCKER()
    hardGrid.setVar(hardSet)
    hardGrid.read()
    energy.show()
    idxCopy = 1
    _mkdir_(saveLig + ' ' + saveProt)
    dockEner = []
    while idxCopy <= num * copy:
        ommd.copy_coor(idxCopy = idxCopy)
        print("Grid minimize OMMD ligand copy " + str(idxCopy))
        minimize.run_sd(nstep = 50)
        minimize.run_abnr(nstep = 1000, tolenr = 1e-3)
        energy.show()
        totalener = energy.get_total()
        write.coor_pdb(str(idxCopy), selection = ligand)
        write.coor_pdb(saveLig + str(idxCopy) + '.pdb',
                       selection = ligand, title = '''Title
                       * The docked pose ID is %d
                       * The total energy is %8.6f '''
                       % (idxCopy, totalener))
        write.coor_pdb(saveProt + str(idxCopy) + '.pdb',
                       selection = receptor, title = '''Title
                       * The docked pose ID is %d
                       * The total energy is %8.6f '''
                       % (idxCopy, totalener))
        dockEner.append(totalener)
        idxCopy += 1

    hardGrid.off()
    hardGrid.clear()
    dock_result = np.zeros((num * copy, 2))
    dock_result[:, 0] = np.asarray(dockEner)
    dock_result[:, 1] = np.arange(num * copy) + 1
    settings.set_verbosity(5)
    settings.set_warn_level(5)
    settings.set_bomb_level(0)

    ## Loop through following generations
    idxGen = 2
    radius = 1
    n_cluster = 0
    while n_cluster <= 0:
        cluster_mmtsb(radius = radius, name = '[0-9]*')
        n_cluster = _fcdocker_cluster_check_(total = num * copy)
        radius += 0.5
    cluster_result = top_N_cluster(N = top_N_result, total = num * copy)

    while idxGen <= generation:
        print("Flexible docking for generation: " + str(idxGen))
        if flag_suppress_print:
            settings.set_verbosity(1)
            settings.set_warn_level(1)
            settings.set_bomb_level(-1)

	## Crossover
        final_result, pair_list = FCDOCKER_crossover(cluster_result, dock_result, num = num, copy = copy)
        _mkdir_(crossoverLig + ' ' + crossoverProt)
        for idx in np.arange(num * copy) + 1:
            ## Copy ligand
            ligandPose = saveLig + str(pair_list[idx - 1, 0]) + ".pdb"
            placement = crossoverLig + str(idx) + ".pdb"
            _cp_(ligandPose, placement)

            ## Copy protein
            proteinPose = saveProt + str(pair_list[idx - 1, 1]) + ".pdb"
            placement = crossoverProt + str(idx) + ".pdb"
            _cp_(proteinPose, placement)

        ## OpenMM docking for second generation
        FCDOCKER_mutation(final_result, pair_list, receptorSel = receptor, flexSel = flexsc,
                          ligSel = ligand, ligPDB = ligPDB, ligSeg = ligSeg,
                          saveLig = saveLig, saveProt = saveProt,
                          crossoverLig = crossoverLig, crossoverProt = crossoverProt,
                          hardGridFile = hardGridFile,  nativeGridFile = nativeGridFile,
                          placementDir = placementDir, num = num, copy = copy,
                          threshold = threshold_mutate, flag_form = flag_form)

        idxCopy = 1
        while idxCopy <= num * copy:
            read.pdb(placementDir + str(idxCopy) + '.pdb', resid = True)
            ommd.set_coor(idxCopy = idxCopy)
            print("Set OMMD ligand copy " + str(idxCopy))
            idxCopy += 1
        default_ommd_sian(ommd)

        _rm_('[0-9]* cluster.log cluster_list')

        hardSet = {'selection': flexsc, 'flag_form' : flag_form, 'gridFile': hardGridFile}
        hardGrid = grid.CDOCKER()
        hardGrid.setVar(hardSet)
        hardGrid.read()
        energy.show()
        idxCopy = 1
        _mkdir_(saveLig + ' ' + saveProt)
        dockEner = []
        while idxCopy <= num * copy:
            ommd.copy_coor(idxCopy = idxCopy)
            print("Grid minimize OMMD ligand copy " + str(idxCopy))
            minimize.run_sd(nstep = 50)
            minimize.run_abnr(nstep = 1000, tolenr = 1e-3)
            energy.show()
            totalener = energy.get_total()
            write.coor_pdb(str(idxCopy), selection = ligand)
            write.coor_pdb(saveLig + str(idxCopy) + '.pdb',
                           selection = ligand, title = '''Title
                           * The docked pose ID is %d
                           * The total energy is %8.6f '''
                           % (idxCopy, totalener))
            write.coor_pdb(saveProt + str(idxCopy) + '.pdb',
                           selection = receptor, title = '''Title
                           * The docked pose ID is %d
                           * The total energy is %8.6f '''
                           % (idxCopy, totalener))
            dockEner.append(totalener)
            idxCopy += 1

        hardGrid.off()
        hardGrid.clear()
        dock_result = np.zeros((num * copy, 2))
        dock_result[:, 0] = np.asarray(dockEner)
        dock_result[:, 1] = np.arange(num * copy) + 1
        settings.set_verbosity(5)
        settings.set_warn_level(5)
        settings.set_bomb_level(0)

	## Cluster docked poses
        radius = 1
        n_cluster = 0
        while n_cluster <= 0:
            cluster_mmtsb(radius = radius, name = '[0-9]*')
            n_cluster = _fcdocker_cluster_check_(total = num * copy)
            radius += 0.5
        if idxGen < generation:
            cluster_result = top_N_cluster(N = top_N_result, total = num * copy)
        else:
            cluster_result = top_N_cluster(N = top_N_result + 5, total = num * copy)

        idxGen += 1

    ## Clear OMMD system
    ommd.clear()

    ## Analyze clustering result after final generation is done
    settings.set_verbosity(1)
    settings.set_warn_level(1)
    settings.set_bomb_level(-1)
    entropy, clusterID, sc_entropy = FCDOCKER_calc_dihedral(flexchain, cluster_result,
                                     saveProt, protein_dihedral)
    settings.set_verbosity(5)
    settings.set_warn_level(5)
    settings.set_bomb_level(0)
    cluster_size = []
    for cluster in np.unique(cluster_result[:, 1]):
        cluster_size.append(len(cluster_result[cluster_result[:, 1] == cluster]))

    ## Get list of pdb needed for all atom explicit minimization
    if flag_fast_grid:
        pdbID = np.ones(len(clusterID))
        for cluster in clusterID:
            tmpEner = []
            pdbIdx = cluster_result[cluster_result[:, 1] == cluster][:, 0]
            for pdb in pdbIdx:
                tmpEner.append(dock_result[dock_result[:, 1] == pdb][:, 0])
            cResult = pd.DataFrame()
            cResult['PDB'] = np.asarray(pdbIdx)
            cResult['Energy'] = np.asarray(tmpEner)
            cResult = cResult.sort_values(by = ['Energy'], ignore_index = True)
            pdbID[cluster - 1] = cResult.iloc[0]['PDB']
        pdbID = pdbID.astype(int)
    else:
        tmp_entropy = []
        for cluster in clusterID:
            tmp = np.ones(cluster_size[cluster - 1]) * entropy[cluster - 1]
            tmp_entropy = tmp_entropy + tmp.tolist()
        pdbID = cluster_result[:, 0]
        clusterID = cluster_result[:, 1]
        pdbID = pdbID.astype(int)
        entropy = np.asarray(tmp_entropy)

    ## Read in the explicit protein and ligand
    cons_fix.turn_off()
    psf.delete_atoms()
    read.psf_card(receptorPSF, append = True)
    read.pdb(receptorPDB, resid = True)
    read.sequence_pdb(ligPDB)
    settings.set_bomb_level(-1)        ## for 3-mem ring
    gen.new_segment(seg_name = ligSeg)
    settings.set_bomb_level(0)
    read.pdb(ligPDB, resid = True)
    for idx, row in flexchain.iterrows():
        resid = str(row['res_id'])
        segid = row['seg_id']
        tmpI = pycharmm.SelectAtoms().by_res_id(res_id = resid)
        tmpJ = pycharmm.SelectAtoms().by_seg_id(seg_id = segid)
        if idx == 0:
            flexSideChain = tmpI & tmpJ
        else:
            flexSideChain = flexSideChain | (tmpI & tmpJ)
    ligand = pycharmm.SelectAtoms().by_seg_id(ligSeg)
    flexPart = flexSideChain | ligand
    fixPart = flexPart.__invert__()

    ## Minimize docked pose with explicit protein atoms
    vdw = []
    elec = []
    totalEner = []
    _mkdir_('tmpligand/ tmpprot/')
    cons_fix.setup(fixPart)
    if flag_suppress_print:
        settings.set_verbosity(1)
        settings.set_warn_level(1)
        settings.set_bomb_level(-1)

    for idx in np.arange(len(pdbID)):
        read.pdb(saveLig + str(pdbID[idx]) + ".pdb", resid = True)
        read.pdb(saveProt + str(pdbID[idx]) + ".pdb", resid = True)
        minimize.run_sd(nstep = 50)
        minimize.run_abnr(nstep = 1000, tolenr = 0.001)
        tmptotal = energy.get_total() + entropy[idx]
        tmpelec = energy.get_elec()
        tmpvdw = energy.get_vdw()
        totalEner.append(tmptotal)
        elec.append(tmpelec)
        vdw.append(tmpvdw)

        write.coor_pdb('tmpligand/' + str(pdbID[idx]) + '.pdb',
                       selection = ligand, title = '''Title
                       * The docked pose ID is %d
                       * The cluster ID is %d
                       * The total energy is %8.6f
                       * The vdw energy is %8.6f
                       * The elec energy is %8.6f
                       * The entropy is %8.6f '''
                       % (pdbID[idx], clusterID[idx], tmptotal,
                       tmpvdw, tmpelec, entropy[idx]))

        write.coor_pdb('tmpprot/' + str(pdbID[idx]) + '.pdb',
                       selection = flexSideChain, title = '''Title
                       * The docked pose ID is %d
                       * The cluster ID is %d
                       * The total energy is %8.6f
                       * The vdw energy is %8.6f
                       * The elec energy is %8.6f
                       * The entropy is %8.6f '''
                       % (pdbID[idx], clusterID[idx], tmptotal,
                       tmpvdw, tmpelec, entropy[idx]))

    settings.set_verbosity(5)
    settings.set_warn_level(5)
    settings.set_bomb_level(0)

    ## Save docking result
    clusterResult = pd.DataFrame()
    if flag_fast_grid:
        clusterResult["total_energy"] = totalEner
        clusterResult["enthalpy"] = np.asarray(totalEner) - entropy
        clusterResult["vdw"] = vdw
        clusterResult["elec"] = elec
        clusterResult["entropy"] = entropy
        clusterResult["cluster_size"] = cluster_size
        clusterResult["PDB_name"] = pdbID
        clusterResult["cluster_id"] = clusterID
    else:
        explicitResult = pd.DataFrame()
        explicitResult["total_energy"] = totalEner
        explicitResult["enthalpy"] = np.asarray(totalEner) - entropy
        explicitResult["vdw"] = vdw
        explicitResult["elec"] = elec
        explicitResult["entropy"] = entropy
        explicitResult["PDB_name"] = pdbID
        explicitResult["cluster_id"] = cluster_result[:, 1]

        vdw = []
        elec = []
        pdbID = []
        entropy = []
        totalEner = []
        clusterID = np.unique(clusterID)
        for cluster in clusterID:
            tmp_result = explicitResult.loc[explicitResult['cluster_id'] == cluster]
            tmp_result = tmp_result.sort_values(by = ['enthalpy'], ignore_index = True)
            pdbID.append(tmp_result.iloc[0]['PDB_name'])
            tmp_result = tmp_result.to_numpy()

            ## Get probability and ensemble average
            enthalpy = tmp_result[:, 1]
            enthalpy = enthalpy - np.amin(enthalpy)
            prob = - (1 / 0.593) * enthalpy
            prob = np.exp(prob)
            prob = prob / np.sum(prob)
            tmptotal = np.sum(tmp_result[:, 1] * prob)
            tmpvdw = np.sum(tmp_result[:, 2] * prob)
            tmpelec = np.sum(tmp_result[:, 3] * prob)
            tmpentropy = np.sum(tmp_result[:, 4] * prob)
            tmpcluster = np.sum(tmp_result[:, 6] * prob)

            vdw.append(tmpvdw)
            elec.append(tmpelec)
            entropy.append(tmpentropy)
            totalEner.append(tmptotal + tmpentropy)

        pdbID = np.asarray(pdbID)
        pdbID = pdbID.astype(int)
        clusterResult["total_energy"] = totalEner
        clusterResult["enthalpy"] = np.asarray(totalEner) - np.asarray(entropy)
        clusterResult["vdw"] = vdw
        clusterResult["elec"] = elec
        clusterResult["entropy"] = entropy
        clusterResult["cluster_size"] = cluster_size
        clusterResult["PDB_name"] = pdbID
        clusterResult["cluster_id"] = clusterID

    dockResult = pd.DataFrame()
    dockResult["enthalpy"] = dock_result[:, 0]
    dockResult["PDB_name"] = dock_result[:, 1].astype(int)
    dockResult = dockResult.sort_values(by = ["enthalpy"], ignore_index = True)

    ## Sort dataframe and save results
    clusterResult = clusterResult.sort_values(by = [sort_energy], ignore_index = True)
    pdbList = clusterResult['PDB_name'].tolist()
    idx = 1
    _mkdir_(saveLigFinal + ' ' + saveProtFinal)
    for pdb in pdbList:
        ## Copy Ligand
        source = 'tmpligand/' + str(pdb) + '.pdb'
        target = saveLigFinal + 'top_' + str(idx) + '.pdb'
        _cp_(source, target)

        ## Copy protein
        source = 'tmpprot/' + str(pdb) + '.pdb'
        target = saveProtFinal + 'top_' + str(idx) + '.pdb'
        _cp_(source, target)

        idx += 1

    ## Clean and save results
    _mkdir_(saveDir)
    _mv_('cluster.log', saveDir)
    clusterResult.to_csv(saveDir + 'clusterResult.tsv', sep = '	')
    dockResult.to_csv(saveDir + 'dockResult.tsv', sep = '	')

    if not flag_fast_grid:
        explicitResult = explicitResult.sort_values(by = [sort_energy], ignore_index = True)
        explicitResult.to_csv(saveDir + 'explicitResult.tsv', sep = '	')
    if flag_save_placement: _mv_(placementDir, saveDir)
    if flag_save_all:
        _mkdir_(saveDir + 'dock_pose/')
        _mv_(saveLig + ' ' + saveProt, saveDir + 'dock_pose/')
    if flag_save_cluster:
        _mkdir_(saveDir + 'cluster/')
        _mv_(saveLigFinal, saveDir + 'cluster/ligand/')
        _mv_(saveProtFinal, saveDir + 'cluster/protein/')
    if flag_save_crossover and generation > 1:
        _mkdir_(saveDir + 'crossover/')
        _mv_(crossoverLig, saveDir + 'crossover/')
        _mv_(crossoverProt, saveDir + 'crossover/')
    if flag_delete_grid: _rm_(nativeGridFile + ' ' + softGridFile + ' ' + hardGridFile)

    _rm_(saveLig + ' ' + saveProt + ' ' + saveLigFinal + ' ' + saveProtFinal + ' ' + crossoverLig + ' ' + crossoverProt)
    _rm_('cluster* [0-9]* tmp* *crd ligand_rotamer.pdb ' + placementDir)

    return clusterResult, dockResult
