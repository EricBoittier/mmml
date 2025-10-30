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

"""Functions for CHARMM grid and docking calculation 

Created by Yujin Wu (wyujin@umich.edu)

See CHARMM documentation [Grid](<https://academiccharmm.org/documentation/version/c47b1/grid>)
and [fftdock](<https://academiccharmm.org/documentation/version/c47b1>)
for more information

Functions
=========
- `generate_with_pyopencl` -- generate grids for a given structure
"""

import ctypes
import numpy as np
#import pyopencl as cl

import pycharmm
import pycharmm.lib as lib
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.coor as coor
import pycharmm.param as param
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.energy as energy
import pycharmm.cons_fix as cons_fix
import pycharmm.charmm_file as charmm_file

class Grid(ctypes.Structure):
    """Settings for python Class Grid 

    Attributes
    ----------
    xCen : float
        x-axis center of the grid box
    yCen : float
        y-axis center of the grid box
    zCen : float
        z-axis center of the grid box
    xMax : float
        length of the grid box along x-axis 
    yMax : float 
        length of the grid box along y-axis 
    zMax : float
        length of the grid box along z-axis 
    rcta : float	
        left cutoff of the customizable grid
    rctb : float	
        right cutoff of the customizable grid
    hMax : float	
        welldepth of the customizable grid
    dGrid : float	
        grid space
    emax : float	
        VdwEmax 
    maxe : float	
        elecReplEmax
    mine : float	
        elecAttrEmax 
    dielec : float	
        dielectric constant 
    gridForce : float	
        grid edge force 
    probes : str
        probe segment ID
    gpuID : int
        gpu ID for gpu grid generation
    gridU : int	
        unit for access grid file
    probeU : int	
        unit for access probe file for gpu grid gen 
    flag_gpu : bool
        True for use GPU, False for CPU 
    flag_rdie : bool
        True for rdie, False for cdie
    flag_form : bool
        True for formatted output, False for binary
    flag_grhb : bool
        True for turn on customizable hbond grids
    gridFile : str
        grid file name
    probeFile : str
        probe file name
    nbond_opt : dict
        nbond option, use default setups

    """
    
    def __init__(self, 
                 xCen = 0.0, 
                 yCen = 0.0, 
                 zCen = 0.0,
                 xMax = 10.0,
                 yMax = 10.0,
                 zMax = 10.0,
                 rcta = 0.0,
                 rctb = 0.0,
                 hMax = 0.0,
                 dGrid = 0.5,
                 emax = 1.0,
                 maxe = 1.0,
                 mine = -1.0,
                 dielec = 3.0,
                 gridForce = 300.0,
                 probes = "PROB", 
                 gpuID = 0,
                 gridU = 1,
                 probeU = 2, 
                 flag_gpu = True, 
                 flag_rdie = True, 
                 flag_form = False, 
                 flag_grhb = False, 
                 gridFile = None,
                 probeFile = None,
                 nbond_opt= None):
        self.xCen = xCen
        self.yCen = yCen
        self.zCen = zCen
        self.xMax = xMax
        self.yMax = yMax
        self.zMax = zMax
        self.rcta = rcta
        self.rctb = rctb
        self.hMax = hMax
        self.dGrid = dGrid
        self.emax = emax
        self.maxe = maxe
        self.mine = mine
        self.dielec = dielec 
        self.gridForce = gridForce
        self.probes = probes
        self.gpuID = gpuID
        self.gridU = gridU
        self.probeU = probeU
        self.flag_gpu = flag_gpu
        self.flag_rdie = flag_rdie
        self.flag_form = flag_form
        self.flag_grhb = flag_grhb
        self.gridFile = gridFile
        self.probeFile = probeFile
        self.nbond_opt = nbond_opt

    def setVar(self, var):
        """Check and set up the variables

        Parameters
        ----------
        var : python built-in types

        Returns
        -------
        AttributeError
            if the variables is not decleard in the Grid class. 
        """
        for key, value in var.items():
            try:
                getattr(self, key)
                setattr(self, key, value)
            except AttributeError:
                print("Please check settings, ", 
                "Grid Object has no attribute " + str(key))

    def generate(self):
        """Generate grids for docking

        Returns
        -------
        int
            one indicates success, any other value indicates failure

        """
        status = False
        if self.gridFile is None:
            if self.flag_form:
                default_gridFile = "grid-emax-" + str(self.emax) + "-mine-" + str(self.mine) + "-maxe-" + str(self.maxe) + ".txt" 
            else:
                default_gridFile = "grid-emax-" + str(self.emax) + "-mine-" + str(self.mine) + "-maxe-" + str(self.maxe) + ".bin" 
        else:
            default_gridFile = self.gridFile

        ## Convert general data to c_type 
        c_gridU = ctypes.c_int(self.gridU)
        c_xcen = ctypes.c_double(self.xCen)
        c_ycen = ctypes.c_double(self.yCen)
        c_zcen = ctypes.c_double(self.zCen)
        c_xmax = ctypes.c_double(self.xMax)
        c_ymax = ctypes.c_double(self.yMax)
        c_zmax = ctypes.c_double(self.zMax)
        c_rcta = ctypes.c_double(self.rcta)
        c_rctb = ctypes.c_double(self.rctb)
        c_hmax = ctypes.c_double(self.hMax)
        c_dgrid = ctypes.c_double(self.dGrid)
        c_gridForce = ctypes.c_double(self.gridForce)
        c_grhbon = (ctypes.c_bool)(self.flag_grhb)
        c_form = (ctypes.c_bool)(self.flag_form)

        ## Open grid file to access
        grid_file = charmm_file.CharmmFile(file_name = default_gridFile,
                    file_unit = self.gridU, formatted = self.flag_form,
                    read_only = False)
        grid_file.open()

        ## Decide whether or use GPU or CPU to perform grid generation
        try:
            if callable(lib.charmm.grid_fftgen):
                if self.flag_gpu:
                    print("Use GPU to generate grids")
                    tmp_flag_gpu_gen = True
                else:
                    print('''GPU grid generation function is ready, \n
                    but not selected. Use CPU instead''')
                    tmp_flag_gpu_gen = False
        except AttributeError:
            print('''GPU grid generation function is not callable. \n
            Use CPU to generate Grids''')
            tmp_flag_gpu_gen = False 

        ## Set up probes if using CPU, and update non-bond lists
        if not tmp_flag_gpu_gen:
            tmp = pycharmm.SelectAtoms(seg_id = self.probes)
            if np.sum(np.asarray(list(tmp))) > 0: psf.delete_atoms(tmp)

            read.sequence_string(self.probes)
            gen.new_segment(seg_name = self.probes)
            probe_selection = pycharmm.SelectAtoms(seg_id = self.probes)
            probe_selection = np.asarray(list(probe_selection))
            xyz = coor.get_positions()
            xyz.loc[probe_selection, 'x'] = self.xCen
            xyz.loc[probe_selection, 'y'] = self.yCen
            xyz.loc[probe_selection, 'z'] = self.zCen
            coor.set_positions(xyz)

            fix_atom = pycharmm.SelectAtoms(seg_id = self.probes).__invert__()
            cons_fix.setup(fix_atom)
            lingo.charmm_script('skipe all excl vdw elec')
            energy.show()

        if self.nbond_opt is None:
            print("Use default non-bond set up")
            if self.flag_rdie:
                nbonds_script = pycharmm.UpdateNonBondedScript(
                  atom = True, switch = True, vswitch = True, soft = True,
                  vdwe = True, elee = True, rdie = True, cutnb = 999, ctofnb = 999,
                  ctonnb = 999, emax = self.emax, mine = self.mine, maxe = self.maxe,
                  epsilon = self.dielec).run()
            else: 
                nbonds_script = pycharmm.UpdateNonBondedScript(
                  atom = True, switch = True, vswitch = True, soft = True,
                  vdwe = True, elee = True, cdie = True, cutnb = 999, ctofnb = 999,
                  ctonnb = 999, emax = self.emax, mine = self.mine, maxe = self.maxe,
                  epsilon = self.dielec).run()
        else:
            print("Use user-defined non-bond set up")
            nbonds_script = pycharmm.UpdateNonBondedScript(**self.nbond_opt).run() 
            
        ## Generate grids
        if tmp_flag_gpu_gen:
            ## Prepare
            if self.probeFile is None:
                raise AttributeError('Probe file is not defined. Use function setVar() to declear file name first')
            probe_file = charmm_file.CharmmFile(file_name = self.probeFile,
                         file_unit = self.probeU, formatted = True)
            probe_file.open()
   
            ## Get system information
            Islct = np.ones(psf.get_natom())
            Islct = Islct.astype(int)
   
            ## Convert data to c_type 
            c_emax = ctypes.c_double(self.emax)
            c_maxe = ctypes.c_double(self.maxe)
            c_mine = ctypes.c_double(self.mine)
            c_gpuID = ctypes.c_int(self.gpuID)
            c_probeU = ctypes.c_int(self.probeU)
            c_natom = ctypes.c_int(len(Islct))
            c_dielec = ctypes.c_double(self.dielec)
            c_islct = (ctypes.c_int * len(Islct))(*Islct)
            c_rdie = (ctypes.c_bool)(self.flag_rdie)
   
            ## Grid calculation with CHARMM library (api_grid.F90)
            status = lib.charmm.grid_fftgen(ctypes.byref(c_islct), 
                                            ctypes.byref(c_rdie), 
                                            ctypes.byref(c_dielec),
                                            ctypes.byref(c_natom), 
                                            ctypes.byref(c_gridForce), 
                                            ctypes.byref(c_xcen), 
                                            ctypes.byref(c_ycen), 
                                            ctypes.byref(c_zcen), 
                                            ctypes.byref(c_xmax), 
                                            ctypes.byref(c_ymax), 
                                            ctypes.byref(c_zmax), 
                                            ctypes.byref(c_dgrid), 
                                            ctypes.byref(c_rcta), 
                                            ctypes.byref(c_rctb), 
                                            ctypes.byref(c_hmax), 
                                            ctypes.byref(c_emax), 
                                            ctypes.byref(c_maxe), 
                                            ctypes.byref(c_mine), 
                                            ctypes.byref(c_gridU),
                                            ctypes.byref(c_probeU),
                                            ctypes.byref(c_grhbon),
                                            ctypes.byref(c_form),
                                            ctypes.byref(c_gpuID))
            probe_file.close()
            status = bool(status)

        else:
            ## Get system information
            probe_atoms = pycharmm.SelectAtoms(seg_id = self.probes)
            probe_atoms = np.asarray(list(probe_atoms))
            probes = np.asarray(param.get_vdwr())
            probes = probes[probe_atoms]
            numProbes = len(probes)
  
            if self.flag_grhb:
                numGrid = numProbes + 3
            else:
                numGrid = numProbes + 1
  
            Islct = np.ones(psf.get_natom())
            Islct = Islct * probe_atoms
            Islct = Islct.astype(int)
            Jslct = Islct
            GridSlct = np.where(Islct == 1)[0] + 1
            GridSlct = GridSlct.astype(int)
  
            ## Convert data to c_type 
            c_ngrid = ctypes.c_int(numGrid)
            c_natom = ctypes.c_int(len(Islct))
            c_numProbes = ctypes.c_int(numProbes)
            c_islct = (ctypes.c_int * len(Islct))(*Islct)
            c_jslct = (ctypes.c_int * len(Islct))(*Islct)
            c_gridslct = (ctypes.c_int * len(GridSlct))(*GridSlct)
            c_gridRadii = (ctypes.c_double * len(probes))(*probes)

            ## Grid calculation with CHARMM library (api_grid.F90)
            status = lib.charmm.grid_generate(ctypes.byref(c_islct), 
                                              ctypes.byref(c_jslct), 
                                              ctypes.byref(c_gridslct), 
                                              ctypes.byref(c_natom), 
                                              ctypes.byref(c_numProbes), 
                                              ctypes.byref(c_gridRadii), 
                                              ctypes.byref(c_gridForce), 
                                              ctypes.byref(c_xcen), 
                                              ctypes.byref(c_ycen), 
                                              ctypes.byref(c_zcen), 
                                              ctypes.byref(c_xmax), 
                                              ctypes.byref(c_ymax), 
                                              ctypes.byref(c_zmax), 
                                              ctypes.byref(c_dgrid), 
                                              ctypes.byref(c_rcta), 
                                              ctypes.byref(c_rctb), 
                                              ctypes.byref(c_hmax), 
                                              ctypes.byref(c_ngrid), 
                                              ctypes.byref(c_gridU),
                                              ctypes.byref(c_grhbon),
                                              ctypes.byref(c_form))
            status = bool(status)

        ## Generation result
        grid_file.close()
        lib.charmm.grid_clear()
        return status

class CDOCKER(Grid):
    """Settings for python Class CDOCKER

    """

    def __init__(self, selection = None):
        """
        Parameters
        ----------
        selection: pycharmm.SelectAtoms
            selection of atoms that are used for grid energy calculation
        """
        self.selection = selection
        super().__init__('CDOCKER')

    def read(self): 
        """Read grid on selected atoms

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        ## Check atom selection 
        if self.selection is None:
            raise AttributeError("CDOCKER.read() atom selection is required for interaction with grid. Use setVar()")
        else:
            print("Grid is applied to specified atoms")

        ## Check file name
        if self.gridFile is None:
            raise AttributeError('Grid file is not defined. Use function setVar() to declear file name first')

        ## Open grid file to access
        grid_file = charmm_file.CharmmFile(file_name = self.gridFile,
                    file_unit = self.gridU, formatted = self.flag_form)
        grid_file.open()

        ## Prepare atom selection 
        ligand_atoms = np.asarray(list(self.selection))
        GridSlct = np.ones(psf.get_natom())
        GridSlct = GridSlct * ligand_atoms
        GridSlct = GridSlct.astype(int)
        GridAtm = GridSlct
        GridHBAtm = np.zeros(len(GridSlct)).astype(int)
        Natom = len(GridSlct)
        NAtmGrd = np.sum(ligand_atoms)
 
        ## Input variables
        c_natom = ctypes.c_int(Natom)
        c_gridU = ctypes.c_int(self.gridU)
        c_NAtmGrd = ctypes.c_int(NAtmGrd)
        c_gridForm = (ctypes.c_bool)(self.flag_form)
        c_gridHBon = (ctypes.c_bool)(self.flag_grhb)
        c_gridSlct = (ctypes.c_int * len(GridSlct))(*GridSlct)
 
        ## Use CHARMM API function to read in the grid 
        status = lib.charmm.grid_read(ctypes.byref(c_natom),
                                      ctypes.byref(c_NAtmGrd),
                                      ctypes.byref(c_gridU),
                                      ctypes.byref(c_gridSlct),
                                      ctypes.byref(c_gridHBon),
                                      ctypes.byref(c_gridForm))
 
        ## Pass results to python variables
        grid_file.close()
        status = bool(status) 
        return status 

    def on(self):
        """Turn on grid energy calculation

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        ## Prepare 
        ligand_atoms = np.asarray(list(self.selection))
        GridSlct = np.ones(psf.get_natom())
        GridSlct = GridSlct * ligand_atoms
        GridSlct = GridSlct.astype(int)
        GridAtm = GridSlct
        GridHBAtm = np.zeros(len(GridSlct)).astype(int)
        Natom = len(GridSlct)
        NAtmGrd = np.sum(ligand_atoms)
 
        ## Input variables
        c_natom = ctypes.c_int(Natom)
        c_NAtmGrd = ctypes.c_int(NAtmGrd)
        c_gridHBon = (ctypes.c_bool)(self.flag_grhb)
        c_gridSlct = (ctypes.c_int * len(GridSlct))(*GridSlct)
 
        ## Use CHARMM API function to set grid on
        status = lib.charmm.grid_on(ctypes.byref(c_gridSlct),
                                    ctypes.byref(c_gridHBon),
                                    ctypes.byref(c_NAtmGrd),
                                    ctypes.byref(c_natom))
        status = bool(status)
        return status 

    def off(self):
        """Turn off grid energy calculation

        Returns
        -------
        status : int
            one indicates success, any other value indicates failure
        """
        status = lib.charmm.grid_off()
        status = bool(status)
        return status 

    def clear(self):
        """Clear grid, free memory allocation for grid calculation

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        status = lib.charmm.grid_clear()
        status = bool(status)
        return status 

class OMMD(Grid):
    """Settings for python Class OMMD 

    Attributes
    ----------
    numCopy : int
        Number of copies in parallel simulated annealing
    softGridUnit : int
        unit for access soft grid file
    hardGridUnit : int 
        unit for access hard grid file
    softGridFile : str
        soft grid file name
    hardGridFile : str
        hard grid file name
    flag_form : bool
        True for formatted output, false for binary
    flag_grhb : bool
        True for turn on customizable hbond grids
    fix_select : list?
        Selection of fixed atoms
    flex_select : list?
        Selection of flexible atoms
    soft : float
        Percentage of soft grid potential 
    hard : float
        Percentage of hard grid potential
    emax : float
        VdwEmax 
    mine : float
        elecAttrEmax 
    maxe : float
        elecReplEmax
    eps : float
        dielectric constant 
    steps : int	
        number of simulation steps 
    heatFrq : int
        heat frequence 
    startTemp : float
        starting temperatue 
    endTemp : float
        end temperature
    incrTemp : float
        increasement of temperature 
    """
    def __init__(self, 
                 numCopy = 1,
                 softGridUnit = 1,
                 hardGridUnit = 2,
                 softGridFile = None,
                 hardGridFile = None,
                 flag_form = False,
                 flag_grhb = False,
                 fix_select = None,
                 flex_select = None,
                 soft = 1.0, 
                 hard = 0.0, 
                 emax = 1.0, 
                 mine = -1.0, 
                 maxe = 1.0, 
                 eps = 1.0,
                 steps = 10000,
                 heatFrq = 50,
                 startTemp = 100,
                 endTemp = 500,
                 incrTemp = 1):

        ## OMMD system parameter
        self.numCopy = numCopy
        self.softGridUnit = softGridUnit
        self.hardGridUnit = hardGridUnit
        self.softGridFile = softGridFile
        self.hardGridFile = hardGridFile
        self.flag_form = flag_form
        self.flag_grhb = flag_grhb
        self.fix_select = fix_select
        self.flex_select = flex_select

        ## OMMD grid softness parameter
        self.soft = soft
        self.hard = hard
        self.emax = emax
        self.mine = mine
        self.maxe = maxe
        self.eps = eps

        ## OMMD simulated annealing parameter
        self.steps = steps
        self.heatFrq = heatFrq
        self.startTemp = startTemp
        self.endTemp = endTemp
        self.incrTemp = incrTemp

        ## Inhert Grid class
        super().__init__('OMMD')

    def create(self):
        """Create the OpenMM docking system

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        ## Prepare 
        natom = psf.get_natom()

        if self.fix_select == None:
            in_fix_select = np.zeros(natom).astype(int)
        else:
            fix_atoms = np.asarray(list(self.fix_select)) 
            in_fix_select = np.ones(natom) * fix_atoms
            in_fix_select = in_fix_select.astype(int)
            
        if self.flex_select is None:
            print("All atoms are considered flexible particles")
            self.flex_select = pycharmm.SelectAtoms().all_atoms()
        else:
            print("Specific atoms are considered flexible particles")
        flex_atoms = np.asarray(list(self.flex_select))
        in_flex_select = np.ones(natom) * flex_atoms
        in_flex_select = in_flex_select.astype(int)
 
        ## Open soft grid file and hard grid file
        if self.softGridFile is None:
            raise AttributeError('Soft grid file is not defined. Use function setVar() to declear file name first')
        soft_grid_file = charmm_file.CharmmFile(
                         file_name = self.softGridFile,
                         file_unit = self.softGridUnit, 
                         formatted = self.flag_form)
        soft_grid_file.open()

        if self.hardGridFile is None:
            raise AttributeError('Hard grid file is not defined. Use function setVar() to declear file name first')
        hard_grid_file = charmm_file.CharmmFile(
                         file_name = self.hardGridFile,
                         file_unit = self.hardGridUnit, 
                         formatted = self.flag_form)
        hard_grid_file.open()
      
        ## Convert data to c_type 
        c_natom = ctypes.c_int(natom)
        c_numCopy = ctypes.c_int(self.numCopy)
        c_form = (ctypes.c_bool)(self.flag_form)
        c_softU = ctypes.c_int(self.softGridUnit)
        c_hardU = ctypes.c_int(self.hardGridUnit)
        c_grhbon = (ctypes.c_bool)(self.flag_grhb)
        c_fixSelect = (ctypes.c_int * natom)(*in_fix_select)
        c_flexSelect = (ctypes.c_int * natom)(*in_flex_select)
 
        ## Grid calculation with CHARMM library (api_grid.F90)
        status = lib.charmm.ommd_create(ctypes.byref(c_softU), 
                                        ctypes.byref(c_hardU), 
                                        ctypes.byref(c_form), 
                                        ctypes.byref(c_grhbon), 
                                        ctypes.byref(c_natom), 
                                        ctypes.byref(c_fixSelect), 
                                        ctypes.byref(c_flexSelect), 
                                        ctypes.byref(c_numCopy))
        status = bool(status)
        return status 

    def energy(self):
        """Calculate and print OpenMM docking system energy

        Returns
        -------
        status: int
                 one indicates success, any other value indicates failure
        """
        c_grhbon = (ctypes.c_bool)(self.flag_grhb)
        status = lib.charmm.ommd_energy(ctypes.byref(c_grhbon))
        status = bool(status)
        return status 

    def set_coor(self, idxCopy):
        """Set OpenMM docking flex part atom positions

        Parameters
        ----------
        idxCopy: int
                  index of the OpenMM docking copy

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        c_idxCopy = ctypes.c_int(idxCopy) 
        status = lib.charmm.ommd_set_position(ctypes.byref(c_idxCopy))
        status = bool(status)
        return status 

    def copy_coor(self, idxCopy):
        """Copy OpenMM docking flex part atom positions to main coordinates

        Parameters
        ----------
        idxCopy: int
            index of the OpenMM docking copy

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        c_idxCopy = ctypes.c_int(idxCopy) 
        status = lib.charmm.ommd_copy_position(ctypes.byref(c_idxCopy))
        status = bool(status)
        return status 

    def change_softness(self):
        """Change grid softness

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        ## Prepare
        c_soft = ctypes.c_double(self.soft) 
        c_hard = ctypes.c_double(self.hard) 
        c_emax = ctypes.c_double(self.emax) 
        c_mine = ctypes.c_double(self.mine) 
        c_maxe = ctypes.c_double(self.maxe) 
        c_eps = ctypes.c_double(self.eps) 
 
        ## Call function
        status = lib.charmm.ommd_change_softness(ctypes.byref(c_soft),
                                                 ctypes.byref(c_hard),
                                                 ctypes.byref(c_emax),
                                                 ctypes.byref(c_mine),
                                                 ctypes.byref(c_maxe),
                                                 ctypes.byref(c_eps))
        status = bool(status)
        return status 

    def simulated_annealing(self):
        """Run OpenMM docking simulated annealing

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        ## Prepare
        c_steps = ctypes.c_int(self.steps)
        c_heatFrq = ctypes.c_int(self.heatFrq)
        c_endTemp = ctypes.c_double(self.endTemp) 
        c_incrTemp = ctypes.c_double(self.incrTemp) 
        c_startTemp = ctypes.c_double(self.startTemp) 
 
        ## Call function
        status = lib.charmm.ommd_sian(ctypes.byref(c_steps),
                                      ctypes.byref(c_heatFrq),
                                      ctypes.byref(c_startTemp),
                                      ctypes.byref(c_endTemp),
                                      ctypes.byref(c_incrTemp))
        status = bool(status)
        return status 

    def clear(self):
        """Clear the OpenMM docking system, free all memory allocation. 

        Returns
        -------
        status: int
            one indicates success, any other value indicates failure
        """
        natom = psf.get_natom()
        c_natom = ctypes.c_int(natom)
        c_grhbon = (ctypes.c_bool)(self.flag_grhb)
        status = lib.charmm.ommd_clear(ctypes.byref(c_grhbon),
	                               ctypes.byref(c_natom))
        status = bool(status)
        return status 

def _get_boxSize(center, length, dgrid):
    """calculate boxsize of the grid

   :param center: center of X/Y/Z axis
   :param length: length of X/Y/Z axis
   :param dgrid: grid spacing
   :return bool: true if successful
   :return np.array: np array of the X/Y/Z coordinates
    """
    if dgrid <= 0 or length / dgrid < 2:
        return False, None
    else:
        xmin = center - length / 2
        xmax = xmin + round(length / dgrid) * dgrid
        return True, np.linspace(xmin, xmax, int(round(length / dgrid) + 1))

def _vdwGrid(receptor, gridSpace, probes, vdwEmax):
    """ calculate vdw grids point energy

    receptor = [x-coor, y-coor, z-coor, rmin, eps]
    eParam = [r, r-min/r, eps_sqrt, vdw_constant, r-cutoff, beta]
    maxE --> maximum VDW grid energy

   :param receptor: receptor atom coordinates and corresponding charges
   :param gridSpace: grid point coordinates
   :param vdwEmax: maximum VDW grid point energy 
   :return np.array: grid point vdw energy 
    """

    ## Separate receptor atoms to attractive atom and repulsive atom
    eParam = np.zeros((len(receptor), 6))
    eParam[:, 1] = receptor[:, -2] + probes       ## Get rmin first
    eParam[:, 2] = np.sqrt(np.absolute(receptor[:, -1]))    ## eqs_sqrt
    eParam[:, 3] = 1 + np.sqrt(1 + 0.5 * vdwEmax / eParam[:, 2])  ## vdw_constant
    eParam[:, 4] = eParam[:, 1] / np.power(eParam[:, 3], 1/6)     ## r-cutoff
    eParam[:, 5] = 24 * eParam[:, 2] / vdwEmax * eParam[:, 3] * (eParam[:, 3] - 1) ## beta

    ## Set up variables for pyopencl calculation
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    nAtoms = np.int32(len(receptor))
    vdwEmax = np.float32(vdwEmax)
    x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 0].astype(np.float32))
    y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 1].astype(np.float32))
    z = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 2].astype(np.float32))

    rmin = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 1].astype(np.float32))
    eqs_sqrt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 2].astype(np.float32))
    rCutoff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 4].astype(np.float32))
    beta = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 5].astype(np.float32))

    gridX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 0].astype(np.float32))
    gridY = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 1].astype(np.float32))
    gridZ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 2].astype(np.float32))

    ## Set up kernels
    prg = cl.Program(ctx, """
    __kernel void elec(
      __global const float * gridX,
      __global const float * gridY,
      __global const float * gridZ,
      __global const float * x,
      __global const float * y,
      __global const float * z,
      __global const float * rmin,
      __global const float * eqs_sqrt,
      __global const float * rCutoff,
      __global const float * beta,
  int nAtoms,
  float vdwEmax,
  __global float *r,
  __global float *result)
    {
  // Local variables
  int atomIdx;
  int gid = get_global_id(0);
  float atomx, atomy, atomz;

  // Loop atoms
  for (atomIdx=0; atomIdx<nAtoms; ++atomIdx) {
    atomx = x[atomIdx];
    atomy = y[atomIdx];
    atomz = z[atomIdx];
    r[gid] = sqrt((atomx-gridX[gid])*(atomx-gridX[gid]) + (atomy-gridY[gid])*(atomy-gridY[gid]) + (atomz-gridZ[gid])*(atomz-gridZ[gid]));

    // Calculation based on cutoff distance
      if (r[gid] > rCutoff[atomIdx]) {
        result[gid] += eqs_sqrt[atomIdx] * (pow(rmin[atomIdx] / r[gid], 12.0f) - 2.0f * pow(rmin[atomIdx] / r[gid], 6.0f));
      } else {
        result[gid] += vdwEmax * (1.0f - 0.5f * pow(r[gid] / rCutoff[atomIdx], beta[atomIdx]));
      }
  }
    }
    """).build()

    ## Run kernels
    r = cl.Buffer(ctx, mf.WRITE_ONLY, gridSpace[:, 0].nbytes)
    result = cl.Buffer(ctx, mf.WRITE_ONLY, gridSpace[:, 0].nbytes)
    knl = prg.elec
    knl(queue, gridSpace[:, 0].shape, None, gridX, gridY, gridZ, x, y, z, rmin, eqs_sqrt, rCutoff, beta, 
    nAtoms, vdwEmax, r, result)

    ## Return results
    gridEner = np.empty_like(gridSpace[:, 0]).astype(np.float32)
    cl.enqueue_copy(queue, gridEner, result)
    return gridEner

def _elecGrid(receptor, gridSpace, elecReplEmax, elecAttrEmax, dielec):
    """ calculate electrostatic grid point energy

    receptor = [x-coor, y-coor, z-coor, charge]
    eParam = [r, electrostatic_constant, maxE, r-cutoff, alpha]
    maxE --> elecReplEmax for charge > 0; elecAttrEmax for charge < 0

   :param receptor: receptor atom coordinates and corresponding charges
   :param gridSpace: grid point coordinates
   :param elecReplEmax: maximum electrostatic repulsion energy
   :param elecAttrEmax: maximum electrostatic attractive energy
   :param dielec: distance dielectric constant
   :return np.array: grid point electrostatic energy 
    """

    ## Separate receptor atoms to attractive atom and repulsive atom
    CCELEC_charmm = 332.0716      ## !!! come back and check YW
    receptor = receptor[receptor[:, -1] != 0]
    eParam = np.zeros((len(receptor), 5))
    eParam[:, 1] = CCELEC_charmm * receptor[:, -1] / dielec
    eParam[:, 2][eParam[:, 1] > 0] = elecReplEmax
    eParam[:, 2][eParam[:, 1] < 0] = elecAttrEmax
    eParam[:, 3] = np.sqrt(2 * np.absolute(eParam[:, 1] / eParam[:, 2]))
    eParam[:, 4] = eParam[:, 2] / np.power(eParam[:, 3], 2) / 2
    
    ## Set up variables for pyopencl calculation
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    nAtoms = np.int32(len(receptor))
    x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 0].astype(np.float32))
    y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 1].astype(np.float32))
    z = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = receptor[:, 2].astype(np.float32))

    maxE = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 2].astype(np.float32))
    alpha = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 4].astype(np.float32))
    eConst = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 1].astype(np.float32))
    rCutoff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = eParam[:, 3].astype(np.float32))

    gridX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 0].astype(np.float32))
    gridY = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 1].astype(np.float32))
    gridZ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = gridSpace[:, 2].astype(np.float32))

    ## Set up kernels
    prg = cl.Program(ctx, """
    __kernel void elec(
      __global const float * gridX,
      __global const float * gridY,
      __global const float * gridZ,
      __global const float * x,
      __global const float * y,
      __global const float * z,
      __global const float * eConst,
      __global const float * maxE,
      __global const float * rCutoff,
      __global const float * alpha,
  int nAtoms,
  __global float *r,
  __global float *result)
    {
  // Local variables
  int atomIdx;
  int gid = get_global_id(0);
  float atomx, atomy, atomz;

  // Loop atoms
  for (atomIdx=0; atomIdx<nAtoms; ++atomIdx) {
    atomx = x[atomIdx];
    atomy = y[atomIdx];
    atomz = z[atomIdx];
    r[gid] = sqrt((atomx-gridX[gid])*(atomx-gridX[gid]) + (atomy-gridY[gid])*(atomy-gridY[gid]) + (atomz-gridZ[gid])*(atomz-gridZ[gid]));

    // Calculation based on cutoff distance
      if (r[gid] > rCutoff[atomIdx]) {
        result[gid] += eConst[atomIdx] / (r[gid] * r[gid]);
      } else {
        result[gid] += maxE[atomIdx] - alpha[atomIdx] * r[gid] * r[gid]; 
      }
  }
    }
    """).build()

    ## Run kernels
    r = cl.Buffer(ctx, mf.WRITE_ONLY, gridSpace[:, 0].nbytes)
    result = cl.Buffer(ctx, mf.WRITE_ONLY, gridSpace[:, 0].nbytes)
    knl = prg.elec
    knl(queue, gridSpace[:, 0].shape, None, gridX, gridY, gridZ, x, y, z, eConst, maxE, rCutoff, alpha, nAtoms, r, result)

    ## Return results
    gridEner = np.empty_like(gridSpace[:, 0]).astype(np.float32)
    cl.enqueue_copy(queue, gridEner, result)
    return gridEner

def generate_with_pyopencl(probes, gridU, formatted = False, 
    gridHBond = False, **kwargs):
    """generate grids for a given structure

    The grid generation utilize numpy package to facilitate grid calculation on CPU.
    Grid generation requires receptor atom selection, probes atom selection, grid center,
    box size and parameters for customizable grids. 

    The generated grids are passed to api_grid.F90 to output results. 

    Parameters
    ----------
    probes : array
        probe radii used for vdw grid generation and interpertation of grid number. 
    in_gridU: int
        CHARMM unit for which file to access, should be the same as the unit in the previous CHARMM command. 
    in_gridForm : bool
        True for txt file, False for binary file. 
    **kwargs: dict
        names and values for grid generation (see _configure function)

    Returns
    -------
    bool
        true for success, false if there was an error 
    """
    ## Update parameters and pass value
    grid_opts = _configure(**kwargs)

    ## Generate grid box
    gridSpace = None
    qx, x = _get_boxSize(grid_opts.xCen, grid_opts.xMax, grid_opts.dGrid)
    qy, y = _get_boxSize(grid_opts.yCen, grid_opts.yMax, grid_opts.dGrid)
    qz, z = _get_boxSize(grid_opts.zCen, grid_opts.zMax, grid_opts.dGrid)
    if qx and qy and qz: 
        gridSpace = np.vstack(np.meshgrid(x, y, z, indexing = 'ij')).reshape(3, -1).T
    else:
        print(f'Please Check Grid Dimension Settings\n',
        f'X-axis step up ==> {qx}\n',
        f'Y-axis step up ==> {qy}\n',
        f'Z-axis step up ==> {qz}\n')

    ## Create grid pot 
    if gridHBond:
        numGrid = len(probes) + 3
    else:
        numGrid = len(probes) + 1
    gridPot = np.zeros((len(gridSpace) * numGrid, 5))
    for idxGrid in np.arange(numGrid):
        gridPot[idxGrid * len(gridSpace): (idxGrid + 1) * len(gridSpace), 0:3] = gridSpace
        gridPot[idxGrid * len(gridSpace): (idxGrid + 1) * len(gridSpace), -1] = idxGrid

    ## Calculate vdw grids 
    receptor = np.zeros((psf.get_natom(), 5))
    receptor[:, 0:3] = coor.get_positions() 
    receptor[:, -2] = param.get_vdwr()
    receptor[:, -1] = param.get_epsilon()
    for idxGrid in np.arange(len(probes)):
        probeRadii = probes[idxGrid]
        gridPot[idxGrid * len(gridSpace): (idxGrid + 1) * len(gridSpace), 3] = _vdwGrid(receptor, gridSpace, probeRadii, grid_opts.emax)

    ## Calculate hdonr and acceptor grid
    if gridHBond:
        idxGrid += 1
        idxGrid += 1

    ## Calculate elec grid
    idxGrid += 1
    receptor = np.zeros((psf.get_natom(), 4))
    receptor[:, 0:3] = coor.get_positions()
    receptor[:, -1] = param.get_charge()
    gridPot[idxGrid * len(gridSpace): (idxGrid + 1) * len(gridSpace), 3] = _elecGrid(receptor, gridSpace, grid_opts.maxe, grid_opts.mine, grid_opts.dielec)

    ## Convert data to ctype
    c_gridU = ctypes.c_int(gridU)
    c_ngrid = ctypes.c_int(numGrid)
    c_ngridX = ctypes.c_int(len(x))
    c_ngridY = ctypes.c_int(len(y))
    c_ngridZ = ctypes.c_int(len(z))
    c_numProbes = ctypes.c_int(len(probes))
    c_xcen = ctypes.c_double(grid_opts.xCen)
    c_ycen = ctypes.c_double(grid_opts.yCen)
    c_zcen = ctypes.c_double(grid_opts.zCen)
    c_xmax = ctypes.c_double(grid_opts.xMax)
    c_ymax = ctypes.c_double(grid_opts.yMax)
    c_zmax = ctypes.c_double(grid_opts.zMax)
    c_dgrid = ctypes.c_double(grid_opts.dGrid)
    c_gridForce = ctypes.c_double(grid_opts.gridForce)
    c_gridPot = (ctypes.c_double * len(gridPot))(*gridPot[:, 3])
    c_gridRadii = (ctypes.c_double * len(probes))(*probes)
    c_grhbon = (ctypes.c_bool)(gridHBond)
    c_form = (ctypes.c_bool)(formatted)

    ## Output grid generation results
    status = lib.charmm.grid_write(ctypes.byref(c_xcen), 
                                   ctypes.byref(c_ycen), 
                                   ctypes.byref(c_zcen), 
                                   ctypes.byref(c_xmax), 
                                   ctypes.byref(c_ymax), 
                                   ctypes.byref(c_zmax), 
                                   ctypes.byref(c_dgrid), 
                                   ctypes.byref(c_numProbes),
                                   ctypes.byref(c_gridForce), 
                                   ctypes.byref(c_ngrid), 
                                   ctypes.byref(c_ngridX), 
                                   ctypes.byref(c_ngridY), 
                                   ctypes.byref(c_ngridZ), 
                                   ctypes.byref(c_gridPot), 
                                   ctypes.byref(c_gridRadii), 
                                   ctypes.byref(c_gridU),
           ctypes.byref(c_grhbon),
                                   ctypes.byref(c_form))
    status = bool(status)
    return gridPot, status 

