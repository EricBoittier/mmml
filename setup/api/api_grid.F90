!> routines for pyCHARMM grid setup and calculation
!> written by Yujin Wu, wyujin@umich.edu
module api_grid
  implicit none
contains

  subroutine warn_no_fftdock()
    use stream, only: outu
    implicit none
    call wrndie(-1, "<api_grid>", &
         "fftdock or openmm features not available; " // &
         "try configuring from scratch with --with-fftdock; " // &
         "ensure openmm is correctly installed and detected")
  end subroutine warn_no_fftdock

#if KEY_GRID==1
  !> @brief calculate the grid point energy and output the grid file with CHARMM cpu
  !
  !> @param[in] Islct, Jslct : selection of atoms for grid calculation
  !> @param[in] GridSlct :  selection of probe atoms
  !> @param[in] natom : total number of atoms in the system
  !> @param[in] numProbes : number of probe atoms
  !> @param[in] GridRadii : probe vdw radius
  !> @param[in] GridForce : grid edge forces
  !> @param[in] XGridCenter, YGridCenter, ZGridCenter : center of the grid box
  !> @param[in] XGridMax, YGridMax, ZGridMax : size of the grid box
  !> @param[in] Rcta, Rctb, Hmax : parameter for hydrogen bond grids
  !> @param[in] DGrid : grid spacing
  !> @param[in] NGrid : number of vdw grid + 2 hbond grids + 1 elec grid
  !> @param[in] NGridX, NGridY, NGridZ : grid point along each axis
  !> @param[in] GridPot : grid point energy
  !> @param[in] GridUnit : CHARMM unit to access
  !> @param[in] GridHBon : True for include hydrogen bond grid
  !> @param[in] GridForm : True for formatted file, False for binary format
  !> @return success 1 <==> no error
  function grid_generate(Islct, Jslct, GridSlct, &
           natom, numProbes, GridRadii, GridForce, &
           XGridCenter, YGridCenter, ZGridCenter, &
           XGridMax, YGridMax, ZGridMax, DGrid, &
           in_Rcta, in_Rctb, in_Hmax, NGrid, GridUnit, &
           GridHBon, GridForm) bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use api_util, only: f2c_logical
       use grid_dock, only: CalcGrid, WriteGrid, GridPot, &
                            NGridX, NGridY, NGridZ, RCTA, RCTB, Hmax
       use stream, only: outu, prnlev
       use string
       use ctitla
       use memory
       implicit none

       ! Declare variables
       logical :: qSuccess, Form, qGrHBon
       integer(c_int) :: success
       logical(c_bool), intent(in) :: GridHBon, GridForm
       integer(c_int), intent(in) :: NGrid, natom, numProbes, GridUnit
       real(c_double), intent(in) :: XGridCenter, YGridCenter, ZGridCenter, &
                                     XGridMax, YGridMax, ZGridMax, DGrid, &
                                     in_Hmax, in_Rcta, in_Rctb, GridForce
       integer(c_int), intent(in), dimension(natom) :: Islct, Jslct
       integer(c_int), intent(in), dimension(numProbes) :: GridSlct
       real(c_double), intent(in), dimension(numProbes) :: GridRadii
       real(chm_real) :: Rij

       ! Prepare
       Form = .False.
       qGrHBon = .False.
       If (GridForm) Form = .True.
       If (GridHBon) qGrHBon = .True.
       NGridX = XGridMax / DGrid + 2
       NGridY = YGridMax / DGrid + 2
       NGridZ = ZGridMax / DGrid + 2

       RIJ = (in_RCTA - in_RCTB) * (in_RCTA - in_RCTB)
       IF (RIJ .eq. 0) Then
          IF (PrnLev .ge. 2) Write(OutU, '(a)') &
             'Please check h-bond grid potential setup'
          RCTA = 0
          RCTB = 1
          HMax = 0 ! This will set all energy value to be 0
       Else
          RCTA = in_RCTA
          RCTB = in_RCTB
          Hmax = in_Hmax
       EndIf

       ! Output grid information
       If (PrnLev .ge. 2) Then
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Grid center x= ",XGridCenter,&
           " y= ",YGridCenter," z= ",ZGridCenter
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Grid length x= ",XGridMax,&
              " y= ",YGridMax," z= ",ZGridMax
          Write(OutU,'(a,i5,a,i5,a,i5)') "Grid points number x= ",NGridX,&
              " y= ",NGridY," z= ",NGridZ
          Write(OutU,'(a,f12.5)') "Grid space = ", DGrid
          Write(OutU,'(a,f12.5,a)') "Grid force constant = ", GridForce," kcal/mol/A^2"
          If (qGrHBon) Then
             Write(OutU, '(a,i4,a)') "Grid potentials will be set-up for", &
                  NGrid-3, " atom types plus 2 hydrogen bond plus electrostatics"
             Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Hydrogen bond grid rcta=",&
                   rcta, " rctb= ",rctb," cutoff= ",hmax
          Else
             Write(OutU, '(a,i4,a)') "Grid potentials will be set-up for", &
                  NGrid-1, " atom types plus plus electrostatics"
          Endif
       Endif

       ! Allocate grid pot
       call chmalloc('api_grid.src','grid_generate','GridPot', &
                      NGrid,NGridX,NGridY,NGridZ,crl=GridPot)

       ! Actual calculating grid point energy
       call CalcGrid(Islct, JSlct, GridSlct, &
            XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            Rcta, Rctb, Hmax, &
            NGrid, NGridX, NGridY, NGridZ, &
            GridPot, GridRadii, GridUnit, qGrHBon, Form)

       ! Actual writing out the grid file
       call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            GridForce, NGrid, NGridX, NGridY, NGridZ, &
            GridPot, GridRadii, GridUnit, qGrHbon, Form)

       ! Deallocate grid pot
       call chmdealloc('api_grid.src','grid_generate','GridPot', &
                        NGrid,NGridX,NGridY,NGridZ,crl=GridPot)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_generate

  !> @brief output the grid generation results in charmm grid format
  !> @brief we might not need this api
  !
  !> @param[in] DGrid : grid spacing
  !> @param[in] GridPot : grid point energy
  !> @param[in] GridRadii : probe vdw radius
  !> @param[in] GridForce : grid edge forces
  !> @param[in] GridUnit : CHARMM unit to access
  !> @param[in] NGridX, NGridY, NGridZ : grid point along each axis
  !> @param[in] XGridMax, YGridMax, ZGridMax : size of the grid box
  !> @param[in] NGrid : number of vdw grid + 2 hbond grids + 1 elec grid
  !> @param[in] GridForm : True for formatted file, False for binary format
  !> @param[in] XGridCenter, YGridCenter, ZGridCenter : center of the grid box
  !> @return success 1 <==> no error
  function grid_write(XGridCenter, YGridCenter, ZGridCenter, &
           XGridMax, YGridMax, ZGridMax, DGrid, numProbes, &
           GridForce, NGrid, NGridX, NGridY, NGridZ, &
           GridPot, GridRadii, GridUnit, GridHBon, GridForm &
           ) bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use api_util, only: f2c_logical
       use grid_dock, only: WriteGrid
       use stream, only: outu, prnlev
       use string
       use ctitla
       use memory
       implicit none

       ! Declare variables
       logical :: qSuccess, Form, qGrHBon
       integer :: tmp_i, tmp_j, tmp_k, tmp_l, tmp_idx
       real(chm_real), dimension(NGrid,NGridX,NGridY,NGridZ) :: tmp_grid_pot
       integer(c_int) :: success
       logical(c_bool), intent(in) :: GridHBon, GridForm
       integer(c_int), intent(in) :: NGrid, NGridX, NGridY, NGridZ, &
                                     numProbes, GridUnit
       real(c_double), intent(in) :: XGridCenter, YGridCenter, ZGridCenter, &
                                     XGridMax, YGridMax, ZGridMax, DGrid, &
                                     GridForce
       real(c_double), intent(in), dimension(NGrid*NGridX*NGridY*NGridZ) :: GridPot
       real(c_double), intent(in), dimension(numProbes) :: GridRadii

       ! Prepare
       Form = .False.
       qGrHBon = .False.
       qSuccess = .False.
       If (GridForm) Form = .True.
       If (GridHBon) qGrHBon = .True.

       ! Process data
       do tmp_l = 1, NGrid
         do tmp_i = 1, NGridX
           do tmp_j = 1, NGridY
             do tmp_k = 1, NGridZ
             tmp_idx = (((tmp_l - 1) * NGridX + tmp_i - 1) * NGridY &
                 + tmp_j - 1) * NGridZ + tmp_k
             tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k) = GridPot(tmp_idx)
             enddo
           enddo
         enddo
       enddo

       ! Actual writing out the grid file
       Call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            GridForce, NGrid, NGridX, NGridY, NGridZ, &
            tmp_grid_pot, GridRadii, GridUnit, qGrHbon, Form)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_write

  !> @brief read in the grid point energy and include grid energy calculation
  !>        in the main energy routine
  !
  !> @param[in] Natom : total number of atoms in the system
  !> @param[in] in_NAtmGrd : number of atoms for grid energy calculation
  !> @param[in] in_gridU : CHARMM unit to access
  !> @param[in] GridSlct : index of atoms for grid energy calculation
  !> @param[in] in_gridHBond : True for including hydrogen bond grid energy
  !> @param[in] in_gridForm : True for formatted file, False for binary format
  !> @return success 1 <==> no error
  function grid_read(Natom, in_NAtmGrd, in_gridU, GridSlct, &
           in_gridHBond, in_gridForm) bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use param_store, only: set_param
       use api_util, only: f2c_logical
       use grid_dock, only: QGrid, QGridOK, QGrHBon, &
                            GridAtm, GridHBAtm, NGrid0, &
                            NAtmGrd, GridRadii, GridPot, &
                            NGrid, NGridX, NGridY, NGridZ, &
                            XGridCenter, YGridCenter, ZGridCenter, &
                            XGridMax, YGridMax, ZGridMax, DGrid, GridForce, &
                            ReadGrid, GridAtmMap
       use energym, only: qeterm, grvdw, grelec, grhbon
       use stream, only: outu, prnlev
       use chm_kinds
       use string
       use ctitla
       use memory
       implicit none

       ! Input variables
       logical(c_bool), intent(in) :: in_gridHBond, in_gridForm
       integer(c_int), intent(in) :: Natom, in_NAtmGrd, in_gridU
       integer(c_int), intent(in), dimension(Natom) :: GridSlct

       ! Output variables
       logical :: qSuccess = .False.
       integer(c_int) :: success

       ! Passing CHARMM variable
       logical :: Form
       Form = .False.
       QGrid = .False.
       QGridOK = .False.
       QGrHBON = .False.
       If (in_gridForm) Form = .True.
       If (in_gridHBond) QGrHBon = .True.

       ! Actual reading of the grid parameters
       If (.not. Form) Then
          If(Prnlev .ge.2) Write(OutU,'(a,i4)') &
               'Grid potentials read from binary file on unit', &
               in_gridU
          Call Rdtitl(TitleB, NtitlB, in_gridU, -1)
          Call Wrtitl(TitleB, NtitlB, OutU, 0)
          Read(in_gridU) NGrid, NGridx, NGridy, NGridz
          Read(in_gridU) XGridCenter, YGridCenter, ZGridCenter
          Read(in_gridU) XGridMax, YGridMax, ZGridMax, &
               DGrid, GridForce
       Else
          Write(OutU,'(a,i4)') &
               'Grid potentials read from formatted file on unit', &
               in_gridU
          Call Rdtitl(TitleB, NtitlB, in_gridU, 0)
          Call Wrtitl(TitleB, NtitlB, OutU, 0)
          Read(in_gridU,'(4(i5,1x))') NGrid, NGridx, NGridy, NGridz
          Read(in_gridU,'(3(f12.5,1x))') XGridCenter, YGridCenter, ZGridCenter
          Read(in_gridU,'(5(f12.5,1x))') XGridMax, YGridMax, ZGridMax, &
               DGrid, GridForce
       Endif

       ! Set up parameter
       Call set_param("XGRIDMAX", XGridMax)
       Call set_param("YGRIDMAX", YGridMax)
       Call set_param("ZGRIDMAX", ZGridMax)
       Call set_param("XGRIDCEN", XGridCenter)
       Call set_param("YGRIDCEN", YGridCenter)
       Call set_param("ZGRIDCEN", ZGridCenter)

       !  Output data for read Parameters
       If (PrnLev .ge. 2 .and. QGrHBON) Write(OutU,'(a,i4,a)') &
          'GridSetUp: Grid potentials will be set-up for', &
          NGrid-3,' atom types plus 2 hydrogen bond plus electrostatics'
       If (PrnLev .ge. 2 .and. .not. QGrHBON) Write(OutU,'(a,i4,a)') &
          'GridSetUp: Grid potentials will be set-up for', &
          NGrid-1,' atom types plus electrostatics'
       If (PrnLev .ge. 2) Write(OutU,'(a,i4)') &
          'GridSetUp: and written to unit', in_gridU
       If (PrnLev .ge. 2) Write(OutU,'(a,3f10.5)') &
          'GridSetUp: Grid centered at', &
          XGridCenter, YGridCenter, ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(3(a,f10.5,a,f10.5/))') &
          'GridSetUp: Grid runs from (X)', &
          -XGridMax/2+XGridCenter,' -', XGridMax/2+XGridCenter, &
          'GridSetUp: Grid runs from (Y)', &
          -YGridMax/2+YGridCenter,' -', YGridMax/2+YGridCenter, &
          'GridSetUp: Grid runs from (Z)', &
          -ZGridMax/2+ZGridCenter,' -', ZGridMax/2+ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.5)') &
          'GridSetUp: With a grid spacing of', DGrid
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.3,a)') &
          'GridSetUp: Force constant at grid edge set to ', &
          GridForce,' kcal/mol/A^2'
       If (PrnLev .ge. 2) Write(OutU,'(a,i8,a)') &
          'GridSetUp: Allocating ', NGrid*NGridX*NGridY*NGridZ, &
          'Real(chm_Real) words for grid potentials.'
       !--- Whether or not use hydrogen bond grids in calculation ---
       If (PrnLev .ge. 2 .and. QGrHBon) Write(OutU, '(a)') &
          'Hydrogen bond grids will be used'
       If (PrnLev .ge. 2 .and. .not. QGrHBon) Write(OutU, '(a)') &
          'Hydrogen bond grids will not be used'

       call chmalloc('api_grid.src','grid_read','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)
       If (QGrHBon) Then
          NGrid0 = NGrid - 3
       Else
          NGrid0 = NGrid - 1
       Endif
       call chmalloc('api_grid.src','grid_read','GridRadii',NGrid0,crl=GridRadii)

       call ReadGrid(GridPot, GridRadii, NGrid0, &
            NGrid, NGridX, NGridY, NGridZ, &
            in_gridU, Form)

       ! Set-up atom based grids on based on current psf
       call chmalloc('api_grid.src','grid_read','GridAtm',Natom,intg=GridAtm)
       call chmalloc('api_grid.src','grid_read','GridHBAtm',Natom,intg=GridHBAtm)
       call GridAtmMap(GridSlct, GridAtm, GridHBAtm, QGrHBon, NGrid, GridRadii)

       If(Prnlev .ge. 2) Write(OutU,'(a,i5,a)') &
          'Grid mapping set-up for', in_NAtmGrd,' atoms'
       NAtmGrd = Natom

       ! Set logical variable
       QGrid = .True.
       QGridOK = .True.
       qeterm(grvdw) = .True.
       qeterm(grelec) = .True.
       qeterm(grhbon) = .False.
       If (QGrHBON) qeterm(grhbon) = .True.

       ! Pass variable for output
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_read

  !> @brief turn on grid for selected atoms
  !
  !> @param[in] GridSlct : index of atoms for grid energy calculation
  !> @param[in] in_gridHBond : True for including hydrogen bond grid energy
  !> @param[in] in_NAtmGrd : number of atoms for grid energy calculation
  !> @param[in] Natom : total number of atoms in the system
  !> @return success 1 <==> no error
  function grid_on(GridSlct, in_gridHBond, in_NAtmGrd, Natom) bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use stream, only: outu, prnlev
       use api_util, only: f2c_logical
       use energym, only: qeterm, grvdw, grelec, grhbon
       use grid_dock, only: QGrid, QGridOK, QGrHBon, GridAtm, NGrid, &
                            GridRadii, GridHBAtm, NAtmGrd, GridAtmMap
       use chm_kinds
       use string
       use memory
       implicit none

       ! Input variables
       logical(c_bool), intent(in) :: in_gridHBond
       integer(c_int), intent(in) :: Natom, in_NAtmGrd
       integer(c_int), intent(in), dimension(Natom) :: GridSlct

       ! Output variables
       logical :: qSuccess
       integer(c_int) :: success

       ! Passing CHARMM variable
       QGrHBon = .False.
       If (in_gridHBond) QGrHBon = .True.

       ! Actual setting up grid atom selection
       If (.not. QGridOK) Call WrnDie(1,'<GridSetUp>', &
            'Grid not set-up, use read first')
       If (QGridOK .and. QGrid .and. Prnlev .ge.2) &
            Write(OutU,'(a)')' Grid energy already on'
       If (.not. QGrid) Then
          If (Prnlev .ge. 2) Write(OutU,'(a)') &
             'Grid energy will be used in energy calculations'
          !--- Whether or not use hydrogen bond grids in calculation ---
          If (PrnLev .ge. 2 .and. QGrHBon) Write(OutU, '(a)') &
             'Hydrogen bond grids will be used'
          If (PrnLev .ge. 2 .and. .not. QGrHBon) Write(OutU, '(a)') &
             'Hydrogen bond grids will not be used'

          ! Set-up atom based grids on based on current psf
          call chmalloc('api_grid.src','grid_on','GridAtm',Natom,intg=GridAtm)
          call chmalloc('api_grid.src','grid_on','GridHBAtm',Natom,intg=GridHBAtm)
          call GridAtmMap(GridSlct, GridAtm, GridHBAtm, QGrHBon, NGrid, GridRadii)

          If (Prnlev .ge. 2) Write(OutU,'(a,i5,a)') &
             'Grid mapping set-up for', NAtmGrd,' atoms'
          NAtmGrd = Natom

          ! Set logical variable
          QGrid = .True.
          qeterm(grvdw) = .True.
          qeterm(grelec) = .True.
          qeterm(grhbon) = .False.
          If (QGrHBON) qeterm(grhbon) = .True.
       EndIf

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_on

  !> @brief turn off grid
  !
  !> @return success 1 <==> no error
  function grid_off() bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use stream, only: outu, prnlev
       use api_util, only: f2c_logical
       use grid_dock, only: QGrid, GridAtm, GridHBAtm, NAtmGrd
       use chm_kinds
       use string
       use memory
       implicit none

       ! Output variables
       logical :: qSuccess
       integer(c_int) :: success

       ! Actual reading of the grid parameters
       If (.not. QGrid .and. PrnLev .ge.2) &
          Write(OutU,'(a)')' Grid energy already off'
       If (QGrid) Then
          If (PrnLev .ge. 2) Write(OutU,'(a)') &
             'Grid energy will not be used in energy calculations'
          QGrid = .False.
          ! Free-up atom based grids on based on current psf
          call chmdealloc('api_grid.src','grid_off','GridAtm',NAtmGrd,intg=GridAtm)
          call chmdealloc('api_grid.src','grid_off','GridHBAtm',NAtmGrd,intg=GridHBAtm)
          If (PrnLev .ge. 2) Write(OutU,'(a,i5,a)') &
             'Space for', NAtmGrd,' atoms freed'
       Endif

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_off

  !> @brief clear grid
  !
  !> @return success 1 <==> no error
  function grid_clear() bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use stream, only: outu, prnlev
       use api_util, only: f2c_logical
       use grid_dock, only: QGrid, QGridOK, QGrHBon, &
                            GridAtm, GridHBAtm, NAtmGrd, &
                            GridRadii, GridPot, NGrid, &
                            NGridX, NGridY, NGridZ
       use chm_kinds
       use string
       use memory
       implicit none

       ! Output variables
       logical :: qSuccess
       integer(c_int) :: success

       ! Actual reading of the grid parameters
       If (QGrid) Then
          call chmdealloc('api_grid.src','grid_clear','GridAtm',NAtmGrd,intg=GridAtm)
          call chmdealloc('api_grid.src','grid_clear','GridHBAtm',NAtmGrd,intg=GridHBAtm)
       Endif
       If (QGridOK) Then
          IF (QGrHBon) Then
             call chmdealloc('api_grid.src','grid_clear','GridRadii',NGrid-3,crl=GridRadii)
          Else
             call chmdealloc('api_grid.src','grid_clear','GridRadii',NGrid-1,crl=GridRadii)
          EndIf
          call chmdealloc('api_grid.src','grid_clear','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)
       Endif
       If (PrnLev .ge. 2) Write(OutU,'(a)') &
          'All space for grid potentials freed'
       QGrid = .False.
       QGridOK = .False.

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function grid_clear

#if KEY_FFTDOCK == 1
  !> @brief calculate the grid point energy and output the grid file with GPU
  !
  !> @param[in] flag_select : selection of atoms for grid calculation
  !> @param[in] flag_rdie : True for rdie, False for cdie
  !> @param[in] in_dielec : dielectric constant
  !> @param[in] natom : total number of atoms in the system
  !> @param[in] GridForce : grid edge forces
  !> @param[in] Xcen, Ycen, Zcen: center of the grid box
  !> @param[in] Xmax, Ymax, Zmax : size of the grid box
  !> @param[in] DGrid : grid spacing
  !> @param[in] Rcta, Rctb, Hmax : parameter for hydrogen bond grids
  !> @param[in] Emax, Maxe, Mine : parameter for soft grid potentials
  !> @param[in] GridUnit : CHARMM unit to access grid file
  !> @param[in] ProbeU : CHARMM unit to access probe file
  !> @param[in] GridHBon : True for include hydrogen bond grid
  !> @param[in] GridForm : True for formatted file, False for binary format
  !> @param[in] gpuID : gpu ID
  !> @return success 1 <==> no error
  function grid_fftgen(flag_select, flag_rdie, in_dielec, &
           natom, GridForce, Xcen, Ycen, Zcen, Xmax, &
           Ymax, Zmax, DGrid, Rcta, Rctb, Hmax, Emax, Maxe, Mine, &
           GridUnit, ProbeUnit, GridHBon, GridForm, &
           gpuID) bind(c) result(success)

       use, intrinsic :: iso_c_binding
       use fftdock, only: generate_potential_grid_gpu, &
                          write_potential_grid, &
                          read_probes_radii, NumGrid, DGrid4, &
                          RIJ, XGridMin, YGridMin, ZGridMin, &
                          VdwEmax, ElecReplEmax, ElecAttrEmax, &
                          fa, fb, gmax, dielec, CCELEC_charmm4, &
                          GridPot, GridRadii, GridRadii4, &
                          SelectAtomsParameters, ElecMode, &
                          num_select_atoms, XGridNum, YGridNum, &
                          ZGridNum, XGridLen, YGridLen, ZGridLen, &
                          XGridCenter, YGridCenter, ZGridCenter
       use psf, only: cg, iac, ndon, nacc, iacc, ihd1
       use consta, only : CCELEC_CHARMM
       use param, only: vdwr, eff, itc
       use api_util, only: f2c_logical
       use grid_dock, only: WriteGrid
       use stream, only: outu, prnlev
       use coord, only: x, y, z
       use string
       use ctitla
       use memory

#if KEY_CUDA == 1
       use fftdock, only: allocate_gpu_id
#elif KEY_OPENCL == 1
       use fftdock, only: fftdock_device_init
#endif

       implicit none

       ! Declare variables
       logical :: qSuccess, Form, flag_hbond
       integer(c_int) :: success
       logical(c_bool), intent(in) :: GridHBon, GridForm, flag_rdie
       integer(c_int), intent(in) :: natom, GridUnit, ProbeUnit, gpuID
       real(c_double), intent(in) :: Xcen, Ycen, Zcen, &
                                     Xmax, Ymax, Zmax, DGrid, &
                                     Hmax, Rcta, Rctb, GridForce, &
                                     Emax, Maxe, Mine, in_dielec
       integer(c_int), intent(in), dimension(natom) :: flag_select
       integer :: tmp_l, tmp_i, tmp_j, tmp_k, idx, tmp_idx, &
                  HDonor(Natom), HAcceptor(Natom), &
                  num_grid_points, tot_num_grid_points, &
                  num_parameters, HBatm

       ! Read in probes
       call Read_Probes_Radii(ProbeUnit)
       call chmalloc('api_grid.src', 'grid_fftgen', 'GridRadii4', &
            NumGrid - 3, cr4 = GridRadii4)
       GridRadii4 = GridRadii

       ! Pass variables
       Form = .False.
       flag_hbond = .False.
       If (GridForm) Form = .True.
       If (GridHBon) flag_hbond = .True.
       IF (flag_rdie) Then
           ElecMode = 1
       Else
           ElecMode = 0
       Endif
       GMAX = HMAX
       RIJ = (RCTA - RCTB) * (RCTA - RCTB)
       IF (RIJ .eq. 0) Then
          IF (PrnLev .ge. 2) Write(OutU, '(a)') &
             'Please check h-bond grid potential setup'
          RIJ = 1
          GMax = 0 ! This will set all energy value to be 0
       EndIf
       FA = -4 * GMAX / RIJ
       FB = 0.5 * (RCTA + RCTB)
       XGridLen = Xmax
       YGridLen = Ymax
       ZGridLen = Zmax
       XGridCenter = Xcen
       YGridCenter = Ycen
       ZGridCenter = Zcen
       XGridMin = XGridCenter - XGridLen/2.0
       YGridMin = YGridCenter - YGridLen/2.0
       ZGridMin = ZGridCenter - ZGridLen/2.0
       XGridNum = XGridLen / DGrid + 2
       YGridNum = YGridLen / DGrid + 2
       ZGridNum = ZGridLen / DGrid + 2
       CCELEC_charmm4 = CCELEC_charmm
       ElecReplEmax = Maxe
       ElecAttrEmax = Mine
       Dielec = in_dielec
       VdwEmax = Emax
       DGrid4 = DGrid
       num_grid_points = XGridNum * YGridNum * ZGridNum
       tot_num_grid_points = num_grid_points * NumGrid

       ! Set up hydrogen bond donor and acceptor index
       HDonor = 0
       HAcceptor = 0
       If (NDON.GT.0.and.NDON.LE.NATOM) Then
          Do HBatm = 1, NDON
             HDonor(IHD1(HBatm)) = 1
          Enddo
       Endif
       If (NACC.GT.0.and.NACC.LE.NATOM) Then
          Do HBatm = 1, NACC
             HAcceptor(IACC(HBatm)) = 1
          Enddo
       Endif

       ! Build selected atoms parameters array
       ! Parameters order for each atom: X,Y,Z,Epsilon,VDWRadii,Charge,Hdonor,Hacceptor
       idx = 1
       num_parameters = 8
       call chmalloc('api_grid.src', 'grid_fftgen', &
           'SelectAtomsParameters', Natom*num_parameters, &
           cr4 = SelectAtomsParameters)
       Do tmp_i = 1, natom
          If (flag_select(tmp_i) > 0) Then
          SelectAtomsParameters((idx-1)*num_parameters+1)=X(tmp_i)
          SelectAtomsParameters((idx-1)*num_parameters+2)=Y(tmp_i)
          SelectAtomsParameters((idx-1)*num_parameters+3)=Z(tmp_i)
          SelectAtomsParameters((idx-1)*num_parameters+4)=EFF(ITC(IAC(tmp_i))) !well depth
          SelectAtomsParameters((idx-1)*num_parameters+5)=VDWR(ITC(IAC(tmp_i))) !vdw radii
          SelectAtomsParameters((idx-1)*num_parameters+6)=CG(tmp_i) !charge
          SelectAtomsParameters((idx-1)*num_parameters+7)=HDonor(tmp_i) !donor
          SelectAtomsParameters((idx-1)*num_parameters+8)=HAcceptor(tmp_i) !acceptor
          idx=idx+1
          Endif
       Enddo
       num_select_atoms = idx - 1

       ! Output grid information
       If (PrnLev .ge. 2) Then
          Write(OutU,'(a)') "<FFTDOCK> Generate protein grid using GPU"
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Grid center x= ",XGridCenter,&
           " y= ",YGridCenter," z= ",ZGridCenter
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Grid length x= ",XGridLen,&
              " y= ",YGridLen," z= ",ZGridLen
          Write(OutU,'(a,i5,a,i5,a,i5)') "Grid points number x= ",XGridNum,&
              " y= ",YGridNum," z= ",ZGridNum
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Grid minimum dimension x=",&
              XGridMin, " y= ",YGridMin," z= ",ZGridMin
          Write(OutU,'(a,f12.5,a,f12.5,a,f12.5)') "Hydrogen bond grid rcta=",&
              rcta, " rctb= ",rctb," cutoff= ",gmax
          Write(OutU,'(a,f12.5)') "Grid space = ", DGrid
          Write(OutU,'(a,i10)') "Number of grid points per type = ", num_grid_points
          Write(OutU,'(a,f12.5,a)') "Grid force constant = ", GridForce," kcal/mol/A^2"
          Write(OutU,'(a,f12.5)') "Electrostatics softcore repulsion emax  = ", &
              ElecReplEmax
          Write(OutU,'(a,f12.5)') "Electrostatics softcore attraction emax  = ", &
              ElecAttrEmax
          Write(OutU,'(a,f12.5)') "Van der Waals softcore emax  = ", VdwEmax
          If (flag_rdie) Then
             Write(OutU,'(a,f12.5)') "Using radial dependent dielectric constant = ",Dielec
          Else
             Write(OutU,'(a,f12.5)') "Using constant dielectric constant = ",Dielec
          Endif
          Write(OutU,'(a)') "  Probe       Radii"
          Do tmp_i = 1, NumGrid - 3
              Write(OutU,'(i7,f12.6)') tmp_i, GridRadii4(tmp_i)
          Enddo
          Write(OutU,'(a,i5)') "Total number of grid types", NumGrid
          Write(OutU,'(a,i10)') "Total number of grid points = ", tot_num_grid_points
          Write(OutU,'(a,i10)') "Total number of atoms used for protein grid = ",idx-1
       Endif

       ! Actual grid calculation on gpu
       call chmalloc('api_grid.src', 'grid_fftgen', 'GridPot', &
            tot_num_grid_points, cr4 = GridPot)
       GridPot(:) = 0.0D0

#if KEY_CUDA == 1
       call allocate_gpu_id(gpuid)
#elif KEY_OPENCL == 1
       call fftdock_device_init(gpuid)
#endif

       call generate_potential_grid_gpu(NumGrid, num_select_atoms, DGrid4, &
            XGridNum, YGridNum, ZGridNum, XGridMin, YGridMin, ZGridMin, &
            Fa, Fb, Gmax, VdwEmax, ElecAttrEmax, ElecReplEmax, CCELEC_charmm4, &
            ElecMode, Dielec, SelectAtomsParameters, GridPot, GridRadii4, gpuid)

       ! Actual writing out the grid file
       call Write_Potential_Grid(XGridCenter, YGridCenter, ZGridCenter, &
            XGridLen, YGridLen, ZGridLen, DGrid, GridForce, NumGrid, &
            XGridNum, YGridNum, ZGridNum, GridPot, GridRadii, GridUnit, &
            flag_hbond, Form)

       ! Deallocate array
       call chmdealloc('api_grid.src', 'grid_fftgen', 'GridPot', &
                        tot_num_grid_points, cr4 = GridPot)
       call chmdealloc('api_grid.src', 'grid_fftgen', 'GridRadii4', &
                        NumGrid - 3, cr4 = GridRadii4)
       call chmdealloc('api_grid.src', 'grid_fftgen', &
                       'SelectAtomsParameters', num_select_atoms*num_parameters, &
                        cr4 = SelectAtomsParameters)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
   end function grid_fftgen
#else /* we have no fftdock compiled in */
     function grid_fftgen() bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
       integer(c_int) :: success
       call warn_no_fftdock()
       success = -1
     end function grid_fftgen
#endif /* KEY_FFTDOCK == 1 */

#if KEY_FFTDOCK == 1 && KEY_OPENMM == 1
  !> @brief Set up OMMD docking system
  !
  !> @param[in] softGridUnit : CHARMM unit for the soft grid
  !> @param[in] hardGridUnit : CHARMM unit for the hard grid
  !> @param[in] GridForm : True for formatted file, False for binary format
  !> @param[in] GridHBon : True for include hydrogen bond grid
  !> @param[in] natom : total number of atoms in the system
  !> @param[in] in_fix_select : selection of fixed atoms
  !> @param[in] in_flex_select : selection of flexible atoms
  !> @param[in] in_numCopyFlex : number of flexible particle copies
  !> @return success 1 <==> no error
  function ommd_create(softGridUnit, hardGridunit, GridForm, &
           GridHBon, natom, in_fix_select, in_flex_select, &
           in_numCopyFlex) bind(c) result(success)

       use, intrinsic :: iso_c_binding, only: c_int, c_bool, c_double
       use openmm_dock, only: SoftGridPot, HardGridPot, GridRadii, Form, &
                              omm_select, fix_select, flex_select, &
                              hdonor_select, hacceptor_select, charmmIdx2ommIdx, &
                              num_copy_flex, num_fix_atom, num_flex_atom, &
                              Read_Grid_Potentials, Create_System, &
                              Create_Context
       use psf, only: iac, ndon, nacc, iacc, ihd1
       use api_util, only: f2c_logical
       use stream, only: outu, prnlev
       use ctitla
       use memory
       implicit none

       ! Declare variables
       logical :: qSuccess, flag_hbond
       integer :: i, hbatm
       integer(c_int) :: success
       integer(c_int), intent(in) :: softGridUnit, hardGridUnit, &
                                     in_numCopyFlex, natom
       integer(c_int), intent(in), dimension(Natom) :: in_fix_select, in_flex_select
       logical(c_bool), intent(in) :: GridHBon, GridForm

       ! Read in and prepare grids for the OpenMM system
       Form = .False.
       flag_hbond = .False.
       If (GridForm) Form = .True.
       If (GridHBon) flag_hbond = .True.
       If (PrnLev .ge. 2) Write(OutU, '(a)') " Read in grid potentials for OpenMM docking system"
       call Read_Grid_Potentials(softGridUnit, hardGridUnit, flag_hbond, Form)

       ! Allocate array and pass variables from input
       call chmalloc("api_grid.src", "grid_omm_create", 'omm_select', natom, intg = omm_select)
       call chmalloc("api_grid.src", "grid_omm_create", 'fix_select', natom, intg = fix_select)
       call chmalloc("api_grid.src", "grid_omm_create", 'flex_select', natom, intg = flex_select)
       call chmalloc("api_grid.src", "grid_omm_create", 'hdonor_select', natom, intg = hdonor_select)
       call chmalloc("api_grid.src", "grid_omm_create", 'hacceptor_select', natom, intg = hacceptor_select)
       call chmalloc("api_grid.src", "grid_omm_create", 'charmmIdx2ommIdx', &
                      natom, intg = charmmIdx2ommIdx)
       omm_select(:) = 0
       hdonor_select(:) = 0
       hacceptor_select(:) = 0
       fix_select = in_fix_select
       flex_select = in_flex_select
       num_copy_flex = in_numCopyFlex

       !---------------------Flexible and fix atom index--------------------------
       num_fix_atom = 0
       num_flex_atom = 0
       do i = 1, nAtom
          if (fix_select(i) == 1) then
             num_fix_atom = num_fix_atom + 1
             omm_select(i) = 1
          endif
          if (flex_select(i) == 1) then
             num_flex_atom = num_flex_atom + 1
             omm_select(i) = 1
          endif
       enddo
       If (PrnLev .ge. 2) Write(OutU, '(a,i10)') " Number of fixed atoms = ", num_fix_atom
       If (PrnLev .ge. 2) Write(OutU, '(a,i10)') " Number of flexible atoms = ", num_flex_atom

       !--------------Hydrogen bond donor and acceptor index-----------------------
       ! Whether or not use hydrogen bond in energy calculation
       if (flag_hbond) then
         If (PrnLev .ge. 2) Write(OutU, '(a)') ' Hydrogen bond grids will be used'
         if (NACC.GT.0.and.NACC.LE.NATOM) then
            do HBatm = 1, NACC
               hacceptor_select(IACC(HBatm)) = 1
            enddo
         end if

         if (NDON.GT.0.and.NDON.LE.NATOM) then
            do HBatm = 1, NDON
               hdonor_select(IHD1(HBatm)) = 1
            enddo
         endif
       else
         If (PrnLev .ge. 2) Write(OutU, '(a)') ' Hydrogen bond grids will not be used'
       endif

       ! Create OpenMM docking system
       If (PrnLev .ge. 2) Write(OutU, '(a)') " Create OpenMM docking system"
       call Create_System(omm_select, fix_select, flex_select, hdonor_select, &
                          hacceptor_select, charmmIdx2ommIdx, num_copy_flex, &
                          num_fix_atom, num_flex_atom, flag_hbond)
       call Create_Context()

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_create

  !> @brief Print OMMD system energy
  !
  !> @param[in] GridHBon : True for include hydrogen bond grid
  !> @return success 1 <==> no error
  function ommd_energy(GridHBon) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int, c_bool
       use openmm_dock, only: Print_Energy
       use api_util, only: f2c_logical

       ! Declare variables
       implicit none
       integer(c_int) :: success
       logical :: qSuccess, flag_hbond
       logical(c_bool), intent(in) :: GridHBon

       ! Pass variable
       flag_hbond = .False.
       If (GridHBon) flag_hbond = .True.

       ! Call ommd energy function
       call Print_Energy(flag_hbond)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_energy

  !> @brief Clear OMMD system
  !
  !> @param[in] GridHBon : True for include hydrogen bond grid
  !> @param[in] natom : total number of atoms in the system
  !> @return success 1 <==> no error
  function ommd_clear(GridHBon, natom) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int, c_bool
       use openmm_dock, only: SoftGridPot, HardGridPot, GridRadii, &
                              omm_select, fix_select, flex_select, &
                              hdonor_select, hacceptor_select, charmmIdx2ommIdx, &
                              num_copy_flex, num_fix_atom, num_flex_atom, &
                              NumGrid, XGridNum, YGridNum, ZGridNum, context, system
       use api_util, only: f2c_logical
       use memory, only: chmalloc, chmdealloc
       use chm_kinds

       ! Declare variables
       implicit none
       integer :: NGrid0, num_grid_points
       logical :: qSuccess, flag_hbond
       integer(c_int) :: success
       integer(c_int), intent(in) :: natom
       logical(c_bool), intent(in) :: GridHBon

       ! Pass variable
       flag_hbond = .False.
       If (GridHBon) flag_hbond = .True.

       ! Clear OpenMM docking system
       num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
       if (flag_hbond) then
          NGrid0 = NumGrid - 3
       else
          NGrid0 = NumGrid - 1
       endif

       call chmdealloc("api_grid.src", "ommd_clear", 'omm_select', natom, intg = omm_select)
       call chmdealloc("api_grid.src", "ommd_clear", 'fix_select', natom, intg = fix_select)
       call chmdealloc("api_grid.src", "ommd_clear", 'flex_select', natom, intg = flex_select)
       call chmdealloc("api_grid.src", "ommd_clear", 'hdonor_select', natom, intg = hdonor_select)
       call chmdealloc("api_grid.src", "ommd_clear", 'hacceptor_select', natom, intg = hacceptor_select)
       call chmdealloc("api_grid.src", "ommd_clear", 'charmmIdx2ommIdx', natom, intg = charmmIdx2ommIdx)
       call chmdealloc("api_grid.src", "ommd_clear", 'SoftGridPot', num_grid_points, crl = SoftGridPot)
       call chmdealloc("api_grid.src", "ommd_clear", 'HardGridPot', num_grid_points, crl = HardGridPot)
       call chmdealloc("api_grid.src", "ommd_clear", 'GridRadii', NGrid0, crl = GridRadii)

       call OpenMM_System_destroy(system)
       call OpenMM_Context_destroy(context)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_clear

  !> @brief Change OpenMM docking system softness
  !
  !> @param[in] in_soft, in_hard : percentage of soft/hard grid
  !> @param[in] in_emax : parameter for soft core E_vdw
  !> @param[in] in_mine, in_maxe, in_eps : parameter for soft core E_elec
  !> @return success 1 <==> no error
  function ommd_change_softness(in_soft, in_hard, in_emax, &
           in_mine, in_maxe, in_eps) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int, c_double
       use openmm_dock, only: soft_p, hard_p, emax, mine, &
                              maxe, eps, Change_Grid_Softness
       use api_util, only: f2c_logical

       ! Declare variables
       implicit none
       logical :: qSuccess
       integer(c_int) :: success
       real(c_double), intent(in) :: in_soft, in_hard, in_emax, &
                                     in_mine, in_maxe, in_eps

       ! Pass variable
       soft_p = in_soft
       hard_p = in_hard
       emax = in_emax
       mine = in_mine
       maxe = in_maxe
       eps = in_eps

       ! Call function to change grid softness
       call Change_Grid_Softness(soft_p, hard_p, emax, mine, maxe, eps)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_change_softness

  !> @brief Run OpenMM docking simulated annealing
  !
  !> @param[in] in_steps : Number of steps
  !> @param[in] in_heat_frq : heat frequence
  !> @param[in] in_start_temp : starting temperature
  !> @param[in] in_end_temp : ending temperature
  !> @param[in] in_incr_temp : increasement of temperature
  !> @return success 1 <==> no error
  function ommd_sian(in_steps, in_heat_frq, in_start_temp, &
           in_end_temp, in_incr_temp) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int, c_double
       use openmm_dock, only: num_steps, heat_frq, start_temp, &
                              end_temp, incr_temp, Run_Simulated_Anneling
       use api_util, only: f2c_logical

       ! Declare variables
       implicit none
       logical :: qSuccess
       integer(c_int) :: success
       integer, intent(in) :: in_steps, in_heat_frq
       real(c_double), intent(in) :: in_start_temp, in_end_temp, in_incr_temp

       ! Pass variable
       num_steps = in_steps
       heat_frq = in_heat_frq
       end_temp = in_end_temp
       incr_temp = in_incr_temp
       start_temp = in_start_temp

       ! Call function to run simulated annealing
       call Run_Simulated_Anneling(num_steps, heat_frq, start_temp, end_temp, incr_temp)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_sian

  !> @brief Set atoms postion in the OpenMM system from main coordinates
  !
  !> @param[in] in_idxcopy : index of the flexible atoms copy
  !> @return success 1 <==> no error
  function ommd_set_position(in_idxcopy) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int
       use openmm_dock, only: flex_select, charmmIdx2ommIdx, num_flex_atom, &
                              set_positions, context, state
       use api_util, only: f2c_logical
       use OpenMM

       ! Declare variables
       implicit none
       logical :: qSuccess
       integer(c_int) :: success
       integer, intent(in) :: in_idxcopy

       ! Call function to run simulated anneling
       call set_positions(flex_select, charmmIdx2ommIdx, num_flex_atom, in_idxcopy)
       call OpenMM_Context_getState(context, OpenMM_State_Energy, Openmm_False, state)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_set_position

  !> @brief Copy atoms postion from the OpenMM system to main coordinates
  !
  !> @param[in] in_idxcopy : index of the flexible atoms copy
  !> @return success 1 <==> no error
  function ommd_copy_position(in_idxcopy) bind(c) result(success)
       use, intrinsic :: iso_c_binding, only: c_int
       use openmm_dock, only: flex_select, charmmIdx2ommIdx, num_flex_atom, &
                              copy_positions_to_main
       use api_util, only: f2c_logical
       use OpenMM

       ! Declare variables
       implicit none
       logical :: qSuccess
       integer(c_int) :: success
       integer, intent(in) :: in_idxcopy

       ! Call function to run simulated anneling
       call copy_positions_to_main(flex_select, charmmIdx2ommIdx, num_flex_atom, in_idxcopy)

       ! Output result
       qSuccess = .True.
       success = f2c_logical(qSuccess)
  end function ommd_copy_position
#else /* we have no fftdock or openmm compiled in */
  function ommd_create() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_create

  function ommd_energy() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_energy

  function ommd_clear() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_clear

  function ommd_change_softness() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_change_softness

  function ommd_sian() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_sian

  function ommd_set_position() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_set_position

  function ommd_copy_position() bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: success
    call warn_no_fftdock()
    success = -1
  end function ommd_copy_position
#endif /* KEY_FFTDOCK == 1 && KEY_OPENMM == 1 */
#endif /* KEY_GRID */
end module api_grid
