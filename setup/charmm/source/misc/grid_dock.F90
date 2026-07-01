module grid_dock
  use chm_kinds
  use dimens_fcm

  implicit none
#if KEY_GRID==1 /*grid_fcm*/
  !     This common block contains the data structure for the
  !     Grid potential for docking
  ! ---------------------------------------------------------
  ! Common block:
  ! QGRID   logical flag
  !
  LOGICAL QGRID, QGridOK, QGrHBON


  Real(chm_Real),Allocatable,Dimension(:) :: GridRadii
  Real(chm_Real),Allocatable,Dimension(:,:,:,:) :: GridPot
  Integer,Allocatable,Dimension(:) :: GridAtm, gridhbatm
  Integer,Parameter,Dimension(1) :: GridNoSelection = (/ 0 /)
  Integer NGrid, NGrid0, NGridX, NGridY, NGridZ, NAtmGrd
  Real(chm_Real) XGridCenter, YGridCenter, ZGridCenter,  &
       RCTA, RCTB, HMAX, &
       XGridMax, YGridMax, ZGridMax, DGrid, GridForce
  !
#endif /* (grid_fcm)*/

contains

#if KEY_GRID==0 /*grid_main*/
  Subroutine GRIDSet(COMLYN,COMLEN)
    Character(len=*) Comlyn
    Integer Comlen
    CALL WRNDIE(-1,'<GRIDSET>','GRID code is not compiled.')
    return
  end Subroutine GRIDSet
#else /* (grid_main)*/

  subroutine grid_init()
    qgrid=.false.
    qgridok=.false.
    return
  end subroutine grid_init

  Subroutine GRIDSet(COMLYN,COMLEN)
    !
    !  This routine sets-up a vdW and electrostatic grid for later
    !  use in grid-based docking with CHARMM.
    !  Written by C.L. Brooks III, The Scripps Research Institute
    !             November/December, 2000
    !

    use chm_kinds
    use select
    use consta
    use dimens_fcm
    use number
    use psf
    use coord 
    use stream
    use string
    use ctitla
    use memory
    use parallel
    use param_store, only: set_param
    
    implicit none
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    ! Variables:
    !   QGridGenerate => Main logical to initiate grid potential calculation
    !   QRead         => Read grid potential from GridUnit
    !   QClear        => Initialize all grid arrays, free space
    !   QOn           => Use grid potential and forces in energy calculations
    !   QOff          => Don't use grid potential and forces in energy calculations
    !   QPrint        => Print grid based potentials to OutU
    !   QGrHBON       => Turn on / off hydrogen bond grid 
    !   Force         => Force constant for quadratic extension of grid potential
    !   NGrid         => Number of grid potentials (minus one for electrostatcs)
    !   XGridCenter   => Geometric center of grid in X-Dimension
    !   YGridCenter   => Geometric center of grid in Y-Dimension
    !   ZGridCenter   => Geometric center of grid in Z-Dimension
    !   XGridMax      => Extent of grid about the origin in X-Dimension
    !   YGridMax      => Extent of grid about the origin in Y-Dimension
    !   ZGridMax      => Extent of grid about the origin in Z-Dimension
    !   DGrid         => Spacing between grid points
    !   OutUnit       => Unit for output of grid potentials
    !   Form          => Write grid potential in fomatted output
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    !  Usage:
    !
    ! grid generate select <atom selection> end -
    !   { xcen <Real> ycen <Real> zcen <Real> } -
    !   { xmax <Real> ymax <Real> zmax <Real> } -
    !   { RCTA <Real> RCTB <Real> hmax <Real> } -
    !   { dgrid <Real> } { Force <Real> }       -
    !   { OutUnit <Integer> } { Formatted }     -
    !   { Print }
    !
    ! grid read select <atom selection> end     -
    !      Unit <Integer> { Formatted }         -
    !      { Print } { GRHB }
    !
    ! grid on select <atom selection> end { GRHB }
    !
    ! grid off
    !
    ! grid clear
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


    Integer,Allocatable,Dimension(:) :: islct, jslct, gridslct
    Character(len=*) Comlyn
    Integer Comlen

    Logical QGridGenerate, Form, QRead, QClear, QOn, QOff, QPrint
    Logical err
    Integer I
    Integer GridUnit

    QOn = .False.
    QOff = .False.
    QRead = .False.
    QClear = .False.
    QPrint = .False.
    QGrHBON = .False.
    QGridGenerate = .False.

    !  Parse the Parameters
    QOn = (IndxA(comLyn,comLen,'ON').GT.0)
    QOff = (IndxA(comLyn,comLen,'OFF').GT.0)
    QRead = (IndxA(comLyn,comLen,'READ').GT.0)
    QClear = (IndxA(comLyn,comLen,'CLEA').GT.0)
    QPrint = (IndxA(comLyn,comLen,'PRIN').GT.0)
    QGrHBON = (IndxA(comLyn,comLen,'GRHB').GT.0)
    QGridGenerate = (IndxA(comLyn,comLen,'GENE').GT.0)

    !  Number of grids to set-up (equal to number of radii (plus one for electrostatic))
    If (QGridGenerate) Then
       ! natom: num of atoms in the psf. This variable is from the module ltm/psf   
       call chmalloc('grid.src','GRIDSet','islct',natom,intg=islct)
       call chmalloc('grid.src','GRIDSet','jslct',natom,intg=jslct)
       
       ! The atom selection is a selection of 26 probe atoms 
       ! probes, so NGrid0 = 26 
       NGrid0 = AtmSel(comLyn, comLen, islct, .FALSE.)
       call chmalloc('grid.src','GRIDSet','Gridslct',NGrid0,intg=Gridslct)
       
       ! crl means chm_Real. see the file memory/alloc_mod.src              
       call chmalloc('grid.src','GRIDSet','GridRadii',NGrid0,crl=GridRadii)
           
       ! Use QGrHBON to control whether include hydrogen bond grids or not. 
       If (QGrHBON) Then
          NGrid = NGrid0 + 3 
          If (PrnLev .ge. 2) Write(OutU, '(a)') &
             "Grid generation with hydrogen bond grids"
       Else
          NGrid = NGrid0 + 1
          If (PrnLev .ge. 2) Write(OutU, '(a)') &
             "Grid generation without hydrogen bond grids"
       EndIf
           
       !  Get maximum energy for grid edges
       GridForce = GTrmf(Comlyn,Comlen,'FORC', ThrHun )
       !  Specify origin of grid
       XGridCenter = GTrmf(Comlyn,Comlen,'XCEN', Zero )
       YGridCenter = GTrmf(Comlyn,Comlen,'YCEN', Zero )
       ZGridCenter = GTrmf(Comlyn,Comlen,'ZCEN', Zero )
       !  Extent of grid about the origin
       XGridMax = GTrmf(Comlyn,Comlen,'XMAX', Zero )
       YGridMax = GTrmf(Comlyn,Comlen,'YMAX', Zero )
       ZGridMax = GTrmf(Comlyn,Comlen,'ZMAX', Zero )
       !  Specify the hbond energy function 
       RCTA = GTrmf(Comlyn,Comlen,'RCTA', 0.0 )
       RCTB = GTrmf(Comlyn,Comlen,'RCTB', 1.0 )
       Hmax = GTrmf(Comlyn,Comlen,'HMAX', -1.0 )
       If (XGridMax .le. Zero) Then
          CALL WRNDIE(1,'<GridSetUp>','XMAX not set')
       EndIf
       If (YGridMax .le. Zero) Then
          CALL WRNDIE(1,'<GridSetUp>','YMAX not set')
       EndIf
       If (ZGridMax .le. Zero) Then
          CALL WRNDIE(1,'<GridSetUp>','ZMAX not set')
       EndIf
       
       !  Get optional grid spacing
       DGrid = GTrmf(Comlyn,Comlen,'DGRI', Half )
       !  What unit to write the grid potentials
       GridUnit = GTrmI(Comlyn,Comlen,'OUTU', OutU)
       Form = .False.
       Form = (IndxA(comLyn,comLen,'FORM').GT.0)
       If (GridUnit .eq. OutU) Form = .True.
       !  Write data for parsed Parameters
       If (PrnLev .ge. 2 .and. QGrHBON) Then
            Write(OutU,'(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NGrid-3,' atom types plus 2 hydrogen bond plus electrostatics'
            Write(OutU,'(a,3f10.5)') &
            ' GridSetUp: Hydrogen bond grid potential', &
            Rcta, Rctb, Hmax 
       EndIf
       If (PrnLev .ge. 2 .and. .not. QGrHBON) Write(OutU, '(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NGrid-1,' atom types plus electrostatics'
       If (PrnLev .ge. 2) Write(OutU,'(a,i4)') &
            ' GridSetUp: and written to unit', GridUnit
       If (PrnLev .ge. 2) Write(OutU,'(a,3f10.5)') &
            ' GridSetUp: Grid centered at', &
            XGridCenter, YGridCenter, ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(3(a,f10.5,a,f10.5/))') &
            ' GridSetUp: Grid runs from (X)', &
            -XGridMax/2+XGridCenter,' -', XGridMax/2+XGridCenter, &
            ' GridSetUp: Grid runs from (Y)', &
            -YGridMax/2+YGridCenter,' -', YGridMax/2+YGridCenter, &
            ' GridSetUp: Grid runs from (Z)', &
            -ZGridMax/2+ZGridCenter,' -', ZGridMax/2+ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.5)') &
            ' GridSetUp: With a grid spacing of', DGrid
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.3,a)') &
            ' GridSetUp: Force constant at grid edge set to ', &
            GridForce,' kcal/mol/A^2'

       call set_param("XGRIDMAX", XGridMax)
       call set_param("YGRIDMAX", YGridMax)
       call set_param("ZGRIDMAX", ZGridMax)

       call set_param("XGRIDCEN", XGridCenter)
       call set_param("YGRIDCEN", YGridCenter)
       call set_param("ZGRIDCEN", ZGridCenter)
       
       call set_param("RCTA", RCTA)
       call set_param("RCTB", RCTB)
       call set_param("HMAX", HMAX)
       
       !  Determine space needs for grids
       NGridX = XGridMax / DGrid + 2
       NGridY = YGridMax / DGrid + 2
       NGridZ = ZGridMax / DGrid + 2
       XGridMax = XGridMax
       YGridMax = YGridMax
       ZGridMax = ZGridMax
       If (PrnLev .ge. 2) Write(OutU,'(a,i8,a)') &
            ' GridSetUp: Allocating ', NGrid*NGridX*NGridY*NGridZ, &
            ' Real(chm_Real) words for grid potentials.'
       call chmalloc('grid.src','GRIDSet','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)

       Call CalcGrid(ISlct,JSlct, GridSlct, &
            XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            RCTA, RCTB, HMAX, &
            NGrid, NGridX, NGridY, NGridZ, &
            GridPot, Gridradii, GridUnit, &
            QGrHBon, Form)

       Call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            GridForce, NGrid, NGridX, NGridY, NGridZ, &
            GridPot, GridRadii, GridUnit, QGrHBon, &
            Form)

       If (QPrint) Then
          Call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
               XGridMax, YGridMax, ZGridMax, DGrid, &
               GridForce, NGrid, NGridX, NGridY, NGridZ, &
               GridPot, GridRadii, OutU, QGrHBon, &
               .True.)
       EndIf

       call chmdealloc('grid.src','GRIDSet','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)
       call chmdealloc('grid.src','GRIDSet','islct',natom,intg=islct)
       call chmdealloc('grid.src','GRIDSet','jslct',natom,intg=jslct)
       call chmdealloc('grid.src','GRIDSet','Gridslct',NGrid0,intg=Gridslct)
       call chmdealloc('grid.src','GRIDSet','GridRadii',NGrid0,crl=GridRadii)

    ElseIf (QRead) Then

       If(QGridOK) &
            Call WrnDie(1,'<GridSetUp>', &
            'Grid already set-up, use clear first')
#if KEY_PARALLEL==1
       !      ----- Master does the reading -----
       !if(iolev.ge.0)then
       IF (MYNOD == 0) THEN
#endif 
          !  What unit to read the grid potentials from
          GridUnit = GTrmI(Comlyn,Comlen,'UNIT', OutU)
          Form = .False.
          Form = (IndxA(comLyn,comLen,'FORM').GT.0)
          If (GridUnit .eq. OutU) Form = .True.

          !  Read binary file of Grid Potentials

          If (.not. Form) Then
             If(PrnLev .ge.2) Write(OutU,'(a,i4)') &
                  ' Grid potentials read from binary file on unit', &
                  GridUnit
             Call Rdtitl(TitleB, NtitlB, GridUnit, -1)
             Call Wrtitl(TitleB, NtitlB, OutU, 0)
             Read(GridUnit) NGrid, NGridX, NGridY, NGridZ
             Read(GridUnit) XGridCenter, YGridCenter, ZGridCenter
             Read(GridUnit) XGridMax, YGridMax, ZGridMax, &
                  DGrid, GridForce
             
          Else

             Write(OutU,'(a,i4)') &
                  ' Grid potentials read from formatted file on unit', &
                  GridUnit
             Call Rdtitl(TitleB, NtitlB, GridUnit, 0)
             Call Wrtitl(TitleB, NtitlB, OutU, 0)
             Read(GridUnit,'(4(i5,1x))') NGrid, NGridX, NGridY, NGridZ
             Read(GridUnit,'(3(f12.5,1x))') &
                  XGridCenter, YGridCenter, ZGridCenter
             Read(GridUnit,'(5(f12.5,1x))') &
                  XGridMax, YGridMax, ZGridMax, DGrid, GridForce

          Endif
#if KEY_PARALLEL==1
          call set_param("XGRIDMAX", XGridMax)
          call set_param("YGRIDMAX", YGridMax)
          call set_param("ZGRIDMAX", ZGridMax)
          
          call set_param("XGRIDCEN", XGridCenter)
          call set_param("YGRIDCEN", YGridCenter)
          call set_param("ZGRIDCEN", ZGridCenter)
          
       endif
       CALL PSND4(NGrid,1)
       CALL PSND8(xgridcenter, 1)
#endif 

       !  Write data for read Parameters
       If (PrnLev .ge. 2 .and. QGrHBON) Write(OutU,'(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NGrid-3,' atom types plus 2 hydrogen bond plus electrostatics'
       If (PrnLev .ge. 2 .and. .not. QGrHBON) Write(OutU,'(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NGrid-1,' atom types plus electrostatics'
       If (PrnLev .ge. 2) Write(OutU,'(a,i4)') &
            ' GridSetUp: and written to unit', GridUnit
       If (PrnLev .ge. 2) Write(OutU,'(a,3f10.5)') &
            ' GridSetUp: Grid centered at', &
            XGridCenter, YGridCenter, ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(3(a,f10.5,a,f10.5/))') &
            ' GridSetUp: Grid runs from (X)', &
            -XGridMax/2+XGridCenter,' -', XGridMax/2+XGridCenter, &
            ' GridSetUp: Grid runs from (Y)', &
            -YGridMax/2+YGridCenter,' -', YGridMax/2+YGridCenter, &
            ' GridSetUp: Grid runs from (Z)', &
            -ZGridMax/2+ZGridCenter,' -', ZGridMax/2+ZGridCenter
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.5)') &
            ' GridSetUp: With a grid spacing of', DGrid
       If (PrnLev .ge. 2) Write(OutU,'(a,f10.3,a)') &
            ' GridSetUp: Force constant at grid edge set to ', &
            GridForce,' kcal/mol/A^2'
       If (PrnLev .ge. 2) Write(OutU,'(a,i8,a)') &
            ' GridSetUp: Allocating ', NGrid*NGridX*NGridY*NGridZ, &
            ' Real(chm_Real) words for grid potentials.'
       !--- Whether or not use hydrogen bond grids in calculation ---
       If (PrnLev .ge. 2 .and. QGrHBon) Write(OutU, '(a)') &
          'Hydrogen bond grids will be used'
       If (PrnLev .ge. 2 .and. .not. QGrHBon) Write(OutU, '(a)') &
          'Hydrogen bond grids will not be used'

       call chmalloc('grid.src','GRIDSet','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)
       If (QGrHBon) Then
          NGrid0 = NGrid - 3 
       Else
          NGrid0 = NGrid - 1 
       EndIf
       Call chmalloc('grid.src','GRIDSet','GridRadii',NGrid0,crl=GridRadii)

       Call ReadGrid(GridPot, GridRadii, NGrid0, &
            NGrid, NGridX, NGridY, NGridZ, &
            GridUnit, Form)
       QGrid = .True.
       QGridOK = .True.

       If (QPrint) Then
          Call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
               XGridMax, YGridMax, ZGridMax, DGrid, &
               GridForce, NGrid, NGridX, NGridY, NGridZ, &
               GridPot, GridRadii, OutU, QGrHBon, &
               .True.)
       EndIf

       ! Set-up atom based grids on based on current psf
       call chmalloc('grid.src','GRIDSet','GridSlct',Natom,intg=GridSlct)
       call chmalloc('grid.src','GRIDSet','GridAtm',Natom,intg=GridAtm)
       call chmalloc('grid.src','GRIDSet','GridHBAtm',Natom,intg=GridHBAtm)
       NAtmGrd = AtmSel (comLyn, comLen, GridSlct, .FALSE.)
       Call GridAtmMap(GridSlct, GridAtm, GridHBAtm, QGrHBon, NGrid, GridRadii)

       If (PrnLev .ge. 2) Write(OutU,'(a,i5,a)') &
          'Grid mapping set-up for', NAtmGrd,' atoms'

       call chmdealloc('grid.src','GRIDSet','GridSlct',Natom,intg=GridSlct)
       NAtmGrd = Natom
    ElseIf (QOn) Then
       If (.not. QGridOK) Call WrnDie(1,'<GridSetUp>', &
          'Grid not set-up, use read first')
       If (QGridOK .and. QGrid .and. PrnLev .ge.2) &
          Write(OutU,'(a)')' Grid energy already on'
       If (.not. QGrid) Then
          If (PrnLev .ge. 2) Write(OutU,'(a)') &
             ' Grid energy will be used in energy calculations'
          QGrid = .True.
          ! Set-up atom based grids on based on current psf
          call chmalloc('grid.src','GRIDSet','GridSlct',Natom,intg=GridSlct)
          call chmalloc('grid.src','GRIDSet','GridAtm',Natom,intg=GridAtm)
          call chmalloc('grid.src','GRIDSet','GridHBAtm',Natom,intg=GridHBAtm)
          NAtmGrd = AtmSel (comLyn, comLen, GridSlct, .FALSE.)
          Call GridAtmMap(GridSlct, GridAtm, GridHBAtm, QGrHBon, NGrid, GridRadii)

          If (PrnLev .ge. 2) Write(OutU,'(a,i5,a)') &
             'Grid mapping set-up for', NAtmGrd,' atoms'

          call chmdealloc('grid.src','GRIDSet','GridSlct',Natom,intg=GridSlct)
          NAtmGrd = Natom
       Endif

    ElseIf (QOff) Then
       If (.not. QGrid .and. PrnLev .ge.2) &
          Write(OutU,'(a)')' Grid energy already off'
       If (QGrid) Then
          If (PrnLev .ge. 2) Write(OutU,'(a)') &
             'Grid energy will not be used in energy calculations'
          QGrid = .False.
          ! Free-up atom based grids on based on current psf
          call chmdealloc('grid.src','GRIDSet','GridAtm',NAtmGrd,intg=GridAtm)
          call chmdealloc('grid.src','GRIDSet','GridHBAtm',NAtmGrd,intg=GridHBAtm)
          If (PrnLev .ge. 2) Write(OutU,'(a,i5,a)') &
             'Space for', NAtmGrd,' atoms freed'
       Endif

    ElseIf (QClear) Then
       If(QGrid) Then 
          call chmdealloc('grid.src','GRIDSet','GridAtm',NAtmGrd,intg=GridAtm)
          call chmdealloc('grid.src','GRIDSet','GridHBAtm',NAtmGrd,intg=GridHBAtm)
       Endif
       If(QGridOK) Then
          IF (QGrHBon) Then
             call chmdealloc('grid.src','GRIDSet','GridRadii',NGrid-3,crl=GridRadii)
          Else 
             call chmdealloc('grid.src','GRIDSet','GridRadii',NGrid-1,crl=GridRadii)
          EndIf
          call chmdealloc('grid.src','GRIDSet','GridPot',NGrid,NGridX,NGridY,NGridZ,crl=GridPot)
       Endif
       If (PrnLev .ge. 2) Write(OutU,'(a)') &
          'All space for grid potentials freed'
       QGrid = .False.
       QGridOK = .False.
    EndIf

    Return
  End Subroutine GRIDSet

  Subroutine GridEnergy(EvdW, Elec, Ehbo, &
       XGridCenter, YGridCenter, ZGridCenter, &
       XGridMax, YGridMax, ZGridMax, DGrid, &
       GridForce, Chg, X, Y, Z, DX, DY, DZ, &
       QGrvdW, QGrElec, QCalcHBon, QGrHBON, &
       NFirst, Nlast, Qslct, ISlct) !! Yujin
    
    ! This routine evaluates the energy of atoms using a grid-based
    ! potential for vdW and electrostatics
    use chm_kinds
    use number
    use dimens_fcm
    use param
    use psf
    implicit none

    Logical QGrvdW, QGrElec, QCalcHBon, QGrHBON, QSlct
    Integer NFirst, Nlast, ISlct(*)
    Real(chm_Real) EvdW, Elec, Ehbo
    Real(chm_Real) Chg(*), X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
    Real(chm_Real) XGridCenter, YGridCenter, ZGridCenter, &
         XGridMax, YGridMax, ZGridMax, DGrid, GridForce


    Integer Igrd, IX, IY, IZ, I, NBegin, NEnd
    Real(chm_Real) EGvdW, EGElec, EDonor, EAcceptor, q, r, s, qd, rd, sd
    Real(chm_Real) Eps, E, DxTmp, DyTmp, DzTmp
    Real(chm_Real) qdp, rdp, sdp, qm, rm, sm
    Logical LXEdge, UXEdge, LYEdge, UYEdge, LZEdge, UZEdge
    logical thisone

    !yw...06-AUG-2002
    save  qdp, rdp, sdp, qd, rd, sd

    EGvdW = Zero
    EGelec = Zero
    EDonor = Zero
    EAcceptor = Zero
    NBegin = NFirst
    NEnd = NLast

    If (Nfirst .eq. 0 .or. Nlast .eq. 0) then
       Nbegin = 1
       Nend   = NatmGrd
    EndIf

    Do I = NBegin, NEnd
       if(qslct)then
          thisone=islct(i) .ne. 0
       else
          thisone=.true.
       endif
       If (thisone) Then
          DxTmp = Zero
          DyTmp = Zero
          DzTmp = Zero
          IGrd = GridAtm(i)
          If(IGrd.gt.0) Then
             Eps = Sqrt(-Eff(ItC(IaC(i))))

             q = (X(i) - XGridCenter + XGridMax/Two + TenM8)
             qd = q / DGrid
             LXEdge = ( (q-TenM8) .lt. Zero )
             UXEdge = ( (q-TenM8) .gt. XGridMax )
             IX = Max(Int(qd) + 1,1)
             IX = Min(IX,NGridX-1)      
             qd = qd - (IX-1)

             r = (Y(i) - YGridCenter + YGridMax/Two + TenM8)
             rd = r / DGrid
             LYEdge = ( (r-TenM8) .lt. Zero )
             UYEdge = ( (r-TenM8) .gt. YGridMax )
             IY = Max(Int(rd) + 1,1)
             IY = Min(IY, NGridY-1)    
             rd = rd - (IY-1)

             s = (Z(i) - ZGridCenter + ZGridMax/Two + TenM8)
             sd = s / DGrid
             LZEdge = ( (s-TenM8) .lt. Zero )
             UZEdge = ( (s-TenM8) .gt. ZGridMax )
             IZ = Max(Int(sd) + 1,1)
             IZ = Min(IZ, NGridZ-1)   
             sd = sd - (IZ-1)

             !      Write(6,*)'i eps, X, Y, Z, qd, rd, sd ',
             !     &     i, eff(itc(iac(i))), x(i), y(i), z(i), qd, rd, sd
             !      Write(6,*)' IGrd IX, IY, IZ GridPot',
             !     &     IGrd, IX, IY, IZ, GridPot(Igrd, IX, IY, IZ)

             qm = Zero
             rm = Zero
             sm = Zero
             If(LXEdge .or. UXEdge) Then
                qm = One
                qdp = qd
                qd = Zero
                If(UXEdge) qd = Zero
             EndIf

             If(LYEdge .or. UYEdge) Then
                rm = One
                rdp = rd
                rd = Zero
                If(UYEdge) rd = Zero
             EndIf

             If(LZEdge .or. UZEdge) Then
                sm = One
                sdp = sd
                sd = Zero
                If(UZEdge) sd = Zero
             EndIf

             !      Write(6,*)'qm, qdp, qd, rm, rdp, rd, sm, sdp, sd',
             !     *     qm, qdp, qd, rm, rdp, rd, sm, sdp, sd
             !     Energy - vdW
             If(QGrvdW) Then

                E = (One - qd)*(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + qd*(One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1)

                !
                E = E + Half*GridForce*DGrid*DGrid/Eps* &
                     (qm*qdp*qdp + rm*rdp*rdp + sm*sdp*sdp)
                !     Dx
                If(.not.(LXEdge .or. UXEdge)) &
                     DxTmp = Eps*( -(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + (One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     - rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + (One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dy
                If(.not.(LYEdge .or. UYEdge)) &
                     DyTmp = Eps*( -(One - qd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - qd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     - qd*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dz
                If(.not.(LZEdge .or. UZEdge)) &
                     DzTmp = Eps*( -(One - qd)*(One - rd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     - qd*rd*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !
                EGvdW = EGvdW &
                     + Eps*E  !GridPot(Igrd,IX,IY,IZ)

             EndIf            !If QGrvdW
            
             !     Energy - elec
             If(Chg(i).ne.Zero.and.QGrElec) Then
                E = (One - qd)*(One - rd)*(One - sd) &
                     *GridPot(NGrid,IX,IY,IZ) &
                                !
                     + qd*(One - rd)*(One - sd)*GridPot(NGrid,IX+1,IY,IZ) &
                     + (One - qd)*rd*(One - sd)*GridPot(NGrid,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*sd*GridPot(NGrid,IX,IY,IZ+1) &
                                !
                     + qd*rd*(One - sd)*GridPot(NGrid,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*sd*GridPot(NGrid,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*sd*GridPot(NGrid,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*sd*GridPot(NGrid,IX+1,IY+1,IZ+1)
                !
                E = E + Half*GridForce*DGrid*DGrid/Chg(i)* &
                     (qm*qdp*qdp + rm*rdp*rdp + sm*sdp*sdp)
                !
                !     Dx
                If(.not.(LXEdge .or. UXEdge)) &
                     DxTmp = DxTmp + Chg(i)*( -(One - rd)*(One - sd) &
                     *GridPot(NGrid,IX,IY,IZ) &
                                !
                     + (One - rd)*(One - sd)*GridPot(NGrid,IX+1,IY,IZ) &
                     - rd*(One - sd)*GridPot(NGrid,IX,IY+1,IZ) &
                     - (One - rd)*sd*GridPot(NGrid,IX,IY,IZ+1) &
                                !
                     + rd*(One - sd)*GridPot(NGrid,IX+1,IY+1,IZ) &
                     - rd*sd*GridPot(NGrid,IX,IY+1,IZ+1) &
                     + (One - rd)*sd*GridPot(NGrid,IX+1,IY,IZ+1) &
                                !
                     + rd*sd*GridPot(NGrid,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dy
                If(.not.(LYEdge .or. UYEdge)) &
                     DyTmp = DyTmp + Chg(i)*( -(One - qd)*(One - sd) &
                     *GridPot(NGrid,IX,IY,IZ) &
                                !
                     - qd*(One - sd)*GridPot(NGrid,IX+1,IY,IZ) &
                     + (One - qd)*(One - sd)*GridPot(NGrid,IX,IY+1,IZ) &
                     - (One - qd)*sd*GridPot(NGrid,IX,IY,IZ+1) &
                                !
                     + qd*(One - sd)*GridPot(NGrid,IX+1,IY+1,IZ) &
                     + (One - qd)*sd*GridPot(NGrid,IX,IY+1,IZ+1) &
                     - qd*sd*GridPot(NGrid,IX+1,IY,IZ+1) &
                                !
                     + qd*sd*GridPot(NGrid,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dz
                If(.not.(LZEdge .or. UZEdge)) &
                     DzTmp = DzTmp + Chg(i)*( -(One - qd)*(One - rd) &
                     *GridPot(NGrid,IX,IY,IZ) &
                                !
                     - qd*(One - rd)*GridPot(NGrid,IX+1,IY,IZ) &
                     - (One - qd)*rd*GridPot(NGrid,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*GridPot(NGrid,IX,IY,IZ+1) &
                                !
                     - qd*rd*GridPot(NGrid,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*GridPot(NGrid,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*GridPot(NGrid,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*GridPot(NGrid,IX+1,IY+1,IZ+1) ) / DGrid
                !
                EGElec = EGelec + Chg(i)*E !GridPot(NGrid,IX,IY,IZ)
             EndIf
            
             !! Added by Yujin Wu 2020/02/17
             !! Delete extra force from the edge
             !     Energy - hbonding -- acceptor
             !  Write(6,*)'GridHBAtm', GridHBAtm
             Igrd = NGrid - 2
             If(GridHBAtm(i).eq.1.and.QGrHBON) Then
                E = (One - qd)*(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + qd*(One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1)
                !
                !     Dx
                If(.not.(LXEdge .or. UXEdge)) &
                     DxTmp = DxTmp + ( -(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + (One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     - rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + (One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dy
                If(.not.(LYEdge .or. UYEdge)) &
                     DyTmp = DyTmp + ( -(One - qd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - qd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     - qd*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dz
                If(.not.(LZEdge .or. UZEdge)) &
                     DzTmp = DzTmp + ( -(One - qd)*(One - rd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     - qd*rd*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !
                EAcceptor = EAcceptor + E !GridPot(Igrd,IX,IY,IZ)
             EndIf
            
             !     Energy - hbonding -- donor 
             Igrd = NGrid - 1
             If(GridHBAtm(i).eq.-1.and.QGrHBON) Then
                E = (One - qd)*(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + qd*(One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1)
                !
                !     Dx
                If(.not.(LXEdge .or. UXEdge)) &
                     DxTmp = DxTmp + ( -(One - rd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     + (One - rd)*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - rd*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - rd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + rd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     - rd*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + (One - rd)*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + rd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dy
                If(.not.(LYEdge .or. UYEdge)) &
                     DyTmp = DyTmp + ( -(One - qd)*(One - sd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - sd)*GridPot(Igrd,IX+1,IY,IZ) &
                     + (One - qd)*(One - sd)*GridPot(Igrd,IX,IY+1,IZ) &
                     - (One - qd)*sd*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     + qd*(One - sd)*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*sd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     - qd*sd*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*sd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !     Dz
                If(.not.(LZEdge .or. UZEdge)) &
                     DzTmp = DzTmp + ( -(One - qd)*(One - rd) &
                     *GridPot(Igrd,IX,IY,IZ) &
                                !
                     - qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ) &
                     - (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ) &
                     + (One - qd)*(One - rd)*GridPot(Igrd,IX,IY,IZ+1) &
                                !
                     - qd*rd*GridPot(Igrd,IX+1,IY+1,IZ) &
                     + (One - qd)*rd*GridPot(Igrd,IX,IY+1,IZ+1) &
                     + qd*(One - rd)*GridPot(Igrd,IX+1,IY,IZ+1) &
                                !
                     + qd*rd*GridPot(Igrd,IX+1,IY+1,IZ+1) ) / DGrid
                !
                EDonor = EDonor + E !GridPot(Igrd,IX,IY,IZ)
             EndIf
            
          !! End of editing (Yujin Wu)        
  
          EndIf

          Dx(i) = Dx(i) + DxTmp + GridForce*qm*qdp*DGrid*DGrid
          Dy(i) = Dy(i) + DyTmp + GridForce*rm*rdp*DGrid*DGrid
          Dz(i) = Dz(i) + DzTmp + GridForce*sm*sdp*DGrid*DGrid

       Endif

    EndDo

    Elec = Elec + EGelec
    EvdW = EvdW + EGvdW
    Ehbo = Ehbo + EDonor + EAcceptor

    Return
  End Subroutine GridEnergy

  Subroutine GridAtmMap(GrdSlct, GridAtm, GridHBAtm, QGrHBon, NGrid, GridRadii)
    !  This subroutine fills an Natom length array with pointers into
    !  appropriate grid potential based on vdW radii
    use chm_kinds
    use number
    use dimens_fcm
    use psf
    use param
    implicit none

    Integer :: GrdSlct(Natom), GridAtm(Natom), GridHBAtm(Natom), NGrid
    Real(chm_Real) DGrid, GridRadii(NGrid)
    logical :: QGrHBon

    Real(chm_Real) Min, radius
    Integer Iatm, Igrd, Jgrd, HBatm, vdwradii
    
    !   H-bond grid
    GridHBAtm = 0
    If (QGrHBon) then
       vdwradii = NGrid -3
    Else
       vdwradii = NGrid -1
    EndIf

    Do Iatm = 1, Natom
       If(GrdSlct(Iatm).gt.0) Then
          radius = VdwR(ITC(IAC(Iatm)))
          Min = Hundrd
          IGrd = 0
          Do JGrd = 1, vdwradii             !! Change by Yujin (NGrid - 3)
             If ( Abs( radius - Gridradii(JGrd)) .le. Min ) Then
                Min = Abs( radius - Gridradii(JGrd))
                IGrd = JGrd
             Endif
          Enddo
          If(Igrd .le. 0 .or. Min .gt. PtOne) &
               Call WrnDie(1,'<GridAtmMap>', &
               'Atom radius differs significantly from references')
          GridAtm(Iatm) = IGrd
       Else
          GridAtm(Iatm) = -1
       Endif

    EndDo
    !          Write(6,'(2(i5,1x))')
    !     &      (iatm,GridAtm(iatm), iatm=1,Natom)

    If (NACC.GT.0.and.NACC.LE.NATOM) Then
       Do HBatm = 1, NACC
          GridHBAtm(IACC(HBatm)) = 1           !! 1 for acceptor
       Enddo
    EndIf

    If (NDON.GT.0.and.NDON.LE.NATOM) Then
       Do HBatm = 1, NDON
          GridHBAtm(IHD1(HBatm)) = -1           !! -1 for donor
       Enddo
    EndIf
    Return
  End Subroutine GridAtmMap

  Subroutine ReadGrid(Gridpot, Gridradii, NGrid0, &
       NGrid, NGridX, NGridY, NGridZ, &
       GridU, Form)
    ! This subroutine reads the grid potential from unit GridU
    ! in either formatted (FORM=TRUE) or unformatted forms.
    use chm_kinds
    use stream
    use parallel
    implicit none
    Integer NGrid, NGrid0, NGridX, NGridY, NGridZ, GridU
    Real(chm_Real) GridPot(NGrid, NGridX, NGridY, NGridZ),  &
         GridRadii(NGrid0)
    Logical Form
    !
    Real(chm_Real) XTmp, YTmp, ZTmp
    Integer IGrd, IX, IY, IZ
    Character(len=80) Key

#if KEY_PARALLEL==1
    !      ----- Master does the reading -----
    IF (MYNOD == 0) THEN
!    if(iolev.ge.0)then
#endif 
       If(.not. Form) Then

          Read(GridU) GridRadii
          Read(GridU) GridPot

       Else
       
          Read(GridU,'(5(f12.5,1x))') &
                 (GridRadii(IGrd), IGrd=1, NGrid0)  !! With hbond grids 

          Do Igrd = 1, NGrid
             Read(GridU,'(a)') Key
             Do IX = 1, NGridX
                Do IY = 1, NGridY
                   Do IZ = 1, NGridZ
                      Read(GridU,'(3f12.6,1x,e12.6)') &
                           XTmp, YTmp, &
                           ZTmp, GridPot(IGrd,IX, IY, IZ)
                   EndDo
                EndDo
             EndDo
          EndDo

       Endif

#if KEY_PARALLEL==1
    endif
    CALL PSND8(GridRadii(1:NGrid0), NGrid0)           
    CALL PSND8m(GridPot,NGrid*NGridX*NGridY*NGridZ)
#endif 
    !
    Return
  End Subroutine ReadGrid

  Subroutine CalcGrid(Islct, Jslct, GridSlct, &
       XGridCenter, YGridCenter, ZGridCenter, &
       XGridMax, YGridMax, ZGridMax, DGrid, &
       RCTA, RCTB, HMAX, &
       NGrid, NGridX, NGridY, NGridZ, &
       GridPot, GridRadii, GridU, QGrHBon, Form)

#if KEY_FLUCQ==1
    use flucqm,only:fqcfor
#endif

    use chm_kinds
    use chm_types
    use memory
    use dimens_fcm
    use stream
    use code
    use coord
    use coordc
    use energym
    use bases_fcm
    use psf
    use inbnd
    use param
    use image
#if KEY_FLUCQ==1
    use flucq
#endif 
    implicit none

    Integer,Allocatable,Dimension(:) :: Iskip
    Real(chm_Real),Allocatable,Dimension(:) :: RTemp
    Integer ISlct(*), JSlct(*), GridSlct(*)
    Integer NGrid, NGridX, NGridY, NGridZ, GridU
    Real(chm_Real) GridPot(NGrid, NGridx, NGridY, NGridZ)
    Real(chm_Real) XGridCenter, YGridCenter, ZGridCenter, &
         XGridMax, YGridMax, ZGridMax, DGrid, &
         RCTA, RCTB, HMAX, &
         GridRadii(NGrid)
    Real(chm_Real) XJ, YJ, ZJ 
    Real(chm_Real) XI, YI, ZI 
    Real(chm_Real) DX, DY, DZ, RIJ, RIJ2, FA, FB, EHB, ETmp
    Integer LENT
    Logical Form
    Logical QGrHBon
    !
    !
    Integer I, IX, IY, IZ, IGrid, JGrid, Icnt, HBatm, vdwGrid
    Logical LDUP

    ! hwm: per BRB's suggestion on 220722
    !LENT=MAX(NBONDT,NTHETT,NPHIT,NIMPHT &
    LENT=MAX(NBONDT2,NTHETT,NPHIT,NIMPHT &
#if KEY_CMAP==1
         ,NCRTT &      
#endif
         )
    call chmalloc('grid.src','CalcGrid','Iskip',LENT,intg=Iskip)
    call chmalloc('grid.src','CalcGrid','RTemp',natom,crl=RTemp)

    ! Write out basic Parameters for atom grids
    If (PrnLev .ge. 2) Write(OutU,'(a)') &
         ' GridSetUp:  Grid #    Radius'
    Icnt = 0
    Do I = 1, Natom
       If(Islct(i).ne.0) Then
          Icnt = Icnt + 1
          GridSlct(Icnt) = I
          GridRadii(Icnt) = VDWR(ITC(IAC(I)))
          If (PrnLev .gt. 2) Write(OutU,'(a,2x,i4,2x,f10.5)') &
               ' GridSetUp:', Icnt, VDWR(ITC(IAC(I)))
       EndIf
    EndDo

    !!---------------------Vdw and elec grid setup----------------------
    !! Num of vdw grids
    If (QGrHBon) Then
       vdwGrid = NGrid - 3
    Else
       vdwGrid = NGrid - 1
    EndIf

    !! Atom selection
    Do I = 1, Natom
       Islct(I) = 1
       Jslct(I) = 1
    Enddo

    Do Igrid = 1, vdwGrid
       Islct(Gridslct(IGrid)) = 0
       Jslct(Gridslct(IGrid)) = 0
    EndDo

    !! Calculation starts
    Do Igrid = 1, vdwGrid
       LDUP = .True.
       ISlct(GridSlct(IGrid)) = 1
       Do IX = 1, NGridX
          X(Gridslct(IGrid)) = XGridCenter - XGridMax / 2 &
               + (IX-1)*DGrid
          Do IY = 1, NGridY
             Y(Gridslct(IGrid)) = YGridCenter - YGridMax / 2 &
                  + (IY-1)*DGrid
             Do IZ = 1, NGridZ
                Z(Gridslct(IGrid)) = ZGridCenter - ZGridMax / 2 &
                     + (IZ-1)*DGrid
                Call Inter2(X, Y, Z, Islct, JSlct, &
                     ISkip, RTemp, &
#if KEY_FLUCQ==1
                     QFLUC,FQCFOR,                & 
#endif
                     BIMAG%IMATTR,BIMAG%IMATPT, &
                     LDUP, .FALSE., .FALSE.)
                LDUP = .False.
                GridPot(IGrid,IX, IY, IZ) = ETerm(VDW)
                If(IGrid .eq. 1) &
                     GridPot(NGrid,IX, IY, IZ) = ETerm(Elec)+Eterm(GBEnr)
             EndDo
          EndDo
       EndDo
      
       !! Clear previous grid probe selection 
       ISlct(GridSlct(IGrid)) = 0
    EndDo

    IF (QGrHBon) Then
    !!----------Calculate parameters for hydrogen bond grids----------
    RIJ = (RCTA - RCTB) * (RCTA - RCTB)  
    If (RIJ.EQ.0) Then
       If (PrnLev .ge. 2) Write(OutU, '(a)') & 
          'Please check h-bond grid potential setup'
       RIJ = 1
       HMax = 0  ! This will set all energy value to be 0
    EndIf
    FA = -4 * HMax / RIJ
    FB = 0.5 *(RCTA + RCTB)
   
    !!--------------------------Donor grid setup----------------------
    !! Prepare atom selection
    Do I = 1, NAtom
       ISlct(I) = 0
    EndDo

    Do JGrid = 1, vdwGrid
       Islct(Gridslct(JGrid)) = 0
    EndDo

    If (NDON.GT.0.and.NDON.LE.NATOM) Then
       Do HBatm = 1, NDON
          ISlct(IHD1(HBatm)) = 1
       EndDo
    EndIf

    !! Calculation starts
    IGrid = NGrid - 2
    Do IX = 1, NGridX                                      !! Grid point
       XI = XGridCenter - XGridMax / 2 + (IX-1)*DGrid
       Do IY = 1, NGridY
          YI = YGridCenter - YGridMax / 2 + (IY-1)*DGrid
          Do IZ = 1, NGridZ
             ZI = ZGridCenter - ZGridMax / 2 + (IZ-1)*DGrid

             EHB = 0 
             Do HBatm = 1, NAtom
                ETmp = 0
                If (ISlct(HBatm).eq.1) Then                !! Receptor coor
                   XJ = X(HBatm)
                   YJ = Y(HBatm)
                   ZJ = Z(HBatm)
                  
                   DX = XI - XJ 
                   DY = YI - YJ 
                   DZ = ZI - ZJ 
                   RIJ2 = DX * DX + DY * DY + DZ * DZ
                   RIJ = SQRT(RIJ2)

                   If (RIJ.gt.RCTA.AND.RIJ.lt.RCTB) then
                      ETmp = FA * (RIJ - FB) * (RIJ - FB) + HMAX
                   Else
                      ETmp = 0
                   EndIf 

                EndIf
                EHB = EHB + ETmp
             EndDo

             GridPot(IGrid, IX, IY, IZ) = EHB 
          EndDo
       EndDo
    EndDo

    !!--------------------------Acceptor grid setup----------------------
    !! Prepare atom selection 
    Do I = 1, NAtom
       ISlct(I) = 0
    EndDo

    Do JGrid = 1, vdwGrid
       Islct(Gridslct(JGrid)) = 0
    EndDo

    If (NACC.GT.0.and.NACC.LE.NATOM) Then
       Do HBatm = 1, NACC
          ISlct(IACC(HBatm)) = 1
       EndDo
    EndIf

    !! Calculation starts
    IGrid = NGrid - 1
    Do IX = 1, NGridX                                      !! Grid coor
       XI = XGridCenter - XGridMax / 2 + (IX-1)*DGrid
       Do IY = 1, NGridY
          YI= YGridCenter - YGridMax / 2 + (IY-1)*DGrid
          Do IZ = 1, NGridZ
             ZI = ZGridCenter - ZGridMax / 2 + (IZ-1)*DGrid

             EHB = 0 
             Do HBatm = 1, NAtom
                ETmp = 0 
                If (ISlct(HBatm).eq.1) Then                !! Receptor coor 
                   XJ = X(HBatm)
                   YJ = Y(HBatm)
                   ZJ = Z(HBatm)

                   DX = XI - XJ 
                   DY = YI - YJ 
                   DZ = ZI - ZJ 
                   RIJ2 = DX * DX + DY * DY + DZ * DZ
                   RIJ = SQRT(RIJ2)

                   If (RIJ.gt.RCTA.AND.RIJ.lt.RCTB) then
                      ETmp = FA * (RIJ - FB) * (RIJ - FB) + HMAX
                   Else
                      ETmp = 0 
                   EndIf 

                EndIf
                EHB = EHB + ETmp
             EndDo

             GridPot(IGrid, IX, IY, IZ) = EHB 
          EndDo
       EndDo
    EndDo
    EndIF

    !!--------------------------End of grid calculation----------------------
    call chmdealloc('grid.src','CalcGrid','Iskip_sv',LENT,intg=Iskip)
    call chmdealloc('grid.src','CalcGrid','RTemp_sv',natom,crl=RTemp)
    Return

  End Subroutine CalcGrid

  Subroutine WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
       XGridMax, YGridMax, ZGridMax, DGrid, &
       GridForce, NGrid, NGridX, NGridY, NGridZ, &
       GridPot, GridRadii, GridU, QGrHBon, Form)

    !
    !
    use chm_kinds
    use ctitla
    use stream
    implicit none
    !
    !

    Integer NGrid, NGridX, NGridY, NGridZ, GridU
    Real(chm_Real) GridPot(NGrid, NGridx, NGridY, NGridZ)
    Real(chm_Real) XGridCenter, YGridCenter, ZGridCenter, &
         XGridMax, YGridMax, ZGridMax, DGrid, &
         GridForce, GridRadii(NGrid)
    Logical Form
    Logical QGrHBon
    !
    !
    Real(chm_Real) XTmp, YTmp, ZTmp
    Integer I, IX, IY, IZ, IGrd, vdwGrid

    IF(IOLEV.LT.0) RETURN

    If (QGrHBon) Then
       vdwGrid = NGrid - 3
    Else
       vdwGrid = NGrid - 1
    EndIf
    If (PrnLev .ge. 2) Write(OutU, '(a,i4)') &
       'Total number of grids for writing', NGrid

    !  Write out binary file of Grid Potentials
    If (.not. Form) Then
       Write(OutU,'(a,i4)') &
            'Grid potentials written to binary file on unit', &
            GridU
       Call Wrtitl(Titlea, Ntitla, GridU, -1)
       Write(GridU) NGrid, NGridX, NGridY, NGridZ
       Write(GridU) XGridCenter, YGridCenter, ZGridCenter
       Write(GridU) XGridMax, YGridMax, ZGridMax, DGrid, GridForce
       Write(GridU) GridRadii(1:vdwGrid)
       Write(GridU) GridPot

    Else

       Write(OutU,'(a,i4)') &
            'Grid potentials written to formatted file on unit', &
            GridU
       Call Wrtitl(Titlea, Ntitla, GridU, 0)
       Write(GridU,'(4(i5,1x))') NGrid, NGridX, NGridY, NGridZ
       Write(GridU,'(3(f12.5,1x))') &
            XGridCenter, YGridCenter, ZGridCenter
       Write(GridU,'(5(f12.5,1x))') &
            XGridMax, YGridMax, ZGridMax, DGrid, GridForce
       Write(GridU,'(5(f12.5,1x))') (GridRadii(i), i=1, vdwGrid)
       Do Igrd = 1, NGrid
          Write(GridU,'(a,i4)')' Grid #', Igrd
          Do IX = 1, NGridX
             XTmp = XGridCenter - XGridMax / 2 &
                  + (IX-1)*DGrid
             Do IY = 1, NGridY
                YTmp = YGridCenter - YGridMax / 2 &
                     + (IY-1)*DGrid
                Do IZ = 1, NGridZ
                   ZTmp = ZGridCenter - ZGridMax / 2 &
                        + (IZ-1)*DGrid
                   Write(GridU,'(3f12.6,1x,e12.6)') &
                        XTmp, YTmp, &
                        ZTmp, GridPot(IGrd,IX, IY, IZ)
                EndDo
             EndDo
          EndDo
       EndDo
    EndIf
    Return
  END Subroutine WriteGrid

#endif /* (grid_main)*/

end module grid_dock

