module shell
  use chm_kinds
  use chm_types
  use dimens_fcm
  implicit none
!
  !  shell.fcm - Contains global variables for solvent shell
  !              decomposition
  !
  !  TR 24. 10. 2002
  !
  !  MXSHL   - Maximum number of shells
  !  QSHELL  - is shell in use ?
  !  QSHLUP  - is shell-data up to date ?
  !  QSHATM  - if .true. -> don't expand whole residues
  !            (default=.false.)
  !  QSHIMG  - is shell to use images (default=.true.)
  !
  !  NSHL    - Number of shells
  !  NASHL   - Number of atoms in each shell
  !  NSHBLK  - Number of atoms in the bulk (outside the last shell)
  !  SHTHK   - Shell thickness (NEW: is now an array to allow for shells
  !                                  with different thickness)
  !  SHTHK2  - Shell thickness**2 (NEW: to allow for shells with different
  !                                thickness, compute less squares in SHLXDI!)
  !  NSHSLU  - Number of solute atoms
  !  NSHSLV  - Number of solvent atoms
  !
  !  SOLUL  - pointer to list of solute atoms
  !  SOLVL  - pointer to list of solvent atoms
  !  SHLLST -  pointer for the list of atoms in each shell
  !  SHBLKL - pointer for tle list of bulk atoms
  !
  INTEGER,PARAMETER :: MAXSHL=50
  !
  LOGICAL QSHELL,QSHLUP,QSHATM,QSHIMG

  INTEGER,save :: NSHL,NASHL(1:MAXSHL),NSHBLK
  INTEGER,save :: NSHSLU,NSHSLV

  real(chm_real)  SHTHK(MAXSHL),SHTHK2(MAXSHL)

  INTEGER,allocatable,dimension(:) :: SOLUL,SOLVL,SHBLKL

  integer,allocatable,dimension(:,:) :: shllst

contains

#if KEY_SHELL==0 /*shell_main*/
  SUBROUTINE SHLINI
    CALL WRNDIE(-1,'<SHELL>','SHELL is not currently compiled.')
    RETURN
  end SUBROUTINE SHLINI
#else /* (shell_main)*/

  subroutine shell_init()
    qshell=.false.
    qshatm=.false.
    qshlup=.false.
    qshimg=.false.
    nshl=1
    nshslu=0
    nshslv=0
    shthk(1)=4.75d0
    nshblk=0
    nashl(1:maxshl)=0
    return
  end subroutine shell_init

  SUBROUTINE SHLINI
    !
    !     SHLINI initializes the parameters and data-structures for solvent
    !     shell decomposition
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use comand
  use coord
  use image
  use number
  use psf
  use stream
  use memory
  use exfunc
  use string
    !
    integer,allocatable,dimension(:) :: ISEL
    real(chm_real)  OSHTH
    INTEGER I,ONSHL,NSLV,NSLU
    CHARACTER(len=4)  WRD
    !
    !
    !     check if user wants to clear all SHELL related variables
    IF(INDXA(COMLYN,COMLEN,'OFF') > 0) THEN
       IF(QSHELL) THEN
          call chmdealloc('shell.src','SHLINI','SHBLKL',NATOM,intg=SHBLKL)
          call chmdealloc('shell.src','SHLINI','SHLLST',NATOM,NSHL,intg=SHLLST)
          call chmdealloc('shell.src','SHLINI','SOLUL',NATOM,intg=SOLUL)
          call chmdealloc('shell.src','SHLINI','SOLVL',NATOM,intg=SOLVL)
          NASHL(1:maxshl)=0
          NSHBLK=0
          !           restore defaults
          NSHL=0
          SHTHK(1)=4.75D0
          SHTHK2(1)=4.75D0**2
          NSHSLU=0
          NSHSLV=0
          QSHELL=.FALSE.
          QSHLUP=.FALSE.
          QSHATM=.FALSE.
          QSHIMG=.FALSE.
       ELSE
          CALL WRNDIE(3,'<SHELL>','SHELL is currently not in use.')
       ENDIF
       RETURN
    ELSE IF(INDXA(COMLYN,COMLEN,'UPDA') > 0) THEN
       !     check if user wants to do a shell update
       IF(QSHELL) THEN
          CALL SHLUPD
          !
          !     This block of code is currently disabled. The general idea is that
          !     a logical flag (QSHLUP) tells an application using SHELL whether the
          !     data is up to date so that if several independent SHELL-using functions
          !     are invoked for the same coordinate-set/trajectory-frame only the first
          !     one calls SHLUPD while the other can safely assume that all is up to
          !     date and well.
          !     BUT: currently this flag is not set to .false. when anything in the
          !     coordinates/psf changes so a user calling SHELL UPDATE manually would
          !     have to use 'FORCE' all but the first time (e.g. in a traj read loop)
          !     which is not what was intended.
          !
          !     This line is only to parse the keyword from the command line to avoid
          !     messages from XTRANE. REMOVE if 'FORCE' is to be put to its original
          !     use!
          IF(INDXA(COMLYN,COMLEN,'FORC') > 0) CONTINUE
          !TR
          !TR            IF((INDXA(COMLYN,COMLEN,'FORC') > 0).OR.(.NOT.QSHLUP)) THEN
          !TR               CALL SHLUPD
          !TR            ELSE
          !TR               WRITE(OUTU,810) 
          !TR 810           FORMAT(/' SHELL seems to be up to date. --  ',
          !TR     $              'use FORCe to force an update'/)
          !TR            ENDIF
       ELSE
          CALL WRNDIE(3,'<SHELL>','SHELL is currently not in use.')
       ENDIF
       RETURN
    ELSE IF(INDXA(COMLYN,COMLEN,'STAT') > 0) THEN
       !     user wants output of shell data
       IF(QSHELL) THEN
          CALL SHLPRN(NATOM)
       ELSE
          CALL WRNDIE(3,'<SHELL>','SHELL is currently not in use.')
       ENDIF
       RETURN
    ELSE IF(INDXA(COMLYN,COMLEN,'DEFI') > 0) THEN
       !     user wants to convert shell to definition
       IF(QSHELL) THEN
          IF(.NOT.QSHLUP) &
               CALL WRNDIE(1,'<SHELL>','SHELL is NOT up to date.')
          CALL SHLDEF(COMLYN,COMLEN)
       ELSE
          CALL WRNDIE(3,'<SHELL>','SHELL is currently not in use.')
       ENDIF
       RETURN
    ENDIF
    !
    !
    !     NONE OF THE ABOVE -> INITIALIZE/MODIFY
    !
    ONSHL=NSHL
    NSHL=GTRMI(COMLYN,COMLEN,'NSHL',NSHL)
    !     check if we want atom-only selection
    IF(INDXA(COMLYN,COMLEN,'ATOM') > 0) QSHATM=.TRUE.
    !
    !     check if there are images -> ues them
    IF(NATIM > NATOM) QSHIMG=.TRUE.
    !     check if we want to go without images
    IF(INDXA(COMLYN,COMLEN,'NOIM') > 0) QSHIMG=.FALSE.
    !
    !     check for sanity
    !     check for NSHL too large/small
    IF(NSHL > MAXSHL) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,839) NSHL,MAXSHL,MAXSHL
839    FORMAT(' ***** WARNING ***** from SHELL -- NSHL, ',I6, &
            ' was larger than ',I6 &
            ,'.'/' It will be set to ',I6,'.')
       NSHL=MAXSHL
    ENDIF
    IF(NSHL <= 0) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,840) NSHL
840    FORMAT(' ***** WARNING ***** from SHELL -- NSHL, ',I6, &
            ' was lower than 1.'/' It will be set to 1.')
       NSHL=1
    ENDIF
    !
    !     allocate space if SHELL is not in use
    IF(.NOT.QSHELL) THEN
       call chmalloc('shell.src','SHLINI','SOLUL',NATOM,intg=SOLUL)
       call chmalloc('shell.src','SHLINI','SOLVL',NATOM,intg=SOLVL)
       call chmalloc('shell.src','SHLINI','SHBLKL',NATOM,intg=SHBLKL)
       call chmalloc('shell.src','SHLINI','SHLLST',NATOM,NSHL,intg=SHLLST)
       QSHLUP=.FALSE.
    ENDIF
    !     free SHLLST if the number of shells has changed while
    !     SHELL is in use
    IF((QSHELL).AND.(NSHL /= ONSHL)) THEN
       call chmdealloc('shell.src','SHLINI','SHLLST',NATOM,ONSHL,intg=SHLLST)
       call chmalloc('shell.src','SHLINI','SHLLST',NATOM,NSHL,intg=SHLLST)
       QSHLUP=.FALSE.
    ENDIF
    !
    !     now process the atom selections
    NSLU=0
    NSLV=0
    NSLU=INDXA(COMLYN,COMLEN,'SOLU')
    NSLV=INDXA(COMLYN,COMLEN,'SOLV')
    !
    call chmalloc('shell.src','SHLINI','ISEL',NATOM,intg=ISEL)
    IF((NSLU > 0).OR.(NSLV.GT.0)) THEN
       IF(NSLV <= 0) THEN
          !           only read solute
          CALL SHLSEL(COMLYN,COMLEN,solul,NSHSLU,ISEL)
       ELSE IF(NSLU <= 0) THEN
          !           only read solvent
          CALL SHLSEL(COMLYN,COMLEN,solvl,NSHSLV,ISEL)
       ELSE IF(NSLU < NSLV) THEN
          !           read solute and then solvent
          CALL SHLSEL(COMLYN,COMLEN,solul,NSHSLU,ISEL)
          CALL SHLSEL(COMLYN,COMLEN,solvl,NSHSLV,ISEL)
       ELSE
          !           read solvent and then solute
          CALL SHLSEL(COMLYN,COMLEN,solvl,NSHSLV,ISEL)
          CALL SHLSEL(COMLYN,COMLEN,solul,NSHSLU,ISEL)
       ENDIF
       QSHLUP=.FALSE.
    ENDIF
    call chmdealloc('shell.src','SHLINI','ISEL',NATOM,intg=ISEL)
    !
    !     check for sanity
    IF(NSHSLU <= 0) THEN
       CALL WRNDIE(-3,'<SHELL>','No solute selected  (NSHSLU=0).')
    ENDIF
    IF(NSHSLV <= 0) THEN
       CALL WRNDIE(-3,'<SHELL>','No solvent selected (NSHSLV=0).')
    ENDIF
    !
    !
    !     NEW: by now all of the command line should have been parsed
    !          now implement our new version with different shell thickness
    !          i.e. look for keyword (SHBOundaries) and parse NSHL real numbers
    WRD=NEXTA4(COMLYN,COMLEN)
    IF(WRD == 'SHTH') THEN
       !        i.e. use classic setup (shells of identical thickness)
       OSHTH=SHTHK(1)
       SHTHK(1)=NEXTF(COMLYN,COMLEN)
       !      if the shell-thickness changed -> we're not up to date
       IF(OSHTH /= SHTHK(1)) QSHLUP=.FALSE.
       !        fill SHTHK()
       DO I=1,NSHL
          SHTHK(I)=SHTHK(1)*I
          SHTHK2(I)=SHTHK(I)**2
       ENDDO
    ELSEIF(WRD == 'SHBO') THEN
       DO I=1,NSHL
          SHTHK(I)=NEXTF(COMLYN,COMLEN)
          SHTHK2(I)=SHTHK(I)**2
       ENDDO
       DO I=2,NSHL
          IF((WRNLEV >= 2).AND.(SHTHK(I) <= SHTHK(I-1))) &
               CALL WRNDIE(3,'<SHELL>', &
               'SHELL borders must be increasing!.')
       ENDDO
    ELSE
       CALL WRNDIE(1,'<SHELL>','unknown keyword '//WRD//' found')
    ENDIF
    !     check shell thickness
    IF(SHTHK(1) <= 0) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,841) SHTHK(1)
841    FORMAT(' ***** WARNING ***** from SHELL -- SHTH, ',F10.4, &
            ' was lower or equal 0.'/' It will be set to 4.75')
       SHTHK(1)=4.75D0
    ENDIF
    !
    !     everything seems to be OK -> GO
    QSHELL=.TRUE.
    !
    !     print statistics
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,850) NSHL,SHTHK(1),NSHSLU,NSHSLV
850    FORMAT(/' SHLINI> Shell setup summary:', &
            /'         Number of shells:             ',I2, &
            /'         Shell thickness(1):        ',F10.4,' A', &
            /'         Number of solute atoms:  ',I7, &
            /'         Number of solvent atoms: ',I7,/)
       IF(QSHATM) THEN
          WRITE(OUTU,'(A)') '         Selecting atoms'
       ELSE
          WRITE(OUTU,'(A)') '         Selecting residues'
       ENDIF
       IF(QSHIMG) THEN
          WRITE(OUTU,'(A)') '         Using images'
       ELSE
          WRITE(OUTU,'(A)') '         Using only primary atoms'
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE SHLINI
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE SHLUPD
    !
    !     updates the shell information
    !     most of this was taken/adapted from images/nbndgcm.src
    !     by Mike Crowley
    !
    !     This is just the outer wrapper to be able to call the routine
    !     without arguments (if other parts just want the shell to be
    !     up to date without having render the code unreadable)
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use exfunc
    !
    !
  use bases_fcm
  use coord
  use image
  use number
  use psf
  use stream
  use timerm
  use memory
  use machutil,only:timrb,timre

    integer,allocatable,dimension(:) :: STSHEL
    INTEGER I,NTIM,NLAST
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE,NTIMA
    INTEGER HPN0U,HPN1U,HPN0V,HPN1V,HPM0U,HPM1U,HPM0V,HPM1V
    INTEGER HPN0M,HPN1M,HPM0M,HPM1M
    real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
    real(chm_real) X0,Y0,Z0,RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI
    real(chm_real) :: MARGIN = pt01
    !
    !
    !     provide for case without images (NATIM=0) or if we don't use them
    IF((.NOT.QSHIMG).OR.(NATIM <= NATOM)) THEN
       NTIM=0
       NLAST=NATOM
    ELSE
       NTIM=NATIM-NATOM
       NLAST=NATIM
    ENDIF
    !
    !---- Initial housekeeping --------------------------------------------
    !
    IF(PRNLEV > 5) WRITE(OUTU,'(A)') ' Using SHELL CUBE search'
    !
    IF (TIMER > 0) CALL TIMRB
    !
    !---- Find bounding box around molecule -------------------------------
    !
    XMIN = X(1)
    XMAX = XMIN
    YMIN = Y(1)
    YMAX = YMIN
    ZMIN = Z(1)
    ZMAX = ZMIN
    DO I = 2, NLAST
       XMIN = MIN(XMIN,X(I))
       YMIN = MIN(YMIN,Y(I))
       ZMIN = MIN(ZMIN,Z(I))
       XMAX = MAX(XMAX,X(I))
       YMAX = MAX(YMAX,Y(I))
       ZMAX = MAX(ZMAX,Z(I))
    END DO
    XD = XMAX - XMIN
    YD = YMAX - YMIN
    ZD = ZMAX - ZMIN
    !
    !---- Establish cube parameters ---------------------------------------
    !     one shell
    RHI = SHTHK(1)
    RHIINV = ONE/SHTHK(1)
    !     last shell
    !tr      RHIMX = SHTHK(1) * NSHL
    RHIMX = SHTHK(NSHL)
    RHIMXI = ONE/RHIMX
    !     cube-lenght
    !tr      RHIM = SHTHK(1) * (NSHL+1)
    !tr                 maybe + 1
    RHIM = SHTHK(NSHL)
    RHIMIN = ONE / RHIM
    !
    NCUBX = INT((XD/RHIM) + MARGIN + 1)
    NCUBY = INT((YD/RHIM) + MARGIN + 1)
    NCUBZ = INT((ZD/RHIM) + MARGIN + 1)
    NCUBE = NCUBX*NCUBY*NCUBZ
    !
    X0 = XMIN - 0.5 * (NCUBX * RHIM - XD)
    Y0 = YMIN - 0.5 * (NCUBY * RHIM - YD)
    Z0 = ZMIN - 0.5 * (NCUBZ * RHIM - ZD)
    !
    !     array to store which shell an atom is in (primary + images)
    !     we sort it out after calling SHLXDI
    call chmalloc('shell.src','SHLUPD','STSHEL',NATom+ntIM,intg=STSHEL)
    !
    !---- Call SHLUP2 -------------------------------------------------------
    !
    !     SHLUP2: the 2nd part of Shell-Update: needs to have all nice
    !             things we have handy to work with it
    !             -> go one level down
    !
    CALL SHLUP2 (RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI, &
         X0,Y0,Z0,NCUBX,NCUBY,NCUBZ,NCUBE, &
         NATOM,Nlast, &
         STSHEL, &
         BIMAG%IMATTR)
    !
    !
    !---- Clean up after building of atom list -----------------------------
    !
    call chmdealloc('shell.src','SHLUPD','STSHEL',NATom+ntIM,intg=STSHEL)
    !     
    IF (TIMER >= 1) THEN
       IF (PRNLEV >= 2) WRITE(OUTU,*) 'Total time in SHLUPD: '
       CALL TIMRE
       CALL TIMRB
    END IF                    ! (TIMER == 1)
    !
    !     we seem to be up to date...
    QSHLUP=.TRUE.
    !
    RETURN
  END SUBROUTINE SHLUPD
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE SHLUP2 (RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI, &
       X0,Y0,Z0,NCUBX,NCUBY,NCUBZ,NCUBE, &
       NAT,NATIM, &
       SHELL, &
       IMATTR)
    !
    !     updates the shell information
    !     2nd part of the subroutine does admin work which needs the
    !     arrays handy
    !     most of this was taken/adapted from images/nbndgcm.src
    !     by Mike Crowley
    !
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use exfunc
    !
  use coord
  use psf
  use stream
  use memory

    real(chm_real) RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI,X0,Y0,Z0
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE
    INTEGER NAT,NATIM
    INTEGER SHELL(natim)
    INTEGER IMATTR(*)
    !
    INTEGER I,J,K,L,N,NTIM,NTIMA,TMP
    integer,allocatable,dimension(:) :: STSH1,STSH2,STSH3

    integer,allocatable,dimension(:),target :: space_n0u,space_m0u, &
         space_n0mu,space_m0mu
    INTEGER,pointer,dimension(:) :: HPN0U,HPN1U,HPN0V,HPN1V, &
         HPM0U,HPM1U,HPM2U,HPM0V,HPM1V,HPM2V, &
         HPN0MU,HPN1MU,HPN0MV,HPN1MV, &
         HPM0MU,HPM1MU,HPM2MU,HPM0MV,HPM1MV,HPM2MV
    !
    !     compute number of images
    NTIM=NATIM-NAT
    IF(NTIM < 0) NTIM=0
    !
    !     Initialize SHELL array which should finally hold, which
    !     shell which atom is in...
    DO I=1,NAT+NTIM
       SHELL(I)=0
    ENDDO
    !
    !---- Allocate work areas for XDIST -----------------------------------
    !
    !     we need 2/3 lists to juggle for: primary/images
    !                                      atomnumbers/cubenumbers
    !                                      solute/solvent
    !
    !     ........ primary ........
    call chmalloc('shell.src','SHLUP2','space_N0U',4*NATOM,intg=space_N0U)
    HPN0U   => space_n0u(0*natom+1 : 1*natom)
    HPN1U   => space_n0u(1*natom+1 : 2*natom)
    HPN0V   => space_n0u(2*natom+1 : 3*natom)
    HPN1V   => space_n0u(3*natom+1 : 4*natom)

    call chmalloc('shell.src','SHLUP2','space_M0U',6*NCUBE,intg=space_M0U)
    HPM0U   => space_m0u(0*ncube+1 : 1*ncube)
    HPM1U   => space_m0u(1*ncube+1 : 2*ncube)
    HPM2U   => space_m0u(2*ncube+1 : 3*ncube)
    HPM0V   => space_m0u(3*ncube+1 : 4*ncube)
    HPM1V   => space_m0u(4*ncube+1 : 5*ncube)
    HPM2V   => space_m0u(5*ncube+1 : 6*ncube)

    !     ........ images ........
    NTIMA=NTIM+1
    call chmalloc('shell.src','SHLUP2','space_N0MU',4*NTIMA,intg=space_N0MU)
    HPN0MU  => space_n0mu(0*ntima+1 : 1*ntima)
    HPN1MU  => space_n0mu(1*ntima+1 : 2*ntima)
    HPN0MV  => space_n0mu(2*ntima+1 : 3*ntima)
    HPN1MV  => space_n0mu(3*ntima+1 : 4*ntima)

    call chmalloc('shell.src','SHLUP2','space_M0MU',6*NCUBE,intg=space_M0MU)
    HPM0MU  => space_m0mu(0*ncube+1 : 1*ncube)
    HPM1MU  => space_m0mu(1*ncube+1 : 2*ncube)
    HPM2MU  => space_m0mu(2*ncube+1 : 3*ncube)
    HPM0MV  => space_m0mu(3*ncube+1 : 4*ncube)
    HPM1MV  => space_m0mu(4*ncube+1 : 5*ncube)
    HPM2MV  => space_m0mu(5*ncube+1 : 6*ncube)
    !
    !     work arrays for SHLXDI
    call chmalloc('shell.src','SHLUP2','STSH1',NTIMA,intg=STSH1)
    call chmalloc('shell.src','SHLUP2','STSH2',NTIMA,intg=STSH2)
    call chmalloc('shell.src','SHLUP2','STSH3',NATOM,intg=STSH3)
    !     
    !
    !---- Call SHLXDI -----------------------------------------------------
    !
    !     SHLXDI solves the basic distance-search problem
    !
    CALL SHLXDI (NATOM,NATIM,X,Y,Z, &
         RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI, &
         X0,Y0,Z0,NCUBX,NCUBY,NCUBZ,NCUBE,NSHL, &
         SHTHK2, &
         IMATTR,SOLUL,SOLVL,SHELL,NSHSLU,NSHSLV, &
         HPM0U,HPM1U,HPM2U, &
         HPN0U,HPN1U, &
         HPM0V,HPM1V,HPM2V, &
         HPN0V,HPN1V, &
         HPM0MU,HPM1MU,HPM2MU, &
         HPM0MV,HPM1MV,HPM2MV, &
         HPN0MU,HPN1MU, &
         HPN0MV,HPN1MV, &
         STSH1,STSH2,STSH3)
    !
    !
    !---- Clean up after building the shell-list --------------------------
    !
    call chmdealloc('shell.src','SHLUP2','STSH1',NTIMA,intg=STSH1)
    call chmdealloc('shell.src','SHLUP2','STSH2',NTIMA,intg=STSH2)
    call chmdealloc('shell.src','SHLUP2','STSH3',NATOM,intg=STSH3)
    !
    call chmdealloc('shell.src','SHLUP2','space_M0MU',6*NCUBE,intg=space_M0MU)
    call chmdealloc('shell.src','SHLUP2','space_N0MU',4*NTIMA,intg=space_N0MU)
    call chmdealloc('shell.src','SHLUP2','space_M0U',6*NCUBE,intg=space_M0U)
    call chmdealloc('shell.src','SHLUP2','space_N0U',4*NATOM,intg=space_N0U)
    !
    !---- update the shell information from the SHELL array  --------------
    !
    !     loop over all solvent atoms. if SHELL(solvent-atom)=0
    !     this means that it was in a cube too far off any solute-atom
    !     to be processed -> set shell to nshl+1 (i.e. bulk).
    !     afterwards all atoms with a SHELL of 0 are non-shell atoms
    J=NSHL+1
    DO I=1,NSHSLV
       N=SOLVL(I)
       IF(SHELL(N) <= 0) SHELL(N)=J
    ENDDO
    !
    !     loop over all images: find the primary atom in IMATTR and if it
    !     is in NSHL+1 this means, that it is a SOLVENT atom and still not
    !     assigned to a shell -> we can use the shell the image is in
    DO I=NATOM+1,NATIM
       N=IMATTR(I)
       IF((SHELL(N) == J).AND.(SHELL(I) > 0)) &
            SHELL(N)=SHELL(I)
    ENDDO
    !
    !     introduce BYRES like behaviour: SOLVENT atom in a residue
    !               that is in the LOWEST shell decides in which shell
    !               the whole residue is
    !               BEWARE: assumes that all atoms in a residue are in a
    !                       contigious sequence
    !                       IBASE(IRES)+1 ... IBASE(IRES+1)
    !     this loops are ugly & nested and could be improved...
    IF(.NOT.QSHATM) THEN
       DO I=1,NRES
          J=IBASE(I)+1
          IF(I < NRES) THEN
             K=IBASE(I+1)
          ELSE
             K=NATOM
          ENDIF
          !
          !           find lowest shell an atom of RES is in
          TMP=0
          DO L=J,K
             IF(SHELL(L) > 0) THEN
                IF(TMP > 0) THEN
                   TMP=MIN(SHELL(L),TMP)
                ELSE
                   TMP=SHELL(L)
                ENDIF
             ENDIF
          ENDDO
          !           and put all atoms into that shell
          IF(TMP > 0) THEN
             DO L=J,K
                SHELL(L)=TMP
             ENDDO
          ENDIF
       ENDDO
    ENDIF
    !
    !     now fill the SHLLST array
    DO I=1,NSHL
       NASHL(I)=0
    ENDDO
    NSHBLK=0
    !
    DO I=1,NATOM
       IF(SHELL(I) > NSHL) THEN
          !           i.e. atom is in a shell beyond the last shell
          NSHBLK=NSHBLK+1
          SHBLKL(NSHBLK)=I
       ELSE IF (SHELL(I) > 0) THEN
          !           i.e. atom is in a shell selected
          NASHL(SHELL(I))=NASHL(SHELL(I))+1
          SHLLST(NASHL(SHELL(I)),SHELL(I))=I
       ENDIF
       !        if atom was not processed here it is not in any shell...
    ENDDO
    !
    !     all admin is done -> return to 1st part of SHLUPD
    !
    RETURN
  END SUBROUTINE SHLUP2
  !
  !----------------------------------------------------------------------
  !

  !======================================================================
  ! Main subroutine
  !======================================================================
  !
  SUBROUTINE SHLXDI(NAT,NATIM,X,Y,Z, &
       RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI, &
       X0,Y0,Z0,NCUBX,NCUBY,NCUBZ,NCUBE,NSHL, &
       SHTHK2, &
       IMATTR,SHSLU,SHSLV,SHELL,NSHSLU,NSHSLV, &
       LSTM0U,LSTM1U,LSTM2U,LSTN0U,LSTN1U, &
       LSTM0V,LSTM1V,LSTM2V,LSTN0V,LSTN1V, &
       LSTM0MU,LSTM1MU,LSTM2MU, &
       LSTM0MV,LSTM1MV,LSTM2MV, &
       LSTN0MU,LSTN1MU, &
       LSTN0MV,LSTN1MV, &
       IMSLVL,IMSLUL,SOLFLG)
    !
    !     updates the shell information, this is the work subroutine
    !     most of this was taken/adapted from images/nbndgcm.src
    !     by Mike Crowley
    !
    !     possible speedups: don't disentangle the intertwined linked lists
    !                        since we don't need to sort the lists
    !                        -> work with them directly
    !                       &: work with integer-arithmetics in decisions
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !

  use new_timer,only:timer_start,timer_stop,T_setgrd,t_bldlst 

  use exfunc
  use stream

    INTEGER NAT,NATIM
    real(chm_real)  X(*),Y(*),Z(*),RHI,RHIINV,RHIM,RHIMIN,RHIMX,RHIMXI
    real(chm_real)  X0,Y0,Z0
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE,NSHL
    real(chm_real) SHTHK2(*)
    INTEGER IMATTR(*),SHSLU(*),SHSLV(*),SHELL(*)
    INTEGER NSHSLU,NSHSLV
    INTEGER LSTM0U(*),LSTM1U(*),LSTM2U(*),LSTN0U(*),LSTN1U(*)
    INTEGER LSTM0V(*),LSTM1V(*),LSTM2V(*),LSTN0V(*),LSTN1V(*)
    INTEGER LSTM0MU(*),LSTM1MU(*),LSTM2MU(*)
    INTEGER LSTM0MV(*),LSTM1MV(*),LSTM2MV(*)
    INTEGER LSTN0MU(*),LSTN1MU(*)
    INTEGER LSTN0MV(*),LSTN1MV(*)
    INTEGER IMSLVL(*),IMSLUL(*),SOLFLG(*)
    !
    INTEGER I,J,K,N,N2,M,M2,IMU,IMV,TMP,NLO,NHI,NLO2,NHI2
    INTEGER NIX,NIX2,NBRN,NBRNIM,NIMSLV,NIMSLU
    INTEGER NCUBXY,NTIM,MX,MY,MZ,NBR27,MADD(27)
    INTEGER NSMPM,NSMPPR,NSMPPP
    real(chm_real)  RHIMX2,XX,YY,ZZ,DDX,DDY,DDZ,XTMP
    LOGICAL DOIMAGES
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    call timer_start(T_setgrd)
    !----------------------------------------------------------------------
    !
    !     how many images are there?
    NTIM=NATIM-NAT
    IF(NTIM < 0) NTIM=0
    !
    DOIMAGES=QSHIMG
    IF(DOIMAGES.AND.(NTIM == 0)) THEN
       CALL WRNDIE(1,'<SHLDEF>' &
            ,'I was told to use IMAGES but none are present.')
       DOIMAGES=.FALSE.
    ENDIF
    !
    !---- Compute frequently used intermediate scalar results -------------
    !
    RHIMX2 = RHIMX * RHIMX
    NCUBXY = NCUBX * NCUBY
    !
    !----- do some output if desired --------------------------------------
    !
    IF (PRNLEV >= 6) THEN
       WRITE (OUTU,'(A)') &
            ' SHLXDI Building particle interaction list using grid'
       WRITE (OUTU,800) '  Number of primary particles    =', NAT
       WRITE (OUTU,800) '  Number of image particles      =', NTIM
       WRITE (OUTU,800) '  Number of cells in X dimension =', ncubx
       WRITE (OUTU,800) '  Number of cells in Y dimension =', ncuby
       WRITE (OUTU,800) '  Number of cells in Z dimension =', ncubz
       WRITE (OUTU,800) '  Number of cells, total         =', ncube
       WRITE (OUTU,801) '  Cell size                      =', RHIM
800    FORMAT(A,I7)
801    FORMAT(A,F7.3)
    END IF
    !
    !---- Prepare MADD array-----------------------------------------------
    !
    !     The MADD array is defined so that for a given cube M,
    !     the 27-cube neighborhood consists of the cubes M+MADD(1..27).
    !
    DO MX = -1, 1
       DO MY = -1, 1
          DO MZ = -1, 1
             NBR27 = (MX+1)*9 + (MY+1)*3 + (MZ+1) + 1
             MADD(NBR27) = MX + NCUBX * (MY + NCUBY * MZ)
          ENDDO
       ENDDO
    ENDDO
    !
    !     Remove duplicate offsets from MADD, or we will get duplicate
    !     atoms and therefore array-bound problems down the line.  Any
    !     element that is set to NCUBE is effectively deleted since later
    !     M+NCUBE is out of the range 1..NCUBE.
    !
    !     Crude method OK since max number of operations is 13 * 27 = 351
    !
    !
    DO NBR27 = 1, 27
       DO TMP = 1, NBR27-1
          IF (MADD(TMP) == MADD(NBR27)) MADD(NBR27) = NCUBE + 1
       END DO
    END DO
    !
    !---- Make lists of image solute/solvent atoms ------------------------
    !
    NIMSLV=0
    NIMSLU=0
    IF(DOIMAGES) THEN
       !        solute:
       !        first make a SELE like flag array for quick lookup if an atom
       !        is a solvent atom
       DO I=1,NAT
          SOLFLG(I)=0
       ENDDO
       DO I=1,NSHSLU
          SOLFLG(SHSLU(I))=1
       ENDDO
       !        now look up all images
       DO I=NAT+1,NATIM
          J=IMATTR(I)
          IF(SOLFLG(J) == 1) THEN
             NIMSLU=NIMSLU+1
             IMSLUL(NIMSLU)=I
          ENDIF
       ENDDO
       !
       !        solvent (the same as above)
       DO I=1,NAT
          SOLFLG(I)=0
       ENDDO
       DO I=1,NSHSLV
          SOLFLG(SHSLV(I))=1
       ENDDO
       DO I=NAT+1,NATIM
          J=IMATTR(I)
          IF(SOLFLG(J) == 1) THEN
             NIMSLV=NIMSLV+1
             IMSLVL(NIMSLV)=I
          ENDIF
       ENDDO
    ENDIF
    !
    !---- Prepare a list of particles for each cube -----------------------
    !
    !     First, temporarily set up LSTN1 such that for a given atom N,
    !     LSTN1(N) is the index of the cube it's in.  This will be
    !     discarded after the next step.
    !
    !     We need to do this for solute and solvent separately so that
    !     we then can pick only cubes containing solute atoms and put
    !     solvent molecules from this + surounding cubes into the right
    !     shell
    !
    !     solute
    DO N = 1, NSHSLU
       K=SHSLU(N)
       LSTN1U(N) = ( INT((Z(K)-Z0)*RHIMIN)  * NCUBY &
            + INT((Y(K)-Y0)*RHIMIN)) * NCUBX &
            + INT((X(K)-X0)*RHIMIN) + 1
    ENDDO
    !     solvent
    DO N = 1, NSHSLV
       K=SHSLV(N)
       LSTN1V(N) = ( INT((Z(K)-Z0)*RHIMIN)  * NCUBY &
            + INT((Y(K)-Y0)*RHIMIN)) * NCUBX &
            + INT((X(K)-X0)*RHIMIN) + 1
    ENDDO
    !     images:
    IF(DOIMAGES) THEN
       !        solute
       DO N = 1, NIMSLU
          K=IMSLUL(N)
          LSTN1MU(N) = ( INT((Z(K)-Z0)*RHIMIN)  * NCUBY &
               + INT((Y(K)-Y0)*RHIMIN)) * NCUBX &
               + INT((X(K)-X0)*RHIMIN) + 1
          k=imattr(k)
       ENDDO
       !        solvent
       DO N = 1, NIMSLV
          K=IMSLVL(N)
          LSTN1MV(N) = ( INT((Z(K)-Z0)*RHIMIN)  * NCUBY &
               + INT((Y(K)-Y0)*RHIMIN)) * NCUBX &
               + INT((X(K)-X0)*RHIMIN) + 1
       ENDDO
    ENDIF
    !
    !     Invert LSTN1:  Set up LSTM0/LSTN0 as intertwined linked lists
    !     such that for a given cube M, you can recursively read off the
    !     atoms that it contains.  This will be discarded after the next
    !     step.
    !
    !     Vectorizable.
    DO M = 1, NCUBE
       LSTM0U(M) = 0
       LSTM0V(M) = 0
       LSTM0MU(M) = 0
       LSTM0MV(M) = 0
    END DO
    !
    !     solute
    DO N = 1, NSHSLU
       M = LSTN1U(N)
       LSTN0U(N) = LSTM0U(M)
       LSTM0U(M) = N
    ENDDO
    !     solvent
    DO N = 1, NSHSLV
       M = LSTN1V(N)
       LSTN0V(N) = LSTM0V(M)
       LSTM0V(M) = N
    ENDDO
    !     images
    IF(DOIMAGES) THEN
       !        solute
       DO N =  1, NIMSLU
          M = LSTN1MU(N)
          LSTN0MU(N) = LSTM0MU(M)
          LSTM0MU(M) = N
       END DO
       !        solvent
       DO N =  1, NIMSLV
          M = LSTN1MV(N)
          LSTN0MV(N) = LSTM0MV(M)
          LSTM0MV(M) = N
       END DO
    ENDIF
    !
    !     we set up the LSTN1's as lists of ATOMNUMBERS of the solute/solvent
    !     primary/image atoms in the given cube M
    !     we need begin and end pointers for the list since we don't use _all_
    !     atoms but only those in solute/solvent. so some cubes may not contain
    !     _any_ atoms of interest. thus LISTM1U/V(M-1) may sometimes be = 0 which
    !     is _not_ the beginning of atoms in cube M.
    !     so: LSTM1U/V/M...(M)  is the start pointer and
    !         LSTM2U/V/M...(M)  is the last element in the list
    !
    !     BEWARE: up to now the numbers in the LST.N.. lists were only
    !             pointers to the solute/solvent/... lists
    !             -> convert to real atom numbers now !
    !
    I  = 0
    J  = 0
    IMU = 0
    IMV = 0
    DO M = 1, NCUBE
       !
       !        .... primary .....
       !        solute
       N = LSTM0U(M)
       LSTM1U(M) = 0
       LSTM2U(M) = 0
       DO WHILE (N /= 0)
          I = I + 1
          !           set begin pointer if this is the first element
          IF(LSTM1U(M) <= 0) LSTM1U(M)=I
          LSTN1U(I) = SHSLU(N)
          N = LSTN0U(N)
       ENDDO
       IF(LSTM1U(M) > 0) LSTM2U(M) = I
       !        solvent
       N = LSTM0V(M)
       LSTM1V(M) = 0
       LSTM2V(M) = 0
       DO WHILE (N /= 0)
          J = J + 1
          !           set begin pointer if this is the first element
          IF(LSTM1V(M) <= 0) LSTM1V(M)=J
          LSTN1V(J) = SHSLV(N)
          N = LSTN0V(N)
       ENDDO
       IF(LSTM1V(M) > 0) LSTM2V(M) = J
       !
       !        .... images .....
       IF(DOIMAGES) THEN
          !           .... solute ....
          N = LSTM0MU(M)
          LSTM1MU(M) = 0
          LSTM2MU(M) = 0
          DO WHILE (N /= 0)
             IMU = IMU + 1
             !              set begin pointer if this is the first element
             IF(LSTM1MU(M) <= 0) LSTM1MU(M)=IMU
             LSTN1MU(IMU) = IMSLUL(N)
             N = LSTN0MU(N)
          ENDDO
          IF(LSTM1MU(M) > 0) LSTM2MU(M) = IMU
          !           .... solvent ....
          N = LSTM0MV(M)
          LSTM1MV(M) = 0
          LSTM2MV(M) = 0
          DO WHILE (N /= 0)
             IMV = IMV + 1
             !              set begin pointer if this is the first element
             IF(LSTM1MV(M) <= 0) LSTM1MV(M)=IMV
             LSTN1MV(IMV) = IMSLVL(N)
             N = LSTN0MV(N)
          ENDDO
          IF(LSTM1MV(M) > 0) LSTM2MV(M) = IMV
       ENDIF
       !
    ENDDO
    !
    !---- AND FINALLY: check all pairs ------------------------------------
    !
    !     Now we loop over all cubes M.
    !     If M contains PRIMARY SOLUTE atoms -> do this cube.
    !     set up two lists of PRIMARY/IMAGE SOLVENT atoms in the
    !     surrounding 27 cubes. then check all pairs with PRIMARY
    !     SOLVENT.
    !     If we are doing images -> check paris with IMAGE SOLVENT
    !     and if the cube contains IMAGE SOLUTE check with both
    !     types of SOLVENT
    !
    !
    !---- Start of major loop over cubes M --------------------------------
    !
    call timer_stop(T_setgrd)
    call timer_start(T_bldlst)
    !
    !---- Init of counters ------------------------------------------------
    !
    NSMPM=0
    NSMPPR=0
    NSMPPP=0
    !
    DO M = 1,NCUBE
       !
       !------- Determine range of atoms LSTN1(NLO..NHI) in center cube M ----
       !
       !        only do this cube if it contains PRIMARY solute atoms
       IF(LSTM1U(M) > 0) THEN
          !
          NSMPM=NSMPM+1
          !
          NLO=LSTM1U(M)
          NHI = LSTM2U(M)
          !
          !---------- Set LSTN0 equal to an array of neighbor atoms -------------
          !
          !           LSTN0 will be an array of the indices of atoms which are in
          !           the cube M and its neighbors.  In the outer loop, M2 takes on
          !           up to 27 indices of cubes adjacent to M.  For each M2,
          !           LSTN1(NLO2..NHI2) is the set of atoms in M2.  We traverse
          !           that list backwards so we end up with little runs of
          !           ascending order.
          !   
          !           Only build the list of solvent molecules since we only
          !           compute solute-solvent interactions
          !   
          NBRN = 0
          NBRNIM = 0
          DO NBR27 = 1, 27
             !
             !              Propose an M2, and if it's in range, then...
             M2 = M + MADD(NBR27)
             IF (M2 >= 1 .AND. M2 <= NCUBE) THEN ! ----------
                !
                !                 ... PRIMARY ...
                !                 Set NLO2..NHI2 equal to range of indices into LSTN1
                NLO2 = LSTM1V(M2)
                NHI2 = LSTM2V(M2)
                !
                !                 Loop over those indices, filling LSTN0 with atom numbers
                IF(NLO2 > 0) THEN
                   DO NIX2 = NHI2, NLO2, -1
                      NBRN = NBRN + 1
                      LSTN0V(NBRN) = LSTN1V(NIX2)
                   END DO
                ENDIF
                !
                !                 ... IMAGES ...
                IF(DOIMAGES) THEN
                   NLO2 = LSTM1MV(M2)
                   NHI2 = LSTM2MV(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRNIM = NBRNIM + 1
                         LSTN0MV(NBRNIM) = LSTN1MV(NIX2)
                      END DO
                   ENDIF
                ENDIF
                !
             END IF           ! ----------
          END DO
          !
          !
          !
          !---------- For every PRIMARY solute atom in M test all prim/img solvent
          !
          !           GO: loop over all PRIMARY SOLUTE atoms in the center cube
          !
          DO NIX = NLO, NHI
             !
             !              Record often-used parameters for solute atom N
             N = LSTN1U(NIX)
             XX = X(N)
             YY = Y(N)
             ZZ = Z(N)
             !
             !              Start this run of N2's (i.e. SOLVENT atoms)
             !------------- primary solvent ----------------------------------------
             DO I = 1, NBRN
                !
                NSMPPR=NSMPPR+1
                !
                N2 = LSTN0V(I)
                DDX = X(N2) - XX
                DDY = Y(N2) - YY
                DDZ = Z(N2) - ZZ
                !                 distance solute atom <-> solvent atom
                XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                !                 the shell this gets the solvent atom into
                IF(XTMP <= RHIMX2) THEN
                   NSMPPP=NSMPPP+1
                   !                     TMP=INT(SQRT(XTMP)/RHI)+1
                   TMP=SHLDEC(XTMP,SHTHK2,NSHL)
                ELSE
                   TMP=NSHL+1
                ENDIF
                IF((SHELL(N2) <= 0).OR.(TMP < SHELL(N2))) &
                     SHELL(N2)=TMP
             END DO
             !------------- End primary solvent ------------------------------------
             !------------- image solvent ------------------------------------------
             IF(DOIMAGES)THEN
                DO I = 1, NBRNIM
                   !
                   NSMPPR=NSMPPR+1
                   !
                   N2 = LSTN0MV(I)
                   DDX = X(N2) - XX
                   DDY = Y(N2) - YY
                   DDZ = Z(N2) - ZZ
                   XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                   IF(XTMP <= RHIMX2) THEN
                      NSMPPP=NSMPPP+1
                      !                        TMP=INT(SQRT(XTMP)/RHI)+1
                      TMP=SHLDEC(XTMP,SHTHK2,NSHL)
                   ELSE
                      TMP=NSHL+1
                   ENDIF
                   !
                   IF((SHELL(N2) <= 0).OR.(TMP < SHELL(N2))) &
                        SHELL(N2)=TMP
                END DO
                !              end if(doimages)
             ENDIF
             !------------- End image solvent --------------------------------------
             !
             !           end loop over PRIMARY solute atoms (NIX)
          ENDDO
          !        end if(lstm1u(m) > 0) <- check if central cube contains solute
       ENDIF
       !
       !------- loop over all IMAGE solute (if wanted) -----------------------
       IF(DOIMAGES) THEN
          IF(LSTM1MU(M) > 0) THEN
             NSMPM=NSMPM+1
             !
             NLO=LSTM1MU(M)
             NHI=LSTM2MU(M)
             !
             !------------- Set LSTN0 equal to an array of neighbor atoms ----------
             !
             NBRN = 0
             NBRNIM = 0
             DO NBR27 = 1, 27
                M2 = M + MADD(NBR27)
                IF (M2 >= 1 .AND. M2 <= NCUBE) THEN ! ----------
                   !                    ... PRIMARY ...
                   NLO2 = LSTM1V(M2)
                   NHI2 = LSTM2V(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRN = NBRN + 1
                         LSTN0V(NBRN) = LSTN1V(NIX2)
                      END DO
                   ENDIF
                   !                    ... IMAGES ...
                   NLO2 = LSTM1MV(M2)
                   NHI2 = LSTM2MV(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRNIM = NBRNIM + 1
                         LSTN0MV(NBRNIM) = LSTN1MV(NIX2)
                      END DO
                   ENDIF
                END IF                            ! ----------
             END DO
             !
             !------------- For every IMAGE solute atom in M test all prim/img solvent
             !
             DO NIX = NLO, NHI
                N = LSTN1MU(NIX)
                XX = X(N)
                YY = Y(N)
                ZZ = Z(N)
                !
                !---------------- primary solvent -------------------------------------
                DO I = 1, NBRN
                   !
                   NSMPPR=NSMPPR+1
                   N2 = LSTN0V(I)
                   DDX = X(N2) - XX
                   DDY = Y(N2) - YY
                   DDZ = Z(N2) - ZZ
                   XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                   IF(XTMP <= RHIMX2) THEN
                      NSMPPP=NSMPPP+1
                      !                        TMP=INT(SQRT(XTMP)/RHI)+1
                      TMP=SHLDEC(XTMP,SHTHK2,NSHL)
                   ELSE
                      TMP=NSHL+1
                   ENDIF
                   !
                   IF((SHELL(N2) <= 0).OR.(TMP < SHELL(N2))) &
                        SHELL(N2)=TMP
                END DO
                !---------------- End primary solvent ---------------------------------
                !---------------- image solvent ---------------------------------------
                DO I = 1, NBRNIM
                   !
                   NSMPPR=NSMPPR+1
                   N2 = LSTN0MV(I)
                   DDX = X(N2) - XX
                   DDY = Y(N2) - YY
                   DDZ = Z(N2) - ZZ
                   XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                   IF(XTMP <= RHIMX2) THEN
                      NSMPPP=NSMPPP+1
                      !                        TMP=INT(SQRT(XTMP)/RHI)+1
                      TMP=SHLDEC(XTMP,SHTHK2,NSHL)
                   ELSE
                      TMP=NSHL+1
                   ENDIF
                   !
                   IF((SHELL(N2) <= 0).OR.(TMP < SHELL(N2))) &
                        SHELL(N2)=TMP
                END DO
                !---------------- End image solvent -----------------------------------
                !              end loop over IMAGE solute
             ENDDO
             !           end if(LSTM1MU(M) > 0) <- check if this cube has IMG solute
          ENDIF
          !        end if(doimages)
       ENDIF
       !
       !     end loop over all cubes (M)
    ENDDO
    !     
    !---- End of major loop over cubes M ----------------------------------
    !
    !---- do some output if desired ---------------------------------------
    !     
    !
    IF (PRNLEV >= 6) THEN
       WRITE (OUTU,802) 
       WRITE (OUTU,803) '  Number of cubes sampled        =', NSMPM
       WRITE (OUTU,803) '  Number of pairs evaluated      =', NSMPPR
       WRITE (OUTU,803) '  Number of atoms in a shell     =', NSMPPP
       WRITE (OUTU,803) '  Number of pairs for brute-force=' &
            , (NSHSLU*NSHSLV)
802    FORMAT(/' SHLXDI Statistics:')
803    FORMAT(A,I14)
    END IF
    !
    !
    CALL TIMER_STOP(T_BLDLST)
    !
    !---- we are done... --------------------------------------------------
    !     
    RETURN
  END SUBROUTINE SHLXDI
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE SHLPRN(NAT)
    !
    !     print shell data/statistics
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use exfunc
  use stream

    INTEGER NAT
    INTEGER I,J
    !
    if(prnlev >= 2) then
       WRITE(OUTU,'(A)') ' SHLPRN> SHELL statistics requested'
       IF (.NOT.QSHLUP) WRITE(OUTU,800)
       WRITE(OUTU,801) NSHL,SHTHK(1),NSHSLU,NSHSLV
       !     
       DO I=1,NSHL
          WRITE(OUTU,809) I,SHTHK(I)
       ENDDO
       WRITE(OUTU,'(A)')
       !     
       IF(QSHATM) THEN
          WRITE(OUTU,'(A)') '         Selecting atoms'
       ELSE
          WRITE(OUTU,'(A)') '         Selecting residues'
       ENDIF
       IF(QSHIMG) THEN
          WRITE(OUTU,'(A)') '         Using images'
       ELSE
          WRITE(OUTU,'(A)') '         Using only primary atoms'
       ENDIF
       !     
       WRITE(OUTU,'(A)') ''
       DO I=1,NSHL
          WRITE(OUTU,802) I,NASHL(I)
       ENDDO
       WRITE(OUTU,803) NSHBLK
    endif
    !
800 FORMAT(/' *** WARNING *** SHELL is not up to date!'/)
801 FORMAT(/'         Number of shells defined ',I7, &
         /'         Shell thickness (1)        ',F10.4, &
         /'         Solute atoms             ',I7, &
         /'         Solvent atoms            ',I7,/'')
809 FORMAT( '         Shell ',I7,' extends to ',F10.4,' A')
    !
802 FORMAT( '         Number of atoms in shell ',I7,' : ',I7)
803 FORMAT(/'         Number of atoms in bulk          : ',I7)
    !
    IF (PRNLEV >= 7) THEN
       WRITE (OUTU,804) NSHSLU
       WRITE (OUTU,808) (SOLUL(J),J=1,NSHSLU)
804    FORMAT(/'         Atom numbers for solute  (',I7,' atoms)')
       WRITE (OUTU,805) NSHSLV
       WRITE (OUTU,808) (SOLVL(J),J=1,NSHSLV)
805    FORMAT(/'         Atom numbers for solvent (',I7,' atoms)')
    ENDIF
    !
    IF (PRNLEV >= 6) THEN
       DO I=1,NSHL
          WRITE (OUTU,806) I,NASHL(I)
          WRITE (OUTU,808) (SHLLSt(J,I),J=1,NASHL(I))
806       FORMAT(/'         Atom numbers for shell:  ',I7,'(' &
               ,I7,' atoms)')
       ENDDO
    ENDIF
    !
    IF (PRNLEV >= 7) THEN
       WRITE (OUTU,807) NSHBLK
       WRITE (OUTU,808) (SHBLKL(J),J=1,NSHBLK)
807    FORMAT(/'         Atom numbers for bulk    (',I7,' atoms)')
    END IF
    !
808 FORMAT(10(1X,I7))
    !
    RETURN
  END SUBROUTINE SHLPRN
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE SHLDEF(ST,STLEN)
    !
    !     convert a shell to a selection keyword
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use exfunc
    !
  use selctam
  use coord
  use psf
  use memory
  use string

    integer,pointer,dimension(:) :: IPTR
    CHARACTER(len=*) ST
    INTEGER STLEN
    !
    INTEGER J,NUMP,NSHDEF
    !
    !     find out, what we want to convert...
    NSHDEF=0
    IF(INDXA(ST,STLEN,'SOLU') > 0) THEN
       NSHDEF=-3
    ELSE IF(INDXA(ST,STLEN,'SOLV') > 0) THEN
       NSHDEF=-2
    ELSE IF(INDXA(ST,STLEN,'BULK') > 0) THEN
       NSHDEF=-1
    ELSE
       NSHDEF=GTRMI(ST,STLEN,'SHEL',NSHDEF)
    ENDIF
    !
    IF(NSHDEF == 0) THEN
       CALL WRNDIE(0,'<SHLDEF>' &
            ,'SOLUte or SOLVent or BULK or SHELl N must be present.')
       RETURN
    ENDIF
    !
    !     now process
    NUMP=NUMSKY+1
    call chmalloc('shell.src','SHLDEF','IPTR',NATOM,intgp=IPTR)

    CALL TRIME(ST,STLEN)
    IF(STLEN <= 0) GOTO 900
    CALL NEXTWD(ST,STLEN,NAMSKY(NUMP),MNAMSK,LNAMSK(NUMP))
    CALL TRIME(ST,STLEN)
    IF(STLEN > 0) GOTO 900
    !
    CALL SHLFLG(NSHDEF,NATOM,NSHL,NSHSLU,NSHSLV,NSHBLK,NASHL,MAXSHL, &
         solul,solvl,SHLLST, &
         shblkl,IPTR)
    !
    DO J=1,NUMSKY
       IF(EQST(NAMSKY(NUMP),LNAMSK(NUMP),NAMSKY(J),LNAMSK(J))) THEN
          !           i.e. this definition name already exists -> change & exit
          call chmdealloc('shell.src','SHLDEF','PTRSKY(J)%a', &
               LENSKY(J),intgp=PTRSKY(J)%a)
          PTRSKY(J)%a => IPTR
          LENSKY(J)=NATOM
          RETURN
       ENDIF
    ENDDO
    !
    IF(NUMP >= MAXSKY) THEN
       CALL WRNDIE(0,'<SHLDEF>','Overflow in number of definitions.')
       RETURN
    ENDIF
    !
    NUMSKY=NUMP
    LENSKY(NUMP)=NATOM
    PTRSKY(NUMP)%a => IPTR
    RETURN
    !
    ! to crap-out
900 CONTINUE
    CALL WRNDIE(0,'<SHLDEF>','Syntax error in SEHLl DEFIne command')
    call chmdealloc('shell.src','SHLDEF','IPTR',NATOM,intgp=IPTR)

    RETURN
  END SUBROUTINE SHLDEF
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE SHLFLG(NSHDEF,NAT,NSHL,NSHSLU,NSHSLV,NSHBLK,NASHL, &
       MAXSHL,SOLUL,SOLVL,SHLLST,SHBLKL,IFLAG)
    !
    !     fill IFLAG to represent a SELEct ... END array where all atoms
    !     in shell NSHDEF are selected
    !
    !     conventions:
    !     NSHDEF = -3 : select solute
    !              -2 : select solvent
    !              -1 : select bulk (solvent outside last shell)
    !              >0 : select shell NSHDEF
    !
    !     Tibor Rudas Oct 2002 - Nov 2002
    !
  use exfunc
  use stream

    INTEGER NSHDEF,NAT,NSHL,NSHSLU,NSHSLV,NSHBLK,NASHL(*),MAXSHL
    INTEGER SOLUL(*),SOLVL(*),SHLLST(NAT,NSHL),SHBLKL(*),IFLAG(*)
    !
    !
    INTEGER I
    !
    !     init array
    DO I=1,NAT
       IFLAG(I)=0
    ENDDO
    !
    IF(NSHDEF == -3) THEN
       !        solute
       IF (PRNLEV >= 6) WRITE(OUTU,'(A)') ' SHLFLG converting solute'
       DO I=1,NSHSLU
          IFLAG(SOLUL(I))=1
       ENDDO
    ELSE IF(NSHDEF == -2) THEN
       !        solvent
       IF (PRNLEV >= 6) WRITE(OUTU,'(A)') ' SHLFLG converting solvent'
       DO I=1,NSHSLV
          IFLAG(SOLVL(I))=1
       ENDDO
    ELSE IF(NSHDEF == -1) THEN
       !        bulk
       IF (PRNLEV >= 6) WRITE(OUTU,'(A)') ' SHLFLG converting bulk'
       DO I=1,NSHBLK
          IFLAG(SHBLKL(I))=1
       ENDDO
    ELSE
       !        shell nr NSHDEF
       IF((NSHDEF <= 0).AND.(NSHDEF > MAXSHL)) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,800) NSHDEF,MAXSHL,1
800       FORMAT(' **** WARNING **** from SHLFLG -- selected shell, ' &
               ,I6,' was larger than ',I6 &
               ,'.'/' It will be set to ',I6,'.')
          NSHDEF=1
       ENDIF
       IF (PRNLEV >= 6) WRITE(OUTU,'(A,I7)')  &
            ' SHLFLG converting shell',NSHDEF
       DO I=1,NASHL(NSHDEF)
          IFLAG(SHLLST(I,NSHDEF))=1
       ENDDO

    ENDIF
    !
    !
    RETURN
  END SUBROUTINE SHLFLG
  !
  !----------------------------------------------------------------------
  !

  INTEGER FUNCTION SHLDEC(R,BOUNDS,NBOUND)
    !
    !     decide into which intervall of BOUNDS(1) ..... BOUNDS(NBOUND)
    !     R belongs (i.e. return I so that BOUNDS(I-1) < R <= BOUNDS(I)
    !
    !     returns NBOUND+1 if larger than BOUNDS(NBOUND)
    !
    !     Tibor Rudas Mar 2004 - 
    !

    real(chm_real) R,BOUNDS(*)
    INTEGER NBOUND
    !
    !
    INTEGER I
    !
    !     do _not_ check for R>BOUNDS(NBOUND) first since this case
    !        should be handled above (and should not call this routine at all)
    !
    DO I=1,NBOUND
       IF(R <= BOUNDS(I)) THEN
          SHLDEC=I
          RETURN
       ENDIF
    ENDDO
    SHLDEC=NBOUND+1
    !
    RETURN
  END FUNCTION SHLDEC
#endif /*  (shell_main)*/

end module shell

