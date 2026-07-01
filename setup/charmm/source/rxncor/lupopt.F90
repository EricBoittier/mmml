module lup
#if KEY_RXNCOR==1

contains

  SUBROUTINE LUPOPT(COMLY1,COMLE1)
    !-----------------------------------------------------------------------
    !     Basic function:
    !     Top level routine for LUPSUB:
    !     Implement Ron Elber's Locally Updated Planes Algorithm
    !
    !     K. Kuczera, Lawrence, KS 03/08/97
    !
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use exfunc
    use psf
    use memory
    use lupcom
    implicit none
    real(chm_real),allocatable,dimension(:) :: IREX
    real(chm_real),allocatable,dimension(:) :: IREY
    real(chm_real),allocatable,dimension(:) :: IREZ
    type(chm_array),allocatable,dimension(:) :: IBXP
    type(chm_array),allocatable,dimension(:) :: IBYP
    type(chm_array),allocatable,dimension(:) :: IBZP
    real(chm_real),allocatable,dimension(:) :: IBPE
    real(chm_real),allocatable,dimension(:) :: IBPP
    real(chm_real),allocatable,dimension(:) :: IBSD
    CHARACTER(len=*) COMLY1
    INTEGER COMLE1
    ! ... MXPATH is the max number of path points
    INTEGER, PARAMETER :: MXPATH=200
    !
    call chmalloc('lupopt.src','LUPOPT','IREX',NATOM,crl=IREX)
    call chmalloc('lupopt.src','LUPOPT','IREY',NATOM,crl=IREY)
    call chmalloc('lupopt.src','LUPOPT','IREZ',NATOM,crl=IREZ)
    call chmalloc_chm_array('lupopt.src','LUPOPT','IBXP',MXPATH,IBXP)
    call chmalloc_chm_array('lupopt.src','LUPOPT','IBYP',MXPATH,IBYP)
    call chmalloc_chm_array('lupopt.src','LUPOPT','IBZP',MXPATH,IBZP)
    call chmalloc('lupopt.src','LUPOPT','IBPE',MXPATH,crl=IBPE)
    call chmalloc('lupopt.src','LUPOPT','IBPP',MXPATH,crl=IBPP)
    call chmalloc('lupopt.src','LUPOPT','IBSD',MXPATH,crl=IBSD)

    CALL LUPSUB(COMLY1,COMLE1,IREX,IREY,IREZ, &
         IBXP,IBYP,IBZP, &
         IBPE,IBPP,IBSD,MXPATH)
    !
    ! ... release allocated space
    call chmdealloc('lupopt.src','LUPOPT','IREX',NATOM,crl=IREX)
    call chmdealloc('lupopt.src','LUPOPT','IREY',NATOM,crl=IREY)
    call chmdealloc('lupopt.src','LUPOPT','IREZ',NATOM,crl=IREZ)
    call chmdealloc_chm_array('lupopt.src','LUPOPT','IBXP',MXPATH,IBXP)
    call chmdealloc_chm_array('lupopt.src','LUPOPT','IBYP',MXPATH,IBYP)
    call chmdealloc_chm_array('lupopt.src','LUPOPT','IBZP',MXPATH,IBZP)
    call chmdealloc('lupopt.src','LUPOPT','IBPE',MXPATH,crl=ibpe)
    call chmdealloc('lupopt.src','LUPOPT','IBPP',MXPATH,crl=ibpp)
    call chmdealloc('lupopt.src','LUPOPT','IBSD',MXPATH,crl=ibsd)
    !
    RETURN
  END SUBROUTINE LUPOPT

  SUBROUTINE LUPSUB(COMLY1,COMLE1,XREF,YREF,ZREF, &
       IBX,IBY,IBZ,EPATH,EPATHP,SDSTEP,MXPATH)
    !-----------------------------------------------------------------------
    !     Basic function:
    !     Implement Ron Elber's Locally Updated Planes Algorithm
    !     Work is in Caretsian space, path optimized by constrained
    !     steepest descent.
    !     See C.Choi & R.Elber, J.Chem.Phys. 94:751-760 (1991).
    !
    !     1. An initial conformational path is read in
    !        a) linear interpolation between MAIN and COMP coordinates
    !        b) series of structures in CHARMM COOR files (see LUPINI)
    !     2. For each structure inside path (i.e. excluding ends):
    !        a. Path vector is computed
    !        b. A steepest descent step is taken, subject to
    !           rigid-body and path constraints (see LUPCNS)
    !        c. Step is accepted if energy decreases
    !        d. Convergence is checked by monitoring energy change
    !     3. If procedure has converged along whole path, stop;
    !           otherwise return to step 2, possibly decreasing step.
    !
    !     Variables :
    !     Command line contains control parameters for LUP procedure
    !       including the optimization. Only SD optomization should be used
    !     XREF,YREF,ZREF - work coordinates
    !     NATOM - number of atoms
    !     IBX,IBY,IBZ - HEAP pointers to structures along path
    !     MXPATH - maximum number of path points, set in LUPOPT
    !     EPATH,EPATHP - arrays of current and previous cycle energies
    !       of the path structures
    !
    !     K. Kuczera, Lawrence, KS 03/08/97
    !
    !-----------------------------------------------------------------------

    use chm_kinds
    use chm_types
#if KEY_CHEQ==1
    use cheq,only:qcg                  
#endif

    use dimens_fcm
    use exfunc
    use number
    use memory
    use stream
    use string
    !
    use coord
    use coordc
    use ctitla
    use deriv
    use cvio
    use energym
    use lupcns_module, only: lupcns
    use lupcom
    use psf
    !
    use bases_fcm
    use hbondm
    use image
    use inbnd

    use minmiz_module, only: minmiz

    implicit none
    !
    !
    real(chm_real),allocatable,dimension(:) :: LUPCX
    real(chm_real),allocatable,dimension(:) :: LUPCY
    real(chm_real),allocatable,dimension(:) :: LUPCZ
    integer,allocatable,dimension(:) :: FREEAT

    CHARACTER(len=*) COMLY1
    CHARACTER(len=50) COMLY2
    INTEGER COMLE1,MXPATH
    !
    type(chm_array):: IBX(MXPATH),IBY(MXPATH),IBZ(MXPATH)
    real(chm_real) EPATH(MXPATH),EPATHP(MXPATH),SDSTEP(MXPATH)
    real(chm_real) XREF(NATOM),YREF(NATOM),ZREF(NATOM)
    !
    ! local variables
    real(chm_real) EPSENR,FACT,EDIFF,EDMAX,EDRMS,STEP,GNORM,S
    INTEGER NPATH,NTRO,INITP,MXCYC,NCYC,NCONV,LPRI,IOPT,I,J
    INTEGER COMLE2,IPVO,IF,IL,NPATO,NOK,I1
    !
    INTEGER  LUPDIM, IX, IY, IZ
    ! parameters used in WRITCV
    INTEGER      NFREAT,NDEGF
    !
    LOGICAL QDONE
    !
    ! ... 0. Print header, set main parameters in lupcom.f90 ,
    !        pointers, defaults, etc.
    !
    WRITE(OUTU,*)
    WRITE(OUTU,*) ' >>>>>> Reaction path optimization by Locally', &
         ' Updated Planes <<<<<<'
    WRITE(OUTU,*)
    QLUPCS = .TRUE.
    NLUPC = 7
    !
    LUPDIM = NLUPC*NATOM
    call chmalloc('lupopt.src','LUPSUB','LUPCX',LUPDIM,crl=LUPCX)
    call chmalloc('lupopt.src','LUPSUB','LUPCY',LUPDIM,crl=LUPCY)
    call chmalloc('lupopt.src','LUPSUB','LUPCZ',LUPDIM,crl=LUPCZ)
    !
    NTRO   = 21
    INITP  = 1
    MXCYC  = 100
    EPSENR = 0.01
    STEP   = 0.01
    IPVO   = 1
    LPRI   = 1
    DO I=1,MXPATH
       EPATH(I)=ZERO
       EPATHP(I)=ZERO
    END DO
    !
    ! ... 1. Process the LUPOPT command line
    !
    !
    NPATH  =GTRMI(COMLY1, COMLE1, 'NPAT',  MXPATH)
    NTRO   =GTRMI(COMLY1, COMLE1, 'UOUT',      21)
    INITP  =GTRMI(COMLY1, COMLE1, 'INIT',       1)
    EPSENR =GTRMF(COMLY1, COMLE1, 'EPSE',  PT0001)
    STEP   =GTRMF(COMLY1, COMLE1, 'STEP',    PT01)
    MXCYC  =GTRMI(COMLY1, COMLE1, 'MAXC',     100)
    IPVO   =GTRMI(COMLY1, COMLE1, 'IPVO',       1)
    LPRI   =GTRMI(COMLY1, COMLE1, 'LPRI',       1)
    IOPT   =GTRMI(COMLY1, COMLE1, 'IOPT',       1)
    !
    WRITE(OUTU,*) ' LUPOPT> Path length                 = ', NPATH
    WRITE(OUTU,*) ' LUPOPT> Path output unit            = ', NTRO
    WRITE(OUTU,*) ' LUPOPT> Path initialization  mode   = ', INITP
    WRITE(OUTU,*) ' LUPOPT> Energy convergence set at   = ', EPSENR
    WRITE(OUTU,*) ' LUPOPT> Optimization step set to    = ', STEP
    WRITE(OUTU,*) ' LUPOPT> No. of optimization cycles  = ', MXCYC
    !
    IF (NPATH < 1 .OR. NPATH > MXPATH) THEN
       WRITE(OUTU,*) ' LUPOPT> ****Error: Illegal number of points'
       WRITE(OUTU,*) '         NPATH must be between 1 and ',MXPATH
       CALL WRNDIE(-4,'LUPOPT','Redefine number of path points')
    END IF
    !
    DO I=1,NPATH
       SDSTEP(I)=STEP
    END DO
    !
    IF(IOPT == 2) THEN
       WRITE(OUTU,*)  ' ...Using LUP step-by-step SD optimizer'
    ELSE
       WRITE(OUTU,*)  ' ...Using standard  SD optimizer'
    END IF
    !
    ! ...save what remains of command line for use in main loop
    !
    IF(IPVO == 2) THEN
       WRITE(OUTU,*)  ' ...Using symmetric path vectors: I-1 -> I+1'
    ELSE
       WRITE(OUTU,*)  ' ...Using standard  path vectors: I   -> I+1'
    END IF
    !
    ! ...save what remains of command line for use in main loop
    !
    COMLY2 = COMLY1
    COMLE2 = COMLE1
    !
    ! ... 2. Initialize the path and save structures on HEAP
    !
    IF(INITP == 1) THEN
       WRITE(OUTU,*) ' LUPOPT> ...Straight line path from MAIN to COMP' &
            ,' in Cartesian space'
       DO I=1,NPATH
          !
          FACT = I-1
          FACT = FACT/(NPATH-1)
          DO J=1,NATOM
             XREF(J) = X(J)*(ONE-FACT) + XCOMP(J)*FACT
             YREF(J) = Y(J)*(ONE-FACT) + YCOMP(J)*FACT
             ZREF(J) = Z(J)*(ONE-FACT) + ZCOMP(J)*FACT
          END DO
          !
          call chmalloc('lupopt.src','LUPSUB','IBX(I)',NATOM,crl=IBX(I)%a)
          call chmalloc('lupopt.src','LUPSUB','IBY(I)',NATOM,crl=IBY(I)%a)
          call chmalloc('lupopt.src','LUPSUB','IBZ(I)',NATOM,crl=IBZ(I)%a)

          CALL COPYD(XREF,YREF,ZREF, &
               IBX(I)%a,IBY(I)%a,IBZ(I)%a,NATOM)
          !
       END DO  ! I=1,NPATH
    END IF    ! INITP == 1
    !
    IF(INITP == 2) THEN
       WRITE(OUTU,*) ' LUPOPT> ...Reading individual structures '
       CALL LUPINI(IBX,IBY,IBZ,NPATH)
    END IF
    !
    IF(INITP < 1 .OR. INITP > 2) THEN
       WRITE(OUTU,*) ' LUPOPT> ****Error: Illegal initialization mode'
       CALL WRNDIE(-4,'LUPOPT',' Change path initialization')
    END IF
    !
    ! ... Calculate end-point energies once for completeness
    !
    I=1

    CALL UPDATE(COMLY1,COMLE1,ibx(i)%a,iby(i)%a,ibx(i)%a,WMAIN,.TRUE., &
         .TRUE.,.TRUE.,.TRUE.,.TRUE. &
         ,0,[zero],[zero],[zero],[zero],[zero],[zero])

    CALL ENERGY(ibx(i)%a,iby(i)%a,ibz(i)%a,DX,DY,DZ,BNBND,BIMAG,1)
    EPATH(I) = EPROP(EPOT)
    !
    I=NPATH

    CALL UPDATE(COMLY1,COMLE1,ibx(i)%a,iby(i)%a,ibx(i)%a,WMAIN,.TRUE., &
         .TRUE.,.TRUE.,.TRUE.,.TRUE., &
         0,[zero],[zero],[zero],[zero],[zero],[zero])

    CALL ENERGY(ibx(i)%a,iby(i)%a,ibz(i)%a,DX,DY,DZ,BNBND,BIMAG,1)
    EPATH(I) = EPROP(EPOT)
    !
    ! ... 3. Main loop: iterate over central path structures
    !        Save initial structure in XREF,YREF,ZREF
    !        Optimize with constraints
    !        Check convergence
    !        Save optimized structure to HEAP
    !
    IF=2
    IL=NPATH - 1
    NPATO=NPATH-2
    !
    QDONE = .FALSE.
    NCYC = 0
100 NCYC = NCYC + 1
    NCONV = 0
    NOK   = 0
    EDMAX = -1.0D6
    EDRMS = ZERO
    !
    DO I=IF,IL
       !
       ! ... Retrieve current structure from HEAP and save copy for reference
       !     + put path direction vector into COMP coordinates
       !
       CALL COPYD(IBX(I)%a,IBY(I)%a,IBZ(I)%a, &
            X,Y,Z,NATOM)
       !
       IF(IPVO == 2) THEN
          ! ...Symmetric path vectors, from structure I-1 to I+1
          I1 = I + 1
          CALL COPYD(IBX(I1)%a,IBY(I1)%a,IBZ(I1)%a, &
               XCOMP,YCOMP,ZCOMP,NATOM)
          I1 = I - 1
          CALL COPYD(IBX(I1)%a,IBY(I1)%a,IBZ(I1)%a, &
               XREF,YREF,ZREF,NATOM)
          !
          DO J=1,NATOM
             XCOMP(J) = XCOMP(J) - XREF(J)
             YCOMP(J) = YCOMP(J) - YREF(J)
             ZCOMP(J) = ZCOMP(J) - ZREF(J)
          END DO
          !
       ELSE
          ! ...Standard path vectors, from structure I to I+1
          I1 = I + 1
          CALL COPYD(IBX(I1)%a,IBY(I1)%a,IBZ(I1)%a, &
               XCOMP,YCOMP,ZCOMP,NATOM)
          DO J=1,NATOM
             XCOMP(J) = XCOMP(J) - X(J)
             YCOMP(J) = YCOMP(J) - Y(J)
             ZCOMP(J) = ZCOMP(J) - Z(J)
          END DO
          !
       END IF ! IPVO == 2
       !
       ! ...Save current path point for reference
       DO J=1,NATOM
          XREF(J) = X(J)
          YREF(J) = Y(J)
          ZREF(J) = Z(J)
       END DO
       !
       ! ... Optimization
       !
       COMLE1=COMLE2
       COMLY1 = COMLY2(1:COMLE2)
       !
       ! ... choose CHARMM optimizer or local step-by-step optimizer
       ! ... for CHARMM - supply minimzation options in command line
       !
       IF(IOPT /= 2) THEN
          CALL MINMIZ(COMLY1,COMLE1)
          EPATH(I) = EPROP(EPOT)
       ELSE
          !
          ! ... Perform a steepest descent step with LUP constraints
          !
          ! .....update lists, calculate energy and gradient
          !
          CALL UPDATE(COMLY1,COMLE1,X,Y,Z,WMAIN,.TRUE., &
               .TRUE.,.TRUE.,.TRUE.,.TRUE., &
               0,[zero],[zero],[zero],[zero],[zero],[zero])
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
          ! ....apply LUP constraints
          !
          CALL LUPCNS(X,Y,Z,XCOMP,YCOMP,ZCOMP,DX,DY,DZ,AMASS,NATOM, &
               LUPCX,LUPCY,LUPCZ)
          ! .....calculate gradient norm
          !
          GNORM = ZERO
          DO J=1,NATOM
             GNORM = GNORM + DX(J)**2 + DY(J)**2 + DZ(J)**2
          END DO
          GNORM = SQRT(GNORM/NATOM)
          ! ......perform SD step
          !
          IF (GNORM  >  TENM5) THEN
             S = SDSTEP(I)/GNORM
          ELSE
             S = ZERO
          ENDIF
          !
          DO J=1,NATOM
             X(J) = X(J) - S*DX(J)
             Y(J) = Y(J) - S*DY(J)
             Z(J) = Z(J) - S*DZ(J)
          END DO
          ! ......recalculate energy
          !
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
          !
          EPATH(I) = EPROP(EPOT)
          !
          ! .......adjust step size for each path point
          EDIFF = EPATH(I)-EPATHP(I)
          IF(EDIFF > 0) THEN
             SDSTEP(I) = SDSTEP(I)/TWO
          ELSE
             SDSTEP(I) = SDSTEP(I)*TWOPT4
          ENDIF
          !
       ENDIF  ! IOPT /= 2
       !
       !
       ! ... Check single point convergence using energy criterion
       !
       IF(NCYC > 1) THEN
          EDIFF = EPATH(I)-EPATHP(I)
          IF(EDIFF > 0) THEN
             ! .......reject step
             IF(PRNLEV >= 2) THEN
                WRITE(OUTU,*) ' LUPOPT> Warning: E increase at point I=',I &
                     ,' in cycle', NCYC
             ENDIF
             EPATH(I) = EPATHP(I)
             DO J=1,NATOM
                X(J) = XREF(J)
                Y(J) = YREF(J)
                Z(J) = ZREF(J)
             END DO
          ELSE
             ! .......accept step
             NOK = NOK + 1
          END IF
          IF(ABS(EDIFF) < EPSENR) NCONV=NCONV+1
       ELSE
          NCONV = 0
       END IF
       !
       ! ...measure rate of convergence
       EDIFF = ABS(EPATH(I)-EPATHP(I))
       IF(EDIFF > EDMAX) EDMAX=EDIFF
       EDRMS = EDRMS + EDIFF**2
       !
       EPATHP(I) = EPATH(I)
       !
       ! ... Store current coordinates back on HEAP
       !
       CALL COPYD(X,Y,Z, &
            IBX(I)%a,IBY(I)%a,IBZ(I)%a,NATOM)
       !
    END DO   ! I=IF,IL
    !
    ! ... 4. End of main loop :  check path convergence
    !
    IF(NCONV >= NPATO .OR. NCYC >= MXCYC) QDONE = .TRUE.
    !
    ! ... Test printout
    !
    IF(MOD(NCYC,LPRI) == 0 .OR. QDONE) THEN
       WRITE(OUTU,*) '   Path energies and steps at cycle', NCYC
       DO I=1,NPATH
          WRITE(OUTU,'(1X,I4,2X,F12.4,2X,F12.6)') I,EPATH(I),SDSTEP(I)
       END DO
       EDRMS=SQRT(EDRMS/NPATO)
       WRITE(OUTU,900) EDMAX,EDRMS
900    FORMAT(1X,' Maximum E change was ',F12.4,' Rms E change was', &
            F12.4,' kcal/mol')
    END IF
    !
    IF(.NOT.QDONE) GO TO 100
    !
    ! ... End of LUP procedure
    !
    IF(NCONV == NPATO) THEN
       WRITE(OUTU,*) ' LUPOPT> Path optimization completed in', &
            NCYC, ' steps'
    END IF
    IF(NCONV < NPATO) THEN
       WRITE(OUTU,*) ' LUPOPT> Path optimization NOT completed in', &
            NCYC, ' steps'
    END IF
    !
    ! ... 5. Save path to binary trajectory file
    !
    NDEGF=3*NATOM-6
    NFREAT = NATOM
    call chmalloc('lupopt.src','LUPSUB','FREEAT',NFREAT,intg=FREEAT)
    !
    DO I=1,NPATH
       CALL WRITCV(IBX(I)%a,IBY(I)%a,IBZ(I)%a, &
#if KEY_CHEQ==1
            CG,QCG,                                       & 
#endif
            NATOM, &
            FREEAT,NFREAT,1,I,NDEGF,ZERO,1,NPATH,TITLEA, &
            NTITLA,NTRO,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
    END DO
    !
    call chmdealloc('lupopt.src','LUPSUB','FREEAT',NFREAT,intg=freeat)
    !
    ! ... 6. Release HEAP space
    !
    DO I=1,NPATH
       call chmdealloc('lupopt.src','LUPSUB','IBX(I)',NATOM,crl=IBX(I)%a)
       call chmdealloc('lupopt.src','LUPSUB','IBY(I)',NATOM,crl=IBY(I)%a)
       call chmdealloc('lupopt.src','LUPSUB','IBZ(I)',NATOM,crl=IBZ(I)%a)
    END DO
    !
    call chmdealloc('lupopt.src','LUPSUB','LUPCX',LUPDIM,crl=LUPCX)
    call chmdealloc('lupopt.src','LUPSUB','LUPCY',LUPDIM,crl=LUPCY)
    call chmdealloc('lupopt.src','LUPSUB','LUPCZ',LUPDIM,crl=LUPCZ)
    !
    RETURN
  END SUBROUTINE LUPSUB

  SUBROUTINE LUPINI(IBX,IBY,IBZ,NPATH)
    !---------------------------------------------------------------------
    !  Initialize LUP path by reading in a series of individual
    !  CHARMM structures
    !
    !   K.Kuczera, Lawrence, KS 10-Mar-97
    !---------------------------------------------------------------------
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use exfunc
    use comand
    use coord
    use psf
    use machio,only:vopen
    use stream
    use memory
    use ctitla
    use coorio_mod,only:coorio
    implicit none
    !
    INTEGER NPATH
    type(chm_array) IBX(NPATH),IBY(NPATH),IBZ(NPATH)
    !
    INTEGER COMLE1,I,NUNIT
    INTEGER ICNTRL(20)
    CHARACTER(len=4) COMLY1
    LOGICAL EOF,QFLAG
    !
    !
    DO I=1,NPATH
       !
       ! ...read coordinate file name from command line
       !
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
            .TRUE.,'LUPINI> ')
       !
       ! ...open the coordinate file for read
       !
       NUNIT = 99
       CALL VOPEN(NUNIT,COMLYN(1:COMLEN),'FORMATTED','READ',QFLAG,0)
       IF(QFLAG) THEN
          CALL WRNDIE(-4,'<LUPINI>','Could not open file.')
       END IF
       !
       ! ...read in CHARMM formatted coordinates
       COMLY1 = 'CARD'
       COMLE1 = LEN(COMLY1)
       CALL COORIO(-1,NUNIT,COMLY1,COMLE1,TITLEB,NTITLB, &
            ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
            RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
       !
       ! ...store path structures on HEAP
       !
       call chmalloc('lupopt.src','LUPINI','IBX(I)',NATOM,crl=IBX(I)%a)
       call chmalloc('lupopt.src','LUPINI','IBY(I)',NATOM,crl=IBY(I)%a)
       call chmalloc('lupopt.src','LUPINI','IBZ(I)',NATOM,crl=IBZ(I)%a)
       CALL COPYD(X,Y,Z,IBX(I)%a,IBY(I)%a,IBZ(I)%a,NATOM)
       !
       ! ...write a message
       !
       WRITE(OUTU,*) ' ...structure ',I,' read from  ',COMLYN(1:COMLEN)
    END DO
    RETURN
  END SUBROUTINE LUPINI
  !
#endif 

end module lup

