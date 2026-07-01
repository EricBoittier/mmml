module ssbpm
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  public :: ssbp0, ssbp1, compot,ssbpm_init      !routines
  public :: fqlmtim,fqlmtim1,fqlmtim2,qssbp,qcavi,rmax,drmax2, &
       acav,bcav,lstssbp,ntssbp,ipa !variables

  !========================================================================
  ! SSBP.FCM : Solvent boundary potential (D.Beglov and B.Roux 1993).
  !
  ! QSSBP            logical flag to setup the solvent term
  ! QFIX              logical flag to use flexible/fixed RMAX of the region
  ! QKIRK             logical flag to setup dielectric reaction field part
  ! QCAVI             logical flag to setup the cavity part
  ! QHSR              logical flag to setup P*V+ sigma*S part
  ! QANGU             logical flag to setup Angular potential
  ! LSTSSBP            list of atoms under the solvent boundary potential
  ! NMULT             number of multipoles in reaction field part
  ! IPA               number of the atom corresponding to RMAX
  ! DIECST            dielectric constant
  ! DRMAX1            deltaRMAX for the reaction field part
  ! DRMAX2            deltaRMAX for the cavity part
  ! PRESI             Pressure in the system
  ! FXCAS             RMAX if it is fixed
  ! ENFA, ENFB, DRHA  Parameters of angular potential
  ! STENS             Surface tension constant
  ! I....             Allocatable arrays necessary for the reaction field calculations.

  real(chm_real) fqlmtim,fqlmtim1,fqlmtim2
  integer,allocatable,dimension(:) :: LSTSSBP, LPTSSBP
  ! Logical
  LOGICAL QFIX, QKIRK, QCAVI, QHSR, QANGU, QSSBP, QEMPI &
       ,Q4SIT,Q5SIT ! DRUDE
  !     COMMON / BSSBP1 / QFIX, QKIRK, &
  !                       QCAVI,QHSR,QANGU,QSSBP, QEMPI &
  !          ,Q4SIT,Q5SIT ! DRUDE

  real(chm_real), allocatable, dimension(:) :: IRADIST,ITVAR12,IRLR3,IZYR3,IXRL,IZXR3, IYRL, &
       IXR2,   IYR2,   IZR2,  IQLMSX, IQLMSY, IQLMSZ, &
       IFACT, IAPOL, IAPOL1, ICOMR, ICOMI, IRCH, ITVAR, &
       IYLMR1, IYLMR2, IYLMR3,IYLMR4, &
       IYLMI1, IYLMI2, IYLMI3,IYLMI4
  ! INTEGER
  INTEGER NTSSBP, NPSSBP,LMA, LMA2, NALM, NMULT, IPA
  !      COMMON / QSSBP2 / &
  !                    NTSSBP, NPSSBP, &
  !                    LMA,   LMA2, NALM, NMULT, IPA

  ! REAL
  real(chm_real) RMAX, DIECST, DRMAX1, DRMAX2, PRESI, EMPI1, EMPI2 &
       ,FXCAS, CANG(5), ACAV(10), BCAV(7), DRHA, STENS
  !      COMMON / QSSBP3 / RMAX, DIECST, DRMAX1, DRMAX2, &
  !                        PRESI, FXCAS, CANG, ACAV, BCAV, DRHA, &
  !                        STENS, EMPI1, EMPI2
#if KEY_IF==1 || KEY_SAVEFCM==1
  
#endif
  !      SAVE / BSSBP1 /
#if KEY_ENDIF==1
  
#endif
  !
contains
  subroutine ssbpm_init()
    use number,only:zero
    fqlmtim=zero
    fqlmtim1=zero
    fqlmtim2=zero
    return
  end subroutine ssbpm_init

  SUBROUTINE SSBP0(ISLCT,JSLCT)
    !-----------------------------------------------------------------------
    ! Spherical Solvent Boundary Potential (SSBP)
    !
    ! Authors:  Dmitrii Beglov and Benoit Roux  (1994)
    !           Department of Chemistry, University of Montreal
    !
    ! Current implementation of the method described in
    ! Beglov & Roux, J. Chem. Phys., 100:9050 (1994).  The method follows
    ! from a rigorous reduction of the multi-dimensional hconfiguration
    ! integral from N solvent molecules (e.g., thermodynamica limit, 10**23)
    ! to "n" solvent molecules (e.g., 1 to 1000).
    ! The SSBP potential corresponds to a constant temperature and constant
    ! pressure system.
    ! The non-bonded interactions must be treated with EXTENDED electrostatics.
    !
    !
    use chm_kinds
    use memory
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord

    implicit none
    INTEGER ISLCT(*),JSLCT(*)
    !
    !-----------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4) WRD
    INTEGER LL12,I
    real(chm_real) PARAM
    LOGICAL ERR
    ERR = .FALSE.
    fqlmtim=zero
    fqlmtim2=zero
    ! Read Input, make initialisation or reset of necessary
    ! allocatable arrays, make choice of a performance
    ! ----------------------------------------------------
    IF(PRNLEV.GT.2)WRITE(OUTU,'(A)') &
         ' Spherical Solvent Boundary Potential (SSBP) '

    IF(QSSBP)THEN

       IF(PRNLEV.GT.2) WRITE(OUTU,'(A)') ' SSBP RESET TO ZERO'
       call chmdealloc('ssbp.src','SSBP0','IRADIST',NATOM,crl=IRADIST)
       call chmdealloc('ssbp.src','SSBP0','LSTSSBP',NATOM,intg=LSTSSBP)
       call chmdealloc('ssbp.src','SSBP0','LPTSSBP',NATOM,intg=LPTSSBP)
       IF(QKIRK)THEN
          call chmdealloc('ssbp.src','SSBP0','ITVAR12',NATOM,crl=ITVAR12)
          call chmdealloc('ssbp.src','SSBP0','IRLR3',NATOM,crl=IRLR3)
          call chmdealloc('ssbp.src','SSBP0','IZYR3',NATOM,crl=IZYR3)
          call chmdealloc('ssbp.src','SSBP0','IXRL',NATOM,crl=IXRL)
          call chmdealloc('ssbp.src','SSBP0','IZXR3',NATOM,crl=IZXR3)
          call chmdealloc('ssbp.src','SSBP0','IYRL',NATOM,crl=IYRL)
          call chmdealloc('ssbp.src','SSBP0','IXR2',NATOM,crl=IXR2)
          call chmdealloc('ssbp.src','SSBP0','IYR2',NATOM,crl=IYR2)
          call chmdealloc('ssbp.src','SSBP0','IZR2',NATOM,crl=IZR2)
          call chmdealloc('ssbp.src','SSBP0','IQLMSX',NATOM,crl=IQLMSX)
          call chmdealloc('ssbp.src','SSBP0','IQLMSY',NATOM,crl=IQLMSY)
          call chmdealloc('ssbp.src','SSBP0','IQLMSZ',NATOM,crl=IQLMSZ)
          call chmdealloc('ssbp.src','SSBP0','IYLMR1',NATOM,crl=IYLMR1)
          call chmdealloc('ssbp.src','SSBP0','IYLMR2',NATOM,crl=IYLMR2)
          call chmdealloc('ssbp.src','SSBP0','IYLMR3',NATOM,crl=IYLMR3)
          call chmdealloc('ssbp.src','SSBP0','IYLMR4',NATOM,crl=IYLMR4)
          call chmdealloc('ssbp.src','SSBP0','IYLMI1',NATOM,crl=IYLMI1)
          call chmdealloc('ssbp.src','SSBP0','IYLMI2',NATOM,crl=IYLMI2)
          call chmdealloc('ssbp.src','SSBP0','IYLMI3',NATOM,crl=IYLMI3)
          call chmdealloc('ssbp.src','SSBP0','IYLMI4',NATOM,crl=IYLMI4)
          call chmdealloc('ssbp.src','SSBP0','IFACT',LMA2,crl=IFACT)
          call chmdealloc('ssbp.src','SSBP0','IAPOL',LMA,crl=IAPOL)
          call chmdealloc('ssbp.src','SSBP0','IAPOL1',LMA,crl=IAPOL1)
          call chmdealloc('ssbp.src','SSBP0','ICOMR',NALM,crl=ICOMR)
          call chmdealloc('ssbp.src','SSBP0','ICOMI',NALM,crl=ICOMI)
          call chmdealloc('ssbp.src','SSBP0','IRCH',NALM,crl=IRCH)
          call chmdealloc('ssbp.src','SSBP0','ITVAR',NALM,crl=ITVAR)
          QKIRK   = .FALSE.
       ENDIF

       QSSBP   = .FALSE.
       QCAVI   = .FALSE.
       QHSR    = .FALSE.
       QANGU   = .FALSE.
       QEMPI   = .FALSE.
       Q4SIT   = .FALSE. ! DRUDE
       Q5SIT   = .FALSE. ! DRUDE

    ELSE

       IF(INDXA(COMLYN,COMLEN,'RESE').GT.0) CALL WRNDIE(0,'<SSBP>','SSBP NOT SETUP')

    ENDIF
    ! --------------------------------------------------
    IF(INDXA(COMLYN,COMLEN,'RESE').LE.0)THEN

       ! Default is include all energy terms for the solvent boundary potential
       QFIX   = .FALSE.
       QKIRK = .TRUE.
       QANGU = .TRUE.
       QCAVI = .TRUE.
       QHSR  = .TRUE.
       QEMPI  = .TRUE.
       Q4SIT = .FALSE. ! DRUDE
       Q5SIT = .FALSE. ! DRUDE

       !  Handle the case with fixed radius
       IF(INDXA(COMLYN,COMLEN,'FIXE').GT.0)THEN
          FXCAS  = GTRMF(COMLYN,COMLEN,'RADI',ZERO)
          IF(PRNLEV.GT.2)WRITE(OUTU,'(A)') &
               ' Finite system with FIXED radius (must give the RADIUS)'
          IF(FXCAS.NE.ZERO)THEN
             IF(PRNLEV.GT.2)WRITE(OUTU,'(A,F10.5)') ' FIXED radius of ', FXCAS
          ELSE
             CALL WRNDIE(-1,'<SSBP>', &
                  'Value of the fixed radius is not defined')
          ENDIF
          QFIX   = .TRUE.
          QKIRK = .FALSE.
          QCAVI = .FALSE.
          QHSR  = .FALSE.
          QANGU = .FALSE.
          QEMPI = .FALSE.
          Q4SIT = .FALSE. ! DRUDE
          Q5SIT = .FALSE. ! DRUDE
          IF(INDXA(COMLYN,COMLEN,'KIRK').GT.0) QKIRK=.TRUE.
          IF(INDXA(COMLYN,COMLEN,'CAVI').GT.0) QCAVI=.TRUE.
          IF(INDXA(COMLYN,COMLEN,'HSR') .GT.0) QHSR =.TRUE.
          IF(INDXA(COMLYN,COMLEN,'ANGU').GT.0) QANGU=.TRUE.
          IF(INDXA(COMLYN,COMLEN,'EMPI').GT.0) QEMPI=.TRUE.
          ! BEGIN DRUDE (G. Lamoureux)
          IF(INDXA(COMLYN,COMLEN,'4SIT').GT.0) Q4SIT=.TRUE.
          IF(INDXA(COMLYN,COMLEN,'5SIT').GT.0) Q5SIT=.TRUE.
          IF(Q4SIT.AND.Q5SIT) Q4SIT=.FALSE.
          if(Q4SIT.AND.(PRNLEV.GT.2))then
             write(OUTU,'(2A)') ' Four-site water molecules ', &
                  '(O=1, X=2, H=3, H=4)'
          endif
          if(Q5SIT.AND.(PRNLEV.GT.2))then
             write(OUTU,'(2A)') ' Five-site water molecules ', &
                  '(O=1, X=2, X=3, H=4, H=5)'
          endif
          ! END DRUDE (G. Lamoureux)
       ENDIF

       ! Do you want to turn off specific terms?
       IF(INDXA(COMLYN,COMLEN,'NOKI').GT.0) QKIRK=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'NOCA').GT.0) QCAVI=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'NOHS').GT.0) QHSR =.FALSE.
       IF(INDXA(COMLYN,COMLEN,'NOAN').GT.0) QANGU=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'NEMP').GT.0) QEMPI=.FALSE.

       IF(QKIRK)THEN
          NMULT  = GTRMI(COMLYN,COMLEN,'NMUL',15)
          DIECST=78.4
          DIECST = GTRMF(COMLYN,COMLEN,'DIEC',DIECST)
          IF(PRNLEV.GT.2)WRITE(OUTU,'(A,F6.3)') &
               ' Kirkwood dielectric reaction field, dielectric constant ', DIECST
          IF(PRNLEV.GT.2)WRITE(OUTU,'(A,I10)') &
               ' Number of terms in the multipolar expansion ', NMULT
          DRMAX1=2.8
          DRMAX1 = GTRMF(COMLYN,COMLEN,'DRDI',DRMAX1)
          LMA    = NMULT+1
          LMA2   = NMULT*2+1
          NALM   = NATOM*LMA
          LL12   = NMULT*2
          call chmalloc('ssbp.src','SSBP0','ITVAR12',NATOM,crl=ITVAR12)
          call chmalloc('ssbp.src','SSBP0','IRLR3',NATOM,crl=IRLR3)
          call chmalloc('ssbp.src','SSBP0','IZYR3',NATOM,crl=IZYR3)
          call chmalloc('ssbp.src','SSBP0','IXRL',NATOM,crl=IXRL)
          call chmalloc('ssbp.src','SSBP0','IZXR3',NATOM,crl=IZXR3)
          call chmalloc('ssbp.src','SSBP0','IYRL',NATOM,crl=IYRL)
          call chmalloc('ssbp.src','SSBP0','IXR2',NATOM,crl=IXR2)
          call chmalloc('ssbp.src','SSBP0','IYR2',NATOM,crl=IYR2)
          call chmalloc('ssbp.src','SSBP0','IZR2',NATOM,crl=IZR2)
          call chmalloc('ssbp.src','SSBP0','IQLMSX',NATOM,crl=IQLMSX)
          call chmalloc('ssbp.src','SSBP0','IQLMSY',NATOM,crl=IQLMSY)
          call chmalloc('ssbp.src','SSBP0','IQLMSZ',NATOM,crl=IQLMSZ)
          call chmalloc('ssbp.src','SSBP0','IYLMR1',NATOM,crl=IYLMR1)
          call chmalloc('ssbp.src','SSBP0','IYLMR2',NATOM,crl=IYLMR2)
          call chmalloc('ssbp.src','SSBP0','IYLMR3',NATOM,crl=IYLMR3)
          call chmalloc('ssbp.src','SSBP0','IYLMR4',NATOM,crl=IYLMR4)
          call chmalloc('ssbp.src','SSBP0','IYLMI1',NATOM,crl=IYLMI1)
          call chmalloc('ssbp.src','SSBP0','IYLMI2',NATOM,crl=IYLMI2)
          call chmalloc('ssbp.src','SSBP0','IYLMI3',NATOM,crl=IYLMI3)
          call chmalloc('ssbp.src','SSBP0','IYLMI4',NATOM,crl=IYLMI4)
          call chmalloc('ssbp.src','SSBP0','IFACT',LMA2,crl=IFACT)
          call chmalloc('ssbp.src','SSBP0','IAPOL',LMA,crl=IAPOL)
          call chmalloc('ssbp.src','SSBP0','IAPOL1',LMA,crl=IAPOL1)
          call chmalloc('ssbp.src','SSBP0','ICOMR',NALM,crl=ICOMR)
          call chmalloc('ssbp.src','SSBP0','ICOMI',NALM,crl=ICOMI)
          call chmalloc('ssbp.src','SSBP0','IRCH',NALM,crl=IRCH)
          call chmalloc('ssbp.src','SSBP0','ITVAR',NALM,crl=ITVAR)
          QSSBP=.TRUE.
          call FACTO(LL12,IFACT)
       ENDIF

       IF(QCAVI)THEN
          IF(PRNLEV.GT.2)WRITE(OUTU,'(2A)') &
               ' Cavity potential applied to atom selection ', &
               ' only (based on RISM-HNC with water)'
          DRMAX2=2.6
          DRMAX2 = GTRMF(COMLYN,COMLEN,'DRCA',DRMAX2)
          ! Parameters for the cavity potential
          ACAV(5)=-1.6649500
          ACAV(1)= 0.56198800
          ACAV(2)=-0.072798148
          ACAV(3)= 0.00426122036
          ACAV(4)=-0.0000925233817
          ACAV(6)= 0.0840
          ACAV(7)=15.39333
          BCAV(1)= 1.319978287
          BCAV(2)=-0.840953501
          BCAV(3)=-0.001602388122
          BCAV(4)=-8.392886499
          BCAV(5)=BCAV(2)+BCAV(4)
          BCAV(6)= 1.6
          BCAV(7)=-8.4751210228
          QSSBP=.TRUE.
       ENDIF

       IF(QHSR)THEN
          IF(PRNLEV.GT.2)WRITE(OUTU,'(A)') &
               ' Hard-Sphere-Restriction contribution'
          PRESI  = GTRMF(COMLYN,COMLEN,'PRES',ONE)
          STENS  = 0.033
          STENS  = GTRMF(COMLYN,COMLEN,'SURT',STENS)
          IF(PRNLEV.GT.2)WRITE(OUTU,'(2(A,F10.5))') &
               ' Pressure        ', PRESI, &
               ' Surface tension ', STENS
          QSSBP=.TRUE.
       ENDIF

       ! Empirical Correction
       IF (QEMPI) THEN
          if (prnlev > 2) write(outu, '(A)') 'Empirical Correction'
          PARAM = 1.1
          EMPI1 = GTRMF(COMLYN,COMLEN,'EMP1',PARAM)
          PARAM = 0.008
          EMPI2 = GTRMF(COMLYN,COMLEN,'EMP2',PARAM)
          if (prnlev > 2) write(outu, '(2(A, F10.5))') &
               ' Empirical Parameter 1 ', empi1, &
               ' Empirical Parameter 2 ', empi2
          if (prnlev > 2) write(outu, '(A)') 'Gaussian form'
          QSSBP=.TRUE.
       ENDIF

       IF(QANGU)THEN
          IF(PRNLEV.GT.2)WRITE(OUTU,'(2A)') &
               ' Angular potential for H2O (for ', &
               ' isotropic orientation near the boundary)'
          ! BEGIN DRUDE (G. Lamoureux)
          IF(INDXA(COMLYN,COMLEN,'4SIT').GT.0) Q4SIT=.TRUE.
          IF(INDXA(COMLYN,COMLEN,'5SIT').GT.0) Q5SIT=.TRUE.
          IF(Q4SIT.AND.Q5SIT) Q4SIT=.FALSE.
          if(Q4SIT.AND.(PRNLEV.GT.2))then
             write(OUTU,'(2A)') ' Four-site water molecules ', &
                  '(O=1, X=2, H=3, H=4)'
          endif
          if(Q5SIT.AND.(PRNLEV.GT.2))then
             write(OUTU,'(2A)') ' Five-site water molecules ', &
                  '(O=1, X=2, X=3, H=4, H=5)'
          endif
          ! END DRUDE (G. Lamoureux)
          ! Parameters for the angular potential
          DRHA   = -1.0
          CANG(5)=0.840661
          CANG(4)=- 1.20064327666
          CANG(3)=- 3.067139576698
          CANG(2)=1.766839115555
          CANG(1)= 2.4085919002
          QSSBP=.TRUE.
       ENDIF

       IF(QSSBP)THEN
          CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
          IF(ERR) CALL WRNDIE(-1,'<SSBP>','ATOM SELECTION ERROR')
          call chmalloc('ssbp.src','SSBP0','IRADIST',NATOM,crl=IRADIST)
          call chmalloc('ssbp.src','SSBP0','LSTSSBP',NATOM,intg=LSTSSBP)
          call chmalloc('ssbp.src','SSBP0','LPTSSBP',NATOM,intg=LPTSSBP)
          call STSSBP0(NATOM,ISLCT,JSLCT,LSTSSBP,NTSSBP,LPTSSBP,NPSSBP)
       ENDIF

    ENDIF
    ! -----------------------------------------
    RETURN
  END SUBROUTINE SSBP0
  !
  SUBROUTINE STSSBP0(NATOM,ISLCT,JSLCT,LSTSSBP,NTSSBP,LPTSSBP,NPSSBP)
    !-----------------------------------------------------------------------
    !     Definition of the selection array LSTSSBP(NTSSBP)
    !     input NATOM ISLCT JSLCT
    !     output LSTSSBP,NTSSBP
    !     LSTSSBP- numbers of selected atoms
    !     NTSSBP- total number of selected atoms

    use chm_kinds
    use stream
    use parallel
    implicit none
    ! Input variables
    INTEGER NATOM, ISLCT(*), JSLCT(*)
    ! Output variables
    INTEGER LSTSSBP(*), NTSSBP, LPTSSBP(*), NPSSBP
    ! Local variables
    INTEGER I
    NTSSBP=0
    NPSSBP=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.1)THEN
          NTSSBP=NTSSBP+1
          LSTSSBP(NTSSBP)=I
       ENDIF
       IF(JSLCT(I).EQ.1)THEN
          NPSSBP=NPSSBP+1
          LPTSSBP(NPSSBP)=I
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE STSSBP0

  SUBROUTINE FACTO(NFACT,FACT)
    !-----------------------------------------------------------------------
    !     Calculate FACT=NFACT!
    !
    use chm_kinds
    implicit none
    ! Output variable
    real(chm_real) FACT(*)
    ! Input variable
    INTEGER NFACT
    ! Local variables
    INTEGER I,J
    !
    FACT(1)=1
    IF(NFACT.GT.0)THEN
       DO I=1,NFACT
          J=I+1
          FACT(J)=FACT(I)*I
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE FACTO

  SUBROUTINE SSBP1(NATOM,X,Y,Z,ENSSBP,CG,DX,DY,DZ)
    !-----------------------------------------------------------------------
    !
    ! Total Spherical Solvent Boundary Potential = EKIRK + EHSR + EANGU +
    !
    ! Each subroutine is called according to its logical flag set in MMFP
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real) ENSSBP,ENKIR,ENCAV,ENHSR,ENANGU,EEMPI
    real(chm_real) X(*),Y(*),Z(*),CG(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM
    ENKIR=ZERO
    ENCAV=ZERO
    ENHSR=ZERO
    ENANGU=ZERO
    EEMPI=ZERO
    !
    ! Get the radius from the center of the sphere for all particles
    CALL DEFRAD(QFIX,QKIRK,FXCAS,RMAX,IRADIST, X,Y,Z, &
         NPSSBP,LPTSSBP,NATOM,IPA)

    ! Kirkwood's multipolar expansion for reaction field
    IF (QKIRK)THEN
       CALL SMEXP(QFIX,ENKIR, DX,DY,DZ, NMULT,DIECST,RMAX,DRMAX1, &
            X, Y, Z, CG, IRADIST, IFACT, IAPOL, &
            IAPOL1,ITVAR12, IRLR3, IZYR3, &
            IXRL,  IZXR3,   IYRL,  IXR2, &
            IYR2,  IZR2,    ICOMR, ICOMI, &
            IQLMSX,IQLMSY,  IQLMSZ,IRCH, &
            ITVAR, IYLMR1,  IYLMR2,IYLMR3, &
            IYLMR4,IYLMI1,  IYLMI2,IYLMI3, &
            IYLMI4,NATOM,IPA)
    ENDIF

    ! Cavity potential of mean force calculated from RISM-HNC
    IF (QCAVI)THEN
       CALL SSBCAVP(QFIX,ENCAV,DX,DY,DZ,RMAX,DRMAX2,X,Y,Z, &
            IRADIST,ACAV,BCAV,NTSSBP,LSTSSBP,IPA)
    ENDIF

    ! Hard-Sphere-Restriction contribution = P*V+sigma*S
    IF (QHSR)THEN
       CALL SSBHSRP(QFIX,ENHSR,DX,DY,DZ,RMAX,PRESI,STENS,X,Y,Z,IRADIST,IPA)
    ENDIF

    ! Empirical Correction
    IF (QEMPI) THEN
       CALL SSBEMPI(QFIX,EEMPI,DX,DY,DZ,RMAX,X,Y,Z, &
            IRADIST,NTSSBP,LSTSSBP,IPA,EMPI1,EMPI2)
    ENDIF


    ! Angular correction potential for isotropic distribution at boundary
    IF (QANGU)THEN
       CALL SSBANGP(QFIX,ENANGU,DX,DY,DZ,RMAX,DRHA,X,Y,Z, &
            IRADIST,CANG,NTSSBP,LSTSSBP,IPA,Q4SIT,Q5SIT) ! DRUDE
    ENDIF
    ENSSBP=ENKIR+ENCAV+ENHSR+ENANGU+EEMPI
    RETURN
  END SUBROUTINE SSBP1

  SUBROUTINE DEFRAD(QFIX,QKIRK,FXCAS,RMAX,RADIST,X,Y,Z, &
       NPSSBP,LPTSSBP,NATOM,IPA)
    !-----------------------------------------------------------------------
    !     Calculate radial distance (RADIST) from the center
    !     and RMAX by choice (QFIX/.NOT.QFIX)
    use chm_kinds
    use number
    use param_store, only: set_param

    implicit none

    LOGICAL QKIRK,QFIX
    INTEGER NATOM,IPA,I,J
    INTEGER NPSSBP,LPTSSBP(*)
    real(chm_real) FXCAS,RMAX,RADIST(*),X(*),Y(*),Z(*)
    DO I=1,NATOM
       RADIST(I)=SQRT(X(I)*X(I)+Y(I)*Y(I)+Z(I)*Z(I))
    ENDDO
    IPA=0
    IF(.NOT.QFIX)THEN
       RMAX=ZERO
    ELSE
       RMAX=FXCAS
    ENDIF
    DO  J=1,NPSSBP
       I=LPTSSBP(J)
       IF(RADIST(I).GT.RMAX)THEN
          IF(.NOT.QFIX)THEN
             RMAX=RADIST(I)
             IPA=I
          ELSE
             IF(QKIRK) CALL WRNDIE(-1,'<SSBP>','ATOM IS OUTSIDE OF THE FIXED RMAX')
          ENDIF
       ENDIF
    ENDDO
    call set_param('RMAX',RMAX)
    RETURN
  END SUBROUTINE DEFRAD

  SUBROUTINE SMEXP(QFIX,ESSBKP,DX,DY,DZ,NMULT,DIECST,RMAX,DRMAX, &
       X,Y,Z,CH,RADIST,FACT,APOL, &
       APOL1,TVAR12,RLR3,ZYR3, &
       XRL,ZXR3,YRL,XR2, &
       YR2,ZR2,COMR,COMI, &
       QLMSX,QLMSY,QLMSZ,RCH, &
       TVAR,YLMR1,YLMR2,YLMR3, &
       YLMR4,YLMI1,YLMI2,YLMI3, &
       YLMI4,NATOM,IPA)
    !-----------------------------------------------------------------------
    !     Reaction field (Kirkwood's) part
    use chm_kinds
    use stream
    use number
    use exfunc
    use consta
    use parallel
    use machutil,only:eclock
    implicit none
    LOGICAL QFIX
    COMPLEX*16 COM0,COM
    real(chm_real) ESSBKP,DIECST,RMAX,DRMAX
    real(chm_real) X(*),Y(*),Z(*),CH(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) FACT(*),APOL(*),APOL1(*)
    real(chm_real) YRL(*),XR2(*),YR2(*),ZR2(*)
    real(chm_real) COMR(*),COMI(*)
    real(chm_real) QLMSX(*),QLMSY(*),QLMSZ(*)
    real(chm_real) RCH(*),TVAR(*)
    real(chm_real) YLMR1(*),YLMR2(*),YLMR4(*),YLMR3(*)
    real(chm_real) YLMI1(*),YLMI2(*),YLMI4(*),YLMI3(*)
    real(chm_real) RLR3(*),ZYR3(*),XRL(*),ZXR3(*)
    real(chm_real) RADIST(*),TVAR12(*)
    INTEGER NATOM,IPA,NMULT
    !     local variables
    INTEGER I,J,K,L,M,N,ILP,ICOD
    real(chm_real) CHTOT,DRM1,DDRM1,RMAX1,RSXY,RATOM2,TVARI,RATOM3
    real(chm_real) COEF0,RMAXF,RMAXLU,RMAXS,COEF,COL,COLX,COLY,COLZ
    real(chm_real) QLMS
    real(chm_real) XITMP,RADIMP
    ESSBKP=ZERO
    !     Definition of deltaRmaxDiel (DRM1) from the total charge CHTOT
    !     and RMAX
    fqlmtim1=eclock()
    CHTOT=ZERO
    DO I=1,NATOM
       CHTOT=CHTOT+CH(I)
    ENDDO
    CHTOT=ABS(CHTOT)
    DRM1=DRMAX-CHTOT*1.6*EXP(-0.5*RMAX)
    DDRM1=(DRMAX-DRM1)*0.5+ONE
    RMAX1=RMAX+DRM1

    !     Definitions of some useful arrays for the fast calculation of
    !     multipoles and forces

    DO I=1,NATOM
       TVAR(I)=ONE
       RCH(I)=CH(I)
       COM=(1.0,0.0)
       COMR(I)=1.0
       COMI(I)=0.0

       RADIMP=RADIST(I)
       XITMP=X(I)
       IF((XITMP.LT.RSMALL).AND.(Y(I).LT.RSMALL))THEN
          XITMP=XITMP+RSMALL
          RADIMP=SQRT(XITMP**2+Y(I)**2+Z(I)**2)
       ENDIF

       RSXY=XITMP**2+Y(I)**2
       RATOM2=RADIMP**2
       RATOM3=RATOM2*RADIMP
#if KEY_GNU==1 || KEY_SINGLE==1
       COM0=(cmplx(ZERO,ONE,chm_cmpx)*Y(I)+cmplx(ONE,ZERO,chm_cmpx)*XITMP)/RADIMP
#else /**/
       COM0=((ZERO,ONE)*Y(I)+(ONE,ZERO)*XITMP)/RADIMP
#endif 
       TVARI=Z(I)/RADIMP
       RLR3(I)=RSXY/RATOM3
       ZYR3(I)=Z(I)*Y(I)/RATOM3
       ZXR3(I)=Z(I)*XITMP/RATOM3
       XR2(I)=XITMP/RATOM2
       YR2(I)=Y(I)/RATOM2
       ZR2(I)=Z(I)/RATOM2
       XRL(I)=XITMP/RSXY
       YRL(I)=Y(I)/RSXY
       TVAR12(I)=TVARI/(ONE-TVARI**2)

       if(NMULT.NE.0)THEN

          DO J=1,NMULT
             TVAR(NATOM*J+I)=TVAR(NATOM*(J-1)+I)*TVARI
             RCH(NATOM*J+I)=RCH(NATOM*(J-1)+I)*RADIMP
             COM=COM*COM0
             COMR(NATOM*J+I)=DBLE(COM)
             COMI(NATOM*J+I)=aimag(COM)
          ENDDO

       ENDIF

    ENDDO

    COEF0=-HALF*CCELEC*(DIECST-ONE)
    IF(.NOT.QFIX)THEN
       RMAXF=DDRM1/RMAX1/RMAX
    ENDIF
    RMAXLU=RMAX1
    RMAXS=RMAX1**2
    fqlmtim2=fqlmtim2+eclock()-fqlmtim1

    !     Main loop K=l-index in the multipole Qlm
    DO I=1,NMULT+1
       RMAXLU=RMAXLU/RMAXS
       K=I-1
       N=K

       !     Operations with Legendre's polynomes
       !     Definition
       CALL POLE(N,APOL,FACT)
       !     Duplication
       CALL EQPOL(L,APOL1,N,APOL)
       !     Differentiation
       CALL DPOL(L,APOL1)

       COEF=COEF0/(DIECST+ONE*(K/I))
       COEF=COEF/(2*K+1)
       COEF=COEF*RMAXLU
       IF(.NOT.QFIX)THEN
          COL=(2*K+1)*COEF*RMAXF
          COLX=COL*X(IPA)
          COLY=COL*Y(IPA)
          COLZ=COL*Z(IPA)
       ENDIF
       !     Enclosed Main loop M=m-index in the multipole Qlm
       DO J=1,I
          ICOD=1
          M=J-1

          IF(M.GT.0)THEN
             CALL EQPOL(N,APOL,L,APOL1)
             CALL DPOL(L,APOL1)
             ICOD=2
          ENDIF
          fqlmtim1=eclock()
          CALL FQLM(K,M,NATOM,N,L,APOL,APOL1,FACT, &
               QLMS,QLMSX,QLMSY,QLMSZ,TVAR12,RLR3,ZYR3,XRL,ZXR3, &
               YRL,XR2,YR2,ZR2,COMR,COMI,RCH,TVAR, &
               YLMR1,YLMR2,YLMR3,YLMR4,YLMI1,YLMI2,YLMI3,YLMI4)
          fqlmtim=fqlmtim+eclock()-fqlmtim1
          !     Subroutine calculating Qlm**2=QLMS
          !     and derivatives QLMSX,QLMSY,QLMSZ
          
          QLMS=ICOD*QLMS
#if KEY_PARALLEL==1
          ESSBKP=ESSBKP+COEF*QLMS/NUMNOD
          DO ILP=MYNODP,NATOM,NUMNOD
#else /**/
          ESSBKP=ESSBKP+COEF*QLMS
          DO ILP=1,NATOM
#endif 
             DX(ILP)=DX(ILP)+ICOD*QLMSX(ILP)*COEF
             DY(ILP)=DY(ILP)+ICOD*QLMSY(ILP)*COEF
             DZ(ILP)=DZ(ILP)+ICOD*QLMSZ(ILP)*COEF
          ENDDO
#if KEY_PARALLEL==1
          IF(MYNOD.EQ.0)THEN                        
#endif
             IF(.NOT.QFIX)THEN
                DX(IPA)=DX(IPA)-COLX*QLMS
                DY(IPA)=DY(IPA)-COLY*QLMS
                DZ(IPA)=DZ(IPA)-COLZ*QLMS
             ENDIF
#if KEY_PARALLEL==1
          ENDIF                                     
#endif
          
       ENDDO
    ENDDO
    
    RETURN
  END  SUBROUTINE SMEXP


  SUBROUTINE FQLM(LIND,MIND,NATOM,NPOL,LPOL, &
       APOL,APOL1,FACT,QLMS,QLMSX,QLMSY,QLMSZ, &
       TVAR12,RLR3,ZYR3,XRL,ZXR3,YRL,XR2,YR2,ZR2, &
       COMR,COMI,RCH,TVAR, &
       YLMR1,YLMR2,YLMR3,YLMR4, &
       YLMI1,YLMI2,YLMI3,YLMI4)
    !-----------------------------------------------------------------------
    !     Calculation of the multipole moment Qlm with
    !     l=lind,m=mind
    use chm_kinds
    use number
    use parallel
    implicit none
    INTEGER LIND,MIND,NATOM,NPOL,LPOL

    real(chm_real) QLMS,APOL(*),APOL1(*),FACT(*)
    real(chm_real) QLMSX(*),QLMSY(*),QLMSZ(*)
    real(chm_real) TVAR12(*),RLR3(*),ZYR3(*),XRL(*),ZXR3(*)
    real(chm_real) YRL(*),XR2(*),YR2(*),ZR2(*)
    real(chm_real) COMI(*),COMR(*)
    real(chm_real) RCH(*),TVAR(*)
    real(chm_real) YLMR1(*),YLMR2(*),YLMR4(*),YLMR3(*)
    real(chm_real) YLMI1(*),YLMI2(*),YLMI4(*),YLMI3(*)

    !     Local variables
    INTEGER J,I
    real(chm_real) BVAR,QLMR,QLMI,RC,DVAR,DVAR1,YLMIT,YLMRT
    real(chm_real) YLMIC,YLMRC,DVAR2,BVAR1,BVARR1,BVARI1
    real(chm_real) atemp(2)

    BVAR=(2*LIND+1)*FACT(LIND-MIND+1)/FACT(LIND+MIND+1)
    QLMR=ZERO
    QLMI=ZERO
#if KEY_PARALLEL==1
    DO I=MYNODP,NATOM,NUMNOD
#else /**/
    DO I=1,NATOM
#endif 
       RC=RCH(I+LIND*NATOM)
       DVAR=ZERO
       DVAR1=ZERO
       DO J=1,NPOL+1
          DVAR=DVAR+APOL(J)*TVAR(I+(NPOL+1-J)*NATOM)
       ENDDO
       DO J=1,LPOL+1
          DVAR1=DVAR1+APOL1(J)*TVAR(I+(LPOL+1-J)*NATOM)
       ENDDO
       YLMIT=COMI(I+NATOM*MIND)
       YLMRT=COMR(I+MIND*NATOM)
       YLMIC=YLMIT*DVAR
       YLMRC=YLMRT*DVAR
       DVAR2=DVAR1-DVAR*TVAR12(I)*MIND
       BVAR1=DVAR2*RLR3(I)
       YLMR3(I)=YLMRT*BVAR1
       YLMI3(I)=YLMIT*BVAR1
       BVARR1=-DVAR2*ZYR3(I)
       BVARI1=MIND*XRL(I)
       YLMR2(I)=YLMRT*BVARR1-YLMIC*BVARI1
       YLMI2(I)=YLMIT*BVARR1+YLMRC*BVARI1
       BVARR1=-DVAR2*ZXR3(I)
       BVARI1=-MIND*YRL(I)
       YLMR1(I)=YLMRT*BVARR1-YLMIC*BVARI1
       YLMI1(I)=YLMIT*BVARR1+YLMRC*BVARI1
       YLMI4(I)=YLMIC
       YLMR4(I)=YLMRC
       QLMI=QLMI+RC*YLMI4(I)
       QLMR=QLMR+RC*YLMR4(I)
    ENDDO
#if KEY_PARALLEL==1
    atemp(1)=qlmr
    atemp(2)=qlmi
    CALL GCOMB(atemp,2)
    qlmr=atemp(1)
    qlmi=atemp(2)
#endif 
    QLMS=QLMR*QLMR+QLMI*QLMI
    QLMS=QLMS*BVAR
#if KEY_PARALLEL==1
    DO I=MYNODP,NATOM,NUMNOD
#else /**/
    DO I=1,NATOM
#endif 
       RC=2*BVAR*RCH(I+LIND*NATOM)
       BVAR1=LIND*XR2(I)
       BVARR1=BVAR1*YLMR4(I)+YLMR1(I)
       BVARI1=BVAR1*YLMI4(I)+YLMI1(I)
       QLMSX(I)=RC*(BVARR1*QLMR+BVARI1*QLMI)
       BVAR1=LIND*YR2(I)
       BVARR1=BVAR1*YLMR4(I)+YLMR2(I)
       BVARI1=BVAR1*YLMI4(I)+YLMI2(I)
       QLMSY(I)=RC*(BVARR1*QLMR+BVARI1*QLMI)
       BVAR1=LIND*ZR2(I)
       BVARR1=BVAR1*YLMR4(I)+YLMR3(I)
       BVARI1=BVAR1*YLMI4(I)+YLMI3(I)
       QLMSZ(I)=RC*(BVARR1*QLMR+BVARI1*QLMI)
    ENDDO
    RETURN
  END SUBROUTINE FQLM

  SUBROUTINE DPOL(N,A)
    !-----------------------------------------------------------------------
    !     Differentiation of Legendre's polynome of N degree
    !     with coefficients A.
    use chm_kinds
    implicit none
    real(chm_real) a(*)
    INTEGER N,I

    IF(N.GT.0)THEN
       DO I=1,N
          A(I)=A(I)*(N-I+1)
       ENDDO
       N=N-1
    ELSE
       N=0
       A(1)=0.0
    ENDIF
    RETURN
  END SUBROUTINE DPOL

  SUBROUTINE EQPOL(N1,A1,N2,A2)
    !-----------------------------------------------------------------------
    !     Duplicate a Legendre's polynome of N2 degree with coefficients A2
    use chm_kinds
    implicit none
    real(chm_real) A1(*),A2(*)
    INTEGER I,N1,N2

    DO I=1,N2+1
       A1(I)=A2(I)
    ENDDO
    N1=N2
    RETURN
  END SUBROUTINE EQPOL

  SUBROUTINE POLE(NPOL,APOL,FACT)
    !-----------------------------------------------------------------------
    !     Calculation of the coefficients (APOL) of the Legendre's
    !     polynome of the degree NPOL
    use chm_kinds
    implicit none
    real(chm_real) APOL(*),FACT(*)
    real(chm_real) BARG
    INTEGER NPOL,IBARG,NHAL
    INTEGER I,J,K,L,M

    BARG=DBLE(NPOL)/2.0
    IBARG=INT(BARG)
    IF(ABS(BARG-DBLE(IBARG)).GT.0.1)THEN
       NHAL=(NPOL-1)/2
    ELSE
       NHAL=NPOL/2
    ENDIF
    DO I=1,NPOL+1
       APOL(I)=0.0
    ENDDO
    DO J=1,NHAL+1
       M=J-1
       K=NPOL-M
       L=2*K+1
       BARG=FACT(L)/FACT(J)/FACT(K+1)/FACT(L-NPOL)
       BARG=BARG/(2.0)**NPOL
       APOL(1+2*M)=BARG*(-1)**M
    ENDDO
    RETURN
  END SUBROUTINE POLE

  SUBROUTINE SSBHSRP(QFIX,ESSBP3,DX,DY,DZ,RMAX,PRESI,STENS, &
       X,Y,Z,RADIST,IPA)
    !-----------------------------------------------------------------------
    !     Calculation of Pressure*Volume+Surface_tension*Surface term
    !     Pressure=PRESI, Surface_tension=STENS
    use chm_kinds
    use consta
    use parallel
    implicit none
    LOGICAL QFIX
    INTEGER IPA
    real(chm_real) RMAX,PRESI,STENS, ESSBP3
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*),RADIST(*)
    real(chm_real) AVAL,BVAL,VAL

#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0)THEN                      
#endif
       AVAL=ATMOSP*PRESI*RMAX**3*PI*4.0/3.0
       BVAL=STENS*4.0*PI*RMAX*RMAX
       ESSBP3=AVAL+BVAL
       IF(.NOT.QFIX)THEN
          VAL=(3.0*AVAL+2.0*BVAL)/RMAX
          DX(IPA)=DX(IPA)+VAL*X(IPA)/RADIST(IPA)
          DY(IPA)=DY(IPA)+VAL*Y(IPA)/RADIST(IPA)
          DZ(IPA)=DZ(IPA)+VAL*Z(IPA)/RADIST(IPA)
       ELSE
       ENDIF
#if KEY_PARALLEL==1
    ENDIF                                   
#endif

    RETURN
  END SUBROUTINE SSBHSRP

  SUBROUTINE SSBEMPI(QFIX,ESSBP2,DX,DY,DZ,RMAX,X,Y,Z, &
       RADIST,NTSSBP,LSTSSBP,IPA,EMPI1,EMPI2)
    !-----------------------------------------------------------------------
    !     Calculation of the empirical correction
    use chm_kinds
    use stream
    use number
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none
    LOGICAL QFIX
    INTEGER NTSSBP,LSTSSBP(*),IPA
    real(chm_real) RMAX, ESSBP2, EMPI1,EMPI2,PAR1,PAR2
    INTEGER I,J
    real(chm_real) X(*),Y(*),Z(*),RADIST(*),DX(*),DY(*),DZ(*)
    real(chm_real) POT2,DPOT2
    PAR1 = EMPI1 * 8.88 / RMAX
    PAR2 = EMPI2 * 8.88 * 8.88 / RMAX / RMAX
    ESSBP2=ZERO
    !       Do 10 I=1,ntssbp
#if KEY_PARALLEL==1
    DO I=MYNODP,NTSSBP,NUMNOD
#else /**/
    DO I=1,NTSSBP
#endif 
       J=LSTSSBP(I)
       CALL EMPFORCEG(RADIST(J),POT2,DPOT2,PAR1,PAR2)
       ESSBP2=ESSBP2 + POT2
       IF (J .EQ. IPA) THEN
          DX(IPA)=DX(IPA)-POT2*X(IPA)/RMAX/RMAX
          DY(IPA)=DY(IPA)-POT2*Y(IPA)/RMAX/RMAX
          DZ(IPA)=DZ(IPA)-POT2*Z(IPA)/RMAX/RMAX
       ELSE
          DX(J)=DX(J)+DPOT2*X(J)/RADIST(J)
          DY(J)=DY(J)+DPOT2*Y(J)/RADIST(J)
          DZ(J)=DZ(J)+DPOT2*Z(J)/RADIST(J)
          DX(IPA)=DX(IPA)+ &
               (-POT2/RMAX+POT2*2*EMPI2*8.88*8.88/(RMAX**3)*RADIST(J)**2) &
               *X(IPA)/RMAX
          DY(IPA)=DY(IPA)+ &
               (-POT2/RMAX+POT2*2*EMPI2*8.88*8.88/(RMAX**3)*RADIST(J)**2) &
               *Y(IPA)/RMAX
          DZ(IPA)=DZ(IPA)+ &
               (-POT2/RMAX+POT2*2*EMPI2*8.88*8.88/(RMAX**3)*RADIST(J)**2) &
               *Z(IPA)/RMAX
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE SSBEMPI
  
  SUBROUTINE EMPFORCEG(RMAX,ENG,FORCE, PAR1, PAR2)
    !-----------------------------------------------------------------------
    !     empirical energy and force
    
    use chm_kinds
    use consta
    implicit none
    real(chm_real) RMAX,ENG,FORCE,PAR1,PAR2
    ENG=PAR1*EXP(-PAR2*RMAX*RMAX)
    FORCE=-ENG*PAR2*2.0*RMAX
    RETURN
  END SUBROUTINE EMPFORCEG
  
  SUBROUTINE SSBCAVP(QFIX,ESSBP2,DX,DY,DZ,RMAX,DRCAV,X,Y,Z, &
       RADIST,ACAV,BCAV,NTSSBP,LSTSSBP,IPA)
    !-----------------------------------------------------------------------
    !     Calculation of the Cavity part.
    !     This constribution was calculated from the cavity potential of
    !     mean force between a LJ TIP3P-like oxygen atom and a large
    !     hard-sphere of radius RMAX.
    !     The polynomial approximation has been fitted using mathematica
    !     and yields a boundary potential that is very similar to the
    !     SBOUND method of C. L. Brooks.
    use chm_kinds
    use stream
    use number
    use parallel
    implicit none
    LOGICAL QFIX
    INTEGER NTSSBP,LSTSSBP(*),IPA
    real(chm_real) RMAX, DRCAV, ESSBP2, ACAV(*), BCAV(*)
    INTEGER I,J,PNTSSBP
    real(chm_real) X(*),Y(*),Z(*),RADIST(*),DX(*),DY(*),DZ(*)
    real(chm_real) RMAX2,SHI,DSHI,XARG,POT2,DPOT2,DUT
    RMAX2=RMAX+DRCAV-2.6
    CALL RARA(RMAX2,SHI,DSHI,ACAV)
    ESSBP2=ZERO
    DUT=ZERO
    PNTSSBP=0
#if KEY_PARALLEL==1
    DO I=MYNODP,NTSSBP,NUMNOD
#else /**/
    DO I=1,NTSSBP
#endif 
       PNTSSBP=PNTSSBP+1
       J=LSTSSBP(I)
       XARG=-RMAX2+RADIST(J)
       CALL COMPOT(XARG,POT2,DPOT2,BCAV)
       ESSBP2=ESSBP2+POT2
       DX(J)=DX(J)+DPOT2*X(J)/RADIST(J)
       DY(J)=DY(J)+DPOT2*Y(J)/RADIST(J)
       DZ(J)=DZ(J)+DPOT2*Z(J)/RADIST(J)
       DUT=DUT+DPOT2
    ENDDO
    IF(.NOT.QFIX)THEN
       DSHI=DSHI*PNTSSBP-DUT
       DX(IPA)=DX(IPA)+DSHI*X(IPA)/RADIST(IPA)
       DY(IPA)=DY(IPA)+DSHI*Y(IPA)/RADIST(IPA)
       DZ(IPA)=DZ(IPA)+DSHI*Z(IPA)/RADIST(IPA)
    ENDIF
    ESSBP2=ESSBP2+SHI*PNTSSBP
    RETURN
  END SUBROUTINE SSBCAVP

  SUBROUTINE RARA(XARGA,CAPA,DCAPA,ACAV)
    !-----------------------------------------------------------------------
    !     Calculation of the Function-axis shift of the approximated
    !     cavitypotential
    use chm_kinds
    use stream
    implicit none
    real(chm_real) RAMA,XARGA,CAPA,DCAPA,ACAV(*)
    real(chm_real) XARGA2,XARGA3
    !a0   ACAV(5)=-1.6649500
    !a1   ACAV(1)=0.56198800
    !a2   ACAV(2)=-0.072798148
    !a3   ACAV(3)=0.00426122036
    !a4   ACAV(4)=-0.0000925233817
    !ac   ACAV(6)=0.0840
    !Xc   ACAV(7)=15.39333
    RAMA=XARGA+2.6
    IF(RAMA.LT.5.0)THEN
       IF(PRNLEV.GT.2)WRITE (OUTU,'(A)') &
            ' SSBP CAVITY: RMAX+DRCAV<MIN  APPROXIMATED'
       IF(PRNLEV.GT.2)WRITE (outu,'(A,F8.4)') &
            '         MIN = 5.0 A, RMAX+DRCAV = ', RAMA
    ENDIF
    XARGA2=XARGA**2
    XARGA3=XARGA2*XARGA
    IF(XARGA.GT.ACAV(7))THEN
       CAPA=ACAV(6)
       DCAPA=0.0
    ELSE
       CAPA=ACAV(5)+ACAV(1)*XARGA+ACAV(2)*XARGA2+ACAV(3)*XARGA3+ &
            ACAV(4)*XARGA*XARGA3
       DCAPA=ACAV(1)+2*XARGA*ACAV(2)+3*ACAV(3)*XARGA2+4*ACAV(4)*XARGA3
    ENDIF
    CAPA=CAPA+8.5
    RETURN
  END SUBROUTINE RARA

  SUBROUTINE COMPOT(XARGB,CAPB,DCAPB,BCAV)
    !-----------------------------------------------------------------------
    !     Approximated calculation of the cavity potential
    use chm_kinds
    implicit none
    real(chm_real) XARGB,XARGB2,CAPB,DCAPB,BCAV(*)
    !     Coefficients:
    !     BCAV(1)= 1.319978287
    !     BCAV(2)=-0.840953501
    !     BCAV(3)=-0.001602388122
    !     BCAV(4)=-8.392886499
    !     BCAV(5)=BCAV(2)+BCAV(4)
    !     BCAV(6)= 1.6
    !     BCAV(7)=-8.4751210228
    IF(XARGB.LT.-5.0)THEN
       CAPB=BCAV(7)
       DCAPB=0.0
    ELSE
       XARGB2=XARGB**2
       IF(XARGB.GT.0.0)THEN
          CAPB=BCAV(5)+BCAV(6)*XARGB2
          DCAPB=2.0*XARGB*BCAV(6)
       ELSE
          CAPB=BCAV(2)/(1.0+XARGB2/BCAV(1))+BCAV(3)*XARGB2+BCAV(4)
          DCAPB= 2.0*XARGB* &
               (BCAV(3)-BCAV(2)/BCAV(1)/(1.0+XARGB2/BCAV(1))**2)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE COMPOT

  SUBROUTINE SSBANGP(QFIX,SSBPE5,DX,DY,DZ,RMAX5,DRHA,X,Y,Z, &
       RADIST,CANG,NTSSBP,LSTSSBP,IPA,Q4SIT,Q5SIT) ! DRUDE
    !-----------------------------------------------------------------------
    !     Calculation of the Angular Potential
    use chm_kinds
    use stream
    use number
    use parallel
    implicit none
    LOGICAL QFIX
    LOGICAL Q4SIT,Q5SIT ! DRUDE
    INTEGER NTSSBP,IPA,LSTSSBP(*)
    real(chm_real) RMAX5,X(*),Y(*),Z(*),RADIST(*),SSBPE5, &
         DX(*),DY(*),DZ(*),DRHA,CANG(*)
    !     Local variables:
    INTEGER I,J
    real(chm_real) RIPA,DUTIPA,RADLOC,VARDP,VANG,FRCNST,DPOT5,POT5
    real(chm_real) VALO(3),DVALO(3),DVALH(2,3),VALH(2,3)
    RIPA=RMAX5
    IF(.NOT.QFIX)RIPA=RADIST(IPA)
    SSBPE5=ZERO
    DUTIPA=ZERO
#if KEY_PARALLEL==1
    DO I=MYNODP,NTSSBP,NUMNOD
#else /**/
    DO I=1,NTSSBP
#endif 
       J=LSTSSBP(I)
       RADLOC=RADIST(J)
       VARDP=RADLOC-RIPA-DRHA
       VALO(1)=X(J)
       VALO(2)=Y(J)
       VALO(3)=Z(J)
       ! BEGIN DRUDE (G. Lamoureux)
       if(Q4SIT) then
          VALH(1,1)=X(J+2)
          VALH(1,2)=Y(J+2)
          VALH(1,3)=Z(J+2)
          VALH(2,1)=X(J+3)
          VALH(2,2)=Y(J+3)
          VALH(2,3)=Z(J+3)
       elseif(Q5SIT) then
          VALH(1,1)=X(J+3)
          VALH(1,2)=Y(J+3)
          VALH(1,3)=Z(J+3)
          VALH(2,1)=X(J+4)
          VALH(2,2)=Y(J+4)
          VALH(2,3)=Z(J+4)
       else
          ! END DRUDE (G. Lamoureux)
          VALH(1,1)=X(J+1)
          VALH(1,2)=Y(J+1)
          VALH(1,3)=Z(J+1)
          VALH(2,1)=X(J+2)
          VALH(2,2)=Y(J+2)
          VALH(2,3)=Z(J+2)
          ! BEGIN DRUDE (G. Lamoureux)
       endif
       ! END DRUDE (G. Lamoureux)
       IF(VARDP.GT.0.0)THEN
          CALL ANGLAIP(RADLOC,VALO,VALH,DVALO,DVALH,VANG,CANG)
          DPOT5=VARDP*VANG
          DUTIPA=DUTIPA+DPOT5
          POT5=0.5*DPOT5*VARDP
          FRCNST=0.5*VARDP**2
          SSBPE5=SSBPE5+POT5
          DX(J)=DX(J)+DPOT5*VALO(1)/RADLOC+FRCNST*DVALO(1)
          DY(J)=DY(J)+DPOT5*VALO(2)/RADLOC+FRCNST*DVALO(2)
          DZ(J)=DZ(J)+DPOT5*VALO(3)/RADLOC+FRCNST*DVALO(3)
          ! BEGIN DRUDE (G. Lamoureux)
          if(Q4SIT) then
             DX(J+2)=DX(J+2)+FRCNST*DVALH(1,1)
             DY(J+2)=DY(J+2)+FRCNST*DVALH(1,2)
             DZ(J+2)=DZ(J+2)+FRCNST*DVALH(1,3)
             DX(J+3)=DX(J+3)+FRCNST*DVALH(2,1)
             DY(J+3)=DY(J+3)+FRCNST*DVALH(2,2)
             DZ(J+3)=DZ(J+3)+FRCNST*DVALH(2,3)
          elseif(Q5SIT) then
             DX(J+3)=DX(J+3)+FRCNST*DVALH(1,1)
             DY(J+3)=DY(J+3)+FRCNST*DVALH(1,2)
             DZ(J+3)=DZ(J+3)+FRCNST*DVALH(1,3)
             DX(J+4)=DX(J+4)+FRCNST*DVALH(2,1)
             DY(J+4)=DY(J+4)+FRCNST*DVALH(2,2)
             DZ(J+4)=DZ(J+4)+FRCNST*DVALH(2,3)
          else
             ! END DRUDE (G. Lamoureux)
             DX(J+1)=DX(J+1)+FRCNST*DVALH(1,1)
             DY(J+1)=DY(J+1)+FRCNST*DVALH(1,2)
             DZ(J+1)=DZ(J+1)+FRCNST*DVALH(1,3)
             DX(J+2)=DX(J+2)+FRCNST*DVALH(2,1)
             DY(J+2)=DY(J+2)+FRCNST*DVALH(2,2)
             DZ(J+2)=DZ(J+2)+FRCNST*DVALH(2,3)
             ! BEGIN DRUDE (G. Lamoureux)
          endif
          ! END DRUDE (G. Lamoureux)
       ENDIF
    ENDDO
    IF(.NOT.QFIX)THEN
       DX(IPA)=DX(IPA)-DUTIPA*X(IPA)/RADIST(IPA)
       DY(IPA)=DY(IPA)-DUTIPA*Y(IPA)/RADIST(IPA)
       DZ(IPA)=DZ(IPA)-DUTIPA*Z(IPA)/RADIST(IPA)
    ELSE
    ENDIF
    RETURN
  END SUBROUTINE SSBANGP

  SUBROUTINE ANGLAIP(RADLOC,VALO,VALH,DVALO,DVALH,VANG,CANG)
    !-----------------------------------------------------------------------
    !     Calculation of the angular part of the angular potential
    !     This contribution was developed empirically to make more
    !     isotropic the orientational distribution function of the waters
    !     near the outer shell.
    !     Warning:  It works only for a 3 site water models such as TIP3P.
    !     The order of the atoms in the psf MUST BE (oxygen,hydrogen,hydrogen)
    !
    use chm_kinds
    use number
    implicit none
    INTEGER I,J
    real(chm_real) OHVEC(2,3),RADLOC,VALO(3),DVALO(3),DVALH(2,3), &
         VALH(2,3)
    real(chm_real) EXAN(2),REBMUN1(2),REBMUN2(2),REBMUN3(2),CO(2), &
         DEXAN(2)
    real(chm_real) CO1,CO2,CO3,CO4,VANG,CANG(*),SCALMUL,OHABS
    ! Coeffitients:
    ! CANG(5)=0.840661
    ! CANG(4)=- 1.20064327666
    ! CANG(3)=- 3.067139576698
    ! CANG(2)=1.766839115555
    ! CANG(1)= 2.4085919002
    DO J=1,2
       SCALMUL=ZERO
       DO I=1,3
          OHVEC(J,I)=VALH(J,I)-VALO(I)
          SCALMUL=SCALMUL+VALO(I)*OHVEC(J,I)
       ENDDO
       OHABS=OHVEC(J,1)**2+OHVEC(J,2)**2+OHVEC(J,3)**2
       REBMUN1(J)=1.0/RADLOC/SQRT(OHABS)
       REBMUN2(J)=SCALMUL/OHABS
       REBMUN3(J)=SCALMUL/RADLOC**2
       CO(J)=SCALMUL*REBMUN1(J)
       CO1=CO(J)
       CO2=CO1*CO(J)
       CO3=CO2*CO(J)
       CO4=CO3*CO(J)
       DEXAN(J)=2* &
            (4.0*CANG(1)*CO3+3.0*CANG(2)*CO2+2.0*CANG(3)*CO1+CANG(4))
       EXAN(J)=2* &
            (CANG(1)*CO4+CANG(2)*CO3+CANG(3)*CO2+CANG(4)*CO1+CANG(5))
    ENDDO
    !
    DO I=1,3
       DVALH(1,I)=DEXAN(1)*REBMUN1(1)*(-REBMUN2(1)*OHVEC(1,I)+VALO(I))
       DVALH(2,I)=DEXAN(2)*REBMUN1(2)*(-REBMUN2(2)*OHVEC(2,I)+VALO(I))
       DVALO(I)=DEXAN(1)*REBMUN1(1)* &
            (-VALO(I)*(REBMUN3(1)+1.0)+OHVEC(1,I)*(1.0+REBMUN2(1)))
       DVALO(I)=DVALO(I)+DEXAN(2)*REBMUN1(2)* &
            (-VALO(I)*(REBMUN3(2)+1.0)+OHVEC(2,I)*(1.0+REBMUN2(2)))
    ENDDO
    !
    VANG=EXAN(1)+EXAN(2)
    RETURN
  END  SUBROUTINE ANGLAIP

end module ssbpm

