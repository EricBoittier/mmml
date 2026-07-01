module hqbmm
  use chm_kinds
  use dimens_fcm
  implicit none

#if KEY_HQBM==1 /*hqbm_outer*/
  !
  !   Emanuele Paci <10-02-97>
  !   This common file contains hqbm defaults which are read
  !   from the parameter file. They are processed in the routines EMBIAS
  !   *************** MULTIPLE REACTION COORDINATE VERSION ***************
  !
  !      DFALPHA - Bias constraint scale factor
  !      RCUT    - Distance cutoff for counting contacts in native
  !                structure
  !      TOL     - Additional tolerance added to rcut for counting native
  !                contacts in non-native structures
  !      BETA    - Beta in heavisides smoothing function for phi native cont.
  !      BETH    - Beta in heavisides smoothing function for h bonds.
  !      HCUT    - Cutoff distance for hydrogen bonds
  !      HXFC  - Fudge factor for contribution of native contacts to
  !                HX protection
  !      HXFH  - Fudge factor for contribution of hydrogen bonds to
  !                HX protection
  !      NSEL    - Number of selected atoms
  !      ISEL    - bored
  !      JEEF1   - use EEF1 GREF-GSOLV to measure burial rather than
  !                number of native contacts
  !      JNHCON  - use amide nitrogen - sidechain heavy atom native
  !                contacts instead of just heavy atom contacts
  !      JSPLIT  - print contributions from h-bonds and burial separately on
  !                output units.
  !      JHQFIX  - fixes target rc to initial value
  !      JRZERO  - sets target rc to zero
  !      NCALPH  - alpha exponent to first native contact term in rc4
  !      JNONN   - whether to use only native, or all contacts in the
  !                bias
  !      Junk:
  !      RMIN, RMAX, JDELTA,
  !
  !-----------------------------------------------------------------------
  !   Flags or values not present in this file may not be specified as
  !   defaults when reading a card parameter file is read.  See PARRDR
  !   for more information.  All variables here MUST also be parsed
  !   in PARRDR.
  !

  character(len=*), parameter,private :: FILE = 'hqbm.src'
  !------------------------------------------------------
  !     REAL DECLARATIONS
  real(chm_real) BETA,RCUT,TOL
  real(chm_real) DFALPHA1,GAMMA1,XIMAX1
  real(chm_real) DFALPHA2,GAMMA2,XIMAX2
  real(chm_real) DFALPHA3,DFAALPHA3,XIMAX3,GAMMA3,AGAMMA3
  real(chm_real) DFALPHA4,XIMAX4,GAMMA4,BETC,BETH,HCUT,HXFC,HXFH, &
       HXCTON,HXCTOF
  real(chm_real) DFALPHA5,XIMAX5,RC5TAR
  real(chm_real) DFALPHA6,XIMAX6,GAMMA6
  real(chm_real) DFALPHA8,XIMAX8
  real(chm_real) RMIN,RMAX,JDELTA
  real(chm_real) DFALPHA9,XIMAX9
  real(chm_real) DFALPHA10,XIMAX10

  real(chm_real),allocatable,dimension(:) :: RC1BUF
  real(chm_real),allocatable,dimension(:) :: RC2BUF
  real(chm_real),allocatable,dimension(:) :: PHIBUF
  real(chm_real),allocatable,dimension(:) :: HXBUF
  real(chm_real),allocatable,dimension(:) :: RC5BUF
  real(chm_real),allocatable,dimension(:) :: NOEBUF
  real(chm_real),allocatable,dimension(:) :: RDCBUF
  real(chm_real),allocatable,dimension(:) :: S2XBUF, S2YBUF, S2ZBUF
  real(chm_real),allocatable,dimension(:) :: J3BUF
  real(chm_real),allocatable,dimension(:) :: PSIBUF
  !------------------------------------------------------
  !     INTEGER DECLARATIONS
  INTEGER EXCL,HQSTEP
  INTEGER,PARAMETER :: MAXSEL = 1024
  INTEGER,PARAMETER :: MAXSLL = 8*MAXSEL
  INTEGER,PARAMETER :: SIZEIJ = MAXSEL*(MAXSEL-1)/2
  !     MAXSEL must be equal or larger to the number of degrees of freedom used
  !     in the definition of the reaction coordinate in HQBM
  INTEGER,allocatable,dimension(:) :: EMSLCT
  integer :: NSEL1,ISEL1(MAXSEL),IUNJUJ1,IUNKUK1,IUNJLS1,IUNRF1,IUNREF,RC1BL
  INTEGER NSEL2,ISEL2(MAXSEL),IUNJUJ2,IUNKUK2,IUNJLS2,IUNRF2,RC2BL
  INTEGER NSEL3,ISEL3(MAXSEL),IUNJUJ3,IUNJUJ3A,IUNKUK3,IUNJLS3,IUNDMP3, &
       IUNPHI,PHIBL
  INTEGER NSEL4,ISEL4(MAXSLL),IUNJUJ4,IUNKUK4,IUNDMP4,IUNPRO, &
       ISELO(MAXSEL),ISELN(MAXSEL),ISELH(MAXSEL),NSELO,NSELN,NSELH,HXBL
  INTEGER NSEL5,ISEL5(MAXSEL),IUNJUJ5,IUNJLS5,RC5BL
  INTEGER IUNJUJ6,IUNNOE,IUNDMP6,NOEBL
  INTEGER IUNRDC,RDCBL
  INTEGER IUNJUJ8,IUNDMP8,IUNS2,S2BL
  INTEGER IUNJUJ9,IUNDMP9,IUNKUK9,IUNJ3,J3BL
  INTEGER IUNJUJ10,IUNJLS10,IUNKUK10,IUNDMP10,PSIBL,IUNPSI, &
       NSEL10,ISEL10(MAXSEL)
  INTEGER,PARAMETER :: MAXPSI = 10
  !------------------------------------------------------
  !     LOGICAL DECLARATIONS
  LOGICAL JRC1,JAWAY1,JRLD1,JSMD1,JRF1,JRF2,JHQFIX1,JNOEN1
  LOGICAL JRC2,JAWAY2,JRLD2,JSMD2,JHQFIX2,JNOEN2
  LOGICAL JRC3,JAWAY3,JRLD3,JSMD3,JRZERO3,JHQFIX3,JCOMB3,JAVPHI, &
       JNOEN3
  LOGICAL JRC4,JAWAY4,JSMD4,JNONN,JEEF1,JNHCON,JRZERO4,JHQFIX4, &
       JSPLIT,JNOEN4
  LOGICAL JRC5,JRLD5,JNOEN5
  LOGICAL JRC6,JAWAY6,JSMD6,JSIXP,JLIN,JRZERO6,JHQFIX6,JNOEN6
  LOGICAL JRC7,JNOEN7
  LOGICAL JRC8,JHQFIX8,JRZERO8
  LOGICAL JRC9,JNOEN9,JRZERO9
  LOGICAL JRC10,JHQFIX10,JRZERO10,JRLD10

  ! arrays formerly declared in RCn routines, dimension(sizeij), save
  integer, allocatable, dimension(:) :: IL1, JL1, IL2, JL2, IL3, JL3, &
       IL4, JL4, IL5, JL5, IL6, JL6
  real(chm_real), allocatable, dimension(:) :: RIJ1, RCIJ1, RIJ2, RCIJ2, &
       RIJ5, RCIJ5, RIJ6, LBOUN6, UBOUN6

contains
  subroutine allocate_hqbm(natom)
    use memory
    integer natom
    character(len=*),parameter :: routine_name="allocate_hqbm"

    if(allocated(emslct) .and. size(emslct) >= natom) then
       return
    elseif(allocated(emslct)) then
       call chmdealloc(FILE,routine_name,'emslct ', &
            size(emslct),intg=emslct)
    endif
    call chmalloc(FILE,routine_name,'emslct ',natom,intg=emslct)
!    call chmalloc(FILE,routine_name,'atsy ',maxa,crl=atsy)
!    call chmalloc(FILE,routine_name,'atsz ',maxa,crl=atsz)

    return
  end subroutine allocate_hqbm
  subroutine deallocate_hqbm
    use memory
    character(len=*),parameter :: routine_name="allocate_hqbm"

    call chmdealloc(FILE,routine_name,'emslct ',size(emslct),intg=emslct)
!    call chmalloc(FILE,routine_name,'atsy ',maxa,crl=atsy)
!    call chmalloc(FILE,routine_name,'atsz ',maxa,crl=atsz)

    return
  end subroutine deallocate_hqbm


  !   *************** MULTIPLE REACTION COORDINATE VERSION ***************
  !
  ! For reference:
  !       RC1: Radius of gyration/distance restraints
  !       RC2: Variation on RC1
  !       RC3/PHI: Phi-value constraints
  !       RC4/HX: Hydrogen exchange constraints
  !       RC5: As for RC1, but drive system towards target value and hold
  !              it there
  !       RC6/NOE: NOE-based constraints
  !       RC7/RDC: Residual dipolar couplings
  !       RC8/S2: Order parameters
  !       RC9/J3: Scalar couplings
  !       RC10/PSI: Psi-value constraints?
  !       RC11/RMS: Constrain RMSD of ensemble
  !
  ! TODO:
  !       use separate beta and tol values for phi and hx
  !       implement psi values
  !       implement rdc's
  !
  !----------------------------------------------------------------------
  SUBROUTINE HQINIT
    !----------------------------------------------------------------------
    !     initialize rxn crds when charmm starts
    use energym
    implicit none
    HQBM = .false.
    JRC1 = .false.
    JRC2 = .false.
    JRC3 = .false.
    JRC4 = .false.
    JRC5 = .false.
    JRC6 = .false.
    JRC7 = .false.
    JRC8 = .false.
    JRC9 = .false.
    JRC10 = .false.
    HQSTEP = 1
  END SUBROUTINE HQINIT
  !
  !----------------------------------------------------------------------
  SUBROUTINE HQFIN
    !----------------------------------------------------------------------
    !         clean memory, for what it's worth,
    !         and turn off all hqbm constraints
    use memory
    use energym
    implicit none
    character(len=*), parameter :: PROC = 'HQFIN'
    IF(ALLOCATED(EMSLCT)) &
          call chmdealloc(FILE,PROC,'emslct ',size(emslct),intg=emslct)
    IF (JRC1) THEN
       if (allocated(RC1BUF)) call chmdealloc(FILE,PROC,'RC1BUF',RC1BL,crl=RC1BUF)
       call chmdealloc(FILE,PROC,'IL1',size(IL1),intg=IL1)
       call chmdealloc(FILE,PROC,'JL1',size(JL1),intg=JL1)
       call chmdealloc(FILE,PROC,'RIJ1',size(RIJ1),crl=RIJ1)
       call chmdealloc(FILE,PROC,'RCIJ1',size(RCIJ1),crl=RCIJ1)
    ENDIF
    IF (JRC2) THEN
       if (allocated(RC2BUF)) call chmdealloc(FILE,PROC,'RC2BUF',RC2BL,crl=RC2BUF)
       call chmdealloc(FILE,PROC,'IL2',size(IL2),intg=IL2)
       call chmdealloc(FILE,PROC,'JL2',size(JL2),intg=JL2)
       call chmdealloc(FILE,PROC,'RIJ2',size(RIJ2),crl=RIJ2)
       call chmdealloc(FILE,PROC,'RCIJ2',size(RCIJ2),crl=RCIJ2)
    ENDIF
    IF (JRC3) THEN
       if (allocated(PHIBUF)) call chmdealloc(FILE,PROC,'PHIBUF',PHIBL,crl=PHIBUF)
       call chmdealloc(FILE,PROC,'IL3',size(IL3),intg=IL3)
       call chmdealloc(FILE,PROC,'JL3',size(JL3),intg=JL3)
    ENDIF
    IF (JRC4) THEN
       if (allocated(HXBUF)) call chmdealloc(FILE,PROC,'HXBUF',HXBL,crl=HXBUF)
       call chmdealloc(FILE,PROC,'IL4',size(IL4),intg=IL4)
       call chmdealloc(FILE,PROC,'JL4',size(JL4),intg=JL4)
    ENDIF
    IF (JRC5) THEN
       if (allocated(RC5BUF)) call chmdealloc(FILE,PROC,'RC5BUF',RC5BL,crl=RC5BUF)
       call chmdealloc(FILE,PROC,'IL5',size(IL5),intg=IL5)
       call chmdealloc(FILE,PROC,'JL5',size(JL5),intg=JL5)
       call chmdealloc(FILE,PROC,'RIJ5',size(RIJ5),crl=RIJ5)
       call chmdealloc(FILE,PROC,'RCIJ5',size(RCIJ5),crl=RCIJ5)
    ENDIF
    IF (JRC6) THEN
       if (allocated(NOEBUF)) call chmdealloc(FILE,PROC,'NOEBUF',NOEBL,crl=NOEBUF)
       call chmdealloc(FILE,PROC,'IL6',size(IL6),intg=IL6)
       call chmdealloc(FILE,PROC,'JL6',size(JL6),intg=JL6)
       call chmdealloc(FILE,PROC,'RIJ6',size(RIJ6),crl=RIJ6)
       call chmdealloc(FILE,PROC,'LBOUN6',size(LBOUN6),crl=LBOUN6)
       call chmdealloc(FILE,PROC,'UBOUN6',size(UBOUN6),crl=UBOUN6)
    ENDIF
    IF (allocated(RDCBUF)) THEN
       call chmdealloc(FILE,PROC,'RDCBUF',RDCBL,crl=RDCBUF)
    ENDIF
    IF (allocated(S2XBUF)) THEN
       call chmdealloc(FILE,PROC,'S2XBUF',S2BL,crl=S2XBUF)
       call chmdealloc(FILE,PROC,'S2YBUF',S2BL,crl=S2YBUF)
       call chmdealloc(FILE,PROC,'S2ZBUF',S2BL,crl=S2ZBUF)
    ENDIF
    IF (allocated(J3BUF)) THEN
       call chmdealloc(FILE,PROC,'J3BUF',J3BL,crl=J3BUF)
    ENDIF
    IF (allocated(PSIBUF)) THEN
       call chmdealloc(FILE,PROC,'PSIBUF',PSIBL,crl=PSIBUF)
    ENDIF
    JRC1 = .false.
    JRC2 = .false.
    JRC3 = .false.
    JRC4 = .false.
    JRC5 = .false.
    JRC6 = .false.
    JRC7 = .false.
    JRC8 = .false.
    JRC9 = .false.
    JRC10 = .false.
    ! TODO: add statements to reset all hqbm parms to defaults
    HQSTEP = 1
  END SUBROUTINE HQFIN
  !
  !----------------------------------------------------------------------
  SUBROUTINE HQBMINI
    !----------------------------------------------------------------------
    !     Author: Emanuele Paci <02-29-00>
    !     This routine is invoked at the beginning of each CHARMM run.
    !     if the option HQBM is called
    !
    use comand
    use psf
    use coord
    use deriv
    use memory
    use stream
    use string

    implicit none
    real(chm_real) EU
    !     read parameters for each type of rc, and init bias routines
    !
    !     no initial call to energy required, each bias initializes
    !     itself separately
    !
    !     added 'friendly' names for biases
    !
    if(.not.allocated(emslct)) call allocate_hqbm(natom)
    if(allocated(emslct) .and. size(emslct)/=natom) &
         call allocate_hqbm(natom)
    IF (INDXA(COMLYN, COMLEN, 'RESET') .GT. 0) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
            ' HQBMINI> RESETTING ALL HQBM BIASES!'
       CALL HQFIN
       if(allocated(emslct)) call deallocate_hqbm
       RETURN
    ENDIF
    IF (INDXA(COMLYN, COMLEN, 'UPAL') .GT. 0) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
            ' HQBMINI> UPDATING ALPHA VALUES!'
       CALL UPALPH
       RETURN
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
         ' HQBMINI> MULTI-DIMENSIONAL HQBM INITIALIZATION'
    IF (INDXA(COMLYN, COMLEN, 'RC1') .GT. 0) THEN
       CALL RC1INI
       CALL RC1BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF (INDXA(COMLYN, COMLEN, 'RC2') .GT. 0) THEN
       CALL RC2INI
       CALL RC2BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC3') .GT. 0) .OR.  &
         (INDXA(COMLYN, COMLEN, 'PHI') .GT. 0)) THEN
       CALL RC3INI
       CALL RC3BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC4') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'HX') .GT. 0)) THEN
       CALL RC4INI
       CALL RC4BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF (INDXA(COMLYN, COMLEN, 'RC5') .GT. 0) THEN
       CALL RC5INI
       CALL RC5BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC6') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'NOE') .GT. 0)) THEN
       CALL RC6INI
       CALL RC6BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC7') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'RDC') .GT. 0)) THEN
       CALL RC7INI
       CALL RC7BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
#if KEY_ENSEMBLE==1
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC8') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'S2') .GT. 0)) THEN
       CALL RC8INI
       CALL RC8BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
#endif 
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC9') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'J3') .GT. 0)) THEN
       CALL RC9INI
       CALL RC9BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
#if KEY_ENSEMBLE==1
    ELSE IF ((INDXA(COMLYN, COMLEN, 'RC10') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN, 'PSI') .GT. 0)) THEN
       CALL RC10INI
       CALL RC10BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
#endif 
    ELSE
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
            ' HQBMINI> Reaction Coordinate 1 taken by default'
       CALL RC1INI
       CALL RC1BIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.true.,.true.)
    END IF
    IF (IOLEV.GT.0) WRITE(OUTU,*) &
         'reaction coordinates initialised so far, 1/2/3/4/5/6/7/8/9 ?', &
         jrc1,jrc2,jrc3,jrc4,jrc5,jrc6,jrc7,jrc8,jrc9
    IF (INDXA(COMLYN, COMLEN, 'ANAL') .GT. 0) CALL HQANAL
    RETURN
  END SUBROUTINE HQBMINI
  !
  !----------------------------------------------------------------------
  SUBROUTINE UPALPH
    !     update alpha values without doing a full 'hqbm reset'
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    A=GTRMF(COMLYN,COMLEN,'RC1',MINONE)
    IF(A.GE.0.0) DFALPHA1=A
    A=GTRMF(COMLYN,COMLEN,'RC2',MINONE)
    IF(A.GE.0.0) DFALPHA2=A
    A=GTRMF(COMLYN,COMLEN,'RC3',MINONE)
    IF(A.GE.0.0) DFALPHA3=A
    A=GTRMF(COMLYN,COMLEN,'PHI',MINONE)
    IF(A.GE.0.0) DFALPHA3=A
    A=GTRMF(COMLYN,COMLEN,'RC4',MINONE)
    IF(A.GE.0.0) DFALPHA4=A
    A=GTRMF(COMLYN,COMLEN,'HX',MINONE)
    IF(A.GE.0.0) DFALPHA4=A
    A=GTRMF(COMLYN,COMLEN,'RC5',MINONE)
    IF(A.GE.0.0) DFALPHA5=A
    A=GTRMF(COMLYN,COMLEN,'RC6',MINONE)
    IF(A.GE.0.0) DFALPHA6=A
    !      A=GTRMF(COMLYN,COMLEN,'RC7',-1.0)
    !      IF(A.GE.0.0) DFALPHA7=A
    A=GTRMF(COMLYN,COMLEN,'RC8',MINONE)
    IF(A.GE.0.0) DFALPHA8=A
    A=GTRMF(COMLYN,COMLEN,'RC9',MINONE)
    IF(A.GE.0.0) DFALPHA9=A
    A=GTRMF(COMLYN,COMLEN,'J3',MINONE)
    IF(A.GE.0.0) DFALPHA9=A
    RETURN
  END SUBROUTINE UPALPH
  !----------------------------------------------------------------------
  SUBROUTINE RC1INI
    !     initialize RC1 from CHARMM command line
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I

    IF (JRC1) THEN
       CALL WRNDIE(-1,'<RC1INI>', &
            'Error: RC1 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC1=.TRUE.
    HQBM = .TRUE.
    JAWAY1 =.FALSE.
    JSMD1 =.FALSE.
    JHQFIX1 =.FALSE.
    JNOEN1=.FALSE.
    IUNJLS1=-1
    IUNJUJ1=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    !      IUNKUK1=GTRMI(COMLYN,COMLEN,'IUNK',97)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'AWAY') .GT. 0) JAWAY1 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) JSMD1 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX1 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN1 =.TRUE.
    IF (JSMD1) THEN
       !        GAMMA is the SMD pulling speed
       A=GTRMF(COMLYN,COMLEN,'GAMMA',FMARK)
       GAMMA1=A
       IF (PRNLEV.GE.2) WRITE (outu, '(A,F8.3)')  &
            ' HQBM> GAMMA1 = ', GAMMA1
    END IF
    IF (JHQFIX1.AND.JSMD1) THEN
       CALL WRNDIE(-1,'<RC1INI>', &
            'Error: Cannot use FIX and SMD together!!!')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA1=A
    ELSE
       DFALPHA1=0.0D0
    ENDIF
    !         to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    IUNJLS1=GTRMI(COMLYN,COMLEN,'READLIST',-1)
    IUNRF1=GTRMI(COMLYN,COMLEN,'READREF',-1)
    IF ((IUNJLS1.GE.0).AND.(IUNRF1.GE.0)) THEN
       CALL WRNDIE(-1,'<RC1INI>', &
            'Error: Cannot use READLIST and READREF together!!!')
    ENDIF

    IF (IUNJLS1 .NE. -1) THEN
       JRLD1=.TRUE.
    ELSE IF (IUNRF1 .NE. -1) THEN
       JRF1=.TRUE.
    ELSE
       NSEL1 = 0
       DO I=1,NATOM
          IF (EMSLCT(I) .EQ. 1) THEN
             NSEL1 = NSEL1 + 1
             ISEL1(NSEL1) = I
          END IF
       END DO
    END IF
    !        If the pairs involved in the reaction coordinate are not
    !        read from the list, check that the number of pairs in not
    !        too large
    IF (NSEL1 .GT. MAXSEL) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))') &
            ' RC1INI> MAXSEL=',MAXSEL, &
            ' IS LOWER THAN NSEL1=',NSEL1
       CALL WRNDIE(-1,'<RC1INI>', &
            'CHANGE hqbm.f90 AND RECOMPILE')
    END IF
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX1=A
    ELSE
       XIMAX1=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC1INI> nsel1',nsel1
    RETURN
  END SUBROUTINE RC1INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC2INI
    !     initialize RC2 from CHARMM command line
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC2) THEN
       CALL WRNDIE(-1,'<RC2INI>', &
            'Error: RC2 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC2 = .TRUE.
    HQBM = .TRUE.
    JAWAY2 =.FALSE.
    JSMD2 =.FALSE.
    JHQFIX2 =.FALSE.
    JNOEN2=.FALSE.
    IUNJLS2=-1
    IUNJUJ2=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNKUK2=GTRMI(COMLYN,COMLEN,'IUNK',97)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'AWAY') .GT. 0) JAWAY2 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) JSMD2 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX2 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN2 =.TRUE.
    IF (JSMD2) THEN
       !        GAMMA is the SMD pulling speed
       A=GTRMF(COMLYN,COMLEN,'GAMMA',FMARK)
       GAMMA2=A
       IF (PRNLEV.GE.2) WRITE (outu, '(A,F8.3)')  &
            ' HQBM> GAMMA2 = ', GAMMA2
    END IF
    IF (JHQFIX2.AND.JSMD2) THEN
       CALL WRNDIE(-1,'<RC2INI>', &
            'Error: Cannot use FIX and SMD together!!!')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA2=A
    ELSE
       DFALPHA2=0.0D0
    ENDIF
    !         to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    IUNJLS2=GTRMI(COMLYN,COMLEN,'READLIST',-1)
    IF (IUNJLS2 .NE. -1) THEN
       JRLD2=.TRUE.
    ELSE
       NSEL2 = 0
       DO I=1,NATOM
          IF (EMSLCT(I) .EQ. 1) THEN
             NSEL2 = NSEL2 + 1
             ISEL2(NSEL2) = I
          END IF
       END DO
    END IF
    IF (NSEL2 .GT. MAXSEL) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))') &
            ' RC2INI> MAXSEL=',MAXSEL, &
            ' IS LOWER THAN NSEL2=',NSEL2
       CALL WRNDIE(-1,'<RC2INI>', &
            'CHANGE hqbm.f90 AND RECOMPILE')
    END IF
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX2=A
    ELSE
       XIMAX2=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC2INI> nsel2',nsel2
    RETURN
  END SUBROUTINE RC2INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC3INI
    !     initialize RC3 from CHARMM command line
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC3) THEN
       CALL WRNDIE(-1,'<RC3INI>', &
            'Error: RC3 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC3 = .TRUE.
    HQBM = .TRUE.
    JAWAY3 =.FALSE.
    JSMD3 =.FALSE.
    JHQFIX3 =.FALSE.
    JRZERO3=.FALSE.
    JNOEN3=.FALSE.
    JAVPHI=.FALSE.
    IUNJLS3=-1
    IUNJUJ3=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNKUK3=GTRMI(COMLYN,COMLEN,'IUNK',97)
    IUNJUJ3A=GTRMI(COMLYN,COMLEN,'AIUNJ',98)
    JCOMB3=.FALSE.
    !HX   unit for dumping detailed output (e.g. calc. phi vals)
    IUNDMP3=GTRMI(COMLYN,COMLEN,'IUND',-1)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'AWAY') .GT. 0) JAWAY3 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) JSMD3 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX3 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN3 =.TRUE.
    !     JCOMB3 - use all-against-all combinations to make native
    !     contact list (e.g. for T2 bias)
    IF (INDXA(COMLYN, COMLEN, 'COMB') .GT. 0) JCOMB3 =.TRUE.
#if KEY_ENSEMBLE==1
    IF (INDXA(COMLYN, COMLEN, 'AVEP') .GT. 0) JAVPHI =.TRUE.
#endif 
    IF (JSMD3) THEN
       !        GAMMA is the SMD pulling speed
       A=GTRMF(COMLYN,COMLEN,'GAMMA',FMARK)
       GAMMA3=A
       IF (PRNLEV.GE.2) WRITE (OUTU, '(A,F8.3)')  &
            ' HQBM> GAMMA3 = ', GAMMA3
       A=GTRMF(COMLYN,COMLEN,'AGAMMA',FMARK)
       AGAMMA3=A
       IF (PRNLEV.GE.2) WRITE (OUTU, '(A,F8.3)')  &
            ' HQBM> AGAMMA3 = ', AGAMMA3
    END IF
    IF (JHQFIX3.AND.JSMD3) THEN
       CALL WRNDIE(-1,'<RC3INI>', &
            'Error: Cannot use FIX and SMD together!!!')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA3=A
    ELSE
       DFALPHA3=0.0D0
    ENDIF
#if KEY_ENSEMBLE==1
    A=GTRMF(COMLYN,COMLEN,'AALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFAALPHA3=A
    ELSE
       DFAALPHA3=0.0D0
    ENDIF
#endif 
    !         to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)

    IUNJLS3=GTRMI(COMLYN,COMLEN,'READLIST',-1)

    IF (IUNJLS3 .NE. -1) THEN
       JRLD3=.TRUE.
    ELSE
       NSEL3 = 0
       DO I=1,NATOM
          IF (EMSLCT(I) .EQ. 1) THEN
             NSEL3 = NSEL3 + 1
             ISEL3(NSEL3) = I
          END IF
       END DO
    END IF
    !        If the pairs involved in the reaction coordinate are not
    !        read from the list, check that the number of pairs in not
    !        too large
    IF (NSEL3 .GT. MAXSEL) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))') &
            ' RC3INI> MAXSEL=',MAXSEL, &
            ' IS LOWER THAN NSEL3=',NSEL3
       CALL WRNDIE(-1,'<RC3INI>', &
            'CHANGE hqbm.f90 AND RECOMPILE')
    END IF
    !     what is iunref for ???
    IUNREF=GTRMI(COMLYN,COMLEN,'IUNR',94)
    IUNPHI=GTRMI(COMLYN,COMLEN,'IUNP',95)
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO3 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'BETA',FMARK)
    IF(A.GE.0.0) THEN
       BETA=A
    ELSE
       BETA=0.0D0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'EXCL',FMARK)
    IF(A.GE.0.0) THEN
       EXCL=A
    ELSE
       EXCL=4
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'RCUT',FMARK)
    IF(A.GE.0.0) THEN
       RCUT=A
    ELSE
       RCUT=8.5D0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'TOL',FMARK)
    IF(A.GE.0.0) THEN
       TOL=A
    ELSE
       TOL=0.0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
         ' RC 3 corresponds to the phi cost function'
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX3=A
    ELSE
       XIMAX3=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC3INI> nsel3',nsel3
    RETURN
  END SUBROUTINE RC3INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC4INI
    !     initialize RC4 from CHARMM command line
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC4) THEN
       CALL WRNDIE(-1,'<RC4INI>', &
            'Error: RC4 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC4 = .TRUE.
    HQBM = .TRUE.
    JAWAY4 =.FALSE.
    JSMD4 =.FALSE.
    JEEF1 =.FALSE.
    JNHCON =.FALSE.
    JHQFIX4 =.FALSE.
    JRZERO4=.FALSE.
    JSPLIT = .FALSE.
    JNONN = .FALSE.
    JNOEN4=.FALSE.
    IUNJUJ4=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNKUK4=GTRMI(COMLYN,COMLEN,'IUNK',97)
    !HX   unit for dumping detailed output (e.g. calc. phi vals)
    IUNDMP4=GTRMI(COMLYN,COMLEN,'IUND',-1)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'AWAY') .GT. 0) JAWAY4 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) JSMD4 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX4 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN4 =.TRUE.
    !HX   Use eef1 energy for burial term
    IF (INDXA(COMLYN, COMLEN, 'EEF1') .GT. 0) JEEF1 =.TRUE.
    !HX   Use (amide N) --- (heavy atom) based contacts
    IF (INDXA(COMLYN, COMLEN, 'NHCON') .GT. 0) JNHCON =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SPLIT') .GT. 0) JSPLIT =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NONN') .GT. 0) JNONN =.TRUE.
    IF (JEEF1.AND.JNHCON) THEN
       CALL WRNDIE(-1,'<RC4INI>', &
            'Error: cannot use both EEF1 and NHCON')
    ENDIF
    IF (JSMD4) THEN
       !        GAMMA is the SMD pulling speed
       A=GTRMF(COMLYN,COMLEN,'GAMMA',FMARK)
       GAMMA4=A
       IF (PRNLEV.GE.2) WRITE (outu, '(A,F8.3)')  &
            ' HQBM> GAMMA4 = ', GAMMA4
    END IF
    IF (JHQFIX4.AND.JSMD4) THEN
       CALL WRNDIE(-1,'<RC4INI>', &
            'Error: Cannot use FIX and SMD together!!!')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA4=A
    ELSE
       DFALPHA4=0.0D0
    ENDIF
    !     to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    NSEL4 = 0
    !     No iunjls for rc4 !
    DO I=1,NATOM
       IF (EMSLCT(I) .EQ. 1) THEN
          NSEL4 = NSEL4 + 1
          ISEL4(NSEL4) = I
       END IF
    END DO
    !     ... we need another three atom selections for rc4
    !       Only O...H distance considered at the moment. Nitrogen
    !       selection is only used when EEF1-based definition is
    !       in operation.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    NSELO = 0
    DO I=1,NATOM
       IF (EMSLCT(I) .EQ. 1) THEN
          IF (PRNLEV.GE.2) WRITE (OUTU, *) 'O atom selected: ', I
          NSELO = NSELO + 1
          ISELO(NSELO) = I
       END IF
    END DO
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    NSELN = 0
    DO I=1,NATOM
       IF (EMSLCT(I) .EQ. 1) THEN
          IF (PRNLEV.GE.2) WRITE (OUTU, *) 'N atom selected: ', I
          NSELN = NSELN + 1
          ISELN(NSELN) = I
       END IF
    END DO
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
    NSELH = 0
    DO I=1,NATOM
       IF (EMSLCT(I) .EQ. 1) THEN
          IF (PRNLEV.GE.2) WRITE (OUTU, *) 'H atom selected: ', I
          NSELH = NSELH + 1
          ISELH(NSELH) = I
       END IF
    END DO
    !        If the pairs involved in the reaction coordinate are not
    !        read from the list, check that the number of pairs in not
    !        too large
    IF (NSEL4 .GT. MAXSLL) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))') &
            ' RC4INI> MAXSEL=',MAXSEL, &
            ' IS LOWER THAN NSEL4=',NSEL4
       CALL WRNDIE(-1,'<RC4INI>', &
            'CHANGE hqbm.f90 AND RECOMPILE')
    END IF
    IUNPRO=GTRMI(COMLYN,COMLEN,'IUNP',95)
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO4 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'BETA',FMARK)
    IF(A.GE.0.0) THEN
       BETA=A
    ELSE
       BETA=0.0D0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'EXCL',FMARK)
    IF(A.GE.0.0) THEN
       EXCL=A
    ELSE
       EXCL=4
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'RCUT',FMARK)
    IF(A.GE.0.0) THEN
       RCUT=A
    ELSE
       RCUT=8.5D0
    ENDIF
    IF (JNONN) THEN
       HXCTON=GTRMF(COMLYN,COMLEN,'CUTON',EIGHT)
       HXCTOF=GTRMF(COMLYN,COMLEN,'CUTOF',TEN)
       IF (HXCTON.GE.HXCTOF) CALL WRNDIE(-1,'<RC4INI>', &
            'Error: CUTON must be less than CUTOF')
       IF (HXCTON.LE.RCUT) CALL WRNDIE(-1,'<RC4INI>', &
            'Error: CUTON must be greater than RCUT')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'TOL',FMARK)
    IF(A.GE.0.0) THEN
       TOL=A
    ELSE
       TOL=A
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
         ' RC 4 corresponds to the HX protection cost function'
    !HX   HCUT is the hydrogen-oxygen distance.
    A=GTRMF(COMLYN,COMLEN,'HCUT',FMARK)
    IF(A.GE.0.0) THEN
       HCUT=A
    ELSE
       HCUT=2.4
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'BETH',FMARK)
    IF(A.GE.0.0) THEN
       BETH=A
    ELSE
       BETH=BETA
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'HXFC',FMARK)
    IF(A.GE.0.0) THEN
       HXFC=A
    ELSE IF (JEEF1) THEN
       !        default for eef1 energy (as yet unknown!)
       HXFC=1.0
    ELSE
       !        default for all-atom side-chain contacts
       HXFC=0.1
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'HXFH',FMARK)
    IF(A.GE.0.0) THEN
       HXFH=A
    ELSE
       !        default for hydrogen bonds with 2.4 A cut-off
       HXFH=5.0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX4=A
    ELSE
       XIMAX4=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC4INI> nsel4',nsel4

    RETURN
  END SUBROUTINE RC4INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC5INI
    !     initialize RC5 from CHARMM command line
    !     similar to RC1, except drives towards target value and holds
    !     system there; no smd yet
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC5) THEN
       CALL WRNDIE(-1,'<RC5INI>', &
            'Error: RC5 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC5=.TRUE.
    HQBM = .TRUE.
    JNOEN5 = .FALSE.
    IUNJLS5=-1
    RC5TAR = 0.0
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN5 =.TRUE.
    IUNJUJ5=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA5=A
    ELSE
       DFALPHA5=0.0D0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'TARGET',FMARK)
    IF(A.GE.0.0) THEN
       RC5TAR=A
    ELSE
       RC5TAR=0.0D0
    ENDIF
    !         to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)

    IUNJLS5=GTRMI(COMLYN,COMLEN,'READLIST',-1)

    IF (IUNJLS5 .NE. -1) THEN
       JRLD5=.TRUE.
    ELSE
       NSEL5 = 0
       DO I=1,NATOM
          IF (EMSLCT(I) .EQ. 1) THEN
             NSEL5 = NSEL5 + 1
             ISEL5(NSEL5) = I
          END IF
       END DO
    END IF
    !        If the pairs involved in the reaction coordinate are not
    !        read from the list, check that the number of pairs in not
    !        too large
    IF (NSEL5 .GT. MAXSEL) THEN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))') &
            ' RC5INI> MAXSEL=',MAXSEL, &
            ' IS LOWER THAN NSEL5=',NSEL5
       CALL WRNDIE(-1,'<RC5INI>', &
            'CHANGE hqbm.f90 AND RECOMPILE')
    END IF
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX5=A
    ELSE
       XIMAX5=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC5INI> nsel5',nsel5
    RETURN
  END SUBROUTINE RC5INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC6INI
    !     initialize RC6 from CHARMM command line
    !     NOE'S
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use select
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC6) THEN
       CALL WRNDIE(-1,'<RC6INI>', &
            'Error: RC6 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC6=.TRUE.
    HQBM = .TRUE.
    JAWAY6 =.FALSE.
    JSMD6 =.FALSE.
    JHQFIX6 =.FALSE.
    JRZERO6=.FALSE.
    JSIXP = .FALSE.
    JLIN = .FALSE.
    JNOEN6=.FALSE.
    IUNJUJ6=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNDMP6=GTRMI(COMLYN,COMLEN,'IUND',-1)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'AWAY') .GT. 0) JAWAY6 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) JSMD6 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX6 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO6 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SIXT') .GT. 0) JSIXP =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'LINE') .GT. 0) JLIN =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN6 =.TRUE.
    IF (JSIXP.AND.JLIN) THEN
       IF (IOLEV.GT.0) WRITE(OUTU,'(A)') &
            '!! Can only use one of linear, third or sixth power averaging'
       CALL WRNDIE(-1,'<RC6INI>', &
            'Error: Cannot use SIXT and LINE together!!!')
    ENDIF
    IF (JSMD6) THEN
       !        GAMMA is the SMD speed
       A=GTRMF(COMLYN,COMLEN,'GAMMA',FMARK)
       GAMMA6=A
       IF (PRNLEV.GE.2) WRITE (outu, '(A,F8.3)')  &
            ' HQBM> GAMMA6 = ', GAMMA6
    END IF
    IF (JHQFIX6.AND.JSMD6) THEN
       CALL WRNDIE(-1,'<RC6INI>', &
            'Error: Cannot use FIX and SMD together!!!')
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA6=A
    ELSE
       DFALPHA6=0.0D0
    ENDIF
    !         to select atoms.
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)

    IUNNOE=GTRMI(COMLYN,COMLEN,'IUNN',-1)
    IF (IUNNOE.LE.0) THEN
       CALL WRNDIE(-1,'<RC6INI>', &
            'Error: Must specify NOEs in unit IUNNOE!!!')
    ENDIF

    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX6=A
    ELSE
       XIMAX6=99999999.9D0
    ENDIF
    RETURN
  END SUBROUTINE RC6INI
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC7INI
    !     initialize RC7 from CHARMM command line
    !     Initialize RDC's
    !----------------------------------------------------------------------
    !     To be written!
  END SUBROUTINE RC7INI
  !----------------------------------------------------------------------
#if KEY_ENSEMBLE==1
  SUBROUTINE RC8INI
    !     initialize RC8 from CHARMM command line
    !     ORDER PARAMETERS (S2)
    !----------------------------------------------------------------------
    use energym
    use string
    use comand
    use coord
    use deriv
    use number
    use psf
    use stream
    use ensemble
    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC8) THEN
       CALL WRNDIE(-1,'<RC8INI>', &
            'Error: RC8 is already set up, use HQBM RESET first!!!')
    ENDIF
    IF (NENSEM.LT.2) THEN
       CALL WRNDIE(-1,'<RC8INI>', &
            'Error: RC8 (S2) makes no sense for less than two replicas!!!')
    ENDIF
    JRC8=.TRUE.
    HQBM = .TRUE.
    JHQFIX8 =.FALSE.
    JRZERO8 =.FALSE.
    IUNJUJ8=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNDMP8=GTRMI(COMLYN,COMLEN,'IUND',-1)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO8 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX8 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA8=A
    ELSE
       DFALPHA8=0.0D0
    ENDIF

    IUNS2=GTRMI(COMLYN,COMLEN,'IUNS',-1)
    IF (IUNS2.LE.0) THEN
       CALL WRNDIE(-1,'<RC8INI>', &
            'Error: Must specify order parameters in unit IUNS2!!!')
    ENDIF

    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX8=A
    ELSE
       XIMAX8=99999999.9D0
    ENDIF
    RETURN
  END SUBROUTINE RC8INI
#endif 
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC9INI
    !     initialize RC9 from CHARMM command line
    !     RC9/J3    Three-bond scalar coupling
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use deriv
    use number
    use psf
    use stream
    use string

    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC9) THEN
       CALL WRNDIE(-1,'<RC9INI>', &
            'Error: RC9 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC9=.TRUE.
    HQBM = .TRUE.
    JNOEN9=.FALSE.
    IUNJUJ9=GTRMI(COMLYN,COMLEN,'IUNJ',-1)
    IF (IUNJUJ9.LT.0) CALL WRNDIE(-1,'<RC9INI>', &
         'Error: Must give IUNJ unit for RC9/J3')
    IUNDMP9=GTRMI(COMLYN,COMLEN,'IUND',-1)
    IUNKUK9=GTRMI(COMLYN,COMLEN,'IUNK',-1)
    IF (IUNKUK9.LT.0) CALL WRNDIE(-1,'<RC9INI>', &
         'Error: Must give IUNK unit for RC9/J3')
    IUNJ3=GTRMI(COMLYN,COMLEN,'JUNIT',-1)
    IF (IUNJ3.LT.0) CALL WRNDIE(-1,'<RC9INI>', &
         'Error: Need to specify unit JUNIT with scalar coupling data')
    IF (INDXA(COMLYN, COMLEN, 'NOEN') .GT. 0) JNOEN9 =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO9 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA9=A
    ELSE
       DFALPHA9=0.0D0
    ENDIF
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX9=A
    ELSE
       XIMAX9=99999999.9D0
    ENDIF
    RETURN
  END SUBROUTINE RC9INI

#if KEY_ENSEMBLE==1
  !----------------------------------------------------------------------
  SUBROUTINE RC10INI
    !     initialize RC10 from CHARMM command line
    !     PSI-value
    !----------------------------------------------------------------------
    use energym
    use string
    use comand
    use coord
    use deriv
    use number
    use psf
    use stream
    implicit none
    real(chm_real) A
    INTEGER I
    IF (JRC10) THEN
       CALL WRNDIE(-1,'<RC10INI>', &
            'Error: RC10 is already set up, use HQBM RESET first!!!')
    ENDIF
    JRC10 = .TRUE.
    HQBM = .TRUE.
    JHQFIX10 =.FALSE.
    JRZERO10=.FALSE.
    IUNJLS10=-1
    IUNJUJ10=GTRMI(COMLYN,COMLEN,'IUNJ',96)
    IUNKUK10=GTRMI(COMLYN,COMLEN,'IUNK',97)
    !HX   unit for dumping detailed output (e.g. calc. psi vals)
    IUNDMP10=GTRMI(COMLYN,COMLEN,'IUND',-1)
    !     Parameter AWAY is used to drive the system toward increasing
    !     values of the reaction coordinate
    IF (INDXA(COMLYN, COMLEN, 'FIX') .GT. 0) JHQFIX10 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA10=A
    ELSE
       DFALPHA10=0.0D0
    ENDIF
    IUNPSI=GTRMI(COMLYN,COMLEN,'IUNP',-1)
    !      IF (IUNJLS10 .NE. -1) THEN
    !         JRLD10=.TRUE.
    !      ELSE
    !         NSEL10 = 0
    !         DO I=1,MAXAIM
    !            IF (EMSLCT(I) .EQ. 1) THEN
    !               NSEL10 = NSEL10 + 1
    !               ISEL10(NSEL10) = I
    !            END IF
    !         END DO
    !      END IF
    !        If the pairs involved in the reaction coordinate are not
    !        read from the list, check that the number of pairs in not
    !        too large
    !      IF (NSEL10 .GT. MAXSEL) THEN
    !         IF (PRNLEV.GE.2) WRITE(OUTU,'(2(A,I5))')
    !     1        ' RC10INI> MAXSEL=',MAXSEL,
    !     2        ' IS LOWER THAN NSEL9=',NSEL10
    !         CALL WRNDIE(-1,'<RC10INI>',
    !     1        'CHANGE hqbm.f90 AND RECOMPILE')
    !      END IF
    !     what is iunref for ???
    !      IUNREF=GTRMI(COMLYN,COMLEN,'IUNR',94)
    IF (INDXA(COMLYN, COMLEN, 'ZERO') .GT. 0) JRZERO10 =.TRUE.
    A=GTRMF(COMLYN,COMLEN,'BETA',FMARK)
    IF(A.GE.0.0) THEN
       BETA=A
    ELSE
       BETA=0.0D0
    ENDIF
    !      A=GTRMF(COMLYN,COMLEN,'EXCL',FMARK)
    !      IF(A.GE.0.0) THEN
    !         EXCL=A
    !      ELSE
    !         EXCL=4
    !      ENDIF
    A=GTRMF(COMLYN,COMLEN,'RCUT',FMARK)
    !      IF(A.GE.0.0) THEN
    !         RCUT=A
    !      ELSE
    !         RCUT=8.5D0
    !      ENDIF
    A=GTRMF(COMLYN,COMLEN,'TOL',FMARK)
    IF(A.GE.0.0) THEN
       TOL=A
    ELSE
       TOL=0.0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
         ' RC 10 corresponds to the psi cost function'
    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX10=A
    ELSE
       XIMAX10=99999999.9D0
    ENDIF
    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')' RC10INI> nsel10',nsel10
    RETURN
  END SUBROUTINE RC10INI

#endif 
  !----------------------------------------------------------------------
  SUBROUTINE HQBIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE)
    !     master routine for bias energies -- dispatches jobs
    !     to routines for reaction coordinates on which biases
    !     have been placed
    !     EU: bias energy (output)
    !     X,Y,Z: coordinates (input)
    !     DX,DY,DZ: forces (output)
    !     NATOMX: number of atoms (input)
    !     QFORCE: logical (input) telling whether to evaluate forces
    !----------------------------------------------------------------------
    implicit none
    real(chm_real) EU,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    INTEGER NATOMX
    LOGICAL QFORCE
    !     local vbls from here on
    IF (JRC1) CALL RC1BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE, &
         .false.)
    IF (JRC2) CALL RC2BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE, &
         .false.)
    IF (JRC3) CALL RC3BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
    IF (JRC4) CALL RC4BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
    IF (JRC5) CALL RC5BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
    IF (JRC6) CALL RC6BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
    IF (JRC7) CALL RC7BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
#if KEY_ENSEMBLE==1
    IF (JRC8) CALL RC8BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
#endif 
    IF (JRC9) CALL RC9BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
#if KEY_ENSEMBLE==1
    IF (JRC10) CALL RC10BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX, &
         QFORCE,.false.)
#endif 
    HQSTEP = HQSTEP + 1
    RETURN
  END SUBROUTINE HQBIAS
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC1BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC1
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    !     Author: Emanuele Paci <07-02-97>
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    real(chm_real) FCN,AUXR,THETA,COSPHI
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES
    INTEGER I,J,L,L1,L2,K,II,JJ,LL,NSELL
    INTEGER IRES,JRES,LRES
    real(chm_real) XIJ,YIJ,ZIJ,R,RCUT1,TOTCON
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0, &
         XI,XIA,DXI,ALPHA,ALPHA2, &
         AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C,PSIM
    SAVE XI,XIA,RHO0,NSELL,RCUT1

    ALPHA=DFALPHA1
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC1BIAS> CALLING HQBM BIAS RC1'
          WRITE(OUTU,'(A,F19.8)')' RC1BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC1BIAS> THE RC1 PERTURBATION DRIVES THE SYSTEM'
          IF (JAWAY1) WRITE(OUTU,'(A)') &
               ' RC1BIAS>  AWAY FROM THE REFERENCE CONFIGURATION'
          IF (.NOT.JAWAY1) WRITE(OUTU,'(A)') &
               ' RC1BIAS>  THROUGH THE REFERENCE CONFIGURATION'
       END IF

       IF (.NOT.(JRLD1.OR.JRF1)) THEN
          IF (PRNLEV.GE.2) THEN
             WRITE(OUTU,'(A)') &
                  ' RC1BIAS> COORDINATES OF THE RC1 REFERENCE CONFIG:'
             DO II=1,NSEL1
                I = ISEL1(II)
                WRITE(OUTU,'(A9,I5,3F12.4)') &
                     'RC1BIAS> ',I,XCOMP(I),YCOMP(I),ZCOMP(I)
             END DO
          ENDIF

          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,'(A)') &
                  ' RC1BIAS> COORDINATES OF THE RC1 INITIAL CONFIG.:'
             DO II=1,NSEL1
                I = ISEL1(II)
                WRITE(OUTU,'(A9,I5,3F12.4)')  &
                     'RC1BIAS> ',I,X(I),Y(I),Z(I)
             END DO
          ENDIF
       END IF

       call chmalloc(FILE,'RC1BIAS','IL1',SIZEIJ,intg=IL1)
       call chmalloc(FILE,'RC1BIAS','JL1',SIZEIJ,intg=JL1)
       call chmalloc(FILE,'RC1BIAS','RCIJ1',SIZEIJ,crl=RCIJ1)
       IF (JRLD1) THEN
          READ(IUNJLS1,*)NSELL
          DO I=1,NSELL
             READ(IUNJLS1,*)IL1(I),JL1(I)
          END DO
       ELSE IF (JRF1) THEN
          READ(IUNJLS1,*)NSELL
          DO I=1,NSELL
             READ(IUNJLS1,*)IL1(I),JL1(I),RCIJ1(I)
          END DO
       ELSE
          L=0
          DO II=1,NSEL1-1
             DO JJ=II+1,NSEL1
                L=L+1
                IL1(L)=ISEL1(II)
                JL1(L)=ISEL1(JJ)
             END DO
          END DO
          NSELL=L
       END IF

       RHO=0
       call chmalloc(FILE,'RC1BIAS','RIJ1',NSELL,crl=RIJ1)
       DO L=1,NSELL
          I=IL1(L)
          J=JL1(L)
          IF (.NOT.JRF1) THEN
             XCIJ=XCOMP(I)-XCOMP(J)
             YCIJ=YCOMP(I)-YCOMP(J)
             ZCIJ=ZCOMP(I)-ZCOMP(J)
             RCIJ1(L)=SQRT(XCIJ*XCIJ+YCIJ*YCIJ+ZCIJ*ZCIJ)
          ENDIF
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ1(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN1)) THEN
          RC1BL = NSELL*NENSEM
          call chmalloc('hqbm.src','RC1BIAS','RC1BUF',RC1BL,crl=RC1BUF)
          CALL ENSAVE(RIJ1,RC1BUF,NSELL)
       ENDIF
       DO L=1,NSELL
#endif 
          RHO = RHO+(RIJ1(L)-RCIJ1(L))**2
       END DO
       RHO = RHO/NSELL
       RHO0 = RHO
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC1BIAS> INITIAL DISTANCE FOR RC1 = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' RC1BIAS> NUMBER OF DISTANCES = ',NSELL
       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO L=1,NSELL
          I=IL1(L)
          J=JL1(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ1(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF (NENSEM.NE.1.AND.(.NOT.JNOEN1))  &
            CALL ENSAVE(RIJ1,RC1BUF,NSELL)
       DO L=1,NSELL
#endif 
          RHO = RHO + (RIJ1(L)-RCIJ1(L))**2
       END DO

       RHO = RHO/NSELL
       XI=RHO
    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    !HX:
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF (.NOT. JSMD1) THEN
          IF ((((.NOT. JAWAY1) .AND. (XI .GT. XIA)) &
               .OR. ((JAWAY1) .AND. (XI .LT. XIA))).OR.JHQFIX1) THEN
             DXI=XI-XIA
#if KEY_ENSEMBLE==1
             IF (JNOEN1) THEN
                EU=ALPHA2*DXI**2
             ELSE
                EU=ALPHA2*REAL(NENSEM)*DXI**2
             ENDIF
#else /**/
             EU=ALPHA2*DXI**2
#endif 
             BUX=ALPHA*DXI
             BUX=TWO*BUX/(NSELL)
             !        Compute forces
             DO L=1,NSELL
                I=IL1(L)
                J=JL1(L)
                XIJ=X(I)-X(J)
                YIJ=Y(I)-Y(J)
                ZIJ=Z(I)-Z(J)
                AUX = BUX * (RIJ1(L)-RCIJ1(L))/RIJ1(L)
                DX(I)=DX(I)+AUX*XIJ
                DY(I)=DY(I)+AUX*YIJ
                DZ(I)=DZ(I)+AUX*ZIJ
                DX(J)=DX(J)-AUX*XIJ
                DY(J)=DY(J)-AUX*YIJ
                DZ(J)=DZ(J)-AUX*ZIJ
             END DO
          ELSE
             EU=ZERO
             XIA=XI
          END IF

       ELSE IF (JSMD1) THEN

          DXI=XI-XIA
#if KEY_ENSEMBLE==1
          IF (JNOEN1) THEN
             EU=ALPHA2*DXI**2
          ELSE
             EU=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EU=ALPHA2*DXI**2
#endif 
          BUX=ALPHA*DXI
          BUX=TWO*BUX/(NSELL)
          !           Compute forces
          DO L=1,NSELL
             I=IL1(L)
             J=JL1(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             AUX = BUX * (RIJ1(L)-RCIJ1(L))/RIJ1(L)
             DX(I)=DX(I)+AUX*XIJ
             DY(I)=DY(I)+AUX*YIJ
             DZ(I)=DZ(I)+AUX*ZIJ
             DX(J)=DX(J)-AUX*XIJ
             DY(J)=DY(J)-AUX*YIJ
             DZ(J)=DZ(J)-AUX*ZIJ
          END DO

          XIA=XIA+GAMMA1*TIMEST

       END IF
    END IF

    IF (((IOLEV.GE.0).OR.JNOEN1).AND.(.NOT.QINIT)) THEN
       IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
          WRITE(IUNJUJ1,'(I9,3(1X,F14.6))')HQSTEP &
               ,XI,XIA,EU
       ENDIF
    ENDIF
    IF (XI .GT. XIMAX1) CALL WRNDIE(-1,'<RC1BIAS>', &
         'Reaction coordinate RC1 exceeded maximum value.')
    RETURN
  END SUBROUTINE RC1BIAS
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC2BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC2
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    !     Author: Emanuele Paci <07-02-97>
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    real(chm_real) FCN,AUXR,THETA,COSPHI
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES

    INTEGER I,J,L,L1,L2,K,II,JJ,LL,NSELL
    INTEGER IRES,JRES,LRES
    real(chm_real) XIJ,YIJ,ZIJ,R,RCUT1
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0,XI,XIA,DXI,ALPHA,ALPHA2,AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C
    SAVE XI,XIA,RHO0,NSELL,RCUT1

    ALPHA=DFALPHA2
    ALPHA2=ALPHA/TWO
    EU=0.0
    ! ****************************
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC2BIAS> CALLING HQBM BIAS FOR RC2'
          WRITE(OUTU,'(A,F19.8)')' RC2BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC2BIAS> THE RC2 PERTURBATION DRIVES THE SYSTEM'
          IF (JAWAY2) WRITE(OUTU,'(A)') &
               ' RC2BIAS>  AWAY FROM THE REFERENCE CONFIGURATION'
          IF (.NOT.JAWAY2) WRITE(OUTU,'(A)') &
               ' RC2BIAS>  THROUGH THE REFERENCE CONFIGURATION'
       END IF

       IF (.NOT.JRLD2) THEN
          IF (PRNLEV.GE.2) THEN
             WRITE(OUTU,'(A)') &
                  ' RC2BIAS> COORDINATES OF RC2 REFERENCE CONFIG:'
             DO II=1,NSEL2
                I = ISEL2(II)
                WRITE(OUTU,'(A9,I5,3F12.4)') &
                     'RC2BIAS> ',I,XCOMP(I),YCOMP(I),ZCOMP(I)
             END DO
          ENDIF

          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,'(A)')  &
                  ' RC2BIAS> COORDINATES OF RC2 INITIAL CONFIG.:'
             DO II=1,NSEL2
                I = ISEL2(II)
                WRITE(OUTU,'(A9,I5,3F12.4)') 'RC2BIAS> ' &
                     ,I,X(I),Y(I),Z(I)
             END DO
          ENDIF
       END IF

       call chmalloc(FILE,'RC2BIAS','IL2',SIZEIJ,intg=IL2)
       call chmalloc(FILE,'RC2BIAS','JL2',SIZEIJ,intg=JL2)
       IF (.NOT.JRLD2) THEN
          L=0
          DO II=1,NSEL2-1
             DO JJ=II+1,NSEL2
                L=L+1
                IL2(L)=ISEL2(II)
                JL2(L)=ISEL2(JJ)
             END DO
          END DO
          NSELL=L
       ELSE
          READ(IUNJLS2,*)NSELL
          DO I=1,NSELL
             READ(IUNJLS2,*)IL2(I),JL2(I)
          END DO
       END IF

       RHO=0
       call chmalloc(FILE,'RC2BIAS','RIJ2',NSELL,crl=RIJ2)
       call chmalloc(FILE,'RC2BIAS','RCIJ2',NSELL,crl=RCIJ2)
       DO L=1,NSELL
          I=IL2(L)
          J=JL2(L)
          XCIJ=XCOMP(I)-XCOMP(J)
          YCIJ=YCOMP(I)-YCOMP(J)
          ZCIJ=ZCOMP(I)-ZCOMP(J)
          RCIJ2(L)=SQRT(XCIJ*XCIJ+YCIJ*YCIJ+ZCIJ*ZCIJ)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ2(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
          CUX = (RIJ2(L)-RCIJ2(L))/RCIJ2(L)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN2)) THEN
          RC2BL = NSELL*NENSEM
          call chmalloc('hqbm.src','RC2BIAS','RC2BUF',RC2BL,crl=RC2BUF)
          CALL ENSAVE(RIJ2,RC2BUF,NSELL)
       ENDIF
       DO L=1,NSELL
#endif 
          RHO = RHO + DEXP(-(CUX*CUX))
       END DO
       RHO = 1.0D0 - RHO/NSELL
       RHO0 = RHO
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC2BIAS> INITIAL DISTANCE = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' RC2BIAS> NUMBER OF DISTANCES = ',NSELL

       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO L=1,NSELL
          I=IL2(L)
          J=JL2(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ2(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN2))  &
            CALL ENSAVE(RIJ2,RC2BUF,NSELL)
       DO L=1,NSELL
#endif 
          CUX = (RIJ2(L)-RCIJ2(L))/RCIJ2(L)
          RHO = RHO + DEXP(-(CUX*CUX))
       END DO

       RHO = 1.0D0 - RHO/NSELL

       XI=RHO
    END IF
    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    !HX:
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF (.NOT. JSMD2) THEN
          IF ((((.NOT. JAWAY2) .AND. (XI .GT. XIA)) &
               .OR. ((JAWAY2) .AND. (XI .LT. XIA))).OR.JHQFIX2) THEN
             DXI=XI-XIA
#if KEY_ENSEMBLE==1
             IF (JNOEN2) THEN
                EU=ALPHA2*DXI**2
             ELSE
                EU=ALPHA2**REAL(NENSEM)*DXI**2
             ENDIF
#else /**/
             EU=ALPHA2*DXI**2
#endif 
             BUX=ALPHA*DXI
             BUX=TWO*BUX/NSELL
             !
             !        Compute forces
             DO L=1,NSELL
                I=IL2(L)
                J=JL2(L)
                XIJ=X(I)-X(J)
                YIJ=Y(I)-Y(J)
                ZIJ=Z(I)-Z(J)
                CUX = (RIJ2(L)-RCIJ2(L))/RCIJ2(L)
                AUX = BUX * (CUX/RCIJ2(L))*DEXP(-(CUX*CUX))/RIJ2(L)
                DX(I)=DX(I)+AUX*XIJ
                DY(I)=DY(I)+AUX*YIJ
                DZ(I)=DZ(I)+AUX*ZIJ
                DX(J)=DX(J)-AUX*XIJ
                DY(J)=DY(J)-AUX*YIJ
                DZ(J)=DZ(J)-AUX*ZIJ
             END DO
          ELSE
             EU=ZERO
             XIA=XI
          END IF

       ELSE IF (JSMD2) THEN

          DXI=XI-XIA
#if KEY_ENSEMBLE==1
          IF (JNOEN2) THEN
             EU=ALPHA2*DXI**2
          ELSE
             EU=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EU=ALPHA2*DXI**2
#endif 
          BUX=ALPHA*DXI
          BUX=TWO*BUX/NSELL
          !
          !        Compute forces
          DO L=1,NSELL
             I=IL2(L)
             J=JL2(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             AUX = BUX * (RIJ2(L)-RCIJ2(L))/RIJ2(L)
             CUX = (RIJ2(L)-RCIJ2(L))/RCIJ2(L)
             AUX = BUX * (CUX/RCIJ2(L))*DEXP(-(CUX*CUX))/RIJ2(L)
             DX(I)=DX(I)+AUX*XIJ
             DY(I)=DY(I)+AUX*YIJ
             DZ(I)=DZ(I)+AUX*ZIJ
             DX(J)=DX(J)-AUX*XIJ
             DY(J)=DY(J)-AUX*YIJ
             DZ(J)=DZ(J)-AUX*ZIJ
          END DO

          XIA=XIA+GAMMA2*TIMEST

       END IF

    END IF

    IF (((IOLEV.GE.0).OR.JNOEN2).AND.(.NOT.QINIT)) THEN
       IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
          WRITE(IUNJUJ2,'(I9,3(1X,F14.6))')HQSTEP &
               ,XI,XIA,EU
       ENDIF
    ENDIF

    IF (XI .GT. XIMAX2) CALL WRNDIE(-1,'<RC2BIAS>', &
         'Reaction coordinate RC2 exceeded maximum value.')

    RETURN
  END SUBROUTINE RC2BIAS
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC3BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC3
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    !     Author: Emanuele Paci <07-02-97>
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    real(chm_real),allocatable,dimension(:),save :: PHI,NNS,NNT,NNTLOC
    real(chm_real),dimension(maxres) :: NNS0, NNT0, PHICAL
    INTEGER INDL(MAXSEL),INDH(MAXSEL),HDONOR(MAXSEL),HACCEP(MAXSEL), &
         NHBI,PIRES

    INTEGER I,J,L,L1,L2,K,II,JJ,LL
    INTEGER IOS
    INTEGER EXPNPHI
    INTEGER IRES,JRES,LRES,TCONI,CNOW
    real(chm_real) XIJ,YIJ,ZIJ,R,RCUT1,TOTCON
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0,XI,XIA,DXI,ALPHA,ALPHA2,AUX,BUX,CUX
    real(chm_real) ARHO,ARHO0,APHIs,APHIe,AXI,AXIA,AALPHA,AALPHA2
    real(chm_real) CX,CY,CZ,A,B,C,PSIM
    real(chm_real) EUstd,EUavg
    SAVE XI,XIA,RHO0,ARHO0,APHIe,AXI,AXIA
    SAVE EXPNPHI,INDL,INDH,RCUT1
    character(len=*),parameter :: routine_name="rc3bias"

    if( .not. allocated(phi) ) then
       call chmalloc(FILE,routine_name,'PHI    ',maxres,crl=PHI   )
       call chmalloc(FILE,routine_name,'NNS    ',maxres,crl=NNS   )
       call chmalloc(FILE,routine_name,'NNT    ',maxres,crl=NNT   )
       call chmalloc(FILE,routine_name,'NNTLOC ',maxres,crl=NNTLOC)
    endif

    ALPHA=DFALPHA3
    ALPHA2=ALPHA/TWO
    AALPHA=DFAALPHA3
    AALPHA2=AALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    !
    ! ****************************
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC3BIAS> CALLING HQBM BIAS'
          WRITE(OUTU,'(A,F19.8)')' RC3BIAS> ALPHA3=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC3BIAS> THE RC3 PERTURBATION DRIVES THE SYSTEM'
          IF (JAWAY3) WRITE(OUTU,'(A)') &
               ' RC3BIAS>  AWAY FROM THE REFERENCE CONFIGURATION'
          IF (.NOT.JAWAY3) WRITE(OUTU,'(A)') &
               ' RC3BIAS>  THROUGH THE REFERENCE CONFIGURATION'
       END IF

       RHO=0.0

       RCUT1=RCUT+TOL
       DO IRES=1,NRES
          PHI(IRES)=-1.0
       END DO
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A,F5.1)')' Native contacts exist if |r_i-r_j|<' &
               ,RCUT
          WRITE(OUTU,'(A,F5.1)') &
               ' Contacts exist if i,j form a native contact and |r_i-r_j|<' &
               ,RCUT1
          WRITE(OUTU,'(A,I3)')'                      and EXCL=',EXCL
       ENDIF
#if KEY_PARALLEL==1
       IF (IOLEV.GT.0) THEN
#endif 
          DO
             READ(IUNPHI,*,iostat=IOS)I,PHI(I)
             IF (IOS /= 0) EXIT
             IF (PRNLEV.GE.2) WRITE(OUTU,*)' HQBM> RESIDUE ',I,' -> PHI' &
                  ,PHI(I)
          ENDDO

#if KEY_PARALLEL==1
       ENDIF
       CALL PSND8(PHI,MAXRES)
#endif 

       !     nns(i) is the number of contacts of residue i

       DO I=1,NRES
          NNS(I)=0
          NNT(I)=0
          NNS0(I)=0
          NNT0(I)=0
       END DO

       EXPNPHI=0

       DO I=1,NRES
          IF (PHI(I).NE.-1.0) THEN
             EXPNPHI=EXPNPHI+1
          END IF
       END DO

       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') &
            ' RC3BIAS> TOTAL NUMBER OF EXPERIMENTAL PHI',EXPNPHI

       !        Computes the number of native contacts for each residue

       call chmalloc(FILE,'RC3BIAS','IL3',SIZEIJ,intg=IL3)
       call chmalloc(FILE,'RC3BIAS','JL3',SIZEIJ,intg=JL3)
       IF (JCOMB3) THEN
          !             write(outu,'(a)') 'Combinatorial initialization'
          CALL COMBINI(IL3,JL3,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL3,NSEL3)
       ELSE
          CALL NNTINI(IL3,JL3,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL3,NSEL3)
       ENDIF

       IF (JAVPHI) THEN
          DO I=1,NRES
             NNTLOC(I) = NNT(I)
          END DO
       END IF

       EXPNPHI=0

       IF ((IOLEV.GE.0).OR.JNOEN3) THEN
          WRITE(IUNKUK3,'(80a)')'# i          phi(i,t=0)             ' &
               ,'nns         nnt      nns0   nnt0'
          WRITE(IUNKUK3,'(80a)') &
               '#       smoothed    unsmoothed       '
       ENDIF
       APHIe = 0.0
       DO I=1,NRES
          IF (PHI(I).NE.-1.0) THEN
             IF (NNS(I).NE.0) THEN
                EXPNPHI=EXPNPHI+1
                APHIe = APHIe + PHI(I)
             ENDIF
             IF ((NNS(I).EQ.0).AND.(PRNLEV.GE.2)) WRITE(OUTU,*) &
                  ' RC3BIAS> RESIDUE',I, &
                  'HAS ZERO NEIGHBORS AND WILL BE DISREGRDED'
          END IF
          IF ((IOLEV.GE.0).OR.JNOEN3) THEN
             IF (NNS(I).NE.0.AND.NNS0(I).NE.0) WRITE(IUNKUK3 &
                  ,'(i3,4(1x,f11.5),3x,2(f7.1))')I,NNT(I)/NNS(I) &
                  ,NNT0(I)/NNS0(I),NNS(I),NNT(I),NNS0(I),NNT0(I)
             IF (NNS(I).EQ.0.AND.NNS0(I).NE.0) WRITE(IUNKUK3 &
                  ,'(i3,4(1x,f11.5),3x,2(f7.1))')I,9999. &
                  ,NNT0(I)/NNS0(I),NNS(I),NNT(I),NNS0(I),NNT0(I)
             IF (NNS(I).NE.0.AND.NNS0(I).EQ.0) WRITE(IUNKUK3 &
                  ,'(i3,4(1x,f11.5),3x,2(f7.1))')I,NNT(I)/NNS(I) &
                  ,9999.,NNS(I),NNT(I),NNS0(I),NNT0(I)
          ENDIF
       END DO
       APHIe = APHIe / REAL(EXPNPHI)
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,*) &
               ' RC3BIAS> CORRECTED NUMBER OF EXPERIMENTAL PHI ', EXPNPHI
          WRITE(OUTU,*) &
               ' RC3BIAS> AVERAGE EXPERIMENTAL PHI = ', APHIe
       ENDIF

#if KEY_ENSEMBLE==1
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN3)) THEN
          PHIBL = NRES*NENSEM
          call chmalloc('hqbm.src','RC3BIAS','PHIBUF',PHIBL,crl=PHIBUF)
          CALL ENSAVE(NNT,PHIBUF,NRES)
       ENDIF
#endif 
       RHO=0
       ARHO=0
       IF (.NOT. JRZERO3) THEN
          DO I=1,NRES
             IF (PHI(I).NE.-1) THEN
                IF (NNS(I).NE.0) RHO=RHO+(NNT(I)/NNS(I)-PHI(I))**2
             END IF
          END DO
          RHO=RHO/REAL(EXPNPHI)
#if KEY_ENSEMBLE==1
          APHIs = 0
          IF (JAVPHI) THEN
             DO I=1,NRES
                IF (PHI(I).NE.-1) THEN
                   IF (NNS(I).NE.0)  &
                        APHIs=APHIs+NNTLOC(I)/NNS(I)
                END IF
             END DO
             APHIs = APHIs / EXPNPHI
             ARHO = (APHIs - APHIe)**2
          ENDIF
#endif 
       ENDIF

       RHO0 = RHO
       ARHO0 = ARHO

       IF (PRNLEV.GE.2) THEN
          IF (JRZERO3) WRITE(OUTU,'(A)') &
               ' RC3BIAS> YOU CHOSE TO KEEP THE PROTEIN AROUND RHO=0', &
               ' RC3BIAS> FOR RC3, AND NOT AROUND THE INITIAL RHO; THUS:'
          WRITE(OUTU,'(A,F13.4)')' RC3BIAS> INITIAL DISTANCE = ',RHO0
#if KEY_ENSEMBLE==1
          IF (JAVPHI) THEN
             WRITE (OUTU,'(A)')  &
                  ' RC3BIAS> CONSTRAINTS WILL ALSO BE PLACED ON AVERAGE'
             WRITE (OUTU,'(A,F13.4)')  &
                  ' RC3BIAS> PHI VALUES; INITIAL DISTANCE FOR THIS = ',ARHO0
             WRITE (OUTU,'(A,F13.4)')  &
                  ' RC3BIAS> INITIAL "SIMULATED" AVERAGE PHI = ', APHIs
          ENDIF
#endif 
       ENDIF

       XI = RHO0
       XIA = XI
       AXI = ARHO0
       AXIA = AXI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       CALL NNTCAL(IL3,JL3,INDH,INDL,NNS,NNT,RCUT1,PHI)
#if KEY_ENSEMBLE==1
       IF (JAVPHI) THEN
          APHIs = 0.0
          DO I=1,NRES
             IF (PHI(I).NE.-1) THEN
                NNTLOC(I) = NNT(I)
                APHIs = APHIs + NNT(I)/NNS(I)
             ENDIF
          END DO
          APHIs = APHIs / EXPNPHI
          ARHO = (APHIs - APHIe)**2
       ENDIF
       AXI=ARHO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN3))  &
            CALL ENSAVE(NNT,PHIBUF,NRES)
#endif 
       RHO=0.0
       DO I=1,NRES
          IF (PHI(I).NE.-1) THEN
             IF (NNS(I).NE.0) RHO=RHO+(NNT(I)/NNS(I)-PHI(I))**2
          END IF
       END DO
       RHO=RHO/REAL(EXPNPHI)

       XI=RHO
       ! ****************************
    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...
    !
    ! *************** CONSTRAINT ON EACH PHI VALUE *****************
    !
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF (.NOT. JSMD3) THEN
          IF ((((.NOT. JAWAY3) .AND. (XI .GT. XIA)) &
               .OR. ((JAWAY3) .AND. (XI .LT. XIA))).OR.JHQFIX3) THEN
             !               write (outu,'(a)') ' calculating forces ...'
             DXI=XI-XIA
#if KEY_ENSEMBLE==1
             IF (JNOEN3) THEN
                EUstd=ALPHA2*DXI**2
             ELSE
                EUstd=ALPHA2*REAL(NENSEM)*DXI**2
             ENDIF
#else /**/
             EUstd=ALPHA2*DXI**2
#endif 
             EU = EU + EUstd
             BUX=-TWO*ALPHA*BETA*DXI/EXPNPHI
             !              Compute forces for all selected atoms
             DO II=1,NSEL3
                I=ISEL3(II)
                DO LRES=1,NRES
                   IF (PHI(LRES).NE.-1 .AND. (NNS(LRES).NE.0)) THEN
                      CX=0.0
                      CY=0.0
                      CZ=0.0
                      DO K=INDL(LRES),INDH(LRES)
                         L=IL3(K)
                         J=JL3(K)
                         IF (I .EQ. L) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            C=A*B/R
                            CX=CX+C*XIJ
                            CY=CY+C*YIJ
                            CZ=CZ+C*ZIJ
                         ELSE IF (I .EQ. J) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            C=A*B/R
                            CX=CX-C*XIJ
                            CY=CY-C*YIJ
                            CZ=CZ-C*ZIJ
                         END IF
                      END DO
                      CUX=(NNT(LRES)/NNS(LRES)-PHI(LRES))/NNS(LRES)
                      DX(I)=DX(I)+BUX*CUX*CX
                      DY(I)=DY(I)+BUX*CUX*CY
                      DZ(I)=DZ(I)+BUX*CUX*CZ
                   END IF
                END DO
             END DO
          ELSE
             EUstd=ZERO
             XIA=XI
          END IF

       ELSE IF (JSMD3) THEN
          DXI=XI-XIA
#if KEY_ENSEMBLE==1
          IF (JNOEN3) THEN
             EUstd=ALPHA2*DXI**2
          ELSE
             EUstd=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EUstd=ALPHA2*DXI**2
#endif 
          EU = EU + EUstd
          BUX=-TWO*ALPHA*BETA*DXI/EXPNPHI
          !           Compute forces for all selected atoms
          DO II=1,NSEL3
             I=ISEL3(II)
             DO LRES=1,NRES
                IF (PHI(LRES).NE.-1 .AND. (NNS(LRES).NE.0)) THEN
                   CX=0.0
                   CY=0.0
                   CZ=0.0
                   DO K=INDL(LRES),INDH(LRES)
                      L=IL3(K)
                      J=JL3(K)
                      IF (I .EQ. L) THEN
                         XIJ=X(L)-X(J)
                         YIJ=Y(L)-Y(J)
                         ZIJ=Z(L)-Z(J)
                         R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                         A=EXP(BETA*(R-RCUT1))
                         B=1.0/((1.0+A)**2)
                         C=A*B/R
                         CX=CX+C*XIJ
                         CY=CY+C*YIJ
                         CZ=CZ+C*ZIJ
                      ELSE IF (I .EQ. J) THEN
                         XIJ=X(L)-X(J)
                         YIJ=Y(L)-Y(J)
                         ZIJ=Z(L)-Z(J)
                         R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                         A=EXP(BETA*(R-RCUT1))
                         B=1.0/((1.0+A)**2)
                         C=A*B/R
                         CX=CX-C*XIJ
                         CY=CY-C*YIJ
                         CZ=CZ-C*ZIJ
                      END IF
                   END DO
                   CUX=(NNT(LRES)/NNS(LRES)-PHI(LRES))/NNS(LRES)
                   DX(I)=DX(I)+BUX*CUX*CX
                   DY(I)=DY(I)+BUX*CUX*CY
                   DZ(I)=DZ(I)+BUX*CUX*CZ
                END IF
             END DO
          END DO

          XIA=MAX(ZERO,(XIA+GAMMA3*TIMEST))
       END IF
       ! end of 'IF (QFORCE) THEN ...':
    END IF

#if KEY_ENSEMBLE==1
    !
    ! *************** CONSTRAINT ON AVERAGE PHI VALUE *****************
    !
    ! Note: The irony is that although this only works with the ENSEMBLE
    !       keyword, it is in fact only a single replica constraint
    !       intended to keep individual replicas from going towards opposite
    !       sides of the barrier.
    !
    IF (QFORCE.AND.AALPHA.NE.0.0) THEN
       IF (.NOT. JSMD3) THEN
          IF ((((.NOT. JAWAY3) .AND. (AXI .GT. AXIA)) &
               .OR. ((JAWAY3).AND.(AXI.LT.AXIA))).OR.JHQFIX3) THEN
             DXI=AXI-AXIA
             EUavg=AALPHA2*DXI**2
             EU = EU + EUavg
             BUX=-TWO*AALPHA*BETA*DXI/EXPNPHI
             !              Compute forces for all selected atoms
             DO II=1,NSEL3
                I=ISEL3(II)
                DO LRES=1,NRES
                   IF (PHI(LRES).NE.-1 .AND. (NNS(LRES).NE.0)) THEN
                      CX=0.0
                      CY=0.0
                      CZ=0.0
                      DO K=INDL(LRES),INDH(LRES)
                         L=IL3(K)
                         J=JL3(K)
                         IF (I .EQ. L) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            C=A*B/R
                            CX=CX+C*XIJ
                            CY=CY+C*YIJ
                            CZ=CZ+C*ZIJ
                         ELSE IF (I .EQ. J) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            C=A*B/R
                            CX=CX-C*XIJ
                            CY=CY-C*YIJ
                            CZ=CZ-C*ZIJ
                         END IF
                      END DO
                      CUX=(APHIs-APHIe)/NNS(LRES)
                      DX(I)=DX(I)+BUX*CUX*CX
                      DY(I)=DY(I)+BUX*CUX*CY
                      DZ(I)=DZ(I)+BUX*CUX*CZ
                   END IF
                END DO
             END DO
          ELSE
             EUavg=ZERO
             AXIA=AXI
          END IF

       ELSE IF (JSMD3) THEN
          DXI=AXI-AXIA
          EUavg=AALPHA2*DXI**2
          EU = EU + EUavg
          BUX=-TWO*AALPHA*BETA*DXI/EXPNPHI
          !           Compute forces for all selected atoms
          DO II=1,NSEL3
             I=ISEL3(II)
             DO LRES=1,NRES
                IF (PHI(LRES).NE.-1 .AND. (NNS(LRES).NE.0)) THEN
                   CX=0.0
                   CY=0.0
                   CZ=0.0
                   DO K=INDL(LRES),INDH(LRES)
                      L=IL3(K)
                      J=JL3(K)
                      IF (I .EQ. L) THEN
                         XIJ=X(L)-X(J)
                         YIJ=Y(L)-Y(J)
                         ZIJ=Z(L)-Z(J)
                         R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                         A=EXP(BETA*(R-RCUT1))
                         B=1.0/((1.0+A)**2)
                         C=A*B/R
                         CX=CX+C*XIJ
                         CY=CY+C*YIJ
                         CZ=CZ+C*ZIJ
                      ELSE IF (I .EQ. J) THEN
                         XIJ=X(L)-X(J)
                         YIJ=Y(L)-Y(J)
                         ZIJ=Z(L)-Z(J)
                         R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                         A=EXP(BETA*(R-RCUT1))
                         B=1.0/((1.0+A)**2)
                         C=A*B/R
                         CX=CX-C*XIJ
                         CY=CY-C*YIJ
                         CZ=CZ-C*ZIJ
                      END IF
                   END DO
                   CUX=(APHIs-APHIe)/NNS(LRES)
                   DX(I)=DX(I)+BUX*CUX*CX
                   DY(I)=DY(I)+BUX*CUX*CY
                   DZ(I)=DZ(I)+BUX*CUX*CZ
                END IF
             END DO
          END DO

          AXIA=MAX(ZERO,(AXIA+AGAMMA3*TIMEST))
       END IF
    END IF
    ! end average phi constraint
#endif 
    !HX
    IF ((IOLEV.GE.0).OR.JNOEN3) THEN
       IF(QINIT) THEN
          IF(IUNDMP3.NE.-1)THEN
             J=0
             DO I=1,NRES
                IF (PHI(I).NE.-1.AND.NNS(I).NE.0) THEN
                   J=J+1
                   PHICAL(J) = REAL(I)
                END IF
             END DO
             WRITE(IUNDMP3,'(7(I9))')(NINT(PHICAL(II)),II=1,EXPNPHI)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ3,'(I9,3(1X,F14.6))')HQSTEP &
                  ,XI,XIA,EUstd
             IF(IUNDMP3.NE.-1)THEN
                J=0
                DO I=1,NRES
                   IF (PHI(I).NE.-1.AND.NNS(I).NE.0) THEN
                      J=J+1
                      PHICAL(J) = NNT(I)/NNS(I)
                   END IF
                END DO
                WRITE(IUNDMP3,'(7(F8.3,1X))')( PHICAL(II),II=1,EXPNPHI)
             ENDIF
          ENDIF
       ENDIF
    ENDIF

#if KEY_ENSEMBLE==1
    IF (JAVPHI) THEN
       IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) then
          WRITE(IUNJUJ3A,'(I9,3(1X,F14.6))')HQSTEP &
               ,AXI,AXIA,EUavg
       ENDIF
    ENDIF
#endif 

    IF (XI .GT. XIMAX3) THEN
       IF ((IOLEV.GT.0).OR.JNOEN3) WRITE(OUTU,'(A,F8.3,A,F8.3)') &
            'XI = ', XI, ' XIMAX3 = ', XIMAX3
       CALL GFLUSH(OUTU)
       CALL WRNDIE(-1,'<RC3BIAS>', &
            'Reaction coordinate RC3 exceeded maximum value.')
    ENDIF

    RETURN
  END SUBROUTINE RC3BIAS
  !-----------------------------------------------------------------
  LOGICAL FUNCTION CONSDO(CTYPE,SIM,EXPT)
    !-----------------------------------------------------------------
    !     Whether protection factor contributes to constraint energy
    !-----------------------------------------------------------------
    implicit none
    INTEGER CTYPE
    real(chm_real) SIM, EXPT
    IF (CTYPE == 0) THEN
       ! full harmonic constraint
       CONSDO = .TRUE.
    ELSE IF (CTYPE == 1) THEN
       ! half-harmonic constraint: upper bound
       IF (SIM > EXPT) THEN
          CONSDO = .TRUE.
       ENDIF
    ELSE IF (CTYPE == -1) THEN
       ! half-harmonic constraint: lower bound
       IF (SIM < EXPT) THEN
          CONSDO = .TRUE.
       ENDIF
    ELSE
       CALL WRNDIE(-1,'<CONSDO>', &
            'Error: Unknown constraint type')
    ENDIF
    RETURN
  END FUNCTION CONSDO
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC4BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC4
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
    use inbnd
#if KEY_PARALLEL==1
    use parallel      
#endif
    use ensemble
    use memory
    use chutil,only:getres
    implicit none
    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    real(chm_real),allocatable,dimension(:),save :: NNS,NNT,SNNS,SNNT, NNH, HXPROC
    real(chm_real),dimension(maxres) :: NNS0, NNT0, HXSIM, PHICAL
    INTEGER INDL(MAXSEL),INDH(MAXSEL),HDONOR(MAXSEL),HACCEP(MAXSEL), &
         NHBI,PIRES,SINDH(MAXSEL),SINDL(MAXSEL)
    integer,allocatable,dimension(:),save :: CTYPE

    INTEGER I,J,L,L1,L2,K,II,JJ,LL
    INTEGER IOS
    INTEGER NHXPRO
    INTEGER IRES,JRES,LRES,TCONI,CNOW
    real(chm_real) XIJ,YIJ,ZIJ,R,R2,RCUT1,TOTCON
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0,XI,XIA,DXI,ALPHA,ALPHA2,AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C,PSIM,SW,DSW,DRCUT,RO3RON,CTON2,CTOF2
    real(chm_real) CHI,DCHI
#if KEY_ENSEMBLE==1
    CHARACTER TBUF*(ENSBFL)
#endif 
    SAVE XI,XIA,RHO0
    SAVE INDL,INDH,RCUT1,NHXPRO
    SAVE SINDH,SINDL,DRCUT,CTON2,CTOF2
    character(len=*),parameter :: routine_name="rc4bias"

    if( .not. allocated(nns))then
       call chmalloc(FILE,routine_name,'NNS    ',maxres,crl=NNS   )
       call chmalloc(FILE,routine_name,'NNT    ',maxres,crl=NNT   )
       call chmalloc(FILE,routine_name,'SNNS   ',maxres,crl=SNNS  )
       call chmalloc(FILE,routine_name,'SNNT   ',maxres,crl=SNNT  )
       call chmalloc(FILE,routine_name,'NNH    ',maxres,crl=NNH   )
       call chmalloc(FILE,routine_name,'HXPROC ',maxres,crl=HXPROC)
       call chmalloc(FILE,routine_name,'ctype  ',maxres,intg=ctype)
    endif

    !     ------------------------------------------------------------
    !     some useful constants
    ALPHA=DFALPHA4
    ALPHA2=ALPHA/TWO
    IF (JNONN) THEN
       DRCUT = 1.0/(HXCTOF**2-HXCTON**2)**3
       RO3RON = HXCTOF**2+3.0*HXCTON**2
       CTON2 = HXCTON**2
       CTOF2 = HXCTOF**2
    ENDIF
    EU=0.0
    !     ------------------------------------------------------------
    !     ------------------------------------------------------------
    !     initialisation
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC4BIAS> CALLING HQBM BIAS'
          WRITE(OUTU,'(A,F19.8)')' RC4BIAS> ALPHA4=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC4BIAS> THE RC4 PERTURBATION DRIVES THE SYSTEM'
          IF (JAWAY4) WRITE(OUTU,'(A)') &
               ' RC4BIAS>  AWAY FROM THE REFERENCE CONFIGURATION'
          IF (.NOT.JAWAY4) WRITE(OUTU,'(A)') &
               ' RC4BIAS>  THROUGH THE REFERENCE CONFIGURATION'
          IF (JNONN) THEN
             WRITE(OUTU,'(A)')  &
                  ' RC4BIAS> USING ALL CONTACTS (INCLUDING NON-NATIVE)'
             WRITE(OUTU,'(A)')  &
                  ' RC4BIAS> ** BEWARE! YOU ARE INVOLVED IN THE TESTING **'
             WRITE(OUTU,'(A)')  &
                  ' RC4BIAS> ** OF THIS FEATURE ...                     **'
          ELSE
             WRITE(OUTU,'(A)')  &
                  ' RC4BIAS> USING ONLY NATIVE CONTACTS'
          END IF
       END IF
       !
       !     ------------------------------------------------------------
       !         Read experimental protection factors
       DO IRES=1,NRES
          HXPROC(IRES)=-1.0
       END DO
       IF (PRNLEV.GE.2) WRITE (OUTU,'(a,i5)')  &
            ' RC4BIAS> READING PROTECTION FACTORS FROM UNIT ', IUNPRO
       DO
          READ(IUNPRO,*,iostat=IOS)I,HXPROC(I),CTYPE(I)
          IF (IOS /= 0) EXIT
          IF ((CTYPE(I).LT.-1).OR.(CTYPE(I).GT.1)) THEN
             !             Catch constraint type not in {-1,0,1}
             !             Most likely caused by old constraint file format
             WRITE(OUTU,*) 'CONSTRAINT TYPE ', CTYPE(I), ' IS UNKNOWN'
             CALL WRNDIE(-1,'<RC4BIAS>', &
                  'Error: Unknown constraint type')
          ENDIF
          IF (PRNLEV.GE.2) WRITE(OUTU,*)' HQBM> RESIDUE ',I, &
               ' -> logPexp',HXPROC(I), ' Constraint type: ', CTYPE(I)
       ENDDO
       NHXPRO=0
       DO I=1,NRES
          IF (HXPROC(I).NE.-1.0) THEN
             NHXPRO=NHXPRO+1
          END IF
       END DO
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') &
            ' RC4BIAS> TOTAL NUMBER OF EXPTL PROTECTION FACTORS',NHXPRO
       !
       !     ------------------------------------------------------------
       !         set up native/non-native contact and initial hbond lists
       IF (JEEF1) THEN
          IF (JNONN) CALL WRNDIE(-1,'<RC4BIAS>', &
               'Non-native contacts not implemented for EEF1 burial')
          IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
               ' RC4BIAS> Using EEF1 energy terms to determine amide burial'
#if KEY_ASPENER==1
          CALL EEFBUR(NNT,NNS,ISELN,NSELN,EMSLCT,.true.)         
#endif
          CALL BBHBON(X,Y,Z,ISELO,ISELH,NSELO,NSELH,NNH, &
               HCUT,BETH,.true.,HDONOR,HACCEP,NHBI)
       ELSE
          RCUT1=RCUT+TOL
          IF (PRNLEV.GE.2) THEN
             IF (JNONN) THEN
                WRITE(OUTU,*) &
                     'Contacts exist for i,j if |r_i-r_j|<',RCUT
                WRITE(OUTU,'(A,I3)')'                 and EXCL=',EXCL
                WRITE(OUTU,*)'Applying a switching function between ', &
                     HXCTON, ' and ', HXCTOF, ' A'
             ELSE
                WRITE(OUTU,'(A,F5.1)') &
                     ' Native contacts exist if |r_i-r_j|<',RCUT
                WRITE(OUTU,'(A,F5.1)') &
                     'Contacts exist if i,j form a native contact and |r_i-r_j|<' &
                     ,RCUT1
                WRITE(OUTU,'(A,I3)')'                  and EXCL=',EXCL
             ENDIF
          ENDIF
          call chmalloc(FILE,'RC4BIAS','IL4',SIZEIJ,intg=IL4)
          call chmalloc(FILE,'RC4BIAS','JL4',SIZEIJ,intg=JL4)
          IF (JNHCON) THEN
             !             use NH --- side-chain contacts
             IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' RC4BIAS> Using (amide H)-(heavy atom) ' &
                  // ' contacts to determine amide burial'
             !             nns(i) is the number of contacts of residue i in the native state
             !             nnt(i) is the working list of native (or total if jnonn)
             !                                  contacts for some conformation
             !             nns0, nnt0 - the above quantities without a switching function
             IF (JNONN) THEN
                CALL NONNTC(XCOMP,YCOMP,ZCOMP,IL4,JL4,INDH,INDL,NNS, &
                     HXPROC,ISEL4,NSEL4)
                CALL NONNTC(X,Y,Z,IL4,JL4,INDH,INDL,NNT, &
                     HXPROC,ISEL4,NSEL4)
             ELSE
                CALL NHNNTI(IL4,JL4,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL4, &
                     NSEL4)
             ENDIF
             CALL GFLUSH(OUTU)
          ELSE
             IF (PRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' RC4BIAS> Using standard heavy atom' &
                  // ' contacts to determine amide burial'
             IF (JNONN) THEN
                CALL NONNTC(XCOMP,YCOMP,ZCOMP,IL4,JL4,INDH,INDL,NNS, &
                     HXPROC,ISEL4,NSEL4)
                CALL NONNTC(X,Y,Z,IL4,JL4,INDH,INDL,NNT, &
                     HXPROC,ISEL4,NSEL4)
             ELSE
                CALL NNTINI(IL4,JL4,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL4, &
                     NSEL4)
             ENDIF
          ENDIF
          CALL BBHBON(X,Y,Z,ISELO,ISELH,NSELO,NSELH,NNH, &
               HCUT,BETH,.true.,HDONOR,HACCEP,NHBI)
       END IF
       !
       !     ------------------------------------------------------------
       !         write iunk file
       IF ((IOLEV.GE.0).OR.JNOEN4) WRITE(IUNKUK4,'(A4,1X,4(A9,1X))')  &
            '# i ','Psim(i)','nns(i)','nnt(i)', 'nnh(i)'
       DO I=1,NRES
          PSIM=HXFC*NNT(I)+HXFH*NNH(I)
          HXSIM(I) = PSIM
          IF((IOLEV.GE.0).OR.JNOEN4)WRITE(IUNKUK4,'(I5,4(F8.3,1X))')  &
               I, PSIM, NNS(I), NNT(I), NNH(I)
       END DO
       !
       !     ------------------------------------------------------------
       !         calculate initial rho
#if KEY_ENSEMBLE==1
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN4)) THEN
          HXBL = NRES*NENSEM
          call chmalloc('hqbm.src','RC4BIAS','HXBUF',HXBL,crl=HXBUF)
          CALL ENSAVE(HXSIM,HXBUF,NRES)
       ENDIF
#endif 
       RHO=0.0
       IF (.NOT.JRZERO4) THEN
          DO I=1,NRES
             IF (HXPROC(I).NE.-1) THEN
                IF(CONSDO(CTYPE(I),HXSIM(I),HXPROC(I)))THEN
                   RHO=RHO+(HXSIM(I)-HXPROC(I))**2
                ENDIF
             END IF
          END DO
          RHO=RHO/REAL(NHXPRO)
       ENDIF
       RHO0 = RHO
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC4BIAS> INITIAL RC4DISTANCE = ', RHO0
       XI = RHO0
       XIA = XI
       !
       !     ------------------------------------------------------------
       !     ------------------------------------------------------------
       !     Do everytime that this routine is called except the first
    ELSE
       !     ------------------------------------------------------------
       !         compute current contacts and rho
       IF (JEEF1) THEN
#if KEY_ASPENER==1
          CALL EEFBUR(NNT,NNS,ISELN,NSELN,EMSLCT,.false.)        
#endif
       ELSE IF (JNONN) THEN
          CALL NONNTC(X,Y,Z,IL4,JL4,INDH,INDL,NNT, &
               HXPROC,ISEL4,NSEL4)
       ELSE
          CALL NNTCAL(IL4,JL4,INDH,INDL,NNS,NNT,RCUT1,HXPROC)
       END IF
       CALL BBHBON(X,Y,Z,ISELO,ISELH,NSELO,NSELH,NNH, &
            HCUT,BETH,.true.,HDONOR,HACCEP,NHBI)

       RHO=0.0
       DO I=1,NRES
          IF (HXPROC(I).NE.-1) THEN
             HXSIM(I)=HXFC*NNT(I)+HXFH*NNH(I)
#if KEY_ENSEMBLE==1
          END IF
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN4))  &
            CALL ENSAVE(HXSIM,HXBUF,NRES)
       DO I=1,NRES
          IF (HXPROC(I).NE.-1) THEN
#endif 
             IF (CONSDO(CTYPE(I),HXSIM(I),HXPROC(I))) THEN
                RHO=RHO+(HXSIM(I)-HXPROC(I))**2
             ENDIF
          END IF
       END DO
       RHO=RHO/REAL(NHXPRO)
       XI=RHO
    END IF
    !
    !     ------------------------------------------------------------
    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       !     ------------------------------------------------------------
       !       BMD PART
       IF (.NOT. JSMD4) THEN
          IF ((((.NOT. JAWAY4) .AND. (XI .GT. XIA)) &
               .OR. ((JAWAY4) .AND. (XI .LT. XIA))).OR.JHQFIX4) THEN
             IF (JEEF1) THEN
                CALL WRNDIE(-1,'<RC4BIAS>', &
                     'Error: Forces not implemented for EEF1; analysis only')
             ELSE
                DXI=XI-XIA
#if KEY_ENSEMBLE==1
                IF (JNOEN4) THEN
                   EU=ALPHA2*DXI**2
                ELSE
                   EU=ALPHA2*REAL(NENSEM)*DXI**2
                ENDIF
#else /**/
                EU=ALPHA2*DXI**2
#endif 
                !     ------------------------------------------------------------
                !           Compute forces for all selected atoms
                !           first do native contact part
                BUX=-TWO*ALPHA*DXI*HXFC/REAL(NHXPRO)
                DO II=1,NSEL4
                   I=ISEL4(II)
                   DO LRES=1,NRES
                      IF (HXPROC(LRES).NE.-1) THEN
                         CX=0.0
                         CY=0.0
                         CZ=0.0
                         IF (.NOT.CONSDO(CTYPE(LRES),HXSIM(LRES), &
                              HXPROC(LRES))) CONTINUE  ! cycle?
                         DO K=INDL(LRES),INDH(LRES)
                            L=IL4(K)
                            J=JL4(K)
                            IF (I .EQ. L) THEN
                               XIJ=X(L)-X(J)
                               YIJ=Y(L)-Y(J)
                               ZIJ=Z(L)-Z(J)
                               R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                               A=EXP(BETA*(R-RCUT1))
                               B=1.0/((1.0+A)**2)
                               IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                  SW=DRCUT*(CTON2-R**2) &
                                       *(CTOF2+2.0*R**2-3.0*CTON2)
                                  DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                  CHI=1.0/(1.0+A)
                                  DCHI=A*B*BETA
                                  C=(SW*DCHI+DSW*CHI)/R
                               ELSE
                                  C=A*B*BETA/R
                               ENDIF
                               CX=CX+C*XIJ
                               CY=CY+C*YIJ
                               CZ=CZ+C*ZIJ
                            ELSE IF (I .EQ. J) THEN
                               XIJ=X(L)-X(J)
                               YIJ=Y(L)-Y(J)
                               ZIJ=Z(L)-Z(J)
                               R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                               A=EXP(BETA*(R-RCUT1))
                               B=1.0/((1.0+A)**2)
                               IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                  SW=DRCUT*(CTON2-R**2) &
                                       *(CTOF2+2.0*R**2-3.0*CTON2)
                                  DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                  CHI=1.0/(1.0+A)
                                  DCHI=A*B*BETA
                                  C=(SW*DCHI+DSW*CHI)/R
                               ELSE
                                  C=A*B*BETA/R
                               ENDIF
                               CX=CX-C*XIJ
                               CY=CY-C*YIJ
                               CZ=CZ-C*ZIJ
                            END IF
                         END DO
                         CUX=HXSIM(LRES)-HXPROC(LRES)
                         DX(I)=DX(I)+BUX*CUX*CX
                         DY(I)=DY(I)+BUX*CUX*CY
                         DZ(I)=DZ(I)+BUX*CUX*CZ
                      END IF
                   END DO
                END DO
                !     ------------------------------------------------------------
                !           ...also calculate forces for amide H's
                IF (JNHCON) THEN
                   DO II=1,NSELH
                      I=ISELH(II)
                      DO LRES=1,NRES
                         IF (HXPROC(LRES).NE.-1) THEN
                            CX=0.0
                            CY=0.0
                            CZ=0.0
                            !                       Take care of half-well constraints
                            IF (.NOT.CONSDO(CTYPE(LRES),HXSIM(LRES), &
                                 HXPROC(LRES))) CONTINUE  ! cycle?

                            DO K=INDL(LRES),INDH(LRES)
                               L=IL4(K)
                               J=JL4(K)
                               IF (I .EQ. L) THEN
                                  XIJ=X(L)-X(J)
                                  YIJ=Y(L)-Y(J)
                                  ZIJ=Z(L)-Z(J)
                                  R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                                  A=EXP(BETA*(R-RCUT1))
                                  B=1.0/((1.0+A)**2)
                                  IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                     SW=DRCUT*(CTON2-R**2) &
                                          *(CTOF2+2.0*R**2-3.0*CTON2)
                                     DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                     CHI=1.0/(1.0+A)
                                     DCHI=A*B*BETA
                                     C=(SW*DCHI+DSW*CHI)/R
                                  ELSE
                                     C=A*B*BETA/R
                                  ENDIF
                                  CX=CX+C*XIJ
                                  CY=CY+C*YIJ
                                  CZ=CZ+C*ZIJ
                               ELSE IF (I .EQ. J) THEN
                                  XIJ=X(L)-X(J)
                                  YIJ=Y(L)-Y(J)
                                  ZIJ=Z(L)-Z(J)
                                  R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                                  A=EXP(BETA*(R-RCUT1))
                                  B=1.0/((1.0+A)**2)
                                  IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                     SW=DRCUT*(CTON2-R**2) &
                                          *(CTOF2+2.0*R**2-3.0*CTON2)
                                     DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                     CHI=1.0/(1.0+A)
                                     DCHI=A*B*BETA
                                     C=(SW*DCHI+DSW*CHI)/R
                                  ELSE
                                     C=A*B*BETA/R
                                  ENDIF
                                  CX=CX-C*XIJ
                                  CY=CY-C*YIJ
                                  CZ=CZ-C*ZIJ
                               END IF
                            END DO
                            CUX=HXSIM(LRES)-HXPROC(LRES)
                            DX(I)=DX(I)+BUX*CUX*CX
                            DY(I)=DY(I)+BUX*CUX*CY
                            DZ(I)=DZ(I)+BUX*CUX*CZ
                         END IF
                      END DO
                   END DO
                ENDIF
                !     ------------------------------------------------------------
                ! ... then hbond part ...
                BUX=-TWO*ALPHA*BETH*DXI*HXFH/REAL(NHXPRO)
                DO II=1,NSELO
                   DO JJ=1,NSELH
                      I=ISELO(II)
                      J=ISELH(JJ)
                      IRES=GETRES(I,IBASE,NREST)
                      JRES=GETRES(J,IBASE,NREST)
                      IF (ABS(IRES-JRES).LE.2) CONTINUE  ! cycle?
                      IF (HXPROC(IRES).NE.-1.OR.HXPROC(JRES).NE.-1) THEN
                         XIJ=X(I)-X(J)
                         YIJ=Y(I)-Y(J)
                         ZIJ=Z(I)-Z(J)
                         R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                         A=EXP(BETH*(R-HCUT))
                         B=1.0/((1.0+A)**2)
                         C=A*B/R
                      ENDIF
                      IF (HXPROC(IRES).NE.-1) THEN
                         IF (CONSDO(CTYPE(IRES),HXSIM(IRES),HXPROC(IRES))) THEN
                            CUX=(HXSIM(IRES)-HXPROC(IRES))*BUX
                            CX=CUX*C*XIJ
                            CY=CUX*C*YIJ
                            CZ=CUX*C*ZIJ
                            DX(I)=DX(I)+CX
                            DY(I)=DY(I)+CY
                            DZ(I)=DZ(I)+CZ
                            DX(J)=DX(J)-CX
                            DY(J)=DY(J)-CY
                            DZ(J)=DZ(J)-CZ
                         ENDIF
                      ENDIF
                      IF (HXPROC(JRES).NE.-1) THEN
                         IF (CONSDO(CTYPE(JRES),HXSIM(JRES),HXPROC(JRES))) THEN
                            CUX=(HXSIM(JRES)-HXPROC(JRES))*BUX
                            CX=CUX*C*XIJ
                            CY=CUX*C*YIJ
                            CZ=CUX*C*ZIJ
                            DX(I)=DX(I)+CX
                            DY(I)=DY(I)+CY
                            DZ(I)=DZ(I)+CZ
                            DX(J)=DX(J)-CX
                            DY(J)=DY(J)-CY
                            DZ(J)=DZ(J)-CZ
                         ENDIF
                      ENDIF
                   ENDDO
                ENDDO
             END IF
          ELSE
             EU=ZERO
             XIA=XI
          END IF
          !     ------------------------------------------------------------
          !       SMD PART (not well tested)
       ELSE IF (JSMD4) THEN
          IF (JEEF1) THEN
             !         use eef1 energies
             CALL WRNDIE(-1,'<RC4BIAS>', &
                  'Error: Forces not implemented for EEF1; analysis only')
          ELSE
             DXI=XI-XIA
#if KEY_ENSEMBLE==1
             IF (JNOEN4) THEN
                EU=ALPHA2*DXI**2
             ELSE
                EU=ALPHA2*REAL(NENSEM)*DXI**2
             ENDIF
#else /**/
             EU=ALPHA2*DXI**2
#endif 
             !     ------------------------------------------------------------
             !           Compute forces for all selected atoms
             !           first do native contact part
             BUX=-TWO*ALPHA*DXI*HXFC/REAL(NHXPRO)
             DO II=1,NSEL4
                I=ISEL4(II)
                DO LRES=1,NRES
                   IF (HXPROC(LRES).NE.-1) THEN
                      CX=0.0
                      CY=0.0
                      CZ=0.0
                      DO K=INDL(LRES),INDH(LRES)
                         L=IL4(K)
                         J=JL4(K)
                         IF (I .EQ. L) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            IF (JNONN.AND.(R.GT.HXCTON)) THEN
                               SW=DRCUT*(CTON2-R**2) &
                                    *(CTOF2+2.0*R**2-3.0*CTON2)
                               DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                               CHI=1.0/(1.0+A)
                               DCHI=A*B*BETA
                               C=(SW*DCHI+DSW*CHI)/R
                            ELSE
                               C=A*B*BETA/R
                            ENDIF
                            CX=CX+C*XIJ
                            CY=CY+C*YIJ
                            CZ=CZ+C*ZIJ
                         ELSE IF (I .EQ. J) THEN
                            XIJ=X(L)-X(J)
                            YIJ=Y(L)-Y(J)
                            ZIJ=Z(L)-Z(J)
                            R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                            A=EXP(BETA*(R-RCUT1))
                            B=1.0/((1.0+A)**2)
                            IF (JNONN.AND.(R.GT.HXCTON)) THEN
                               SW=DRCUT*(CTON2-R**2) &
                                    *(CTOF2+2.0*R**2-3.0*CTON2)
                               DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                               CHI=1.0/(1.0+A)
                               DCHI=A*B*BETA
                               C=(SW*DCHI+DSW*CHI)/R
                            ELSE
                               C=A*B*BETA/R
                            ENDIF
                            CX=CX-C*XIJ
                            CY=CY-C*YIJ
                            CZ=CZ-C*ZIJ
                         END IF
                      END DO
                      CUX=HXSIM(LRES)-HXPROC(LRES)
                      DX(I)=DX(I)+BUX*CUX*CX
                      DY(I)=DY(I)+BUX*CUX*CY
                      DZ(I)=DZ(I)+BUX*CUX*CZ
                   END IF
                END DO
             END DO
             !     ------------------------------------------------------------
             !           ...also calculate forces for amide H's
             IF (JNHCON) THEN
                DO II=1,NSELH
                   I=ISELH(II)
                   DO LRES=1,NRES
                      IF (HXPROC(LRES).NE.-1) THEN
                         CX=0.0
                         CY=0.0
                         CZ=0.0
                         DO K=INDL(LRES),INDH(LRES)
                            L=IL4(K)
                            J=JL4(K)
                            IF (I .EQ. L) THEN
                               XIJ=X(L)-X(J)
                               YIJ=Y(L)-Y(J)
                               ZIJ=Z(L)-Z(J)
                               R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                               A=EXP(BETA*(R-RCUT1))
                               B=1.0/((1.0+A)**2)
                               IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                  SW=DRCUT*(CTON2-R**2) &
                                       *(CTOF2+2.0*R**2-3.0*CTON2)
                                  DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                  CHI=1.0/(1.0+A)
                                  DCHI=A*B*BETA
                                  C=(SW*DCHI+DSW*CHI)/R
                               ELSE
                                  C=A*B*BETA/R
                               ENDIF
                               CX=CX+C*XIJ
                               CY=CY+C*YIJ
                               CZ=CZ+C*ZIJ
                            ELSE IF (I .EQ. J) THEN
                               XIJ=X(L)-X(J)
                               YIJ=Y(L)-Y(J)
                               ZIJ=Z(L)-Z(J)
                               R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                               A=EXP(BETA*(R-RCUT1))
                               B=1.0/((1.0+A)**2)
                               IF (JNONN.AND.(R.GT.HXCTON)) THEN
                                  SW=DRCUT*(CTON2-R**2) &
                                       *(CTOF2+2.0*R**2-3.0*CTON2)
                                  DSW=DRCUT*(-8.0*R**3+2.0*R*RO3RON)
                                  CHI=1.0/(1.0+A)
                                  DCHI=A*B*BETA
                                  C=(SW*DCHI+DSW*CHI)/R
                               ELSE
                                  C=A*B*BETA/R
                               ENDIF
                               CX=CX-C*XIJ
                               CY=CY-C*YIJ
                               CZ=CZ-C*ZIJ
                            END IF
                         END DO
                         CUX=HXSIM(LRES)-HXPROC(LRES)
                         DX(I)=DX(I)+BUX*CUX*CX
                         DY(I)=DY(I)+BUX*CUX*CY
                         DZ(I)=DZ(I)+BUX*CUX*CZ
                      END IF
                   END DO
                END DO
             ENDIF
             !     ------------------------------------------------------------
             ! ... then hbond part ...
             BUX=-TWO*ALPHA*BETH*DXI*HXFH/REAL(NHXPRO)
             DO II=1,NSELO
                DO JJ=1,NSELH
                   I=ISELO(II)
                   J=ISELH(JJ)
                   IRES=GETRES(I,IBASE,NREST)
                   JRES=GETRES(J,IBASE,NREST)
                   IF (ABS(IRES-JRES).LE.2) CONTINUE  ! cycle?
                   IF (HXPROC(IRES).NE.-1.OR.HXPROC(JRES).NE.-1) THEN
                      XIJ=X(I)-X(J)
                      YIJ=Y(I)-Y(J)
                      ZIJ=Z(I)-Z(J)
                      R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                      A=EXP(BETH*(R-HCUT))
                      B=1.0/((1.0+A)**2)
                      C=A*B/R
                   ENDIF
                   IF (HXPROC(IRES).NE.-1) THEN
                      CUX=(HXSIM(IRES)-HXPROC(IRES))*BUX
                      CX=CUX*C*XIJ
                      CY=CUX*C*YIJ
                      CZ=CUX*C*ZIJ
                      DX(I)=DX(I)+CX
                      DY(I)=DY(I)+CY
                      DZ(I)=DZ(I)+CZ
                      DX(J)=DX(J)-CX
                      DY(J)=DY(J)-CY
                      DZ(J)=DZ(J)-CZ
                   ENDIF
                   IF (HXPROC(JRES).NE.-1) THEN
                      CUX=(HXSIM(JRES)-HXPROC(JRES))*BUX
                      CX=CUX*C*XIJ
                      CY=CUX*C*YIJ
                      CZ=CUX*C*ZIJ
                      DX(I)=DX(I)+CX
                      DY(I)=DY(I)+CY
                      DZ(I)=DZ(I)+CZ
                      DX(J)=DX(J)-CX
                      DY(J)=DY(J)-CY
                      DZ(J)=DZ(J)-CZ
                   ENDIF
                ENDDO
             ENDDO
             XIA=MAX(ZERO,(XIA+GAMMA4*TIMEST))
          END IF
       ENDIF
    END IF
    !     ------------------------------------------------------------
    !     ------------------------------------------------------------
    !     Write iunj and iund output if required
    IF ((IOLEV.GE.0).OR.JNOEN4) THEN
       IF(QINIT) THEN
          IF(IUNDMP4.NE.-1)THEN
             J=0
             DO I=1,NRES
                IF (HXPROC(I).NE.-1) THEN
                   J=J+1
                   PHICAL(J)=REAL(I)
                END IF
             END DO
             WRITE(IUNDMP4,'(7(I9))')( NINT(PHICAL(II)),II=1,NHXPRO)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ4,'(I9,2(1X,F14.6),1X,E14.6)')HQSTEP &
                  ,XI,XIA,EU
             IF(IUNDMP4.NE.-1)THEN
                !                always dump total protection factor
                J=0
                DO I=1,NRES
                   IF (HXPROC(I).NE.-1) THEN
                      J=J+1
                      PHICAL(J)=HXSIM(I)
                   END IF
                END DO
                WRITE(IUNDMP4,'(7(F8.3,1X))') (PHICAL(II), II=1,NHXPRO)
                !                if writing detailed output also dump hbond + burial contrib
                IF (JSPLIT) THEN
                   !                    hbonds
                   J=0
                   DO I=1,NRES
                      IF (HXPROC(I).NE.-1) THEN
                         J=J+1
                         PHICAL(J)=HXFH*NNH(I)
                      END IF
                   END DO
                   WRITE(IUNDMP4,'(7(F8.3,1X))') &
                        (PHICAL(II),II=1,NHXPRO)
                   !                    burial
                   J=0
                   DO I=1,NRES
                      IF (HXPROC(I).NE.-1) THEN
                         J=J+1
                         PHICAL(J)=HXFC*NNT(I)
                      END IF
                   END DO
                   WRITE(IUNDMP4,'(7(F8.3,1X))') &
                        (PHICAL(II),II=1,NHXPRO)
                ENDIF
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    !     ------------------------------------------------------------
    !     check whether rho exceeds specified max. if so, exit and
    !     dump coordinates
    IF (XI .GT. XIMAX4) THEN
#if KEY_ENSEMBLE==1
       DO I=1,NRES
          IF (HXPROC(I).NE.-1) THEN
             HXSIM(I)=HXFC*NNT(I)+HXFH*NNH(I)
             IF (CONSDO(CTYPE(I),HXSIM(I),HXPROC(I))) THEN
                RHO=RHO+(HXSIM(I)-HXPROC(I))**2
             ENDIF
          END IF
       END DO
       RHO=RHO/REAL(NHXPRO)
       WRITE(TBUF,'(A,I4,A,F12.5)') ' RC4BIAS> NODE ', WHOIAM,  &
            ': MY XI = ', RHO
       CALL ENSPRN(OUTU,TBUF,ENSBFL)
       !         dump coords to node00.pdb, node01.pdb, etc...
       CALL ENSDMP()
#endif 
       IF (IOLEV.GT.0) WRITE(OUTU,*) 'RC4 = ',  &
            XI, '(MAX=', XIMAX4,')'
       CALL WRNDIE(-1,'<RC4BIAS>', &
            'Reaction coordinate RC4 exceeded maximum value.')
    ENDIF

    RETURN
  END SUBROUTINE RC4BIAS
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC5BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC5
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    !     Author: Emanuele Paci <07-02-97>
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    real(chm_real) FCN,AUXR,THETA,COSPHI
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES
    INTEGER I,J,L,L1,L2,K,II,JJ,LL,NSELL
    INTEGER IRES,JRES,LRES
    real(chm_real) XIJ,YIJ,ZIJ,R,RCUT1,TOTCON
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0, &
         XI,XIAMIN,XIAMAX,DXI,ALPHA,ALPHA2, &
         AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C,PSIM
    SAVE XI,XIAMIN,XIAMAX,RHO0,NSELL,RCUT1

    ALPHA=DFALPHA5
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC5BIAS> CALLING HQBM BIAS RC5'
          WRITE(OUTU,'(A,F19.8)')' RC5BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC5BIAS> THE RC5 PERTURBATION DRIVES THE SYSTEM'
          WRITE(OUTU,'(A)') &
               ' RC5BIAS> TOWARDS THE TARGET REACTION COORDINATE'
       END IF

       IF (.NOT.JRLD5) THEN
          IF (PRNLEV.GE.2) THEN
             WRITE(OUTU,'(A)') &
                  ' RC5BIAS> COORDINATES OF THE RC5 REFERENCE CONFIG:'
             DO II=1,NSEL5
                I = ISEL5(II)
                WRITE(OUTU,'(A9,I5,3F12.4)') &
                     'RC5BIAS> ',I,XCOMP(I),YCOMP(I),ZCOMP(I)
             END DO
          ENDIF

          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,'(A)') &
                  ' RC5BIAS> COORDINATES OF THE RC5 INITIAL CONFIG.:'
             DO II=1,NSEL5
                I = ISEL5(II)
                WRITE(OUTU,'(A9,I5,3F12.4)')  &
                     'RC5BIAS> ',I,X(I),Y(I),Z(I)
             END DO
          ENDIF
       END IF

       call chmalloc(FILE,'RC5BIAS','IL5',SIZEIJ,intg=IL5)
       call chmalloc(FILE,'RC5BIAS','JL5',SIZEIJ,intg=JL5)
       IF (.NOT.JRLD5) THEN
          L=0
          DO II=1,NSEL5-1
             DO JJ=II+1,NSEL5
                L=L+1
                IL5(L)=ISEL5(II)
                JL5(L)=ISEL5(JJ)
             END DO
          END DO
          NSELL=L
       ELSE
          READ(IUNJLS5,*)NSELL
          DO I=1,NSELL
             READ(IUNJLS5,*)IL5(I),JL5(I)
          END DO
       END IF

       RHO=0
       call chmalloc(FILE,'RC5BIAS','RIJ5',NSELL,crl=RIJ5)
       call chmalloc(FILE,'RC5BIAS','RCIJ5',NSELL,crl=RCIJ5)
       DO L=1,NSELL
          I=IL5(L)
          J=JL5(L)
          XCIJ=XCOMP(I)-XCOMP(J)
          YCIJ=YCOMP(I)-YCOMP(J)
          ZCIJ=ZCOMP(I)-ZCOMP(J)
          RCIJ5(L)=SQRT(XCIJ*XCIJ+YCIJ*YCIJ+ZCIJ*ZCIJ)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ5(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN5)) THEN
          RC5BL = NSELL*NENSEM
          call chmalloc('hqbm.src','RC5BIAS','RC5BUF',RC5BL,crl=RC5BUF)
          CALL ENSAVE(RIJ5,RC5BUF,NSELL)
       ENDIF
       DO L=1,NSELL
#endif 
          RHO = RHO+(RIJ5(L)-RCIJ5(L))**2
       END DO
       RHO = RHO/NSELL
       RHO0 = RHO
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC5BIAS> INITIAL DISTANCE FOR RC5 = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC5BIAS> TARGET DISTANCE FOR RC5 = ',RC5TAR
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' RC5BIAS> NUMBER OF DISTANCES = ',NSELL
       XI = RHO0
       IF (XI.GT.RC5TAR) THEN
          XIAMIN = RC5TAR
          XIAMAX = XI
       ELSE
          XIAMAX = RC5TAR
          XIAMIN = XI
       ENDIF
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F10.4)') &
            ' RC5BIAS> INITIAL CONSTRAINT MINIMUM = ',XIAMIN
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F10.4)') &
            ' RC5BIAS> INITIAL CONSTRAINT MAXIMUM = ',XIAMAX
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO L=1,NSELL
          I=IL5(L)
          J=JL5(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ5(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN5))  &
            CALL ENSAVE(RIJ5,RC5BUF,NSELL)
       DO L=1,NSELL
#endif 
          RHO = RHO + (RIJ5(L)-RCIJ5(L))**2
       END DO

       RHO = RHO/NSELL
       XI=RHO
    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    !HX:
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF ((XI .GT. XIAMAX) &
            .OR. (XI .LT. XIAMIN)) THEN
          IF (XI.GT.XIAMAX) THEN
             DXI=XI-XIAMAX
          ELSE
             DXI=XI-XIAMIN
          ENDIF
#if KEY_ENSEMBLE==1
          IF (JNOEN5) THEN
             EU=ALPHA2*DXI**2
          ELSE
             EU=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EU=ALPHA2*DXI**2
#endif 
          BUX=ALPHA*DXI
          BUX=TWO*BUX/REAL(NSELL)
          !              Compute forces
          DO L=1,NSELL
             I=IL5(L)
             J=JL5(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             AUX = BUX * (RIJ5(L)-RCIJ5(L))/RIJ5(L)
             DX(I)=DX(I)+AUX*XIJ
             DY(I)=DY(I)+AUX*YIJ
             DZ(I)=DZ(I)+AUX*ZIJ
             DX(J)=DX(J)-AUX*XIJ
             DY(J)=DY(J)-AUX*YIJ
             DZ(J)=DZ(J)-AUX*ZIJ
          END DO
       ELSE
          EU=ZERO
          !               XIA=XI
          IF (XI.LE.XIAMAX) THEN
             IF (XI.GT.RC5TAR) THEN
                XIAMAX = XI
             ELSE
                XIAMAX = RC5TAR
             ENDIF
          ENDIF
          IF (XI.GE.XIAMIN) THEN
             IF (XI.LT.RC5TAR) THEN
                XIAMIN = XI
             ELSE
                XIAMIN = RC5TAR
             ENDIF
          ENDIF
       END IF
    END IF

    IF (((IOLEV.GE.0).OR.JNOEN5).AND.(.NOT.QINIT)) THEN
       IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
          WRITE(IUNJUJ5,'(I9,4(1X,F10.3))')HQSTEP &
               ,XI,XIAMIN,XIAMAX,EU
       ENDIF
    ENDIF
    IF (XI .GT. XIMAX5) CALL WRNDIE(-1,'<RC5BIAS>', &
         'Reaction coordinate RC5 exceeded maximum value.')
    RETURN
  END SUBROUTINE RC5BIAS
  !
  !-----------------------------------------------------------------
  FUNCTION NOEDEV(SIM,LBND,HBND) result(noedev_1)
    !-----------------------------------------------------------------
    !     useful for flat-bottomed double harmonic potential
    !-----------------------------------------------------------------
    implicit none

    real(chm_real) SIM, LBND, HBND, noedev_1
    IF (SIM.GT.HBND) THEN
       NOEDEV_1 = SIM-HBND
    ELSE IF (SIM.LT.LBND) THEN
       NOEDEV_1 = SIM-LBND
    ELSE
       NOEDEV_1 = 0.0
    ENDIF
    RETURN
  END FUNCTION NOEDEV
  !----------------------------------------------------------------------
  SUBROUTINE RC6BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !     NOE BIAS
    !----------------------------------------------------------------------
    !     bias for RC6
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    real(chm_real) FCN,AUXR,THETA,COSPHI
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES
    INTEGER I,J,I1,I2,J1,J2,L,L1,L2,K,II,JJ,LL,NSELL
    INTEGER IRES,JRES,LRES
    real(chm_real) XIJ,YIJ,ZIJ,R,RCUT1,TOTCON
    real(chm_real) XXI,YYI,ZZI,XXJ,YYJ,ZZJ
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0, &
         XI,XIA,DXI,ALPHA,ALPHA2, &
         AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C,PSIM,PCOR
    SAVE XI,XIA,RHO0,NSELL,RCUT1

    ALPHA=DFALPHA6
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC6BIAS> CALLING HQBM BIAS RC6'
          WRITE(OUTU,'(A,F19.8)')' RC6BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC6BIAS> THE RC6 (NOE) PERTURBATION DRIVES THE SYSTEM'
          IF (JAWAY6) WRITE(OUTU,'(A)') &
               ' RC6BIAS>  AWAY FROM THE REFERENCE CONFIGURATION'
          IF (.NOT.JAWAY6) WRITE(OUTU,'(A)') &
               ' RC6BIAS>  THROUGH THE REFERENCE CONFIGURATION'
       END IF

       IF (JSIXP) THEN
          WRITE(OUTU,'(A)') &
               ' RC6BIAS> USING SIXTH POWER DISTANCE AVERAGING'
       ELSE IF (JLIN) THEN
          WRITE(OUTU,'(A)') &
               ' RC6BIAS> USING LINEAR DISTANCE AVERAGING'
       ELSE
          WRITE(OUTU,'(A)') &
               ' RC6BIAS> USING THIRD POWER DISTANCE AVERAGING'
       ENDIF
       READ(IUNNOE,*)NSELL
       call chmalloc(FILE,'RC6BIAS','IL6',NSELL,intg=IL6)
       call chmalloc(FILE,'RC6BIAS','JL6',NSELL,intg=JL6)
       call chmalloc(FILE,'RC6BIAS','LBOUN6',NSELL,crl=LBOUN6)
       call chmalloc(FILE,'RC6BIAS','UBOUN6',NSELL,crl=UBOUN6)
       DO I=1,NSELL
          READ(IUNNOE,*)IL6(I),JL6(I),LBOUN6(I),UBOUN6(I)
       END DO

       RHO=0
       call chmalloc(FILE,'RC6BIAS','RIJ6',NSELL,crl=RIJ6)
       DO L=1,NSELL
          I=IL6(L)
          J=JL6(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ6(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN6)) THEN
          NOEBL = NSELL*NENSEM
          call chmalloc('hqbm.src','RC6BIAS','NOEBUF',NOEBL,crl=NOEBUF)
          IF (JSIXP) THEN
             CALL ENSAV6(RIJ6,NOEBUF,NSELL)
          ELSE IF (JLIN) THEN
             CALL ENSAVE(RIJ6,NOEBUF,NSELL)
          ELSE
             CALL ENSAV3(RIJ6,NOEBUF,NSELL)
          ENDIF
       ENDIF
       DO L=1,NSELL
#endif 
          RHO = RHO+ NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L))**2
       END DO
       RHO = RHO/NSELL
       IF (JRZERO6) THEN
          RHO0 = 0.0
       ELSE
          RHO0 = RHO
       ENDIF

       IF ((IOLEV.GT.0).OR.JNOEN6) THEN
          WRITE(OUTU,'(A,I6,A)') ' NOEBIAS> ', NSELL,  &
               ' NOE CONSTRAINTS READ IN:'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          WRITE(OUTU,'(A6,A6,A8,A8,A8)') 'I','J','LOWER','UPPER', &
               'CURRENT'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          DO I=1,NSELL
             IF (RIJ6(I).GT.UBOUN6(I)+1.0) THEN
                WRITE(OUTU,'(I6,I6,F8.3,F8.3,F8.3,1X,A)')  &
                     IL6(I),JL6(I),LBOUN6(I),UBOUN6(I),RIJ6(I),'**'
             ELSE IF (RIJ6(I).GT.UBOUN6(I)+0.5) THEN
                WRITE(OUTU,'(I6,I6,F8.3,F8.3,F8.3,1X,A)')  &
                     IL6(I),JL6(I),LBOUN6(I),UBOUN6(I),RIJ6(I),'*'
             ELSE
                WRITE(OUTU,'(I6,I6,F8.3,F8.3,F8.3)')  &
                     IL6(I),JL6(I),LBOUN6(I),UBOUN6(I),RIJ6(I)
             ENDIF
          END DO
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
       ENDIF

       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC6BIAS> INITIAL DISTANCE FOR RC6/NOE = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' RC6BIAS> NUMBER OF DISTANCE CONSTRAINTS = ',NSELL
       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO L=1,NSELL
          I=IL6(L)
          J=JL6(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          RIJ6(L)=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN6)) THEN
          IF (JSIXP) THEN
             CALL ENSAV6(RIJ6,NOEBUF,NSELL)
          ELSE IF (JLIN) THEN
             CALL ENSAVE(RIJ6,NOEBUF,NSELL)
          ELSE
             CALL ENSAV3(RIJ6,NOEBUF,NSELL)
          ENDIF
       ENDIF
       DO L=1,NSELL
#endif 
          RHO = RHO+ NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L))**2
       END DO
       RHO = RHO/NSELL
       XI=RHO
    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    !HX:
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF (.NOT. JSMD6) THEN
          IF ((((.NOT. JAWAY6) .AND. (XI .GT. XIA)) &
               .OR. ((JAWAY6) .AND. (XI .LT. XIA))).OR.JHQFIX6) THEN
             DXI=XI-XIA
#if KEY_ENSEMBLE==1
             IF (JNOEN6) THEN
                EU=ALPHA2*DXI**2
             ELSE
                EU=ALPHA2*REAL(NENSEM)*DXI**2
             ENDIF
#else /**/
             EU=ALPHA2*DXI**2
#endif 
             BUX=ALPHA*DXI
             BUX=TWO*BUX/(NSELL)
             !        Compute forces
             DO L=1,NSELL
                I=IL6(L)
                J=JL6(L)
                XIJ=X(I)-X(J)
                YIJ=Y(I)-Y(J)
                ZIJ=Z(I)-Z(J)
#if KEY_ENSEMBLE==1
                RNIJ = SQRT(XIJ**2+YIJ**2+ZIJ**2)
                IF (JNOEN6) THEN
                   PCOR = 1.0
                ELSE IF (JSIXP) THEN
                   PCOR = (RIJ6(L)/RNIJ)**7
                ELSE IF (JLIN) THEN
                   PCOR = 1.0
                ELSE
                   PCOR = (RIJ6(L)/RNIJ)**4
                ENDIF
                AUX=BUX*PCOR*(NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L)))/RNIJ
#else /**/
                AUX = BUX*(NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L)))/RIJ6(L)
#endif 
                DX(I)=DX(I)+AUX*XIJ
                DY(I)=DY(I)+AUX*YIJ
                DZ(I)=DZ(I)+AUX*ZIJ
                DX(J)=DX(J)-AUX*XIJ
                DY(J)=DY(J)-AUX*YIJ
                DZ(J)=DZ(J)-AUX*ZIJ
             END DO
          ELSE
             EU=ZERO
             XIA=XI
          END IF

       ELSE IF (JSMD6) THEN

          DXI=XI-XIA
#if KEY_ENSEMBLE==1
          IF (JNOEN6) THEN
             EU=ALPHA2*DXI**2
          ELSE
             EU=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EU=ALPHA2*DXI**2
#endif 
          BUX=ALPHA*DXI
          BUX=TWO*BUX/(NSELL)
          !           Compute forces
          DO L=1,NSELL
             I=IL6(L)
             J=JL6(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
#if KEY_ENSEMBLE==1
             RNIJ = SQRT(XIJ**2+YIJ**2+ZIJ**2)
             IF (JNOEN6) THEN
                PCOR = 1.0
             ELSE IF (JSIXP) THEN
                PCOR = (RIJ6(L)/RNIJ)**7
             ELSE IF (JLIN) THEN
                PCOR = 1.0
             ELSE
                PCOR = (RIJ6(L)/RNIJ)**4
             ENDIF
             AUX=BUX*PCOR*(NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L)))/RNIJ
#else /**/
             AUX = BUX*(NOEDEV(RIJ6(L),LBOUN6(L),UBOUN6(L)))/RIJ6(L)
#endif 
             DX(I)=DX(I)+AUX*XIJ
             DY(I)=DY(I)+AUX*YIJ
             DZ(I)=DZ(I)+AUX*ZIJ
             DX(J)=DX(J)-AUX*XIJ
             DY(J)=DY(J)-AUX*YIJ
             DZ(J)=DZ(J)-AUX*ZIJ
          END DO

          XIA=XIA+GAMMA6*TIMEST

       END IF
    END IF

    IF ((IOLEV.GE.0).OR.JNOEN6) THEN
       IF(QINIT) THEN
          IF(IUNDMP6.NE.-1)THEN
             WRITE(IUNDMP6,'(7(I9))')( IL6(II),II=1,NSELL)
             WRITE(IUNDMP6,'(7(I9))')( JL6(II),II=1,NSELL)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ6,'(I9,3(1X,F14.6))')HQSTEP &
                  ,XI,XIA,EU
             IF(IUNDMP6.NE.-1)THEN
                !                always dump total protection factor
                WRITE(IUNDMP6,'(7(F8.3,1X))')( RIJ6(II),II=1,NSELL)
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    IF (XI .GT. XIMAX6) CALL WRNDIE(-1,'<RC6BIAS>', &
         'Reaction coordinate RC6/NOE exceeded maximum value.')
    RETURN
  END SUBROUTINE RC6BIAS
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC7BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !     RDC BIAS
    !----------------------------------------------------------------------
    !     bias for RC7
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    implicit none
    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    ! -------------------
    !     to be implemented
    ! -------------------
  END SUBROUTINE RC7BIAS
  !
  !----------------------------------------------------------------------
#if KEY_ENSEMBLE==1
  SUBROUTINE RC8BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !     ORDER PARAMETER (S2) BIAS
    !----------------------------------------------------------------------
    !     bias for RC8
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
    use string
    use comand
#if KEY_PARALLEL==1
    use parallel
#endif 
    use ensemble
    use corman_mod, only: corcom
    use memory
    implicit none
    real(chm_real) EU
    real(chm_real) FCN,AUXR,THETA,COSPHI
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES
    INTEGER I,J,I1,I2,J1,J2,L,L1,L2,K,II,JJ,LL,NSELL
    INTEGER IL(SIZEIJ),JL(SIZEIJ)
    INTEGER IRES,JRES,LRES
    real(chm_real) XIJ,YIJ,ZIJ,S2(SIZEIJ),RIJ(SIZEIJ),R,RCUT1
    real(chm_real) BXIJ(SIZEIJ),BYIJ(SIZEIJ),BZIJ(SIZEIJ)
    real(chm_real) XCIJ,YCIJ,ZCIJ,S2EXPT(SIZEIJ),RNIJ
    real(chm_real) RHO,RHO0, &
         XI,XIA,DXI,ALPHA,ALPHA2, &
         AUX,BUX,CUX
    real(chm_real) CX,CY,CZ,A,B,C,PSIM,PCOR
    SAVE XI,XIA,RHO0,S2EXPT,IL,JL,NSELL,RCUT1,RIJ

    ALPHA=DFALPHA8
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC8BIAS> CALLING HQBM BIAS RC8 (S2)'
          WRITE(OUTU,'(A,F19.8)')' RC8BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC8BIAS> NOTE THAT THIS COMMAND AUTOMATICALLY INVOKES '
          WRITE(OUTU,'(A)') &
               ' RC8BIAS> "SHAKE BOND PARAM" FOLLOWED BY "COOR SHAKE"'
          WRITE(OUTU,'(A)') &
               ' RC8BIAS> SO BOND LENGTHS TAKEN FROM PARAMETERS'
       END IF
       !        and we make good on our promise ...
       WRITE(COMLYN,'(A)') 'BOND PARAM'
       COMLEN = STRLNG(COMLYN)
       CALL SHKCOM(COMLYN, COMLEN)
       WRITE(COMLYN,'(A)') 'SHAKE'
       COMLEN = STRLNG(COMLYN)
       CALL CORCOM(COMLYN, COMLEN)

       READ(IUNS2,*)NSELL
       DO I=1,NSELL
          READ(IUNS2,*)IL(I),JL(I),S2EXPT(I)
       END DO

       DO L=1,NSELL
          I=IL(L)
          J=JL(L)
          BXIJ(L)=X(I)-X(J)
          BYIJ(L)=Y(I)-Y(J)
          BZIJ(L)=Z(I)-Z(J)
          !           this is the final (fixed) value of RIJ that will be
          !           used for the whole simulation (since SHAKE BOND in use)
          RIJ(L)=SQRT(BXIJ(L)**2+BYIJ(L)**2+BZIJ(L)**2)
       END DO
       S2BL = NSELL*NENSEM
       call chmalloc('hqbm.src','RC8BIAS','S2XBUF',S2BL,crl=S2XBUF)
       call chmalloc('hqbm.src','RC8BIAS','S2YBUF',S2BL,crl=S2YBUF)
       call chmalloc('hqbm.src','RC8BIAS','S2ZBUF',S2BL,crl=S2ZBUF)
       CALL ENSS2(BXIJ,BYIJ,BZIJ,S2XBUF,S2YBUF,S2ZBUF,NSELL,RIJ,S2)
       RHO=0
       IF (.NOT. JRZERO8) THEN
          DO L=1,NSELL
             RHO = RHO+ (S2(L)-S2EXPT(L))**2
          END DO
          RHO = RHO/NSELL
       ENDIF
       RHO0 = RHO
       IF (IOLEV.GT.0) THEN
          WRITE(OUTU,'(A,I6,A)') ' RC8BIAS> ', NSELL,  &
               ' ORDER PARAMETER CONSTRAINTS READ IN:'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          WRITE(OUTU,'(A6,A6,3X,A8,A8)') 'I','J','S2(EXPT)','S2(SIM)'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          DO I=1,NSELL
             WRITE(OUTU,'(I6,I6,3X,F8.3,F8.3)')  &
                  IL(I),JL(I),S2EXPT(I),S2(I)
          END DO
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
       ENDIF
       IF (PRNLEV.GE.2) THEN
          IF (JRZERO8) WRITE(OUTU,'(A)') &
               ' RC8BIAS> YOU CHOSE TO KEEP THE PROTEIN AROUND RHO=0', &
               ' RC8BIAS> FOR RC8, AND NOT AROUND THE INITIAL RHO; THUS:'
          WRITE(OUTU,'(A,F13.4)') &
               ' RC8BIAS> INITIAL DISTANCE FOR RC8/S2 = ',RHO0
          WRITE(OUTU,'(A,I7)') &
               ' RC8BIAS> NUMBER OF ORDER PARAMETER CONSTRAINTS = ',NSELL
       ENDIF
       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO L=1,NSELL
          I=IL(L)
          J=JL(L)
          BXIJ(L)=X(I)-X(J)
          BYIJ(L)=Y(I)-Y(J)
          BZIJ(L)=Z(I)-Z(J)
       END DO
       CALL ENSS2(BXIJ,BYIJ,BZIJ,S2XBUF,S2YBUF,S2ZBUF,NSELL,RIJ,S2)
       DO L=1,NSELL
          RHO = RHO+ (S2(L)-S2EXPT(L))**2
       END DO
       RHO = RHO/NSELL
       XI=RHO
    END IF
    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF ((XI .GT. XIA).OR.JHQFIX8) THEN
          DXI=XI-XIA
          EU=ALPHA2*REAL(NENSEM)*DXI**2
          BUX=ALPHA*DXI
          BUX=TWELVE*BUX/REAL(NSELL)
          !        Compute forces
          !        N.B. BXIJ etc are changed by ENSS2 from cartesian components
          !        to the appropriate factors for force calculation!
          DO L=1,NSELL
             I=IL(L)
             J=JL(L)
             AUX = BUX*(S2(L)-S2EXPT(L))/(RIJ(L)**4)
             DX(I)=DX(I)+AUX*BXIJ(L)
             DY(I)=DY(I)+AUX*BYIJ(L)
             DZ(I)=DZ(I)+AUX*BZIJ(L)
             DX(J)=DX(J)-AUX*BXIJ(L)
             DY(J)=DY(J)-AUX*BYIJ(L)
             DZ(J)=DZ(J)-AUX*BZIJ(L)
          END DO
       ELSE
          EU=ZERO
          XIA=XI
       END IF
    END IF

    IF (IOLEV.GE.0) THEN
       IF(QINIT) THEN
          IF(IUNDMP8.NE.-1)THEN
             WRITE(IUNDMP8,'(7(I9))')( IL(II),II=1,NSELL)
             WRITE(IUNDMP8,'(7(I9))')( JL(II),II=1,NSELL)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ8,'(I9,3(1X,F14.6))')HQSTEP &
                  ,XI,XIA,EU
             CALL GFLUSH(IUNJUJ8)
             IF(IUNDMP8.NE.-1)THEN
                WRITE(IUNDMP8,'(7(F8.3,1X))')( S2(II),II=1,NSELL)
                CALL GFLUSH(IUNDMP8)
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    IF (XI .GT. XIMAX8) CALL WRNDIE(-1,'<RC8BIAS>', &
         'Reaction coordinate RC8/S2 exceeded maximum value.')
    RETURN
  END SUBROUTINE RC8BIAS
#endif 
  !
  !----------------------------------------------------------------------
  SUBROUTINE RC9BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !     SCALAR COUPLING BIAS
    !----------------------------------------------------------------------
    !     bias for RC9
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use eef1_mod
    use consta
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    INTEGER PIRES
    INTEGER I,J,K,L,II,NSELL
    INTEGER IOS
    INTEGER IRES,JRES,LRES
    !     karplus parameters
    real(chm_real) KA(MAXSEL),KB(MAXSEL),KC(MAXSEL),KOFF(MAXSEL)
    !     P,Q,R,S sets defining dihedrals for 3J couplings
    INTEGER TI(MAXSEL),TJ(MAXSEL),TK(MAXSEL),TL(MAXSEL)
    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ,AX,AY,AZ,BX,BY,BZ
    real(chm_real) RA2,RB2,RG2,RG,RGR,RA2R,RB2R,RABR,CP,SP,CD,SD,CPD
    real(chm_real) SPD,DJDP,FG,HG
    real(chm_real) RCUT1
    real(chm_real) FGA,HGB,GAA,GBB,DTFX,DTFY,DTFZ,DTGX,DTGY,DTGZ
    real(chm_real) DTHX,DTHY,DTHZ,DFX,DFY,DFZ,DGX,DGY,DGZ,DHX,DHY,DHZ
    real(chm_real) J3(MAXSEL),SIMJ(MAXSEL)
    real(chm_real) RHO,RHO0, &
         XI,XIA,DXI,ALPHA,ALPHA2, &
         AUX,BUX
    real(chm_real) A,B,C
    real(chm_real), parameter :: RXMIN=0.005D0,RXMIN2=0.000025D0
    SAVE XI,XIA,RHO0,J3,TI,TJ,TK,TL,KA,KB,KC,KOFF,NSELL,RCUT1

    ALPHA=DFALPHA9
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC9BIAS> CALLING HQBM BIAS RC9'
          WRITE(OUTU,'(A,F19.8)')' RC9BIAS> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' RC9BIAS> THE RC9 (J3) PERTURBATION DRIVES THE SYSTEM'
          WRITE(OUTU,'(A)') &
               ' RC9BIAS>  THROUGH THE REFERENCE CONFIGURATION'
          WRITE(OUTU,*)' RC9BIAS> READING SCALAR COUPLING CONSTRAINTS'
          WRITE(OUTU,*) '  '
          WRITE(OUTU,'(A24,A36,A9)') '< TORSION ATOM INDICES >', &
               '<   KARPLUS EQUATION COEFFICIENTS  >', &
               '<  EXPT >'
          WRITE(OUTU,'(4(A5,1X),(5(A8,1X)))') 'I', 'J', 'K', 'L', &
               'A', 'B', 'C', 'OFFSET', 'J3'
       END IF

       I=1
       DO
          READ(IUNJ3,*,iostat=IOS)  &
               TI(I),TJ(I),TK(I),TL(I),KA(I),KB(I), &
               KC(I),KOFF(I),J3(I)
          IF (IOS /= 0) EXIT
          IF (PRNLEV.GE.2) WRITE(OUTU,'(4(I5,1X),(5(F8.3,1X)))')  &
               TI(I),TJ(I),TK(I),TL(I),KA(I),KB(I),KC(I),KOFF(I),J3(I)
          I=I+1
       ENDDO
       NSELL = I-1

       RHO=0
       DO II=1,NSELL
          I=TI(II)
          J=TJ(II)
          K=TK(II)
          L=TL(II)
          ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
          FX=X(I)-X(J)
          FY=Y(I)-Y(J)
          FZ=Z(I)-Z(J)
          GX=X(J)-X(K)
          GY=Y(J)-Y(K)
          GZ=Z(J)-Z(K)
          HX=X(L)-X(K)
          HY=Y(L)-Y(K)
          HZ=Z(L)-Z(K)
          ! A=F^G, B=H^G
          AX=FY*GZ-FZ*GY
          AY=FZ*GX-FX*GZ
          AZ=FX*GY-FY*GX
          BX=HY*GZ-HZ*GY
          BY=HZ*GX-HX*GZ
          BZ=HX*GY-HY*GX
          ! RG=|G|, RGR=1/|G|
          RA2=AX*AX+AY*AY+AZ*AZ
          RB2=BX*BX+BY*BY+BZ*BZ
          RG2=GX*GX+GY*GY+GZ*GZ
          RG=SQRT(RG2)
          ! Warnings have been simplified.
          !          IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          !            IF((WRNLEV.GE.5)) THEN
          !              WRITE(OUTU,20) IPHI,I,J,K,L
          !   20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/
          !     1          ' derivatives may be affected for atoms:',4I5)
          !            ENDIF
          !            GOTO 160
          !          ENDIF
          !
          RGR=ONE/RG
          RA2R=ONE/RA2
          RB2R=ONE/RB2
          RABR=SQRT(RA2R*RB2R)
          ! CP=cos(phi)
          CP=(AX*BX+AY*BY+AZ*BZ)*RABR
          ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
          ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
          SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
          CD=COS(KOFF(II)*DEGRAD)
          SD=SIN(KOFF(II)*DEGRAD)
          CPD=CP*CD-SP*SD
          SIMJ(II) = KA(II)*CPD*CPD + KB(II)*CPD + KC(II)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN9)) THEN
          J3BL = NSELL*NENSEM
          call chmalloc('hqbm.src','RC9BIAS','J3BUF',J3BL,crl=J3BUF)
          CALL ENSAVE(SIMJ,J3BUF,NSELL)
       ENDIF
       DO II=1,NSELL
#endif 
          RHO = RHO+(SIMJ(II)-J3(II))**2
       END DO
       RHO = RHO/NSELL
       IF (JRZERO9) THEN
          RHO0 = 0.0
       ELSE
          RHO0 = RHO
       ENDIF

       IF ((IOLEV.GT.0).OR.JNOEN9) THEN
          WRITE(OUTU,'(A,I6,A)') ' J3BIAS> ', NSELL,  &
               ' J3 CONSTRAINTS READ IN:'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          WRITE(OUTU,'(A24,A9,A9)') '< TORSION ATOM INDICES >', &
               '< JEXPT >', '< JSIM  >'
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
          DO II=1,NSELL
             WRITE(OUTU,'(4(I5,1X),2(F8.3,1X))') &
                  TI(II),TJ(II),TK(II),TL(II),J3(II),SIMJ(II)
          END DO
          WRITE(OUTU,'(A)')  &
               '---------------------------------------------------------'
       ENDIF

       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' RC9BIAS> INITIAL DISTANCE FOR RC9/J3 = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' RC9BIAS> NUMBER OF J3 CONSTRAINTS = ',NSELL
       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0
       DO II=1,NSELL
          I=TI(II)
          J=TJ(II)
          K=TK(II)
          L=TL(II)
          ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
          FX=X(I)-X(J)
          FY=Y(I)-Y(J)
          FZ=Z(I)-Z(J)
          GX=X(J)-X(K)
          GY=Y(J)-Y(K)
          GZ=Z(J)-Z(K)
          HX=X(L)-X(K)
          HY=Y(L)-Y(K)
          HZ=Z(L)-Z(K)
          ! A=F^G, B=H^G
          AX=FY*GZ-FZ*GY
          AY=FZ*GX-FX*GZ
          AZ=FX*GY-FY*GX
          BX=HY*GZ-HZ*GY
          BY=HZ*GX-HX*GZ
          BZ=HX*GY-HY*GX
          ! RG=|G|, RGR=1/|G|
          RA2=AX*AX+AY*AY+AZ*AZ
          RB2=BX*BX+BY*BY+BZ*BZ
          RG2=GX*GX+GY*GY+GZ*GZ
          RG=SQRT(RG2)
          ! Warnings have been simplified.
          !          IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          !            IF((WRNLEV.GE.5)) THEN
          !              WRITE(OUTU,20) IPHI,I,J,K,L
          !   20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/
          !     1          ' derivatives may be affected for atoms:',4I5)
          !            ENDIF
          !            GOTO 160
          !          ENDIF
          !
          RGR=ONE/RG
          RA2R=ONE/RA2
          RB2R=ONE/RB2
          RABR=SQRT(RA2R*RB2R)
          ! CP=cos(phi)
          CP=(AX*BX+AY*BY+AZ*BZ)*RABR
          ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
          ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
          SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
          CD=COS(KOFF(II)*DEGRAD)
          SD=SIN(KOFF(II)*DEGRAD)
          CPD=CP*CD-SP*SD
          SIMJ(II) = KA(II)*CPD*CPD + KB(II)*CPD + KC(II)
#if KEY_ENSEMBLE==1
       END DO
       IF ((NENSEM.NE.1).AND.(.NOT.JNOEN9)) THEN
          CALL ENSAVE(SIMJ,J3BUF,NSELL)
       ENDIF
       DO II=1,NSELL
#endif 
          RHO = RHO+(SIMJ(II)-J3(II))**2
       END DO
       RHO = RHO/NSELL
       XI = RHO
    ENDIF
    !
    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...

    IF (QFORCE.AND.ALPHA.NE.0.0) THEN
       IF (XI .GT. XIA) THEN
          DXI=XI-XIA
#if KEY_ENSEMBLE==1
          IF (JNOEN9) THEN
             EU=ALPHA2*DXI**2
          ELSE
             EU=ALPHA2*REAL(NENSEM)*DXI**2
          ENDIF
#else /**/
          EU=ALPHA2*DXI**2
#endif 
          BUX=ALPHA*DXI
          BUX=TWO*BUX/(NSELL)
          !        Compute forces
          DO II=1,NSELL
             I=TI(II)
             J=TJ(II)
             K=TK(II)
             L=TL(II)
             ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
             FX=X(I)-X(J)
             FY=Y(I)-Y(J)
             FZ=Z(I)-Z(J)
             GX=X(J)-X(K)
             GY=Y(J)-Y(K)
             GZ=Z(J)-Z(K)
             HX=X(L)-X(K)
             HY=Y(L)-Y(K)
             HZ=Z(L)-Z(K)
             ! A=F^G, B=H^G
             AX=FY*GZ-FZ*GY
             AY=FZ*GX-FX*GZ
             AZ=FX*GY-FY*GX
             BX=HY*GZ-HZ*GY
             BY=HZ*GX-HX*GZ
             BZ=HX*GY-HY*GX
             ! RG=|G|, RGR=1/|G|
             RA2=AX*AX+AY*AY+AZ*AZ
             RB2=BX*BX+BY*BY+BZ*BZ
             RG2=GX*GX+GY*GY+GZ*GZ
             RG=SQRT(RG2)
             ! Warnings have been simplified.
             !          IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
             !            IF((WRNLEV.GE.5)) THEN
             !              WRITE(OUTU,20) IPHI,I,J,K,L
             !   20         FORMAT(' EPHI: WARNING.  dihedral',I5,' is almost linear.'/
             !     1          ' derivatives may be affected for atoms:',4I5)
             !            ENDIF
             !            GOTO 160
             !          ENDIF
             !
             RGR=ONE/RG
             RA2R=ONE/RA2
             RB2R=ONE/RB2
             RABR=SQRT(RA2R*RB2R)
             ! CP=cos(phi)
             CP=(AX*BX+AY*BY+AZ*BZ)*RABR
             ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
             ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
             SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
             CD=COS(KOFF(II)*DEGRAD)
             SD=SIN(KOFF(II)*DEGRAD)
             CPD=CP*CD-SP*SD
             SPD=SP*CD+CP*SD
             DJDP = -SPD*(2.0*KA(II)*CPD + KB(II))
             AUX = BUX * (SIMJ(II)-J3(II)) * DJDP
             !
             ! Compute derivatives wrt catesian coordinates.
             ! this section is for first derivatives only.
             !
             ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
             !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
             FG=FX*GX+FY*GY+FZ*GZ
             HG=HX*GX+HY*GY+HZ*GZ
             FGA=FG*RA2R*RGR
             HGB=HG*RB2R*RGR
             GAA=-RA2R*RG
             GBB=RB2R*RG
             ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
             DTFX=GAA*AX
             DTFY=GAA*AY
             DTFZ=GAA*AZ
             DTGX=FGA*AX-HGB*BX
             DTGY=FGA*AY-HGB*BY
             DTGZ=FGA*AZ-HGB*BZ
             DTHX=GBB*BX
             DTHY=GBB*BY
             DTHZ=GBB*BZ
             ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
             DFX=AUX*DTFX
             DFY=AUX*DTFY
             DFZ=AUX*DTFZ
             DGX=AUX*DTGX
             DGY=AUX*DTGY
             DGZ=AUX*DTGZ
             DHX=AUX*DTHX
             DHY=AUX*DTHY
             DHZ=AUX*DTHZ
             ! Distribute over Ri.
             DX(I)=DX(I)+DFX
             DY(I)=DY(I)+DFY
             DZ(I)=DZ(I)+DFZ
             DX(J)=DX(J)-DFX+DGX
             DY(J)=DY(J)-DFY+DGY
             DZ(J)=DZ(J)-DFZ+DGZ
             DX(K)=DX(K)-DHX-DGX
             DY(K)=DY(K)-DHY-DGY
             DZ(K)=DZ(K)-DHZ-DGZ
             DX(L)=DX(L)+DHX
             DY(L)=DY(L)+DHY
             DZ(L)=DZ(L)+DHZ
             !
             ! 160            CONTINUE
             !
          END DO
       ELSE
          EU=ZERO
          XIA=XI
       END IF

    END IF

    IF ((IOLEV.GE.0).OR.JNOEN9) THEN
       IF(QINIT) THEN
          IF(IUNDMP9.NE.-1)THEN
             WRITE(IUNDMP9,'(7(I9))')( TI(II),II=1,NSELL)
             WRITE(IUNDMP9,'(7(I9))')( TJ(II),II=1,NSELL)
             WRITE(IUNDMP9,'(7(I9))')( TK(II),II=1,NSELL)
             WRITE(IUNDMP9,'(7(I9))')( TL(II),II=1,NSELL)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ9,'(I9,3(1X,F14.6))')HQSTEP &
                  ,XI,XIA,EU
             IF(IUNDMP9.NE.-1)THEN
                WRITE(IUNDMP9,'(7(F8.3,1X))')( SIMJ(II),II=1,NSELL)
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    IF (XI .GT. XIMAX9) THEN
       WRITE(OUTU,*) '   RC9 = ', XI
       WRITE(OUTU,*) 'XIMAX9 = ', XIMAX9
       CALL WRNDIE(-1,'<RC9BIAS>', &
            'Reaction coordinate RC9/J3 exceeded maximum value.')
    ENDIF
    RETURN
  END SUBROUTINE RC9BIAS

#if KEY_ENSEMBLE==1
  !----------------------------------------------------------------------
  SUBROUTINE RC10BIAS(EU,X,Y,Z,DX,DY,DZ,NATOMX,QFORCE,QINIT)
    !----------------------------------------------------------------------
    !     bias for RC10
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !     QFORCE - calculate forces? (not needed for analysis)
    !     QINIT - first time routine called - do we need to initialize?
    !
    use coordc
    use number
    use contrl
    use stream
    use psf
    use reawri
    use ensemble
    use memory
    implicit none
    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    LOGICAL QFORCE,QINIT
    !     MAXSEL is defined in hqbm.f90
    !     all local vbls from here -------------------------
    !
    INTEGER IPSI(MAXPSI),JPSI(MAXPSI),PIRES
    !,EXPNPSI
    INTEGER I,J,L,L1,L2,K,II,JJ,LL
    INTEGER IOS
    INTEGER EXPNPSI
    INTEGER IRES,JRES,LRES,TCONI,CNOW
    real(chm_real) XIJ,YIJ,ZIJ,R,TOTCON
    real(chm_real) XCIJ,YCIJ,ZCIJ,RNIJ
    real(chm_real) RHO,RHO0,XI,XIA,DXI,ALPHA,ALPHA2,AUX,BUX,CUX
    !      real(chm_real) ARHO,ARHO0,APSIs,APSIe,AXI,AXIA
    real(chm_real) CX,CY,CZ,A,B,C,PSIM
    real(chm_real) PSI(MAXPSI),RPSI(MAXPSI),PSICAL(MAXPSI)
    SAVE XI,XIA,RHO0,PSI,RPSI,IPSI,JPSI,EXPNPSI

    ALPHA=DFALPHA10
    ALPHA2=ALPHA/TWO
    EU=0.0
    !     The first time HQBIAS is called, compute the distance RHO between
    !     the reference structure (COMP) and the initial configuration.
    !     The reaction coordinate is RHO(t).
    !     Do the first time that this routine is called
    !
    !C ****************************
    IF (QINIT) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' RC10BIAS> CALLING HQBM BIAS'
          WRITE(OUTU,'(A,F19.8)')' RC10BIAS> ALPHA10=',ALPHA
       END IF

       RHO=0.0

       !         RCUT1=RCUT+TOL
       EXPNPSI=0
       I=1
       DO
          READ(IUNPSI,*,iostat=IOS)IPSI(I),JPSI(I),PSI(I)
          IF (IOS /= 0) EXIT
          IF (I.GT.MAXPSI) CALL WRNDIE(-1,'<RC10BIAS>', &
               'Number of psi values exceeds MAXPSI.')
          RPSI(I)=SQRT((X(IPSI(I))-X(JPSI(I)))**2 &
               +(Y(IPSI(I))-Y(JPSI(I)))**2 &
               +(Z(IPSI(I))-Z(JPSI(I)))**2)+TOL
          IF (PRNLEV.GE.2) THEN
             WRITE(OUTU,*) ' HQBM> RESIDUE ',IPSI(I), JPSI(I),  &
                  ' = ' ,PSI(I), RPSI(I)
          ENDIF
          EXPNPSI=EXPNPSI+1
          I=I+1
       ENDDO

       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') &
            ' RC10BIAS> TOTAL NUMBER OF EXPERIMENTAL PSI',EXPNPSI

       PSIBL = EXPNPSI*NENSEM
       call chmalloc('hqbm.src','RC10BIAS','PSIBUF',PSIBL,crl=PSIBUF)
       RHO=0.0
       IF (.NOT. JRZERO10) THEN
          DO I=1,EXPNPSI
             RNIJ=SQRT((X(IPSI(I))-X(JPSI(I)))**2 &
                  +(Y(IPSI(I))-Y(JPSI(I)))**2 &
                  +(Z(IPSI(I))-Z(JPSI(I)))**2)
             A=EXP(BETA*(RNIJ-RPSI(I)))
             PSICAL(I) = (1.0/(1.0+A))**2
          END DO
          CALL ENSAVE(PSICAL,PSIBUF,EXPNPSI)
          DO I=1,EXPNPSI
             RHO = RHO+ (PSICAL(I)-PSI(I))**2
          END DO
          RHO=RHO/REAL(EXPNPSI)
       ENDIF

       RHO0 = RHO

       IF (PRNLEV.GE.2) THEN
          IF (JRZERO10) WRITE(OUTU,'(A)') &
               ' RC10BIAS> YOU CHOSE TO KEEP THE PROTEIN AROUND RHO=0', &
               ' RC10BIAS> FOR RC10, AND NOT AROUND THE INITIAL RHO; THUS:'
          WRITE(OUTU,'(A,F13.4)')' RC10BIAS> INITIAL DISTANCE = ',RHO0
       ENDIF

       XI = RHO0
       XIA = XI
       !     Do everytime that this routine is called except the first
       ! ****************************
    ELSE
       RHO=0.0
       DO I=1,EXPNPSI
          R=SQRT((X(IPSI(I))-X(JPSI(I)))**2 &
               +(Y(IPSI(I))-Y(JPSI(I)))**2 &
               +(Z(IPSI(I))-Z(JPSI(I)))**2)
          A=EXP(BETA*(R-RPSI(I)))
          PSICAL(I) = (1.0/(1.0+A))**2
       END DO
       CALL ENSAVE(PSICAL,PSIBUF,EXPNPSI)
       DO I=1,EXPNPSI
          RHO = RHO+ (PSICAL(I)-PSI(I))**2
       END DO
       RHO=RHO/REAL(EXPNPSI)

       XI=RHO
       ! ****************************
    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   BUX       x ....         x ...
    !
    ! *************** CONSTRAINT ON EACH PSI VALUE *****************
    !
    IF (QFORCE.AND.ALPHA.GT.RSMALL) THEN
       IF ((XI .GT. XIA).OR.JHQFIX10) THEN
          !               write (outu,'(a)') ' calculating forces ...'
          DXI=XI-XIA

          EU =  ALPHA2*REAL(NENSEM)*DXI**2
          BUX=-TWO*ALPHA*BETA*DXI/EXPNPSI
          !              Compute forces for all selected atoms
          DO II=1,EXPNPSI
             I=IPSI(II)
             J=JPSI(II)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
             A=EXP(BETA*(R-RPSI(II)))
             B=1.0/((1.0+A)**2)
             C=A*B/R
             CX=C*XIJ
             CY=C*YIJ
             CZ=C*ZIJ
             CUX=(PSICAL(II)-PSI(II))
             DX(I)=DX(I)+BUX*CUX*CX
             DY(I)=DY(I)+BUX*CUX*CY
             DZ(I)=DZ(I)+BUX*CUX*CZ
             DX(J)=DX(J)-BUX*CUX*CX
             DY(J)=DY(J)-BUX*CUX*CY
             DZ(J)=DZ(J)-BUX*CUX*CZ
          END DO
       ELSE
          EU=ZERO
          XIA=XI
       END IF
    END IF

    IF (IOLEV.GE.0) THEN
       IF(QINIT) THEN
          IF(IUNDMP10.NE.-1)THEN
             WRITE(IUNDMP10,'(7(I9))')(IPSI(II),II=1,EXPNPSI)
             WRITE(IUNDMP10,'(7(I9))')(JPSI(II),II=1,EXPNPSI)
          ENDIF
       ELSE
          IF(MOD(HQSTEP,NPRINT).EQ.0.AND.HQSTEP.GE.0) THEN
             WRITE(IUNJUJ10,'(I9,3(1X,F14.6))')HQSTEP &
                  ,XI,XIA,EU
             IF(IUNDMP10.NE.-1)THEN
                WRITE(IUNDMP10,'(7(F8.3,1X))')(PSICAL(II),II=1,EXPNPSI)
             ENDIF
          ENDIF
       ENDIF
    ENDIF

    IF (XI .GT. XIMAX10) THEN
       IF (IOLEV.GT.0) WRITE(OUTU,'(A,F8.3,A,F8.3)') &
            'XI = ', XI, ' XIMAX10 = ', XIMAX10
       CALL GFLUSH(OUTU)
       CALL WRNDIE(-1,'<RC10BIAS>', &
            'Reaction coordinate RC10 exceeded maximum value.')
    ENDIF

    RETURN
  END SUBROUTINE RC10BIAS

#endif 
#if KEY_ASPENER==1
  SUBROUTINE EEFBUR(NNT,NNS,ISELN,NSELN,ATOMON,QINIT)
    !----------------------------------------------------------------------
    !     Calculates burial term from eef1 energies
    !     NNT = number of native contacts (actually energies!)
    !     QINIT = whether to fill NNT0
    !----------------------------------------------------------------------
    use psf
    use coord
    use coordc
    use deriv
    use stream
    use econtmod
    use energym
    use inbnd
    use eef1_mod
    use bases_fcm
    use chutil,only:getres
    implicit none
    real(chm_real) NNT(*),NNS(*)
    INTEGER ISELN(*),ATOMON(*),NSELN
    LOGICAL QINIT
    !     local
    INTEGER II,I,IRES
    real(chm_real) EU
    !     QC: bug fix 06/27/07

    DO II=1,NSELN
       I=ISELN(II)
       IRES=GETRES(I,IBASE,NREST)
       NNT(IRES) = ABS(GREFI(I) - GSOLV(I))
    ENDDO

    IF (QINIT) THEN
       !        Call eef1 with comparison coords
       CALL NBONDS(XCOMP,YCOMP,ZCOMP,BNBND,BIMAG)
#if KEY_ASPENER==1
       CALL EEF1EN(EU,XCOMP,YCOMP,ZCOMP,DX,DY,DZ,.false., &
            ECONT,1,NATOM,1, &
            NGRP,BNBND%JNBG,BNBND%INBLOG, &
            BNBND%INB14,BNBND%IBLO14,.FALSE., &
            .FALSE. &
            )
#endif 

       DO II=1,NSELN
          I=ISELN(II)
          IRES=GETRES(I,IBASE,NREST)
          NNS(IRES) = ABS(GREFI(I) - GSOLV(I))
       ENDDO
       !
       !        reset to regular coords (to be safe)
       CALL NBONDS(X,Y,Z,BNBND,BIMAG)
#if KEY_ASPENER==1
       CALL EEF1EN(EU,X,Y,Z,DX,DY,DZ,.false., &
            ECONT,1,NATOM,1, &
            NGRP,BNBND%JNBG,BNBND%INBLOG, &
            BNBND%INB14,BNBND%IBLO14,.FALSE., &
            .FALSE. &
            )
#endif 
    ENDIF
    RETURN
  END SUBROUTINE EEFBUR
#endif 
  !
  !C
  !      SUBROUTINE RC9AUX(I,J,K,L,X,Y,Z,DX,DY,DZ,J3S,J3E,QFORCE)
  !C----------------------------------------------------------------------
  !C     Calculates J3 and optionally forces for RC9
  !C----------------------------------------------------------------------
  !      INTEGER I,J,K,L
  !      real(chm_real) J3S,J3E
  !      LOGICAL QFORCE
  !
  !      RETURN
  !      END
  !
  SUBROUTINE BBHBON(XL,YL,ZL,ISELO,ISELH,NSELO,NSELH,NNH, &
       HCUT,BETH,QSMTH,HDONOR,HACCEP,NHBI)
    !----------------------------------------------------------------------
    ! calc. bb hydrogen bonds
    !----------------------------------------------------------------------
    use psf
    use stream
    use chutil,only:getres
    implicit none
    real(chm_real) XL(*),YL(*),ZL(*),NNH(*),HCUT,BETH
    INTEGER ISELO(*),ISELH(*),HDONOR(*),HACCEP(*),NHBI, &
         NSELO,NSELH
    LOGICAL QSMTH
    !     local
    INTEGER II, JJ, I, J, IRES, JRES
    real(chm_real) DX,DY,DZ,RIJ
    DO I=1,NRES
       NNH(I) = 0.0
    ENDDO
    NHBI=0
    DO II=1,NSELO
       DO JJ=1,NSELH
          I=ISELO(II)
          J=ISELH(JJ)
          IRES=GETRES(I,IBASE,NREST)
          JRES=GETRES(J,IBASE,NREST)
          ! hardwired exclusion
          IF (ABS(IRES-JRES).GE.3) THEN
             DX=XL(I)-XL(J)
             DY=YL(I)-YL(J)
             DZ=ZL(I)-ZL(J)
             RIJ=SQRT(DX*DX+DY*DY+DZ*DZ)
             IF (QSMTH) THEN
                NNH(JRES)=NNH(JRES)+1.0/(1.0+EXP(BETH*(RIJ-HCUT)))
             ELSE IF (RIJ.LT.HCUT) THEN
                ! assuming only one bb hbond possible for each amide
                NHBI=NHBI+1
                HDONOR(NHBI)=J
                HACCEP(NHBI)=I
                NNH(JRES)=NNH(JRES)+1.0
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE BBHBON

  SUBROUTINE NNTINI(IL,JL,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL,NSEL)
    !----------------------------------------------------------------------
    !           Computes the number of native contacts for each residue
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use coordc
    use number
    use psf
    use stream
    use chutil,only:getres
    implicit none
    INTEGER IL(*),JL(*),INDH(*),INDL(*),ISEL(*),NSEL
    real(chm_real) NNS(*),NNT(*),NNS0(*),NNT0(*)
    ! local
    INTEGER L,PIRES,II,JJ,IRES,JRES,I,J,EX
    real(chm_real) XIJ,YIJ,ZIJ,XCIJ,YCIJ,ZCIJ,RNIJ,RIJ,RCUT1,R,RC
    EX = EXCL
    RC = RCUT
    RCUT1 = RCUT + TOL

    DO I=1,NRES
       NNS(I)=0
       NNT(I)=0
       NNS0(I)=0
       NNT0(I)=0
    END DO

    L=0
    PIRES=-1
    DO II=1,NSEL
       DO JJ=1,NSEL
          !
          !         Get the atom number I corresponding to the IIth selected atom
          I = ISEL(II)
          J = ISEL(JJ)
          !
          !         Get the residue number IRES corresponding to the atom number I
          IRES=GETRES(I,IBASE,NREST)
          JRES=GETRES(J,IBASE,NREST)
          !            if restype not ala and atomtype cb continue
          !            fcm/psf.f90
          !            fcm/exfunc.f90
          !             IF (GETRSN(?).EQ.'ALA'.AND.ATYPE(I).EQ.' CA ') CONTINUE
          !             IF (GETRSN(?).EQ.'ALA'.AND.ATYPE(J).EQ.' CA ') CONTINUE

          IF (ABS(IRES-JRES).GE.EX) THEN

             XCIJ=XCOMP(I)-XCOMP(J)
             YCIJ=YCOMP(I)-YCOMP(J)
             ZCIJ=ZCOMP(I)-ZCOMP(J)
             RNIJ=SQRT(XCIJ*XCIJ+YCIJ*YCIJ+ZCIJ*ZCIJ)

             IF (RNIJ.LE.RC) THEN
                L=L+1
                IL(L)=I
                JL(L)=J
                IF (RNIJ.LE.RCUT1) NNS0(IRES)=NNS0(IRES)+1
                NNS(IRES)=NNS(IRES)+1./(1.+EXP(BETA*(RNIJ-RCUT1)))
                IF (IRES.NE.PIRES) THEN
                   INDL(IRES)=L
                   PIRES=IRES
                END IF
                INDH(IRES)=L
                !                  IL(L) and JL(L) are the last pair of atoms in the list
                !                  which are native contacts for reside IRES
                XIJ=X(I)-X(J)
                YIJ=Y(I)-Y(J)
                ZIJ=Z(I)-Z(J)
                R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                IF (R.LE.RCUT1) NNT0(IRES)=NNT0(IRES)+1
                NNT(IRES)=NNT(IRES)+1./(1.+EXP(BETA*(R-RCUT1)))
             END IF
          END IF
       END DO
    END DO
    RETURN
  END SUBROUTINE NNTINI

  SUBROUTINE COMBINI(IL,JL,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL,NSEL)
    !----------------------------------------------------------------------
    !           initialize native contacts as all-against-all combinations
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use coordc
    use number
    use psf
    use stream
    use chutil,only:getres
    implicit none
    INTEGER IL(*),JL(*),INDH(*),INDL(*),ISEL(*),NSEL
    real(chm_real) NNS(*),NNT(*),NNS0(*),NNT0(*)
    ! local
    INTEGER L,PIRES,II,JJ,IRES,JRES,I,J,EX
    real(chm_real) XIJ,YIJ,ZIJ,XCIJ,YCIJ,ZCIJ,RNIJ,RIJ,RCUT1,R,RC
    EX = EXCL
    RC = RCUT
    RCUT1 = RCUT + TOL

    DO I=1,NRES
       NNS(I)=0.0
       NNT(I)=0.0
       NNS0(I)=0.0
       NNT0(I)=0.0
    END DO

    L=0
    PIRES=-1
    DO II=1,NSEL
       DO JJ=1,NSEL
          !
          !         Get the atom number I corresponding to the IIth selected atom
          I = ISEL(II)
          J = ISEL(JJ)
          !
          !         Get the residue number IRES corresponding to the atom number I
          IRES=GETRES(I,IBASE,NREST)
          JRES=GETRES(J,IBASE,NREST)

          IF (ABS(IRES-JRES).GE.EX) THEN
             L=L+1
             IL(L)=I
             JL(L)=J
             NNS0(IRES)=NNS0(IRES)+1.0
             NNS(IRES)=NNS(IRES)+1.0
             IF (IRES.NE.PIRES) THEN
                INDL(IRES)=L
                PIRES=IRES
             END IF
             INDH(IRES)=L
             !                  IL(L) and JL(L) are the last pair of atoms in the list
             !                  which are native contacts for reside IRES
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
             IF (R.LE.RCUT1) NNT0(IRES)=NNT0(IRES)+1.0
             NNT(IRES)=NNT(IRES)+1./(1.+EXP(BETA*(R-RCUT1)))
          END IF
       END DO
    END DO
    RETURN
  END SUBROUTINE COMBINI

  SUBROUTINE NNTCAL(IL,JL,INDH,INDL,NNS,NNT,RCUT1,EXPDAT)
    !----------------------------------------------------------------------
    !           Computes the number of native contacts for each residue
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use coordc
    use number
    use psf
    use stream
    implicit none
    INTEGER IL(*),JL(*),INDH(*),INDL(*)
    real(chm_real) NNS(*),NNT(*),RCUT1,EXPDAT(*)
    ! local
    INTEGER L,IRES,JRES,I,J
    real(chm_real) XIJ,YIJ,ZIJ,RIJ,R
    !           Loop over all residues
    DO IRES=1,NRES
       IF (EXPDAT(IRES).NE.-1 .AND. (NNS(IRES).NE.0)) THEN
          !              Loop over all pairs of neighboring atoms involving residue IRES
          NNT(IRES)=0
          DO L=INDL(IRES),INDH(IRES)
             I=IL(L)
             J=JL(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
             NNT(IRES)=NNT(IRES)+1.0/(1.0+EXP(BETA*(R-RCUT1)))
          END DO
       END IF
    END DO
    RETURN
  END SUBROUTINE NNTCAL

  SUBROUTINE NONNTC(XX,YY,ZZ,IL,JL,INDH,INDL,NNS,HXPROC,ISEL,NSEL)
    !----------------------------------------------------------------------
    !           Calculates list of total contacts less than HXCTOF
    !           A switching function is applied on the contacts between
    !           HXCTON and HXCTOF.
    !
    !           IL,JL,INDH,INDL are updated with the current list of
    !           contacts for the purpose of force calculation
    !
    !           Ideally, this would use a non-bonded list, however this
    !           is complicated by the type of non-bonded list used
    !           (e.g. group lists as used by EEF1 are not really suitable)
    !----------------------------------------------------------------------
    use energym
    use comand
    use number
    use psf
    use stream
    use chutil,only:getres
    implicit none
    INTEGER IL(*),JL(*),INDH(*),INDL(*),ISEL(*),NSEL
    real(chm_real) NNS(*),XX(*),YY(*),ZZ(*),HXPROC(*)
    !     local
    INTeGER I,J,II,JJ,IRES,JRES,L,PIRES
    real(chm_real) XIJ,YIJ,ZIJ,RIJ,SW,DRCUT,CTON2,CTOF2
    real(chm_real) inside, outside
    !      write(outu,*) 'hxcton = ', hxcton
    !      write(outu,*) 'hxctof = ', hxctof
    !      write(outu,*) 'rcut = ', rcut
    call gflush(outu)

    DO I=1,NRES
       NNS(I)=0.0
    END DO
    DRCUT = 1.0/(HXCTOF**2-HXCTON**2)**3
    CTON2 = HXCTON**2
    CTOF2 = HXCTOF**2
    !      inside = 0.0
    !      outside = 0.0
    !
    IF (JNHCON) THEN
       L=0
       PIRES=-1
       DO II=1,NSELH
          DO JJ=1,NSEL
             !
             !           Get the atom number I corresponding to the IIth selected atom
             I = ISELH(II)
             J = ISEL(JJ)
             !
             !            write(outu,*) ii,jj,i,j
             call gflush(outu)
             !           Get the residue number IRES corresponding to the atom number I
             IRES=GETRES(I,IBASE,NREST)
             JRES=GETRES(J,IBASE,NREST)
             !               write(outu,*) ires,jres
             !               call gflush(outu)

             IF (ABS(IRES-JRES).GE.EXCL) THEN

                !                  write(outu,*) 'considering contact...'
                !                  call gflush(outu)
                XIJ=XX(I)-XX(J)
                YIJ=YY(I)-YY(J)
                ZIJ=ZZ(I)-ZZ(J)
                RIJ=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                !                  write(outu,*) xij,yij,zij,rij
                !                  call gflush(outu)

                IF (RIJ.LE.HXCTOF) THEN
                   L=L+1
                   IL(L)=I
                   JL(L)=J
                   IF (RIJ.LE.HXCTON) THEN
                      NNS(IRES)=NNS(IRES) &
                           +1./(1.+EXP(BETA*(RIJ-RCUT)))
                      !                         inside = inside
                      !     1                               +1./(1.+EXP(BETA*(RIJ-RCUT)))
                   ELSE IF (RIJ.LE.HXCTOF) THEN
                      SW = DRCUT*(CTOF2-RIJ**2)*(CTOF2+2.0*RIJ**2 &
                           -3.0*CTON2)
                      NNS(IRES)=NNS(IRES) &
                           + SW/(1.+EXP(BETA*(RIJ-RCUT)))
                      !                         outside = outside
                      !     1                            + SW/(1.+EXP(BETA*(RIJ-RCUT)))
                   ENDIF

                   IF (IRES.NE.PIRES) THEN
                      INDL(IRES)=L
                      PIRES=IRES
                   END IF
                   INDH(IRES)=L

                   !                    IL(L) and JL(L) are the last pair of atoms in the list
                   !                    which are native contacts for reside IRES
                END IF
             END IF
          END DO
       END DO

    ELSE
       L=0
       PIRES=-1
       DO II=1,NSEL
          DO JJ=1,NSEL
             !
             !           Get the atom number I corresponding to the IIth selected atom
             I = ISEL(II)
             J = ISEL(JJ)
             !
             !           Get the residue number IRES corresponding to the atom number I
             IRES=GETRES(I,IBASE,NREST)
             JRES=GETRES(J,IBASE,NREST)

             IF (ABS(IRES-JRES).GE.EXCL) THEN

                XIJ=XX(I)-XX(J)
                YIJ=YY(I)-YY(J)
                ZIJ=ZZ(I)-ZZ(J)
                RIJ=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)

                IF (RIJ.LE.HXCTOF) THEN
                   L=L+1
                   IL(L)=I
                   JL(L)=J
                   IF (RIJ.LE.HXCTON) THEN
                      NNS(IRES)=NNS(IRES) &
                           +1./(1.+EXP(BETA*(RIJ-RCUT)))
                   ELSE
                      SW = DRCUT*(CTOF2-RIJ**2)*(CTOF2+2.0*RIJ**2 &
                           -3.0*CTON2)
                      NNS(IRES)=NNS(IRES) &
                           + SW/(1.+EXP(BETA*(RIJ-RCUT)))
                   ENDIF

                   IF (IRES.NE.PIRES) THEN
                      INDL(IRES)=L
                      PIRES=IRES
                   END IF
                   INDH(IRES)=L

                   !                    IL(L) and JL(L) are the last pair of atoms in the list
                   !                    which are native contacts for reside IRES
                END IF
             END IF
          END DO
       END DO
    ENDIF
    !      write(outu,*) 'inside = ',inside
    !      write(outu,*) 'outside = ',outside
    call gflush(outu)
  END SUBROUTINE NONNTC

  SUBROUTINE NHNNTI(IL,JL,INDH,INDL,NNS,NNT,NNT0,NNS0,ISEL,NSEL)
    !----------------------------------------------------------------------
    !           Computes the number of native contacts for each residue
    !           using (amide H) --- (sc heavy atom) dists
    !----------------------------------------------------------------------
    use energym
    use comand
    use coord
    use coordc
    use number
    use psf
    use stream
    use chutil,only:getres
    implicit none
    INTEGER IL(*),JL(*),INDH(*),INDL(*),ISEL(*),NSEL
    real(chm_real) NNS(*),NNT(*),NNS0(*),NNT0(*)
    ! local
    INTEGER L,PIRES,II,JJ,IRES,JRES,I,J,EX
    real(chm_real) XIJ,YIJ,ZIJ,XCIJ,YCIJ,ZCIJ,RNIJ,RIJ,RCUT1,R,RC
    SAVE RCUT1

    EX = EXCL
    RC = RCUT
    RCUT1 = RCUT + TOL

    DO I=1,NRES
       NNS(I)=0
       NNT(I)=0
       NNS0(I)=0
       NNT0(I)=0
    END DO

    L=0
    PIRES=-1
    DO II=1,NSELH
       DO JJ=1,NSEL
          !
          !        Get the atom number I corresponding to the IIth selected atom
          I = ISELH(II)
          J = ISEL(JJ)
          !
          !        Get the residue number IRES corresponding to the atom number I
          IRES=GETRES(I,IBASE,NREST)
          JRES=GETRES(J,IBASE,NREST)

          IF (ABS(IRES-JRES).GE.EX) THEN

             XCIJ=XCOMP(I)-XCOMP(J)
             YCIJ=YCOMP(I)-YCOMP(J)
             ZCIJ=ZCOMP(I)-ZCOMP(J)
             RNIJ=SQRT(XCIJ*XCIJ+YCIJ*YCIJ+ZCIJ*ZCIJ)

             IF (RNIJ.LE.RC) THEN
                L=L+1
                IL(L)=I
                JL(L)=J
                IF (RNIJ.LE.RCUT1) NNS0(IRES)=NNS0(IRES)+1
                NNS(IRES)=NNS(IRES)+1./(1.+EXP(BETA*(RNIJ-RCUT1)))
                IF (IRES.NE.PIRES) THEN
                   INDL(IRES)=L
                   PIRES=IRES
                END IF
                INDH(IRES)=L
                !                 IL(L) and JL(L) are the last pair of atoms in the list
                !                 which are native contacts for reside IRES
                XIJ=X(I)-X(J)
                YIJ=Y(I)-Y(J)
                ZIJ=Z(I)-Z(J)
                R=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
                IF (R.LE.RCUT1) NNT0(IRES)=NNT0(IRES)+1
                NNT(IRES)=NNT(IRES)+1./(1.+EXP(BETA*(R-RCUT1)))
             END IF
          END IF
       END DO
    END DO
    RETURN
  END SUBROUTINE NHNNTI

  SUBROUTINE HQANAL
    !----------------------------------------------------------------------
    ! runs through a trajectory and computes properties relevant to
    ! various reaction coordinates
    !----------------------------------------------------------------------
    use psf
    use coord
    use coordc
    use deriv
    use cvio
    use stream
    use string
    use econtmod
    use energym
    use comand
    use number
    use ctitla
    use inbnd
    use memory
    use eef1_mod
    use bases_fcm
    use contrl

    implicit none
    real(chm_real4),allocatable,dimension(:) :: TEMP
    integer,allocatable,dimension(:) :: FREEAT
    real(chm_real),allocatable,dimension(:) :: CHARG
    INTEGER XL, YL, ZL, NUNIT, FIRSTU, BEGIN, SKIP, STOP, &
         NFREAT, IUNIT, NDEGF, ISTEPX, ISTATS, NSAV, NFILE, &
         NPRIN
    !     +         ,ATOMON
    real(chm_real) DELTA, EU
    LOGICAL QCG
    CHARACTER(len=4), parameter :: HDR='CORD'

    !     check for nprint update
    NPRIN=GTRMI(COMLYN,COMLEN,'NPRI',-1)
    IF (NPRIN.GT.0) NPRINT = NPRIN
    call chmalloc('hqbm.src','HQANAL','TEMP',NATOM,cr4=TEMP)
    call chmalloc('hqbm.src','HQANAL','FREEAT',NATOM,intg=FREEAT)
    call chmalloc('hqbm.src','HQANAL','CHARG',NATOM,crl=CHARG)

    IF (JEEF1) THEN
       !       need to call EEF1EN to update solvation terms
       !       if using eef1 for burial
       CALL NBONDS(X,Y,Z,BNBND,BIMAG)
#if KEY_ASPENER==1
       CALL EEF1EN(EU,X,Y,Z,DX,DY,DZ,.false.,ECONT,1,NATOM, &
            1,NGRP,BNBND%JNBG,BNBND%INBLOG, &
            BNBND%INB14,BNBND%IBLO14,.FALSE., &
            .FALSE. &
            )
#endif 
    ENDIF
    !     loop over trajectory -------------
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
    IUNIT = FIRSTU
    QCG = .false.
    ISTATS=1
    DO WHILE (ISTATS .GE. 0)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CHARG,QCG, & 
#endif
            TEMP,NATOM, &
            FREEAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,NFILE,ISTEPX,ISTATS, &
            NDEGF,DELTA,BEGIN,STOP,SKIP, &
            NSAV,HDR,HDR,TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       IF (JEEF1) THEN
          !          need to call EEF1EN to update solvation terms
          !          if using eef1 for burial
          CALL NBONDS(X,Y,Z,BNBND,BIMAG)
#if KEY_ASPENER==1
          CALL EEF1EN(EU,X,Y,Z,DX,DY,DZ,.false.,ECONT,1,NATOM, &
               1,NGRP,BNBND%JNBG,BNBND%INBLOG, &
               BNBND%INB14,BNBND%IBLO14,.FALSE., &
               .FALSE. &
               )
#endif 
       ENDIF
       CALL HQBIAS(EU,X,Y,Z,DX,DY,DZ,NATOM,.false.)
    ENDDO
    call chmdealloc('hqbm.src','HQANAL','CHARG',NATOM,crl=CHARG)
    call chmdealloc('hqbm.src','HQANAL','FREEAT',NATOM,intg=FREEAT)
    call chmdealloc('hqbm.src','HQANAL','TEMP',NATOM,cr4=TEMP)
    RETURN
  END SUBROUTINE HQANAL

#endif /*       (hqbm_outer)*/
end module hqbmm

