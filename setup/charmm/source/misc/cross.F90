! --------------------------------------------------------------------
! 07/12/04: Wrapping of cross.src into MODULE CROSS for global, 
! allocatable array usage.
!
! Redefined arrays (replaces COMMON BLOCK arrays in saveforces.fcm and 
! cross.fcm)
!
! 08/11/23: 
! - Convert remaining fixed size FORTRAN77 arrays to 
!   allocatable ones. 
! - Merge all data from cross.fcm into module header.
! - Fixed missing counter increase in exclusion list generator.
! - Fixed bug when exclusions between identical atoms occured.
! - Fixed wrong indexing in old coordinate and forces array updater
!   and reloader while running dynamics.
! - Fixed error where double counted exclusions were generated
! - Added non-bonding inclusion list to handle PSF with bond which gets
!   broken on a specific surface.
! - Fixed wrong VdW parameter selection in EDELTA
! - Include removal of PSF improper interactions in CREMOVE
! --------------------------------------------------------------------

module cross

  use chm_kinds

  implicit none


!++++++++++++Variables used in the CROSS module++++++++++++++++++++++++++
!
!
! 2. Logical flags
!
! QCROS    - Flag to indicate that surface crossing can occur. 
! QCON     - Flag that indicates that surface crossing is in progress
! QCFOUND  - Flag to indicate that a crossing was found during the previous
!            timestep
! QXTIME   - Flag to indicate that time needs to be rewinded
! QTANH    - Flag to indicate that tanh switching function is used during
!            crossing  
! QCOS     - Defunct flag to indicate that the cosine switching function is 
!            used
! QCEND    - Flag to indicate the final step in the crossing procedure
!            is reached 
! QUOUT    - Flag to indicate that crossing geometries are to be dumped
! QCBON(K) - Array of flags to indicate if a bond is present on PES K
! QCBON(NCRSU+1) - Array of flags to indicate if a bond is present in PSF 
! QCANG(K) - Array of flags to indicate if an angle is present on PES K
! QCANG0(NCRSU+1) - Array of flags to indicate if an angle is present in PSF
! QCDIH(K) - Array of flags to indicate if a dihedral is present on PES K
! QCDIH(NCRSU+1) - Array of flags to indicate if a dihedral is present in PSF
!
! 3. Pointer arrays
!
! ICATOM() - Array of PSF atom number for crossing atoms   
! ICINDX() - Array of crossing atom index (if available) for all atoms,
!            equals -1 for atoms not in crossing zone. 
! ICBON1() - PSF atom number of first atom in crossing bond
! ICBON2() - PSF atom number of second atom in crossing bond
! ICBONI() - Corresponding index for bond in PSF arrays
! ICANG1() - PSF atom number of first atom in crossing angle
! ICANG2() - PSF atom number of central atom in crossing angle
! ICANG3() - PSF atom number of third atom in crossing angle
! ICANGI() - Corresponding index for angle in PSF arrays
! ICDIH1() - PSF atom number of first atom in crossing dihedral 
! ICDIH2() - PSF atom number of first central atom in crossing dihedral 
! ICDIH3() - PSF atom number of second central atom in crossing dihedral 
! ICDIH4() - PSF atom number of fourth atom in crossing dihedral 
! ICDIHI() - Corresponding index for dihedral in PSF arrays
! ICEXC(K,:,:)- List of pairs of crossing atoms (PSF number) to exclude from 
!            nonbonded interactions on PES K
! ICEXC(N+1,:,:)- List of pairs of crossing atoms (PSF number) to exclude from 
!            nonbonded interactions on PES 0
! ICINC(K,:,:)- List of pairs of crossing atoms (PSF number) to include from 
!            nonbonded interactions on PES K
! ICINC(N+1,:,:)- List of pairs of crossing atoms (PSF number) to include from 
!            nonbonded interactions on PES 0
! IC14(K,:,:) - List of 1-4 exclusion atom pairs involving crossing atoms 
!            on PES K
! IC14(N+1,:,:) - List of 1-4 exclusion atom pairs involving crossing atoms 
!            in the PSF
! ICIN14(K,:,:) - List of 1-4 inclusion atom pairs involving crossing atoms 
!            on PES K
! ICIN14(N+1,:,:) - List of 1-4 inclusion atom pairs involving crossing atoms 
!            in the PSF
!
! 4. Counters
!
! NCRUN    - Number of times the RXMD command was called
! NCRSU    - Number of user defined surfaces
! NCRAT    - Number of atoms defined in CROS input
! NCBON    - Number of covalent bonds defined in CROS input = NCHAR+NCMOR 
! NCHAR    - Number of harmonic bonds defined in CROS input
! NCMOR    - Number of Morse bonds defined in CROS input
! NCANG    - Number of angles defined in CROS input
! NCDIH    - Number of dihedrals defined in CROS input
! NISUR    - Number of possible surface crossings
! NCEXC(K) - Number of non-bonded exclusions generated for PES K
! NCEXC(N+1)   - Number of non-bonded exclusions for the PSF including
!            crossing atoms
! NCINC(K) - Number of non-bonded inclusions generated for PES K
! NCINC(N+1) - Number of non-bonded inclusions for the PSF including
!            crossing atoms
! NC14(N+1)    - Number of 1-4 exclusions in the PSF involving crossing atoms
! NC14(K)    - Number of 1-4 exclusions generated for PES K
! NCIN14(N+1)  - Number of 1-4 inclusions in the PSF involving crossing atoms
! NCIN14(K)  - Number of 1-4 inclusions generated for PES K
! OLDSU    - Number of old surface during a crossing
! NEWSU    - Number of new surface during a crossing 
! LOWSU    - Number of current surface when no crossing in progress
! CROSSCOUNT - Counts total number of steps in Surface crossing dynamics
! XCROSCOUNT - Counts steps during the crossing procedure
! CTIME    - Keeps track of simulation time
!
! 5. Force field parameters
!
! ECCHG(K,:) - Atomic charges on PES K 
! VCEPS(K,:) - vdW epsilon parameters on PES K
! VCRAD1(K,:) - vdW radii on PES K
! KCBON1(K,:) - Harmonic bond force constants for PES K
! RCBON1(K,:) - Equil. bond lengths for harmonic bonds on PES K 
! DCMOR1(K,:) - Diss. Energies for Morse bonds on PES K
! BCMOR1(K,:) - Beta parameters for Morse bonds on PES K 
! RCMOR1(K,:) - Equil. bond lengths for Morse bonds on PES K
! KCANG1(K,:) - Force constants for angle bending on PES K
! TCANG1(K,:) - Equil. angles on PES K
! MCDIH1(K,:) - Multiplicities of dihedrals on PES K
! KCDIH1(K,:) - Force constants for dihedrals on PES K 
! PCDIH1(K,:) - Phase shifts of dihedrals on PES K
! PCDIC1(K,:) - Cosine of  phase shifts of dihedrals on PES K
! PCDIS1(K,:) - Sinus of phase shifts of dihedrals on PES K
!
! 6. Input parameters
! 
! CPARU    - Unit number of CROSS parameter file. 
! XCROSU   - Unit number for writing crossing geometries
! CSHIFT   - Added shift between the two PESs
! defSHIF  - default value of CSHIFT 
! XTIME    - Mixing time in timesteps.
! defXTIM  - Default value of XTIM
! BARRTOL  - Allows crossing for surfaces with energies within BARRTOL (even ).
!            Useful for asymptotically degenerate PES  
! defBTOL  - Default value of BTOL
! XCROSF   - Frequency of checking energy difference between the two PES
! defXFRQ  - Default value of XCROSF
! BCHK     - Cutoff distance below which energies of bond forming crossings 
!            are getting checked
!
! 7. Arrays for backing up coordinates, velocities etc.
!
! (  X[XYZ](,) - Array that stores coordinates from recent steps
!    XSTEP[XYZ](,) - Array that stores the step vectors
!    XSTEPOX[XYZ](,) - Array that stores the previous step vectors
!    XGAMMA(,) - Array to store random forces from recents steps   )
!
!
! XSEED()   - Array to store random seeds from recents steps
! XFACTOR() - Array that stores precomputed values of the switching function
!             during the crossing
!
! 8. Other
!
! ICDIR     - shows direction of crossing and has 3 possible values:
!             1: crossing from lower to higher surface is taking place 
!             0: No crossing in progress, 
!            -1: crossing from higher to lower surface is taking place
!
!
! 9. Passed variables between routines for use in surface crossing procedure
!
! SAVEDSEED   - the recalled random seed
! XSTRT       - flag for when the current coordinates should be swapped
!               with recalled ones
!
! 10. Variables used for special VdW interaction between bond forming atom
!     pairs and the related 1-3 interactions
!
! NBVDW       - Number of special VdW interaction atom pairs
! BVDWSURF    - Array of flags indicating for each pair of VdW interactions on which
!               ARMD surface they become active
! BVDWEXC1    - Array storing PSF number of atom 1 treated with special VdW interactions
! BVDWEXC2    - Array storing PSF number of atom 2 treated with special VdW interactions
! BVDWEPS     - Stores epsilon parameter of special VdW interaction
! BVDWSIG     - Stores sigma parameter of special VdW interaction
! REPP(NCBON) - Stores repulsive exponent of all user defined VdW functions
! ATTP(NCBON) - Stores attractive exponent of all the user defined VdW funcions


  logical,public,save :: QCON,QTANH,QCOS,QCFOUND,QXTIME, &
       QCROS,QCEND,QUOUT,XSTRT

  logical,public,allocatable,dimension(:),save :: QON

  integer,public,save :: NCRUN,NCRSU,NISUR,NCRAT,NCBON,  &
       NCHAR,NCMOR,NCANG,NCDIH,NBVDW,LOWSU,OLDSU,NEWSU

  integer,public,allocatable,dimension(:),save :: NCEXC, &
       NC14,NCINC,NCIN14,ICDIR,ICINDX

  integer,public,save :: XTIME,XRESTARTTIME,CPARU,CROSSCOUNT

  real(chm_real),public,allocatable,dimension(:),save :: SSHIFT, &
       SBARTO

  real(chm_real),public,save :: BCHK,CTIME,SAVEDSEED

  real(chm_real),public, parameter :: defBTOL=0.1,defShift=4

  integer,public,save :: XCROSU,XCROSF,XCROSCOUNT

  integer,public, parameter :: DEFXFRQ=1, DEFXTIM=10

  real(chm_real),allocatable,dimension(:),save :: XFACTOR, &
       XSEED

  real(chm_real),public,allocatable,dimension(:),save :: &
       SAVEDSX,SAVEDSY,SAVEDSZ,SAVEDXX,SAVEDXY,SAVEDXZ,SAVEDSOX, &
       SAVEDSOY, SAVEDSOZ

  real(chm_real),public,allocatable,dimension(:),save :: &
       SAVEDGAMMA, CURRENTGAMMA

  real(chm_real),public,allocatable,dimension(:,:),save :: XX,XY, &
       XZ,XSTEPX,XSTEPY,XSTEPZ,XSTEPOX,XSTEPOY,XSTEPOZ,XGAMMA

  logical,allocatable,dimension(:,:),save :: QCBON,QCANG,QCDIH

  integer,allocatable,dimension(:),save :: ICATOM,ICBON1,ICBON2, &
       ICBONI,ICANG1,ICANG2,ICANG3,ICANGI,ICDIH1,ICDIH2,ICDIH3, &
       ICDIH4,ICDIHI

  integer,allocatable,dimension(:,:,:),save :: ICEXC,IC14,ICINC, &
       ICIN14

  integer,allocatable,dimension(:,:),save :: MCDIH

  real(chm_real),allocatable,dimension(:,:),save :: ECCHG,VCEPS, &
       VCRAD,KCBON,RCBON,DCMOR,BCMOR,RCMOR,KCANG,TCANG,KCDIH, &
       PCDIH,PCDIC,PCDIS

  integer,public,save,allocatable,dimension(:) :: BVDWEXC1,BVDWEXC2
  real(chm_real),allocatable,dimension(:),save :: BVDWEPS, &
                                                  BVDWSIG,REPP,ATTP
  logical,public,save :: BFDVDW
  logical,public,allocatable,dimension(:,:),save :: BVDWSURF


  !====================================================================
  ! Leave global array definition and enter module subroutine section
  !====================================================================

contains

  !====================================================================
  ! INITIALIZATION
  !====================================================================
#if KEY_RMD==1 /*cross*/
  subroutine cross_iniall_init()
    qcon    = .false.
    qcros   = .false.
    xstrt   = .false.
    ncrun   = 0
    ncrat   = 0
    ncbon   = 0
    ncang   = 0
    ncmor   = 0
    ncdih   = 0
    qcfound = .false.
    qcend   = .false.
    qxtime  = .false.
    crosscount  =  0
    xcroscount  =  0
    xrestarttime = 1000000000
    return
  end subroutine cross_iniall_init

  SUBROUTINE CROSSINIT
    !     
    !     Interprets the options CROSS command, reads the CROSS 
    !     parameter file into suitable arrays and generates lists  
    !     of exclusions and/or 1-4 interactions   
    !
    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_ltm
    use number
    use cnst_fcm
    use image
   
    ! Loop counters, atom number storages, decision flags etc.
    INTEGER I,J,K,L,N,M,IB1,IB2,IB3,IB4,JB1,JB2,IDUM, &
            CNT1,CNT2,CNT3
    INTEGER IBB0,IBB1,IBB2,IBB3,NEWEXC,NEWINC
    LOGICAL EXFL,INFL,INF14,EXF14

    ! Temporary array sizes
    INTEGER TNCRA,TNCBO,TNCMO,TNCAN,TNCDI

    ! Temporary data storage
    integer,allocatable,dimension(:) :: TICBON1,TICBON2, &
         TICBONI,TICMOR1,TICMOR2,TICMORI
    logical,allocatable,dimension(:,:) :: TQCBON,TQCMOR
    integer,allocatable,dimension(:,:,:) :: TMPEXC,TMPC14, &
         TMPINC,TMPIN14
    real(kind=8),allocatable,dimension(:,:) :: TMPSHIF,TMPBART

    real(kind=8),allocatable,dimension(:) :: TMPREA

    integer,allocatable,dimension(:,:) :: TMPOVLA

    INTEGER SURF1,SURF2,MAXCEXC,MAXCIC14
    real(chm_real) COEFF,TNCRS,READSH,READBA
    CHARACTER(len=4) LBT


    !==================================================================
    ! Process command line
    !

    NCRUN = NCRUN+1
    !-------------------------------------------------------------------
    ! Choice of crossing algorithm:
    ! saved for later use /jonas
    ! "TANH" algorithm (only one available):
    !-------------------------------------------------------------------

    QTANH = (IndxA(ComLyn,ComLen,'TANH').gt.0)
    QTANH = .true.

    !-------------------------------------------------------------------
    ! Unit for reading parameters
    !-------------------------------------------------------------------

    CparU = GTRMI(ComLyn,ComLen,'UPAR',-1)
    if (CparU.eq.-1) CALL WrnDie (-3,'<CROSS>', &
         'no parameter file for CROSS')

    !-------------------------------------------------------------------
    ! Number of time steps over which crossing is to take place
    !-------------------------------------------------------------------

    XTIME = GTRMI(ComLyn,ComLen,'XTIM',defXTIM)

    ! XTIME should be an odd number so that crossing
    ! trajectory is symmetric about the crossing point

    IF (MOD(XTIME,2) .eq. 0) THEN
       XTIME=XTIME+1
    ENDIF
    ! allocate memory for XTIME dependent arrays
    IF (allocated(XFACTOR)) THEN
      deallocate(XFACTOR)
    ENDIF
    allocate(XFACTOR(XTIME))
    !-------------------------------------------------------------------
    ! Unit for writing crossing geometry and frequency for looking
    ! after crossings
    !-------------------------------------------------------------------

    XCROSU = GTRMI(ComLyn,ComLen,'UNIT',-1)
    XCROSF = GTRMI(ComLyn,ComLen,'FREQ',defXFRQ)
    IF (XCROSU.eq.-1) THEN
       quOut = .false.
    ELSE
       quOut= .true. 
    ENDIF

    !-------------------------------------------------------------------
    ! Check if user defines cutoff distance for bond formation checks
    !-------------------------------------------------------------------
      BCHK = GTRMF(ComLyn,ComLen,'BCHK',-1.0_CHM_REAL)


    !====================================================================
    ! Read and interpret parameter file
    !====================================================================

    !----------------------------------------------------------------
    !  Surface parameters
    !----------------------------------------------------------------

    READ(CparU,*) LBT, TNCRS

    IF(LBT.NE.'SURF') CALL WRNDIE(-2,'<CROSSINIT>', &
         'Missing SURF segment in input')

    ! Check if CROSSINIT was called for the first time and
    ! allocate memory for arrays, if called for a second or later time
    ! dellocate the old memory first. If NCRSU .EQ. 0 deallocate the
    ! old memory anyway.
    IF (TNCRS.GT.0 .AND. NCRUN.EQ.1) THEN
      NCRSU=TNCRS
    ! Evaluate number of possible surface crossings according to number of
    ! user defined surfaces
      NISUR=(TNCRS/2)*(TNCRS-1)
    ! Allocate memory for unsorted arrays used for SHIFT and BARTol read-in 
      allocate(TMPSHIF(NCRSU,NCRSU),TMPBART(NCRSU,NCRSU))
    !  Allocate memory for saved surface defining arrays
      allocate(SSHIFT(NISUR),SBARTO(NISUR),QON(NCRSU),ICDIR(NISUR), &
         NCEXC(NCRSU+1),NC14(NCRSU+1),NCINC(NCRSU),NCIN14(NCRSU))
    ELSEIF (TNCRS.GT.0 .AND. NCRSU.NE.0 .AND. NCRUN.GT.1) THEN
    ! Allocate memory for unsorted arrays used for SHIFT and BARTol read-in 
    ! Dellocate/Allocate memory for saved surface defining arrays
      NCRSU=TNCRS
      NISUR=(NCRSU/2)*(NCRSU-1)
      deallocate(SSHIFT,SBARTO,NCEXC,NC14,NCINC,NCIN14,QON,ICDIR)
      allocate(TMPSHIF(NCRSU,NCRSU),TMPBART(NCRSU,NCRSU))
      allocate(SSHIFT(NISUR),SBARTO(NISUR),QON(NCRSU),ICDIR(NISUR), &
         NCEXC(NCRSU+1),NC14(NCRSU+1),NCINC(NCRSU),NCIN14(NCRSU))
    ELSEIF (TNCRS.GT.0 .AND. NCRSU.EQ.0 .AND. NCRUN.GT.1) THEN
      NCRSU=TNCRS
      NISUR=(NCRSU/2)*(NCRSU-1)
    ! Allocate memory for unsorted arrays used for SHIFT and BARTol read-in 
      allocate(TMPSHIF(NCRSU,NCRSU),TMPBART(NCRSU,NCRSU))
    ! Allocate memory for saved surface defining arrays
      allocate(SSHIFT(NISUR),SBARTO(NISUR),QON(NCRSU),ICDIR(NISUR), &
         NCEXC(NCRSU+1),NC14(NCRSU+1),NCINC(NCRSU),NCIN14(NCRSU))
    ELSEIF (TNCRS.LE.1 .AND. NCRSU.NE.0 .AND. NCRUN.GT.1) THEN
      deallocate(SSHIFT,SBARTO,NCEXC,NC14,NCINC,NCIN14,QON,ICDIR)
      CALL WRNDIE(-2,'<CROSSINIT>', &
         'More than one surface needed for RMD!')
    ENDIF
    ! Set TMPSHIF and TMPBART initially to zero
    DO I=1, NCRSU
      DO J=1, NCRSU
        TMPSHIF(I,J)=0.0_CHM_REAL
        TMPBART(I,J)=0.0_CHM_REAL
      ENDDO
    ENDDO

    ! Read surface SHIFT values relative to surface 1
    DO I=1, NCRSU-1
      READ(CparU,*) SURF1, SURF2, READSH
      IF ((SURF1.EQ.SURF2).OR.(SURF1.NE.1.AND.SURF2.NE.1)) THEN
        CALL WRNDIE(-2,'<CROSSINIT>', &
        'Inter-surface energetics only assignable relative to surface 1')
      ELSE
        TMPSHIF(SURF1,SURF2)=READSH
      ENDIF
    ENDDO

    ! Get order from TMPSHIF and copy it to SSHIFT and 
    ! evaluate remaining surface shifts
    ! Surface order within array is 1-2, 1-3, ... 1-N ..., (N-1)-N
    K=1
    DO I=1, NCRSU-1
      DO J=I+1, NCRSU
        IF  ((I.NE.J).AND.((TMPSHIF(I,J).NE.0.0_CHM_REAL).AND. &
             (TMPSHIF(J,I).NE.0.0_CHM_REAL))) THEN
          CALL WRNDIE(-2,'<CROSSINIT>', &
               'surface energy shifts assigned twice')
        ELSEIF ((I.NE.J).AND.((TMPSHIF(I,J).NE.0.0_CHM_REAL).OR. &
                (TMPSHIF(J,I).NE.0.0_CHM_REAL))) THEN
          IF ((I.NE.J).AND.(I.EQ.1)) THEN
            SSHIFT(K)=TMPSHIF(I,J)
            K=K+1
          ELSEIF ((I.NE.J).AND.(J.EQ.1)) THEN
            SSHIFT(K)=TMPSHIF(J,I)
            K=K+1
          ENDIF
        ENDIF
      ENDDO
    ENDDO

    !  L=NCRSU
    DO I=1, NCRSU-1
      DO J=1, NCRSU
        IF ((I.GT.1).AND.(J.GT.1).AND.(I.LT.J)) THEN
          SSHIFT(K)=SSHIFT(J-1)-SSHIFT(I-1)
          K=K+1
        ENDIF
      ENDDO
    ENDDO

    !  Read surface BARRTOL values between each surface pair
    READ(CparU,*) LBT
    IF (LBT.NE.'BART') THEN
      WRITE(OUTU,*)'CROSSINIT> ', &
        'No Barrier Tolerance defined. Default to 10e-4 kcal/mol'

      DO I=1, NISUR
          SBARTO(I)=0.0001_CHM_REAL
      ENDDO

      BACKSPACE(CparU)

    ELSE
    !  Read surface BarrTol values
      DO I=1, NISUR
        READ(CparU,*) SURF1, SURF2, READBA
        IF (SURF1.EQ.SURF2) THEN
          CALL WRNDIE(-2,'<CROSSINIT>', &
          'Cannot assign BARRTOL between identical surfaces')
        ELSE
          TMPBART(SURF1,SURF2)=READBA
        ENDIF
      ENDDO

    !  Get order from TMPBART and copy it to SBARTO array. 
    !  Surface order within arrays is 1-2, 1-3, ..., 2-3, 2-4, 
    !  ..., (N-1)-N
      K=1
      DO I=1, NCRSU-1
        DO J=1, NCRSU
          IF  ((I.NE.J).AND.((TMPBART(I,J).NE.0.0_CHM_REAL).AND. &
               (TMPBART(J,I).NE.0.0_CHM_REAL))) THEN
            CALL WRNDIE(-2,'<CROSSINIT>', & 
                 'surface BarrTol assigned twice')

          ELSEIF ((I.NE.J).AND.((TMPBART(I,J).NE.0.0_CHM_REAL).OR. &
                  (TMPBART(J,I).NE.0.0_CHM_REAL))) THEN

            IF ((TMPBART(I,J).NE.0.0_CHM_REAL).AND.(I.LT.J)) THEN
              SBARTO(K)=TMPBART(I,J)
              K=K+1
            ELSEIF ((TMPBART(J,I).NE.0.0_CHM_REAL).AND.(I.LT.J)) THEN
              SBARTO(K)=TMPBART(J,I)
              K=K+1
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF

    IF (allocated(TMPBART)) deallocate(TMPBART)
    IF (allocated(TMPSHIF)) deallocate(TMPSHIF)

    ! ----------------------------------------------------------------
    ! Nonbonding interactions
    ! ----------------------------------------------------------------

    ! Allocate memory for ICINDX. If already allocated deallocate first.
    IF (allocated(ICINDX)) deallocate(ICINDX)
    allocate(ICINDX(NATOM))

    ! Set ICINDX initially to -1
    DO I=1,NATOM
      ICINDX(I)=-1
    ENDDO

    READ(CparU,*) LBT, TNCRA

    IF(LBT.NE.'ATOM') CALL WRNDIE(-2,'<CROSSINIT>', &
       'Missing ATOM segment in input') 

    ! Check if CROSSINIT was called for the first time and
    ! allocate memory for arrays, if called for a second or later time
    ! dellocate the old memory first. If NCRAT .EQ. 0 deallocate the
    ! old memory anyway.

    IF (TNCRA.GT.0 .AND. NCRUN.EQ.1) THEN
      NCRAT=TNCRA
      allocate(ICATOM(NCRAT),ECCHG(NCRSU,NCRAT), &
               VCEPS(NCRSU,NCRAT),VCRAD(NCRSU,NCRAT))
    ELSEIF (TNCRA.GT.0 .AND. NCRAT.NE.0 .AND. NCRUN.GT.1) THEN
      NCRAT=TNCRA
      deallocate(ICATOM,ECCHG,VCEPS,VCRAD)
      allocate(ICATOM(NCRAT),ECCHG(NCRSU,NCRAT), &
               VCEPS(NCRSU,NCRAT),VCRAD(NCRSU,NCRAT))
    ELSEIF (TNCRA.GT.0 .AND. NCRAT.EQ.0 .AND. NCRUN.GT.1) THEN
      NCRAT=TNCRA
      allocate(ICATOM(NCRAT),ECCHG(NCRSU,NCRAT), &
               VCEPS(NCRSU,NCRAT),VCRAD(NCRSU,NCRAT))
    ELSEIF (TNCRA.EQ.0 .AND. NCRAT.NE.0 .AND. NCRUN.GT.1) THEN
      NCRAT=TNCRA
      deallocate(ICATOM,ECCHG,VCEPS,VCRAD)
    ELSE
      NCRAT=TNCRA
    ENDIF

    ! Fill NON-BOND parameter arrays of every atom in a surface 
    ! crossing region
    IF (NCRAT.GT.0) THEN
      allocate(TMPREA(3*NCRSU+1))
      DO I = 1,NCRAT 
        READ(CparU,*) (TMPREA(J), J=1, 3*NCRSU+1)
        DO J=1,3*NCRSU+1
          IF (J.EQ.1) THEN 
            ICATOM(I)=TMPREA(J)
          ELSEIF (MOD((J+1),3).EQ.0) THEN
            ECCHG((J+1)/3,I)=TMPREA(J)
          ELSEIF (MOD(J,3).EQ.0) THEN
            VCEPS(J/3,I)=TMPREA(J)
          ELSEIF (MOD((J-1),3).EQ.0) THEN
            VCRAD((J-1)/3,I)=TMPREA(J)
          ENDIF
        ENDDO
          ICINDX(ICATOM(I))=I  
      ENDDO
      IF (allocated(TMPREA)) deallocate(TMPREA)
    ENDIF

    !----------------------------------------------------------------
    !  Harmonic bonds
    !----------------------------------------------------------------
    READ(CparU,*) LBT,TNCBO

    IF(LBT.NE.'BOND') CALL WRNDIE(-2,'<CROSSINIT>', &
                  'Missing BOND segment in input') 
    !     Check if CROSSINIT was called for the first time and
    !     allocate memory for arrays, if called for a second or later time
    !     dellocate the old memory first. If NCBON .EQ. 0 deallocate the
    !     old memory anyway.
    IF (TNCBO.GT.0 .AND. NCRUN.EQ.1) THEN
        NCBON=TNCBO
      allocate(TQCBON(NCRSU+1,NCBON),TICBONI(NCBON),TICBON1(NCBON), &
               TICBON2(NCBON),KCBON(NCRSU,NCBON),RCBON(NCRSU,NCBON))
    ELSEIF (TNCBO.GT.0 .AND. NCBON.NE.0 .AND. NCRUN.GT.1) THEN
      NCBON=TNCBO
      deallocate(KCBON,RCBON)

      IF (NCMOR.GT.0) THEN
        deallocate(ICBONI,ICBON1,ICBON2,QCBON)
      ENDIF

      allocate(TQCBON(NCRSU+1,NCBON),TICBONI(NCBON),TICBON1(NCBON), &
               TICBON2(NCBON),KCBON(NCRSU,NCBON),RCBON(NCRSU,NCBON))

    ELSEIF (TNCBO.GT.0 .AND. NCBON.EQ.0 .AND. NCRUN.GT.1) THEN
      NCBON=TNCBO
      allocate(TQCBON(NCRSU+1,NCBON),TICBONI(NCBON),TICBON1(NCBON), &
               TICBON2(NCBON),KCBON(NCRSU,NCBON),RCBON(NCRSU,NCBON))

    ELSEIF (TNCBO.EQ.0 .AND. NCBON.NE.0 .AND. NCRUN.GT.1) THEN
      NCBON=TNCBO
      deallocate(KCBON,RCBON)

      IF (NCMOR.GT.0) THEN
        deallocate(ICBONI,ICBON1,ICBON2,QCBON)
      ENDIF

    ELSE
      NCBON=TNCBO
    ENDIF


    !  Fill BOND parameter arrays of every atom pair in a surface 
    !  crossing region

    NCHAR=NCBON
    IF(NCHAR.GT.0) THEN
      allocate(TMPREA(2*NCRSU+2))
      DO I = 1,NCHAR
        READ(CparU,*) (TMPREA(J), J=1, 2*NCRSU+2)
        DO J=1,2*NCRSU+2
          IF (J.EQ.1) THEN 
            TICBON1(I)=TMPREA(J)
          ELSEIF (J.EQ.2) THEN
            TICBON2(I)=TMPREA(J)
          ELSEIF (MOD((J-1),2).EQ.0) THEN
            KCBON((J-1)/2,I)=TMPREA(J)

            IF (KCBON((J-1)/2,I).LT.0._chm_real) THEN
              TQCBON((J-1)/2,I)=.FALSE.
            ELSE 
              TQCBON((J-1)/2,I)=.TRUE. 
            ENDIF

          ELSEIF (MOD((J-2),2).EQ.0) THEN
            RCBON((J-2)/2,I)=TMPREA(J)
          ENDIF
        ENDDO
        IF (TICBON1(I).GT.TICBON2(I)) THEN
          IDUM = TICBON2(I)
          TICBON2(I) = TICBON1(I)
          TICBON1(I) = IDUM
        ENDIF
      ENDDO
      IF (allocated(TMPREA)) deallocate(TMPREA)
    ENDIF

    !-------------------------------------------------------------------
    ! Morse Bonds
    !-------------------------------------------------------------------

    READ(CparU,*) LBT,TNCMO

    IF(LBT.NE.'MORS') CALL WRNDIE(-2,'<CROSSINIT>', &
                  'Missing MORSE segment in input') 

    !     Check if CROSSINIT was called for the first time and
    !     allocate memory for arrays, if called for a second or later time
    !     dellocate the old memory first. If NCMOR .EQ. 0 deallocate the
    !     old memory anyway.
    IF (TNCMO.GT.0 .AND. NCRUN.EQ.1) THEN
      NCMOR=TNCMO
      allocate(TICMORI(NCMOR),TICMOR1(NCMOR),TICMOR2(NCMOR), &
               DCMOR(NCRSU,NCMOR),RCMOR(NCRSU,NCMOR), &
               BCMOR(NCRSU,NCMOR),TQCMOR(NCRSU+1,NCMOR))
    ELSEIF (TNCMO.GT.0 .AND. NCMOR.NE.0 .AND. NCRUN.GT.1) THEN
      NCMOR=TNCMO
      deallocate(DCMOR,RCMOR,BCMOR)
      allocate(TICMORI(NCMOR),TICMOR1(NCMOR),TICMOR2(NCMOR), &
               DCMOR(NCRSU,NCMOR),RCMOR(NCRSU,NCMOR), &
               BCMOR(NCRSU,NCMOR),TQCMOR(NCRSU+1,NCMOR))
    ELSEIF (TNCMO.GT.0 .AND. NCMOR.EQ.0 .AND. NCRUN.GT.1) THEN
      NCMOR=TNCMO
      allocate(TICMORI(NCMOR),TICMOR1(NCMOR),TICMOR2(NCMOR), &
               DCMOR(NCRSU,NCMOR),RCMOR(NCRSU,NCMOR), &
               BCMOR(NCRSU,NCMOR),TQCMOR(NCRSU+1,NCMOR))
    ELSEIF (TNCMO.EQ.0 .AND. NCMOR.NE.0 .AND. NCRUN.GT.1) THEN
      NCMOR=TNCMO
      deallocate(DCMOR,RCMOR,BCMOR)
    ELSE
      NCMOR=TNCMO
    ENDIF

    ! Make NCBON counter the sum of NCHAR+NCMOR (for exclusion lists)
    NCBON=NCHAR+NCMOR

    ! Allocate memory for unified bond arrays (Harmonic & Morse)
    IF (.NOT. allocated(QCBON)) THEN
      allocate(QCBON(NCRSU+1,NCBON))
    ENDIF

    IF (.NOT. allocated(ICBON1)) THEN
      allocate(ICBON1(NCBON),ICBON2(NCBON),ICBONI(NCBON))
    ENDIF

    ! Copy values from harmonic bonds into saved public arrays
    DO L=1, NCRSU
      DO I=1, NCHAR
        QCBON(L,I)=TQCBON(L,I)
      ENDDO
    ENDDO

    DO I=1, NCHAR
      ICBON1(I)=TICBON1(I)
      ICBON2(I)=TICBON2(I)
      ICBONI(I)=TICBONI(I)
    ENDDO

    ! Read Morse potential parameters
    IF(NCMOR.GT.0) THEN
      allocate(TMPREA(3*NCRSU+2))
      DO I = 1,NCMOR
        READ(CparU,*) (TMPREA(J), J=1, 3*NCRSU+2)

        DO J=1,3*NCRSU+2
          IF (J.EQ.1) THEN 
            TICMOR1(I)=TMPREA(J)
          ELSEIF (J.EQ.2) THEN
            TICMOR2(I)=TMPREA(J)
          ELSEIF (MOD((J),3).EQ.0) THEN
            DCMOR((J)/3,I)=TMPREA(J)
            IF (DCMOR((J)/3,I).LT.0._chm_real) THEN
              TQCMOR((J)/3,I) = .FALSE.
            ELSE 
              TQCMOR((J)/3,I) = .TRUE. 
            ENDIF

          ELSEIF (MOD((J-1),3).EQ.0) THEN
            RCMOR((J-1)/3,I)=TMPREA(J)
          ELSEIF (MOD((J-2),3).EQ.0) THEN
            BCMOR((J-2)/3,I)=TMPREA(J)
          ENDIF
        ENDDO
        IF (TICMOR1(I).GT.TICMOR2(I)) THEN
          IDUM = TICMOR2(I)
          TICMOR2(I) = TICMOR1(I)
          TICMOR1(I) = IDUM
        ENDIF
      ENDDO
      IF (allocated(TMPREA)) deallocate(TMPREA)
    ENDIF


    ! Add values from Morse bond definitions to harmonic bond arrays
    DO L=1, NCRSU
      DO I=1, NCMOR
        J=I+NCHAR
        QCBON(L,J)=TQCMOR(L,I)
      ENDDO
    ENDDO

    DO I=1, NCMOR
      J=I+NCHAR
      ICBON1(J)=TICMOR1(I)
      ICBON2(J)=TICMOR2(I)
      ICBONI(J)=TICMORI(I)
    ENDDO

    IF(NCHAR.GT.0) THEN
      deallocate(TICBON1,TICBON2,TICBONI,TQCBON)
    ENDIF

    IF(NCMOR.GT.0) THEN
      deallocate(TICMOR1,TICMOR2,TICMORI,TQCMOR)
    ENDIF

  !------------------------------------------------------------------
  !  Angles
  !------------------------------------------------------------------

    READ(CparU,*) LBT,TNCAN
    IF(LBT.NE.'ANGL') CALL WRNDIE(-2,'<CROSSINIT>', &
                'Missing ANGL segment in input') 

    !     Check if CROSSINIT was called for the first time and
    !     allocate memory for arrays, if called for a second or later time
    !     dellocate the old memory first. If NCANG .EQ. 0 deallocate the
    !     old memory anyway.
    IF (TNCAN.GT.0 .AND. NCRUN.EQ.1) THEN
      NCANG=TNCAN
      allocate(ICANG1(NCANG),ICANG2(NCANG),ICANG3(NCANG), &
               ICANGI(NCANG),KCANG(NCRSU,NCANG), &
               TCANG(NCRSU,NCANG),QCANG(NCRSU+1,NCANG))
    ELSEIF (TNCAN.GT.0 .AND. NCANG.NE.0 .AND. NCRUN.GT.1) THEN
      NCANG=TNCAN
      deallocate(ICANG1,ICANG2,ICANG3,ICANGI,KCANG,TCANG,QCANG)
      allocate(ICANG1(NCANG),ICANG2(NCANG),ICANG3(NCANG), &
               ICANGI(NCANG),KCANG(NCRSU,NCANG), &
               TCANG(NCRSU,NCANG),QCANG(NCRSU+1,NCANG))
    ELSEIF (TNCAN.GT.0 .AND. NCANG.EQ.0 .AND. NCRUN.GT.1) THEN
      NCANG=TNCAN
      allocate(ICANG1(NCANG),ICANG2(NCANG),ICANG3(NCANG), &
               ICANGI(NCANG),KCANG(NCRSU,NCANG), &
               TCANG(NCRSU,NCANG),QCANG(NCRSU+1,NCANG))
    ELSEIF (TNCAN.EQ.0 .AND. NCANG.NE.0 .AND. NCRUN.GT.1) THEN
      NCANG=TNCAN
      deallocate(ICANG1,ICANG2,ICANG3,ICANGI,KCANG,TCANG,QCANG)
    ELSE
      NCANG=TNCAN
    ENDIF

    IF(NCANG.GT.0) THEN
      allocate(TMPREA(2*NCRSU+3))
      DO I = 1,NCANG
        READ(CparU,*) (TMPREA(J), J=1, 2*NCRSU+3)

        DO J=1,2*NCRSU+3
          IF (J.EQ.1) THEN 
            ICANG1(I)=TMPREA(J)
          ELSEIF (J.EQ.2) THEN
            ICANG2(I)=TMPREA(J)
          ELSEIF (J.EQ.3) THEN
            ICANG3(I)=TMPREA(J)

          ELSEIF (MOD((J-2),2).EQ.0) THEN
            KCANG((J-2)/2,I)=TMPREA(J)

              IF (KCANG((J-2)/2,I).LT.0._chm_real) THEN
                QCANG((J-2)/2,I) = .FALSE.
              ELSE 
                QCANG((J-2)/2,I) = .TRUE. 
              ENDIF

          ELSEIF (MOD((J-3),2).EQ.0) THEN
            TCANG((J-3)/2,I)=TMPREA(J)
            TCANG((J-3)/2,I)=TCANG((J-3)/2,I)*DEGRAD
          ENDIF
        ENDDO

        QCANG(NCRSU+1,I)=.FALSE.

        DO J=1,NTHETA
          IF(ICANG2(I).EQ.JT(J)) THEN
            IF((ICANG1(I).EQ.IT(J).AND.ICANG3(I).EQ.KT(J)).OR. &
               (ICANG1(I).EQ.KT(J).AND.ICANG3(I).EQ.IT(J))) THEN
 
              QCANG(NCRSU+1,I)=.TRUE.
              ICANGI(I)=J 

            ENDIF
          ENDIF
        ENDDO

      ENDDO
      IF (allocated(TMPREA)) deallocate(TMPREA)
    ENDIF

  !----------------------------------------------------------------------
  ! Dihedrals
  !----------------------------------------------------------------------

     READ(CparU,*) LBT,TNCDI 

    IF(LBT.NE.'DIHE') CALL WRNDIE(-2,'<CROSSINIT>', &
                'Missing DIHE segment in input') 


    !     Check if CROSSINIT was called for the first time and
    !     allocate memory for arrays, if called for a second or later time
    !     dellocate the old memory first. If NCDIH .EQ. 0 deallocate the
    !     old memory anyway.
    IF (TNCDI.GT.0 .AND. NCRUN.EQ.1) THEN
      NCDIH=TNCDI
      allocate(ICDIH1(NCDIH),ICDIH2(NCDIH),ICDIH3(NCDIH), &
               ICDIH4(NCDIH),ICDIHI(NCDIH),KCDIH(NCRSU,NCDIH), &
               MCDIH(NCRSU,NCDIH),PCDIH(NCRSU,NCDIH), &
               QCDIH(NCRSU+1,NCDIH),PCDIC(NCRSU,NCDIH), &
               PCDIS(NCRSU,NCDIH))
    ELSEIF (TNCDI.GT.0 .AND. NCDIH.NE.0 .AND. NCRUN.GT.1) THEN
      NCDIH=TNCDI
      deallocate(ICDIH1,ICDIH2,ICDIH3,ICDIH4,ICDIHI, &
                 KCDIH,MCDIH,PCDIH,QCDIH,PCDIC,PCDIS)
        allocate(ICDIH1(NCDIH),ICDIH2(NCDIH),ICDIH3(NCDIH), &
               ICDIH4(NCDIH),ICDIHI(NCDIH),KCDIH(NCRSU,NCDIH), &
               MCDIH(NCRSU,NCDIH),PCDIH(NCRSU,NCDIH), &
               QCDIH(NCRSU+1,NCDIH),PCDIC(NCRSU,NCDIH), &
               PCDIS(NCRSU,NCDIH))
    ELSEIF (TNCDI.GT.0 .AND. NCDIH.EQ.0 .AND. NCRUN.GT.1) THEN
      NCDIH=TNCDI
      allocate(ICDIH1(NCDIH),ICDIH2(NCDIH),ICDIH3(NCDIH), &
               ICDIH4(NCDIH),ICDIHI(NCDIH),KCDIH(NCRSU,NCDIH), &
               MCDIH(NCRSU,NCDIH),PCDIH(NCRSU,NCDIH), &
               QCDIH(NCRSU+1,NCDIH),PCDIC(NCRSU,NCDIH), &
               PCDIS(NCRSU,NCDIH))
    ELSEIF (TNCDI.EQ.0 .AND. NCDIH.NE.0 .AND. NCRUN.GT.1) THEN
      NCDIH=TNCDI
      deallocate(ICDIH1,ICDIH2,ICDIH3,ICDIH4,ICDIHI, &
                 KCDIH,MCDIH,PCDIH,QCDIH,PCDIC,PCDIS)
    ELSE
      NCDIH=TNCDI
    ENDIF

    IF(NCDIH.GT.0) THEN
      allocate(TMPREA(3*NCRSU+4))
      DO I = 1,NCDIH

        READ(CparU,*) (TMPREA(J), J=1, 3*NCRSU+4)

        DO J=1,3*NCRSU+4
          IF (J.EQ.1) THEN 
            ICDIH1(I)=TMPREA(J)
          ELSEIF (J.EQ.2) THEN
            ICDIH2(I)=TMPREA(J)
          ELSEIF (J.EQ.3) THEN
            ICDIH3(I)=TMPREA(J)
          ELSEIF (J.EQ.4) THEN
            ICDIH4(I)=TMPREA(J)
          ELSEIF (MOD((J-2),3).EQ.0) THEN
            KCDIH((J-2)/3,I)=TMPREA(J)

            IF (KCDIH((J-2)/3,I).LT.0._chm_real) THEN
              QCDIH((J-2)/3,I) = .FALSE.
            ELSE 
              QCDIH((J-2)/3,I) = .TRUE.
            ENDIF

          ELSEIF (MOD((J-3),3).EQ.0) THEN
            MCDIH((J-3)/3,I)=TMPREA(J)
          ELSEIF (MOD((J-4),3).EQ.0) THEN
            PCDIH((J-4)/3,I)=TMPREA(J)
            PCDIH((J-4)/3,I) = PCDIH((J-4)/3,I)*DEGRAD
            PCDIC((J-4)/3,I) = COS(PCDIH((J-4)/3,I))
            PCDIS((J-4)/3,I) = SIN(PCDIH((J-4)/3,I))
          ENDIF
        ENDDO

        QCDIH(NCRSU+1,I)=.FALSE. 

        DO J = 1,NPHI
          IF((ICDIH1(I).EQ.IP(J).AND.ICDIH4(I).EQ.LP(J)).OR. &
            (ICDIH1(I).EQ.LP(J).AND.ICDIH4(I).EQ.IP(J))) THEN
            IF((ICDIH2(I).EQ.JP(J).AND.ICDIH3(I).EQ.KP(J)).OR. &
              (ICDIH2(I).EQ.KP(J).AND.ICDIH3(I).EQ.JP(J))) THEN

              QCDIH(NCRSU+1,I)=.TRUE.
              ICDIHI(I)=J
            ENDIF
          ENDIF
        ENDDO 

      ENDDO
      IF (allocated(TMPREA)) deallocate(TMPREA)
    ENDIF


    !====================================================================
    ! Check if every atom included in the BOND or MORS section is 
    ! also defined in the ATOM section.
    ! If not, print a warning message as energy evaluation can go
    ! wrong in certain cases when this is not the case.
    !====================================================================
    !

    CNT1=2
    DO I=1,NCBON

      IF (CNT1.EQ.2) THEN
        CNT1=0
        DO J=1,NCRAT
          IF (ICATOM(J).EQ.ICBON1(I).OR. &
              ICATOM(J).EQ.ICBON2(I)) CNT1=CNT1+1
        ENDDO

        IF (CNT1.LT.2) THEN
          WRITE(OUTU,*)' '
          WRITE(OUTU,*)' '
          WRITE(OUTU,810) 
          WRITE(OUTU,*)'CROSSINIT> BOND or MORS definition detected including'
          WRITE(OUTU,*)'CROSSINIT> atoms which are not defined in ATOM section.'
          WRITE(OUTU,*)'CROSSINIT> This situation might introduce inconsistencies'
          WRITE(OUTU,*)'CROSSINIT> in the energy comparison of ARMD.'
          WRITE(OUTU,810)
          CALL WRNDIE(0,'<CROSSINIT>', &
                        'More ATOM definitions suggested')
        ENDIF
      ENDIF
    ENDDO


    !====================================================================
    ! Write out the input parameters
    !====================================================================
    !

    IF(PRNLEV.GE.5) THEN
       WRITE(OUTU,800)
       WRITE(OUTU,801)
       WRITE(OUTU,802) qTANH
       WRITE(OUTU,805) XTIME,XCROSU,XCROSF
       WRITE(OUTU,800)
       IF (BCHK.GT.0.0_CHM_REAL) WRITE(OUTU,796)
       IF (BCHK.GT.0.0_CHM_REAL) WRITE(OUTU,797) BCHK
       WRITE(OUTU,800)
       WRITE(OUTU,803) NCRSU
       WRITE(OUTU,800)
       WRITE(OUTU,798)
       WRITE(OUTU,799)
       K=1
       DO I=1, NCRSU-1
         DO J=I+1, NCRSU
           WRITE(OUTU,804) I,J,SSHIFT(K),SBARTO(K)
           K=K+1
         ENDDO
       ENDDO
       WRITE(OUTU,800)
     ENDIF
800 FORMAT(76('-'))
801 FORMAT(' CROSSINIT> Input values for XING options:')
802 FORMAT(' CROSSINIT> TANH = ',L5)
796 FORMAT(' CROSSINIT> Cutoff distance checking for possible' &
              ' bond formations:')
797 FORMAT(' CROSSINIT> BCHK = ',F3.1)
803 FORMAT(' CROSSINIT> Number of user defined surfaces = ',I2)
798 FORMAT(' CROSSINIT> Surface energetics:')
799 FORMAT(' CROSSINIT> Surfaces         BarrShift   BarrTol')
804 FORMAT(' CROSSINIT> ',I2, ' and ',I2, '      ',F8.3,'   ',F8.3)
805 FORMAT(' CROSSINIT> XTIME   =',I8  ,'   XCROSU=',I8, &
           '   XCROSF =',I8)

    !====================================================================

    !====================================================================
    ! Write out the force field parameters:
    !

    IF(PRNLEV.GE.5) THEN

      WRITE(OUTU,810)
      DO J=1, NCRSU
        WRITE(OUTU,806) J
        WRITE(OUTU,810)
        WRITE(OUTU,807) NCRAT
806     FORMAT(' CROSSINIT> The following parameters were read for ', &
               'surface',I2)
807     FORMAT(' CROSSINIT> Number of atoms in crossing zone: ',I8)
        IF(PRNLEV.GE.6) THEN
          WRITE(OUTU,808)
808       FORMAT(' CROSSINIT>   Atom  Chg    Eps    Sigma')
          DO I=1,NCRAT   

            WRITE(OUTU,809) ICATOM(I),ECCHG(J,I),VCEPS(J,I),VCRAD(J,I) 
          ENDDO
809       FORMAT(' CROSSINIT> ',I6,3F8.4)
        ENDIF

        WRITE(OUTU,810) 
810     FORMAT(' CROSSINIT> ',50('='))

        WRITE(OUTU,811) NCBON
811     FORMAT(' CROSSINIT> Number of bonds in crossing zone: ',I8)

        IF(PRNLEV.GE.6) THEN
          IF(NCHAR.GT.0) THEN     
            WRITE(OUTU,812)
812         FORMAT(' CROSSINIT> Harmonic bonds')
            WRITE(OUTU,813) 
813         FORMAT(' CROSSINIT>  Atom1 Atom2   FC      Req      ') 
            DO I=1,NCHAR
              IF (KCBON(J,I).GT.0.0_CHM_REAL) THEN
                WRITE(OUTU,814) ICBON1(I),ICBON2(I),KCBON(J,I), &
                                RCBON(J,I) 
              ELSE 
                WRITE(OUTU,815) ICBON1(I),ICBON2(I) 
              ENDIF
            ENDDO
          ENDIF 
        ENDIF
814     FORMAT(' CROSSINIT> ',2I6,2F8.3)
815     FORMAT(' CROSSINIT> ',2I6,'   -         -')
        WRITE(OUTU,901) NCMOR
901     FORMAT(' CROSSINIT> Number of Morse Bonds:',I8)
        IF(PRNLEV.GE.6) THEN
          IF(NCMOR.GT.0) THEN   
            WRITE(OUTU,902) 
902         FORMAT(' CROSSINIT> Atom1 Atom2  De     Req     beta   ')
            DO I=1,NCMOR
              IF (QCBON(J,I+NCHAR)) THEN
                WRITE(OUTU,907) ICBON1(I+NCHAR),ICBON2(I+NCHAR), &
                                DCMOR(J,I),RCMOR(J,I),BCMOR(J,I) 
              ELSE
                WRITE(OUTU,903) ICBON1(I+NCHAR),ICBON2(I+NCHAR)
903             FORMAT(' CROSSINIT> ',2I6,'   -      -     -')
              ENDIF
            ENDDO 
          ENDIF
        ENDIF
907     FORMAT(' CROSSINIT> ',2I6,3F8.3)

        WRITE(OUTU,810) 
        WRITE(OUTU,910) NCANG
910     FORMAT(' CROSSINIT> Number of angles in crossing zone: ',I8)
        IF(PRNLEV.GE.6) THEN 
          IF(NCANG.GT.0) THEN 
            WRITE(OUTU,911) 
911         FORMAT(' CROSSINIT> Atom1 Atom2 Atom3  FC     Theq   ')     
            DO I=1,NCANG
              IF (KCANG(J,I).GT.0.0_CHM_REAL) THEN
                WRITE(OUTU,912) ICANG1(I),ICANG2(I),ICANG3(I), &
                                KCANG(J,I),TCANG(J,I) 
              ELSE
                WRITE(OUTU,915) ICANG1(I),ICANG2(I),ICANG3(I)
915             FORMAT(' CROSSINIT> ',3I6,'   -        -')
              ENDIF
            ENDDO 
          ENDIF
        ENDIF 
912     FORMAT(' CROSSINIT> ',3I6,2F8.3)

        WRITE(OUTU,810) 
        WRITE(OUTU,916) NCDIH
916     FORMAT(' CROSSINIT> Number of dihedrals in crossing zone: ',I8)
        IF(PRNLEV.GE.6) THEN
          IF(NCDIH.GT.0) THEN
            WRITE(OUTU,917) 
917         FORMAT(' CROSSINIT>  Atom1 Atom2 Atom3 Atom4   FC    ', &
                   'Mult    Phi     ')
            DO I=1,NCDIH
              IF (KCDIH(J,I).GT.0.0_CHM_REAL) THEN
                WRITE(OUTU,918) ICDIH1(I),ICDIH2(I),ICDIH3(I), &
                    ICDIH4(I),KCDIH(J,I),MCDIH(J,I),PCDIH(J,I)
              ELSE
                WRITE(OUTU,921) ICDIH1(I),ICDIH2(I),ICDIH3(I),ICDIH4(I)
921             FORMAT(' CROSSINIT> ',4I6,'   -       -       -')
              ENDIF
            ENDDO
          ENDIF   
        ENDIF
918     FORMAT(' CROSSINIT> ',4I6,F8.3,I4,'   ',F8.3)
 
        WRITE(OUTU,800)
        WRITE(OUTU,810)

        ENDDO
    ! ------Leaving surface loop----------------------------

        WRITE(OUTU,810) 
      ENDIF


    !================================================================
    ! Build exclusion lists for 1-2,1-3 and list of 1-4 interactions 
    !================================================================
    DO I=1, NCRSU+1
      NCEXC(I)=0
      NC14(I)=0
    ENDDO

    DO I=1, NCRSU
      NCINC(I)=0
      NCIN14(I)=0
    ENDDO

    ! Deallocate memory for bond, angle and dihedral exclusion lists of 
    ! CROSSINIT has been previously called already.
    IF (NCRUN.GT.1) deallocate(ICEXC,IC14,ICINC,ICIN14)

    ! Allocate temporary memory for nonbond-exclusion arrays
    ! The arraysize is temporarily set to a large number. 
    ! When the exact arraysizes are evaluated this data is transfered 
    ! to ICEXC and ICINC.
    MAXCEXC=200*NCBON+100
    allocate(TMPEXC(NCRSU+1,MAXCEXC,2),TMPINC(NCRSU,MAXCEXC,2))

    DO I=1,NCRSU+1
      DO J=1,MAXCEXC
        TMPEXC(I,J,1)=0.0_CHM_REAL
        TMPEXC(I,J,2)=0.0_CHM_REAL
      ENDDO
    ENDDO

    DO I=1,NCRSU
      DO J=1,MAXCEXC
        TMPINC(I,J,1)=0.0_CHM_REAL
        TMPINC(I,J,2)=0.0_CHM_REAL
      ENDDO
    ENDDO

    ! Allocate temporary memory for 1-4 interaction arrays
    ! The arraysize is temporarily set to a large number. 
    ! When the exact arraysize is evaluated this data is transfered to
    ! arrays IC14 and ICIN14
    MAXCIC14=200*NCBON+100
    allocate(TMPC14(NCRSU+1,MAXCIC14,2),TMPIN14(NCRSU,MAXCIC14,2))

    DO I=1,NCRSU+1
      DO J=1,MAXCIC14
        TMPC14(I,J,1)=0.0_CHM_REAL
        TMPC14(I,J,2)=0.0_CHM_REAL
      ENDDO
    ENDDO

    DO I=1,NCRSU
      DO J=1,MAXCIC14
        TMPIN14(I,J,1)=0.0_CHM_REAL
        TMPIN14(I,J,2)=0.0_CHM_REAL
      ENDDO
    ENDDO

    DO I=1,NCBON
      IB1=ICBON1(I)
      IB2=ICBON2(I) 
      QCBON(NCRSU+1,I)=.FALSE. 

    !-----------------------------------------------------------------
    ! Check if bond already exists in PSF  
    !-----------------------------------------------------------------
      DO J=1,NBOND
        IF(((IB1.EQ.IB(J)).AND.(IB2.EQ.JB(J))).OR.((IB1.EQ.JB(J)) &
             .AND.(IB2.EQ.IB(J)))) THEN 
          QCBON(NCRSU+1,I)=.TRUE.
          ICBONI(I)=J
        ENDIF  
      ENDDO
    ENDDO

    !------------------------------------------------------------------
    ! If not, exclusions should be made
    !------------------------------------------------------------------
    DO I=1,NCBON
      IB1=ICBON1(I)
      IB2=ICBON2(I) 

      IF (.NOT.QCBON(NCRSU+1,I)) THEN
    ! Loop over each surface
        DO L=1, NCRSU

          !------------------------------------------------------------
          ! 1-2 exclusions
          !------------------------------------------------------------
          IF (QCBON(L,I).AND.(IB1.NE.IB2)) THEN
            NCEXC(L)=NCEXC(L)+1
            IF (NCEXC(L).GT.MAXCEXC) THEN
              CALL WRNDIE(-2,'<CROSSINIT>','Too many exclusions')
            ENDIF

            TMPEXC(L,NCEXC(L),1) = IB1
            TMPEXC(L,NCEXC(L),2) = IB2
          ENDIF

          !----------------------------------------------------------------
          ! 1-3 exclusions
          !---------------------------------------------------------------

          ! Identify all neighboring atoms of excluded bonds 
          ! (they have angle potential from PSF which need to be 
          ! substracted on the corresponding surfaces) 
          DO J=1,NBOND

            IF(IB1.EQ.IB(J).OR.IB1.EQ.JB(J).OR.IB2.EQ.IB(J).OR. &
               IB2.EQ.JB(J)) THEN

              IF(IB1.EQ.IB(J)) THEN 
                IBB1=JB(J)
                IBB2=IB2
                IBB0=IB1  
              ENDIF

              IF(IB1.EQ.JB(J)) THEN
                IBB1=IB(J)
                IBB2=IB2
                IBB0=IB1  
              ENDIF

              IF(IB2.EQ.IB(J)) THEN
                IBB1=JB(J)
                IBB2=IB1
                IBB0=IB2 
              ENDIF
 
              IF(IB2.EQ.JB(J)) THEN
                IBB1=IB(J)    
                IBB2=IB1
                IBB0=IB2
              ENDIF

              IF(QCBON(L,I).AND.(IBB1.NE.IBB2)) THEN
                EXFL=.true.
                ! Check if bond between IBB0 and IBB1 is removed on L,
                ! if so set EXFL back to .false.
                DO K=1,NCBON
                  IF (((ICBON1(K).EQ.IBB0).AND.(ICBON2(K).EQ.IBB1)) &
                      .OR.((ICBON1(K).EQ.IBB1).AND. &
                      (ICBON2(K).EQ.IBB0))) THEN

                    IF (.NOT.QCBON(L,K)) EXFL=.false. 

                  ELSEIF (((ICBON1(K).EQ.IBB0).AND. &
                         (ICBON2(K).EQ.IBB2)).OR. &
                         ((ICBON1(K).EQ.IBB2).AND. &
                         (ICBON2(K).EQ.IBB0))) THEN

                    IF (.NOT.QCBON(L,K)) EXFL=.false. 

                  ENDIF
                ENDDO

                IF (EXFL) THEN
                  NCEXC(L)=NCEXC(L)+1
                  IF (NCEXC(L).GT.MAXCEXC) THEN              
                    CALL WRNDIE(-2,'<CROSSINIT>', &
                                'Too many exclusions')
                  ENDIF

                  TMPEXC(L,NCEXC(L),1)=IBB2
                  TMPEXC(L,NCEXC(L),2)=IBB1
                ENDIF
              ENDIF

              !------------------------------------------------------------
              ! 1-4 interactions Case 1: 1-new bond-2-old bond-3-old bond-4
              !------------------------------------------------------------
              DO K=1,NBOND
                IF (IB(K).EQ.IBB1.OR.JB(K).EQ.IBB1) THEN 
                  IF (.NOT.(JB(K).EQ.IBB0.OR.IB(K).EQ.IBB0)) THEN
                    IF (IB(K).EQ.IBB1) IBB3 = JB(K)
                    IF (JB(K).EQ.IBB1) IBB3 = IB(K)

                    IF(QCBON(L,I).AND.(IBB2.NE.IBB3)) THEN
                      NC14(L)=NC14(L)+1
                      IF (NC14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                               'Too many 1-4 interactions')     
                      ENDIF
                      TMPC14(L,NC14(L),1)=IBB2
                      TMPC14(L,NC14(L),2)=IBB3 
                    ENDIF

                  ENDIF 
                ENDIF

                !-------------------------------------------------------------
                ! 1-4 interactions Case 2: 1-old bond-2-new bond-3-old bond-4
                !-------------------------------------------------------------

                IF (IB(K).EQ.IBB2.OR.JB(K).EQ.IBB2) THEN
                  IF (IB(K).EQ.IBB2) IBB3 = JB(K)
                  IF (JB(K).EQ.IBB2) IBB3 = IB(K)

                  IF(QCBON(L,I).AND.(IBB1.NE.IBB3)) THEN
                    NC14(L)=NC14(L)+1 
                    IF (NC14(L).GT.MAXCIC14) THEN
                      CALL WRNDIE(-2,'<CROSSINIT>', &
                       'Too many 1-4 interactions')     
                    ENDIF
                    TMPC14(L,NC14(L),1)=IBB1
                    TMPC14(L,NC14(L),2)=IBB3 
                  ENDIF

                ENDIF 
              ENDDO

              !--------------------------------------------------------------
              !--1-4 interactions Case 3: 1-new bond-2-old bond-3-new bond-4
              !--------------------------------------------------------------

              DO K=1,NCBON
                IF ((K.NE.I).AND.(.NOT.QCBON(NCRSU+1,K))) THEN
                  IF(ICBON1(K).EQ.IBB1.OR.ICBON2(K).EQ.IBB1) THEN 
                    IF (ICBON1(K).EQ.IBB1) IBB3 = ICBON2(K)
                    IF (ICBON2(K).EQ.IBB1) IBB3 = ICBON1(K)

                    IF(QCBON(L,I).AND.QCBON(L,K).AND. &
                       (IBB2.NE.IBB3)) THEN
                      NC14(L)=NC14(L)+1 
                      IF (NC14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                                 'Too many 1-4 interactions')
                      ENDIF
                      TMPC14(L,NC14(L),1)=IBB2
                      TMPC14(L,NC14(L),2)=IBB3 
                    ENDIF
                  ENDIF
                ENDIF
              ENDDO
              !-------------end of loop K----------------------
            ENDIF
          ENDDO
          !-------------end of loop J--------------------------

          !--------------------------------------------------------------
          ! Check for 1-3 and 1-4 over multiple new bonds

          !--------------------------------------------------------------
          ! 1-3 Case 2: 1-new bond-2-new bond-3             
          !--------------------------------------------------------------

          DO J=1,NCBON
            IF((J.NE.I).AND.(.NOT.QCBON(NCRSU+1,J))) THEN  
              IF((ICBON1(J).EQ.IB1.OR.ICBON2(J).EQ.IB2) &
                .OR.(ICBON1(J).EQ.IB2.OR.ICBON2(J).EQ.IB1)) THEN

                IF (IB1.EQ.ICBON1(J)) THEN
                  IBB2=ICBON2(J)
                  IBB1=IB2
                  IBB0=IB1 
                ENDIF

                IF (IB1.EQ.ICBON2(J)) THEN
                  IBB2=ICBON1(J)
                  IBB1=IB2
                  IBB0=IB1 
                ENDIF

                IF (IB2.EQ.ICBON2(J)) THEN
                  IBB2=ICBON1(J) 
                  IBB1=IB1
                  IBB0=IB2
                ENDIF

                IF (IB2.EQ.ICBON1(J)) THEN
                  IBB2=ICBON2(J) 
                  IBB1=IB1
                  IBB0=IB2 
                ENDIF

                IF(QCBON(L,I).AND.QCBON(L,J).AND. &
                   (IBB1.NE.IBB2)) THEN

                  EXFL=.true.
                  ! Check if bond between IBB0 and IBB1 is removed on L,
                  ! if so set EXFL back to .false.
                  DO K=1,NCBON
                    IF (((ICBON1(K).EQ.IBB0).AND.(ICBON2(K).EQ.IBB1)) &
                       .OR.((ICBON1(K).EQ.IBB1).AND. &
                        (ICBON2(K).EQ.IBB0))) THEN

                      IF (.NOT.QCBON(L,K)) EXFL=.false. 
         
                    ELSEIF (((ICBON1(K).EQ.IBB0).AND. &
                           (ICBON2(K).EQ.IBB2)).OR.((ICBON1(K).EQ.IBB2) &
                           .AND.(ICBON2(K).EQ.IBB0))) THEN

                      IF (.NOT.QCBON(L,K)) EXFL=.false. 
         
                    ENDIF
                  ENDDO

                  IF (EXFL) THEN
                    NCEXC(L)=NCEXC(L)+1 
                    IF (NCEXC(L).GT.MAXCEXC) THEN
                      CALL WRNDIE(-2,'<CROSSINIT>', &
                             'Too many exclusions')
                    ENDIF
       !   WRITE(OUTU,*)'Second 1-3 exclusions on:',L,IBB1,IBB2,IBB0
                    TMPEXC(L,NCEXC(L),1)=IBB1
                    TMPEXC(L,NCEXC(L),2)=IBB2
                  ENDIF
 
                ENDIF

                !-------------------------------------------------------------
                !--1-4 interactions Case 4:1-new bond-2-new bond-3-old bond-4
                !------------------------------------------------------------- 

                DO K=1,NBOND
                  IF (IB(K).EQ.IBB2.OR.JB(K).EQ.IBB2) THEN 
                    IF (IB(K).EQ.IBB2) IBB3 = JB(K)
                    IF (JB(K).EQ.IBB2) IBB3 = IB(K)

                    IF(QCBON(L,I).AND.QCBON(L,J).AND. &
                       (IBB1.NE.IBB3)) THEN
                      NC14(L)=NC14(L)+1 
                      IF (NC14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', & 
                               'Too many 1-4 interactions')     
                      ENDIF
                      TMPC14(L,NC14(L),1)=IBB1
                      TMPC14(L,NC14(L),2)=IBB3 
                    ENDIF

                  ENDIF
                ENDDO
                !------------End of loop K---------------------------

                !-------------------------------------------------------------
                ! 1-4 interactions Case 5: 1-new bond-2-new bond-3-new bond-4
                !-------------------------------------------------------------

                DO K=1,NCBON
                  IF(((K.NE.I).AND.(K.NE.J)).AND. &
                     (.NOT.QCBON(NCRSU+1,K)) ) THEN
                    IF((ICBON1(K).EQ.IBB2).OR. &
                       (ICBON2(K).EQ.IBB2)) THEN

                      IF(ICBON1(K).EQ.IBB2) THEN
                        IBB3=ICBON2(K)  
                      ELSE
                        IBB3=ICBON1(K)
                      ENDIF

                      IF(QCBON(L,I).AND.QCBON(L,J).AND. &
                         QCBON(L,K).AND.(IBB1.NE.IBB3)) THEN
                        NC14(L)=NC14(L)+1 
                        IF (NC14(L).GT.MAXCIC14) THEN
                          CALL WRNDIE(-2,'<CROSSINIT>', &
                                  'Too many 1-4 interactions')     
                        ENDIF
                        TMPC14(L,NC14(L),1)=IBB1
                        TMPC14(L,NC14(L),2)=IBB3 
                      ENDIF

                    ENDIF
                  ENDIF
                ENDDO 
                !---------------End of loop K-------------------    
              ENDIF 
            ENDIF
          ENDDO
          !---------------End of loop J------------------------- 
        ENDDO
        !---------------End of loop L---------------------------

      ENDIF
    ENDDO
    !---------------End of loop I-------------------------------


    !==================================================================
    !  Build a list of standard exclusions for PSF surface involving 
    !  crossing atoms (part of ATOM section)
    !==================================================================

    NCEXC(NCRSU+1)=0

    DO I=1,NCRAT
      IB1=ICATOM(I)         
      !-------1-2         
         DO J=1,NBOND
           IF((IB(J).EQ.IB1).OR.(JB(J).EQ.IB1)) THEN
             IF(IB(J).EQ.IB1) IB2=JB(J)
             IF(JB(J).EQ.IB1) IB2=IB(J)

             NCEXC(NCRSU+1)=NCEXC(NCRSU+1)+1
             IF (NCEXC(NCRSU+1).GT.MAXCEXC) THEN
               CALL WRNDIE(-2,'<CROSSINIT>', &
                         'Too many exclusions')     
             ENDIF
             TMPEXC(NCRSU+1,NCEXC(NCRSU+1),1)=IB1
             TMPEXC(NCRSU+1,NCEXC(NCRSU+1),2)=IB2
             !-------1-3
             DO K=1,NBOND
               IF((IB(K).EQ.IB2).OR.(JB(K).EQ.IB2)) THEN
                 IF(IB(K).EQ.IB2) IB3=JB(K)  
                 IF(JB(K).EQ.IB2) IB3=IB(K)  

                 IF(.NOT.(IB3.EQ.IB1)) THEN
                   NCEXC(NCRSU+1)=NCEXC(NCRSU+1)+1
                   IF (NCEXC(NCRSU+1).GT.MAXCEXC) THEN
                     CALL WRNDIE(-2,'<CROSSINIT>', &
                               'Too many exclusions')     
                   ENDIF
     !   WRITE(OUTU,*)'ATOM section exclusion detected'
                   TMPEXC(NCRSU+1,NCEXC(NCRSU+1),1)=IB1
                   TMPEXC(NCRSU+1,NCEXC(NCRSU+1),2)=IB3
                   !-------1-4
                   DO L=1,NBOND
                     IF(IB(L).EQ.IB3.OR.JB(L).EQ.IB3) THEN 
                       IF(IB(L).EQ.IB3) IB4=JB(L)
                       IF(JB(L).EQ.IB3) IB4=IB(L)

                       IF(.NOT.(IB4.EQ.IB2.OR.IB4.EQ.IB1)) THEN
                         NC14(NCRSU+1)=NC14(NCRSU+1)+1
                         IF (NC14(NCRSU+1).GT.MAXCIC14) THEN
                           CALL WRNDIE(-2,'<CROSSINIT>', &
                                  'Too many 1-4 interactions')     
                         ENDIF
                         TMPC14(NCRSU+1,NC14(NCRSU+1),1)=IB1
                         TMPC14(NCRSU+1,NC14(NCRSU+1),2)=IB4
                       ENDIF
                     ENDIF
                   ENDDO
                   !-------end 1-4
                 ENDIF
               ENDIF
             ENDDO
             !-------end 1-3
           ENDIF
         ENDDO
         !-------end 1-2
       ENDDO
       !-------end cross atom loop

  !===========================================================
  ! Check for double counted exclusion.
  !===========================================================

       DO L=1, NCRSU+1

  ! DO I=1,NC14(L)
  !  WRITE(OUTU,*)'1-4 excl before, L, IB1, IB2', L, TMPC14(L,I,1),TMPC14(L,I,2)
  ! ENDDO

         ! Check 1-2 & 1-3
         IF (NCEXC(L).GT.1) THEN

             NEWEXC=NCEXC(L)
             CNT2=0
             DO I=1,(NCEXC(L)-1)
               IF (I.LE.(NEWEXC-1)) THEN
                 IB1=TMPEXC(L,I,1)
                 IB2=TMPEXC(L,I,2)

                 CNT1=0
                 DO J=I+1,NCEXC(L)
                   IF (J.LE.NCEXC(L)-CNT2) THEN 
                     JB1=TMPEXC(L,J-CNT1,1)
                     JB2=TMPEXC(L,J-CNT1,2)

                     IF ((IB1.EQ.JB1.AND.IB2.EQ.JB2).OR. &
                         (IB1.EQ.JB2.AND.IB2.EQ.JB1)) THEN

                       DO K=J-CNT1,NEWEXC
                         TMPEXC(L,K,1)=TMPEXC(L,K+1,1)
                         TMPEXC(L,K,2)=TMPEXC(L,K+1,2)
                       ENDDO

                       CNT1=CNT1+1
                       NEWEXC=NEWEXC-1

                     ENDIF
                   ENDIF
                 ENDDO
                 CNT2=CNT2+CNT1
               ENDIF
             ENDDO
             NCEXC(L)=NEWEXC

         ENDIF

         ! Check 1-4
         IF (NC14(L).GT.1) THEN

             NEWEXC=NC14(L)
             CNT2=0
             DO I=1,(NC14(L)-1)
               IF (I.LE.(NEWEXC-1)) THEN
                 IB1=TMPC14(L,I,1)
                 IB2=TMPC14(L,I,2)

                 CNT1=0
                 DO J=I+1,NC14(L)
                   IF (J.LE.NC14(L)-CNT2) THEN 
                     JB1=TMPC14(L,J-CNT1,1)
                     JB2=TMPC14(L,J-CNT1,2)

                     IF ((IB1.EQ.JB1.AND.IB2.EQ.JB2).OR. &
                        (IB1.EQ.JB2.AND.IB2.EQ.JB1)) THEN

                       DO K=J-CNT1,NEWEXC
                         TMPC14(L,K,1)=TMPC14(L,K+1,1)
                         TMPC14(L,K,2)=TMPC14(L,K+1,2)
                       ENDDO

                       CNT1=CNT1+1
                       NEWEXC=NEWEXC-1
 
                     ENDIF
                   ENDIF
                 ENDDO
                 CNT2=CNT2+CNT1
               ENDIF
             ENDDO
             NC14(L)=NEWEXC

         ENDIF

         ! Check if 1-4 exclusions already show up as 1-2 or 1-3
         ! and remove them from 1-4
         IF ((NCEXC(L).GT.0).AND.(NC14(L).GT.0)) THEN

           NEWEXC=NC14(L)
           DO I=1,NCEXC(L)

             IB1=TMPEXC(L,I,1)
             IB2=TMPEXC(L,I,2)
             CNT1=0
             DO J=1,NC14(L)
               IF ((IB1.EQ.TMPC14(L,J,1).AND.IB2.EQ. &
                 TMPC14(L,J,2)).OR.IB2.EQ.TMPC14(L,J,1) &
                 .AND.IB1.EQ.TMPC14(L,J,2)) THEN

                 DO M=1,NEWEXC
                   IF (M.GT.J-CNT1) THEN
 
                     TMPC14(L,M-1,1)=TMPC14(L,M,1)
                     TMPC14(L,M-1,2)=TMPC14(L,M,2)

                   ENDIF
                 ENDDO
                 CNT1=CNT1+1
                 NEWEXC=NEWEXC-1
               ENDIF
             ENDDO
           ENDDO
           NC14(L)=NEWEXC
         ENDIF

 !  DO I=1,NC14(L)
 !   WRITE(OUTU,*)'1-4 excl after double, L, IB1, IB2', L, TMPC14(L,I,1),TMPC14(L,I,2)
 !  ENDDO

       ENDDO



  !----------------------------------------------------------------
  ! SL: 10/03/24 Remove 1-4 exclusions which are bonded to each
  !              other in the PSF but not on L. This can happen for 
  !              cyclic transition state structures!!!!

       DO L=1, NCRSU

        IF (NC14(L).GT.0) THEN


          CNT1=0
          NEWEXC=NC14(L)
          DO I=1,NEWEXC
            IB1=TMPC14(L,I-CNT1,1)
            IB2=TMPC14(L,I-CNT1,2)
            !Check for the PSF bond
            DO J=1,NBOND
              IF ((IB1.EQ.IB(J).AND.IB2.EQ.JB(J)).OR. &
                 (IB2.EQ.IB(J).AND.IB1.EQ.JB(J))) THEN
                !Check if there is no bond defined on L

                DO K=1,NCBON

                  IF ((ICBON1(K).EQ.IB1.AND.ICBON2(K) &
                    .EQ.IB2).OR.((ICBON2(K).EQ.IB1.AND. &
                    ICBON1(K).EQ.IB2)).AND.QCBON(L,K)) THEN
                  
                    DO M=1,NEWEXC

                      IF (M.GT.I-CNT1) THEN
                        TMPC14(L,M-1,1)=TMPC14(L,M,1)
                        TMPC14(L,M-1,2)=TMPC14(L,M,2)
                      ENDIF
                    ENDDO
                    CNT1=CNT1+1
                    NEWEXC=NEWEXC-1
                  ENDIF
                ENDDO
              ENDIF
            ENDDO

          ENDDO
          NC14(L)=NEWEXC

        ENDIF

  ! DO I=1,NC14(L)
  !  WRITE(OUTU,*)'1-4 excl after cycle, L, IB1, IB2',L, TMPC14(L,I,1),TMPC14(L,I,2)
  ! ENDDO

      ENDDO
      ! ------------ End of surface loop L ------------------------

  !===================END of exclusion part==========================


  !==================== Inclusion part ==============================
  ! 08/12/1 SL: Build non-bond inclusion list for all bonds 
  !             present on the PSF but missing on a specific surface. 

      DO I=1,NCBON
        IB1=ICBON1(I)
        IB2=ICBON2(I)

        ! ... Check if bond is present on PSF
        IF (QCBON(NCRSU+1,I)) THEN

          !------------------------------------------------------------------
          ! 1-2 inclusions
          !------------------------------------------------------------------

          DO L=1, NCRSU
            IF (.NOT.QCBON(L,I)) THEN

              NCINC(L)=NCINC(L)+1
              IF (NCINC(L).GT.MAXCEXC) THEN
                CALL WRNDIE(-2,'<CROSSINIT>', &
                     'Too many inclusions')
              ENDIF
              TMPINC(L,NCINC(L),1) = IB1
              TMPINC(L,NCINC(L),2) = IB2

            ENDIF

            !----------------------------------------------------------------
            ! 1-3 inclusions coming from PSF bonds
            !---------------------------------------------------------------

            ! ... Identify all atoms of a neighboring bond opposite 
            ! ... to the bond breakage to include this atom in the inclusion list
            IF (.NOT.QCBON(L,I)) THEN
              DO J=1,NBOND
                ! identify bonds opposite to IB1
                INFL=.false.
                IF((IB2.EQ.IB(J)).AND.(JB(J).NE.IB1)) THEN
                  IBB0=IB1
                  IBB1=IB2
                  IBB2=JB(J)
                  IF (.NOT.QCBON(L,I)) INFL=.true.
                ENDIF

                IF((IB2.EQ.JB(J)).AND.(IB(J).NE.IB1)) THEN
                  IBB0=IB1
                  IBB1=IB2
                  IBB2=IB(J)
                  IF (.NOT.QCBON(L,I)) INFL=.true.
                ENDIF

                ! identify bonds opposite to IB2
                IF((IB1.EQ.IB(J)).AND.(JB(J).NE.IB2)) THEN
                  IBB0=IB2
                  IBB1=IB1
                  IBB2=JB(J)
                  IF (.NOT.QCBON(L,I)) INFL=.true.
                ENDIF

                IF((IB1.EQ.JB(J)).AND.(IB(J).NE.IB2)) THEN
                  IBB0=IB2
                  IBB1=IB1
                  IBB2=IB(J)
                  IF (.NOT.QCBON(L,I)) INFL=.true.
                ENDIF


                IF (INFL.AND.(IBB0.NE.IBB2)) THEN
                  NCINC(L)=NCINC(L)+1
                  IF (NCINC(L).GT.MAXCEXC) THEN
                    CALL WRNDIE(-2,'<CROSSINIT>', &
                       'Too many inclusions')
                  ENDIF
                  TMPINC(L,NCINC(L),1) = IBB0
                  TMPINC(L,NCINC(L),2) = IBB2
                ENDIF

                !---------------------------------------------------------------
                ! Identify 1-4 interactions coming from PSF bonds
                !---------------------------------------------------------------
                IF (INFL) THEN
                  DO K=1,NBOND

                    INF14=.false.
                    IF((IBB2.EQ.IB(K)).AND.(JB(K).NE.IBB1)) THEN 
                      IBB3=JB(K)
                      INF14=.true.
                    ELSEIF((IBB2.EQ.JB(K)).AND. &
                           (IB(K).NE.IBB1)) THEN 
                      IBB3=IB(K)
                      INF14=.true.
                    ENDIF


                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO
                  ! ------- End of loop K ----------------------

                  !---------------------------------------------------------------
                  ! Identify 1-4 interactions coming from RMD bonds
                  !---------------------------------------------------------------

                  DO K=1,NCBON

                    IF((IBB2.EQ.ICBON1(K)).AND. &
                       (ICBON2(K).NE.IBB1)) THEN 

                      IF (QCBON(L,K).AND.INFL.AND. &
                         (IBB0.NE.ICBON2(K))) THEN
                        NCIN14(L)=NCIN14(L)+1
                        IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                           'Too many inclusions')
                        ENDIF
                        TMPIN14(L,NCIN14(L),1) = IBB0
                        TMPIN14(L,NCIN14(L),2) = ICBON2(K)
                      ENDIF

                    ENDIF

                    IF((IBB2.EQ.ICBON2(K)).AND. &
                      (ICBON1(K).NE.IBB1)) THEN 

                      IF (QCBON(L,K).AND.INFL.AND. &
                         (IBB0.NE.ICBON1(K))) THEN
                        NCIN14(L)=NCIN14(L)+1
                        IF (NCIN14(L).GT.MAXCIC14) THEN
                          CALL WRNDIE(-2,'<CROSSINIT>', &
                             'Too many inclusions')
                        ENDIF
                        TMPIN14(L,NCIN14(L),1) = IBB0
                        TMPIN14(L,NCIN14(L),2) = ICBON1(K)
                      ENDIF

                    ENDIF

                  ENDDO
                  ! ---------- End of K loop
                ENDIF
              ENDDO
              ! --------- End of J loop

              !---------------------------------------------------------
              ! 1-3 inclusions coming from RMD bonds
              !---------------------------------------------------------

              DO J=1,NCBON
                ! ... Identify all atoms of a neighboring bond opposite 
                ! ... to the bond breakage to include this atom in the inclusion list
                INFL=.false.
                IF ((IB1.EQ.ICBON1(J)).AND. &
                    (IB2.NE.ICBON2(J))) THEN
                  IBB0=IB2
                  IBB1=IB1
                  IBB2=ICBON2(J)
                  IF (QCBON(L,J).AND.QCBON(NCRSU+1,I) &
                      .AND.QCBON(NCRSU+1,J)) INFL=.true.

                ELSEIF ((IB2.EQ.ICBON1(J)).AND. &
                       (IB1.NE.ICBON2(J))) THEN
                  IBB0=IB1
                  IBB1=IB2
                  IBB2=ICBON2(J)
                  IF (QCBON(L,J).AND.QCBON(NCRSU+1,I) &
                      .AND.QCBON(NCRSU+1,J)) INFL=.true.

                ELSEIF ((IB1.EQ.ICBON2(J)).AND. &
                       (IB2.NE.ICBON1(J))) THEN
                  IBB0=IB2
                  IBB1=IB1
                  IBB2=ICBON1(J)
                  IF (QCBON(L,J).AND.QCBON(NCRSU+1,I) &
                      .AND.QCBON(NCRSU+1,J)) INFL=.true.

                ELSEIF ((IB2.EQ.ICBON2(J)).AND. &
                       (IB1.NE.ICBON1(J))) THEN
                  IBB0=IB1
                  IBB1=IB2
                  IBB2=ICBON1(J)
                  IF (QCBON(L,J).AND.QCBON(NCRSU+1,I) &
                      .AND.QCBON(NCRSU+1,J)) INFL=.true.

                ENDIF
      
                IF (INFL.AND.(IBB0.NE.IBB2)) THEN
                  NCINC(L)=NCINC(L)+1
                  IF (NCINC(L).GT.MAXCEXC) THEN
                    CALL WRNDIE(-2,'<CROSSINIT>', &
                       'Too many inclusions')
                  ENDIF

                  TMPINC(L,NCINC(L),1) = IBB0
                  TMPINC(L,NCINC(L),2) = IBB2
                ENDIF

                !---------------------------------------------------------------
                ! Identify 1-4 interactions coming from PSF bonds
                !---------------------------------------------------------------
                IF (INFL) THEN
                  DO K=1,NBOND

                    INF14=.false.
                    IF((IBB2.EQ.IB(K)).AND.(JB(K).NE.IBB1)) THEN 
                      IBB3=JB(K)
                      IF(.NOT.QCBON(L,I).AND.INFL) INF14=.true.
                    ENDIF
                    IF((IBB2.EQ.JB(K)).AND.(IB(K).NE.IBB1)) THEN 
                      IBB3=IB(K)
                      IF(.NOT.QCBON(L,I).AND.INFL) INF14=.true.
                    ENDIF

                    ! ... For a found PSF bond check if IBB2-IBB3 is not deleted 
                    ! ... by RMD parameter file for each surface
                    IF (INF14) THEN
                      DO M=1,NCBON
                        IF (((IBB2.EQ.ICBON1(M)).AND. &
                           (IBB3.EQ.ICBON2(M))).OR. &
                           ((IBB3.EQ.ICBON1(M)).AND. &
                            (IBB2.EQ.ICBON2(M)))) THEN
                          IF (.NOT.QCBON(L,M)) INF14=.false.
                        ENDIF
                      ENDDO
                    ENDIF

                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO
                  ! ---------- End of loop K -------------------

                  !---------------------------------------------------------------
                  ! Identify 1-4 interactions coming from RMD bonds
                  !---------------------------------------------------------------

                  DO K=1,NCBON

                    IF((IBB2.EQ.ICBON1(K)).AND. &
                      (ICBON2(K).NE.IBB1)) THEN 

                      IF (QCBON(L,K).AND.INFL.AND. &
                        (IBB0.NE.ICBON2(K))) THEN
                        NCIN14(L)=NCIN14(L)+1
                        IF (NCIN14(L).GT.MAXCIC14) THEN
                          CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                        ENDIF
                        TMPIN14(L,NCIN14(L),1) = IBB0
                        TMPIN14(L,NCIN14(L),2) = ICBON2(K)
                      ENDIF

                    ENDIF

                    IF((IBB2.EQ.ICBON2(K)).AND. &
                       (ICBON1(K).NE.IBB1)) THEN 

                      IF (QCBON(L,K).AND.INFL.AND. &
                        (IBB0.NE.ICBON1(K))) THEN
                        NCIN14(L)=NCIN14(L)+1
                        IF (NCIN14(L).GT.MAXCIC14) THEN
                          CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                        ENDIF
                        TMPIN14(L,NCIN14(L),1) = IBB0 
                        TMPIN14(L,NCIN14(L),2) = ICBON1(K)
                      ENDIF

                    ENDIF

                  ENDDO
                  ! ---------- End of K loop
                ENDIF
              ENDDO
            ! --------- End of J loop
            ENDIF

            ! ---------------------------------------------------------------
            ! ... Identify 1-4 inclusions having broken bond in the center
            ! ---------------------------------------------------------------

            ! ... Case 1: Outer bonds both in PSF


            IF (.NOT.QCBON(L,I)) THEN
              DO J=1,NBOND
                ! identify neighboring PSF bonds of IB1
                INFL=.false.
                IF ((IB1.EQ.IB(J)).AND.(JB(J).NE.IB2)) THEN
                  IBB0=JB(J)
                  IBB1=IB1
                  INFL=.true.
                ELSEIF((IB1.EQ.JB(J)).AND.(IB(J).NE.IB2))THEN
                  IBB0=IB(J)
                  IBB1=IB1
                  INFL=.true.
                ENDIF
                ! Find neighboring PSF bonds on the IB2 side
                IF (INFL) THEN
                  DO K=1,NBOND
                    INF14=.false.
                    ! identify bonds of IB2
                    IF((IB2.EQ.IB(K)).AND.(JB(K).NE.IB1))THEN
                      IBB2=IB2
                      IBB3=JB(K)
                      IF (.NOT.QCBON(L,I)) INF14=.true.
                    ELSEIF((IB2.EQ.JB(K)).AND. &
                          (IB(K).NE.IB1))THEN
                      IBB2=IB2
                      IBB3=JB(K)
                      IF (.NOT.QCBON(L,I)) INF14=.true.
                    ENDIF
                    ! Check if detected outer bonds were not removed by RMD
                    IF (INF14) THEN
                      DO M=1, NCBON
                        IF ((ICBON1(M).EQ.IBB0).AND. &
                           (ICBON2(M).EQ.IBB1).AND. &
                           (.NOT.QCBON(L,M))) THEN
                          INF14=.false.
                        ELSEIF ((ICBON2(M).EQ.IBB0).AND. &
                               (ICBON1(M).EQ.IBB1).AND. &
                               (.NOT.QCBON(L,M))) THEN
                          INF14=.false.
                        ELSEIF ((ICBON1(M).EQ.IBB2).AND. &
                               (ICBON2(M).EQ.IBB3).AND. &
                               (.NOT.QCBON(L,M))) THEN
                          INF14=.false.
                        ELSEIF ((ICBON2(M).EQ.IBB2).AND. &
                               (ICBON1(M).EQ.IBB3).AND. &
                               (.NOT.QCBON(L,M))) THEN
                          INF14=.false.
                        ENDIF
                      ENDDO
                    ENDIF

                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO

                  ! ... Case 2: IB2 bond from RMD
                  DO K=1,NCBON
                    INF14=.false.
                    IF((ICBON1(K).EQ.IB2).AND. &
                      (ICBON2(K).NE.IB1)) THEN
                      IBB3=ICBON2(K)
                      IF(QCBON(L,K).AND.(.NOT.QCBON(L,I))) INF14=.true.
                    ELSEIF((ICBON2(K).EQ.IB2).AND. &
                          (ICBON1(K).NE.IB1)) THEN
                      IBB3=ICBON1(K)
                      IF(QCBON(L,K).AND.(.NOT.QCBON(L,I))) INF14=.true.
                    ENDIF
                    ! Check if PSF bond (IBB0-IBB1) was not removed
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IBB0).AND. &
                         (ICBON2(M).EQ.IBB1).AND. &
                         (.NOT.QCBON(L,M))) THEN
                        IF (INF14) INF14=.false.
                      ELSEIF ((ICBON2(M).EQ.IBB0).AND. &
                             (ICBON1(M).EQ.IBB1).AND. &
                             (.NOT.QCBON(L,M))) THEN
                        IF (INF14) INF14=.false.
                      ENDIF
                    ENDDO

                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                           'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO
                  ! ----------- End of K loop
                ENDIF
              ENDDO
              ! ----------- End of J loop

              ! ... Looking from the other side of the broken bond
              DO J=1,NBOND
              ! identify neighboring PSF bonds of IB2
                INFL=.false.
                IF ((IB2.EQ.IB(J)).AND.(JB(J).NE.IB1)) THEN
                  IBB0=JB(J)
                  IBB1=IB2
                  INFL=.true.
                ELSEIF((IB2.EQ.JB(J)).AND.(IB(J).NE.IB1))THEN
                  IBB0=IB(J)
                  IBB1=IB2
                  INFL=.true.
                ENDIF

                ! Find neighboring PSF bonds on the IB1 side
                IF (INFL) THEN

                  ! ... Case 2: IB1 bond from RMD
                  DO K=1,NCBON
                    INF14=.false.
                    IF((ICBON1(K).EQ.IB1).AND. &
                      (ICBON2(K).NE.IB2)) THEN
                      IBB3=ICBON2(K)
                      IF(QCBON(L,K).AND.(.NOT.QCBON(L,I))) INF14=.true.
                    ELSEIF((ICBON2(K).EQ.IB1).AND. &
                          (ICBON1(K).NE.IB2)) THEN
                      IBB3=ICBON1(K)
                      IF(QCBON(L,K).AND.(.NOT.QCBON(L,I))) INF14=.true.
                    ENDIF
                    ! Check if PSF bond (IBB0-IBB1) was not removed
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IBB0).AND. &
                         (ICBON2(M).EQ.IBB1).AND. &
                         (.NOT.QCBON(L,M))) THEN
                        IF (INF14) INF14=.false.
                      ELSEIF ((ICBON2(M).EQ.IBB0).AND. &
                             (ICBON1(M).EQ.IBB1).AND. &
                             (.NOT.QCBON(L,M))) THEN
                        IF (INF14) INF14=.false.
                      ENDIF
                    ENDDO

                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO
                  ! ---------- End of K loop
                ENDIF
              ENDDO
              ! ---------- End of J loop

              ! ... Case 4: All three bonds are from RMD
              DO J=1,NCBON
                INFL=.false.
                IF((ICBON1(J).EQ.IB1).AND.(IB2.NE.ICBON2(J)))THEN
                  IBB0=ICBON2(J)
                  IBB1=IB1
                  INFL=.true.
                ELSEIF((ICBON2(J).EQ.IB1).AND. &
                      (IB2.NE.ICBON1(J)))THEN
                  IBB0=ICBON1(J)
                  IBB1=IB1
                  INFL=.true.
                ENDIF

                IF (INFL) THEN
                  DO K=1,NCBON
                    INF14=.false.
                    IF((ICBON1(K).EQ.IB2).AND.(IB1.NE.ICBON2(K)))THEN
                      IBB3=ICBON2(K)
                      IBB1=IB2
                      IF (QCBON(L,J).AND.QCBON(L,K).AND. &
                         (.NOT.QCBON(L,I))) INF14=.true.
                    ELSEIF((ICBON2(K).EQ.IB2).AND. &
                          (IB1.NE.ICBON1(K)))THEN
                      IBB3=ICBON1(K)
                      IBB1=IB2
                      IF (QCBON(L,J).AND.QCBON(L,K).AND. &
                         (.NOT.QCBON(L,I))) INF14=.true.
                    ENDIF

                    IF (INF14.AND.(IBB0.NE.IBB3)) THEN
                      NCIN14(L)=NCIN14(L)+1
                      IF (NCIN14(L).GT.MAXCIC14) THEN
                        CALL WRNDIE(-2,'<CROSSINIT>', &
                            'Too many inclusions')
                      ENDIF
                      TMPIN14(L,NCIN14(L),1) = IBB0
                      TMPIN14(L,NCIN14(L),2) = IBB3
                    ENDIF

                  ENDDO
                  ! ----------- End of K loop 
                ENDIF
              ENDDO
              ! ----------- End of J loop
            ENDIF
          ENDDO
          ! --------- End of L loop 
        ENDIF
      ENDDO
      ! -------- End of I loop


  !===========================================================
  ! Check for double counted inclusion.
  !===========================================================


      DO L=1, NCRSU
        ! Check 1-2 & 1-3 inclusions
        IF (NCINC(L).GT.1) THEN

          NEWINC=NCINC(L)
          CNT2=0
          DO I=1,(NCINC(L)-1)
            IF (I.LE.(NEWINC-1)) THEN
              IB1=TMPINC(L,I,1)
              IB2=TMPINC(L,I,2)

              CNT1=0
              DO J=I+1,NCINC(L)
                IF (J.LE.NCINC(L)-CNT2) THEN 
                  JB1=TMPINC(L,J-CNT1,1)
                  JB2=TMPINC(L,J-CNT1,2)

                  IF ((IB1.EQ.JB1.AND.IB2.EQ.JB2).OR. &
                      (IB1.EQ.JB2.AND.IB2.EQ.JB1)) THEN

                    DO K=J-CNT1,NEWINC
                        TMPINC(L,K,1)=TMPINC(L,K+1,1)
                        TMPINC(L,K,2)=TMPINC(L,K+1,2)  
                    ENDDO

                    CNT1=CNT1+1
                    NEWINC=NEWINC-1

                  ENDIF
                ENDIF
              ENDDO
              CNT2=CNT2+CNT1
            ENDIF
          ENDDO
          NCINC(L)=NEWINC

        ENDIF

        ! Check 1-4 inclusions
        IF (NCIN14(L).GT.1) THEN

          NEWINC=NCIN14(L)
          CNT2=0
          DO I=1,(NCIN14(L)-1)
            IF (I.LE.(NEWINC-1)) THEN
              IB1=TMPIN14(L,I,1)
              IB2=TMPIN14(L,I,2)

              CNT1=0
              DO J=I+1,NCIN14(L)
                IF (J.LE.NCIN14(L)-CNT2) THEN
                  JB1=TMPIN14(L,J-CNT1,1)
                  JB2=TMPIN14(L,J-CNT1,2)

                  IF ((IB1.EQ.JB1.AND.IB2.EQ.JB2).OR. &
                     (IB1.EQ.JB2.AND.IB2.EQ.JB1)) THEN

                    DO K=J-CNT1,NEWINC
                        TMPIN14(L,K,1)=TMPIN14(L,K+1,1)
                        TMPIN14(L,K,2)=TMPIN14(L,K+1,2) 
                    ENDDO

                    CNT1=CNT1+1
                    NEWINC=NEWINC-1

                  ENDIF
                ENDIF
              ENDDO
              CNT2=CNT2+CNT1
            ENDIF
          ENDDO

          NCIN14(L)=NEWINC

        ENDIF

        ! Check if 1-4 inclusions already show up as 1-2 or 1-3
        ! and remove them from 1-4
        IF ((NCINC(L).GT.0).AND.(NCIN14(L).GT.0)) THEN

          NEWINC=NCIN14(L)
          DO I=1,NCINC(L)

            IB1=TMPINC(L,I,1)
            IB2=TMPINC(L,I,2)
            CNT1=0
            DO J=1,NCIN14(L)
              IF ((IB1.EQ.TMPIN14(L,J,1).AND.IB2.EQ. &
                TMPIN14(L,J,2)).OR.IB2.EQ.TMPIN14(L,J,1) &
                .AND.IB1.EQ.TMPIN14(L,J,2)) THEN

                DO M=1,NEWINC
                  IF (M.GT.J-CNT1) THEN

                    TMPIN14(L,M-1,1)=TMPIN14(L,M,1)
                    TMPIN14(L,M-1,2)=TMPIN14(L,M,2)

                  ENDIF
                ENDDO
                CNT1=CNT1+1
                NEWINC=NEWINC-1
              ENDIF
            ENDDO
          ENDDO
          NCIN14(L)=NEWINC
        ENDIF

      ENDDO
      ! ------------ End of surface loop L ------------------------

  !----------------------------------------------------------------
  ! SL: 10/03/25 Remove 1-4 inclusions which are bonded to each
  !              other in the PSF. This can happen for cyclic 
  !              transition state structures!!!!

      DO L=1, NCRSU

        IF (NCIN14(L).GT.0) THEN

          CNT1=0
          NEWINC=NCIN14(L)
          DO I=1,NEWINC
            IB1=TMPIN14(L,I-CNT1,1)
            IB2=TMPIN14(L,I-CNT1,2)

            DO J=1,NBOND
              IF ((IB1.EQ.IB(J).AND.IB2.EQ.JB(J)).OR. &
                   IB2.EQ.IB(J).AND.IB1.EQ.JB(J)) THEN

                DO M=1,NEWINC
                  IF (M.GT.I-CNT1) THEN

                    TMPIN14(L,M-1,1)=TMPIN14(L,M,1)
                    TMPIN14(L,M-1,2)=TMPIN14(L,M,2)

                  ENDIF
                ENDDO
                CNT1=CNT1+1
                NEWINC=NEWINC-1
              ENDIF
            ENDDO
          ENDDO
          NCIN14(L)=NEWINC

        ENDIF



  !===============================================================
  ! Initial determination of non-bond exclusions and inclusions
  ! is done now. The next part takes care of cases when multiple
  ! bonds get formed and broken on a specific surface at the same 
  ! time. 
  !===============================================================


  ! --------------------------------------------------------------
  ! SL: 10/03/24
  ! Check if any nonbond exclusion is also present as an identical 
  ! inclusion. If so, multiple exclusion/inclusion overlaps are 
  ! possible depending on the setup of the RMD parameter file. 
  ! This needs a more careful analysis of the bond network.

  ! Allocate memory for TMPOVLA which stores information on needed deletions
  ! in TMPEXC and TMPINC lists
        IF (NCEXC(L).GE.NCINC(L)) THEN 
          allocate(TMPOVLA(NCEXC(L),2))
          DO I=1,NCEXC(L)
            TMPOVLA(I,1)=0
            TMPOVLA(I,2)=0
          ENDDO
        ELSEIF (NCEXC(L).LT.NCINC(L)) THEN
          allocate(TMPOVLA(NCINC(L),2))
          DO I=1,NCINC(L)
            TMPOVLA(I,1)=0
            TMPOVLA(I,2)=0
          ENDDO
        ENDIF

        ! Set CNT1 to one
        CNT1=1

        ! First compare NCINC/NCEXC lists
        DO I=1,NCINC(L)

          IB1=TMPINC(L,I,1)
          IB2=TMPINC(L,I,2)
          DO J=1,NCEXC(L)

            IF (((IB1.EQ.TMPEXC(L,J,1)).AND. &
               IB2.EQ.TMPEXC(L,J,2)).OR. &
               ((IB2.EQ.TMPEXC(L,J,1)).AND. &
               IB1.EQ.TMPEXC(L,J,2))) THEN
 
              ! Atom pair found in both lists:
              ! Define flag to check if the 1-3 interaction is 
              ! defined over two bonds on the current surface
              INFL=.false.

              ! Search RMD bonds including atom IB1 or IB2
              DO K=1,NCBON

                ! ... ICBON1
                IF (ICBON1(K).EQ.IB1) THEN
                  DO M=1,NCBON
                    ! Search for a bond connection definition of 
                    ! IB1 and IB2 in NCBON
                    IF ((K.NE.M).AND.((IB2.EQ.ICBON1(M) &
                   .AND.ICBON2(M).EQ.ICBON2(K)).OR. &
                    (IB2.EQ.ICBON2(M).AND. &
                    ICBON1(M).EQ.ICBON2(K)))) THEN
                      ! K and M are two neigboring bonds
                      ! Now see if both are present on current surface L 
                      ! and set INFL .true. as soon as at least one 
                      ! bonded network was found
                      IF (QCBON(L,K).AND.QCBON(L,M)) INFL=.true.

                    ENDIF
                  ENDDO
                ENDIF

                ! ... ICBON2
                IF (ICBON2(K).EQ.IB1) THEN
                  DO M=1,NCBON
                    ! Search for a bond connection definition 
                    ! of IB1 and IB2 in NCBON
                    IF ((K.NE.M).AND.((IB2.EQ.ICBON1(M) &
                       .AND.ICBON2(M).EQ.ICBON1(K)).OR. &
                       (IB2.EQ.ICBON2(M).AND. &
                       ICBON1(M).EQ.ICBON1(K)))) THEN
                      ! K and M are two neighboring bonds
                      ! Now see if both are present on current surface L 
                      ! and set INFL .true. as soon as at least one 
                      ! bonded network was found
                      IF (QCBON(L,K).AND.QCBON(L,M)) INFL=.true.

                    ENDIF
                  ENDDO
                ENDIF

              ENDDO

              ! Check if there are any bonded connections of IB1-X-IB2 in the PSF
              EXFL=.false.
              DO K=1,NBOND
                IF (IB1.EQ.IB(K)) THEN
                  DO M=1,NBOND
                    IF ((IB2.EQ.IB(M).AND.JB(K).EQ.JB(M)).OR. &
                     (IB2.EQ.JB(M).AND.JB(K).EQ.IB(M))) THEN
                      ! 1-3 bond network detected in PSF
                      EXFL=.true.
                    ENDIF 
                  ENDDO
                ENDIF

                IF (IB1.EQ.JB(K)) THEN
                  DO M=1,NBOND
                    IF ((IB2.EQ.IB(M).AND.IB(K).EQ.JB(M)).OR. &
                        (IB2.EQ.JB(M).AND.IB(K).EQ.IB(M))) THEN
                      ! 1-3 bond network detected in PSF
                      EXFL=.true.
                    ENDIF 
                  ENDDO
                ENDIF

              ENDDO

              ! Store information on possible deletions 
              ! in TMPEXC and TMPINC
              IF ((INFL.AND.EXFL).OR. &
                 (.NOT.INFL.AND..NOT.EXFL)) THEN 

                TMPOVLA(CNT1,1)=J
                TMPOVLA(CNT1,2)=I
              ELSEIF (INFL.AND..NOT.EXFL) THEN

                TMPOVLA(CNT1,2)=I
              ELSEIF (.NOT.INFL.AND.EXFL) THEN 

                TMPOVLA(CNT1,1)=J
              ENDIF

              CNT1=CNT1+1

            ENDIF
          ENDDO
        ENDDO

        ! Remove all overlapping 1-3 inclusions and 
        ! exclusions from the corresponding lists TMPINC and TMPEXC.
        IF (NCINC(L).GT.0) THEN
          NEWINC=NCINC(L)
          CNT1=0
          DO I=1,NEWINC
            DO J=1,NCINC(L)
              IF (I.EQ.TMPOVLA(J,2)) THEN

                DO K=I-CNT1,NEWINC

                  TMPINC(L,K,1)=TMPINC(L,K+1,1)
                  TMPINC(L,K,2)=TMPINC(L,K+1,2)

                ENDDO
                NEWINC=NEWINC-1
                CNT1=CNT1+1
              ENDIF
            ENDDO
            NCINC(L)=NEWINC
          ENDDO
        ENDIF

        IF (NCEXC(L).GT.0) THEN
          NEWEXC=NCEXC(L)

          CNT1=0
          DO I=1,NEWEXC
            DO J=1,NCEXC(L)
              IF (I.EQ.TMPOVLA(J,1)) THEN

                DO K=I-CNT1,NEWEXC

                  TMPEXC(L,K,1)=TMPEXC(L,K+1,1)
                  TMPEXC(L,K,2)=TMPEXC(L,K+1,2)

                ENDDO
                NEWEXC=NEWEXC-1
                CNT1=CNT1+1
              ENDIF
            ENDDO

            NCEXC(L)=NEWEXC
          ENDDO
        ENDIF

        IF (allocated(TMPOVLA)) deallocate(TMPOVLA)


  !-----------------------------------------------------------------
  ! SL: 10/03/25 Careful verification of non-bond 1-4 exclusion and
  !              inclusion lists
  !
  ! -------------- First do exclusions -----------------------------


        ! Loop over 1-4 exclusion list
        CNT1=0
        NEWEXC=NC14(L)
        DO I=1,NEWEXC
          EXFL=.false.
          INFL=.false.
          INF14=.false.
          IB1=TMPC14(L,I-CNT1,1)
          IB2=TMPC14(L,I-CNT1,2)

          ! Check if 1-4 connectivity is already defined in PSF
          DO J=1,NBOND
            IF ((IB1.EQ.IB(J)).AND.IB2.NE.JB(J)) IBB1=JB(J)
            IF ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)) IBB1=IB(J)
            IF ((IB2.EQ.IB(J)).AND.IB1.NE.JB(J)) IBB2=JB(J)
            IF ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)) IBB2=IB(J)

            IF ((IB1.EQ.IB(J).AND.IB2.NE.JB(J)).OR. &
                ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)).AND. &
                .NOT.EXFL) THEN
              ! IB1
              DO K=1,NBOND
                IF (IB2.EQ.IB(K).AND.JB(K).NE.IB1) IBB2=JB(K)
                IF (IB2.EQ.JB(K).AND.IB(K).NE.IB1) IBB2=IB(K)

                IF ((IB2.EQ.IB(K).AND.JB(K).NE.IB1).OR. &
                    (IB2.EQ.JB(K).AND.IB(K).NE.IB1)) THEN
                  DO M=1,NBOND
                    ! If 1-4 connectivity detected in PSF, set flag
                    IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                       .OR.(IB(M).EQ.IBB2.AND. &
                       JB(M).EQ.IBB1)) EXFL=.true.

                  ENDDO
                ENDIF
              ENDDO
            ENDIF

            IF ((IB2.EQ.IB(J).AND.IB1.NE.JB(J)).OR. &
                ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)).AND. &
                .NOT.EXFL) THEN
              ! IB2
              DO K=1,NBOND
                IF (IB1.EQ.IB(K).AND.JB(K).NE.IB2) IBB1=JB(K)
                IF (IB1.EQ.JB(K).AND.IB(K).NE.IB2) IBB1=IB(K)

                IF ((IB1.EQ.IB(K).AND.JB(K).NE.IB2).OR. &
                    (IB1.EQ.JB(K).AND.IB(K).NE.IB2)) THEN
                  DO M=1,NBOND
                    ! 1-4 connectivity detected in PSF, set flag
                    IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                      .OR.(IB(M).EQ.IBB2.AND. &
                       JB(M).EQ.IBB1)) EXFL=.true.

                  ENDDO
                ENDIF
              ENDDO
            ENDIF

          ENDDO

          ! -----------------------------------------------------------
          ! Check if the same 1-4 connectivity is present on surface L
          ! -----------------------------------------------------------

          ! First check the cases where at least one of the outer bonds is
          ! defined for surface L

          ! EXF14 defines if one of the outer bonds  was found for surface L
          EXF14=.false.
          DO J=1,NCBON
            IF (IB1.EQ.ICBON1(J).AND.IB2.NE.ICBON2(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB1=ICBON2(J)
            IF (IB1.EQ.ICBON2(J).AND.IB2.NE.ICBON1(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB1=ICBON1(J)

            IF (IB2.EQ.ICBON1(J).AND.IB1.NE.ICBON2(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB2=ICBON2(J)
            IF (IB2.EQ.ICBON2(J).AND.IB1.NE.ICBON1(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB2=ICBON1(J)

            ! If bond IB1-IBB1 on surface L detected
            IF (((IB1.EQ.ICBON1(J).AND.IB2.NE.ICBON2(J) &
                .AND.QCBON(L,J)).OR.(IB1.EQ.ICBON2(J).AND. &
                 IB2.NE.ICBON1(J).AND.QCBON(L,J))).AND. &
                 .NOT.INFL) THEN

              ! Outer bond definition was found on surface L
              EXF14=.true.

              ! INF14 defines if the other outer bond was found for surface L
              INF14=.false.
              DO K=1,NCBON
                IF (IB2.EQ.ICBON1(K).AND.IB1.NE.ICBON2(K) &
                   .AND.QCBON(L,K)) IBB2=ICBON2(K)
                IF (IB2.EQ.ICBON2(K).AND.IB1.NE.ICBON1(K) &
                   .AND.QCBON(L,K)) IBB2=ICBON1(K)

                ! If bond IB2-IBB2 on surface L detected
                IF (((IB2.EQ.ICBON1(K).AND.IB1.NE.ICBON2(K)) &
                   .OR.(IB2.EQ.ICBON2(K).AND.IB1.NE.ICBON1(K))) &
                   .AND.QCBON(L,K)) THEN

                  ! Second outer bond definition was found on surface L
                  INF14=.true.
                  DO M=1,NCBON
                    ! If bond IBB1-IBB2 on surface L detected
                    IF (((ICBON1(M).EQ.IBB1.AND.ICBON2(M).EQ.IBB2) &
                       .OR.(ICBON1(M).EQ.IBB2.AND. &
                       ICBON2(M).EQ.IBB1)).AND.QCBON(L,M)) THEN 
                      ! Activate 1-4 interaction flag 
                      INFL=.true.
   !     WRITE(OUTU,*)'Check 1: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                    ! Reset INF14 to .false. in case IBB1-IBB2 is 
                    ! deactivated on surface L
                    ELSEIF (((ICBON1(M).EQ.IBB1.AND. &
                           ICBON2(M).EQ.IBB2) &
                           .OR.(ICBON1(M).EQ.IBB2.AND. &
                           ICBON2(M).EQ.IBB1)).AND. &
                           .NOT.QCBON(L,M)) THEN
                      INF14=.false.
                    ENDIF
                  ENDDO 
   !     WRITE(OUTU,*)'Check 2: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                  ! No 1-4 interaction detected yet: Check if IBB1-IBB2 
                  ! bond comes from PSF and is not explicitly defined on
                  ! surface L
                  IF (.NOT.INFL) THEN
                    DO M=1,NBOND
                      IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                         .OR.(IB(M).EQ.IBB2.AND. &
                         JB(M).EQ.IBB1)) THEN

                        INFL=.true.
    !    WRITE(OUTU,*)'Check 3: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                        ! Set INFL back to .false. in case IBB1-IBB2 is
                        ! defined in RMD parameter file but deactivated
                        ! on surface L 
                        DO N=1,NCBON 
                          IF (((ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2).OR. &
                               (ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.

   !   WRITE(OUTU,*)'Check 4: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                        ENDDO
                      ENDIF
                    ENDDO
                  ENDIF
                ENDIF
              ENDDO
   !     WRITE(OUTU,*)'Check 5: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
              ! If bond IB2-IBB2 not detected on surface L
              IF (.NOT.INF14.AND..NOT.INFL) THEN

                ! Check if IB2-IBB2 is defined as a PSF bond
                !
                ! CNT2 is used as a flag to get out of the K loop in 
                ! case the 1-4 interaction was confirmed by one of the
                ! bonds in NBOND
                CNT2=0 
                DO K=1,NBOND
                  IF (IB(K).EQ.IB2.AND. &
                     JB(K).NE.IB1.AND.CNT2.EQ.0) IBB2=JB(K)
                  IF (JB(K).EQ.IB2.AND. &
                     IB(K).NE.IB1.AND.CNT2.EQ.0) IBB2=IB(K)

                  IF (((IB(K).EQ.IB2.AND.JB(K).NE.IB1).OR. &
                      (JB(K).EQ.IB2.AND.IB(K).NE.IB1)) &
                     .AND.CNT2.EQ.0) THEN
                    INFL=.true.
   !     WRITE(OUTU,*)'Check 6: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                    ! Check if same bond is deactivated on
                    ! surface L
                    DO M=1,NCBON
                      ! Set INFL back to .false. in case IB2-IBB2 is
                      ! defined in RMD parameter file but deactivated
                      ! on surface L 
                      IF (((ICBON1(M).EQ.IB2.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IB2)).AND. &
                          (.NOT.QCBON(L,M))) INFL=.false.

   !   WRITE(OUTU,*)'Check 7: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                    ENDDO

   !     WRITE(OUTU,*)'Check 8: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                    ! At this point IB2-IBB2 was found to be present and 
                    ! uniquely defined on PSF, check for IBB1-IBB2 bond
                    IF (INFL) THEN

                      ! Check if IBB1-IBB2 is defined on surface L but
                      ! deactivated. Set INFL back to .false. in that case
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                            ICBON2(M).EQ.IBB2).OR. &
                            (ICBON2(M).EQ.IBB1.AND. &
                            ICBON1(M).EQ.IBB2)).AND. &
                            (.NOT.QCBON(L,M))) INFL=.false.

   !     WRITE(OUTU,*)'Check 9: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                      ENDDO
                    ENDIF

                    ! If not defined for surface L check if IBB1-IBB2 is 
                    ! defined on PSF
                    IF (INFL) THEN
           !         CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                           .OR.(IB(M).EQ.IBB2.AND. &
                           JB(M).EQ.IBB1)) THEN
                          ! IBB1-IBB2 should not be defined in RMD in this case
                          DO N=1,NCBON
                            IF (((ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2).OR. &
                               (ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.
                          ENDDO
           !             IF (CNT3.EQ.1) INFL=.false.

  !   IF (CNT2.EQ.1)  WRITE(OUTU,*)'Check 11: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2, INFL

                        ENDIF
                      ENDDO
                    ENDIF
  !  WRITE(OUTU,*)'Check 12: IB1, IB2,IBB1, IBB2, INFL',IB1, IB2,IBB1,IBB2, INFL
                    ! If IBB1-IBB2 not in PSF and not in RMD set INFL to .false.
                    IF (INFL) THEN
                      ! CNT3 is used as a flag to mark a detected bond in NBOND or NCBON
                      CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                           .OR.(IB(M).EQ.IBB2.AND. &
                           JB(M).EQ.IBB1)) CNT3=1

                      ENDDO
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                             ICBON2(M).EQ.IBB2).OR. &
                             (ICBON1(M).EQ.IBB2.AND. &
                             ICBON2(M).EQ.IBB1)).AND. &
                             QCBON(L,M)) CNT3=1
                      ENDDO
                      ! Reset INFL in case no bond was detected
                      IF (CNT3.EQ.0) INFL=.false.
   ! IF (CNT2.EQ.0)  WRITE(OUTU,*)'Check 13: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL,CNT2
                    ENDIF
                    IF (INFL.AND.CNT2.EQ.0) CNT2=1
                  ENDIF
                ENDDO
              ENDIF

            ! If bond IB2-IBB2 on surface L detected
            ELSEIF (((IB2.EQ.ICBON1(J).AND.IB1.NE.ICBON2(J) &
                .AND.QCBON(L,J)).OR.(IB2.EQ.ICBON2(J).AND. &
                 IB1.NE.ICBON1(J).AND.QCBON(L,J))).AND..NOT.INFL) &
                 THEN

              ! Outer bond definition was found on surface L
              EXF14=.true.

              ! INF14 defines if the other outer bond was found for surface L
              INF14=.false.
              DO K=1,NCBON
                IF (IB1.EQ.ICBON1(K).AND. &
                    IB2.NE.ICBON2(K).AND. &
                    QCBON(L,K)) IBB1=ICBON2(K)
                IF (IB1.EQ.ICBON2(K).AND. &
                   IB2.NE.ICBON1(K).AND. &
                    QCBON(L,K)) IBB1=ICBON1(K)

                ! If bond IB1-IBB1 on surface L detected
                IF (((IB1.EQ.ICBON1(K).AND. &
                    IB2.NE.ICBON2(K)).OR.(IB1.EQ.ICBON2(K).AND. &
                   IB2.NE.ICBON1(K))).AND.QCBON(L,K)) THEN

                  ! Second outer bond definition was found on surface L
                  INF14=.true.
                  DO M=1,NCBON
                    ! If bond IBB1-IBB2 on surface L detected
                    IF (((ICBON1(M).EQ.IBB2.AND.ICBON2(M).EQ.IBB1) &
                       .OR.(ICBON1(M).EQ.IBB1.AND. &
                       ICBON2(M).EQ.IBB2)).AND.QCBON(L,M)) THEN 
                      ! Activate 1-4 interaction flag 
                      INFL=.true.

                    ! Reset INF14 to .false. in case IBB1-IBB2 is 
                    ! deactivated on surface L
                    ELSEIF (((ICBON1(M).EQ.IBB2.AND. &
                             ICBON2(M).EQ.IBB1) &
                             .OR.(ICBON1(M).EQ.IBB1.AND. &
                            ICBON2(M).EQ.IBB2)).AND. &
                            .NOT.QCBON(L,M)) THEN
                      INF14=.false.
                    ENDIF
                  ENDDO 

                  ! No 1-4 interaction detected yet: Check if IBB1-IBB2 
                  ! bond comes from PSF and is not explicitly defined on
                  ! surface L
                  IF (.NOT.INFL) THEN
                    DO M=1,NBOND
                      IF((IB(M).EQ.IBB2.AND.JB(M).EQ.IBB1) &
                         .OR.(IB(M).EQ.IBB1.AND. &
                         JB(M).EQ.IBB2)) THEN

                        INFL=.true.

                        ! Set INFL back to .false. in case IBB1-IBB2 is
                        ! defined in RMD parameter file but deactivated
                        ! on surface L 
                        DO N=1,NCBON 
                          IF (((ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1).OR. &
                               (ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.
                        ENDDO
                      ENDIF
                    ENDDO
                  ENDIF
                ENDIF
              ENDDO

              ! If bond IB1-IBB1 not detected on surface L
              IF (.NOT.INF14.AND..NOT.INFL) THEN
                ! Check if IB1-IBB1 is defined as a PSF bond
                !
                ! CNT2 is used as a flag to get out of the K loop in 
                ! case the 1-4 interaction was confirmed by one of the
                ! bonds in NBOND
                CNT2=0
                DO K=1,NBOND
                  IF (IB(K).EQ.IB1.AND. &
                     JB(K).NE.IB2.AND.CNT2.EQ.0) IBB1=JB(K)
                  IF (JB(K).EQ.IB1.AND. &
                     IB(K).NE.IB2.AND.CNT2.EQ.0) IBB1=IB(K)
 
                  IF (((IB(K).EQ.IB1.AND.JB(K).NE.IB2).OR. &
                      (JB(K).EQ.IB1.AND.IB(K).NE.IB2)) &
                     .AND.CNT2.EQ.0) THEN
                    INFL=.true.

                    ! Check if same bond is deactivated on
                    ! surface L
                    DO M=1,NCBON
                      ! Set INFL back to .false. in case IB1-IBB1 is
                      ! defined in RMD parameter file but deactivated
                      ! on surface L
                      IF (((ICBON1(M).EQ.IB1.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IB1)).AND. &
                          (.NOT.QCBON(L,M))) INFL=.false.
                    ENDDO
        !            IF (CNT2.EQ.0.AND.INFL) CNT2=1
        !          ENDIF
        !        ENDDO

                    ! At this point IB1-IBB1 was found to be present and 
                    ! uniquely defined on PSF, check for IBB1-IBB2 bond
                    IF (INFL) THEN
                      ! Check if IBB1-IBB2 is defined on surface L but
                      ! deactivated. Set INFL back to .false. in that case
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB2.AND. &
                           ICBON2(M).EQ.IBB1).OR. &
                           (ICBON2(M).EQ.IBB2.AND. &
                            ICBON1(M).EQ.IBB1)).AND. &
                           (.NOT.QCBON(L,M))) INFL=.false.
                      ENDDO
                    ENDIF

                    ! If not defined for surface L check if IBB1-IBB2 is 
                    ! defined on PSF
                    IF (INFL) THEN
        !              CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB2.AND.JB(M).EQ.IBB1) &
                           .OR.(IB(M).EQ.IBB1.AND. &
                           JB(M).EQ.IBB2)) THEN
                          ! IBB1-IBB2 should not be defined in RMD in this case
                          DO N=1,NCBON
                            IF (((ICBON1(N).EQ.IBB1.AND. &
                             ICBON2(N).EQ.IBB2).OR. &
                             (ICBON1(N).EQ.IBB2.AND. &
                             ICBON2(N).EQ.IBB1)).AND. &
                             .NOT.QCBON(L,N)) INFL=.false.
                          ENDDO
        !                  IF (CNT3.EQ.1) INFL=.false.
                        ENDIF
                      ENDDO
                    ENDIF

                    ! If IBB1-IBB2 not in PSF and not in RMD set INFL to .false.
                    IF (INFL) THEN
                      ! CNT3 is used as a flag to mark a detected bond in NBOND or NCBON
                      CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                             .OR.(IB(M).EQ.IBB2.AND. &
                             JB(M).EQ.IBB1)) CNT3=1
                      ENDDO
                      DO N=1,NCBON
                        IF (((ICBON1(N).EQ.IBB1.AND. &
                             ICBON2(N).EQ.IBB2).OR. &
                             (ICBON1(N).EQ.IBB2.AND. &
                             ICBON2(N).EQ.IBB1)) &
                             .AND.QCBON(L,N)) CNT3=1
                      ENDDO
                      ! Reset INFL in case no bond was detected
                      IF (CNT3.EQ.0) INFL=.false.
                    ENDIF
                    IF (INFL.AND.CNT2.EQ.0) CNT2=1
                  ENDIF
                ENDDO
              ENDIF
            ENDIF

          ENDDO

          ! If up to now no 1-4 interaction was detected do last possible check:
          ! IB1-IBB1 & IB2-IBB2 uniquely defined by PSF and IBB1-IBB2 
          ! defined by surface L
          IF (.NOT.INFL.AND..NOT.EXF14) THEN             ! New 2
   !  WRITE(OUTU,*)'Check 14: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL
          DO J=1,NBOND
            IF ((IB1.EQ.IB(J)).AND.IB2.NE.JB(J)) IBB1=JB(J)
            IF ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)) IBB1=IB(J)
            IF ((IB2.EQ.IB(J)).AND.IB1.NE.JB(J)) IBB2=JB(J)
            IF ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)) IBB2=IB(J)

            ! Check IB1-IBB1 first
            IF ((IB1.EQ.IB(J).AND.IB2.NE.JB(J)).OR. &
                ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J))) THEN

              INF14=.false. ! used as flag to check if bond is defined in RMD
              DO K=1,NCBON
                IF ((ICBON1(K).EQ.IB1.AND. &
                    ICBON2(K).EQ.IBB1).OR. &
                    (ICBON2(K).EQ.IB1.AND. &
                    ICBON1(K).EQ.IBB1)) INF14=.true. 
              ENDDO

              ! If INF14 is still .false. IB1-IBB1 is uniquely described by PSF
              ! -> continue
              IF (.NOT. INF14) THEN
                DO K=1,NBOND
                  IF ((IB2.EQ.IB(K)).AND.IB1.NE.JB(K)) IBB2=JB(K)
                  IF ((IB2.EQ.JB(K)).AND.IB1.NE.IB(K)) IBB2=IB(K)

                  IF ((IB2.EQ.IB(K).AND.IB1.NE.JB(K)).OR. &
                     ((IB2.EQ.JB(K)).AND.IB1.NE.IB(K))) THEN

                    ! Insure that IB2-IBB2 is not defined in RMD
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IB2.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON2(M).EQ.IB2.AND. &
                          ICBON1(M).EQ.IBB2)) INF14=.true.
                    ENDDO

                    ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
                    ! -> continue
                    IF (.NOT. INF14) THEN

                      ! Check if IBB1-IBB2 is active on surface L
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IBB1)).AND. &
                          QCBON(L,M)) INFL=.true.

    !  WRITE(OUTU,*)'Check 15: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                      ENDDO
                    ENDIF
                  ENDIF
                ENDDO
              ENDIF

            ! Alternatively check IB2-IBB2
            ELSEIF ((IB2.EQ.IB(J).AND.IB1.NE.JB(J)).OR. &
                ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J))) THEN

              INF14=.false. ! used as flag to check if bond is defined in RMD             
              DO K=1,NCBON
                IF ((ICBON1(K).EQ.IB2.AND. &
                    ICBON2(K).EQ.IBB2).OR. &
                    (ICBON2(K).EQ.IB2.AND. &
                    ICBON1(K).EQ.IBB2)) INF14=.true. 
              ENDDO

              ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
              ! -> continue
              IF (.NOT. INF14) THEN
                DO K=1,NBOND
                  IF ((IB1.EQ.IB(K)).AND.IB2.NE.JB(K)) IBB1=JB(K)
                  IF ((IB1.EQ.JB(K)).AND.IB2.NE.IB(K)) IBB1=IB(K)

                  IF ((IB1.EQ.IB(K).AND.IB2.NE.JB(K)).OR. &
                     ((IB1.EQ.JB(K)).AND.IB2.NE.IB(K))) THEN

                    ! Insure that IB2-IBB2 is not defined in RMD
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IB1.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON2(M).EQ.IB1.AND. &
                          ICBON1(M).EQ.IBB1)) INF14=.true.
                    ENDDO

                    ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
                    ! -> continue
                    IF (.NOT. INF14) THEN

                      ! Check if IBB1-IBB2 is active on surface L
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IBB2)).AND. &
                          QCBON(L,M)) INFL=.true.

  !  WRITE(OUTU,*)'Check 16: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                      ENDDO
                    ENDIF
                  ENDIF
                ENDDO
              ENDIF

            ENDIF
          ! --------- End of surface L 1-4 connectivity check
          ENDDO
          ENDIF


          ! Correct exclusion list according to 
          ! EXFL and INFL status:
          ! The 1-4 exclusion needs to be removed if 
          ! the connection is present on PSF (EXFL), 
          ! or missing in both lists (.NOT.EXFL .AND. .NOT.INFL).

   !  WRITE(OUTU,*)'L,IB1,IB2,IBB1,IBB2,EXFL,INFL',L,IB1,IB2,IBB1,IBB2,EXFL,INFL

          IF (EXFL.OR.(.NOT.EXFL.AND..NOT.INFL)) THEN
            DO J=1,NEWEXC
              IF (J.GT.I-CNT1) THEN

                TMPC14(L,J-1,1)=TMPC14(L,J,1)
                TMPC14(L,J-1,2)=TMPC14(L,J,2)

              ENDIF
            ENDDO
            CNT1=CNT1+1
            NEWEXC=NEWEXC-1
          ENDIF

        ENDDO
        NC14(L)=NEWEXC


  ! ------------- Do the same for inclusions ----------------------

        ! Loop over 1-4 inclusion list
        CNT1=0
        NEWINC=NCIN14(L)
        DO I=1,NEWINC
          EXFL=.false.
          INFL=.false.
          INF14=.false.

          IB1=TMPIN14(L,I-CNT1,1)
          IB2=TMPIN14(L,I-CNT1,2)

          ! Check if 1-4 connectivity is already defined in PSF
          DO J=1,NBOND
            IF ((IB1.EQ.IB(J)).AND.IB2.NE.JB(J)) IBB1=JB(J)
            IF ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)) IBB1=IB(J)
            IF ((IB2.EQ.IB(J)).AND.IB1.NE.JB(J)) IBB2=JB(J)
            IF ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)) IBB2=IB(J)

            IF ((IB1.EQ.IB(J).AND.IB2.NE.JB(J)).OR. &
                ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)).AND. &
                .NOT.EXFL) THEN
              ! IB1
              DO K=1,NBOND
                IF (IB2.EQ.IB(K).AND.JB(K).NE.IB1) IBB2=JB(K)
                IF (IB2.EQ.JB(K).AND.IB(K).NE.IB1) IBB2=IB(K)

                IF ((IB2.EQ.IB(K).AND.JB(K).NE.IB1).OR. &
                    (IB2.EQ.JB(K).AND.IB(K).NE.IB1)) THEN
                  DO M=1,NBOND
                    ! If 1-4 connectivity detected in PSF, set flag
                    IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                       .OR.(IB(M).EQ.IBB2.AND. &
                       JB(M).EQ.IBB1)) EXFL=.true.

                  ENDDO
                ENDIF
              ENDDO
            ENDIF

            IF ((IB2.EQ.IB(J).AND.IB1.NE.JB(J)).OR. &
                ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)).AND. &
                .NOT.EXFL) THEN
              ! IB2
              DO K=1,NBOND
                IF (IB1.EQ.IB(K).AND.JB(K).NE.IB2) IBB1=JB(K)
                IF (IB1.EQ.JB(K).AND.IB(K).NE.IB2) IBB1=IB(K)

                IF ((IB1.EQ.IB(K).AND.JB(K).NE.IB2).OR. &
                    (IB1.EQ.JB(K).AND.IB(K).NE.IB2)) THEN
                  DO M=1,NBOND
                    ! 1-4 connectivity detected in PSF, set flag
                    IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                      .OR.(IB(M).EQ.IBB2.AND. &
                       JB(M).EQ.IBB1)) EXFL=.true.

                  ENDDO
                ENDIF
              ENDDO
            ENDIF

          ENDDO

          ! -----------------------------------------------------------
          ! Check if the same 1-4 connectivity is present on surface L
          ! -----------------------------------------------------------

          ! First check the cases where at least one of the outer bonds is
          ! defined for surface L

          ! EXF14 defines if one of the outer bonds  was found for surface L
          EXF14=.false.
          DO J=1,NCBON
            IF (IB1.EQ.ICBON1(J).AND.IB2.NE.ICBON2(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB1=ICBON2(J)
            IF (IB1.EQ.ICBON2(J).AND.IB2.NE.ICBON1(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB1=ICBON1(J)

            IF (IB2.EQ.ICBON1(J).AND.IB1.NE.ICBON2(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB2=ICBON2(J)
            IF (IB2.EQ.ICBON2(J).AND.IB1.NE.ICBON1(J) &
                .AND.QCBON(L,J).AND..NOT.INFL) IBB2=ICBON1(J)

            ! If bond IB1-IBB1 on surface L detected
            IF (((IB1.EQ.ICBON1(J).AND.IB2.NE.ICBON2(J) &
                .AND.QCBON(L,J)).OR.(IB1.EQ.ICBON2(J).AND. &
                 IB2.NE.ICBON1(J).AND.QCBON(L,J))).AND. &
                 .NOT.INFL) THEN

              ! Outer bond definition was found on surface L
              EXF14=.true.

              ! INF14 defines if the other outer bond was found for surface L
              INF14=.false.
              DO K=1,NCBON
                IF (IB2.EQ.ICBON1(K).AND.IB1.NE.ICBON2(K) &
                   .AND.QCBON(L,K)) IBB2=ICBON2(K)
                IF (IB2.EQ.ICBON2(K).AND.IB1.NE.ICBON1(K) &
                   .AND.QCBON(L,K)) IBB2=ICBON1(K)

                ! If bond IB2-IBB2 on surface L detected
                IF (((IB2.EQ.ICBON1(K).AND.IB1.NE.ICBON2(K)) &
                   .OR.(IB2.EQ.ICBON2(K).AND.IB1.NE.ICBON1(K))) &
                   .AND.QCBON(L,K)) THEN

                  ! Second outer bond definition was found on surface L
                  INF14=.true.
                  DO M=1,NCBON
                    ! If bond IBB1-IBB2 on surface L detected
                    IF (((ICBON1(M).EQ.IBB1.AND.ICBON2(M).EQ.IBB2) &
                       .OR.(ICBON1(M).EQ.IBB2.AND. &
                       ICBON2(M).EQ.IBB1)).AND.QCBON(L,M)) THEN 
                      ! Activate 1-4 interaction flag 
                      INFL=.true.
   !     WRITE(OUTU,*)'Check 1: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                    ! Reset INF14 to .false. in case IBB1-IBB2 is 
                    ! deactivated on surface L
                    ELSEIF (((ICBON1(M).EQ.IBB1.AND. &
                           ICBON2(M).EQ.IBB2) &
                           .OR.(ICBON1(M).EQ.IBB2.AND. &
                           ICBON2(M).EQ.IBB1)).AND. &
                           .NOT.QCBON(L,M)) THEN
                      INF14=.false.
                    ENDIF
                  ENDDO 
   !     WRITE(OUTU,*)'Check 2: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                  ! No 1-4 interaction detected yet: Check if IBB1-IBB2 
                  ! bond comes from PSF and is not explicitly defined on
                  ! surface L
                  IF (.NOT.INFL) THEN
                    DO M=1,NBOND
                      IF((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                         .OR.(IB(M).EQ.IBB2.AND. &
                         JB(M).EQ.IBB1)) THEN

                        INFL=.true.
    !    WRITE(OUTU,*)'Check 3: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                        ! Set INFL back to .false. in case IBB1-IBB2 is
                        ! defined in RMD parameter file but deactivated
                        ! on surface L 
                        DO N=1,NCBON 
                          IF (((ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2).OR. &
                               (ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.

   !   WRITE(OUTU,*)'Check 4: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                        ENDDO
                      ENDIF
                    ENDDO
                  ENDIF
                ENDIF
              ENDDO
   !     WRITE(OUTU,*)'Check 5: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
              ! If bond IB2-IBB2 not detected on surface L
              IF (.NOT.INF14.AND..NOT.INFL) THEN

                ! Check if IB2-IBB2 is defined as a PSF bond
                !
                ! CNT2 is used as a flag to get out of the K loop in 
                ! case the 1-4 interaction was confirmed by one of the
                ! bonds in NBOND
                CNT2=0 
                DO K=1,NBOND
                  IF (IB(K).EQ.IB2.AND. &
                     JB(K).NE.IB1.AND.CNT2.EQ.0) IBB2=JB(K)
                  IF (JB(K).EQ.IB2.AND. &
                     IB(K).NE.IB1.AND.CNT2.EQ.0) IBB2=IB(K)

                  IF (((IB(K).EQ.IB2.AND.JB(K).NE.IB1).OR. &
                      (JB(K).EQ.IB2.AND.IB(K).NE.IB1)) &
                     .AND.CNT2.EQ.0) THEN
                    INFL=.true.
   !     WRITE(OUTU,*)'Check 6: IB1, IB2, INFL',IB1,IB2,IBB1,IBB2,INFL
                    ! Check if same bond is deactivated on
                    ! surface L
                    DO M=1,NCBON
                      ! Set INFL back to .false. in case IB2-IBB2 is
                      ! defined in RMD parameter file but deactivated
                      ! on surface L 
                      IF (((ICBON1(M).EQ.IB2.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IB2)).AND. &
                          (.NOT.QCBON(L,M))) INFL=.false.

   !   WRITE(OUTU,*)'Check 7: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                    ENDDO

   !     WRITE(OUTU,*)'Check 8: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                    ! At this point IB2-IBB2 was found to be present and 
                    ! uniquely defined on PSF, check for IBB1-IBB2 bond
                    IF (INFL) THEN

                      ! Check if IBB1-IBB2 is defined on surface L but
                      ! deactivated. Set INFL back to .false. in that case
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                            ICBON2(M).EQ.IBB2).OR. &
                            (ICBON2(M).EQ.IBB1.AND. &
                            ICBON1(M).EQ.IBB2)).AND. &
                            (.NOT.QCBON(L,M))) INFL=.false.

   !     WRITE(OUTU,*)'Check 9: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2,INFL
                      ENDDO
                    ENDIF

                    ! If not defined for surface L check if IBB1-IBB2 is 
                    ! defined on PSF
                    IF (INFL) THEN
           !         CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                           .OR.(IB(M).EQ.IBB2.AND. &
                           JB(M).EQ.IBB1)) THEN
                          ! IBB1-IBB2 should not be defined in RMD in this case
                          DO N=1,NCBON
                            IF (((ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2).OR. &
                               (ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.
                          ENDDO
           !             IF (CNT3.EQ.1) INFL=.false.

  !   IF (CNT2.EQ.1)  WRITE(OUTU,*)'Check 11: IB1, IB2, INFL',IB1, IB2,IBB1,IBB2, INFL

                        ENDIF
                      ENDDO
                    ENDIF
  !  WRITE(OUTU,*)'Check 12: IB1, IB2,IBB1, IBB2, INFL',IB1, IB2,IBB1,IBB2, INFL
                    ! If IBB1-IBB2 not in PSF and not in RMD set INFL to .false.
                    IF (INFL) THEN
                      ! CNT3 is used as a flag to mark a detected bond in NBOND or NCBON
                      CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                           .OR.(IB(M).EQ.IBB2.AND. &
                           JB(M).EQ.IBB1)) CNT3=1

                      ENDDO
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                             ICBON2(M).EQ.IBB2).OR. &
                             (ICBON1(M).EQ.IBB2.AND. &
                             ICBON2(M).EQ.IBB1)).AND. &
                             QCBON(L,M)) CNT3=1
                      ENDDO
                      ! Reset INFL in case no bond was detected
                      IF (CNT3.EQ.0) INFL=.false.
   ! IF (CNT2.EQ.0)  WRITE(OUTU,*)'Check 13: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL,CNT2
                    ENDIF
                    IF (INFL.AND.CNT2.EQ.0) CNT2=1
                  ENDIF
                ENDDO
              ENDIF

            ! If bond IB2-IBB2 on surface L detected
            ELSEIF (((IB2.EQ.ICBON1(J).AND.IB1.NE.ICBON2(J) &
                .AND.QCBON(L,J)).OR.(IB2.EQ.ICBON2(J).AND. &
                 IB1.NE.ICBON1(J).AND.QCBON(L,J))).AND..NOT.INFL) &
                 THEN

              ! Outer bond definition was found on surface L
              EXF14=.true.

              ! INF14 defines if the other outer bond was found for surface L
              INF14=.false.
              DO K=1,NCBON
                IF (IB1.EQ.ICBON1(K).AND. &
                    IB2.NE.ICBON2(K).AND. &
                    QCBON(L,K)) IBB1=ICBON2(K)
                IF (IB1.EQ.ICBON2(K).AND. &
                   IB2.NE.ICBON1(K).AND. &
                    QCBON(L,K)) IBB1=ICBON1(K)

                ! If bond IB1-IBB1 on surface L detected
                IF (((IB1.EQ.ICBON1(K).AND. &
                    IB2.NE.ICBON2(K)).OR.(IB1.EQ.ICBON2(K).AND. &
                   IB2.NE.ICBON1(K))).AND.QCBON(L,K)) THEN

                  ! Second outer bond definition was found on surface L
                  INF14=.true.
                  DO M=1,NCBON
                    ! If bond IBB1-IBB2 on surface L detected
                    IF (((ICBON1(M).EQ.IBB2.AND.ICBON2(M).EQ.IBB1) &
                       .OR.(ICBON1(M).EQ.IBB1.AND. &
                       ICBON2(M).EQ.IBB2)).AND.QCBON(L,M)) THEN 
                      ! Activate 1-4 interaction flag 
                      INFL=.true.

                    ! Reset INF14 to .false. in case IBB1-IBB2 is 
                    ! deactivated on surface L
                    ELSEIF (((ICBON1(M).EQ.IBB2.AND. &
                             ICBON2(M).EQ.IBB1) &
                             .OR.(ICBON1(M).EQ.IBB1.AND. &
                            ICBON2(M).EQ.IBB2)).AND. &
                            .NOT.QCBON(L,M)) THEN
                      INF14=.false.
                    ENDIF
                  ENDDO 

                  ! No 1-4 interaction detected yet: Check if IBB1-IBB2 
                  ! bond comes from PSF and is not explicitly defined on
                  ! surface L
                  IF (.NOT.INFL) THEN
                    DO M=1,NBOND
                      IF((IB(M).EQ.IBB2.AND.JB(M).EQ.IBB1) &
                         .OR.(IB(M).EQ.IBB1.AND. &
                         JB(M).EQ.IBB2)) THEN

                        INFL=.true.

                        ! Set INFL back to .false. in case IBB1-IBB2 is
                        ! defined in RMD parameter file but deactivated
                        ! on surface L 
                        DO N=1,NCBON 
                          IF (((ICBON1(N).EQ.IBB2.AND. &
                               ICBON2(N).EQ.IBB1).OR. &
                               (ICBON1(N).EQ.IBB1.AND. &
                               ICBON2(N).EQ.IBB2)).AND. &
                               .NOT.QCBON(L,N)) INFL=.false.
                        ENDDO
                      ENDIF
                    ENDDO
                  ENDIF
                ENDIF
              ENDDO

              ! If bond IB1-IBB1 not detected on surface L
              IF (.NOT.INF14.AND..NOT.INFL) THEN
                ! Check if IB1-IBB1 is defined as a PSF bond
                !
                ! CNT2 is used as a flag to get out of the K loop in 
                ! case the 1-4 interaction was confirmed by one of the
                ! bonds in NBOND
                CNT2=0
                DO K=1,NBOND
                  IF (IB(K).EQ.IB1.AND. &
                     JB(K).NE.IB2.AND.CNT2.EQ.0) IBB1=JB(K)
                  IF (JB(K).EQ.IB1.AND. &
                     IB(K).NE.IB2.AND.CNT2.EQ.0) IBB1=IB(K)
 
                  IF (((IB(K).EQ.IB1.AND.JB(K).NE.IB2).OR. &
                      (JB(K).EQ.IB1.AND.IB(K).NE.IB2)) &
                     .AND.CNT2.EQ.0) THEN
                    INFL=.true.

                    ! Check if same bond is deactivated on
                    ! surface L
                    DO M=1,NCBON
                      ! Set INFL back to .false. in case IB1-IBB1 is
                      ! defined in RMD parameter file but deactivated
                      ! on surface L
                      IF (((ICBON1(M).EQ.IB1.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IB1)).AND. &
                          (.NOT.QCBON(L,M))) INFL=.false.
                    ENDDO

                    ! At this point IB1-IBB1 was found to be present and 
                    ! uniquely defined on PSF, check for IBB1-IBB2 bond
                    IF (INFL) THEN
                      ! Check if IBB1-IBB2 is defined on surface L but
                      ! deactivated. Set INFL back to .false. in that case
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB2.AND. &
                           ICBON2(M).EQ.IBB1).OR. &
                           (ICBON2(M).EQ.IBB2.AND. &
                            ICBON1(M).EQ.IBB1)).AND. &
                           (.NOT.QCBON(L,M))) INFL=.false.
                      ENDDO
                    ENDIF

                    ! If not defined for surface L check if IBB1-IBB2 is 
                    ! defined on PSF
                    IF (INFL) THEN
        !              CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB2.AND.JB(M).EQ.IBB1) &
                           .OR.(IB(M).EQ.IBB1.AND. &
                           JB(M).EQ.IBB2)) THEN
                          ! IBB1-IBB2 should not be defined in RMD in this case
                          DO N=1,NCBON
                            IF (((ICBON1(N).EQ.IBB1.AND. &
                             ICBON2(N).EQ.IBB2).OR. &
                             (ICBON1(N).EQ.IBB2.AND. &
                             ICBON2(N).EQ.IBB1)).AND. &
                             .NOT.QCBON(L,N)) INFL=.false.
                          ENDDO
        !                  IF (CNT3.EQ.1) INFL=.false.
                        ENDIF
                      ENDDO
                    ENDIF

                    ! If IBB1-IBB2 not in PSF and not in RMD set INFL to .false.
                    IF (INFL) THEN
                      ! CNT3 is used as a flag to mark a detected bond in NBOND or NCBON
                      CNT3=0
                      DO M=1,NBOND
                        IF ((IB(M).EQ.IBB1.AND.JB(M).EQ.IBB2) &
                             .OR.(IB(M).EQ.IBB2.AND. &
                             JB(M).EQ.IBB1)) CNT3=1
                      ENDDO
                      DO N=1,NCBON
                        IF (((ICBON1(N).EQ.IBB1.AND. &
                             ICBON2(N).EQ.IBB2).OR. &
                             (ICBON1(N).EQ.IBB2.AND. &
                             ICBON2(N).EQ.IBB1)) &
                             .AND.QCBON(L,N)) CNT3=1
                      ENDDO
                      ! Reset INFL in case no bond was detected
                      IF (CNT3.EQ.0) INFL=.false.
                    ENDIF
                    IF (INFL.AND.CNT2.EQ.0) CNT2=1
                  ENDIF
                ENDDO
              ENDIF
            ENDIF

          ENDDO

          ! If up to now no 1-4 interaction was detected do last possible check:
          ! IB1-IBB1 & IB2-IBB2 uniquely defined by PSF and IBB1-IBB2 
          ! defined by surface L
          IF (.NOT.INFL.AND..NOT.EXF14) THEN             ! New 2
   !  WRITE(OUTU,*)'Check 14: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL
          DO J=1,NBOND
            IF ((IB1.EQ.IB(J)).AND.IB2.NE.JB(J)) IBB1=JB(J)
            IF ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J)) IBB1=IB(J)
            IF ((IB2.EQ.IB(J)).AND.IB1.NE.JB(J)) IBB2=JB(J)
            IF ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J)) IBB2=IB(J)

            ! Check IB1-IBB1 first
            IF ((IB1.EQ.IB(J).AND.IB2.NE.JB(J)).OR. &
                ((IB1.EQ.JB(J)).AND.IB2.NE.IB(J))) THEN

              INF14=.false. ! used as flag to check if bond is defined in RMD
              DO K=1,NCBON
                IF ((ICBON1(K).EQ.IB1.AND. &
                    ICBON2(K).EQ.IBB1).OR. &
                    (ICBON2(K).EQ.IB1.AND. &
                    ICBON1(K).EQ.IBB1)) INF14=.true. 
              ENDDO

              ! If INF14 is still .false. IB1-IBB1 is uniquely described by PSF
              ! -> continue
              IF (.NOT. INF14) THEN
                DO K=1,NBOND
                  IF ((IB2.EQ.IB(K)).AND.IB1.NE.JB(K)) IBB2=JB(K)
                  IF ((IB2.EQ.JB(K)).AND.IB1.NE.IB(K)) IBB2=IB(K)

                  IF ((IB2.EQ.IB(K).AND.IB1.NE.JB(K)).OR. &
                     ((IB2.EQ.JB(K)).AND.IB1.NE.IB(K))) THEN

                    ! Insure that IB2-IBB2 is not defined in RMD
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IB2.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON2(M).EQ.IB2.AND. &
                          ICBON1(M).EQ.IBB2)) INF14=.true.
                    ENDDO

                    ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
                    ! -> continue
                    IF (.NOT. INF14) THEN

                      ! Check if IBB1-IBB2 is active on surface L
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IBB2).OR. &
                          (ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IBB1)).AND. &
                          QCBON(L,M)) INFL=.true.

    !  WRITE(OUTU,*)'Check 15: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                      ENDDO
                    ENDIF
                  ENDIF
                ENDDO
              ENDIF

            ! Alternatively check IB2-IBB2
            ELSEIF ((IB2.EQ.IB(J).AND.IB1.NE.JB(J)).OR. &
                ((IB2.EQ.JB(J)).AND.IB1.NE.IB(J))) THEN

              INF14=.false. ! used as flag to check if bond is defined in RMD             
              DO K=1,NCBON
                IF ((ICBON1(K).EQ.IB2.AND. &
                    ICBON2(K).EQ.IBB2).OR. &
                    (ICBON2(K).EQ.IB2.AND. &
                    ICBON1(K).EQ.IBB2)) INF14=.true. 
              ENDDO

              ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
              ! -> continue
              IF (.NOT. INF14) THEN
                DO K=1,NBOND
                  IF ((IB1.EQ.IB(K)).AND.IB2.NE.JB(K)) IBB1=JB(K)
                  IF ((IB1.EQ.JB(K)).AND.IB2.NE.IB(K)) IBB1=IB(K)

                  IF ((IB1.EQ.IB(K).AND.IB2.NE.JB(K)).OR. &
                     ((IB1.EQ.JB(K)).AND.IB2.NE.IB(K))) THEN

                    ! Insure that IB2-IBB2 is not defined in RMD
                    DO M=1,NCBON
                      IF ((ICBON1(M).EQ.IB1.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON2(M).EQ.IB1.AND. &
                          ICBON1(M).EQ.IBB1)) INF14=.true.
                    ENDDO

                    ! If INF14 is still .false. IB2-IBB2 is uniquely described by PSF
                    ! -> continue
                    IF (.NOT. INF14) THEN

                      ! Check if IBB1-IBB2 is active on surface L
                      DO M=1,NCBON
                        IF (((ICBON1(M).EQ.IBB2.AND. &
                          ICBON2(M).EQ.IBB1).OR. &
                          (ICBON1(M).EQ.IBB1.AND. &
                          ICBON2(M).EQ.IBB2)).AND. &
                          QCBON(L,M)) INFL=.true.

  !  WRITE(OUTU,*)'Check 16: IB1, IB2, IBB1, IBB2, INFL',IB1,IB2,IBB1,IBB2,INFL

                      ENDDO
                    ENDIF
                  ENDIF
                ENDDO
              ENDIF

            ENDIF
          ! --------- End of surface L 1-4 connectivity check
          ENDDO
          ENDIF

          ! Correct inclusion list according to EXFL and INFL status:
          ! The 1-4 inclusion needs to be removed if the connection 
          ! is not present on PSF (.NOT. EXFL), or present in both 
          ! lists (EXFL .AND. INFL).
          IF (.NOT.EXFL.OR.(EXFL.AND.INFL)) THEN
            DO J=1,NEWINC
              IF (J.GT.I-CNT1) THEN

                TMPIN14(L,J-1,1)=TMPIN14(L,J,1)
                TMPIN14(L,J-1,2)=TMPIN14(L,J,2)

              ENDIF
            ENDDO
              CNT1=CNT1+1
              NEWINC=NEWINC-1
          ENDIF

        ENDDO
        NCIN14(L)=NEWINC



      ENDDO
      ! ------------ End of surface loop L ---------

      
  ! ==================================================================
  ! At this point all exclusions and inclusions were carefuly 
  ! determined. Move the lists into global arrays of exact size.
  ! ==================================================================
  !
  ! ------------------------------------------------------------------
  ! Evaluating largest NCINC and NCIN14 cell (saved in J and K) of all 
  ! surfaces first. Subsequently, allocate static memory for ICINC and 
  ! ICIN14 of size NCRSU,J/K,2 and transfer content of temporary 
  ! arrays into these.
  ! ------------------------------------------------------------------
      
      J=0
      K=0

      DO L=1, NCRSU
        IF (NCINC(L).GT.J) THEN
          J=NCINC(L)
        ENDIF
        IF (NCIN14(L).GT.K) THEN
          K=NCIN14(L)
        ENDIF
      ENDDO

      allocate(ICINC(NCRSU,J,2),ICIN14(NCRSU,K,2))

      DO L=1,NCRSU
        DO I=1,NCINC(L)
          ICINC(L,I,1)=TMPINC(L,I,1)
          ICINC(L,I,2)=TMPINC(L,I,2)
        ENDDO

        DO I=NCINC(L)+1,J
          ICINC(L,I,1)=0
          ICINC(L,I,2)=0
        ENDDO

        DO I=1,NCIN14(L)
          ICIN14(L,I,1)=TMPIN14(L,I,1)
          ICIN14(L,I,2)=TMPIN14(L,I,2)
        ENDDO

        DO I=NCIN14(L)+1,K
          ICIN14(L,I,1)=0
          ICIN14(L,I,2)=0
        ENDDO

      ENDDO

      ! Free temporary memory:
      IF(allocated(TMPINC)) deallocate(TMPINC)
      IF(allocated(TMPIN14)) deallocate(TMPIN14)


  ! ------------------------------------------------------------------
  ! Evaluating largest NCEXC and NC14 cell (saved in J and K) of all 
  ! surfaces first. Subsequently, allocate static memory for ICEXC 
  ! and IC14 of size NCRSU+1,J/K,2 and transfer content of temporary 
  ! arrays into these.
  ! ------------------------------------------------------------------
      
      J=0
      K=0
      DO L=1, NCRSU+1
        IF (NCEXC(L).GT.J) THEN
          J=NCEXC(L)
        ENDIF
        IF (NC14(L).GT.K) THEN
          K=NC14(L)
        ENDIF
      ENDDO

      allocate(ICEXC(NCRSU+1,J,2),IC14(NCRSU+1,K,2))

      DO L=1,NCRSU+1
        DO I=1,NCEXC(L)
          ICEXC(L,I,1)=TMPEXC(L,I,1)
          ICEXC(L,I,2)=TMPEXC(L,I,2)
        ENDDO

        DO I=NCEXC(L)+1,J
          ICEXC(L,I,1)=0
          ICEXC(L,I,2)=0
        ENDDO

        DO I=1,NC14(L)
          IC14(L,I,1)=TMPC14(L,I,1)
          IC14(L,I,2)=TMPC14(L,I,2)
        ENDDO
        DO I=NC14(L)+1,K
          IC14(L,I,1)=0
          IC14(L,I,2)=0
        ENDDO

      ENDDO

      ! Free temporary memory:
      IF (allocated(TMPEXC)) deallocate(TMPEXC)
      IF (allocated(TMPC14)) deallocate(TMPC14)



      ! ------------------------------------------------------------------
      ! Check if tetra-cyclic states are defined in any of the RMD 
      ! potentials. If so, all the related bonds must be removed from the
      ! PSF file in order to subsequently guess the correct non-bond 
      ! exclusion and inclusions on these potentials!!!!
      ! If these bonds are still present in the PSF, a warning message is 
      ! printed out to indicate this restriction.
      ! 
      ! Caution: This check does not care about tetra-cyclic states
      !          defined partly by PSF and partly by RMD parameter file!
      ! ------------------------------------------------------------------
      !
      DO I=1,NCRSU

        DO J=1,NCBON

          IF (QCBON(I,J)) IB1=ICBON1(J)
          IF (QCBON(I,J)) IB2=ICBON2(J)    
   
          IF (QCBON(I,J)) THEN
            DO K=1,NCBON

              IF (ICBON1(K).EQ.IB1.AND.ICBON2(K).NE.IB2.AND. &
                QCBON(I,K)) IBB1=ICBON2(K)
              IF (ICBON2(K).EQ.IB1.AND.ICBON1(K).NE.IB2.AND. &
                QCBON(I,K)) IBB1=ICBON1(K)

              IF (((ICBON1(K).EQ.IB1.AND.ICBON2(K).NE.IB2).OR. &
                (ICBON2(K).EQ.IB1.AND.ICBON1(K).NE.IB2)).AND. &
                 QCBON(I,K)) THEN

                DO L=1,NCBON
                  IF (ICBON1(L).EQ.IBB1.AND.ICBON2(L).NE.IB1 &
                      .AND.ICBON2(L).NE.IB2.AND. &
                      QCBON(I,L)) IBB2=ICBON2(L)
                  IF (ICBON2(L).EQ.IBB1.AND.ICBON1(L).NE.IB1 &
                      .AND.ICBON1(L).NE.IB2.AND. &
                      QCBON(I,L)) IBB2=ICBON1(L)

                  IF (((ICBON1(L).EQ.IBB1.AND.ICBON2(L).NE.IB1 &
                      .AND.ICBON2(L).NE.IB2).OR. &
                    (ICBON2(L).EQ.IBB1.AND.ICBON1(L).NE.IB1 &
                      .AND.ICBON2(L).NE.IB2)).AND.QCBON(I,L)) THEN

                    DO M=1,NCBON

                      ! Need to check again if all 4 bonds are connected
                      ! and present on surface I
                      IF (((ICBON1(M).EQ.IB2.AND. &
                          ICBON2(M).EQ.IBB2) &
                          .OR.(ICBON2(M).EQ.IB2.AND. &
                          ICBON1(M).EQ.IBB2)).AND.QCBON(I,M)) THEN

                      ! tetra-cyclic state detected on surface I
                      ! Print warning if any of these bonds is also 
                      ! defined in the PSF file.
                        IF (QCBON(NCRSU+1,J).OR.QCBON(NCRSU+1,K) &
                           .OR.QCBON(NCRSU+1,L).OR. &
                           QCBON(NCRSU+1,M)) THEN

                           WRITE(OUTU,FMT = '(A,A,I3)') 'CROSSINIT> ', &
                          'Tetra-cyclic state detected on surface', I 
                           WRITE(OUTU,810)
                          CALL WRNDIE(0,'<CROSSINIT>', &
                          'Please remove all related bonds from the PSF.')

                        ENDIF
                      ENDIF 
                    ENDDO

                  ENDIF
                ENDDO

              ENDIF
            ENDDO
        
          ENDIF
        ENDDO

      ENDDO 



  !==================================================================
  ! Print out exclusion lists
  !

      IF(PRNLEV.GE.5) THEN 

        WRITE(OUTU,922)
922     FORMAT(' CROSSINIT> Nonbond-Exclusion lists generated:')
        WRITE(OUTU,*)'CROSSINIT>    This list shows the non-bonding interactions'
        WRITE(OUTU,*)'CROSSINIT>    which are removed on each surface because'
        WRITE(OUTU,*)'CROSSINIT>    they are present according to the PSF bond '
        WRITE(OUTU,*)'CROSSINIT>    definitions.'
        DO L=1, NCRSU+1

          WRITE(OUTU,810)
          IF (L.LE.NCRSU) THEN
            WRITE(OUTU,923) L, NCEXC(L) 
          ELSE
            WRITE(OUTU,923) 0, NCEXC(L)
            WRITE(OUTU,*)'CROSSINIT>    This list contains 1-2 & 1-3 interactions of'
            WRITE(OUTU,*)'CROSSINIT>    PSF bonds which also have at least one member'
            WRITE(OUTU,*)'CROSSINIT>    listed in the ATOM section. These 1-2 & 1-3'
            WRITE(OUTU,*)'CROSSINIT>    interactions are introduced in case the PSF'
            WRITE(OUTU,*)'CROSSINIT>    bond is removed from a certain RXMD surface.'
          ENDIF
923       FORMAT(' CROSSINIT> Number of new 1-2 & 1-3 exclusions', &
                ' in ',I2,': ',I3)
          WRITE(OUTU,924)
924       FORMAT(' CROSSINIT> Atom 1 Atom 2') 
          IF(PRNLEV.GE.6) THEN
            DO I=1,NCEXC(L)
              WRITE(OUTU,930) ICEXC(L,I,1),ICEXC(L,I,2)
            ENDDO
          ENDIF
          WRITE(OUTU,810)
          IF (L.LE.NCRSU) THEN
            WRITE(OUTU,925) L,NC14(L)
          ELSE
            WRITE(OUTU,925) 0,NC14(L)
            WRITE(OUTU,*)'CROSSINIT>    This list contains 1-4 interactions of PSF'
            WRITE(OUTU,*)'CROSSINIT>    bonds which also have at least one member'
            WRITE(OUTU,*)'CROSSINIT>    listed in the ATOM section. These 1-4'
            WRITE(OUTU,*)'CROSSINIT>    interactions are introduced in case the PSF'
            WRITE(OUTU,*)'CROSSINIT>    bond is removed from a certain RXMD surface.'
          ENDIF
925       FORMAT(' CROSSINIT> Number of new 1-4 exclusions', &
                 ' in ',I2,': ',I3)
          WRITE(OUTU,924) 
          IF(PRNLEV.GE.6) THEN 
            DO I=1,NC14(L)
              WRITE(OUTU,930) IC14(L,I,1),IC14(L,I,2)
            ENDDO
          ENDIF  
          WRITE(OUTU,810)
        ENDDO

        WRITE(OUTU,800)
930     FORMAT(' CROSSINIT> ',(2I6)) 



  !==================================================================
  ! Print out inclusion lists 
  !
        WRITE(OUTU,810) 
        WRITE(OUTU,810) 
        WRITE(OUTU,222)
222     FORMAT(' CROSSINIT> Nonbond-Inclusion lists generated:')   
        WRITE(OUTU,*)'CROSSINIT>    This list shows the non-bonding interactions'
        WRITE(OUTU,*)'CROSSINIT>    which are introduced on each surface because'
        WRITE(OUTU,*)'CROSSINIT>    they are missing according to the PSF bond '
        WRITE(OUTU,*)'CROSSINIT>    definitions.'
        DO L=1, NCRSU
     
          WRITE(OUTU,810) 
          WRITE(OUTU,223) L, NCINC(L) 
223       FORMAT(' CROSSINIT> Number of new 1-2 & 1-3 inclusions', &
                 ' in ',I2,': ',I3)
          WRITE(OUTU,224)
224       FORMAT(' CROSSINIT> Atom 1 Atom 2') 
          IF(PRNLEV.GE.6) THEN
            DO I=1,NCINC(L)
              WRITE(OUTU,230) ICINC(L,I,1),ICINC(L,I,2)
            ENDDO
          ENDIF
          WRITE(OUTU,810)
          IF (L.LE.NCRSU) THEN
            WRITE(OUTU,225) L,NCIN14(L)
          ELSE
            WRITE(OUTU,225) 0,NCIN14(L)
          ENDIF
225       FORMAT(' CROSSINIT> Number of new 1-4 inclusions', &
                 ' in ',I2,': ',I4)
          WRITE(OUTU,224) 
          IF(PRNLEV.GE.6) THEN 
            DO I=1,NCIN14(L)
              WRITE(OUTU,230) ICIN14(L,I,1),ICIN14(L,I,2)
            ENDDO
          ENDIF  
          WRITE(OUTU,810)
        ENDDO
      ENDIF

      WRITE(OUTU,800)
230   FORMAT(' CROSSINIT> ',(2I6)) 


  !========END of non-bond ex- & inclusion setup ====================

  !=======User defined VDW potentials for dissociating bonds=========
  ! This section is called if the user gives the BFDVDW option to RXMD.
  ! It is designed to modify the standard CHARMM nonbonding LJ 12-6 
  ! potential of atom pairs which have a broken bond defined in one or
  ! more surfaces in the RMD BOND or MORS parameter section. The 
  ! special LJ potential is applied only on the surfaces where the 
  ! specific bonds are broken (NB interaction occurs) and only for the
  ! 1-2 and 1-3 interactions among the possible bonding partners.
  ! VATT, another option in RXMD is the exponent of the attractive LJ
  ! part and VREP the exponent of the repulsive part. Specific VdW 
  ! epsilon and VdW radii values can be given in the BOND or MORS 
  ! section. The force constant in the BOND section is read as epsilon
  ! (must be negative in order to be recognized as a broken bond) and
  ! equil. distance of the bond is read as the VdW radius. In case of a
  ! MORS bond which is broken VdW epsilon and radius is stored in the
  ! dissociation energy variable (must be negative!) and the equil. 
  ! distance variable


      LBT=NEXTA4(COMLYN,COMLEN)
      IF (LBT.EQ.'BVDW') THEN 
        BFDVDW=.true.

        ! Read atoms and parameter from CparU 
        READ(CparU,*) LBT,NBVDW
        IF(LBT.NE.'BVDW') CALL WRNDIE(-2,'<CROSSINIT>', &
         'Missing BVDW segment in input')

        WRITE(OUTU,810)
        WRITE(OUTU,810)
        WRITE(OUTU,*)'CROSSINIT> Modified VdW functions for dissociable'
        WRITE(OUTU,*)'CROSSINIT> bonds activated.'
        WRITE(OUTU,*)'CROSSINIT>'
        WRITE(OUTU,*)'CROSSINIT> Atom1  Atom2  surf. eps. sig. attr-exp. rep-exp.'
        WRITE(OUTU,810)

        ! Deallocate memory if previously activated
        IF (allocated(BVDWEXC1)) deallocate(BVDWEXC1,BVDWEXC2,BVDWEPS, &
                                           BVDWSIG,ATTP,REPP,BVDWSURF)
        ! Allocate memory for BVDW usage
        allocate(BVDWEXC1(NBVDW),BVDWEXC2(NBVDW),BVDWEPS(NBVDW), &
                 BVDWSIG(NBVDW),ATTP(NBVDW),REPP(NBVDW), &
                 BVDWSURF(NBVDW,NCRSU))

        ! Set all BVDW surface flags initially to false
        DO L=1,NBVDW
          DO J=1,NCRSU
            BVDWSURF(L,J)=.false.
          ENDDO
        ENDDO

        ! Read in BVDW parameters
        DO L=1,NBVDW
          CNT1=0 !Used as a counter for the detection of BVDW interactions
          READ(CparU,*) BVDWEXC1(L),BVDWEXC2(L),BVDWEPS(L), &
                        BVDWSIG(L),REPP(L),ATTP(L)

          ! Find out if interaction is a 1-2 (listed under BOND or MORS)
          DO I=1,NCBON

            IF ((BVDWEXC1(L).EQ.ICBON1(I).AND. &
                BVDWEXC2(L).EQ.ICBON2(I)).OR. &
                (BVDWEXC2(L).EQ.ICBON1(I).AND. &
                BVDWEXC1(L).EQ.ICBON2(I))) THEN

              CNT1=CNT1+1 !1-2 interaction detected
              ! Activate BVDW flag for surfaces where bond is missing
              DO J=1,NCRSU

                IF (.NOT.QCBON(J,I)) BVDWSURF(L,J)=.true.
              ENDDO
            ENDIF
          ENDDO

          ! If its not a 1-2 interaction, it must be a 1-3 interaction listed in the ANGL section
          IF (CNT1.EQ.0) THEN
            DO K=1,NCANG

              IF ((ICANG1(K).EQ.BVDWEXC1(L).AND. &
                  ICANG3(K).EQ.BVDWEXC2(L)) &
                 .OR.(ICANG1(K).EQ.BVDWEXC2(L).AND. &
                  ICANG3(K).EQ.BVDWEXC1(L))) THEN

                DO J=1,NCRSU
                  IF (.NOT.QCANG(J,K)) BVDWSURF(L,J)=.true.
                ENDDO

              ENDIF

            ENDDO
          ENDIF
        ENDDO

        ! Write out all detected BVDW interactions 
233     FORMAT(' CROSSINIT>',I6,' ',I6,'  ',I2,'   ',F5.3, &
                           ' ',F5.3,' ',F4.1,'     ',F4.1)
        DO L=1,NBVDW
          DO J=1,NCRSU
            IF (BVDWSURF(L,J)) THEN
              WRITE(OUTU,233) BVDWEXC1(L),BVDWEXC2(L), &
                              J,BVDWEPS(L), &
                              BVDWSIG(L),ATTP(L),REPP(L)
            ENDIF
          ENDDO
        ENDDO

        WRITE(OUTU,*)'CROSSINIT>'
        WRITE(OUTU,800)

      ! If BVDW option has not defined deactivate corresponding flag
      ELSE
        BFDVDW=.false.
      ENDIF
  !================END of user defined Vdw potentials================



  !=================================================================
  ! Fill up array of factors (0-->1) for crossing
  ! Does this really make sense. Let it be for now /jonas 
  !
      
      IF (QTANH) THEN
         N=(XTIME-1)/2
         COEFF=2.4_CHM_REAL/DBLE(N)
         IF(PRNLEV.GE.6 ) WRITE(OUTU,810)
         DO I = 1, XTIME
            XFACTOR(I) = (TANH(COEFF*(I-(N+1.0_CHM_REAL)))+1.0_CHM_REAL)/2.0_CHM_REAL
  !           XFACTOR(I) = (TANH(0.6*(I-5_chm_real))+1_chm_real)/2_chm_real
           IF(PRNLEV.GE.6 ) THEN
             WRITE(OUTU,810) 
             WRITE(OUTU,931)  
             WRITE(OUTU,932) I,XFACTOR(I)
  !           WRITE(OUTU,810)
           ENDIF 
         ENDDO
      ENDIF
931   FORMAT(' CROSSINIT> Discretized switch function')
932   FORMAT(' CROSSINIT> Step Factor',I6,F6.3)
      IF(PRNLEV.GE.6 ) WRITE(OUTU,800)

  !==================================================================

  !==================================================================
  ! Effectively only need (XTIME+1)/2 points to be stored
  ! in arrays
 
      XTIME=(XTIME+1)/2

  !==================================================================
  ! Check that final value of XTIME < MAXXTIME 
  ! otherwise array overflow!

  !      IF ( XTIME .GE. MAXXTIME ) THEN
  !         CALL WRNDIE(-2,'<XINGINIT>','XTIME exceeds limit') 
  !      ENDIF

  !==================================================================
  ! Resetting the crossing algorithm for a fresh (re-)start   
  !

      DO L=1,NCRSU
        QON(L)=.FALSE.
      ENDDO
      QCON=.FALSE.
      QCROS=.FALSE.
      XRESTARTTIME=CROSSCOUNT+XTIME+2
      QCFOUND=.FALSE.
      QXTIME=.FALSE.
      XSTRT=.FALSE.
      QCEND=.FALSE.

    RETURN
  END SUBROUTINE CROSSINIT

  !====================================================================C
  ! SURFACE CROSSING ALGORITHM - THE MAIN ENGINE                       C
  !====================================================================C

  SUBROUTINE ECROSS(EU,X,Y,Z,DX,DY,DZ,NATOMX) 
    !====================================================================C     
    ! Main subroutine for doing surface crossing and dissociation     
    !
    ! Jonas Danielsson 2006
    ! Stephan Lutz 2009
    !====================================================================C
    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    ! ... For BFDVDW
    use energym
    !
    real(chm_real) EU,FACTOR
    INTEGER NATOMX,I
    LOGICAL J
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)


    !===================================================================
    ! This should not be necessary.
    !
    IF (NCRUN.EQ.0) RETURN
      !=================================================================

      !=================================================================
      ! Zero the energy contribution
      !
      EU = zero
      !=================================================================

      !=================================================================
      !  Initial step: Check which surface we are on and set parameters
      !
      J=.FALSE.
      DO I=1, NCRSU
        IF (QON(I)) THEN
          J=.TRUE.
        ENDIF 
      ENDDO

      IF ((.NOT.J).AND.(.NOT.QCON)) THEN
        CALL CSETSURFACE(X,Y,Z,NATOMX)
        CALL CSETCHARGES(0.0_CHM_REAL) 
      ENDIF 


      !==============================================================
      !  Check that the non-bonded interactions are handled as we like it
      !==============================================================
      !
      IF (.NOT. lElec) CALL WrnDie(-2,'<CROSS>','Elec Required')
      IF (.NOT. lCons) CALL WrnDie(-2,'<CROSS>','Cdie Required')
      IF (.NOT. lShft) CALL WrnDie(-2,'<CROSS>','Shift Required')
      IF (NBXMOD.LT.3) CALL WrnDie(-2,'<CROSS>', &
                         'Only Nbxmod 3,4 and 5 supported')

      !==============================================================
      ! The backtracking algorithm is only active during dynamics 
      !==============================================================

      IF(DYNAMQ) THEN
        ! Update the counter for number of calls
        !
        CROSSCOUNT = CROSSCOUNT + 1
        !------------------------------------------------------------
        ! Prepare and detect crossings 
        CALL CSTEP(X,Y,Z,DX,DY,DZ,NATOMX)

        !------------------------------------------------------------
        ! Set the mixing factor between surface involved in crossing

        CALL CSETFACTOR(FACTOR)


      ELSE
  !==================================================================
  ! When no dynamics, just set the system on the lowest surface
  !

        CALL CSETSURFACE(X,Y,Z,NATOMX)
        CALL CSETCHARGES(FACTOR)

  !==================================================================
      ENDIF

  !==================================================================  
  ! When crossing is in progress set the charges
  !

      IF(QCON.OR.QCEND) THEN
        CALL CSETCHARGES(FACTOR) 
      ENDIF
  !==================================================================
  ! Calculating Energies and Forces
  !==================================================================

      !--------------------------------------------------------------
      !   Remove the PSF-encoded surface 
      !--------------------------------------------------------------

      CALL CREMOVE(EU,X,Y,Z,DX,DY,DZ,NATOMX)
      !--------------------------------------------------------------
      !   Add the user-defined PESs in the right mix
      !--------------------------------------------------------------

      CALL CFORCES(EU,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)


      !--------------------------------------------------------------
      !   Replace VDW interactions with UVDW if requested
      !--------------------------------------------------------------
      IF (BFDVDW.AND.QETERM(VDW)) THEN
 
        CALL RVDW123(EU,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)

      ENDIF

      !--------------------------------------------------------------
      !   Add the SHIFT to the energy
      !   If no crossing in progress and LOWSU=1 SSHIFT is zero by 
      !   definition.
      IF ((.NOT.QCON).AND.(LOWSU.EQ.1)) THEN
        EU = EU

      !   If no crossing in progress and LOWSU.GT.1 substract 
      !   corresponding SHIFT to surface 1
      ELSEIF ((.NOT.QCON).AND.(LOWSU.GT.1)) THEN
        EU = EU - SSHIFT(LOWSU-1)

      !   If crossing in progress substract SSHIFT(1-OLDSU)*FACTOR and 
      !   SSHIFT(1-NEWSU)*(ONE-FACTOR)
      ELSEIF (QCON) THEN
        EU = EU-FACTOR*SSHIFT(OLDSU-1)-(ONE-FACTOR)*SSHIFT(NEWSU-1)

   !  WRITE(OUTU,*)' ECROSS> FACTOR, EU:', FACTOR,EU

      ENDIF
      !-------------------------------------------------------------

      IF(PRNLEV.GE.7) THEN
        WRITE(OUTU,5)
        WRITE(OUTU,20) EU
        WRITE(OUTU,5)

5       FORMAT(' ECROSS>======================================')
20      FORMAT(' ECROSS> Added final energy: ',F8.3)
      ENDIF

      !=============================================================

    RETURN

  END SUBROUTINE ECROSS

  !=================================================================
  !  CROSSING HELP ROUTINES                                         
  !=================================================================

  SUBROUTINE CSETFACTOR(FACTOR)
    !===============================================================
    ! Sets the mixing factor of OLDSU and NEWSU on the current PES
    !  Jonas Danielsson 2006, Stephan Lutz 2008
    !===============================================================
    use dimens_fcm
    use number    
    use stream   

    real(chm_real) FACTOR

    !--------------------------------------------------------------
    ! Is crossing in progress?
    !--------------------------------------------------------------
    IF (QCON) THEN
      FACTOR=ONE-XFACTOR(XCROSCOUNT) 

    ELSE 
      FACTOR=ZERO
    ENDIF
 
    IF(PRNLEV.GE.7) THEN
         WRITE(OUTU,5) 
         WRITE(OUTU,10) FACTOR
         WRITE(OUTU,5)
    ENDIF
 
5   FORMAT(' CSETFACTOR>======================================')
10  FORMAT(' CSETFACTOR> New FACTOR: ',F6.3)
    RETURN 
  END SUBROUTINE CSETFACTOR

  !===============================================================


  SUBROUTINE CBARRIER(X,Y,Z,NATOMX)
    !=============================================================     
    !     This subroutine checks for crossings                        
    !     Jonas Danielsson                                              
    !=============================================================
    use dimens_fcm
    use number
    use reawri
    use stream
    use psf
    use cnst_fcm
    use contrl
    use consta
    !
    ! passed variables
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX) 
    ! local variables
    real(chm_real) TMPED,ED,TIME,CUSHIFT,CUBARTO
    real(chm_real) BNDDIST
    INTEGER I,J,K,L
    LOGICAL FIRST,DELCHK


    !-------------------------------------------------------------
    ! Set time
    !-------------------------------------------------------------

    TIME = MDSTEP*TIMEST

    !-------------------------------------------------------------
    ! Check energy difference between present and all remaining 
    ! surfaces and add the corresponding shift
    !-------------------------------------------------------------

    K=1
    ! ... FIRST becomes .true. after the first energy comparison
    !     is done
    FIRST=.FALSE.
    DO I=1, NCRSU-1
      DO J=I+1, NCRSU

        ! Check for current surface
        IF (I.EQ.LOWSU) THEN
          ! Compare absolute surface shifts to evaluate shift 
          ! and tolerance signs
          CUBARTO=SBARTO(K)

          ! See if there is a cutoff distance defined for bond 
          ! formations and if so, check if distance of possibly new 
          ! formed bonds is within this cutoff to allow for 
          ! EDELTA call.
          DELCHK=.true.
          IF (BCHK .GT. 0.0_CHM_REAL) THEN

            DO L=1, NCBON
              IF (.NOT. QCBON(I,L) .AND. QCBON(J,L)) THEN
                ! Possible bond formation detected.
                ! Calculates distance between bond forming partners
                BNDDIST=DISTANCE(X(ICBON1(L)),Y(ICBON1(L)), &
                        Z(ICBON1(L)),X(ICBON2(L)),Y(ICBON2(L)), &
                        Z(ICBON2(L)))
                ! set EDELTA check flag to false if distance beyond cutoff
                IF (BNDDIST .GT. BCHK) DELCHK=.false. 

              ENDIF
            ENDDO

          ENDIF

          IF (DELCHK) THEN

            CALL EDELTA(J,TMPED,X,Y,Z,NATOMX)
            TMPED = TMPED + CUBARTO
            IF (.NOT.FIRST) THEN
              ED=TMPED
              OLDSU=I
              NEWSU=J
              FIRST=.TRUE.
            ENDIF
            IF (TMPED.LT.ED) THEN
              ED=TMPED
              OLDSU=I
              NEWSU=J
            ENDIF

          ENDIF

        ELSEIF (J.EQ.LOWSU) THEN
          ! Compare absolute surface shifts to evaluate shift and 
          ! tolerance signs
          ! IF (I.EQ.1) THEN
          CUBARTO=SBARTO(K)

          ! See if there is a cutoff distance defined for bond 
          ! formations and if check if distance of possibly new 
          ! formed bonds is within this cutoff to allow for EDELTA call.
          DELCHK=.true.
          IF (BCHK .GT. 0.0_CHM_REAL) THEN

            DO L=1, NCBON
              IF (.NOT. QCBON(J,L) .AND. QCBON(I,L)) THEN
                ! Possible bond formation detected.
                ! Calculates distance between bond forming partners
                BNDDIST=DISTANCE(X(ICBON1(L)),Y(ICBON1(L)), &
                        Z(ICBON1(L)),X(ICBON2(L)),Y(ICBON2(L)), &
                        Z(ICBON2(L)))
                ! set EDELTA check flag to false if distance beyond cutoff
                IF (BNDDIST .GT. BCHK) DELCHK=.false. 

              ENDIF
            ENDDO

          ENDIF

          IF (DELCHK) THEN

            CALL EDELTA(I,TMPED,X,Y,Z,NATOMX)
            TMPED = TMPED + CUBARTO
            IF (.NOT.FIRST) THEN
              ED=TMPED
              OLDSU=J
              NEWSU=I
              FIRST=.TRUE.
            ENDIF
            IF (TMPED.LT.ED) THEN
              ED=TMPED
              OLDSU=J
              NEWSU=I
            ENDIF

          ENDIF
        ENDIF

        K=K+1
      ENDDO
    ENDDO

    IF ((MOD(CROSSCOUNT,XCROSF).EQ.0).AND.FIRST) THEN
      ! IF(PRNLEV.GE.5) THEN
      WRITE(OUTU,5) 
      WRITE(OUTU,8887)
      WRITE(OUTU,8888) TIME,ED,OLDSU,NEWSU
      ! ENDIF
    ENDIF

8887 FORMAT(' CBARRIER>    TIME     Delta-E    current', &
            '/closest surface')
8888 FORMAT(' CBARRIER> ',f9.4,F13.5,'   ',I2,'      ',I2 )

    !-----------------------------------------------------------
    ! If crossing was found, dump the geometry and set QCFOUND
    !-----------------------------------------------------------
       
    IF (ED.LT.ZERO) THEN

      QCFOUND = .TRUE.

      IF(QUOUT) THEN 
        CALL CPDBDUMP(NATOMX,X,Y,Z,TIME)
      ENDIF

      IF(PRNLEV.GE.4) THEN
        WRITE(OutU,5)
        WRITE(OutU,10) NEWSU
        WRITE(OutU,20) TIME
        WRITE(OutU,30) 
      ENDIF

10    FORMAT(' CBARRIER> Crossing to state ',I1)
20    FORMAT(' CBARRIER> Crossing occured at t = ',F10.3)
30    FORMAT(' CBARRIER> Restoring old coordinates and starting', &
             ' crossing procedure.')            

    ENDIF

5     FORMAT( ' CBARRIER>===================================')
    RETURN
  END SUBROUTINE CBARRIER

  !================================================================

  SUBROUTINE CSETSURFACE(X,Y,Z,NATOMX)
  !================================================================   
  !     This subroutine puts the system on the lowest surface when
  !     no information about current surface energies is available   
  !                                        JD 060214
  !     Modified for multisurface evaluation SL 090403
  !================================================================
    use dimens_fcm
    use number
    use reawri
    use stream
    use inbnd
    use psf
    use cnst_fcm
    use contrl
    use consta
    use param
    use energym
    !
    INTEGER NATOMX,L
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),ED,LOWEN

    INTEGER I,J,K,II,JJ,KK,LL,IX,MULT 
    real(chm_real) EBND1,EANG1,EDIH1,EEL1,EVDW1,EEXCL1,EBOUN1,ENBND1
    real(chm_real) DEBND,DEANG,DEDIH,DEEL,DEVDW,DEEXCL,DEBOUN,DENBND
    real(chm_real) EINC1,EINV1,EINE1,EIV141,EIE141,E3P1

    real(chm_real) ETOT1
    real(chm_real) E,E1,E2,E12,REQ,FCON,TEQ,PEQ,Q11,Q12,Q21,Q22,R, &
                   DE,BETA
    real(chm_real) SIG11,SIG12,SIG21,SIG22,EFF11,EFF12,EFF21,EFF22
    real(chm_real) SIG112,SIG122,EFF112,EFF122
    real(chm_real) EEXV1,EEXV01,EEXE1,EEXE01
    real(chm_real) E14V1,E14E1,E14E01,E14V01
    real(chm_real) C2OFNB,C2ONNB,CTOFNB1,CTOFNB2,CTOFNB4,FUNCT
    real(chm_real) S,RX,RY,RZ,SIGQ1,SIGQ2,EFF1,EFF2,RUL3,RIJL, &
                   RIJU,DR4
    real(chm_real) SIGQ12,EPS1

    !-----------------------------------------------------------------
    ! Print some info
    !-----------------------------------------------------------------
     
    ! IF(PRNLEV.GE.5) THEN  
    WRITE(OutU,5) 
    WRITE(OutU,10) 
    WRITE(OutU,20)
5   FORMAT(' CSURFACE> ==================================') 
10  FORMAT(' CSETSURFACE> No surface or crossing active')
20  FORMAT(' CSETSURFACE> Checking energies to set surface')        
    !     ENDIF


    !-----------------------------------------------------------------
    ! Loop over all surfaces NCRSU
    !-----------------------------------------------------------------
    DO L=1, NCRSU

      !==============================================================
      ! BONDED PART
      !
      EBOUN1=ZERO  

      !--------------------------------------------------------------
      ! Bonds
      !--------------------------------------------------------------

      EBND1=ZERO

      IF((NCBON.GT.0).AND.QETERM(BOND)) THEN

        IF(NCHAR.GT.0) THEN

          DO I=1,NCHAR

            II=ICBON1(I)
            JJ=ICBON2(I) 

            IF(QCBON(L,I)) THEN
              REQ=RCBON(L,I)
              FCON=KCBON(L,I)
              E=UBOND(II,JJ,FCON,REQ,X,Y,Z,NATOMX)
              EBND1=EBND1+E
     !  WRITE(OUTU,*)'Harm. bond added:',II, JJ, E,'kcal/mol'
            ENDIF

          ENDDO

        ENDIF

        IF(NCMOR.GT.0) THEN

          DO I=1,NCMOR

            II=ICBON1(I+NCHAR)
            JJ=ICBON2(I+NCHAR) 

            IF(QCBON(L,I+NCHAR)) THEN
              REQ=RCMOR(L,I)
              DE=DCMOR(L,I)
              BETA=BCMOR(L,I) 
              E=MORSE(II,JJ,DE,BETA,REQ,X,Y,Z,NATOMX)
              EBND1=EBND1+E
    !   WRITE(OUTU,*)'Morse. bond added:',II, JJ, E,'kcal/mol'
            ENDIF 

          ENDDO

        ENDIF

      ENDIF
 
      EBOUN1=EBOUN1+EBND1

      ! End bonds----------------------------------------------   

      !--------------------------------------------------------
      ! Angles
      !--------------------------------------------------------

      EANG1=ZERO

      IF((NCANG.GT.0).AND.QETERM(ANGLE)) THEN

        DO I=1,NCANG

          II=ICANG1(I)
          JJ=ICANG2(I)
          KK=ICANG3(I)

          IF(QCANG(L,I)) THEN
            TEQ=TCANG(L,I)
            FCON=KCANG(L,I)
            E=UANGLE(II,JJ,KK,FCON,TEQ,X,Y,Z,NATOMX)
            EANG1=EANG1+E
    !   WRITE(OUTU,*)'Angle added:',II, JJ, KK, E,'kcal/mol'
          ENDIF

        ENDDO

      ENDIF

      EBOUN1=EBOUN1+EANG1

      ! End angles----------------------------------------------

      !---------------------------------------------------------
      ! Dihedrals
      !---------------------------------------------------------

      EDIH1=ZERO

      IF((NCDIH.GT.0).AND.QETERM(DIHE)) THEN

        DO I=1,NCDIH

          II=ICDIH1(I)
          JJ=ICDIH2(I)
          KK=ICDIH3(I)
          LL=ICDIH4(I)

          IF(QCDIH(L,I)) THEN
            PEQ=PCDIH(L,I)
            FCON=KCDIH(L,I)
            MULT=MCDIH(L,I)
            E=DIHD(II,JJ,KK,LL,FCON,PEQ,MULT,X,Y,Z,NATOMX)
            EDIH1=EDIH1+E
    !   WRITE(OUTU,*)'Dih. added:',II, JJ, KK, LL, E,'kcal/mol'
          ENDIF 

        ENDDO

      ENDIF

      EBOUN1=EBOUN1+EDIH1

      ! End dihedrals-------------------------------------------

      ! END BONDED
      !=========================================================

      !=========================================================
      ! NON-BONDED        
      !

      ENBND1=ZERO

      !---------------------------------------------------------
      ! Zero everything
      !---------------------------------------------------------
      EVDW1=ZERO
      EEL1=ZERO
    !  E3P1=ZERO

      EEXV1=ZERO
      EEXE1=ZERO
      E14V1=ZERO
      E14E1=ZERO

      EEXV01=ZERO
      EEXE01=ZERO
      E14V01=ZERO
      E14E01=ZERO

      EINC1=ZERO
      EINV1=ZERO
      EINE1=ZERO
      EIV141=ZERO
      EIE141=ZERO

      IF(QETERM(ELEC).OR.QETERM(VDW)) THEN 

        !-----------------------------------------------------------
        ! precalculating some switching and shifting stuff  
        !-----------------------------------------------------------
        C2OFNB=CTOFNB*CTOFNB
        C2ONNB=CTONNB*CTONNB
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        CTOFNB1=ONE/CTOFNB
        CTOFNB2=CTOFNB1*CTOFNB1
        CTOFNB4=CTOFNB2*CTOFNB2

        !-----------------------------------------------------------
        ! Main Nonbonded Loop
        !-----------------------------------------------------------
        IF((NCRAT.GT.0).AND.(QETERM(ELEC).OR.QETERM(VDW))) THEN

          DO I=1,NCRAT

            SIG11=VCRAD(L,I)
            EFF11=VCEPS(L,I)

            II=ICATOM(I)
            Q11=ECCHG(L,I)

            DO J=1,NATOMX

              IF((II.NE.J).AND.(ICINDX(J).LT.I)) THEN

                RX=X(II)-X(J)
                RY=Y(II)-Y(J)
                RZ=Z(II)-Z(J)

                S=RX*RX+RY*RY+RZ*RZ

                IF (S.LT.C2OFNB) THEN

                  IF(ICINDX(J).GT.0) THEN

                    SIG12=VCRAD(L,ICINDX(J))
                    EFF12=VCEPS(L,ICINDX(J)) 

                    Q12=ECCHG(L,ICINDX(J))

                  ELSE
                    SIG12=VDWR(ITC(IAC(J)))
                    EFF12=EFF(ITC(IAC(J)))

                    Q12=CG(J)

                  ENDIF

                  SIGQ1=(SIG11+SIG12)**2
                  EFF1=SQRT(EFF11*EFF12)
                  R=SQRT(S)

                  !---------------------------------------------
                  ! VdW 
                  !---------------------------------------------

                  IF(QETERM(VDW)) THEN 

                    E1=UVDW(SIGQ1,EFF1,R) 

                    !  Switching function stuff
                    IF (S.GT.C2ONNB) THEN
                      RIJL=C2ONNB-S
                      RIJU=C2OFNB-S
                      FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                      E1=(FUNCT*E1)
                    ENDIF
                    EVDW1=EVDW1+E1
     !  WRITE(OUTU,*)'VdW added:',II, J, E1,'kcal/mol'
                  ENDIF 
                  ! End Van der Waal----------------------------

                  !---------------------------------------------
                  ! Coulomb 
                  !---------------------------------------------   

                  ! Shifting function

                  IF(QETERM(ELEC)) THEN 
                    DR4=S*S
                    FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4

                    ! Calculate electrostatic potential

                    E1=UELEC(Q11,Q12,R)*FUNCT
                    EEL1=EEL1+E1
    !   WRITE(OUTU,*)'Elec added:',II, J, E1,'kcal/mol'
                  ENDIF 

                ENDIF
              ENDIF
            ENDDO
          ENDDO
        ENDIF

        ! End Coulomb part---------------------------------------

        ! Bond breaking/forming VdW part-------------------------
        IF (BFDVDW.AND.QETERM(VDW)) THEN
          DO I=1,NBVDW
            IF (BVDWSURF(I,L)) THEN
              II=BVDWEXC1(I)
              JJ=BVDWEXC2(I)

              IF(ICINDX(II).GT.0) THEN
                SIG11=VCRAD(L,ICINDX(II))
                EFF11=VCEPS(L,ICINDX(II))
              ELSE
                SIG11=VDWR(ITC(IAC(II)))
                EFF11=EFF(ITC(IAC(II)))
              ENDIF
  
              IF(ICINDX(JJ).GT.0) THEN
                SIG12=VCRAD(L,ICINDX(JJ))
                EFF12=VCEPS(L,ICINDX(JJ))
              ELSE
                SIG12=VDWR(ITC(IAC(JJ)))
                EFF12=EFF(ITC(IAC(JJ)))
              ENDIF

              RX=X(II)-X(JJ)
              RY=Y(II)-Y(JJ)
              RZ=Z(II)-Z(JJ)

              S=RX*RX+RY*RY+RZ*RZ

              IF (S.LT.C2OFNB) THEN
              ! ... Remove 1-2 CHARMM LJ interaction
                R=SQRT(S)
                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                EVDW1=EVDW1-E1
    !   WRITE(OUTU,*)'VdW removed:',II, JJ, E1,'kcal/mol'
                ! ... Add user defined LJ interaction
                E1=URVDW(BVDWSIG(I)**2.0_CHM_REAL,BVDWEPS(I), &
                         REPP(I),ATTP(I),R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-(ATTP(I)/2.0_CHM_REAL)*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                EVDW1=EVDW1+E1
     !  WRITE(OUTU,*)'1-2 VdW added:',II, J, E1,'kcal/mol'
              ENDIF
          
            ENDIF
          ENDDO

        ENDIF

        ! End bond breaking/forming VdW part--------------------

  !-------------------------------------------------------------      
  ! Exclusions and 1-4 interactions
  !-------------------------------------------------------------

  ! Surface 1 --------------------------------------------------

  !-------------------------------------------------------------
  ! Exclusions
  !-------------------------------------------------------------

        IF((NCEXC(L).GT.0).AND.(QETERM(ELEC).OR.QETERM(VDW))) THEN

          DO I=1,NCEXC(L)

            II=ICEXC(L,I,1)
            JJ=ICEXC(L,I,2)
         
            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              SIGQ1=(SIG11+SIG12)**2
              EFF1=SQRT(EFF11*EFF12)
              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF
         
                EEXV1=EEXV1+E1

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT
                EEXE1=EEXE1+E1

              ENDIF

            ENDIF
          ENDDO 
        ENDIF

        !-------------------------------------------------------
        ! 1-2 & 1-3 Inclusions
        !-------------------------------------------------------

        IF(NCINC(L).GT.0) THEN

          DO I=1,NCINC(L)

            II=ICINC(L,I,1)
            JJ=ICINC(L,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              SIGQ1=(SIG11+SIG12)**2
              EFF1=SQRT(EFF11*EFF12)
              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                EINV1=EINV1+E1
     !  WRITE(OUTU,*)'VdW added:',II,JJ, E1,'kcal/mol'
              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT
                EINE1=EINE1+E1
     !  WRITE(OUTU,*)'Elec. added:',II,JJ, E1,'kcal/mol'
              ENDIF

            ENDIF
          ENDDO 
        ENDIF


        !-------------------------------------------------------
        ! 1-4 interactions
        !-------------------------------------------------------

        ! Exclusions
        IF((NBXMOD.GE.4).AND.(NC14(L).GT.0)) THEN

          DO I=1,NC14(L)

            II=IC14(L,I,1)
            JJ=IC14(L,I,2)
         
            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))
              SIG21=SIG11
              EFF21=EFF11  
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG21=VDWR(ITC(IAC(II))+MAXATC)
                EFF21=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))
              SIG22=SIG12
              EFF22=EFF12  
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG22=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)
 
            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ2=(SIG21+SIG22)**2
                  EFF2=SQRT(EFF21*EFF22)
                  E2=UVDW(SIGQ2,EFF2,R)
                ENDIF
          
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  IF (NBXMOD.EQ.5) E2=FUNCT*E2
                ENDIF

                E14V1=E14V1+E1
     !  WRITE(OUTU,*)'VdW removed:',II,JJ, E1,'kcal/mol'
                IF (NBXMOD.EQ.5) E14V1=E14V1-E2
     !  WRITE(OUTU,*)'VdW added:',II,JJ, E2,'kcal/mol'
              ENDIF 

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT
                  E14E1=E14E1+E1
      ! WRITE(OUTU,*)'Elec. removed:',II,JJ, E1,'kcal/mol'
                ENDIF     

              ENDIF
            ENDIF
          ENDDO 
        ENDIF


        ! 1-4 inclusions

        IF((NBXMOD.GE.4).AND.(NCIN14(L).GT.0)) THEN

          DO I=1,NCIN14(L)

            II=ICIN14(L,I,1)
            JJ=ICIN14(L,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))
              SIG21=SIG11
              EFF21=EFF11  
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG21=VDWR(ITC(IAC(II))+MAXATC)
                EFF21=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))
              SIG22=SIG12
              EFF22=EFF12  
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG22=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ2=(SIG21+SIG22)**2
                  EFF2=SQRT(EFF21*EFF22)
                  E2=UVDW(SIGQ2,EFF2,R)
                ENDIF

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  IF (NBXMOD.EQ.5) E2=FUNCT*E2
                ENDIF

                EIV141=EIV141+E1
     !  WRITE(OUTU,*)'VdW added:',II,JJ, E1,'kcal/mol'
                IF (NBXMOD.EQ.5) EIV141=EIV141-E2
     !  WRITE(OUTU,*)'VdW removed:',II,JJ, E2,'kcal/mol'
              ENDIF 

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT
                  EIE141=EIE141+E1
     !  WRITE(OUTU,*)'Calc. added:',II,JJ, E1,'kcal/mol'
                ENDIF

              ENDIF
            ENDIF
          ENDDO 
        ENDIF

        ! End ex- & inclusions and 1-4 interactions

  ! Surface 0 (modified non-bonding interactions )  ------

  !-------------------------------------------------------
  ! Exclusions
  !-------------------------------------------------------
        IF(NCEXC(NCRSU+1).GT.0) THEN

          DO I=1,NCEXC(NCRSU+1)

            II=ICEXC(NCRSU+1,I,1)
            JJ=ICEXC(NCRSU+1,I,2)
              
            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))

            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
              SIG21=SIG11
              EFF21=EFF11
              Q21=Q11  
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))

            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)

            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R)

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)

                ENDIF

                EEXV01=EEXV01+E1
     !  WRITE(OUTU,*)'Surf 0 VdW added:',II,JJ, E1,'kcal/mol'
              ENDIF

               IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT

                EEXE01=EEXE01+E1
     !  WRITE(OUTU,*)'Surf 0 Calc. added:',II,JJ, E1,'kcal/mol'
              ENDIF    

            ENDIF
          ENDDO
        ENDIF
 
        !-----------------------------------------------------
        ! 1-4 interactions
        !-----------------------------------------------------

        IF((NBXMOD.GE.4).AND.(NC14(NCRSU+1).GT.0)) THEN

          DO I=1,NC14(NCRSU+1)

            II=IC14(NCRSU+1,I,1)
            JJ=IC14(NCRSU+1,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(L,ICINDX(II))
              EFF11=VCEPS(L,ICINDX(II))
              Q11=ECCHG(L,ICINDX(II))
              SIG112=SIG11
              EFF112=EFF11

            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
              SIG112=VDWR(ITC(IAC(II))+MAXATC)
              EFF112=EFF(ITC(IAC(II))+MAXATC)

            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(L,ICINDX(JJ))
              EFF12=VCEPS(L,ICINDX(JJ))
              Q12=ECCHG(L,ICINDX(JJ))
              SIG122=SIG12
              EFF122=EFF12

            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
              SIG122=VDWR(ITC(IAC(JJ))+MAXATC)
              EFF122=EFF(ITC(IAC(JJ))+MAXATC)

            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ12=(SIG112+SIG122)**2
                  EPS1=SQRT(EFF112*EFF122)

                  E12=UVDW(SIGQ12,EPS1,R) 

                ENDIF

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)

                  IF(NBXMOD.EQ.5) THEN
                    E12=(FUNCT*E12)

                  ENDIF  

                ENDIF

                E14V01=E14V01+E1
     !  WRITE(OUTU,*)'Surf 0 VdW added:',II,JJ, E1,'kcal/mol'
                IF(NBXMOD.EQ.5) THEN 
                  E14V01=E14V01-E12
    !   WRITE(OUTU,*)'Surf 0 VdW removed:',II,JJ, E12,'kcal/mol'
                ENDIF

              ENDIF

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT

                  E14E01=E14E01+E1
     !  WRITE(OUTU,*)'Surf 0 Calc. added:',II,JJ, E1,'kcal/mol'
                ENDIF

              ENDIF 
            ENDIF
          ENDDO
        ENDIF
      ENDIF  

      ! End Surface 0 exclusions----------------------------------

      EEXCL1=EEXV1+EEXE1+E14E1+E14V1+EEXV01+EEXE01+E14V01+E14E01
      EINC1=EINV1+EINE1+EIV141+EIE141

      ! END EX- & INCLUSIONS -------------------------------------

      EEL1 =EEL1-EEXE1-E14E1-EEXE01-E14E01+EINE1+EIE141
      EVDW1=EVDW1-EEXV1-E14V1-EEXV01-E14V01+EINV1+EIV141

      ENBND1=EEL1+EVDW1
 
  ! END NON-BONDED
  !===============================================================
  ! Add it all up
  !===============================================================
      ETOT1=EBOUN1+ENBND1

      ! Add absolute surface shifts
      IF (L.NE.1) THEN
        ETOT1=EBOUN1+ENBND1-SSHIFT(L-1)
        IF (ETOT1.LT.LOWEN) THEN 
          LOWSU=L
          LOWEN=ETOT1
        ENDIF 
      ELSE
        LOWEN=ETOT1
        LOWSU=L
      ENDIF

    IF (( PRNLEV.GE.4).OR.DYNAMQ) WRITE(OutU,29) L, ETOT1

    ENDDO
    !-------- Exit surface loop L -----------------------------------

  !------------------------------------------------------------------
  ! Activate the lowest surface
  !------------------------------------------------------------------
    IF (PRNLEV.GE.4) WRITE(OutU,30) LOWSU

    DO I=1,NCRSU
      IF (I.NE.LOWSU) THEN
        QON(I)=.FALSE.
      ELSE
        QON(I)=.TRUE.
      ENDIF
    ENDDO

29  FORMAT(' CSETSURFACE> Absolute energy of surface ' &
            ,I2,' = ',F11.3)
30  FORMAT(' CSETSURFACE> System on surface ',I2)

    IF ((PRNLEV.GE.4).OR.DYNAMQ) WRITE(OutU,5)


    RETURN
  END SUBROUTINE CSETSURFACE



  !==================================================================
  !   ENERGY DIFFERENCE EVALUATION                                       
  !==================================================================
  SUBROUTINE EDELTA(SELSU,ED,X,Y,Z,NATOMX)
    !     
    ! Calculates the energy difference between the two energy surfaces defined    
    ! for the surface crossing procedure.    
    !    
    !                           JD 060105
    !
    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
    use energym
    !
      INTEGER NATOMX,SELSU
      real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),ED

      INTEGER I,J,K,II,JJ,KK,LL,IX,MULT 
      real(chm_real) EBND1,EANG1,EDIH1,EEL1,EVDW1,EEXCL1, &
                     EBOUN1,ENBND1
      real(chm_real) EBND2,EANG2,EDIH2,EEL2,EVDW2,EEXCL2, &
                     EBOUN2,ENBND2
      real(chm_real) DEBND,DEANG,DEDIH,DEEL,DEVDW,DEEXCL, &
                     DEBOUN,DENBND
      real(chm_real) EINC1,EINV1,EINE1,EIV141,EIE141
      real(chm_real) EINC2,EINV2,EINE2,EIV142,EIE142
      real(chm_real) ERVDW1,ERVDW2

      real(chm_real) ETOT1,ETOT2
      real(chm_real) E,E1,E2,E12,E22,REQ,FCON,TEQ,PEQ,Q11,Q12, &
                     Q21,Q22,R,DE,BETA
      real(chm_real) SIG11,SIG12,SIG21,SIG22,EFF11,EFF12, &
                     EFF21,EFF22
      real(chm_real) SIG112,SIG122,SIG212,SIG222,EFF112,EFF122, &
                     EFF212,EFF222
      real(chm_real) EEXV1,EEXV2,EEXV01,EEXV02,EEXE1,EEXE2, &
                     EEXE01,EEXE02
      real(chm_real) E14V1,E14V2,E14E1,E14E2,E14E01,E14E02, &
                     E14V01,E14V02
      real(chm_real) C2OFNB,C2ONNB,CTOFNB1,CTOFNB2,CTOFNB4,FUNCT
      real(chm_real) S,RX,RY,RZ,SIGQ1,SIGQ2,EFF1,EFF2,RUL3,RIJL, &
                     RIJU,DR4
      real(chm_real) SIGQ12,SIGQ22,EPS1,EPS2
      real(chm_real) SSHIFT1,SSHIFT2,DELSHIF

      !--------------------------------------------------------
      ! Find present surface shifts if available
      IF (SELSU.GT.1) SSHIFT1=SSHIFT(SELSU-1)
      IF (LOWSU.GT.1) SSHIFT2=SSHIFT(LOWSU-1)

      !========================================================
      ! BONDED PART
      !

      EBOUN1=ZERO  
      EBOUN2=ZERO

      !--------------------------------------------------------
      ! Bonds
      !--------------------------------------------------------

      EBND1=ZERO
      EBND2=ZERO

      IF((NCBON.GT.0).AND.QETERM(BOND)) THEN

        IF(NCHAR.GT.0) THEN

          DO I=1,NCHAR

            II=ICBON1(I)
            JJ=ICBON2(I) 

            IF(QCBON(SELSU,I)) THEN
              REQ=RCBON(SELSU,I)
              FCON=KCBON(SELSU,I)
              E=UBOND(II,JJ,FCON,REQ,X,Y,Z,NATOMX)
              EBND1=EBND1+E
            ENDIF 

            IF(QCBON(LOWSU,I)) THEN
              REQ=RCBON(LOWSU,I)
              FCON=KCBON(LOWSU,I)
              E=UBOND(II,JJ,FCON,REQ,X,Y,Z,NATOMX)
              EBND2=EBND2+E
            ENDIF 

          ENDDO

        ENDIF

        IF(NCMOR.GT.0) THEN

          DO I=1,NCMOR

            II=ICBON1(I+NCHAR)
            JJ=ICBON2(I+NCHAR) 

            IF(QCBON(SELSU,I+NCHAR)) THEN
              REQ=RCMOR(SELSU,I)
              DE=DCMOR(SELSU,I)
              BETA=BCMOR(SELSU,I) 
              E=MORSE(II,JJ,DE,BETA,REQ,X,Y,Z,NATOMX)
              EBND1=EBND1+E
            ENDIF 

            IF(QCBON(LOWSU,I+NCHAR)) THEN
              REQ=RCMOR(LOWSU,I)
              DE=DCMOR(LOWSU,I)
              BETA=BCMOR(LOWSU,I)
              E=MORSE(II,JJ,DE,BETA,REQ,X,Y,Z,NATOMX)
              EBND2=EBND2+E
            ENDIF 

          ENDDO

        ENDIF

      ENDIF

      EBOUN1=EBOUN1+EBND1
      EBOUN2=EBOUN2+EBND2

      DEBND=EBND1-EBND2

      ! End bonds----------------------------------------------   

      !--------------------------------------------------------
      ! Angles
      !--------------------------------------------------------

      EANG1=ZERO
      EANG2=ZERO

      IF((NCANG.GT.0).AND.QETERM(ANGLE)) THEN

        DO I=1,NCANG

          II=ICANG1(I)
          JJ=ICANG2(I)
          KK=ICANG3(I)

          IF(QCANG(SELSU,I)) THEN
            TEQ=TCANG(SELSU,I)
            FCON=KCANG(SELSU,I)
            E=UANGLE(II,JJ,KK,FCON,TEQ,X,Y,Z,NATOMX)
            EANG1=EANG1+E
          ENDIF

          IF(QCANG(LOWSU,I)) THEN
            TEQ=TCANG(LOWSU,I)
            FCON=KCANG(LOWSU,I)
            E=UANGLE(II,JJ,KK,FCON,TEQ,X,Y,Z,NATOMX)
            EANG2=EANG2+E
          ENDIF

        ENDDO

      ENDIF

      EBOUN1=EBOUN1+EANG1
      EBOUN2=EBOUN2+EANG2

      DEANG=EANG1-EANG2
   
      ! End angles----------------------------------------------

      !---------------------------------------------------------
      ! Dihedrals
      !---------------------------------------------------------

      EDIH1=ZERO
      EDIH2=ZERO

      IF((NCDIH.GT.0).AND.QETERM(DIHE)) THEN

        DO I=1,NCDIH

          II=ICDIH1(I)
          JJ=ICDIH2(I)
          KK=ICDIH3(I)
          LL=ICDIH4(I)

          IF(QCDIH(SELSU,I)) THEN
            PEQ=PCDIH(SELSU,I)
            FCON=KCDIH(SELSU,I)
            MULT=MCDIH(SELSU,I)
            E=DIHD(II,JJ,KK,LL,FCON,PEQ,MULT,X,Y,Z,NATOMX)
            EDIH1=EDIH1+E
          ENDIF 

          IF(QCDIH(LOWSU,I)) THEN
            PEQ=PCDIH(LOWSU,I)
            FCON=KCDIH(LOWSU,I)
            MULT=MCDIH(LOWSU,I)
            E=DIHD(II,JJ,KK,LL,FCON,PEQ,MULT,X,Y,Z,NATOMX)
            EDIH2=EDIH2+E
          ENDIF 

        ENDDO

      ENDIF

      EBOUN1=EBOUN1+EDIH1
      EBOUN2=EBOUN2+EDIH2

      DEDIH=EDIH1-EDIH2 

      DEBOUN=EBOUN1-EBOUN2  
      ! End dihedrals-------------------------------------------

      ! END BONDED
      !=========================================================

      !=========================================================
      ! NON-BONDED        
      !

      ENBND1=ZERO
      ENBND2=ZERO

      !---------------------------------------------------------
      ! Zero everything
      !---------------------------------------------------------

      EVDW1=ZERO
      EVDW2=ZERO
      EEL1=ZERO
      EEL2=ZERO
      EEXV1=ZERO
      EEXE1=ZERO
      E14V1=ZERO
      E14E1=ZERO
      EEXV2=ZERO
      EEXE2=ZERO
      E14V2=ZERO
      E14E2=ZERO
      EEXV01=ZERO
      EEXE01=ZERO
      EEXV02=ZERO
      EEXE02=ZERO
      E14V01=ZERO
      E14E01=ZERO
      E14V02=ZERO
      E14E02=ZERO
    !  E3P1=ZERO
    !  E3P2=ZERO
    !  E3P=ZERO

      EINC1=ZERO
      EINV1=ZERO
      EINE1=ZERO
      EIV141=ZERO
      EIE141=ZERO
      EINC2=ZERO
      EINV2=ZERO
      EINE2=ZERO
      EIV142=ZERO
      EIE142=ZERO


      IF(QETERM(ELEC).OR.QETERM(VDW)) THEN 

        !----------------------------------------------------------
        ! precalculating some switching and shifting stuff  
        !----------------------------------------------------------
        C2OFNB=CTOFNB*CTOFNB
        C2ONNB=CTONNB*CTONNB
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        CTOFNB1=ONE/CTOFNB
        CTOFNB2=CTOFNB1*CTOFNB1
        CTOFNB4=CTOFNB2*CTOFNB2

        !----------------------------------------------------------
        ! Main Nonbonded Loop
        !----------------------------------------------------------

        ! Stuff listed in ATOM section
        IF((NCRAT.GT.0).AND.(QETERM(ELEC).OR.QETERM(VDW))) THEN

          DO I=1,NCRAT

            SIG11=VCRAD(SELSU,I)
            EFF11=VCEPS(SELSU,I)
            SIG21=VCRAD(LOWSU,I)
            EFF21=VCEPS(LOWSU,I)
            II=ICATOM(I)
            Q11=ECCHG(SELSU,I)
            Q21=ECCHG(LOWSU,I)

            DO J=1,NATOMX

              IF((II.NE.J).AND.(ICINDX(J).LT.I)) THEN

                RX=X(II)-X(J)
                RY=Y(II)-Y(J)
                RZ=Z(II)-Z(J)

                S=RX*RX+RY*RY+RZ*RZ

                IF (S.LT.C2OFNB) THEN

                  IF(ICINDX(J).GT.0) THEN
                    SIG12=VCRAD(SELSU,ICINDX(J))
                    EFF12=VCEPS(SELSU,ICINDX(J)) 
                    SIG22=VCRAD(LOWSU,ICINDX(J))
                    EFF22=VCEPS(LOWSU,ICINDX(J))
                    Q12=ECCHG(SELSU,ICINDX(J))
                    Q22=ECCHG(LOWSU,ICINDX(J))
                  ELSE
                    SIG12=VDWR(ITC(IAC(J)))
                    EFF12=EFF(ITC(IAC(J)))
                    SIG22=SIG12
                    EFF22=EFF12
                    Q12=CG(J)
                    Q22=Q12 
                  ENDIF

                  SIGQ1=(SIG11+SIG12)**2
                  EFF1=SQRT(EFF11*EFF12)
                  SIGQ2=(SIG21+SIG22)**2
                  EFF2=SQRT(EFF21*EFF22) 
                  R=SQRT(S)

                  !---------------------------------------------
                  ! VdW 
                  !---------------------------------------------

                  IF(QETERM(VDW)) THEN 

                    E1=UVDW(SIGQ1,EFF1,R) 
                    E2=UVDW(SIGQ2,EFF2,R)
               
                    !  Switching function stuff

                    IF (S.GT.C2ONNB) THEN
                      RIJL=C2ONNB-S
                      RIJU=C2OFNB-S
                      FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                      E1=(FUNCT*E1)
                      E2=(FUNCT*E2) 
                    ENDIF
                    EVDW1=EVDW1+E1
                    EVDW2=EVDW2+E2
                  ENDIF 
                  ! End Van der Waal----------------------------

                  !---------------------------------------------
                  ! Coulomb 
                  !---------------------------------------------   

                  ! Shifting function

                  IF(QETERM(ELEC)) THEN 
                    DR4=S*S
                    FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4

                    ! Calculate electrostatic potential

                    E1=UELEC(Q11,Q12,R)*FUNCT
                    E2=UELEC(Q21,Q22,R)*FUNCT

                    EEL1=EEL1+E1
                    EEL2=EEL2+E2
                  ENDIF 

                ENDIF
              ENDIF
            ENDDO
          ENDDO
        ENDIF

        ! End ATOM section stuff
        !-------------------------------------------------------

        ! End Coulomb part--------------------------------------


        !-------------------------------------------------------
        ! Exclusions and 1-4 interactions
        !-------------------------------------------------------

        ! Surface 1 --------------------------------------------

        !-------------------------------------------------------
        ! Exclusions
        !-------------------------------------------------------

        IF((NCEXC(SELSU).GT.0).AND.(QETERM(ELEC).OR.QETERM(VDW))) THEN

          DO I=1,NCEXC(SELSU)

            II=ICEXC(SELSU,I,1)
            JJ=ICEXC(SELSU,I,2)
         
            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              SIGQ1=(SIG11+SIG12)**2
              EFF1=SQRT(EFF11*EFF12)
              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                EEXV1=EEXV1+E1

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT
                EEXE1=EEXE1+E1
              ENDIF

            ENDIF
          ENDDO 
        ENDIF


        !-----------------------------------------------------
        ! 1-2 & 1-3 Inclusions
        !-----------------------------------------------------
        IF(NCINC(SELSU).GT.0) THEN

          DO I=1,NCINC(SELSU)

            II=ICINC(SELSU,I,1)
            JJ=ICINC(SELSU,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              SIGQ1=(SIG11+SIG12)**2
              EFF1=SQRT(EFF11*EFF12)
              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                EINV1=EINV1+E1

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT
                EINE1=EINE1+E1
              ENDIF

            ENDIF
          ENDDO 
        ENDIF


        !----------------------------------------------------
        ! 1-4 interactions
        !----------------------------------------------------

        ! Exclusions

        IF((NBXMOD.GE.4).AND.(NC14(SELSU).GT.0)) THEN

          DO I=1,NC14(SELSU)

            II=IC14(SELSU,I,1)
            JJ=IC14(SELSU,I,2)
         
            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
              SIG21=SIG11
              EFF21=EFF11  
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG21=VDWR(ITC(IAC(II))+MAXATC)
                EFF21=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
              SIG22=SIG12
              EFF22=EFF12  
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG22=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)
             
                E1=UVDW(SIGQ1,EFF1,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ2=(SIG21+SIG22)**2
                  EFF2=SQRT(EFF21*EFF22)
                  E2=UVDW(SIGQ2,EFF2,R)
                ENDIF
          
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  IF (NBXMOD.EQ.5) E2=FUNCT*E2
                ENDIF

                E14V1=E14V1+E1

                IF (NBXMOD.EQ.5) E14V1=E14V1-E2

              ENDIF 

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT
                  E14E1=E14E1+E1
                ENDIF     

              ENDIF
            ENDIF
          ENDDO 
        ENDIF


        ! 1-4 inclusions

        IF((NBXMOD.GE.4).AND.(NCIN14(SELSU).GT.0)) THEN

          DO I=1,NCIN14(SELSU)

            II=ICIN14(SELSU,I,1)
            JJ=ICIN14(SELSU,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
              SIG21=SIG11
              EFF21=EFF11  
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG21=VDWR(ITC(IAC(II))+MAXATC)
                EFF21=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q11=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
              SIG22=SIG12
              EFF22=EFF12  
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG22=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q12=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ2=(SIG21+SIG22)**2
                  EFF2=SQRT(EFF21*EFF22)
                  E2=UVDW(SIGQ2,EFF2,R)
                ENDIF

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  IF (NBXMOD.EQ.5) E2=FUNCT*E2
                ENDIF

                EIV141=EIV141+E1

                IF (NBXMOD.EQ.5) EIV141=EIV141-E2
              ENDIF 

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT
                  EIE141=EIE141+E1
                ENDIF

              ENDIF
            ENDIF
          ENDDO 
        ENDIF

        ! End Surface 1 ex- & inclusions and 1-4 interactions
 

  ! Lowest surface ------------------------------------------
        !----------------------------------------------------
        ! Exclusions
        !----------------------------------------------------
        IF(NCEXC(LOWSU).GT.0) THEN

          DO I=1,NCEXC(LOWSU)

            II=ICEXC(LOWSU,I,1)
            JJ=ICEXC(LOWSU,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
            ELSE
              SIG21=VDWR(ITC(IAC(II)))
              EFF21=EFF(ITC(IAC(II)))
              Q21=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
            ELSE
              SIG22=VDWR(ITC(IAC(JJ)))
              EFF22=EFF(ITC(IAC(JJ)))
              Q22=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN 

                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E2=UVDW(SIGQ2,EFF2,R) 

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E2=(FUNCT*E2)
                ENDIF

                EEXV2=EEXV2+E2

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E2=UELEC(Q21,Q22,R)*FUNCT
                EEXE2=EEXE2+E2     
              ENDIF

            ENDIF
          ENDDO 
        ENDIF

        !-----------------------------------------------------
        ! 1-2 & 1-3 Inclusions
        !-----------------------------------------------------
        IF(NCINC(LOWSU).GT.0) THEN

          DO I=1,NCINC(LOWSU)

            II=ICINC(LOWSU,I,1)
            JJ=ICINC(LOWSU,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
            ELSE
              SIG21=VDWR(ITC(IAC(II)))
              EFF21=EFF(ITC(IAC(II)))
              Q21=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
            ELSE
              SIG22=VDWR(ITC(IAC(JJ)))
              EFF22=EFF(ITC(IAC(JJ)))
              Q22=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN 

                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E2=UVDW(SIGQ2,EFF2,R) 

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E2=(FUNCT*E2)
                ENDIF

                EINV2=EINV2+E2

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E2=UELEC(Q21,Q22,R)*FUNCT
                EINE2=EINE2+E2
              ENDIF

            ENDIF
          ENDDO 
        ENDIF

        !------------------------------------------------------
        ! 1-4 interactions
        !------------------------------------------------------
        IF((NBXMOD.GE.4).AND.(NC14(LOWSU).GT.0)) THEN

          DO I=1,NC14(LOWSU)

            II=IC14(LOWSU,I,1)
            JJ=IC14(LOWSU,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
              SIG11=SIG21
              EFF11=EFF21
            ELSE
              SIG21=VDWR(ITC(IAC(II)))
              EFF21=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG11=VDWR(ITC(IAC(II))+MAXATC)
                EFF11=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q21=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
              SIG12=SIG22
              EFF12=EFF22
            ELSE
              SIG22=VDWR(ITC(IAC(JJ)))
              EFF22=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG12=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF12=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q22=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN
                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E2=UVDW(SIGQ2,EFF2,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ1=(SIG11+SIG12)**2
                  EFF1=SQRT(EFF11*EFF12)
                  E1=UVDW(SIGQ1,EFF1,R)
                ENDIF
          
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E2=(FUNCT*E2)

                  IF (NBXMOD.EQ.5) E1=FUNCT*E1

                ENDIF

                E14V2=E14V2+E2
                IF (NBXMOD.EQ.5) E14V2=E14V2-E1

              ENDIF 

              IF (QETERM(ELEC)) THEN   

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E2=UELEC(Q21,Q22,R)*FUNCT
                  E14E2=E14E2+E2
                ENDIF

              ENDIF     
            ENDIF
          ENDDO 
        ENDIF

        !----------------------------------------------------
        ! 1-4 inclusions
        !----------------------------------------------------
        IF((NBXMOD.GE.4).AND.(NCIN14(LOWSU).GT.0)) THEN
          DO I=1,NCIN14(LOWSU)

            II=ICIN14(LOWSU,I,1)
            JJ=ICIN14(LOWSU,I,2)
         
            IF(ICINDX(II).GT.0) THEN
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
              SIG11=SIG21
              EFF11=EFF21
            ELSE
              SIG21=VDWR(ITC(IAC(II)))
              EFF21=EFF(ITC(IAC(II)))

              IF (NBXMOD.EQ.5) THEN
                SIG11=VDWR(ITC(IAC(II))+MAXATC)
                EFF11=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              Q21=CG(II)
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
              SIG12=SIG22
              EFF12=EFF22
            ELSE
              SIG22=VDWR(ITC(IAC(JJ)))
              EFF22=EFF(ITC(IAC(JJ)))

              IF (NBXMOD.EQ.5) THEN
                SIG12=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF12=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              Q22=CG(JJ)
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN
                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E2=UVDW(SIGQ2,EFF2,R) 

                IF (NBXMOD.EQ.5) THEN
                  SIGQ1=(SIG11+SIG12)**2
                  EFF1=SQRT(EFF11*EFF12)
                  E1=UVDW(SIGQ1,EFF1,R)
                ENDIF

                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E2=(FUNCT*E2)

                  IF (NBXMOD.EQ.5) E1=FUNCT*E1

                ENDIF

                EIV142=EIV142+E2

                IF (NBXMOD.EQ.5) EIV142=EIV142-E1


              ENDIF 

              IF (QETERM(ELEC)) THEN   

                IF (NBXMOD.GE.4) THEN 
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E2=UELEC(Q21,Q22,R)*FUNCT
                  EIE142=EIE142+E2
                ENDIF

              ENDIF
            ENDIF
          ENDDO 
        ENDIF

  ! End lower surface exclusions and 1-4 interactions-----

  ! Surface 0 (both)--------------------------------------

        !-------------------------------------------------
        ! Exclusions
        !-------------------------------------------------
        IF(NCEXC(NCRSU+1).GT.0) THEN

          DO I=1,NCEXC(NCRSU+1)

            II=ICEXC(NCRSU+1,I,1)
            JJ=ICEXC(NCRSU+1,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
              SIG21=SIG11
              EFF21=EFF11
              Q21=Q11  
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
              SIG22=SIG12
              EFF22=EFF12
              Q22=Q12
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)
                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E1=UVDW(SIGQ1,EFF1,R) 
                E2=UVDW(SIGQ2,EFF2,R)

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  E2=(FUNCT*E2)
                ENDIF

                EEXV01=EEXV01+E1
                EEXV02=EEXV02+E2

              ENDIF

              IF(QETERM(ELEC)) THEN  
                DR4=S*S
                FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                E1=UELEC(Q11,Q12,R)*FUNCT
                E2=UELEC(Q21,Q22,R)*FUNCT 
                EEXE01=EEXE01+E1
                EEXE02=EEXE02+E2  
              ENDIF    

            ENDIF
          ENDDO
        ENDIF
 
        !--------------------------------------------------------
        ! 1-4 interactions
        !--------------------------------------------------------
        IF((NBXMOD.GE.4).AND.(NC14(NCRSU+1).GT.0)) THEN

          DO I=1,NC14(NCRSU+1)

            II=IC14(NCRSU+1,I,1)
            JJ=IC14(NCRSU+1,I,2)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(SELSU,ICINDX(II))
              EFF11=VCEPS(SELSU,ICINDX(II))
              Q11=ECCHG(SELSU,ICINDX(II))
              SIG112=SIG11
              EFF112=EFF11
              SIG21=VCRAD(LOWSU,ICINDX(II))
              EFF21=VCEPS(LOWSU,ICINDX(II))
              Q21=ECCHG(LOWSU,ICINDX(II))
              SIG212=SIG21
              EFF212=EFF21  
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              Q11=CG(II)
              SIG112=VDWR(ITC(IAC(II))+MAXATC)
              EFF112=EFF(ITC(IAC(II))+MAXATC)
              SIG21=SIG11
              EFF21=EFF11
              Q21=Q11  
              SIG212=SIG112
              EFF212=EFF112
            ENDIF

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(SELSU,ICINDX(JJ))
              EFF12=VCEPS(SELSU,ICINDX(JJ))
              Q12=ECCHG(SELSU,ICINDX(JJ))
              SIG122=SIG12
              EFF122=EFF12
              SIG22=VCRAD(LOWSU,ICINDX(JJ))
              EFF22=VCEPS(LOWSU,ICINDX(JJ))
              Q22=ECCHG(LOWSU,ICINDX(JJ))
              SIG222=SIG22
              EFF222=EFF22 
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              Q12=CG(JJ)
              SIG122=VDWR(ITC(IAC(JJ))+MAXATC)
              EFF122=EFF(ITC(IAC(JJ))+MAXATC)
              SIG22=SIG12
              EFF22=EFF12
              Q22=Q12
              SIG222=SIG122
              EFF222=EFF122
            ENDIF

            RX=X(II)-X(JJ)
            RY=Y(II)-Y(JJ)
            RZ=Z(II)-Z(JJ)

            S=RX*RX+RY*RY+RZ*RZ

            IF (S.LT.C2OFNB) THEN

              R=SQRT(S)

              IF(QETERM(VDW)) THEN

                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)
                SIGQ2=(SIG21+SIG22)**2
                EFF2=SQRT(EFF21*EFF22)
                E1=UVDW(SIGQ1,EFF1,R) 
                E2=UVDW(SIGQ2,EFF2,R)

                IF (NBXMOD.EQ.5) THEN
                  SIGQ12=(SIG112+SIG122)**2
                  EPS1=SQRT(EFF112*EFF122)
                  SIGQ22=(SIG212+SIG222)**2
                  EPS2=SQRT(EFF212*EFF222)
                  E12=UVDW(SIGQ12,EPS1,R) 
                  E22=UVDW(SIGQ22,EPS2,R)
                ENDIF

                IF (S.GT.C2ONNB) THEN
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                  E2=(FUNCT*E2)

                  IF(NBXMOD.EQ.5) THEN
                    E12=(FUNCT*E12)
                    E22=(FUNCT*E22)
                  ENDIF  

                ENDIF

                E14V01=E14V01+E1
                E14V02=E14V02+E2

                IF(NBXMOD.EQ.5) THEN 
                  E14V01=E14V01-E12
                  E14V02=E14V02-E22
                ENDIF

              ENDIF

              IF(QETERM(ELEC)) THEN

                IF (NBXMOD.GE.4) THEN
                  DR4=S*S
                  FUNCT=1.0_chm_real-2.0_chm_real*S*CTOFNB2+DR4*CTOFNB4
                  E1=UELEC(Q11,Q12,R)*FUNCT
                  E2=UELEC(Q21,Q22,R)*FUNCT 
                  E14E01=E14E01+E1
                  E14E02=E14E02+E2
                ENDIF

              ENDIF 
            ENDIF
          ENDDO
        ENDIF
  ! End Surface 0 exclusions---------------------------------

        !----------------------------------------------------
        ! When BFDVDW was requested
        !----------------------------------------------------
        IF(BFDVDW.AND.QETERM(VDW)) THEN
          ERVDW1=ZERO
          ERVDW2=ZERO
          DO I=1,NBVDW
            IF (BVDWSURF(I,SELSU)) THEN
              II=BVDWEXC1(I)
              JJ=BVDWEXC2(I)

              IF(ICINDX(II).GT.0) THEN
                SIG11=VCRAD(SELSU,ICINDX(II))
                EFF11=VCEPS(SELSU,ICINDX(II))
              ELSE
                SIG11=VDWR(ITC(IAC(II)))
                EFF11=EFF(ITC(IAC(II)))
              ENDIF

              IF(ICINDX(JJ).GT.0) THEN
                SIG12=VCRAD(SELSU,ICINDX(JJ))
                EFF12=VCEPS(SELSU,ICINDX(JJ))
              ELSE
                SIG12=VDWR(ITC(IAC(JJ)))
                EFF12=EFF(ITC(IAC(JJ)))
              ENDIF

              RX=X(II)-X(JJ)
              RY=Y(II)-Y(JJ)
              RZ=Z(II)-Z(JJ)

              S=RX*RX+RY*RY+RZ*RZ

              IF (S.LT.C2OFNB) THEN

                ! ... Remove 1-2 CHARMM LJ interaction
                R=SQRT(S)
                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E1=UVDW(SIGQ1,EFF1,R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                ERVDW1=ERVDW1-E1

                ! ... Add user defined LJ interaction
                E1=URVDW(BVDWSIG(I)**2.0_CHM_REAL,BVDWEPS(I), &
                         REPP(I),ATTP(I),R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-(ATTP(I)/2.0_CHM_REAL)*RIJL)*RUL3
                  E1=(FUNCT*E1)
                ENDIF

                ERVDW1=ERVDW1+E1
              ENDIF

            ENDIF


            ! ... BFDVDW on LOWSU
            IF (BVDWSURF(I,LOWSU)) THEN
              II=BVDWEXC1(I)
              JJ=BVDWEXC2(I)

              IF(ICINDX(II).GT.0) THEN
                SIG11=VCRAD(LOWSU,ICINDX(II))
                EFF11=VCEPS(LOWSU,ICINDX(II))
              ELSE
                SIG11=VDWR(ITC(IAC(II)))
                EFF11=EFF(ITC(IAC(II)))
              ENDIF

              IF(ICINDX(JJ).GT.0) THEN
                SIG12=VCRAD(LOWSU,ICINDX(JJ))
                EFF12=VCEPS(LOWSU,ICINDX(JJ))
              ELSE
                SIG12=VDWR(ITC(IAC(JJ)))
                EFF12=EFF(ITC(IAC(JJ)))
              ENDIF

              RX=X(II)-X(JJ)
              RY=Y(II)-Y(JJ)
              RZ=Z(II)-Z(JJ)

              S=RX*RX+RY*RY+RZ*RZ

              IF (S.LT.C2OFNB) THEN

                ! ... Remove 1-2 CHARMM LJ interaction
                R=SQRT(S)
                SIGQ1=(SIG11+SIG12)**2
                EFF1=SQRT(EFF11*EFF12)

                E2=UVDW(SIGQ1,EFF1,R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                  E2=(FUNCT*E2)
                ENDIF

                ERVDW2=ERVDW2-E2

                ! ... Add user defined LJ interaction
                IF (I.LE.NCHAR) THEN
                  SIG11=RCBON(LOWSU,I)/2.0_CHM_REAL
                  SIGQ1=RCBON(LOWSU,I)**2
                  EFF11=KCBON(LOWSU,I)
                  EFF1=ABS(EFF11)
                ELSE
                  SIG11=RCMOR(LOWSU,I-NCHAR)/2.0_CHM_REAL
                  SIGQ1=RCMOR(LOWSU,I-NCHAR)**2
                  EFF11=DCMOR(LOWSU,I-NCHAR)
                  EFF1=ABS(EFF11)
                ENDIF


                E2=URVDW(BVDWSIG(I)**2.0_CHM_REAL,BVDWEPS(I), &
                         REPP(I),ATTP(I),R)

                ! ... Switching function
                IF (S.GT.C2ONNB) THEN 
                  RIJL=C2ONNB-S
                  RIJU=C2OFNB-S
                  FUNCT=RIJU*RIJU*(RIJU-(ATTP(I)/2.0_CHM_REAL)*RIJL)*RUL3
                  E2=(FUNCT*E2)
                ENDIF

                ERVDW2=ERVDW2+E2
              ENDIF

            ENDIF
          ENDDO 
        ENDIF

        ! End of special BFDVDW treatment --------------------------

      ENDIF

      EEXCL1=EEXV1+EEXE1+E14E1+E14V1+EEXV01+EEXE01+E14V01+E14E01
      EEXCL2=EEXV2+EEXE2+E14E2+E14V2+EEXV02+EEXE02+E14V02+E14E02

      EINC1=EINV1+EINE1+EIV141+EIE141
      EINC2=EINV2+EINE2+EIV142+EIE142

      ! END EX- & INCLUSIONS -----------------------------------------

      EEL1 =EEL1-EEXE1-E14E1-EEXE01-E14E01+EINE1+EIE141
      EVDW1=EVDW1-EEXV1-E14V1-EEXV01-E14V01+EINV1+EIV141+ERVDW1

      EEL2=EEL2-EEXE2-E14E2-EEXE02-E14E02+EINE2+EIE142
      EVDW2=EVDW2-EEXV2-E14V2-EEXV02-E14V02+EINV2+EIV142+ERVDW2

      DEEL=EEL1-EEL2
      DEVDW=EVDW1-EVDW2
      DEEXCL=EEXCL1-EEXCL2

      ENBND1=EEL1+EVDW1
      ENBND2=EEL2+EVDW2
 
      DENBND=ENBND1-ENBND2
 
  ! END NON-BONDED
  !=============================================================

  !=============================================================
  ! Add it all up
  !=============================================================

      ! ETOT1=EBOUN1+ENBND1
      ! ETOT2=EBOUN2+ENBND2

      ! Add absolute surface shifts
      IF (SELSU.EQ.1) THEN
        ETOT1=EBOUN1+ENBND1
        ETOT2=EBOUN2+ENBND2-SSHIFT2
        DELSHIF=SSHIFT2

      ELSEIF (LOWSU.EQ.1) THEN
        ETOT1=EBOUN1+ENBND1-SSHIFT1
        ETOT2=EBOUN2+ENBND2
        DELSHIF=-SSHIFT1

      ELSE
        ETOT1=EBOUN1+ENBND1-SSHIFT1
        ETOT2=EBOUN2+ENBND2-SSHIFT2
        DELSHIF=SSHIFT1-SSHIFT2
      ENDIF

      ED = ETOT1-ETOT2

  ! End final adding----------------------------------------------
  !===============================================================

  !===============================================================
  ! Debug writeout of different terms
      IF(PRNLEV.GE.6) THEN
     !  WRITE(OUTU,10) EEXCL1,EEXCL2,DEEXCL
        WRITE(OUTU,5) 
        WRITE(OUTU,10) SELSU,LOWSU 
        WRITE(OUTU,20) EEL1,EEL2,DEEL
        WRITE(OUTU,30) EVDW1,EVDW2,DEVDW
        WRITE(OUTU,40) EBOUN1,EBOUN2,DEBOUN
        WRITE(OUTU,50) ENBND1,ENBND2,DENBND
        WRITE(OUTU,70) SSHIFT1,SSHIFT2,DELSHIF
        WRITE(OUTU,60) ETOT1,ETOT2,ED

        WRITE(OUTU,5)
      ENDIF
  !===============================================================
5     FORMAT(' EDELTA>=====================================')
10    FORMAT(' EDELTA> Energy comparison between surfaces ', &
              I2,' and ',I2,':')
  ! 10   FORMAT('EDELTA> ex-tot: Excl1  Excl2 De: ',3F10.3)  
20    FORMAT(' EDELTA> el-tot:  Eel1  Eell2 De: ',3F10.3)
30    FORMAT(' EDELTA> evdw-tot: Ev1   Ev2  De: ',3F10.3)
40    FORMAT(' EDELTA> eboun-tot: Eb1  Eb2  De: ',3F10.3)
50    FORMAT(' EDELTA> enbon-tot: Enb1 Enb2 De: ',3F10.3)
60    FORMAT(' EDELTA> e-tot:     E1   E2   De: ',3F10.3)
70    FORMAT(' EDELTA> e-shift:  Esh1  Esh2 De: ',3F10.3)

    RETURN
  END SUBROUTINE EDELTA

  !===================================================================
  ! HELP FUNCTIONS FOR ENERGY EVALUATION
  !===================================================================

  real(chm_real) FUNCTION DISTANCE(XI,YI,ZI,XJ,YJ,ZJ)
    !    
    !     The distance between atoms I and J
    !
    !     Oren Becker Feb-26-1992
    !----------------------------------------------------------------- 
    !
    !MM##FORTRAN

    ! passed variables
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ
    ! local variables
    real(chm_real) RX, RY, RZ, R2, R

    RX = XI - XJ
    RY = YI - YJ
    RZ = ZI - ZJ

    R2 = RX*RX + RY*RY + RZ*RZ
    R = SQRT(R2)

    DISTANCE = R

    RETURN
  END FUNCTION DISTANCE


  real(chm_real) FUNCTION  UELEC(Q1,Q2,R)
    !     
    !     Electrostatic interaction between two point charges Q1 and Q1 at
    !     a distance R.
    !
    !     Oren Becker Feb-27-1992
    !--------------------------------------------------------------------- 
    !
    use number

    ! passed variables
    real(chm_real) Q1, Q2, R
    ! local variables
    real(chm_real) EPS, CONST, RT, E

    EPS = one
    CONST = 332.0716_chm_real/EPS
    RT = one/R

    E = CONST*Q1*Q2*RT
      
    UELEC = E
    RETURN
  END FUNCTION UELEC

  !===================================================================

  real(chm_real) FUNCTION UVDW(Rm2,Emin,R)
    !     
    !     VDW interaction
    !
    !     Oren Becker Feb-27-1992
    !--------------------------------------------------------------------- 
    !
    use number

    ! passed variables
    real(chm_real) Rm2, Emin, R
    ! local variables
    real(chm_real) R2,RR,RR6,A,E 

    R2 = R*R
    RR = Rm2/R2
    RR6 = RR*RR*RR
    A = RR6*RR6 - TWO*RR6
    E = Emin*A
      
    UVDW = E
    RETURN
  END FUNCTION UVDW

  !===================================================================

  real(chm_real) FUNCTION URVDW(Rm2,Emin,RRREP,RRATT,R)
    !     
    !     User modified VDW interaction
    !
    !     SL 090705
    !--------------------------------------------------------------------- 
    !
    use number

    ! passed variables
    real(chm_real) Rm2,Emin,RRREP,RRATT,R
    ! local variables
    real(chm_real) R2,RR,A,E 

    R2 = R*R
    RR = SQRT(Rm2/R2)
    A = RR**RRREP - TWO*(RR**RRATT)
    E = Emin*A
      
    URVDW = E
    RETURN
  END FUNCTION URVDW

  !====================================================================

  real(chm_real) FUNCTION UANGLE(I,J,K,kTH,THeq,X,Y,Z,NATOMX)
    !                      result(E)

    !     
    !     The energy of an harmonic angle
    !
    !     Oren Becker Feb-14-1992
    !--------------------------------------------------------------------- 
    !
    use number
    use dimens_fcm
    !
    ! passed variables
    INTEGER,intent(in) :: NATOMX, I,J,K
    REAL(chm_real),intent(in) :: X(NATOMX),Y(NATOMX),Z(NATOMX), &
                                 kTH, THeq

    real(chm_real) DXI, DYI, DZI, RI2, RI, RIR, DXIR, DYIR, DZIR
    real(chm_real) DXJ, DYJ, DZJ, RJ2, RJ, RJR, DXJR, DYJR, DZJR


    ! local variables
    real(chm_real) CST, AT, DA

    DXI = X(I) - X(J)
    DYI = Y(I) - Y(J)
    DZI = Z(I) - Z(J)
    DXJ = X(K) - X(J)
    DYJ = Y(K) - Y(J)
    DZJ = Z(K) - Z(J)

    RI2 = DXI*DXI + DYI*DYI + DZI*DZI
    RJ2 = DXJ*DXJ + DYJ*DYJ + DZJ*DZJ
    RI = SQRT(RI2)
    RJ = SQRT(RJ2)
  
    RIR = one/RI
    RJR = one/RJ

    DXIR = DXI*RIR
    DYIR = DYI*RIR
    DZIR = DZI*RIR
    DXJR = DXJ*RJR
    DYJR = DYJ*RJR
    DZJR = DZJ*RJR
      
    CST = DXIR*DXJR + DYIR*DYJR + DZIR*DZJR
    AT = ACOS(CST)

    DA = AT - THeq
    UANGLE = kTH*DA*DA
      
    RETURN
  END FUNCTION UANGLE

  !===================================================================

  real(chm_real) FUNCTION UBOND(I,J,FCON,REQ,X,Y,Z,NATOMX)
    !                      result(E)

    !
    !     The energy of an harmonic bond
    !
    !     JD 060105
    !
    use dimens_fcm

    ! passed variables
    INTEGER I,J,NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),FCON,REQ

    ! local variables
    real(chm_real) RX, RY, RZ, R2, R, DD

    RX = X(I) - X(J)
    RY = Y(I) - Y(J)
    RZ = Z(I) - Z(J)

    R2 = RX*RX + RY*RY + RZ*RZ
    R = SQRT(R2)


    DD= R-REQ
    UBOND=FCON*DD*DD


    RETURN
  END FUNCTION UBOND

  !================================================================

  real(chm_real) FUNCTION MORSE(I,J,DE,BETA,RMIN,X,Y,Z,NATOMX)
    !                      result(E)
    !     
    !     The energy of a Morse potential
    !
    !     Oren Becker Feb-14-1992
    !--------------------------------------------------------------------- 
    !
    use dimens_fcm
    use number
    !M##FORTRAN
    ! passed variables
    INTEGER NATOMX, I, J
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),DE, BETA, RMIN

    ! local variables
    real(chm_real) RX, RY, RZ, R2, R, Q, A, B
    !
    ! external functions
    !      real(kind=8) :: distance
    !      EXTERNAL DISTANCE

    ! local variables
    !      REAL*8 RX, RY, RZ, R2, R

    RX = X(I) - X(J)
    RY = Y(I) - Y(J)
    RZ = Z(I) - Z(J)

    R2 = RX*RX + RY*RY + RZ*RZ
    R = SQRT(R2)

    Q = R-RMIN
    B = MINONE*BETA*Q
    A = TWO*B
      
    MORSE = DE*(1.0_chm_real+EXP(A)-TWO*EXP(B))
    ! morse = E

    RETURN
  END FUNCTION MORSE

 
  !=================================================================

  real(chm_real) FUNCTION DIHD(I,J,K,L,FORCE,PHASE,IPER,X,Y,Z,NATOMX)
    !
    !     CALCULATES A TORSION ANGLE AND ITS ENERGY.
    !
    !     Oren Becker Feb-26-1992, adapted from EPHI By B.R. Brooks 1981
    !-------------------------------------------------------------------
    !     ENERGY TERMS FOR DIHEDRALS TERMS ARE EXPRESSED AS A FUNCTION
    !     OF COSINE PHI, AS OPPOSED TO EXPRESSING IT IN TERMS OF PHI
    !     THIS AVOIDS ALL PROBLEMS AS DIHEDRALS BECOME PLANAR.
    !     THE FUNCTIONAL FORMS ARE:
    !     E = K* (1.0 + PHASE* F(PERIODICITY,COS(PHI)) )
    !     WHERE
    !     F(1,C) = C
    !     F(2,C) = 2C**2 - 1
    !     F(3,C) = 4C**3 - 3C
    !     F(4,C) = 8C**4 - 8C**2 + 1
    !     F(5,C) = UNUSED
    !     F(6,C) = 32C**6 -48C**4 + 18C**2 -1
    !     AND  PHASE = 1.0 OR -1.0
    !
    !     By Bernard R. Brooks    1981
    !================================================
    !
    use number
    use consta
    use stream
    !
    ! passed variables
    INTEGER I,J,K,L,IPER,NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),FORCE,PHASE
    !MM##FORTRAN
    ! local variables
    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA,RB,RAR,RBR
    real(chm_real) AXR,AYR,AZR,BXR,BYR,BZR,CT,E,CT2,E1,ARG
    real(chm_real) RG2,RG,RGR,RABR,ST,CA,SA,AP
    !
    !MM      real(chm_real)  TWOPI,RXMIN,RMIN,RSMIN
    !MM      PARAMETER (TWOPI=TWO*PI)
    real(chm_real), PARAMETER :: RXMIN=0.005_CHM_REAL,RMIN=0.05_CHM_REAL,RSMIN=0.01_CHM_REAL
    !
    E=ZERO
    !----Bond Vectors
    FX=X(I)-X(J)
    FY=Y(I)-Y(J)
    FZ=Z(I)-Z(J)
    GX=X(J)-X(K)
    GY=Y(J)-Y(K)
    GZ=Z(J)-Z(K)
    HX=X(L)-X(K)
    HY=Y(L)-Y(K)
    HZ=Z(L)-Z(K)
    !-----A = F x G, B = H x G -----------------------------------C
    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX
    !
    RA2=AX*AX+AY*AY+AZ*AZ
    RB2=BX*BX+BY*BY+BZ*BZ
    RG2=GX*GX+GY*GY+GZ*GZ

    RG=DSQRT(RG2)
    RA=DSQRT(RA2)
    RB=DSQRT(RB2)

    RGR=ONE/RG

    !------Check for linear dihedrals----------------------------C
    IF(RA.LE.RXMIN) THEN
       WRITE(OUTU,20) I,J,K,L
20     FORMAT(' DIHD: WARNING.  dihedral is almost linear.'/ &
            ' derivatives may be affected for atoms:',4I5)
       RAR=SQRT(GX*GX+GY*GY+GZ*GZ)
       IF(RAR.LT.RMIN) GOTO 160
       IF(RA/RAR.LT.RMIN) GOTO 160
    ENDIF
    RAR=ONE/RA
    IF(RB.LE.RXMIN) THEN
       WRITE(OUTU,20) I,J,K,L
       RBR=SQRT(GX*GX+GY*GY+GZ*GZ)
       IF(RBR.LT.RMIN) GOTO 160
       IF(RB/RBR.LT.RMIN) GOTO 160
    ENDIF
    RBR=ONE/RB
    RABR=RAR*RBR
    !---Normalise A and B
    AXR=AX*RAR
    AYR=AY*RAR
    AZR=AZ*RAR
    BXR=BX*RBR
    BYR=BY*RBR
    BZR=BZ*RBR
    !
    CT=AXR*BXR+AYR*BYR+AZR*BZR
    !
    ST=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
    IF(IPER.NE.0) THEN
       IF (IPER.EQ.1) THEN
          E1=CT
       ELSE IF (IPER.EQ.2) THEN
          E1=TWO*CT*CT-ONE
       ELSE IF (IPER.EQ.3) THEN
          CT2=CT*CT
          E1=CT*(FOUR*CT2-THREE)
       ELSE IF (IPER.EQ.4) THEN
          CT2=CT*CT
          E1=ONE+CT2*EIGHT*(CT2-ONE)
       ELSE IF (IPER.EQ.6) THEN
          CT2=CT*CT
          E1=CT2*(CT2*(CT2*32._CHM_REAL-48._CHM_REAL)+18._CHM_REAL)-ONE
       ELSE IF (IPER.EQ.5) THEN
          CT2=CT*CT
          E1=CT*(CT2*(CT2*16._CHM_REAL-20._CHM_REAL)+FIVE)
       ELSE IF (IPER.EQ.0) THEN
          E1=ZERO
       ELSE
          WRITE(OUTU,*) I,J,K,L,IPER
          WRITE(OUTU,40) I,J,K,L,IPER
40        FORMAT(' BAD PERIOD: (I,J,K,L,IPER)',5I5)
          CALL WRNDIE(-3,'<DIHD>  ',  &
               'BAD PERIODICITY IN LIST FOR DIHEDRAL ANGLES')
       ENDIF
       !
       ARG=FORCE
       IF(PHASE.NE.ZERO) THEN
          ARG=-ARG
          IF(DABS(PI-PHASE).GT.PT01) THEN
             CALL WRNDIE(-3,'<DIHD>  ', &
                  'BAD PHASE IN LIST FOR DIHEDRAL ANGLES')
          ENDIF
       ENDIF

       E=E+FORCE+ARG*E1

    ELSE

       CA=CT*COS(PHASE)+ST*SIN(PHASE)
       SA=ST*COS(PHASE)-CT*SIN(PHASE)
       IF (CA.GT.PTONE ) THEN
          AP=ASIN(SA)
       ELSE
          AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
       ENDIF

       E=FORCE*AP*AP

    ENDIF

160 CONTINUE
    DIHD = E
    RETURN
  END FUNCTION DIHD


  !===============================================================
  ! DATA GATHERING AND CONTROL FOR SURFACE CROSSING                  
  !===============================================================

  SUBROUTINE CSTEP(X,Y,Z,DX,DY,DZ,NATOMX)
    !
    ! Subroutine which backs up the coordinate, force and velocity arrays
    ! for use in the surface crossing procedure
    !
    use number
    use stream
    use dimens_fcm
    use reawri
    ! Variables ----------------------------------------------

    INTEGER NATOMX, I,J,K,XINDEX,INDEXMO,INDEXMOO,INDEXPO
    INTEGER NATOM,NATOM2,NATOM3
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX), ED

    NATOM=NATOMX
    NATOM2=NATOM+NATOM
    NATOM3=NATOM2+NATOM

    !---------------------------------------------------------
    ! Recrossing:
    ! Updating the temp arrays must be done before calculating 
    ! energies (for _preceeding_ time step)
    ! since the arrays will be modified later.
    !--------------------------------------------------------        
    !

    IF(PRNLEV.GE.7) WRITE(OUTU,5)   

    ! Can we start looking for new crossings ?--------------
    IF ((NCRUN.GT.0) .AND. .NOT.QCROS .AND. &
         CROSSCOUNT .EQ. XRESTARTTIME) THEN

      IF(PRNLEV.GE.7) THEN 
        WRITE(OUTU,15) CROSSCOUNT
15      FORMAT(' CSTEP> Resetting QCROS at: ',I8)
      ENDIF

      QCROS = .TRUE. 
      XRESTARTTIME=0
    ENDIF

    !-------------------------------------------------------

    ! Have we done the final update of parameters ?----------
    IF(QCEND .AND. XCROSCOUNT.EQ.(2*XTIME)) THEN

      QCEND=.FALSE.

      ! IF(PRNLEV.GE.6) THEN  
      WRITE(OUTU,20) CROSSCOUNT  
20    FORMAT(' CSTEP> Resetting QCEND at: ',I8)
      ! ENDIF

    ENDIF

    !--------------------------------------------------------
    

    ! New crossing detected (at previous step)?--------------

    IF (QCFOUND) THEN

      QCON=.TRUE.
      K=1
      DO I=1, NCRSU-1
        DO J=I+1, NCRSU
          IF ((I.EQ.OLDSU).AND.(J.EQ.NEWSU)) THEN
            ICDIR(K)=1
          ELSEIF ((I.EQ.NEWSU).AND.(J.EQ.OLDSU)) THEN
            ICDIR(K)=-1
          ENDIF
          K=K+1
        ENDDO
      ENDDO

      !  IF(PRNLEV.GE.6) THEN 
      WRITE(OUTU,30) CROSSCOUNT
30    FORMAT(' CSTEP> Setting QCON at: ',I8)         
      !  ENDIF 

      QON(LOWSU)=.FALSE.

      !  IF(PRNLEV.GE.6) WRITE(OUTU,40) OLDSU, NEWSU
40    FORMAT(' CSTEP> Surface crossing from: ', &
             I2,' to ',I2,' detected')  

      QXTIME=.TRUE.
      QCFOUND=.FALSE.
      XSTRT=.TRUE.
      XCROSCOUNT = 0

    ENDIF

    ! End new crossing-----------------------------------------

    ! During crossing==========================================
    IF (QCROS .AND. QCON) THEN

    ! First step-----------------------------------------------
      IF (XCROSCOUNT .EQ. 0) THEN
         ! IF(PRNLEV.GE.6) THEN
        WRITE(OUTU,50) CROSSCOUNT
50      FORMAT(' CSTEP> First step: ',I8)  
        ! ENDIF    

        XINDEX=MOD(CROSSCOUNT,XTIME)+1
        INDEXMO=MOD(CROSSCOUNT+1,XTIME)+1
        INDEXPO=MOD(CROSSCOUNT-1,XTIME)+1

        ! Picking up saved data for recreating at T-XTIME-------------

        SAVEDSEED=XSEED(INDEXMO)

        DO I=1,NATOMX
          SAVEDSX(I)=XSTEPX(INDEXMO,I)
          SAVEDSY(I)=XSTEPY(INDEXMO,I)
          SAVEDSZ(I)=XSTEPZ(INDEXMO,I)
          SAVEDSOX(I)=XSTEPOX(INDEXMO,I)
          SAVEDSOY(I)=XSTEPOY(INDEXMO,I)
          SAVEDSOZ(I)=XSTEPOZ(INDEXMO,I)
          SAVEDXX(I)=XX(XINDEX,I)
          SAVEDXY(I)=XY(XINDEX,I)
          SAVEDXZ(I)=XZ(XINDEX,I)
          SAVEDGAMMA(I)=XGAMMA(INDEXMO,I)
          SAVEDGAMMA(I+NATOM)=XGAMMA(INDEXMO,I+NATOM)
          SAVEDGAMMA(I+NATOM2)=XGAMMA(INDEXMO,I+NATOM2)
          SAVEDGAMMA(I+NATOM3)=XGAMMA(INDEXMO,I+NATOM3)
        ENDDO          

        ! End first step------------------------------------------

        ! A Crossing can not occur during another crossing--------
      ELSE IF (QCROS.AND.XCROSCOUNT.GT.0) THEN

        QCROS=.FALSE.

        !  IF(PRNLEV.GE.6) THEN           
        WRITE(OUTU,60) CROSSCOUNT 
60      FORMAT(' CSTEP> QCROS Reset at: ',I8) 
        !  ENDIF

      ENDIF

      XCROSCOUNT = XCROSCOUNT + 1

      ! During the crossing----------------------------------------

      ELSE IF (QCON .AND. .NOT. QCROS) THEN

      ! The final step---------------------------------------------

        IF (XCROSCOUNT .EQ. (2*XTIME-1)) THEN

          QCON = .FALSE.
          QCEND=.TRUE.
          XCROSCOUNT = XCROSCOUNT + 1

          !          IF(PRNLEV.GE.6) THEN
          WRITE(OUTU,70) XCROSCOUNT
70        FORMAT(' CSTEP> Crossing finished: XCROSCOUNT = ',I8)
          !          ENDIF

          ! Calculate when we can start looking for crossings again
          XRESTARTTIME=CROSSCOUNT+XTIME+2

          !          IF(PRNLEV.GE.6) THEN 
          WRITE(OUTU,80) XRESTARTTIME
80        FORMAT(' CSTEP> Recrossing will be reactivated at', &
                 ' step ',I8)
          !          ENDIF

          ! Activate the correct surface
          K=1
          DO I=1, NCRSU-1
            DO J=I+1, NCRSU
              IF (ICDIR(K).EQ.1) THEN
                LOWSU=J
                NEWSU=0
                OLDSU=0
                QON(J)=.TRUE.
                ICDIR(K)=0
              ELSEIF (ICDIR(K).EQ.-1) THEN
                LOWSU=I
                NEWSU=0
                OLDSU=0
                QON(I)=.TRUE.
                ICDIR(K)=0
              ENDIF
              K=K+1
            ENDDO
          ENDDO

          !          IF(PRNLEV.GE.6) THEN 
          WRITE(OUTU,90) LOWSU
90        FORMAT(' CSTEP> Crossing to surface: ',I4, &
                 ' finished') 
          !          ENDIF 

          ! End final step-----------------------------------------------

        ELSE
          XCROSCOUNT = XCROSCOUNT + 1
        ENDIF 

      ENDIF

      ! End during crossing===========================================

      ! No crossing in progress---------------------------------------
      IF ((NCRUN.GT.0) .AND. .NOT. QCON) THEN
        !        WRITE(OUTU,*) "CSTEP> BACKING UP POSITIONS"
        ! Find index to write to
        ! Cycle through temp array using MOD() -- avoids recopying.

        XINDEX=MOD(CROSSCOUNT,XTIME)+1
        INDEXMO=MOD(CROSSCOUNT-1,XTIME)+1
        INDEXMOO=MOD(CROSSCOUNT-2,XTIME)+1

        ! Check for surface crossing every XCROSF steps

        !        IF (MOD(CROSSCOUNT,XCROSF) .EQ. 0 .AND. QCROS) THEN
        ! ... SL: Do surface comparison in each step
        IF (QCROS) THEN
          !           WRITE(OUTU,*) "CSTEP> Calling CBARRIER"
          CALL CBARRIER(X,Y,Z,NATOMX)
        ENDIF

        ! Save information of current timestep-------------------------
        XSEED(XINDEX)=ISEED

        ! Copy position and displacement arrays into temp arrays-------
        DO I=1,NATOMX
          XX(XINDEX,I)=X(I)
          XY(XINDEX,I)=Y(I)
          XZ(XINDEX,I)=Z(I)
          XSTEPOX(XINDEX,I)=XX(INDEXMO,I)-XX(INDEXMOO,I)
          XSTEPOY(XINDEX,I)=XY(INDEXMO,I)-XY(INDEXMOO,I)
          XSTEPOZ(XINDEX,I)=XZ(INDEXMO,I)-XZ(INDEXMOO,I)
          XSTEPX(XINDEX,I)=XX(XINDEX,I)-XX(INDEXMO,I)
          XSTEPY(XINDEX,I)=XY(XINDEX,I)-XY(INDEXMO,I)
          XSTEPZ(XINDEX,I)=XZ(XINDEX,I)-XZ(INDEXMO,I)
          XGAMMA(XINDEX,I)=CURRENTGAMMA(I)
          XGAMMA(XINDEX,I+NATOM)=CURRENTGAMMA(I+NATOM)
          XGAMMA(XINDEX,I+NATOM2)=CURRENTGAMMA(I+NATOM2)
          XGAMMA(XINDEX,I+NATOM3)=CURRENTGAMMA(I+NATOM3)
        ENDDO

      ENDIF

      ! End No Crossing----------------------------------------------
      IF(PRNLEV.GE.7) WRITE(OUTU,5)

5     FORMAT(' CSTEP>=======================================')

    RETURN
  END SUBROUTINE CSTEP


!===================================================================
! CALCULATING SURFACE CROSSING FORCES                              
!===================================================================

  SUBROUTINE CREMOVE(EU,X,Y,Z,DX,DY,DZ,NATOMX)
    !     
    ! Subroutine to remove the PSF-encoded PES during 
    ! surface crossing simulations
    !                                   JD 060103
    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use code
    use param
    use consta
    use energym


    ! Variables ==================================================
    INTEGER I,J,IX,II,JJ,KK,LL,MULT,NATOMX
    real(chm_real) EU,E,EBND0,EANG0,EDIH0,EVDW0,ETOT0
    real(chm_real) FCON,REQ,TEQ,PEQ,PEQC,PEQS,SIG1,SIG2,EFF1,EFF2
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)

    ! End Variables===============================================
    !      WRITE(OUTU,*) "CREMOVE> Inside"
       
    ETOT0=ZERO

    ! Bonds=======================================================
    EBND0=ZERO 
    IF(NCBON.GT.0.AND.QETERM(BOND)) THEN
      DO I=1,NCBON 
        IF(QCBON(NCRSU+1,I)) THEN
          II=ICBON1(I)
          JJ=ICBON2(I) 
          IX=ICB(ICBONI(I))
          FCON=CBC(IX)
          REQ=CBB(IX)  
          CALL XBOND(E,II,JJ,FCON,REQ,X,Y,Z,DX,DY,DZ, &
                  NATOMX,MINONE)
          EBND0=EBND0+E
    !   WRITE(OUTU,*)'CREM: Harm. bond removed:',II,JJ, E,'kcal/mol'
        ENDIF  
      ENDDO
    ENDIF 

    ETOT0=ETOT0+EBND0

    ! End Bonds===================================================       

    !      WRITE(OUTU,*)"CREMOVE> After bonds: EBND0, ETOT0: "
    !     &              ,EBND0,ETOT0   

    ! Angles======================================================

    EANG0=ZERO

    IF(NCANG.GT.0.AND.QETERM(ANGLE)) THEN
      DO I=1,NCANG 
        IF(QCANG(NCRSU+1,I)) THEN
          II=ICANG1(I)
          JJ=ICANG2(I) 
          KK=ICANG3(I)
          IX=ICT(ICANGI(I))
          FCON=CTC(IX)
          TEQ=CTB(IX)
          CALL XANGLE(E,II,JJ,KK,FCON,TEQ,X,Y,Z,DX,DY,DZ, &
                  NATOMX,MINONE)
          EANG0=EANG0+E
    !   WRITE(OUTU,*)'CREM: Angle removed:',II,JJ,KK, E,'kcal/mol'
        ENDIF
      ENDDO
    ENDIF 

    ETOT0=ETOT0+EANG0

    ! End Angles==================================================

    !      WRITE(OUTU,*)"CREMOVE> After angles: EANG0, ETOT0: "
    !     &              ,EANG0,ETOT0   

    ! Dihedrals====================================================

    EDIH0=ZERO 

    IF((NCDIH.GT.0).AND.(QETERM(DIHE))) THEN 
      DO I=1,NCDIH
        IF(QCDIH(NCRSU+1,I)) THEN
          II=ICDIH1(I)
          JJ=ICDIH2(I)
          KK=ICDIH3(I)
          LL=ICDIH4(I)
          IX=ICP(ICDIHI(I))
          FCON=CPC(IX)
          MULT=CPD(IX)
          PEQ=CPB(IX)
          PEQS=CPSIN(IX)
          PEQC=CPCOS(IX)
          CALL XPHI(E,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS, &
                  X,Y,Z,DX,DY,DZ,NATOMX,MINONE)
          EDIH0=EDIH0+E
    !   WRITE(OUTU,*)'CREM: Dihed. removed:',II,JJ,KK,LL, E,'kcal/mol'
        ENDIF
      ENDDO
    ENDIF
    ETOT0=ETOT0+EDIH0

    !===========================================================

    !      WRITE(OUTU,*)"CREMOVE> After dihedrals: EDIH0, ETOT0: "
    !     &              ,EDIH0,ETOT0   

    ! vdW ======================================================

    EVDW0=ZERO   
    IF(QETERM(VDW)) THEN 
      IF(NCRAT.GT.ZERO) THEN
        DO I=1,NCRAT
          II=ICATOM(I)
          SIG1=VDWR(ITC(IAC(II)))
          EFF1=EFF(ITC(IAC(II)))
                
          DO J=1,NATOMX
            IF((II.NE.J).AND.(ICINDX(J).LT.I)) THEN
              SIG2=VDWR(ITC(IAC(J)))
              EFF2=EFF(ITC(IAC(J)))
              CALL XVDW(E,II,J,X,Y,Z,DX,DY,DZ,SIG1,EFF1,SIG2,EFF2, &
                        NATOMX,MINONE)
              EVDW0=EVDW0+E
    !   WRITE(OUTU,*)'CREM: Surf 0 VdW removed:',II,J, E,'kcal/mol'
            ENDIF
          ENDDO
        ENDDO
      ENDIF

      ! End vdW=======================================================

      ! vdW exclusions===============================================

      IF(NCEXC(NCRSU+1).GT.0) THEN
        DO I=1,NCEXC(NCRSU+1) 
          II=ICEXC(NCRSU+1,I,1)
          JJ=ICEXC(NCRSU+1,I,2)
          SIG1=VDWR(ITC(IAC(II)))
          EFF1=EFF(ITC(IAC(II)))
          SIG2=VDWR(ITC(IAC(JJ)))
          EFF2=EFF(ITC(IAC(JJ)))
          CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG1,EFF1,SIG2,EFF2, &
               NATOMX,ONE)
          EVDW0=EVDW0-E
     !  WRITE(OUTU,*)'CREM: Surf 0 VdW added:',II,JJ, E,'kcal/mol'
        ENDDO
      ENDIF
      !      WRITE(OUTU,*)"CREMOVE> After vdW-Excl: EVDW0: ",EVDW0 

      ! End vdW exclusions==========================================

      ! 1-4 interactions============================================

      IF ((NBXMOD.GE.4).AND.(NC14(NCRSU+1).GT.0)) THEN
        DO I=1,NC14(NCRSU+1)
          II=IC14(NCRSU+1,I,1)
          JJ=IC14(NCRSU+1,I,2)
          SIG1=VDWR(ITC(IAC(II)))
          EFF1=EFF(ITC(IAC(II)))
          SIG2=VDWR(ITC(IAC(JJ)))
          EFF2=EFF(ITC(IAC(JJ)))
          CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG1,EFF1,SIG2,EFF2, &
                  NATOMX,ONE)
          EVDW0=EVDW0-E
     !  WRITE(OUTU,*)'CREM: NBXMOD4 -E:',II,JJ, E,'kcal/mol'
          IF (NBXMOD.EQ.5) THEN
            SIG1=VDWR(ITC(IAC(II))+MAXATC)
            EFF1=EFF(ITC(IAC(II))+MAXATC)
            SIG2=VDWR(ITC(IAC(JJ))+MAXATC)
            EFF2=EFF(ITC(IAC(JJ))+MAXATC)
        
            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG1,EFF1,SIG2,EFF2, &
                        NATOMX,MINONE)
            EVDW0=EVDW0+E
    !   WRITE(OUTU,*)'CREM: NBXMOD5 +E:',II,JJ, E,'kcal/mol'
          ENDIF
        ENDDO
      ENDIF
    ENDIF
 
    ETOT0=ETOT0+EVDW0

    ! End 1-4 interactions========================================

    ! Printout====================================================
    IF(PRNLEV.GE.7) THEN
      WRITE(OUTU,15)
      WRITE(OUTU,10) 
      WRITE(OUTU,20) EBND0   
      WRITE(OUTU,30) EANG0   
      WRITE(OUTU,40) EDIH0
      WRITE(OUTU,50) EVDW0 
      WRITE(OUTU,60) ETOT0
      WRITE(OUTU,15)  
    ENDIF
    EU=EU-ETOT0
 
15  FORMAT(' CREMOVE> =====================================')   
10  FORMAT(' CREMOVE> Removed Energy Terms In CROSS')
20  FORMAT(' CREMOVE> Bonds:        ',F10.3)
30  FORMAT(' CREMOVE> Angles:       ',F10.3)
40  FORMAT(' CREMOVE> Dihedrals:    ',F10.3)
50  FORMAT(' CREMOVE> van der Waal: ',F10.3)
60  FORMAT(' CREMOVE> Total:        ',F10.3)
 
    !=============================================================
    RETURN
  END SUBROUTINE CREMOVE

  !================================================================C

  SUBROUTINE CFORCES(EU,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)
    !     
    ! Subroutine to add the current PES during 
    ! surface crossing simulations
    !                                   JD 060103
    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use code
    use param
    use consta
    use energym


    ! Variables=====================================================C

    INTEGER I,J,IX,II,JJ,KK,LL,MULT,NATOMX
    real(chm_real) EU,E,EBND1,EANG1,EDIH1,EEL1,EVDW1,ETOT1,DITES
    real(chm_real) EBND2,EANG2,EDIH2,EEL2,EVDW2,ETOT2,EVDW01,EVDW02
    real(chm_real) EBNDVDW,EBNDVDW1,EBNDVDW2,EBNDEL,EBNDEL1,EBNDEL2
    real(chm_real) EBND,EANG,EDIH,EEL,EVDW,ETOT,E1,E2,RE1,RE2,DE
    real(chm_real) FCON,REQ,TEQ,PEQ,PEQC,PEQS,EVAN,EVAN1,EVAN2,BETA
    real(chm_real) SIG11,SIG12,SIG21,SIG22,EFF11,EFF12,EFF21,EFF22
    real(chm_real) SIG112,SIG122,SIG212,SIG222,EFF112,EFF122,EFF212
    real(chm_real) EFF222,RE12,E12,RE22,E22 
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)

    real(chm_real) FACTOR,FACTOR2  

    ! End Variables================================================C

    !     WRITE(OUTU,*) "CFORCES> Inside"

    ETOT=ZERO
    ETOT1=ZERO
    ETOT2=ZERO

    FACTOR2=ONE-FACTOR

    !     WRITE(OUTU,*) "CFORCES> FACTOR, FACTOR2: ", FACTOR,
    !     &                FACTOR2 

    !---------BONDS------------------------------------------------
      
    EBND=ZERO 
    EBND1=ZERO
    EBND2=ZERO

    IF((NCBON.GT.0).AND.QETERM(BOND)) THEN
      ! If crossing in progress mix OLDSU and NEWSU FF
      IF(QCON) THEN
          
        IF(NCHAR.GT.0) THEN 
          DO I=1,NCHAR

            IF(QCBON(OLDSU,I)) THEN
              II=ICBON1(I)
              JJ=ICBON2(I)
              FCON=KCBON(OLDSU,I)                   
              REQ=RCBON(OLDSU,I) 
              CALL XBOND(E,II,JJ,FCON,REQ,X,Y,Z,DX,DY,DZ, &
                         NATOMX,FACTOR)
              EBND1=EBND1+E

            ENDIF

            IF(QCBON(NEWSU,I)) THEN
              II=ICBON1(I)
              JJ=ICBON2(I)
              FCON=KCBON(NEWSU,I)                   
              REQ=RCBON(NEWSU,I) 
              CALL XBOND(E,II,JJ,FCON,REQ,X,Y,Z,DX,DY,DZ, &
                         NATOMX,FACTOR2)
              EBND2=EBND2+E 
            ENDIF
          ENDDO
        ENDIF

        IF(NCMOR.GT.0) THEN 
          DO I=1,NCMOR
            IF(QCBON(OLDSU,I+NCHAR)) THEN
              II=ICBON1(I+NCHAR)
              JJ=ICBON2(I+NCHAR)
              DE=DCMOR(OLDSU,I)                   
              REQ=RCMOR(OLDSU,I)
              BETA=BCMOR(OLDSU,I)
              CALL XMORSE(E,II,JJ,DE,BETA,REQ,X,Y,Z,DX,DY,DZ, &
                          NATOMX,FACTOR)
              EBND1=EBND1+E 
            ENDIF

            IF(QCBON(NEWSU,I+NCHAR)) THEN
              II=ICBON1(I+NCHAR)
              JJ=ICBON2(I+NCHAR)
              DE=DCMOR(NEWSU,I)                   
              REQ=RCMOR(NEWSU,I)
              BETA=BCMOR(NEWSU,I) 
              CALL XMORSE(E,II,JJ,DE,BETA,REQ,X,Y,Z,DX,DY,DZ, &
                          NATOMX,FACTOR2)
              EBND2=EBND2+E 
            ENDIF
          ENDDO
        ENDIF  
        EBND=FACTOR*EBND1+FACTOR2*EBND2

        ! If no crossing in progress add only FF of LOWSU
      ELSE

        IF(NCHAR.GT.0) THEN
          DO I=1,NCHAR

            IF(QCBON(LOWSU,I)) THEN
              II=ICBON1(I)
              JJ=ICBON2(I)
              FCON=KCBON(LOWSU,I)
              REQ=RCBON(LOWSU,I) 
              CALL XBOND(E,II,JJ,FCON,REQ,X,Y,Z,DX,DY,DZ, &
                         NATOMX,1.0_chm_real)
              EBND1=EBND1+E
    !   WRITE(OUTU,*)'CFORC: Harm. bond added:',II,JJ, E,'kcal/mol'
            ENDIF
          ENDDO
        ENDIF

        IF(NCMOR.GT.0) THEN 
          DO I=1,NCMOR
            IF(QCBON(LOWSU,I+NCHAR)) THEN
              II=ICBON1(I+NCHAR)
              JJ=ICBON2(I+NCHAR)
              DE=DCMOR(LOWSU,I)
              REQ=RCMOR(LOWSU,I)
              BETA=BCMOR(LOWSU,I) 
              CALL XMORSE(E,II,JJ,DE,BETA,REQ,X,Y,Z,DX,DY,DZ, &
                          NATOMX,1.0_CHM_REAL)
              EBND1=EBND1+E
     !  WRITE(OUTU,*)'CFORC: Morse bond added:',II,JJ, E,'kcal/mol'
            ENDIF
          ENDDO
        ENDIF  
        EBND=EBND1
      ENDIF
    ENDIF 


    !-------------Nonbond inclusions ----------------------------

    EBNDVDW=ZERO
    EBNDVDW1=ZERO
    EBNDVDW2=ZERO
    EBNDEL=ZERO
    EBNDEL1=ZERO
    EBNDEL2=ZERO

    IF(QETERM(VDW).OR.QETERM(ELEC)) THEN

      ! If crossing in progress mix OLDSU and NEWSU surfaces
      IF(QCON) THEN
        IF(NCINC(OLDSU).GT.0) THEN 

          ! OLDSU
          DO I=1,NCINC(OLDSU)
            II=ICINC(OLDSU,I,1)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(OLDSU,ICINDX(II))
              E1=VCEPS(OLDSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            JJ=ICINC(OLDSU,I,2)

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(OLDSU,ICINDX(JJ))
              E2=VCEPS(OLDSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF


            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2, &
                        E2,NATOMX,FACTOR)
              EBNDVDW1=EBNDVDW1+E
            ENDIF

            IF(QETERM(ELEC)) THEN  
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)
              EBNDEL1=EBNDEL1+E
            ENDIF

          ENDDO

          ! NEWSU
          DO I=1,NCINC(NEWSU)
            II=ICINC(NEWSU,I,1)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(NEWSU,ICINDX(II))
              E1=VCEPS(NEWSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            JJ=ICINC(NEWSU,I,2)

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(NEWSU,ICINDX(JJ))
              E2=VCEPS(NEWSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF

            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2, &
                        E2,NATOMX,FACTOR2)
              EBNDVDW2=EBNDVDW2+E
            ENDIF

            IF(QETERM(ELEC)) THEN  
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR2)
              EBNDEL2=EBNDEL2+E
            ENDIF

          ENDDO

        ENDIF

        !-------------End 1-2 & 1-3 inclusions---------------------

        !-------------1-4 inclusions-------------------------------
        IF((NCIN14(OLDSU).GT.0).AND.(NBXMOD.GE.4)) THEN 

          DO I=1, NCIN14(OLDSU)

            II=ICIN14(OLDSU,I,1)
            JJ=ICIN14(OLDSU,I,2)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(OLDSU,ICINDX(II))
              E1=VCEPS(OLDSU,ICINDX(II)) 
              RE12=RE1
              E12=E1
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              RE12=VDWR(ITC(IAC(II))+MAXATC)
              E1=EFF(ITC(IAC(II)))
              E12=EFF(ITC(IAC(II))+MAXATC)
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(OLDSU,ICINDX(JJ))
              E2=VCEPS(OLDSU,ICINDX(JJ)) 
              RE22=RE2
              E22=E2
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              RE22=VDWR(ITC(IAC(JJ))+MAXATC)
              E2=EFF(ITC(IAC(JJ)))
              E22=EFF(ITC(IAC(JJ))+MAXATC)
            ENDIF

            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                         NATOMX,FACTOR)

              EBNDVDW1=EBNDVDW1+E 
            ENDIF

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN 

              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)
              EBNDEL1=EBNDEL1+E
            ENDIF

            IF(QETERM(VDW).AND.(NBXMOD.EQ.5)) THEN

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                        E22,NATOMX,-FACTOR)
              EBNDVDW1=EBNDVDW1-E

            ENDIF

          ENDDO
        ENDIF

        IF((NCIN14(NEWSU).GT.0).AND.(NBXMOD.GE.4)) THEN 

          DO I=1, NCIN14(NEWSU)

            II=ICIN14(NEWSU,I,1)
            JJ=ICIN14(NEWSU,I,2)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(NEWSU,ICINDX(II))
              E1=VCEPS(NEWSU,ICINDX(II)) 
              RE12=RE1
              E12=E1
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              RE12=VDWR(ITC(IAC(II))+MAXATC)
              E1=EFF(ITC(IAC(II)))
              E12=EFF(ITC(IAC(II))+MAXATC)
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(NEWSU,ICINDX(JJ))
              E2=VCEPS(NEWSU,ICINDX(JJ)) 
              RE22=RE2
              E22=E2
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              RE22=VDWR(ITC(IAC(JJ))+MAXATC)
              E2=EFF(ITC(IAC(JJ)))
              E22=EFF(ITC(IAC(JJ))+MAXATC)
            ENDIF

            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                        NATOMX,FACTOR2)

              EBNDVDW2=EBNDVDW2+E 
            ENDIF

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN 

              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR2)
              EBNDEL2=EBNDEL2+E
            ENDIF

            IF(QETERM(VDW).AND.(NBXMOD.EQ.5)) THEN

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                        E22,NATOMX,-FACTOR2)
              EBNDVDW2=EBNDVDW2-E

            ENDIF

          ENDDO
        ENDIF
      EBNDVDW=FACTOR*EBNDVDW1+FACTOR2*EBNDVDW2
      EBNDEL=FACTOR*EBNDEL1+FACTOR2*EBNDEL2


      ELSE
        ! System on LOWSU
        IF(NCINC(LOWSU).GT.0) THEN

          DO I=1,NCINC(LOWSU)

            II=ICINC(LOWSU,I,1)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(LOWSU,ICINDX(II))
              E1=VCEPS(LOWSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            JJ=ICINC(LOWSU,I,2)

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(LOWSU,ICINDX(JJ))
              E2=VCEPS(LOWSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF

            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2, &
                         E2,NATOMX,1.0_CHM_REAL)
              EBNDVDW1=EBNDVDW1+E
    !   WRITE(OUTU,*)'CFORC: VdW incl. added:',II,JJ, E,'kcal/mol'
            ENDIF

            IF(QETERM(ELEC)) THEN   
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,1.0_CHM_REAL)
              EBNDEL1=EBNDEL1+E
    !   WRITE(OUTU,*)'CFORC: Elec. incl. added:',II,JJ, E,'kcal/mol'
            ENDIF

          ENDDO
        ENDIF

        !-------------End 1-2 & 1-3 inclusions------------------------

        !-------------1-4 inclusions----------------------------------
        IF((NCIN14(LOWSU).GT.0).AND.(NBXMOD.GE.4)) THEN 

          DO I=1, NCIN14(LOWSU)

            II=ICIN14(LOWSU,I,1)
            JJ=ICIN14(LOWSU,I,2)

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(LOWSU,ICINDX(II))
              E1=VCEPS(LOWSU,ICINDX(II)) 
              RE12=RE1
              E12=E1
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              RE12=VDWR(ITC(IAC(II))+MAXATC)
              E1=EFF(ITC(IAC(II)))
              E12=EFF(ITC(IAC(II))+MAXATC)
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(LOWSU,ICINDX(JJ))
              E2=VCEPS(LOWSU,ICINDX(JJ)) 
              RE22=RE2
              E22=E2
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              RE22=VDWR(ITC(IAC(JJ))+MAXATC)
              E2=EFF(ITC(IAC(JJ)))
              E22=EFF(ITC(IAC(JJ))+MAXATC)
            ENDIF

            IF (QETERM(VDW)) THEN
              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                        NATOMX,1.0_CHM_REAL)
              EBNDVDW1=EBNDVDW1+E 
    !   WRITE(OUTU,*)'CFORC: VdW incl. added:',II,JJ, E,'kcal/mol'
            ENDIF

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN 

              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,1.0_CHM_REAL)
              EBNDEL1=EBNDEL1+E
    !   WRITE(OUTU,*)'CFORC: Elec. incl. added:',II,JJ, E,'kcal/mol'
            ENDIF

            IF(QETERM(VDW).AND.(NBXMOD.EQ.5)) THEN

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                        E22,NATOMX,-1.0_CHM_REAL)
              EBNDVDW1=EBNDVDW1-E 
    !   WRITE(OUTU,*)'CFORC: NBXMOD 5 1:',II,JJ, E,'kcal/mol'
            ENDIF

          ENDDO
        ENDIF

        EBNDVDW=EBNDVDW1
        EBNDEL=EBNDEL1

      ENDIF

    ENDIF

    ! ----------- End 1-4 inclusions ----------------------------

    !-------------END BONDS--------------------------------------
      
    !      WRITE(OUTU,*) "CFORCES> EBND1, EBND2, EBND: ", 
    !     &               EBND1,EBND2,EBND 
    !-------------ANGLES-----------------------------------------

    EANG=ZERO
    EANG1=ZERO
    EANG2=ZERO

    IF((NCANG.GT.0).AND.QETERM(ANGLE)) THEN
    ! If crossing in progress mix OLDSU and NEWSU FF
      IF (QCON) THEN
        DO I=1,NCANG
          IF(QCANG(OLDSU,I)) THEN
            II=ICANG1(I)
            JJ=ICANG2(I)
            KK=ICANG3(I) 
            FCON=KCANG(OLDSU,I)                   
            TEQ=TCANG(OLDSU,I) 
            CALL XANGLE(E,II,JJ,KK,FCON,TEQ,X,Y,Z,DX,DY,DZ, &
                        NATOMX,FACTOR)
            EANG1=EANG1+E

          ENDIF

          IF(QCANG(NEWSU,I)) THEN
            II=ICANG1(I)
            JJ=ICANG2(I)
            KK=ICANG3(I) 
            FCON=KCANG(NEWSU,I)                   
            TEQ=TCANG(NEWSU,I) 
            CALL XANGLE(E,II,JJ,KK,FCON,TEQ,X,Y,Z,DX,DY,DZ, &
                        NATOMX,FACTOR2)
            EANG2=EANG2+E 
          ENDIF
        ENDDO
        EANG=FACTOR*EANG1+FACTOR2*EANG2

      ! If no crossing in progress add only FF of LOWSU
      ELSE
        DO I=1,NCANG

          IF(QCANG(LOWSU,I)) THEN
            II=ICANG1(I)
            JJ=ICANG2(I)
            KK=ICANG3(I) 
            FCON=KCANG(LOWSU,I)                   
            TEQ=TCANG(LOWSU,I) 
            CALL XANGLE(E,II,JJ,KK,FCON,TEQ,X,Y,Z,DX,DY,DZ, &
                         NATOMX,1.0_CHM_REAL)
            EANG1=EANG1+E
     !  WRITE(OUTU,*)'CFORC: Angle added:',II,JJ,KK, E,'kcal/mol'
          ENDIF
        ENDDO
        EANG=EANG1
      ENDIF
    ENDIF

    !--------------END ANGLES----------------------------------------

    !      WRITE(OUTU,*) "CFORCES> EANG1, EANG2, EANG: " ,
    !     &               EANG1,EANG2,EANG 


    !---------------DIHEDRALS----------------------------------------

    EDIH=ZERO
    EDIH1=ZERO
    EDIH2=ZERO

    IF((NCDIH.GT.0).AND.QETERM(DIHE)) THEN
      ! If crossing in progress mix OLDSU and NEWSU FF
      IF (QCON) THEN
        DO I=1,NCDIH
          IF(QCDIH(OLDSU,I)) THEN
            II=ICDIH1(I)
            JJ=ICDIH2(I)
            KK=ICDIH3(I)
            LL=ICDIH4(I) 
            FCON=KCDIH(OLDSU,I)
            MULT=MCDIH(OLDSU,I)
            PEQ=PCDIH(OLDSU,I)
            PEQC=PCDIC(OLDSU,I)
            PEQS=PCDIS(OLDSU,I) 
            CALL XPHI(E,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,X,Y,Z, &
                      DX,DY,DZ,NATOMX,FACTOR)
            EDIH1=EDIH1+E
          ENDIF

          IF(QCDIH(NEWSU,I)) THEN
            II=ICDIH1(I)
            JJ=ICDIH2(I)
            KK=ICDIH3(I)
            LL=ICDIH4(I) 
            FCON=KCDIH(NEWSU,I)
            MULT=MCDIH(NEWSU,I)
            PEQ=PCDIH(NEWSU,I)
            PEQC=PCDIC(NEWSU,I)
            PEQS=PCDIS(NEWSU,I) 
            CALL XPHI(E,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,X,Y,Z, &
                      DX,DY,DZ,NATOMX,FACTOR2) 
            EDIH2=EDIH2+E
          ENDIF

        ENDDO
        EDIH=FACTOR*EDIH1+FACTOR2*EDIH2

      ! If no crossing in progress add only FF of LOWSU
      ELSE
        DO I=1,NCDIH
          IF(QCDIH(LOWSU,I)) THEN
            II=ICDIH1(I)
            JJ=ICDIH2(I)
            KK=ICDIH3(I)
            LL=ICDIH4(I) 
            FCON=KCDIH(LOWSU,I)
            MULT=MCDIH(LOWSU,I)
            PEQ=PCDIH(LOWSU,I)
            PEQC=PCDIC(LOWSU,I)
            PEQS=PCDIS(LOWSU,I) 
            CALL XPHI(E,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,X,Y,Z, &
                       DX,DY,DZ,NATOMX,1.0_CHM_REAL)
            EDIH1=EDIH1+E
    !   WRITE(OUTU,*)'CFORC: Dihed. added:',II,JJ,KK,LL, E,'kcal/mol'
          ENDIF
        ENDDO
        EDIH=EDIH1
      ENDIF
    ENDIF

    !      WRITE(OUTU,*) "CFORCES> EDIH1, EDIH2, EDIH: ", 
    !     &               EDIH1,EDIH2,EDIH 

    !------------END DIHEDRALS--------------------------------------
      
    !-------------VAN DER WAALS-------------------------------------
    EVAN1=ZERO 
    EVAN2=ZERO
    EVAN=ZERO 
  
    IF((NCRAT.GT.0).AND.QETERM(VDW)) THEN
      ! If crossing in progress mix OLDSU and NEWSU FF
      IF (QCON) THEN

        DO I=1,NCRAT

          II=ICATOM(I)
          SIG11=VCRAD(OLDSU,I)
          EFF11=VCEPS(OLDSU,I)
          SIG12=VCRAD(NEWSU,I)
          EFF12=VCEPS(NEWSU,I) 

          DO J=1,NATOMX
            IF((II.NE.J).AND.(ICINDX(J).LT.I)) THEN
              IF(ICINDX(J).GT.0) THEN 
                SIG21=VCRAD(OLDSU,ICINDX(J))
                EFF21=VCEPS(OLDSU,ICINDX(J))
                SIG22=VCRAD(NEWSU,ICINDX(J))
                EFF22=VCEPS(NEWSU,ICINDX(J))
              ELSE
                SIG21=VDWR(ITC(IAC(J)))
                EFF21=EFF(ITC(IAC(J)))
                SIG22=SIG21
                EFF22=EFF21
              ENDIF

              !                IF(.NOT.QON2) THEN
              CALL XVDW(E,II,J,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                         SIG21,EFF21,NATOMX,FACTOR)
              !               WRITE(OUTU,*) "CFORCES> In VDW LOOP. 
              EVAN1=EVAN1+E
              !             ENDIF

              !            IF(.NOT.QON1) THEN
              CALL XVDW(E,II,J,X,Y,Z,DX,DY,DZ,SIG12,EFF12, &
                        SIG22,EFF22,NATOMX,FACTOR2)
              EVAN2=EVAN2+E
              !              ENDIF
            ENDIF
          ENDDO
        ENDDO
        EVAN=FACTOR*EVAN1+FACTOR2*EVAN2
      ELSE
        ! If no crossing in progress add only FF of LOWSU
        DO I=1,NCRAT

          II=ICATOM(I)
          SIG11=VCRAD(LOWSU,I)
          EFF11=VCEPS(LOWSU,I)

          DO J=1,NATOMX
            IF((II.NE.J).AND.(ICINDX(J).LT.I)) THEN
              IF(ICINDX(J).GT.0) THEN 
                SIG21=VCRAD(LOWSU,ICINDX(J))
                EFF21=VCEPS(LOWSU,ICINDX(J))
              ELSE
                SIG21=VDWR(ITC(IAC(J)))
                EFF21=EFF(ITC(IAC(J)))
              ENDIF

              CALL XVDW(E,II,J,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                        SIG21,EFF21,NATOMX,1.0_CHM_REAL)
 
              EVAN1=EVAN1+E
     !  WRITE(OUTU,*)'CFORC: NBXMOD 3 0 -E:',II,J, E,'kcal/mol'
            ENDIF
          ENDDO
        ENDDO
        EVAN=EVAN1
      ENDIF
    ENDIF

      
    !-------------END VDW-----------------------------------------

    !--------------EXCLUSIONS-------------------------------------

    EEL=ZERO
    EVDW=ZERO
    EEL1=ZERO
    EVDW1=ZERO
    EEL2=ZERO  
    EVDW2=ZERO
    EVDW01=ZERO
    EVDW02=ZERO

    IF(QETERM(VDW).OR.QETERM(ELEC)) THEN      
      IF((NCEXC(NCRSU+1).GT.0).AND.QETERM(VDW)) THEN
        ! If crossing in progress mix OLDSU and NEWSU FF
        IF (QCON) THEN
          DO I=1,NCEXC(NCRSU+1)
            II=ICEXC(NCRSU+1,I,1)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(OLDSU,ICINDX(II))
              SIG21=VCRAD(NEWSU,ICINDX(II))
              EFF11=VCEPS(OLDSU,ICINDX(II))
              EFF21=VCEPS(NEWSU,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
              SIG21=SIG11
              EFF21=EFF11
            ENDIF

            JJ=ICEXC(NCRSU+1,I,2)

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(OLDSU,ICINDX(JJ))
              SIG22=VCRAD(NEWSU,ICINDX(JJ))
              EFF12=VCEPS(OLDSU,ICINDX(JJ))
              EFF22=VCEPS(NEWSU,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
              SIG22=SIG12
              EFF22=EFF12
            ENDIF

            !            IF (.NOT.QON2) THEN
            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                       SIG12,EFF12,NATOMX,-FACTOR)
            EVDW01=EVDW01+E
            !          ENDIF

            !          IF (.NOT.QON1) THEN
            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG21,EFF21, &
                      SIG22,EFF22,NATOMX,-FACTOR2)
            EVDW02=EVDW02+E
            !          ENDIF
          ENDDO
        ELSE
          ! If no crossing in progress add only FF of LOWSU
          DO I=1,NCEXC(NCRSU+1)
            II=ICEXC(NCRSU+1,I,1)

            IF(ICINDX(II).GT.0) THEN
              SIG11=VCRAD(LOWSU,ICINDX(II))
              EFF11=VCEPS(LOWSU,ICINDX(II))
            ELSE
              SIG11=VDWR(ITC(IAC(II)))
              EFF11=EFF(ITC(IAC(II)))
            ENDIF

            JJ=ICEXC(NCRSU+1,I,2)

            IF(ICINDX(JJ).GT.0) THEN
              SIG12=VCRAD(LOWSU,ICINDX(JJ))
              EFF12=VCEPS(LOWSU,ICINDX(JJ))
            ELSE
              SIG12=VDWR(ITC(IAC(JJ)))
              EFF12=EFF(ITC(IAC(JJ)))
            ENDIF

            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                       SIG12,EFF12,NATOMX,-1.0_CHM_REAL)
            EVDW01=EVDW01+E
    !   WRITE(OUTU,*)'CFORC: NBXMOD 3 0 excl. +E:',II,JJ, E,'kcal/mol'
          ENDDO
        ENDIF
      ENDIF

      ! If crossing in progress mix OLDSU and NEWSU (next clause) FF
      IF ((QCON).AND.(NCEXC(OLDSU).GT.0)) THEN

        DO I=1,NCEXC(OLDSU)

          II=ICEXC(OLDSU,I,1)
          JJ=ICEXC(OLDSU,I,2)

          IF(QETERM(ELEC)) THEN  
            CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-FACTOR)
            EEL1=EEL1+E       
          ENDIF
         
          ! Pick up correct vdw-parameters
          IF(QETERM(VDW)) THEN

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(OLDSU,ICINDX(II))
              E1=VCEPS(OLDSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(OLDSU,ICINDX(JJ))
              E2=VCEPS(OLDSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF

            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                      NATOMX,-FACTOR)
            EVDW1=EVDW1+E

          ENDIF 
        ENDDO
      ENDIF  
 
      IF ((QCON).AND.(NCEXC(NEWSU).GT.0)) THEN
        DO I=1,NCEXC(NEWSU)
          II=ICEXC(NEWSU,I,1)
          JJ=ICEXC(NEWSU,I,2)

          IF(QETERM(ELEC)) THEN
            CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-FACTOR2)
            EEL2=EEL2+E
          ENDIF

          ! Pick up correct vdw-parameters
          IF(QETERM(VDW)) THEN 

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(NEWSU,ICINDX(II))
              E1=VCEPS(NEWSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(NEWSU,ICINDX(JJ))
              E2=VCEPS(NEWSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF

            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                      NATOMX,-FACTOR2)
            EVDW2=EVDW2+E

          ENDIF 
        ENDDO
      ENDIF

      ! If no crossing in progress add only FF of LOWSU
      IF ((.NOT.QCON).AND.(NCEXC(LOWSU).GT.0)) THEN

        DO I=1,NCEXC(LOWSU)

          II=ICEXC(LOWSU,I,1)
          JJ=ICEXC(LOWSU,I,2)

          IF(QETERM(ELEC)) THEN  
            CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-1.0_CHM_REAL)
            EEL1=EEL1+E
     !  WRITE(OUTU,*)'CFORC: Excl. elec. added:',II,JJ, E,'kcal/mol'
          ENDIF

          ! Pick up correct vdw-parameters

          IF(QETERM(VDW)) THEN

            IF(ICINDX(II).GT.ZERO) THEN
              RE1=VCRAD(LOWSU,ICINDX(II))
              E1=VCEPS(LOWSU,ICINDX(II)) 
            ELSE
              RE1=VDWR(ITC(IAC(II)))
              E1=EFF(ITC(IAC(II)))
            ENDIF

            IF(ICINDX(JJ).GT.ZERO) THEN
              RE2=VCRAD(LOWSU,ICINDX(JJ))
              E2=VCEPS(LOWSU,ICINDX(JJ)) 
            ELSE
              RE2=VDWR(ITC(IAC(JJ)))
              E2=EFF(ITC(IAC(JJ)))
            ENDIF

            CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                      NATOMX,-1.0_CHM_REAL)
            EVDW1=EVDW1+E
     !  WRITE(OUTU,*)'CFORC: Excl. VdW added:',II,JJ, E,'kcal/mol'
          ENDIF 
        ENDDO
      ENDIF

      !-------------End 1-2/3 Exclusions--------------------------------

      !------------- 1-4 Exclusions-----------------------------
      IF (NBXMOD.GE.4) THEN

        IF((NC14(NCRSU+1).GT.0).AND.QETERM(VDW)) THEN
          ! If crossing in progress mix OLDSU and NEWSU FF
          IF (QCON) THEN
            DO I=1,NC14(NCRSU+1)

              II=IC14(NCRSU+1,I,1)

              IF(ICINDX(II).GT.0) THEN
                SIG11=VCRAD(OLDSU,ICINDX(II))
                SIG112=SIG11 
                SIG21=VCRAD(NEWSU,ICINDX(II))
                SIG212=SIG21
                EFF11=VCEPS(OLDSU,ICINDX(II))
                EFF112=EFF11
                EFF21=VCEPS(NEWSU,ICINDX(II))
                EFF212=EFF21
              ELSE
                SIG11=VDWR(ITC(IAC(II)))
                SIG112=VDWR(ITC(IAC(II))+MAXATC)
                EFF11=EFF(ITC(IAC(II)))
                EFF112=EFF(ITC(IAC(II))+MAXATC)
                SIG21=SIG11
                SIG212=SIG112
                EFF21=EFF11
                EFF212=SIG112
              ENDIF

              JJ=IC14(NCRSU+1,I,2)

              IF(ICINDX(JJ).GT.0) THEN
                SIG12=VCRAD(OLDSU,ICINDX(JJ))
                SIG122=SIG12 
                SIG22=VCRAD(NEWSU,ICINDX(JJ))
                SIG222=SIG22
                EFF12=VCEPS(OLDSU,ICINDX(JJ))
                EFF122=EFF12 
                EFF22=VCEPS(NEWSU,ICINDX(JJ))
                EFF222=EFF22
              ELSE
                SIG12=VDWR(ITC(IAC(JJ)))
                SIG122=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF12=EFF(ITC(IAC(JJ)))
                EFF122=EFF(ITC(IAC(JJ))+MAXATC)
                SIG22=SIG12
                SIG222=SIG122 
                EFF22=EFF12
                EFF222=EFF122
              ENDIF

              !          IF (.NOT.QON2) THEN

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                        SIG12,EFF12,NATOMX,-FACTOR)
              EVDW01=EVDW01+E

              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG112,EFF112, &
                          SIG122,EFF122,NATOMX,FACTOR)
                EVDW01=EVDW01-E
              ENDIF

              !            ENDIF

              !  IF (.NOT.QON1) THEN

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG21,EFF21, &
                        SIG22,EFF22,NATOMX,-FACTOR2)
              EVDW02=EVDW02+E

              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG212,EFF212, &
                          SIG222,EFF222,NATOMX,FACTOR2)
                EVDW02=EVDW02-E
              ENDIF  

              !           ENDIF
            ENDDO

          ELSE

            ! If no crossing in progress add only FF of LOWSU
            DO I=1,NC14(NCRSU+1)

              II=IC14(NCRSU+1,I,1)

              IF(ICINDX(II).GT.0) THEN
                SIG11=VCRAD(LOWSU,ICINDX(II))
                SIG112=SIG11 
                EFF11=VCEPS(LOWSU,ICINDX(II))
                EFF112=EFF11
              ELSE
                SIG11=VDWR(ITC(IAC(II)))
                SIG112=VDWR(ITC(IAC(II))+MAXATC)
                EFF11=EFF(ITC(IAC(II)))
                EFF112=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              JJ=IC14(NCRSU+1,I,2)

              IF(ICINDX(JJ).GT.0) THEN
                SIG12=VCRAD(LOWSU,ICINDX(JJ))
                SIG122=SIG12 
                EFF12=VCEPS(LOWSU,ICINDX(JJ))
                EFF122=EFF12 
              ELSE
                SIG12=VDWR(ITC(IAC(JJ)))
                SIG122=VDWR(ITC(IAC(JJ))+MAXATC)
                EFF12=EFF(ITC(IAC(JJ)))
                EFF122=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG11,EFF11, &
                        SIG12,EFF12,NATOMX,-1.0_CHM_REAL)
              EVDW01=EVDW01+E
    !   WRITE(OUTU,*)'CFORC: NBXMOD 4 0 excl. +E:',II,JJ, E,'kcal/mol'
              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,SIG112,EFF112, &
                          SIG122,EFF122,NATOMX,1.0_CHM_REAL)
                EVDW01=EVDW01-E
    !   WRITE(OUTU,*)'CFORC: NBXMOD 5 0 excl. -E:',II,JJ, E,'kcal/mol'
              ENDIF

            ENDDO
          ENDIF
        ENDIF

        ! If crossing in progress mix OLDSU and NEWSU FF
        IF ((QCON).AND.(NC14(OLDSU).GT.0)) THEN

          DO I=1,NC14(OLDSU)
            II=IC14(OLDSU,I,1)
            JJ=IC14(OLDSU,I,2)

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN 
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-FACTOR)
              EEL1=EEL1+E       
            ENDIF

            ! Pick up correct vdw-parameters
            IF(QETERM(VDW)) THEN

              IF(ICINDX(II).GT.ZERO) THEN
                RE1=VCRAD(OLDSU,ICINDX(II))
                E1=VCEPS(OLDSU,ICINDX(II)) 
                RE12=RE1
                E12=E1
              ELSE
                RE1=VDWR(ITC(IAC(II)))
                RE12=VDWR(ITC(IAC(II))+MAXATC)            
                E1=EFF(ITC(IAC(II)))
                E12=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              IF(ICINDX(JJ).GT.ZERO) THEN
                RE2=VCRAD(OLDSU,ICINDX(JJ))
                E2=VCEPS(OLDSU,ICINDX(JJ)) 
                RE22=RE2
                E22=E2
              ELSE
                RE2=VDWR(ITC(IAC(JJ)))
                RE22=VDWR(ITC(IAC(JJ))+MAXATC)            
                E2=EFF(ITC(IAC(JJ)))
                E22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                        NATOMX,-FACTOR)
              EVDW1=EVDW1+E 

              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                          E22,NATOMX,FACTOR)
                EVDW1=EVDW1-E 
              ENDIF

            ENDIF
          ENDDO
        ENDIF

        IF ((QCON).AND.(NC14(NEWSU).GT.0)) THEN

          DO I=1,NC14(NEWSU)

            II=IC14(NEWSU,I,1)
            JJ=IC14(NEWSU,I,2)

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-FACTOR2)
              EEL2=EEL2+E       
            ENDIF 

            ! Pick up correct vdw-parameters
            IF(QETERM(VDW)) THEN     

              IF(ICINDX(II).GT.ZERO) THEN
                RE1=VCRAD(NEWSU,ICINDX(II))
                RE12=RE1 
                E1=VCEPS(NEWSU,ICINDX(II))
                E12=E1  
              ELSE
                RE1=VDWR(ITC(IAC(II)))
                RE12=VDWR(ITC(IAC(II))+MAXATC)
                E1=EFF(ITC(IAC(II)))
                E12=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              IF(ICINDX(JJ).GT.ZERO) THEN
                RE2=VCRAD(NEWSU,ICINDX(JJ))
                RE22=RE2 
                E2=VCEPS(NEWSU,ICINDX(JJ))
                E22=E2 
              ELSE
                RE2=VDWR(ITC(IAC(JJ)))
                RE22=VDWR(ITC(IAC(JJ))+MAXATC)
                E2=EFF(ITC(IAC(JJ)))
                E22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                        NATOMX,-FACTOR2)
              EVDW2=EVDW2+E 

              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                          E22,NATOMX,FACTOR2)
                EVDW2=EVDW2-E 
              ENDIF

            ENDIF  
          ENDDO
        ENDIF

        ! If no crossing in progress add only FF of LOWSU
        IF ((.NOT.QCON).AND.(NC14(LOWSU).GT.0)) THEN

          DO I=1,NC14(LOWSU)
            II=IC14(LOWSU,I,1)
            JJ=IC14(LOWSU,I,2)

            IF((NBXMOD.GE.4).AND.QETERM(ELEC)) THEN 
              CALL XELEC(E,II,JJ,X,Y,Z,DX,DY,DZ,NATOMX,-1.0_CHM_REAL)
              EEL1=EEL1+E
    !   WRITE(OUTU,*)'CFORC: NBXMOD 4 Excl. Elec. +E:',II,JJ, E,'kcal/mol'     
            ENDIF

            ! Pick up correct vdw-parameters
            IF(QETERM(VDW)) THEN

              IF(ICINDX(II).GT.ZERO) THEN
                RE1=VCRAD(LOWSU,ICINDX(II))
                E1=VCEPS(LOWSU,ICINDX(II)) 
                RE12=RE1
                E12=E1
              ELSE
                RE1=VDWR(ITC(IAC(II)))
                RE12=VDWR(ITC(IAC(II))+MAXATC)
                E1=EFF(ITC(IAC(II)))
                E12=EFF(ITC(IAC(II))+MAXATC)
              ENDIF

              IF(ICINDX(JJ).GT.ZERO) THEN
                RE2=VCRAD(LOWSU,ICINDX(JJ))
                E2=VCEPS(LOWSU,ICINDX(JJ)) 
                RE22=RE2
                E22=E2
              ELSE
                RE2=VDWR(ITC(IAC(JJ)))
                RE22=VDWR(ITC(IAC(JJ))+MAXATC)
                E2=EFF(ITC(IAC(JJ)))
                E22=EFF(ITC(IAC(JJ))+MAXATC)
              ENDIF

              CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE1,E1,RE2,E2, &
                        NATOMX,-1.0_CHM_REAL)
              EVDW1=EVDW1+E 
    !   WRITE(OUTU,*)'CFORC: NBXMOD 4 Excl. +E:',II,JJ, E,'kcal/mol'  
              IF(NBXMOD.EQ.5) THEN
                CALL XVDW(E,II,JJ,X,Y,Z,DX,DY,DZ,RE12,E12,RE22, &
                          E22,NATOMX,1.0_CHM_REAL)
                EVDW1=EVDW1-E
    !   WRITE(OUTU,*)'CFORC: NBXMOD 5 Excl. -E:',II,JJ, E,'kcal/mol' 
              ENDIF

            ENDIF
          ENDDO
        ENDIF

      ENDIF
    ENDIF
 
    IF(.NOT.QCON) THEN
      EEL=EEL1
      EVDW=EVDW1+EVDW01
    ELSE 
      EEL=FACTOR*EEL1+FACTOR2*EEL2
      EVDW=FACTOR*(EVDW1+EVDW01)+FACTOR2*(EVDW2+EVDW02) 
    ENDIF 

    !------------END 1-4 interactions---------------------------

    !-------------SUMMATION-------------------------------------

    ETOT1=EBND1+EBNDVDW1+EBNDEL1+EANG1+EDIH1 &
           +EVAN1-EEL1-EVDW1-EVDW01
    ETOT2=EBND2+EBNDVDW2+EBNDEL2+EANG2+EDIH2 &
           +EVAN2-EEL2-EVDW2-EVDW02
      
    ETOT=EBND+EBNDVDW+EBNDEL+EANG+EDIH+EVAN-EEL-EVDW

    IF(PRNLEV.GE.7) THEN

      WRITE(OUTU,5)
      IF (QCON) THEN
        WRITE(OUTU,80) OLDSU, NEWSU
      ELSE 
        WRITE(OUTU,90) LOWSU
      ENDIF
      WRITE(OUTU,10) 
      WRITE(OUTU,20) EBND 
      WRITE(OUTU,30) EANG 
      WRITE(OUTU,40) EDIH 
      WRITE(OUTU,50) -EEL
      WRITE(OUTU,60) (EVAN-EVDW) 
      WRITE(OUTU,70) ETOT 
      WRITE(OUTU,5)
    ENDIF 

5   FORMAT(' CFORCES> ===================================')
10  FORMAT(' CFORCES> Energy added from specified surfaces')
20  FORMAT(' CFORCES> Bonds            :',F10.3)
30  FORMAT(' CFORCES> Angles           :',F10.3)
40  FORMAT(' CFORCES> Dihedrals        :',F10.3)
50  FORMAT(' CFORCES> Electrostatics   :',F10.3)
60  FORMAT(' CFORCES> Van der Waals    :',F10.3)
70  FORMAT(' CFORCES> TOTAL            :',F10.3)        
80  FORMAT(' CFORCES> Surfaces ',I2,' and ',I2)
90  FORMAT(' CFORCES> Surface ',I2)
       
    EU=EU+ETOT  

    RETURN
  END SUBROUTINE CFORCES

  !=============================================================
  ! SURFACE CROSSING FORCES - HELP SUBROUTINES                  
  !=============================================================

  SUBROUTINE XBOND(EB,I,J,FORCE,REQ,X,Y,Z,DX,DY,DZ, &
                   NATOMX,FACTOR)

    ! Based on EBONDFS() from energy/enefscal.src
    !
    ! Calculates bond energies and forces
    ! Preprocessor commands have been removed 
    !
    use number
    use stream

    real(chm_real) EB,FORCE,REQ,FACTOR
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    INTEGER I,J

    real(chm_real) RX,RY,RZ,S,R,DB,DF,DXI,DYI,DZI

    !      WRITE(OUTU,*) I,J,FORCE,REQ,NATOMX,FACTOR

    EB=ZERO

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    S=SQRT(RX*RX+RY*RY+RZ*RZ)
    R=2.0/S

    DB=S-REQ
    DF=FORCE*DB
    EB=EB+DF*DB

    DF=DF*R
    DXI=RX*DF*FACTOR
    DYI=RY*DF*FACTOR
    DZI=RZ*DF*FACTOR

    !      WRITE(OUTU,*) "XBOND> FORCE: ",I,J,DB,DF

    DX(I)=DX(I)+DXI 
    DY(I)=DY(I)+DYI
    DZ(I)=DZ(I)+DZI
    DX(J)=DX(J)-DXI
    DY(J)=DY(J)-DYI
    DZ(J)=DZ(J)-DZI

    RETURN
  END SUBROUTINE XBOND

  !====================================================================

  SUBROUTINE XMORSE(EM,I,J,DE,BETA,RMIN,X,Y,Z,DX,DY,DZ, &
                        NATOMX,FACTOR)
    !     
    !     The energy and forces of a Morse potential
    !
    !     Based on MORSE()
    !--------------------------------------------------------------------- 
    !
    use dimens_fcm
    use number
    !M##FORTRAN
    !
    ! passed variables
    INTEGER NATOMX, I, J
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX),DE, BETA, RMIN
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX),EM,FACTOR

    ! local variables
    real(chm_real) RX, RY, RZ, R2, R, RR, Q, A, B
    real(chm_real) DXI,DYI,DZI,DF

    EM=ZERO

    !      WRITE(6,*) " *** DE, BETA, RMIN = ",DE,BETA,RMIN

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    R=SQRT(RX*RX+RY*RY+RZ*RZ)

    Q = R-RMIN
    B = MINONE*BETA*Q
    A = TWO*B
      
    EM = DE*(1.0_chm_real+EXP(A)-TWO*EXP(B))

    ! Force (dV/dr)
    DF=2.0_chm_real*BETA*DE*(EXP(B)-EXP(A))
    ! Force (dr/d{xyz})
    RR=1.0_chm_real/R
    ! Components as for harmonic bond
    DXI=RX*DF*RR*FACTOR
    DYI=RY*DF*RR*FACTOR
    DZI=RZ*DF*RR*FACTOR
    !     WRITE(6,*) " VALUES: ",DF,R,FACTOR,DXI,DYI,DZI
    !     WRITE(6,*) "VALUES: ",I,J,Q,DF

    DX(I)=DX(I)+DXI 
    DY(I)=DY(I)+DYI
    DZ(I)=DZ(I)+DZI
    DX(J)=DX(J)-DXI
    DY(J)=DY(J)-DYI
    DZ(J)=DZ(J)-DZI
      
    RETURN
  END SUBROUTINE XMORSE

  !====================================================================

  SUBROUTINE XANGLE(ET,I,J,K,FORCE,TEQR,X,Y,Z,DX,DY,DZ, &
                    NATOMX,FACTOR)

    !     Calculates bond angles, angle energies and forces
    !     Based on subroutine EANGLE() from energy/eintern.src
    !     Equilibrium angle is passed in _radians_
    !
    use number
    use dimens_fcm
    use stream
    use consta
    use fast
    !
    real(chm_real) ET
    INTEGER I,J,K,NATOMX
    real(chm_real) FORCE,TEQR,FACTOR
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    !
    real(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,RI2,RJ2,RI,RJ
    real(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,CST, &
                   AT,DA,DF,E
    real(chm_real) ST2R,STR,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ
    real(chm_real) DFX,DFY,DFZ,DGX,DGY,DGZ,DDF,SMALLV
    !
    !   WRITE(OUTU,*) "#IN : ",I,J,K,FORCE,TEQR,NATOMX,FACTOR
    !   WRITE(OUTU,*) X(I),Y(I),Z(I)
    !   WRITE(OUTU,*) X(J),Y(J),Z(J)
    !   WRITE(OUTU,*) X(K),Y(K),Z(K)
      
    ET=ZERO

    SMALLV=RPRECI
    !
    DXI=X(I)-X(J)
    DYI=Y(I)-Y(J)
    DZI=Z(I)-Z(J)
    DXJ=X(K)-X(J)
    DYJ=Y(K)-Y(J)
    DZJ=Z(K)-Z(J)
    RI2=DXI*DXI+DYI*DYI+DZI*DZI
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

    RI=SQRT(RI2)
    RJ=SQRT(RJ2)
    RIR=ONE/RI
    RJR=ONE/RJ
    DXIR=DXI*RIR
    DYIR=DYI*RIR
    DZIR=DZI*RIR

    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR
    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR
    !
    IF(ABS(CST).GE.COSMAX) THEN
      IF(ABS(CST).GT.ONE) CST=SIGN(ONE,CST)
      AT=ACOS(CST)
      DA=AT-TEQR
      IF(ABS(DA).GT.0.1) THEN
        WRITE(OUTU,10) I,J,K
10      FORMAT(' WARNING FROM EANGLE. Angle is almost linear.', &
              /' Derivatives may be affected for atoms:',3I5)
        WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
        WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
        WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
        WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
        WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
        WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
101     FORMAT(5X,A,5F15.5)
      ENDIF
    ENDIF
    !
    AT=ACOS(CST)
    !
    DA=AT-TEQR


    DF=FORCE*DA
    DDF=FORCE

    E=DF*DA
    ET=ET+E
    DF=DF+DF
    !    WRITE(OUTU,*) "XANGLE> : ",I,J,K,DA,AT,TEQR,FORCE,ET      

    ! ... Workaround for different cosine comparison in SLOW and
    ! ... FAST angle routines. Check which one are set by CHARMM

    ! ... If slow angle routine (EANGLE) is used
    IF ((FASTER.EQ.-1).OR.(LFAST.EQ.-1)) THEN
      IF(ABS(CST).GE.0.999) THEN
        ST2R=ONE/(ONE-CST*CST+SMALLV)
        STR=SQRT(ST2R)
        IF(TEQR.LT.PT001) THEN
          DF=MINTWO*FORCE*(ONE+DA*DA*SIXTH)
        ELSE IF(PI-TEQR.LT.PT001) THEN
          DF=TWO*FORCE*(ONE+DA*DA*SIXTH)
        ELSE
          DF=-DF*STR
        ENDIF
      ELSE
        ST2R=ONE/(ONE-CST*CST)
        STR=SQRT(ST2R)
        DF=-DF*STR
      ENDIF

    ! ... If fast angle routine (EANGLFS) is used
    ELSE
      IF(ABS(CST).GE.0.999999) THEN
        ST2R=ONE/(ONE-CST*CST+SMALLV)
        STR=SQRT(ST2R)
        IF(TEQR.LT.PT001) THEN
          DF=MINTWO*FORCE*(ONE+DA*DA*SIXTH)
        ELSE IF(PI-TEQR.LT.PT001) THEN
          DF=TWO*FORCE*(ONE+DA*DA*SIXTH)
        ELSE
          DF=-DF*STR
        ENDIF
      ELSE
        ST2R=ONE/(ONE-CST*CST)
        STR=SQRT(ST2R)
        DF=-DF*STR
      ENDIF
    ENDIF

    !
    DTXI=RIR*(DXJR-CST*DXIR)
    DTXJ=RJR*(DXIR-CST*DXJR)
    DTYI=RIR*(DYJR-CST*DYIR)
    DTYJ=RJR*(DYIR-CST*DYJR)
    DTZI=RIR*(DZJR-CST*DZIR)
    DTZJ=RJR*(DZIR-CST*DZJR)
    !
    DFX=DF*DTXI*FACTOR
    DGX=DF*DTXJ*FACTOR
    DX(I)=DX(I)+DFX
    DX(K)=DX(K)+DGX
    DX(J)=DX(J)-DFX-DGX
    !
    DFY=DF*DTYI*FACTOR
    DGY=DF*DTYJ*FACTOR
    DY(I)=DY(I)+DFY
    DY(K)=DY(K)+DGY
    DY(J)=DY(J)-DFY-DGY
    !
    DFZ=DF*DTZI*FACTOR
    DGZ=DF*DTZJ*FACTOR
    DZ(I)=DZ(I)+DFZ
    DZ(K)=DZ(K)+DGZ
    DZ(J)=DZ(J)-DFZ-DGZ

    RETURN
  END SUBROUTINE XANGLE

  !==================================================================


  SUBROUTINE XPHI(EP,I,J,K,L,FCONS,MULT,PEQR,PEQCOS,PEQSIN,X,Y,Z, &
                  DX,DY,DZ,NATOMX,FACTOR)
    !===================================================
    ! This calculates dihedral angle,energy and forces
    ! based on ephi in eintern.src
    !====================================================

    use dimens_fcm
    use number
    use econtmod
    use stream

    real(chm_real) EP
    INTEGER I,J,K,L,MULT,NATOMX
    real(chm_real) FCONS,PEQR,PEQCOS,PEQSIN,FACTOR
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)

    INTEGER NWARN,NWARNX,IPER,NPER
    real(chm_real) CPBIC,E1,DF1,DDF1,FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA2R,RB2R,RG2,RG, &
                   RGR,RGR2
    real(chm_real) RABR,CP,AP,SP,E,DF,DDF,CA,SA,ARG,APR
    real(chm_real) GAA,GBB,FG,HG,FGA,HGB,FGRG2,HGRG2,DFRG3
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    real(chm_real) DTFX,DTFY,DTFZ,DTHX,DTHY,DTHZ,DTGX,DTGY,DTGZ

    real(chm_real), PARAMETER :: RXMIN=0.005_CHM_REAL,RXMIN2=0.000025_CHM_REAL

    EP=ZERO
    NWARN=0
    NWARNX=0


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

    IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
       NWARN=NWARN+1
      IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
        WRITE(OUTU,20) I,J,K,L
20      FORMAT(' EPHI: WARNING.  dihedral is almost linear.'/ &
               ' derivatives may be affected for atoms:',4I5)
      ENDIF
      return
    ENDIF

    RGR=ONE/RG
    RA2R=ONE/RA2
    RB2R=ONE/RB2
    RABR=SQRT(RA2R*RB2R)
    ! CP=cos(phi)
    CP=(AX*BX+AY*BY+AZ*BZ)*RABR
    ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
    ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
    SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)

    IF(MULT.NE.0) THEN 

      E=ZERO
      DF=ZERO
      DDF=ZERO
      IPER=MULT
      IF (IPER.LT.0) THEN
        IPER=-IPER
      ENDIF
      !
      E1=ONE
      DF1=ZERO
      !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
       DO NPER=1,IPER
         DDF1=E1*CP-DF1*SP
         DF1=E1*SP+DF1*CP
         E1=DDF1
       ENDDO
       E1=E1*PEQCOS+DF1*PEQSIN
       DF1=DF1*PEQCOS-DDF1*PEQSIN
       DF1=-IPER*DF1
       DDF1=-IPER*IPER*E1
       E1=ONE+E1
       !
       IF (IPER.EQ.0) THEN
         E1=ONE
         DF1=ZERO
         DDF1=ZERO
       ENDIF
       !
       ARG=FCONS
       E=E+ARG*E1
       DF=DF+ARG*DF1
       DDF=DDF+ARG*DDF1
       !
     ELSE
       CA=CP*PEQCOS+SP*PEQSIN
       SA=SP*PEQCOS-CP*PEQSIN
       IF (CA.GT.PTONE ) THEN
         AP=ASIN(SA)
       ELSE
         AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
       ENDIF

       DDF=TWO*FCONS
       DF=DDF*AP
       E=HALF*DF*AP
          
    ENDIF

    ! Energy Done

    EP=EP+E

    ! BEGIN: Derivatives

    ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
    !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
    FG=FX*GX+FY*GY+FZ*GZ
    HG=HX*GX+HY*GY+HZ*GZ
    FGA=FG*RA2R*RGR
    HGB=HG*RB2R*RGR
    GAA=-RA2R*RG
    GBB=RB2R*RG

    ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. 

    DTFX=GAA*AX
    DTFY=GAA*AY
    DTFZ=GAA*AZ
    DTGX=FGA*AX-HGB*BX
    DTGY=FGA*AY-HGB*BY
    DTGZ=FGA*AZ-HGB*BZ
    DTHX=GBB*BX
    DTHY=GBB*BY
    DTHZ=GBB*BZ

    !    DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

    DFX=DF*DTFX
    DFY=DF*DTFY
    DFZ=DF*DTFZ
    DGX=DF*DTGX
    DGY=DF*DTGY
    DGZ=DF*DTGZ
    DHX=DF*DTHX
    DHY=DF*DTHY
    DHZ=DF*DTHZ

    !   Distribute over Ri.

    DX(I)=DX(I)+FACTOR*DFX
    DY(I)=DY(I)+FACTOR*DFY
    DZ(I)=DZ(I)+FACTOR*DFZ
    DX(J)=DX(J)-FACTOR*(DFX-DGX)
    DY(J)=DY(J)-FACTOR*(DFY-DGY)
    DZ(J)=DZ(J)-FACTOR*(DFZ-DGZ)
    DX(K)=DX(K)-FACTOR*(DHX+DGX)
    DY(K)=DY(K)-FACTOR*(DHY+DGY)
    DZ(K)=DZ(K)-FACTOR*(DHZ+DGZ)
    DX(L)=DX(L)+FACTOR*DHX
    DY(L)=DY(L)+FACTOR*DHY
    DZ(L)=DZ(L)+FACTOR*DHZ

     !     WRITE(OUTU,*) "XPHI> Added components: ",FACTOR,DTFX,DTGX,
     !     &               DTHX,DF  

    RETURN
  END SUBROUTINE XPHI

  !===================================================================

  SUBROUTINE XELEC(E,I,J,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)
    !
    ! Electrostatic energy and forces. Based on FUNCTION ELEC()
    ! and includes shifting function
    !
    !--------------------------------------------------------------------- 
    !
    use dimens_fcm
    use number
    use stream
    use psf
    use param
    use inbnd
    !
    ! passed variables
    real(chm_real) E
    INTEGER I,J,NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX),FACTOR
    ! local variables
    real(chm_real) RX,RY,RZ,DR2,DR4,DR,RM1,RM2,DF,DEX,DEY,DEZ
    real(chm_real) CTOFNB1,CTOFNB2,CTOFNB4

    E=ZERO
    !      WRITE(OUTU,*) 'XELEC>',I,J,FACTOR,CTOFNB,CG(I),CG(J)

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    DR2=RX*RX+RY*RY+RZ*RZ
    CTOFNB2=CTOFNB*CTOFNB
    ! Check cutoff distance:
    IF (DR2.LE.CTOFNB2) THEN
       ! Distances
       DR=SQRT(DR2)
       DR4=DR2*DR2
       RM1=ONE/DR
       RM2=RM1*RM1
       CTOFNB1=ONE/CTOFNB
       CTOFNB2=CTOFNB1*CTOFNB1
       CTOFNB4=CTOFNB2*CTOFNB2

       ! Calculate electrostatic potential
       E=332.0716_CHM_REAL*CG(I)*CG(J)*RM1* &
            (1.0_chm_real-2.0_chm_real*DR2*CTOFNB2+DR4*CTOFNB4)
       ! Calculate Forces
       ! DE/DR
       DF=-332.0716_CHM_REAL*CG(I)*CG(J)*(RM2+2.0_chm_real*CTOFNB2-3.0_chm_real*DR2*CTOFNB4)
       ! DE/DR*DR/Dq
       DEX=DF*RX*RM1*FACTOR
       DEY=DF*RY*RM1*FACTOR
       DEZ=DF*RZ*RM1*FACTOR
       ! Add on forces
       DX(I)=DX(I)+ DEX
       DY(I)=DY(I)+ DEY
       DZ(I)=DZ(I)+ DEZ
       DX(J)=DX(J)- DEX
       DY(J)=DY(J)- DEY
       DZ(J)=DZ(J)- DEZ
       !
    ENDIF

    RETURN
  END SUBROUTINE XELEC

  !======================================================================

  SUBROUTINE XVDW(E,I,J,X,Y,Z,DX,DY,DZ,R1,E1,R2,E2, &
       NATOMX,FACTOR)
    !     
    !     VDW interaction and forces. Based on FUNCTION VDW()
    !     and uses the distance dependent switching function
    !
    !--------------------------------------------------------------------- 
    !
    use number
    use stream
    use inbnd
    !
    ! passed variables
    real(chm_real) R1,R2,E1,E2      
    real(chm_real) E,FACTOR
    INTEGER I,J,NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    ! local variables
    real(chm_real) RX,RY,RZ,S,RM2,SIG2,SIG6,SIG12,EN
    real(chm_real) EMIN,RUL12,RMIN2,DEN,DF,DEX,DEY,DEZ
    ! Variables for the switching function
    real(chm_real) RIJL,RIJU,C2ONNB,C2OFNB,RUL3,DFN,FUNCT

    E=ZERO

    !      WRITE(OUTU,*) " VDW: ",I,J,R1,E1,R2,E2,FACTOR

    ! Distances
    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    S=RX*RX+RY*RY+RZ*RZ

    ! Switching function stuff
    C2ONNB=CTONNB*CTONNB
    C2OFNB=CTOFNB*CTOFNB

    IF (S.LT.C2OFNB) THEN

       RMIN2=(R1+R2)**2
       EMIN=SQRT(E1*E2)
       RM2=1.0_chm_real/S

       SIG2=RMIN2*RM2
       SIG6=SIG2*SIG2*SIG2
       SIG12=SIG6*SIG6
       IF (S.GT.C2ONNB) THEN
          RIJL=C2ONNB-S
          RIJU=C2OFNB-S
          RUL3=ONE/(C2OFNB-C2ONNB)**3
          RUL12=RUL3*TWELVE
          FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
          DFN=RIJL*RIJU*RUL12
          EN=EMIN*(SIG12-SIG6-SIG6)
          E=(FUNCT*EN)
          DEN=EMIN*RM2*TWELVE*(SIG6-SIG12)
          DF=DFN*EN+FUNCT*DEN
       ELSE
          E=EMIN*(SIG12-SIG6-SIG6)
          DF=EMIN*RM2*12.0*(SIG6-SIG12)
       ENDIF

       ! Calculate force components
       DEX=DF*RX*FACTOR
       DEY=DF*RY*FACTOR
       DEZ=DF*RZ*FACTOR
       ! Add on forces
       DX(I)=DX(I)+ DEX
       DY(I)=DY(I)+ DEY
       DZ(I)=DZ(I)+ DEZ
       DX(J)=DX(J)- DEX
       DY(J)=DY(J)- DEY
       DZ(J)=DZ(J)- DEZ        

    ENDIF
    !
    !      WRITE(OUTU,*) " XVDW: ",I,J,S,E

    RETURN
  END SUBROUTINE XVDW


  SUBROUTINE RVDW123(EU,X,Y,Z,DX,DY,DZ,NATOMX,FACTOR)


    use dimens_fcm
    use number
    use stream
    use psf
    use code
    use param
    use inbnd
    use consta

    ! ... Passed variables
    INTEGER NATOMX
    real(chm_real) EU,X(NATOMX),Y(NATOMX),Z(NATOMX), &
                   DX(NATOMX),DY(NATOMX),DZ(NATOMX),FACTOR

    ! ... Local variables
    INTEGER I,J
    real(chm_real) EBNDVDW,EBNDVDW1,EBNDVDW2,RE1,E1,RE2,E2, &
                   E,FACTOR2

    EBNDVDW=ZERO
    EBNDVDW1=ZERO
    EBNDVDW2=ZERO
    E=ZERO

    FACTOR2=ONE-FACTOR

    !-------------------------------------------------------
    ! ... System on LOWSU
    !-------------------------------------------------------
    IF (.NOT.QCON) THEN
      DO I=1,NBVDW
        ! ... Check if bond is broken on LOWSU
        IF (BVDWSURF(I,LOWSU)) THEN

          ! ... Remove standard CHARMM-FF LJ potential between bonding partners
          IF(ICINDX(BVDWEXC1(I)).GT.ZERO) THEN
            RE1=VCRAD(LOWSU,ICINDX(BVDWEXC1(I)))
            E1=VCEPS(LOWSU,ICINDX(BVDWEXC1(I))) 
          ELSE
            RE1=VDWR(ITC(IAC(BVDWEXC1(I))))
            E1=EFF(ITC(IAC(BVDWEXC1(I))))
          ENDIF

          IF(ICINDX(BVDWEXC2(I)).GT.ZERO) THEN
            RE2=VCRAD(LOWSU,ICINDX(BVDWEXC2(I)))
            E2=VCEPS(LOWSU,ICINDX(BVDWEXC2(I))) 
          ELSE
            RE2=VDWR(ITC(IAC(BVDWEXC2(I))))
            E2=EFF(ITC(IAC(BVDWEXC2(I))))
          ENDIF

          CALL XVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                    RE1,E1,RE2,E2,NATOMX,-1.0_CHM_REAL)
          EBNDVDW=EBNDVDW-E

          ! ... Add user defined LJ potential
          CALL USRVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                      BVDWSIG(I),BVDWEPS(I),ATTP(I),REPP(I), &
                      NATOMX,1.0_CHM_REAL)

          EBNDVDW=EBNDVDW+E

        ENDIF
      ENDDO

      EU=EU+EBNDVDW

      ! ----------------------------------------------------------------------
      ! ... if crossing in progress mix OLDSU and NEWSU
      ! ----------------------------------------------------------------------
    ELSE
      DO I=1,NBVDW
        ! ... OLDSU
        IF (BVDWSURF(I,OLDSU)) THEN

          ! ... Remove standard CHARMM-FF LJ potential between bonding partners
          IF(ICINDX(BVDWEXC1(I)).GT.ZERO) THEN
            RE1=VCRAD(OLDSU,ICINDX(BVDWEXC1(I)))
            E1=VCEPS(OLDSU,ICINDX(BVDWEXC1(I))) 
          ELSE
            RE1=VDWR(ITC(IAC(BVDWEXC1(I))))
            E1=EFF(ITC(IAC(BVDWEXC1(I))))
          ENDIF

          IF(ICINDX(BVDWEXC2(I)).GT.ZERO) THEN
            RE2=VCRAD(OLDSU,ICINDX(BVDWEXC2(I)))
            E2=VCEPS(OLDSU,ICINDX(BVDWEXC2(I))) 
          ELSE
            RE2=VDWR(ITC(IAC(BVDWEXC2(I))))
            E2=EFF(ITC(IAC(BVDWEXC2(I))))
          ENDIF

          CALL XVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                     RE1,E1,RE2,E2,NATOMX,-FACTOR)
          EBNDVDW1=EBNDVDW1-E

          ! ... Add user defined LJ potential
          CALL USRVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                      BVDWSIG(I),BVDWEPS(I),ATTP(I),REPP(I), &
                      NATOMX,FACTOR)

          EBNDVDW1=EBNDVDW1+E

        ENDIF

        ! ... NEWSU-------------------------------------------
        IF (BVDWSURF(I,NEWSU)) THEN

        ! ... Remove standard CHARMM-FF LJ potential between bonding partners
          IF(ICINDX(BVDWEXC1(I)).GT.ZERO) THEN
            RE1=VCRAD(NEWSU,ICINDX(BVDWEXC1(I)))
            E1=VCEPS(NEWSU,ICINDX(BVDWEXC1(I))) 
          ELSE
            RE1=VDWR(ITC(IAC(BVDWEXC1(I))))
            E1=EFF(ITC(IAC(BVDWEXC1(I))))
          ENDIF

          IF(ICINDX(BVDWEXC2(I)).GT.ZERO) THEN
            RE2=VCRAD(NEWSU,ICINDX(BVDWEXC2(I)))
            E2=VCEPS(NEWSU,ICINDX(BVDWEXC2(I))) 
          ELSE
            RE2=VDWR(ITC(IAC(BVDWEXC2(I))))
            E2=EFF(ITC(IAC(BVDWEXC2(I))))
          ENDIF

          CALL XVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                    RE1,E1,RE2,E2,NATOMX,-FACTOR2)
          EBNDVDW2=EBNDVDW2-E

          ! ... Add user defined LJ potential
          CALL USRVDW(E,BVDWEXC1(I),BVDWEXC2(I),X,Y,Z,DX,DY,DZ, &
                      BVDWSIG(I),BVDWEPS(I),ATTP(I),REPP(I), &
                      NATOMX,FACTOR2)
          EBNDVDW2=EBNDVDW2+E

        ENDIF
      ENDDO

      EBNDVDW=FACTOR*EBNDVDW1+FACTOR2*EBNDVDW2
      EU=EU+EBNDVDW

    ENDIF

    RETURN
  END SUBROUTINE RVDW123

  SUBROUTINE USRVDW(E,I,J,X,Y,Z,DX,DY,DZ,R1,E1,EXP1,EXP2, &
                    NATOMX,FACTOR)
    !     
    !     Power modifiable VDW interaction and forces. Based on 
    !     FUNCTION VDW() and uses the distance dependent switching 
    !     function
    !
    !---------------------------------------------------------------- 
    !
    use number
    use stream
    use inbnd
    !
    ! passed variables
    real(chm_real) R1,E1,EXP1,EXP2    
    real(chm_real) E,FACTOR
    INTEGER I,J,NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    ! local variables
    real(chm_real) RX,RY,RZ,S,RM2,SIG,SIG6,SIG12,EN
    real(chm_real) SIGREP,SIGATT
    real(chm_real) EMIN,RUL12,RMIN2,DEN,DF,DEX,DEY,DEZ
    ! Variables for the switching function
    real(chm_real) RIJL,RIJU,C2ONNB,C2OFNB,RUL3,DFN,FUNCT

    E=ZERO

    ! Distances
    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    S=RX*RX+RY*RY+RZ*RZ

    ! Switching function stuff
    C2ONNB=CTONNB*CTONNB
    C2OFNB=CTOFNB*CTOFNB

    IF (S.LT.C2OFNB) THEN
      RMIN2=R1**2
      EMIN=E1
      RM2=1.0_chm_real/S

      IF (S.GT.C2ONNB) THEN
        SIG6=(RMIN2**(EXP1/2.0_CHM_REAL))*(RM2**(EXP1/2.0_CHM_REAL))
        SIG12=(RMIN2**(EXP2/2.0_CHM_REAL))*(RM2**(EXP2/2.0_CHM_REAL))
        RIJL=C2ONNB-S
        RIJU=C2OFNB-S
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*EXP2
        FUNCT=RIJU*RIJU*(RIJU-(EXP1/2.0_CHM_REAL)*RIJL)*RUL3
        DFN=RIJL*RIJU*RUL12
        EN=EMIN*(SIG12-SIG6-SIG6)
        E=(FUNCT*EN)
        DEN=EMIN*RM2*EXP2*(SIG6-SIG12)
        DF=DFN*EN+FUNCT*DEN
      ELSE

        SIG=SQRT(RMIN2*RM2)
        SIGREP=SIG**EXP2
        SIGATT=SIG**EXP1
        E=EMIN*(SIGREP-SIGATT-SIGATT)
        DF=EMIN*RM2*(2.0_CHM_REAL*EXP1*SIGATT-EXP2*SIGREP)
      ENDIF

      ! Calculate force components
      DEX=DF*RX*FACTOR
      DEY=DF*RY*FACTOR
      DEZ=DF*RZ*FACTOR
      ! Add on forces
      DX(I)=DX(I)+ DEX
      DY(I)=DY(I)+ DEY
      DZ(I)=DZ(I)+ DEZ
      DX(J)=DX(J)- DEX
      DY(J)=DY(J)- DEY
      DZ(J)=DZ(J)- DEZ        
 
    ENDIF
    !
    RETURN
  END SUBROUTINE USRVDW


  !====================================================================C
  ! CHANGING NON-BONDED PARAMETERS DURING SURFACE CROSSING             C
  !====================================================================C

  SUBROUTINE CSETCHARGES(FACTOR)
    !
    ! Subroutine which resets the value of the charges during surface 
    ! crossing
    !                                    JD 060104

    use cheq,only:cpcgimg

    use chm_types
    use dimens_fcm
    use number
    use stream
    use psf
    use image
    use bases_fcm


    INTEGER I
    real(chm_real) FACTOR,NEWCHG 

    IF (QCON) THEN

      DO I=1,NCRAT
        CG(ICATOM(I))=FACTOR*ECCHG(OLDSU,I)+ &
                      (ONE-FACTOR)*ECCHG(NEWSU,I)
        IF(PRNLEV.GE.6) WRITE(OUTU,10) ICATOM(I),CG(ICATOM(I))
      ENDDO
    ELSE
      DO I=1,NCRAT
        CG(ICATOM(I))=ECCHG(LOWSU,I)
        IF(PRNLEV.GE.6) WRITE(OUTU,10) ICATOM(I),CG(ICATOM(I))
      ENDDO
    ENDIF

10  FORMAT(' CSETCHARGES> New Charge of atom ',I5,': ',F6.3)

    ! Update the charges on the image atoms (this subroutine is
    ! borrowed from the fluctuating charge implementation)

    CALL CPCGIMG(BIMAG%IMATTR)

    RETURN 
  END SUBROUTINE CSETCHARGES

  !=================================================================

  SUBROUTINE CPDBDUMP(NATOMX,X,Y,Z,TIME)
    use dimens_fcm
    use number
    use stream
    use string
    use psf
    use chutil,only:atomid

    INTEGER NATOMX
    real(chm_real) X(*),Y(*),Z(*),TIME 

    CHARACTER(len=4) HDR,SID,RID,REN,AC,ARID,ATYPEI
    INTEGER IRES,IPT,I,L       

    WRITE(XCROSU,'(A,F8.3)') 'REMARK CROSSING GEOMETRY AT T= ',TIME

    DO IRES=1,NRES
       DO I=IBASE(IRES)+1,IBASE(IRES+1)
          CALL ATOMID(I,SID,RID,REN,AC)
          L=4
          CALL TRIME(RID,L)
          ARID='    '
          IF (L.EQ.4.OR.RID(L:L).GE.'A') THEN
             ARID(4-L+1:4)=RID(1:L)
          ELSE
             ARID(4-L:3)=RID(1:L)
          ENDIF
          ! shift atom names when they exceed 3 characters
          IF (ATYPE(I)(4:4) == ' ') THEN
             ATYPEI=' '//ATYPE(I)(1:3)
          ELSE
             ATYPEI=ATYPE(I)
          ENDIF
          ! the SEGID is written to the last four characters of the line
          !brb..07-FEB-99 Change default occupancy from zero to one
          WRITE(XCROSU, &
               '(A,I5,1X,A4,1X,A4,2X,A4,3X,3F8.3,2F6.2,6X,A4)') &
               'ATOM  ',I,ATYPEI,REN,ARID,X(I),Y(I),Z(I),1.0,0.0 &
               ,SID

       ENDDO
    ENDDO
    ! write END statement for PDB file
    WRITE(XCROSU,'(A)') 'END'
    !       End Procedure WRITE-PDB-FILE
  END SUBROUTINE CPDBDUMP


  ! Routines for memory allocation and deallocation
  ! For simple energy or dynamics calculations
  SUBROUTINE ALLRMD(NATOMX)


    INTEGER NATOMX

    allocate(XX(XTIME,NATOMX), &
              XY(XTIME,NATOMX), &
              XZ(XTIME,NATOMX), &
              XSTEPX(XTIME,NATOMX), &
              XSTEPY(XTIME,NATOMX), &
              XSTEPZ(XTIME,NATOMX), &
              XSTEPOX(XTIME,NATOMX), &
              XSTEPOY(XTIME,NATOMX), &
              XSTEPOZ(XTIME,NATOMX), &
              XGAMMA(XTIME,4*NATOMX), &
              XSEED(XTIME))

    allocate(SAVEDSX(NATOMX), SAVEDSY(NATOMX), &
               SAVEDSZ(NATOMX), SAVEDXX(NATOMX), &
               SAVEDXY(NATOMX), SAVEDXZ(NATOMX), &
               SAVEDSOX(NATOMX), SAVEDSOY(NATOMX), &
               SAVEDSOZ(NATOMX))

    allocate(SAVEDGAMMA(4*NATOMX), &
               CURRENTGAMMA(4*NATOMX))

    RETURN
  END SUBROUTINE ALLRMD

  SUBROUTINE FREERMD

    deallocate(XX,XY,XZ,XSTEPX,XSTEPY, &
                XSTEPZ,XSTEPOX,XSTEPOY,XSTEPOZ, &
                XGAMMA,XSEED)

    deallocate(SAVEDSX,SAVEDSY,SAVEDSZ, &
               SAVEDXX,SAVEDXY,SAVEDXZ,SAVEDSOX, &
               SAVEDSOY,SAVEDSOZ,SAVEDGAMMA, &
               CURRENTGAMMA)

    RETURN
  END SUBROUTINE FREERMD

#else /* (cross)*/
  SUBROUTINE NULL_CROSS
    RETURN
  END SUBROUTINE NULL_CROSS
#endif /* (cross)*/
      
  !=====================================================================
  ! End of module cross
  !=====================================================================
 end module cross




