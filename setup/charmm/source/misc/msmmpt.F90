! Print Level in MS-MMPT:
! Level 2: General information
! Level 6: Detailed information (i.e. weights and energies of states) for each MD-Step
! Level 7: Energy information
! Level 8: Developmental information (Bond connection)
! Level10: Intensive developmental information (Unrecommended)
!
! Usage:
! IF PRMLPE[115]<-100, NON-SWITCHABLE MMPT
! IF 0>PRMLPE[115]>-100, MIXING FUNCTION ONLY
! IF 100>PRMLPE[115]>0, MMPT SWITCHING
! IF PRMLPE[115]>100, MMPT SWITCHING WITH A FIXED DONOR
! ABS(PRMLPE[115])=DELTA_V
! PRMLPE[116]= [0.0,~],   A force constant on activating pseudo-dihedral
!                         energy terms of X-D-A-Y
! PRMLPE[117] consists of xxx.yyy
!                         xxx represents for D-A MAX_DISTANCE
!                         yyy is the range of nonbond cutoff, (0,999.0], shift mode only!!
! PRMLPE[118]= [0.0,~),   A force constant on activating pseudo-angular
!                         energy terms of X-D-H*/H*-A-Y
! PRMLPE[119]= [90.0,180.0],   An equillibrium of pseudo-angular terms
!                         of X-D-H*/H*-A-Y
! PRMLPE[120~123],        refers to which point charge model to employ
!   1) fixed TIP3P(SPC/E) point charge: PRMLPE[120]=PRMLPE[122]=0.D0;
!   2) fixed point charge model with a net charge of +1:
!                         PRMLPE[123] must be negative;
!   3) charge transfer model:
!   3-1) linear function on exponential index:
!                         PRMLPE[122]=0.d0
!                         PRMLPE[120,121]>0.d0
!   3-2) square root function on exponential index:
!                         PRMLPE[120~122]>0.d0
! PRMLPE[124~126],        Sizes of the cubic box as (X,Y,Z)
! PRMLPE[127~137],        External FF parameters acted on H5O2p

! CHARMM Element source/misc/mmpt.src 1.3
! 27/12/2017
! - Corrected NONBONDED List for JNB and IMJNB
! 23/11/2017
! - Options to search PT motifs from the 1st to 2nd solvation shell
! 30/10/2017
! - Major correction: reorder EMSMMPT as calc_E(H5O2p) -> calc_wg(E_i) -> calc_E(nonbond)
! 30/10/2017
! - Re-write SUBROUTINE EWAT with one-dimensioned var r1*r2*(r1^2+r2^2)/(r1^3+r2^3)
! 28/10/2017
! - Introduce an intermediate surface to transform between (k_par,b_par)
!   and (k_wat,b_wat)
! 18/10/2017
! - Using E(H5O2+) to calculate w_i instead of POT_TOT of one EVB state
! 09/10/2017
! - energy/eintern.src modified for image distance
! 26/10/2016
! - extend PT motifs to the secondary solvation shell
! 04/10/2016
! - add new weight functions
! 09/06/2016
! - extend selection motifs up to the 1st solvation shell for ARMD implementation (planned)
!
! 05/10/2015:
! - include correction of VSWITCH
! 09/03/2015:
! - 'intra'-molecular nonbonded interactions are replaced with extra
!   angular bending and torsion
! - include a charge transfer model
!
! 20/10/2014:
! - NONBMMPT is re-arranged as a two-dimensional array
! - implement ARMD Multi-surface to make DH-A motif switchable
!
! CHARMM Element source/misc/mmpt.src 1.3
! 11/11/01:
! - bugs fixed and MMPT can run in parallel with good scaling
!
! 10/11/15:
! - upgrade to Fortran95-compatiable as CHARMM c36a5
! - include a new potentail NLM (see JCP, 133, 064503)
! -
! -
! 09/11/28:
! - Wrapping of mmpt.src and mmpt.fcm into MODULE MMPT.
! - Convert all the fixed size FORTRAN77 arrays to
!   allocatable ones.
! - Fixed the bug that transfering protons have to be the fourth
! - Introducing a new potential LPE with the function
!   form V(r,rho,theta)=\sum V(r,rho)P(costheta)
! -------------------------------------------------------------------

module msmmpt_fcm

  use chm_kinds
  implicit none

  !====================================================================
  ! MMPT GLOBAL VARIABLES
  !====================================================================
  !     SDMPRMU    - Unit number of MMPT SDM parameter file.
  !     SSMPRMU    - Unit number of MMPT SSM parameter file.
  !     ASMPRMU    - Unit number of MMPT ASM parameter file.
  !     QSDM       - FLAG TO INVOKE SDM POTENTIAL FUNCTION
  !     QSSM       - FLAG TO INVOKE SSM POTENTIAL FUNCTION
  !     QASM       - FLAG TO INVOKE ASM POTENTIAL FUNCTION
  !     MORE TO BE ADDED
  !      HBRDNUM -> HYDROGEN BRIDGE COUNTER

  INTEGER, SAVE :: SDMPRMU, SSMPRMU, ASMPRMU, LPEPRMU, NLMPRMU,MSPRMU
  LOGICAL, SAVE :: QSSM, QSDM, QASM, QLPE, QNLM

  INTEGER, SAVE :: NPRMNHN, NPRMOHO, NPRMNHO, NPRMLPE, NPRMNLM, NPRMMUL
  real(chm_real),ALLOCATABLE,dimension(:),save :: PRMNHN, PRMOHO, &
     PRMNHO, PRMLPE, PRMNLM, PRMMUL, XTMP, WDDX,WDDY,WDDZ,ECONSFC,ECONSEQ, &
     GOO,GOH,GHO,GHH
  real(chm_real),ALLOCATABLE,dimension(:,:),save :: DISTMAT
  real(chm_real),dimension(4),save :: DSX,DSY,DSZ,HSX,HSY,HSZ

  INTEGER, SAVE :: HBRDGU, HBRDNUM,  DIHENUM, IMPRNUM, NUMCONS, &
  MAXMTFN0,MAXMTFNUM,MAXMCBNUM, MMPTSTP, MMPTSKP, MMPTCYC, MMPTMAXSTEP, &
  MMPTSMODE,LASTSTEP
  LOGICAL, SAVE :: MMPTSFLAG,MMPTMXFLAG

  integer,ALLOCATABLE,dimension(:), save :: NONBNUM, BONDNUM, ANGLNUM, &
  WATOMM,ECATOMA,ECATOMB,ECATOMC,ECIFLAG
  INTEGER,ALLOCATABLE,DIMENSION(:,:),SAVE :: HBRDATOM, &
  DIHEMMPT, IMPRMMPT, MMPTCHRG, INITCHRG, HBRDFLAG,NBLIST,MMPTNBL
  INTEGER,ALLOCATABLE,DIMENSION(:,:,:),SAVE :: NONBMMPT, ANGLMMPT, BONDMMPT
  CHARACTER*3,ALLOCATABLE,DIMENSION(:),SAVE :: POTTYPE

  real(chm_real), SAVE :: SCLOHO,SCLNHN,SCLNHO

  !declare constant parameters
  real(chm_real), save :: PRM_DELTAV,PXSIZE, &
  PYSIZE,PZSIZE,PRM_NONB_CUTOFF,PRM_EPS_O,PRM_EPS_H,PRM_RMIN_O,PRM_RMIN_H, &
  MMPTTIME0,MMPTTIME1,PRM_NONB_LIST,PRM_NONB_CUTON,PRM_STEPSIZE, &
  PRM_EPS_Cl,PRM_EPS_Na,PRM_RMIN_Cl,PRM_RMIN_Na,PRM_GOFRMAX,PRM_GOFRBIN
  integer, save :: PRM_MAXMTFNUM,PRM_MAXMCBNUM,PRM_DA_LENGTH,PRM_HA_LENGTH, &
  WMODE,PRM_IF_SHELL2
  !internal (flag) parameters
  integer, save :: PRM_INTERNAL_IF_READ_NB,PRM_INTERNAL_CCP
  real(chm_real), save :: PRM_INTERNAL_EINDEX, PRM_INTERNAL_LONG_OO, &
  PRM_INTERNAL_FCONST_DHA_ANGLE,PRM_INTERNAL_EMIN,PRM_INTERNAL_WG_CUT, &
  PRM_INTERNAL_CCA,PRM_INTERNAL_CCB,PRM_INTERNAL_CCC,PRM_INTERNAL_DA_DIST



  !====================================================================
  ! Leave global array definition and enter module subroutine section
  !====================================================================

contains

#if KEY_MSMMPT==1
  !====================================================================
  ! INITIALIZATION
  !====================================================================

  SUBROUTINE MSMMPTINIT

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel
#endif
    use inbnd
    use chm_types
    use contrl
    use bases_fcm

    INTEGER I,J,K,L,BC,MA,MD,MM, MULTI
    INTEGER ATOM1,ATOM2,ATOM3,ATOM4
    real(chm_real) AFCONST,TMIN,DFCONST,PMIN

    INTEGER AC,ITH
    INTEGER DC, IPHI
    INTEGER IC, STATU
!      CHARACTER*4 STRING
    INTEGER HB, VAR1, VAR2
    real(chm_real) RX,RY,RZ,DHA,TMP

!   NEW PARAMETER FILES FOR HYDROGEN AND ACCEPTOR SIDE ATOMS
    INTEGER HPRMU,NAPAR,NDPAR
    INTEGER DUM1, DUM2, DUM3
    logical :: file_exists


#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN
#endif

    !c'mmptinit: give numbers to parameters
!    PRM_MAXMTFNUM=100  !maximum available PT motifs per proton
    PRM_DELTAV=-1.D0 !c'emmpt-parameter: w_i=exp(-(v_i-v_min)/delta_v)
!    PRM_MAXMCBNUM=50

    PRM_DA_LENGTH=100
    PRM_HA_LENGTH=7
    PRM_NONB_LIST=CUTNB
    PRM_NONB_CUTOFF=CTOFNB
    PRM_NONB_CUTON=CTONNB
    PRM_EPS_O=-0.1521d0
    PRM_EPS_H=-0.0460d0
    PRM_EPS_Cl=-0.26d0
    PRM_EPS_Na=-0.26d0
    PRM_RMIN_O=1.7682d0
    PRM_RMIN_H=0.2245d0
    PRM_RMIN_Cl=2.06d0
    PRM_RMIN_Na=1.65d0
    !g(r)
    PRM_GOFRBIN=0.02d0
    PRM_GOFRMAX=10.D0
    PRM_STEPSIZE=0.005D0


    !Internal param
    PRM_INTERNAL_IF_READ_NB=1 !IF ==0, NBond params READ FROM CHARMM (SLOW)
                              !IF <>0, READ FROM external FILE (default)

    PRM_INTERNAL_WG_CUT=0.00000000001d0
                              ! Weight cutoff in Multi-state scheme
    ! A state with weight below PRM_INTERNAL_WG_CUT should be neglected in calculating total pot

    PRM_INTERNAL_EMIN=0.0d0
    ! custom EMIN (if not specified, emin=minimum state energy)
    ! If PRM_INTERNAL_EINDEX<>1.0, PRM_INTERNAL_EMINmust be specified
    ! A recommended value is the minimized energy for the current motif
    ! (for H5O2+, this value is -33.62 under the given MMPT parameters)

    PRM_INTERNAL_EINDEX=1.D0  ! Value==1.0 (default);
    ! If value <> 1.0 (0.0~100.): a beta-index is introduced into
    !calculating weight=exp(-[(E_j-E_min)/delta_E]^beta_idx)
    !(See eq(7.5) in Thesis_by_Z.Xu, var=0.7 is recommended)

    PRM_INTERNAL_LONG_OO=800.D0! Penalty constrain on a too-long D-A distance within a PT motif
    PRM_INTERNAL_FCONST_DHA_ANGLE=400.D0 ! Penalty constrain on a too-large D-H-A angle within a PT motif
    !The above two are applied for energy-conservation purpose

    PRM_INTERNAL_DA_DIST=4.0d0 ! maximum allowed D-A distance to be selected

    PRM_INTERNAL_CCP=1        ! FLAG>=0: DEFINE A FIXED CHARGE FOR H*
                              ! FLAG==(-1): DEFINE CHARGES FOR ALL ATOMS IN MOTIF
                              ! such as XD--D....H*....A--YA
    PRM_INTERNAL_CCA=-0.75    ! CG(D or A)
    PRM_INTERNAL_CCB=0.5      ! CG(H*)
    PRM_INTERNAL_CCC=0.5      ! CG(XD or YA)

    CALL CPU_TIME(MMPTTIME0)


    MMPTSTP=-1
    MMPTSKP=3
    MMPTCYC=0
    MMPTMAXSTEP=99999999
    MMPTMXFLAG=.TRUE.
    MMPTSFLAG=.FALSE.
    INQUIRE(FILE="extra.cycle", EXIST=file_exists)

    if (file_exists) then
      open (unit = 31, file = "extra.cycle")
      READ (31,*) MMPTCYC,MMPTMAXSTEP,MMPTSMODE,PRM_STEPSIZE
      IF (MMPTSMODE.EQ.1) MMPTSFLAG=.TRUE.
      close (31)
    endif


    INQUIRE(FILE="extra.wmod", EXIST=file_exists)

    WMODE=2
    if (file_exists) then
      open (unit = 31, file = "extra.wmod")
      READ (31,*) WMODE
      close (31)
    endif

    INQUIRE(FILE="extra.shell", EXIST=file_exists)

    PRM_IF_SHELL2=1
    if (file_exists) then
      open (unit = 31, file = "extra.shell")
      READ (31,*) PRM_IF_SHELL2
      close (31)
    endif

    INQUIRE(FILE="extra.cons", EXIST=file_exists)

    if (file_exists) then
      open (unit = 31, file = "extra.cons")
      READ (31,*) NUMCONS

      IF (NUMCONS.GT.0) THEN

        ALLOCATE(ECATOMA(NUMCONS),ECATOMB(NUMCONS),ECATOMC(NUMCONS), &
        ECONSFC(NUMCONS),ECONSEQ(NUMCONS),ECIFLAG(NUMCONS))

        DO I = 1,NUMCONS
          READ (31,*) ECATOMA(I),ECATOMB(I),ECATOMC(I), &
          ECONSFC(I),ECONSEQ(I),ECIFLAG(I)
        ENDDO

      ENDIF
    endif

    WRITE(OUTU,*) 'MSMMPT WARNING> THIS VERSION ONLY APPLIES TO WATER MOLECULE'


    if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> MSMMPTINIT HAS BEEN CALLED.'

!   ALLOCATE THE ARRAIES FOR READING PARA
    CALL ALLOCFIRMS

    QSSM = .FALSE.
    QSDM = .FALSE.
    QASM = .FALSE.
    QLPE = .FALSE.
    QNLM = .FALSE.

    !define the number of active protons
    HBRDGU = GTRMI(COMLYN,COMLEN,'UHBR',-1)
    IF (HBRDGU.EQ.-1) THEN
      CALL WrnDie (-3,'<MISCOM>',' No hydrogen bridge file for MMPT')
    ELSE
      HBRDNUM=0
      STATU=0
      DO WHILE (.TRUE.)
         HBRDNUM=HBRDNUM+1
         READ(UNIT=HBRDGU, FMT=*, IOSTAT=STATU) HBRDATOM(HBRDNUM,1),&
         HBRDATOM(HBRDNUM,2),HBRDATOM(HBRDNUM,3),POTTYPE(HBRDNUM)
         IF(STATU.NE.0) EXIT
      ENDDO
    ENDIF
    HBRDNUM=HBRDNUM-1

    !define maximum number of motifs
    MAXMTFN0=2*3*5
    MAXMTFNUM=MAXMTFN0*3*5
    MAXMCBNUM=MAXMTFNUM**HBRDNUM


    if (PRNLEV >= 2) then
       WRITE(OUTU,*) ' '
       WRITE(OUTU,*) 'MSMMPT> FOUND',HBRDNUM,' HYDROGEN BOND(S) IN FILE:'
    endif
    DO J=1,HBRDNUM
      if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT>',HBRDATOM(J,1), &
      HBRDATOM(J,2),HBRDATOM(J,3),POTTYPE(J)
      IF (POTTYPE(J) .EQ. 'SSM') THEN
        QSSM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'SDM') THEN
        QSDM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'ASM') THEN
        QASM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'LPE') THEN
        QLPE = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'NLM') THEN
        QNLM = .TRUE.
      ENDIF
    ENDDO

!   PLAUSIBILITY CHECK: DO ATOMS HAVE THE POSSIBLE DONOR, HYDROGEN AND
!   ACCEPTOR ATOM TYPE (IE FIRST AND LAST MUST BE OXYGEN OR NITROGEN
!   ATOMS AND MIDDLE MUST BE A HYDROGEN ATOM)
    DO I=1,HBRDNUM
      IF (ATYPE(HBRDATOM(I,1))(1:1) .EQ. 'N' .or. &
        ATYPE(HBRDATOM(I,1))(1:1) .EQ. 'O' ) THEN
        CONTINUE
      ELSE
        WRITE(OUTU,*) ' ',ATYPE(HBRDATOM(I,1))(1:1)
        WRITE(OUTU,*) 'MSMMPT> FIRST ATOM IN HYDROGEN BOND IS NOT A DONOR ATOM'
        STOP
      ENDIF

      IF (ATYPE(HBRDATOM(I,3))(1:1) .EQ. 'N' .OR. &
        ATYPE(HBRDATOM(I,3))(1:1) .EQ. 'O' ) THEN
        CONTINUE
      ELSE
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> THIRD ATOM IN HYDROGEN BOND IS NOT AN ACCEPTOR ATOM'
        STOP
      ENDIF

      IF (ATYPE(HBRDATOM(I,2))(1:1) .EQ. 'H') THEN
        CONTINUE
      ELSE
        WRITE(OUTU,*) 'MSMMPT> SECOND ATOM IN HYDROGEN BOND IS NOT A HYDROGEN ATOM'
        STOP
      ENDIF
    ENDDO

!      PROCESS COMMAND LINE
!      READ IN PARAMETER SET / Symmetric Double Minimum (SDM)
!      DERIVED FROM N-H...N BOND IN N2H4+ PROTOTYPE PSF; HENCE NHN<->SDM
      IF (QSDM) THEN
        SDMPRMU = GTRMI(COMLYN,COMLEN,'USDM',-1)
        IF (SDMPRMU.EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No SDM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNHN
            READ(SDMPRMU, *) PRMNHN(I)
20        ENDDO
          if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> SDM parameter set has been read.'
          IF(PRNLEV.GE.7) THEN
            DO I=1, NPRMNHN
              WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMNHN(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

!      READ IN PARAMETER SET / Symmetric Single Minimum (SSM)
!      DERIVED FROM O-H...O BOND IN O2H5+ PROTOTYPE PSF; HENCE OHO<->SSM

      IF (QSSM) THEN
        SSMPRMU = GTRMI(COMLYN,COMLEN,'USSM',-1)
        IF (SSMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No SSM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMOHO
            READ(SSMPRMU, *) PRMOHO(I)
21        ENDDO
          if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> SSM parameter set has been read.'
          IF(PRNLEV.GE.7) THEN
            DO I=1, NPRMOHO
              WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMOHO(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

!      READ IN PARAMETER SET / Symmetric Single Minimum (SSM)
!      DERIVED FROM O-H...O BOND IN NH4OH2+ PROTOTYPE PSF; HENCE NHO<->ASM
      IF (QASM) THEN
        ASMPRMU = GTRMI(COMLYN,COMLEN,'UASM',-1)
        IF (ASMPRMU.EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No ASM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNHO
            READ(ASMPRMU, *) PRMNHO(I)
          ENDDO
          if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> ASM parameter set has been read.'
          IF(PRNLEV.GE.7) THEN
            DO I=1, NPRMNHO
              WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMNHO(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

      IF (QLPE) THEN
        LPEPRMU = GTRMI(COMLYN,COMLEN,'ULPE',-1)
        IF (LPEPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No LPE parameter file for MMPT')
        ELSE
          DO I=1 , NPRMLPE
            READ(LPEPRMU, *) PRMLPE(I)
          ENDDO
          if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> LPE parameter set has been read.'
          IF(PRNLEV.GE.7) THEN
            DO I=1, NPRMLPE
              WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMLPE(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

      IF (QNLM) THEN
        NLMPRMU = GTRMI(COMLYN,COMLEN,'UNLM',-1)
        IF (NLMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No NLM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNLM
            READ(NLMPRMU, *) PRMNLM(I)
          ENDDO
          if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> NLM parameter set has been read.'
          IF(PRNLEV.GE.7) THEN
            DO I=1, NPRMNLM
              WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMNLM(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF


      !READ MS-MMPT PARAMS
      MSPRMU = GTRMI(COMLYN,COMLEN,'UMUL',-1)
      IF (MSPRMU .EQ.-1) THEN
        CALL WrnDie (-3,'<MISCOM>',' No Multi-state parameter file for MMPT')
      ELSE
        DO I=1 , NPRMMUL
          READ(MSPRMU, *) PRMMUL(I)
        ENDDO
        if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> Multi-state parameter set has been read.'
        IF(PRNLEV.GE.7) THEN
          DO I=1, NPRMMUL
            WRITE(OUTU,*) 'MSMMPT>  PARAM', I, PRMMUL(I)
          ENDDO
        ENDIF


        PRM_DELTAV=PRMMUL(1)


        IF (PRMMUL(2).LT.0.D0) THEN
          PRMMUL(2)=0.d0
        ENDIF

        IF (PRMMUL(3).lt.0.D0) THEN
          PRMMUL(3)=0.D0
        ENDIF

        IF (PRMMUL(4).LT.90.D0 .OR. PRMMUL(4).GT.180.D0) THEN
          PRMMUL(4)=180.D0
        ENDIF

        IF (PRMMUL(22).LE.0.D0) THEN
          PXSIZE=0.d0
        ELSE
          PXSIZE=PRMMUL(22)
          PBCXSIZE=PRMMUL(22)
        ENDIF

        IF (PRMMUL(23).LE.0.D0) THEN
          PYSIZE=0.d0
        ELSE
          PYSIZE=PRMMUL(23)
          PBCYSIZE=PRMMUL(23)
        ENDIF

        IF (PRMMUL(24).LE.0.D0) THEN
          PZSIZE=0.d0
        ELSE
          PZSIZE=PRMMUL(24)
          PBCZSIZE=PRMMUL(24)
        ENDIF

        PRM_EPS_O=PRMMUL(18)
        PRM_RMIN_O=PRMMUL(19)
        PRM_EPS_H=PRMMUL(20)
        PRM_RMIN_H=PRMMUL(21)

      ENDIF

      SCLOHO = GTRMF(COMLYN,COMLEN,'SSMS',ONE)
      IF (SCLOHO .EQ. ONE ) THEN
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> SSM ENERGIES WITHOUT SCALING'
      ELSE
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> SSM SCALING FACTOR IS ', SCLOHO
      ENDIF
      SCLNHN = GTRMF(COMLYN,COMLEN,'SDMS',ONE)
      IF (SCLNHN .EQ. ONE ) THEN
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> SDM ENERGIES WITHOUT SCALING'
      ELSE
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> SDM SCALING FACTOR IS ', SCLNHN
      ENDIF
      SCLNHO = GTRMF(COMLYN,COMLEN,'ADMS',ONE)
      IF (SCLNHO .EQ. ONE ) THEN
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> ASM ENERGIES WITHOUT SCALING'
      ELSE
         if (PRNLEV >= 2) WRITE(OUTU,*)'MSMMPT> ASM SCALING FACTOR IS ', SCLNHO
      ENDIF

!     ALLOCATE THE ARRAIES FOR MMPT BONDS, ANGLES, ETC.
      CALL ALLOCSECMS(HBRDNUM)

!     CHECK NONBONDED CUTOFF
      IF (PRM_NONB_LIST.GT.PRM_NONB_CUTOFF .AND. &
      PRM_NONB_CUTOFF.GT.PRM_NONB_CUTON) THEN
        CONTINUE
      ELSE
        CALL WrnDie (-3,'<MISCOM>', &
        'MSMMPTINIT> WRONG NONBONDED CUTOFF')
      ENDIF
!     INITIALIZE MMPTNBL
      DO I=1,NATOM
        DO J=1,NATOM
          MMPTNBL(I,J)=0
        ENDDO
      ENDDO

      !g(r)
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        TMP=PXSIZE
        IF (PYSIZE.LT.TMP) TMP=PYSIZE
        IF (PZSIZE.LT.TMP) TMP=PZSIZE
        IF (TMP.LT.PRM_GOFRMAX) PRM_GOFRMAX=TMP
      ENDIF

      !DEFINE NUMBER OF ATOMS FOR GLOBAL USE
      MMPTNATOMX=NATOM

!     INITIALIZE THE DHA LIST
      DO HB=1, HBRDNUM
        HBRDFLAG(HB,1)=HBRDATOM(HB,1)
        HBRDFLAG(HB,2)=HBRDATOM(HB,2)
        HBRDFLAG(HB,3)=HBRDATOM(HB,3)
        HBRDFLAG(HB,4)=-1
      ENDDO

!      GET BOND NUMBERS OF DONOR AND H ATOM BONDS
!c'mmpt_fcm: may not allow 'BOND H O' in definition
      DO HB=1, HBRDNUM
        BONDNUM(HB)=0
        DO MM=1, NBOND
          IF (HBRDATOM(HB,1).EQ.IB(MM) .AND. &
          HBRDATOM(HB,2).EQ.JB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            IF (BONDNUM(HB).GT.1) THEN
              BONDMMPT(HB,BONDNUM(HB),1)=BONDMMPT(HB,1,1)
              BONDMMPT(HB,BONDNUM(HB),2)=BONDMMPT(HB,1,2)
              BONDMMPT(HB,BONDNUM(HB),3)=BONDMMPT(HB,1,3)
              BONDMMPT(HB,BONDNUM(HB),4)=BONDMMPT(HB,1,4)
            ENDIF
            BONDMMPT(HB,1,1)=MM
            BONDMMPT(HB,1,2)=IB(MM)
            BONDMMPT(HB,1,3)=JB(MM)
            BONDMMPT(HB,1,4)=1
            GOTO 530
          ELSE IF (HBRDATOM(HB,1).EQ.JB(MM) .AND. &
          HBRDATOM(HB,2).EQ.IB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            IF (BONDNUM(HB).GT.1) THEN
              BONDMMPT(HB,BONDNUM(HB),1)=BONDMMPT(HB,1,1)
              BONDMMPT(HB,BONDNUM(HB),2)=BONDMMPT(HB,1,2)
              BONDMMPT(HB,BONDNUM(HB),3)=BONDMMPT(HB,1,3)
              BONDMMPT(HB,BONDNUM(HB),4)=BONDMMPT(HB,1,4)
            ENDIF
            BONDMMPT(HB,1,1)=MM
            BONDMMPT(HB,1,2)=JB(MM)
            BONDMMPT(HB,1,3)=IB(MM)
            BONDMMPT(HB,1,4)=1
            GOTO 530
          ENDIF

          IF (HBRDATOM(HB,1).EQ.IB(MM) .AND. &
          HBRDATOM(HB,2).NE.JB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            BONDMMPT(HB,BONDNUM(HB),1)=MM
            BONDMMPT(HB,BONDNUM(HB),2)=IB(MM)
            BONDMMPT(HB,BONDNUM(HB),3)=JB(MM)
            BONDMMPT(HB,BONDNUM(HB),4)=2
            GOTO 530
          ELSE IF (HBRDATOM(HB,1).EQ.JB(MM) .AND. &
          HBRDATOM(HB,2).NE.IB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            BONDMMPT(HB,BONDNUM(HB),1)=MM
            BONDMMPT(HB,BONDNUM(HB),2)=JB(MM)
            BONDMMPT(HB,BONDNUM(HB),3)=IB(MM)
            BONDMMPT(HB,BONDNUM(HB),4)=2
            GOTO 530
          ENDIF

          IF (HBRDATOM(HB,3).EQ.IB(MM) .AND. &
          HBRDATOM(HB,2).NE.JB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            BONDMMPT(HB,BONDNUM(HB),1)=MM
            BONDMMPT(HB,BONDNUM(HB),2)=IB(MM)
            BONDMMPT(HB,BONDNUM(HB),3)=JB(MM)
            BONDMMPT(HB,BONDNUM(HB),4)=-2
            GOTO 530
          ELSE IF (HBRDATOM(HB,3).EQ.JB(MM) .AND. &
          HBRDATOM(HB,2).NE.IB(MM)) THEN
            BONDNUM(HB)=BONDNUM(HB)+1
            BONDMMPT(HB,BONDNUM(HB),1)=MM
            BONDMMPT(HB,BONDNUM(HB),2)=JB(MM)
            BONDMMPT(HB,BONDNUM(HB),3)=IB(MM)
            BONDMMPT(HB,BONDNUM(HB),4)=-2
            GOTO 530
          ENDIF

530  CONTINUE
        ENDDO


        IF (BONDMMPT(HB,1,4).NE.1) CALL WrnDie (-3,'<MISCOM>', &
        'MSMMPTINIT> Failed to define D-H in BONDMMPT')

      ENDDO

!      GET ALL ANGLES AND ANGLE NUMBERS CONNECTED TO H BRIDGE
      DO HB=1,HBRDNUM
        ANGLNUM(HB)=0
        DO ITH=1, NTHETA
!      GET THE ANGLES WHICH NEED TO BE SWITCHED off AT
!      DONOR SIDE (SEARCH FOR ANGLES IN PSF WHICH
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
!          IF (HBRDATOM(HB,2).EQ.IT(ITH) &
!              .OR.HBRDATOM(HB,2).EQ.JT(ITH) &!H is never in the mid
!              .OR.HBRDATOM(HB,2).EQ.KT(ITH)) THEN
          !FLAG: XD-D-H = 1; XA-A-H = -1; XD-D-XD2 = 2; XA-A-XA2 = -2;
          IF ((HBRDATOM(HB,2).EQ.IT(ITH) .OR. HBRDATOM(HB,2).EQ.KT(ITH))) THEN
            IF (HBRDATOM(HB,1).EQ.JT(ITH)) THEN
              ANGLNUM(HB)=ANGLNUM(HB)+1
              ANGLMMPT(HB,ANGLNUM(HB),1)=ITH ! >0 means existing angular bond
              ANGLMMPT(HB,ANGLNUM(HB),2)=IT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),3)=JT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),4)=KT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),5)=1
            ENDIF
          ELSE
            IF (HBRDATOM(HB,1).EQ.JT(ITH)) THEN
              ANGLNUM(HB)=ANGLNUM(HB)+1
              ANGLMMPT(HB,ANGLNUM(HB),1)=ITH ! >0 means existing angular bond
              ANGLMMPT(HB,ANGLNUM(HB),2)=IT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),3)=JT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),4)=KT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),5)=2
            ELSE IF (HBRDATOM(HB,3).EQ.JT(ITH)) THEN
              ANGLNUM(HB)=ANGLNUM(HB)+1
              ANGLMMPT(HB,ANGLNUM(HB),1)=ITH ! >0 means existing angular bond
              ANGLMMPT(HB,ANGLNUM(HB),2)=IT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),3)=JT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),4)=KT(ITH)
              ANGLMMPT(HB,ANGLNUM(HB),5)=-2
            ENDIF
          ENDIF
        ENDDO
      ENDDO
!      GET THE ANGLES WHICH NEED TO BE SWITCHED on AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT ARE
!      BONDED TO DONOR ATOM. USE TO SET UP NEW ANGLE)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
      DO HB=1,HBRDNUM
        DO MM=1, NBOND
          IF (HBRDATOM(HB,3).EQ.IB(MM)) THEN
             ANGLNUM(HB)=ANGLNUM(HB)+1
             ANGLMMPT(HB,ANGLNUM(HB),1)=0 ! =0 means non-existing angular bond
             ANGLMMPT(HB,ANGLNUM(HB),2)=JB(MM)
             ANGLMMPT(HB,ANGLNUM(HB),3)=IB(MM)
             ANGLMMPT(HB,ANGLNUM(HB),4)=HBRDATOM(HB,2)
             ANGLMMPT(HB,ANGLNUM(HB),5)=-1
          ELSE IF (HBRDATOM(HB,3).EQ.JB(MM)) THEN
             ANGLNUM(HB)=ANGLNUM(HB)+1
             ANGLMMPT(HB,ANGLNUM(HB),1)=0
             ANGLMMPT(HB,ANGLNUM(HB),2)=IB(MM)
             ANGLMMPT(HB,ANGLNUM(HB),3)=JB(MM)
             ANGLMMPT(HB,ANGLNUM(HB),4)=HBRDATOM(HB,2)
             ANGLMMPT(HB,ANGLNUM(HB),5)=-1
          ENDIF
        ENDDO
      ENDDO


!      WE NEED TO GET THE CORRECT FORCE CONSTANTS.
!      SEARCH FOR ANGLES OF THE SAME KIND I.E. SAME
!      PARAMETER TYPE CODE (IAC) IN THE SAME ORDER.
!      TO DO: IF YOU DO NOT HAVE ANY YOU HAVE TO FIND OUT HOW
!      THE FORCE CONSTANT TABLES ARE CREATED AFTER
!      SYSTEMS HAS BEEN INITIALIZED. PSF -> ICT
    DO HB=1,HBRDNUM
      DO I=1,ANGLNUM(HB)
        IF (ANGLMMPT(HB,I,1).EQ.0) THEN
          DO ITH=1, NTHETA
            IF (IAC(ANGLMMPT(HB,I,2)).EQ.IAC(IT(ITH))        &
                 .AND.IAC(ANGLMMPT(HB,I,3)).EQ.IAC(JT(ITH))   &
                 .AND.IAC(ANGLMMPT(HB,I,4)).EQ.IAC(KT(ITH))) THEN
                ANGLMMPT(HB,I,1)=ITH

                goto 470
            ENDIF
            IF (IAC(ANGLMMPT(HB,I,4)).EQ.IAC(IT(ITH))       &
                 .AND.IAC(ANGLMMPT(HB,I,3)).EQ.IAC(JT(ITH))  &
                 .AND.IAC(ANGLMMPT(HB,I,2)).EQ.IAC(KT(ITH))) THEN
                ANGLMMPT(HB,I,1)=ITH

                goto 470
            ENDIF
          ENDDO

          CALL WrnDie (-3,'<MISCOM>','MSMMPTINIT> Failed to &
          find reference angles')

470       continue
        ENDIF
      ENDDO
    ENDDO

!!need to be implemented for dihedral & improper
!     GET ALL DIHEDRAL ANGLES CONNECTED TO H BRIDGE
      DIHENUM=0
!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT
!      DONOR SIDE (SEARCH FOR DIHEDRALS IN PSF WHICH
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
      DO HB=1,HBRDNUM
        DO IPHI=1, NPHI
          IF (HBRDATOM(HB,2).EQ.IP(IPHI)      &
              .OR.HBRDATOM(HB,2).EQ.JP(IPHI)  &
              .OR.HBRDATOM(HB,2).EQ.KP(IPHI)  &
              .OR.HBRDATOM(HB,2).EQ.LP(IPHI)) THEN
             DIHENUM=DIHENUM+1
             DIHEMMPT(DIHENUM,1)=IPHI
             DIHEMMPT(DIHENUM,2)=IP(IPHI)
             DIHEMMPT(DIHENUM,3)=JP(IPHI)
             DIHEMMPT(DIHENUM,4)=KP(IPHI)
             DIHEMMPT(DIHENUM,5)=LP(IPHI)
             DIHEMMPT(DIHENUM,6)=1
          ENDIF
        ENDDO
      ENDDO

!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT FORM
!      ANGLE WITH DONOR ATOM. USE TO SET UP NEW DIHEDRAL)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
     DO HB=1,HBRDNUM
         DO ITH=1,NTHETA
           IF (HBRDATOM(HB,3).EQ.IT(ITH)) THEN
               DIHENUM=DIHENUM+1
               DIHEMMPT(DIHENUM,1)=0
               DIHEMMPT(DIHENUM,2)=KT(ITH)
               DIHEMMPT(DIHENUM,3)=JT(ITH)
               DIHEMMPT(DIHENUM,4)=IT(ITH)
               DIHEMMPT(DIHENUM,5)=HBRDATOM(HB,2)
               DIHEMMPT(DIHENUM,6)=-1
!      NOT SURE WHAT HAPPENS IF ACCEPTOR ATOM IS SOMETHING
!      ELSE THAN FIRST OR LAST ATOM IN ORDER OF ANGLE DEFINITION
! ... IF NEEDED USE THIS WITH MODIFIED ORDER
           ELSE IF (HBRDATOM(HB,3).EQ.KT(ITH)) THEN
               DIHENUM=DIHENUM+1
               DIHEMMPT(DIHENUM,1)=0
               DIHEMMPT(DIHENUM,2)=IT(ITH)
               DIHEMMPT(DIHENUM,3)=JT(ITH)
               DIHEMMPT(DIHENUM,4)=KT(ITH)
               DIHEMMPT(DIHENUM,5)=HBRDATOM(HB,2)
               DIHEMMPT(DIHENUM,6)=-1
           ENDIF
         ENDDO
      ENDDO

!      TO GET THE FORCE CONSTANTS
      DO I=1,DIHENUM
        IF (DIHEMMPT(I,1).EQ.0) THEN
          DO IPHI=1, NPHI
            IF (IAC(DIHEMMPT(I,2)).EQ.IAC(IP(IPHI))        &
                .AND.IAC(DIHEMMPT(I,3)).EQ.IAC(JP(IPHI))   &
                .AND.IAC(DIHEMMPT(I,4)).EQ.IAC(KP(IPHI))   &
                .AND.IAC(DIHEMMPT(I,5)).EQ.IAC(LP(IPHI)))  &
               THEN
               DIHEMMPT(I,1)=IPHI
            ENDIF
          ENDDO
        ENDIF
      ENDDO


!     GET ALL IMPROPER DIHEDRAL ANGLES CONNECTED TO H BRIDGE
      IMPRNUM=0
      DO HB=1,HBRDNUM
        DO IPHI=1, NIMPHI
!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT
!      DONOR SIDE (SEARCH FOR DIHEDRALS IN PSF WHICH
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
          IF (HBRDATOM(HB,2).EQ.IM(IPHI)       &
              .OR.HBRDATOM(HB,2).EQ.JM(IPHI)   &
              .OR.HBRDATOM(HB,2).EQ.KM(IPHI)   &
              .OR.HBRDATOM(HB,2).EQ.LM(IPHI)) THEN
             IMPRNUM=IMPRNUM+1
             IMPRMMPT(IMPRNUM,1)=IPHI
             IMPRMMPT(IMPRNUM,2)=IM(IPHI)
             IMPRMMPT(IMPRNUM,3)=JM(IPHI)
             IMPRMMPT(IMPRNUM,4)=KM(IPHI)
             IMPRMMPT(IMPRNUM,5)=LM(IPHI)
             IMPRMMPT(IMPRNUM,6)=1
          ENDIF
        ENDDO
      ENDDO

!      GET IMPROPERS WHICH NEED TO BE SWITCHED AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT FORM
!      ANGLE WITH DONOR ATOM AND ATOM IS MIDDLE ATOM.
!.... USE ANGLE ATOMS TO SET UP NEW IMPROPER)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
      DO HB=1,HBRDNUM
        DO ITH=1,NTHETA
          IF (HBRDATOM(HB,3).EQ.JT(ITH)) THEN
             IMPRNUM=IMPRNUM+1
             IMPRMMPT(IMPRNUM,1)=0
             IMPRMMPT(IMPRNUM,2)=JT(ITH)
             IMPRMMPT(IMPRNUM,3)=IT(ITH)
             IMPRMMPT(IMPRNUM,4)=KT(ITH)
             IMPRMMPT(IMPRNUM,5)=HBRDATOM(HB,2)
             IMPRMMPT(IMPRNUM,6)=-1
          ENDIF
        ENDDO
      ENDDO

!      TO GET THE FORCE CONSTANTS
      DO I=1,IMPRNUM
        IF (IMPRMMPT(I,1).EQ.0) THEN
          DO IPHI=1, NIMPHI
            IF (IAC(IMPRMMPT(I,2)).EQ.IAC(IM(IPHI))       &
                 .AND.IAC(IMPRMMPT(I,3)).EQ.IAC(JM(IPHI))  &
                 .AND.IAC(IMPRMMPT(I,4)).EQ.IAC(KM(IPHI))  &
                 .AND.IAC(IMPRMMPT(I,5)).EQ.IAC(LM(IPHI))) &
                 THEN
                IMPRMMPT(I,1)=IPHI
            ENDIF
          ENDDO
        ENDIF
      ENDDO


!      IF H ATOM IS CONNECTED TO ATOMS ON ACCEPTOR SIDE SO THAT UNCOMMON
!      INTERACTIONS BETWEEN ATOM TYPES OCCUR THE USER NEEDS TO PROVIDE
!      THEM: READ MISSING PARAMETER FROM FILE
!      ANGLES:
!      THEY NEED TO BE ADDED TO CHARMMS INTERNAL STORAGE OF ANGLE
!      PARAMETER.
!      NCT NUMBER OF ANGLE PARAMETER
!      CTC FORCE CONSTANT
!      CTB EQUILIBRIUM ANGLE
!      THEY ARE ACCESSED TROUGH ANGLE CODE LOOK UP ARRAY -> ICT
!     .WHERE ALL NTHETA ANGLE CODES ARE STORED
!      ICT(X) -> CTC(ICT(X)) AND CTB(ICT(X))
!      DIHEDRALS:

      HPRMU = GTRMI(COMLYN,COMLEN,'UHPM',-1)
      IF (HPRMU.EQ.-1) THEN
         if (PRNLEV >= 2) WRITE(OUTU,*) 'MSMMPT> NO HBOND PARAMETER FILE PROVIDED!'
      ELSE
!     SET NUMBER OF MISSING ANGLE RESP DIHEDRAL PARAMETER
        MA=1
        MD=1
!     READ NEW ANGLE PARAMETER
        if (PRNLEV >= 2) WRITE(OUTU,*) 'READING ANGLE PARMS FROM UNIT', HPRMU
!       todo: check if file exists
        READ(UNIT=HPRMU,FMT=*) NAPAR
        if (PRNLEV >= 2) WRITE(OUTU,*) 'NUMBER OF ANGLE PARMS TO BE READ:',NAPAR
        DO I=1,NAPAR
          if (PRNLEV >= 2) WRITE(OUTU,*) 'READING LINE',I
          READ(UNIT=HPRMU,FMT=*) ATOM1,ATOM2,ATOM3, AFCONST, TMIN
          if (PRNLEV >= 2) WRITE(UNIT=OUTU,FMT=*) ATOM1,ATOM2,ATOM3, AFCONST, TMIN
          DO HB=1,HBRDNUM
            DO J=1,ANGLNUM(HB)
              IF (ATOM1.EQ.IAC(ANGLMMPT(HB,J,2)).AND. &
                  ATOM2.EQ.IAC(ANGLMMPT(HB,J,3)).AND.  &
                  ATOM3.EQ.IAC(ANGLMMPT(HB,J,4))) THEN
  !     SET ANGLE NUMBER
                    ANGLMMPT(HB,J,1)=NTHETA+MA
  !     EXTEND ICT
                    ICT(NTHETA+MA)=NCT+MA
  !     EXTEND CTC AND CTB
                    CTC(NCT+MA)=AFCONST
                    CTB(NCT+MA)=TMIN*PI/180.D0
  !     INCREASE MISSING ANGLE COUNTER
                    MA=MA+1
              ENDIF
            ENDDO
          ENDDO
        ENDDO
!     READ DIHEDRAL PARAMETER
        NDPAR=0
        READ(UNIT=HPRMU,FMT=*,END=680) NDPAR
        if (PRNLEV >= 2) WRITE(OUTU,*) 'NUMBER OF DIHE PARMS TO BE READ:',NDPAR
        DO I=1,NDPAR
          READ(UNIT=HPRMU,FMT=*,END=680) ATOM1,ATOM2,ATOM3,ATOM4, &
          DFCONST, MULTI, PMIN
          if (PRNLEV >= 2) WRITE(UNIT=OUTU,FMT=*) &
          ATOM1,ATOM2,ATOM3,ATOM4, DFCONST, MULTI, PMIN
          DO J=1,DIHENUM
            IF (ATOM1.EQ.IAC(DIHEMMPT(J,2)).AND.                   &
                  ATOM2.EQ.IAC(DIHEMMPT(J,3)).AND.                  &
                  ATOM3.EQ.IAC(DIHEMMPT(J,4)).AND.                  &
                  ATOM4.EQ.IAC(DIHEMMPT(J,5))) THEN
!     SET DIHEDRAL NUMBER
                  DIHEMMPT(J,1)=NPHI+MD
!     EXTEND ICP
                  ICP(NPHI+MD)=NCP+MD
!     EXTEND CPC, CPD, AND CPB
                  CPC(NCP+MD)=DFCONST
                  CPD(NCP+MD)=MULTI
                  CPB(NCP+MD)=PMIN*PI/180.D0
!     INCREASE MISSING DIHEDRAL COUNTER
                  MD=MD+1
            ENDIF
          ENDDO
        ENDDO
      ENDIF

!95   FORMAT(I4)
!96   FORMAT(3I4,2F8.4)
!97   FORMAT(4I4,F8.4,I4,F8.4)
!98   FORMAT(4I4,2X,F8.4,2X,I4,2X,F8.4)
680   CONTINUE


!      IF NO FORCE PARAMETER HAVE BEEN FOUND THEN STOP
      DO HB=1,HBRDNUM
        DO I=1,ANGLNUM(HB)
           IF (ANGLMMPT(HB,I,1).EQ.0) THEN
              if (PRNLEV >= 2) then
                 WRITE(OUTU,*) '<MSMMPTINIT> COULD NOT FIND ANGLE PARAMETER.'
                 WRITE(OUTU,*) '           PSF AND TYPE:',       &
                      ANGLMMPT(HB,I,2),' ',ATYPE(ANGLMMPT(HB,I,2)),' ',  &
                      IAC(ANGLMMPT(HB,I,2)),                         &
                      ANGLMMPT(HB,I,3),' ',ATYPE(ANGLMMPT(HB,I,3)),' ',  &
                      IAC(ANGLMMPT(HB,I,3)),                         &
                      ANGLMMPT(HB,I,4),' ',ATYPE(ANGLMMPT(HB,I,4)),' ',  &
                      IAC(ANGLMMPT(HB,I,4))
                 WRITE(OUTU,*) '           PLEASE PROVIDE PARAMETER!'
              endif
          ENDIF
        ENDDO
      ENDDO


      DO I=1,DIHENUM
        IF (DIHEMMPT(I,1).EQ.0) THEN
           if (PRNLEV >= 2) then
              WRITE(OUTU,*) '<MSMMPTINIT> COULD NOT FIND DIHEDRAL PARAMETER'
              WRITE(OUTU,*) '           PSF AND TYPE:',       &
                   DIHEMMPT(I,2),' ',ATYPE(DIHEMMPT(I,2)),' ',  &
                   IAC(DIHEMMPT(I,2)),                         &
                   DIHEMMPT(I,3),' ',ATYPE(DIHEMMPT(I,3)),' ',  &
                   IAC(DIHEMMPT(I,3)),                         &
                   DIHEMMPT(I,4),' ',ATYPE(DIHEMMPT(I,4)),' ',  &
                   IAC(DIHEMMPT(I,4)),                         &
                   DIHEMMPT(I,5),' ',ATYPE(DIHEMMPT(I,5)),' ',  &
                   IAC(DIHEMMPT(I,5))
              WRITE(OUTU,*) '           PLEASE PROVIDE PARAMETER!'
           endif
        ENDIF
      ENDDO

      DO HB=1,HBRDNUM
        DO I=1,ANGLNUM(HB)
         IF (ANGLMMPT(HB,I,1).EQ.0) THEN
            STOP
         ENDIF
        ENDDO
      ENDDO

      DO I=1,DIHENUM
         IF (DIHEMMPT(I,1).EQ.0) THEN
            STOP
         ENDIF
      ENDDO

!     INITIALIZATION OF NON-BOND INTERACTIONS ARE CARRIED OUT
!     IN SEPERATED SOUBROUTINE NBONDINIT
      CALL NBONDINITMS(BNBND%INBLO,BNBND%JNB)

!     FCM
!     FLUCTUATING CHARGE MODEL - COMMENT OUT IN STARNARD MMPT
!      STORE PSF CHARGES FOR DIPOLMOMENT CALCULATIONS
!       WRITE(OUTU,*) '    '
!       WRITE(OUTU,*) 'MSMMPT> STORED INITIAL PSF CHARGES FOR DIPOL  &
!       CALCULATION'
!       DO I=1, HBRDNUM
!          INITCHRG(I,1)=CG(HBRDATOM(I,1))
!          INITCHRG(I,2)=CG(HBRDATOM(I,2))
!          INITCHRG(I,3)=CG(HBRDATOM(I,3))
!          WRITE(OUTU,*)'', INITCHRG(I,1),INITCHRG(I,2), INITCHRG(I,3)
!       ENDDO


!     EXCLUDE IMPROPERS FOR NOW. SET IMPROPER NUMBER TO ZERO
      IMPRNUM=0
!      WRITE(OUTU,*) 'MSMMPT> IMPROPER DIHEDRALS HAVE BEEN SWITCHED OFF!'

!     PRINT OUT THE MSMMPTINIT INFORMATIONS
      if (PRNLEV >= 2) then
         WRITE(OUTU,*) ' '
         WRITE(OUTU,*) 'MSMMPT> ENERGIES AND FORCES OF FOLLOWING '
         WRITE(OUTU,*) '      INTERACTIONS WILL BE REMOVED OR MODIFIED'
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      BONDED TERMS: FLAG  1  MEANS TERM EXISTS'
         WRITE(OUTU,*) '                    FLAG -1  MEANS TERM IS NEW'
         WRITE(OUTU,*) '      BONDS:'
         WRITE(OUTU,*) '      NO    ATOM I     ATOM J   FLAG'
         DO I=1, HBRDNUM
          DO J=1,BONDNUM(I)
            WRITE(OUTU,811) BONDMMPT(I,J,1),BONDMMPT(I,J,2), &
            ATYPE(BONDMMPT(I,J,2)),BONDMMPT(I,J,3),ATYPE(BONDMMPT(I,J,3)), &
            BONDMMPT(I,J,4)
          ENDDO
         ENDDO
         WRITE(OUTU,*) '      ANGLES:'
         WRITE(OUTU,*) '      NO    ATOM I     ATOM J     ATOM K   FLAG'
         DO HB=1,HBRDNUM
           DO I=1,ANGLNUM(HB)
             WRITE(OUTU,812) ANGLMMPT(HB,I,1),ANGLMMPT(HB,I,2),  &
             ATYPE(ANGLMMPT(HB,I,2)),ANGLMMPT(HB,I,3),ATYPE(ANGLMMPT(HB,I,3)), &
             ANGLMMPT(HB,I,4),ATYPE(ANGLMMPT(HB,I,4)),ANGLMMPT(HB,I,5)
           ENDDO
         ENDDO
         WRITE(OUTU,*) '      DIHEDRALS:'
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J      ATOM K     ATOM' &
              ,' L   FLAG'
         DO I=1,DIHENUM
            WRITE(OUTU,813) DIHEMMPT(I,1),DIHEMMPT(I,2),ATYPE(DIHEMMPT(I,2)),   &
                 DIHEMMPT(I,3),ATYPE(DIHEMMPT(I,3)),                 &
                 DIHEMMPT(I,4),ATYPE(DIHEMMPT(I,4)),                 &
                 DIHEMMPT(I,5),ATYPE(DIHEMMPT(I,5)),DIHEMMPT(I,6)
         ENDDO
         WRITE(OUTU,*) '      IMPROPERS:'
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J      ATOM K    ATOM'   &
              ,' L   FLAG'
         DO I=1,IMPRNUM
            WRITE(OUTU,813) IMPRMMPT(I,1),IMPRMMPT(I,2),ATYPE(IMPRMMPT(I,2)),   &
                 IMPRMMPT(I,3),ATYPE(IMPRMMPT(I,3)),                 &
                 IMPRMMPT(I,4),ATYPE(IMPRMMPT(I,4)),                 &
                 IMPRMMPT(I,5),ATYPE(IMPRMMPT(I,5)),IMPRMMPT(I,6)
         ENDDO
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      NONBONDED TERMS: FLAG  1  MEANS TERM IS NEW'
         WRITE(OUTU,*) '                       FLAG -1  MEANS TERM EXISTS'
         WRITE(OUTU,*) '      SPECIAL 1-4 VDW: FLAG  14  MEANS TERM IS NEW'
         WRITE(OUTU,*) '      SPECIAL 1-4 VDW: FLAG -14 MEANS TERM EXISTS'
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      NONBONDED:'
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J    FLAG'
! WRITE DOWN NONBMMPT[..]
        DO I=1,HBRDNUM
         DO J=1,NONBNUM(I)
            WRITE(OUTU,814) NONBMMPT(I,J,1),NONBMMPT(I,J,2), &
            ATYPE(NONBMMPT(I,J,2)),NONBMMPT(I,J,3), &
            ATYPE(NONBMMPT(I,J,3)),NONBMMPT(I,J,4)
         ENDDO
        ENDDO
      endif


811 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,2X,I6)
812 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,1X,I6)
813 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,1X,I6)
814 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,1X,I6)

#if KEY_PARALLEL==1
    ENDIF
#endif

  END SUBROUTINE MSMMPTINIT


!      ______________________________________________________________

!      NBONDED INTERACTIONS AFFECTED BY MMPT

  SUBROUTINE NBONDINITMS(INBLO,JNB)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image
    use inbnd

    INTEGER H,J,ITMP,JSTART,JEND
    INTEGER NBPREV,L,I,NB,HB,NONB2H(NATOM),INBLO(*),JNB(*)

    DO H=1,HBRDNUM
      NONBNUM(H)=0
    ENDDO

! SET HYDROGEN - ACCEPTOR NONB INTERACTION
    DO H=1,HBRDNUM
       NONBNUM(H)=1
       NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
       NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,3)
       NONBMMPT(H,NONBNUM(H),3)=HBRDATOM(H,2)
       NONBMMPT(H,NONBNUM(H),4)=2
    ENDDO

! SET DONOR - HYDROGEN NONB INTERACTION
    DO H=1,HBRDNUM
       NONBNUM(H)=NONBNUM(H)+1
       NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
       NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,1)
       NONBMMPT(H,NONBNUM(H),3)=HBRDATOM(H,2)
       NONBMMPT(H,NONBNUM(H),4)=-2
    ENDDO

! SET DONOR - ACCEPTOR NONB INTERACTION
    DO H=1,HBRDNUM
       NONBNUM(H)=NONBNUM(H)+1
       NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
       NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,1)
       NONBMMPT(H,NONBNUM(H),3)=HBRDATOM(H,3)
       NONBMMPT(H,NONBNUM(H),4)=0
    ENDDO

! SET ANGLE CONNECTED ATOMS - HYDROGEN NONB INTERACTION
! FOR EACH HBOND SEARCH IN EACH ANGLE TERM THE ATOM WHICH IS NOT
! PART OF HBOND
! ASSUMES THAT ANGLE SET UP IS ALWAYS OF THE KIND XA-A-H, OR H-A-XA
! ON ACCEPTOR SIDE OR SIMILAR ON DONOR SIDE LIKE XD-D-H, OR H-D-XD
! WHERE A AND D STAND FOR ACEPTOR, RESP DONOR, H IS HYDROGEN ATOM AND
! XD, RESP XA IS THE THIRD ATOM IN THE STRUCTURE FORMING THE ANGLE
!
! SOME NONBONDED INTERACTIONS ARE NOT COMPUTED. CHARMM HAS CERTAIN
! EXCLUSION MODES (CONTROLLED THROUGH NBXMOD KEYWORD):
! NBXMOD 3 EXCLUDES 1-2 AND 1-3 INTERACTION, IE EXCLUDE NONBONDED
! CONTRIBUTION FROM ATOMS CONNECTED TROUGHT BONDS OR ANGLES
! NBXMOD 4 ADDITIONALLY EXCLUDES 1-4 INTERACTIONS, IE ATOMS CONNECTED
! TROUGH DIHEDRAL ANGLES
!
! CORRECT PROCESSING OF EXCLUSION LISTS OF NONBONDED INTERACTIONS HAS
! BEEN IMPLEMENTED ONLY FOR MODES 3,4,AND 5
! SHOULD USE THE DEFAULT 5
! ------------------------- NBXMOD 3 -------------------------------

    IF ((NBXMOD .EQ. 3) .OR. (NBXMOD .EQ. 5)) THEN
      DO H=1,HBRDNUM
        DO I=1,ANGLNUM(H)
         if (PRNLEV >= 2) WRITE(OUTU,*) I, ANGLMMPT(H,I,2), &
         ANGLMMPT(H,I,3),ANGLMMPT(H,I,4)
        ENDDO
      ENDDO

      DO H=1,HBRDNUM
        DO I=1,ANGLNUM(H)
!   HANDLE 1-3 NONB INTERACTION FOR H-A-Y ORDER
          IF (ANGLMMPT(H,I,2).EQ.HBRDATOM(H,2) .AND. &
          ABS(ANGLMMPT(H,I,5)).EQ.1) THEN
            NONBNUM(H)=NONBNUM(H)+1
            NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
            NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
            NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,I,4)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (ANGLMMPT(H,I,5).EQ.1) THEN
              NONBMMPT(H,NONBNUM(H),4)=-1
            ELSE
              NONBMMPT(H,NONBNUM(H),4)=1
            ENDIF

            !ADD D-H-A-Y/X-D-H-A IN REMOVAL LIST
            NONBNUM(H)=NONBNUM(H)+1
            NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
            NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,I,4)
            NONBMMPT(H,NONBNUM(H),4)=3
            IF (ANGLMMPT(H,I,5).EQ.1) THEN
              NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,3)
            ELSE
              NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,1)
            ENDIF
!   HANDLE 1-3 NONB INTERACTION FOR Y-A-H ORDER
          ELSE IF (ANGLMMPT(H,I,4).EQ.HBRDATOM(H,2)) THEN
            NONBNUM(H)=NONBNUM(H)+1
            NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
            NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
            NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,I,2)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (ANGLMMPT(H,I,5).EQ.1) THEN
              NONBMMPT(H,NONBNUM(H),4)=-1
            ELSE
              NONBMMPT(H,NONBNUM(H),4)=1
            ENDIF

            !ADD D-H-A-Y/X-D-H-A IN REMOVAL LIST
            NONBNUM(H)=NONBNUM(H)+1
            NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
            NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,I,2)
            NONBMMPT(H,NONBNUM(H),4)=3
            IF (ANGLMMPT(H,I,5).EQ.1) THEN
              NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,3)
            ELSE
              NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,1)
            ENDIF

          ENDIF
        ENDDO
      ENDDO

      !!prmlpe label
      !add XD-D-H..A-XA and XA-A-H..D-XD in REMOVAL list
      DO H=1,HBRDNUM
        DO I=1,ANGLNUM(H)
          IF (ABS(ANGLMMPT(H,I,5)).EQ.1) THEN
            DO J=I+1,ANGLNUM(H)
              IF (ABS(ANGLMMPT(H,J,5)).EQ.1) THEN
                IF(((ANGLMMPT(H,I,3).EQ.HBRDATOM(H,1)).AND. &
                (ANGLMMPT(H,J,3).EQ.HBRDATOM(H,3))).OR. &
                ((ANGLMMPT(H,I,3).EQ.HBRDATOM(H,3)).AND. &
                (ANGLMMPT(H,J,3).EQ.HBRDATOM(H,1)))) THEN
                  NONBNUM(H)=NONBNUM(H)+1
                  NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
                  IF (ANGLMMPT(H,I,2).EQ.HBRDATOM(H,2)) THEN
                    NONBMMPT(H,NONBNUM(H),2)=ANGLMMPT(H,I,4)
                  ELSE
                    NONBMMPT(H,NONBNUM(H),2)=ANGLMMPT(H,I,2)
                  ENDIF
                  IF (ANGLMMPT(H,J,2).EQ.HBRDATOM(H,2)) THEN
                    NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,J,4)
                  ELSE
                    NONBMMPT(H,NONBNUM(H),3)=ANGLMMPT(H,J,2)
                  ENDIF
                  NONBMMPT(H,NONBNUM(H),4)=3
                ENDIF
              ENDIF
            ENDDO
          ENDIF
        ENDDO
      ENDDO
    ENDIF

!   ------------- EXTENSION NBXMOD 4 ---------------------------------

    IF (NBXMOD .EQ. 4) THEN
      DO I=1,DIHENUM
        DO H=1, HBRDNUM

!   HANDLE 1-4 NONB INTERACTION FOR H-A-XA-XB ORDER
          IF (DIHEMMPT(I,2) .EQ. HBRDATOM(H,2)) THEN
             NONBNUM(H)=NONBNUM(H)+1
             NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
             NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
             NONBMMPT(H,NONBNUM(H),3)=DIHEMMPT(I,5)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   DIHEDRAL TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW DIHEDRAL TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(H,NONBNUM(H),4)=-1
            ELSE
               NONBMMPT(H,NONBNUM(H),4)=1
            ENDIF
!   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-H ORDER
          ELSE IF (DIHEMMPT(I,5).EQ.HBRDATOM(H,2)) THEN
                NONBNUM(H)=NONBNUM(H)+1
                NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
                NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
                NONBMMPT(H,NONBNUM(H),3)=DIHEMMPT(I,2)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(H,NONBNUM(H),4)=-1
            ELSE
               NONBMMPT(H,NONBNUM(H),4)=1
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF


!   ------------- EXTENSION NBXMOD 5 ---------------------------------
!   IF TRANSFERED HYDROGEN ATOM IS CONNECTED TO OTHER ATOMS THROUGH
!   DIHEDRAL ANGLES *AND* IF THESE ATOM PAIRS USE SPECIAL 1-4 VDW
!   PARAMETERS THESE 1-4 INTERACTIONS MUST BE SWITCHED FROM SPECIAL
!   TO STANDARD ON THE DONOR SIDE AND VICE VERSA ON THE ACCEPTOR SIDE

    IF (NBXMOD .EQ. 5) THEN
!   HANDLE 1-4 NONB EXCLUSION (NBXMOD 5)
      DO I=1,DIHENUM
        DO H=1, HBRDNUM
!   HANDLE 1-4 NONB INTERACTION FOR H-A-XA-XB ORDER
          IF (DIHEMMPT(I,2) .EQ. HBRDATOM(H,2)) THEN
             NONBNUM(H)=NONBNUM(H)+1
             NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
             NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
             NONBMMPT(H,NONBNUM(H),3)=DIHEMMPT(I,5)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL TERMS ON ACCEPTOR SIDE HAVE FLAG -14
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(H,NONBNUM(H),4)=-14
            ELSE
               NONBMMPT(H,NONBNUM(H),4)=14
            ENDIF
!   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-H ORDER
          ELSE IF (DIHEMMPT(I,5).EQ.HBRDATOM(H,2)) THEN
              NONBNUM(H)=NONBNUM(H)+1
              NONBMMPT(H,NONBNUM(H),1)=NONBNUM(H)
              NONBMMPT(H,NONBNUM(H),2)=HBRDATOM(H,2)
              NONBMMPT(H,NONBNUM(H),3)=DIHEMMPT(I,2)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL 1-4  TERMS ON ACCEPTOR SIDE HAVE FLAG -14
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(H,NONBNUM(H),4)=-14
            ELSE
               NONBMMPT(H,NONBNUM(H),4)=14
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF


!   SORT NB PAIRS BY ATOM TYPE, I.E. TRANSFERED HYDROGEN
!   COMES SECOND
    DO H=1,HBRDNUM
      DO I=1,NONBNUM(H)
        IF (NONBMMPT(H,I,2).NE.HBRDATOM(H,2)) THEN
              CONTINUE
        ELSE
          ITMP=NONBMMPT(H,I,2)
          NONBMMPT(H,I,2)=NONBMMPT(H,I,3)
          NONBMMPT(H,I,3)=ITMP
        ENDIF
      ENDDO
    ENDDO

  END SUBROUTINE NBONDINITMS


! ... _______________________________________________________

  SUBROUTINE EBONDMSPT(MM,I,J,EB,DXBRM,DYBRM,DZBRM)

! ... routine calculates classical bond energy terms
! ... and forces for specific bonds
!      results or stored in ebmmpt and dxbrm, dxbrm
! ... based on charmm ebond subroutine

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use cnst_fcm
    use psf
    use param
    use block_fcm
    use number
    use image

    INTEGER MM
    real(chm_real) EB,RX,RY,RZ,S1,S2,DB,DF,DXBRM,DYBRM,DZBRM
    real(chm_real) R,ERM
    INTEGER I,II,J,IC

    ERM=0.0D0
    IC=ICB(MM)

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (RX.GT.0.5D0*PXSIZE) THEN
        RX=RX-PXSIZE
      ELSE IF (RX.LE.-0.5D0*PXSIZE) THEN
        RX=RX+PXSIZE
      ENDIF
      IF (RY.GT.0.5D0*PYSIZE) THEN
        RY=RY-PYSIZE
      ELSE IF (RY.LE.-0.5D0*PYSIZE) THEN
        RY=RY+PYSIZE
      ENDIF
      IF (RZ.GT.0.5D0*PZSIZE) THEN
        RZ=RZ-PZSIZE
      ELSE IF (RZ.LE.-0.5D0*PZSIZE) THEN
        RZ=RZ+PZSIZE
      ENDIF
    ENDIF

    S2=RX*RX + RY*RY + RZ*RZ
    S1=SQRT(S2)

    DB=S1-CBB(IC)
    DF=CBC(IC)*DB
    EB=DF*DB

    R=2.D0/S1
    DF=DF*R

    DXBRM=DF*RX
    DYBRM=DF*RY
    DZBRM=DF*RZ

    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> BOND ATOMS ATOM I',I,'  ATOM J', J
      WRITE(OUTU,*) 'MSMMPT> BOND ENERGY',EB
      WRITE(OUTU,*) 'MSMMPT> BOND FORCE ATOM I',DXBRM,DYBRM,DZBRM
      WRITE(OUTU,*) 'MSMMPT> BOND FORCE ATOM J',-DXBRM,-DYBRM,-DZBRM
      WRITE(OUTU,*) ' '
      IF(PRNLEV.GT.9) THEN
        WRITE(OUTU,*) 'MSMMPT> BOND PARAMETERS USED',CBB(IC),CBC(IC)
        WRITE(OUTU,*) ' '
      ENDIF
    ENDIF

  END SUBROUTINE EBONDMSPT
! ... _______________________________________________________


  SUBROUTINE EANGLMSPT(M,I,J,K,EA,DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
!     THIS ROUTINE CALCULATES THE ENERGY AND FORCE OF A BOND ANGLE
!     OF ATOMS CONNECTED TO A TRANSFER HYDROGEN ATOM.
!     THE ROUTINE IS BASED ON THE EANGLE SUBROUTINE BY BROOKS.
!     SVEN.LAMMERS@UNIBAS.CH AUG 2003

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,K,IC

    real(chm_real) RI2,RJ2,RI,RJ,RIR,RJR,          &
          DXI,DYI,DZI,DXJ,DYJ,DZJ,         &
          DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,   &
          CST, AT, STR, ST2R,              &
          DA,DF,EA,                        &
          DTXI,DTYI,DTZI,DTXJ,DTYJ,DTZJ,   &
          DFX,DGX,DFY,DGY,DFZ,DGZ




    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK

    EA=0.0D0
    IC=ICT(M)

    DXI=X(I)-X(J)
    DYI=Y(I)-Y(J)
    DZI=Z(I)-Z(J)
    DXJ=X(K)-X(J)
    DYJ=Y(K)-Y(J)
    DZJ=Z(K)-Z(J)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXI.GT.0.5D0*PXSIZE) THEN
        DXI=DXI-PXSIZE
      ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
        DXI=DXI+PXSIZE
      ENDIF
      IF (DYI.GT.0.5D0*PYSIZE) THEN
        DYI=DYI-PYSIZE
      ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
        DYI=DYI+PYSIZE
      ENDIF
      IF (DZI.GT.0.5D0*PZSIZE) THEN
        DZI=DZI-PZSIZE
      ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
        DZI=DZI+PZSIZE
      ENDIF

      IF (DXJ.GT.0.5D0*PXSIZE) THEN
        DXJ=DXJ-PXSIZE
      ELSE IF (DXJ.LE.-0.5D0*PXSIZE) THEN
        DXJ=DXJ+PXSIZE
      ENDIF
      IF (DYJ.GT.0.5D0*PYSIZE) THEN
        DYJ=DYJ-PYSIZE
      ELSE IF (DYJ.LE.-0.5D0*PYSIZE) THEN
        DYJ=DYJ+PYSIZE
      ENDIF
      IF (DZJ.GT.0.5D0*PZSIZE) THEN
        DZJ=DZJ-PZSIZE
      ELSE IF (DZJ.LE.-0.5D0*PZSIZE) THEN
        DZJ=DZJ+PZSIZE
      ENDIF
    ENDIF

    RI2=DXI*DXI+DYI*DYI+DZI*DZI
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

    RI=SQRT(RI2)
    RJ=SQRT(RJ2)

    RIR=1.D0/RI
    RJR=1.D0/RJ

    DXIR=DXI*RIR
    DYIR=DYI*RIR
    DZIR=DZI*RIR
    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR

    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

    AT=ACOS(CST)

    DA=AT-CTB(IC)
    DF=CTC(IC)*DA

    EA=DF*DA
    DF=DF+DF

    ST2R=1.D0/(1.D0-CST*CST)
    STR=SQRT(ST2R)
    DF=-DF*STR

    DTXI=RIR*(DXJR-CST*DXIR)
    DTXJ=RJR*(DXIR-CST*DXJR)
    DTYI=RIR*(DYJR-CST*DYIR)
    DTYJ=RJR*(DYIR-CST*DYJR)
    DTZI=RIR*(DZJR-CST*DZIR)
    DTZJ=RJR*(DZIR-CST*DZJR)

    DFX=DF*DTXI
    DGX=DF*DTXJ

    DFY=DF*DTYI
    DGY=DF*DTYJ

    DFZ=DF*DTZI
    DGZ=DF*DTZJ

    DXAI=DFX
    DXAJ=-DFX-DGX
    DXAK=DGX

    DYAI=DFY
    DYAJ=-DFY-DGY
    DYAK=DGY

    DZAI=DFZ
    DZAJ=-DFZ-DGZ
    DZAK=DGZ


    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> ANGLE ATOMS ATOM I',I,'  ATOM J', J,'  ATOM K', K
      WRITE(OUTU,*) 'MSMMPT> ANGLE ENERGY',EA
      WRITE(OUTU,*) 'MSMMPT> ANGLE FORCE ATOM I',DXAI,DYAI,DZAI
      WRITE(OUTU,*) 'MSMMPT> ANGLE FORCE ATOM J',DXAJ,DYAJ,DZAJ
      WRITE(OUTU,*) 'MSMMPT> ANGLE FORCE ATOM K',DXAK,DYAK,DZAK
      WRITE(OUTU,*) ' '
      IF(PRNLEV.GT.9) THEN
        WRITE(OUTU,*) 'MSMMPT> ANGLE PARAMETERS USED',CTB(IC),CTC(IC)
        WRITE(OUTU,*) ' '
      ENDIF
    ENDIF

    RETURN
  END SUBROUTINE EANGLMSPT


!      _______________________________________________________

  SUBROUTINE NBNDMMPT(NB,I,MYJ,INBLO,JNB,MAXROW, &
         DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM, &
         DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1, &
         DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2, &
         ENBMMPT,EELMMPT)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,J1,IC,INBLO(*),JNB(*),NB,MYJ,jpr,npr,itemp
    !INTEGER PXL,PYL,PZL,PXI,PYI,PZI,PMAX
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,      &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          EELMMPT,ENBMMPT, DF,DF2,ERM

    INTEGER IOFF(193),I1,IACI,MAXROW

    real(chm_real)  DXIRM,DXJRM,DYIRM,DYJRM,DZIRM,DZJRM
    real(chm_real)  DXIRM1,DXJRM1,DYIRM1,DYJRM1,DZIRM1,DZJRM1
    real(chm_real)  DXIRM2,DXJRM2,DYIRM2,DYJRM2,DZIRM2,DZJRM2
    real(chm_real) XYZ, CTOFNB,CTONNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) RDIST,C2ONNB,C3OFON,VSWI
    real(chm_real) DXIT,DYIT,DZIT


!     INITIALIZE THE CODE LOOK UP OFFSETS
    J=0
    DO M=1,NATC
      IOFF(M)=J
      J=J+M
    ENDDO

!   TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.

    CTOFNB=PRM_NONB_CUTOFF
    CTONNB=PRM_NONB_CUTON
!   CGF EQUALS CCELEC/EPS
    CGF= 332.0716D0
!   HERE E14FAC IS ALWAYS SET TO ONE
    E14FAC=1.D0

    C2OFNB=CTOFNB*CTOFNB

!   SHIFTED DIELECTRIC COEFFICIENTS
    C2ROF2=MINTWO/C2OFNB
    CHROF2=-HALF/C2OFNB

    EELMMPT=0.D0
    ENBMMPT=0.D0

    DXIRM=0.D0
    DYIRM=0.D0
    DZIRM=0.D0
    DXJRM=0.D0
    DYJRM=0.D0
    DZJRM=0.D0

    DXIRM1=0.D0
    DYIRM1=0.D0
    DZIRM1=0.D0
    DXJRM1=0.D0
    DYJRM1=0.D0
    DZJRM1=0.D0

    DXIRM2=0.D0
    DYIRM2=0.D0
    DZIRM2=0.D0
    DXJRM2=0.D0
    DYJRM2=0.D0
    DZJRM2=0.D0

    I1=ITC(IAC(I))
    IACI=IOFF(I1)

    CGT=CGF*CG(I)
    J=MYJ
    CGT2=CGT

    J1=ITC(IAC(J))
    IF (I1.LT.J1) THEN
       IC=IOFF(J1)+I1
    ELSE
       IC=IACI+J1
    ENDIF

    CRXI=X(I)
    CRYI=Y(I)
    CRZI=Z(I)

    DXI=CRXI-X(MYJ)
    DYI=CRYI-Y(MYJ)
    DZI=CRZI-Z(MYJ)
    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXI.GT.0.5D0*PXSIZE) THEN
        DXI=DXI-PXSIZE
      ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
        DXI=DXI+PXSIZE
      ENDIF
      IF (DYI.GT.0.5D0*PYSIZE) THEN
        DYI=DYI-PYSIZE
      ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
        DYI=DYI+PYSIZE
      ENDIF
      IF (DZI.GT.0.5D0*PZSIZE) THEN
        DZI=DZI-PZSIZE
      ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
        DZI=DZI+PZSIZE
      ENDIF
    ENDIF

    S=DXI*DXI+DYI*DYI+DZI*DZI
    RDIST=SQRT(S)

    IF (RDIST.LT.CTOFNB) THEN
      R2 = 1.0/S
      R1 = SQRT(R2)

      DF=0.0D0

!   VAN DER WAALS with cutoff in switch mode
      SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
      SIG6=SIG2*SIG2*SIG2
      SIG12=SIG6*SIG6

      IF (RDIST.LT.CTONNB) THEN
        ENBMMPT=ENBMMPT+CNBB(IC)*(SIG12-SIG6-SIG6)
        DF=CNBB(IC)*R2*12.D0*(SIG6-SIG12)
      ELSE
        C2ONNB=CTONNB*CTONNB
        C3OFON=(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)
        VSWI=(S-C2OFNB)*(S-C2OFNB)*(C2OFNB+2.D0*S-3.D0*C2ONNB)/C3OFON
        ENBMMPT=ENBMMPT+CNBB(IC)*(SIG12-SIG6-SIG6)*VSWI
        DF=CNBB(IC)*R2*12.D0*(SIG6-SIG12)*VSWI+CNBB(IC)* &
        (SIG12-SIG6-SIG6)*12.D0*(S-C2OFNB)*(S-C2ONNB)/C3OFON
      ENDIF



!   ADD -DXIT TO J AND ADD DXIT TO I
      DXIT=DXI*DF
      DYIT=DYI*DF
      DZIT=DZI*DF

      DXIRM1=DXIRM1+DXIT
      DYIRM1=DYIRM1+DYIT
      DZIRM1=DZIRM1+DZIT

      DXJRM1=DXJRM1-DXIT
      DYJRM1=DYJRM1-DYIT
      DZJRM1=DZJRM1-DZIT

!   ELECTROSTATIC
      G1=CGT2*CG(J)*R1
      G2=G1*S*C2ROF2   !QxQxRx(-2)/rcof^2
      G3=G2*S*CHROF2   !QxQxR^3/rcof^4

      EELMMPT=EELMMPT+(G1+G2+G3)
      DF2=R2*(G2-G1+THREE*G3)

!   ADD -DXIT TO J AND ADD DXIT TO I
      DXIT=DXI*DF2
      DYIT=DYI*DF2
      DZIT=DZI*DF2

      DXIRM2=DXIRM2+DXIT
      DYIRM2=DYIRM2+DYIT
      DZIRM2=DZIRM2+DZIT

      DXJRM2=DXJRM2-DXIT
      DYJRM2=DYJRM2-DYIT
      DZJRM2=DZJRM2-DZIT


!   TOTAL
      DF=DF+DF2

!   ADD -DXIT TO J AND ADD DXIT TO I
      DXIT=DXI*DF
      DYIT=DYI*DF
      DZIT=DZI*DF

      DXIRM=DXIRM+DXIT
      DYIRM=DYIRM+DYIT
      DZIRM=DZIRM+DZIT

      DXJRM=DXJRM-DXIT
      DYJRM=DYJRM-DYIT
      DZJRM=DZJRM-DZIT

      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> NONBONDED ATOM [I,J] ',I,J, &
        ' ELSTAT', EELMMPT,' VDW', ENBMMPT,' DIST',RDIST
        WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
        !WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM I',DXIRM2,DYIRM2,DZIRM2
        !WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM J',DXJRM2,DYJRM2,DZJRM2
        IF(PRNLEV.GT.9) THEN
          WRITE(OUTU,*) ' '
          WRITE(OUTU,*) 'MSMMPT> VDW PARAMETER USED:',  &
             RSCLF(I),RSCLF(J),CNBA(IC),CNBB(IC), IC
        ENDIF
      ENDIF
    ENDIF


  END SUBROUTINE NBNDMMPT

!      _______________________________________________________

  SUBROUTINE EVDW14MSPT(NB,I,MYJ,I14,INBLO,JNB,MAXROW, &
         DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM,EVDWMMPT)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,J1,IC,INBLO(*),JNB(*),NB,MYJ,jpr,npr,itemp
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,              &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          EVDWMMPT, DF,ERM

    INTEGER IOFF(193),I1,IACI,MAXROW,I14

    real(chm_real)  DXIRM,DXJRM,  &
           DYIRM,DYJRM,   &
           DZIRM,DZJRM

    real(chm_real) XYZ, CTOFNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) DXIT,DYIT,DZIT


!     INITIALIZE THE CODE LOOK UP OFFSETS
    J=0
    DO M=1,NATC
      IOFF(M)=J
      J=J+M
    ENDDO

!  TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.
    CTOFNB=PRM_NONB_CUTOFF
!   CGF EQUALS CCELEC/EPS
    CGF= 332.0716D0
!   HERE E14FAC IS ALWAYS SET TO ONE
    E14FAC=1.D0
    C2OFNB=CTOFNB*CTOFNB

!   SHIFTED DIELECTRIC COEFFICIENTS
    C2ROF2=MINTWO/C2OFNB
    CHROF2=-HALF/C2OFNB

    EVDWMMPT=0.D0

    DXIRM=0.D0
    DYIRM=0.D0
    DZIRM=0.D0
    DXJRM=0.D0
    DYJRM=0.D0
    DZJRM=0.D0

!   CALCULATE FORCE OF INTERACTION I - J
    CRXI=X(I)
    CRYI=Y(I)
    CRZI=Z(I)

    I1=ITC(IAC(I))
    IACI=IOFF(I1)

    CGT=CGF*CG(I)

    DF=0.0D0

    J=MYJ

    CGT2=CGT

!   RETRIEVE VDW PARAMETER FOR ATOM PAIR
    IF (I14 .EQ. -14) THEN
      J1=ITC(IAC(J))
      IF (I1.LT.J1) THEN
          IC=IOFF(J1)+I1+MAXROW
      ELSE
          IC=IACI+J1+MAXROW
      ENDIF
    ELSE
      J1=ITC(IAC(J))
      IF (I1.LT.J1) THEN
          IC=IOFF(J1)+I1
      ELSE
          IC=IACI+J1
      ENDIF
    ENDIF

    DXI=CRXI-X(MYJ)
    DYI=CRYI-Y(MYJ)
    DZI=CRZI-Z(MYJ)
    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXI.GT.0.5D0*PXSIZE) THEN
        DXI=DXI-PXSIZE
      ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
        DXI=DXI+PXSIZE
      ENDIF
      IF (DYI.GT.0.5D0*PYSIZE) THEN
        DYI=DYI-PYSIZE
      ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
        DYI=DYI+PYSIZE
      ENDIF
      IF (DZI.GT.0.5D0*PZSIZE) THEN
        DZI=DZI-PZSIZE
      ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
        DZI=DZI+PZSIZE
      ENDIF
    ENDIF

    S=DXI*DXI+DYI*DYI+DZI*DZI

    R2=1.0/S
    R1 = SQRT(R2)

    IF (R1.LT.CTOFNB) THEN
!   VAN DER WAALS
      SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
      SIG6=SIG2*SIG2*SIG2
      SIG12=SIG6*SIG6

      EVDWMMPT=(CNBB(IC)*(SIG12-SIG6-SIG6))
      DF=CNBB(IC)*R2*12.D0*(SIG6-SIG12)

      DXIT=DXI*DF
      DYIT=DYI*DF
      DZIT=DZI*DF

!   ADD -DXIT TO J AND ADD DXIT TO I

      DXIRM=DXIT
      DYIRM=DYIT
      DZIRM=DZIT

      DXJRM=-DXIT
      DYJRM=-DYIT
      DZJRM=-DZIT

      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> SPECIAL 1-4 ATOMS ATOM I',I,' ATOM J',J
        WRITE(OUTU,*) 'MSMMPT> NONBONDED ENERGY 1-4 VDW', EVDWMMPT
        WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MSMMPT> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
        IF(PRNLEV.GT.9) THEN
          WRITE(OUTU,*) ' '
          WRITE(OUTU,*) 'MSMMPT> VDW PARAMETER USED:',  &
             RSCLF(I),RSCLF(J),CNBA(IC),CNBB(IC), IC
        ENDIF
      ENDIF
    ENDIF

  END SUBROUTINE EVDW14MSPT
!      _______________________________________________________


  SUBROUTINE EPHIMSPT(IPHI,I,J,K,L,NPHI,ICP,CPD,CPCOS,CPSIN,CPC,&
         erm,DXIRM,DXJRM,DXKRM,DXLRM,                               &
             DYIRM,DYJRM,DYKRM,DYLRM,                               &
             DZIRM,DZJRM,DZKRM,DZLRM)

!     VARIABLES
!     CPD Periodicity of the dihedral energy
!     CPB Phase shift (delta) for the dihedral energy
!     CPCOS COS(CPB)
!     CPSIN SIN(CPB)
!     CPC Force constant for the dihedral energy


    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use consta
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER ICP(*),CPD(*)
    real(chm_real) CPC(*),CPCOS(*),CPSIN(*)
    INTEGER I,J,K,L,IPER,NPER,IPHI,nphi,ic

    real(chm_real) e,ap,arg,erm

    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ,                        &
          AX,AY,AZ,BX,BY,BZ,                                  &
          RA2,RB2,RG2,RG,                                     &
          RGR,RA2R,RB2R,RABR,                                 &
          CP,SP,                                              &
          DF,DDF,E1,DF1,DDF1,                                 &
          FG,HG,FGA,HGB,GAA,GBB,                              &
          DTFX,DTFY,DTFZ,DTGX,DTGY,DTGZ,DTHX,DTHY,DTHZ,       &
          DFX,DFY,DFZ,DGX,DGY,DGZ,DHX,DHY,DHZ,                &
          CA,SA
    LOGICAL LREP
    real(chm_real)  DXIRM,DXJRM,DXKRM,DXLRM,      &
           DYIRM,DYJRM,DYKRM,DYLRM,       &
           DZIRM,DZJRM,DZKRM,DZLRM

    ERM=0.0D0

    IC=ICP(IPHI)

    FX=X(I)-X(J)
    FY=Y(I)-Y(J)
    FZ=Z(I)-Z(J)
    GX=X(J)-X(K)
    GY=Y(J)-Y(K)
    GZ=Z(J)-Z(K)
    HX=X(L)-X(K)
    HY=Y(L)-Y(K)
    HZ=Z(L)-Z(K)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (FX.GT.0.5D0*PXSIZE) THEN
        FX=FX-PXSIZE
      ELSE IF (FX.LE.-0.5D0*PXSIZE) THEN
        FX=FX+PXSIZE
      ENDIF
      IF (FY.GT.0.5D0*PYSIZE) THEN
        FY=FY-PYSIZE
      ELSE IF (FY.LE.-0.5D0*PYSIZE) THEN
        FY=FY+PYSIZE
      ENDIF
      IF (FZ.GT.0.5D0*PZSIZE) THEN
        FZ=FZ-PZSIZE
      ELSE IF (FZ.LE.-0.5D0*PZSIZE) THEN
        FZ=FZ+PZSIZE
      ENDIF

      IF (GX.GT.0.5D0*PXSIZE) THEN
        GX=GX-PXSIZE
      ELSE IF (GX.LE.-0.5D0*PXSIZE) THEN
        GX=GX+PXSIZE
      ENDIF
      IF (GY.GT.0.5D0*PYSIZE) THEN
        GY=GY-PYSIZE
      ELSE IF (GY.LE.-0.5D0*PYSIZE) THEN
        GY=GY+PYSIZE
      ENDIF
      IF (GZ.GT.0.5D0*PZSIZE) THEN
        GZ=GZ-PZSIZE
      ELSE IF (GZ.LE.-0.5D0*PZSIZE) THEN
        GZ=GZ+PZSIZE
      ENDIF

      IF (HX.GT.0.5D0*PXSIZE) THEN
        HX=HX-PXSIZE
      ELSE IF (HX.LE.-0.5D0*PXSIZE) THEN
        HX=HX+PXSIZE
      ENDIF
      IF (HY.GT.0.5D0*PYSIZE) THEN
        HY=HY-PYSIZE
      ELSE IF (HY.LE.-0.5D0*PYSIZE) THEN
        HY=HY+PYSIZE
      ENDIF
      IF (HZ.GT.0.5D0*PZSIZE) THEN
        HZ=HZ-PZSIZE
      ELSE IF (HZ.LE.-0.5D0*PZSIZE) THEN
        HZ=HZ+PZSIZE
      ENDIF

    ENDIF

    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX

    RA2=AX*AX+AY*AY+AZ*AZ
    RB2=BX*BX+BY*BY+BZ*BZ
    RG2=GX*GX+GY*GY+GZ*GZ
    RG=SQRT(RG2)

    RGR=ONE/RG
    RA2R=ONE/RA2
    RB2R=ONE/RB2
    RABR=SQRT(RA2R*RB2R)

    CP=(AX*BX+AY*BY+AZ*BZ)*RABR
    SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)

!   SETUP PROPER DIHEDRALS
!   CPD  Periodicity of the dihedral energy

    IF (CPD(IC).NE.0) THEN
      E=ZERO
      DF=ZERO
      DDF=ZERO

      IPER=CPD(IC)

      IF (IPER.GE.0) THEN
        LREP=.FALSE.
      ELSE
        LREP=.TRUE.
        IPER=-IPER
      ENDIF

      E1=ONE
      DF1=ZERO

      DO NPER=1,IPER
          DDF1=E1*CP-DF1*SP
          DF1=E1*SP+DF1*CP
          E1=DDF1
      enddo

      E1=E1*CPCOS(IC)+DF1*CPSIN(IC)
      DF1=DF1*CPCOS(IC)-DDF1*CPSIN(IC)
      DF1=-IPER*DF1
      DDF1=-IPER*IPER*E1
      E1=ONE+E1

      IF (IPER.EQ.0) THEN
          E1=ONE
          DF1=ZERO
          DDF1=ZERO
      ENDIF


!   SETUP FOR IMPROPER DIHEDRALS
    ELSE

!   Calculation  of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
      CA=CP*CPCOS(IC)+SP*CPSIN(IC)
      SA=SP*CPCOS(IC)-CP*CPSIN(IC)
      IF (CA.GT.PTONE ) THEN
          AP=ASIN(SA)
      ELSE
          AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
      ENDIF

      DDF=TWO*CPC(IC)
      DF=DDF*AP
      E=HALF*DF*AP
    ENDIF

    ARG=CPC(IC)
    E=E+ARG*E1
    DF=DF+ARG*DF1
    DDF=DDF+ARG*DDF1

    FG=FX*GX+FY*GY+FZ*GZ
    HG=HX*GX+HY*GY+HZ*GZ
    FGA=FG*RA2R*RGR
    HGB=HG*RB2R*RGR
    GAA=-RA2R*RG
    GBB=RB2R*RG

    DTFX=GAA*AX
    DTFY=GAA*AY
    DTFZ=GAA*AZ
    DTGX=FGA*AX-HGB*BX
    DTGY=FGA*AY-HGB*BY
    DTGZ=FGA*AZ-HGB*BZ
    DTHX=GBB*BX
    DTHY=GBB*BY
    DTHZ=GBB*BZ

    DFX=DF*DTFX
    DFY=DF*DTFY
    DFZ=DF*DTFZ
    DGX=DF*DTGX
    DGY=DF*DTGY
    DGZ=DF*DTGZ
    DHX=DF*DTHX
    DHY=DF*DTHY
    DHZ=DF*DTHZ

    ERM=E

    DXIRM=DFX
    DXJRM=-DFX+DGX
    DXKRM=-DHX-DGX
    DXLRM=DHX

    DYIRM=DFY
    DYJRM=-DFY+DGY
    DYKRM=-DHY-DGY
    DYLRM=DHY

    DZIRM=DFZ
    DZJRM=-DFZ+DGZ
    DZKRM=-DHZ-DGZ
    DZLRM=DHZ

    IF(PRNLEV.GT.8) THEN
      IF (CPD(IC).NE.0) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL ATOMS ATOM I',I,'  ATOM J', J   &
              ,'  ATOM K', K,'  ATOM L', L
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL ENERGY',E
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL FORCE ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL FORCE ATOM J',DXJRM,DYJRM,DZJRM
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL FORCE ATOM K',DXKRM,DYKRM,DZKRM
        WRITE(OUTU,*) 'MSMMPT> DIHEDRAL FORCE ATOM L',DXLRM,DYLRM,DZLRM
        WRITE(OUTU,*) ' '
          IF(PRNLEV.GT.9) THEN
            WRITE(OUTU,*) 'MSMMPT> DIHEDRAL PARAMETERS USED', &
                 CPC(IC),CPCOS(IC),CPSIN(IC),CPD(IC)
            WRITE(OUTU,*) ' '
          ENDIF
      ELSE
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> IMPROPER ATOMS ATOM I',I,'  ATOM J', J    &
             ,'  ATOM K', K,'  ATOM L', L
        WRITE(OUTU,*) 'MSMMPT> IMPROPER ENERGY',E
        WRITE(OUTU,*) 'MSMMPT> IMPROPER FORCE ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MSMMPT> IMPROPER FORCE ATOM J',DXJRM,DYJRM,DZJRM
        WRITE(OUTU,*) 'MSMMPT> IMPROPER FORCE ATOM K',DXKRM,DYKRM,DZKRM
        WRITE(OUTU,*) 'MSMMPT> IMPROPER FORCE ATOM L',DXLRM,DYLRM,DZLRM
        WRITE(OUTU,*) ' '
        IF(PRNLEV.GT.9) THEN
             WRITE(OUTU,*) 'MSMMPT> IMPROPER PARAMETERS USED',  &
                CPC(IC),CPCOS(IC),CPSIN(IC),CPD(IC)
             WRITE(OUTU,*) ' '
        ENDIF
      ENDIF
    ENDIF

  END SUBROUTINE EPHIMSPT


!      _________________________________________________________________



  SUBROUTINE SWITCHMSPT(I,J,K,SWF,DSWFRNN,DSWFRNH,                &
                           DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)

!     Force and energy are controlled by a switching function.
!     They are gradually turned off or on according to the relative
!     distance between the involved donor resp acceptor atom and
!     the transfer hydrogen atom.

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER I,J,K
    real(chm_real) RNN,SNN,DXNN,DYNN,DZNN,        &
          RNH,SNH,DXNH,DYNH,DZNH,                 &
          DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ,    &
          SIGMA,KAPPA,DKAPPA,                     &
          THF,SWF,DSWFRNN,DSWFRNH

!   cartesian distances donor - hydrogen

    DXNH=X(I)-X(J)
    DYNH=Y(I)-Y(J)
    DZNH=Z(I)-Z(J)

!   cartesian distances donor - acceptor

    DXNN=X(I)-X(K)
    DYNN=Y(I)-Y(K)
    DZNN=Z(I)-Z(K)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXNH.GT.0.5D0*PXSIZE) THEN
        DXNH=DXNH-PXSIZE
      ELSE IF (DXNH.LE.-0.5D0*PXSIZE) THEN
        DXNH=DXNH+PXSIZE
      ENDIF
      IF (DYNH.GT.0.5D0*PYSIZE) THEN
        DYNH=DYNH-PYSIZE
      ELSE IF (DYNH.LE.-0.5D0*PYSIZE) THEN
        DYNH=DYNH+PYSIZE
      ENDIF
      IF (DZNH.GT.0.5D0*PZSIZE) THEN
        DZNH=DZNH-PZSIZE
      ELSE IF (DZNH.LE.-0.5D0*PZSIZE) THEN
        DZNH=DZNH+PZSIZE
      ENDIF


      IF (DXNN.GT.0.5D0*PXSIZE) THEN
        DXNN=DXNN-PXSIZE
      ELSE IF (DXNN.LE.-0.5D0*PXSIZE) THEN
        DXNN=DXNN+PXSIZE
      ENDIF
      IF (DYNN.GT.0.5D0*PYSIZE) THEN
        DYNN=DYNN-PYSIZE
      ELSE IF (DYNN.LE.-0.5D0*PYSIZE) THEN
        DYNN=DYNN+PYSIZE
      ENDIF
      IF (DZNN.GT.0.5D0*PZSIZE) THEN
        DZNN=DZNN-PZSIZE
      ELSE IF (DZNN.LE.-0.5D0*PZSIZE) THEN
        DZNN=DZNN+PZSIZE
      ENDIF
    ENDIF


    SNN=DXNN*DXNN+DYNN*DYNN+DZNN*DZNN
    RNN=SQRT(SNN)

    RNH=(DXNH*DXNN+DYNH*DYNN+DZNH*DZNN)/RNN !RDH*costheta

!   derivatives
    DRNHX=DXNH/RNN
    DRNHY=DYNH/RNN
    DRNHZ=DZNH/RNN

    DRNNX=DXNN/RNN
    DRNNY=DYNN/RNN
    DRNNZ=DZNN/RNN


!   switch function
!    INSUFFIECIENT APPROXIMATION
!    KAPPA=-3.625D0+2.7D0*RNN
!    DKAPPA=2.7D0

!    IMPROVED KAPPA
    KAPPA=0.4999988862D0*RNN*RNN+0.6014307884D-5*RNN-0.8024645983D-5
    DKAPPA=0.9999977724D0*RNN+0.6014307884D-5

    SIGMA=2.D0

    THF=TANH(SIGMA*(RNH*RNN-KAPPA))

    SWF=(THF+1.D0)/2.D0

    DSWFRNH=(1.D0-THF*THF)/2.D0*SIGMA*RNN
!    DSWFRNN=(1.D0-THF*THF)/2.D0*SIGMA*(RNH-DKAPPA)
!    dswfrnn=dswfrnn-DSWFRNH*rnh/rnn

    DSWFRNN=(1.D0-THF*THF)/2.D0*SIGMA*(-1.D0*DKAPPA)
    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) 'MSMMPT> SWITCH FUNCTION RETURNS',SWF
    ENDIF
  END SUBROUTINE SWITCHMSPT


! ... _______________________________________________________

  SUBROUTINE EMSMMPT(EU,X,Y,Z,DX,DY,DZ,NATOMX)
! main subroutine to add MMPT energy and derivatives
!
!
    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    use code
    use bases_fcm
    use chm_types
    use reawri, only:timest
    !use coordc

    real(chm_real) EU,ETMP
    real(chm_real) ERM,DXBRM,DYBRM,DZBRM
    real(chm_real) GAMMA_k,GAMMA_CUT,GAMMA_DV
    INTEGER NATOMX,I,J,K,L
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!     SWITCH STUFF
    real(chm_real) DX2(NATOMX),DY2(NATOMX),DZ2(NATOMX)


!c'emmpt: variable for proton shuttling
    INTEGER HBI, ITMP,MMPTSEED
    INTEGER, DIMENSION(HBRDNUM) :: II,MTFNUM,MDANUM,MHANUM
    INTEGER MPTLV,MCBI,MCBJ,FLAGPT,FLAGAH,MCBNUM,EMINI,FLAGSHELL
    INTEGER, DIMENSION(HBRDNUM,PRM_DA_LENGTH) ::  MPTDATOM
    INTEGER, DIMENSION(HBRDNUM,PRM_HA_LENGTH) ::  MPTHATOM
!    integer MPTCBTEMP(HBRDNUM,3)
    INTEGER, DIMENSION(3) :: DHATEMP,DHAT2
    !MPTLV: LEVEL OF MPTCOMBI IN LOOP, END WITH HBRDNUM
    !MTFNUM: MTFNUM(I) is pointer to MPTMOTIF(I) in the i-th excessive proton
    !MPTCBTEMP: temporary array to MPTCOMBI
    !MCBNUM: pointer to MPTCOMBI
    !MDANUM: pointer to MPTDATOM, 2*MDANUM+1 to MPTHATOM
    !FLAGHA: FLAG ON IF 'H' IS ACCEPTOR BONDED
    !FLAGPT: FLAG ON IF ONE MOTIF CONFLICTS WITH OTHERS
    !EMIN&EMINI: EMIN(EMINI) is the minimum energy of all states
    REAL(CHM_REAL) EMIN,EPOT,EVMPT,WGSUM,MMPTTIME,DELTAVN,ETEMP

    REAL*8 FXTMP,FYTMP,FZTMP,FXXTMP,FYYTMP,FZZTMP
    REAL*8 XTOR,YTOR,ZTOR,DXI,DYI,DZI
    REAL*8 MAX_WEIGHT,WEIGHT2,EMIN1,EMIN2,RM1,RM2,DTMP,D(3)
    integer MINI1,MINI2
    INTEGER ATMA(3),ATMB(3),FLAGLOOP
    INTEGER EMMPTLOOP,MAXLOOP

    logical :: file_exists,MMPTPRNFLAG,LOOP_NOT_EXIT
    integer mnum,a1,a2,a3,iread,jy,NUMLOOP,FLAG1,FLAG2,IDX7
    INTEGER IFSWAP01,IFSWAP02,MCBLAST

    !EDAMPT: Switch generic energies by E(D..H-A)-E(D-H..A)
    !EDADX-YZ: gradient of potential by EDAMPT(I)
    !EDAFLAG: Switch from '0' to '1' if EDAMPT(I) is calculated

!c'emmpt: global variable in arrays
    INTEGER,DIMENSION(HBRDNUM,MAXMTFNUM,7) :: MPTMOTIF
    INTEGER,DIMENSION(MAXMCBNUM,HBRDNUM,7) :: MPTCOMBI
    INTEGER,DIMENSION(MAXMCBNUM,HBRDNUM) :: MCBPOS
    INTEGER,DIMENSION(HBRDNUM,MAXMTFNUM) :: EMPTFLAG
    INTEGER,DIMENSION(HBRDNUM,MAXMTFN0) :: EDAFLAG

    INTEGER,DIMENSION(MAXMCBNUM) :: MCBWFLAG
    REAL(CHM_REAL),DIMENSION(HBRDNUM,MAXMTFN0) :: EDAMPT
    REAL(CHM_REAL),DIMENSION(HBRDNUM,MAXMTFN0,NATOMX+1) :: EDADX,EDADY,EDADZ
    real(chm_real),DIMENSION(HBRDNUM,MAXMTFNUM,NATOMX+1) :: EMPTDX,EMPTDY, &
    EMPTDZ,EWMPTDX,EWMPTDY,EWMPTDZ
    real(chm_real),DIMENSION(HBRDNUM,MAXMTFNUM) :: EMPT,EWMPT
    real(chm_real),DIMENSION(MAXMCBNUM,NATOMX+1) :: EMCBDX,EMCBDY,EMCBDZ, &
    EWMCBDX,EWMCBDY,EWMCBDZ
    real(chm_real),DIMENSION(MAXMCBNUM) :: EMCB,EWMCB,MCBWG
    !MPTMOTIF is an [HBRDNUM*MTFNUM(:)*3] matrix, stores
    !          potential PT motifs for excessive protons
    !MPTCOMBI is an [MCBNUM*HBRDNUM*3] matrix, stores
    !          proper combinations of PT motifs
    !MCBPOS(I,J) is the pointer of MPTCOMBI(I,J)
    !          in MPTMOTIF(J)
    !EMPT: Switch energies on MPTMOTIF(I,J) by EMSMMPT(DH-A,HBI)
    !EMPTFLAG: Switch from '0' to '1' if EMPT(I,J) is calculated
    !EMPTDX-YZ: gradient of potential by EDAMPT(I,J)
    !EMCB: TOTAL CORRECTED ENERGIES FOR EACH STATE
    !EMCBDX-YZ: gradient of potential by EMCB(MCBI)
    !MCBWG is an [MCBNUM] array to store unnormalized weight,
    !         the summation is WGSUM
    !EW*** are the energies and gradients used for weight

    REAL(CHM_REAL) MPTRAND,STEPSIZE
!c'emmpt: end of variables definition

!    real(chm_real) tstart,tstart0,tend

!    CALL CPU_TIME(MMPTTIME1)
!    WRITE(OUTU,*) 'TEST-TIME>',MMPTTIME1-MMPTTIME0
!    MMPTTIME0=MMPTTIME1
!    call cpu_time(tstart)
!    tstart0=tstart

!    WRITE(OUTU,*) 'TEST-MD>',MDSTEP,TIMEST
    IF (HBRDNUM.EQ.0) THEN
      RETURN
    ENDIF

    IF (MMPTSMODE.EQ.2) THEN
      IF (DYNAMQ .AND. MMPTSKP.GE.1) THEN
        MMPTSTP=MDSTEP
        STEPSIZE=TIMEST
      ELSE
        IF (PRNLEV >= 6 .OR. MMPTSFLAG) THEN
          MMPTSTP=MMPTSTP+1
          STEPSIZE=PRM_STEPSIZE
        ENDIF
      ENDIF
    ELSE
      IF (DYNAMQ) THEN ! .AND. .NOT.(MMPTSFLAG.eq.2 .AND. MMPTSKP.eq.0)) THEN
        MMPTSTP=MDSTEP
        STEPSIZE=TIMEST
      ELSE
        IF (PRNLEV >= 6 .OR. MMPTSFLAG) THEN
          MMPTSTP=MMPTSTP+1
          STEPSIZE=PRM_STEPSIZE
        ENDIF
      ENDIF
    ENDIF
!write(outu,*) 'TEST> ',MDSTEP,LASTSTEP,MMPTSTP,MMPTSKP,DYNAMQ
    MMPTPRNFLAG=.false.
    ! hwm: to fix MOD() problem when MMPTCYC=0
    IF (PRNLEV >= 6 ) THEN
       MMPTPRNFLAG=.TRUE.
    ELSE IF (MMPTCYC.GT.0) THEN
       IF (MMPTSFLAG .AND. MOD(MMPTSTP,MMPTCYC).EQ.0 .AND. &
            (MMPTSTP.GT.LASTSTEP .OR. (.not. DYNAMQ))) THEN
          MMPTPRNFLAG=.TRUE.
       ENDIF
    ELSE
       CONTINUE
    ENDIF
    !    IF (PRNLEV >= 6 .OR. (MMPTSFLAG .AND. MMPTCYC.GT.0 .AND. &
!    MOD(MMPTSTP,MMPTCYC).EQ.0 .AND. (MMPTSTP.GT.LASTSTEP .OR. (.not. DYNAMQ)))) THEN
!      MMPTPRNFLAG=.TRUE.
!    ENDIF

    DO I=1,HBRDNUM
      !if donor changes
      IF (HBRDFLAG(I,4).EQ.1) THEN
        CALL NONBUPDATE(I,BNBND%INBLO,BNBND%JNB,BIMAG%IMATTR, &
        BIMAG%IMBLO,BIMAG%IMJNB,NATOMX,NATOMT,1)
      !if donor is unchanged
      ELSE
        CALL NONBUPDATE(I,BNBND%INBLO,BNBND%JNB,BIMAG%IMATTR, &
        BIMAG%IMBLO,BIMAG%IMJNB,NATOMX,NATOMT,0)
      ENDIF
    ENDDO

    DO I=1,HBRDNUM
      IF (HBRDFLAG(I,4).GE.0) THEN
        IF (HBRDFLAG(I,4).EQ.1) THEN
          CALL BONDSWAP(HBRDATOM(I,1),HBRDATOM(I,2), &
          HBRDATOM(I,3))
          IF (PRNLEV.GE.6) WRITE (OUTU,*) 'MSMMPT-MCB> SWAP COMPLETE:', &
          HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3)
        ENDIF

        CALL MTFSWITCH(I,HBRDFLAG(I,1:3),BNBND%INBLO,BNBND%JNB,0)

        if (PRNLEV .GE. 6) then!.and. HBRDFLAG(I,4).NE.2) then
          WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH COMPLETE:'
          WRITE (OUTU,2090) '   ',HBRDATOM(I,1), HBRDATOM(I,2), &
          HBRDATOM(I,3), '=>',HBRDFLAG(I,1),HBRDFLAG(I,2),HBRDFLAG(I,3)
        endif

2090  FORMAT(A3,I4,I4,I4,A3,I4,I4,I4)

        HBRDATOM(I,1)=HBRDFLAG(I,1)
        HBRDATOM(I,2)=HBRDFLAG(I,2)
        HBRDATOM(I,3)=HBRDFLAG(I,3)
      ENDIF
    ENDDO

!    call cpu_time(TEND)
!    write(outu,*) 'TEST-DHA-SWAP>',tend-tstart
!    tstart=tend

    !re-INITIALIZE
    DO I=1,HBRDNUM
      HBRDFLAG(I,4)=-1
    ENDDO
    !initialize distance matrix
    DO I=1,NATOMX
      DO J=1,NATOMX
        DISTMAT(I,J)=-9999.D0
      ENDDO
    ENDDO


!    call cpu_time(TEND)
!    write(outu,*) 'TEST-DIST-INIT>',tend-tstart
!    tstart=tend

    EU = 0.d0

!     REMARKS ON CONSTRAINTS:
! ... IF ALL HYDROGEN BRIDGE ATOMS ARE CONSTRAINT NO MMPT ENERGY AND FORCES
!     ARE CALCULATED

!c'emmpt: if (ProtonTransportation)
    DO HBI=1,HBRDNUM
      !FIND CLOSEST Four DONORS/acceptors AROUND OHO
      !c'emmpt: need further development on non-'O' atoms
      CALL FINDDA(MPTDATOM(HBI,:),HBRDATOM(HBI,1),HBRDATOM(HBI,3),-1,-1, &
      MDANUM(HBI),X,Y,Z,NATOMX)
      !OUTPUT: MPTDATOM(HBI,DANUM)={D,A,A1~A4}
!WRITE(OUTU,*) 'TEST> MDANUM-',HBI,'=',MDANUM(HBI)
      !FIND ALL BONDED HYDROGENS around DH-A motif (up to 5)
      CALL FINDHA(MPTDATOM(HBI,:),MPTHATOM(HBI,:),HBRDATOM(HBI,2), &
      MDANUM(HBI),MHANUM(HBI))

      !DEFINE PT MOTIF
      MTFNUM(HBI)=0
      DO I=1,2 !NEW DONORS FROM D OR A
        DO J=(I+1),MDANUM(HBI)
          DO K=1,MHANUM(HBI)
            FLAGAH=0      !IDENTIFY IF 'H' IS ACCEPTOR BONDED
            IF (MPTHATOM(HBI,K).LT.0) THEN
              MPTHATOM(HBI,K)=0-MPTHATOM(HBI,K)
              FLAGAH=1
            ENDIF

            !IDENTIFY IF NEW MOTIF PASSES
            !Account the 1st motif even if it has bad configuration
            IF (IFMOTIF(HBI,MPTDATOM(HBI,I),MPTHATOM(HBI,K), &
            MPTDATOM(HBI,J),X,Y,Z,NATOMX).EQ.1) THEN
              MTFNUM(HBI) = MTFNUM(HBI)+1
              MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTDATOM(HBI,I) !DONOR
              MPTMOTIF(HBI,MTFNUM(HBI),2)=MPTHATOM(HBI,K) !HYDROGEN
              MPTMOTIF(HBI,MTFNUM(HBI),3)=MPTDATOM(HBI,J) !ACCEPTOR
              MPTMOTIF(HBI,MTFNUM(HBI),4)=-1
              MPTMOTIF(HBI,MTFNUM(HBI),5)=-1
              MPTMOTIF(HBI,MTFNUM(HBI),6)=-1
              MPTMOTIF(HBI,MTFNUM(HBI),7)=MTFNUM(HBI)

              !SWITCH ACCEPTOR TO DONOR
              IF (FLAGAH.EQ.1 .AND. I.EQ.1) THEN
                ITMP=MPTMOTIF(HBI,MTFNUM(HBI),1)
                MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTMOTIF(HBI,MTFNUM(HBI),3)
                MPTMOTIF(HBI,MTFNUM(HBI),3)=ITMP
              ENDIF

!              IF(I.EQ.1 .AND. J.EQ.2 .AND. MPTHATOM(HBI,K).NE.) THEN
!                CALL WrnDie (-3,'<MISCOM>','MSMMPT-MPT> Error:
!c'emmpt: new donors must be from D or A
!!CHECK excessive protons can be only bonded to D or A
!!check energy equality if donor switches
            ELSE
              IF (MTFNUM(HBI).EQ.0) then
                CALL WrnDie (0,'<MISCOM>', &
                            'MSMMPT-MPT> ERROR: The primary motif is rejected')

                MTFNUM(HBI)=MTFNUM(HBI)+1
                MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTDATOM(HBI,I) !DONOR
                MPTMOTIF(HBI,MTFNUM(HBI),2)=MPTHATOM(HBI,K) !HYDROGEN
                MPTMOTIF(HBI,MTFNUM(HBI),3)=MPTDATOM(HBI,J) !ACCEPTOR
                MPTMOTIF(HBI,MTFNUM(HBI),4)=-1
                MPTMOTIF(HBI,MTFNUM(HBI),5)=-1
                MPTMOTIF(HBI,MTFNUM(HBI),6)=-1
                MPTMOTIF(HBI,MTFNUM(HBI),7)=MTFNUM(HBI)
                !SWITCH ACCEPTOR TO DONOR
                IF (FLAGAH.EQ.1 .AND. I.EQ.1) THEN
                  ITMP=MPTMOTIF(HBI,MTFNUM(HBI),1)
                  MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTMOTIF(HBI,MTFNUM(HBI),3)
                  MPTMOTIF(HBI,MTFNUM(HBI),3)=ITMP
                ENDIF
              ENDIF
            ENDIF

            !IF (FLAGAH.EQ.1) THEN
            !  MPTHATOM(HBI,K)=0-MPTHATOM(HBI,K)
            !ENDIF
          ENDDO
        ENDDO
      ENDDO

      IF (MTFNUM(HBI).EQ.0) then
         write(outu,*)'MSMMPT-MPT> ERROR: FOUND NO',MTFNUM(HBI), &
           ' MOTIFS IN THE',HBI,'(th) EXCESS PROTON'
         CALL WrnDie (-3,'<MISCOM>','MSMMPT-MPT> ERROR')
      endif

      IF (PRM_IF_SHELL2.EQ.1) THEN
        NUMLOOP=MTFNUM(HBI)
        DO I=2,NUMLOOP
          CALL FINDDA(MPTDATOM(HBI,:),-MPTMOTIF(HBI,I,1),MPTMOTIF(HBI,I,3), &
          HBRDATOM(HBI,1),HBRDATOM(HBI,3),MDANUM(HBI),X,Y,Z,NATOMX)

          IF (MDANUM(HBI).GT.2) THEN
            CALL FINDHA(MPTDATOM(HBI,:),MPTHATOM(HBI,:),MPTMOTIF(HBI,I,2), &
            MDANUM(HBI),MHANUM(HBI))


            DO J=3,MDANUM(HBI)
              DO K=1,MHANUM(HBI)
                FLAGAH=0      !IDENTIFY IF 'H' IS ACCEPTOR BONDED
                IF (MPTHATOM(HBI,K).LT.0) THEN
                  MPTHATOM(HBI,K)=0-MPTHATOM(HBI,K)
                  FLAGAH=1
                ENDIF

                !IDENTIFY IF NEW MOTIF PASSES
                !Account the 1st motif even if it has bad configuration
                IF (IFMOTIF(HBI,MPTDATOM(HBI,2),MPTHATOM(HBI,K), &
                MPTDATOM(HBI,J),X,Y,Z,NATOMX).EQ.1) THEN
                  MTFNUM(HBI) = MTFNUM(HBI)+1
                  MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTMOTIF(HBI,I,1) !DONOR
                  MPTMOTIF(HBI,MTFNUM(HBI),2)=MPTMOTIF(HBI,I,2) !HYDROGEN
                  MPTMOTIF(HBI,MTFNUM(HBI),3)=MPTMOTIF(HBI,I,3) !ACCEPTOR
                  MPTMOTIF(HBI,MTFNUM(HBI),4)=MPTDATOM(HBI,2)
                  MPTMOTIF(HBI,MTFNUM(HBI),5)=MPTHATOM(HBI,K)
                  MPTMOTIF(HBI,MTFNUM(HBI),6)=MPTDATOM(HBI,J)
                  MPTMOTIF(HBI,MTFNUM(HBI),7)=MPTMOTIF(HBI,I,7)

                  DO L=1,(MTFNUM(HBI)-1)
                    IF (MPTMOTIF(HBI,L,4).NE.-1) THEN
                      IF (MPTMOTIF(HBI,MTFNUM(HBI),4).EQ.MPTMOTIF(HBI,L,6) &
                      .AND. MPTMOTIF(HBI,MTFNUM(HBI),6).EQ.MPTMOTIF(HBI,L,4) &
                      .AND. MPTMOTIF(HBI,MTFNUM(HBI),5).EQ.MPTMOTIF(HBI,L,5) &
                      .AND. MPTMOTIF(HBI,MTFNUM(HBI),2).EQ.MPTMOTIF(HBI,L,2) &
                      .AND. MPTMOTIF(HBI,MTFNUM(HBI),2).EQ.MPTMOTIF(HBI,L,5)) THEN
                        MTFNUM(HBI) = MTFNUM(HBI)-1
                      ENDIF
                    ENDIF
                  ENDDO

                  !SWITCH ACCEPTOR TO DONOR
                  IF (FLAGAH.EQ.1 .AND. I.EQ.1) THEN
                    ITMP=MPTMOTIF(HBI,MTFNUM(HBI),1)
                    MPTMOTIF(HBI,MTFNUM(HBI),1)=MPTMOTIF(HBI,MTFNUM(HBI),3)
                    MPTMOTIF(HBI,MTFNUM(HBI),3)=ITMP
                  ENDIF
                ENDIF

              ENDDO
            ENDDO
          ENDIF
        ENDDO

        DO I=NUMLOOP+1,MTFNUM(HBI)
          FLAGLOOP=1
          J=I
          DO WHILE(FLAGLOOP.EQ.1 .AND. J.GT.0)
            IF (MPTMOTIF(HBI,J,1).NE.MPTMOTIF(HBI,J-1,1) .OR. &
            MPTMOTIF(HBI,J,2).NE.MPTMOTIF(HBI,J-1,2) .OR. &
            MPTMOTIF(HBI,J,3).NE.MPTMOTIF(HBI,J-1,3)) THEN
              DO K=1,7
                ITMP=MPTMOTIF(HBI,J,K)
                MPTMOTIF(HBI,J,K)=MPTMOTIF(HBI,J-1,K)
                MPTMOTIF(HBI,J-1,K)=ITMP
              ENDDO
              J=J-1
            ELSE
              FLAGLOOP=0
            ENDIF
          ENDDO
        ENDDO

      ENDIF

      IF (PRNLEV >= 7) then
        WRITE(OUTU,*) 'MSMMPT-MPT> FOUND ',MTFNUM(HBI), &
        ' MOTIFS IN THE',HBI,'(th) EXCESS PROTON'
        WRITE(OUTU,*) '    (H+)  NO.   ATOM-D   ATOM-H   ATOM-A'
        DO I=1,MTFNUM(HBI)
          IF (MPTMOTIF(HBI,I,4).GT.0) THEN
            WRITE(OUTU,2190) HBI,I,MPTMOTIF(HBI,I,1),ATYPE(MPTMOTIF(&
            HBI,I,1)),MPTMOTIF(HBI,I,2),ATYPE(MPTMOTIF(HBI,I,2)),&
            MPTMOTIF(HBI,I,3),ATYPE(MPTMOTIF(HBI,I,3)),MPTMOTIF(HBI,I,4), &
            ATYPE(MPTMOTIF(HBI,I,4)),MPTMOTIF(HBI,I,5),ATYPE(MPTMOTIF(HBI,I,5)), &
            MPTMOTIF(HBI,I,6),ATYPE(MPTMOTIF(HBI,I,6)),MPTMOTIF(HBI,I,7)
          ELSE
            WRITE(OUTU,2190) HBI,I,MPTMOTIF(HBI,I,1),ATYPE(MPTMOTIF(&
            HBI,I,1)),MPTMOTIF(HBI,I,2),ATYPE(MPTMOTIF(HBI,I,2)),&
            MPTMOTIF(HBI,I,3),ATYPE(MPTMOTIF(HBI,I,3)),MPTMOTIF(HBI,I,4), &
            'X',MPTMOTIF(HBI,I,5),'X',MPTMOTIF(HBI,I,6),'X',MPTMOTIF(HBI,I,7)
          ENDIF
        ENDDO
2190 FORMAT('     ',I4,1X,I4,1X,I4,1X,A3,1X,I4,1X,A3,1X,I4,1X,A3, &
1X,I4,1X,A3,1X,I4,1X,A3,1X,I4,1X,A3,1X,I4)
      ENDIF
    ENDDO


!####################################################
!##
!## DEFINE ALL POSSIBLE COMBINATIONS OF MULTI-MOTIFS
!##
!####################################################

    MPTLV=1
    II(1)=1
    MCBNUM=0
    DO WHILE (MPTLV.GT.0)
!WRITE(OUTU,*) 'TEST> ',MPTLV,II(MPTLV),MCBNUM
      IF (II(MPTLV).LE.MTFNUM(MPTLV)) THEN
!c'emmpt: Grothuss principles should be obeyed by
!1) Donors, acceptors and excessive protons are not shared;
!2) A donor can be an acceptor in ANOTHER MOTIF? need to be implemented
!!But TEMPORARILY TURN DOWN CONDITION_2 WITH NO VIOLATION ALLOWED
        FLAGPT=1
        IF (MPTMOTIF(I,II(MPTLV),4).GT.0) THEN
          ATMB(1)=MPTMOTIF(I,II(MPTLV),4)
          ATMB(2)=MPTMOTIF(I,II(MPTLV),5)
          ATMB(3)=MPTMOTIF(I,II(MPTLV),6)
        ELSE
          ATMB(1)=MPTMOTIF(I,II(MPTLV),1)
          ATMB(2)=MPTMOTIF(I,II(MPTLV),2)
          ATMB(3)=MPTMOTIF(I,II(MPTLV),3)
        ENDIF
        DO I=1,MPTLV-1
          IF (MPTMOTIF(I,II(I),4).GT.0) THEN
            ATMA(1)=MPTMOTIF(I,II(I),4)
            ATMA(2)=MPTMOTIF(I,II(I),5)
            ATMA(3)=MPTMOTIF(I,II(I),6)
          ELSE
            ATMA(1)=MPTMOTIF(I,II(I),1)
            ATMA(2)=MPTMOTIF(I,II(I),2)
            ATMA(3)=MPTMOTIF(I,II(I),3)
          ENDIF
          IF (ATMA(1).EQ.ATMB(1)) THEN
            FLAGPT=0
            GOTO 2380
          ENDIF
          IF (ATMA(1).EQ.ATMB(3)) THEN
            FLAGPT=0
            GOTO 2380
          ENDIF
          IF (ATMA(3).EQ.ATMB(1)) THEN
            FLAGPT=0
            GOTO 2380
          ENDIF
          IF (ATMA(3).EQ.ATMB(3)) THEN
            FLAGPT=0
            GOTO 2380
          ENDIF
          IF (ATMA(2).EQ.ATMB(2)) THEN
            FLAGPT=0
            GOTO 2380
          ENDIF
        ENDDO

2380    CONTINUE

        IF (FLAGPT.EQ.1) THEN
          IF (MPTLV.EQ.HBRDNUM) THEN
            MCBNUM=MCBNUM+1
            IF (MCBNUM.GT.MAXMCBNUM) THEN
              CALL WrnDie (-3,'<MISCOM>', &
                          'MSMMPT-MPT> Exceeded potential motifs')
            ENDIF
            DO I=1,HBRDNUM  !WRITE DOWN MPTCBTEMP into MPTCOMBI
              MPTCOMBI(MCBNUM,I,1)=MPTMOTIF(I,II(I),1)
              MPTCOMBI(MCBNUM,I,2)=MPTMOTIF(I,II(I),2)
              MPTCOMBI(MCBNUM,I,3)=MPTMOTIF(I,II(I),3)
              MPTCOMBI(MCBNUM,I,4)=MPTMOTIF(I,II(I),4)
              MPTCOMBI(MCBNUM,I,5)=MPTMOTIF(I,II(I),5)
              MPTCOMBI(MCBNUM,I,6)=MPTMOTIF(I,II(I),6)
              MPTCOMBI(MCBNUM,I,7)=MPTMOTIF(I,II(I),7)
              MCBPOS(MCBNUM,I)=II(I)
            ENDDO
            II(MPTLV)=II(MPTLV)+1
          ELSE
            MPTLV=MPTLV+1
            II(MPTLV)=1
          ENDIF
        ELSE
          II(MPTLV)=II(MPTLV)+1
        ENDIF
      ELSE
        MPTLV=MPTLV-1
        IF (MPTLV.GT.0) THEN
          II(MPTLV)=II(MPTLV)+1
        ENDIF
      ENDIF
    ENDDO

    IF (PRNLEV >= 8) THEN
      WRITE(OUTU,*) '----------------------------------------'
      WRITE(OUTU,*) 'MSMMPT-MPT> FOUND ',MCBNUM, &
      ' COMBINATIONS OF DH-A MOTIFS IN TOTAL'
      WRITE(OUTU,*) '     NO. (H+)   ATOM-D    ATOM-H    ATOM-A     &
      ATOM-D    ATOM-H    ATOM-A'
      DO I=1,MCBNUM
        DO J=1,HBRDNUM
          IF (MPTCOMBI(I,J,4).GT.0) THEN
            WRITE(OUTU,2280) I,J,MPTCOMBI(I,J,1),ATYPE(MPTCOMBI(I,J,1)), &
            MPTCOMBI(I,J,2),ATYPE(MPTCOMBI(I,J,2)), &
            MPTCOMBI(I,J,3),ATYPE(MPTCOMBI(I,J,3)), &
            MPTCOMBI(I,J,4),ATYPE(MPTCOMBI(I,J,4)), &
            MPTCOMBI(I,J,5),ATYPE(MPTCOMBI(I,J,5)), &
            MPTCOMBI(I,J,6),ATYPE(MPTCOMBI(I,J,6)), &
            MPTCOMBI(I,J,7)
          ELSE
            WRITE(OUTU,2280) I,J,MPTCOMBI(I,J,1),ATYPE(MPTCOMBI(I,J,1)), &
            MPTCOMBI(I,J,2),ATYPE(MPTCOMBI(I,J,2)), &
            MPTCOMBI(I,J,3),ATYPE(MPTCOMBI(I,J,3)), &
            MPTCOMBI(I,J,4),'X', &
            MPTCOMBI(I,J,5),'X', &
            MPTCOMBI(I,J,6),'X', &
            MPTCOMBI(I,J,7)
          ENDIF
        ENDDO
        IF (I.LT.MCBNUM) WRITE(OUTU,*) '  '
      ENDDO
      WRITE(OUTU,*) '----------------------------------------'
2280  FORMAT('     ',I4,1X,I4,1X,I4,1X,A4,1X,I4,1X,A4,1X,I4,1X,A4 &
,1X,I4,1X,A4,1X,I4,1X,A4,1X,I4,1X,A4,1X,I4)
    ENDIF

!    call cpu_time(TEND)
!    write(outu,*) 'TEST-MTF>',tend-tstart
!    tstart=tend


!####################################################
!##
!## INITIALIZATION
!##
!####################################################

    !INITIALIZE EDAFLAG
    DO I=1,HBRDNUM
      DO J=1,MTFNUM(I)
        EMPTFLAG(I,J)=0
      ENDDO
    ENDDO

    !INITIALIZE EMCB~DXYZ
    DO I=1,MCBNUM
      EMCB(I)=0.D0
      DO J=1,NATOMX+1
        EMCBDX(I,J)=0.D0
        EMCBDY(I,J)=0.D0
        EMCBDZ(I,J)=0.D0
      ENDDO
    ENDDO

    !INITIALIZE EMPT-DXYZ
    DO I=1,HBRDNUM
      DO J=1,MTFNUM(I)
        EMPT(I,J)=0.D0
        DO K=1,NATOMX+1
          EMPTDX(I,J,K)=0.D0
          EMPTDY(I,J,K)=0.D0
          EMPTDZ(I,J,K)=0.D0
        ENDDO
      ENDDO
    ENDDO

    IF (WMODE.GE.1) THEN
      !INITIALIZE EW***
      DO I=1,MCBNUM
        MCBWFLAG(I)=0
        EWMCB(I)=0.D0
        !DO J=1,NATOMX
        !  EWMCBDX(I,J)=0.D0
        !  EWMCBDY(I,J)=0.D0
        !  EWMCBDZ(I,J)=0.D0
        !ENDDO
      ENDDO
      MCBWFLAG(MCBNUM+1)=0

      DO I=1,HBRDNUM
        DO J=1,MTFNUM(I)
          EWMPT(I,J)=0.D0
          !DO K=1,NATOMX
          !  EWMPTDX(I,J,K)=0.D0
          !  EWMPTDY(I,J,K)=0.D0
          !  EWMPTDZ(I,J,K)=0.D0
          !ENDDO
        ENDDO
      ENDDO
    ENDIF

    !c'emmpt: initialize EDAMPT,EDADX-YZ

    DO I=1,HBRDNUM
      DO K=1,MPTMOTIF(I,MTFNUM(I),7)
        EDAFLAG(I,K)=0
        EDAMPT(I,K)=0.D0
        DO J=1,NATOMX
          EDADX(I,K,J)=0.D0
          EDADY(I,K,J)=0.D0
          EDADZ(I,K,J)=0.D0
        ENDDO
      ENDDO
    ENDDO

!    call cpu_time(TEND)
!    write(outu,*) 'TEST-INIT-02>',tend-tstart
!    tstart=tend


!####################################################
!##
!## CALCULATE ENERGIES (BONDED TERMS) FOR PT MOTIFS
!##
!####################################################

    IF (PRM_DELTAV.LT.-0.D0) THEN
      MCBNUM=1 !prmlpe label
      WMODE=0
    ENDIF

    IF (WMODE.EQ.0) THEN
      MAXLOOP=1
    ELSE
      MAXLOOP=2
    ENDIF

!!Warning: Only apply to ONE excess proton

    !DO EMMPTLOOP=1,MAXLOOP
    EMMPTLOOP=1
    LOOP_NOT_EXIT=.TRUE.
    DO WHILE ((EMMPTLOOP.LE.MAXLOOP) .AND. LOOP_NOT_EXIT)
      IF (EMMPTLOOP.EQ.2) THEN
        !INITIALIZE EMCB~DXYZ IN LOOP-02
        DO I=1,MCBNUM
          EMCB(I)=0.D0
          DO J=1,NATOMX+1
            EMCBDX(I,J)=0.D0
            EMCBDY(I,J)=0.D0
            EMCBDZ(I,J)=0.D0
          ENDDO
        ENDDO

        !PROCESS THE FLAG AND PRINT IN LOOP-02
        DO I=1,HBRDNUM
          ITMP=-1
          DO J=1,MCBNUM
            IF (MPTCOMBI(J,I,4).EQ.-1) THEN
              IF (MCBWFLAG(J).EQ.1) THEN
                ITMP=-1
              ELSE
                ITMP=J
              ENDIF
            ELSE
              IF (MCBWFLAG(J).EQ.1 .AND. ITMP.NE.-1) THEN
                MCBWFLAG(ITMP)=1
                ITMP=-1
              ENDIF
            ENDIF
          ENDDO
        ENDDO

      ENDIF

      DO HBI=1,HBRDNUM

        FLAG1=0
        FLAG2=0
        IFSWAP01=0
        IFSWAP02=0
        IDX7=1

        ATMA(1)=HBRDATOM(HBI,1)
        ATMA(2)=HBRDATOM(HBI,2)
        ATMA(3)=HBRDATOM(HBI,3)

        MCBLAST=1

        IF (PRNLEV.GE.7 .AND. EMMPTLOOP.EQ.2) THEN
          WRITE(OUTU,*) '----------------------------------------------'
          WRITE (OUTU,*) 'MSMMPT-MCB> WEIGHT-FLAGGED MOTIF LIST',HBI
          DO J=1,MCBNUM
            IF (MCBWFLAG(J).EQ.1) THEN
              WRITE(OUTU,*) 'MOTIF',J,':', &
              MPTCOMBI(J,HBI,1),MPTCOMBI(J,HBI,2),MPTCOMBI(J,HBI,3), &
              MPTCOMBI(J,HBI,4),MPTCOMBI(J,HBI,5),MPTCOMBI(J,HBI,6), &
              MPTCOMBI(J,HBI,7)
            ENDIF
          ENDDO
        ENDIF

        DO MCBI=1,MCBNUM+1

          IF (EMMPTLOOP.EQ.1 .OR. MCBWFLAG(MCBI).EQ.1 .OR. MCBI.EQ.MCBNUM+1) THEN
            !Donor stays the same

!DEFINE FLAG2 OF D-H-A SWAP
            !binary boolean for FLAG2 -
            ! 0: 0->0 || 1: 1->0 || 2: 0->1 || 3: 1->1
            IF (MCBI.LE.MCBNUM) THEN
              IF (MPTCOMBI(MCBI,HBI,1).EQ.ATMA(1)) THEN
                IF (FLAG2.EQ.0) THEN
                  FLAG2=0
                ELSE IF (FLAG2.EQ.2 .OR. FLAG2.EQ.3) THEN
                  FLAG2=1
                ELSE
                  CALL WrnDie (-3,'<MISCOM>', &
                  ' MMPT MPT> Error: Impossible to turn FLAG2 from 1 to 1')
                ENDIF
              ELSE IF (MPTCOMBI(MCBI,HBI,1).EQ.ATMA(3)) then
                IF (FLAG2.EQ.0 .OR. FLAG2.EQ.1) THEN
                  FLAG2=2
                ELSE
                  FLAG2=3
                ENDIF
              ELSE
                CALL WrnDie (-3,'<MISCOM>', &
                ' MMPT MPT> Error: New donor is not D or A')
              ENDIF

!DEFINE FLAG1 OF A-H-X SWAP
              IF (MPTCOMBI(MCBI,HBI,4).EQ.-1) THEN
                ! 1/0 ==> 0
                IF (FLAG1.EQ.1 .OR. FLAG1.EQ.0) THEN
                  FLAG1=0
                ! 2/3 ==> 1
                ELSE
                  FLAG1=1
                ENDIF
              ELSE
                ! 1/0 ==> 2
                IF (FLAG1.EQ.1 .OR. FLAG1.EQ.0) THEN
                  FLAG1=2
                ! 2/3 ==> 3
                ELSE
                  IF (MCBI.GT.1 &
                  .AND. MPTCOMBI(MCBI,HBI,1).EQ.MPTCOMBI(MCBLAST,HBI,1) &
                  .AND. MPTCOMBI(MCBI,HBI,2).EQ.MPTCOMBI(MCBLAST,HBI,2) &
                  .AND. MPTCOMBI(MCBI,HBI,3).EQ.MPTCOMBI(MCBLAST,HBI,3)) THEN
                    FLAG1=3
                  ELSE
                    CALL WrnDie (-3,'<MISCOM>', &
                    ' MMPT MPT> Error: Impossible order of MTF')
                  ENDIF
                ENDIF
              ENDIF

              IDX7=MPTCOMBI(MCBI,HBI,7)

              if (PRNLEV .GE. 7) then
                WRITE(outu,*) '------------------------------'
                WRITE(outu,*) 'MSMMPT-MCB> MCBI,FLAG1&2: ',MCBI,flag1,flag2
              endif

!CALC EMMPT2.MODE2 ENERGY FOR A-H-X SWAP
              IF (EMMPTLOOP.EQ.2 .AND. (FLAG1.EQ.2 .OR. FLAG1.EQ.3)) THEN
                IF (EDAFLAG(HBI,IDX7).EQ.0) THEN
!WRITE(OUTU,*) '!CALL EMMPT2-M2-F1> ', HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                  CALL EMMPT2(EDAMPT(HBI,IDX7),ETMP,X,Y,Z,EDADX(HBI,IDX7,:), &
                  EDADY(HBI,IDX7,:),EDADZ(HBI,IDX7,:), &
                  DX2(:),DY2(:),DZ2(:),NATOMX,HBI,2)
                  EDAFLAG(HBI,IDX7)=1
                ENDIF

                EMCB(MCBI)=EMCB(MCBI)+EDAMPT(HBI,IDX7)
                DO I=1,NATOMX
                  EMCBDX(MCBI,I)=EMCBDX(MCBI,I)+EDADX(HBI,IDX7,I)
                  EMCBDY(MCBI,I)=EMCBDY(MCBI,I)+EDADY(HBI,IDX7,I)
                  EMCBDZ(MCBI,I)=EMCBDZ(MCBI,I)+EDADZ(HBI,IDX7,I)
                ENDDO
              ENDIF
            ELSE
              FLAG1=-1
              FLAG2=-1

              if (PRNLEV .ge. 7) then
              WRITE(outu,*) '------------------------------'
              WRITE(outu,*) 'MSMMPT-MCB> FINALIZE AN ITERATION: LOOP',EMMPTLOOP,MCBI
              endif
            ENDIF

!PROCEED A-H-X SWAP IF NECESSARY
            IF (FLAG1.EQ.2 .OR. FLAG1.EQ.1 .OR. &
            (MCBI.EQ.(MCBNUM+1) .AND. MPTCOMBI(MCBLAST,HBI,4).NE.-1)) THEN
              IF (PRNLEV >= 7) THEN
                IF (FLAG1.EQ.2) THEN
                  WRITE (OUTU,*) 'MSMMPT-MCB> SWAP A-H-Y FORWARD:', &
                  HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                ELSE
                  WRITE (OUTU,*) 'MSMMPT-MCB> SWAP A-H-Y BACKWARD:', &
                  HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                ENDIF

                IF (PRNLEV.GE.8) THEN
                  CALL PRINTSWAP()
                  WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                ENDIF
              ENDIF

              IF (FLAG1.EQ.2) THEN
                IF (IFSWAP01.EQ.0) THEN
                  IFSWAP01=1
                ELSE
                  CALL WrnDie (-3,'<MISCOM>', 'MSMMPT-MCB> WRONG SWAP-01-1')
                ENDIF

                CALL BONDSWAP(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3))

              ELSE IF (FLAG1.EQ.1) THEN
                IF (IFSWAP01.EQ.1) THEN
                  IFSWAP01=0
                ELSE
                  CALL WrnDie (-3,'<MISCOM>', 'MSMMPT-MCB> WRONG SWAP-01-2')
                ENDIF

                CALL BONDSWAP(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3))

              ELSE IF (MCBI.EQ.(MCBNUM+1) .AND. IFSWAP01.EQ.1) THEN
                CALL BONDSWAP(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3))
                IFSWAP01=0
              ENDIF

              IF (PRNLEV >= 8) THEN
                CALL PRINTSWAP()
                WRITE(OUTU,*) '----------------------------------------'
              ENDIF

              ITMP=HBRDATOM(HBI,1)
              HBRDATOM(HBI,1)=HBRDATOM(HBI,3)
              HBRDATOM(HBI,3)=ITMP
            ENDIF

!PROCEED A-H-X BACKWARD SWITCH
            IF (MCBI.EQ.(MCBNUM+1) .OR. (MCBI.GT.1 .AND. &
            IDX7.NE.MPTCOMBI(MCBLAST,HBI,7))) THEN

              CALL MTFSWITCH(HBI,DHAT2,BNBND%INBLO,BNBND%JNB,0)
              IF (PRNLEV >= 7) THEN
                WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH A-H-Y BACKWARD:', &
                DHAT2(1),DHAT2(2),DHAT2(3)
                IF (PRNLEV.GE.8) THEN
                  CALL PRINTMMPT()
                  WRITE(OUTU,*) '----------------------------------------'
                ENDIF
              ENDIF

              HBRDATOM(HBI,1)=DHAT2(1)
              HBRDATOM(HBI,2)=DHAT2(2)
              HBRDATOM(HBI,3)=DHAT2(3)

            ENDIF

!CALC EMMPT2.MODE2 ENERGY FOR D-H-A SWAP
            IF (MCBI.LE.MCBNUM .AND. (FLAG2.EQ.2 .OR. FLAG2.EQ.3)) THEN
              IF (EMMPTLOOP.EQ.2) THEN
                IF (EDAFLAG(HBI,1).EQ.0) THEN
!WRITE(OUTU,*) 'CALL EMMPT2-M2-F2> ', HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                  CALL EMMPT2(EDAMPT(HBI,1),ETMP,X,Y,Z,EDADX(HBI,1,:), &
                  EDADY(HBI,1,:),EDADZ(HBI,1,:),DX2(:),DY2(:),DZ2(:),NATOMX,HBI,2)
                  EDAFLAG(HBI,1)=1
                ENDIF

                EMCB(MCBI)=EMCB(MCBI)+EDAMPT(HBI,1)
                DO I=1,NATOMX
                  EMCBDX(MCBI,I)=EMCBDX(MCBI,I)+EDADX(HBI,1,I)
                  EMCBDY(MCBI,I)=EMCBDY(MCBI,I)+EDADY(HBI,1,I)
                  EMCBDZ(MCBI,I)=EMCBDZ(MCBI,I)+EDADZ(HBI,1,I)
                ENDDO
              ENDIF

!PROCEED D-H-A SWAP IF NECESSARY
              IF (FLAG2.EQ.2 .AND. IFSWAP02.EQ.0) THEN
                IF (PRNLEV >= 7) THEN
                  IF (PRNLEV.GE.8) THEN
                    CALL PRINTSWAP()
                    WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                  ENDIF
                  WRITE (OUTU,*) 'MSMMPT-MCB> SWAP D-H-A FORWARD:', &
                  HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                ENDIF

                IFSWAP02=1
                CALL BONDSWAP(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3))

                IF (PRNLEV >= 8) THEN
                  CALL PRINTSWAP()
                  WRITE(OUTU,*) '----------------------------------------'
                ENDIF

                ITMP=HBRDATOM(HBI,1)
                HBRDATOM(HBI,1)=HBRDATOM(HBI,3)
                HBRDATOM(HBI,3)=ITMP
              ENDIF
            ELSE IF (IFSWAP02.EQ.1 .AND. (MCBI.EQ.(MCBNUM+1) .OR. &
            FLAG2.EQ.1)) THEN
              IF (PRNLEV >= 7) THEN
                IF (PRNLEV.GE.8) THEN
                  CALL PRINTSWAP()
                  WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                ENDIF
                WRITE (OUTU,*) 'MSMMPT-MCB> SWAP D-H-A BACKWARD:', &
                HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
              ENDIF

              CALL BONDSWAP(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3))

              IF (PRNLEV >= 8) THEN
                CALL PRINTSWAP()
                WRITE(OUTU,*) '----------------------------------------'
              ENDIF

              DHATEMP(1)=HBRDATOM(HBI,3)
              DHATEMP(2)=HBRDATOM(HBI,2)
              DHATEMP(3)=HBRDATOM(HBI,1)

              CALL MTFSWITCH(HBI,DHATEMP,BNBND%INBLO,BNBND%JNB,0)

              IF (PRNLEV >= 7) THEN
                WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH D-H-A RECOVERY:', &
                DHATEMP(1),DHATEMP(2),DHATEMP(3)

                IF (PRNLEV.GE.8) THEN
                  CALL PRINTMMPT()
                  WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                ENDIF
              ENDIF

              ITMP=HBRDATOM(HBI,1)
              HBRDATOM(HBI,1)=HBRDATOM(HBI,3)
              HBRDATOM(HBI,3)=ITMP
            ENDIF

!PROCEED A-H-X SWITCH IF NECESSARY
            IF (MCBI.LE.MCBNUM) THEN
              IF (FLAG1.EQ.2 .OR. FLAG1.EQ.3) THEN
                IF (PRNLEV >= 7) THEN
                  IF (PRNLEV.GE.8) THEN
                    CALL PRINTMMPT()
                    WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                  ENDIF

                  WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH A-H-Y FORWARD:', &
                  MPTCOMBI(MCBI,HBI,4),MPTCOMBI(MCBI,HBI,5),MPTCOMBI(MCBI,HBI,6)
                ENDIF

                CALL MTFSWITCH(HBI,MPTCOMBI(MCBI,HBI,4:6),BNBND%INBLO,BNBND%JNB,0)

                IF (PRNLEV >= 8) THEN
                  CALL PRINTMMPT()
                  WRITE(OUTU,*) '----------------------------------------'
                ENDIF
              ELSE
                IF (PRNLEV >= 7) THEN
                  IF (PRNLEV.GE.8) THEN
                    CALL PRINTMMPT()
                    WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                  ENDIF

                  WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH A-H-Y FORWARD:', &
                  MPTCOMBI(MCBI,HBI,1),MPTCOMBI(MCBI,HBI,2),MPTCOMBI(MCBI,HBI,3)
                ENDIF

                CALL MTFSWITCH(HBI,MPTCOMBI(MCBI,HBI,1:3),BNBND%INBLO,BNBND%JNB,0)

                IF (PRNLEV >= 8) THEN
                  CALL PRINTMMPT()
                  WRITE(OUTU,*) '----------------------------------------'
                ENDIF
              ENDIF

!CALC EMMPT2.MODE1
              IF (FLAG1.EQ.2 .OR. FLAG1.EQ.3) THEN
                DHATEMP(1)=HBRDATOM(HBI,1)
                DHATEMP(2)=HBRDATOM(HBI,2)
                DHATEMP(3)=HBRDATOM(HBI,3)
                HBRDATOM(HBI,1)=MPTCOMBI(MCBI,HBI,4)
                HBRDATOM(HBI,2)=MPTCOMBI(MCBI,HBI,5)
                HBRDATOM(HBI,3)=MPTCOMBI(MCBI,HBI,6)
              ELSE
                DHAT2(1)=HBRDATOM(HBI,1)
                DHAT2(2)=HBRDATOM(HBI,2)
                DHAT2(3)=HBRDATOM(HBI,3)
                HBRDATOM(HBI,1)=MPTCOMBI(MCBI,HBI,1)
                HBRDATOM(HBI,2)=MPTCOMBI(MCBI,HBI,2)
                HBRDATOM(HBI,3)=MPTCOMBI(MCBI,HBI,3)
              ENDIF

            !c'emmpt: EVALUATE ENERGY E_PT
              IF (WMODE.GE.1) THEN
!WRITE(OUTU,*) 'CALL EMMPT2-M1> ', HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)
                IF (EMMPTLOOP.EQ.1) THEN
                  CALL EMMPT2(EMPT(HBI,MCBPOS(MCBI,HBI)), &
                  EWMPT(HBI,MCBPOS(MCBI,HBI)),X,Y,Z, &
                  EMPTDX(HBI,MCBPOS(MCBI,HBI),:),EMPTDY(HBI,MCBPOS(MCBI,HBI),:), &
                  EMPTDZ(HBI,MCBPOS(MCBI,HBI),:),EWMPTDX(HBI,MCBPOS(MCBI,HBI),:), &
                  EWMPTDY(HBI,MCBPOS(MCBI,HBI),:),EWMPTDZ(HBI,MCBPOS(MCBI,HBI),:), &
                  NATOMX,HBI,1)
                ELSE
                  CALL EMMPT2(EMPT(HBI,MCBPOS(MCBI,HBI)),ETMP,X,Y,Z, &
                  EMPTDX(HBI,MCBPOS(MCBI,HBI),:),EMPTDY(HBI,MCBPOS(MCBI,HBI),:), &
                  EMPTDZ(HBI,MCBPOS(MCBI,HBI),:),DX2(:),DY2(:),DZ2(:), &
                  NATOMX,HBI,3)
                ENDIF
              ELSE
                CALL EMMPT2(EMPT(HBI,MCBPOS(MCBI,HBI)),ETMP,X,Y,Z, &
                EMPTDX(HBI,MCBPOS(MCBI,HBI),:),EMPTDY(HBI,MCBPOS(MCBI,HBI),:), &
                EMPTDZ(HBI,MCBPOS(MCBI,HBI),:),DX2(:),DY2(:),DZ2(:), &
                NATOMX,HBI,1)

                CALL EMMPT2(EMPT(HBI,MCBPOS(MCBI,HBI)),ETMP,X,Y,Z, &
                EMPTDX(HBI,MCBPOS(MCBI,HBI),:),EMPTDY(HBI,MCBPOS(MCBI,HBI),:), &
                EMPTDZ(HBI,MCBPOS(MCBI,HBI),:),DX2(:),DY2(:),DZ2(:), &
                NATOMX,HBI,3)
              ENDIF

              IF (FLAG1.EQ.2 .OR. FLAG1.EQ.3) THEN
                CALL MTFSWITCH(HBI,DHATEMP,BNBND%INBLO,BNBND%JNB,0)

                IF (PRNLEV >= 7) THEN
                  WRITE (OUTU,*) 'MSMMPT-MCB> SWITCH A-H-Y RECOVERY:', &
                  DHATEMP(1),DHATEMP(2),DHATEMP(3)

                  IF (PRNLEV.GE.8) THEN
                    CALL PRINTMMPT()
                    WRITE(OUTU,*) '- - - - - - - - - - - - - - - - - - - - '
                  ENDIF
                ENDIF

                HBRDATOM(HBI,1)=DHATEMP(1)
                HBRDATOM(HBI,2)=DHATEMP(2)
                HBRDATOM(HBI,3)=DHATEMP(3)
              ENDIF

              EMCB(MCBI)=EMCB(MCBI)+EMPT(HBI,MCBPOS(MCBI,HBI))
              DO I=1,NATOMX
                EMCBDX(MCBI,I)=EMCBDX(MCBI,I)+EMPTDX(HBI,MCBPOS(MCBI,HBI),I)
                EMCBDY(MCBI,I)=EMCBDY(MCBI,I)+EMPTDY(HBI,MCBPOS(MCBI,HBI),I)
                EMCBDZ(MCBI,I)=EMCBDZ(MCBI,I)+EMPTDZ(HBI,MCBPOS(MCBI,HBI),I)
                IF (PRNLEV >= 10)  WRITE(OUTU,*) 'TEST DX-YZ: EMCB-EMPT(', &
                MCBI,',',I,')',EMCBDX(MCBI,I),EMCBDY(MCBI,I),EMCBDZ(MCBI,I)
              ENDDO

              IF (WMODE.GE.1 .AND. EMMPTLOOP.EQ.1) THEN
                EWMCB(MCBI)=EWMCB(MCBI)+EWMPT(HBI,MCBPOS(MCBI,HBI))
                IF (HBI.EQ.1) THEN
                  DO I=1,NATOMX
                    EWMCBDX(MCBI,I)=EWMPTDX(HBI,MCBPOS(MCBI,HBI),I)
                    EWMCBDY(MCBI,I)=EWMPTDY(HBI,MCBPOS(MCBI,HBI),I)
                    EWMCBDZ(MCBI,I)=EWMPTDZ(HBI,MCBPOS(MCBI,HBI),I)
                  ENDDO
                ELSE
                  DO I=1,NATOMX
                    EWMCBDX(MCBI,I)=EWMCBDX(MCBI,I)+EWMPTDX(HBI,MCBPOS(MCBI,HBI),I)
                    EWMCBDY(MCBI,I)=EWMCBDY(MCBI,I)+EWMPTDY(HBI,MCBPOS(MCBI,HBI),I)
                    EWMCBDZ(MCBI,I)=EWMCBDZ(MCBI,I)+EWMPTDZ(HBI,MCBPOS(MCBI,HBI),I)
                  ENDDO
                ENDIF
              ENDIF

              !positioning the center of charge
              IF (MMPTPRNFLAG .AND. (WMODE.EQ.0 .OR. EMMPTLOOP.EQ.2)) THEN
                EMCBDX(MCBI,NATOMX+1)=EMCBDX(MCBI,NATOMX+1)+ &
                    EMPTDX(HBI,MCBPOS(MCBI,HBI),NATOMX+1)/real(HBRDNUM)
                EMCBDY(MCBI,NATOMX+1)=EMCBDY(MCBI,NATOMX+1)+ &
                    EMPTDY(HBI,MCBPOS(MCBI,HBI),NATOMX+1)/real(HBRDNUM)
                EMCBDZ(MCBI,NATOMX+1)=EMCBDZ(MCBI,NATOMX+1)+ &
                    EMPTDZ(HBI,MCBPOS(MCBI,HBI),NATOMX+1)/real(HBRDNUM)
              ENDIF

!IF (EMPTFLAG(HBI,MCBPOS(MCBI,HBI)).EQ.1) THEN
!CONTINUE
!ELSE
!!positioning the center of charge
!IF (PRNLEV.GE.7) THEN
!!WRITE (OUTU,*) 'MSMMPT TEST> ',xtmp(1),xtmp(2),xtmp(3)
!  EMPTDX(HBI,MCBPOS(MCBI,HBI),NATOMX+1)=XTMP(1)
!  EMPTDY(HBI,MCBPOS(MCBI,HBI),NATOMX+1)=XTMP(2)
!  EMPTDZ(HBI,MCBPOS(MCBI,HBI),NATOMX+1)=XTMP(3)
!ENDIF
!
!EMPTFLAG(HBI,MCBPOS(MCBI,HBI))=1
!ENDIF
            ENDIF
          ENDIF

          IF (MCBI.GT.1 .AND. (EMMPTLOOP.EQ.1 .OR. &
          MCBWFLAG(MCBI).EQ.1)) MCBLAST=MCBI

        ENDDO
      ENDDO

!####################################################
!##
!## IF ARMD IS DEACTIVATED
!##
!####################################################

      IF (PRM_DELTAV.LT.-0.D0 .AND. EMMPTLOOP.EQ.1) THEN !!prmlpe label
        EU=EMCB(1)
        DO I=1,NATOMX
          DX(I)=DX(I)+EMCBDX(1,I)
          DY(I)=DY(I)+EMCBDY(1,I)
          DZ(I)=DZ(I)+EMCBDZ(1,I)
        ENDDO

        !positioning the center of charge
        IF (MMPTPRNFLAG) WRITE(OUTU,'(1X,A20,F12.5,F12.5,F12.5)') &
        'MSMMPT-MPT> COOR-CEC: ', &
        EMCBDX(1,NATOMX+1),EMCBDY(1,NATOMX+1),EMCBDZ(1,NATOMX+1)
        EMCBDX(1,NATOMX+1)=0.d0
        EMCBDY(1,NATOMX+1)=0.d0
        EMCBDZ(1,NATOMX+1)=0.d0
        XTMP(1)=0.D0
        XTMP(2)=0.D0
        XTMP(3)=0.D0

        LOOP_NOT_EXIT=.FALSE.
      ENDIF

!####################################################
!##
!## evaluate w(i) by global function
!##
!####################################################

      IF (LOOP_NOT_EXIT) THEN
        IF (EMMPTLOOP.EQ.1) THEN
          IF (WMODE.GE.1) THEN
            IF (PRM_INTERNAL_EINDEX.EQ.1.D0) THEN
              EMINI=1
              EMIN=EWMCB(EMINI)
              DO I=1,MCBNUM
                IF (EMIN.GT.EWMCB(I)) THEN
                  EMIN=EWMCB(I)
                  EMINI=I
                ENDIF
              ENDDO
            ELSE
              EMIN=PRM_INTERNAL_EMIN
            ENDIF
          ELSE
            EMINI=1
            EMIN=EMCB(EMINI)
            DO I=1,MCBNUM
              IF (EMIN.GT.EMCB(I)) THEN
                EMIN=EMCB(I)
                EMINI=I
              ENDIF
            ENDDO
          ENDIF

          WGSUM=0.D0

          IF (WMODE.GE.1) THEN
            IF (PRM_INTERNAL_EINDEX.EQ.1.D0) THEN
              DO I=1,MCBNUM
                MCBWG(I)=DEXP(-1.D0*((EWMCB(I)-EMIN)/PRM_DELTAV))
                WGSUM=WGSUM+MCBWG(I)
              ENDDO
            ELSE
              DO I=1,MCBNUM
                MCBWG(I)=DEXP(-1.D0*((EWMCB(I)-EMIN)/PRM_DELTAV)**PRM_INTERNAL_EINDEX)
                WGSUM=WGSUM+MCBWG(I)
              ENDDO
            ENDIF
          ELSE
            DO I=1,MCBNUM
              MCBWG(I)=DEXP(-1.D0*((EMCB(I)-EMIN)/PRM_DELTAV))
              WGSUM=WGSUM+MCBWG(I)
            ENDDO
          ENDIF
        !w_i=w_i^0/sum(s_j)

          DO I=1,MCBNUM
            IF (MCBWG(I)/WGSUM.GE.PRM_INTERNAL_WG_CUT) MCBWFLAG(I)=1 !0.00000000001D0) MCBWFLAG(I)=1
          ENDDO
        !EMMPTLOOP==2
        ELSE
          WGSUM=0.D0

          DO I=1,MCBNUM
            MCBWG(I)=0.D0
            IF (PRM_INTERNAL_EINDEX.EQ.1.D0) THEN
              IF (MCBWFLAG(I).eq.1) THEN
                MCBWG(I)=DEXP(-1.D0*((EWMCB(I)-EMIN)/PRM_DELTAV))
                WGSUM=WGSUM+MCBWG(I)
              ENDIF
            ELSE
              IF (MCBWFLAG(I).eq.1) THEN
                MCBWG(I)=DEXP(-1.D0*((EWMCB(I)-EMIN)/PRM_DELTAV)**PRM_INTERNAL_EINDEX)
                WGSUM=WGSUM+MCBWG(I)
              ENDIF
            ENDIF
          ENDDO
        ENDIF
      ENDIF
!    call cpu_time(TEND)
!    write(outu,*) 'TEST-LOOP>',tend-tstart
!    tstart=tend
      EMMPTLOOP=EMMPTLOOP+1
    ENDDO

!####################################################
!##
!## CALCULATE GLOBAL TOTAL POTENTIAL ENERGY
!##
!####################################################

    IF (LOOP_NOT_EXIT) THEN
      !c'emmpt, calculate mixed energy by E=sum(w_i*E_i)
      EPOT=0.D0
      DO I=1,MCBNUM
        if (MCBWFLAG(I).EQ.1 .OR. WMODE.EQ.0)  &
        EPOT=EPOT+MCBWG(I)/WGSUM*EMCB(I)
      ENDDO

      EU=EPOT

  !WRITE(OUTU,*) 'TEST DX-YZ> ',EMCBDX(1,1),EMCBDY(1,1),EMCBDZ(1,1)

      !positions of the center of charge
      IF (MMPTPRNFLAG) THEN
        XTMP(1)=0.D0
        XTMP(2)=0.D0
        XTMP(3)=0.D0
        DO I=1,MCBNUM
          IF (MCBWFLAG(I).EQ.1 .OR. WMODE.EQ.0) THEN
            DXI=EMCBDX(I,NATOMX+1)-EMCBDX(1,NATOMX+1)
            DYI=EMCBDY(I,NATOMX+1)-EMCBDY(1,NATOMX+1)
            DZI=EMCBDZ(I,NATOMX+1)-EMCBDZ(1,NATOMX+1)

            IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
              IF (DXI.GT.0.5D0*PXSIZE) THEN
                DXI=DXI-PXSIZE
              ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
                DXI=DXI+PXSIZE
              ENDIF
              IF (DYI.GT.0.5D0*PYSIZE) THEN
                DYI=DYI-PYSIZE
              ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
                DYI=DYI+PYSIZE
              ENDIF
              IF (DZI.GT.0.5D0*PZSIZE) THEN
                DZI=DZI-PZSIZE
              ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
                DZI=DZI+PZSIZE
              ENDIF
            ENDIF

            XTMP(1)=XTMP(1)+MCBWG(I)/WGSUM*(EMCBDX(1,NATOMX+1)+DXI)
            XTMP(2)=XTMP(2)+MCBWG(I)/WGSUM*(EMCBDY(1,NATOMX+1)+DYI)
            XTMP(3)=XTMP(3)+MCBWG(I)/WGSUM*(EMCBDZ(1,NATOMX+1)+DZI)
          ENDIF
        ENDDO
        WRITE(OUTU,'(1X,A20,F12.5,F12.5,F12.5)') 'MSMMPT-MPT> COOR-CEC: ', &
        XTMP(1),XTMP(2),XTMP(3)
        DO I=1,MCBNUM
          EMCBDX(I,NATOMX+1)=0.d0
          EMCBDY(I,NATOMX+1)=0.d0
          EMCBDZ(I,NATOMX+1)=0.d0
        ENDDO

        !dipole
        XTMP(1)=0.D0
        XTMP(2)=0.D0
        XTMP(3)=0.D0
        XTOR=0.D0
        YTOR=0.D0
        ZTOR=0.D0
        DO I=1,NATOMX
          XTOR=XTOR+X(I)
          YTOR=YTOR+Y(I)
          ZTOR=ZTOR+Z(I)
        ENDDO
        XTOR=XTOR/REAL(NATOMX)
        YTOR=YTOR/REAL(NATOMX)
        ZTOR=ZTOR/REAL(NATOMX)
        DO I=1,NATOMX
          XTMP(1)=XTMP(1)+(X(I)-XTOR)*CG(I)
          XTMP(2)=XTMP(2)+(Y(I)-YTOR)*CG(I)
          XTMP(3)=XTMP(3)+(Z(I)-ZTOR)*CG(I)
        ENDDO
        IF (MMPTPRNFLAG) WRITE(OUTU,'(1X,A18,F12.5,F12.5,F12.5)') &
        'MSMMPT-MPT> DIPOLE: ',XTMP(1),XTMP(2),XTMP(3)
        XTMP(1)=0.D0
        XTMP(2)=0.D0
        XTMP(3)=0.D0
  !      DO I=1,NATOMX
  !        XTMP(1)=XTMP(1)+MMPTVX(I)*CG(I)
  !        XTMP(2)=XTMP(2)+MMPTVY(I)*CG(I)
  !        XTMP(3)=XTMP(3)+MMPTVZ(I)*CG(I)
  !!IF (MMPTPRNFLAG) WRITE(OUTU,'(A10,1X,I5,F8.3,F8.3,F8.3,F8.3,F8.3,F8.3)') &
  !!'VELO>',i,XCOMP(I),YCOMP(I),ZCOMP(I),X(I),Y(I),Z(I)
  !      ENDDO
        IF (MMPTPRNFLAG) WRITE(OUTU,'(1X,A18,F12.5,F12.5,F12.5,F12.5)') &
        'MSMMPT-MPT> VDIP: ',real(MMPTSTP)*STEPSIZE,XTMP(1),XTMP(2),XTMP(3)

        !calculate center of mass
        XTMP(1)=0.D0
        XTMP(2)=0.D0
        XTMP(3)=0.D0
        DTMP=0.D0

        DO I=1,NATOMX
          DTMP=DTMP+AMASS(I)
          XTMP(1)=XTMP(1)+X(I)*AMASS(I)
          XTMP(2)=XTMP(2)+Y(I)*AMASS(I)
          XTMP(3)=XTMP(3)+Z(I)*AMASS(I)
        ENDDO
        XTMP(1)=XTMP(1)/DTMP
        XTMP(2)=XTMP(2)/DTMP
        XTMP(3)=XTMP(3)/DTMP
        !WRITE(OUTU,'(1X,A21,F12.5,F12.5,F12.5)') 'MSMMPT-MPT> COOR-COFM: ', &
        !XTMP(1),XTMP(2),XTMP(3)
        DO I=1,MCBNUM
          EMCBDX(I,NATOMX+1)=0.d0
          EMCBDY(I,NATOMX+1)=0.d0
          EMCBDZ(I,NATOMX+1)=0.d0
        ENDDO
      ENDIF

      !print out effective states
      IF (MMPTPRNFLAG) THEN


        WRITE(OUTU,*) 'MSMMPT-MPT> ENERGY of MMPT'

        WRITE(OUTU,*) '----------------------------------------&
        ---------------------------------'
        WRITE(OUTU,*) '     No.                  MOTIF                              &
        EMSMMPT            EW               Weight        R(O-O)'
        DO I=1,MCBNUM
          IF (MCBWFLAG(I).EQ.1 .OR. WMODE.EQ.0) THEN
            IF (MPTCOMBI(I,1,4).EQ.-1) THEN
              DTMP=MDIST(MPTCOMBI(I,1,1),MPTCOMBI(I,1,3))
            ELSE
              DTMP=MDIST(MPTCOMBI(I,1,4),MPTCOMBI(I,1,6))
            ENDIF

            IF (MPTCOMBI(I,1,4).GT.-1) THEN
              A1=MPTCOMBI(I,1,4)
              A2=MPTCOMBI(I,1,5)
              A3=MPTCOMBI(I,1,6)
            ELSE
              A1=MPTCOMBI(I,1,1)
              A2=MPTCOMBI(I,1,2)
              A3=MPTCOMBI(I,1,3)
            ENDIF

            IF (A3.LT.A1) THEN
              ITMP=A1
              A1=A3
              A3=ITMP
            ENDIF

            WRITE(OUTU,2500) I,' [ ',MPTCOMBI(I,1,1),'   ', &
            MPTCOMBI(I,1,2),'   ',MPTCOMBI(I,1,3) &
            ,'   ',MPTCOMBI(I,1,4),'   ',MPTCOMBI(I,1,5),'   ', &
            MPTCOMBI(I,1,6),' ] ',EMCB(I),EWMCB(I),MCBWG(I)/WGSUM,DTMP,&
            '   ',MMPTSTP,' XCODE',a1,a2,a3
          ENDIF
        ENDDO
        WRITE(OUTU,*) 'TOTAL: ',EPOT
        WRITE(OUTU,*) '----------------------------------------&
        ---------------------------------'
2500  FORMAT('    ',I4,A3,I4,A3,I4,A3,I4,A3,I4,A3,I4,A3,I4,A3,2X, &
F12.5,2X,F12.5,1X,F20.12,2X,F12.5,A3,I10,A6,I4.4,I4.4,I4.4)
      ENDIF

      !c'emmpt: calculate gradient of energy
      DO I=1,NATOMX
        DX2(I)=0.D0
        DY2(I)=0.D0
        DZ2(I)=0.D0
      ENDDO

      IF (WMODE.GE.1) THEN
        IF (PRM_INTERNAL_EINDEX .EQ. 1) THEN
          DO J=1,MCBNUM
            IF (MCBWFLAG(J).EQ.1) THEN
              ETEMP=(EMCB(J)-EPOT)/PRM_DELTAV
              DO I=1,NATOMX
                DX2(I)=DX2(I)+MCBWG(J)*(EMCBDX(J,I)-ETEMP*EWMCBDX(J,I))
                DY2(I)=DY2(I)+MCBWG(J)*(EMCBDY(J,I)-ETEMP*EWMCBDY(J,I))
                DZ2(I)=DZ2(I)+MCBWG(J)*(EMCBDZ(J,I)-ETEMP*EWMCBDZ(J,I))
              ENDDO
            ENDIF
          ENDDO
        ELSE
          DO J=1,MCBNUM
            IF (MCBWFLAG(J).EQ.1) THEN
              ETEMP=(EMCB(J)-EPOT)*PRM_INTERNAL_EINDEX/PRM_DELTAV*((EWMCB(J)-EMIN)/PRM_DELTAV)**(PRM_INTERNAL_EINDEX-1.d0)
              DO I=1,NATOMX
                DX2(I)=DX2(I)+MCBWG(J)*(EMCBDX(J,I)-ETEMP*EWMCBDX(J,I))
                DY2(I)=DY2(I)+MCBWG(J)*(EMCBDY(J,I)-ETEMP*EWMCBDY(J,I))
                DZ2(I)=DZ2(I)+MCBWG(J)*(EMCBDZ(J,I)-ETEMP*EWMCBDZ(J,I))
              ENDDO
            ENDIF
          ENDDO
        ENDIF
      ELSE
        DO J=1,MCBNUM
          ETEMP=(EMCB(J)-EPOT)/PRM_DELTAV
          DO I=1,NATOMX
            DX2(I)=DX2(I)+MCBWG(J)*EMCBDX(J,I)*(1.0D0-ETEMP)
            DY2(I)=DY2(I)+MCBWG(J)*EMCBDY(J,I)*(1.0D0-ETEMP)
            DZ2(I)=DZ2(I)+MCBWG(J)*EMCBDZ(J,I)*(1.0D0-ETEMP)
          ENDDO
        ENDDO
      ENDIF
  !do i=1,natomx
  !  WRITE(OUTU,*) 'Force_INI > ',DX(I),DY(I),DZ(I)
  !enddo
      DO I=1,NATOMX

        DX2(I)=DX2(I)/WGSUM
        DY2(I)=DY2(I)/WGSUM
        DZ2(I)=DZ2(I)/WGSUM

        DX(I)=DX(I)+DX2(I)
        DY(I)=DY(I)+DY2(I)
        DZ(I)=DZ(I)+DZ2(I)

      ENDDO
  !if (PRNLEV.ge.7) then
  !!do i=1,natomx
  !!  WRITE(OUTU,*) 'Force_ADD > ',DX2(I),DY2(I),DZ2(I)
  !!enddo
  !do i=1,natomx
  !  WRITE(OUTU,*) 'Force_FIN > ',DX(I),DY(I),DZ(I)
  !enddo
  !endif
      !c'emmpt: switch the motif


      MAX_WEIGHT=0.D0
      EMIN1=999999999.D0
      EMIN2=999999999.D0
      RM1=0.d0
      RM2=0.d0
      MINI1=0
      MINI2=0
      WEIGHT2=-1.D0

      DO I=1,MCBNUM
        IF (MCBWFLAG(I).EQ.1 .OR. WMODE.EQ.0) THEN
          IF (MMPTPRNFLAG) THEN
            IF (EWMCB(I).LT.EMIN2) THEN
              IF (EWMCB(I).LT.EMIN1) THEN
                MCBJ=I
                WEIGHT2=MAX_WEIGHT
                MAX_WEIGHT=MCBWG(I)/WGSUM
                EMIN2=EMIN1
                EMIN1=EWMCB(I)

                RM2=RM1
                RM1=MDIST(MPTCOMBI(I,1,1),MPTCOMBI(I,1,3))

                MINI2=MINI1
                MINI1=I
              ELSE
                WEIGHT2=MCBWG(I)/WGSUM
                EMIN2=EWMCB(I)

                RM2=MDIST(MPTCOMBI(I,1,1),MPTCOMBI(I,1,3))

                MINI2=I
              ENDIF
            ENDIF
          ELSE
            IF (MCBWG(I)/WGSUM.GT.MAX_WEIGHT) THEN
              MCBJ=I
              MAX_WEIGHT=MCBWG(I)/WGSUM

              !GOTO 2790
            ENDIF
          ENDIF
        ENDIF
      ENDDO

      !print out analysis
      IF (MMPTPRNFLAG) THEN
        ITMP=0
        A1=-1
        A2=0
        DO I=1,MCBNUM
          IF (MCBWFLAG(I).EQ.1) THEN
            ITMP=ITMP+1
            IF (MPTCOMBI(I,1,7).NE.A1) THEN
              A2=A2+1
              A1=MPTCOMBI(I,1,7)
            ENDIF
          ENDIF
        ENDDO
        K=A2
        J=ITMP

        IF (MPTCOMBI(MINI1,1,4).GT.-1) THEN
          A1=MPTCOMBI(MINI1,1,4)
          A2=MPTCOMBI(MINI1,1,5)
          A3=MPTCOMBI(MINI1,1,6)
        ELSE
          A1=MPTCOMBI(MINI1,1,1)
          A2=MPTCOMBI(MINI1,1,2)
          A3=MPTCOMBI(MINI1,1,3)
        ENDIF
        IF (MPTCOMBI(MINI2,1,4).GT.-1) THEN
          ATMB(1)=MPTCOMBI(MINI2,1,4)
          ATMB(2)=MPTCOMBI(MINI2,1,5)
          ATMB(3)=MPTCOMBI(MINI2,1,6)
        ELSE
          ATMB(1)=MPTCOMBI(MINI2,1,1)
          ATMB(2)=MPTCOMBI(MINI2,1,2)
          ATMB(3)=MPTCOMBI(MINI2,1,3)
        ENDIF
        IF (A1.EQ.ATMB(1)) THEN
          D(1)=MDIST(A3,A2)-MDIST(A2,A1)
          D(2)=MDIST(ATMB(1),ATMB(2))-MDIST(ATMB(2),ATMB(3))
        ELSE IF (A1.EQ.ATMB(3)) THEN
          D(1)=MDIST(A3,A2)-MDIST(A2,A1)
          D(2)=MDIST(ATMB(3),ATMB(2))-MDIST(ATMB(2),ATMB(1))
        ELSE IF (A3.EQ.ATMB(1)) THEN
          D(1)=MDIST(A1,A2)-MDIST(A2,A3)
          D(2)=MDIST(ATMB(1),ATMB(2))-MDIST(ATMB(2),ATMB(3))
        ELSE IF (A3.EQ.ATMB(3)) THEN
          D(1)=MDIST(A1,A2)-MDIST(A2,A3)
          D(2)=MDIST(ATMB(3),ATMB(2))-MDIST(ATMB(2),ATMB(1))
        ELSE
          D(1)=-999.0D0
          D(2)=-999.0D0
        ENDIF

        IF (A3.LT.A1) THEN
          ITMP=A1
          A1=A3
          A3=ITMP
        ENDIF
        WRITE(OUTU, '(A36,f12.5,1x,I4,I4,I4,I4,A20,I4,I4)') &
        ' MMPT-MPT> NSTATE-NCLASS-MIN1-MIN2: ',real(MMPTSTP)*STEPSIZE,MCBNUM, MPTCOMBI(MCBNUM,1,7),MINI1, MINI2, &
        ' EFFECTIVE(NS-NC): ',J,K
        WRITE(OUTU, '(A22,f12.5,1x,F8.6,A8,F8.6,2x,I4.4,I4.4,I4.4)') &
        ' MMPT-MPT> MAXWEIGHT: ',real(MMPTSTP)*STEPSIZE,MAX_WEIGHT,'WDIFF: ',MAX_WEIGHT-WEIGHT2,A1,A2,A3
        WRITE(OUTU, '(A18,F12.5,A8,F12.5,A5,F8.5,A5,F8.5,A8,F8.5,F8.5)') &
        ' MMPT-MPT> EMIN1: ',EMIN1,' EDIFF: ',EMIN2-EMIN1, &
        ' R1: ',RM1,' R2: ',RM2,' V1-V2: ', D(1),D(2)


        !calculate index for O* and H* and delta(Nature 1999)
        !one excess proton only
        A1=MPTCOMBI(MINI1,1,1)
        A2=MPTCOMBI(MINI1,1,2)
        A3=MPTCOMBI(MINI1,1,3)

        D(1)=MDIST(A1,A2)-MDIST(A3,A2)
        IF (D(1).GT.0) A1=A3
        D(1)=ABS(D(1))
        ATMA(1)=A2
        ITMP=1

  !if (mini1.ne.1 .and. a1.ne.MPTCOMBI(MINI1,1,1)) then
  !write(outu,*) 'TEST>',mini1,a1,a2
  !endif

        DO I=2,BONDNUM(1)
          IF (BONDMMPT(1,I,2).EQ.A1) THEN
            A2=BONDMMPT(1,I,3)
            ITMP=ITMP+1
            ATMA(ITMP)=A2
            D(ITMP)=9999999.D0
            DO J=2,MCBNUM
              IF (MPTCOMBI(J,1,1).EQ.A1 .AND. MPTCOMBI(J,1,2).EQ.A2 &
              .AND. MPTCOMBI(J,1,4).EQ.-1) THEN
                DTMP=ABS(MDIST(A2,MPTCOMBI(J,1,3))-MDIST(A2,A1))
                IF (D(ITMP).GT.DTMP) D(ITMP)=DTMP
              ENDIF
            ENDDO
          ELSE IF (BONDMMPT(1,I,3).EQ.A1) THEN
            A2=BONDMMPT(1,I,2)
            ITMP=ITMP+1
            ATMA(ITMP)=A2
            D(ITMP)=9999999.D0
            DO J=2,MCBNUM
              IF (MPTCOMBI(J,1,1).EQ.A1 .AND. MPTCOMBI(J,1,2).EQ.A2 &
              .AND. MPTCOMBI(J,1,4).EQ.-1) THEN
                DTMP=ABS(MDIST(A2,MPTCOMBI(J,1,3))-MDIST(A2,A1))
                IF (D(ITMP).GT.DTMP) D(ITMP)=DTMP
              ENDIF
            ENDDO
          ENDIF
        ENDDO

        DTMP=9999999.D0
        DO I=1,3
          IF (DTMP.GT.D(I)) THEN
            DTMP=D(I)
            ITMP=ATMA(I)
          ENDIF
        ENDDO
        A2=ITMP
        !calculate hopping distance for O* and H* from their initial positions
        IF ((DYNAMQ .and. MMPTSTP.LE.MMPTCYC) .or. MMPTSTP.LT.MMPTCYC) THEN
          !initialization
          DSX(1)=X(A1)
          DSY(1)=Y(A1)
          DSZ(1)=Z(A1)
          DSX(2)=X(A1)
          DSY(2)=Y(A1)
          DSZ(2)=Z(A1)

          HSX(1)=X(A2)
          HSY(1)=Y(A2)
          HSZ(1)=Z(A2)
          HSX(2)=X(A2)
          HSY(2)=Y(A2)
          HSZ(2)=Z(A2)

          DSX(4)=0.0D0
          DSY(4)=0.0D0
          DSZ(4)=0.0D0
          HSX(4)=0.0D0
          HSY(4)=0.0D0
          HSZ(4)=0.0D0
        ENDIF
        DSX(3)=X(A1)
        DSY(3)=Y(A1)
        DSZ(3)=Z(A1)
        IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
          DXI=DSX(3)-DSX(2)
          DYI=DSY(3)-DSY(2)
          DZI=DSZ(3)-DSZ(2)
          IF (DXI.GT.0.5D0*PXSIZE) THEN
            DSX(4)=DSX(4)-1.D0
          ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
            DSX(4)=DSX(4)+1.D0
          ENDIF
          IF (DYI.GT.0.5D0*PYSIZE) THEN
            DSY(4)=DSY(4)-1.D0
          ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
            DSY(4)=DSY(4)+1.D0
          ENDIF
          IF (DZI.GT.0.5D0*PZSIZE) THEN
            DSZ(4)=DSZ(4)-1.D0
          ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
            DSZ(4)=DSZ(4)+1.D0
          ENDIF
        ENDIF
        D(1)=SQRT((DSX(3)+DSX(4)*PXSIZE-DSX(1))**2+(DSY(3)+DSY(4)*PYSIZE-DSY(1))**2 &
        +(DSZ(3)+DSZ(4)*PZSIZE-DSZ(1))**2)
        DSX(2)=DSX(3)
        DSY(2)=DSY(3)
        DSZ(2)=DSZ(3)
        !continue
        HSX(3)=X(A2)
        HSY(3)=Y(A2)
        HSZ(3)=Z(A2)
        IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
          DXI=HSX(3)-HSX(2)
          DYI=HSY(3)-HSY(2)
          DZI=HSZ(3)-HSZ(2)
          IF (DXI.GT.0.5D0*PXSIZE) THEN
            HSX(4)=HSX(4)-1.D0
          ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
            HSX(4)=HSX(4)+1.D0
          ENDIF
          IF (DYI.GT.0.5D0*PYSIZE) THEN
            HSY(4)=HSY(4)-1.D0
          ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
            HSY(4)=HSY(4)+1.D0
          ENDIF
          IF (DZI.GT.0.5D0*PZSIZE) THEN
            HSZ(4)=HSZ(4)-1.D0
          ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
            HSZ(4)=HSZ(4)+1.D0
          ENDIF
        ENDIF
        D(2)=SQRT((HSX(3)+HSX(4)*PXSIZE-HSX(1))**2+(HSY(3)+HSY(4)*PYSIZE-HSY(1))**2 &
        +(HSZ(3)+HSZ(4)*PZSIZE-HSZ(1))**2)
        HSX(2)=HSX(3)
        HSY(2)=HSY(3)
        HSZ(2)=HSZ(3)
        WRITE(OUTU, '(A35,F12.5,I8,I8,F12.5,F12.5,F12.5)') &
        ' MMPT-MPT> TIME-D-HS-DELTA-D/H-JUMP: ',real(MMPTSTP)*STEPSIZE, &
        A1,A2,DTMP,D(1),D(2)

        !calculate g(r) for O* and H*
        I=1
        ITMP=0
        do WHILE (I.LE.NBOND .AND. ITMP.LT.3)
          IF (IB(I).EQ.A1) THEN
            ITMP=ITMP+1
            ATMA(ITMP)=JB(I)
          ELSE IF (JB(I).EQ.A1) THEN
            ITMP=ITMP+1
            ATMA(ITMP)=IB(I)
          ENDIF
          I=I+1
        ENDDO

        IF (ITMP.EQ.2) THEN
          ITMP=MPTCOMBI(MINI1,1,2)
          IF (ITMP.NE.ATMA(1) .AND. ITMP.NE.ATMA(2)) THEN
            CONTINUE
          ELSE
            ITMP=MPTCOMBI(1,1,2)
          ENDIF
          ATMA(3)=ITMP
        ENDIF

        D(1)=99.d0
        D(2)=99.d0
        D(3)=99.d0
        DO I=1,NATOMX
          IF (I.NE.A1 .AND. I.NE.ATMA(1) .AND. I.NE.ATMA(2) .AND. I.NE.ATMA(3)) THEN
            IF (ATYPE(I)(1:1) .EQ. 'H') THEN
              DTMP=MDIST(A1,I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GOH(ITMP)=GOH(ITMP)+1.d0
              ENDIF

              !DTMP=MDIST(A2,I)
              !IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
              !  ITMP=NINT(DTMP/PRM_GOFRBIN)
              !  GHH(ITMP)=GHH(ITMP)+1.d0
              !ENDIF
              DTMP=MDIST(ATMA(1),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHH(ITMP)=GHH(ITMP)+1.d0
              ENDIF
              DTMP=MDIST(ATMA(2),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHH(ITMP)=GHH(ITMP)+1.d0
              ENDIF
              DTMP=MDIST(ATMA(3),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHH(ITMP)=GHH(ITMP)+1.d0
              ENDIF
            ELSE IF (ATYPE(I)(1:1) .EQ. 'O') THEN
              DTMP=MDIST(A1,I)
              IF (DTMP.LT.D(3)) THEN
                D(3)=DTMP
                IF (D(3).LT.D(2)) THEN
                  XTOR=D(2)
                  D(2)=D(3)
                  D(3)=XTOR
                  IF (D(2).LT.D(1)) THEN
                    XTOR=D(1)
                    D(1)=D(2)
                    D(2)=XTOR
                  ENDIF
                ENDIF
              ENDIF
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GOO(ITMP)=GOO(ITMP)+1.d0
              ENDIF

              !DTMP=MDIST(A2,I)
              !IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
              !  ITMP=NINT(DTMP/PRM_GOFRBIN)
              !  GHO(ITMP)=GHO(ITMP)+1.d0
              !ENDIF
              DTMP=MDIST(ATMA(1),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHO(ITMP)=GHO(ITMP)+1.d0
              ENDIF
              DTMP=MDIST(ATMA(2),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHO(ITMP)=GHO(ITMP)+1.d0
              ENDIF
              DTMP=MDIST(ATMA(3),I)
              IF (DTMP.LT.(PRM_GOFRMAX+PRM_GOFRBIN/2.D0)) THEN
                ITMP=NINT(DTMP/PRM_GOFRBIN)
                GHO(ITMP)=GHO(ITMP)+1.d0
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDIF
      !WRITE(OUTU, '(A22,F12.5,F12.5,F12.5)') &
      !' MMPT-MPT> EIGEN-O-O: ',D(1),D(2),D(3)


      IF (MMPTSTP.GE.(MMPTMAXSTEP-1) .AND. MMPTMXFLAG) THEN
        MMPTMXFLAG=.FALSE.
        DO I=1,NINT(PRM_GOFRMAX/PRM_GOFRBIN)
          DTMP=4.D0*3.1415926536d0*(REAL(I)*PRM_GOFRBIN)**2*PRM_GOFRBIN* &
          real(INT(MMPTSTP/MMPTCYC))*0.03342d0
          GOO(I)=GOO(I)/DTMP
          GOH(I)=GOH(I)/DTMP/2.d0
          GHO(I)=GHO(I)/DTMP/3.D0
          GHH(I)=GHH(I)/DTMP/6.d0
          WRITE(OUTU,'(A11,1X,F12.5,1X,F12.5,1X,F12.5,1X,F12.5,1X,F12.5)') &
          ' MMPT-GOFR>',REAL(I)*PRM_GOFRBIN,GOO(I),GOH(I),GHO(I),GHH(I)
        ENDDO
      ENDIF


!2790  CONTINUE


      IF (MCBJ.NE.1) THEN
        DO I=1,HBRDNUM
          IF (MPTCOMBI(MCBJ,I,1).EQ.HBRDATOM(I,1) .AND. &
          MPTCOMBI(MCBJ,I,2).EQ.HBRDATOM(I,2) .AND. &
          MPTCOMBI(MCBJ,I,3).EQ.HBRDATOM(I,3)) THEN
            CONTINUE
          ELSE
            HBRDFLAG(I,1)=MPTCOMBI(MCBJ,I,1)
            HBRDFLAG(I,2)=MPTCOMBI(MCBJ,I,2)
            HBRDFLAG(I,3)=MPTCOMBI(MCBJ,I,3)
            IF (MPTCOMBI(MCBJ,I,1).EQ.HBRDATOM(I,3)) THEN
              HBRDFLAG(I,4)=1
            ELSE
              HBRDFLAG(I,4)=0
            ENDIF
          ENDIF
        ENDDO
      ELSE
        CONTINUE
      ENDIF
    ENDIF

!call cpu_time(TEND)
!write(outu,*) 'TEST-WG>',tend-tstart
!tstart=tend

    ETMP=EU

    IF (NUMCONS.GT.0) THEN
      CALL ECONSTRAINT(NATOMX,EU,X,Y,Z,DX,DY,DZ)

      IF (PRNLEV >= 7) THEN
      write(outu,*) 'MSMMPT-ECONS> ',EU-ETMP
      endif
    ENDIF


    IF (MMPTCYC.GT.0) THEN
      IF (MMPTSMODE.NE.2) THEN
        IF (MMPTSFLAG) THEN
          LASTSTEP=MMPTSTP
        ELSE
          IF (MMPTSKP.GT.0 .AND. MMPTSTP.EQ.1) THEN
            MMPTSKP=MMPTSKP-1
          ENDIF

          IF (MMPTSKP.EQ.0) MMPTSFLAG=.TRUE.
        ENDIF
      ELSE
        IF (MMPTSFLAG) THEN
          LASTSTEP=MMPTSTP
        ELSE
          IF (MMPTSKP.GT.0 .AND. (MMPTSTP.EQ.1 .OR. &
          (MMPTSTP.GT.0 .AND. MMPTSTP.LE.LASTSTEP))) THEN
            MMPTSKP=MMPTSKP-1
          ENDIF

          IF (MMPTSKP.EQ.0)  THEN
            MMPTSFLAG=.TRUE.
            MMPTSTP=-1
          ELSE IF (MMPTSKP.EQ.1) THEN
            LASTSTEP=MMPTSTP
          ENDIF
        ENDIF
      ENDIF
    ENDIF
!call cpu_time(TEND)
!write(outu,*) 'TEST-TIME-0>',tend-tstart0
!write(outu,*) 'TEST------------------------'

  END SUBROUTINE EMSMMPT



!      _______________________________________________________



  SUBROUTINE PLZFCT(I,DA,HA,AA)

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
    use coord

      INTEGER I,DA,HA,AA
      real(chm_real) RDA,RDH,RHO,PLZFCTDA,PLZFCTHA,PLZFCTAA

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((X(AA)-X(DA))**2+(Y(AA)-Y(DA))**2+(Z(AA)-Z(DA))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((X(HA)-X(DA))**2+(Y(HA)-Y(DA))**2+(Z(HA)-Z(DA))**2)


!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)

!      FUNCTION OF POLARIZATION OF DONOR ATOM
      PLZFCTDA = EXP(-12.32D0*(RHO+0.17D0))-0.01
!      FUNCTION OF POLARIZATION OF ACCEPTOR ATOM
      PLZFCTAA = EXP(12.32D0*(RHO-1.17D0))-0.01
!      FUNCTION OF POLARIZATION OF HYDROGEN ATOM
!      PLZFCTHA = -PLZFCTDA-PLZFCTAA

      CG(AA)=INITCHRG(I,3)-INITCHRG(I,3)*PLZFCTAA
      CG(DA)=INITCHRG(I,1)-INITCHRG(I,1)*PLZFCTDA
      CG(HA)=INITCHRG(I,2)+INITCHRG(I,1)*PLZFCTDA+INITCHRG(I,3)*PLZFCTAA


!      WRITE(OUTU,*)  CG(DA)+CG(HA)+CG(AA)

  END SUBROUTINE PLZFCT


!      _______________________________________________________




  SUBROUTINE EMSPTOHO(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
!    use deriv
!    use coord


!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      real(chm_real) DXI,DYI,DZI
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3)

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,          &
            DRHODY1, DRHODY2, DRHODY3,           &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,           &
            DRDADY1, DRDADY2, DRDADY3,           &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,        &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,        &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3

     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,     &
            DTHETADY1, DTHETADY2, DTHETADY3,     &
            DTHETADZ1, DTHETADZ2, DTHETADZ3


     real(chm_real) DX1, DX2, DX3,                       &
            DY1, DY2, DY3,                       &
            DZ1, DZ2, DZ3

!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD




!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)


      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        DXI=XA(2)-XA(1)
        DYI=YA(2)-YA(1)
        DZI=ZA(2)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(2)=XA(2)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(2)=XA(2)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(2)=YA(2)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(2)=YA(2)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)+PZSIZE
        ENDIF

        DXI=XA(3)-XA(1)
        DYI=YA(3)-YA(1)
        DZI=ZA(3)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(3)=XA(3)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(3)=XA(3)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(3)=YA(3)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(3)=YA(3)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)+PZSIZE
        ENDIF
      ENDIF

!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)



!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG







!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.8) THEN
      write(outu,*) 'entering OHO mmpt routine'
      WRITE(OUTU,*) 'MSMMPT>   RDA', RDA
      WRITE(OUTU,*) 'MSMMPT>   RDH', RDH
      WRITE(OUTU,*) 'MSMMPT>   RHO', RHO
      WRITE(OUTU,*) 'MSMMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MSMMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MSMMPT>   THETA', THETA
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMOHO(1)*(1-DEXP(-PRMOHO(2)*(RDA-PRMOHO(3))))**2+PRMOHO(4)
      B =PRMOHO(5)+PRMOHO(6)*RDA
      RE=PRMOHO(7)*DEXP(-PRMOHO(8)*RDA)+PRMOHO(9)
      H1=PRMOHO(10)



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMOHO(1)*(1.D0-DEXP(-PRMOHO(2)*(RDA-PRMOHO(3))))    &
            *PRMOHO(2)*DEXP(-PRMOHO(2)*(RDA-PRMOHO(3)))
      DB = PRMOHO(6)
      DRE= -PRMOHO(7)*PRMOHO(8)*DEXP(-PRMOHO(8)*RDA)
      DH1=0.D0


!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)

      V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                            &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                        &
         -DEQ+PRMOHO(11)

     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                       &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))                 &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                  &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))                 &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))       &
              -DDEQ



     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
               *B*DEXP(-B*(RHO-RE))                                   &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))                   &
               *B*DEXP(-B*(1-RHO-RE))


!      HARMONIC RESTRAINT:

      HR =  H1*THETA**2

      DHRDTHET = 2.D0*H1*THETA

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0
      DV0DRDA=SCLOHO*DV0DRDA
      DV0DRHO=SCLOHO*DV0DRHO

!      SUM TERMS

      VTOT = V0 + HR

      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))          &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))        &
             /(( RDA-1.6D0)**2*RDA)

     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))        &
             /(( RDA-1.6D0)**2*RDA)

     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))        &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA






!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E.
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE
!      D(THETA)/D(U)*D(U)/D(X1)


      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1
      DTHETADX2=DTDCOST*DCOSTDX2
      DTHETADX3=DTDCOST*DCOSTDX3

      DTHETADY1=DTDCOST*DCOSTDY1
      DTHETADY2=DTDCOST*DCOSTDY2
      DTHETADY3=DTDCOST*DCOSTDY3

      DTHETADZ1=DTDCOST*DCOSTDZ1
      DTHETADZ2=DTDCOST*DCOSTDZ2
      DTHETADZ3=DTDCOST*DCOSTDZ3


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3


      IF(PRNLEV.GT.8) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT ENERGY: ', VTOT
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',AA,DX3,DY3,DZ3

      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EMSPTOHO






!      _______________________________________________________
!      _______________________________________________________


  SUBROUTINE EMSPTNHN(DA,HA,AA,EU)



    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
    use deriv
    use coord


!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      real(chm_real) DXI,DYI,DZI
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3)

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,     &
            DRHODY1, DRHODY2, DRHODY3,      &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,      &
            DRDADY1, DRDADY2, DRDADY3,      &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,   &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,   &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3

     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,&
            DTHETADY1, DTHETADY2, DTHETADY3,&
            DTHETADZ1, DTHETADZ2, DTHETADZ3


     real(chm_real) DX1, DX2, DX3,                  &
            DY1, DY2, DY3,                  &
            DZ1, DZ2, DZ3

!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD


!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) =X(DA)
      YA(1) =Y(DA)
      ZA(1) =Z(DA)

      XA(2) =X(HA)
      YA(2) =Y(HA)
      ZA(2) =Z(HA)

      XA(3) =X(AA)
      YA(3) =Y(AA)
      ZA(3) =Z(AA)

      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        DXI=XA(2)-XA(1)
        DYI=YA(2)-YA(1)
        DZI=ZA(2)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(2)=XA(2)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(2)=XA(2)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(2)=YA(2)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(2)=YA(2)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)+PZSIZE
        ENDIF

        DXI=XA(3)-XA(1)
        DYI=YA(3)-YA(1)
        DZI=ZA(3)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(3)=XA(3)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(3)=XA(3)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(3)=YA(3)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(3)=YA(3)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)+PZSIZE
        ENDIF
      ENDIF


!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))       &
             +(YA(3)-YA(1))*(YA(2)-YA(1))        &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)



!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG





!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)





      IF(PRNLEV.GT.8) THEN
      write(outu,*) 'entering NHN mmpt routine'
      WRITE(OUTU,*) 'MSMMPT>   RDA', RDA
      WRITE(OUTU,*) 'MSMMPT>   RDH', RDH
      WRITE(OUTU,*) 'MSMMPT>   RHO', RHO
      WRITE(OUTU,*) 'MSMMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MSMMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MSMMPT>   THETA', THETA
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMNHN(1)*(1-DEXP(-PRMNHN(2)*(RDA-PRMNHN(3))))**2+PRMNHN(4)
      B =PRMNHN(5)+PRMNHN(6)*RDA
      RE=PRMNHN(7)*DEXP(-PRMNHN(8)*RDA)+PRMNHN(9)
      H1=PRMNHN(10)



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMNHN(1)*(1.D0-DEXP(-PRMNHN(2)*(RDA-PRMNHN(3))))   &
            *PRMNHN(2)*DEXP(-PRMNHN(2)*(RDA-PRMNHN(3)))
     DB = PRMNHN(6)
     DRE= -PRMNHN(7)*PRMNHN(8)*DEXP(-PRMNHN(8)*RDA)
     DH1=0.D0


!     POTENTIAL FUNKTION (DOUBLE MORSE POTENTIAL)

     V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                         &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                    &
         -DEQ+PRMNHN(11)

     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                   &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                  &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))             &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2              &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))             &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))   &
              -DDEQ



     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                  &
               *B*DEXP(-B*(RHO-RE))                               &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))               &
               *B*DEXP(-B*(1-RHO-RE))




!      HARMONIC RESTRAINT:

      HR =  H1*THETA**2

      DHRDTHET = 2.D0*H1*THETA


!      POTENTIAL ENERGY SCALING

      V0=SCLNHN*V0
      DV0DRDA=SCLNHN*DV0DRDA
      DV0DRHO=SCLNHN*DV0DRHO

!      SUM TERMS

      VTOT = V0 + HR

      EU = EU + VTOT



!      RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =(-XA(2)+XA(1))/(RDH*(RDA-1.6))  &
             -((RDH-0.8D0)*(-XA(3)+XA(1)))     &
             /((RDA-1.6D0)**2*RDA)
     DRHODX2 =(XA(2)-XA(1))/(RDH*(RDA-1.6D0))
     DRHODX3 =-(RDH-0.8D0)*(XA(3)-XA(1))       &
             /((RDA-1.6D0)**2*RDA)

     DRHODY1 =(-YA(2)+YA(1))/(RDH*(RDA-1.6))   &
             -((RDH-0.8D0)*(-YA(3)+YA(1)))     &
             /((RDA-1.6D0)**2*RDA)
     DRHODY2 =(YA(2)-YA(1))/(RDH*(RDA-1.6D0))
     DRHODY3 =-(RDH-0.8D0)*(YA(3)-YA(1))       &
             /((RDA-1.6D0)**2*RDA)

     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*(RDA-1.6))   &
             -((RDH-0.8D0)*(-ZA(3)+ZA(1)))     &
             /((RDA-1.6D0)**2*RDA)
     DRHODZ2 =(ZA(2)-ZA(1))/(RDH*(RDA-1.6D0))
     DRHODZ3 =-(RDH-0.8D0)*(ZA(3)-ZA(1))       &
             /((RDA-1.6D0)**2*RDA)




!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA



!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E.
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE
!      D(THETA)/D(U)*D(U)/D(X1)


      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                      &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1
      DTHETADX2=DTDCOST*DCOSTDX2
      DTHETADX3=DTDCOST*DCOSTDX3

      DTHETADY1=DTDCOST*DCOSTDY1
      DTHETADY2=DTDCOST*DCOSTDY2
      DTHETADY3=DTDCOST*DCOSTDY3

      DTHETADZ1=DTDCOST*DCOSTDZ1
      DTHETADZ2=DTDCOST*DCOSTDZ2
      DTHETADZ3=DTDCOST*DCOSTDZ3


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3


      IF(PRNLEV.GT.8) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT ENERGY: ', VTOT
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',AA,DX3,DY3,DZ3

      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EMSPTNHN






!      _______________________________________________________





  SUBROUTINE EMSPTNHO(DA,HA,AA,EU)




    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use deriv
    use param
!    use energym
    use coord

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      real(chm_real) DXI,DYI,DZI
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3)

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,       &
            DRHODY1, DRHODY2, DRHODY3,        &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,        &
            DRDADY1, DRDADY2, DRDADY3,        &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,     &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,     &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3

     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,  &
            DTHETADY1, DTHETADY2, DTHETADY3,  &
            DTHETADZ1, DTHETADZ2, DTHETADZ3


      real(chm_real) DX1, DX2, DX3,     &
            DY1, DY2, DY3,      &
            DZ1, DZ2, DZ3

!     PARAMETER

     real(chm_real) DEQ1, DEQ2, BETA1, BETA2, REQ1, REQ2, C, T,     &
            DDEQ1, DDEQ2, DBETA1, DBETA2, DREQ1, DREQ2, DC, DT


      real(chm_real) DOTPROD


      integer i

!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) =X(DA)
      YA(1) =Y(DA)
      ZA(1) =Z(DA)

      XA(2) =X(HA)
      YA(2) =Y(HA)
      ZA(2) =Z(HA)

      XA(3) =X(AA)
      YA(3) =Y(AA)
      ZA(3) =Z(AA)


      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        DXI=XA(2)-XA(1)
        DYI=YA(2)-YA(1)
        DZI=ZA(2)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(2)=XA(2)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(2)=XA(2)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(2)=YA(2)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(2)=YA(2)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)+PZSIZE
        ENDIF

        DXI=XA(3)-XA(1)
        DYI=YA(3)-YA(1)
        DZI=ZA(3)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(3)=XA(3)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(3)=XA(3)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(3)=YA(3)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(3)=YA(3)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)+PZSIZE
        ENDIF
      ENDIF

!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)



!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG








!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.8) THEN
      write(outu,*) 'entering NHO mmpt routine'
      WRITE(OUTU,*) 'MSMMPT>   RDA', RDA
      WRITE(OUTU,*) 'MSMMPT>   RDH', RDH
      WRITE(OUTU,*) 'MSMMPT>   RHO', RHO
      WRITE(OUTU,*) 'MSMMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MSMMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MSMMPT>   THETA', THETA
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER, PARAMETER FUNCTION ARE FUNCTIONS
!      OF RDA


      DEQ1 =PRMNHO(1)*(1.D0-dexp(-PRMNHO(2)*(RDA-PRMNHO(3))))**2       &
          +PRMNHO(4)


     BETA1=PRMNHO(5)/(1.D0+DEXP(-PRMNHO(6)*(RDA-PRMNHO(7))))

!     write(outu,*) 'beta1', beta1
!     write(outu,*) 'using', prmnho(5),prmnho(6),prmnho(7)


     REQ1 =PRMNHO(8)*(1.D0-dexp(-PRMNHO(9)*(RDA-PRMNHO(10))))**2       &
          +PRMNHO(11)

     DEQ2 =PRMNHO(12)*(1.D0-dexp(-PRMNHO(13)*(RDA-PRMNHO(14))))**2     &
          +PRMNHO(15)

     BETA2=PRMNHO(16)/(1.D0+DEXP(-PRMNHO(17)*(RDA-PRMNHO(18))))

     REQ2 =PRMNHO(19)*(1.D0-dexp(-PRMNHO(20)*(RDA-PRMNHO(21))))**2     &
          +PRMNHO(22)

      C    =PRMNHO(23)*(1.D0-dexp(-PRMNHO(24)*(RDA-PRMNHO(25))))**2    &
          +PRMNHO(26)

      T    =PRMNHO(27)

      IF(PRNLEV.GT.8) THEN
      DO I=1, NPRMNHO
         write(outu,*) 'MSMMPT>  PARAM', I, PRMNHO(I)
      ENDDO
      ENDIF

!      write(outu,*) 'parameter functions return:'
!      write(outu,*) deq1,beta1,req1,deq2,beta2,req2,c,t


!      DERIVE PARAMETER FUNCTIONS P(RDA)


      ddeq1 = 2*prmnho(1)*(1-exp(-prmnho(2)*(rda-prmnho(3))))*         &
               prmnho(2)*exp(-prmnho(2)*(rda-prmnho(3)))


     dbeta1 =  prmnho(5)*prmnho(6)*exp(-prmnho(6)*(rda-prmnho(7)))/    &
               (1+exp(-prmnho(6)*(rda-prmnho(7))))**2


!     write(outu,*) 'dbeta1', dbeta1
!     write(outu,*) 'using', prmnho(5),prmnho(6),prmnho(7)



     dreq1 = 2*prmnho(8)*(1-exp(-prmnho(9)*(rda-prmnho(10))))*         &
               prmnho(9)*exp(-prmnho(9)*(rda-prmnho(10)))

     ddeq2 = 2*prmnho(12)*(1-exp(-prmnho(13)*(rda-prmnho(14))))*       &
               prmnho(13)*exp(-prmnho(13)*(rda-prmnho(14)))

     dbeta2 =  prmnho(16)*prmnho(17)*exp(-prmnho(17)*(rda-prmnho(18)))/&
               (1+exp(-prmnho(17)*(rda-prmnho(18))))**2

     dreq2 = 2*prmnho(19)*(1-exp(-prmnho(20)*(rda-prmnho(21))))*       &
               prmnho(20)*exp(-prmnho(20)*(rda-prmnho(21)))


     dc    = 2*prmnho(23)*(1-exp(-prmnho(24)*(rda-prmnho(25))))*       &
               prmnho(24)*exp(-prmnho(24)*(rda-prmnho(25)))



!      write(outu,*) 'derivative parameter functions return:'
!      write(outu,*) ddeq1,dbeta1,dreq1,ddeq2,dbeta2,dreq2,dc,dt





!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)
! ... in contrast to eptoho or eptnhn we have two independent potentials

      v0 = Deq1*(1.d0-dexp(-beta1*(rho-Req1)))**2                      &
         +Deq2*(1.d0-dexp(-beta2*(Req2-rho)))**2                       &
         -c



     dv0drda = dDeq1*(1-exp(-beta1*(rho-Req1)))**2                     &
             -2*Deq1*(1-exp(-beta1*(rho-Req1)))*                       &
              (-dbeta1*(rho-Req1)+beta1*dReq1)*exp(-beta1*(rho-Req1))  &
              +dDeq2*(1-exp(-beta2*(Req2-rho)))**2                     &
             -2*Deq2*(1-exp(-beta2*(Req2-rho)))*                       &
              (-dbeta2*(Req2-rho)-beta2*dReq2)*exp(-beta2*(Req2-rho))  &
             -dc


!      write(outu,*) 'dv0drda', dv0drda


!      write(outu,*) dDeq1*(1-exp(-beta1*(rho-Req1)))**2
!      write(outu,*) 'using',ddeq1,beta1,rho,req1

!      write(outu,*) -2*Deq1*(1-exp(-beta1*(rho-Req1)))

       dv0drho = 2*Deq1*(1-exp(-beta1*(rho-Req1)))*  &
                beta1*exp(-beta1*(rho-Req1))         &
               -2*Deq2*(1-exp(-beta2*(Req2-rho)))*   &
                beta2*exp(-beta2*(Req2-rho))


!      write(outu,*) 'dv0drho', dv0drho



!      HARMONIC RESTRAINT:

      HR =  T*THETA**2

      DHRDTHET = 2.D0*T*THETA

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0
      DV0DRDA=SCLOHO*DV0DRDA
      DV0DRHO=SCLOHO*DV0DRHO

!      SUM TERMS

      VTOT = V0 + HR

      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))       &
             /(( RDA-1.6D0)**2*RDA)
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))         &
             /(( RDA-1.6D0)**2*RDA)

     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))            &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))       &
             /(( RDA-1.6D0)**2*RDA)
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))         &
             /(( RDA-1.6D0)**2*RDA)

     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))            &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))       &
             /(( RDA-1.6D0)**2*RDA)
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))         &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA


!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E.
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE
!      D(THETA)/D(U)*D(U)/D(X1)


      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1
      DTHETADX2=DTDCOST*DCOSTDX2
      DTHETADX3=DTDCOST*DCOSTDX3

      DTHETADY1=DTDCOST*DCOSTDY1
      DTHETADY2=DTDCOST*DCOSTDY2
      DTHETADY3=DTDCOST*DCOSTDY3

      DTHETADZ1=DTDCOST*DCOSTDZ1
      DTHETADZ2=DTDCOST*DCOSTDZ2
      DTHETADZ3=DTDCOST*DCOSTDZ3


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3


      IF(PRNLEV.GT.8) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT ENERGY: ', VTOT
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',AA,DX3,DY3,DZ3

      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EMSPTNHO
!      _________________________________________________________________________

  SUBROUTINE EMSPTOHOLE(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)
!!remove 'use deriv' and 'use coord' also to be applied for other models
!     NEW PES FORMAT: THE LEGENDRE EXPANSION
!     SEPT. 2007

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
!    use deriv
    use param
!    use energym
!    use coord


!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO
      real(chm_real) R1,R2
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      real(chm_real) DXI,DYI,DZI
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!      VTOT: TOTAL MMPT ENERGY


      real(chm_real) VTOT, V0
      INTEGER L
      PARAMETER (L=10)
      real(chm_real) V(L)
      real(chm_real) FX(3), FY(3), FZ(3)

      real(chm_real) THETA, COSTHETA, SINTHETA,ALPHA
!      PARTIAL DERIVATIVES
      real(chm_real) DVDRDA, DVDRHO, DVDTHET, DTDCOST
      real(chm_real) DV0DRDA, DV0DRHO
      real(chm_real) DVLDRDA(L), DVLDRHO(L)

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,    &
            DRHODY1, DRHODY2, DRHODY3,     &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,     &
            DRDADY1, DRDADY2, DRDADY3,     &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,  &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,  &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3


     real(chm_real) DX1, DX2, DX3, DY1, DY2, DY3, DZ1, DZ2, DZ3

!      PARAMETER
      real(chm_real) PRMA0, PRMA1, PRMA2, PRMA3, PRMA4
      real(chm_real) PRMB0(L), PRMB1(L), PRMB2(L)
!      AND THEIR DERIVATIVE TO RDA
      real(chm_real) DPRMA0, DPRMA1, DPRMA2, DPRMA3, DPRMA4
      real(chm_real) PRMA5, DPRMA5
      real(chm_real) DPRMB0(L), DPRMB1(L), DPRMB2(L)

!     LOOP INDEX FOR CALCULATING POTENTIAL
      INTEGER IL

      real(chm_real) DOTPROD,DOTPRODDHA

!     LEGENDRE POLYNOMIALS
      real(chm_real) P, DP
      DIMENSION P(0:L)
      DIMENSION DP(0:L)




!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)


      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        DXI=XA(2)-XA(1)
        DYI=YA(2)-YA(1)
        DZI=ZA(2)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(2)=XA(2)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(2)=XA(2)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(2)=YA(2)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(2)=YA(2)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)+PZSIZE
        ENDIF

        DXI=XA(3)-XA(1)
        DYI=YA(3)-YA(1)
        DZI=ZA(3)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(3)=XA(3)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(3)=XA(3)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(3)=YA(3)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(3)=YA(3)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)+PZSIZE
        ENDIF
      ENDIF

!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)
!      IF
!      SINTHETA = DSQRT(1.D0-COSTHETA**2)

!      THETA = DACOS(COSTHETA)

!

!      CONVERSION RAD -> DEG

!      THETA = THETA*RADDEG






!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)


      IF(PRNLEV.GE.7) THEN
        THETA = DACOS(COSTHETA)
        THETA = THETA*RADDEG
        DOTPRODDHA =(XA(2)-XA(3))*(XA(2)-XA(1))   &
        +(YA(2)-YA(3))*(YA(2)-YA(1))+(ZA(2)-ZA(3))*(ZA(2)-ZA(1))
        ALPHA=DACOS(DOTPRODDHA/(RDH*DSQRT((XA(2)-XA(3))**2+ &
        (YA(2)-YA(3))**2+(ZA(2)-ZA(3))**2)))
        ALPHA=ALPHA*RADDEG
        WRITE(OUTU,*) 'MSMMPT-MOTIF> D-H-A=',DA,HA,AA
        WRITE(OUTU,4030) 'MSMMPT-MOTIF> (DA,DH,RHO,THETA,ALPHA)=(', &
        RDA,' ,',RDH,' ,',RHO,' ,',THETA,' ,',ALPHA,')'
4030 FORMAT(A38,F10.3,A2,F10.3,A2,F10.3,A2,F10.3,A2,F10.3,A1)
      ENDIF



!      POTENTIAL FUNCTION

!     ZEROTH ORDER
!      INITIALIZE PARAMETER
!      AND DERIVE PARAMETER FUNCTIONS P(RDA):
      PRMA0=PRMLPE(1)*(TANH(PRMLPE(2)*(RDA-PRMLPE(3)))+PRMLPE(4))
      PRMA1=PRMLPE(5)*(TANH(PRMLPE(6)*(RDA-PRMLPE(7)))+PRMLPE(8))
      PRMA2=PRMLPE(9)*(TANH(PRMLPE(10)*(RDA-PRMLPE(11)))+PRMLPE(12))
      PRMA3=PRMLPE(13)*(TANH(PRMLPE(14)*(RDA-PRMLPE(15)))+PRMLPE(16))
      PRMA4=PRMLPE(17)*(TANH(PRMLPE(18)*(RDA-PRMLPE(19)))+PRMLPE(20))
      PRMA5=PRMLPE(21)*(TANH(PRMLPE(22)*(RDA-PRMLPE(23)))+PRMLPE(24))

      DPRMA0=PRMLPE(1)*PRMLPE(2)                             &
            *(1.D0-(TANH(PRMLPE(2)*(RDA-PRMLPE(3))))**2)
     DPRMA1=PRMLPE(5)*PRMLPE(6)                              &
            *(1.D0-(TANH(PRMLPE(6)*(RDA-PRMLPE(7))))**2)
     DPRMA2=PRMLPE(9)*PRMLPE(10)                             &
            *(1.D0-(TANH(PRMLPE(10)*(RDA-PRMLPE(11))))**2)
     DPRMA3=PRMLPE(13)*PRMLPE(14)                            &
            *(1.D0-(TANH(PRMLPE(14)*(RDA-PRMLPE(15))))**2)
     DPRMA4=PRMLPE(17)*PRMLPE(18)                            &
            *(1.D0-(TANH(PRMLPE(18)*(RDA-PRMLPE(19))))**2)
     DPRMA5=PRMLPE(21)*PRMLPE(22)                            &
            *(1.D0-(TANH(PRMLPE(22)*(RDA-PRMLPE(23))))**2)



      V0 = PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))**2             &
         +PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))**2         &
         -1.D0*PRMA5                                            &
         +PRMA3*DEXP(-PRMA4*(RHO-0.5D0)**2)

     DV0DRDA = DPRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))**2        &
              -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))       &
              *(-DPRMA1*(RHO-PRMA2)+PRMA1*DPRMA2)               &
              *DEXP(-PRMA1*(RHO-PRMA2))                         &
              +DPRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))**2   &
              -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))  &
              *(-DPRMA1*(1.D0-RHO-PRMA2)+PRMA1*DPRMA2)          &
              *DEXP(-PRMA1*(1.D0-RHO-PRMA2))                    &
              -1.D0*DPRMA5                                      &
              +DPRMA3*DEXP(-PRMA4*(RHO-0.5D0)**2)               &
              -1.D0*PRMA3*DPRMA4                                &
              *DEXP(-PRMA4*(RHO-0.5D0)**2)*(RHO-0.5D0)**2

      DV0DRHO = 2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))      &
                *PRMA1*DEXP(-PRMA1*(RHO-PRMA2))                 &
               -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2))) &
                *PRMA1*DEXP(-PRMA1*(1.D0-RHO-PRMA2))            &
               -2.D0*PRMA3*PRMA4*(RHO-0.5D0)                    &
                *DEXP(-PRMA4*(RHO-0.5D0)**2)

!      V0=0.D0
!      DV0DRDA = 0.D0
!      DV0DRHO =0.D0

!      HIGHER ORDER

      DO IL=1, L
        PRMB0(IL)=PRMLPE(9*IL+16)
        PRMB1(IL)=PRMLPE(9*IL+17)                                    &
                 *(TANH(PRMLPE(9*IL+18)*(RDA-PRMLPE(9*IL+19)))       &
                   +PRMLPE(9*IL+20))
       PRMB2(IL)=PRMLPE(9*IL+21)+                                    &
                 PRMLPE(9*IL+22)*(RDA-PRMLPE(9*IL+23))**2+           &
                 PRMLPE(9*IL+24)*(RDA-PRMLPE(9*IL+23))**4


       DPRMB1(IL)=PRMLPE(9*IL+17)*PRMLPE(9*IL+18)*                   &
            (1.D0-(TANH(PRMLPE(9*IL+18)*(RDA-PRMLPE(9*IL+19))))**2)
       DPRMB2(IL)=2.D0*PRMLPE(9*IL+22)*(RDA-PRMLPE(9*IL+23))+        &
               4.0D0*PRMLPE(9*IL+24)*(RDA-PRMLPE(9*IL+23))**3

        V(IL)=0.D0
        DVLDRDA(IL)=0.D0
        DVLDRHO(IL)=0.D0
        IF ((IL.EQ.1).OR.(IL.EQ.3)) THEN
!        Here we directly skip terms that equil zero
        V(IL)=PRMB0(IL)                                              &
             +PRMB1(IL)/(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2))

       DVLDRDA(IL)=DPRMB1(IL)*((RHO-0.5D0)**2-PRMB1(IL)**2)          &
                   /(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2)**2)     &
                   -DPRMB2(IL)*PRMB1(IL)                             &
                   /(((RHO-0.5D0)**2+PRMB1(IL)**2)*PRMB2(IL)**2)

       DVLDRHO(IL)=-2.D0*PRMB1(IL)*(RHO-0.5D0)                       &
                   /(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2)**2)
        ENDIF

      ENDDO

      CALL LGNDMS(L, COSTHETA, P, DP)

!     SUM ALL THE TERMS

      VTOT = V0+0.D0
      DVDRDA = DV0DRDA
      DVDRHO = DV0DRHO
      DVDTHET= 0.D0



      DO  IL=1, L
        VTOT = VTOT + V(IL)*P(IL)
        DVDRDA = DVDRDA + DVLDRDA(IL)*P(IL)
        DVDRHO = DVDRHO + DVLDRHO(IL)*P(IL)
!       HERE WE ACCTUALLY CALCULATE d(V)/d(cos(theta))
        DVDTHET = DVDTHET + DP(IL)*V(IL)
      ENDDO

      IF(PRNLEV.GT.8) THEN
      write(outu,*) 'Print every part of potential'
      WRITE(OUTU,*) 'MSMMPT>   V0', V0
      WRITE(OUTU,*) 'MSMMPT>   V1', V(1)
      WRITE(OUTU,*) 'MSMMPT>   V3', V(3)
      WRITE(OUTU,*) 'MSMMPT>   V', V(2), V(8), V(9), V(10)
      WRITE(OUTU,*) 'MSMMPT>   V', V(4), V(5), V(6), V(7)
      WRITE(OUTU,*) 'MSMMPT>   P1', P(1)
      WRITE(OUTU,*) 'MSMMPT>   P3', P(3)
      WRITE(OUTU,*) 'MSMMPT>   DP1', DP(1)
      WRITE(OUTU,*) 'MSMMPT>   DP3', DP(3)
      WRITE(OUTU,*) 'MSMMPT>   DVDRDA', DVDRDA
      WRITE(OUTU,*) 'MSMMPT>   DVDRHO', DVDRHO
      WRITE(OUTU,*) 'MSMMPT>   DVDTHET', DVDTHET
      ENDIF

!      POTENTIAL ENERGY SCALING
!     AND ALSO ADD MINUS SIGN
      VTOT= -1.D0*SCLOHO*VTOT
      DVDRDA=-1.D0*SCLOHO*DVDRDA
      DVDRHO=-1.D0*SCLOHO*DVDRHO
      DVDTHET=-1.D0*SCLOHO*DVDTHET

!      ADD TO THE TOTAL ENERGY



      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))       &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))   &
             /(( RDA-1.6D0)**2*RDA)
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))     &
             /(( RDA-1.6D0)**2*RDA)

     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))        &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))   &
             /(( RDA-1.6D0)**2*RDA)
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))     &
             /(( RDA-1.6D0)**2*RDA)

     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))        &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))   &
             /(( RDA-1.6D0)**2*RDA)
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))     &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA








      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                      &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)




!      CALCULATE FORCES

      DX1 = DVDRHO*DRHODX1 + DVDRDA*DRDADX1 + DVDTHET*DCOSTDX1
      DX2 = DVDRHO*DRHODX2 + DVDRDA*DRDADX2 + DVDTHET*DCOSTDX2
      DX3 = DVDRHO*DRHODX3 + DVDRDA*DRDADX3 + DVDTHET*DCOSTDX3

      DY1 = DVDRHO*DRHODY1 + DVDRDA*DRDADY1 + DVDTHET*DCOSTDY1
      DY2 = DVDRHO*DRHODY2 + DVDRDA*DRDADY2 + DVDTHET*DCOSTDY2
      DY3 = DVDRHO*DRHODY3 + DVDRDA*DRDADY3 + DVDTHET*DCOSTDY3

      DZ1 = DVDRHO*DRHODZ1 + DVDRDA*DRDADZ1 + DVDTHET*DCOSTDZ1
      DZ2 = DVDRHO*DRHODZ2 + DVDRDA*DRDADZ2 + DVDTHET*DCOSTDZ2
      DZ3 = DVDRHO*DRHODZ3 + DVDRDA*DRDADZ3 + DVDTHET*DCOSTDZ3


      IF(PRNLEV.GT.8) THEN

        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> PT ENERGY: ', VTOT
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',DA,DX1,DY1,DZ1
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',HA,DX2,DY2,DZ2
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',AA,DX3,DY3,DZ3
        WRITE(OUTU,*) 'MSMMPT> D DX1',DRHODX1,DRDADX1
        WRITE(OUTU,*) 'MSMMPT> D DX2',DRHODX2,DRDADX2
        WRITE(OUTU,*) 'MSMMPT> D DX3',DRHODX3,DRDADX3
        WRITE(OUTU,*) 'MSMMPT> D DY1',DRHODY1,DRDADY1
        WRITE(OUTU,*) 'MSMMPT> D DY2',DRHODY2,DRDADY2
        WRITE(OUTU,*) 'MSMMPT> D DY3',DRHODY3,DRDADY3
        WRITE(OUTU,*) 'MSMMPT> D DZ1',DRHODZ1,DRDADZ1
        WRITE(OUTU,*) 'MSMMPT> D DZ2',DRHODZ2,DRDADZ2
        WRITE(OUTU,*) 'MSMMPT> D DZ3',DRHODZ3,DRDADZ3
        WRITE(OUTU,*) 'end>   V0, VTOT', V0, VTOT

      ENDIF
      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EMSPTOHOLE

! _________________________________________________________________________

  SUBROUTINE EMSPTNL(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)


    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
!    use deriv
!    use coord


!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      real(chm_real) DXI,DYI,DZI
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3) ,dhrdrda,dhrdrho,dhrdxyz,optts,rah1

      real(chm_real) THETA, ALPHA, COSTHETA, DOTPRODDHA, sintheta , &
      kmin,kts,dmin,dts,dvar
!      PARTIAL DERIVATIVES
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST, kfin, dfin

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,          &
            DRHODY1, DRHODY2, DRHODY3,           &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,           &
            DRDADY1, DRDADY2, DRDADY3,           &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,        &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,        &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3

     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,     &
            DTHETADY1, DTHETADY2, DTHETADY3,     &
            DTHETADZ1, DTHETADZ2, DTHETADZ3


     real(chm_real) DX1, DX2, DX3,                       &
            DY1, DY2, DY3,                       &
            DZ1, DZ2, DZ3

!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD




!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)


      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        DXI=XA(2)-XA(1)
        DYI=YA(2)-YA(1)
        DZI=ZA(2)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(2)=XA(2)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(2)=XA(2)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(2)=YA(2)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(2)=YA(2)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(2)=ZA(2)+PZSIZE
        ENDIF

        DXI=XA(3)-XA(1)
        DYI=YA(3)-YA(1)
        DZI=ZA(3)-ZA(1)
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          XA(3)=XA(3)-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          XA(3)=XA(3)+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          YA(3)=YA(3)-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          YA(3)=YA(3)+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          ZA(3)=ZA(3)+PZSIZE
        ENDIF
      ENDIF

!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
              +(YA(3)-YA(1))*(YA(2)-YA(1))    &
              +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)
      sintheta=DSQRT(1.D0-COSTHETA**2)


!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG
      dvar=RDH*sintheta






!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH*costheta - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GE.7) THEN
        ALPHA=DACOS(DOTPRODDHA/(RDH*DSQRT((XA(2)-XA(3))**2+ &
        (YA(2)-YA(3))**2+(ZA(2)-ZA(3))**2)))
        ALPHA=ALPHA*RADDEG
        WRITE(OUTU,*) 'MSMMPT-MOTIF> D-H-A=',DA,HA,AA
        WRITE(OUTU,4840) 'MSMMPT-MOTIF> (DA,DH,RHO,THETA,ALPHA)=(', &
        RDA,' ,',RDH,' ,',RHO,' ,',THETA,' ,',ALPHA,')'
4840 FORMAT(A38,F10.3,A2,F10.3,A2,F10.3,A2,F10.3,A2,F10.3,A1)
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMNLM(1)*(1-DEXP(-PRMNLM(2)*(RDA-PRMNLM(3))))**2+PRMNLM(4)
      B =PRMNLM(5)+PRMNLM(6)*RDA
      RE=PRMNLM(7)*DEXP(-PRMNLM(8)*RDA)+PRMNLM(9)
      H1=PRMNLM(10)
      kmin= PRMNLM(10)
      kts=PRMNLM(12)
      dmin=PRMNLM(13)
      dts=PRMNLM(14)



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMNLM(1)*(1.D0-DEXP(-PRMNLM(2)*(RDA-PRMNLM(3))))  &
            *PRMNLM(2)*DEXP(-PRMNLM(2)*(RDA-PRMNLM(3)))
      DB = PRMNLM(6)
      DRE= -PRMNLM(7)*PRMNLM(8)*DEXP(-PRMNLM(8)*RDA)
      DH1=0.D0


!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)

      V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                            &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                        &
         -DEQ+PRMNLM(11)

     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                       &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))                 &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                  &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))                 &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))       &
              -DDEQ



     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
               *B*DEXP(-B*(RHO-RE))                                   &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))                   &
               *B*DEXP(-B*(1-RHO-RE))


!      HARMONIC RESTRAINT:

        kfin=kts+kmin*V0
        dfin=dts+dmin*(rho-0.5d0)**2
      HR = 0.5d0*kfin*(dvar-dfin)**2

      DHRDTHET = kfin*(dvar-dfin)*RDH*costheta

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0
      DV0DRDA=SCLOHO*DV0DRDA
      DV0DRHO=SCLOHO*DV0DRHO

!      SUM TERMS

      VTOT = V0 + HR

      EU = EU + VTOT

! ... RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =costheta*(-XA(2)+XA(1))/(RDH*(RDA-1.6))       &
             -((costheta*RDH-0.8D0)*(-XA(3)+XA(1)))          &
             /((RDA-1.6D0)**2*RDA)
     DRHODX2 =costheta*(XA(2)-XA(1))/(RDH*(RDA-1.6D0))
     DRHODX3 =-(costheta*RDH-0.8D0)*(XA(3)-XA(1))            &
             /((RDA-1.6D0)**2*RDA)

     DRHODY1 =costheta*(-YA(2)+YA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-YA(3)+YA(1)))          &
             /((RDA-1.6D0)**2*RDA)
     DRHODY2 =costheta*(YA(2)-YA(1))/(RDH*(RDA-1.6D0))
     DRHODY3 =-(costheta*RDH-0.8D0)*(YA(3)-YA(1))            &
             /((RDA-1.6D0)**2*RDA)

     DRHODZ1 =costheta*(-ZA(2)+ZA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-ZA(3)+ZA(1)))          &
             /((RDA-1.6D0)**2*RDA)
     DRHODZ2 =costheta*(ZA(2)-ZA(1))/(RDH*(RDA-1.6D0))
     DRHODZ3 =-(costheta*RDH-0.8D0)*(ZA(3)-ZA(1))            &
             /((RDA-1.6D0)**2*RDA)



!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA






!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E.
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE
!      D(THETA)/D(U)*D(U)/D(X1)


      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1
      DTHETADX2=DTDCOST*DCOSTDX2
      DTHETADX3=DTDCOST*DCOSTDX3

      DTHETADY1=DTDCOST*DCOSTDY1
      DTHETADY2=DTDCOST*DCOSTDY2
      DTHETADY3=DTDCOST*DCOSTDY3

      DTHETADZ1=DTDCOST*DCOSTDZ1
      DTHETADZ2=DTDCOST*DCOSTDZ2
      DTHETADZ3=DTDCOST*DCOSTDZ3

!    plus deriv of DthetaDx
      DRHODx1=DRHODx1+DCOSTDx1*RDH/(RDA-1.6d0)
      DRHODy1=DRHODy1+DCOSTDy1*RDH/(RDA-1.6d0)
      DRHODz1=DRHODz1+DCOSTDz1*RDH/(RDA-1.6d0)
      DRHODx2=DRHODx2+DCOSTDx2*RDH/(RDA-1.6d0)
      DRHODy2=DRHODy2+DCOSTDy2*RDH/(RDA-1.6d0)
      DRHODz2=DRHODz2+DCOSTDz2*RDH/(RDA-1.6d0)
      DRHODx3=DRHODx3+DCOSTDx3*RDH/(RDA-1.6d0)
      DRHODy3=DRHODy3+DCOSTDy3*RDH/(RDA-1.6d0)
      DRHODz3=DRHODz3+DCOSTDz3*RDH/(RDA-1.6d0)

!    new deriv of RDH and RHO
        dhrdrda=0.5d0*kmin*(dvar-dfin)**2*DV0DRDA
        dhrdrho=0.5d0*kmin*(dvar-dfin)**2*DV0Drho-kfin*(dvar-dfin)*2d0*dmin*(rho-0.5d0)
        dhrdxyz=kfin*(dvar-dfin)*sintheta/RDH

!      CALCULATE FORCES

      DX1 = (DV0DRHO+dhrdrho)*DRHODX1 + (DV0DRDA+dhrdrda)*DRDADX1   &
      + DHRDTHET*DTHETADX1 +dhrdxyz*(-XA(2)+XA(1))
     DX2 = (DV0DRHO+dhrdrho)*DRHODX2 + (DV0DRDA+dhrdrda)*DRDADX2    &
      + DHRDTHET*DTHETADX2 +dhrdxyz*(XA(2)-XA(1))
     DX3 = (DV0DRHO+dhrdrho)*DRHODX3 + (DV0DRDA+dhrdrda)*DRDADX3    &
      + DHRDTHET*DTHETADX3

     Dy1 = (DV0DRHO+dhrdrho)*DRHODy1 + (DV0DRDA+dhrdrda)*DRDADy1    &
      + DHRDTHET*DTHETADy1 +dhrdxyz*(-yA(2)+yA(1))
     Dy2 = (DV0DRHO+dhrdrho)*DRHODy2 + (DV0DRDA+dhrdrda)*DRDADy2    &
      + DHRDTHET*DTHETADy2 +dhrdxyz*(yA(2)-yA(1))
     Dy3 = (DV0DRHO+dhrdrho)*DRHODy3 + (DV0DRDA+dhrdrda)*DRDADy3    &
      + DHRDTHET*DTHETADy3

     Dz1 = (DV0DRHO+dhrdrho)*DRHODz1 + (DV0DRDA+dhrdrda)*DRDADz1    &
      + DHRDTHET*DTHETADz1 +dhrdxyz*(-zA(2)+zA(1))
     Dz2 = (DV0DRHO+dhrdrho)*DRHODz2 + (DV0DRDA+dhrdrda)*DRDADz2    &
      + DHRDTHET*DTHETADz2 +dhrdxyz*(zA(2)-zA(1))
     Dz3 = (DV0DRHO+dhrdrho)*DRHODz3 + (DV0DRDA+dhrdrda)*DRDADz3    &
      + DHRDTHET*DTHETADz3


      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> PT ENERGY: ', VTOT
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT> V0 + HR: ', V0,HR
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',DA,DX1,DY1,DZ1
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',HA,DX2,DY2,DZ2
        WRITE(OUTU,*) 'MSMMPT> PT FORCES ON',AA,DX3,DY3,DZ3
      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EMSPTNL


!      _________________________________________________________________________

!      LEGENDRE POLYNOMIALS
  SUBROUTINE LGNDMS(LNUM,X,P,DP)
!  SUBROUTINE TO GENERATE LEGENDRE POLYNOMIALS P_L(X)
!      FOR L = 0,1,...,LNUM WITH GIVEN X.

      real(chm_real) X, P, DP
      DIMENSION P(0:LNUM)
      DIMENSION DP(0:LNUM)
      INTEGER L, LNUM

      P(0) = 1.D0
      P(1) = X
      DO L=1, LNUM-1
        P(L+1) = ((2.D0*DBLE(L)+1.D0)*X*P(L)-DBLE(L)*P(L-1))/DBLE(L+1)
      ENDDO

!      RECURSIVE RELATION FOR DERIVATIVES

      DP(0) = 0.D0
      DP(1) = 1.D0


!      IF COS THETA = 1 THEN USE RESULTS DIRECTLY AND AVOID
!      DIVISION
      IF (X.EQ.1.D0) THEN
         DP(2) =  3.D0
         DP(3) =  6.D0
         DP(4) = 10.D0
         DP(5) = 15.D0
         DP(6) = 21.D0
         DP(7) = 28.D0
         DP(8) = 36.D0
         DP(9) = 45.D0
         DP(10)= 55.D0
         DP(11)= 66.D0
      ELSE

      DO L=2, LNUM
        DP(L) = DBLE(L)*(X*P(L)-P(L-1))/(X**2-1)
!        WRITE(22,*) L, P(L), P(L-1), X,(X**2-1), DP(L)
      ENDDO
      ENDIF

      RETURN
  END SUBROUTINE LGNDMS

  SUBROUTINE ALLOCFIRMS
    use number
    use psf

    INTEGER NHBNUM, NHBNUM3, NHBNUM4, NHBNUM7,I

    NPRMOHO=11
    NPRMNHN=11
    NPRMNHO=27
    NPRMLPE=114
    NPRMNLM=14
    NHBNUM=200
    NPRMMUL=24

    ALLOCATE(PRMNHN(NPRMNHN),       &
            PRMOHO(NPRMOHO),        &
            PRMNHO(NPRMNHO),        &
            PRMLPE(NPRMLPE),        &
            PRMNLM(NPRMNLM),        &
            PRMMUL(NPRMMUL),        &
            POTTYPE(NHBNUM),        &
            HBRDATOM(NHBNUM,3),     &
            XTMP(3)    &
            )


    RETURN
  END SUBROUTINE ALLOCFIRMS

  SUBROUTINE ALLOCSECMS(NHBNUM)

    use psf

    INTEGER NHBNUM, NHBNUM3, NHBNUM4, NHBNUM7,I

    NHBNUM3=NHBNUM*12
    NHBNUM4=NHBNUM*18
    NHBNUM7=NHBNUM*30

    ALLOCATE(BONDMMPT(NHBNUM,12,4),     &
            MMPTCHRG(NHBNUM,3),      &
            INITCHRG(NHBNUM,3),      &
            ANGLMMPT(NHBNUM,12,5),     &
            DIHEMMPT(NHBNUM4,6),     &
            IMPRMMPT(NHBNUM4,6),     &
            HBRDFLAG(NHBNUM,4),      &
            NONBMMPT(NHBNUM,30+NATOM*10,4),   &
            NONBNUM(NHBNUM),          &
            BONDNUM(NHBNUM),          &
            ANGLNUM(NHBNUM),           &
            WDDX(10),           &
            WDDY(10),           &
            WDDZ(10),           &
            WATOMM(10)          &
            )
    allocate(GOO(1000), &
             GOH(1000), &
             GHO(1000), &
             GHH(1000))
    DO I=1,1000
      GOO(I)=0.D0
      GOH(I)=0.D0
      GHO(I)=0.D0
      GHH(I)=0.D0
    ENDDO
    ALLOCATE(MMPTNBL(NATOM,NATOM))
    ALLOCATE(DISTMAT(NATOM,NATOM))
    ALLOCATE(NBLIST(NATOM,NATOM))

    RETURN
  END SUBROUTINE ALLOCSECMS

  SUBROUTINE FINDDA(DA,D1,D2,D3,D4,DNUM,X,Y,Z,NATOM2)

!##INCLUDE '~/charmm_fcm/impnon.fcm'
    use stream
    use string
    use code
    use psf
    use param
    use dimens_fcm

    INTEGER D1,D2,D3,D4,DNUM,NATOM2
    INTEGER,DIMENSION(PRM_DA_LENGTH) :: DA
    real(chm_real) X(NATOM2),Y(NATOM2),Z(NATOM2),DD(PRM_DA_LENGTH)
    real(chm_real) DXX,DYY,DZZ

    INTEGER I,J,ITMP,STMP
    INTEGER cnt1,cnt2

    cnt1=0
    cnt2=0

!c'FINDDA: initialization
    DO I=1,PRM_DA_LENGTH
      DA(I)=-1
      DD(I)=99.9D0
    ENDDO
    DA(1)=ABS(D1)
    DA(2)=D2

    DO I=1,NATOM2
      IF (ATYPE(I)(1:1).EQ.'O' .AND. I.NE.ABS(D1) .AND. I.NE.D2 .AND. &
      I.NE.D3 .AND. I.NE.D4) THEN
        IF (D1.GT.0) THEN
          DXX=X(I)-X(D1)
          DYY=Y(I)-Y(D1)
          DZZ=Z(I)-Z(D1)

          ! In a situation of PBC
          IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
            IF (DXX.GT.0.5D0*PXSIZE) THEN
              DXX=DXX-PXSIZE
  !write(outu,*) 'TEST-Warning1> ',DXX+PXSIZE
            ELSE IF (DXX.LE.-0.5D0*PXSIZE) THEN
              DXX=DXX+PXSIZE
  !write(outu,*) 'TEST-Warning2> ',DXX-PXSIZE
            ENDIF
            IF (DYY.GT.0.5D0*PYSIZE) THEN
              DYY=DYY-PYSIZE
            ELSE IF (DYY.LE.-0.5D0*PYSIZE) THEN
              DYY=DYY+PYSIZE
            ENDIF
            IF (DZZ.GT.0.5D0*PZSIZE) THEN
              DZZ=DZZ-PZSIZE
            ELSE IF (DZZ.LE.-0.5D0*PZSIZE) THEN
              DZZ=DZZ+PZSIZE
            ENDIF
          ENDIF
          DD(1)=SQRT(DXX*DXX+DYY*DYY+DZZ*DZZ)
        ELSE
          DD(1)=99.D0
        ENDIF

        DXX=X(I)-X(D2)
        DYY=Y(I)-Y(D2)
        DZZ=Z(I)-Z(D2)

        ! In a situation of PBC
        IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
          IF (DXX.GT.0.5D0*PXSIZE) THEN
            DXX=DXX-PXSIZE
          ELSE IF (DXX.LE.-0.5D0*PXSIZE) THEN
            DXX=DXX+PXSIZE
          ENDIF
          IF (DYY.GT.0.5D0*PYSIZE) THEN
            DYY=DYY-PYSIZE
          ELSE IF (DYY.LE.-0.5D0*PYSIZE) THEN
            DYY=DYY+PYSIZE
          ENDIF
          IF (DZZ.GT.0.5D0*PZSIZE) THEN
            DZZ=DZZ-PZSIZE
          ELSE IF (DZZ.LE.-0.5D0*PZSIZE) THEN
            DZZ=DZZ+PZSIZE
          ENDIF
        ENDIF
        DD(2)=SQRT(DXX*DXX+DYY*DYY+DZZ*DZZ)

        IF (DD(2)<DD(1)) DD(1)=DD(2)
        IF (DD(1)<PRM_INTERNAL_DA_DIST) THEN
          ITMP=I
          DO J=3,PRM_DA_LENGTH
            IF (DD(1)<DD(J)) THEN
              DD(2)=DD(J)
              DD(J)=DD(1)
              DD(1)=DD(2)
              STMP=DA(J)
              DA(J)=ITMP
              ITMP=STMP
            ENDIF
          ENDDO
        ENDIF
      ENDIF
    ENDDO

    DO I=1,PRM_DA_LENGTH
      IF (DA(I).GT.0) THEN
        DNUM=I

        IF (I.GE.PRM_DA_LENGTH) WRITE (OUTU,*) 'MSMMPT-FINDDA> Warning: &
        The number of acceptors is about to exceed.'
      ENDIF
    ENDDO

    IF (D1.LT.0) DA(1)=D1

    RETURN
  END SUBROUTINE FINDDA

  SUBROUTINE FINDHA(DA,HA,HA0,DNUM,HNUM)
!##INCLUDE '~/charmm_fcm/impnon.fcm'
    use stream
    use string
    use coord
    use code
    use psf
    use param
    use dimens_fcm

    INTEGER DNUM
    INTEGER,DIMENSION(PRM_DA_LENGTH) :: DA
    INTEGER,DIMENSION(PRM_HA_LENGTH) :: HA
    INTEGER I,J,K,HA0,ITMP,JJ,HNUM

    K=0

    IF (DA(1).LT.0) THEN
      K=K+1
      HA(K)=HA0
    ENDIF

    JJ=1
    DO I=1,NBOND
      DO J=1,2 !adopt only D and A, or do J=1,DNUM
        IF (ATYPE(JB(I))(1:1).EQ.'H' .AND. DA(J).EQ.IB(I)) THEN
          K=K+1
          HA(K)=JB(I)
          IF (J.EQ.2) HA(K)=0-HA(K)  !LABEL ON ACCEPTOR BONDED ATOM
        ELSE IF (ATYPE(IB(I))(1:1).EQ.'H' .AND. DA(J).EQ.JB(I)) THEN
          K=K+1
          HA(K)=IB(I)
          IF (J.EQ.2) HA(K)=0-HA(K)  !LABEL ON ACCEPTOR BONDED ATOM
        ENDIF

        IF (K.EQ.PRM_HA_LENGTH) WRITE(OUTU,*) 'MSMMPT FINDHA> &
        Warning: MHANUM comes into exceeding'
      ENDDO
    ENDDO

    HNUM=K !store number of selected hydrogen atoms'

    DO I=1,K
      IF (HA(I).EQ.HA0) THEN !move H to the 1st position
        ITMP=HA(I)
        HA(I)=HA(1)
        HA(1)=ITMP
        GOTO 4890
      ENDIF
    ENDDO

    CALL WrnDie (-3,'<MISCOM>','MSMMPT-MPT> ERROR: former DH-A not found')

4890  continue

    RETURN
  END SUBROUTINE FINDHA

!c'IFMOTIF: Check if DH-A is a proper motif
!           1) DA<4.0A and 2) DA^2>|DH^2+AH^2|*gamma
!           3) If new Donor is A, new Hydrogen can not be H_x-(D)
!              other than H; if new Donor is D, new Hydrogen can not be
!              H_y-(A) other than H.
  INTEGER FUNCTION IFMOTIF(HBI,I,J,K,X,Y,Z,NATOM2)

    use stream
    use string
    use code
    use psf
    use param
    use dimens_fcm

    INTEGER HBI,I,J,K,NATOM2,L
    real(chm_real) X(NATOM2),Y(NATOM2),Z(NATOM2)
    real(chm_real) DXIJ,DYIJ,DZIJ,DXIJ2,DYIJ2,DZIJ2,COSALPHA
    REAL(CHM_REAL) RDA,RDH,RAH
    logical :: file_exists
    integer NMOTIF,A1,A2,A3,IDX

    INQUIRE(FILE="extra.mtf", EXIST=file_exists)

    if (file_exists .AND. 1.eq.1) then
      open (unit = 32, file = "extra.mtf")
      READ (32,*) NMOTIF

      DO IDX = 1,NMOTIF
        READ (32,*) A1,A2,A3
        IF (A1.EQ.I .AND. A2.EQ.J .AND. A3.EQ.K) THEN
          IFMOTIF=1
          CLOSE(32)
          RETURN
        ENDIF
      ENDDO

      CLOSE(32)
      IFMOTIF=0
      RETURN
    ENDIF

    IF (I.EQ.HBRDATOM(HBI,3) .AND. J.NE.HBRDATOM(HBI,2)) THEN
      DO L=1,NBOND
        IF ((IB(L).EQ.J .AND. JB(L).EQ.HBRDATOM(HBI,1)) .OR. &
        (JB(L).EQ.J .AND. IB(L).EQ.HBRDATOM(HBI,1))) THEN
          IFMOTIF=0
          IF (PRNLEV.GE.9) WRITE(OUTU,*) 'MSMMPT IFMOTIF> D-H-A', &
          I,J,K, 'RETURN-01'
          RETURN
        ENDIF
      ENDDO
    ELSE IF (I.EQ.HBRDATOM(HBI,1) .AND. J.NE.HBRDATOM(HBI,2)) THEN
      DO L=1,NBOND
        IF ((IB(L).EQ.J .AND. JB(L).EQ.HBRDATOM(HBI,3)) .OR. &
        (JB(L).EQ.J .AND. IB(L).EQ.HBRDATOM(HBI,3))) THEN
          IFMOTIF=0
          IF (PRNLEV.GE.9) WRITE(OUTU,*) 'MSMMPT IFMOTIF> D-H-A', &
          I,J,K, 'RETURN-02'
          RETURN
        ENDIF
      ENDDO
    ENDIF


    DXIJ=X(I)-X(K)
    DYIJ=Y(I)-Y(K)
    DZIJ=Z(I)-Z(K)
    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXIJ.GT.0.5D0*PXSIZE) THEN
        DXIJ=DXIJ-PXSIZE
      ELSE IF (DXIJ.LE.-0.5D0*PXSIZE) THEN
        DXIJ=DXIJ+PXSIZE
      ENDIF
      IF (DYIJ.GT.0.5D0*PYSIZE) THEN
        DYIJ=DYIJ-PYSIZE
      ELSE IF (DYIJ.LE.-0.5D0*PYSIZE) THEN
        DYIJ=DYIJ+PYSIZE
      ENDIF
      IF (DZIJ.GT.0.5D0*PZSIZE) THEN
        DZIJ=DZIJ-PZSIZE
      ELSE IF (DZIJ.LE.-0.5D0*PZSIZE) THEN
        DZIJ=DZIJ+PZSIZE
      ENDIF
    ENDIF

    RDA=DXIJ*DXIJ+DYIJ*DYIJ+DZIJ*DZIJ

    IF (RDA.GE.PRM_INTERNAL_DA_DIST*PRM_INTERNAL_DA_DIST .OR. RDA.LE.3.6D0) THEN
      IFMOTIF=0
      IF (PRNLEV.GE.9) WRITE(OUTU,*) 'MSMMPT IFMOTIF> D-H-A', &
      I,J,K, RDA,'RETURN-03'
      RETURN
    ENDIF

    DXIJ=X(I)-X(J)
    DYIJ=Y(I)-Y(J)
    DZIJ=Z(I)-Z(J)
    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXIJ.GT.0.5D0*PXSIZE) THEN
        DXIJ=DXIJ-PXSIZE
      ELSE IF (DXIJ.LE.-0.5D0*PXSIZE) THEN
        DXIJ=DXIJ+PXSIZE
      ENDIF
      IF (DYIJ.GT.0.5D0*PYSIZE) THEN
        DYIJ=DYIJ-PYSIZE
      ELSE IF (DYIJ.LE.-0.5D0*PYSIZE) THEN
        DYIJ=DYIJ+PYSIZE
      ENDIF
      IF (DZIJ.GT.0.5D0*PZSIZE) THEN
        DZIJ=DZIJ-PZSIZE
      ELSE IF (DZIJ.LE.-0.5D0*PZSIZE) THEN
        DZIJ=DZIJ+PZSIZE
      ENDIF
    ENDIF
    RDH=DXIJ*DXIJ+DYIJ*DYIJ+DZIJ*DZIJ

    DXIJ2=X(K)-X(J)
    DYIJ2=Y(K)-Y(J)
    DZIJ2=Z(K)-Z(J)
    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXIJ2.GT.0.5D0*PXSIZE) THEN
        DXIJ2=DXIJ2-PXSIZE
      ELSE IF (DXIJ2.LE.-0.5D0*PXSIZE) THEN
        DXIJ2=DXIJ2+PXSIZE
      ENDIF
      IF (DYIJ2.GT.0.5D0*PYSIZE) THEN
        DYIJ2=DYIJ2-PYSIZE
      ELSE IF (DYIJ2.LE.-0.5D0*PYSIZE) THEN
        DYIJ2=DYIJ2+PYSIZE
      ENDIF
      IF (DZIJ2.GT.0.5D0*PZSIZE) THEN
        DZIJ2=DZIJ2-PZSIZE
      ELSE IF (DZIJ2.LE.-0.5D0*PZSIZE) THEN
        DZIJ2=DZIJ2+PZSIZE
      ENDIF
    ENDIF
    RAH=DXIJ2*DXIJ2+DYIJ2*DYIJ2+DZIJ2*DZIJ2

    COSALPHA=(DXIJ*DXIJ2+DYIJ*DYIJ2+DZIJ*DZIJ2)/SQRT(RDH)/SQRT(RAH)
    IF (COSALPHA.GT.0.0d0) THEN !>COS_ALPHA
      IFMOTIF=0
      IF (PRNLEV.GE.9) WRITE(OUTU,*) 'MSMMPT IFMOTIF> D-H-A', &
      I,J,K, (DXIJ*DXIJ2+DYIJ*DYIJ2+DZIJ*DZIJ2),SQRT(RDH),SQRT(RAH), &
      COSALPHA,'RETURN-04'
      RETURN
    ENDIF
    IF (PRNLEV.GE.9) WRITE(OUTU,*) 'MSMMPT IFMOTIF> D-H-A', &
    I,J,K, 'GAMMA=',RDA/(RDH+RAH)

    IFMOTIF=1

    RETURN
  END FUNCTION IFMOTIF

  REAL(chm_real) FUNCTION MDIST(I,J)

! ... routine flags whether a nonbond interaction is under cutoff

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    real(chm_real) RX,RY,RZ,S2,S1
!    real(chm_real) :: MDIST
    integer I,J


    IF (DISTMAT(I,J).GT.0.D0) THEN
      MDIST=DISTMAT(I,J)
      return
    ELSE
      RX=X(I)-X(J)
      RY=Y(I)-Y(J)
      RZ=Z(I)-Z(J)

      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        IF (RX.GT.0.5D0*PXSIZE) THEN
          RX=RX-PXSIZE
        ELSE IF (RX.LE.-0.5D0*PXSIZE) THEN
          RX=RX+PXSIZE
        ENDIF
        IF (RY.GT.0.5D0*PYSIZE) THEN
          RY=RY-PYSIZE
        ELSE IF (RY.LE.-0.5D0*PYSIZE) THEN
          RY=RY+PYSIZE
        ENDIF
        IF (RZ.GT.0.5D0*PZSIZE) THEN
          RZ=RZ-PZSIZE
        ELSE IF (RZ.LE.-0.5D0*PZSIZE) THEN
          RZ=RZ+PZSIZE
        ENDIF
      ENDIF

      S2=RX*RX+RY*RY+RZ*RZ
      S1=SQRT(S2)

      DISTMAT(I,J)=S1
      DISTMAT(J,I)=S1

      MDIST=S1
      return
    ENDIF

    RETURN
  END FUNCTION MDIST


  SUBROUTINE IMGOFFSET(I,J,RX,RY,RZ)

! ... routine flags whether a nonbond interaction is under cutoff

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    real(chm_real) RX,RY,RZ,S2,S1
!    real(chm_real) :: MDIST
    integer I,J


    RX=X(J)-X(I)
    RY=Y(J)-Y(I)
    RZ=Z(J)-Z(I)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (RX.GT.0.5D0*PXSIZE) THEN
        RX=RX-PXSIZE
      ELSE IF (RX.LE.-0.5D0*PXSIZE) THEN
        RX=RX+PXSIZE
      ENDIF
      IF (RY.GT.0.5D0*PYSIZE) THEN
        RY=RY-PYSIZE
      ELSE IF (RY.LE.-0.5D0*PYSIZE) THEN
        RY=RY+PYSIZE
      ENDIF
      IF (RZ.GT.0.5D0*PZSIZE) THEN
        RZ=RZ-PZSIZE
      ELSE IF (RZ.LE.-0.5D0*PZSIZE) THEN
        RZ=RZ+PZSIZE
      ENDIF
    ENDIF

    S2=RX*RX+RY*RY+RZ*RZ
    S1=SQRT(S2)

    DISTMAT(I,J)=S1
    DISTMAT(J,I)=S1

  END SUBROUTINE IMGOFFSET

!c'emmpt: this subroutine is to check if a diatomic pair has a image nonbond
!interaction

  INTEGER FUNCTION IFIMAGE(I,J)

! ... routine flags whether a nonbond interaction is under cutoff

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    real(chm_real) RX,RY,RZ,S1,S2
    INTEGER I,J,FLAG

    IF (DISTMAT(I,J).GT.0.D0) THEN
      S1=DISTMAT(I,J)
    ELSE
      RX=X(I)-X(J)
      RY=Y(I)-Y(J)
      RZ=Z(I)-Z(J)

      ! In a situation of PBC
      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        IF (RX.GT.0.5D0*PXSIZE) THEN
          RX=RX-PXSIZE
        ELSE IF (RX.LE.-0.5D0*PXSIZE) THEN
          RX=RX+PXSIZE
        ENDIF
        IF (RY.GT.0.5D0*PYSIZE) THEN
          RY=RY-PYSIZE
        ELSE IF (RY.LE.-0.5D0*PYSIZE) THEN
          RY=RY+PYSIZE
        ENDIF
        IF (RZ.GT.0.5D0*PZSIZE) THEN
          RZ=RZ-PZSIZE
        ELSE IF (RZ.LE.-0.5D0*PZSIZE) THEN
          RZ=RZ+PZSIZE
        ENDIF
      ENDIF

      S2=RX*RX+RY*RY+RZ*RZ
      S1=SQRT(S2)

      DISTMAT(I,J)=S1
      DISTMAT(J,I)=S1
    ENDIF

    IF (S1.GE.PRM_NONB_CUTOFF) THEN
      IFIMAGE=0
      RETURN
    ENDIF
!    ENDIF

    IFIMAGE=1
    RETURN

  END FUNCTION IFIMAGE

  SUBROUTINE NBONDMOD(ATOMD,ATOMH,ATOMA,NBNUM,H)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use inbnd

    INTEGER H,J,ITMP,JSTART,JEND,NBNUM,ATOMD,ATOMH,ATOMA
    INTEGER NBPREV,L,I,NB,HB,NONB2H(NATOM)
    INTEGER IMSEED(NATOM),IMFLAG(NATOM),IMNUM,A1,A2,A3,NATOM2

    NATOM2=NATOM

    IMNUM=0

    IMNUM=IMNUM+1
    IMSEED(IMNUM)=ATOMD
    IMFLAG(IMNUM)=21

    IMNUM=IMNUM+1
    IMSEED(IMNUM)=ATOMH
    IMFLAG(IMNUM)=22

    IMNUM=IMNUM+1
    IMSEED(IMNUM)=ATOMA
    IMFLAG(IMNUM)=23

    DO I=1,BONDNUM(H)
      A1=BONDMMPT(H,I,2)
      A2=BONDMMPT(H,I,3)

      IF (A1.EQ.IMSEED(1) .OR. A2.EQ.IMSEED(1)) THEN
        IF (A2.EQ.IMSEED(2) .OR. A1.EQ.IMSEED(2))  THEN
          CONTINUE
        ELSE
          IMNUM=IMNUM+1
          IMSEED(IMNUM)=A1+A2-IMSEED(1)
          IMFLAG(IMNUM)=24
        ENDIF
      ELSE IF (A1.EQ.IMSEED(3) .OR. A2.EQ.IMSEED(3)) THEN
        IMNUM=IMNUM+1
        IMSEED(IMNUM)=A1+A2-IMSEED(3)
        IMFLAG(IMNUM)=25
      ENDIF
    ENDDO

    !check if any nonbond not in JNB() list
    !initialization (partially)
    DO I=1,IMNUM
      DO J=1,NATOM2
        NBLIST(IMSEED(I),J)=0
        NBLIST(J,IMSEED(I))=0
      ENDDO
      NBLIST(IMSEED(I),IMSEED(I))=-1
    ENDDO

    DO I=1,BONDNUM(H)
      A1=BONDMMPT(H,I,2)
      A2=BONDMMPT(H,I,3)
      NBLIST(A1,A2)=-1
      NBLIST(A2,A1)=-1
    ENDDO

    DO I=1,ANGLNUM(H)
      A1=ANGLMMPT(H,I,2)
      A2=ANGLMMPT(H,I,4)
      NBLIST(A1,A2)=-1
      NBLIST(A2,A1)=-1
    ENDDO

    !NOT IMPLEMENTED FOR DIHEDRAL

    DO I=1,NONBNUM(H)
      A1=NONBMMPT(H,I,2)
      A2=NONBMMPT(H,I,3)
      NBLIST(A1,A2)=-1
      NBLIST(A2,A1)=-1
    ENDDO

    DO I=1,IMNUM
      DO J=1,NATOM2

        IF (NBLIST(IMSEED(I),J).EQ.0 .AND. IFIMAGE(IMSEED(I),J).EQ.1) THEN
          NBNUM=NBNUM+1
          NONBMMPT(H,NBNUM,1)=NBNUM
          NONBMMPT(H,NBNUM,2)=IMSEED(I)
          NONBMMPT(H,NBNUM,3)=J
          NONBMMPT(H,NBNUM,4)=IMFLAG(I)
          NBLIST(IMSEED(I),J)=1
          NBLIST(J,IMSEED(I))=1


        ENDIF
      ENDDO
    ENDDO

  END SUBROUTINE NBONDMOD

!c'emmpt: this subroutine is to calculate changes of energy components
!
  SUBROUTINE EMMPT2(EU,EW,X,Y,Z,DX,DY,DZ,WDX,WDY,WDZ,NATOMX,HBI,EMODE)
! main subroutine to add MMPT energy and derivatives
! do not 'use coord'
!

    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    use code
    use bases_fcm
    use chm_types

    real(chm_real) EU,ETMP,FCST,EQUIDEG
    real(chm_real) BNDHYD,BEQHYD,AGLHYD,AEQHYD
    real(chm_real) BNDWAT,BEQWAT,AGLWAT,AEQWAT
    real(chm_real) FC1,FC2,EQUI1,EQUI2,FCDIH,DDSWF
    real(chm_real) CCA,CCB,CCC,CCP_TMP,CGO,CGH,CGE
    real(chm_real) EPSI_O,RMIN_O,EPSI_H,RMIN_H
    real(chm_real) EPSI_H_LOCAL,RMIN_H_LOCAL
    real(chm_real) PSI1,PSI2,PSI3,PSI4,PHI1,PHI2,&
    DELTA_C,EDIH_CORR,EDIH_FINI,SWFTMP
    real(chm_real) DCGE,DCGEDX,DCGEDY,DCGEDZ,RDA,RXDA,RYDA,RZDA
    INTEGER NATOMX,EMODE,HBI,ATOM1,ATOM2,ATOM3,QFLAG(NATOMX)
    INTEGER DIHATOM1,DIHATOM2,DIHATOM3,DIHATOM4
    INTEGER VDW_FLAG,EXNBFLAG_CG,EXNBFLAG_VDW,CCP
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!EWEIGHT-00
    real(chm_real) WDX(NATOMX),WDY(NATOMX),WDZ(NATOMX)
    real(chm_real) EW,EWTEMP
    real(chm_real) QXTMP,QYTMP,QZTMP
    real(chm_real) RMIN_ATOMJ,EPS_ATOMJ,RMIN_ATOMI,EPS_ATOMI
    INTEGER WNATMX,WJ,ATMP,DATM(PRM_DA_LENGTH),DNUM
    logical MMPTPFLAG2

    !type(nonbondDataStructure) BNBND

!     SWITCH STUFF
    real(chm_real)  SWF, &
                    DSWFRNN,DSWFRNH,DRNHX,DRNHY,DRNHZ, &
                    DRNNX,DRNNY,DRNNZ



!     -----------------------------------------------------------------


    real(chm_real) EB,EA, EP,EELMMPT,ENBMMPT,ENBMMPT3,EVDW1MMPT,EVDW2MMPT
    real(chm_real) EELTOT,EVDWTOT
    real(chm_real) ERM,DXBRM,DYBRM,DZBRM
!      real(chm_real) SUM

    INTEGER I,J,K,HB,NBNUM_LOCAL

    real(chm_real) EIMPROP

    real(chm_real) DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM

    real(chm_real) DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1
    real(chm_real) DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2
    real(chm_real) DXIRM3,DYIRM3,DZIRM3,DXJRM3,DYJRM3,DZJRM3


    real(chm_real) ED,DXDI,DXDJ,DXDK,DXDL,  &
              DYDI,DYDJ,DYDK,DYDL,  &
              DZDI,DZDJ,DZDK,DZDL

    real*8 EI,DXII,DXIJ,DXIK,DXIL,  &
              DYII,DYIJ,DYIK,DYIL,  &
              DZII,DZIJ,DZIK,DZIL

    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK
    real(chm_real) DX1,DX2,DY1,DY2,DZ1,DZ2

    real(chm_real) RX,RY,RZ,S1,S2,RB,BETA,BSWF,DDBSWF
    real(chm_real) BETA2,BETA3,AMPLIFIER,AMPREAL

    real(chm_real) XCOM,YCOM,ZCOM,DIPX,DIPY,DIPZ,DIPM

    real*8 timep1,timep2

    if (PRNLEV>=10) then
      WRITE (OUTU,*) '---------------------X,Y,Z---------------------'
      DO I=1,NATOMX
        WRITE(OUTU,*) '|',X(I),Y(I),Z(I),'|'
      ENDDO
      WRITE (OUTU,*) '-----------------------------------------------'
    endif

    if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> MODE:',EMODE

    IF(.not. (EMODE.GE.1 .AND. EMODE.LE.3)) THEN
      CALL WrnDie (-3,'<MISCOM>','MSMMPT-MPT> Error: EMSMMPT-MODE should &
      be in [1-3]')
    ENDIF

    IF(EMODE.EQ.1 .AND. WMODE.GE.1) THEN
      DO I=1,10
        WATOMM(I)=0
      ENDDO
    ENDIF

!    EU = 0.d0
    EELTOT=0.D0
    EVDWTOT=0.D0
!    EW = 0.D0

    DXIRM=0.D0
    DYIRM=0.D0
    DZIRM=0.d0

    DXJRM=0.D0
    DYJRM=0.D0
    DZJRM=0.d0

    DXIRM1=0.D0
    DYIRM1=0.D0
    DZIRM1=0.d0

    DXJRM1=0.D0
    DYJRM1=0.D0
    DZJRM1=0.d0

    DXIRM2=0.D0
    DYIRM2=0.D0
    DZIRM2=0.d0

    DXJRM2=0.D0
    DYJRM2=0.D0
    DZJRM2=0.d0


!c'emmpt: specify switching function of D-H...A
    CALL SWITCHMSPT(HBRDATOM(HBI,1),HBRDATOM(HBI,2),  &
         HBRDATOM(HBI,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
         DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)

    IF((POTTYPE(HBI).EQ.'LPE').OR.(POTTYPE(HBI).EQ.'SSM').OR. &
    (POTTYPE(HBI).EQ.'SDM').OR.(POTTYPE(HBI).EQ.'NLM')) THEN
      FC2=PRMMUL(3)
      EQUI2=PRMMUL(4)/180.d0*PI
      FCDIH=PRMMUL(2)

      IF (PRM_INTERNAL_CCP.GE.0) THEN
        CCA=0.0D0
        CCB=LOG(1.D0-PRMMUL(5))
        CCC=0.0D0
        CCP=PRM_INTERNAL_CCP
      ELSE
        CCA=PRM_INTERNAL_CCA
        CCB=PRM_INTERNAL_CCB
        CCC=PRM_INTERNAL_CCC
        CCP=PRM_INTERNAL_CCP
      ENDIF

      BNDHYD=PRMMUL(6) !parameters for H3O+
      BEQHYD=PRMMUL(7)
      AGLHYD=PRMMUL(8)
      AEQHYD=PRMMUL(9)/180.d0*PI
      BNDWAT=PRMMUL(10) !parameters for H2O
      BEQWAT=PRMMUL(11)
      AGLWAT=PRMMUL(12)
      AEQWAT=PRMMUL(13)/180.d0*PI
      EPSI_O=PRMMUL(14)
      RMIN_O=PRMMUL(15)
      EPSI_H=PRMMUL(16)
      RMIN_H=PRMMUL(17)
    ELSE
      CALL WrnDie (-3,'<MISCOM>','MSMMPT-MPT> ERROR: Only available with &
      SSM/SDM, LPE and NLM')
    ENDIF

    !c'emmpt: put charges in variables
    CGO=CG(HBRDATOM(HBI,1))
    CGH=CG(HBRDATOM(HBI,2))
    if (PRNLEV>=7 .and. EMODE.eq.3) &
    write (OUTU,*) 'MSMMPT-EMMPT2> Original charge [D-H-A]:', &
    CG(HBRDATOM(HBI,1)),CG(HBRDATOM(HBI,2)),CG(HBRDATOM(HBI,3))

    !c'emmpt: if MODE 2, then SWF=1,DSWFRNN=0,DSWFRNH=0
    IF (EMODE.EQ.2) THEN
      SWF=1.D0
      DSWFRNN=0.D0
      DSWFRNH=0.D0
      FC2=0.D0
      FCDIH=0.D0
      CCA=0.D0
      CCB=0.D0
      CCC=0.D0
      CCP=0

      BNDHYD=0.0D0
      BEQHYD=0.0D0
      AGLHYD=0.0D0
      AEQHYD=0.0D0
      BNDWAT=0.0D0
      BEQWAT=0.0D0
      AGLWAT=0.0D0
      AEQWAT=0.0D0
      EPSI_O=0.0D0
      RMIN_O=-999.D0
      EPSI_H=0.0D0
      RMIN_H=-999.D0
    ENDIF

    !c'emmpt: define if there is extra modifications for nonbond interactions
    IF (EMODE.EQ.3) THEN

      MMPTPFLAG2=.false.

    ! hwm: to fix MOD() problem when MMPTCYC=0
    IF (PRNLEV >= 6 ) THEN
       MMPTPFLAG2=.TRUE.
    ELSE IF (MMPTCYC.GT.0) THEN
       IF (MMPTSFLAG .AND. MOD(MMPTSTP,MMPTCYC).EQ.0 .AND. &
            (MMPTSTP.GT.LASTSTEP .OR. (.not. DYNAMQ))) THEN
          MMPTPFLAG2=.TRUE.
       ENDIF
    ELSE
       CONTINUE
    ENDIF
!      IF (PRNLEV >= 6 .OR. (MMPTSFLAG .AND. MMPTCYC.GT.0 .AND. &
!      MOD(MMPTSTP,MMPTCYC).EQ.0 .AND. (MMPTSTP.GT.LASTSTEP .OR. (.not. DYNAMQ)))) THEN
!        MMPTPFLAG2=.TRUE.
!      ENDIF

      IF (CCA.EQ.0.D0 .AND. CCB.EQ.0.D0 .AND. CCC.EQ.0.D0 ) THEN
        EXNBFLAG_CG=0
      ELSE
        EXNBFLAG_CG=1
      ENDIF

      IF (RMIN_O.GT.-0.0000001D0 .OR. RMIN_H.GT.-0.0000001D0) THEN
        EXNBFLAG_VDW=1
      ELSE
        EXNBFLAG_VDW=0
      ENDIF
    ENDIF

    if (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> SWF=',SWF
    IF (EMODE.EQ.3 .and. PRNLEV.GE.7) WRITE(OUTU,*) 'MSMMPT-MOTIF> D-H-A=', &
    HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3)

!c'emmpt: if DH-A atoms are restricted, skip E_PT or E_AH
    IF (IMOVE(HBRDATOM(HBI,1)).GT.0 .AND. &
    IMOVE(HBRDATOM(HBI,2)).GT.0 .AND. &
    IMOVE(HBRDATOM(HBI,3)).GT.0)  GOTO 5420
    CALL FINDDA(DATM,HBRDATOM(HBI,1),HBRDATOM(HBI,3),-1,-1,DNUM,X,Y,Z,NATOMX)

!c'emmpt: add E_PT at mode 1 or E_AH at MODE 2
    IF (EMODE.EQ.1) THEN
      IF(POTTYPE(HBI) .EQ. 'SDM') CALL  &
        EMSPTNHN(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3),EU)
      IF(POTTYPE(HBI) .EQ. 'SSM') THEN
        CALL EMSPTOHO(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3), &
        EU,X,Y,Z,DX,DY,DZ,NATOMX)

        CALL EPSEUDOMMPT(HBRDATOM(HBI,2),HBRDATOM(HBI,1), &
        HBRDATOM(HBI,3),EA,PRMOHO(10),0.D0,0,DXAI,DYAI,DZAI, &
        DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

        EU=EU-EA*SWF
        ETMP=-EA*SWF

        DX1=DSWFRNH*DRNNX*EA
        DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
        DY1=DSWFRNH*DRNNY*EA
        DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
        DZ1=DSWFRNH*DRNNZ*EA
        DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

        DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-(DXAI*SWF)
        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DXAJ*SWF)
        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-(DXAK*SWF)
        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-DX1-DX2
        DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

        DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-(DYAI*SWF)
        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DYAJ*SWF)
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-(DYAK*SWF)
        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-DY1-DY2
        DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

        DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-(DZAI*SWF)
        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZAJ*SWF)
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-(DZAK*SWF)
        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-DZ1-DZ2
        DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

        CALL EPSEUDOMMPT(HBRDATOM(HBI,2),HBRDATOM(HBI,3), &
        HBRDATOM(HBI,1),EA,PRMOHO(10),0.D0,0,DXAI,DYAI,DZAI,   &
        DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

        EU=EU+EA*SWF
        ETMP=ETMP+EA*SWF

        IF(PRNLEV.GT.6) THEN
          WRITE(OUTU,*) 'MSMMPT-MPT> Energy correction on SSM model', &
          ETMP
        ENDIF

        DX1=DSWFRNH*DRNNX*EA
        DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
        DY1=DSWFRNH*DRNNY*EA
        DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
        DZ1=DSWFRNH*DRNNZ*EA
        DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

        DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+(DXAI*SWF)
        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+(DXAJ*SWF)
        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DXAK*SWF)
        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX1+DX2
        DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

        DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+(DYAI*SWF)
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+(DYAJ*SWF)
        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DYAK*SWF)
        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY1+DY2
        DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

        DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+(DZAI*SWF)
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+(DZAJ*SWF)
        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZAK*SWF)
        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ1+DZ2
        DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
      ENDIF
      IF(POTTYPE(HBI) .EQ. 'ASM') CALL  &
        EMSPTNHO(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3),EU)
      IF(POTTYPE(HBI) .EQ. 'LPE') THEN
!        IF (HBRDATOM(HBI,1).LT.HBRDATOM(HBI,3)) THEN
           CALL EMSPTOHOLE(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3), &
           EU,X,Y,Z,DX,DY,DZ,NATOMX)
!        ELSE
!          CALL EMSPTOHOLE(HBRDATOM(HBI,3),HBRDATOM(HBI,2),HBRDATOM(HBI,1), &
!          EU,X,Y,Z,DX,DY,DZ,NATOMX)
!        ENDIF
      ENDIF
      IF(POTTYPE(HBI) .EQ. 'NLM') CALL  &
        EMSPTNL(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3), &
        EU,X,Y,Z,DX,DY,DZ,NATOMX)

      IF (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> EMSPT=',EU

      !EWEIGHT-01
      IF (WMODE.GE.1) THEN
        EW=EU
        DO I=1,NATOMX
          WDX(I)=DX(I)
          WDY(I)=DY(I)
          WDZ(I)=DZ(I)
        ENDDO
        IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EMSPT=',EW
      ENDIF

    ELSE IF (EMODE.EQ.2) THEN
      !c'emmpt: only adapted to OHO/NHN
      !!needed to be implemented for OHN
      CALL EBONDMSPT(BONDMMPT(HBI,1,1),HBRDATOM(HBI,2), &
      HBRDATOM(HBI,3),ERM,DXBRM,DYBRM,DZBRM)

      EU=EU+ERM

      DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DXBRM
      DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DYBRM
      DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZBRM

      DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DXBRM
      DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DYBRM
      DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZBRM

      if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_HA=',EU

    ENDIF

5420   CONTINUE

    IF (EMODE.EQ.1) THEN
      !c'emmpt: pseudo potential, ONLY IN MODE 1
      EQUIDEG=135.D0/180.d0*PI

      CALL EPSEUDOMMPT(HBRDATOM(HBI,1),HBRDATOM(HBI,2), &
      HBRDATOM(HBI,3),EA,PRM_INTERNAL_FCONST_DHA_ANGLE,EQUIDEG,-2,DXAI,DYAI,DZAI,   &
      DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

      IF (EA.GT.0.D0) THEN
        EU=EU+EA

        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DXAI
        DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DXAJ
        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DXAK

        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DYAI
        DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DYAJ
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DYAK

        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZAI
        DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZAJ
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZAK
      ENDIF

      if (PRNLEV>=7 .AND. EA.GT.0.D0) WRITE (OUTU,*) &
      'MSMMPT-EMMPT2> E_DHA_corr=',EA

      !EWEIGHT-02
      IF (WMODE.GE.1 .AND. EA.GT.0.D0) THEN
        EW=EW+EA
        WDX(HBRDATOM(HBI,1))=WDX(HBRDATOM(HBI,1))+DXAI
        WDX(HBRDATOM(HBI,2))=WDX(HBRDATOM(HBI,2))+DXAJ
        WDX(HBRDATOM(HBI,3))=WDX(HBRDATOM(HBI,3))+DXAK

        WDY(HBRDATOM(HBI,1))=WDY(HBRDATOM(HBI,1))+DYAI
        WDY(HBRDATOM(HBI,2))=WDY(HBRDATOM(HBI,2))+DYAJ
        WDY(HBRDATOM(HBI,3))=WDY(HBRDATOM(HBI,3))+DYAK

        WDZ(HBRDATOM(HBI,1))=WDZ(HBRDATOM(HBI,1))+DZAI
        WDZ(HBRDATOM(HBI,2))=WDZ(HBRDATOM(HBI,2))+DZAJ
        WDZ(HBRDATOM(HBI,3))=WDZ(HBRDATOM(HBI,3))+DZAK

        IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_ALPHA=',EA
      ENDIF
    ENDIF
!c'emmpt: modify bond energy
      IF (EMODE.NE.3) THEN
      DO I=1,BONDNUM(HBI)
        ETMP=EU

        IF (BEQHYD.GT.0.D0 .AND. BEQWAT.GT.0.D0) THEN
          IF (I.EQ.1 .OR. EMODE.EQ.1) THEN
            CALL EBONDMSPT(BONDMMPT(HBI,I,1),BONDMMPT(HBI,I,2), &
            BONDMMPT(HBI,I,3),ERM,DXBRM,DYBRM,DZBRM)

            EU = EU - ERM

            DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))-DXBRM
            DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))-DYBRM
            DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))-DZBRM

            DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))+DXBRM
            DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))+DYBRM
            DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))+DZBRM

            IF (ABS(BONDMMPT(HBI,I,4)).EQ.2) THEN

              !EWEIGHT-03
              IF (WMODE.GE.2 .AND. EMODE.EQ.1) THEN
                EWTEMP=EU
                WNATMX=3
                WATOMM(1)=HBRDATOM(HBI,1)
                WATOMM(2)=HBRDATOM(HBI,2)
                WATOMM(3)=HBRDATOM(HBI,3)
                IF (BONDMMPT(HBI,I,2).NE.WATOMM(1) .AND. BONDMMPT(HBI,I,2).NE.WATOMM(2) &
                .AND. BONDMMPT(HBI,I,2).NE.WATOMM(3)) THEN
                  WNATMX=WNATMX+1
                  WATOMM(WNATMX)=BONDMMPT(HBI,I,2)
                ENDIF
                IF (BONDMMPT(HBI,I,3).NE.WATOMM(1) .AND. BONDMMPT(HBI,I,3).NE.WATOMM(2) &
                .AND. BONDMMPT(HBI,I,3).NE.WATOMM(3) .AND. &
                BONDMMPT(HBI,I,3).NE.BONDMMPT(HBI,I,2)) THEN
                  WNATMX=WNATMX+1
                  WATOMM(WNATMX)=BONDMMPT(HBI,I,3)
                ENDIF

                DO WJ=1,WNATMX
                  ATMP=WATOMM(WJ)
                  WDDX(WJ)=DX(ATMP)
                  WDDY(WJ)=DY(ATMP)
                  WDDZ(WJ)=DZ(ATMP)
                ENDDO
              ENDIF
              IF (BONDMMPT(HBI,I,4).EQ.2) THEN !(1-swf)*H3O+swf*H2O



                CALL EPSEUDOBOND(BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3),ERM, &
                BNDHYD, BEQHYD ,0,DXBRM,DYBRM,DZBRM)

                EU = EU + ERM*(1.D0-SWF)
                DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))+DXBRM*(1.D0-SWF)
                DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))+DYBRM*(1.D0-SWF)
                DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))+DZBRM*(1.D0-SWF)

                DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))-DXBRM*(1.D0-SWF)
                DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))-DYBRM*(1.D0-SWF)
                DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))-DZBRM*(1.D0-SWF)

                DX1=DSWFRNH*DRNNX*ERM
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM

                DY1=DSWFRNH*DRNNY*ERM
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM

                DZ1=DSWFRNH*DRNNZ*ERM
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-DX1-DX2
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-DY1-DY2
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-DZ1-DZ2
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2


                CALL EPSEUDOBOND(BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3),ERM, &
                BNDWAT, BEQWAT ,0,DXBRM,DYBRM,DZBRM)

                EU = EU + ERM*SWF
                DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))+DXBRM*SWF
                DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))+DYBRM*SWF
                DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))+DZBRM*SWF

                DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))-DXBRM*SWF
                DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))-DYBRM*SWF
                DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))-DZBRM*SWF

                DX1=DSWFRNH*DRNNX*ERM
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM

                DY1=DSWFRNH*DRNNY*ERM
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM

                DZ1=DSWFRNH*DRNNZ*ERM
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX1+DX2
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY1+DY2
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ1+DZ2
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                EU=EU-(BEQHYD-BEQWAT)**2/(1.D0/((1.D0-SWF)*BNDHYD)+ &
                1.D0/(SWF*BNDWAT))

                DDSWF=0.D0-(BEQHYD-BEQWAT)**2*BNDHYD*BNDWAT/(SWF*BNDWAT+ &
                (1.D0-SWF)*BNDHYD)**2*((1.D0-SWF)**2*BNDHYD-SWF**2*BNDWAT)

                DX1=DSWFRNH*DRNNX*DDSWF
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

                DY1=DSWFRNH*DRNNY*DDSWF
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

                DZ1=DSWFRNH*DRNNZ*DDSWF
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
              ELSE IF (BONDMMPT(HBI,I,4).EQ.-2) THEN !swf*H3O+(1-swf)*H2O

                CALL EPSEUDOBOND(BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3),ERM, &
                BNDHYD, BEQHYD ,0,DXBRM,DYBRM,DZBRM)

                EU = EU + ERM*SWF
                DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))+DXBRM*SWF
                DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))+DYBRM*SWF
                DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))+DZBRM*SWF

                DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))-DXBRM*SWF
                DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))-DYBRM*SWF
                DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))-DZBRM*SWF

                DX1=DSWFRNH*DRNNX*ERM
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM

                DY1=DSWFRNH*DRNNY*ERM
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM

                DZ1=DSWFRNH*DRNNZ*ERM
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX1+DX2
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY1+DY2
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ1+DZ2
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                CALL EPSEUDOBOND(BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3),ERM, &
                BNDWAT, BEQWAT ,0,DXBRM,DYBRM,DZBRM)

                EU = EU + ERM*(1.D0-SWF)
                DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))+DXBRM*(1.D0-SWF)
                DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))+DYBRM*(1.D0-SWF)
                DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))+DZBRM*(1.D0-SWF)

                DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))-DXBRM*(1.D0-SWF)
                DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))-DYBRM*(1.D0-SWF)
                DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))-DZBRM*(1.D0-SWF)

                DX1=DSWFRNH*DRNNX*ERM
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM

                DY1=DSWFRNH*DRNNY*ERM
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM

                DZ1=DSWFRNH*DRNNZ*ERM
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-DX1-DX2
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-DY1-DY2
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-DZ1-DZ2
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

                EU=EU-(BEQWAT-BEQHYD)**2/(1.D0/((1.D0-SWF)*BNDWAT)+ &
                1.D0/(SWF*BNDHYD))

                DDSWF=0.D0-(BEQHYD-BEQWAT)**2*BNDHYD*BNDWAT/(SWF*BNDWAT+ &
                (1.D0-SWF)*BNDHYD)**2*((1.D0-SWF)**2*BNDHYD-SWF**2*BNDWAT)

                DX1=DSWFRNH*DRNNX*DDSWF
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

                DY1=DSWFRNH*DRNNY*DDSWF
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

                DZ1=DSWFRNH*DRNNZ*DDSWF
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
              ENDIF

              !EWEIGHT-03 AT WMODE=2
              IF (WMODE.EQ.2 .AND. EMODE.EQ.1) THEN

                EW=EW+(EU-EWTEMP)
                DO WJ=1,WNATMX
                  ATMP=WATOMM(WJ)
                  WDX(ATMP)=WDX(ATMP)+(DX(ATMP)-WDDX(WJ))
                  WDY(ATMP)=WDY(ATMP)+(DY(ATMP)-WDDY(WJ))
                  WDZ(ATMP)=WDZ(ATMP)+(DZ(ATMP)-WDDZ(WJ))
                ENDDO

                IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EB_OH=', &
                EU-EWTEMP,0.D0,BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3)

                !IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EB_OH=',EU-EWTEMP, &
                !BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3)
              ENDIF
            ENDIF

            if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_BOND_corr=', &
            EU-ETMP,BONDMMPT(HBI,I,2),BONDMMPT(HBI,I,3)

          ENDIF
        ELSE
  !WRITE (OUTU,*) 'TEST-02>'
          IF (I.EQ.1) THEN
            CALL EBONDMSPT(BONDMMPT(HBI,I,1),BONDMMPT(HBI,I,2), &
            BONDMMPT(HBI,I,3),ERM,DXBRM,DYBRM,DZBRM)

            EU = EU - ERM

            DX(BONDMMPT(HBI,I,2))=DX(BONDMMPT(HBI,I,2))-DXBRM
            DY(BONDMMPT(HBI,I,2))=DY(BONDMMPT(HBI,I,2))-DYBRM
            DZ(BONDMMPT(HBI,I,2))=DZ(BONDMMPT(HBI,I,2))-DZBRM

            DX(BONDMMPT(HBI,I,3))=DX(BONDMMPT(HBI,I,3))+DXBRM
            DY(BONDMMPT(HBI,I,3))=DY(BONDMMPT(HBI,I,3))+DYBRM
            DZ(BONDMMPT(HBI,I,3))=DZ(BONDMMPT(HBI,I,3))+DZBRM
          ELSE IF (I.GT.1 .AND. EMODE.EQ.1 .AND. WMODE.GE.2) THEN
            CALL EBONDMSPT(BONDMMPT(HBI,I,1),BONDMMPT(HBI,I,2), &
            BONDMMPT(HBI,I,3),ERM,DXBRM,DYBRM,DZBRM)

            EW = EW + ERM

            WDX(BONDMMPT(HBI,I,2))=WDX(BONDMMPT(HBI,I,2))+(DXBRM)
            WDY(BONDMMPT(HBI,I,2))=WDY(BONDMMPT(HBI,I,2))+(DYBRM)
            WDZ(BONDMMPT(HBI,I,2))=WDZ(BONDMMPT(HBI,I,2))+(DZBRM)

            WDX(BONDMMPT(HBI,I,3))=WDX(BONDMMPT(HBI,I,3))-(DXBRM)
            WDY(BONDMMPT(HBI,I,3))=WDY(BONDMMPT(HBI,I,3))-(DYBRM)
            WDZ(BONDMMPT(HBI,I,3))=WDZ(BONDMMPT(HBI,I,3))-(DZBRM)
          ENDIF
        ENDIF
      ENDDO
    ENDIF
    !PSEUDO-BOND WHEN IF D-A ARE TOO CLOSE OR TOO FAR
    IF (EMODE.EQ.1 .AND. PRM_DELTAV.GT.0.D0) THEN

      CALL EPSEUDOBOND(HBRDATOM(HBI,1),HBRDATOM(HBI,3),ERM,1000.0D0, &
      2.0D0,-2,DXBRM,DYBRM,DZBRM)

      IF (ERM.GT.0.D0) THEN
        EU = EU + ERM

        DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DXBRM
        DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DYBRM
        DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZBRM

        DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DXBRM
        DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DYBRM
        DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZBRM
      ENDIF

      if (PRNLEV>=7 .AND. ERM.GT.0.D0) WRITE (OUTU,*) &
      'MSMMPT-EMMPT2> E_DA_2A=',ERM
      !EWEIGHT-04
      IF (WMODE.GE.1 .AND. ERM.GT.0.D0) THEN
        EW=EW+ERM

        WDX(HBRDATOM(HBI,1))=WDX(HBRDATOM(HBI,1))+DXBRM
        WDY(HBRDATOM(HBI,1))=WDY(HBRDATOM(HBI,1))+DYBRM
        WDZ(HBRDATOM(HBI,1))=WDZ(HBRDATOM(HBI,1))+DZBRM

        WDX(HBRDATOM(HBI,3))=WDX(HBRDATOM(HBI,3))-DXBRM
        WDY(HBRDATOM(HBI,3))=WDY(HBRDATOM(HBI,3))-DYBRM
        WDZ(HBRDATOM(HBI,3))=WDZ(HBRDATOM(HBI,3))-DZBRM

        IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EB_OO_SHORT=',ERM
      ENDIF

      IF (PRM_INTERNAL_LONG_OO.GE.0.D0) THEN
        !!c'mmpt: addtional potential wall if D-A is close to cutoff
        CALL EPSEUDOBOND(HBRDATOM(HBI,1),HBRDATOM(HBI,3),ERM,PRM_INTERNAL_LONG_OO, &
        PRM_INTERNAL_DA_DIST-0.8d0,2,DXBRM,DYBRM,DZBRM)

        IF (ERM.GT.0.D0) THEN
          EU = EU + ERM

          DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DXBRM
          DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DYBRM
          DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZBRM

          DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DXBRM
          DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DYBRM
          DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZBRM
        ENDIF

        !IF (PRNLEV>=7 .AND. ERM.GT.0.D0) WRITE (OUTU,*) &
        !'MSMMPT-EMMPT2> E_DA_cutoff=',ERM
        !EWEIGHT-05
        IF (WMODE.GE.1 .AND. ERM.GT.0.D0) THEN
          EW=EW+ERM

          WDX(HBRDATOM(HBI,1))=WDX(HBRDATOM(HBI,1))+DXBRM
          WDY(HBRDATOM(HBI,1))=WDY(HBRDATOM(HBI,1))+DYBRM
          WDZ(HBRDATOM(HBI,1))=WDZ(HBRDATOM(HBI,1))+DZBRM

          WDX(HBRDATOM(HBI,3))=WDX(HBRDATOM(HBI,3))-DXBRM
          WDY(HBRDATOM(HBI,3))=WDY(HBRDATOM(HBI,3))-DYBRM
          WDZ(HBRDATOM(HBI,3))=WDZ(HBRDATOM(HBI,3))-DZBRM

          IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EB_OO_LONG=',ERM
        ENDIF
      ENDIF
    ENDIF

    ETMP=EU

!c'emmpt: identify angle energies by swithcing function
!      ASSUMES THAT TRANSFERED HYDROGEN ATOM IS ALWAYS FIRST OR FOURTH
!      ATOM IN MMPT ANGLE DEFINITION AND SECOND IN MMPT HYDROGEN BOND
!      DEFINITION
    IF (EMODE.NE.3) THEN
      DO I=1,ANGLNUM(HBI)

        !CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ANGLMMPT(HBI,I,2),     &
        !ANGLMMPT(HBI,I,3),ANGLMMPT(HBI,I,4),EA,                 &
        !DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

  !      CHECK IF ANGLE IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
  !      TO SWITCHING FUNCTION

        ETMP=EU
        IF (ABS(ANGLMMPT(HBI,I,5)).EQ.1 .AND. EMODE.EQ.1) THEN
          ATOM1=ANGLMMPT(HBI,I,2)
          ATOM2=ANGLMMPT(HBI,I,3)
          ATOM3=ANGLMMPT(HBI,I,4)

          IF (AEQHYD.GT.0.D0 .AND. AEQWAT.GT.0.D0 .AND. EQUI2.GT.0) THEN

            IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
              CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU-EA

              DX(ATOM1)=DX(ATOM1)-DXAI
              DX(ATOM2)=DX(ATOM2)-DXAJ
              DX(ATOM3)=DX(ATOM3)-DXAK

              DY(ATOM1)=DY(ATOM1)-DYAI
              DY(ATOM2)=DY(ATOM2)-DYAJ
              DY(ATOM3)=DY(ATOM3)-DYAK

              DZ(ATOM1)=DZ(ATOM1)-DZAI
              DZ(ATOM2)=DZ(ATOM2)-DZAJ
              DZ(ATOM3)=DZ(ATOM3)-DZAK
            ENDIF

            !EWEIGHT-06
            IF (WMODE.GE.2) THEN
              EWTEMP=EU
              WNATMX=3
              WATOMM(1)=HBRDATOM(HBI,1)
              WATOMM(2)=HBRDATOM(HBI,2)
              WATOMM(3)=HBRDATOM(HBI,3)

              IF (ATOM1.NE.WATOMM(1) .AND. ATOM1.NE.WATOMM(2) .AND. &
              ATOM1.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM1
              ENDIF
              IF (ATOM2.NE.WATOMM(1) .AND. ATOM2.NE.WATOMM(2) .AND. &
              ATOM2.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM2
              ENDIF
              IF (ATOM3.NE.WATOMM(1) .AND. ATOM3.NE.WATOMM(2) .AND. &
              ATOM3.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM3
              ENDIF

              DO WJ=1,WNATMX
                ATMP=WATOMM(WJ)
                WDDX(WJ)=DX(ATMP)
                WDDY(WJ)=DY(ATMP)
                WDDZ(WJ)=DZ(ATMP)
              ENDDO
  !write(outu,*) 'TEST-06>',WNATMX,WATOMM(1),WATOMM(2),WATOMM(3),WATOMM(4),WATOMM(5),WATOMM(6)
            ENDIF

            IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
              !c'emmpt: pseudo potential on X-D-H, ONLY IN MODE 1
  !WRITE (OUTU,*),'TEST-04>'
              IF (EMODE.EQ.1) THEN

                CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLHYD,AEQHYD,0, &
                DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

                EU=EU+(1.D0-SWF)*EA

                DX(ATOM1)=DX(ATOM1)+DXAI*(1.D0-SWF)
                DX(ATOM2)=DX(ATOM2)+DXAJ*(1.D0-SWF)
                DX(ATOM3)=DX(ATOM3)+DXAK*(1.D0-SWF)

                DY(ATOM1)=DY(ATOM1)+DYAI*(1.D0-SWF)
                DY(ATOM2)=DY(ATOM2)+DYAJ*(1.D0-SWF)
                DY(ATOM3)=DY(ATOM3)+DYAK*(1.D0-SWF)

                DZ(ATOM1)=DZ(ATOM1)+DZAI*(1.D0-SWF)
                DZ(ATOM2)=DZ(ATOM2)+DZAJ*(1.D0-SWF)
                DZ(ATOM3)=DZ(ATOM3)+DZAK*(1.D0-SWF)

                DX1=DSWFRNH*DRNNX*EA
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

                DY1=DSWFRNH*DRNNY*EA
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

                DZ1=DSWFRNH*DRNNZ*EA
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

                CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,FC2,EQUI2,0, &
                DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

                EU=EU+EA*SWF

                DX(ATOM1)=DX(ATOM1)+(DXAI*SWF)
                DX(ATOM2)=DX(ATOM2)+(DXAJ*SWF)
                DX(ATOM3)=DX(ATOM3)+(DXAK*SWF)

                DY(ATOM1)=DY(ATOM1)+(DYAI*SWF)
                DY(ATOM2)=DY(ATOM2)+(DYAJ*SWF)
                DY(ATOM3)=DY(ATOM3)+(DYAK*SWF)

                DZ(ATOM1)=DZ(ATOM1)+(DZAI*SWF)
                DZ(ATOM2)=DZ(ATOM2)+(DZAJ*SWF)
                DZ(ATOM3)=DZ(ATOM3)+(DZAK*SWF)

                DX1=DSWFRNH*DRNNX*EA
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

                DY1=DSWFRNH*DRNNY*EA
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

                DZ1=DSWFRNH*DRNNZ*EA
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2


                EU=EU-(AEQHYD-EQUI2)**2/(1.D0/((1.D0-SWF)*AGLHYD)+ &
                1.D0/(SWF*FC2))

                DDSWF=0.D0-(AEQHYD-EQUI2)**2*AGLHYD*FC2/(SWF*FC2+(1.D0-SWF)* &
                AGLHYD)**2*((1.D0-SWF)**2*AGLHYD-SWF**2*FC2)

                DX1=DSWFRNH*DRNNX*DDSWF
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

                DY1=DSWFRNH*DRNNY*DDSWF
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

                DZ1=DSWFRNH*DRNNZ*DDSWF
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_ANGL_corr=', &
                EU-ETMP, ATOM1, ATOM2, ATOM3

                !EWEIGHT-06
                IF (WMODE.GE.2) THEN
                  EW=EW+(EU-EWTEMP)
                  DO WJ=1,WNATMX
                    ATMP=WATOMM(WJ)
                    WDX(ATMP)=WDX(ATMP)+(DX(ATMP)-WDDX(WJ))
                    WDY(ATMP)=WDY(ATMP)+(DY(ATMP)-WDDY(WJ))
                    WDZ(ATMP)=WDZ(ATMP)+(DZ(ATMP)-WDDZ(WJ))
                  ENDDO

                  IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOHs=', &
                  EU-EWTEMP,ATOM1,ATOM2,ATOM3
                ENDIF

              !pseudo-dihedral energies for X-D-A-Y, only in MODE 1
                DO J=1,ANGLNUM(HBI)
                  IF (ANGLMMPT(HBI,J,5).EQ.-1) THEN
                    IF (ANGLMMPT(HBI,I,2).EQ.HBRDATOM(HBI,2)) THEN
                      DIHATOM1=ANGLMMPT(HBI,I,4)
                    ELSE
                      DIHATOM1=ANGLMMPT(HBI,I,2)
                    ENDIF
                    DIHATOM2=ANGLMMPT(HBI,I,3)
                    IF (ANGLMMPT(HBI,J,2).EQ.HBRDATOM(HBI,2)) THEN
                      DIHATOM4=ANGLMMPT(HBI,J,4)
                    ELSE
                      DIHATOM4=ANGLMMPT(HBI,J,2)
                    ENDIF
                    DIHATOM3=ANGLMMPT(HBI,J,3)

                    CALL EPSEUDODIHEDRAL(DIHATOM1,DIHATOM2,DIHATOM3,DIHATOM4,FCDIH,ED, &
                    DXDI,DXDJ,DXDK,DXDL,DYDI,DYDJ,DYDK,DYDL,DZDI,DZDJ,DZDK,DZDL)

                    EU=EU+ED

                    DX(DIHATOM1)=DX(DIHATOM1)+DXDI
                    DX(DIHATOM2)=DX(DIHATOM2)+DXDJ
                    DX(DIHATOM3)=DX(DIHATOM3)+DXDK
                    DX(DIHATOM4)=DX(DIHATOM4)+DXDL

                    DY(DIHATOM1)=DY(DIHATOM1)+DYDI
                    DY(DIHATOM2)=DY(DIHATOM2)+DYDJ
                    DY(DIHATOM3)=DY(DIHATOM3)+DYDK
                    DY(DIHATOM4)=DY(DIHATOM4)+DYDL

                    DZ(DIHATOM1)=DZ(DIHATOM1)+DZDI
                    DZ(DIHATOM2)=DZ(DIHATOM2)+DZDJ
                    DZ(DIHATOM3)=DZ(DIHATOM3)+DZDK
                    DZ(DIHATOM4)=DZ(DIHATOM4)+DZDL

                    !EWEIGHT-06-DIH
                    IF (WMODE.GE.2) THEN
                      EW=EW+(ED)

                      WDX(DIHATOM1)=WDX(DIHATOM1)+(DXDI)
                      WDX(DIHATOM2)=WDX(DIHATOM2)+(DXDJ)
                      WDX(DIHATOM3)=WDX(DIHATOM3)+(DXDK)
                      WDX(DIHATOM4)=WDX(DIHATOM4)+(DXDL)

                      WDY(DIHATOM1)=WDY(DIHATOM1)+(DYDI)
                      WDY(DIHATOM2)=WDY(DIHATOM2)+(DYDJ)
                      WDY(DIHATOM3)=WDY(DIHATOM3)+(DYDK)
                      WDY(DIHATOM4)=WDY(DIHATOM4)+(DYDL)

                      WDZ(DIHATOM1)=WDZ(DIHATOM1)+(DZDI)
                      WDZ(DIHATOM2)=WDZ(DIHATOM2)+(DZDJ)
                      WDZ(DIHATOM3)=WDZ(DIHATOM3)+(DZDK)
                      WDZ(DIHATOM4)=WDZ(DIHATOM4)+(DZDL)

                      IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EDI=',ED, &
                      DIHATOM1,DIHATOM2,DIHATOM3,DIHATOM4
                      if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_DIHE=', &
                      ED,DIHATOM1,DIHATOM2,DIHATOM3,DIHATOM4
                    ENDIF
                  ENDIF
                ENDDO
              ENDIF

            ELSE

              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLHYD,AEQHYD,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+SWF*EA

              DX(ATOM1)=DX(ATOM1)+DXAI*SWF
              DX(ATOM2)=DX(ATOM2)+DXAJ*SWF
              DX(ATOM3)=DX(ATOM3)+DXAK*SWF

              DY(ATOM1)=DY(ATOM1)+DYAI*SWF
              DY(ATOM2)=DY(ATOM2)+DYAJ*SWF
              DY(ATOM3)=DY(ATOM3)+DYAK*SWF

              DZ(ATOM1)=DZ(ATOM1)+DZAI*SWF
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*SWF
              DZ(ATOM3)=DZ(ATOM3)+DZAK*SWF

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,FC2,EQUI2,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+EA*(1.D0-SWF)

              DX(ATOM1)=DX(ATOM1)+DXAI*(1.D0-SWF)
              DX(ATOM2)=DX(ATOM2)+DXAJ*(1.D0-SWF)
              DX(ATOM3)=DX(ATOM3)+DXAK*(1.D0-SWF)

              DY(ATOM1)=DY(ATOM1)+DYAI*(1.D0-SWF)
              DY(ATOM2)=DY(ATOM2)+DYAJ*(1.D0-SWF)
              DY(ATOM3)=DY(ATOM3)+DYAK*(1.D0-SWF)

              DZ(ATOM1)=DZ(ATOM1)+DZAI*(1.D0-SWF)
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*(1.D0-SWF)
              DZ(ATOM3)=DZ(ATOM3)+DZAK*(1.D0-SWF)

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

              EU=EU-(AEQHYD-EQUI2)**2/(1.D0/((1.D0-SWF)*FC2)+ &
              1.D0/(SWF*AGLHYD))

              DDSWF=0.D0-(AEQHYD-EQUI2)**2*FC2*AGLHYD/(SWF*AGLHYD+(1.D0-SWF)* &
              FC2)**2*((1.D0-SWF)**2*FC2-SWF**2*AGLHYD)

              DX1=DSWFRNH*DRNNX*DDSWF
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

              DY1=DSWFRNH*DRNNY*DDSWF
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

              DZ1=DSWFRNH*DRNNZ*DDSWF
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_ANGL_corr=', &
              EU-ETMP, ATOM1, ATOM2, ATOM3

              !EWEIGHT-06
              IF (WMODE.GE.2) THEN
                EW=EW+(EU-EWTEMP)
                DO WJ=1,WNATMX
                  ATMP=WATOMM(WJ)
                  WDX(ATMP)=WDX(ATMP)+(DX(ATMP)-WDDX(WJ))
                  WDY(ATMP)=WDY(ATMP)+(DY(ATMP)-WDDY(WJ))
                  WDZ(ATMP)=WDZ(ATMP)+(DZ(ATMP)-WDDZ(WJ))
                ENDDO
    !WRITE (OUTU,*) 'EW-06>',EU-EWTEMP
                IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOHs=', &
                EU-EWTEMP,ATOM1,ATOM2,ATOM3
              ENDIF
            ENDIF

          ELSE
            IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
              CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU-EA*SWF

              DX(ATOM1)=DX(ATOM1)-DXAI*SWF
              DX(ATOM2)=DX(ATOM2)-DXAJ*SWF
              DX(ATOM3)=DX(ATOM3)-DXAK*SWF

              DY(ATOM1)=DY(ATOM1)-DYAI*SWF
              DY(ATOM2)=DY(ATOM2)-DYAJ*SWF
              DY(ATOM3)=DY(ATOM3)-DYAK*SWF

              DZ(ATOM1)=DZ(ATOM1)-DZAI*SWF
              DZ(ATOM2)=DZ(ATOM2)-DZAJ*SWF
              DZ(ATOM3)=DZ(ATOM3)-DZAK*SWF

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

              !EWEIGHT-07
              EW=EW+(EA*(1.0D0-SWF))
              WDX(ATOM1)=WDX(ATOM1)+(DXAI*(1.D0-SWF))
              WDX(ATOM2)=WDX(ATOM2)+(DXAJ*(1.D0-SWF))
              WDX(ATOM3)=WDX(ATOM3)+(DXAK*(1.D0-SWF))

              WDY(ATOM1)=WDY(ATOM1)+(DYAI*(1.D0-SWF))
              WDY(ATOM2)=WDY(ATOM2)+(DYAJ*(1.D0-SWF))
              WDY(ATOM3)=WDY(ATOM3)+(DYAK*(1.D0-SWF))

              WDZ(ATOM1)=WDZ(ATOM1)+(DZAI*(1.D0-SWF))
              WDZ(ATOM2)=WDZ(ATOM2)+(DZAJ*(1.D0-SWF))
              WDZ(ATOM3)=WDZ(ATOM3)+(DZAK*(1.D0-SWF))

              WDX(HBRDATOM(HBI,1))=WDX(HBRDATOM(HBI,1))-((DX1+DX2))
              WDX(HBRDATOM(HBI,2))=WDX(HBRDATOM(HBI,2))+(DX1)
              WDX(HBRDATOM(HBI,3))=WDX(HBRDATOM(HBI,3))+(DX2)

              WDY(HBRDATOM(HBI,1))=WDY(HBRDATOM(HBI,1))-((DY1+DY2))
              WDY(HBRDATOM(HBI,2))=WDY(HBRDATOM(HBI,2))+(DY1)
              WDY(HBRDATOM(HBI,3))=WDY(HBRDATOM(HBI,3))+(DY2)

              WDZ(HBRDATOM(HBI,1))=WDZ(HBRDATOM(HBI,1))-((DZ1+DZ2))
              WDZ(HBRDATOM(HBI,2))=WDZ(HBRDATOM(HBI,2))+(DZ1)
              WDZ(HBRDATOM(HBI,3))=WDZ(HBRDATOM(HBI,3))+(DZ2)

              IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOHs=', &
              EA*(1.0D0-SWF),ATOM1,ATOM2,ATOM3
            ELSE
              CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+EA*SWF

              DX(ATOM1)=DX(ATOM1)+DXAI*SWF
              DX(ATOM2)=DX(ATOM2)+DXAJ*SWF
              DX(ATOM3)=DX(ATOM3)+DXAK*SWF

              DY(ATOM1)=DY(ATOM1)+DYAI*SWF
              DY(ATOM2)=DY(ATOM2)+DYAJ*SWF
              DY(ATOM3)=DY(ATOM3)+DYAK*SWF

              DZ(ATOM1)=DZ(ATOM1)+DZAI*SWF
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*SWF
              DZ(ATOM3)=DZ(ATOM3)+DZAK*SWF

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              !EWEIGHT-08
              EW=EW+(EA*SWF)

              WDX(ATOM1)=WDX(ATOM1)+(DXAI*SWF)
              WDX(ATOM2)=WDX(ATOM2)+(DXAJ*SWF)
              WDX(ATOM3)=WDX(ATOM3)+(DXAK*SWF)

              WDY(ATOM1)=WDY(ATOM1)+(DYAI*SWF)
              WDY(ATOM2)=WDY(ATOM2)+(DYAJ*SWF)
              WDY(ATOM3)=WDY(ATOM3)+(DYAK*SWF)

              WDZ(ATOM1)=WDZ(ATOM1)+(DZAI*SWF)
              WDZ(ATOM2)=WDZ(ATOM2)+(DZAJ*SWF)
              WDZ(ATOM3)=WDZ(ATOM3)+(DZAK*SWF)

              WDX(HBRDATOM(HBI,1))=WDX(HBRDATOM(HBI,1))+((DX1+DX2))
              WDX(HBRDATOM(HBI,2))=WDX(HBRDATOM(HBI,2))-(DX1)
              WDX(HBRDATOM(HBI,3))=WDX(HBRDATOM(HBI,3))-(DX2)

              WDY(HBRDATOM(HBI,1))=WDY(HBRDATOM(HBI,1))+((DY1+DY2))
              WDY(HBRDATOM(HBI,2))=WDY(HBRDATOM(HBI,2))-(DY1)
              WDY(HBRDATOM(HBI,3))=WDY(HBRDATOM(HBI,3))-(DY2)

              WDZ(HBRDATOM(HBI,1))=WDZ(HBRDATOM(HBI,1))+((DZ1+DZ2))
              WDZ(HBRDATOM(HBI,2))=WDZ(HBRDATOM(HBI,2))-(DZ1)
              WDZ(HBRDATOM(HBI,3))=WDZ(HBRDATOM(HBI,3))-(DZ2)

              IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOHs=', &
              EA*SWF,ATOM1,ATOM2,ATOM3
            ENDIF
          ENDIF
        ELSE IF (ABS(ANGLMMPT(HBI,I,5)).EQ.1 .AND. EMODE.EQ.2) THEN
  !WRITE (OUTU,*) 'TEST-04>'
          ATOM1=ANGLMMPT(HBI,I,2)
          ATOM2=ANGLMMPT(HBI,I,3)
          ATOM3=ANGLMMPT(HBI,I,4)

          IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
            CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
            DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

            EU=EU-EA

            DX(ATOM1)=DX(ATOM1)-DXAI
            DX(ATOM2)=DX(ATOM2)-DXAJ
            DX(ATOM3)=DX(ATOM3)-DXAK

            DY(ATOM1)=DY(ATOM1)-DYAI
            DY(ATOM2)=DY(ATOM2)-DYAJ
            DY(ATOM3)=DY(ATOM3)-DYAK

            DZ(ATOM1)=DZ(ATOM1)-DZAI
            DZ(ATOM2)=DZ(ATOM2)-DZAJ
            DZ(ATOM3)=DZ(ATOM3)-DZAK
          ELSE
            CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
            DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

            EU=EU+EA

            DX(ATOM1)=DX(ATOM1)+DXAI
            DX(ATOM2)=DX(ATOM2)+DXAJ
            DX(ATOM3)=DX(ATOM3)+DXAK

            DY(ATOM1)=DY(ATOM1)+DYAI
            DY(ATOM2)=DY(ATOM2)+DYAJ
            DY(ATOM3)=DY(ATOM3)+DYAK

            DZ(ATOM1)=DZ(ATOM1)+DZAI
            DZ(ATOM2)=DZ(ATOM2)+DZAJ
            DZ(ATOM3)=DZ(ATOM3)+DZAK
          ENDIF
  !WRITE (OUTU,*) 'TEST-05>'
        ELSE IF (ABS(ANGLMMPT(HBI,I,5)).EQ.2 .AND. EMODE.EQ.1) THEN
          IF (AEQHYD.GT.0.D0 .AND. AEQWAT.GT.0.D0 .AND. EQUI2.GT.0.D0) THEN
            ATOM1=ANGLMMPT(HBI,I,2)
            ATOM2=ANGLMMPT(HBI,I,3)
            ATOM3=ANGLMMPT(HBI,I,4)

            CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
            DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

            EU=EU-EA

            DX(ATOM1)=DX(ATOM1)-DXAI
            DX(ATOM2)=DX(ATOM2)-DXAJ
            DX(ATOM3)=DX(ATOM3)-DXAK

            DY(ATOM1)=DY(ATOM1)-DYAI
            DY(ATOM2)=DY(ATOM2)-DYAJ
            DY(ATOM3)=DY(ATOM3)-DYAK

            DZ(ATOM1)=DZ(ATOM1)-DZAI
            DZ(ATOM2)=DZ(ATOM2)-DZAJ
            DZ(ATOM3)=DZ(ATOM3)-DZAK

            !EWEIGHT-09
            IF (WMODE.GE.2) THEN

              EWTEMP=EU
              WNATMX=3
              WATOMM(1)=HBRDATOM(HBI,1)
              WATOMM(2)=HBRDATOM(HBI,2)
              WATOMM(3)=HBRDATOM(HBI,3)
              IF (ATOM1.NE.WATOMM(1) .AND. ATOM1.NE.WATOMM(2) .AND. &
              ATOM1.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM1
              ENDIF
              IF (ATOM2.NE.WATOMM(1) .AND. ATOM2.NE.WATOMM(2) .AND. &
              ATOM2.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM2
              ENDIF
              IF (ATOM3.NE.WATOMM(1) .AND. ATOM3.NE.WATOMM(2) .AND. &
              ATOM3.NE.WATOMM(3)) THEN
                WNATMX=WNATMX+1
                WATOMM(WNATMX)=ATOM3
              ENDIF

              DO WJ=1,WNATMX
                ATMP=WATOMM(WJ)
                WDDX(WJ)=DX(ATMP)
                WDDY(WJ)=DY(ATMP)
                WDDZ(WJ)=DZ(ATMP)
              ENDDO
            ENDIF

            IF (ANGLMMPT(HBI,I,5).EQ.2) THEN
              !c'emmpt: pseudo potential on X-D-H, ONLY IN MODE 1
              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLHYD,AEQHYD,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+(1.D0-SWF)*EA

              DX(ATOM1)=DX(ATOM1)+DXAI*(1.D0-SWF)
              DX(ATOM2)=DX(ATOM2)+DXAJ*(1.D0-SWF)
              DX(ATOM3)=DX(ATOM3)+DXAK*(1.D0-SWF)

              DY(ATOM1)=DY(ATOM1)+DYAI*(1.D0-SWF)
              DY(ATOM2)=DY(ATOM2)+DYAJ*(1.D0-SWF)
              DY(ATOM3)=DY(ATOM3)+DYAK*(1.D0-SWF)

              DZ(ATOM1)=DZ(ATOM1)+DZAI*(1.D0-SWF)
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*(1.D0-SWF)
              DZ(ATOM3)=DZ(ATOM3)+DZAK*(1.D0-SWF)

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2

              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLWAT,AEQWAT,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+EA*SWF

              DX(ATOM1)=DX(ATOM1)+DXAI*SWF
              DX(ATOM2)=DX(ATOM2)+DXAJ*SWF
              DX(ATOM3)=DX(ATOM3)+DXAK*SWF

              DY(ATOM1)=DY(ATOM1)+DYAI*SWF
              DY(ATOM2)=DY(ATOM2)+DYAJ*SWF
              DY(ATOM3)=DY(ATOM3)+DYAK*SWF

              DZ(ATOM1)=DZ(ATOM1)+DZAI*SWF
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*SWF
              DZ(ATOM3)=DZ(ATOM3)+DZAK*SWF

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2


              EU=EU-(AEQHYD-AEQWAT)**2/(1.D0/((1.D0-SWF)*AGLHYD)+ &
              1.D0/(SWF*AGLWAT))

              DDSWF=0.D0-(AEQHYD-AEQWAT)**2*AGLHYD*AGLWAT/(SWF*AGLWAT+ &
              (1.D0-SWF)*AGLHYD)**2*((1.D0-SWF)**2*AGLHYD-SWF**2*AGLWAT)

              DX1=DSWFRNH*DRNNX*DDSWF
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

              DY1=DSWFRNH*DRNNY*DDSWF
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

              DZ1=DSWFRNH*DRNNZ*DDSWF
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_ANGL_corr=', &
              EU-ETMP, ATOM1, ATOM2, ATOM3

            ELSE IF (ANGLMMPT(HBI,I,5).EQ.-2) THEN
              !c'emmpt: pseudo potential on X-D-H, ONLY IN MODE 1
              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLHYD,AEQHYD,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+SWF*EA

              DX(ATOM1)=DX(ATOM1)+DXAI*SWF
              DX(ATOM2)=DX(ATOM2)+DXAJ*SWF
              DX(ATOM3)=DX(ATOM3)+DXAK*SWF

              DY(ATOM1)=DY(ATOM1)+DYAI*SWF
              DY(ATOM2)=DY(ATOM2)+DYAJ*SWF
              DY(ATOM3)=DY(ATOM3)+DYAK*SWF

              DZ(ATOM1)=DZ(ATOM1)+DZAI*SWF
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*SWF
              DZ(ATOM3)=DZ(ATOM3)+DZAK*SWF

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              CALL EPSEUDOMMPT(ATOM1,ATOM2,ATOM3,EA,AGLWAT,AEQWAT,0, &
              DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

              EU=EU+EA*(1.D0-SWF)

              DX(ATOM1)=DX(ATOM1)+DXAI*(1.D0-SWF)
              DX(ATOM2)=DX(ATOM2)+DXAJ*(1.D0-SWF)
              DX(ATOM3)=DX(ATOM3)+DXAK*(1.D0-SWF)

              DY(ATOM1)=DY(ATOM1)+DYAI*(1.D0-SWF)
              DY(ATOM2)=DY(ATOM2)+DYAJ*(1.D0-SWF)
              DY(ATOM3)=DY(ATOM3)+DYAK*(1.D0-SWF)

              DZ(ATOM1)=DZ(ATOM1)+DZAI*(1.D0-SWF)
              DZ(ATOM2)=DZ(ATOM2)+DZAJ*(1.D0-SWF)
              DZ(ATOM3)=DZ(ATOM3)+DZAK*(1.D0-SWF)

              DX1=DSWFRNH*DRNNX*EA
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA

              DY1=DSWFRNH*DRNNY*EA
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA

              DZ1=DSWFRNH*DRNNZ*EA
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2


              EU=EU-(AEQHYD-AEQWAT)**2/(1.D0/((1.D0-SWF)*AGLWAT)+ &
              1.D0/(SWF*AGLHYD))

              DDSWF=0.D0-(AEQHYD-AEQWAT)**2*AGLWAT*AGLHYD/(SWF*AGLHYD+ &
              (1.D0-SWF)*AGLWAT)**2*((1.D0-SWF)**2*AGLWAT-SWF**2*AGLHYD)

              DX1=DSWFRNH*DRNNX*DDSWF
              DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

              DY1=DSWFRNH*DRNNY*DDSWF
              DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

              DZ1=DSWFRNH*DRNNZ*DDSWF
              DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

              DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
              DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
              DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

              DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
              DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
              DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

              DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
              DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
              DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

              if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_ANGL_corr=', &
              EU-ETMP, ATOM1, ATOM2, ATOM3
            ENDIF

            !EWEIGHT-09
            IF (WMODE.GE.2) THEN
              EW=EW+(EU-EWTEMP)
              DO WJ=1,WNATMX
                ATMP=WATOMM(WJ)
                WDX(ATMP)=WDX(ATMP)+(DX(ATMP)-WDDX(WJ))
                WDY(ATMP)=WDY(ATMP)+(DY(ATMP)-WDDY(WJ))
                WDZ(ATMP)=WDZ(ATMP)+(DZ(ATMP)-WDDZ(WJ))
              ENDDO
              IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOH=', &
              EU-EWTEMP,ATOM1,ATOM2,ATOM3
            ENDIF
          ELSE IF(WMODE.GE.2) THEN
            ATOM1=ANGLMMPT(HBI,I,2)
            ATOM2=ANGLMMPT(HBI,I,3)
            ATOM3=ANGLMMPT(HBI,I,4)

            CALL EANGLMSPT(ANGLMMPT(HBI,I,1),ATOM1,ATOM2,ATOM3,EA, &
            DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

            EW=EW+(EA)

            WDX(ATOM1)=WDX(ATOM1)+(DXAI)
            WDX(ATOM2)=WDX(ATOM2)+(DXAJ)
            WDX(ATOM3)=WDX(ATOM3)+(DXAK)

            WDY(ATOM1)=WDY(ATOM1)+(DYAI)
            WDY(ATOM2)=WDY(ATOM2)+(DYAJ)
            WDY(ATOM3)=WDY(ATOM3)+(DYAK)

            WDZ(ATOM1)=WDZ(ATOM1)+(DZAI)
            WDZ(ATOM2)=WDZ(ATOM2)+(DZAJ)
            WDZ(ATOM3)=WDZ(ATOM3)+(DZAK)

            IF (PRNLEV.GE.7) WRITE (OUTU,*) 'MSMMPT-EW > EA_HOH=', &
            EU-EWTEMP,ATOM1,ATOM2,ATOM3
          ENDIF
        ENDIF
      ENDDO
    ENDIF

!subtract pseudoenergy of dihedral and redistribute gradients into DH-A
!!only applies for water
!!and now switched off
    IF (EMODE.EQ.1 .and. FCDIH.ne.0.d0 .and. 0.eq.1) THEN

      PSI1=((1.D0-SWF)*AGLHYD*AEQHYD+SWF*AGLWAT*AEQWAT)/ &
      ((1.D0-SWF)*AGLHYD+SWF*AGLWAT)
      PSI2=((1.D0-SWF)*AGLHYD*AEQHYD+SWF*FC2*EQUI2)/ &
      ((1.D0-SWF)*AGLHYD+SWF*FC2)
      PHI1=2.D0*ASIN(SIN(PSI1/2.)/SIN(PSI2))
      PSI3=(SWF*AGLHYD*AEQHYD+(1.D0-SWF)*AGLWAT*AEQWAT)/ &
      (SWF*AGLHYD+(1.D0-SWF)*AGLWAT)
      PSI4=(SWF*AGLHYD*AEQHYD+(1.D0-SWF)*FC2*EQUI2)/ &
      (SWF*AGLHYD+(1.D0-SWF)*FC2)
      PHI2=2.D0*ASIN(SIN(PSI3/2.)/SIN(PSI4))

      EDIH_CORR=-FCDIH*4.D0*(1.D0-COS(PHI1)*COS(PHI2))
      EU=EU+EDIH_CORR

      IF (SWF.LE.0.5D0) THEN
        DELTA_C=0.000001D0
      ELSE
        DELTA_C=-0.000001D0
      ENDIF

      SWFTMP=SWF+DELTA_C
      PSI1=((1.D0-SWFTMP)*AGLHYD*AEQHYD+SWFTMP*AGLWAT*AEQWAT)/ &
      ((1.D0-SWFTMP)*AGLHYD+SWFTMP*AGLWAT)
      PSI2=((1.D0-SWFTMP)*AGLHYD*AEQHYD+SWFTMP*FC2*EQUI2)/ &
      ((1.D0-SWFTMP)*AGLHYD+SWFTMP*FC2)
      PHI1=2.D0*ASIN(SIN(PSI1/2.)/SIN(PSI2))
      PSI3=(SWFTMP*AGLHYD*AEQHYD+(1.D0-SWFTMP)*AGLWAT*AEQWAT)/ &
      (SWFTMP*AGLHYD+(1.D0-SWFTMP)*AGLWAT)
      PSI4=(SWFTMP*AGLHYD*AEQHYD+(1.D0-SWFTMP)*FC2*EQUI2)/ &
      (SWFTMP*AGLHYD+(1.D0-SWFTMP)*FC2)
      PHI2=2.D0*ASIN(SIN(PSI3/2.)/SIN(PSI4))

      EDIH_FINI=-FCDIH*4.D0*(1.D0-COS(PHI1)*COS(PHI2))
      DDSWF=(EDIH_FINI-EDIH_CORR)/DELTA_C

      !DDSWF = FCDIH/0.1D0*(2.D0*DIH3*(SWF-0.5D0)**2 &
      !-DIH2)*(2.D0*SWF-1.D0)

      DX1=DSWFRNH*DRNNX*DDSWF
      DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF

      DY1=DSWFRNH*DRNNY*DDSWF
      DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF

      DZ1=DSWFRNH*DRNNZ*DDSWF
      DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF

      DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
      DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
      DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

      DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
      DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
      DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

      DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
      DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
      DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

      if (PRNLEV>=7) WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_DIH_corr=', EDIH_CORR
    ENDIF

!      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND

    IF (PRNLEV.GE.7 .AND. EMODE.EQ.1) WRITE (OUTU,*) 'MSMMPT-EW > EW_SUM=', EW


    ETMP=EU !'TEST EMMPT2>'

!!CLOSE DOWN DIHEDRALS AND IMPROPERS TEMPORARILY
!!DON'T FORGET PUT 'DO J=1,HBRDNUM' TO THE EXTERIOR ITERATION
!OFFIF (1.EQ.0) THEN
!OFF!     DIHEDRALS
!OFF!      REMARKS ON CONSTRAINTS:
!OFF! ... UNCLEAR WHAT HAPPENS TO DIHEDRALS
!OFF
!OFF      DO I=1,DIHENUM
!OFF      CALL EPHIMSPT(DIHEMMPT(I,1),DIHEMMPT(I,2),               &
!OFF                   DIHEMMPT(I,3),DIHEMMPT(I,4),DIHEMMPT(I,5),  &
!OFF                   NPHI,ICP,CPD,CPCOS,CPSIN,CPC,               &
!OFF          ED,DXDI,DXDJ,DXDK,DXDL,                              &
!OFF             DYDI,DYDJ,DYDK,DYDL,                              &
!OFF             DZDI,DZDJ,DZDK,DZDL)
!OFF
!OFF
!OFF
!OFF
!OFF
!OFF!      THE TRANSFERED HYDROGEN ATOM IS EITHER FOURTH ATOM OR SECOND
!OFF!      ATOM IN DIHEDRAL DEFINITION AND ALWAYS SECOND IN HYDROGEN BOND
!OFF!      DEFINITION
!OFF
!OFF         DO J=1, HBRDNUM
!OFF            IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2) .OR.         &
!OFF               DIHEMMPT(I,2).EQ.HBRDATOM(J,2) ) THEN
!OFF              CALL SWITCHMSPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
!OFF                   HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
!OFF                   DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
!OFF               GOTO 110
!OFF!      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
!OFF            ENDIF
!OFF         ENDDO
!OFF
!OFF!      CHECK IF DIHEDRAL IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
!OFF!      TO SWITCHING FUNCTION
!OFF
!OFF 110     CONTINUE
!OFF
!OFF         IF (DIHEMMPT(I,6).GT.0) THEN
!OFF
!OFF            EU=EU-ED*SWF
!OFF
!OFF            IF(PRNLEV.GT.7) THEN
!OFF      WRITE(OUTU,*) 'MSMMPT> DIHEDRAL ENERGY REMOVED ACCORDING    &
!OFF     TO SWITCH',  ED*SWF
!OFF            ENDIF
!OFF
!OFF            DX2=DSWFRNH*DRNHX*ED+DSWFRNN*DRNNX*ED
!OFF            DX1=DSWFRNH*DRNNX*ED
!OFF
!OFF            DY2=DSWFRNH*DRNHY*ED+DSWFRNN*DRNNY*ED
!OFF            DY1=DSWFRNH*DRNNY*ED
!OFF
!OFF            DZ2=DSWFRNH*DRNHZ*ED+DSWFRNN*DRNNZ*ED
!OFF            DZ1=DSWFRNH*DRNNZ*ED
!OFF2,ATOM3
!OFF          IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
!OFF            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))-(DXDI*SWF)
!OFF            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))-(DXDJ*SWF)
!OFF            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))-(DXDK*SWF+DX1+DX2)
!OFF            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))-(DXDL*SWF-DX1)
!OFF            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
!OFF
!OFF            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))-(DYDI*SWF)
!OFF            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))-(DYDJ*SWF)
!OFF            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))-(DYDK*SWF+DY1+DY2)
!OFF            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))-(DYDL*SWF-DY1)
!OFF            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
!OFF
!OFF            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))-(DZDI*SWF)
!OFF            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))-(DZDJ*SWF)
!OFF            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))-(DZDK*SWF+DZ1+DZ2)
!OFF            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))-(DZDL*SWF-DZ1)
!OFF            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
!OFF          ENDIF
!OFF          IF (DIHEMMPT(I,2).EQ.HBRDATOM(J,2)) THEN
!OFF            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))-(DXDL*SWF)
!OFF            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))-(DXDK*SWF)
!OFF            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))-(DXDJ*SWF+DX1+DX2)
!OFF            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))-(DXDI*SWF-DX1)
!OFF            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
!OFF
!OFF            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))-(DYDL*SWF)
!OFF            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))-(DYDK*SWF)
!OFF            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))-(DYDJ*SWF+DY1+DY2)
!OFF            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))-(DYDI*SWF-DY1)
!OFF            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
!OFF
!OFF            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))-(DZDL*SWF)
!OFF            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))-(DZDK*SWF)
!OFF            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))-(DZDJ*SWF+DZ1+DZ2)
!OFF            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))-(DZDI*SWF-DZ1)
!OFF            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
!OFF          ENDIF
!OFF
!OFF
!OFF         ELSE
!OFF
!OFF            EU=EU+ED*SWF
!OFF
!OFF            IF(PRNLEV.GT.7) THEN
!OFF      WRITE(OUTU,*) 'MSMMPT> DIHEDRAL ENERGY ADDED ACCORDING  &
!OFF     TO SWITCH',ED*SWF
!OFF            ENDIF
!OFF
!OFF            DX2=DSWFRNH*DRNHX*ED+DSWFRNN*DRNNX*ED
!OFF            DX1=DSWFRNH*DRNNX*ED
!OFF
!OFF            DY2=DSWFRNH*DRNHY*ED+DSWFRNN*DRNNY*ED
!OFF            DY1=DSWFRNH*DRNNY*ED
!OFF
!OFF            DZ2=DSWFRNH*DRNHZ*ED+DSWFRNN*DRNNZ*ED
!OFF            DZ1=DSWFRNH*DRNNZ*ED
!OFF
!OFF
!OFF          IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
!OFF            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))+DXDI*SWF
!OFF            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))+DXDJ*SWF
!OFF            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))+DXDK*SWF-DX2
!OFF            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))+DXDL*SWF-DX1
!OFF            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
!OFF
!OFF            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))+DYDI*SWF
!OFF            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))+DYDJ*SWF
!OFF            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))+DYDK*SWF-DY2
!OFF            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))+DYDL*SWF-DY1
!OFF            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
!OFF
!OFF            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))+DZDI*SWF
!OFF            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))+DZDJ*SWF
!OFF            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))+DZDK*SWF-DZ2
!OFF            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))+DZDL*SWF-DZ1
!OFF            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
!OFF          ENDIF
!OFF          IF (DIHEMMPT(I,2).EQ.HBRDATOM(J,2)) THEN
!OFF            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))+DXDL*SWF
!OFF            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))+DXDK*SWF
!OFF            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))+DXDJ*SWF-DX2
!OFF            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))+DXDI*SWF-DX1
!OFF            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
!OFF
!OFF            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))+DYDL*SWF
!OFF            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))+DYDK*SWF
!OFF            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))+DYDJ*SWF-DY2
!OFF            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))+DYDI*SWF-DY1
!OFF            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
!OFF
!OFF            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))+DZDL*SWF
!OFF            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))+DZDK*SWF
!OFF            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))+DZDJ*SWF-DZ2
!OFF            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))+DZDI*SWF-DZ1
!OFF            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
!OFF          ENDIF
!OFF         ENDIF
!OFF      ENDDO
!OFF
!OFF
!OFF
!OFF
!OFF!      IMPROPER DIHEDRAL ANGLES
!OFF!      REMARKS ON CONSTRAINTS:
!OFF! ... UNCLEAR WHAT HAPPENS TO IMPROPER DIHEDRALS
!OFF
!OFF      DO I=1, IMPRNUM
!OFF      CALL EPHIMSPT(IMPRMMPT(I,1),IMPRMMPT(I,2),             &
!OFF                   IMPRMMPT(I,3),IMPRMMPT(I,4),IMPRMMPT(I,5),&
!OFF                   NPHI,ICP,CPD,CPCOS,CPSIN,CPC,             &
!OFF         EI ,DXII,DXIJ,DXIK,DXIL,                            &
!OFF             DYII,DYIJ,DYIK,DYIL,                            &
!OFF             DZII,DZIJ,DZIK,DZIL)
!OFF
!OFF!      ASSUMES THAT TRANSFERED HYDROGEN ATOM IS ALWAYS FOURTH
!OFF!      ATOM IN IMPROPER DEFINITION AND SECOND IN HYDROGEN BOND
!OFF!      DEFINITION
!OFF         DO J=1, HBRDNUM
!OFF            IF (IMPRMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
!OFF               CALL SWITCHMSPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
!OFF                   HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX,  &
!OFF                   DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
!OFF               GOTO 120
!OFF!      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
!OFF            ENDIF
!OFF         ENDDO
!OFF
!OFF!      CHECK IF IMPROPER IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
!OFF!      TO SWITCHING FUNCTION
!OFF
!OFF 120     IF (IMPRMMPT(I,6).GT.0) THEN
!OFF
!OFF            EU=EU-EI*SWF
!OFF
!OFF            IF(PRNLEV.GT.7) THEN
!OFF               WRITE(OUTU,*) 'MSMMPT> IMPROPER ENERGY REMOVED ACCORDING   &
!OFF     TO SWITCH',EI*SWF
!OFF            ENDIF
!OFF            DX1=DSWFRNH*DRNNX*EI
!OFF            DX2=DSWFRNN*DRNNX*EI+DSWFRNH*DRNHX*EI
!OFF
!OFF            DY1=DSWFRNH*DRNNY*EI
!OFF            DY2=DSWFRNN*DRNNY*EI+DSWFRNH*DRNHY*EI
!OFF
!OFF            DZ1=DSWFRNH*DRNNZ*EI
!OFF            DZ2=DSWFRNN*DRNNZ*EI+DSWFRNH*DRNHZ*EI
!OFF
!OFF            DX(IMPRMMPT(I,2))=DX(IMPRMMPT(I,2))-(DXII*SWF+DX1+DX2)
!OFF            DX(IMPRMMPT(I,3))=DX(IMPRMMPT(I,3))-(DXIJ*SWF)
!OFF            DX(IMPRMMPT(I,4))=DX(IMPRMMPT(I,4))-(DXIK*SWF)
!OFF            DX(IMPRMMPT(I,5))=DX(IMPRMMPT(I,5))-(DXIL*SWF-DX1)
!OFF            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
!OFF
!OFF            DY(IMPRMMPT(I,2))=DY(IMPRMMPT(I,2))-(DYII*SWF+DY1+DY2)
!OFF            DY(IMPRMMPT(I,3))=DY(IMPRMMPT(I,3))-(DYIJ*SWF)
!OFF            DY(IMPRMMPT(I,4))=DY(IMPRMMPT(I,4))-(DYIK*SWF)
!OFF            DY(IMPRMMPT(I,5))=DY(IMPRMMPT(I,5))-(DYIL*SWF-DY1)
!OFF            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
!OFF
!OFF            DZ(IMPRMMPT(I,2))=DZ(IMPRMMPT(I,2))-(DZII*SWF+DZ1+DZ2)
!OFF            DZ(IMPRMMPT(I,3))=DZ(IMPRMMPT(I,3))-(DZIJ*SWF)
!OFF            DZ(IMPRMMPT(I,4))=DZ(IMPRMMPT(I,4))-(DZIK*SWF)
!OFF            DZ(IMPRMMPT(I,5))=DZ(IMPRMMPT(I,5))-(DZIL*SWF-DZ1)
!OFF            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
!OFF
!OFF
!OFF
!OFF         ELSE
!OFF
!OFF            EU=EU+EI*SWF
!OFF
!OFF
!OFF            IF(PRNLEV.GT.7) THEN
!OFF               WRITE(OUTU,*) 'MSMMPT> IMPROPER ENERGY ADDED ACCORDING &
!OFF     TO SWITCH',EI*SWF
!OFF            ENDIF
!OFF
!OFF            DX1=DSWFRNH*DRNNX*EI
!OFF            DX2=DSWFRNN*DRNNX*EI+DSWFRNH*DRNHX*EI
!OFF
!OFF            DY1=DSWFRNH*DRNNY*EI
!OFF            DY2=DSWFRNN*DRNNY*EI+DSWFRNH*DRN2,ATOM3
!OFF
!OFF            DZ1=DSWFRNH*DRNNZ*EI
!OFF            DZ2=DSWFRNN*DRNNZ*EI+DSWFRNH*DRNHZ*EI
!OFF
!OFF            DX(IMPRMMPT(I,2))=DX(IMPRMMPT(I,2))+DXII*SWF-DX2
!OFF            DX(IMPRMMPT(I,3))=DX(IMPRMMPT(I,3))+DXIJ*SWF
!OFF            DX(IMPRMMPT(I,4))=DX(IMPRMMPT(I,4))+DXIK*SWF
!OFF            DX(IMPRMMPT(I,5))=DX(IMPRMMPT(I,5))+DXIL*SWF-DX1
!OFF            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
!OFF
!OFF            DY(IMPRMMPT(I,2))=DY(IMPRMMPT(I,2))+DYII*SWF-DY2
!OFF            DY(IMPRMMPT(I,3))=DY(IMPRMMPT(I,3))+DYIJ*SWF
!OFF            DY(IMPRMMPT(I,4))=DY(IMPRMMPT(I,4))+DYIK*SWF
!OFF            DY(IMPRMMPT(I,5))=DY(IMPRMMPT(I,5))+DYIL*SWF-DY1
!OFF            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
!OFF
!OFF            DZ(IMPRMMPT(I,2))=DZ(IMPRMMPT(I,2))+DZII*SWF-DZ2
!OFF            DZ(IMPRMMPT(I,3))=DZ(IMPRMMPT(I,3))+DZIJ*SWF
!OFF            DZ(IMPRMMPT(I,4))=DZ(IMPRMMPT(I,4))+DZIK*SWF
!OFF            DZ(IMPRMMPT(I,5))=DZ(IMPRMMPT(I,5))+DZIL*SWF-DZ1
!OFF            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
!OFF
!OFF
!OFF
!OFF         ENDIF
!OFF      ENDDO
!OFF
!OFF!!CLOSE DOWN DIHEDRALS AND IMPROPERS TEMPORARILY
!OFFENDIF

!      REMARKS ON CONSTRAINTS:
! ... UNCLEAR WHAT HAPPENS TO NONBONDED INTERACTION

!c'emmpt: change NONBNUM->NONBNUM(HBI);
!         NONBMMPT->NONBMMPT(HBI,NONBNUM(HBI),J)
    IF (EMODE.EQ.3 .OR. EMODE.EQ.2) THEN

      IF (MMPTPFLAG2) THEN
        DO I=1,NATOMX
          QFLAG(I)=0
        ENDDO
        QXTMP=0.d0
        QYTMP=0.d0
        QZTMP=0.d0
      ENDIF

      NBNUM_LOCAL=NONBNUM(HBI)
      CALL NBONDMOD(HBRDATOM(HBI,1),HBRDATOM(HBI,2),HBRDATOM(HBI,3), &
      NBNUM_LOCAL,HBI)

      RMIN_H_LOCAL=0.2245d0
      EPSI_H_LOCAL=-0.046d0

      DO I=1, NBNUM_LOCAL

  !        STANDARD OR SPECIAL VDW FORCES

        IF (ABS(NONBMMPT(HBI,I,4)) .NE. 14) THEN

          IF (.NOT.(NONBMMPT(HBI,I,4).GT.20 .AND. &
          EXNBFLAG_CG.EQ.0 .AND. EXNBFLAG_VDW.EQ.0)) THEN
            !using local subroutine for nonbond calculations
            IF (PRM_INTERNAL_IF_READ_NB.eq.0.d0) THEN
              CALL NBNDMMPT(NONBMMPT(HBI,I,1),NONBMMPT(HBI,I,2), &
              NONBMMPT(HBI,I,3),BNBND%INBLO,BNBND%JNB,MAXCN,  &
              DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM, &
              DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1, &
              DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2, &
              ENBMMPT,EELMMPT)

            ELSE
              RMIN_ATOMI=0.001d0
              EPS_ATOMI=0.d0
              IF (ATYPE(NONBMMPT(HBI,I,2))(1:1) .EQ. 'H') THEN
                RMIN_ATOMI=PRM_RMIN_H
                EPS_ATOMI=PRM_EPS_H
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,2))(1:1) .EQ. 'O') THEN
                RMIN_ATOMI=PRM_RMIN_O
                EPS_ATOMI=PRM_EPS_O
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,2))(1:1) .EQ. 'C') THEN
                RMIN_ATOMI=PRM_RMIN_Cl
                EPS_ATOMI=PRM_EPS_Cl
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,2))(1:1) .EQ. 'S') THEN
                RMIN_ATOMI=PRM_RMIN_Na
                EPS_ATOMI=PRM_EPS_Na
              ELSE
                WRITE(OUTU,*) 'MSMMPT-EMMPT2> Warning: NONBond-Atom-I  ', &
                ATYPE(NONBMMPT(HBI,I,3)),'not included'
              ENDIF

              RMIN_ATOMJ=0.001d0
              EPS_ATOMJ=0.d0
              IF (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'H') THEN
                RMIN_ATOMJ=PRM_RMIN_H
                EPS_ATOMJ=PRM_EPS_H
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'O') THEN
                RMIN_ATOMJ=PRM_RMIN_O
                EPS_ATOMJ=PRM_EPS_O
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'C') THEN
                RMIN_ATOMJ=PRM_RMIN_Cl
                EPS_ATOMJ=PRM_EPS_Cl
              ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'S') THEN
                RMIN_ATOMJ=PRM_RMIN_Na
                EPS_ATOMJ=PRM_EPS_Na
              ELSE
                WRITE(OUTU,*) 'MSMMPT-EMMPT2> Warning: NONBond-Atom-J  ', &
                ATYPE(NONBMMPT(HBI,I,3)),'not included'
              ENDIF

              IF(PRNLEV.GT.8) THEN
                WRITE(OUTU,*) ' '
                WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED-CALL EPSI(I,J) RMIN(I,J) ', &
                EPS_ATOMI,EPS_ATOMJ,RMIN_ATOMI,RMIN_ATOMJ
                WRITE(OUTU,*) PRM_EPS_O,PRM_EPS_H
                WRITE(OUTU,*) PRMMUL(18),PRMMUL(19),PRMMUL(20),PRMMUL(21)
              ENDIF
              CALL EPSEUDONONB(NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3), &
              SQRT(EPS_ATOMI*EPS_ATOMJ),(RMIN_ATOMI+RMIN_ATOMJ), &
              DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM, &
              DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1, &
              DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2, &
              ENBMMPT,EELMMPT)
            ENDIF

          ENDIF

  !write (outu,*) 'TEST-NBMODE>',NONBMMPT(HBI,I,4)
          IF (NONBMMPT(HBI,I,4).EQ.0) THEN
            !c'emmpt: -[D-A], only in mode 1
            IF (EMODE.EQ.3) THEN
              EU=EU-(EELMMPT+ENBMMPT)
              EELTOT=EELTOT-EELMMPT
              EVDWTOT=EVDWTOT-ENBMMPT

              DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-DXIRM
              DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-DYIRM
              DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-DZIRM

              DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-DXJRM
              DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-DYJRM
              DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-DZJRM
            ENDIF
          ELSE IF (ABS(NONBMMPT(HBI,I,4)).EQ.2) THEN
            !c'emmpt: -[H-A] in mode 1 & 2,
            IF (NONBMMPT(HBI,I,4).EQ.2) THEN

              EU=EU-(EELMMPT+ENBMMPT)
              EELTOT=EELTOT-EELMMPT
              EVDWTOT=EVDWTOT-ENBMMPT

              DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-DXIRM
              DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-DYIRM
              DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-DZIRM

              DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-DXJRM
              DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-DYJRM
              DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-DZJRM

            !c'emmpt: +[D-H] in mode2,
            ELSE
              IF (EMODE.EQ.3) THEN
                CONTINUE
              ELSE
                EU=EU+(EELMMPT+ENBMMPT)
                EELTOT=EELTOT+EELMMPT
                EVDWTOT=EVDWTOT+ENBMMPT

                DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+DXIRM
                DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+DYIRM
                DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+DZIRM

                DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+DXJRM
                DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+DYJRM
                DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+DZJRM
              ENDIF
            ENDIF
          !c'emmpt: H-A-X/X-D-H
          ELSE IF (ABS(NONBMMPT(HBI,I,4)).EQ.1) THEN
            IF ((NONBMMPT(HBI,I,2).NE.HBRDATOM(HBI,2)) .AND. &
            (NONBMMPT(HBI,I,3).NE.HBRDATOM(HBI,2))) THEN
              CALL WrnDie(-3,'<MISCOM>', &
                         'MSMMPT-MPT> Error: NOT H-X nonbonded interactions')
            ENDIF

  !            CALL SWITCHMSPT(HBRDATOM(HBI,1),HBRDATOM(HBI,2),  &
  !                 HBRDATOM(HBI,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
  !                 DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)

  !130         continue

            !c'emmpt: -E_elec[H-A-Y]-E_vdw[H-A-Y]*SWF in mode 1 & 2
            IF (NONBMMPT(HBI,I,4).EQ.1) THEN
  !            REMOVE 1-4 INTERACTION AND ADD SWITCHED ENERGIE AND FORCE, IE
  !            READ AS  -(EELMMPT+ENBMMPT)+(EELMMPT+ENBMMPT)*(1-SWF)
              !EU=EU-ENBMMPT*SWF-EELMMPT
              EU=EU-ENBMMPT-EELMMPT
              EELTOT=EELTOT-EELMMPT
              EVDWTOT=EVDWTOT-ENBMMPT

              !REMOVE DERIVATIVE OF VDW
              !DX1=DSWFRNH*DRNNX*(ENBMMPT)
              !DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(ENBMMPT)

              !DY1=DSWFRNH*DRNNY*(ENBMMPT)
              !DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(ENBMMPT)

              !DZ1=DSWFRNH*DRNNZ*(ENBMMPT)
              !DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(ENBMMPT)

              !DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-(DXIRM1*SWF)
              !DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-(DXJRM1*SWF-DX1)
              !DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2
              !DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-DX2-DX1

              !DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-(DYIRM1*SWF)
              !DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-(DYJRM1*SWF-DY1)
              !DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2
              !DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-DY2-DY1


              !DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-(DZIRM1*SWF)
              !DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-(DZJRM1*SWF-DZ1)
              !DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2
              !DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-DZ2-DZ1

              !REMOVE DERIVATIVE OF ELEC
              DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-(DXIRM)
              DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-(DXJRM)
              DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-(DYIRM)
              DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-(DYJRM)
              DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-(DZIRM)
              DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-(DZJRM)
            !c'emmpt: +[X-D-H] in mode 2, +E_vdw[X-D-H]*swf in mode 1
            ELSE
              IF (EMODE.EQ.2) THEN
                EU=EU+(EELMMPT+ENBMMPT)
                EELTOT=EELTOT+EELMMPT
                EVDWTOT=EVDWTOT+ENBMMPT

                DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+(DXIRM)
                DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+(DXJRM)
                DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+(DYIRM)
                DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+(DYJRM)
                DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+(DZIRM)
                DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+(DZJRM)

              ELSE
                !EU=EU+ENBMMPT*SWF
                !EVDWTOT=EVDWTOT+ENBMMPT
                continue

                !REMOVE DERIVATIVE OF VDW
              !  DX1=DSWFRNH*DRNNX*(ENBMMPT)
              !  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(ENBMMPT)

              !  DY1=DSWFRNH*DRNNY*(ENBMMPT)
              !  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(ENBMMPT)

              !  DZ1=DSWFRNH*DRNNZ*(ENBMMPT)
              !  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(ENBMMPT)

              !  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+(DXIRM1*SWF)
              !  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+(DXJRM1*SWF-DX1)
              !  DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
              !  DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1

              !  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+(DYIRM1*SWF)
              !  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+(DYJRM1*SWF-DY1)
              !  DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
              !  DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1

              !  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+(DZIRM1*SWF)
              !  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+(DZJRM1*SWF-DZ1)
              !  DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
              !  DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1
              ENDIF
            ENDIF
          !c'emmpt: D-H-A-XA/XD-D-H-A/XD-D-H-A-XA, -E_elec[] IN MODE 1
          ELSE IF (NONBMMPT(HBI,I,4).EQ.3 .AND. EMODE.EQ.3) THEN
            EU=EU-ENBMMPT-EELMMPT
            EELTOT=EELTOT-EELMMPT
            EVDWTOT=EVDWTOT-ENBMMPT

            DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-(DXIRM)
            DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-(DXJRM)
            DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-(DYIRM)
            DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-(DYJRM)
            DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-(DZIRM)
            DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-(DZJRM)
          !c'emmpt: inter-molecular interaction between w2p and environments
          ELSE IF (NONBMMPT(HBI,I,4).GT.20 .AND. EMODE.EQ.3) THEN

            IF (EXNBFLAG_CG.EQ.1) THEN
  !write(outu,*) 'TEST-hey there> ',CCP
              !charge transfer model
              IF (CCP.GT.-1) THEN
                CGE=EXP(CCB)
                !! equi-partition the charges
                CCP_TMP=3.d0-(CGO+2.d0)/CGE

  !write(outu,*) 'TEST-hey there> ',CCP,NONBMMPT(HBI,I,4)
                !D-env
                IF (NONBMMPT(HBI,I,4).EQ.21) THEN
                  EU=EU+CCP_TMP*CGE/CGO*(1.D0-SWF)*EELMMPT
                  EELTOT=EELTOT+CCP_TMP*CGE/CGO*(1.D0-SWF)*EELMMPT
  !write(outu,*) 'TEST-EEL> ',CCP*CGE/CGO*(1.D0-SWF)*EELMMPT
                  !calculating the point charge
                  IF (MMPTPFLAG2 .AND. QFLAG(NONBMMPT(HBI,I,2)).EQ.0) THEN
                    QFLAG(NONBMMPT(HBI,I,2))=1
                    XTMP(1)=ABS(CGO+CCP_TMP*CGE)*(1.D0-SWF)

                    CALL IMGOFFSET(HBRDATOM(HBI,2),NONBMMPT(HBI,I,2),RX,RY,RZ)
                    QXTMP=QXTMP+XTMP(1)*(X(HBRDATOM(HBI,2))+RX)
                    QYTMP=QYTMP+XTMP(1)*(Y(HBRDATOM(HBI,2))+RY)
                    QZTMP=QZTMP+XTMP(1)*(Z(HBRDATOM(HBI,2))+RZ)
                  ENDIF

                  SWFTMP=CCP_TMP*CGE/CGO*(1.D0-SWF)
                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+DXIRM2*SWFTMP
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+DXJRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+DYIRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+DYJRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+DZIRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+DZJRM2*SWFTMP


                  SWFTMP=(-CCP_TMP*CGE/CGO*EELMMPT)
                  DX1=DSWFRNH*DRNNX*SWFTMP
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*SWFTMP
                  DY1=DSWFRNH*DRNNY*SWFTMP
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*SWFTMP
                  DZ1=DSWFRNH*DRNNZ*SWFTMP
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*SWFTMP

                  DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1
                  DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                  DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
                  DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1
                  DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                  DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
                  DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1
                  DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                  DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                !H-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.22) THEN
                  EU=EU-EELMMPT+(1.D0-CGE)/CGH*EELMMPT
                  EELTOT=EELTOT-EELMMPT+(1.D0-CGE)/CGH*EELMMPT

                  !calculating the point charge
                  IF (MMPTPFLAG2 .AND. QFLAG(NONBMMPT(HBI,I,2)).EQ.0) THEN
                    QFLAG(NONBMMPT(HBI,I,2))=1
                    XTMP(1)=ABS(1.D0-CGE)

                    CALL IMGOFFSET(HBRDATOM(HBI,2),NONBMMPT(HBI,I,2),RX,RY,RZ)
                    QXTMP=QXTMP+XTMP(1)*(X(HBRDATOM(HBI,2))+RX)
                    QYTMP=QYTMP+XTMP(1)*(Y(HBRDATOM(HBI,2))+RY)
                    QZTMP=QZTMP+XTMP(1)*(Z(HBRDATOM(HBI,2))+RZ)
                  ENDIF

                  SWFTMP=(1.D0-(1.D0-CGE)/CGH)
                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-DXIRM2*SWFTMP
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-DXJRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-DYIRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-DYJRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-DZIRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-DZJRM2*SWFTMP

                !A-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.23) THEN
                  EU=EU+CCP_TMP*CGE/CGO*SWF*EELMMPT
                  EELTOT=EELTOT+CCP_TMP*CGE/CGO*SWF*EELMMPT

                  !calculating the point charge
                  IF (MMPTPFLAG2 .AND. QFLAG(NONBMMPT(HBI,I,2)).EQ.0) THEN
                    QFLAG(NONBMMPT(HBI,I,2))=1
                    XTMP(1)=ABS(CGO+CCP_TMP*CGE)*SWF

                    CALL IMGOFFSET(HBRDATOM(HBI,2),NONBMMPT(HBI,I,2),RX,RY,RZ)
                    QXTMP=QXTMP+XTMP(1)*(X(HBRDATOM(HBI,2))+RX)
                    QYTMP=QYTMP+XTMP(1)*(Y(HBRDATOM(HBI,2))+RY)
                    QZTMP=QZTMP+XTMP(1)*(Z(HBRDATOM(HBI,2))+RZ)
                  ENDIF

                  SWFTMP=CCP_TMP*CGE/CGO*SWF
                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                  DXIRM2*SWFTMP
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                  DXJRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                  DYIRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                  DYJRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                  DZIRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                  DZJRM2*SWFTMP

                  DX1=DSWFRNH*DRNNX*(CCP_TMP*CGE/CGO*EELMMPT)
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(CCP_TMP*CGE/CGO*EELMMPT)
                  DY1=DSWFRNH*DRNNY*(CCP_TMP*CGE/CGO*EELMMPT)
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(CCP_TMP*CGE/CGO*EELMMPT)
                  DZ1=DSWFRNH*DRNNZ*(CCP_TMP*CGE/CGO*EELMMPT)
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(CCP_TMP*CGE/CGO*EELMMPT)

                  DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1
                  DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                  DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
                  DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1
                  DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                  DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
                  DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1
                  DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                  DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                !XD-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.24) THEN
                  EU=EU+0.5D0*(1.D0-CCP_TMP)*CGE/CGH*(1.D0-SWF)*EELMMPT
                  EELTOT=EELTOT+0.5D0*(1.D0-CCP_TMP)*CGE/CGH*(1.D0-SWF)*EELMMPT

                  !calculating the point charge
                  IF (MMPTPFLAG2 .AND. QFLAG(NONBMMPT(HBI,I,2)).EQ.0) THEN
                    QFLAG(NONBMMPT(HBI,I,2))=1
                    XTMP(1)=ABS(CGH+0.5D0*(1.D0-CCP_TMP)*CGE)*(1.D0-SWF)

                    CALL IMGOFFSET(HBRDATOM(HBI,2),NONBMMPT(HBI,I,2),RX,RY,RZ)
                    QXTMP=QXTMP+XTMP(1)*(X(HBRDATOM(HBI,2))+RX)
                    QYTMP=QYTMP+XTMP(1)*(Y(HBRDATOM(HBI,2))+RY)
                    QZTMP=QZTMP+XTMP(1)*(Z(HBRDATOM(HBI,2))+RZ)
                  ENDIF

                  SWFTMP=0.5D0*(1.D0-CCP_TMP)*CGE/CGH*(1.D0-SWF)
                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                  DXIRM2*SWFTMP
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                  DXJRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                  DYIRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                  DYJRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                  DZIRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                  DZJRM2*SWFTMP

                  SWFTMP=(-0.5D0*(1.D0-CCP_TMP)*CGE/CGH*EELMMPT)
                  DX1=DSWFRNH*DRNNX*SWFTMP
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*SWFTMP
                  DY1=DSWFRNH*DRNNY*SWFTMP
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*SWFTMP
                  DZ1=DSWFRNH*DRNNZ*SWFTMP
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*SWFTMP

                  DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1
                  DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                  DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
                  DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1
                  DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                  DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
                  DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1
                  DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                  DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                !XA-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.25) THEN
                  EU=EU+0.5D0*(1.D0-CCP_TMP)*CGE/CGH*SWF*EELMMPT
                  EELTOT=EELTOT+0.5D0*(1.D0-CCP_TMP)*CGE/CGH*SWF*EELMMPT

                  !calculating the point charge
                  IF (MMPTPFLAG2 .AND. QFLAG(NONBMMPT(HBI,I,2)).EQ.0) THEN
                    QFLAG(NONBMMPT(HBI,I,2))=1
                    XTMP(1)=ABS(CGH+0.5D0*(1.D0-CCP_TMP)*CGE)*SWF

                    CALL IMGOFFSET(HBRDATOM(HBI,2),NONBMMPT(HBI,I,2),RX,RY,RZ)
                    QXTMP=QXTMP+XTMP(1)*(X(NONBMMPT(HBI,I,2))+RX)
                    QYTMP=QYTMP+XTMP(1)*(Y(NONBMMPT(HBI,I,2))+RY)
                    QZTMP=QZTMP+XTMP(1)*(Z(NONBMMPT(HBI,I,2))+RZ)
                  ENDIF

                  SWFTMP=0.5D0*(1.D0-CCP_TMP)*CGE/CGH*SWF
                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                  DXIRM2*SWFTMP
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                  DXJRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                  DYIRM2*SWFTMP
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                  DYJRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                  DZIRM2*SWFTMP
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                  DZJRM2*SWFTMP

                  SWFTMP=(0.5D0*(1.D0-CCP_TMP)*CGE/CGH*EELMMPT)
                  DX1=DSWFRNH*DRNNX*SWFTMP
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*SWFTMP
                  DY1=DSWFRNH*DRNNY*SWFTMP
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*SWFTMP
                  DZ1=DSWFRNH*DRNNZ*SWFTMP
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*SWFTMP

                  DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1
                  DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                  DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
                  DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1
                  DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                  DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
                  DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1
                  DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                  DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2

                ENDIF
              !fixed point charges, net charge is +1
              ELSE

                !D-env; A-env
                IF (NONBMMPT(HBI,I,4).EQ.21 .OR. NONBMMPT(HBI,I,4).EQ.23) THEN
                  EU=EU+(CCA-CGO)/CGO*EELMMPT
                  EELTOT=EELTOT+(CCA-CGO)/CGO*EELMMPT

                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+DXIRM2*(CCA-CGO)/CGO
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+DXJRM2*(CCA-CGO)/CGO
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+DYIRM2*(CCA-CGO)/CGO
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+DYJRM2*(CCA-CGO)/CGO
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+DZIRM2*(CCA-CGO)/CGO
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+DZJRM2*(CCA-CGO)/CGO

                !H-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.22) THEN
                  EU=EU+(CCB-CGH)/CGH*EELMMPT
                  EELTOT=EELTOT+(CCB-CGH)/CGH*EELMMPT

                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+DXIRM2*(CCB-CGH)/CGH
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+DXJRM2*(CCB-CGH)/CGH
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+DYIRM2*(CCB-CGH)/CGH
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+DYJRM2*(CCB-CGH)/CGH
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+DZIRM2*(CCB-CGH)/CGH
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+DZJRM2*(CCB-CGH)/CGH

                !XD-env; XA-env
                ELSE IF (NONBMMPT(HBI,I,4).EQ.24 .OR. NONBMMPT(HBI,I,4).EQ.25) THEN
                  EU=EU+(CCC-CGH)/CGH*EELMMPT
                  EELTOT=EELTOT+(CCC-CGH)/CGH*EELMMPT

                  DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+DXIRM2*(CCC-CGH)/CGH
                  DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+DXJRM2*(CCC-CGH)/CGH
                  DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+DYIRM2*(CCC-CGH)/CGH
                  DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+DYJRM2*(CCC-CGH)/CGH
                  DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+DZIRM2*(CCC-CGH)/CGH
                  DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+DZJRM2*(CCC-CGH)/CGH
                ENDIF
              ENDIF
            ENDIF

            !external vdw
            !!water only
            ENBMMPT3=0.d0
            VDW_FLAG=0

            RMIN_ATOMJ=0.001d0
            EPS_ATOMJ=0.d0
            IF (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'H') THEN
              RMIN_ATOMJ=PRM_RMIN_H
              EPS_ATOMJ=PRM_EPS_H
            ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'O') THEN
              RMIN_ATOMJ=PRM_RMIN_O
              EPS_ATOMJ=PRM_EPS_O
            ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'C') THEN
              RMIN_ATOMJ=PRM_RMIN_Cl
              EPS_ATOMJ=PRM_EPS_Cl
            ELSE IF  (ATYPE(NONBMMPT(HBI,I,3))(1:1) .EQ. 'S') THEN
              RMIN_ATOMJ=PRM_RMIN_Na
              EPS_ATOMJ=PRM_EPS_Na
            ELSE
              WRITE(OUTU,*) 'MSMMPT-EMMPT2> Warning: NONBond-Atom-J  ', &
              ATYPE(NONBMMPT(HBI,I,3)),'not included'
            ENDIF

            IF (EXNBFLAG_VDW.EQ.1) THEN
              !shift vdw parameters of oxygen
              !parallel if operation *3
              IF (NONBMMPT(HBI,I,4).EQ.21 .OR. NONBMMPT(HBI,I,4).EQ.23) THEN
                IF (RMIN_O .GT. -0.0000001D0) THEN
                  VDW_FLAG=1
                  CALL EPSEUDOVDW(NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3),ENBMMPT3, &
                  SQRT(EPSI_O*EPS_ATOMJ),(RMIN_O+RMIN_ATOMJ), &
                  DXIRM3,DYIRM3,DZIRM3,DXJRM3,DYJRM3,DZJRM3)
                ENDIF
              !!only applied with SPC water model, give H* a TIP3P parameter
              ELSE IF (NONBMMPT(HBI,I,4).EQ.22 .AND. 1.EQ.1) THEN
                VDW_FLAG=1
                CALL EPSEUDOVDW(NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3),ENBMMPT3, &
                SQRT(EPS_ATOMJ*EPSI_H_LOCAL),(RMIN_ATOMJ+RMIN_H_LOCAL), &
                DXIRM3,DYIRM3,DZIRM3,DXJRM3,DYJRM3,DZJRM3)
              !shift vdw parameters of hydrogen
              ELSE
                IF (RMIN_H .GT. -0.0000001D0) THEN
                  VDW_FLAG=1
                  CALL EPSEUDOVDW(NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3),ENBMMPT3, &
                  SQRT(EPS_ATOMJ*EPSI_H),(RMIN_ATOMJ+RMIN_H), &
                  DXIRM3,DYIRM3,DZIRM3,DXJRM3,DYJRM3,DZJRM3)
                ENDIF
              ENDIF
            ELSE
              IF (NONBMMPT(HBI,I,4).EQ.22 .AND. 1.EQ.1) THEN
                VDW_FLAG=1
                CALL EPSEUDOVDW(NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3),ENBMMPT3, &
                SQRT(EPS_ATOMJ*EPSI_H_LOCAL),(RMIN_ATOMJ+RMIN_H_LOCAL), &
                DXIRM3,DYIRM3,DZIRM3,DXJRM3,DYJRM3,DZJRM3)
              ENDIF
            ENDIF
              !shift vdw parameters of hydrogen


            IF (VDW_FLAG.EQ.1 .AND. (ENBMMPT3.NE.0.D0 .OR. ENBMMPT.NE.0.D0)) THEN
              IF (NONBMMPT(HBI,I,4).EQ.21 .OR. NONBMMPT(HBI,I,4).EQ.24) THEN
                EU=EU+(1.D0-SWF)*(ENBMMPT3-ENBMMPT)
                EVDWTOT=EVDWTOT+(1.D0-SWF)*(ENBMMPT3-ENBMMPT)

                DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                (DXIRM3-DXIRM1)*(1.D0-SWF)
                DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                (DXJRM3-DXJRM1)*(1.D0-SWF)
                DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                (DYIRM3-DYIRM1)*(1.D0-SWF)
                DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                (DYJRM3-DYJRM1)*(1.D0-SWF)
                DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                (DZIRM3-DZIRM1)*(1.D0-SWF)
                DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                (DZJRM3-DZJRM1)*(1.D0-SWF)

                DX1=DSWFRNH*DRNNX*(ENBMMPT3-ENBMMPT)
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(ENBMMPT3-ENBMMPT)
                DY1=DSWFRNH*DRNNY*(ENBMMPT3-ENBMMPT)
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(ENBMMPT3-ENBMMPT)
                DZ1=DSWFRNH*DRNNZ*(ENBMMPT3-ENBMMPT)
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(ENBMMPT3-ENBMMPT)

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))+DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))+DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))+DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2
              ELSE IF (NONBMMPT(HBI,I,4).EQ.23 .OR. NONBMMPT(HBI,I,4).EQ.25) THEN
                EU=EU+SWF*(ENBMMPT3-ENBMMPT)
                EVDWTOT=EVDWTOT+SWF*(ENBMMPT3-ENBMMPT)

                DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                (DXIRM3-DXIRM1)*SWF
                DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                (DXJRM3-DXJRM1)*SWF
                DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                (DYIRM3-DYIRM1)*SWF
                DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                (DYJRM3-DYJRM1)*SWF
                DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                (DZIRM3-DZIRM1)*SWF
                DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                (DZJRM3-DZJRM1)*SWF

                DX1=DSWFRNH*DRNNX*(ENBMMPT3-ENBMMPT)
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(ENBMMPT3-ENBMMPT)
                DY1=DSWFRNH*DRNNY*(ENBMMPT3-ENBMMPT)
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(ENBMMPT3-ENBMMPT)
                DZ1=DSWFRNH*DRNNZ*(ENBMMPT3-ENBMMPT)
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(ENBMMPT3-ENBMMPT)

                DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+(DX1+DX2)
                DX(HBRDATOM(HBI,2))=DX(HBRDATOM(HBI,2))-DX1
                DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2

                DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+(DY1+DY2)
                DY(HBRDATOM(HBI,2))=DY(HBRDATOM(HBI,2))-DY1
                DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2

                DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+(DZ1+DZ2)
                DZ(HBRDATOM(HBI,2))=DZ(HBRDATOM(HBI,2))-DZ1
                DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
              ELSE IF (NONBMMPT(HBI,I,4).EQ.22 .and. 1.eq.1) THEN !!switch on
                EU=EU+(ENBMMPT3-ENBMMPT)
                EVDWTOT=EVDWTOT+(ENBMMPT3-ENBMMPT)

                DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+ &
                (DXIRM3-DXIRM1)
                DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+ &
                (DXJRM3-DXJRM1)
                DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+ &
                (DYIRM3-DYIRM1)
                DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+ &
                (DYJRM3-DYJRM1)
                DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+ &
                (DZIRM3-DZIRM1)
                DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+ &
                (DZJRM3-DZJRM1)
              ENDIF
            ENDIF

          ENDIF


  !!nothing to do for 1-4 VDW
        ELSE
          CALL WrnDie(-3,'<MISCOM>', &
          'MSMMPT-MPT> Error: Currently only for H2O systems. No 1-4 interactions')

          CALL EVDW14MSPT(NONBMMPT(HBI,I,1),NONBMMPT(HBI,I,2), &
          NONBMMPT(HBI,I,3),NONBMMPT(HBI,I,4),BNBND%INBLO,BNBND%JNB, &
          MAXCN,DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1,EVDW1MMPT)


          CALL EVDW14MSPT(NONBMMPT(HBI,I,1),NONBMMPT(HBI,I,2), &
          NONBMMPT(HBI,I,3),-(NONBMMPT(HBI,I,4)),BNBND%INBLO, &
          BNBND%JNB,MAXCN,DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2, &
          EVDW2MMPT)

  !          DO J=1, HBRDNUM
  !             IF (NONBMMPT(I,2).EQ.HBRDATOM(J,2)                       &
  !                 .OR. NONBMMPT(I,3).EQ.HBRDATOM(J,2)) THEN
  !                CALL SWITCHMSPT(HBRDATOM(J,1),HBRDATOM(J,2),          &
  !                     HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX,         &
  !                     DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
  !                 GOTO 140
  !                     STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
  !             ENDIF
  !          ENDDO


  !     .            REMOVE 1-4 INTERACTION AND ADD SWITCHED ENERGIE AND FORCE, IE
  !     .            READ AS  -(EVD1WMMPT)+(EVDW1MMPT)*(1-SWF)
  ! 140      continue
          EU=EU-(EVDW1MMPT)*SWF
          EVDWTOT=EVDWTOT-(EVDW1MMPT)*SWF

          DX1=DSWFRNH*DRNNX*(EVDW1MMPT)
          DX2=DSWFRNN*DRNNX*(EVDW1MMPT)+DSWFRNH*DRNHX*(EVDW1MMPT)

          DY1=DSWFRNH*DRNNY*(EVDW1MMPT)
          DY2=DSWFRNN*DRNNY*(EVDW1MMPT)+DSWFRNH*DRNHY*(EVDW1MMPT)

          DZ1=DSWFRNH*DRNNZ*(EVDW1MMPT)
          DZ2=DSWFRNN*DRNNZ*(EVDW1MMPT)+DSWFRNH*DRNHZ*(EVDW1MMPT)


          DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))-(DXIRM1*SWF)
          DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))-(DXJRM1*SWF-DX1)
          DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))+DX2
          DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))-DX2-DX1

          DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))-(DYIRM1*SWF)
          DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))-(DYJRM1*SWF-DY1)
          DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))+DY2
          DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))-DY2-DY1


          DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))-(DZIRM1*SWF)
          DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))-(DZJRM1*SWF-DZ1)
          DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))+DZ2
          DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))-DZ2-DZ1

          IF(PRNLEV.GT.8) THEN
            WRITE(OUTU,*) 'MSMMPT> VDW1 ENERGY ACCORDING TO SWITCH',  &
              (EVDW1MMPT)*(1.D0-SWF)
          ENDIF

  !                 ADD STANDARD VDW FORCE

          EU=EU+(EVDW2MMPT)*SWF
          EVDWTOT=EVDWTOT+(EVDW2MMPT)*SWF

          DX1=DSWFRNH*DRNNX*(EVDW2MMPT)
          DX2=DSWFRNN*DRNNX*(EVDW2MMPT)+DSWFRNH*DRNHX*(EVDW2MMPT)

          DY1=DSWFRNH*DRNNY*(EVDW2MMPT)
          DY2=DSWFRNN*DRNNY*(EVDW2MMPT)+DSWFRNH*DRNHY*(EVDW2MMPT)

          DZ1=DSWFRNH*DRNNZ*(EVDW2MMPT)
          DZ2=DSWFRNN*DRNNZ*(EVDW2MMPT)+DSWFRNH*DRNHZ*(EVDW2MMPT)


          DX(NONBMMPT(HBI,I,2))=DX(NONBMMPT(HBI,I,2))+(DXIRM2*SWF)
          DX(NONBMMPT(HBI,I,3))=DX(NONBMMPT(HBI,I,3))+(DXJRM2*SWF-DX1)
          DX(HBRDATOM(HBI,3))=DX(HBRDATOM(HBI,3))-DX2
          DX(HBRDATOM(HBI,1))=DX(HBRDATOM(HBI,1))+DX2+DX1

          DY(NONBMMPT(HBI,I,2))=DY(NONBMMPT(HBI,I,2))+(DYIRM2*SWF)
          DY(NONBMMPT(HBI,I,3))=DY(NONBMMPT(HBI,I,3))+(DYJRM2*SWF-DY1)
          DY(HBRDATOM(HBI,3))=DY(HBRDATOM(HBI,3))-DY2
          DY(HBRDATOM(HBI,1))=DY(HBRDATOM(HBI,1))+DY2+DY1


          DZ(NONBMMPT(HBI,I,2))=DZ(NONBMMPT(HBI,I,2))+(DZIRM2*SWF)
          DZ(NONBMMPT(HBI,I,3))=DZ(NONBMMPT(HBI,I,3))+(DZJRM2*SWF-DZ1)
          DZ(HBRDATOM(HBI,3))=DZ(HBRDATOM(HBI,3))-DZ2
          DZ(HBRDATOM(HBI,1))=DZ(HBRDATOM(HBI,1))+DZ2+DZ1

          IF(PRNLEV.GT.8) THEN
            WRITE(OUTU,*) 'MSMMPT> VDW2 ENERGY ACCORDING TO SWITCH', &
                 (EVDW2MMPT)*SWF
          ENDIF

        ENDIF
        IF (PRNLEV.GE.9 .and. (NONBMMPT(HBI,I,4).lt.20 .or. &
        EXNBFLAG_CG.EQ.1 .or. EXNBFLAG_VDW.EQ.1)) THEN
          WRITE (OUTU,*) 'MSMMPT-EMMPT2> E_NB=',EU-ETMP, &
          NONBMMPT(HBI,I,1),NONBMMPT(HBI,I,2),NONBMMPT(HBI,I,3), &
          NONBMMPT(HBI,I,4)
          ETMP=EU
        ENDIF
      ENDDO
    ENDIF


    IF (MMPTPFLAG2) THEN
      XTMP(1)=ABS(1.D0-CGE)+ABS(CGO+CCP_TMP*CGE)+ &
      2.D0*ABS(CGH+0.5D0*(1.D0-CCP_TMP)*CGE)

      DX(NATOMX+1)=QXTMP/XTMP(1)
      DY(NATOMX+1)=QYTMP/XTMP(1)
      DZ(NATOMX+1)=QZTMP/XTMP(1)
    ENDIF

    IF(EXNBFLAG_CG.EQ.1 .and. CCP.GT.-1 .AND. EMODE.EQ.3) THEN
      IF(PRNLEV.GT.7) THEN
        WRITE(OUTU,*) 'MSMMPT-EMMPT2> FLUCTUATING CHARGES [D,H,A,HD,HA]: '
        WRITE(OUTU,*) CGO+CCP_TMP*CGE*(1.D0-SWF) ,1.D0-CGE ,&
        CGO+CCP_TMP*CGE*SWF ,&
        CGH+0.5D0*(1.D0-CCP_TMP)*CGE*(1.D0-SWF) ,&
        CGH+0.5D0*(1.D0-CCP_TMP)*CGE*SWF
      ENDIF
    ENDIF

    IF (PRNLEV.GE.7) THEN
      IF (EMODE.GE.2) THEN
        WRITE(OUTU,*) 'MSMMPT-EMMPT2> EELTOTAL:',EELTOT
        WRITE(OUTU,*) 'MSMMPT-EMMPT2> EVDWTOTAL:',EVDWTOT
      ENDIF

      WRITE(OUTU,*) 'MSMMPT-EMMPT2> EU:',EU

      IF (EMODE.EQ.1) WRITE(OUTU,*) 'MSMMPT-EMMPT2> EW:',EW
      IF (PRNLEV.GE.10) THEN
        DO I=1,NATOMX
          WRITE(OUTU,*) 'MSMMPT-EMMPT2> WDXYZ: ',WDX(I),WDY(I),WDZ(I)
        ENDDO
      ENDIF
    ENDIF


  END SUBROUTINE EMMPT2

  SUBROUTINE PRINTMMPT()
    use stream
    use string

    INTEGER I,J,H

    WRITE(OUTU,*) 'MSMMPTBOND LIST OF DH-A MOTIFS'

    WRITE(OUTU,*) 'BONDMMPT:'
    DO I=1,HBRDNUM
      DO J=1,BONDNUM(I)
        WRITE (OUTU,6010) '(',BONDMMPT(I,J,1),')-[',BONDMMPT(I,J,2),'-', &
        BONDMMPT(I,J,3),']-(',BONDMMPT(I,J,4),')'
      ENDDO
    ENDDO

    WRITE(OUTU,*) 'ANGLMMPT:'
    DO H=1,HBRDNUM
      DO I=1,ANGLNUM(H)
        WRITE (OUTU,6011) '(',ANGLMMPT(H,I,1),')-[',ANGLMMPT(H,I,2),'-', &
        ANGLMMPT(H,I,3),'-',ANGLMMPT(H,I,4),']-(',ANGLMMPT(H,I,5),')'
      ENDDO
    ENDDO

    WRITE(OUTU,*) 'NONBMMPT:'
    DO I=1,HBRDNUM
      DO J=1,NONBNUM(I)
        WRITE (OUTU,6012) '(',NONBMMPT(I,J,1),')-[',NONBMMPT(I,J,2), &
        '-',NONBMMPT(I,J,3),']-(',NONBMMPT(I,J,4),')'
      ENDDO
    ENDDO

6010 FORMAT(A1,I4,A3,I4,A1,I4,A3,I4,A1)
6011 FORMAT(A1,I4,A3,I4,A1,I4,A1,I4,A3,I4,A1)
6012 FORMAT(A1,I4,A3,I4,A1,I4,A3,I4,A1)

  END SUBROUTINE PRINTMMPT

  SUBROUTINE PRINTSWAP()
    use bases_fcm
    use block_fcm
    use chm_types
    use cnst_fcm
    use code
    use comand
    use consta
    use contrl
    use dimens_fcm
    use inbnd
    use number
    use param
    use psf
    use stream
    use string

    INTEGER I,J

    WRITE(OUTU,*) 'Non- & BONDED LIST'

    WRITE(OUTU,*) 'Bond:'
    DO I=1,NBOND
      WRITE (OUTU,6060) 'B(',I,'): ',IB(I),JB(I)
    ENDDO

    WRITE(OUTU,*) 'Angular bond:'
    DO I=1,NTHETA
      WRITE (OUTU,6061) 'T(',I,'): ',IT(I),JT(I),KT(I)
    ENDDO

    WRITE(OUTU,*) 'Non-bond:'
    !DO I=1,HBRDNUM
    !  DO J=1,NONBNUM(I)
    !    WRITE (OUTU,5987) '(',NONBMMPT(I,J,1),')-[',NONBMMPT(I,J,2), &
    !    '-',NONBMMPT(I,J,3),']-(',NONBMMPT(I,J,4),')'
    !  ENDDO
    !ENDDO

6060 FORMAT(A4,I6,A3,I6,I6)
6061 FORMAT(A4,I6,A3,I6,I6,I6)

  END SUBROUTINE PRINTSWAP

  SUBROUTINE MTFSWITCH(HBI,DA2,INBLO,JNB,SMODE)
!SMODE=1 REQUIRES CONVERSION OF NONBMMPT WITH NEGATIVE FLAGS INTO
!REGULAR NONBOND LIST
!ONLY applies when a swap is called but nonbond is not yet updated

    use block_fcm
    use cnst_fcm
    use code
    use comand
    use consta
    use coord
    use dimens_fcm
    use image
    use inbnd
    use number
    use param
    use psf
    use stream
    use string


    INTEGER DA1(3),DA2(3),HBI,I,J,H, &
            ITH,ITMP,NONB2H(NATOM)
    INTEGER JSTART,JEND,INBLO(*),JNB(*)
    INTEGER BKP1(NONBNUM(HBI)),BKP2(NONBNUM(HBI)),BKPNUM,SMODE

    DA1(1)=HBRDATOM(HBI,1)
    DA1(2)=HBRDATOM(HBI,2)
    DA1(3)=HBRDATOM(HBI,3)

    IF (DA1(1).EQ.DA2(1) .AND. DA1(2).EQ.DA2(2) &
    .AND. DA1(3).EQ.DA2(3)) return

    !!new bonding parameters need to be specified if changed
    !update new D-H bond

    BONDNUM(HBI)=0
    DO I=1,NBOND
      IF ((IB(I).EQ.DA2(1) .AND. JB(I).EQ.DA2(2)) .OR. &
      (JB(I).EQ.DA2(1) .AND. IB(I).EQ.DA2(2))) THEN
        BONDNUM(HBI)=BONDNUM(HBI)+1
        IF (BONDNUM(HBI).GT.1) THEN
          BONDMMPT(HBI,BONDNUM(HBI),1)=BONDMMPT(HBI,1,1)
          BONDMMPT(HBI,BONDNUM(HBI),2)=BONDMMPT(HBI,1,2)
          BONDMMPT(HBI,BONDNUM(HBI),3)=BONDMMPT(HBI,1,3)
          BONDMMPT(HBI,BONDNUM(HBI),4)=BONDMMPT(HBI,1,4)
        ENDIF
        BONDMMPT(HBI,1,1)=I
        BONDMMPT(HBI,1,2)=DA2(1)
        BONDMMPT(HBI,1,3)=DA2(2)
        BONDMMPT(HBI,1,4)=1

        GOTO 7165
      ENDIF

      IF ((IB(I).EQ.DA2(1)) .OR. (JB(I).EQ.DA2(1))) THEN
        BONDNUM(HBI)=BONDNUM(HBI)+1
        BONDMMPT(HBI,BONDNUM(HBI),1)=I
        BONDMMPT(HBI,BONDNUM(HBI),2)=IB(I)
        BONDMMPT(HBI,BONDNUM(HBI),3)=JB(I)
        BONDMMPT(HBI,BONDNUM(HBI),4)=2

        GOTO 7165
      ENDIF

      IF ((IB(I).EQ.DA2(3)) .OR. (JB(I).EQ.DA2(3))) THEN
        BONDNUM(HBI)=BONDNUM(HBI)+1
        BONDMMPT(HBI,BONDNUM(HBI),1)=I
        BONDMMPT(HBI,BONDNUM(HBI),2)=IB(I)
        BONDMMPT(HBI,BONDNUM(HBI),3)=JB(I)
        BONDMMPT(HBI,BONDNUM(HBI),4)=-2

        GOTO 7165
      ENDIF

7165 CONTINUE
    ENDDO

    !update new X-DH HA-Y angular bonds
    !DO I=1,ANGLNUM(HBI)
    !  IF ((ANGLMMPT(HBI,I,2).EQ.DA1(2).OR.ANGLMMPT(HBI,I,4).EQ.DA1(2)) &
    !  .AND.(ANGLMMPT(HBI,I,3).EQ.DA1(1).OR.ANGLMMPT(HBI,I,3).EQ.DA1(3))) THEN
    !    ANGLMMPT(HBI,I,1)=-1
    !  ENDIF
    !ENDDO
    !! CHECK THE FOLLOWING CAREFULLY
    J=ANGLNUM(HBI)


    ANGLNUM(HBI)=0
    DO I=1, NTHETA
      IF ((DA2(2).EQ.IT(I) .OR. DA2(2).EQ.KT(I))) THEN
        IF (DA2(1).EQ.JT(I)) THEN
          ANGLNUM(HBI)=ANGLNUM(HBI)+1
          ANGLMMPT(HBI,ANGLNUM(HBI),1)=I ! >0 means existing angular bond
          ANGLMMPT(HBI,ANGLNUM(HBI),2)=IT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),3)=JT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),4)=KT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),5)=1
        ENDIF
      ELSE
        IF (DA2(1).EQ.JT(I)) THEN
          ANGLNUM(HBI)=ANGLNUM(HBI)+1
          ANGLMMPT(HBI,ANGLNUM(HBI),1)=I ! >0 means existing angular bond
          ANGLMMPT(HBI,ANGLNUM(HBI),2)=IT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),3)=JT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),4)=KT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),5)=2
        ELSE IF (DA2(3).EQ.JT(I)) THEN
          ANGLNUM(HBI)=ANGLNUM(HBI)+1
          ANGLMMPT(HBI,ANGLNUM(HBI),1)=I ! >0 means existing angular bond
          ANGLMMPT(HBI,ANGLNUM(HBI),2)=IT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),3)=JT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),4)=KT(I)
          ANGLMMPT(HBI,ANGLNUM(HBI),5)=-2
        ENDIF
      ENDIF
    ENDDO

    DO I=1, NBOND
      IF (DA2(3).EQ.IB(I)) THEN
         ANGLNUM(HBI)=ANGLNUM(HBI)+1
         ANGLMMPT(HBI,ANGLNUM(HBI),1)=0 ! =0 means non-existing angular bond
         ANGLMMPT(HBI,ANGLNUM(HBI),2)=JB(I)
         ANGLMMPT(HBI,ANGLNUM(HBI),3)=IB(I)
         ANGLMMPT(HBI,ANGLNUM(HBI),4)=DA2(2)
         ANGLMMPT(HBI,ANGLNUM(HBI),5)=-1
      ELSE IF (DA2(3).EQ.JB(I)) THEN
         ANGLNUM(HBI)=ANGLNUM(HBI)+1
         ANGLMMPT(HBI,ANGLNUM(HBI),1)=0
         ANGLMMPT(HBI,ANGLNUM(HBI),2)=IB(I)
         ANGLMMPT(HBI,ANGLNUM(HBI),3)=JB(I)
         ANGLMMPT(HBI,ANGLNUM(HBI),4)=DA2(2)
         ANGLMMPT(HBI,ANGLNUM(HBI),5)=-1
      ENDIF
    ENDDO

    IF (J.NE.ANGLNUM(HBI)) THEN
      WRITE(OUTU,*) 'MSMMPT MTFSWITCH> WARNINGS: ANGLNUM(HBI) changes',HBI
      ANGLNUM(HBI)=J
    ENDIF

    !c'emmpt: iac pointer
    DO I=1,ANGLNUM(HBI)
      IF (ANGLMMPT(HBI,I,1).EQ.0) THEN
        DO ITH=1, NTHETA
          IF (IAC(ANGLMMPT(HBI,I,2)).EQ.IAC(IT(ITH))        &
               .AND.IAC(ANGLMMPT(HBI,I,3)).EQ.IAC(JT(ITH))   &
               .AND.IAC(ANGLMMPT(HBI,I,4)).EQ.IAC(KT(ITH))) THEN
              ANGLMMPT(HBI,I,1)=ITH

              goto 6190
          ENDIF
          IF (IAC(ANGLMMPT(HBI,I,4)).EQ.IAC(IT(ITH))       &
               .AND.IAC(ANGLMMPT(HBI,I,3)).EQ.IAC(JT(ITH))  &
               .AND.IAC(ANGLMMPT(HBI,I,2)).EQ.IAC(KT(ITH))) THEN
              ANGLMMPT(HBI,I,1)=ITH

              goto 6190
          ENDIF
        ENDDO

        CALL WrnDie(-3,'<MISCOM>', &
                   'MSMMPTINIT> Failed to find reference angles')

6190    continue
      ENDIF
    ENDDO

!    IF (SMODE.EQ.1) THEN !NON-EXISTING NONBONDS SHOW UP AFTER SWAP
!      BKPNUM=0
!      DO I=1,NONBNUM(HBI)
!        IF(NONBMMPT(HBI,I,4).LT.0) THEN
!          BKPNUM=BKPNUM+1
!          BKP1(BKPNUM)=NONBMMPT(HBI,I,2)
!          BKP2(BKPNUM)=NONBMMPT(HBI,I,3)
!        ENDIF
!      ENDDO
!    ENDIF
    !update nonbonded
    !new H-A
    NONBMMPT(HBI,1,2)=DA2(3)
    NONBMMPT(HBI,1,3)=DA2(2)
    NONBMMPT(HBI,1,4)=2
    !new D-H
    NONBMMPT(HBI,2,2)=DA2(1)
    NONBMMPT(HBI,2,3)=DA2(2)
    NONBMMPT(HBI,2,4)=-2
    !new D-A
    NONBMMPT(HBI,3,2)=DA2(1)
    NONBMMPT(HBI,3,3)=DA2(3)
    NONBMMPT(HBI,3,4)=0

    NONBNUM(HBI)=3
    !!OHO only

!c'emmpt: find subroutine NONBINIT for deltailed explainations
! ------------------------- NBXMOD 3 -------------------------------

    IF ((NBXMOD .EQ. 3) .OR. (NBXMOD .EQ. 5)) THEN

      DO I=1,ANGLNUM(HBI)
!   HANDLE 1-3 NONB INTERACTION FOR HBI-A-X ORDER
        IF (ANGLMMPT(HBI,I,2).EQ.DA2(2) .AND. &
        ABS(ANGLMMPT(HBI,I,5)).EQ.1) THEN
          NONBNUM(HBI)=NONBNUM(HBI)+1
          NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
          NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
          NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,I,4)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
          IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
            NONBMMPT(HBI,NONBNUM(HBI),4)=-1
          ELSE
            NONBMMPT(HBI,NONBNUM(HBI),4)=1
          ENDIF

          !!prmlpe label
          IF (1.eq.1) THEN
            NONBNUM(HBI)=NONBNUM(HBI)+1
            NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
            NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,I,4)
            NONBMMPT(HBI,NONBNUM(HBI),4)=3
            IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(3)
            ELSE
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(1)
            ENDIF
          ENDIF
!   HANDLE 1-3 NONB INTERACTION FOR X-A-HBI ORDER
        ELSE IF (ANGLMMPT(HBI,I,4).EQ.DA2(2) .AND. &
        ABS(ANGLMMPT(HBI,I,5)).EQ.1) THEN
          NONBNUM(HBI)=NONBNUM(HBI)+1
          NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
          NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
          NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,I,2)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
          IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
            NONBMMPT(HBI,NONBNUM(HBI),4)=-1
          ELSE
            NONBMMPT(HBI,NONBNUM(HBI),4)=1
          ENDIF

          !!prmlpe label
          IF (1.eq.1) THEN
            NONBNUM(HBI)=NONBNUM(HBI)+1
            NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
            NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,I,2)
            NONBMMPT(HBI,NONBNUM(HBI),4)=3
            IF (ANGLMMPT(HBI,I,5).EQ.1) THEN
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(3)
            ELSE
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(1)
            ENDIF
          ENDIF
        ENDIF
      ENDDO

      !remove XD-D-H..A-XA and XA-A-H..D-XD nonbond interaction
      DO I=1,ANGLNUM(HBI)
        IF(ABS(ANGLMMPT(HBI,I,5)).EQ.1) THEN
          DO J=I+1,ANGLNUM(HBI)
            IF(ABS(ANGLMMPT(HBI,J,5)).EQ.1) THEN
              IF(((ANGLMMPT(HBI,I,3).EQ.DA2(1)).AND.(ANGLMMPT(HBI,J,3).EQ.DA2(3))) &
              .OR.((ANGLMMPT(HBI,I,3).EQ.DA2(3)).AND. &
              (ANGLMMPT(HBI,J,3).EQ.DA2(1)))) THEN
                NONBNUM(HBI)=NONBNUM(HBI)+1
                NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
                IF (ANGLMMPT(HBI,I,2).EQ.DA2(2)) THEN
                  NONBMMPT(HBI,NONBNUM(HBI),2)=ANGLMMPT(HBI,I,4)
                ELSE
                  NONBMMPT(HBI,NONBNUM(HBI),2)=ANGLMMPT(HBI,I,2)
                ENDIF
                IF (ANGLMMPT(HBI,J,2).EQ.DA2(2)) THEN
                  NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,J,4)
                ELSE
                  NONBMMPT(HBI,NONBNUM(HBI),3)=ANGLMMPT(HBI,J,2)
                ENDIF
                NONBMMPT(HBI,NONBNUM(HBI),4)=3
              ENDIF
            ENDIF
          ENDDO
        ENDIF
      ENDDO
    ENDIF

!   ------------- EXTENSION NBXMOD 4 ---------------------------------

    IF (NBXMOD .EQ. 4) THEN
      DO I=1,DIHENUM
    !   HANDLE 1-4 NONB INTERACTION FOR H-A-XA-XB ORDER
        IF (DIHEMMPT(I,2) .EQ. DA2(2)) THEN
           NONBNUM(HBI)=NONBNUM(HBI)+1
           NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
           NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
           NONBMMPT(HBI,NONBNUM(HBI),3)=DIHEMMPT(I,5)
  !   USE ANGLE FLAGS TO SET NB FLAGS
  !   DIHEDRAL TERMS ON DONOR SIDE HAVE FLAG ONE
  !   NEW DIHEDRAL TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
  !   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
  !   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
          IF (DIHEMMPT(I,6).EQ.1) THEN
             NONBMMPT(HBI,NONBNUM(HBI),4)=-1
          ELSE
             NONBMMPT(HBI,NONBNUM(HBI),4)=1
          ENDIF
  !   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-HBI ORDER
        ELSE IF (DIHEMMPT(I,5).EQ.DA2(2)) THEN
              NONBNUM(HBI)=NONBNUM(HBI)+1
              NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
              NONBMMPT(HBI,NONBNUM(HBI),3)=DIHEMMPT(I,2)
  !   USE ANGLE FLAGS TO SET NB FLAGS
  !   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
  !   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE
  !   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE
  !   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
          IF (DIHEMMPT(I,6).EQ.1) THEN
             NONBMMPT(HBI,NONBNUM(HBI),4)=-1
          ELSE
             NONBMMPT(HBI,NONBNUM(HBI),4)=1
          ENDIF
        ENDIF
      ENDDO
    ENDIF


!   ------------- EXTENSION NBXMOD 5 ---------------------------------
!   IF TRANSFERED HYDROGEN ATOM IS CONNECTED TO OTHER ATOMS THROUGH
!   DIHEDRAL ANGLES *AND* IF THESE ATOM PAIRS USE SPECIAL 1-4 VDW
!   PARAMETERS THESE 1-4 INTERACTIONS MUST BE SWITCHED FROM SPECIAL
!   TO STANDARD ON THE DONOR SIDE AND VICE VERSA ON THE ACCEPTOR SIDE

    IF (NBXMOD .EQ. 5) THEN
!   HANDLE 1-4 NONB EXCLUSION (NBXMOD 5)
      DO I=1,DIHENUM
!   HANDLE 1-4 NONB INTERACTION FOR HBI-A-XA-XB ORDER
        IF (DIHEMMPT(I,2) .EQ. DA2(2)) THEN
           NONBNUM(HBI)=NONBNUM(HBI)+1
           NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
           NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
           NONBMMPT(HBI,NONBNUM(HBI),3)=DIHEMMPT(I,5)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL TERMS ON ACCEPTOR SIDE HAVE FLAG -14
          IF (DIHEMMPT(I,6).EQ.1) THEN
             NONBMMPT(HBI,NONBNUM(HBI),4)=-14
          ELSE
             NONBMMPT(HBI,NONBNUM(HBI),4)=14
          ENDIF
!   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-HBI ORDER
        ELSE IF (DIHEMMPT(I,5).EQ.DA2(2)) THEN
              NONBNUM(HBI)=NONBNUM(HBI)+1
              NONBMMPT(HBI,NONBNUM(HBI),1)=NONBNUM(HBI)
              NONBMMPT(HBI,NONBNUM(HBI),2)=DA2(2)
              NONBMMPT(HBI,NONBNUM(HBI),3)=DIHEMMPT(I,2)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL 1-4  TERMS ON ACCEPTOR SIDE HAVE FLAG -14
          IF (DIHEMMPT(I,6).EQ.1) THEN
             NONBMMPT(HBI,NONBNUM(HBI),4)=-14
          ELSE
             NONBMMPT(HBI,NONBNUM(HBI),4)=14
          ENDIF
        ENDIF
      ENDDO
    ENDIF


  END SUBROUTINE MTFSWITCH

  SUBROUTINE BONDSWAP(DAT,HAT,AAT)

    use bases_fcm
    use block_fcm
    use chm_types
    use cnst_fcm
    use code
    use comand
    use consta
    use contrl
    use dimens_fcm
    use inbnd
    use number
    use param
    use psf
    use stream
    use string

!!regardless of iac[], only in [h2o]n
    INTEGER DAT,HAT,AAT,I,J,K

    !Update dihedral list
    !!improper to be updated
    DO I=1,NPHI
      IF ((IP(I).EQ.HAT .AND. JP(I).EQ.DAT) .OR. &
      (LP(I).EQ.HAT .AND. KP(I).EQ.DAT)) THEN
        IP(I)=-1
      ENDIF
    ENDDO

    J=NPHI
    DO I=1,NTHETA
      IF (IT(I).EQ.AAT) THEN
        J=J+1
        IP(J)=KT(I)
        JP(J)=JT(I)
        KP(J)=IT(I)
        LP(J)=HAT
      ELSE IF (KT(I).EQ.AAT) THEN
        J=J+1
        IP(J)=IT(I)
        JP(J)=JT(I)
        KP(J)=KT(I)
        LP(J)=HAT
      ENDIF
    ENDDO

    DO I=1,NPHI
      IF (IP(I).EQ.-1) THEN
        IF (J.LE.I) GOTO 7310
        IP(I)=IP(J)
        JP(I)=JP(J)
        KP(I)=KP(J)
        LP(I)=LP(J)
        J=J-1
      ENDIF
    ENDDO

7310 CONTINUE

    NPHI=J!!?

    !Update angular bond list
    DO I=1,NTHETA
      IF ((IT(I).EQ.HAT.OR.KT(I).EQ.HAT) .AND. JT(I).EQ.DAT) THEN
        IT(I)=-1
      ENDIF
    ENDDO

    J=NTHETA
    DO I=1,NBOND
      IF (IB(I).EQ.AAT) THEN
        J=J+1
        IT(J)=JB(I)
        JT(J)=IB(I)
        KT(J)=HAT
      ELSE IF (JB(I).EQ.AAT) THEN
        J=J+1
        IT(J)=IB(I)
        JT(J)=JB(I)
        KT(J)=HAT
      ENDIF
    ENDDO

    DO I=1,NTHETA
      IF (IT(I).EQ.-1) THEN
        IF (J.LE.I) GOTO 7350
        IT(I)=IT(J)
        JT(I)=JT(J)
        KT(I)=KT(J)
        J=J-1
      ENDIF
    ENDDO

7350 CONTINUE

    IF(J.NE.NTHETA) THEN
      CALL WrnDie (-3,'<MISCOM>','MSMMPT BONDSWAP> NTHETA is changed')
    ENDIF

    !Update bond list, at last
    DO I=1,NBOND
      IF (DAT.EQ.IB(I) .AND. HAT.EQ.JB(I)) THEN
        IB(I)=AAT
      ELSE IF (DAT.EQ.JB(I) .AND. HAT.EQ.IB(I)) THEN
        JB(I)=AAT
      ENDIF
    ENDDO
    !!iac

  END SUBROUTINE BONDSWAP

  SUBROUTINE NONBUPDATE(HBI,INBLO,JNB,IMAT,IMBLO,IMJNB,NA,NATT,NMODE)

    use dimens_fcm
    use comand
    use stream
    use string
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER I,J,K,HBI,JSTART,JEND,INBLO(*),JNB(*),NA
    INTEGER IMAT(*),IMBLO(*),IMJNB(*),NATT
    INTEGER NB2,NB3,NB4,NMODE
!real(chm_real) tstart,tend

!call cpu_time(tstart)

!    DO I=1,NA
!      DO J=1,NA
!        IF (ABS(MMPTNBL(I,J)).NE.2) MMPTNBL(I,J)=0
!      ENDDO
!    ENDDO
!
!    DO I=1,NATT-NA
!      DO J=1,NA
!        MMPTIMGL(I,J)=0
!      ENDDO
!    ENDDO

    IF (NMODE.EQ.1) THEN
      DO I=1,NONBNUM(HBI)
        NB2=NONBMMPT(HBI,I,2)
        NB3=NONBMMPT(HBI,I,3)
        NB4=NONBMMPT(HBI,I,4)
        IF (NB4.EQ.1 .OR. NB4.EQ.2) THEN
          MMPTNBL(NB2,NB3)=-2
          MMPTNBL(NB3,NB2)=-2
        ELSE IF (NB4.EQ.-1 .OR. NB4.EQ.-2) THEN
          MMPTNBL(NB2,NB3)=2
          MMPTNBL(NB3,NB2)=2
        ENDIF
      ENDDO
    ENDIF

    DO I=1,NBOND
      IF (MMPTNBL(IB(I),JB(I)).EQ.0) THEN
        MMPTNBL(IB(I),JB(I))=-1
        MMPTNBL(JB(I),IB(I))=-1
      ENDIF
    ENDDO

    DO I=1,NTHETA
      IF (MMPTNBL(IT(I),KT(I)).EQ.0) THEN
        MMPTNBL(IT(I),KT(I))=-1
        MMPTNBL(KT(I),IT(I))=-1
      ENDIF
    ENDDO

    JSTART=1
    DO I=1,NA
      JEND=INBLO(I)
      DO J=JSTART,JEND
        IF (MMPTNBL(I,JNB(J)).EQ.0) THEN
          MMPTNBL(I,JNB(J))=1
          MMPTNBL(JNB(J),I)=1
        ENDIF
      ENDDO
      JSTART=INBLO(I)+1
    ENDDO

!call cpu_time(tend)
!write(outu,*) 'TEST---NB>',tend-tstart
!tstart=tend

    JSTART=1
    K=0
    DO I=NA+1,NATT
      JEND=IMBLO(I)
      DO J=JSTART,JEND
        IF (MMPTNBL(IMAT(I),IMJNB(J)).EQ.0) THEN
          K=K+1
          IMJNB(K)=IMJNB(J)
        ENDIF
      ENDDO
      JSTART=IMBLO(I)+1
      IMBLO(I)=K
    ENDDO

!call cpu_time(tend)
!write(outu,*) 'TEST---NB>',tend-tstart
!tstart=tend


    K=0
    DO I=1,NA
      DO J=(I+1),NA
        IF (MMPTNBL(I,J).EQ.1) THEN
          K=K+1
          JNB(K)=J
          MMPTNBL(I,J) = 0
          MMPTNBL(J,I) = 0
        ELSE IF (MMPTNBL(I,J).EQ.2) THEN
          K=K+1
          JNB(K)=J
        ELSE IF (MMPTNBL(I,J).EQ.-1) THEN
          MMPTNBL(I,J) = 0
          MMPTNBL(J,I) = 0
        ENDIF
      ENDDO
      INBLO(I)=K
    ENDDO


!call cpu_time(tend)
!write(outu,*) 'TEST---NB>',tend-tstart
!tstart=tend

  END SUBROUTINE NONBUPDATE


  SUBROUTINE EPSEUDOMMPT(I,J,K,EA,FCST,EQUIDEG,MODE,DXAI,DYAI,DZAI, &
  DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
!     THIS ROUTINE CALCULATES THE PSEUDO-ENERGY OF COUPLED D-H-A
! IF D-H-A>90, EA=0, d(EA)=0
! IF D-H-A<90, EA = 45.0 * (theta-90.0)**2

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,K,MODE

    real(chm_real) RI2,RJ2,RI,RJ,RIR,RJR,          &
          DXI,DYI,DZI,DXJ,DYJ,DZJ,         &
          DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,   &
          CST, AT, STR, ST2R,              &
          DA,DF,EA,                        &
          DTXI,DTYI,DTZI,DTXJ,DTYJ,DTZJ,   &
          DFX,DGX,DFY,DGY,DFZ,DGZ

    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK
    REAL(CHM_REAL) FCST,EQUI,EQUIDEG,EQCUT,DBX,DBX3,DBX4,DBX5

    EA=0.0D0

    EQUI=EQUIDEG

    DXI=X(I)-X(J)
    DYI=Y(I)-Y(J)
    DZI=Z(I)-Z(J)
    DXJ=X(K)-X(J)
    DYJ=Y(K)-Y(J)
    DZJ=Z(K)-Z(J)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXI.GT.0.5D0*PXSIZE) THEN
        DXI=DXI-PXSIZE
      ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
        DXI=DXI+PXSIZE
      ENDIF
      IF (DYI.GT.0.5D0*PYSIZE) THEN
        DYI=DYI-PYSIZE
      ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
        DYI=DYI+PYSIZE
      ENDIF
      IF (DZI.GT.0.5D0*PZSIZE) THEN
        DZI=DZI-PZSIZE
      ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
        DZI=DZI+PZSIZE
      ENDIF

      IF (DXJ.GT.0.5D0*PXSIZE) THEN
        DXJ=DXJ-PXSIZE
      ELSE IF (DXJ.LE.-0.5D0*PXSIZE) THEN
        DXJ=DXJ+PXSIZE
      ENDIF
      IF (DYJ.GT.0.5D0*PYSIZE) THEN
        DYJ=DYJ-PYSIZE
      ELSE IF (DYJ.LE.-0.5D0*PYSIZE) THEN
        DYJ=DYJ+PYSIZE
      ENDIF
      IF (DZJ.GT.0.5D0*PZSIZE) THEN
        DZJ=DZJ-PZSIZE
      ELSE IF (DZJ.LE.-0.5D0*PZSIZE) THEN
        DZJ=DZJ+PZSIZE
      ENDIF
    ENDIF

    RI2=DXI*DXI+DYI*DYI+DZI*DZI
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

    RI=SQRT(RI2)
    RJ=SQRT(RJ2)

    RIR=1.D0/RI
    RJR=1.D0/RJ

    DXIR=DXI*RIR
    DYIR=DYI*RIR
    DZIR=DZI*RIR
    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR

    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

    AT=ACOS(CST)

    IF (MODE.EQ.0) THEN
      CONTINUE
    ELSE IF (MODE.LT.0) THEN
      IF (AT.GE.EQUI) THEN
        EA=0.D0
        DXAI=0.D0
        DXAJ=0.D0
        DXAK=0.D0

        DYAI=0.D0
        DYAJ=0.D0
        DYAK=0.D0

        DZAI=0.D0
        DZAJ=0.D0
        DZAK=0.D0

        RETURN
      ENDIF
    ELSE
      IF (AT.LE.EQUI) THEN
        EA=0.D0
        DXAI=0.D0
        DXAJ=0.D0
        DXAK=0.D0

        DYAI=0.D0
        DYAJ=0.D0
        DYAK=0.D0

        DZAI=0.D0
        DZAJ=0.D0
        DZAK=0.D0

        RETURN
      ENDIF
    ENDIF

    DA=AT-EQUI
    EQCUT=0.5D0
    IF (MODE.EQ.2 .AND. DA.LT.EQCUT) THEN
      DBX=DA/EQCUT
      DBX3=DBX*DBX*DBX
      DBX4=DBX3*DBX
      DBX5=DBX4*DBX
      EA=FCST*DA*DA*(6.D0*DBX5-15.D0*DBX4+10.D0*DBX3)
      DF=FCST*DA*(21.D0*DBX5-45.D0*DBX4+25.D0*DBX3)
      DF=DF+DF
    ELSE IF (MODE.EQ.-2 .AND. DA.GT.0.D0-EQCUT) THEN
      DBX=DA/EQCUT
      DBX3=DBX*DBX*DBX
      DBX4=DBX3*DBX
      DBX5=DBX4*DBX
      EA=FCST*DA*DA*(-6.D0*DBX5-15.D0*DBX4-10.D0*DBX3)
      DF=FCST*DA*(-21.D0*DBX5-45.D0*DBX4-25.D0*DBX3)
      DF=DF+DF
    ELSE
      DF=FCST*DA
      EA=DF*DA
      DF=DF+DF
    ENDIF






    ST2R=1.D0/(1.D0-CST*CST)
    STR=SQRT(ST2R)
    DF=-DF*STR

    DTXI=RIR*(DXJR-CST*DXIR)
    DTXJ=RJR*(DXIR-CST*DXJR)
    DTYI=RIR*(DYJR-CST*DYIR)
    DTYJ=RJR*(DYIR-CST*DYJR)
    DTZI=RIR*(DZJR-CST*DZIR)
    DTZJ=RJR*(DZIR-CST*DZJR)

    DFX=DF*DTXI
    DGX=DF*DTXJ

    DFY=DF*DTYI
    DGY=DF*DTYJ

    DFZ=DF*DTZI
    DGZ=DF*DTZJ

    DXAI=DFX
    DXAJ=-DFX-DGX
    DXAK=DGX

    DYAI=DFY
    DYAJ=-DFY-DGY
    DYAK=DGY

    DZAI=DFZ
    DZAJ=-DFZ-DGZ
    DZAK=DGZ


    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-ANGLE ATOMS (I, J, K):',I,J,K
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-ANGLE ENERGY',EA
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-ANGLE FORCE ATOM I',DXAI,DYAI,DZAI
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-ANGLE FORCE ATOM J',DXAJ,DYAJ,DZAJ
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-ANGLE FORCE ATOM K',DXAK,DYAK,DZAK
    ENDIF

    RETURN
  END SUBROUTINE EPSEUDOMMPT


  SUBROUTINE EPSEUDOBOND(I,J,EB,FCST,EQUI,MODE,DXBRM,DYBRM,DZBRM)

! ... routine calculates classical bond energy terms
! ... and forces for specific bonds
!      results or stored in ebmmpt and dxbrm, dxbrm
! ... based on charmm ebond subroutine

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    real(chm_real) EB,RX,RY,RZ,S1,S2,DB,DF,DXBRM,DYBRM,DZBRM
    real(chm_real) R,ERM,FCST,EQUI,EQCUT,DBX,DBX3,DBX4,DBX5
    INTEGER I,II,J,IC,MODE

    ERM=0.0D0

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (RX.GT.0.5D0*PXSIZE) THEN
        RX=RX-PXSIZE
      ELSE IF (RX.LE.-0.5D0*PXSIZE) THEN
        RX=RX+PXSIZE
      ENDIF
      IF (RY.GT.0.5D0*PYSIZE) THEN
        RY=RY-PYSIZE
      ELSE IF (RY.LE.-0.5D0*PYSIZE) THEN
        RY=RY+PYSIZE
      ENDIF
      IF (RZ.GT.0.5D0*PZSIZE) THEN
        RZ=RZ-PZSIZE
      ELSE IF (RZ.LE.-0.5D0*PZSIZE) THEN
        RZ=RZ+PZSIZE
      ENDIF
    ENDIF

    S2=RX*RX + RY*RY + RZ*RZ
    S1=SQRT(S2)


    DB=S1-EQUI  !equillibrium for penalty

    IF (MODE.EQ.0) THEN
      CONTINUE
    ELSE IF (MODE.LT.0) THEN
      IF (DB.GE.0.D0) THEN
        EB=0.D0
        DXBRM=0.D0
        DYBRM=0.D0
        DZBRM=0.D0

        RETURN
      ENDIF
    ELSE
      IF (DB.LE.0.D0) THEN
        EB=0.D0
        DXBRM=0.D0
        DYBRM=0.D0
        DZBRM=0.D0

        RETURN
      ENDIF
    ENDIF

    EQCUT=0.25D0
    IF (MODE.EQ.2 .AND. DB.LT.EQCUT) THEN
      DBX=DB/EQCUT
      DBX3=DBX*DBX*DBX
      DBX4=DBX3*DBX
      DBX5=DBX4*DBX
      EB=FCST*DB*DB*(6.D0*DBX5-15.D0*DBX4+10.D0*DBX3)
      DF=FCST*DB*(21.D0*DBX5-45.D0*DBX4+25.D0*DBX3)
    ELSE IF (MODE.EQ.-2 .AND. DB.GT.0.D0-EQCUT) THEN
      DBX=DB/EQCUT
      DBX3=DBX*DBX*DBX
      DBX4=DBX3*DBX
      DBX5=DBX4*DBX
      EB=FCST*DB*DB*(-6.D0*DBX5-15.D0*DBX4-10.D0*DBX3)
      DF=FCST*DB*(-21.D0*DBX5-45.D0*DBX4-25.D0*DBX3)
    ELSE
      DF=FCST*DB !force constant for penalty
      EB=DF*DB
    ENDIF

    R=2.D0/S1
    DF=DF*R

    DXBRM=DF*RX
    DYBRM=DF*RY
    DZBRM=DF*RZ

    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT EXTERN> PENALTY-BOND ATOMS ATOM I',I,'  ATOM J', J
      WRITE(OUTU,*) 'MSMMPT EXTERN> PENALTY-BOND ENERGY',EB
      WRITE(OUTU,*) 'MSMMPT EXTERN> PENALTY-BOND FORCE ATOM I',DXBRM,DYBRM,DZBRM
      WRITE(OUTU,*) 'MSMMPT EXTERN> PENALTY-BOND FORCE ATOM J',-DXBRM,-DYBRM,-DZBRM
      WRITE(OUTU,*) ' '
    ENDIF

  END SUBROUTINE EPSEUDOBOND

  SUBROUTINE EPSEUDODIHEDRAL(I,J,K,L,ARG,&
         ERM,DXIRM,DXJRM,DXKRM,DXLRM,                               &
             DYIRM,DYJRM,DYKRM,DYLRM,                               &
             DZIRM,DZJRM,DZKRM,DZLRM)

!     VARIABLES
!     CPD Periodicity of the dihedral energy
!     CPB Phase shift (delta) for the dihedral energy
!     CPCOS COS(CPB)
!     CPSIN SIN(CPB)
!     CPC Force constant for the dihedral energy


    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use consta
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER I,J,K,L,IPER,NPER

    real(chm_real) e,ap,arg,erm

    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ,                        &
          AX,AY,AZ,BX,BY,BZ,                                  &
          RA2,RB2,RG2,RG,                                     &
          RGR,RA2R,RB2R,RABR,                                 &
          CP,SP,                                              &
          DF,DDF,E1,DF1,DDF1,                                 &
          FG,HG,FGA,HGB,GAA,GBB,                              &
          DTFX,DTFY,DTFZ,DTGX,DTGY,DTGZ,DTHX,DTHY,DTHZ,       &
          DFX,DFY,DFZ,DGX,DGY,DGZ,DHX,DHY,DHZ,                &
          CA,SA
    LOGICAL LREP
    real(chm_real)  DXIRM,DXJRM,DXKRM,DXLRM,      &
           DYIRM,DYJRM,DYKRM,DYLRM,       &
           DZIRM,DZJRM,DZKRM,DZLRM

    ERM=0.0D0

    FX=X(I)-X(J)
    FY=Y(I)-Y(J)
    FZ=Z(I)-Z(J)
    GX=X(J)-X(K)
    GY=Y(J)-Y(K)
    GZ=Z(J)-Z(K)
    HX=X(L)-X(K)
    HY=Y(L)-Y(K)
    HZ=Z(L)-Z(K)

    ! In a situation of PBC
    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (FX.GT.0.5D0*PXSIZE) THEN
        FX=FX-PXSIZE
      ELSE IF (FX.LE.-0.5D0*PXSIZE) THEN
        FX=FX+PXSIZE
      ENDIF
      IF (FY.GT.0.5D0*PYSIZE) THEN
        FY=FY-PYSIZE
      ELSE IF (FY.LE.-0.5D0*PYSIZE) THEN
        FY=FY+PYSIZE
      ENDIF
      IF (FZ.GT.0.5D0*PZSIZE) THEN
        FZ=FZ-PZSIZE
      ELSE IF (FZ.LE.-0.5D0*PZSIZE) THEN
        FZ=FZ+PZSIZE
      ENDIF

      IF (GX.GT.0.5D0*PXSIZE) THEN
        GX=GX-PXSIZE
      ELSE IF (GX.LE.-0.5D0*PXSIZE) THEN
        GX=GX+PXSIZE
      ENDIF
      IF (GY.GT.0.5D0*PYSIZE) THEN
        GY=GY-PYSIZE
      ELSE IF (GY.LE.-0.5D0*PYSIZE) THEN
        GY=GY+PYSIZE
      ENDIF
      IF (GZ.GT.0.5D0*PZSIZE) THEN
        GZ=GZ-PZSIZE
      ELSE IF (GZ.LE.-0.5D0*PZSIZE) THEN
        GZ=GZ+PZSIZE
      ENDIF

      IF (HX.GT.0.5D0*PXSIZE) THEN
        HX=HX-PXSIZE
      ELSE IF (HX.LE.-0.5D0*PXSIZE) THEN
        HX=HX+PXSIZE
      ENDIF
      IF (HY.GT.0.5D0*PYSIZE) THEN
        HY=HY-PYSIZE
      ELSE IF (HY.LE.-0.5D0*PYSIZE) THEN
        HY=HY+PYSIZE
      ENDIF
      IF (HZ.GT.0.5D0*PZSIZE) THEN
        HZ=HZ-PZSIZE
      ELSE IF (HZ.LE.-0.5D0*PZSIZE) THEN
        HZ=HZ+PZSIZE
      ENDIF

    ENDIF

    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX

    RA2=AX*AX+AY*AY+AZ*AZ
    RB2=BX*BX+BY*BY+BZ*BZ
    RG2=GX*GX+GY*GY+GZ*GZ
    RG=SQRT(RG2)

    RGR=ONE/RG
    RA2R=ONE/RA2
    RB2R=ONE/RB2
    RABR=SQRT(RA2R*RB2R)

    CP=(AX*BX+AY*BY+AZ*BZ)*RABR
    SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)

!   SETUP PROPER DIHEDRALS
!   CPD  Periodicity of the dihedral energy

    E=ZERO
    DF=ZERO
    DDF=ZERO

    IPER=2

    E1=ONE
    DF1=ZERO

    DO NPER=1,IPER
        DDF1=E1*CP-DF1*SP
        DF1=E1*SP+DF1*CP
        E1=DDF1
    enddo

    E1=E1*1.D0+DF1*0.D0
    DF1=DF1*1.D0-DDF1*0.D0
    DF1=-IPER*DF1
    DDF1=-IPER*IPER*E1
    E1=ONE+E1

    IF (IPER.EQ.0) THEN
        E1=ONE
        DF1=ZERO
        DDF1=ZERO
    ENDIF

    E=E+ARG*E1
    DF=DF+ARG*DF1
    DDF=DDF+ARG*DDF1

    FG=FX*GX+FY*GY+FZ*GZ
    HG=HX*GX+HY*GY+HZ*GZ
    FGA=FG*RA2R*RGR
    HGB=HG*RB2R*RGR
    GAA=-RA2R*RG
    GBB=RB2R*RG

    DTFX=GAA*AX
    DTFY=GAA*AY
    DTFZ=GAA*AZ
    DTGX=FGA*AX-HGB*BX
    DTGY=FGA*AY-HGB*BY
    DTGZ=FGA*AZ-HGB*BZ
    DTHX=GBB*BX
    DTHY=GBB*BY
    DTHZ=GBB*BZ

    DFX=DF*DTFX
    DFY=DF*DTFY
    DFZ=DF*DTFZ
    DGX=DF*DTGX
    DGY=DF*DTGY
    DGZ=DF*DTGZ
    DHX=DF*DTHX
    DHY=DF*DTHY
    DHZ=DF*DTHZ

    ERM=E

    DXIRM=DFX
    DXJRM=-DFX+DGX
    DXKRM=-DHX-DGX
    DXLRM=DHX

    DYIRM=DFY
    DYJRM=-DFY+DGY
    DYKRM=-DHY-DGY
    DYLRM=DHY

    DZIRM=DFZ
    DZJRM=-DFZ+DGZ
    DZKRM=-DHZ-DGZ
    DZLRM=DHZ

    IF(PRNLEV.GT.8) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL ATOMS ATOM I',I,'  ATOM J', J   &
            ,'  ATOM K', K,'  ATOM L', L
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL ENERGY',E
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL [COS(PHI),SIN(PHI)]=',CP,SP
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL FORCE ATOM I',DXIRM,DYIRM,DZIRM
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL FORCE ATOM J',DXJRM,DYJRM,DZJRM
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL FORCE ATOM K',DXKRM,DYKRM,DZKRM
      WRITE(OUTU,*) 'MSMMPT EXTERN> PSEUDO-DIHEDRAL FORCE ATOM L',DXLRM,DYLRM,DZLRM
      WRITE(OUTU,*) ' '

    ENDIF

  END SUBROUTINE EPSEUDODIHEDRAL

  SUBROUTINE EPSEUDOVDW(I,J,ENBMMPT,EPS,RMIN, &
      DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,NB
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,      &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          ENBMMPT,DF

    real(chm_real) DXIRM,DXJRM,DYIRM,DYJRM,DZIRM,DZJRM
    real(chm_real) XYZ, CTOFNB,CTONNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) DXIT,DYIT,DZIT
    real(chm_real) RDIST,C2ONNB,C3OFON,VSWI
    real(chm_real) EPS,RMIN

!   TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.

    CTOFNB=PRM_NONB_CUTOFF
    CTONNB=PRM_NONB_CUTON
    C2OFNB=CTOFNB*CTOFNB

    ENBMMPT=0.D0

    DXIRM=0.D0
    DYIRM=0.D0
    DZIRM=0.D0
    DXJRM=0.D0
    DYJRM=0.D0
    DZJRM=0.D0

    if (EPS.ne.0.d0) then
      CRXI=X(I)
      CRYI=Y(I)
      CRZI=Z(I)

      DXI=CRXI-X(J)
      DYI=CRYI-Y(J)
      DZI=CRZI-Z(J)

      IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
        IF (DXI.GT.0.5D0*PXSIZE) THEN
          DXI=DXI-PXSIZE
        ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
          DXI=DXI+PXSIZE
        ENDIF
        IF (DYI.GT.0.5D0*PYSIZE) THEN
          DYI=DYI-PYSIZE
        ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
          DYI=DYI+PYSIZE
        ENDIF
        IF (DZI.GT.0.5D0*PZSIZE) THEN
          DZI=DZI-PZSIZE
        ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
          DZI=DZI+PZSIZE
        ENDIF
      ENDIF

      S=DXI*DXI+DYI*DYI+DZI*DZI
      RDIST=SQRT(S)

      IF (RDIST.LT.CTOFNB) THEN
        R2=1.0/S
        R1 = SQRT(R2)

        DF=0.0D0

  !   VAN DER WAALS
        SIG2=RMIN*RMIN*R2
        SIG6=SIG2*SIG2*SIG2
        SIG12=SIG6*SIG6

        IF (RDIST.LT.CTONNB) THEN
          ENBMMPT=ENBMMPT+EPS*(SIG12-SIG6-SIG6)
          DF=EPS*R2*12.D0*(SIG6-SIG12)
        ELSE
          C2ONNB=CTONNB*CTONNB
          C3OFON=(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)
          VSWI=(S-C2OFNB)*(S-C2OFNB)*(C2OFNB+2.D0*S-3.D0*C2ONNB)/C3OFON
          ENBMMPT=ENBMMPT+EPS*(SIG12-SIG6-SIG6)*VSWI
          DF=EPS*R2*12.D0*(SIG6-SIG12)*VSWI+EPS*(SIG12-SIG6-SIG6)* &
          12.D0*(S-C2OFNB)*(S-C2ONNB)/C3OFON
        ENDIF

  !   ADD -DXIT TO J AND ADD DXIT TO I
        DXIT=DXI*DF
        DYIT=DYI*DF
        DZIT=DZI*DF

        DXIRM=DXIRM+DXIT
        DYIRM=DYIRM+DYIT
        DZIRM=DZIRM+DZIT
        DXJRM=DXJRM-DXIT
        DYJRM=DYJRM-DYIT
        DZJRM=DZJRM-DZIT

        IF(PRNLEV.GT.8) THEN
          WRITE(OUTU,*) ' '
          WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED ATOMS ATOM I',I,' ATOM J',J
          WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED ENERGY VDW', ENBMMPT
          WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED DIST & EPSILON & RMIN', &
          1.d0/R1,EPS, RMIN
          WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
          WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
        ENDIF
      ENDIF
    endif


    return
  END SUBROUTINE EPSEUDOVDW

  SUBROUTINE EPSEUDONONB(I,J,EPS,RMIN, &
      DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM, &
      DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1, &
      DXIRM2,DYIRM2,DZIRM2,DXJRM2,DYJRM2,DZJRM2, &
      ENBMMPT,EELMMPT)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER M,I,J,NB
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,      &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          ENBMMPT,EELMMPT,DF,DF2

    real(chm_real) DXIRM,DXJRM,DYIRM,DYJRM,DZIRM,DZJRM
    real(chm_real)  DXIRM1,DXJRM1,DYIRM1,DYJRM1,DZIRM1,DZJRM1
    real(chm_real)  DXIRM2,DXJRM2,DYIRM2,DYJRM2,DZIRM2,DZJRM2
    real(chm_real) XYZ, CTOFNB,CTONNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) DXIT,DYIT,DZIT
    real(chm_real) RDIST,C2ONNB,C3OFON,VSWI
    real(chm_real) EPS,RMIN

!   TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.

    CTOFNB=PRM_NONB_CUTOFF
    CTONNB=PRM_NONB_CUTON
    C2OFNB=CTOFNB*CTOFNB

!   SHIFTED DIELECTRIC COEFFICIENTS
    C2ROF2=MINTWO/C2OFNB
    CHROF2=-HALF/C2OFNB

    CGF= 332.0716D0
    ENBMMPT=0.D0
    EELMMPT=0.D0

    DXIRM=0.D0
    DYIRM=0.D0
    DZIRM=0.D0
    DXJRM=0.D0
    DYJRM=0.D0
    DZJRM=0.D0

    DXIRM1=0.D0
    DYIRM1=0.D0
    DZIRM1=0.D0
    DXJRM1=0.D0
    DYJRM1=0.D0
    DZJRM1=0.D0

    DXIRM2=0.D0
    DYIRM2=0.D0
    DZIRM2=0.D0
    DXJRM2=0.D0
    DYJRM2=0.D0
    DZJRM2=0.D0

    CRXI=X(I)
    CRYI=Y(I)
    CRZI=Z(I)

    DXI=CRXI-X(J)
    DYI=CRYI-Y(J)
    DZI=CRZI-Z(J)

    IF (PXSIZE.GT.0.D0 .AND. PYSIZE.GT.0.D0 .AND. PZSIZE.GT.0.D0) THEN
      IF (DXI.GT.0.5D0*PXSIZE) THEN
        DXI=DXI-PXSIZE
      ELSE IF (DXI.LE.-0.5D0*PXSIZE) THEN
        DXI=DXI+PXSIZE
      ENDIF
      IF (DYI.GT.0.5D0*PYSIZE) THEN
        DYI=DYI-PYSIZE
      ELSE IF (DYI.LE.-0.5D0*PYSIZE) THEN
        DYI=DYI+PYSIZE
      ENDIF
      IF (DZI.GT.0.5D0*PZSIZE) THEN
        DZI=DZI-PZSIZE
      ELSE IF (DZI.LE.-0.5D0*PZSIZE) THEN
        DZI=DZI+PZSIZE
      ENDIF
    ENDIF


    S=DXI*DXI+DYI*DYI+DZI*DZI
    RDIST=SQRT(S)

    IF (RDIST.LT.CTOFNB) THEN
      R2=1.0/S
      R1 = SQRT(R2)

      DF=0.0D0

      if (EPS.ne.0.d0) then
    !   VAN DER WAALS
        SIG2=RMIN*RMIN*R2
        SIG6=SIG2*SIG2*SIG2
        SIG12=SIG6*SIG6

        IF (RDIST.LT.CTONNB) THEN
          ENBMMPT=ENBMMPT+EPS*(SIG12-SIG6-SIG6)
          DF=EPS*R2*12.D0*(SIG6-SIG12)
        ELSE
          C2ONNB=CTONNB*CTONNB
          C3OFON=(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)*(C2OFNB-C2ONNB)
          VSWI=(S-C2OFNB)*(S-C2OFNB)*(C2OFNB+2.D0*S-3.D0*C2ONNB)/C3OFON
          ENBMMPT=ENBMMPT+EPS*(SIG12-SIG6-SIG6)*VSWI
          DF=EPS*R2*12.D0*(SIG6-SIG12)*VSWI+EPS*(SIG12-SIG6-SIG6)* &
          12.D0*(S-C2OFNB)*(S-C2ONNB)/C3OFON
        ENDIF

  !   ADD -DXIT TO J AND ADD DXIT TO I
        DXIT=DXI*DF
        DYIT=DYI*DF
        DZIT=DZI*DF

        DXIRM1=DXIRM1+DXIT
        DYIRM1=DYIRM1+DYIT
        DZIRM1=DZIRM1+DZIT

        DXJRM1=DXJRM1-DXIT
        DYJRM1=DYJRM1-DYIT
        DZJRM1=DZJRM1-DZIT
      ENDIF

!   ELECTROSTATIC
      G1=CGF*CG(I)*CG(J)*R1
      G2=G1*S*C2ROF2   !QxQxRx(-2)/rcof^2
      G3=G2*S*CHROF2   !QxQxR^3/rcof^4

      EELMMPT=EELMMPT+(G1+G2+G3)
      DF2=R2*(G2-G1+THREE*G3)

!   ADD -DXIT TO J AND ADD DXIT TO I
      DXIT=DXI*DF2
      DYIT=DYI*DF2
      DZIT=DZI*DF2

      DXIRM2=DXIRM2+DXIT
      DYIRM2=DYIRM2+DYIT
      DZIRM2=DZIRM2+DZIT

      DXJRM2=DXJRM2-DXIT
      DYJRM2=DYJRM2-DYIT
      DZJRM2=DZJRM2-DZIT


!   TOTAL
      DF=DF+DF2

!   ADD -DXIT TO J AND ADD DXIT TO I
      DXIT=DXI*DF
      DYIT=DYI*DF
      DZIT=DZI*DF

      DXIRM=DXIRM+DXIT
      DYIRM=DYIRM+DYIT
      DZIRM=DZIRM+DZIT

      DXJRM=DXJRM-DXIT
      DYJRM=DYJRM-DYIT
      DZJRM=DZJRM-DZIT

      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED ATOMS ATOM I',I,' ATOM J',J
        WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED ENERGY VDW-ELEC', ENBMMPT,EELMMPT
        WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED DIST & EPSILON & RMIN', &
        1.d0/R1,EPS, RMIN
        WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MSMMPT EXTERN> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
      ENDIF
    endif


    return
  END SUBROUTINE EPSEUDONONB

  SUBROUTINE ECONSTRAINT(NAT,EU,X,Y,Z,DX,DY,DZ)

    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    use code
    use bases_fcm
    use chm_types

    INTEGER NAT,I
    real(chm_real) X(NAT),Y(NAT),Z(NAT)
    real(chm_real) DX(NAT),DY(NAT),DZ(NAT)
    real(chm_real) EU,ETMP,DXBRM,DYBRM,DZBRM
    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK

    DO I = 1,NUMCONS

      IF (ECATOMC(I).LE.0) then

        CALL EPSEUDOBOND(ECATOMA(I),ECATOMB(I),ETMP,ECONSFC(I),ECONSEQ(I),ECIFLAG(I), &
        DXBRM,DYBRM,DZBRM)

        EU = EU + ETMP

        DX(ECATOMA(I))=DX(ECATOMA(I))+DXBRM
        DY(ECATOMA(I))=DY(ECATOMA(I))+DYBRM
        DZ(ECATOMA(I))=DZ(ECATOMA(I))+DZBRM

        DX(ECATOMB(I))=DX(ECATOMB(I))-DXBRM
        DY(ECATOMB(I))=DY(ECATOMB(I))-DYBRM
        DZ(ECATOMB(I))=DZ(ECATOMB(I))-DZBRM

      ELSE

        CALL EPSEUDOMMPT(ECATOMA(I),ECATOMB(I),ECATOMC(I),ETMP,ECONSFC(I), &
        ECONSEQ(I)/180.d0*PI,ECIFLAG(I),DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

        EU=EU+ETMP

        DX(ECATOMA(I))=DX(ECATOMA(I))+DXAI
        DX(ECATOMB(I))=DX(ECATOMB(I))+DXAJ
        DX(ECATOMC(I))=DX(ECATOMC(I))+DXAK

        DY(ECATOMA(I))=DY(ECATOMA(I))+DYAI
        DY(ECATOMB(I))=DY(ECATOMB(I))+DYAJ
        DY(ECATOMC(I))=DY(ECATOMC(I))+DYAK

        DZ(ECATOMA(I))=DZ(ECATOMA(I))+DZAI
        DZ(ECATOMB(I))=DZ(ECATOMB(I))+DZAJ
        DZ(ECATOMC(I))=DZ(ECATOMC(I))+DZAK
      ENDIF

    ENDDO

  END SUBROUTINE ECONSTRAINT

  SUBROUTINE IMGTEST(HBI,INBLO,JNB,IMAT,IMBLO,IMJNB,NA,NATT,NMODE,ATMD,ATMH,ATMA)

    use dimens_fcm
    use comand
    use stream
    use string
    use code
    use coord
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER I,J,K,HBI,JSTART,JEND,INBLO(*),JNB(*),NA,NATT
    INTEGER IMAT(*),IMBLO(*),IMJNB(*)
    INTEGER NB2,NB3,NB4,NMODE,ATMD,ATMH,ATMA

    DO I=1,NBOND
      write(outu,*) 'TEST-BND>',IB(I),JB(I)
    ENDDO

    JSTART=1
    DO I=1,NA
      JEND=INBLO(I)
      DO J=JSTART,JEND
        write(outu,*) 'TEST-JNB>',i,jnb(j) &
        ,sqrt((x(i)-x(jnb(j)))**2+(y(i)-y(jnb(j)))**2+(z(i)-z(jnb(j)))**2)
      ENDDO
      JSTART=INBLO(I)+1
    ENDDO

    JSTART=1
    DO I=NA+1,NATT
      JEND=IMBLO(I)
      DO J=JSTART,JEND
        write(outu,*) 'TEST-IMG>',imat(i),IMJNB(J) &
        ,sqrt((x(i)-x(imjnb(j)))**2+(y(i)-y(imjnb(j)))**2+(z(i)-z(imjnb(j)))**2)
      ENDDO
      JSTART=IMBLO(I)+1
    ENDDO

  END SUBROUTINE IMGTEST

#else /**/
  SUBROUTINE NULL_MSMMPT
        RETURN
  END SUBROUTINE NULL_MSMMPT
#endif
!=====================================================================
! End of module MMPT
!=====================================================================
end module msmmpt_fcm
