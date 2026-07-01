module nose_mod
  use chm_kinds
  use memory
  use dimens_fcm
  implicit none
  !
  !     NOSE.FCM CONTAINS THE DATA STRUCTURES USED FOR 
  !     NOSE-HOOVER CONSTANT TEMPERATURE DYNAMICS  
  !
  !     QNOSE               THE  NOSE DYNAMICS LOGICAL FLAG
  !     QNOSP               THE  NOSE DYNAMICS LOGICAL FLAG
  !     SN11, SN12, SN13    EXTRA DEGREES OF FREEDOM      
  !                         FOR VERLET METHOD
  !     SNH,SNHV,SNHF       EXTRA DEGREES OF FREEDOM
  !                         FOR VELOCITY VERLET
  !     RTMPR               REFERENCE TEMPERATURE 
  !     SQM                 THERMAL INERTIA PARAMETER 
  !     NOBL                NUMBER OF HEAT BATH
  !     NSCYC               NUMBER OF CONVERGENCE LOOP
  !     NDGN                NUMBER OF DEGREE OF FREEDOM 
  !
  integer,allocatable,dimension(:) :: INLCKP
  integer,PARAMETER :: MAXNOS=10
  LOGICAL QNOSE,QNOSP  
  real(chm_real),dimension(maxnos) ::  SN11,SN12,SN13,SNH,SNHV,SNHF
  INTEGER NOBL,NSCYC
  real(chm_real),dimension(maxnos) ::  RTMPR,SQM,NDGN 
  real(chm_real)  TOLNS 
  ! BEGIN TPCONTROL (G. Lamoureux edited by Ed Harder)
  !     
  !     QTPCON              Flags for temperature and/or pressure control
  !     QTCON               Flag for temperature control
  !     QPCON               Flag for pressure control
  !     QZONLY              Flag to scale the volume along Z-axis only
  !     QPCONFULL           Flag to scale all directions independently
  !     QNHLANG             Flags for Langevin forces
  !     NHTAU               Timescales: Q = Nf kT tau^2
  !     YSNC,YSNYS          Parameters for Yoshida-Suzuki decomposition
  !     TOLSCF              Tolerance for GRMS on Drudes
  !     MAXSCF              Maximum number of SCF iterations
  LOGICAL QTPCON,QTCON,QPCON,QZONLY,QPCONFULL
  LOGICAL QNHLANG(maxnos)
  real(chm_real)  NHTAU(maxnos)
  INTEGER YSNC,YSNYS
  real(chm_real)  TOLSCF
  INTEGER MAXSCF
  !     
  !     Relative thermostats
  !     
  !     QRELT,QQRELT        Relative thermostats logical flags
  !     RELT_VX, RELT_VY, RELT_VZ, RELT_M
  !     IABST               Index of the absolute thermostat
  !                         1 <= IABST <= NOBL
  !                         IABST = 0 means no absolute thermostat
  !     ABST_VX, ABST_VY, ABST_VZ, ABST_M
  LOGICAL QRELT(MAXNOS),QQRELT
  real(chm_real)  RELT_VX(MAXNOS),RELT_VY(MAXNOS),RELT_VZ(MAXNOS)
  real(chm_real)  RELT_M(MAXNOS)
  INTEGER IABST
  real(chm_real)  ABST_VX,ABST_VY,ABST_VZ
  real(chm_real)  ABST_M
  !     
  !     Langevin thermostat
  !     
  LOGICAL QNHLANGEVIN
  real(chm_real)  NHGAMMA,NHGAMMAD
  !     
  !     Langevin absolute thermostat
  !     
  LOGICAL QCMLANGEVIN
  real(chm_real)  CMGAMMA
  !     
  !     Nose-Hoover chains
  !     
  !     QNHCH               Logical flag for Nose-Hoover chains
  !     NHCH_M              Number of thermostats in each chain (no chain: 1)
  !     NHCH_B              Branka non-equilibrium correction
  !     NHCH_Q              Inertia parameters (analog to SQM)
  !     NHCH_ETA            NH chain generalized coordinates (analog to SNH)
  !     NHCH_ZETA           NH chain generalized velocities (analog to SNHV)
  !     NHCH_G              NH chain generalized forces (analog to SNHF)
  integer,PARAMETER :: NHCH_MAXM = 5
  LOGICAL QNHCH
  INTEGER NHCH_M(maxnos)
  real(chm_real),dimension(maxnos) :: NHCH_B, NHCH_Q
  real(chm_real),dimension(maxnos,nhch_maxm) :: NHCH_ETA, NHCH_ZETA, NHCH_G
  !     
  !     Andersen-Hoover barostat
  !     
  !     QDSCALE_CP          Logical flag to scale Drude displacements
  !     P_CP                Targeted isotropic pressure (PREF)
  !     W_CP                Inertia parameter for isotropic barostat (WREF)
  !     TAU_CP              Timescale: W = Nf kT tau^2
  !     TI_CP               Index of thermostat coupled to the barostat
  !     ETA_CP              Barostat generalized coordinate
  !     ZETA_CP             Barostat generalized velocity
  !     G_CP                Barostat generalized force
  !     RX1_CP,RX2_CP       Scaling matrix for positions
  !     RV1_CP,RV2_CP       Scaling matrix for velocities
  LOGICAL QDSCALE_CP
  real(chm_real)  P_CP, W_CP, TAU_CP
  INTEGER TI_CP
  real(chm_real)  ETA_CP(3,3), ZETA_CP(3,3), G_CP(3,3)
  real(chm_real)  RX1_CP(3,3),RX2_CP(3,3), RV1_CP(3,3),RV2_CP(3,3)

contains

  subroutine nose_init()
    qnosp=.false.
    tolns=1.0d-10
    qnose=.false.
    nscyc=5
    ! begin tpcontrol (g. lamoureux)
    qtpcon = .false.
    qtcon = .false.
    qpcon = .false.
    ! end tpcontrol (g. lamoureux)
    !
    return
  end subroutine nose_init

end module nose_mod

SUBROUTINE NOSECT(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This routine interprets commands dealing with
  !     Nose constant temperature dynamics
  !     by Masa Watanabe
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  use dimens_fcm
  use exfunc
  use number
  use consta
  !
  use psf
  use stream
  use string
  use nose_mod
#if KEY_BLOCK==1
  use block_fcm, only : blasgn    
#endif
  use machutil,only:die

  implicit none
  integer,allocatable,dimension(:) :: ITEMP
  character(len=*) COMLYN
  INTEGER COMLEN
  !
  !     local
  LOGICAL     QEND, QCHECK, EOF, LUSED
  INTEGER     NINT, JINT, I, J, JTEMP
  character(len=4) WRD
  !
  !     begin
  QNOSP=.FALSE.
  QNOSE=.FALSE.
  EOF=.FALSE.
  QCHECK=.FALSE.
  QEND=.FALSE.
  IF (.NOT. QNOSP) THEN
     CALL NOSEKIN(COMLYN,COMLEN)
     IF(NOBL > MAXNOS) CALL DIE
     QNOSP=.TRUE.
     QNOSE=.FALSE.
  ENDIF
  DO  I=1,NOBL
     SQM(I)=RBIG
     RTMPR(I)=ROOMT
  enddo
  !
  lused = .true.
  do while(lused)
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
          EOF,.TRUE.,.TRUE.,'NOSE> ')
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD  ==  '    ') LUSED=.TRUE.
  enddo
  !
  loop20: do while (.true.)
     IF (WRD  ==  'CALL') THEN
        !     Procedure ASSIGN-A-BLOCK
        I=NEXTI(COMLYN,COMLEN)
        IF (I  <=  NOBL) THEN
#if KEY_BLOCK==1
           call chmalloc('nose.src','NOSECT','ITEMP',NATOM,intg=ITEMP)
           CALL BLASGN(COMLYN,COMLEN,ITEMP,NATOM,INLCKP,I)
           call chmdealloc('nose.src','NOSECT','ITEMP',NATOM,intg=ITEMP)
           QCHECK=.TRUE.
           IF(PRNLEV >= 2) WRITE(OUTU,30) I
30         FORMAT(' The selected atoms have been reassigned to block', &
                I4)
#endif 
        ELSE
           CALL WRNDIE(-3,'<NOSE>', &
                'Failed attempt to reassign atoms.  Block number too high.')
        ENDIF
        !
     ELSE IF (WRD  ==  'COEF') THEN
        I=NEXTI(COMLYN,COMLEN)
        SQM(I)=GTRMF(COMLYN,COMLEN,'QREF',ZERO)
        RTMPR(I)=GTRMF(COMLYN,COMLEN,'TREF',ROOMT)
        IF(SQM(I) == ZERO) SQM(I)=100.0
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,987) I
987        FORMAT(' The selected atom in block ',I4,' has')
           WRITE(OUTU,'(2(8X,A,F12.7,A,/))') &
                ' Reference temperature     = ',RTMPR(I),' K.', &
                ' Temp. Coupling Constant   = ',SQM(I),' Kcal sec**2.'
        ENDIF
        !
     ELSE IF (WRD  ==  'NCYC') THEN
        NSCYC=NEXTI(COMLYN,COMLEN)
        IF(NSCYC == 0) NSCYC=5
        !
     ELSE IF (WRD  ==  'TOL ') THEN
        TOLNS=NEXTF(COMLYN,COMLEN)
        WRITE(OUTU,988) TOLNS
988     FORMAT(' NOSE TOLERANCE = ',D10.3)
        !
     ELSE IF (WRD  ==  'END ') THEN
        !     Procedure FINISH-UP-AND-END
        CALL XTRANE(COMLYN,COMLEN,'NOSE')
        GOTO 700
        !
     ELSE IF (WRD  ==  'CLEA') THEN
        !     Procedure CLEAR-BLOCKING
        QNOSP=.FALSE.
        QNOSE=.FALSE.
        IF(PRNLEV >= 2) WRITE(OUTU,560)
560     FORMAT(' NOSE Structure has been turned off.')
        !
     ENDIF
     !
     CALL XTRANE(COMLYN,COMLEN,'NOSE')
     IF (QEND) THEN
        WRD='END '
     ELSE
690     CONTINUE
        CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
             EOF,.TRUE.,.TRUE.,'NOSE> ')
        CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
        WRD=NEXTA4(COMLYN,COMLEN)
        IF (WRD  ==  '    ') LUSED=.TRUE.
        IF (LUSED) GOTO 690
     ENDIF
  enddo loop20   !GOTO 20
  !
700 CONTINUE
  RETURN
END SUBROUTINE NOSECT
  
SUBROUTINE NOSINIT(NATOM,NBLOCK,IBLOCK)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE INITIALIZES THE SYSTEM TO ALL INTERACTION COEFFICI
  !     EQUAL TO ONE AND ALL ATOMS BELONGING TO BLOCK NUMBER ONE.
  !
  !     INPUT/OUTPUT
  use chm_kinds
  implicit none
  INTEGER NATOM,NBLOCK
  INTEGER IBLOCK(*)
  !     LOCAL
  INTEGER I
  !
  !     BEGIN
  IBLOCK(1:natom)=1
  RETURN
END SUBROUTINE NOSINIT

SUBROUTINE NOSEKIN(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     PROCEDURE DO-INITIALIZATION
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use string
  use dimens_fcm
  !
  use nose_mod
  use psf
  implicit none
  character(len=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NINT
  !
  NOBL=0
  NOBL=NEXTI(COMLYN,COMLEN)
  IF (NOBL  ==  0) NOBL=1
  IF (NOBL  >  0) THEN
     IF (.NOT.QNOSP) call chmalloc('nose.src','NOSEKIN','INLCKP',NATOM,intg=INLCKP)
     CALL NOSINIT(NATOM,NOBL,INLCKP)
  ENDIF
  ! SAPATEL
  RETURN
END SUBROUTINE NOSEKIN

#if KEY_CHEQ==1
SUBROUTINE NOSEFQ(COMLYN,COMLEN)
  !-----------------------------------------
  !   this routine sets up multiple heat baths for the fluctuating charge kinetics
  !
  !   adopted from algorithm by M. Watanabe
  !   Sandeep Patel
  !
  !-------------------------------------

  use cheq, only: FQNHSBATH,FQNHSOBATH,FQNHMBATH,FQTEMPBATH, &
       NDGFBATH,cheqbasetup,nqbaths,ifirstbath,ilastbath, &
       allocate_cheq

  use chm_kinds
  !
  use dimens_fcm
  use exfunc
  use number
  !
  use psf
  use stream
  use nose_mod
  use string

  implicit none
  integer,allocatable,dimension(:) :: ITEMP
  character(len=*) COMLYN
  INTEGER COMLEN
  !
  !     local
  LOGICAL     QEND, QCHECK, EOF, LUSED
  INTEGER     NINT, JINT, I, J, K, JTEMP, QDEGF
  character(len=4) WRD
  real(chm_real)      ROOMT
  PARAMETER (ROOMT=298.0D0)

  call allocate_cheq(natom,ngrp)
  CHEQBASETUP = .FALSE.
  EOF = .FALSE.
  QCHECK = .FALSE.
  QEND = .FALSE.

  !     get number of baths
  NQBATHS = NEXTI(COMLYN,COMLEN)

  if (PRNLEV >= 2) write(OUTU,*) " NUMBER OF BATHS = ", NQBATHS

  !     initialize some variables for each bath
  DO K = 1, NQBATHS
     FQNHSBATH(K)=1.0
     FQNHSOBATH(K)=1.0
     FQNHMBATH(K)=0.005
     FQTEMPBATH(K)=1.0
  ENDDO


10 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
       EOF,.TRUE.,.TRUE.,'FQBA> ')
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  WRD=NEXTA4(COMLYN,COMLEN)
  IF (WRD  ==  '    ') LUSED=.TRUE.
  IF (LUSED) GOTO 10
  !
20 CONTINUE
  IF (WRD  ==  'CALL') THEN
     if (prnlev >= 2) write(outu,*) " CALLING "
     !     Procedure ASSIGN-A-BATH
     I=NEXTI(COMLYN,COMLEN)
     IF (I  <=  NQBATHS) THEN
        call chmalloc('nose.src','NOSEFQ','ITEMP',NATOM,intg=ITEMP)
        if (prnlev >= 2) write(outu,*) " CALL ASSIGN BATH "

        CALL FQBATHASGN(COMLYN,COMLEN,ITEMP,NATOM,INLCKP,I,NRES,IBASE,QDEGF)

        NDGFBATH(I) = QDEGF

        if (prnlev >= 2) write(outu, '(A, 2I8)')  &
             "NUMBER OF DEG. OF Q FREEDOM for BAth =",I,QDEGF
        call chmdealloc('nose.src','NOSEFQ','ITEMP',NATOM,intg=ITEMP)
        if (prnlev >= 2) write(outu,*) " DONE ASSIGN Q's"
        QCHECK=.TRUE.
        IF(PRNLEV >= 2) WRITE(OUTU,30) I
30      FORMAT(' The selected atoms have been reassigned to block', &
             I4)
     ELSE
        CALL WRNDIE(-3,'<NOSE>', &
             'Failed attempt to reassign atoms.  Block number too high.')
     ENDIF


     !
  ELSE IF (WRD  ==  'COEF') THEN
     I=NEXTI(COMLYN,COMLEN)
     FQNHMBATH(I)=GTRMF(COMLYN,COMLEN,'QREF',ZERO)
     FQTEMPBATH(I)=GTRMF(COMLYN,COMLEN,'TREF',ROOMT)


  ELSE IF (WRD  ==  'END ') THEN
     !     Procedure FINISH-UP-AND-END
     CALL XTRANE(COMLYN,COMLEN,'FQBA')
     GOTO 700

  ELSE
  ENDIF

  CALL XTRANE(COMLYN,COMLEN,'FQBA')
  IF (QEND) THEN
     WRD='END '
  ELSE
690  CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
          EOF,.TRUE.,.TRUE.,'FQBA> ')
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD  ==  '    ') LUSED=.TRUE.
     IF (LUSED) GOTO 690
  ENDIF
  GOTO 20


700 CONTINUE


  DO K = 1,NQBATHS
     IF (PRNLEV >= 2) write(outu,'(I8,4F16.6,3I8)')  &
          K,FQNHMBATH(K),FQTEMPBATH(K),FQNHSBATH(K), &
          FQNHSOBATH(K), IFIRSTBATH(K),ILASTBATH(K), NDGFBATH(K)
  ENDDO



  CHEQBASETUP = .TRUE.

  RETURN
END SUBROUTINE NOSEFQ

SUBROUTINE FQBATHASGN(COMLYN,COMLEN,ITEMP,NATOM,IBLOCK,INDEX, &
     NRES, IBASE, QDEGF)

  !-------------------------------------------------------
  !    this subroutine handles assignment of charges to specific heat
  !    baths
  !
  !     adopted algorithm from M. Watanabe
  !   Sandeep Patel

  !-----------------------------------------------------------

  use cheq,only: ILASTBATH,IFIRSTBATH

  use chm_kinds
  use dimens_fcm
  use coord
  use select
  use stream
  implicit none
  INTEGER ITEMP(*), IBLOCK(*),IBASE(*)
  INTEGER NATOM, COMLEN, INDEX, ICOUNT,ILASTATM
  character(len=*) COMLYN
  !     LOCAL
  INTEGER I
  INTEGER NFQ,NFQR,QDEGF,NRES,J
  LOGICAL  INRES

  !
  !     BEGIN

  CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)
  ICOUNT=0
  DO I=1, NATOM
     IF (ITEMP(I)  ==  1) THEN
        ICOUNT=ICOUNT+1
        ILASTATM=I
     ENDIF
  ENDDO

  ILASTBATH(INDEX) = ILASTATM
  IFIRSTBATH(INDEX) = ILASTATM - ICOUNT + 1
  !

  !   get number of degrees of freedom

  !C degrees of freedom=no. of FQ atoms-no. of restraints (residues)
  QDEGF = 0
  NFQ=0
  NFQR=0
  DO I=1,NRES
     INRES=.FALSE.
     DO J=IBASE(I)+1,IBASE(I+1)
        IF (ITEMP(J) /= 0) THEN
           NFQ=NFQ+1
           INRES=.TRUE.
        ENDIF
     ENDDO
     IF (INRES) NFQR=NFQR+1
  ENDDO
  IF ((NFQ-NFQR) <= 0) THEN
     CALL WRNDIE(-4,'<FQDEGF>', &
          'Number of charge degrees of freedom zero or negative!')
     QDEGF=0
  ELSE
     QDEGF=NFQ-NFQR
     !        QDEGF = NFQ
     IF (PRNLEV >= 5) WRITE(OUTU,10) NFQ-NFQR
10   FORMAT(' FQDEGF> System has ',I8, &
          ' degrees of charge freedom')
  ENDIF

  RETURN
END SUBROUTINE FQBATHASGN


!----------------------------------------------------------------
!
SUBROUTINE CHEQDEGF(COMLYN,COMLEN,ITEMP,NATOM,IBLOCK,INDEX, &
     NRES,IBASE,QDEGF)
  !
  !
  !
  !    Compute charge degrees of freedom for a Nose-Hoover bath
  !    Adapted from B. Webb's code for fluq
  !
  !
  use chm_kinds
  use dimens_fcm
  use coord
  use select
  use stream
  implicit none
  INTEGER ITEMP(*), IBLOCK(*), IBASE(*)
  INTEGER NATOM, COMLEN, INDEX, ICOUNT,ILASTATM
  character(len=*) COMLYN
  !C     LOCAL
  INTEGER I,NFQ,NFQR,QDEGF,NRES,J
  LOGICAL  INRES
  !C
  !C     BEGIN

  CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)



  !C degrees of freedom=no. of FQ atoms-no. of restraints (residues)
  NFQ=0
  NFQR=0
  DO I=1,NRES
     INRES=.FALSE.
     DO J=IBASE(I)+1,IBASE(I+1)
        IF (ITEMP(J) /= 0) THEN
           NFQ=NFQ+1
           INRES=.TRUE.
        ENDIF
     ENDDO
     IF (INRES) NFQR=NFQR+1
  ENDDO
  IF ((NFQ-NFQR) <= 0) THEN
     CALL WRNDIE(-4,'<FQDEGF>', &
          'Number of charge degrees of freedom zero or negative!')
     QDEGF=0
  ELSE
     !         QDEGF=NFQ-NFQR
     QDEGF = NFQ
     IF (PRNLEV >= 5) WRITE(OUTU,10) NFQ-NFQR
10   FORMAT(' FQDEGF> System has ',I8, &
          ' degrees of charge freedom')
  ENDIF
  RETURN
END SUBROUTINE CHEQDEGF
#endif 

