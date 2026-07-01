module umb
  use chm_kinds
  use dimens_fcm

  implicit none
#if KEY_ADUMB==1 /*adumb*/
  !     Adaptiv Umbrella Potentials
  !
  !     C. Bartels, 1996/1997/1998
  !
  !     Purpose:
  !
  !     To store adaptiv umbrella potentials           
  !
  !
  !  GENERAL UMBRELLA
  !     STONUM     Whether to record statistics or not
  !     WUNUM      Unit for writing out data
  !     RUNUM      Unit for reading histograms
  !     WCUNUM     Unit for writing umbrella coordinates
  !     RCUNUM     Unit for reading umbrella coordinates
  !     WPUNUM     Unit for writing probability factors
  !     WTUNUM     Unit for writing info for themodynamic analysis
  !     NUMBR      Number of defined umbrellas
  !     FUPUM      Frequency for updating forces
  !     NEQUM      Number of equilibration steps
  !     NSIUM      Number of simulations to store
  !     ICUM       Number of calls
  !     STMFUM     Stack location for potential of mean force
  !     STSFUM     Stack location for scaling factor
  !     STEMUM     Stack location for number of counts in each bin
  !     STCAUM     Stack location to use for calulations
  !     STLAUM     Stack location for lagrange multipliers
  !     STFPUM     Stack location for fit parameters
  !     INSTUM     Increments to address stack
  !     INSIUM     Stack Increment for simulations
  !     ISIUM      Pointer to current simulation
  !     TYPUM      Type of umbrella
  !                1 ... Dihedral Angle
  !                2 ... Energy
  !                5 ... Harmonic constraint
  !     NBUM       Number of bins for statistics
  !     TFUM       Number of trigonometrical fit functions
  !     PFUM       Number of polynomial fit functions
  !     COUM       Internal coordinates for umbrella
  !     EUM        Energy of potential
  !     DUM        Derivative of potential to internal coordinates
  !     TUM        Temperature used in simulation 
  !     TEUM       Temperature to use for evaluation
  !     MAXDEV     Threshold to throw out simulations
  !     AGEUM      Aging factor
  !     NEXUM      Number of bins to extrapolate
  !     ADFREQ     Frequency for data collection from dynamics
  ! DATA STRUCTURES NECESSARY FOR EACH OF THE RUNS
  !     STAGUM     Stack location for age factor: old histogram = less important
  !     STHIUM     Stack location for histograms
  !     STUPUM     Stack location for umbrella potential
  !     STDAUM     Stack location of free energies of different simulation
  !     STERUM     Stack location for deviation from average distribution
  !     STFLUM     Stack location for flags indicating whether to use the run
  !                when deriving the PMF
  !     STNSUM     Stack location for number of MD steps
  !
  !
  !  DIHEDRAL UMBRELLAS
  !     NUMPHI     Number of dihedral umbrellas
  !     IUM        First atom of dihedral umbrella
  !     JUM        Second atom of dihedral umbrella
  !     KUM        Third atom of dihedral umbrella
  !     LUM        Fourth atom of dihedral umbrella
  !     IPHIUM     Index of corresponding general umbrella
  !
  !
  !  ENERGY UMBRELLA
  !     IEUM       Index of corresponding general umbrella
  !     EMAXUM     Boundaries beyond which the umbrella potential should
  !     EMINUM     stay approximately constant
  !     TMAXUM     temperatures between which to perform the simulation
  !     TMINUM     
  !
  !  NOE CONSTRAINT UMBRELLA
  !     NNUM       Number of noe data sets
  !     STFNUM     Stack location of forces
  !     ENUM       Energy
  !     LLOWNU     lower and upper limits after which potential should be
  !     LUPPNU     kept constant
  !     variables of type real for NOE data sets (see also noe.fcm)
  !         SCANUM
  !     variables of type integer for NOE data sets (see also noe.fcm)
  !         NUMNUM, NM2NUM
  !     Stack location of NOE integer data structures (see also noe.fcm)
  !         IPTNUM, JPTNUM, INMNUM, JNMNUM, LISNUM
  !     Stack location of NOE real data structures (see also noe.fcm)
  !         RMNNUM, KMNNUM, RMXNUM, KMXNUM, FMXNUM, TCNNUM, AVENUM,
  !         EXPNUM, MINNUM
  !     INUM       Index of corresponding general umbrella
  !     NORMNU     Normalization factor for ENUM
  !     EMAXNU     limits for bins  
  !     EMINNU
  !     ELIMNU     dacay constant umb. coord.=ENUM/(ENUM+ELIMNU)
  !
#if KEY_ADUMB==1
    !  FROM:  umb.fcm
    !
    !  MAXUMP - The maximum number of adaptive umbrella potentials.
    !  MAXNUM - Maximum number of NOE data sets for NOE constraints
    !  C.Bartels, 1996/1998
    !
    integer,parameter :: MAXUMP = 10, MAXNUM = 4
#endif 
  !  INTEGERS
  type umb_arr_rl
     real(chm_real),allocatable,dimension(:) :: a
  end type umb_arr_rl

  type umb_arr_3d
     real(chm_real),allocatable,dimension(:) :: x,y,z
  end type umb_arr_3d

  type umb_arr_intg
     integer,allocatable,dimension(:) :: a
  end type umb_arr_intg

  type umb_arr_log
     logical,allocatable,dimension(:) :: a
  end type umb_arr_log

  type (umb_arr_rl),dimension(maxnum):: RMNNUM, KMNNUM, RMXNUM, KMXNUM, FMXNUM,  &
       TCNNUM, AVENUM, EXPNUM, RSWNUM,SEXNUM

  type (umb_arr_3d),dimension(maxnum):: STFNUM

  type (umb_arr_intg),dimension(maxnum):: IPTNUM, JPTNUM, INMNUM, JNMNUM, LISNUM,RAMNUM

  type (umb_arr_log),dimension(maxnum):: MINNUM

  integer,allocatable,dimension(:,:) :: STHIUM 
  real(chm_real),allocatable,dimension(:,:) :: STUPUM 
  integer,allocatable,dimension(:) :: STNSUM,STFLUM
  real(chm_real),allocatable,dimension(:) :: STEMUM,STCAUM,STERUM,STAGUM
  real(chm_real),allocatable,dimension(:) ::STFPUM,STDAUM,STMFUM,STSFUM

  integer :: NUMPHI, IUM(MAXUMP), JUM(MAXUMP), KUM(MAXUMP),  &
       LUM(MAXUMP), NBUM(MAXUMP), TFUM(MAXUMP),  &
       PFUM(MAXUMP), IPHIUM(MAXUMP), NUMBR, &
       INSTUM(MAXUMP+1), ICUM, FUPUM, NEQUM, INSIUM, ISIUM, &
       NSIUM, WUNUM, RUNUM, TYPUM(MAXUMP), IEUM, WCUNUM, &
       RCUNUM, WPUNUM, WTUNUM, &
       STLAUM, NNUM, INUM,  &
       NUMNUM(MAXNUM), NM2NUM(MAXNUM), &
       NEXUM, ADFREQ
  !
  ! REAL
  real(chm_real) :: COUM(MAXUMP), DUM(MAXUMP), EUM, TUM, EMAXUM, EMINUM, &
       TEUM, TMAXUM, TMINUM, MAXDEV, AGEUM, MINRMS, &
       MAXRMS, ENUM(MAXNUM), NORMNU(MAXNUM),  &
       EMAXNU, EMINNU, SCANUM(MAXNUM), ELIMNU(MAXNUM),  &
       LUPPNU, LLOWNU
  !
  ! LOGICALS
  LOGICAL STONUM
#if KEY_ADUMBRXNCOR==1
! numbrxn -- number of reaction coordinate umbrellas
! rxncorindex -- index in the reaction coordinate list of coordinate i
! umbindex -- index in the umbrella coordinate list of coordinate i
      integer :: numbrxn, rxncorindex(1:maxump), umbindex(1:maxump)
      real(chm_real) :: umbmin(1:maxump),umbmax(1:maxump),maxbias
#endif 

contains

  subroutine adumb_iniall()
    numbr = 0
#if KEY_ADUMBRXNCOR==1
    numbrxn = 0 
#endif
    adfreq = 1
    stonum=.true.
    return
  end subroutine adumb_iniall

#endif /* (adumb)*/
end module umb


#if KEY_ADUMB==1 /*adumb*/
!
! Routines valid for all types of umbrellas
!
SUBROUTINE UMINIT()
  !
  !     Initialize data structures
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use umb
  use stream
  use umbcor
  use memory

  implicit none

  INTEGER I,TS

  ! allocate stack space for statistics

  ICUM=0
  ISIUM=1
  IF(.not. allocated(STMFUM)) THEN
     TS=1
     DO I=1,NUMBR
        INSTUM(I)=TS
        TS=TS*NBUM(I)
     enddo
     INSTUM(NUMBR+1)=TS
     INSIUM=TS

     call chmalloc('umb.src','UMINIT','STHIUM',TS,NSIUM,intg=STHIUM)
     call chmalloc('umb.src','UMINIT','STUPUM',TS,NSIUM,crl=STUPUM)
     call chmalloc('umb.src','UMINIT','STFPUM',TS,crl=STFPUM)
     call chmalloc('umb.src','UMINIT','STDAUM',NSIUM,crl=STDAUM)
     call chmalloc('umb.src','UMINIT','STNSUM',NSIUM,intg=STNSUM)
     call chmalloc('umb.src','UMINIT','STMFUM',TS,crl=STMFUM)
     call chmalloc('umb.src','UMINIT','STSFUM',TS,crl=STSFUM)
     call chmalloc('umb.src','UMINIT','STEMUM',TS,crl=STEMUM)
     call chmalloc('umb.src','UMINIT','STCAUM',TS,crl=STCAUM)
     call chmalloc('umb.src','UMINIT','STERUM',NSIUM,crl=STERUM)
     call chmalloc('umb.src','UMINIT','STAGUM',NSIUM,crl=STAGUM)
     call chmalloc('umb.src','UMINIT','STFLUM',NSIUM,intg=STFLUM)

  ELSE
     IF(WRNLEV >= 2.AND.PRNLEV > 2) WRITE(OUTU,170)
170  FORMAT(' Adaptive umbrella potential existed already')
  ENDIF

  call UMINI2(STHIUM,STUPUM,STDAUM,STNSUM,STMFUM, &
       STEMUM,STFPUM,STFLUM,STSFUM,STAGUM)

  RETURN
end SUBROUTINE UMINIT
!
!-----------------------------------------------------------------------

SUBROUTINE UMINI2(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10)

  !     Initialize statistics

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use number
  use umb
  implicit none
  INTEGER NBIN,NSIM
  real(chm_real)  M2(INSIUM,NSIUM),M3(NSIUM),M5(INSIUM),M6(INSIUM)
  real(chm_real)  M7(INSIUM),M9(INSIUM),M10(NSIUM)
  INTEGER M1(INSIUM,NSIUM),M4(NSIUM),M8(NSIUM)
  !
  INTEGER I,j
  M10(1:nsium)=one
  M3(1:nsium)=one
  M4(1:nsium)=0
  M8(1:nsium)=0
  DO I=1,NSIUM
     M1(1:insium,I)=0
     M2(1:insium,I)=one
  enddo

  M5(1:insium)=zero
  M6(1:insium)=zero
  M7(1:insium)=zero
  M9(1:insium)=zero

  RETURN
END SUBROUTINE UMINI2
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMINI3(M1,M3,M4,M5,CRRDSM,RMSSUM1, &
     RMSSUM2)
  !
  !     Initialize statistics
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use umb
  use umbcor
  use number
  implicit none
  INTEGER M1(INSIUM,NSIUM),M3(NSIUM),M4(NSIUM)
  real(chm_real)  M5(NSIUM),CRRDSM(INSIUM,NDISTA)
  real(chm_real)  RMSSUM1(INSIUM,NRMSDI)
  real(chm_real)  RMSSUM2(INSIUM,NRMSDI)
  !
  INTEGER J,IS,JJ
  !
  M3(ISIUM)=0
  M4(ISIUM)=0
  M5(ISIUM)=1.0

  M1(1:insium, ISIUM)=0

  ! aging of histograms
  M5=M5*AGEUM

  CRRDSM = zero
  RMSSUM1 = zero
  RMSSUM2 = zero
  RETURN
END SUBROUTINE UMINI3

!-----------------------------------------------------------------------

SUBROUTINE UMSTAT(MPA,MUP,MSF,MHI,MEM,CRRDSM,TOAVDI,RMSSUM1, &
     RMSSUM2,TORMSD1,TORMSD2)

  !     update statistics

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use number
  use umb
  use stream
  use umbcor
  !
  implicit none
  real(chm_real)  MUP(INSIUM,NSIUM),MPA(INSIUM),MSF(INSIUM), &
       MEM(INSIUM)
  real(chm_real)  CRRDSM(INSIUM,NDISTA),TOAVDI(INSIUM,NDISTA)
  real(chm_real)  RMSSUM1(INSIUM,NRMSDI),TORMSD1(INSIUM,NRMSDI)
  real(chm_real)  RMSSUM2(INSIUM,NRMSDI),TORMSD2(INSIUM,NRMSDI)
  INTEGER MHI(INSIUM,NSIUM)
  !
  INTEGER TMP,I,DIM,IA(MAXUMP),DIMA
  integer,PARAMETER :: NA=10
  real(chm_real)  RT,X(MAXUMP),E,EA,Y(MAXUMP),EAI
  !
  ! for correlations -RJP 6.16.01
  INTEGER CRUNIT,J,JJ,NDSSTK,RC1STK,RC2STK
  INTEGER CRUNIT1,CRUNIT2,TPFREQ
  real(chm_real) CURAVG,TOTAVG
  real(chm_real) CURAVG1,TOTAVG1
  real(chm_real) CURAVG2,TOTAVG2
  !
  ! **********************************************************
  !
  IF (LCORRV) THEN
     IF (LCNEED) CALL WRNDIE(-5,'<UMSTAT>', &
          'SPECIFY CORRELATIONS BEFORE UMBR INIT COMMAND')
  ENDIF
  RT=KBOLTZ*TUM
  IF(.not. allocated(STHIUM)) THEN
     IF(WRNLEV >= 2.AND.PRNLEV > 2) WRITE(OUTU,170)
170  FORMAT(' Adaptive umbrella potential is not initialized')
  ELSE
     ICUM=ICUM+1
     IF(MOD(ICUM,FUPUM) > NEQUM) THEN
        call UMSTA2(STHIUM,STNSUM,HPCRRD,HPNDIS,HPCRMS1, &
             HPCRMS2,HPCOR1,HPCOR2,HPRMEN,HPRMLS,HPOREN, &
             HPORLS)

     ENDIF
     IF(MOD(ICUM,FUPUM) == 0) THEN
        ! calc factor for umbrella potential from parameters
        DO I=1,INSIUM
           DO DIM=1,NUMBR
              X(DIM)=(FLOAT(MOD(I-1,INSTUM(DIM+1))/INSTUM(DIM))+0.5) &
                   /FLOAT(NBUM(DIM))
           enddo
           E=-(UMFI(MPA,X,0,STCAUM)/RT-MSF(I))
           IF(MHI(I,ISIUM) > 0.AND.MEM(I) == 0.AND.MSF(I).EQ.0.0)THEN
              MSF(I)= -E
              E=0.0
              IF(MSF(I) == 0.0) MSF(I)=1.0E-20
           END IF
           IF(MHI(I,ISIUM) > 0.OR.MEM(I).GT.0) THEN
              MUP(I,ISIUM)=EXP(-E)
           ELSE
              MUP(I,ISIUM)=1.0E-20
           END IF
        enddo
        if( .not. allocated(sthium)) then
           call wrndie(-3,"umstat<umb.src>","sthium not allocated")
        endif
        call UMSTA3(STHIUM,STMFUM,STEMUM,STFPUM,STDAUM, &
             STNSUM,STUPUM,STERUM,STFLUM,STSFUM,STAGUM)

        call UMWRT(STHIUM,STMFUM,STEMUM,STFPUM,STUPUM, &
             STSFUM)

        IF(WUNUM > 0) THEN
           call UMWFI(STHIUM,STMFUM,STEMUM,STFPUM,STUPUM, &
                STSFUM)

        ENDIF
        !------------------------------------------------------
        ! begin correlated variables
        !------------------------------------------------------
        IF(LCORRV) THEN
           TPFREQ = FUPUM*PCORMD
           ! ----------------------------------------------------
           ! sum up correlations and print if appropriate -RJP
           ! calculate the totals for the histograms
           !  distances
           DO I=1,INSIUM
              DO J = 1,NDISTA
                 TOAVDI(I,J) = TOAVDI(I,J) + CRRDSM(I,J)
              ENDDO
              DO J = 1,NRMSDI
                 TORMSD1(I,J) = TORMSD1(I,J) + RMSSUM1(I,J)
                 TORMSD2(I,J) = TORMSD2(I,J) + RMSSUM2(I,J)
              ENDDO
           ENDDO
           ! print out stuff
           IF((PRNLEV >= 2).AND. &
                (MOD(ICUM,TPFREQ) == 0)) THEN
              IF (LPRCRU) THEN !writing to files
                 DO J = 1,NDISTA
                    CRUNIT = CDISUN(J)
                    WRITE(CRUNIT,'(A27,1x,I6,1x,A2,1x,I6,1x, A7,1x,I14)') &
                         'Average vals of distance fr',DISIAT(J),'to',DISJAT(J), &
                         'at step',ICUM
                    DO I=1,INSIUM
                       IF(MHI(I,ISIUM) > 0) THEN
                          CURAVG = CRRDSM(I,J)/MHI(I,ISIUM)
                       ELSE
                          CURAVG = -1
                       ENDIF
                       IF(MEM(I) > 0) THEN
                          TOTAVG = TOAVDI(I,J)/MEM(I)
                       ELSE
                          TOTAVG = -1
                       ENDIF
                       WRITE(CRUNIT,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            J,I,CURAVG,TOTAVG
                    ENDDO !loop over gridpoints
                 ENDDO !loop over distances
              ELSE  !write to standard output
                 DO J = 1,NDISTA
                    WRITE(OUTU,'(A27,1x,I6,1x,A2,1x,I6,1x, A7,1x,I14)') &
                         'Average vals of distance fr',DISIAT(J),'to',DISJAT(J), &
                         'at step',ICUM
                    DO I=1,INSIUM
                       IF(MHI(I,ISIUM) > 0) THEN
                          CURAVG = CRRDSM(I,J)/MHI(I,ISIUM)
                       ELSE
                          CURAVG = -1
                       ENDIF
                       IF(MEM(I) > 0) THEN
                          TOTAVG = TOAVDI(I,J)/MEM(I)
                       ELSE
                          TOTAVG = -1
                       ENDIF
                       WRITE(OUTU,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            J,I,CURAVG,TOTAVG
                    ENDDO !loop over gridpoints
                 ENDDO  !loop over distance variables
              ENDIF
              ! loop over rmsd's
              IF(LPRRMU) THEN !if writing to files
                 DO JJ = 1,NRMSDI
                    CRUNIT1 = CRMSU1(JJ)
                    CRUNIT2 = CRMSU2(JJ)
                    WRITE(CRUNIT1,'(A34,1x,I6,1x,A7,1x,I14)') &
                         'Average RMSDs from Ref #1 for set ',JJ,'at step',ICUM
                    WRITE(CRUNIT2,'(A34,1x,I6,1x,A7,1x,I14)') &
                         'Average RMSDs from Ref #2 for set ',JJ,'at step',ICUM
                    DO I = 1,INSIUM
                       IF(MHI(I,ISIUM) > 0) THEN
                          CURAVG1 = RMSSUM1(I,JJ)/MHI(I,ISIUM)
                          CURAVG2 = RMSSUM2(I,JJ)/MHI(I,ISIUM)
                       ELSE
                          CURAVG1 = -1
                          CURAVG2 = -1
                       ENDIF
                       IF(MEM(I) > 0) THEN
                          TOTAVG1 = TORMSD1(I,JJ)/MEM(I)
                          TOTAVG2 = TORMSD2(I,JJ)/MEM(I)
                       ELSE
                          TOTAVG1 = -1
                          TOTAVG2 = -1
                       ENDIF
                       WRITE(CRUNIT1,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            JJ,I,CURAVG1,TOTAVG1
                       WRITE(CRUNIT2,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            JJ,I,CURAVG2,TOTAVG2
                    ENDDO
                 ENDDO
              ELSE !if writing to standard output
                 ! loop over rmsd's
                 ! do comparisons with first reference structure
                 DO JJ = 1,NRMSDI
                    WRITE(OUTU,'(A34,1x,I6,1x,A7,1x,I14)') &
                         'Average RMSDs from Ref #1 for set ',JJ,'at step',ICUM
                    DO I = 1,INSIUM
                       IF(MHI(I,ISIUM) > 0) THEN
                          CURAVG1 = RMSSUM1(I,JJ)/MHI(I,ISIUM)
                       ELSE
                          CURAVG1 = -1
                       ENDIF
                       IF(MEM(I) > 0) THEN
                          TOTAVG1 = TORMSD1(I,JJ)/MEM(I)
                       ELSE
                          TOTAVG1 = -1
                       ENDIF
                       WRITE(OUTU,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            JJ,I,CURAVG1,TOTAVG1
                    ENDDO  !loop over gridpoints
                 ENDDO !over substructures
                 ! do comparisons with second reference structure
                 DO JJ = 1,NRMSDI
                    WRITE(OUTU,'(A34,1x,I6,1x,A7,1x,I14)') &
                         'Average RMSDs from Ref #2 for set ',JJ,'at step',ICUM
                    DO I = 1,INSIUM
                       IF(MHI(I,ISIUM) > 0) THEN
                          CURAVG2 = RMSSUM2(I,JJ)/MHI(I,ISIUM)
                       ELSE
                          CURAVG2 = -1
                       ENDIF
                       IF(MEM(I) > 0) THEN
                          TOTAVG2 = TORMSD2(I,JJ)/MEM(I)
                       ELSE
                          TOTAVG2 = -1
                       ENDIF
                       WRITE(OUTU,'(I8,1X,I8,1X,F14.8,1X,F14.8)') &
                            JJ,I,CURAVG2,TOTAVG2
                    ENDDO  !loop over gridpoints
                 ENDDO !over substructures
              ENDIF ! if writing to files or standard output
           ENDIF !if prnlev gt 2
           ! C write out correlated structural variable accumulators
           IF ((WCRUNI > 0).AND.(MOD(ICUM,TPFREQ) == 0)) THEN
              call UMWRCOR(HPTOAV,HPTORM1,HPTORM2)

           ENDIF
        ENDIF !if correlations
        !  end correlated variables ------------------------------------
        ISIUM=MOD(ISIUM,NSIUM)+1
        call UMINI3(STHIUM,STNSUM,STFLUM,STAGUM,HPCRRD, &
             HPCRMS1,HPCRMS2)

     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE UMSTAT

!-----------------------------------------------------------------------

SUBROUTINE UMSTA2(MHI,MNS,CRRDSM,DISTAN,RMSSUM1, &
     RMSSUM2,RMSCOR1,RMSCOR2,RMAEND,RMSLST,ORIEND, &
     ORILST)

  !     Add point to statistics

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use umb
  use umbcor

  implicit none
  INTEGER MHI(INSIUM,NSIUM),MNS(NSIUM)
  real(chm_real) CRRDSM(INSIUM,NDISTA)
  real(chm_real) DISTAN(NDISTA),RMSCOR1(NRMSDI)
  real(chm_real) RMSCOR2(NRMSDI)
  real(chm_real) RMSSUM1(INSIUM,NRMSDI)
  real(chm_real) RMSSUM2(INSIUM,NRMSDI)
  INTEGER RMAEND(*),RMSLST(*)
  INTEGER ORIEND(*),ORILST(*)
  INTEGER CPXSTK,CPYSTK,CPZSTK,ATMNSK
  INTEGER C2XSTK,C2YSTK,C2ZSTK

  INTEGER I,IND

  IND=0
  DO I=1,NUMBR
     IF((COUM(I) < 0.0) .OR. (COUM(I) >= 1.0)) return
     IND=IND+INT(COUM(I)*NBUM(I))*INSTUM(I)
  enddo
  MHI(IND+1,ISIUM)=MHI(IND+1,ISIUM)+1
  MNS(ISIUM)=MNS(ISIUM)+1
  ! ************************
  IF(LCORRV) THEN !RJP
     call CORRLVAR(DISTAN,RMSCOR1,RMSCOR2,RMAEND,RMSLST,ORIEND,ORILST,IND, &
          HPCO1X,HPCO1Y,HPCO1Z,HPCO2X,HPCO2Y,HPCO2Z, &
          HPCPYX,HPCPYY,HPCPYZ,HPATMN,HPLORI,HPSYMN, &
          HPSYDL,HPXCPS,HPYCPS,HPZCPS,HPTEMX,HPTEMY, &
          HPTEMZ,HPTE2X,HPTE2Y,HPTE2Z,HPMNIC,HPMXIC, &
          HPSYLS,HPSYEN)

     DO I = 1,NDISTA
        CRRDSM(IND+1,I) = CRRDSM(IND+1,I) + DISTAN(I)
     ENDDO
     DO I = 1,NRMSDI
        RMSSUM1(IND+1,I) = RMSSUM1(IND+1,I) + RMSCOR1(I)
        RMSSUM2(IND+1,I) = RMSSUM2(IND+1,I) + RMSCOR2(I)
     ENDDO
  ENDIF
  ! ***********************

  RETURN
END SUBROUTINE UMSTA2

!-----------------------------------------------------------------------

SUBROUTINE UMSTA3(MHI,MMF,MEM,MPA,MDA,MNS,MUP,MER,MFL,MSF,MAG)

  !     Calculate potential of mean force and fit parameters

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  use umbcor
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:),target :: Wtarg
  real(chm_real),pointer,dimension(:) :: WS=>null(),TS1=>null(),TS2=>null(),TS3=>null()
  real(chm_real),allocatable,dimension(:,:) :: US,VS      

  INTEGER MHI(INSIUM,NSIUM),MNS(NSIUM),MFL(NSIUM)
  real(chm_real)  MMF(INSIUM),MEM(INSIUM),MUP(INSIUM,NSIUM), &
       MDA(NSIUM)
  real(chm_real)  MPA(INSIUM),MER(NSIUM),MSF(INSIUM),MAG(NSIUM)
  !
  real(chm_real),parameter ::  EPS=1.0
  INTEGER, PARAMETER ::MAXIT=500
  !
  INTEGER I,IT,DIM,IS,NCNT,NP,AVN,NDEL,DELIT
  INTEGER N,M,S(MAXUMP)
  real(chm_real)  FACT,EN,DEV,AVDEV,E,RT,ET,MINF,SUM
  LOGICAL MINSET
  INTEGER DUMY !RJP

  !
  ! mark invalid points
  RT=KBOLTZ*TUM
  DO I=1,INSIUM
     IF((MHI(I,ISIUM) == 0).AND.(MEM(I).EQ.0)) MHI(I,ISIUM)=-1
  enddo
  IF(ISIUM > 1) MDA(ISIUM)=MDA(ISIUM-1)
  ! DO UNTIL CONVERGENCE
  DELIT=0
7 loop100: DO  IT=1,MAXIT
     DEV=0.0
     NCNT=0
     ! normalize
     FACT=MDA(1)
     loop35: DO IS=1,NSIUM
        IF (MNS(IS) == ZERO) exit loop35
        MDA(IS)=MDA(IS)/FACT
     enddo loop35
     ! CALC PMF FOR ALL DATA POINTS
     DUMY = 0
     loop80: DO I=1,INSIUM
        ! SUM UP COUNTS !
        SUM=0.0
        loop50: DO IS=1,NSIUM
           IF (MNS(IS) == ZERO) exit loop50
           IF(MHI(I,IS) > 0.AND.MFL(IS) == 0) SUM=SUM+MHI(I,IS)*MAG(IS)
        enddo loop50
        MEM(I)=SUM
        ! CALC FACTOR
        FACT=0.0
        loop60: DO IS=1,NSIUM
           IF(MNS(IS) == 0) exit loop60
           IF(MHI(I,IS) >= 0.AND.MFL(IS) == 0) &
                FACT=FACT+MNS(IS)*MUP(I,IS)/MDA(IS)*MAG(IS)
        enddo loop60
        ! SET NEW PROBABILITIES FOR STATES
        IF(FACT > ZERO) THEN
           MMF(I)=SUM/FACT
        ELSE
           MMF(I)=0.0
        ENDIF
        ! NEXT DATA POINT
     enddo loop80
     ! update factors of individual simulations
     MDA(1:nsium)=zero
     loop30: DO I=1,INSIUM
        DO IS=1,NSIUM
           IF (MNS(IS) == 0) cycle loop30
           IF(MHI(I,IS) >= 0) MDA(IS)=MDA(IS)+MMF(I)*MUP(I,IS)
        enddo
     enddo loop30
     DO IS=1,NSIUM
        IF(MDA(IS) == 0) THEN
           MDA(IS)=1.0
        END IF
     enddo
     ! calc. deviation of expected number to measured number (MHI)
     DO I=1,INSIUM
        EN=0
        loop70: DO IS=1,NSIUM
           IF(MNS(IS) == 0) exit loop70
           IF(MHI(I,IS) >= 0.AND.MFL(IS) == 0) THEN
              ET=MMF(I)*MNS(IS)*MUP(I,IS)/MDA(IS)*MAG(IS)
              EN=EN+ET
           END IF
        enddo loop70
        DEV=DEV+(EN-MEM(I))**2
        NCNT=NCNT+1
     enddo
     ! look whether there were significant changes
     DEV=(DEV/NCNT)**0.5
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,425) IT,DEV,MDA(1),MDA(2),MDA(3)
425     FORMAT(' It ',I4,': ',4G11.3)
     END IF
     IF(DEV < EPS) exit loop100
  enddo loop100

  ! calc average deviation for each of the simulations
  AVDEV=0.0
  AVN=0
  loop110: DO IS=1,NSIUM
     IF(MNS(IS) == ZERO) exit loop110
     NCNT=0
     DEV=0.0
     DO I=1,INSIUM
        IF(MHI(I,IS) >= 0) THEN
           ET=MAX(MMF(I)*MNS(IS)*MUP(I,IS)/MDA(IS),TENM5)
           DEV=DEV+(ET-MHI(I,IS))**2/ET
           NCNT=NCNT+1
        END IF
     enddo
     MER(IS)=DEV/MNS(IS)
     IF(MFL(IS) == 0) THEN
        AVDEV=AVDEV+MER(IS)
        AVN=AVN+1
     END IF
  enddo loop110

  ! look whether some of the runs have to be discarded
  AVDEV=MAX(AVDEV/AVN,MER(ISIUM))*MAXDEV
  NDEL=0
  loop130: DO IS=1,NSIUM
     IF(MNS(IS) == ZERO) exit loop130
     IF((MER(IS) > AVDEV).AND.(.NOT.LCORRV)) THEN
        IF(MFL(IS) == 0) NDEL=NDEL+1
        MFL(IS)=-1
     ELSE
        MFL(IS)=0
     END IF
     IF(PRNLEV >= 2) THEN
        WRITE (OUTU,410) MFL(IS),IS,MER(IS)
410     FORMAT(I2,' Deviation of simulation ',I5,' : ',G12.3)
     END IF
  enddo loop130
  DELIT=DELIT+1
  IF (NDEL > 0.AND.DELIT < 20) GOTO 7
  ! copy pmf into parameter array
  MINSET=.FALSE.
  DO I=1,INSIUM
     IF(MMF(I) > 0.0) THEN
        MMF(I)=RT*(MSF(I)-LOG(MMF(I)))
        IF(.NOT.MINSET) THEN
           MINSET=.TRUE.
           MINF=MMF(I)
        ELSE
           MINF=MIN(MINF,MMF(I))
        END IF
     END IF
  enddo

  MMF(1:insium)=MMF(1:insium)-MINF
  MPA(1:insium)=MMF(1:insium)
  ! extrapolate pmf
  call UMEXTR(MPA,MEM,STCAUM)

  ! set initial sizes
  S(1:numbr)=NBUM(1:numbr)
  ! transform each dimension
  DO DIM=1,NUMBR
     N=2*TFUM(DIM)+PFUM(DIM)
     M=NBUM(DIM)
     call chmalloc('umb.src','UMSTA3','US',M,N,crl=US)  
     call chmalloc('umb.src','UMSTA3','VS',N,N,crl=VS)

     call chmalloc('umb.src','UMSTA3','WS',2*N+2*m,crl=Wtarg)
     ws=>wtarg(1:n)
     ts1=>wtarg(n+1:n+m)
     ts2=>wtarg(n+m+1:n+m+m)
     ts3=>wtarg(n+m+m+1:n+m+m+n)
    
     call UMFIT1(MPA,DIM,N,M,S,US,VS,WS,TS1,TS2,TS3,STEMUM)

     call chmdealloc('umb.src','UMSTA3','WS',2*N+2*m,crl=Wtarg)
     nullify(WS,TS1,TS2,TS3)
     call chmdealloc('umb.src','UMSTA3','US',M,N,crl=US)  
     call chmdealloc('umb.src','UMSTA3','US',n,N,crl=vS)  
     S(DIM)=N
  enddo
  RETURN
END SUBROUTINE UMSTA3
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMWRT(MHI,MMF,MEM,MPA,MUP,MSF)
  !
  !     Write out statistics
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use stream
  use umb
  implicit none
  real(chm_real)  MMF(INSIUM),MEM(INSIUM),MUP(INSIUM,NSIUM)
  real(chm_real)  MPA(INSIUM),MSF(INSIUM)
  INTEGER MHI(INSIUM,NSIUM)
  !
  INTEGER I,DIM,K
  real(chm_real)  E,X(MAXUMP),D1,D2,RT
  !
  IF(PRNLEV < 2) RETURN
  RT=KBOLTZ*TUM
  WRITE(OUTU,420)
420 FORMAT(' Population of different states. ')
  DO I=1,INSIUM
     DO DIM=1,NUMBR
        X(DIM)=(FLOAT(MOD(I-1,INSTUM(DIM+1))/INSTUM(DIM))+0.5) &
             /FLOAT(NBUM(DIM))
     enddo
     E =UMFI(MPA,X,0,STCAUM)
     D1=UMFI(MPA,X,1,STCAUM)
     D2=UMFI(MPA,X,2,STCAUM)
     WRITE(OUTU,425) I, &
          MHI(I,ISIUM),MMF(I),E,MEM(I), &
          (X(K),K=1,NUMBR,1)
425  FORMAT(I4,' : ',I10,' ',3G12.4,10(F8.5,1X))
  enddo
  !
  RETURN
END SUBROUTINE UMWRT
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMDER(MPA)
  !
  !     Return Derivative of Potential with respect to Internal Coordinates
  !
  !     C. Bartels, 1996
  !
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use umb
  !
  implicit none
  real(chm_real)  MPA(INSIUM)
  !
  INTEGER DIM,I,ID,IL,IR,IND
  real(chm_real)  PL,PR
  !
  EUM = UMFI(MPA,COUM,0,STCAUM)
  DO DIM=1,NUMBR
     DUM(DIM) = UMFI(MPA,COUM,DIM,STCAUM)
  enddo

  RETURN
END SUBROUTINE UMDER

!-----------------------------------------------------------------------
!     Compute value of fit function and derivatives at given x
!
!     D<=0   fit function
!     D>0    derivative with respect to the d'th internal coord

FUNCTION UMFI(P,X,D,T) result(umfi_rslt)
  use chm_kinds
  !
  use dimens_fcm
  use consta
  use stream
  use umb
  implicit none
  real(chm_real)  P(*),X(NUMBR),T(NUMBR,*),umfi_rslt
  INTEGER D
  !
  real(chm_real)  F,S
  INTEGER K,L,IND(MAXUMP),DIM,IP
  !
  ! calculate function values
  DO DIM=1,NUMBR
     K=1
     IF(D == DIM) THEN
        DO L=1,TFUM(DIM)
           T(DIM,K)=COS(TWOPI*X(DIM)*FLOAT(L))*TWOPI*FLOAT(L)
           K=K+1
           T(DIM,K)=-SIN(TWOPI*X(DIM)*FLOAT(L))*TWOPI*FLOAT(L)
           K=K+1
        enddo
        T(DIM,K)=0.0
        K=K+1
        IF(PFUM(DIM) > 0) THEN
           T(DIM,K)=1.0
           K=K+1
           DO L=1,PFUM(DIM)-2
              T(DIM,K)=(L+1)*X(DIM)**L
              K=K+1
           enddo
        END IF
     ELSE
        DO L=1,TFUM(DIM)
           T(DIM,K)=SIN(TWOPI*X(DIM)*FLOAT(L))
           K=K+1
           T(DIM,K)=COS(TWOPI*X(DIM)*FLOAT(L))
           K=K+1
        enddo
        IF(PFUM(DIM) > 0) THEN
           T(DIM,K)=1.0
           K=K+1
           DO L=1,PFUM(DIM)-1
              T(DIM,K)=X(DIM)**L
              K=K+1
           enddo
        END IF
     END IF
  enddo
  ! sum up
  F=0.0

  IND(1:numbr)=1
  ! loop over all parameters
  ! DO
  ! calculate index of parameter
  loop200: do while(.true.)
100  IP=1
     do k=1,numbr
        IP=IP+(IND(k)-1)*INSTUM(k)
     enddo
     ! add value
     S=P(IP)
     DO DIM=1,NUMBR
        S=S*T(DIM,IND(DIM))
     enddo
     F=F+S
     ! increment index
     loop190: DO DIM=1,NUMBR
        IND(DIM)=IND(DIM)+1
        IF(IND(DIM) > 2*TFUM(DIM)+PFUM(DIM)) THEN
           IND(DIM)=1
        ELSE!
           cycle loop200
        ENDIF
     enddo loop190
     exit loop200
  enddo loop200

  UMFI_rslt=F
  RETURN
END FUNCTION UMFI
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMFIT2(N,M,U,V,W,F,T1,P)
  !
  !     Fit functions to forces
  !
  !     C. Bartels, 1996, Adapted from PROSA Code of P. Guentert
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use number
  use stream
  implicit none
  INTEGER N,M
  real(chm_real)  U(M,N),V(N,N),W(N),F(M),T1(M),P(N)
  !
  real(chm_real),parameter :: tol=1.0e-4
  real(chm_real)    t,r,x
  integer   i,j,k,l

  call svdcmp(u,m,n,m,n,w,v,t1)

  t=0.0
  do k=1,n
     t=max(t,w(k))
     p(k)=0.0
  enddo
  t=t*tol
  do k=1,n
     if(w(k) > t) then
        r=0.0
        do i=1,m
           r=r+u(i,k)*f(i)
        enddo
        r=r/w(k)
        do l=1,n
           p(l)=p(l)+v(l,k)*r
        enddo
     end if

  enddo
  return
end subroutine umfit2
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMFIT1(MPO,DIM,N,M,S,U,V,W,F,T1,P,MEM)
  !
  !     Fit functions to potential
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use number
  use stream
  use umb
  !
  implicit none
  INTEGER DIM,N,M,S(MAXUMP)
  real(chm_real)  MPO(INSIUM),U(M,N),V(N,N),W(N), &
       F(M),T1(M),P(N),MEM(N)
  !
  INTEGER I,J,K,L,IND,C
  real(chm_real)  X,FF

  ! do for all rows
  loop200: DO I=0,INSIUM/M-1
     IND=MOD(I,INSTUM(DIM))+(I/INSTUM(DIM))*INSTUM(DIM)*M
     DO J=1,NUMBR
        C=MOD(IND,INSTUM(J+1))/INSTUM(J)
        IF(C >= S(J)) cycle loop200
     enddo
     DO J=0,M-1
        F(J+1)=MPO(IND+J*INSTUM(DIM)+1)
        K=1
        X=(FLOAT(J)+0.5)/FLOAT(M)
        DO L=1,TFUM(DIM)
           U(J+1,K)=SIN(TWOPI*X*FLOAT(L))
           K=K+1
           U(J+1,K)=COS(TWOPI*X*FLOAT(L))
           K=K+1
        enddo
        IF(PFUM(DIM) > 0) THEN
           U(J+1,K)=1
           K=K+1
           DO L=1,PFUM(DIM)-1
              U(J+1,K)=X**L
              K=K+1
           enddo
        END IF
     enddo
     !
     CALL UMFIT2(N,M,U,V,W,F,T1,P)
     !
     ! store parameters
     DO J=0,N-1
        MPO(IND+J*INSTUM(DIM)+1)=P(J+1)
     enddo
  enddo loop200
  RETURN
END subroutine umfit1

!-----------------------------------------------------------------------

SUBROUTINE UMWFI(MHI,MMF,MEM,MPA,MUP,MSF)

  !     Write out statistics to file

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use stream
  use umb
  implicit none
  real(chm_real)  MMF(INSIUM),MEM(INSIUM),MPA(INSIUM)
  real(chm_real)  MUP(INSIUM,NSIUM),MSF(INSIUM)
  INTEGER MHI(INSIUM,NSIUM)
  !
  INTEGER I,J,DIM,C,S
  real(chm_real)  RT
  !
  IF(IOLEV < 0) RETURN
  RT=KBOLTZ*TUM
  WRITE(WUNUM,400) NUMBR
400 FORMAT(I3)
  DO DIM=1,NUMBR
     WRITE(WUNUM,410) NBUM(DIM),TFUM(DIM),PFUM(DIM)
410  FORMAT(3I4)
  enddo

  loop20: DO I=1,INSIUM
     DO J=1,NUMBR
        C=MOD(I-1,INSTUM(J+1))/INSTUM(J)
        S=2*TFUM(J)+PFUM(J)
        IF(C >= S) cycle loop20
     enddo
     WRITE(WUNUM,420) MPA(I)
420  FORMAT(G24.16E2)
  enddo loop20

  DO I=1,INSIUM
     WRITE(WUNUM,430) MHI(I,ISIUM),RT*(LOG(MUP(I,ISIUM))+MSF(I))
430  FORMAT(I10,1X,G24.16E2)
  enddo
  !
  !     Make sure everything is put on disk (which is needed on some
  !     machines in case of a job crash
  CALL SAVEIT(WUNUM)
  RETURN
END subroutine umwfi

!-----------------------------------------------------------------------

subroutine umrfi(mhi,mem,mpa,mns,mup,msf)

  !     Read in statistics from file

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use stream
  use umb
  use umbcor
  use parallel,only: psnd4,psnd8
  implicit none

  real(chm_real)  mem(insium),mpa(insium),mup(insium,nsium)
  real(chm_real)  msf(insium)
  integer mhi(insium,nsium),mns(nsium)
  !
  integer i,j,dim,c,s,tmp,nb
  integer n(maxump),t(maxump),p(maxump)
  real(chm_real)  x(maxump),rt,e
  !
  rt=kboltz*tum
  if(iolev > 0) then
     read(runum,*,end=100,err=90) nb
     !     while not eof
5    if(nb /= numbr) then
        if(wrnlev >= 2) then
           WRITE(OUTU,*) '*** Error: Number of umbrella do not match'
        end if
        goto 90
     end if
     do dim=1,numbr
        read(runum,*,err=90) n(dim),t(dim),p(dim)
        if(n(dim) /= nbum(dim).and.wrnlev >= 2) then
           WRITE(OUTU,*) '*** Error: Number of bins do not match'
        endif
        if(t(dim) /= tfum(dim).and.wrnlev >= 2) then
           WRITE(OUTU,*) '*** Warning: Number of trigonometrical', &
                ' functions do not match'
        endif
        if(p(dim) /= pfum(dim).and.wrnlev >= 2) then
           WRITE(OUTU,*) '*** Warning: Number of polynomial functions', &
                ' do not match'
        endif
     enddo

     loop20: do i=1,insium
        do j=1,numbr
           c=mod(i-1,instum(j+1))/instum(j)
           s=2*t(j)+p(j)
           if(c >= s) cycle loop20
        enddo
        read(runum,*,err=90) mpa(i)
     enddo loop20

     ! recalculate umbrella potential factors
     iloop: do i=1,insium
        read(runum,*,err=90) mhi(i,isium),mup(i,isium)
        mup(i,isium)=mup(i,isium)/rt
        if(mhi(i,isium) > 0.and.mem(i) == 0) then
           msf(i)=mup(i,isium)
        end if
        if(mhi(i,isium) > 0.or.mem(i).gt.0) then
           mup(i,isium)=exp(mup(i,isium)-msf(i))
        else
           mup(i,isium)=1.0e-20
        end if
        if(mhi(i,isium) > 0) then
           mem(i)=mem(i)+mhi(i,isium)
           mns(isium)=mns(isium)+mhi(i,isium)
        end if
     enddo iloop
     WRITE(OUTU,*) 'Read histogram ',ISIUM
     read(runum,*,end=100,err=90) nb
     isium=mod(isium,nsium)+1
     call umini3(sthium,stnsum,stflum,stagum,hpcrrd, &
          hpcrms1,hpcrms2)

     goto 5
     !
90   if(wrnlev >= 2) then
        WRITE(OUTU,*) '*** Error in reading statistics file ***'
     end if
  endif                     !  if(iolev > 0)
100 continue
  !
#if KEY_PARALLEL==1
  CALL PSND4(ISIUM,1)
  call PSND4(STFLUM,NSIUM)      
  CALL PSND8(MPA,INSIUM)
  CALL PSND4m(MHI,INSIUM*NSIUM)
  CALL PSND8(MEM,INSIUM)
  CALL PSND8(MSF,INSIUM)
  CALL PSND4(MNS,NSIUM)
  CALL PSND8m(MUP,INSIUM*NSIUM)
#endif 
  call UMSTA3(STHIUM,STMFUM,STEMUM,STFPUM,STDAUM, &
       STNSUM,STUPUM,STERUM,STFLUM,STSFUM,STAGUM)

  call UMWRT(STHIUM,STMFUM,STEMUM,STFPUM,STUPUM, &
       STSFUM)

  ISIUM=MOD(ISIUM,NSIUM)+1
  call UMINI3(STHIUM,STNSUM,STFLUM,STAGUM,HPCRRD, &
       HPCRMS1,HPCRMS2)

  RETURN
END subroutine umrfi

!-----------------------------------------------------------------------

SUBROUTINE UMWRUC(IS)

  !     Write out umbrella coordinates to file

  !     C. Bartels, 1996

  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use stream
  use umb
  implicit none
  INTEGER IS
  !
  INTEGER I
  IF(IOLEV < 0) RETURN
  WRITE(WCUNUM,400) IS,(COUM(I),I=1,NUMBR,1)
400 FORMAT(I10,1X,20G24.16E2)
  !     Make sure everything is put on disk (which is needed on some
  !     machines in case of a job crash
  CALL SAVEIT(WCUNUM)
  RETURN
END SUBROUTINE UMWRUC
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMCHT(TOLD,TNEW,DIM,MSF)
  !
  !     Adapt factors cj,k to new temperature
  !
  !     C. Bartels, 1997
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  implicit none
  !
  real(chm_real)  MSF(INSIUM),TOLD,TNEW
  INTEGER DIM
  !
  real(chm_real)  BOLD,B,FACT,E
  INTEGER I,J,IND,IND2
  !
  !
  BOLD=1.0/KBOLTZ/TOLD
  B   =1.0/KBOLTZ/TNEW
  FACT=B-BOLD
  DO I=0,INSIUM/NBUM(DIM)-1
     IND=MOD(I,INSTUM(DIM))+ &
          (I/INSTUM(DIM))*INSTUM(DIM)*NBUM(DIM)
     DO J=0,NBUM(DIM)-1
        IND2=IND+J*INSTUM(DIM)+1
        IF(MSF(IND2) /= 0.0) THEN
           E=((J+0.5)/FLOAT(NBUM(DIM))-0.25)*(EMAXUM-EMINUM)*2.0 &
                +EMINUM
           MSF(IND2)=MSF(IND2)+E*FACT
           IF(MSF(IND2) == 0.0) MSF(IND2)=1.0E-20
        END IF
     enddo
  enddo
  RETURN
END SUBROUTINE UMCHT
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMEXTR(MMF,MEM,MCA)
  !
  !     Extrapolate potential (= empirical stuff)
  !     likely to change in future versions
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  implicit none
  !
  real(chm_real)  MMF(INSIUM),MEM(INSIUM),MCA(INSIUM)
#if KEY_ADUMBRXNCOR==1
  real(chm_real) beta 
#endif
  !

  INTEGER I,J,IND,IDIM
  real(chm_real)  EN,MINF,MAXF,MINP
  INTEGER,parAMETER :: NSMOOTH=5
  real(chm_real)  VS,V(NSMOOTH),FS(NSMOOTH)
  INTEGER VI,NS,K,IT,SS,L,MAXSET,MINSET
  !
  FS(1)=-0.3
  FS(2)=1.3
  FS(3)=1.0
  FS(4)=1.3
  FS(5)=-0.3
  SS=3
  ! detemine max of sampled bins & copy potential to MCA
  MAXSET=0
  MAXF=0.0
  DO I=1,INSIUM
     MCA(I)=MMF(I)
     IF(MEM(I) > 0) THEN
        IF((MAXSET == 0).OR.(MAXF < MMF(I))) THEN
           MAXF=MMF(I)
           MAXSET=1
        END IF
     END IF
  ENDDO
  DO I=1,INSIUM
     IF(MEM(I) == 0) THEN
        MCA(I)=MAXF
     END IF
  ENDDO
  ! enforce limits of the different umbrellas using the copy of the pot
  !    ... for energy umbrella
  IF(IEUM > 0) THEN
     CALL UMEXTR2(MCA,MEM,IEUM,EMAXUM,EMINUM,TMAXUM,TMINUM)
  END IF
  !    ... for NOE umbrella
  IF(INUM > 0) THEN
     CALL UMEXTR4(MCA,MEM,INUM,LLOWNU,LUPPNU)
  ENDIF
  ! determine min of extrapolations for energy umbrella
  MINSET=0
  MINP=MAXF
  IF(IEUM > 0) THEN
     CALL UMLIM1(MMF,MEM,MCA,IEUM,MINP,MINSET)
  END IF
  ! ... and for noe umbrella
  !      IF(INUM > 0) THEN
  !        CALL UMLIM1(MMF,MEM,MCA,INUM,MINP,MINSET)
  !      ENDIF
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,400) MAXF,MINP
400  FORMAT(' Max. sampled / Min. extrapolated ',G11.3,'/',G11.3)
  END IF
  ! set pot. of bins that were not sampled
  !      EN=MAX(MINP,MAXF)
  EN=MINP
  DO I=1,INSIUM
     IF(MEM(I) == 0.0) THEN
        MMF(I)=EN
     END IF
  ENDDO
  ! enforce limits for energy umbrella
  IF(IEUM > 0) THEN
     CALL UMEXTR2(MMF,MEM,IEUM,EMAXUM,EMINUM,TMAXUM,TMINUM)
  END IF
  ! enforce limits for NOE umbrella
  IF(INUM > 0) THEN
     CALL UMEXTR4(MMF,MEM,INUM,LLOWNU,LUPPNU)
  ENDIF
!JMS 12/2010 cap on biasing potential
#if KEY_ADUMBRXNCOR==1
      beta=1/(kboltz*tum)
      do i=1,insium
         mmf(i)=-kboltz*tum*log(exp(-beta*mmf(i))+exp(-beta*maxbias))
      end do
#endif 
  ! smoothing
  DO IT=1,NEXUM*2
     DO IDIM=1,NUMBR
        DO I=0,INSIUM/NBUM(IDIM)-1
           IND=MOD(I,INSTUM(IDIM))+ &
                (I/INSTUM(IDIM))*INSTUM(IDIM)*NBUM(IDIM)
           NS=MIN(NSMOOTH,NBUM(IDIM))
           DO J=1,NS
              IF(TYPUM(IDIM) == 1) THEN
                 K=MOD(NBUM(IDIM)+J-1-NS/2,NBUM(IDIM))
              ELSE
                 K=MIN(MAX(J-1-NS/2,0),NBUM(IDIM)-1)
              END IF
              V(J)=MMF(IND+K*INSTUM(IDIM)+1)
           ENDDO
           VI=NS-1
           DO J=0,NBUM(IDIM)-1
              IF(TYPUM(IDIM) == 1) THEN
                 K=MOD(NBUM(IDIM)+J+NS/2,NBUM(IDIM))
              ELSE
                 K=MIN(J+NS/2,NBUM(IDIM)-1)
              END IF
              V(VI)=MMF(IND+K*INSTUM(IDIM)+1)
              VI=MOD(VI,NS)+1
              IF(MEM(IND+J*INSTUM(IDIM)+1) == 0) THEN
                 VS=0.0
                 DO K=1,NS
                    L=MOD(VI+K-2,NS)+1
                    VS=VS+V(L)*FS(K)
                 ENDDO
                 MMF(IND+J*INSTUM(IDIM)+1)=VS/SS
              END IF
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE UMEXTR
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMEXTR2(MMF,MEM,DIM,MAE,MIE,MAT,MIT)
  !
  !     Extrapolate potential for energy umbrellas(= empirical stuff)
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  implicit none
  !
  real(chm_real)  MMF(INSIUM),MEM(INSIUM)
  real(chm_real)  MAE,MIE,MAT,MIT
  INTEGER DIM
  !
  INTEGER LB,UB,LI,UI,IND,I,J,SMIN,SMAX,GMIN,J2,LIM
  INTEGER TOTCNT,CENTER,CNT
  real(chm_real)  B1,B2,B3,D1,D2,U1,U2,MAXDEL,DEL,DELT
  !
  ! borders beyond which potential is kept constant
  LB=NBUM(DIM)/4
  UB=(3*NBUM(DIM))/4
  LI=LB*INSTUM(DIM)
  UI=UB*INSTUM(DIM)
  ! max allowed increments per bin
  B1=1.0/KBOLTZ/MIT
  B2=1.0/KBOLTZ/TUM
  B3=1.0/KBOLTZ/MAT
  D1=(B1-B2)/B2
  D2=(B3-B2)/B2
  D1=D1*2.0*(MAE-MIE)/NBUM(DIM)
  D2=D2*2.0*(MAE-MIE)/NBUM(DIM)
  ! fill in gaps and determine first & last bin that was sampled (SMIN,SMAX) and
  ! center of bins that were sampled
  DO I=0,INSIUM/NBUM(DIM)-1
     IND=MOD(I,INSTUM(DIM))+ &
          (I/INSTUM(DIM))*INSTUM(DIM)*NBUM(DIM)
     SMIN=NBUM(DIM)
     SMAX=-1
     GMIN=-1
     CENTER=0
     TOTCNT=0
     DO J=0,NBUM(DIM)-1
        U2=MMF(IND+J*INSTUM(DIM)+1)
        CNT=MEM(IND+J*INSTUM(DIM)+1)
        IF(CNT /= 0) THEN
           TOTCNT=TOTCNT+CNT
           CENTER=CENTER+CNT*J
           SMIN=MIN(SMIN,J)
           SMAX=J
           IF(GMIN >= 0) THEN
              ! a gap has been found, fill it up
              U1=MMF(IND+GMIN*INSTUM(DIM)+1)
              DO J2=GMIN+1,J-1
                 MMF(IND+J2*INSTUM(DIM)+1)= &
                      U1+FLOAT(J2-GMIN)/FLOAT(J-GMIN)*(U2-U1)
              ENDDO
              GMIN=-1
           ENDIF
        ELSE
           IF(SMAX >= 0) THEN
              GMIN=SMAX
           ENDIF
        END IF
     ENDDO
     ! enforce limits, if some bins were sampled
     IF(TOTCNT > 0) THEN
        ! make potential flat within [1,LB) and [UB,NBUM)
        DO J=0,NBUM(DIM)-1
           IF(J < LB) THEN
              MMF(IND+J*INSTUM(DIM)+1)=MMF(IND+LI+1)
           ELSE IF(J > UB) THEN
              MMF(IND+J*INSTUM(DIM)+1)=MMF(IND+UI+1)
           ENDIF
        ENDDO
        ! enforce maximal energy increments
        MAXDEL=0.0
        CENTER=CENTER/TOTCNT
        DELT=0.0
        DO J=0,NBUM(DIM)-1
           IF(J > 0) THEN
              DEL=MMF(IND+(J-1)*INSTUM(DIM)+1)-DELT- &
                   MMF(IND+J*INSTUM(DIM)+1)
              DELT=DELT+MAX(DEL-D1,ZERO)+MIN(DEL-D2,ZERO)
              IF(J == CENTER) THEN
                 MAXDEL=DELT
              ENDIF
           ENDIF
           MMF(IND+J*INSTUM(DIM)+1)=MMF(IND+J*INSTUM(DIM)+1)+DELT
        ENDDO
        ! make sure that enforcing the energy increments does not change the
        ! potential of the central bin
        IF(MAXDEL /= 0.0) THEN
           DO J=0,NBUM(DIM)-1
              MMF(IND+J*INSTUM(DIM)+1)=MMF(IND+J*INSTUM(DIM)+1)-MAXDEL
           ENDDO
        ENDIF
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE UMEXTR2
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMEXTR4(MMF,MEM,DIM,MIC,MAC)
  !
  !     Extrapolate potential for NOE umbrella (= empirical stuff)
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  !
  implicit none
  real(chm_real)  MMF(INSIUM),MEM(INSIUM)
  real(chm_real)  MAC,MIC
  INTEGER DIM
  !
  INTEGER LB,UB,I,IND,J
  real(chm_real) POT
  !
  ! borders beyond which potential is kept constant
  LB=NBUM(DIM)*((MIC-EMINNU)/(EMAXNU-EMINNU))
  UB=NBUM(DIM)*((MAC-EMINNU)/(EMAXNU-EMINNU))
  ! loop over all rows to extrapolate
  ! make potential small outsid of [LB,UB]
  DO I=0,INSIUM/NBUM(DIM)-1
     IND=MOD(I,INSTUM(DIM))+ &
          (I/INSTUM(DIM))*INSTUM(DIM)*NBUM(DIM)
     DO J=0,NBUM(DIM)-1
        IF(J < LB) THEN
           POT=MMF(IND+J*INSTUM(DIM)+1)
           MMF(IND+J*INSTUM(DIM)+1)=MIN(MMF(IND+LB*INSTUM(DIM)+1),POT)
        ENDIF
        IF(J > UB) THEN
           POT=MMF(IND+J*INSTUM(DIM)+1)
           MMF(IND+J*INSTUM(DIM)+1)=MIN(MMF(IND+UB*INSTUM(DIM)+1),POT)
        ENDIF
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE UMEXTR4
!
!-----------------------------------------------------------------------
!
SUBROUTINE UMLIM1(MMF,MEM,MCA,DIM,MINP,MINSET)
  !
  !     Extrapolate potential for energy umbrellas(= empirical stuff)
  !
  !     C. Bartels, 1996
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use number
  use stream
  use umb
  !
  implicit none
  real(chm_real)  MMF(INSIUM),MEM(INSIUM),MCA(INSIUM)
  real(chm_real)  MINP
  INTEGER DIM,MINSET
  !
  INTEGER IND,I,J,SMIN,SMAX
  real(chm_real)  U1,U2,DIF,DEL
  real(chm_real),PARAMETER :: TOL=1.0E-5
  !
  ! determine first & last bin that was sampled (SMIN,SMAX)
  DO I=0,INSIUM/NBUM(DIM)-1
     IND=MOD(I,INSTUM(DIM))+ &
          (I/INSTUM(DIM))*INSTUM(DIM)*NBUM(DIM)
     SMIN=NBUM(DIM)
     SMAX=-1
     DO J=0,NBUM(DIM)-1
        U2=MMF(IND+J*INSTUM(DIM)+1)
        IF(MEM(IND+J*INSTUM(DIM)+1) /= 0) THEN
           SMIN=MIN(SMIN,J)
           SMAX=J
        END IF
     ENDDO
     ! extrapolation (left border)
     DEL=0.0
     IF(SMIN < NBUM(DIM)-1) THEN
        IF (MEM(IND+(SMIN+1)*INSTUM(DIM)+1) > 0) THEN
           DEL=MAX(DEL, &
                MMF(IND+SMIN*INSTUM(DIM)+1)- &
                MMF(IND+(SMIN+1)*INSTUM(DIM)+1))
        ENDIF
     ENDIF
     IF((SMIN < NBUM(DIM)).AND.(SMIN > 0)) THEN
        DIF=MMF(IND+SMIN*INSTUM(DIM)+1)-MCA(IND+SMIN*INSTUM(DIM)+1)
        ! if point is valid for extrapolation, i.e., if the point will not be changed
        ! by enforcing limits, then
        IF(ABS(DIF) < TOL) THEN
           U1=MMF(IND+SMIN*INSTUM(DIM)+1)+NEXUM*DEL
           IF((MINSET == 0).OR.(MINP > U1)) THEN
              MINP=U1
              MINSET=1
           ENDIF
        ENDIF
     ENDIF
     ! extrapolate (right border)
     DEL=0.0
     IF(SMAX > 0) THEN
        IF (MEM(IND+(SMAX-1)*INSTUM(DIM)+1) > 0) THEN
           DEL=MAX(DEL, &
                MMF(IND+SMAX*INSTUM(DIM)+1)- &
                MMF(IND+(SMAX-1)*INSTUM(DIM)+1))
        ENDIF
     ENDIF
     IF((SMAX >= 0).AND.(SMAX < NBUM(DIM)-1)) THEN
        DIF=MMF(IND+SMAX*INSTUM(DIM)+1)-MCA(IND+SMAX*INSTUM(DIM)+1)
        ! if point is valid for extrapolation, i.e., if the point will not be changed
        ! by enforcing limits, then
        IF(ABS(DIF) < TOL) THEN
           U1=MMF(IND+SMAX*INSTUM(DIM)+1)+NEXUM*DEL
           IF((MINSET == 0).OR.(MINP > U1)) THEN
              MINP=U1
              MINSET=1
           ENDIF
        ENDIF
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE UMLIM1
!----------------------------------------------------------------------
!
SUBROUTINE CORRLVAR(DISTAN,RMSCOR1,RMSCOR2,RMAEND, &
     RMSLST,ORIEND,ORILST,IND,CCOR1X,CCOR1Y,CCOR1Z,CCOR2X, &
     CCOR2Y,CCOR2Z,XCOPY,YCOPY,ZCOPY,ATOMIN, &
     LORIEN,DISYMN,SYDLIN,XCOPYS,YCOPYS,ZCOPYS,STEMPX, &
     STEMPY,STEMPZ,STEM2X,STEM2Y,STEM2Z,MINIC,MAXIC, &
     SYMLST,SYMEND)
  !
  !     Calculate the values of the variables that are to be correlated
  !      with the umbrella coordinates
  !
  !     RJ Petrella, 2001
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  !
  use number
  use stream
  use umbcor
  use coord
  use coordc
  use consta
  use bases_fcm
  use intcor_module
  use intcor2,only:fillic,bildc
  use corsubs,only:rotlsq
  !
  implicit none
  INTEGER I,J,KK,JJ,II
  real(chm_real) XX,YY,ZZ,XXR,YYR,ZZR,RMST1,RMST2
  real(chm_real) XXR1,YYR1,ZZR1,XXR2,YYR2,ZZR2
  INTEGER RMAEND(*),RMSLST(*)
  INTEGER ORIEND(*),ORILST(*)
  real(chm_real) DISTAN(*),RMSCOR1(*),RMSCOR2(*)
  real(chm_real) CCOR1X(*),CCOR1Y(*),CCOR1Z(*)
  real(chm_real) CCOR2X(*),CCOR2Y(*),CCOR2Z(*)
  real(chm_real) XCOPY(*),YCOPY(*),ZCOPY(*)
  LOGICAL LORIEN(*)
  real(chm_real) XCOPYS(*),YCOPYS(*),ZCOPYS(*)
  real(chm_real) STEMPX(*),STEMPY(*),STEMPZ(*)
  real(chm_real) STEM2X(*),STEM2Y(*),STEM2Z(*)
  INTEGER DISYMN(*),SYDLIN(*)
  INTEGER MINIC(*),MAXIC(*)
  INTEGER SYMLST(*),SYMEND(*)
  INTEGER ATOMIN(2,NATOM)
  INTEGER STARMS,ENDRMS,ATCNT
  INTEGER IND,NPR,LL
  real(chm_real) RMSMN1,RMSMN2,STPTHE
  INTEGER STAORI,ENDORI,NPO,NPT
  INTEGER ORIATM,ATOMNB
  INTEGER FILDIC, COUNT,STASYM,ENDSYM
  !
  ! *******  CALCULATE DISTANCES *****
  DO KK = 1,NDISTA
     I = DISIAT(KK)
     J = DISJAT(KK)
     XX = X(I) - X(J)
     XX = XX*XX
     YY = Y(I) - Y(J)
     YY = YY*YY
     ZZ = Z(I) - Z(J)
     ZZ = ZZ*ZZ
     DISTAN(KK) = SQRT(XX + YY + ZZ)
  ENDDO
  ! ****** Calculate RMSD's***********
  ! make copies of positions
  DO II = 1,NATOM
     XCOPY(II) = X(II)
     YCOPY(II) = Y(II)
     ZCOPY(II) = Z(II)
  ENDDO
  FILDIC = 0
  DO JJ = 1,NRMSDI  !loop over rmsd substructures
     IF (JJ == 1) THEN
        STARMS = 1
        STAORI = 1
        STASYM = 1
     ELSE
        STARMS = RMAEND(JJ-1) + 1
        STAORI = ORIEND(JJ-1) + 1
        STASYM = SYMEND(JJ-1) + 1
     ENDIF
     ENDRMS = RMAEND(JJ)
     ENDORI = ORIEND(JJ)
     ENDSYM = SYMEND(JJ)
     ! take care of symmetries
     RMSMN1 = 999999999
     RMSMN2 = 999999999
     STPTHE = (360/DISYMN(JJ))
     DO LL = 1,DISYMN(JJ) !loop over dihe angl symmetries
        IF((LL == 2).AND.(FILDIC.EQ.0)) THEN
           CALL FILLIC(icr_struct%LENIC,.FALSE.,.FALSE.,X,Y,Z, &
                icr_struct%B1ic,icr_struct%B2ic, &
                icr_struct%T1ic,icr_struct%T2ic, &
                icr_struct%PIC, icr_struct%IAR, &
                icr_struct%JAR, icr_struct%KAR, &
                icr_struct%LAR, icr_struct%TAR)

           FILDIC = 1
        ENDIF
        IF(LL >= 2) THEN
           DO II = STASYM,ENDSYM
              ATOMNB = SYMLST(II)
              XCOPY(ATOMNB) = ANUM
              YCOPY(ATOMNB) = ANUM
              ZCOPY(ATOMNB) = ANUM
           ENDDO
           CALL INCREMDIH(icr_struct%PIC,STPTHE,SYDLIN(JJ))

           CALL BILDC(MINIC(JJ),MAXIC(JJ),XCOPY,YCOPY,ZCOPY, &
                icr_struct%B1ic,icr_struct%B2ic, &
                icr_struct%T1ic,icr_struct%T2ic, &
                icr_struct%PIC, icr_struct%IAR, &
                icr_struct%JAR, icr_struct%KAR, &
                icr_struct%LAR, icr_struct%TAR, &
                NATOM)

           IF(LL == DISYMN(JJ)) THEN !if last symm, reset
              CALL INCREMDIH(icr_struct%PIC,STPTHE,SYDLIN(JJ))
           ENDIF
        ENDIF
        ! pull out atoms for rmsd calculation
        NPR = 0
        DO I = STARMS,ENDRMS
           ATOMNB = RMSLST(I)
           NPR=NPR+1
           XCOPYS(NPR) = XCOPY(ATOMNB)
           YCOPYS(NPR) = YCOPY(ATOMNB)
           ZCOPYS(NPR) = ZCOPY(ATOMNB)
           STEMPX(NPR) = CCOR1X(ATOMNB)
           STEMPY(NPR) = CCOR1Y(ATOMNB)
           STEMPZ(NPR) = CCOR1Z(ATOMNB)
           STEM2X(NPR) = CCOR2X(ATOMNB)
           STEM2Y(NPR) = CCOR2Y(ATOMNB)
           STEM2Z(NPR) = CCOR2Z(ATOMNB)
        ENDDO
        ! do reorientation if necessary
        IF (LORIEN(JJ)) THEN
           NPO = 0
           NPT = NPR
           DO I = STAORI,ENDORI
              ORIATM = ORILST(I)
              NPT = NPT + 1
              NPO = NPO + 1
              ATOMIN(1,NPO)=NPT
              ATOMIN(2,NPO)=NPT
              ! tack the orientation-determining atoms on
              XCOPYS(NPT) = XCOPY(ORIATM)
              YCOPYS(NPT) = YCOPY(ORIATM)
              ZCOPYS(NPT) = ZCOPY(ORIATM)
              STEMPX(NPT) = CCOR1X(ORIATM)
              STEMPY(NPT) = CCOR1Y(ORIATM)
              STEMPZ(NPT) = CCOR1Z(ORIATM)
              STEM2X(NPT) = CCOR2X(ORIATM)
              STEM2Y(NPT) = CCOR2Y(ORIATM)
              STEM2Z(NPT) = CCOR2Z(ORIATM)
           ENDDO
           CALL ROTLSQ(STEMPX,STEMPY,STEMPZ,NPT, &
                XCOPYS,YCOPYS,ZCOPYS,NPT,ATOMIN,NPO,.FALSE., &
                AMASS,AMASS,.FALSE.,WMAIN,.FALSE.,.FALSE.)
        ENDIF
        RMST1 = 0
        DO I = 1,NPR
           XXR1 = XCOPYS(I) - STEMPX(I)
           YYR1 = YCOPYS(I) - STEMPY(I)
           ZZR1 = ZCOPYS(I) - STEMPZ(I)
           XXR1 = XXR1*XXR1
           YYR1 = YYR1*YYR1
           ZZR1 = ZZR1*ZZR1
           RMST1=RMST1+XXR1+YYR1+ZZR1
        ENDDO
        IF (RMST1 < RMSMN1) RMSMN1 = RMST1
        IF (LORIEN(JJ)) THEN
           CALL ROTLSQ(STEM2X,STEM2Y,STEM2Z,NPT, &
                XCOPYS,YCOPYS,ZCOPYS,NPT,ATOMIN,NPO,.FALSE., &
                AMASS,AMASS,.FALSE.,WMAIN,.FALSE.,.FALSE.)
        ENDIF
        RMST2 = 0
        DO I = 1,NPR
           XXR2 = XCOPYS(I) - STEM2X(I)
           YYR2 = YCOPYS(I) - STEM2Y(I)
           ZZR2 = ZCOPYS(I) - STEM2Z(I)
           XXR2 = XXR2*XXR2
           YYR2 = YYR2*YYR2
           ZZR2 = ZZR2*ZZR2
           RMST2=RMST2+XXR2+YYR2+ZZR2
        ENDDO
        IF (RMST2 < RMSMN2) RMSMN2 = RMST2
     ENDDO !loop over symmetries
     RMSCOR1(JJ)=SQRT(RMSMN1/NPR)
     RMSCOR2(JJ)=SQRT(RMSMN2/NPR)
     ! restore coordinates to positions from before symm ops
     DO II = STARMS,ENDRMS
        ATOMNB = RMSLST(II)
        XCOPY(ATOMNB) =X(ATOMNB)
        YCOPY(ATOMNB) =Y(ATOMNB)
        ZCOPY(ATOMNB) =Z(ATOMNB)
     ENDDO
  ENDDO !loop over rmsd substructures
  RETURN
END SUBROUTINE CORRLVAR
!--------------------------------------------------------------------
SUBROUTINE INCREMDIH(PIC,STEP,ICLINE)
  !
  !     increments one dihedral angle value
  !
  use chm_kinds
  implicit none
  real(chm_real) PIC(*),STEP
  INTEGER ICLINE
  !
  PIC(ICLINE) = PIC(ICLINE) + STEP
  RETURN
END SUBROUTINE INCREMDIH

!--------------------------------------------------------------------
SUBROUTINE INITCORR(CRRDSM,TOAVDI,RMSSUM1,RMSSUM2, &
     TORMSD1,TORMSD2)
  !
  !     Initialize the accumulators for the correlated
  !     variables
  !
  use chm_kinds
  use dimens_fcm
  use umb
  use umbcor
  !
  implicit none
  real(chm_real) CRRDSM(INSIUM,NDISTA),TOAVDI(INSIUM,NDISTA)
  real(chm_real) RMSSUM1(INSIUM,NRMSDI),TORMSD1(INSIUM,NRMSDI)
  real(chm_real) RMSSUM2(INSIUM,NRMSDI),TORMSD2(INSIUM,NRMSDI)
  !      real(chm_real) RMAEND(RMSTRS),RMSLST(RMSMMS)
  !     Local variables
  INTEGER J,IS,JJ
  !
  DO J = 1,NDISTA
     DO IS = 1,INSIUM
        CRRDSM(IS,J) = 0
        TOAVDI(IS,J) = 0
     ENDDO
  ENDDO
  DO JJ = 1,NRMSDI
     DO IS = 1,INSIUM
        RMSSUM1(IS,JJ) = 0
        RMSSUM2(IS,JJ) = 0
        TORMSD1(IS,JJ) = 0
        TORMSD2(IS,JJ) = 0
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE INITCORR

!-----------------------------------------------------------------------
!
SUBROUTINE UMWRCOR(TOAVDI,TORMSD1,TORMSD2)
  !
  !     Write out statistics for correlated structural variables to file
  !
  !     RJP 2002
  !
  use chm_kinds
  use dimens_fcm
  use umb
  use umbcor
  use stream
  implicit none
  !
  real(chm_real) TOAVDI(INSIUM,NDISTA),TORMSD1(INSIUM,NRMSDI)
  real(chm_real) TORMSD2(INSIUM,NRMSDI)
  !
  INTEGER II,JJ
  !
  IF(IOLEV < 0) RETURN
  WRITE(WCRUNI,100) NUMBR,INSIUM,NDISTA,NRMSDI
100 FORMAT(I8,I14,I8,I8)
  DO II = 1,NDISTA
     DO JJ = 1,INSIUM
        WRITE(WCRUNI,200) TOAVDI(JJ,II)
     ENDDO
  ENDDO
200 FORMAT(G24.16E2)
  DO II = 1,NRMSDI
     DO JJ = 1,INSIUM
        WRITE(WCRUNI,300) TORMSD1(JJ,II),TORMSD2(JJ,II)
     ENDDO
  ENDDO
300 FORMAT(G24.16E2,1X,G24.16E2)
  !
  RETURN
END SUBROUTINE UMWRCOR
!
!-----------------------------------------------------------------------
!--------------------------------------------------------------------
!
SUBROUTINE UMRDCOR(TOAVDI,TORMSD1,TORMSD2)
  !
  !     Read in correlated structural variable statistics
  !      from file
  !                 RJP 2002
  !
  use chm_kinds
  use dimens_fcm
  use umb
  use umbcor
  use stream
  !
  implicit none
  real(chm_real) TOAVDI(INSIUM,NDISTA), TORMSD1(INSIUM,NRMSDI)
  real(chm_real) TORMSD2(INSIUM,NRMSDI)
  !     Local variables
  INTEGER II,JJ,NREACR,NGRDPR,NDISTR,NRMSDR
  LOGICAL DIEFLG
  integer rdflag
  !
  IF(IOLEV >= 0)THEN
     DIEFLG = .FALSE.
     do while(.not. dieflg)
10      READ(RCRUNI,*,iostat=rdflag) NREACR,NGRDPR,NDISTR,NRMSDR
        if(rdflag > 0)then       !error in read
           CALL WRNDIE(-5,'<UMRDCOR>', &
                'Error(s) in reading stats file for corr variables')
           return
        elseif(rdflag < 0)then   !end of file reached
           exit
        endif

        IF(NREACR /= NUMBR) THEN
           WRITE(OUTU,'(A47)') &
                '*** Error: Number of react coords does not match '
           DIEFLG = .TRUE.
        ENDIF
        IF(NGRDPR /= INSIUM) THEN
           WRITE(OUTU,'(A47)') &
                '*** Error: Number of grid points does not match '
           DIEFLG = .TRUE.
        ENDIF
        IF(NDISTR /= NDISTA) THEN
           WRITE(OUTU,'(A47)') &
                '*** Error: Number of distances does not match '
           DIEFLG = .TRUE.
        ENDIF
        IF(NRMSDR /= NRMSDI) THEN
           WRITE(OUTU,'(A47)') &
                '*** Error: Number of rmsd structures does not match '
           DIEFLG = .TRUE.
        ENDIF
        IF (DIEFLG) THEN
           CALL WRNDIE(-5,'<UMRDCOR>', &
                'Error(s) in reading stats file for corr variables')
        ENDIF
        DO II = 1,NDISTA
           DO JJ = 1,INSIUM
              READ(RCRUNI,*,iostat=rdflag) TOAVDI(JJ,II)
              if(rdflag > 0)then       !error in read
                 CALL WRNDIE(-5,'<UMRDCOR>', &
                      'Error(s) in reading stats file for corr variables')
                 return
              elseif(rdflag < 0)then   !end of file reached
                 exit
              endif
           ENDDO
        ENDDO
        DO II = 1,NRMSDI
           DO JJ = 1,INSIUM
              READ(RCRUNI,*,iostat=rdflag) TORMSD1(JJ,II),TORMSD2(JJ,II)
              if(rdflag > 0)then       !error in read
                 CALL WRNDIE(-5,'<UMRDCOR>', &
                      'Error(s) in reading stats file for corr variables')
                 return
              elseif(rdflag < 0)then   !end of file reached
                 exit
              endif
           ENDDO
        ENDDO
     enddo
  ENDIF

#if KEY_PARALLEL==1
  CALL PSND8m(TOAVDI,INSIUM*NDISTA)
  CALL PSND8m(TORMSD1,INSIUM*NRMSDI)
  CALL PSND8m(TORMSD2,INSIUM*NRMSDI)
#endif 
  return
end SUBROUTINE UMRDCOR

#endif /* (adumb)*/

SUBROUTINE NULL_UMB
  RETURN
END SUBROUTINE NULL_UMB

