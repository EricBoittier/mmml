module correl_mod
  use chm_kinds
  use dimens_fcm
  implicit none

contains
  SUBROUTINE CORREL(NVIBR,DDV)
    !
    !             CORREL INTERPRETS THE CORRELATION COMMAND AND CALLS TO DO
    !     THE NECESSARY WORK. THE INFORMATION IS PASSED VIA A NUMBER OF CODE
    !     VARIABLES AND THE QAT ARRAY WHICH HOLD ATOM POINTERS.
    !
    !     Original version: Robert Bruccoleri, S. Swaminathan, David Perahia
    !     Completly rewritten: Bernard R. Brooks  9/84
    !
  use chm_kinds
  use dimens_fcm
  use number,only:zero
  use exfunc
    !
  use comand
  use psf
  use stream
  use string
  use coord
    implicit none

    INTEGER NVIBR
    real(chm_real) DDV(*)
    !
    INTEGER MAXSER,MAXSR,MAXTOT

    INTEGER QSIZE
    LOGICAL DOIMAGES
    !
    !
    MAXSER=GTRMI(COMLYN,COMLEN,'MAXS',2)
    MAXTOT=GTRMI(COMLYN,COMLEN,'MAXT',512)
    QSIZE=GTRMI(COMLYN,COMLEN,'MAXA',NATOM)               ! eh050711
    maxsr=maxser+1
    !
    !     parse options and generate lists, unless not asked not to do so
    !sb+ln add testing for presence of NOUPdate keyword
    IF(.NOT. (INDXA(COMLYN,COMLEN,'NOUP')  > 0) )THEN
       CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE., &
            .TRUE.,.TRUE.,.TRUE.,.TRUE.,0,[zero],[zero],[zero],[zero],[zero],[zero])
    ENDIF
    !
    DOIMAGES=.FALSE.
    CALL CORRE1(MAXSER,MAXSR,MAXTOT, &
         QSIZE,NVIBR,DDV, &
         DOIMAGES)
    !
    !
    RETURN
  END SUBROUTINE CORREL

  SUBROUTINE CORRE1(MAXSER,ICORR,MAXTOT, &
       QSIZE,NVIBR,DDV,DOIMAGES) ! tr 20020919
    !-----------------------------------------------------------------------
    !             CORREL INTERPRETS THE CORRELATION COMMAND AND CALLS TO DO
    !     THE NECESSARY WORK. THE INFORMATION IS PASSED VIA A NUMBER OF CODE
    !     VARIABLES AND THE QAT ARRAY WHICH HOLD
    !     POINTERS INTO THE VARIOUS ARRAYS IN THE PSF TO SPECIFY WHAT
    !     INTERNAL COORDINATES OR ATOMIC POSITIONS SHOULD BE CORRELATED.
    !
    !     THE TIMESERIES DATA STRUCTURE
    !     MXNTOT - Maximum number of time entries
    !     NSERIE - Number of time series to compute
    !     SNAME(NSERIE) - Name of series
    !     ICLASS(NSERIE) - class name for each series
    !     STOT(NSERIE) - Number of entries in time series
    !     VELCOD(NSERIE) - Logical flag for time series from velocities
    !     SSKIP(NSERIE) - SKIP value for the time series
    !     SDELTA(NSERIE) - time interval between 1/SKIP steps.
    !     SOFFST(NSERIE) - time of first element.
    !     GECOD(NSERIE) - Flag specifying geometry as opposed to energy
    !     VECCOD(NSERIE) - Vector code (1 or 3)
    !     SERVAL(NSERIE) - Series value (definition depends on ICLASS)
    !     LMASS(NSERIE) - Logical flag for mass weighting of terms
    !     SAVEG(NSERIE) - Average value for time series
    !     SFLCT(NSERIE) - Fluctuation value for each series
    !     SERPT(NSERIE) - Pointer to start of QAT atom descriptor
    !     SERNQ(NSERIE) - Number of entries to average over for each series
    !     TQ(MXNTOT,NSERIE+1) - Time series values
    !     DOIMAGES      - Logical if any of the series needs images
    !
    !     Original version: Robert Bruccoleri, S. Swaminathan, David Perahia
    !     Completly rewritten: Bernard R. Brooks  9/84
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use comand
  use consta
  use psf
  use coord
  use coordc
  use stream
  use ctitla
  use image
  use parallel
  use memory
  use cvio,only:trjspc
  use helix,only:hlxspc
#if KEY_SHELL==1
  use shell,only:qshell,shllst,shblkl,nshl,qshimg      
#endif
#if KEY_PROTO==1
  use proto_mod       
#endif
  use chutil,only:getrsn,getres,makind
  use anacor_m
  use memory
  use secondary_structure,only:cuth,qstrict
  use select
  use string
  use dcor_network  !--> Amitava Roy, 12/12/2016

    implicit none
    !
    INTEGER MAXSER,ICORR,MAXTOT
    INTEGER QSIZE
    INTEGER NVIBR
    real(chm_real) DDV(*)
    LOGICAL DOIMAGES
    !
    character(len=8),allocatable,dimension(:) :: sname,iclass
    INTEGER,allocatable,dimension(:) :: &
         VELCOD,GECOD,VECCOD,STOT,SSKIP,SERPT,SERNQ,ISLCT,JSLCT,sselen,qat
    real(chm_real),allocatable,dimension(:) :: &
         SERVAL,SDELTA,SOFFST,SAVEG,SFLCT
    real(chm_real),allocatable,dimension(:,:) :: TQ
    logical,allocatable,dimension(:) :: LMASS

    CHARACTER(len=8) SID,RID
    INTEGER maxsr
    integer,allocatable,dimension(:,:) :: atomin
    real(chm_real),allocatable,dimension(:,:) :: tbt
    !
    !
    INTEGER VELCD,GECD,VECCD,SERP,SERN
    real(chm_real) SERVL
    CHARACTER(LEN=8) NAME,ICLAS
    LOGICAL MASSWT
    !
    INTEGER NSERIE,ISERIE,JSERIE,NQ,FIRSTU,NSERX,NSERXI
    INTEGER NUNIT,BEGIN,STOP,SKIP,OLDT2E
    INTEGER NTOT,NPRI2,CRSCOD
    INTEGER NAVEG,NSLCT
    INTEGER I800
    CHARACTER(len=4)  MANCOD
    INTEGER IPN,LFA

! rvenable, NHLBI Apr2024 :: split allocatable array into six arrays
! change fixes a fatal CORFUN error for series w. NTOT > 2^28 points
    real(chm_real),allocatable,dimension(:) :: SFA,SFB,SFC,SFD,SFAH,SFBH
    integer siz_sfa

    INTEGER I,J,K,N,IS,IQ
    INTEGER ICODE,IAT,IPT,IRES0,IRES1
    real(chm_real) FACTOR,FACTR2,DELTA,OFFST,XNORM
    LOGICAL LUSED,EOF,OK,ERR
    LOGICAL QDIFF,QFFT,QLTC,QNNORM,QFOLD,QRAMP,QSWIT,QORIENT,QMASS
    !
    INTEGER,parameter ::  WDMAX=80
    integer WDLEN
    CHARACTER(LEN=WDMAX) WD
    !
    INTEGER SELEN
    integer,allocatable,dimension(:) :: KSLCT
    LOGICAL QRSTI, QRSTJ
    INTEGER,parameter :: NSPN=2
    CHARACTER(LEN=2) :: SPN(NSPN)=(/'P1','P2'/)
    !
    INTEGER,PARAMETER :: MARK = -99999999
    !
    ! Variables below are for the function DCOR - Amitava Roy 03/28/2016
    real(chm_real),allocatable,dimension(:,:) :: DCORTQ1,DCORTQ2
    real(chm_real)                            :: DVAR1,DVAR2,DCOV,DCOR
    ! <--- Amitava Roy
    !
    MAXSR=MAXSER+1
    call chmalloc('correl.src','CORREL','VELCOD',MAXSR,intg=VELCOD)
    call chmalloc('correl.src','CORREL','GECOD',MAXSR,intg=GECOD)
    call chmalloc('correl.src','CORREL','VECCOD',MAXSR,intg=VECCOD)
    call chmalloc('correl.src','CORREL','SERVAL',MAXSR,crl=SERVAL)
    call chmalloc('correl.src','CORREL','LMASS',MAXSR,log=LMASS)
    call chmalloc('correl.src','CORREL','STOT',MAXSR,intg=STOT)
    call chmalloc('correl.src','CORREL','SSKIP',MAXSR,intg=SSKIP)
    call chmalloc('correl.src','CORREL','SDELTA',MAXSR,crl=SDELTA)
    call chmalloc('correl.src','CORREL','SOFFST',MAXSR,crl=SOFFST)
    call chmalloc('correl.src','CORREL','SAVEG',MAXSR,crl=SAVEG)
    call chmalloc('correl.src','CORREL','SFLCT',MAXSR,crl=SFLCT)
    call chmalloc('correl.src','CORREL','SERPT',MAXSR,intg=SERPT)
    call chmalloc('correl.src','CORREL','SERNQ',MAXSR,intg=SERNQ)
    call chmalloc('correl.src','CORREL','sname',MAXSR,ch8=sname)
    call chmalloc('correl.src','CORREL','iclass',MAXSR,ch8=iclass)
    call chmalloc('correl.src','CORREL','ISLCT',NATOM,intg=ISLCT)
    call chmalloc('correl.src','CORREL','JSLCT',NATOM,intg=JSLCT)
    call chmalloc('correl.src','CORREL','TQ',MAXTOT,maxsr,crl=TQ)
    call chmalloc('correl.src','CORREL','SSELEN',MAXSR,intg=SSELEN)
    call chmalloc('correl.src','CORREL','QAT',QSIZE,intg=QAT)



    LUSED=.FALSE.
    EOF=.FALSE.
    NQ=0
    NSERIE=0
    !
    !     set up data for the correlation function.
    SNAME(ICORR)='CORR'
    ICLASS(ICORR)='CORR'
    STOT(ICORR)=0
    VELCOD(ICORR)=0
    SSKIP(ICORR)=1
    SDELTA(ICORR)=0.0
    SOFFST(ICORR)=0.0
    GECOD(ICORR)=0
    VECCOD(ICORR)=1
    SERVAL(ICORR)=0.0
    LMASS(ICORR)=.TRUE.
    SERNQ(ICORR)=0
    SERPT(ICORR)=MARK
    !
100 CONTINUE
    CALL XTRANE(COMLYN,COMLEN,'CORREL')
10  CONTINUE
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
         .TRUE.,'CORREL> ')
    CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
    IF(LUSED.AND..NOT.EOF) GOTO 10
    !
    IF(EOF) THEN
       IF(NSTRM  ==  1) then
          call RETURN_corr1
          return
       endif
       CALL PPSTRM(OK)
       EOF=.FALSE.
       GOTO 100
    ENDIF
    WD=NEXTA4(COMLYN,COMLEN)
    IF(WD == '    ') GOTO 100
    !
    IF(WD == 'ENTE') THEN
       ! PROCESS-ENTER-COMMAND

       CALL NEXTWD(COMLYN,COMLEN,WD,WDMAX,WDLEN)
       IF(WDLEN > 8) CALL WRNDIE(0,'<CORREL>', &
            'ENTER name too long, truncated to 8 characters.')
       NAME=NEXTA8(WD,WDLEN)
       !
       WD=NEXTA4(COMLYN,COMLEN)
       !
       IF(WD == 'DUPL') THEN
          !           duplicate an existing time series
          !           get-series-number
          if( .not. func800()) goto 100
805       CONTINUE
          !
          ISERIE=I
          IF(NSERIE+VECCOD(ISERIE) > MAXSER) THEN
             CALL WRNDIE(0,'<CORREL>', &
                  'Maximum number of time series exceeded')
             IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
             GOTO 100
          ENDIF
          !
          N=NSERIE
          DO I=1,VECCOD(ISERIE)
             N=N+1
             K=ISERIE+I-1
             ICLASS(N)=ICLASS(K)
             VELCOD(N)=VELCOD(K)
             GECOD(N)=GECOD(K)
             VECCOD(N)=VECCOD(K)
             SERVAL(N)=SERVAL(K)
             LMASS(N)=LMASS(K)
             STOT(N)=STOT(K)
             SSKIP(N)=SSKIP(K)
             SDELTA(N)=SDELTA(K)
             SOFFST(N)=SOFFST(K)
             SAVEG(N)=SAVEG(K)
             SFLCT(N)=SFLCT(K)
             SERPT(N)=SERPT(K)
             SERNQ(N)=SERNQ(K)
             SNAME(N)=NAME
             DO J=1,MAXTOT
                TQ(J,N)=TQ(J,K)
             ENDDO
          ENDDO
          NSERIE=NSERIE+VECCOD(ISERIE)
          !
       ELSE
          OK=.TRUE.
          VECCD=1
          GECD=1
          IF(INDXA(COMLYN,COMLEN,'GEOM') > 0) GECD=1
          IF(INDXA(COMLYN,COMLEN,'ENER') > 0) GECD=2
          SERVL=0.0
          MASSWT=(INDXA(COMLYN,COMLEN,'MASS') > 0)
          SERP=NQ+1
          SERN=0
          !
          ICODE=-1
          ICLAS=WD
          SELEN=-1
          IF(WD == 'USER') THEN
             ICODE=0
             SERVL=NEXTF(COMLYN,COMLEN)
          ELSE IF(WD == 'ENER') THEN
             CONTINUE
          ELSE IF(WD == 'BOND') THEN
             ICODE=2
          ELSE IF(WD == 'DIST') THEN
             ICODE=2
          ELSE IF(WD == 'ANGL' .OR. WD.EQ.'THET') THEN
             ICODE=3
             ICLAS='ANGL'
          ELSE IF(WD == 'TORS' .OR. WD.EQ.'DIHE') THEN
             ICODE=4
             ICLAS='DIHE'
             SERVL=1.0
             IF(INDXA(COMLYN,COMLEN,'TRAN') > 0) SERVL=-1.0
          ELSE IF(WD == 'IMPR' .OR. WD.EQ.'IMPH') THEN
             ICODE=5
             ICLAS='IMPR'
          ELSE IF(WD == 'HBON') THEN
             ICODE=7
             IF(GECD == 2) GECD=3
             IF(INDXA(COMLYN,COMLEN,'HANG') > 0) GECD=2
             IF(INDXA(COMLYN,COMLEN,'AANG') > 0) GECD=4
             IF(INDXA(COMLYN,COMLEN,'DIST') > 0) GECD=1

          ELSE IF(WD == 'ATOM'.OR.WD.EQ.'FLUC'.OR.WD.EQ.'VECT') THEN
             ! GET-SECONDARY-CODE
             WD=NEXTA4(COMLYN,COMLEN)
             IF(WD == 'X') THEN
                GECD=1
             ELSE IF(WD == 'Y') THEN
                GECD=2
             ELSE IF(WD == 'Z') THEN
                GECD=3
             ELSE IF(WD == 'W') THEN
#if KEY_FOURD==1
                GECD=9
#else /**/
                CALL WRNDIE(-1,'<CORREL>','4D code not compiled')
                GECD=1
#endif 
             ELSE IF(WD == 'R') THEN
                GECD=4
             ELSE IF(WD == 'XYZ') THEN
                GECD=0
                VECCD=3
             ELSE IF(WD == 'DOTP') THEN
                GECD=5
             ELSE IF(WD == 'CROS') THEN
                GECD=5
                VECCD=3
             ELSE
                CALL WRNDIE(0,'<ENTER>', &
                     'Unrecognized secondary code. Command ignored.')
                GOTO 100
             ENDIF
             !
             IF(ICLAS == 'VECT') THEN
                IF(GECD >= 5 .AND. GECD /= 9) THEN
                   ICODE=6
                ELSE
                   ICODE=1
                ENDIF
             ELSE
                IF(GECD >= 5 .AND. GECD /= 9) THEN
                   ICODE=1
                ELSE
                   ICODE=0
                ENDIF
             ENDIF

          ELSE IF(WD == 'GYRA') THEN
             SERVL=GTRMF(COMLYN,COMLEN,'CUT',ANUM)
          ELSE IF(WD == 'DENS') THEN
             SERVL=GTRMF(COMLYN,COMLEN,'CUT',ANUM)
          ELSE IF(WD == 'MODE') THEN
             SERVL=GTRMF(COMLYN,COMLEN,'FREQ',FMARK)
             GECD=NEXTI(COMLYN,COMLEN)
             IF(GECD <= 0 .OR. GECD > NVIBR) THEN
                CALL WRNDIE(-1,'<CORREL>','BAD NORMAL MODE NUMBER')
                GOTO 100
             ENDIF
          ELSE IF(WD == 'TIME') THEN
             IF(INDXA(COMLYN,COMLEN,'AKMA') > 0) THEN
                SERVL=1.0/TIMFAC
             ELSE
                SERVL=1.0
             ENDIF
             !
          ELSE IF(WD == 'SURF')THEN
             !
             ! Same syntax as for COOR SURF, but only analytical method allowed
             ! ASA of selection in ENTER aaa SURF SELE  END is computed in context of
             ! FIRST selection given in TRAJ   SELE  END
             ! lni, december 04
             !
#if KEY_NOMISC==0
             ICODE=0
             SERVL=1.6
             SERVL=GTRMF(COMLYN,COMLEN,'RPRO',SERVL)
             GECD=1
             IF(INDXA(COMLYN,COMLEN,'CONT') > 0) GECD=2
             IF(INDXA(COMLYN,COMLEN,'ACCE') > 0) GECD=1
             IF(INDXA(COMLYN,COMLEN,'WEIG') > 0) GECD=-GECD

#else /**/
             CALL WRNDIE(-1,'<CORREL>','SURFAC code is not compiled')
#endif 
             !
          ELSE IF(WD == 'TEMP') THEN
             SERVL=GTRMF(COMLYN,COMLEN,'NDEG',ZERO)
          ELSE IF(WD == 'ZERO') THEN
             CONTINUE
          ELSE IF(WD == 'HELI') THEN
             !ln..B950330a.ln returning zero problem fix 06/29/95
             GECD=0
             !ln..
             VECCD=3
             CALL HLXSPC(COMLYN,COMLEN)
          ELSE IF(WD == 'CELL') THEN
             WD=NEXTA4(COMLYN,COMLEN)
             IF(WD == 'A   ') THEN
                GECD=1
             ELSE IF(WD == 'B   ') THEN
                GECD=2
             ELSE IF(WD == 'C   ') THEN
                GECD=3
             ELSE IF(WD == 'ALPH') THEN
                GECD=4
             ELSE IF(WD == 'BETA') THEN
                GECD=5
             ELSE IF(WD == 'GAMM') THEN
                GECD=6
             ELSE IF(WD == 'ALL ') THEN
                SERVL=ZERO
                GECD=0
                VECCD=6
             ELSE IF(WD == 'SHAP') THEN
                SERVL=ONE
                GECD=0
                VECCD=6
             ELSE
                CALL WRNDIE(0,'<ENTER>', &
                     'Unrecognized secondary code. Command ignored.')
                GOTO 100
             ENDIF
          ELSE IF(WD == 'INER') THEN
             SERVL=ZERO
             IF(INDXA(COMLYN,COMLEN,'TRAC') > 0) SERVL=ONE
             GECD=0
             ICODE=0
             VECCD=3
             IF(INDXA(COMLYN,COMLEN,'ALL') > 0) THEN
                IF(SERVL ==  ONE)THEN
                   CALL WRNDIE(0,'<ENTER>', &
                        'TRAC and ALL are mutually exclusive. TRAC is used')
                ELSE
                   VECCD=9
                ENDIF
             ENDIF
          ELSE IF(WD == 'RMS ') THEN
             GECD=1
             IF(INDXA(COMLYN,COMLEN,'ORIE') > 0) GECD=2
          ELSE IF(WD == 'DRMS') THEN
             ! Pick off selection(s)
             CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.FALSE.,ERR)
             IF(ERR) THEN
               CALL WRNDIE(0,'<CORRE1>','Selection error.')
               IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
               RETURN
               GOTO 100
             ENDIF
             NSLCT=NSELCT(NATOM,ISLCT)
             IF(ALL(ISLCT == JSLCT)) THEN
               IF(NSLCT == NATOM) THEN
                 SERVL=MINONE ! all atoms selected for both selections
               ELSE     
                 SERVL=ZERO   ! same selection, but not all atoms
                 IF( NSLCT > (QSIZE - NQ)) THEN
                   CALL WRNDIE(0,'<CORRE1>','Overflow of atom list.')
                   IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
                   RETURN
                   GOTO 100
                 ENDIF
                 SERP=NQ+1
                 CALL MAKIND(NATOM,ISLCT,QAT(SERP),NSLCT)
                 SERN=NSLCT
                 NQ=NQ+SERN
               ENDIF
             ELSE    
                 SERVL=NSELCT(NATOM,JSLCT) ! Different explicit selections
                 IF(NSLCT+SERVL > (QSIZE - NQ)) THEN
                   CALL WRNDIE(0,'<CORRE1>','Overflow of atom list.')
                   IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
                   RETURN
                   GOTO 100
                 ENDIF
                 SERP=NQ+1
                 CALL MAKIND(NATOM,ISLCT,QAT(SERP),NSLCT)
                 CALL MAKIND(NATOM,JSLCT,QAT(SERP+NSLCT),J)
                 SERN=NSLCT+SERVL
                 NQ=NQ+SERN
             ENDIF
!
          ELSE IF(WD == 'SECS') THEN
             CUTH=2.6
             CUTH=GTRMF(COMLYN,COMLEN,'CUT',CUTH)
             QSTRICT=INDXA(COMLYN,COMLEN,'STRI') > 0
             VECCD=4
             GECD=0 ! 1: fraction alpha; 2: fraction beta
          ELSE IF(WD == 'PUCK') THEN
             if (indxa(comlyn,comlen,'ATOM') > 0) then
                ! 6-member ring puckering coordinate
                icode = 8
                servl = 1.0
                iclas = 'PUC6'
                ! get-secondary-code
                veccd = 1
                wd = nexta4(comlyn,comlen)
                if (wd == 'Q') then
                   gecd = 1
                elseif (wd == 'THET') then
                   gecd = 2
                elseif (wd == 'PHI') then
                   gecd = 3
                else
                   gecd = 0
                   veccd = 3
                endif
             else
                VECCD=2
                GECD=0
                RID=GTRMA(COMLYN,COMLEN,'RESI')
                SID=GTRMA(COMLYN,COMLEN,'SEGI')
                !              Locate correct residue (default to first segment)
                IF(SID  == ' ') SID=SEGID(1)
                SERVL=GETRSN(SID,RID,'    ', &
                     SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
                !     Define algorithm to use - A&S is default,C&P signalled by
                !     negative RESID
                IF(INDXA(COMLYN,COMLEN,'CP') > 0) SERVL=-SERVL
                !     minimum vector series
             endif
          ELSE IF(WD == 'VECM') THEN
             GECD=0
             VECCD=3
             ICODE=1
             IF(NATIM > NATOM) THEN
                DOIMAGES=.TRUE.
             ELSE
                CALL WRNDIE(-5,'<CORREL>', &
                     'VECM is useless without images (=VECT).')
             ENDIF
             !     try to implement a dipole-moment timeseries
          ELSE IF(WD == 'DIPO') THEN
             GECD=0
             ICODE=0
             VECCD=3
             !tr 04152004   use SERVAL to indicate OXYZ (no recentering)
             !              (and remember reverse logic: keyword present
             !               means CDIPOLE gets .FALSE. !!!)
             SERVL=ONE
             IF(INDXA(COMLYN,COMLEN,'OXYZ') > 0) SERVL=ZERO
          ELSE IF(WD == 'SDIP') THEN
#if KEY_SHELL==1
             !              shell dipole moment
             IF(QSHELL) THEN
                GECD=0
                ICODE=0
                !tr 04152004      use SERVAL to indicate OXYZ (no recentering)
                SERVL=ONE
                IF(INDXA(COMLYN,COMLEN,'OXYZ') > 0) SERVL=ZERO
                !
                !            s     which shell do we want (default=first)
                SELEN=GTRMI(COMLYN,COMLEN,'SHEL',1)
                !                 or do we want bulk (overrides)
                IF(INDXA(COMLYN,COMLEN,'BULK') > 0) SELEN=-1
                VECCD=4
                !                 check for sanity
                IF(SELEN > NSHL) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,870) SELEN,NSHL
870                FORMAT(' ** WARNING ** from CORREL -- Shell, ' &
                        ,I6,' is not defined.'/' It will be set to ' &
                        ,I6,'.')
                   SELEN=NSHL
                ENDIF
                !                 use images if shell needs them
                DOIMAGES = (DOIMAGES.OR.QSHIMG)
             ELSE
                CALL WRNDIE(0,'<CORREL>', &
                     'SHELL is not set up...')
             ENDIF
#else /**/
             CALL WRNDIE(0,'<CORREL>', &
                  'SHELL is not compiled...')
#endif /*  SHELL*/
          ELSE IF(WD == 'SATM') THEN
#if KEY_SHELL==1
             !              atom in shell x?
             IF(QSHELL) THEN
                ICODE=0
                VECCD=1
                !                 which shell do we want (default=first)
                GECD=GTRMI(COMLYN,COMLEN,'SHEL',1)
                !                 or do we want bulk (overrides)
                IF(INDXA(COMLYN,COMLEN,'BULK') > 0) GECD=-1
                !                 check for sanity
                IF(GECD > NSHL) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,870) GECD,NSHL
                   GECD=NSHL
                ENDIF
             ELSE
                CALL WRNDIE(0,'<CORREL>', &
                     'SHELL is not set up...')
             ENDIF
#else /**/
             CALL WRNDIE(0,'<CORREL>', &
                  'SHELL is not compiled...')
#endif /*  SHELL*/
             !-->tr 24012006
          ELSE IF(WD == 'PRDI') THEN
#if KEY_PROTO==1 /*proto*/
             !              dipole moment of a prototype set
             ICODE=0
             VECCD=3
             GECD=0
             SERVL=NEXTI(COMLYN,COMLEN)
             IF((SERVL <= 0).OR.(SERVL > MXPRTO) &
                  .OR.(.NOT.QPRTO( int(SERVL) )))THEN
                CALL WRNDIE(0,'<CORREL>', &
                     'This Protoype is not active...')
                SERVL=0
             ENDIF
#else /* (proto)*/
             CALL WRNDIE(0,'<CORREL>', &
                  'PROTO is not compiled...')
#endif /* (proto)*/
             !<--tr 24012006
             !-->tr 30012006
          ELSE IF(WD == 'PRVE') THEN
#if KEY_PROTO==1 /*proto*/
             !              c.o.m. velocity (or coors) sum of a prototype set
             ICODE=0
             VECCD=3
             GECD=0
             SERVL=NEXTI(COMLYN,COMLEN)
             IF((SERVL <= 0).OR.(SERVL > MXPRTO) &
                  .OR.(.NOT.QPRTO( int(SERVL) )))THEN
                CALL WRNDIE(0,'<CORREL>', &
                     'This Protoype is not active...')
                SERVL=0
             ENDIF
#else /* (proto)*/
             CALL WRNDIE(0,'<CORREL>', &
                  'PROTO is not compiled...')
#endif /* (proto)*/
             !<--tr 30012006
          ELSE
             CALL WRNDIE(0,'<CORREL>', &
                  'Unrecognized time series specification')
             OK=.FALSE.
          ENDIF
          !
          IF(ICODE >= 0) THEN
             CALL INTIC(COMLYN,COMLEN,ICODE,NQ,QAT,QSIZE,SERN,GECD)
          ENDIF
          !
#if KEY_NOMISC==0
          IF(ICLAS == 'SURF'.AND. &
               INDXA(COMLYN,COMLEN,'RESI') > 0) THEN
             ! Find out how many residues are contained in the selection so
             ! one ASA per residue can be returned
             IPT=SERP
             IRES0=0
             VECCD=0
             DO I=1,SERN
                IAT=QAT(IPT)
                IRES1=GETRES(IAT,IBASE,NRES)
                IF(IRES1 /= IRES0)THEN
                   VECCD=VECCD+1
                   IRES0=IRES1
                ENDIF
                IPT=IPT+1
             ENDDO
          ENDIF
#endif 
          !
          IF(NSERIE+VECCD > MAXSER) THEN
             CALL WRNDIE(0,'<CORREL>', &
                  'Maximum number of time series exceeded')
             IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
             GOTO 100
          ENDIF
          !
          N=NSERIE
          DO I=1,VECCD
             N=N+1
             ICLASS(N)=ICLAS
             VELCOD(N)=0
             IF(VECCD > 1) THEN
                GECOD(N)=I+GECD
             ELSE
                GECOD(N)=GECD
             ENDIF
             VECCOD(N)=0
             SERVAL(N)=SERVL
             LMASS(N)=MASSWT
             STOT(N)=0
             SSKIP(N)=1
             SDELTA(N)=0.0
             SOFFST(N)=0.0
             SAVEG(N)=0.0
             SFLCT(N)=0.0
             SERPT(N)=SERP
             SERNQ(N)=SERN
             SNAME(N)=NAME
             SSELEN(N)=SELEN
             DO J=1,MAXTOT
                TQ(J,N)=0.0
             ENDDO
          ENDDO
          VECCOD(NSERIE+1)=VECCD
          NSERIE=NSERIE+VECCD
          SERPT(NSERIE+1)=NQ+1
       ENDIF
       !
    ELSE IF(WD == 'TRAJ') THEN
       ! get-time-series
       CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
       VELCD=0
       IF(INDXA(COMLYN,COMLEN,'VELO') > 0) VELCD=1
       !
       ! parse rms best fit options
       QORIENT=(INDXA(COMLYN,COMLEN,'ORIE') > 0)
       IF(QORIENT) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' Orienting each frame wrt comparison coordinates'
          QMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       ENDIF
       !
       !lni all three principal axes may be requested so TBT needs 3x3 components
       call chmalloc('correl.src','CORRE1','TBT',NSERIE,9,crl=TBT)
       call chmalloc('correl.src','CORRE1','ATOMIN',2,NATOM,intg=ATOMIN)
       call chmalloc('correl.src','CORRE1','KSLCT',NATOM,intg=KSLCT)
       !
       CALL ANACOR(X,Y,Z,XCOMP,YCOMP,ZCOMP, &
            TQ,MAXTOT,NTOT,NUNIT,FIRSTU, &
            BEGIN,STOP,SKIP,DELTA,VELCD, &
            NSERIE,SNAME,STOT,ICLASS,GECOD,VECCOD, &
            SERVAL,LMASS,SAVEG,SFLCT, &
            SERPT,SERNQ, &
            SSELEN,KSLCT,DOIMAGES, &
#if KEY_SHELL==1
            SHLLST,SHBLKL,                 & 
#endif
            QAT,DDV,ISLCT,ATOMIN,WMAIN, &
            QORIENT,QMASS,JSLCT,TBT)
       !
       call chmdealloc('correl.src','CORRE1','TBT',NSERIE,9,crl=TBT)
       call chmdealloc('correl.src','CORRE1','ATOMIN',2,NATOM,intg=ATOMIN)
       call chmdealloc('correl.src','CORRE1','KSLCT',NATOM,intg=KSLCT)
       !
       !        save series specs pertaining to time series just read.
       DO ISERIE=1,NSERIE
          IF(STOT(ISERIE) == 0) THEN
             STOT(ISERIE)=NTOT
             SSKIP(ISERIE)=SKIP
             SDELTA(ISERIE)=DELTA
             SOFFST(ISERIE)=DELTA*SKIP
             VELCOD(ISERIE)=VELCD
          ENDIF
       ENDDO
       !
    ELSE IF(WD == 'READ') THEN
       ! PROCESS-READ-COMMAND
       !        get-series-number
       if( .not. func800()) goto 100
810    CONTINUE
       !
       ISERIE=I
       CALL READTS(ISERIE,NSERX,.FALSE., &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !        process-edit-command
       GOTO 700
       !
    ELSE IF(WD == 'WRIT' .OR. WD.EQ.'PRIN') THEN
       ! process-writ-command

       !        get-series-number
       if( .not. func800()) goto 100

815    CONTINUE
       !
       ISERIE=I
       CALL WRITTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
    ELSE IF(WD == 'EDIT') THEN
       ! process-edit-command
       !        get-series-number
       if( .not. func800()) goto 100

820    CONTINUE
       GOTO 700
       !
       !
    ELSE IF(WD == 'SHOW') THEN
       ! process-show-command
       !        get-series-number
       if( .not. func800()) goto 100

825    CONTINUE
       !
       ISERIE=I
       CALL SHOWTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
       !CC
       IF(PRNLEV >= 4) THEN
          write(outu,423) ' Atom pointers for all time series:'
          write(outu,423) ' SERPT:',(serpt(j),j=1,nserie)
          write(outu,423) ' SERNQ:',(sernq(j),j=1,nserie)
          write(outu,423) ' QAT  :',(qat(j),j=1,nq)
423       format(A,10I6/,(7X,10I6))
       ENDIF
       !CC
       !
    ELSE IF(WD == 'MANT') THEN
       ! process-mantime
       !        get-series-number
       if( .not. func800()) goto 100

830    CONTINUE
       !
       ISERIE=I
       NSERXI=NSERX
       WD=NEXTA4(COMLYN,COMLEN)
       MANCOD=WD
       FACTOR=0.0
       FACTR2=0.0
       NAVEG=0
       IF(WD == 'MULT' .OR. WD.EQ.'DIVI' .OR. WD.EQ.'SHIF' .OR. &
            WD == 'CONT' .OR. WD.EQ.'TEST') FACTOR=NEXTF(COMLYN,COMLEN)
       IF(WD == 'PROB' .OR. WD.EQ.'AVER' .OR. WD.EQ.'DELN' .OR. &
            WD == 'IPOW') NAVEG=NEXTI(COMLYN,COMLEN)

       IF(WD == 'HIST' &
#if KEY_ADUMB==1
            .OR.WD.EQ.'WHIS' &      
#endif
            ) THEN
          FACTOR=NEXTF(COMLYN,COMLEN)
          FACTR2=NEXTF(COMLYN,COMLEN)
          NAVEG=NEXTI(COMLYN,COMLEN)
       ENDIF
       IF(WD == 'STAT' ) THEN
          FACTOR=NEXTF(COMLYN,COMLEN)
          FACTR2=NEXTF(COMLYN,COMLEN)
       ELSEIF(WD == 'MAP ') THEN
          IF(COMLEN == 0)THEN
             FACTOR=0.0
             FACTR2=360.0
          ELSE
             IF(INDXA(COMLYN,COMLEN,'AUTO') >0 ) MANCOD='AMAP'
             FACTOR=NEXTF(COMLYN,COMLEN)
             FACTR2=NEXTF(COMLYN,COMLEN)
             IF(FACTOR == 0.0 .AND. FACTR2 == 0.0)THEN
               IF(MANCOD == 'AMAP')THEN 
                  FACTOR= -120.0
                  FACTR2= +120.0
               ELSE
                  FACTR2=360.0
               ENDIF
             ELSEIF(FACTOR > FACTR2)THEN
                  CALL WRNDIE(-2,'<CORREL>','Incorrect MANTIME MAP interval')
             ENDIF
          ENDIF
       ENDIF
       siz_SFA=1
       IF(WD == 'INTE') then
          siz_sfa = sTOT(ISERIE)
       elseIF(WD == 'MOVI') THEN
          ! BEGIN Moving average (G. Lamoureux)
          NAVEG=NEXTI(COMLYN,COMLEN)
          !          NAVEG is the number of past frames to average
          !          NAVEG < 1 averages over all past frames
          IF(NAVEG < 1) NAVEG=MAXTOT
          siz_SFA=NAVEG
          ! END Moving average (G. Lamoureux)
       END IF
       call chmalloc('correl.src','CORRE1','SFA',siz_sfa,crl=SFA)

       IF(WD == 'COPY'.OR.WD.EQ.'ADD'.OR.WD.EQ.'FLUC'.OR. &
            WD == 'RATI'.OR.WD.EQ.'KMUL'.OR. &
#if KEY_ADUMB==1
            WD == 'WHIS'.OR. &    
#endif
            WD == 'CROS'.OR.WD.EQ.'DOTP') THEN
          !           get-series-number
          if( .not. func800()) goto 100

835       CONTINUE
          !
          JSERIE=I
          IF(NSERX /= NSERXI) THEN
             CALL WRNDIE(0,'<CORREL>', &
                  'MANTIME: VECTOR LENGTHS DONT MATCH.')
          ENDIF
          CALL MANTIM(MAXTOT,MANCOD,NAVEG,FACTOR,FACTR2,ISERIE,JSERIE, &
               SNAME,STOT,SSKIP,SDELTA,SOFFST,VECCOD, &
               TQ(1,ISERIE),TQ(1,JSERIE),SFA)
       ELSE
          CALL MANTIM(MAXTOT,MANCOD,NAVEG,FACTOR,FACTR2,ISERIE,ICORR, &
               SNAME,STOT,SSKIP,SDELTA,SOFFST,VECCOD, &
               TQ(1,ISERIE),TQ,SFA)
       ENDIF
       call chmdealloc('correl.src','CORRE1','SFA',siz_sfa,crl=SFA)

       NSERX=NSERXI
       !        process-show-command
       CALL SHOWTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
       !
    ELSE IF(WD == 'CORF') THEN
       ! process-corfun
       !        get-series-number
       if( .not. func800()) goto 100

840    CONTINUE
       !
       ISERIE=I
       !        get-series-number
       if( .not. func800()) goto 100

845    CONTINUE
       !
       JSERIE=I
       !
       NTOT=GTRMI(COMLYN,COMLEN,'TOTA',MARK)
       IF(NTOT /= MARK) THEN
          IF(NTOT > STOT(ISERIE) .OR. NTOT <= 0) THEN
             CALL WRNDIE(0,'<CORREL>', &
                  'Illegal length. Command ignored.')
             GOTO 100
          ENDIF
       ELSE
          !           for the default time series length,
          !           round down the number of entries to the nearest power of
          !           two less than or equal to half the trajectory length.
          NTOT=2
          DO WHILE(NTOT <= STOT(ISERIE))
             NTOT=NTOT*2
          ENDDO
          NTOT=NTOT/4
          !
          IF(NTOT == 0) THEN
             CALL WRNDIE(0,'<CORREL>', &
                  'Illegal option or length. Command ignored.')
             GOTO 100
          ENDIF

       ENDIF
       !
       QDIFF=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'PROD') > 0) THEN
          QDIFF=.FALSE.
       ELSE IF(INDXA(COMLYN,COMLEN,'DIFF') > 0) THEN
          QDIFF=.TRUE.
       ENDIF
       QFFT=.TRUE.
       IF(INDXA(COMLYN,COMLEN,'FFT') > 0) THEN
          QFFT=.TRUE.
       ELSE IF(INDXA(COMLYN,COMLEN,'DIRE') > 0) THEN
          QFFT=.FALSE.
       ENDIF
       QLTC=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'LTC') > 0) THEN
          QLTC=.TRUE.
       ELSE IF(INDXA(COMLYN,COMLEN,'NLTC') > 0) THEN
          QLTC=.FALSE.
       ENDIF
       IPN=1
       DO I=1,NSPN
          IF(INDXA(COMLYN,COMLEN,SPN(I)) > 0) IPN=I
       ENDDO
       QNNORM=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'NONO') > 0) QNNORM=.TRUE.
       XNORM=GTRMF(COMLYN,COMLEN,'XNOR',ZERO)
       !
       !tr 19.11.2002: had to move this bit above the veccod comparison
       !        if we want to correlate 2 DIPO series with atom selection
       !        in EACH step we need to reduce veccod by 1 since the last (4th)
       !        column contains the number of selected atoms (and we don't want
       !        that to be in our (auto-) correl-function, do we).
       !        QRSTI/J remembers this so that we can reset this afterwards
       QRSTI=.FALSE.
       QRSTJ=.FALSE.
       IF((ICLASS(ISERIE) == 'SDIP').AND.(VECCOD(ISERIE) > 3)) THEN
          QRSTI=.TRUE.
          VECCOD(ISERIE) = VECCOD(ISERIE) - 1
       ENDIF
       IF((ICLASS(JSERIE) == 'SDIP').AND.(VECCOD(JSERIE) > 3)) THEN
          QRSTJ=.TRUE.
          VECCOD(JSERIE) = VECCOD(JSERIE) - 1
       ENDIF
       !
       IF(VECCOD(ISERIE) /= VECCOD(JSERIE)) THEN
          CALL WRNDIE(1,'<CORFUN>','Vector codes dont match.')
       ENDIF
       IF(ISERIE == JSERIE) THEN
          CRSCOD=1
       ELSE
          CRSCOD=2
       ENDIF
       !
       !        copy data into correlation function vector
       STOT(ICORR)=NTOT
       VELCOD(ICORR)=VELCOD(ISERIE)
       SDELTA(ICORR)=SDELTA(ISERIE)*SSKIP(ISERIE)
       SOFFST(ICORR)=ZERO
       GECOD(ICORR)=IPN
       LMASS(ICORR)=QFFT.AND..NOT.QLTC
       !
       NPRI2=2
       DO WHILE(NPRI2 < NTOT+STOT(ISERIE))
          NPRI2=NPRI2*2
       ENDDO
       IF(.NOT.QFFT) NPRI2=STOT(ISERIE)
       LFA=NPRI2
       call chmalloc('correl.src','CORRE1','SFA',LFA,crl=SFA)
       call chmalloc('correl.src','CORRE1','SFB',LFA,crl=SFB)
       call chmalloc('correl.src','CORRE1','SFC',LFA,crl=SFC)
       call chmalloc('correl.src','CORRE1','SFD',LFA,crl=SFD)
       call chmalloc('correl.src','CORRE1','SFAH',LFA,crl=SFAH)
       call chmalloc('correl.src','CORRE1','SFBH',LFA,crl=SFBH)
       CALL CORFUN(TQ(1,ISERIE),TQ(1,JSERIE),MAXTOT, &
            SFA ,SFB,SFC,SFD, &
            SFAH,SFBH,IPN, &
            CRSCOD,VECCOD(ISERIE),QFFT,QLTC,STOT(ISERIE), &
            NTOT,QNNORM,TQ(1,ICORR),NPRI2,QDIFF,XNORM)
       call chmdealloc('correl.src','CORRE1','SFA',LFA,crl=SFA)
       call chmdealloc('correl.src','CORRE1','SFB',LFA,crl=SFB)
       call chmdealloc('correl.src','CORRE1','SFC',LFA,crl=SFC)
       call chmdealloc('correl.src','CORRE1','SFD',LFA,crl=SFD)
       call chmdealloc('correl.src','CORRE1','SFAH',LFA,crl=SFAH)
       call chmdealloc('correl.src','CORRE1','SFBH',LFA,crl=SFBH)
       !
       ISERIE=ICORR
       NSERX=1
       !        now reset the two VECCODS according to QRSTI/J from before the call
       IF(QRSTI) VECCOD(ISERIE) = VECCOD(ISERIE) + 1
       IF(QRSTJ) VECCOD(JSERIE) = VECCOD(JSERIE) + 1
       !        process-show-command
       CALL SHOWTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
    ELSE IF(WD == 'SPEC') THEN
       ! process-spectral-density

       QFOLD=(INDXA(COMLYN,COMLEN,'FOLD') > 0)
       QRAMP=(INDXA(COMLYN,COMLEN,'RAMP') > 0)
       QSWIT=(INDXA(COMLYN,COMLEN,'SWIT') > 0)
       NPRI2=STOT(ICORR)*2
       NPRI2=GTRMI(COMLYN,COMLEN,'SIZE',NPRI2)
       call chmalloc('correl.src','CORRE1','SFA',NPRI2,crl=SFA)
       call chmalloc('correl.src','CORRE1','SFAH',NPRI2,crl=SFAH)

       CALL SPECTR(TQ(1,ICORR),SFA,SFAH,STOT(ICORR), &
            SSKIP(ICORR),SDELTA(ICORR),QFOLD,QRAMP,QSWIT,NPRI2)
       SOFFST(ICORR)=ZERO
       call chmdealloc('correl.src','CORRE1','SFA',NPRI2,crl=SFA)
       call chmdealloc('correl.src','CORRE1','SFAH',NPRI2,crl=SFAH)
       !
       ISERIE=ICORR
       NSERX=1
       !        process-show-command
       CALL SHOWTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
       !mu CLUSTER code
       ! *** Modification to add clustering function, 6/93, M.E. Karpen
    ELSE IF(WD == 'CLUS') THEN
       !        get-series-number
       if( .not. func800()) goto 100

850    CONTINUE
       !
       ISERIE = I
#if KEY_PARALLEL==1
       if ( mynod == 0 ) then     
#endif
          CALL CLUSTR(ISERIE, VECCOD, MAXSER, MAXTOT, TQ, NSERIE, &
               SNAME, ICLASS, STOT)
#if KEY_PARALLEL==1
       endif                      
#endif
       ! *** End of Modification
       !
! --> Amitava Roy 03/27/2016
    ELSE IF(WD == 'DCOR') THEN
       ! get-series-number
       if( .not. func800()) goto 100
       ISERIE=I
       if( .not. func800()) goto 100
       JSERIE=I
       ! CHeck if both the time series have same number of entries
       IF(STOT(ISERIE).NE.STOT(JSERIE)) THEN
         IF(PRNLEV >= 2) WRITE(OUTU,'(A)')"Time series should be of equal length"
         goto 100
       ELSE
          CALL CHMALLOC('correl.src','DCOR','DCORTQ1',STOT(ISERIE),VECCOD(ISERIE),CRL=DCORTQ1)
          CALL CHMALLOC('correl.src','DCOR','DCORTQ2',STOT(JSERIE),VECCOD(JSERIE),CRL=DCORTQ2)
          DCORTQ1=ZERO
          DCORTQ2=ZERO
          DVAR1=ZERO
          DVAR2=ZERO
          DCOV=ZERO
          DCOR=ZERO 

          DO I=1,VECCOD(ISERIE)
           DCORTQ1(:,I)=TQ(1:STOT(ISERIE),ISERIE+I-1)
          ENDDO
          DO I=1,VECCOD(JSERIE)
           DCORTQ2(:,I)=TQ(1:STOT(JSERIE),JSERIE+I-1)
          ENDDO
          CALL DISTCOR(DCORTQ1,VECCOD(ISERIE),DCORTQ2,VECCOD(JSERIE), &
                                  STOT(ISERIE),DVAR1,DVAR2,DCOV,DCOR)
          CALL CHMDEALLOC('correl.src','DCOR','DCORTQ1',STOT(ISERIE),VECCOD(ISERIE),CRL=DCORTQ1)
          CALL CHMDEALLOC('correl.src','DCOR','DCORTQ2',STOT(JSERIE),VECCOD(JSERIE),CRL=DCORTQ2)
          WRITE(OUTU,'(1X,A18,F12.3,A10,F12.3,A12,F12.3,A11,F6.3)')  & 
          "DCOR>      VAR1 = ",DVAR1,"    VAR2 = ",DVAR2,"    COVAR = ",DCOV,"    CORR = ",DCOR
                      
       ENDIF

! <-- End DCOR Amitava Roy 03/27/2016
    ELSE IF(WD == 'END ') THEN
       call return_corr1
       RETURN
    ELSE
       CALL WRNDIE(0,'<CORREL>','UNRECOGNIZED COMMAND')
    ENDIF
    GOTO 100
    !
    !=======================================================================
    ! to process-edit-spec
700 CONTINUE
    ISERIE=I
    VECCD=GTRMI(COMLYN,COMLEN,'VECC',MARK)
    IF(VECCD /= MARK) THEN
       NSERX=VECCD
       IF(ISERIE+NSERX-1 > MAXSER) NSERX=MAXSER-ISERIE+1
       IF(NSERX <= 0) NSERX=1
    ENDIF
    J=GTRMI(COMLYN,COMLEN,'INDE',MARK)
    IF(J == MARK) THEN
       IS=ISERIE
       IQ=ISERIE+NSERX-1
    ELSE
       IS=ISERIE+J-1
       IQ=IS
    ENDIF
    ICLAS=GTRMA(COMLYN,COMLEN,'CLAS')
    GECD=GTRMI(COMLYN,COMLEN,'SECO',MARK)
    NTOT=GTRMI(COMLYN,COMLEN,'TOTA',MARK)
    SKIP=GTRMI(COMLYN,COMLEN,'SKIP',MARK)
    DELTA=GTRMF(COMLYN,COMLEN,'DELT',ANUM)
    OFFST=GTRMF(COMLYN,COMLEN,'OFFS',ANUM)
    SERVL=GTRMF(COMLYN,COMLEN,'VALU',ANUM)
    NAME=GTRMA(COMLYN,COMLEN,'NAME')
    DO JSERIE=IS,IQ
       IF(ICLAS /= ' ') ICLASS(JSERIE)=ICLAS
       IF(GECD /= MARK) GECOD(JSERIE)=GECD
       IF(NTOT /= MARK) STOT(JSERIE)=NTOT
       IF(SKIP /= MARK) SSKIP(JSERIE)=SKIP
       IF(DELTA /= ANUM) SDELTA(JSERIE)=DELTA
       IF(OFFST /= ANUM) SOFFST(JSERIE)=OFFST
       IF(SERVL /= ANUM) SERVAL(JSERIE)=SERVL
       !lni allow veccod to be modified for individual entry
       IF(VECCD /= MARK) THEN
          IF(IS == IQ) THEN
             VECCOD(JSERIE)=VECCD
          ELSE
             VECCOD(JSERIE)=0
          ENDIF
       ENDIF
       IF(NAME /= ' ') SNAME(JSERIE)=NAME
    ENDDO
    IF(VECCD /= MARK.AND.IS.NE.IQ) VECCOD(ISERIE)=NSERX
    !        process-show-command
    CALL SHOWTS(ISERIE,NSERX, &
         MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
         VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
         SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
    GOTO 100
    !=======================================================================
    !

  contains
    logical function func800() result(sub800ok)
      ! to get-series-number
      CALL NEXTWD(COMLYN,COMLEN,WD,WDMAX,WDLEN)
      IF(EQSTA(WD,WDLEN,'ALL')) THEN
         I=1
         NSERX=NSERIE
      ELSE IF(EQSTA(WD,MIN(4,WDLEN),'CORR')) THEN
         I=ICORR
         NSERX=1
      ELSE
         I=0
         DO J=1,NSERIE
            N=8
            CALL TRIME(SNAME(J),N)
            IF(EQST(WD,WDLEN,SNAME(J),N)) THEN
               I=J
               GOTO 250
            ENDIF
         ENDDO
         IF(PRNLEV >= 2) WRITE(OUTU,'(3A)') '"',WD(1:WDLEN), &
              '" doesnt exist. Command ignored.'
         CALL WRNDIE(0,'<CORREL>','Time series not found')

         sub800ok=.false.
         return
250      CONTINUE
         NSERX=VECCOD(I)
         IF(NSERX <= 0) NSERX=1
      ENDIF
      sub800ok=.true.
      return
    end function func800

    subroutine return_corr1
      call chmdealloc('correl.src','CORREL','VELCOD',MAXSR,intg=VELCOD)
      call chmdealloc('correl.src','CORREL','GECOD',MAXSR,intg=GECOD)
      call chmdealloc('correl.src','CORREL','VECCOD',MAXSR,intg=VECCOD)
      call chmdealloc('correl.src','CORREL','SERVAL',MAXSR,crl=SERVAL)
      call chmdealloc('correl.src','CORREL','LMASS',MAXSR,log=LMASS)
      call chmdealloc('correl.src','CORREL','STOT',MAXSR,intg=STOT)
      call chmdealloc('correl.src','CORREL','SSKIP',MAXSR,intg=SSKIP)
      call chmdealloc('correl.src','CORREL','SDELTA',MAXSR,crl=SDELTA)
      call chmdealloc('correl.src','CORREL','SOFFST',MAXSR,crl=SOFFST)
      call chmdealloc('correl.src','CORREL','SAVEG',MAXSR,crl=SAVEG)
      call chmdealloc('correl.src','CORREL','SFLCT',MAXSR,crl=SFLCT)
      call chmdealloc('correl.src','CORREL','SERPT',MAXSR,intg=SERPT)
      call chmdealloc('correl.src','CORREL','SERNQ',MAXSR,intg=SERNQ)
      call chmdealloc('correl.src','CORREL','sname',MAXSR,ch8=sname)
      call chmdealloc('correl.src','CORREL','iclass',MAXSR,ch8=iclass)
      call chmdealloc('correl.src','CORREL','SSELEN',MAXSR,intg=SSELEN)
      call chmdealloc('correl.src','CORREL','TQ',MAXTOT,maxsr,crl=TQ)
      call chmdealloc('correl.src','CORREL','QAT',QSIZE,intg=QAT)
      call chmdealloc('correl.src','CORREL','ISLCT',NATOM,intg=ISLCT)
      call chmdealloc('correl.src','CORREL','JSLCT',NATOM,intg=JSLCT)

      return
    end subroutine return_corr1
    !
  END subroutine corre1

  SUBROUTINE INTIC(COMLYN,COMLEN,ICODE,NQ,QAT,QSIZE,SERN,GECD)
    !
    !     THIS ROUTINE INTERPRETS THE ATOMS SPECIFIED FOR A TIME SERIES
    !     ALSO, CODE VALUES ARE PUT IN IF ENERGY IS DESIRED.
    !
  use chm_kinds
    !
  use memory
    implicit none
    integer,allocatable,dimension(:) :: ZAT
    CHARACTER(LEN=*) COMLYN
    INTEGER COMLEN,ICODE,NQ
    INTEGER QAT(*)
    INTEGER QSIZE,SERN,GECD
    !
    INTEGER NQ1
    !
    call chmalloc('correl.src','INTIC','ZAT',QSIZE-NQ,intg=ZAT)

    CALL INTIC2(COMLYN,COMLEN,ICODE,NQ1,QAT(NQ+1),QSIZE-NQ,SERN,GECD,ZAT)
    call chmdealloc('correl.src','INTIC','ZAT',QSIZE-NQ,intg=ZAT)

    NQ=NQ+NQ1
    RETURN
  END SUBROUTINE INTIC

  SUBROUTINE INTIC2(COMLYN,COMLEN,ICODE,NQ,QAT,QSIZE,SERN,GECD, &
       ZAT)
    !
    !     THIS ROUTINE INTERPRETS THE ATOMS SPECIFIED FOR A TIME SERIES
    !     ALSO, CODE VALUES ARE PUT IN IF ENERGY IS DESIRED.
    !
  use chm_kinds
  use dimens_fcm
  use psf
  use param
  use stream
  use string
  use memory
  use select
  use machutil,only:die
    !
    implicit none
    integer,allocatable,dimension(:) :: ISLCT
    CHARACTER(LEN=*) COMLYN
    INTEGER COMLEN,ICODE,NQ
    INTEGER QAT(*),ZAT(*)
    INTEGER QSIZE,SERN,GECD
    !
    !
    INTEGER ICD,NZAT,MAX,I,J,NS
    !
    ICD=ICODE
    IF(ICODE <= 1) ICD=ICODE+1
    IF(ICODE >= 5) ICD=4
    if (icode == 8) icd = 6
    NZAT=0
    SERN=0
    NQ=0
    CALL TRIME(COMLYN,COMLEN)
    call chmalloc('correl.src','INTIC2','ISLCT',NATOM,intg=ISLCT)

    CALL NXTATM(ZAT,NZAT,QSIZE,COMLYN,COMLEN,ISLCT, &
         SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
    call chmdealloc('correl.src','INTIC2','ISLCT',NATOM,intg=ISLCT)
    !
    IF(NZAT == 0) RETURN
    MAX=NZAT
    IF(ICODE >= 2) MAX=MAX+NZAT/ICD
    IF(MAX > QSIZE) THEN
       CALL WRNDIE(-2,'<INTIC>','Overflow of atom list.')
       RETURN
    ENDIF
    !
    NS=NZAT/ICD
    IF(ICODE /= 7) THEN
       IF(NS*ICD /= NZAT) THEN
          CALL WRNDIE(0,'<INTIC>', &
               'Wrong number of atoms specified. Some ignored')
       ENDIF
    ENDIF
    !
    !
    IF(ICODE == 0) THEN
       ! atom list
       DO I=1,NS
          QAT(I)=ZAT(I)
       ENDDO
       !
    ELSE IF(ICODE == 1) THEN
       ! vector list
       DO I=1,NS*2
          QAT(I)=ZAT(I)
       ENDDO
       !
    ELSE IF(ICODE == 2) THEN
       ! bond list
       J=1
       DO I=1,NS
          QAT(I)=ZAT(J)
          QAT(NS+I)=ZAT(J+1)
          QAT(2*NS+I)=1
          J=J+2
       ENDDO
       IF(GECD == 2) THEN
          !           get code values
          CALL CODES(QAT(2*NS+1),[0],[0],[0], &
               NATOM,IMOVE,IAC,NS,QAT(1),QAT(NS+1), &
               0,[0],[0],[0],0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
               .FALSE.,0,                           & ! DRUDE
#if KEY_CMAP==1
               [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                 & 
#endif
               .TRUE.,.FALSE.)
       ENDIF
       !
    ELSE IF(ICODE == 3) THEN
       ! angle list
       J=1
       DO I=1,NS
          QAT(I)=ZAT(J)
          QAT(NS+I)=ZAT(J+1)
          QAT(2*NS+I)=ZAT(J+2)
          QAT(3*NS+I)=1
          J=J+3
       ENDDO
       IF(GECD == 2) THEN
          !        get code values
          CALL CODES([0],QAT(NS*3+1),[0],[0], &
               NATOM,IMOVE,IAC,0,[0],[0], &
               NS,QAT(1),QAT(NS+1),QAT(2*NS+1), &
               0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
               .FALSE.,0,                           & ! DRUDE
#if KEY_CMAP==1
               [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                 & 
#endif
               .TRUE.,.FALSE.)
       ENDIF
       !
    ELSE IF(ICODE == 4) THEN
       ! dihedral list
       J=1
       DO I=1,NS
          QAT(I)=ZAT(J)
          QAT(NS+I)=ZAT(J+1)
          QAT(2*NS+I)=ZAT(J+2)
          QAT(3*NS+I)=ZAT(J+3)
          QAT(4*NS+I)=1
          J=J+4
       ENDDO
       IF(GECD == 2) THEN
          !           get code values
          CALL CODES([0],[0],QAT(4*NS+1),[0], &
               NATOM,IMOVE,IAC,0,[0],[0],0,[0],[0],[0],NS, &
               QAT(1),QAT(NS+1),QAT(2*NS+1),QAT(3*NS+1), &
               0,[0],[0],[0],[0], &
               .FALSE.,0,                           & ! DRUDE
#if KEY_CMAP==1
               [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                 & 
#endif
               .TRUE.,.FALSE.)
       ENDIF
       !
    ELSE IF(ICODE == 5) THEN
       ! improper dihedral list
       J=1
       DO I=1,NS
          QAT(I)=ZAT(J)
          QAT(NS+I)=ZAT(J+1)
          QAT(2*NS+I)=ZAT(J+2)
          QAT(3*NS+I)=ZAT(J+3)
          QAT(4*NS+I)=1
          J=J+4
       ENDDO
       IF(GECD == 2) THEN
          !           get code values
          CALL CODES([0],[0],[0],QAT(4*NS+1), &
               NATOM,IMOVE,IAC,0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
               NS,QAT(1),QAT(NS+1),QAT(2*NS+1),QAT(3*NS+1), &
               .FALSE.,0,                           & ! DRUDE
#if KEY_CMAP==1
               [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                 & 
#endif
               .TRUE.,.FALSE.)
       ENDIF
       !
    ELSE IF(ICODE == 6) THEN
       ! double vector list
       DO I=1,NS*4
          QAT(I)=ZAT(I)
       ENDDO
       !
    ELSE IF(ICODE == 7) THEN
       ! hydrogen bond element
       NS=1
       IF(NZAT > 4 .OR. NZAT < 2) THEN
          CALL WRNDIE(0,'<INTIC>', &
               'Wrong number of atoms specified. Some ignored')
          NS=0
       ENDIF
       !
       IF(NS == 1) THEN
          IF(NZAT == 2) THEN
             QAT(1)=ZAT(1)
             QAT(2)=0
             QAT(3)=ZAT(2)
             QAT(4)=0
          ENDIF
          IF(NZAT == 3) THEN
             QAT(1)=ZAT(1)
             QAT(2)=ZAT(2)
             QAT(3)=ZAT(3)
             QAT(4)=0
          ENDIF
          IF(NZAT == 4) THEN
             QAT(1)=ZAT(1)
             QAT(2)=ZAT(2)
             QAT(3)=ZAT(3)
             QAT(4)=ZAT(4)
          ENDIF
          !        get code values because we need the energy
          CALL HCODES(QAT(5),1,QAT(1),QAT(3),IAC)
       ENDIF
       !
    else if (icode == 8) then
       ! puckering atom list
       j = 1
       do i=1,ns
          qat(i)      = zat(j)
          qat(ns+i)   = zat(j+1)
          qat(2*ns+i) = zat(j+2)
          qat(3*ns+i) = zat(j+3)
          qat(4*ns+i) = zat(j+4)
          qat(5*ns+i) = zat(j+5)
          j = j + 6
       enddo
       !
    ELSE
       CALL DIE
    ENDIF
    !
    nq = (icode+1)*ns
    if (icode == 5) then
       nq = icode*ns
    elseif (icode == 7) then
       nq = 5*ns
    elseif (icode == 8) then
       nq = 6*ns
    endif
    SERN=NS
    !
    return
  END SUBROUTINE INTIC2
end module correl_mod

