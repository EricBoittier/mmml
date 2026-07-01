#if KEY_TSM==1
! post processor for pert files.
! calculates delta A; delta S(T); delta E
!
!
SUBROUTINE TSMP
  ! START OF DECLARATIONS FROM INCLUDE FILES
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use comand
  use exfunc
  use number
  use stream
  use string
  use memory

  implicit none
  !
  ! END OF DECLARATIONS FROM INCLUDE FILES
  !

  CHARACTER(LEN=4) WRD

  real(chm_real) TEMP,DLTEMP,LMBDAP
  INTEGER NSTPBN
  LOGICAL ERR,PLOT,GO,EOF,TEMPDO,UMBRDO,OK
  real(chm_real) DATOT,DETOT,DSTOT,DA2TOT,DE2TOT,DS2TOT
  real(chm_real) AV2TOT,AV2ATOT,AV2BTOT
  INTEGER PTUNIT,NUNIT
  INTEGER NPL,PSTKSZ

  LOGICAL LCOMP
  LOGICAL CALDER,ISITTI,ONEP,ZEROP
  INTEGER MAXWIN,MAXSRF
  INTEGER BEGIN,STOP,NICP,NWIN,NPROC,NSURF
  LOGICAL LICP,SURF,LINTE,LEAVG
  LOGICAL LUSED
  INTEGER NSKIP,NMAX,MAXPRT

  real(chm_real),allocatable,dimension(:) :: AVDWIN, AINTPL,AINT2PL, AINTIN, AELEIN
  real(chm_real),allocatable,dimension(:) :: AVDWPL,AELEPL,AVDW2PL,AELE2PL

  real(chm_real),allocatable,dimension(:) :: DAPLT, DEPLT, DSPLT, DEX, DSX, DAX
  real(chm_real),allocatable,dimension(:,:) :: ICPLT, AVICX
  integer,allocatable,dimension(:) :: ICTEMP

  real(chm_real),allocatable,dimension(:) :: DAPL, DEPL, DSPL
  logical,allocatable,dimension(:) :: SYMB

  real(chm_real),allocatable,dimension(:) :: LMBDPL, AINCR
  real(chm_real),allocatable,dimension(:) :: AVPL,AVAPL,AVBPL,AV2PL,AV2APL,AV2BPL

  real(chm_real),allocatable,dimension(:) :: AVI,AVIBIN,AVI2,AVT,AVTBIN,AVT2

  real(chm_real),allocatable,dimension(:) :: AV2TPD,AV2TMD,AV2LP,AV2LM,AVTPBN,AVTMBN
  real(chm_real),allocatable,dimension(:) :: AVLP,AVLM,AVLPBN,AVLMBN,ECP,ECP1,ECM,ECM1,EAVG
  real(chm_real),allocatable,dimension(:) :: AVTPDT, AVTMDT, DE, DS

  real(chm_real),allocatable,dimension(:,:) :: AVIC,AVIC2
  real(chm_real),allocatable,dimension(:) :: AV, AV2, AVBIN, DA

  !
  ! SYNTAX:
  !
  !   TSM POST [PSTAck <integer>] [PLOT] [TI] [COMPonents] [ENDPoints]
  !             [NODEriv] [IC] [SURF]
  !             [MAXP <integer>] [MAXW <integer>] [MAXS <integer>]
  !             PSTAck:  array size for plotting x,y points. Default 100.
  !                      needed for thermodynamic integration and or plotting.
  !             PLOT: create PLT2 output.
  !             TI:  thermodynamic integration Delta A = int 0 to 1 <dE/dLambda>
  !             NODEriv only calculate the free energy.
  !
  !             IC specifies that ic perturbation output will be processed
  !             COMP: do Vdw, Elec and Intern component analysis
  !
  !   PROC FIRSt <int> [NUNIt <int>] LAMBda <real> BINSize <int> [CTEM] [NODEriv]
  !        TEMP <real> [UMBRella] DELTa <real> [ZERO] [ONE]
  !        [BEGin <integer>] [STOP <integer>] [EAVG] [SKIP <int>] [NMAX <int>]
  !
  !        FIRSt fortran unit number
  !        NUNIT number of fortran i/o units
  !        LAMBda lambda prime. No meaning if TI is specified.
  !               Level 0 warning issued.
  !        BINSize number of data points per bin for error calculation
  !        CTEMp flag to indicate that average temperature is to be calculated.
  !        TEMP temperature for calculating properties.
  !        UMBRella flag to indicate that umbrella sampling is used.
  !        DELTa temperature increment for finite difference derivatives
  !              (calculate delta E and delta S). No meaning if TI is
  !              specified. Level 0 warning issued.
  !        ONE   indicates that lambda is exactly 1. Overides input lambda in
  !              file.
  !        ZERO  indicates that lambda is exactly 0. Overides input lambda in
  !              file.  Commands ONE and ZERO are mutually exclusive and
  !              are used only in the TI post processor. Level 0 warning issued.
  !        MAXP  maximum number of ic perturbations
  !        MAXW  maximum number of ic perturbation window (nwin in savi command)
  !        MAXS  maximum number of points in surface
  !        SURF  generate thermodynamic surfaces from ic perturbation data
  !        INTE  calculate average interaction energies
  !        BEGI  specifies number of first dataset to use in accumulating
  !              averages (ic perturbations only)
  !        STOP  specifies number of last dataset to use in accumulating
  !              averages (ic perturbations only)
  !        EAVG  calculate <Etot.> and uncertainty for this value of lambda
  !              ignored if TI.
  !        SKIP  skip first nskip records.
  !        NMAX  maximum number of points to read. skips skip number of values
  !              first.
  !
  ! process POST command line
  IF (INDXA(COMLYN,COMLEN,'NODE') /= 0) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,*) &
          'Delta E and Delta S will NOT be calculated.'
     CALDER = .FALSE.
  ELSE
     CALDER = .TRUE.
  END IF
  PLOT = (INDXA(COMLYN,COMLEN,'PLOT') /= 0)
  ISITTI = (INDXA(COMLYN,COMLEN,'TI') /= 0)
  LCOMP = (INDXA(COMLYN,COMLEN,'COMP') /= 0)
  IF (LCOMP.AND.ISITTI) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,*) &
          'VdW, electrostatic and internal components will be computed.'
  ELSE IF(LCOMP) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,*) &
          ' Components specified but will not be computed'
  ENDIF
  ! Added all this:
  IF(INDXA(COMLYN,COMLEN,'INTE') /= 0) THEN
     LINTE = .TRUE.
  ELSE
     LINTE = .FALSE.
  END IF
  LICP = (INDXA(COMLYN,COMLEN,'IC') /= 0)
  SURF = (INDXA(COMLYN,COMLEN,'SURF') /= 0)
  IF(LICP) THEN
     IF(INDX(COMLYN,COMLEN,'MAXP',4) /= 0) THEN
        MAXPRT = GTRMI(COMLYN,COMLEN,'MAXP',-1)
        IF(MAXPRT < 1) THEN
           MAXPRT = 1
           CALL WRNDIE(0,'<TSMP>', &
                'invalid MAXP; it was set to 1.')
        END IF
     ELSE
        MAXPRT = 1
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'MAXP was not on command line; it was set to 1.'
     END IF
     IF(INDX(COMLYN,COMLEN,'MAXW',4) /= 0) THEN
        MAXWIN = GTRMI(COMLYN,COMLEN,'MAXW',-1)
        IF(MAXWIN < 1) THEN
           MAXWIN = 1
           CALL WRNDIE(0,'<TSMP>', &
                'invalid MAXW; it was set to 1.')
        END IF
     ELSE
        MAXWIN = 1
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'MAXW was not on command line; it was set to 1.'
     END IF
     MAXWIN = 2*MAXWIN + 1
     call chmalloc('tsmp.src','TSMP','AV',MAXWIN,crl=AV)
     call chmalloc('tsmp.src','TSMP','AV2',MAXWIN,crl=AV2)
     call chmalloc('tsmp.src','TSMP','AVBIN',MAXWIN,crl=AVBIN)
     call chmalloc('tsmp.src','TSMP','AVIC',MAXPRT,MAXWIN,crl=AVIC)
     call chmalloc('tsmp.src','TSMP','AVIC2',MAXPRT,MAXWIN,crl=AVIC2)
     call chmalloc('tsmp.src','TSMP','DA',MAXWIN,crl=DA)
     IF(CALDER) THEN
        call chmalloc('tsmp.src','TSMP','AVTPDT',MAXWIN,crl=AVTPDT)
        call chmalloc('tsmp.src','TSMP','AVTMDT',MAXWIN,crl=AVTMDT)
        call chmalloc('tsmp.src','TSMP','AV2TPD',MAXWIN,crl=AV2TPD)
        call chmalloc('tsmp.src','TSMP','AV2TMD',MAXWIN,crl=AV2TMD)
        call chmalloc('tsmp.src','TSMP','AV2LP',MAXWIN,crl=AV2LP)
        call chmalloc('tsmp.src','TSMP','AV2LM',MAXWIN,crl=AV2LM)
        call chmalloc('tsmp.src','TSMP','AVTPBN',MAXWIN,crl=AVTPBN)
        call chmalloc('tsmp.src','TSMP','AVTMBN',MAXWIN,crl=AVTMBN)
        call chmalloc('tsmp.src','TSMP','AVLP',MAXWIN,crl=AVLP)
        call chmalloc('tsmp.src','TSMP','AVLM',MAXWIN,crl=AVLM)
        call chmalloc('tsmp.src','TSMP','AVLPBN',MAXWIN,crl=AVLPBN)
        call chmalloc('tsmp.src','TSMP','AVLMBN',MAXWIN,crl=AVLMBN)
        call chmalloc('tsmp.src','TSMP','ECP',MAXWIN,crl=ECP)
        call chmalloc('tsmp.src','TSMP','ECM',MAXWIN,crl=ECM)
        call chmalloc('tsmp.src','TSMP','ECP1',MAXWIN,crl=ECP1)
        call chmalloc('tsmp.src','TSMP','ECM1',MAXWIN,crl=ECM1)
        call chmalloc('tsmp.src','TSMP','EAVG',MAXWIN,crl=EAVG)
        call chmalloc('tsmp.src','TSMP','DE',MAXWIN,crl=DE)
        call chmalloc('tsmp.src','TSMP','DS',MAXWIN,crl=DS)
     ELSE
        call chmalloc('tsmp.src','TSMP','AVTPDT',1,crl=AVTPDT)
        call chmalloc('tsmp.src','TSMP','AVTMDT',1,crl=AVTMDT)
        call chmalloc('tsmp.src','TSMP','AV2TPD',1,crl=AV2TPD)
        call chmalloc('tsmp.src','TSMP','AV2TMD',1,crl=AV2TMD)
        call chmalloc('tsmp.src','TSMP','AV2LP',1,crl=AV2LP)
        call chmalloc('tsmp.src','TSMP','AV2LM',1,crl=AV2LM)
        call chmalloc('tsmp.src','TSMP','AVTPBN',1,crl=AVTPBN)
        call chmalloc('tsmp.src','TSMP','AVTMBN',1,crl=AVTMBN)
        call chmalloc('tsmp.src','TSMP','AVLP',1,crl=AVLP)
        call chmalloc('tsmp.src','TSMP','AVLM',1,crl=AVLM)
        call chmalloc('tsmp.src','TSMP','AVLPBN',1,crl=AVLPBN)
        call chmalloc('tsmp.src','TSMP','AVLMBN',1,crl=AVLMBN)
        call chmalloc('tsmp.src','TSMP','ECP',1,crl=ECP)
        call chmalloc('tsmp.src','TSMP','ECM',1,crl=ECM)
        call chmalloc('tsmp.src','TSMP','ECP1',1,crl=ECP1)
        call chmalloc('tsmp.src','TSMP','ECM1',1,crl=ECM1)
        call chmalloc('tsmp.src','TSMP','EAVG',1,crl=EAVG)
        call chmalloc('tsmp.src','TSMP','DE',1,crl=DE)
        call chmalloc('tsmp.src','TSMP','DS',1,crl=DS)
     END IF
     IF(LINTE) THEN
        call chmalloc('tsmp.src','TSMP','AVI',MAXWIN,crl=AVI)
        call chmalloc('tsmp.src','TSMP','AVIBIN',MAXWIN,crl=AVIBIN)
        call chmalloc('tsmp.src','TSMP','AVI2',MAXWIN,crl=AVI2)
        call chmalloc('tsmp.src','TSMP','AVT',MAXWIN,crl=AVT)
        call chmalloc('tsmp.src','TSMP','AVTBIN',MAXWIN,crl=AVTBIN)
        call chmalloc('tsmp.src','TSMP','AVT2',MAXWIN,crl=AVT2)
     ELSE
        call chmalloc('tsmp.src','TSMP','AVI',1,crl=AVI)
        call chmalloc('tsmp.src','TSMP','AVIBIN',1,crl=AVIBIN)
        call chmalloc('tsmp.src','TSMP','AVI2',1,crl=AVI2)
        call chmalloc('tsmp.src','TSMP','AVT',1,crl=AVT)
        call chmalloc('tsmp.src','TSMP','AVTBIN',1,crl=AVTBIN)
        call chmalloc('tsmp.src','TSMP','AVT2',1,crl=AVT2)
     END IF
     IF(SURF) THEN
        NPROC = 0
        NSURF = 0
        IF(INDX(COMLYN,COMLEN,'MAXS',4) /= 0) THEN
           MAXSRF = GTRMI(COMLYN,COMLEN,'MAXS',-1)
           IF(MAXSRF < 1) THEN
              MAXSRF = 100
              CALL WRNDIE(0,'<TSMP>', &
                   'invalid MAXS; it was set to 100.')
           END IF
        ELSE
           MAXSRF = 100
           IF(WRNLEV >= 2) WRITE(OUTU,*) &
                'MAXS was not on command line; it was set to 100.'
        END IF
        call chmalloc('tsmp.src','TSMP','ICPLT',MAXPRT,MAXSRF,crl=ICPLT)
        call chmalloc('tsmp.src','TSMP','ICTEMP',MAXSRF,intg=ICTEMP)
        call chmalloc('tsmp.src','TSMP','AVICX',MAXPRT,MAXSRF,crl=AVICX)
        call chmalloc('tsmp.src','TSMP','DAPLT',MAXSRF,crl=DAPLT)
        call chmalloc('tsmp.src','TSMP','DAX',MAXSRF,crl=DAX)
        IF(CALDER) THEN
           call chmalloc('tsmp.src','TSMP','DEPLT',MAXSRF,crl=DEPLT)
           call chmalloc('tsmp.src','TSMP','DSPLT',MAXSRF,crl=DSPLT)
           call chmalloc('tsmp.src','TSMP','DEX',MAXSRF,crl=DEX)
           call chmalloc('tsmp.src','TSMP','DSX',MAXSRF,crl=DSX)
        ELSE
           call chmalloc('tsmp.src','TSMP','DEPLT',1,crl=DEPLT)
           call chmalloc('tsmp.src','TSMP','DSPLT',1,crl=DSPLT)
           call chmalloc('tsmp.src','TSMP','DEX',1,crl=DEX)
           call chmalloc('tsmp.src','TSMP','DSX',1,crl=DSX)
        END IF
     ELSE
        call chmalloc('tsmp.src','TSMP','ICPLT',1,1,crl=ICPLT)
        call chmalloc('tsmp.src','TSMP','ICTEMP',1,intg=ICTEMP)
        call chmalloc('tsmp.src','TSMP','AVICX',1,1,crl=AVICX)
        call chmalloc('tsmp.src','TSMP','DAPLT',1,crl=DAPLT)
        call chmalloc('tsmp.src','TSMP','DAX',1,crl=DAX)
        call chmalloc('tsmp.src','TSMP','DEPLT',1,crl=DEPLT)
        call chmalloc('tsmp.src','TSMP','DSPLT',1,crl=DSPLT)
        call chmalloc('tsmp.src','TSMP','DEX',1,crl=DEX)
        call chmalloc('tsmp.src','TSMP','DSX',1,crl=DSX)
     END IF
  END IF
  ! Moved this here from above:
  IF(INDX(COMLYN,COMLEN,'PSTA',4) /= 0) THEN
     PSTKSZ = GTRMI(COMLYN,COMLEN,'PSTA',0)
     IF(LICP) CALL WRNDIE(1,'<TSMP>', &
          'PSTACK not used with ic perturbations.')
     IF(PSTKSZ <= 0.AND..NOT.LICP) THEN
        CALL WRNDIE(0,'<TSMP>', &
             'PSTACK is unacceptable')
        RETURN
     END IF
  ELSE IF(.NOT.LICP) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,*) &
          'Setting PSTACK (variable pstksz) to 100.'
     PSTKSZ = 100
  ELSE
     PSTKSZ = 1
  END IF
  NPL = 0
  IF(ISITTI) THEN
     ONEP =.FALSE.
     ZEROP = .FALSE.
     call chmalloc('tsmp.src','TSMP','AVPL',PSTKSZ,crl=AVPL)
     call chmalloc('tsmp.src','TSMP','AV2PL',PSTKSZ,crl=AV2PL)
     AV2TOT = ZERO
     call chmalloc('tsmp.src','TSMP','LMBDPL',PSTKSZ,crl=LMBDPL)
     call chmalloc('tsmp.src','TSMP','AINCR',PSTKSZ+1,crl=AINCR)
     IF(CALDER) THEN
        call chmalloc('tsmp.src','TSMP','AVAPL',PSTKSZ,crl=AVAPL)
        call chmalloc('tsmp.src','TSMP','AV2APL',PSTKSZ,crl=AV2APL)
        AV2ATOT = ZERO
        call chmalloc('tsmp.src','TSMP','AVBPL',PSTKSZ,crl=AVBPL)
        call chmalloc('tsmp.src','TSMP','AV2BPL',PSTKSZ,crl=AV2BPL)
        AV2BTOT = ZERO
     ELSE
        call chmalloc('tsmp.src','TSMP','AVAPL',1,crl=AVAPL)
        call chmalloc('tsmp.src','TSMP','AVBPL',1,crl=AVBPL)
     END IF
     call chmalloc('tsmp.src','TSMP','AVDWIN',PSTKSZ+1,crl=AVDWIN)
     call chmalloc('tsmp.src','TSMP','AVDWPL',PSTKSZ,crl=AVDWPL)
     call chmalloc('tsmp.src','TSMP','AVDW2PL',PSTKSZ,crl=AVDW2PL)
     call chmalloc('tsmp.src','TSMP','AELEPL',PSTKSZ,crl=AELEPL)
     call chmalloc('tsmp.src','TSMP','AELE2PL',PSTKSZ,crl=AELE2PL)
     call chmalloc('tsmp.src','TSMP','AINTPL',PSTKSZ,crl=AINTPL)
     call chmalloc('tsmp.src','TSMP','AINT2PL',PSTKSZ,crl=AINT2PL)
     call chmalloc('tsmp.src','TSMP','AELEIN',PSTKSZ+1,crl=AELEIN)
     call chmalloc('tsmp.src','TSMP','AINTIN',PSTKSZ+1,crl=AINTIN)
  ELSE
     DATOT = ZERO
     DA2TOT = ZERO
     call chmalloc('tsmp.src','TSMP','LMBDPL',PSTKSZ,crl=LMBDPL)
     call chmalloc('tsmp.src','TSMP','SYMB',PSTKSZ,log=SYMB)
     IF(CALDER) THEN
        DETOT = ZERO
        DE2TOT = ZERO
        DSTOT = ZERO
        DS2TOT = ZERO
        call chmalloc('tsmp.src','TSMP','DAPL',PSTKSZ,crl=DAPL)
        call chmalloc('tsmp.src','TSMP','DEPL',PSTKSZ,crl=DEPL)
        call chmalloc('tsmp.src','TSMP','DSPL',PSTKSZ,crl=DSPL)
     ELSE
        call chmalloc('tsmp.src','TSMP','DAPL',1,crl=DAPL)
        call chmalloc('tsmp.src','TSMP','DEPL',1,crl=DEPL)
        call chmalloc('tsmp.src','TSMP','DSPL',1,crl=DSPL)
     END IF
  END IF
  !
  GO = .TRUE.
50 IF(GO) THEN
     ERR = .FALSE.
     ZEROP = .FALSE.
     ! added eof initialization to fix problems on iris 11/91, clbiii.
     EOF = .FALSE.
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
          .TRUE.,'TSMP> ')
     ! added shf 7/89 so that we can open and close files in the post-processor.
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     IF (EOF) THEN
        IF (NSTRM  ==  1) THEN
           GO = .FALSE.
           GOTO 100
        ELSE
           CALL PPSTRM(OK)
           EOF=.FALSE.
        END IF
     END IF
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD  ==  '    ') THEN
        CONTINUE
     ELSE IF (WRD == 'PROC') THEN
        PTUNIT = GTRMI(COMLYN,COMLEN,'FIRS',-1)
        IF(PTUNIT <= 0) THEN
           ERR = .TRUE.
           CALL WRNDIE(0,'<TSMP>', &
                'error in parsing unit in POST PROC command')
           GOTO 101
        END IF
        IF (INDX(COMLYN,COMLEN,'NUNI',4) /= 0) THEN
           NUNIT = GTRMI(COMLYN,COMLEN,'NUNI',-1)
           IF(NUNIT <= 0) THEN
              ERR = .TRUE.
              CALL WRNDIE(0,'<TSMP>', &
                   'error in parsing # of units in POST PROC command')
              GOTO 101
           ENDIF
        ELSE
           NUNIT=1
        END IF
        IF(INDXA(COMLYN,COMLEN,'ONE') /= 0) THEN
           IF(ISITTI) THEN
              IF(ZEROP) THEN
                 CALL WRNDIE(0,'<TSMP>', &
                      'The ZERO command has been issued. They are mutually exclusive.')
                 GOTO 101
              ELSE
                 ONEP=.TRUE.
              END IF
           ELSE
              CALL WRNDIE(0,'<TSMP>', &
                   'The ONE command issued without TI method being used.')
              GOTO 101
           END IF
        END IF
        IF(INDXA(COMLYN,COMLEN,'ZERO') /= 0) THEN
           IF(ISITTI) THEN
              IF(ONEP) THEN
                 CALL WRNDIE(0,'<TSMP>', &
                      'The ONE command has been issued. They are mutually exclusive.')
                 GOTO 101
              ELSE
                 ZEROP=.TRUE.
              END IF
           ELSE
              CALL WRNDIE(0,'<TSMP>', &
                   'The ZERO command issued without TI method being used.')
              GOTO 101
           END IF
        END IF
        IF(INDX(COMLYN,COMLEN,'LAMB',4) /= 0) THEN
           IF(ISITTI.OR.LICP) THEN
              CALL WRNDIE(0,'<TSMP>', &
                   'lambda prime has no meaning in TI or IC methods.')
              GOTO 101
           ELSE
              LMBDAP = GTRMF(COMLYN,COMLEN,'LAMB',MINONE)
              IF(LMBDAP < ZERO) THEN
                 CALL WRNDIE(0,'<TSMP>', &
                      'lambda prime not found or unacceptable')
                 GOTO 101
              ELSE IF (LMBDAP > ONE) THEN
                 CALL WRNDIE(0,'<TSMP>', &
                      'lambda prime greater than 1')
                 GOTO 101
              END IF
           END IF
        END IF
        NSTPBN = GTRMI(COMLYN,COMLEN,'BINS',0)
        IF(NSTPBN <= 0) THEN
           CALL WRNDIE(0,'<TSMP>', &
                'bin size either zero, negative or input error')
           GOTO 101
        END IF
        IF(INDX(COMLYN,COMLEN,'SKIP',4) /= 0) THEN
           NSKIP = GTRMI(COMLYN,COMLEN,'SKIP',0)
           IF (NSKIP <= 0) THEN
              CALL WRNDIE(0,'<TSMP>', &
                   'skip value is either zero, negative or input error')
              GOTO 101
           END IF
        ELSE
           NSKIP = 0
        ENDIF

        IF(INDX(COMLYN,COMLEN,'NMAX',4) /= 0) THEN
           NMAX = GTRMI(COMLYN,COMLEN,'NMAX',0)
           IF (NMAX <= 0) THEN
              CALL WRNDIE(0,'<TSMP>', &
                   'max step value is either zero, negative or input error')
              GOTO 101
           END IF
        ELSE
           NMAX = 0
        ENDIF
        TEMP = GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
        IF(PRNLEV >= 2) WRITE(OUTU,*) &
             'Temperature is set to ',TEMP
        IF(TEMP <= ZERO) THEN
           CALL WRNDIE(0,'<TSMP>', &
                'temperature is unacceptable')
           GOTO 101
        ENDIF
        IF(INDX(COMLYN,COMLEN,'DELT',4) /= 0) THEN
           IF(ISITTI) THEN
              CALL WRNDIE(0,'<TSMP>', &
                   ' delta T has no meaning in the TI method.')
              GOTO 101
           ELSE
              DLTEMP = GTRMF(COMLYN,COMLEN,'DELT',FMARK)
              IF(DLTEMP <= ZERO) THEN
                 CALL WRNDIE(0,'<TSMP>', &
                      'delta T is unacceptable')
                 GOTO 101
              END IF
           END IF
        ELSE IF(.NOT.ISITTI) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,*) 'Setting DLTEMP to 10.K'
           DLTEMP = 10.
        END IF
        IF (INDXA(COMLYN,COMLEN,'CTEM') /= 0) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,'(2(/A))') &
                'Temperature will be calculated from the average KE.', &
                'This temperature is NOT used in the perturbation calculation.'
           TEMPDO = .TRUE.
        ELSE
           TEMPDO = .FALSE.
        END IF
        IF (INDXA(COMLYN,COMLEN,'UMBR') /= 0) THEN
           UMBRDO = .TRUE.
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'Umbrella sampling corrections will be made', &
                ' to the calculated averages.'
        ELSE
           UMBRDO = .FALSE.
        END IF
        IF(LICP.AND.INDX(COMLYN,COMLEN,'BEGI',4) /= 0) THEN
           BEGIN = GTRMI(COMLYN,COMLEN,'BEGI',-1)
           IF(BEGIN < 1) THEN
              BEGIN = 1
              CALL WRNDIE(0,'<TSMP>', &
                   'invalid BEGI; it was set to 1.')
           END IF
        ELSE IF(LICP) THEN
           BEGIN = 1
           IF(WRNLEV >= 2) WRITE(OUTU,*) &
                'BEGI was not on command line; it was set to 1.'
        END IF
        IF(LICP.AND.INDX(COMLYN,COMLEN,'STOP',4) /= 0) THEN
           STOP = GTRMI(COMLYN,COMLEN,'STOP',-1)
           IF(STOP < 1) THEN
              STOP = -1
              CALL WRNDIE(0,'<TSMP>', &
                   'invalid STOP; it was ignored.')
           END IF
        ELSE IF(LICP) THEN
           STOP = -1
           IF(WRNLEV >= 2) WRITE(OUTU,*) &
                'STOP was not on command line; it was not set.'
        END IF
        IF (INDXA(COMLYN,COMLEN,'EAVG') /= 0) THEN
           LEAVG = .TRUE.
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'Average total energy will be calculated.'
        ELSE
           LEAVG = .FALSE.
        ENDIF
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMP')
           CALL DIEWRN(0)
        END IF
        IF (ISITTI) THEN
           CALL PPOSTTI(PTUNIT,NUNIT,PSTKSZ,NSTPBN,TEMP, &
                ERR,LMBDPL,AVPL,AV2PL, &
                AVAPL,AV2APL,AVBPL, &
                AV2BPL,NPL,UMBRDO,TEMPDO,CALDER,ONEP,ZEROP, &
                AVDWPL,AVDW2PL,AELEPL, &
                AELE2PL,AINTPL,AINT2PL,LCOMP, &
                NSKIP,NMAX)
        ELSE IF (LICP) THEN
           CALL PPOSTIC(PTUNIT,NUNIT,NSTPBN,TEMP,DLTEMP, &
                TEMPDO,CALDER,MAXPRT,MAXWIN,MAXSRF, &
                BEGIN,STOP,NICP,NWIN,NPROC,NSURF, &
                AV,AVTPDT,AVTMDT, &
                AV2,AV2TPD,AV2TMD,AV2LP, &
                AV2LM,AVBIN,AVTPBN,AVTMBN, &
                AVLP,AVLM,AVLPBN,AVLMBN, &
                ECP,ECM,ECP1,ECM1, &
                EAVG,AVI,AVIBIN,AVI2, &
                AVT,AVTBIN,AVT2,AVIC, &
                AVIC2,DA,DE,DS,DAX, &
                DEX,DSX,AVICX,SURF,LINTE)
        ELSE
           CALL PPOST(PTUNIT,NUNIT,PSTKSZ,LMBDAP,NSTPBN,TEMP, &
                DLTEMP, &
                PLOT,ERR,DATOT,DETOT,DSTOT,DA2TOT,DE2TOT,DS2TOT, &
                DAPL,DEPL,DSPL, &
                LMBDPL,SYMB,NPL,UMBRDO,TEMPDO,CALDER, &
                LEAVG,NSKIP,NMAX)
        END IF
        IF(ERR) THEN
           GOTO 101
        END IF
     ELSE IF (WRD == 'END') THEN
        IF(ISITTI) THEN
           CALL TITTOT(PLOT,ERR,CALDER, &
                LMBDPL,AVPL,AV2PL, &
                AVAPL,AV2APL,AVBPL, &
                AV2BPL,TEMP,NPL, &
                AVDWPL,AVDW2PL,AELEPL, &
                AELE2PL,AINTPL,AINT2PL, &
                LCOMP,AINCR,AVDWIN,AELEIN, &
                AINTIN)
        ELSE IF(.NOT.LICP) THEN
           CALL PSTTOT(PLOT,ERR, &
                DATOT,DETOT,DSTOT,DA2TOT,DE2TOT,DS2TOT, &
                DAPL,DEPL,DSPL, &
                LMBDPL,SYMB,NPL,CALDER)
        ELSE
           IF(SURF) CALL ICPTOT(MAXPRT,NICP,NWIN,NSURF,NPROC, &
                AVICX,DAX,DEX, &
                DSX,ICPLT,DAPLT, &
                DEPLT,DSPLT,ICTEMP, &
                CALDER)
        END IF
        GOTO 101
     ELSE
        CALL WRNDIE(0,'<TSMP>','Unknown command')
        GOTO 101
     END IF
     GOTO 50
  END IF
100 CONTINUE
  RETURN
101 CONTINUE

  IF(LICP) THEN
     call chmdealloc('tsmp.src','TSMP','AV',MAXWIN,crl=AV)
     call chmdealloc('tsmp.src','TSMP','AV2',MAXWIN,crl=AV2)
     call chmdealloc('tsmp.src','TSMP','AVBIN',MAXWIN,crl=AVBIN)
     call chmdealloc('tsmp.src','TSMP','AVIC',MAXPRT,MAXWIN,crl=AVIC)
     call chmdealloc('tsmp.src','TSMP','AVIC2',MAXPRT,MAXWIN,crl=AVIC2)
     call chmdealloc('tsmp.src','TSMP','DA',MAXWIN,crl=DA)
     IF(CALDER) THEN
        call chmdealloc('tsmp.src','TSMP','AVTPDT',MAXWIN,crl=AVTPDT)
        call chmdealloc('tsmp.src','TSMP','AVTMDT',MAXWIN,crl=AVTMDT)
        call chmdealloc('tsmp.src','TSMP','AV2TPD',MAXWIN,crl=AV2TPD)
        call chmdealloc('tsmp.src','TSMP','AV2TMD',MAXWIN,crl=AV2TMD)
        call chmdealloc('tsmp.src','TSMP','AV2LP',MAXWIN,crl=AV2LP)
        call chmdealloc('tsmp.src','TSMP','AV2LM',MAXWIN,crl=AV2LM)
        call chmdealloc('tsmp.src','TSMP','AVTPBN',MAXWIN,crl=AVTPBN)
        call chmdealloc('tsmp.src','TSMP','AVTMBN',MAXWIN,crl=AVTMBN)
        call chmdealloc('tsmp.src','TSMP','AVLP',MAXWIN,crl=AVLP)
        call chmdealloc('tsmp.src','TSMP','AVLM',MAXWIN,crl=AVLM)
        call chmdealloc('tsmp.src','TSMP','AVLPBN',MAXWIN,crl=AVLPBN)
        call chmdealloc('tsmp.src','TSMP','AVLMBN',MAXWIN,crl=AVLMBN)
        call chmdealloc('tsmp.src','TSMP','ECP',MAXWIN,crl=ECP)
        call chmdealloc('tsmp.src','TSMP','ECM',MAXWIN,crl=ECM)
        call chmdealloc('tsmp.src','TSMP','ECP1',MAXWIN,crl=ECP1)
        call chmdealloc('tsmp.src','TSMP','ECM1',MAXWIN,crl=ECM1)
        call chmdealloc('tsmp.src','TSMP','EAVG',MAXWIN,crl=EAVG)
        call chmdealloc('tsmp.src','TSMP','DE',MAXWIN,crl=DE)
        call chmdealloc('tsmp.src','TSMP','DS',MAXWIN,crl=DS)
     ELSE
        call chmdealloc('tsmp.src','TSMP','AVTPDT',1,crl=AVTPDT)
        call chmdealloc('tsmp.src','TSMP','AVTMDT',1,crl=AVTMDT)
        call chmdealloc('tsmp.src','TSMP','AV2TPD',1,crl=AV2TPD)
        call chmdealloc('tsmp.src','TSMP','AV2TMD',1,crl=AV2TMD)
        call chmdealloc('tsmp.src','TSMP','AV2LP',1,crl=AV2LP)
        call chmdealloc('tsmp.src','TSMP','AV2LM',1,crl=AV2LM)
        call chmdealloc('tsmp.src','TSMP','AVTPBN',1,crl=AVTPBN)
        call chmdealloc('tsmp.src','TSMP','AVTMBN',1,crl=AVTMBN)
        call chmdealloc('tsmp.src','TSMP','AVLP',1,crl=AVLP)
        call chmdealloc('tsmp.src','TSMP','AVLM',1,crl=AVLM)
        call chmdealloc('tsmp.src','TSMP','AVLPBN',1,crl=AVLPBN)
        call chmdealloc('tsmp.src','TSMP','AVLMBN',1,crl=AVLMBN)
        call chmdealloc('tsmp.src','TSMP','ECP',1,crl=ECP)
        call chmdealloc('tsmp.src','TSMP','ECM',1,crl=ECM)
        call chmdealloc('tsmp.src','TSMP','ECP1',1,crl=ECP1)
        call chmdealloc('tsmp.src','TSMP','ECM1',1,crl=ECM1)
        call chmdealloc('tsmp.src','TSMP','EAVG',1,crl=EAVG)
        call chmdealloc('tsmp.src','TSMP','DE',1,crl=DE)
        call chmdealloc('tsmp.src','TSMP','DS',1,crl=DS)
     END IF
     IF(LINTE) THEN
        call chmdealloc('tsmp.src','TSMP','AVI',MAXWIN,crl=AVI)
        call chmdealloc('tsmp.src','TSMP','AVIBIN',MAXWIN,crl=AVIBIN)
        call chmdealloc('tsmp.src','TSMP','AVI2',MAXWIN,crl=AVI2)
        call chmdealloc('tsmp.src','TSMP','AVT',MAXWIN,crl=AVT)
        call chmdealloc('tsmp.src','TSMP','AVTBIN',MAXWIN,crl=AVTBIN)
        call chmdealloc('tsmp.src','TSMP','AVT2',MAXWIN,crl=AVT2)
     ELSE
        call chmdealloc('tsmp.src','TSMP','AVI',1,crl=AVI)
        call chmdealloc('tsmp.src','TSMP','AVIBIN',1,crl=AVIBIN)
        call chmdealloc('tsmp.src','TSMP','AVI2',1,crl=AVI2)
        call chmdealloc('tsmp.src','TSMP','AVT',1,crl=AVT)
        call chmdealloc('tsmp.src','TSMP','AVTBIN',1,crl=AVTBIN)
        call chmdealloc('tsmp.src','TSMP','AVT2',1,crl=AVT2)
     END IF
     IF(SURF) THEN
        call chmdealloc('tsmp.src','TSMP','ICPLT',MAXPRT,MAXSRF,crl=ICPLT)
        call chmdealloc('tsmp.src','TSMP','ICTEMP',MAXSRF,intg=ICTEMP)
        call chmdealloc('tsmp.src','TSMP','AVICX',MAXPRT,MAXSRF,crl=AVICX)
        call chmdealloc('tsmp.src','TSMP','DAPLT',MAXSRF,crl=DAPLT)
        call chmdealloc('tsmp.src','TSMP','DAX',MAXSRF,crl=DAX)
        IF(CALDER) THEN
           call chmdealloc('tsmp.src','TSMP','DEPLT',MAXSRF,crl=DEPLT)
           call chmdealloc('tsmp.src','TSMP','DSPLT',MAXSRF,crl=DSPLT)
           call chmdealloc('tsmp.src','TSMP','DEX',MAXSRF,crl=DEX)
           call chmdealloc('tsmp.src','TSMP','DSX',MAXSRF,crl=DSX)
        ELSE
           call chmdealloc('tsmp.src','TSMP','DEPLT',1,crl=DEPLT)
           call chmdealloc('tsmp.src','TSMP','DSPLT',1,crl=DSPLT)
           call chmdealloc('tsmp.src','TSMP','DEX',1,crl=DEX)
           call chmdealloc('tsmp.src','TSMP','DSX',1,crl=DSX)
        END IF
     ELSE
        call chmdealloc('tsmp.src','TSMP','ICPLT',1,1,crl=ICPLT)
        call chmdealloc('tsmp.src','TSMP','ICTEMP',1,intg=ICTEMP)
        call chmdealloc('tsmp.src','TSMP','AVICX',1,1,crl=AVICX)
        call chmdealloc('tsmp.src','TSMP','DAPLT',1,crl=DAPLT)
        call chmdealloc('tsmp.src','TSMP','DAX',1,crl=DAX)
        call chmdealloc('tsmp.src','TSMP','DEPLT',1,crl=DEPLT)
        call chmdealloc('tsmp.src','TSMP','DSPLT',1,crl=DSPLT)
        call chmdealloc('tsmp.src','TSMP','DEX',1,crl=DEX)
        call chmdealloc('tsmp.src','TSMP','DSX',1,crl=DSX)
     END IF
  END IF

  IF(ISITTI) THEN
     call chmdealloc('tsmp.src','TSMP','AVPL',PSTKSZ,crl=AVPL)
     call chmdealloc('tsmp.src','TSMP','AV2PL',PSTKSZ,crl=AV2PL)
     call chmdealloc('tsmp.src','TSMP','LMBDPL',PSTKSZ,crl=LMBDPL)
     call chmdealloc('tsmp.src','TSMP','AINCR',PSTKSZ+1,crl=AINCR)
     IF(CALDER) THEN
        call chmdealloc('tsmp.src','TSMP','AVAPL',PSTKSZ,crl=AVAPL)
        call chmdealloc('tsmp.src','TSMP','AV2APL',PSTKSZ,crl=AV2APL)
        call chmdealloc('tsmp.src','TSMP','AVBPL',PSTKSZ,crl=AVBPL)
        call chmdealloc('tsmp.src','TSMP','AV2BPL',PSTKSZ,crl=AV2BPL)
     ELSE
        call chmdealloc('tsmp.src','TSMP','AVAPL',1,crl=AVAPL)
        call chmdealloc('tsmp.src','TSMP','AVBPL',1,crl=AVBPL)
     END IF
     call chmdealloc('tsmp.src','TSMP','AVDWIN',PSTKSZ+1,crl=AVDWIN)
     call chmdealloc('tsmp.src','TSMP','AVDWPL',PSTKSZ,crl=AVDWPL)
     call chmdealloc('tsmp.src','TSMP','AVDW2PL',PSTKSZ,crl=AVDW2PL)
     call chmdealloc('tsmp.src','TSMP','AELEPL',PSTKSZ,crl=AELEPL)
     call chmdealloc('tsmp.src','TSMP','AELE2PL',PSTKSZ,crl=AELE2PL)
     call chmdealloc('tsmp.src','TSMP','AINTPL',PSTKSZ,crl=AINTPL)
     call chmdealloc('tsmp.src','TSMP','AINT2PL',PSTKSZ,crl=AINT2PL)
     call chmdealloc('tsmp.src','TSMP','AELEIN',PSTKSZ+1,crl=AELEIN)
     call chmdealloc('tsmp.src','TSMP','AINTIN',PSTKSZ+1,crl=AINTIN)
  ELSE
     call chmdealloc('tsmp.src','TSMP','LMBDPL',PSTKSZ,crl=LMBDPL)
     call chmdealloc('tsmp.src','TSMP','SYMB',PSTKSZ,log=SYMB)
     IF(CALDER) THEN
        call chmdealloc('tsmp.src','TSMP','DAPL',PSTKSZ,crl=DAPL)
        call chmdealloc('tsmp.src','TSMP','DEPL',PSTKSZ,crl=DEPL)
        call chmdealloc('tsmp.src','TSMP','DSPL',PSTKSZ,crl=DSPL)
     ELSE
        call chmdealloc('tsmp.src','TSMP','DAPL',1,crl=DAPL)
        call chmdealloc('tsmp.src','TSMP','DEPL',1,crl=DEPL)
        call chmdealloc('tsmp.src','TSMP','DSPL',1,crl=DSPL)
     END IF
  END IF

  RETURN
END SUBROUTINE TSMP


SUBROUTINE PPOST(PTUNIT,NUNIT,PSTKSZ,LMBDAP,NSTPBN,TEMP,DT, &
     PLOT,ERR,DATOT,DETOT,DSTOT,DA2TOT,DE2TOT,DS2TOT,DAPL,DEPL, &
     DSPL,LMBDPL,SYMB,NPL,UMBRDO,TEMPDO,CALDER,LEAVG,NSKIP,NMAX)
  !

  use chm_kinds
  use dimens_fcm
  use consta
  use ctitla
  use number
  use stream
  use string

  implicit none
  !
  LOGICAL ERR,PLOT
  real(chm_real) WUMBR,WUMBRB,WUMBRA
  LOGICAL UMBRDO,TEMPDO
  real(chm_real) LMBDPL(*),DAPL(*),DEPL(*),DSPL(*)
  INTEGER PTUNIT,NUNIT,NPL,I,LEFTOV,PSTKSZ,IUNCNT
  INTEGER NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
  LOGICAL SYMB(*)
  real(chm_real) KTMDT,KTPDT,KLP,KLM,KT,ELMBDA,ELMBDP,ULMBDA,ULMBDP
  real(chm_real) UREAC,UPROD,LAMBDA,LMBDAP,TEMP,DT,TMDT,TPDT,JLAMBDA
  real(chm_real) RLAMBDA,PLAMBDA,RPLAMBDA,PPLAMBDA
  real(chm_real) AV,AVTPDT,AVTMDT,AVLP,AVLM,AVTEMP,AVPE,AVKE
  real(chm_real) AV1,AV1TPD,AV1TMD,AV1LP,AV1LM,AV1PE,AV1KE
  real(chm_real) AV2,AV2TPD,AV2TMD,AV2LP,AV2LM,AV2PE,AV2KE
  real(chm_real) DA,DE,DS,DLNZ,ECP,ECM,ECP1,ECM1,EAVG
  real(chm_real) DA2,DE2,DS2
  real(chm_real) AVBIN,AVTPBN,AVTMBN,AVLPBN,AVLMBN,AVPEBN,AVKEBN
  real(chm_real) DATOT,DA2TOT,DETOT,DE2TOT,DSTOT,DS2TOT
  INTEGER N,IOS,M,NSTPBN,NBINS,NNN
  real(chm_real) MMM1,XX1,YY1,YY2,YY3,YY4,YY5,YY6,YY7,YY8
  real(chm_real) AVV0,AVV1,AVV2,AVV3,AVV4
  CHARACTER(len=80) :: ELINE
  INTEGER NMAX,ISKIP,NSKIP
  INTEGER ONSTEP,OPERFRQ,ONDEGF,ONPUMB,OLPOWER
  LOGICAL GO,CALDER,DOIT,LEAVG

  ERR = .FALSE.
  IF (IOLEV < 0) RETURN
  !
  ! Read the title and parameter line from the pert file
  !
  ISKIP = 0
  loop4000: DO IUNCNT = PTUNIT,PTUNIT+NUNIT-1
     CALL RDTITL(TITLEB,NTITLB,IUNCNT,0)
     READ(IUNCNT,'(5(i6,1x))',IOSTAT=IOS,END=950,ERR=950) &
          NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
950  CONTINUE
     IF(IOS /= 0) THEN
        IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3,A)') &
             'Empty file or eof on unit ',PTUNIT,' reading lambda line.'
        CALL WRNDIE(0,'<ppost>',ELINE(1:STRLNG(ELINE)))
        ERR=.TRUE.
        RETURN
     END IF
     IF (IUNCNT == PTUNIT) THEN
        ONSTEP = NSTEP
        OPERFRQ = PERFRQ
        ONDEGF = NDEGF
        ONPUMB = NPUMB
        OLPOWER = LPOWER
     ELSE IF( ONSTEP /= NSTEP.OR.OPERFRQ /= PERFRQ.OR. &
          ONDEGF /= NDEGF.OR.ONPUMB /= NPUMB.OR. &
          OLPOWER /= LPOWER) THEN
        IF(WRNLEV >= 2) THEN
           WRITE(OUTU,*) 'Headers do not match'
           WRITE(OUTU,*) 'Old:'
           WRITE(OUTU,'(1X,A)') 'nstep,perfrq,ndegf,npumb,lpower'
           WRITE(OUTU,'(1X,5(I6,1X))') &
                NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
           WRITE(OUTU,*) 'New:'
           WRITE(OUTU,'(1X,A)') 'onstep,operfrq,ondegf,onpumb,olpower'
           WRITE(OUTU,'(1X,5(I6,1X))') &
                ONSTEP,OPERFRQ,ONDEGF,ONPUMB,OLPOWER
        ENDIF
        CALL DIEWRN(-2)
        ERR = .TRUE.
        RETURN
     ENDIF
     IF(LPOWER == 0) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'LPOWER is equal to zero.  Assuming old file'
        IF(PRNLEV >= 2) WRITE(OUTU,*) &
             'format and setting it to one.'
        LPOWER = 1
     END IF
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'nstep,perfrq,ndegf,npumb,lpower'
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,5(I6,1X))') &
          NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
     IF(UMBRDO.AND.NPUMB == 0) THEN
        CALL WRNDIE(0,'<PPOST>', &
             'UMBRella option chosen but npumb = 0')
        ERR = .TRUE.
        RETURN
     END IF
     !
     !  lambda prime dependence
     !
     RPLAMBDA = (ONE-LMBDAP)**LPOWER
     PPLAMBDA = LMBDAP**LPOWER
     IF (IUNCNT == PTUNIT) THEN
        IF(CALDER) THEN
           TPDT = TEMP + DT
           TMDT = TEMP - DT
           KLP = (ONE/TPDT - ONE/TEMP)/KBOLTZ
           KLM = (ONE/TMDT - ONE/TEMP)/KBOLTZ
           KTPDT = KBOLTZ*TPDT
           KTMDT = KBOLTZ*TMDT
           AVTPDT = ZERO
           AVTMDT = ZERO
           AVLP = ZERO
           AVLM = ZERO
           AVTPBN = ZERO
           AVTMBN = ZERO
           AVLPBN = ZERO
           AVLMBN = ZERO
           AV1TPD = ZERO
           AV1TMD = ZERO
           AV1LP = ZERO
           AV1LM = ZERO
           AV2TPD = ZERO
           AV2TMD = ZERO
           AV2LP = ZERO
           AV2LM = ZERO
        END IF
        !
        M = 0
        N = 0
        NBINS = 0
        KT = KBOLTZ*TEMP
        AV = ZERO
        AVBIN = ZERO
        AV2 = ZERO
        AVTEMP = ZERO
        WUMBRB = ZERO
        !
        AV1 = ZERO
        WUMBR = ZERO
        WUMBRA = ZERO
        IF (LEAVG) THEN
           AVPE = ZERO
           AVKE = ZERO
           AVPEBN = ZERO
           AVKEBN = ZERO
           AV1PE = ZERO
           AV1KE = ZERO
           AV2PE = ZERO
           AV2KE = ZERO
        ENDIF
        !
        !         !scaling constant
        IF(UMBRDO) THEN
           READ(IUNCNT,1000,END=1010,ERR=1010,IOSTAT=IOS) &
                NNN,XX1,LAMBDA,ELMBDA,UREAC,UPROD,YY1,YY2,YY3,YY4,YY5, &
                YY6,YY7,YY8,WUMBR
1000       FORMAT(I12,2(1X,G24.16),/,3(2(G24.16,1X),G24.16,/), &
                2(G24.16,1X),G24.16)
1010       CONTINUE
        ELSE
           READ(IUNCNT,1100,END=1110,ERR=1110,IOSTAT=IOS) &
                NNN,XX1,LAMBDA,ELMBDA,UREAC,UPROD,YY1,YY2,YY3,YY4,YY5, &
                YY6,YY7,YY8
1100       FORMAT(I12,2(1X,G24.16),/,3(2(G24.16,1X),G24.16,/), &
                G24.16,1X,G24.16)
1110       CONTINUE
        END IF
        IF(IOS /= 0) THEN
           IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3,A)') &
                'Empty file or end of file on unit ',PTUNIT,' scaling record.'
           CALL WRNDIE(0,'<ppost>',ELINE)
           ERR=.TRUE.
           RETURN
        END IF
        RLAMBDA = (ONE-LAMBDA)**LPOWER
        PLAMBDA = LAMBDA**LPOWER
        ULMBDA =  RLAMBDA*UREAC + PLAMBDA*UPROD
        ULMBDP =  RPLAMBDA*UREAC +PPLAMBDA*UPROD
        IF(CALDER) THEN
           ELMBDP = ELMBDA -ULMBDA + ULMBDP
           ECP  =  -ELMBDP/KTPDT + ELMBDA/KT
           ECM  =  -ELMBDP/KTMDT + ELMBDA/KT
           ECP1 =  -ELMBDA*KLP
           ECM1 =  -ELMBDA*KLM
           EAVG = ECP - ECM - ECP1 + ECM1
        END IF
        DO I = 1,5
           BACKSPACE (UNIT=IUNCNT,IOSTAT=IOS)
        enddo
        IF(PRNLEV >= 2) WRITE(OUTU,*) &
             'Thermodynamic Properties Calculations: lambda = ', &
             LAMBDA,' lambda prime: ',LMBDAP,' lpower: ',LPOWER
     END IF
     GO = .TRUE.
     ! *************************************************************************
     !         do while(go)
2000 IF (GO)  THEN
        ! *************************************************************************
        IF(UMBRDO) THEN
           READ(IUNCNT,1000,IOSTAT=IOS,END=1210,ERR=1210) &
                NNN,XX1,JLAMBDA,ELMBDA, &
                UREAC,UPROD,YY1,YY2,YY3,YY4,YY5,YY6,YY7,YY8,WUMBR
           WUMBR = EXP(-WUMBR/KT)
1210       CONTINUE
        ELSE
           READ(IUNCNT,1100,IOSTAT=IOS,END=1220,ERR=1220) &
                NNN,XX1,JLAMBDA,ELMBDA, &
                UREAC,UPROD,YY1,YY2,YY3,YY4,YY5,YY6,YY7,YY8
1220       CONTINUE
        END IF
        IF (IOS /= 0) THEN
           IF (N == 0) THEN
              IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3,A)') &
                   'Empty file or end of file on unit ',IUNCNT,' data record.'
              CALL WRNDIE(0,'<ppost>',ELINE)
              ERR=.TRUE.
              RETURN
           ELSE
              REWIND (UNIT=IUNCNT,IOSTAT=IOS)
              cycle loop4000
           END IF
        END IF
        ISKIP = ISKIP + 1
        DOIT = (NSKIP > 0.AND.ISKIP > NSKIP)
        IF (DOIT .OR. NSKIP == 0) THEN
           M = M + 1
           N = N + 1
           IF (NMAX  >  0 .AND. N  >  NMAX) THEN
              N = N - 1
              M =  M - 1
              REWIND (UNIT=IUNCNT,IOSTAT=IOS)
              exit loop4000
           ENDIF
           ULMBDA = RLAMBDA*UREAC + PLAMBDA*UPROD
           ULMBDP = RPLAMBDA*UREAC + PPLAMBDA*UPROD
           IF(CALDER) ELMBDP = ELMBDA -ULMBDA + ULMBDP
           !
           IF (TEMPDO) AVTEMP = AVTEMP + YY8
           IF(UMBRDO) THEN
              WUMBRA = WUMBRA + ONE/WUMBR
              WUMBRB = WUMBRB + ONE/WUMBR
           END IF
           AVV0 = EXP( -(ULMBDP-ULMBDA)/KT)
           IF(CALDER) THEN
              AVV1 = EXP( -ELMBDP/KTPDT + ELMBDA/KT-ECP)
              AVV2 = EXP( -ELMBDP/KTMDT + ELMBDA/KT-ECM)
              AVV3 = EXP( -ELMBDA*KLP-ECP1)
              AVV4 = EXP( -ELMBDA*KLM-ECM1)
           END IF
           IF (UMBRDO) THEN
              AVV0 = AVV0/WUMBR
              IF(LEAVG) ELMBDA = ELMBDA/WUMBR
              IF(CALDER) THEN
                 AVV1 = AVV1/WUMBR
                 AVV2 = AVV2/WUMBR
                 AVV3 = AVV3/WUMBR
                 AVV4 = AVV4/WUMBR
              END IF
           END IF
           AV = AV + AVV0
           IF (LEAVG) THEN
              AVPE=AVPE + ELMBDA
              AVKE=AVKE + YY8
           ENDIF
           IF (CALDER) THEN
              AVTPDT = AVTPDT + AVV1
              AVTMDT = AVTMDT + AVV2
              AVLP = AVLP + AVV3
              AVLM = AVLM + AVV4
           END IF
           !
           AVBIN = AVBIN + AVV0
           IF (LEAVG) THEN
              AVPEBN = AVPEBN + ELMBDA
              AVKEBN = AVKEBN + YY8
           ENDIF
           IF(CALDER) THEN
              AVTPBN = AVTPBN + AVV1
              AVTMBN = AVTMBN + AVV2
              AVLPBN = AVLPBN + AVV3
              AVLMBN = AVLMBN + AVV4
           END IF
           !
           IF (M == NSTPBN) THEN
              M = 0
              NBINS = NBINS + 1
              !
              IF (LEAVG) AV2KE = AV2KE + (AVKEBN/NSTPBN)**2
              IF (UMBRDO) THEN
                 AV1 = AV1 + (AVBIN/WUMBRB)
                 AV2 = AV2 + (AVBIN/WUMBRB)**2
                 IF (LEAVG) THEN
                    AV1PE = AV1PE + AVPEBN/WUMBRB
                    AV2PE = AV2PE + (AVPEBN/WUMBRB)**2
                 ENDIF
                 IF (CALDER) THEN
                    AV1TPD = AV1TPD + (AVTPBN/WUMBRB)
                    AV1TMD = AV1TMD + (AVTMBN/WUMBRB)
                    AV1LP = AV1LP + (AVLPBN/WUMBRB)
                    AV1LM = AV1LM + (AVLMBN/WUMBRB)
                    AV2TPD = AV2TPD + (AVTPBN/WUMBRB)**2
                    AV2TMD = AV2TMD + (AVTMBN/WUMBRB)**2
                    AV2LP = AV2LP + (AVLPBN/WUMBRB)**2
                    AV2LM = AV2LM + (AVLMBN/WUMBRB)**2
                 END IF
              ELSE
                 AV2 = AV2 + (AVBIN/NSTPBN)**2
                 IF (LEAVG) AV2PE = AV2PE + (AVPEBN/NSTPBN)**2
                 IF (CALDER) THEN
                    AV2TPD = AV2TPD + (AVTPBN/NSTPBN)**2
                    AV2TMD = AV2TMD + (AVTMBN/NSTPBN)**2
                    AV2LP = AV2LP + (AVLPBN/NSTPBN)**2
                    AV2LM = AV2LM + (AVLMBN/NSTPBN)**2
                 END IF
              END IF
              !
              AVBIN = ZERO
              IF (LEAVG) THEN
                 AVPEBN = ZERO
                 AVKEBN = ZERO
              ENDIF
              IF (CALDER) THEN
                 AVTPBN = ZERO
                 AVTMBN = ZERO
                 AVLPBN = ZERO
                 AVLMBN = ZERO
              END IF
              WUMBRB = ZERO
              !
           END IF
        ENDIF
        !
        ! *************************************************************************
        !         end do
        GOTO 2000
     ENDIF
     ! *************************************************************************
  enddo loop4000

  REWIND (UNIT=IUNCNT,IOSTAT=IOS)
  !
  IF(TEMPDO) AVTEMP = TWO*(AVTEMP/N)/(KBOLTZ*NDEGF)
  IF (LEAVG) AVKE = AVKE/N
  IF (UMBRDO) THEN
     AV =AV/WUMBRA
     IF (LEAVG) AVPE = AVPE/WUMBRA
     IF (CALDER) THEN
        AVTPDT = AVTPDT/WUMBRA
        AVTMDT = AVTMDT/WUMBRA
        AVLP = AVLP/WUMBRA
        AVLM = AVLM/WUMBRA
     END IF
  ELSE
     AV =AV/N
     IF (LEAVG) AVPE = AVPE/N
     IF (CALDER) THEN
        AVTPDT = AVTPDT/N
        AVTMDT = AVTMDT/N
        AVLP = AVLP/N
        AVLM = AVLM/N
     END IF
  END IF
  !
  MMM1 = ONE/(NBINS*(NBINS-1))
  IF(LEAVG) AV2KE = MMM1*(AV2KE - NBINS*(AVKE**2))
  IF(UMBRDO) THEN
     AV2 = MMM1*(AV2 + AV*(NBINS*AV - TWO*AV1))
     IF (LEAVG) AV2PE =MMM1*(AV2PE + AVPE*(NBINS*AVPE - TWO*AV1PE))
     IF (CALDER) THEN
        AV2TPD = MMM1*(AV2TPD + AVTPDT*(NBINS*AVTPDT - TWO*AV1TPD))
        AV2TMD = MMM1*(AV2TMD + AVTMDT*(NBINS*AVTMDT - TWO*AV1TMD))
        AV2LP = MMM1*(AV2LP + AVLP*(NBINS*AVLP - TWO*AV1LP))
        AV2LM = MMM1*(AV2LM + AVLM*(NBINS*AVLM - TWO*AV1LM))
     END IF
  ELSE
     AV2 = MMM1*(AV2 - NBINS*(AV**2))
     IF(LEAVG) AV2PE = MMM1*(AV2PE - NBINS*(AVPE**2))
     IF (CALDER) THEN
        AV2TPD = MMM1*(AV2TPD - NBINS*(AVTPDT**2))
        AV2TMD = MMM1*(AV2TMD - NBINS*(AVTMDT**2))
        AV2LP = MMM1*(AV2LP - NBINS*(AVLP**2))
        AV2LM = MMM1*(AV2LM - NBINS*(AVLM**2))
     END IF
  END IF
  !
  DA = -KT*LOG(AV)
  IF (CALDER) THEN
     !         contains correction for scaling (eavg):
     DLNZ = ( LOG(AVTPDT) - LOG(AVTMDT) &
          -  LOG(AVLP)   + LOG(AVLM) ) + EAVG
     DLNZ = (ONE/(TWO*DT))*DLNZ
     DS = KBOLTZ*LOG(AV) + KT*DLNZ
     DE = KT*TEMP*DLNZ
  END IF
  !
  DA2 = AV2/(AV*AV)
  IF (CALDER) THEN
     DS2 = ((KT/(TWO*DT))**2)*(AV2TPD/(AVTPDT*AVTPDT) &
          + AV2TMD/(AVTMDT*AVTMDT) + AV2LP/(AVLP*AVLP) &
          + AV2LM/(AVLM*AVLM))
     DE2 = TEMP*TEMP*DS2
     DS2 = KBOLTZ*KBOLTZ*DA2 + DS2
     DA2 = KT*KT*DA2
  END IF
  !
  IF(LMBDAP >= LAMBDA) THEN
     DATOT = DATOT + DA
  ELSE
     DATOT = DATOT - DA
  END IF
  DA2TOT = DA2TOT + DA2
  DA2 = SQRT(DA2)
  IF(CALDER) THEN
     IF(LMBDAP >= LAMBDA) THEN
        DETOT = DETOT + DE
        DSTOT = DSTOT + DS
     ELSE
        DETOT = DETOT - DE
        DSTOT = DSTOT - DS
     END IF
     DE2TOT = DE2TOT + DE2
     DE2 = SQRT(DE2)
     DS2TOT = DS2TOT + DS2
     DS2 = SQRT(DS2)
  END IF
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,*) 'Perturbation results: lambda = ',LAMBDA, &
          ' lambda''= ',LMBDAP,' # of steps: ',N
     WRITE(OUTU,*) 'temperature: ',TEMP,' delta T: ',DT
     WRITE(OUTU,*) 'delta A = ',DA,' +/-',DA2
     IF (CALDER) THEN
        WRITE(OUTU,*) 'delta E = ',DE,' +/-',DE2
        WRITE(OUTU,*) 'delta S = ',DS,' +/-',DS2
     END IF
     IF (LEAVG) THEN
        WRITE(OUTU,*) '<P.E.> = ',AVPE,' +/-',SQRT(AV2PE)
        WRITE(OUTU,*) '<K.E.> = ',AVKE,' +/-',SQRT(AV2KE)
        WRITE(OUTU,*) '<Total E.> = ',AVKE+AVPE,' +/-', &
             SQRT(AV2KE + AV2PE)
     ENDIF
     WRITE(OUTU,*) 'Used ',NBINS,' bins of ',NSTPBN,' steps.'
     IF(TEMPDO) WRITE(OUTU,'(1X,A,F7.2,A)')'Average Temp ',AVTEMP,'K'
  ENDIF
  LEFTOV = N-NBINS*NSTPBN
  IF (LEFTOV /= 0 .AND. WRNLEV >= 2) THEN
     WRITE(OUTU,*) 'Number of steps/bin did not divide into', &
          ' number of steps.'
     WRITE(OUTU,*) LEFTOV, &
          ' steps not included in error analysis'
     WRITE(OUTU,*) 'but were included in average.'
  END IF
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) 'Story so far: '
     WRITE(OUTU,*) 'delta A(lambda) = ',DATOT
     IF (CALDER) THEN
        WRITE(OUTU,*) 'delta E(lambda) = ',DETOT
        WRITE(OUTU,*) 'delta S(lambda) = ',DSTOT
     END IF
     IF(NDEGF /= 0) WRITE(OUTU,*) 'average temp: ',AVTEMP
  ENDIF
  !
  IF(.NOT.PLOT) RETURN
  NPL = NPL +1
  IF (NPL > PSTKSZ) THEN
     CALL WRNDIE(0,'<PPOST>', &
          'number of plot points exceeds maximum set with PSTAck command')
     ERR = .TRUE.
     RETURN
  END IF
  IF(LAMBDA > LMBDAP) THEN
     LMBDPL(NPL) = LAMBDA
     SYMB(NPL) = .TRUE.
  ELSE
     LMBDPL(NPL) = LMBDAP
     SYMB(NPL) = .FALSE.
  END IF
  DAPL(NPL) = DATOT
  IF (CALDER) THEN
     DEPL(NPL) = DETOT
     DSPL(NPL) = DSTOT
  END IF
  !
  RETURN
END SUBROUTINE PPOST

SUBROUTINE PSTTOT(PLOT,ERR, &
     DATOT,DETOT,DSTOT,DA2TOT,DE2TOT,DS2TOT, &
     DAPL,DEPL,DSPL,LMBDPL,SYMB,NPL,CALDER)

  use chm_kinds
  use dimens_fcm
  use stream
  implicit none
  LOGICAL ERR,PLOT,SYMB(*),CALDER
  real(chm_real) LMBDPL(*),DAPL(*),DEPL(*),DSPL(*)
  real(chm_real) DATOT,DA2TOT,DETOT,DE2TOT,DSTOT,DS2TOT
  INTEGER I,NPL
  !
  DA2TOT = SQRT(DA2TOT)
  IF (CALDER) THEN
     DE2TOT = SQRT(DE2TOT)
     DS2TOT = SQRT(DS2TOT)
  END IF
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,'(1X,72(''-''))')
     WRITE(OUTU,*) 'Totals:'
     WRITE(OUTU,*) 'delta A = ',DATOT,' +/-',DA2TOT
     IF (CALDER) THEN
        WRITE(OUTU,*) 'delta E = ',DETOT,' +/-',DE2TOT
        WRITE(OUTU,*) 'delta S = ',DSTOT,' +/-',DS2TOT
     END IF
  ENDIF
  !
  IF(PRNLEV <= 3) RETURN
  IF(.NOT.PLOT) RETURN
  WRITE(OUTU,'(///)')
  WRITE(OUTU,*) 'plot files'
  WRITE(OUTU,'(//,'' delta A'')')
  WRITE(OUTU,*) '0. 0. "O"'
  DO I = 1,NPL
     IF(SYMB(I)) THEN
        WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
             LMBDPL(I),DAPL(I),'"X"'
     ELSE
        WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
             LMBDPL(I),DAPL(I),'"O"'
     END IF
  enddo
  IF (CALDER) THEN
     WRITE(OUTU,'(//,'' delta E'')')
     WRITE(OUTU,*) '0. 0. "O"'
     DO I = 1,NPL
        IF(SYMB(I)) THEN
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),DEPL(I),'"X"'
        ELSE
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),DEPL(I),'"O"'
        END IF
     enddo
     WRITE(OUTU,'(//,'' delta S'')')
     WRITE(OUTU,*) '0. 0. "O"'
     DO I = 1,NPL
        DSPL(I) = DSPL(I)*1000.
        IF(SYMB(I)) THEN
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),DSPL(I),'"X"'
        ELSE
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),DSPL(I),'"O"'
        END IF
     enddo
  END IF
  RETURN
END SUBROUTINE PSTTOT

SUBROUTINE PPOSTTI(PTUNIT,NUNIT,PSTKSZ,NSTPBN,TEMP, &
     ERR,LMBDPL,AVPL,AV2PL,AVAPL,AV2APL,AVBPL,AV2BPL,NPL, &
     UMBRDO,TEMPDO,CALDER,ONEP,ZEROP, &
     AVDWPL,AVDW2PL,AELEPL,AELE2PL,AINTPL,AINT2PL,LCOMP,NSKIP,NMAX)
  !

  use chm_kinds
  use dimens_fcm
  use consta
  use ctitla
  use number
  use stream
  implicit none
  !
  LOGICAL ERR
  real(chm_real) WUMBR,WUMBRB,WUMBRA,KT,TEMP
  LOGICAL UMBRDO,TEMPDO
  real(chm_real) LMBDPL(*),AVPL(*),AV2PL(*),AVAPL(*),AV2APL(*), &
       AVTEMP
  real(chm_real) AVBPL(*),AV2BPL(*)
  real(chm_real) AVDWPL(*),AVDW2PL(*),AELEPL(*), &
       AELE2PL(*),AINTPL(*),AINT2PL(*)
  real(chm_real) AVDW,AVDW2,AELE,AELE2,AINT,AINT2
  real(chm_real) AVDWBIN,AVDW1,AELEBIN,AELE1,AINTBIN,AINT1
  real(chm_real) VPRTVR,VPRTVP,VPRTER,VPRTEP,DULMVA,DULMEA,DULMIA
  LOGICAL LCOMP
  INTEGER PTUNIT,NUNIT,NPL,LEFTOV,PSTKSZ,IUNCNT
  INTEGER NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
  real(chm_real)  ELMBDA,DULMBDA,EDULMBDA
  real(chm_real) UREAC,UPROD,LAMBDA
  real(chm_real) RLAMBDA,PLAMBDA,DRLAMBDA,DPLAMBDA
  real(chm_real) AV,AVA,AVB
  real(chm_real) AV1,AV1A,AV1B
  real(chm_real) AV2,AV2A,AV2B
  real(chm_real) AVBIN,AVBINA,AVBINB
  INTEGER N,IOS,M,NSTPBN,NBINS,NNN
  real(chm_real) MMM1,XX1,YY1,YY2,YY7,YY8
  CHARACTER(len=80) :: ELINE
  LOGICAL GO,CALDER,LREAD,ONEP,ZEROP,DOIT
  INTEGER ONSTEP,OPERFRQ,ONDEGF,ONPUMB,OLPOWER
  INTEGER NSKIP,ISKIP,NMAX
  !
  ! Thermodynamic Integration Post-Processing
  !

  KT = KBOLTZ*TEMP
  ERR = .FALSE.
  IF (IOLEV < 0) RETURN
  !
  ! Read the title and parameter line from the pert file
  !
  ISKIP = 0
  loop4000: DO IUNCNT = PTUNIT,PTUNIT+NUNIT-1
     CALL RDTITL(TITLEB,NTITLB,IUNCNT,0)
     READ(IUNCNT,'(5(i6,1x))',IOSTAT=IOS,END=950,ERR=950) &
          NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
950  CONTINUE
     IF(IOS /= 0) THEN
        IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
             'Empty file or end of file on unit ',PTUNIT
        CALL WRNDIE(0,'<ppost>',ELINE)
        ERR=.TRUE.
        RETURN
     END IF
     IF (IUNCNT == PTUNIT) THEN
        ONSTEP = NSTEP
        OPERFRQ = PERFRQ
        ONDEGF = NDEGF
        ONPUMB = NPUMB
        OLPOWER = LPOWER
     ELSE IF( ONSTEP /= NSTEP.OR.OPERFRQ /= PERFRQ.OR. &
          ONDEGF /= NDEGF.OR.ONPUMB /= NPUMB.OR. &
          OLPOWER /= LPOWER) THEN
        IF(WRNLEV >= 2) THEN
           WRITE(OUTU,*) 'Headers do not match'
           WRITE(OUTU,*) 'Old:'
           WRITE(OUTU,'(1X,A)') 'nstep,perfrq,ndegf,npumb,lpower'
           WRITE(OUTU,'(1X,5(I6,1X))') &
                NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
           WRITE(OUTU,*) 'New:'
           WRITE(OUTU,'(1X,A)') 'onstep,operfrq,ondegf,onpumb,olpower'
           WRITE(OUTU,'(1X,5(I6,1X))') &
                ONSTEP,OPERFRQ,ONDEGF,ONPUMB,OLPOWER
        ENDIF
        CALL DIEWRN(-2)
        ERR = .TRUE.
        RETURN
     ENDIF
     IF(LPOWER == 0) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'LPOWER is equal to zero.  Assuming old file'
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'format and setting it to one.'
        LPOWER = 1
     END IF
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'nstep,perfrq,ndegf,npumb,lpower'
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,5(I6,1X))') &
          NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
     IF(UMBRDO.AND.NPUMB == 0) THEN
        CALL WRNDIE(0,'<PPOST>', &
             'UMBRella option chosen but npumb = 0')
        ERR = .TRUE.
        RETURN
     END IF
     IF (IUNCNT == PTUNIT) THEN
        !
        M = 0
        N = 0
        NBINS = 0
        AV = ZERO
        AVBIN = ZERO
        AV2 = ZERO
        AVTEMP = ZERO
        WUMBRB = ZERO
        IF(CALDER) THEN
           AVA = ZERO
           AVB = ZERO
        END IF
        !
        AV1 = ZERO
        IF(CALDER) THEN
           AV1A = ZERO
           AV1B = ZERO
        END IF
        WUMBR = ZERO
        WUMBRA = ZERO
        !
        IF(CALDER) THEN
           AVBINA = ZERO
           AVBINB = ZERO
           AV2A = ZERO
           AV2B = ZERO
        END IF
        IF(LCOMP) THEN
           AVDWBIN = ZERO
           AVDW1 = ZERO
           AELEBIN = ZERO
           AELE1 = ZERO
           AINTBIN = ZERO
           AINT1 = ZERO
           AVDW = ZERO
           AVDW2 = ZERO
           AELE = ZERO
           AELE2 = ZERO
           AINT = ZERO
           AINT2 = ZERO
        END IF
     END IF
     GO = .TRUE.
     LREAD = .FALSE.
1000 FORMAT(I12,2(1X,G24.16),/,3(2(G24.16,1X),G24.16,/), &
          2(G24.16,1X),G24.16)
1100 FORMAT(I12,2(1X,G24.16),/,3(2(G24.16,1X),G24.16,/), &
          G24.16,1X,G24.16)
     ! *************************************************************************
     !         do while(go)
2000 IF (GO) THEN
        ! *************************************************************************
        IF(UMBRDO) THEN
           READ(IUNCNT,1000,IOSTAT=IOS,END=1210,ERR=1210) &
                NNN,XX1,LAMBDA,ELMBDA, &
                UREAC,UPROD,YY1,YY2,VPRTVR,VPRTVP,VPRTER,VPRTEP, &
                YY7,YY8,WUMBR
           WUMBR = EXP(-WUMBR/KT)
1210       CONTINUE
        ELSE
           READ(IUNCNT,1100,IOSTAT=IOS,END=1220,ERR=1220) &
                NNN,XX1,LAMBDA,ELMBDA, &
                UREAC,UPROD,YY1,YY2,VPRTVR,VPRTVP,VPRTER,VPRTEP, &
                YY7,YY8
1220       CONTINUE
        END IF
        IF(IOS /= 0) THEN
           IF(N == 0) THEN
              IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
                   'Empty file or end of file on unit ',PTUNIT
              CALL WRNDIE(0,'<ppost>',ELINE)
              ERR=.TRUE.
              RETURN
           ELSE
              REWIND (UNIT=IUNCNT,IOSTAT=IOS)
              cycle loop4000
           END IF
        END IF
        ISKIP = ISKIP + 1
        DOIT = (NSKIP > 0.AND.ISKIP > NSKIP)
        IF (DOIT .OR. NSKIP == 0) THEN
           M = M + 1
           N = N + 1
           IF (NMAX  >  0 .AND. N  ==  NMAX) THEN
              N = N - 1
              M =  M - 1
              REWIND (UNIT=IUNCNT,IOSTAT=IOS)
              exit loop4000
           END IF
           IF(.NOT.LREAD) THEN
              LREAD = .TRUE.
              IF (LPOWER == 1) THEN
                 RLAMBDA = ONE-LAMBDA
                 PLAMBDA = LAMBDA
                 DRLAMBDA = MINONE
                 DPLAMBDA = ONE
              ELSE IF(ZEROP) THEN
                 RLAMBDA = ONE
                 PLAMBDA = ZERO
                 DRLAMBDA = -LPOWER
                 DPLAMBDA = ZERO
              ELSE IF (ONEP) THEN
                 RLAMBDA = ZERO
                 PLAMBDA = ONE
                 DRLAMBDA = ZERO
                 DPLAMBDA = LPOWER
              ELSE
                 RLAMBDA = (ONE-LAMBDA)**LPOWER
                 PLAMBDA = LAMBDA**LPOWER
                 DRLAMBDA = -LPOWER*((ONE-LAMBDA)**(LPOWER-1))
                 DPLAMBDA = LPOWER*(LAMBDA**(LPOWER-1))
              END IF
           END IF
           !              dE(lambda)/dlambda
           DULMBDA = DRLAMBDA*UREAC + DPLAMBDA*UPROD
           IF(LCOMP) THEN
              DULMVA = DRLAMBDA*VPRTVR + DPLAMBDA*VPRTVP
              DULMEA = DRLAMBDA*VPRTER + DPLAMBDA*VPRTEP
              DULMIA = DULMBDA - DULMVA - DULMEA
           END IF
           !              E(lambda)dE(lambda)/dlambda
           IF(CALDER) EDULMBDA = ELMBDA*DULMBDA
           !
           IF (TEMPDO) AVTEMP = AVTEMP + YY8
           IF(UMBRDO) THEN
              WUMBRA = WUMBRA + ONE/WUMBR
              WUMBRB = WUMBRB + ONE/WUMBR
           END IF
           IF (UMBRDO) THEN
              DULMBDA = DULMBDA/WUMBR
              IF(CALDER) THEN
                 ELMBDA = ELMBDA/WUMBR
                 EDULMBDA = EDULMBDA/WUMBR
              END IF
              IF (LCOMP) THEN
                 DULMVA = DULMVA/WUMBR
                 DULMEA = DULMEA/WUMBR
                 DULMIA = DULMIA/WUMBR
              END IF
           END IF
           AV = AV + DULMBDA
           AVBIN = AVBIN + DULMBDA
           IF(CALDER) THEN
              AVA = AVA + ELMBDA
              AVB = AVB + EDULMBDA
              !
              AVBINA = AVBINA + ELMBDA
              AVBINB = AVBINB + EDULMBDA
           END IF
           IF (LCOMP) THEN
              AVDW = AVDW + DULMVA
              AELE = AELE + DULMEA
              AINT = AINT + DULMIA
              AVDWBIN = AVDWBIN + DULMVA
              AELEBIN = AELEBIN + DULMEA
              AINTBIN = AINTBIN + DULMIA
           END IF
           !
           IF (M == NSTPBN) THEN
              M = 0
              NBINS = NBINS + 1
              !
              IF (UMBRDO) THEN
                 AV1 = AV1 + (AVBIN/WUMBRB)
                 AV2 = AV2 + (AVBIN/WUMBRB)**2
                 IF(CALDER) THEN
                    AV1A = AV1A + AVBINA/WUMBRB
                    AV1B = AV1B + AVBINB/WUMBRB
                    AV2A = AV2A + (AVBINA/WUMBRB)**2
                    AV2B = AV2B + (AVBINB/WUMBRB)**2
                 END IF
                 IF(LCOMP) THEN
                    AVDW1 = AVDW1 + (AVDWBIN/WUMBRB)
                    AVDW2 = AVDW2 + (AVDWBIN/WUMBRB)**2
                    AELE1 = AELE1 + (AELEBIN/WUMBRB)
                    AELE2 = AELE2 + (AELEBIN/WUMBRB)**2
                    AINT1 = AINT1 + (AINTBIN/WUMBRB)
                    AINT2 = AINT2 + (AINTBIN/WUMBRB)**2
                 END IF
              ELSE
                 AV2 = AV2 + (AVBIN/NSTPBN)**2
                 IF(CALDER) THEN
                    AV2A = AV2A + (AVBINA/NSTPBN)**2
                    AV2B = AV2B + (AVBINB/NSTPBN)**2
                 END IF
                 IF (LCOMP) THEN
                    AVDW2 = AVDW2 + (AVDWBIN/NSTPBN)**2
                    AELE2 = AELE2 + (AELEBIN/NSTPBN)**2
                    AINT2 = AINT2 + (AINTBIN/NSTPBN)**2
                 END IF
              END IF
              !
              AVBIN = ZERO
              IF(CALDER) THEN
                 AVBINA = ZERO
                 AVBINB = ZERO
              END IF
              IF(LCOMP) THEN
                 AVDWBIN = ZERO
                 AELEBIN = ZERO
                 AINTBIN = ZERO
              END IF
              WUMBRB = ZERO
              !
           END IF
        ENDIF
        !
        ! *************************************************************************
        !         end do
        GOTO 2000
     END IF
     ! *************************************************************************
  enddo loop4000
  !
  REWIND (UNIT=IUNCNT,IOSTAT=IOS)
  !
  IF(TEMPDO) AVTEMP = TWO*(AVTEMP/N)/(KBOLTZ*NDEGF)
  IF (UMBRDO) THEN
     AV =AV/WUMBRA
     IF(CALDER) THEN
        AVA =AV/WUMBRA
        AVB =AV/WUMBRA
     END IF
     IF(LCOMP) THEN
        AVDW = AVDW/WUMBRA
        AELE = AELE/WUMBRA
        AINT = AINT/WUMBRA
     END IF
  ELSE
     AV =AV/N
     IF(CALDER) THEN
        AVA = AVA/N
        AVB = AVB/N
     END IF
     IF(LCOMP) THEN
        AVDW = AVDW/N
        AELE = AELE/N
        AINT = AINT/N
     END IF
  END IF
  !
  MMM1 = ONE/(NBINS*(NBINS-1))
  IF(UMBRDO) THEN
     AV2 = MMM1*(AV2 + AV*(NBINS*AV - TWO*AV1))
     IF(CALDER) THEN
        AV2A = MMM1*(AV2A + AVA*(NBINS*AVA - TWO*AV1A))
        AV2B = MMM1*(AV2B + AVB*(NBINS*AVB - TWO*AV1B))
     END IF
     IF(LCOMP) THEN
        AVDW2 = MMM1*(AVDW2 + AVDW*(NBINS*AVDW - TWO*AVDW1))
        AELE2 = MMM1*(AELE2 + AELE*(NBINS*AELE - TWO*AELE1))
        AINT2 = MMM1*(AINT2 + AINT*(NBINS*AINT - TWO*AINT1))
     END IF
  ELSE
     AV2 = MMM1*(AV2 - NBINS*(AV**2))
     IF(CALDER) THEN
        AV2A = MMM1*(AV2A - NBINS*(AVA**2))
        AV2B = MMM1*(AV2B - NBINS*(AVB**2))
     END IF
     IF(LCOMP) THEN
        AVDW2 = MMM1*(AVDW2 - NBINS*(AVDW**2))
        AELE2 = MMM1*(AELE2 - NBINS*(AELE**2))
        AINT2 = MMM1*(AINT2 - NBINS*(AINT**2))
     END IF
  END IF
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU, '(A, F0.6, A, I0)') 'Perturbation results: lambda = ',LAMBDA, &
          ' # steps: ',N
     WRITE(OUTU,*) 'temperature: ',TEMP
     WRITE(OUTU,*) 'Used ',NBINS,' bins of ',NSTPBN,' steps.'
     IF(TEMPDO) WRITE(OUTU,'(1X,A,F7.2,A)') 'Average Temp ',AVTEMP,'K'
     LEFTOV = N-NBINS*NSTPBN
     IF (LEFTOV /= 0) THEN
        WRITE(OUTU,*) 'Number of steps/bin did not divide into', &
             ' number of steps.'
        WRITE(OUTU,*) LEFTOV, &
             ' steps not included in error analysis'
        WRITE(OUTU,*) 'but were included in average.'
     END IF
     WRITE(OUTU,*) '<dE(lambda)/dlambda> ',AV,' +/- ',SQRT(AV2)
     IF(LCOMP) THEN
        WRITE(OUTU,*) &
             '<dE(lambda)/dlambda>(vdW)',AVDW,' +/- ',SQRT(AVDW2)
        WRITE(OUTU,*) &
             '<dE(lambda)/dlambda>(Ele)',AELE,' +/- ',SQRT(AELE2)
        WRITE(OUTU,*) &
             '<dE(lambda)/dlambda>(Int)',AINT,' +/- ',SQRT(AINT2)
     END IF
     IF(CALDER) THEN
        WRITE(OUTU,*)'<E(lambda)> ',AVA,' +/- ',SQRT(AV2A)
        WRITE(OUTU,*)'<E(lambda).dE(lambda)/dlambda> ',AVB,' +/- ', &
             SQRT(AV2B)
     END IF
     WRITE(OUTU,*) &
          '------------------------------------------------------'
  ENDIF
  !
  NPL = NPL +1
  IF (NPL > PSTKSZ) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,*) '<PPOST>'
     IF(WRNLEV >= 2) WRITE(OUTU,*) &
          'The number of integrand points exceeds the'
     IF(WRNLEV >= 2) WRITE(OUTU,*) &
          'maximum set with the PSTAck command.'
     CALL DIEWRN(0)
     ERR = .TRUE.
     RETURN
  END IF
  LMBDPL(NPL) = LAMBDA
  AVPL(NPL) = AV
  AV2PL(NPL) = AV2
  IF(CALDER) THEN
     AVAPL(NPL) = AVA*AV
     AV2APL(NPL) = AVA*AVA*AV2*AV2 + AV*AV*AV2A*AV2A
     AVBPL(NPL) = AVB
     AV2BPL(NPL) = AV2B
  END IF
  IF(LCOMP) THEN
     AVDWPL(NPL) = AVDW
     AVDW2PL(NPL) = AVDW2
     AELEPL(NPL) = AELE
     AELE2PL(NPL) = AELE2
     AINTPL(NPL) = AINT
     AINT2PL(NPL) = AINT2
  END IF
  !
  RETURN
END SUBROUTINE PPOSTTI

!
SUBROUTINE TITTOT(PLOT,ERR,CALDER,LMBDPL,AVPL,AV2PL, &
     AVAPL,AV2APL,AVBPL,AV2BPL,TEMP,NPL, &
     AVDWPL,AVDW2PL,AELEPL,AELE2PL,AINTPL,AINT2PL,LCOMP, &
     AINCR,AVDWIN,AELEIN,AINTIN)

  use chm_kinds
  use consta
  use number
  use param_store, only: set_param
  use stream

  implicit none

  INTEGER NPL
  real(chm_real) LMBDPL(NPL),AVPL(NPL),AV2PL(NPL),AV2TOT, &
       AVAPL(NPL),AV2ATOT
  real(chm_real) AV2APL(NPL),AVBPL(NPL),AV2BPL(NPL),AV2BTOT, &
       TEMP,BETA
  LOGICAL PLOT,ERR,CALDER
  real(chm_real) AVDWPL(NPL),AVDW2PL(NPL),AELEPL(NPL), &
       AELE2PL(NPL),AINTPL(NPL),AINT2PL(NPL),AINCR(*), &
       AVDWIN(*),AELEIN(*),AINTIN(*)
  real(chm_real) AVDW2TOT,AELE2TOT,AINT2TOT,DAVDW,DAELE,DAINT
  LOGICAL LCOMP

  real(chm_real) DA,DE,DS
  INTEGER I
  !
  IF (IOLEV < 0) RETURN
  !
  ! calculate quantities
  !
  !     Zero incremental free energy arrays
  DO I=1, NPL+1
     AINCR(I) = ZERO
     IF (LCOMP) THEN
        AVDWIN(I) = ZERO
        AELEIN(I) = ZERO
        AINTIN(I) = ZERO
     END IF
  ENDDO

  !
  !  we need sorted values of the abscissae
  IF(NPL > 1) CALL REALSORT(LMBDPL,AVPL,AVAPL,AVBPL, &
       AV2PL,AV2APL,AV2BPL,AVDWPL,AVDW2PL,AELEPL, &
       AELE2PL,AINTPL,AINT2PL,NPL)

  !
  !     Do integrations:
  !
  !     da contains integral <dE/dl> and error
  CALL TIINT(DA,AV2TOT,AVPL,LMBDPL,AV2PL,NPL,ERR,AINCR)

  IF(ERR) RETURN
  IF (CALDER) THEN
     !        de contains Integral <E(l)><dE/dl> and deerr contains the uncertainty
     CALL TIINT(DE,AV2ATOT,AVAPL,LMBDPL,AV2APL,NPL,ERR,AVDWIN)
     IF(ERR) RETURN
     !        now get <E(l).dE/dl> integral and error
     CALL TIINT(DS,AV2BTOT,AVBPL,LMBDPL,AV2BPL,NPL,ERR,AVDWIN)
     IF(ERR) RETURN
  END IF

  !     if component values are wanted
  IF(LCOMP) THEN
     CALL TIINT(DAVDW,AVDW2TOT,AVDWPL,LMBDPL,AVDW2PL,NPL,ERR, &
          AVDWIN)
     IF(ERR) RETURN
     CALL TIINT(DAELE,AELE2TOT,AELEPL,LMBDPL,AELE2PL,NPL,ERR, &
          AELEIN)
     IF(ERR) RETURN
     CALL TIINT(DAINT,AINT2TOT,AINTPL,LMBDPL,AINT2PL,NPL,ERR, &
          AINTIN)
     IF(ERR) RETURN
  END IF

  ! now put the terms together to get delta E and delta S
  !     first delta E
  IF(CALDER) THEN
     BETA = ONE/(KBOLTZ*TEMP)
     AV2BTOT = (BETA*BETA)*(AV2BTOT + AV2ATOT)
     AV2ATOT = (AV2TOT + AV2BTOT)
     AV2BTOT = AV2BTOT/(TEMP*TEMP)
     DS = BETA*(DS - DE)
     DE = DA - DS
     DS = -DS/TEMP
  END IF
  AV2TOT = AV2TOT
  DO I=1,NPL+1
     AINCR(I) = AINCR(I)
     IF (LCOMP) THEN
        AVDWIN(I) = AVDWIN(I)
        AELEIN(I) = AELEIN(I)
        AINTIN(I) = AINTIN(I)
     END IF
  ENDDO
  IF(LCOMP) THEN
     AVDW2TOT = AVDW2TOT
     AELE2TOT = AELE2TOT
     AINT2TOT = AINT2TOT
  END IF

     call set_param('DELTAA',da)
     call set_param('DELTAE',de)
     call set_param('DELTAS',ds)

  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,'(1X,72(''-''))')
     WRITE(OUTU,*) 'Thermodynamic Integration Total Values'
     WRITE(OUTU,*) 'delta A = ',DA,' +/-',SQRT(AV2TOT)
     IF(LCOMP) THEN
        IF(AVDW2TOT == ZERO) AVDW2TOT = ONE
        WRITE(OUTU,*) 'delta A(vdW)    = ',DAVDW,' +/-', &
             SQRT(AVDW2TOT)
        IF(AELE2TOT == ZERO) AVDW2TOT = ONE
        WRITE(OUTU,*) 'delta A(Ele)    = ',DAELE,' +/-', &
             SQRT(AELE2TOT)
        IF(AINT2TOT == ZERO) AVDW2TOT = ONE
        WRITE(OUTU,*) 'delta A(Int)    = ',DAINT,' +/-', &
             SQRT(AINT2TOT)
     END IF
     IF (CALDER) THEN
        WRITE(OUTU,*) 'delta E = ',DE,' +/-',SQRT(AV2ATOT)
        WRITE(OUTU,*) 'delta S = ',DS,' +/-',SQRT(AV2BTOT)
     END IF
     IF(.NOT.PLOT) RETURN
     WRITE(OUTU,'(///)')
     WRITE(OUTU,*) 'plot files'
     WRITE(OUTU,'(//,20x,a)')'   <dE(l)/dl>         dA(l)'
     DO I = 1,NPL
        WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,F12.5,2X,A)') &
             LMBDPL(I),AVPL(I),AINCR(I),'"X"'
     enddo
     IF(LCOMP) THEN
        WRITE(OUTU,'(//,20X,A)')' <dE(l)/dl>(vdW)    dA_vdW(l)'
        DO I = 1,NPL
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),AVDWPL(I),AVDWIN(I),'"X"'
        enddo
        WRITE(OUTU,'(//,20x,a)')' <dE(l)/dl>(Ele)    dA_ele(l)'
        DO I = 1,NPL
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),AELEPL(I),AELEIN(I),'"X"'
        enddo
        WRITE(OUTU,'(//,20X,A)')' <dE(l)/dl>(Int)    dA_int(l)'
        DO I = 1,NPL
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),AINTPL(I),AINTIN(I),'"X"'
        enddo
     END IF
     IF (CALDER) THEN
        WRITE(OUTU,'(//,20x,a)')'  d<E(l)>/dl'
        DO I = 1,NPL
           WRITE(OUTU,'(2X,F12.5,2X,F12.5,2X,A)') &
                LMBDPL(I),AVPL(I)-BETA*(AVBPL(I)-AVAPL(I)),'"X"'
        enddo
     END IF
  ENDIF
  RETURN
END SUBROUTINE TITTOT

SUBROUTINE TIINT(SS,SS2ERR,AV,LAMBDA,AVERR,NVAL,ERR,SUM)
  !
  use chm_kinds
  use consta
  use number
  use stream
  implicit none
  ! routine to do spline curve fit to data points
  ! and integrate
  !
  !     av      array holding function values (averages)
  !     averr   squares of errors of averages
  !     lambda  array holding abscissa values (lambda)
  !     nval    integer containing number of values
  !     ss      value of integral
  !     ss2err   uncertainty in integral from propagation of errors
  !     sum     cumulative integral
  !
  !
  INTEGER NVAL,I
  real(chm_real) AV(NVAL),LAMBDA(NVAL),SS,SS2ERR,AVERR(NVAL),SUM(*)
  real(chm_real) COEF
  LOGICAL ERR

  !     Uses simple trapazoidal rule
  ERR = .FALSE.
  IF(NVAL <= 0) THEN
     CALL WRNDIE(0,'<TIint>', &
          ' Zero or negative number of points passed to integrator.')
     ERR = .TRUE.
     RETURN
  END IF
  IF(NVAL == 1) THEN
     SS = AV(1)
     SS2ERR = AVERR(1)
     RETURN
  ENDIF
  !
  !     now get the integral and the error
  COEF = (LAMBDA(2)-LAMBDA(1))*HALF
  SS = COEF*AV(1)
  SS2ERR = COEF*AVERR(1)
  SUM(1) = ZERO
  !     We essentially do the integration twice, once for
  !     the actual integral and the the second for the running integral.
  IF(NVAL > 2) THEN
     DO I = 2,NVAL-1
        COEF = (LAMBDA(I+1)-LAMBDA(I-1))*HALF
        SS = SS + COEF*AV(I)
        SS2ERR = SS2ERR + COEF*AVERR(I)
        COEF = (LAMBDA(I)-LAMBDA(I-1))*HALF
        SUM(I) = SUM(I-1)+COEF*(AV(I)+AV(I-1))
     ENDDO
  ENDIF
  COEF = (LAMBDA(NVAL)-LAMBDA(NVAL-1))*HALF
  SS = SS + COEF*AV(NVAL)
  SS2ERR = SS2ERR + COEF*AVERR(NVAL)
  SUM(NVAL) = SS
  RETURN
END SUBROUTINE TIINT

SUBROUTINE REALSORT(A,B,C,D,E,F,G,H,O,P,Q,R,S,N)
  use chm_kinds
  use number
  implicit none
  !
  !   a: real(chm_real) array to be sorted
  !   b,c,d: real(chm_real) arrays to be arranged according to sorting of a
  !   n: integer scalar containing the number of elements to be sorted.
  INTEGER N
  real(chm_real) A(N),B(N),C(N),D(N),E(N),F(N),G(N), &
       H(N),O(N),P(N),Q(N),R(N),S(N)

  INTEGER I,J,K,L
  real(chm_real) ATEMP,BTEMP,CTEMP,DTEMP,ETEMP,FTEMP,GTEMP, &
       HTEMP,OTEMP,PTEMP,QTEMP,RTEMP,STEMP

  K = N/2 + 1
  L = N
  !     Loop
100 CONTINUE
  IF(K > 1) THEN
     K = K-1
     ATEMP = A(K)
     BTEMP = B(K)
     CTEMP = C(K)
     DTEMP = D(K)
     ETEMP = E(K)
     FTEMP = F(K)
     GTEMP = G(K)
     HTEMP = H(K)
     OTEMP = O(K)
     PTEMP = P(K)
     QTEMP = Q(K)
     RTEMP = R(K)
     STEMP = S(K)
  ELSE
     ATEMP = A(L)
     BTEMP = B(L)
     CTEMP = C(L)
     DTEMP = D(L)
     ETEMP = E(L)
     FTEMP = F(L)
     GTEMP = G(L)
     HTEMP = H(L)
     OTEMP = O(L)
     PTEMP = P(L)
     QTEMP = Q(L)
     RTEMP = R(L)
     STEMP = S(L)
     A(L) = A(1)
     B(L) = B(1)
     C(L) = C(1)
     D(L) = D(1)
     E(L) = E(1)
     F(L) = F(1)
     G(L) = G(1)
     H(L) = H(1)
     O(L) = O(1)
     P(L) = P(1)
     Q(L) = Q(1)
     R(L) = R(1)
     S(L) = S(1)
     L=L-1
     IF(L == 1) THEN
        A(1)=ATEMP
        B(1)=BTEMP
        C(1)=CTEMP
        D(1)=DTEMP
        E(1)=ETEMP
        F(1)=FTEMP
        G(1)=GTEMP
        H(1)=HTEMP
        O(1)=OTEMP
        P(1)=PTEMP
        Q(1)=QTEMP
        R(1)=RTEMP
        S(1)=STEMP
        RETURN
     END IF
  END IF
  I=K
  J=K+K
  !        while (j <= l)
  do while(J <= L) 
     IF(J < L) THEN
        IF(A(J) < A(J+1)) J=J+1
     END IF
     IF(ATEMP < A(J)) THEN
        A(I)=A(J)
        B(I)=B(J)
        C(I)=C(J)
        D(I)=D(J)
        E(I)=E(J)
        F(I)=F(J)
        G(I)=G(J)
        H(I) = H(J)
        O(I) = O(J)
        P(I) = P(J)
        Q(I) = Q(J)
        R(I) = R(J)
        S(I) = S(J)
        I=J
        J=J+J
     ELSE
        J=L+1
     END IF
     !        end while
  enddo
  A(I)=ATEMP
  B(I)=BTEMP
  C(I)=CTEMP
  D(I)=DTEMP
  E(I)=ETEMP
  F(I)=FTEMP
  G(I)=GTEMP
  H(I)=HTEMP
  O(I)=OTEMP
  P(I)=PTEMP
  Q(I)=QTEMP
  R(I)=RTEMP
  S(I)=STEMP
  !     end loop
  GOTO 100
END SUBROUTINE REALSORT

SUBROUTINE PPOSTIC(PTUNIT,NUNIT,NSTPBN,TEMP,DT, &
     TEMPDO,CALDER,MAXICP,MAXWIN,MAXSRF, &
     BEGIN,STOP,NICP,NWIN,NPROC,NSURF,AV,AVTPDT,AVTMDT, &
     AV2,AV2TPD,AV2TMD,AV2LP,AV2LM,AVBIN,AVTPBN,AVTMBN, &
     AVLP,AVLM,AVLPBN,AVLMBN,ECP,ECM,ECP1,ECM1,EAVG, &
     AVI,AVIBIN,AVI2,AVT,AVTBIN,AVT2,AVIC,AVIC2,DA,DE,DS, &
     DAX,DEX,DSX,AVICX,SURF,LINTE)
  !
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use ctitla
  use number
  use stream
  implicit none
  !
  INTEGER MAXWIN,MAXICP,MAXSRF
  real(chm_real) AV(*),AVTPDT(*),AVTMDT(*)
  real(chm_real) AV2(*),AV2TPD(*),AV2TMD(*),AV2LP(*)
  real(chm_real) AV2LM(*),AVBIN(*),AVTPBN(*),AVTMBN(*)
  real(chm_real) AVLP(*),AVLM(*),AVLPBN(*),AVLMBN(*)
  real(chm_real) ECP(*),ECM(*),ECP1(*),ECM1(*),EAVG(*)
  real(chm_real) AVVI,AVI(*),AVIBIN(*),AVI2(*)
  real(chm_real) AVI0,AVI0BN,AVI02
  real(chm_real) AVVT,AVT(*),AVTBIN(*),AVT2(*)
  real(chm_real) AVT0,AVT0BN,AVT02
  real(chm_real) AVIC(MAXICP,*),AVIC2(MAXICP,*),AVICX(MAXICP,*)
  real(chm_real) DA(*),DE(*),DS(*),DAX(*),DEX(*),DSX(*)
  real(chm_real) TEMP,DT
  INTEGER PTUNIT,NUNIT,NSTPBN
  INTEGER NICP,NSURF,NPROC,BEGIN,STOP
  LOGICAL TEMPDO,CALDER,SURF,LINTE
  !
  real(chm_real) KTMDT,KTPDT,KLP,KLM,KT,ELMBDA,ELMBDP,TMDT,TPDT
  real(chm_real) DA2,DE2,DS2,DLNZ,AVTEMP,MMM1
  real(chm_real) AVV0,AVV1,AVV2,AVV3,AVV4
  real(chm_real) HLMBDA,TLMBDA,LMBDAP,DELLP,EILMPP,EILMPM,EILMBD
  real(chm_real) ICVLMD,ICVLPP,ICVLPM
  real(chm_real) DELTA,DELTAX,AKMATI
  INTEGER N,N1,IOS,M,NBINS,NNN
  INTEGER I,J,LEFTOV,IUNCNT,NDEGF,NDEGFX
  INTEGER NWIN,NWIN2,NWINX,NICPX,IW,IW1,IIC,ICTYPE,ICP
  LOGICAL ERR,GO
  CHARACTER(len=80) :: ELINE
  real(chm_real),PARAMETER :: ONE81=181.0D0
  !
  ERR = .FALSE.
  IF(IOLEV < 0) RETURN
  !****CLBIII modification following DJT suggestion, following code not
  !****necessary
  !      IF(STOP-BEGIN <= 0) THEN
  !        CALL WRNDIE(-4,'<PPOSTIC>',
  !     1  'STOP is less than BEGIN.  Nothing done.')
  !        ERR = .TRUE.
  !        RETURN
  !      END IF
  !
  ! Read the header line from the pert file
  !
  loop4000: DO IUNCNT = PTUNIT,PTUNIT+NUNIT-1
     READ(IUNCNT,'(3i6,f12.6)',IOSTAT=IOS,END=950,ERR=950) &
          NICP,NWIN,NDEGF,DELTA
950  CONTINUE
     IF(IOS /= 0) THEN
        IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
             'Empty file or end of file on unit ',IUNCNT
        CALL WRNDIE(0,'<PPOSTIC>',ELINE)
        RETURN
     END IF
     IF(NICP > MAXICP) CALL WRNDIE(-4,'<PPOSTIC>', &
          'MAXICP exceeded; increase MAXP.')
     NWIN2 = 2*NWIN
     IF(NWIN2 > MAXWIN) CALL WRNDIE(-4,'<PPOSTIC>', &
          'MAXWIN exceeded; increase MAXW.')
     IF(TEMPDO.AND.NDEGF <= 0) THEN
        CALL WRNDIE(0,'<PPOSTIC>', &
             'Bad NDEGF in header.  Avg. T will not be calculated.')
        TEMPDO = .FALSE.
     END IF
     !
     IF (IUNCNT == PTUNIT) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(/,1X,A)') &
             'Header: nicp,nwin,ndegf,delta'
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,3I6,F12.6)') &
             NICP,NWIN,NDEGF,DELTA
        !
        ! Initialize
        !
        NICPX = NICP
        NWINX = NWIN
        NDEGFX = NDEGF
        DELTAX = DELTA
        IF(CALDER) THEN
           TPDT = TEMP + DT
           TMDT = TEMP - DT
           KLP = (ONE/TPDT - ONE/TEMP)/KBOLTZ
           KLM = (ONE/TMDT - ONE/TEMP)/KBOLTZ
           KTPDT = KBOLTZ*TPDT
           KTMDT = KBOLTZ*TMDT
           DO I = 1,MAXWIN
              AVTPDT(I) = ZERO
              AVTMDT(I) = ZERO
              AVLP(I) = ZERO
              AVLM(I) = ZERO
              AVTPBN(I) = ZERO
              AVTMBN(I) = ZERO
              AVLPBN(I) = ZERO
              AVLMBN(I) = ZERO
              AV2TPD(I) = ZERO
              AV2TMD(I) = ZERO
              AV2LP(I) = ZERO
              AV2LM(I) = ZERO
           enddo
        END IF
        M = 0
        N = 0
        N1 = 0
        NBINS = 0
        KT = KBOLTZ*TEMP
        AVTEMP = ZERO
        IF(LINTE) THEN
           AVI0 = ZERO
           AVI0BN = ZERO
           AVI02 = ZERO
           AVT0 = ZERO
           AVT0BN = ZERO
           AVT02 = ZERO
        END IF
        DO I = 1,MAXWIN
           AV(I) = ZERO
           AVBIN(I) = ZERO
           AV2(I) = ZERO
           IF(LINTE) THEN
              AVI(I) = ZERO
              AVIBIN(I) = ZERO
              AVI2(I) = ZERO
              AVT(I) = ZERO
              AVTBIN(I) = ZERO
              AVT2(I) = ZERO
           END IF
           DO J = 1,MAXICP
              AVIC(J,I) = ZERO
              AVIC2(J,I) = ZERO
           enddo
        enddo
1100    FORMAT(I7,F10.4,3D16.8)
1101    FORMAT(7X,F10.4,2D16.8)
     END IF
     ERR = (NICP /= NICPX.OR.NWIN /= NWINX.OR. &
          NDEGF /= NDEGFX.OR.DELTA /= DELTAX)
     IF(ERR) THEN
        IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
             'Header mismatch in file on unit ',IUNCNT
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
             'Header: nicp,nwin,ndegf,delta'
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,3I6,F12.6)') &
             NICP,NWIN,NDEGF,DELTA
        CALL WRNDIE(0,'<PPOSTIC>',ELINE)
        RETURN
     END IF
     GO = .TRUE.
     !
     ! Calculate scaling constants for exponentials from first dataset
     !
     IF(CALDER) THEN
        READ(IUNCNT,1100,END=1200,ERR=1200,IOSTAT=IOS) &
             NNN,AKMATI,HLMBDA,TLMBDA,EILMBD
        ELMBDA = HLMBDA - TLMBDA
        DO I = 1,NWIN
           READ(IUNCNT,1101,END=1200,ERR=1200,IOSTAT=IOS) &
                LMBDAP,EILMPP,EILMPM
           IW = 2*I - 1
           IW1 = IW + 1
           DELLP = EILMPP - EILMBD
           ELMBDP = ELMBDA + DELLP
           ECP(IW1) = -ELMBDP/KTPDT + ELMBDA/KT
           ECM(IW1) = -ELMBDP/KTMDT + ELMBDA/KT
           ECP1(IW1) = -ELMBDA*KLP
           ECM1(IW1) = -ELMBDA*KLM
           EAVG(IW1) = ECP(IW1) - ECM(IW1) - ECP1(IW1) + ECM1(IW1)
           DELLP = EILMPM - EILMBD
           ELMBDP = ELMBDA + DELLP
           ECP(IW) = -ELMBDP/KTPDT + ELMBDA/KT
           ECM(IW) = -ELMBDP/KTMDT + ELMBDA/KT
           ECP1(IW) = -ELMBDA*KLP
           ECM1(IW) = -ELMBDA*KLM
           EAVG(IW) = ECP(IW) - ECM(IW) - ECP1(IW) + ECM1(IW)
        enddo
        REWIND (UNIT=IUNCNT,IOSTAT=IOS)
        IF(IOS /= 0) THEN
           IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
                'Error rewinding file on unit ',IUNCNT
           CALL WRNDIE(0,'<PPOSTIC>',ELINE)
           RETURN
        END IF
        READ(IUNCNT,'(3i6,f12.6)',IOSTAT=IOS,END=1200,ERR=1200) &
             NICP,NWIN,NDEGF,DELTA
1200    CONTINUE
        IF(IOS /= 0) THEN
           IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
                'Empty file or end of file on unit ',IUNCNT
           CALL WRNDIE(0,'<PPOSTIC>',ELINE)
           RETURN
        END IF
     END IF
     !
     ! Accumulate the averages
     !
     do while(GO)
        READ(IUNCNT,1100,END=1220,ERR=1220,IOSTAT=IOS) &
             NNN,AKMATI,HLMBDA,TLMBDA,EILMBD
        N1 = N1 + 1
        IF(N1 > STOP.AND.STOP > 0) cycle loop4000
        IF(N1 >= BEGIN) THEN
           ELMBDA = HLMBDA - TLMBDA
           IF(TEMPDO) AVTEMP = AVTEMP + TLMBDA
           IF(LINTE) THEN
              AVI0 = AVI0 + EILMBD
              AVI0BN = AVI0BN + EILMBD
              AVT0 = AVT0 + ELMBDA
              AVT0BN = AVT0BN + ELMBDA
           END IF
        END IF
        DO I = 1,NWIN
           READ(IUNCNT,1101,END=1220,ERR=1220,IOSTAT=IOS) &
                LMBDAP,EILMPP,EILMPM
           IF(N1 >= BEGIN) THEN
              IW = 2*I - 1
              IW1 = IW + 1
              DELLP = EILMPP - EILMBD
              ELMBDP = ELMBDA + DELLP
              AVV0 = EXP( -DELLP/KT)
              AV(IW1) = AV(IW1) + AVV0
              AVBIN(IW1) = AVBIN(IW1) + AVV0
              IF(LINTE) THEN
                 AVVI = EILMPP*AVV0
                 AVI(IW1) = AVI(IW1) + AVVI
                 AVIBIN(IW1) = AVIBIN(IW1) + AVVI
                 AVVT = ELMBDP*AVV0
                 AVT(IW1) = AVT(IW1) + AVVT
                 AVTBIN(IW1) = AVTBIN(IW1) + AVVT
              END IF
              IF(CALDER) THEN
                 AVV1 = EXP( -ELMBDP/KTPDT + ELMBDA/KT - ECP(IW1))
                 AVV2 = EXP( -ELMBDP/KTMDT + ELMBDA/KT - ECM(IW1))
                 AVV3 = EXP( -ELMBDA*KLP - ECP1(IW1))
                 AVV4 = EXP( -ELMBDA*KLM - ECM1(IW1))
                 AVTPDT(IW1) = AVTPDT(IW1) + AVV1
                 AVTMDT(IW1) = AVTMDT(IW1) + AVV2
                 AVLP(IW1) = AVLP(IW1) + AVV3
                 AVLM(IW1) = AVLM(IW1) + AVV4
                 AVTPBN(IW1) = AVTPBN(IW1) + AVV1
                 AVTMBN(IW1) = AVTMBN(IW1) + AVV2
                 AVLPBN(IW1) = AVLPBN(IW1) + AVV3
                 AVLMBN(IW1) = AVLMBN(IW1) + AVV4
              END IF
              DELLP = EILMPM - EILMBD
              ELMBDP = ELMBDA + DELLP
              AVV0 = EXP( -DELLP/KT)
              AV(IW) = AV(IW) + AVV0
              AVBIN(IW) = AVBIN(IW) + AVV0
              IF(LINTE) THEN
                 AVVI = EILMPM*AVV0
                 AVI(IW) = AVI(IW) + AVVI
                 AVIBIN(IW) = AVIBIN(IW) + AVVI
                 AVVT = ELMBDP*AVV0
                 AVT(IW) = AVT(IW) + AVVT
                 AVTBIN(IW) = AVTBIN(IW) + AVVT
              END IF
              IF(CALDER) THEN
                 AVV1 = EXP( -ELMBDP/KTPDT + ELMBDA/KT - ECP(IW))
                 AVV2 = EXP( -ELMBDP/KTMDT + ELMBDA/KT - ECM(IW))
                 AVV3 = EXP( -ELMBDA*KLP - ECP1(IW))
                 AVV4 = EXP( -ELMBDA*KLM - ECM1(IW))
                 AVTPDT(IW) = AVTPDT(IW) + AVV1
                 AVTMDT(IW) = AVTMDT(IW) + AVV2
                 AVLP(IW) = AVLP(IW) + AVV3
                 AVLM(IW) = AVLM(IW) + AVV4
                 AVTPBN(IW) = AVTPBN(IW) + AVV1
                 AVTMBN(IW) = AVTMBN(IW) + AVV2
                 AVLPBN(IW) = AVLPBN(IW) + AVV3
                 AVLMBN(IW) = AVLMBN(IW) + AVV4
              END IF
           END IF
           DO ICP = 1,NICP
              READ(IUNCNT,1212,END=1220,ERR=1220,IOSTAT=IOS) &
                   IIC,ICTYPE,ICVLMD,ICVLPP,ICVLPM
              IF(N1 >= BEGIN) THEN
1201             IF(ICTYPE == 3.AND.ICVLPP < -ONE81) THEN
                    ICVLPP = ICVLPP + THR6TY
                    GOTO 1201
                 END IF
1202             IF(ICTYPE == 3.AND.ICVLPP > ONE81) THEN
                    ICVLPP = ICVLPP - THR6TY
                    GOTO 1202
                 END IF
1203             IF(ICTYPE == 3.AND.ICVLPM < -ONE81) THEN
                    ICVLPM = ICVLPM + THR6TY
                    GOTO 1203
                 END IF
1204             IF(ICTYPE == 3.AND.ICVLPM > ONE81) THEN
                    ICVLPM = ICVLPM - THR6TY
                    GOTO 1204
                 END IF
                 AVIC(ICP,IW1) = AVIC(ICP,IW1) + ICVLPP
                 AVIC2(ICP,IW1) = AVIC2(ICP,IW1) + ICVLPP*ICVLPP
                 AVIC(ICP,IW) = AVIC(ICP,IW) + ICVLPM
                 AVIC2(ICP,IW) = AVIC2(ICP,IW) + ICVLPM*ICVLPM
              END IF
           enddo
1212       FORMAT(9X,2I4,3D16.8)
        enddo
        !
        IF(N1 >= BEGIN) THEN
           M = M + 1
           N = N + 1
           IF (M == NSTPBN) THEN
              M = 0
              NBINS = NBINS + 1
              IF(LINTE) THEN
                 AVI02 = AVI02 + (AVI0BN/NSTPBN)**2
                 AVI0BN = ZERO
                 AVT02 = AVT02 + (AVT0BN/NSTPBN)**2
                 AVT0BN = ZERO
              END IF
              DO I = 1,NWIN2
                 AV2(I) = AV2(I) + (AVBIN(I)/NSTPBN)**2
                 AVBIN(I) = ZERO
                 IF(LINTE) THEN
                    AVI2(I) = AVI2(I) + (AVIBIN(I)/NSTPBN)**2
                    AVIBIN(I) = ZERO
                    AVT2(I) = AVT2(I) + (AVTBIN(I)/NSTPBN)**2
                    AVTBIN(I) = ZERO
                 END IF
                 IF (CALDER) THEN
                    AV2TPD(I) = AV2TPD(I) + (AVTPBN(I)/NSTPBN)**2
                    AV2TMD(I) = AV2TMD(I) + (AVTMBN(I)/NSTPBN)**2
                    AV2LP(I) = AV2LP(I) + (AVLPBN(I)/NSTPBN)**2
                    AV2LM(I) = AV2LM(I) + (AVLMBN(I)/NSTPBN)**2
                    AVTPBN(I) = ZERO
                    AVTMBN(I) = ZERO
                    AVLPBN(I) = ZERO
                    AVLMBN(I) = ZERO
                 END IF
              enddo
           END IF
        END IF
1220    CONTINUE
        IF(IOS /= 0) THEN
           IF(N == 0) THEN
              IF(WRNLEV >= 2) WRITE(ELINE,'(A,I3)') &
                   'Empty file or end of file on unit ',IUNCNT
              CALL WRNDIE(0,'<PPOSTIC>',ELINE)
              ERR=.TRUE.
              RETURN
           ELSE
              cycle loop4000
           END IF
        END IF
     enddo
  enddo loop4000
  REWIND (UNIT=IUNCNT,IOSTAT=IOS)
  IF(N1 < BEGIN) CALL WRNDIE(-4,'<PPOSTIC>', &
       'BEGIN is greater than the number datasets in the file.')
  IF(N < BEGIN-STOP.AND.STOP > 0 .AND. WRNLEV >= 2) WRITE(OUTU, &
       '(1X,A)') 'Warning: read less than BEGIN - STOP datasets.'
  !
  ! Process averages and errors
  !
  IF(PRNLEV >= 2) WRITE(OUTU,'(/,1X,A,I6)') &
       'Total number of steps: ',N
  IF(PRNLEV >= 2) WRITE(OUTU,'(1X,2(A,I6),A)') &
       'Used ',NBINS,' bins of ',NSTPBN,' steps.'
  IF(TEMPDO) THEN
     AVTEMP = TWO*(AVTEMP/N)/(KBOLTZ*NDEGF)
     IF(PRNLEV >= 2) WRITE(OUTU,'(/,1X,A,F9.4,A)') &
          '<T> = ',AVTEMP,' K'
  END IF
  LEFTOV = N-NBINS*NSTPBN
  IF (LEFTOV /= 0) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'Number of steps/bin did not divide into number of steps.'
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,I6,A)') LEFTOV, &
          ' steps not included in error analysis,'
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'but were included in averages.'
  END IF
  IF(PRNLEV >= 2) WRITE(OUTU,4100)
4100 FORMAT(/,' Internal coordinate averages:',//,1X,'pert. no.', &
       2X,'window',11X,'ic value')
  DO ICP = 1,NICP
     DO IW = 1,NWIN2
        AVIC(ICP,IW) = AVIC(ICP,IW)/N
        AVIC2(ICP,IW) = AVIC2(ICP,IW)/N
        AVIC2(ICP,IW) = SQRT(ABS(AVIC(ICP,IW)*AVIC(ICP,IW) &
             - AVIC2(ICP,IW)))
        IF(PRNLEV >= 2) WRITE(OUTU,4200) &
             ICP,IW,AVIC(ICP,IW),AVIC2(ICP,IW)
     enddo
  enddo
4200 FORMAT(3X,I3,7X,I3,5X,F10.5,1X,'+/-',F8.5)
  IF(PRNLEV >= 2) WRITE(OUTU,4300)
4300 FORMAT(/,' Thermodynamics:',/)
  IF(PRNLEV >= 2) WRITE(OUTU,'(1X,2(A,F9.4),A)') &
       'T = ',TEMP,' K; delta T = ',DT,' K'
  IF(PRNLEV >= 2) WRITE(OUTU,4320)
4320 FORMAT(/,1X,'window',17X,'delta A (kcal/mole)')
  IF(NBINS == 1) THEN
     MMM1 = ONE
  ELSE
     MMM1 = ONE/(NBINS*(NBINS-1))
  END IF
  DO I = 1,NWIN2
     AV(I) = AV(I)/N
     DA(I) = -KT*LOG(AV(I))
     AV2(I) = MMM1*(AV2(I) - NBINS*(AV(I)**2))
     DA2 = AV2(I)/(AV(I)*AV(I))
     DA2 = KT*SQRT(DA2)
     IF(PRNLEV >= 2) WRITE(OUTU,4400) I,DA(I),DA2
4400 FORMAT(2X,I3,13X,E13.6,1X,'+/-',1X,E12.6)
  enddo
  IF(CALDER) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,4430)
4430 FORMAT(/,1X,'window',17X,'delta S (kcal/mole*K)')
     DO I = 1,NWIN2
        AVTPDT(I) = AVTPDT(I)/N
        AVTMDT(I) = AVTMDT(I)/N
        AVLP(I) = AVLP(I)/N
        AVLM(I) = AVLM(I)/N
        AV2TPD(I) = MMM1*(AV2TPD(I) - NBINS*(AVTPDT(I)**2))
        AV2TMD(I) = MMM1*(AV2TMD(I) - NBINS*(AVTMDT(I)**2))
        AV2LP(I) = MMM1*(AV2LP(I) - NBINS*(AVLP(I)**2))
        AV2LM(I) = MMM1*(AV2LM(I) - NBINS*(AVLM(I)**2))
        DLNZ = ( LOG(AVTPDT(I)) - LOG(AVTMDT(I)) &
             -  LOG(AVLP(I))   + LOG(AVLM(I)) ) + EAVG(I)
        DLNZ = (ONE/(TWO*DT))*DLNZ
        DS(I) = KBOLTZ*LOG(AV(I)) + KT*DLNZ
        DA2 = AV2(I)/(AV(I)*AV(I))
        DS2 = ((KT/(TWO*DT))**2)*(((AV2TPD(I)/AVTPDT(I))/AVTPDT(I)) &
             + ((AV2TMD(I)/AVTMDT(I))/AVTMDT(I)) &
             + ((AV2LP(I)/AVLP(I))/AVLP(I)) &
             + ((AV2LM(I)/AVLM(I))/AVLM(I)))
        DS2 = KBOLTZ*KBOLTZ*DA2 + DS2
        DS2 = SQRT(DS2)
        IF(PRNLEV >= 2) WRITE(OUTU,4400) I,DS(I),DS2
     enddo
     IF(PRNLEV >= 2) WRITE(OUTU,4530)
4530 FORMAT(/,1X,'window',17X,'delta E (kcal/mole)')
     DO I = 1,NWIN2
        DLNZ = ( LOG(AVTPDT(I)) - LOG(AVTMDT(I)) &
             -  LOG(AVLP(I))   + LOG(AVLM(I)) ) + EAVG(I)
        DLNZ = (ONE/(TWO*DT))*DLNZ
        DE(I) = KT*TEMP*DLNZ
        DS2 = ((KT/(TWO*DT))**2)*(((AV2TPD(I)/AVTPDT(I))/AVTPDT(I)) &
             + ((AV2TMD(I)/AVTMDT(I))/AVTMDT(I)) &
             + ((AV2LP(I)/AVLP(I))/AVLP(I)) &
             + ((AV2LM(I)/AVLM(I))/AVLM(I)))
        DE2 = TEMP*TEMP*DS2
        DE2 = SQRT(DE2)
        IF(PRNLEV >= 2) WRITE(OUTU,4400) I,DE(I),DE2
     enddo
  END IF
  IF(LINTE) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,4600)
4600 FORMAT(/,' Average interaction energies:',/)
     IF(PRNLEV >= 2) WRITE(OUTU,4630)
4630 FORMAT(1X,'window',15X,'avg. int. E (kcal/mole)')
     AVI0 = AVI0/N
     AVI02 = SQRT(MMM1*(AVI02 - NBINS*(AVI0**2)))
     I = 0
     IF(PRNLEV >= 2) WRITE(OUTU,4400) I,AVI0,AVI02
     DO I = 1,NWIN2
        AVI(I) = AVI(I)/N
        AVI2(I) = MMM1*(AVI2(I) - NBINS*(AVI(I)**2))
        AVI(I) = AVI(I)/AV(I)
        AVI2(I) = SQRT((AVI2(I) + AVI(I)*AVI(I)*AV2(I)))/AV(I)
        IF(PRNLEV >= 2) WRITE(OUTU,4400) I,AVI(I),AVI2(I)
     enddo
     IF(PRNLEV >= 2) WRITE(OUTU,4601)
4601 FORMAT(/,' Average total energies:',/)
     IF(PRNLEV >= 2) WRITE(OUTU,4631)
4631 FORMAT(1X,'window',15X,'avg. tot. E (kcal/mole)')
     AVT0 = AVT0/N
     AVT02 = SQRT(MMM1*(AVT02 - NBINS*(AVT0**2)))
     I = 0
     IF(PRNLEV >= 2) WRITE(OUTU,4400) I,AVT0,AVT02
     DO I = 1,NWIN2
        AVT(I) = AVT(I)/N
        AVT2(I) = MMM1*(AVT2(I) - NBINS*(AVT(I)**2))
        AVT(I) = AVT(I)/AV(I)
        AVT2(I) = SQRT((AVT2(I) + AVT(I)*AVT(I)*AV2(I)))/AV(I)
        IF(PRNLEV >= 2) WRITE(OUTU,4400) I,AVT(I),AVT2(I)
     enddo
  END IF
  !
  ! load the pieces into temporary arrays for the surface construction
  !
  IF(SURF) THEN
     IF(NSURF+NWIN2+1 > MAXSRF) &
          CALL WRNDIE(-4,'<PPOSTIC>','maxsrf exceeded; increase MAXS.')
     IF(AVIC(1,1) < AVIC(1,2)) THEN
        I = NSURF + NWIN + 1
        J = NSURF + NWIN
        DO IW = 2,NWIN2,2
           IW1 = IW - 1
           I = I - 1
           J = J + 1
           DAX(I) = DA(IW1)
           DAX(J) = DA(IW)
           IF(CALDER) THEN
              DSX(I) = DS(IW1)
              DSX(J) = DS(IW)
              DEX(I) = DE(IW1)
              DEX(J) = DE(IW)
           END IF
           DO ICP = 1,NICP
              AVICX(ICP,I) = AVIC(ICP,IW1)
              AVICX(ICP,J) = AVIC(ICP,IW)
           enddo
        enddo
     ELSE
        I = NSURF + NWIN2 + 1
        J = NSURF
        DO IW = NWIN2,2,-2
           IW1 = IW - 1
           I = I - 1
           J = J + 1
           DAX(I) = DA(IW1)
           DAX(J) = DA(IW)
           IF(CALDER) THEN
              DSX(I) = DS(IW1)
              DSX(J) = DS(IW)
              DEX(I) = DE(IW1)
              DEX(J) = DE(IW)
           END IF
           DO ICP = 1,NICP
              AVICX(ICP,I) = AVIC(ICP,IW1)
              AVICX(ICP,J) = AVIC(ICP,IW)
           enddo
        enddo
     END IF
     NSURF = NSURF + NWIN2
     NPROC = NPROC + 1
  END IF
  !
  RETURN
END SUBROUTINE PPOSTIC

SUBROUTINE ICPTOT(MAXICP,NICP,NWIN,NSURF,NPROC,IC,DA,DE,DS, &
     ICPLT,DAPLT,DEPLT,DSPLT,ICTEMP,CALDER)
  use chm_kinds
  use number
  use stream
  use timerm
  implicit none
  !
  ! construct and print the thermodynamic surfaces
  !
  !
  INTEGER MAXICP
  real(chm_real) IC(MAXICP,*),DA(*),DE(*),DS(*)
  real(chm_real) ICPLT(MAXICP,*),DAPLT(*),DEPLT(*),DSPLT(*)
  INTEGER ICTEMP(*)
  INTEGER NICP,NWIN,NSURF,NPROC
  INTEGER NWIN2,I,ICNT,N,IMID,IW,ICP,ICNT1
  LOGICAL CALDER
  !
  IF(NSURF == 0) RETURN
  NWIN2 = 2*NWIN
  IF(NPROC > 1) CALL SORTIC(IC,DA,DE,DS,ICTEMP,ICPLT,DAPLT,DEPLT, &
       DSPLT,NWIN2,NICP,NSURF,MAXICP,CALDER)
  DO N = 1,NSURF
     DO I = 1,NICP
        ICPLT(I,N) = IC(I,N)
     enddo
     DAPLT(N) = ZERO
     IF(CALDER) THEN
        DEPLT(N) = ZERO
        DSPLT(N) = ZERO
     END IF
  enddo
  ICNT = 1
  DO N = 1,NPROC
     IMID = ICNT + NWIN
     DO I = 1,NICP
        ICPLT(I,IMID) = HALF*(IC(I,ICNT) + IC(I,ICNT+NWIN2-1))
     enddo
     DAPLT(IMID) = DAPLT(ICNT) - DA(ICNT)
     IF(CALDER) THEN
        DEPLT(IMID) = DEPLT(ICNT) - DE(ICNT)
        DSPLT(IMID) = DSPLT(ICNT) - DS(ICNT)
     END IF
     DO IW = 2,NWIN
        ICNT = ICNT + 1
        DO I = 1,NICP
           ICPLT(I,ICNT) = IC(I,ICNT)
        enddo
        DAPLT(ICNT) = DAPLT(IMID) + DA(ICNT)
        IF(CALDER) THEN
           DEPLT(ICNT) = DEPLT(IMID) + DE(ICNT)
           DSPLT(ICNT) = DSPLT(IMID) + DS(ICNT)
        END IF
     enddo
     ICNT = IMID
     DO IW = NWIN+1,NWIN2
        ICNT1 = ICNT
        ICNT = ICNT + 1
        DO I = 1,NICP
           ICPLT(I,ICNT) = IC(I,ICNT1)
        enddo
        DAPLT(ICNT) = DAPLT(IMID) + DA(ICNT1)
        IF(CALDER) THEN
           DEPLT(ICNT) = DEPLT(IMID) + DE(ICNT1)
           DSPLT(ICNT) = DSPLT(IMID) + DS(ICNT1)
        END IF
     enddo
  enddo
  !
  IF(PRNLEV >= 2) WRITE(OUTU,1000)
1000 FORMAT(/,1X,72('-'))
  IF(PRNLEV >= 2) WRITE(OUTU,1001)
1001 FORMAT(/,1X,'Thermodynamic surfaces:')
  DO ICP = 1,NICP
     IF(PRNLEV >= 2) WRITE(OUTU,1100) ICP
1100 FORMAT(/,1X,'Perturbation #',I3,/)
     IF(CALDER) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,1200)
1200    FORMAT(2X,'ic value',3X,'delta A (kcal/mole)', &
             3X,'delta S (kcal/mole*K)',2X,'delta E (kcal/mole)',/)
        DO I = 1,ICNT
           IF(PRNLEV >= 2) WRITE(OUTU,1300) ICPLT(ICP,I), &
                DAPLT(I),DSPLT(I),DEPLT(I)
1300       FORMAT(1X,F10.5,5X,E13.6,9X,E13.6,10X,E13.6)
        enddo
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,1201)
1201    FORMAT(2X,'ic value',3X,'delta A (kcal/mole)',/)
        DO I = 1,ICNT
           IF(PRNLEV >= 2) WRITE(OUTU,1301) ICPLT(ICP,I),DAPLT(I)
1301       FORMAT(1X,F10.5,5X,E13.6)
        enddo
     END IF
  enddo
  !
  RETURN
END SUBROUTINE ICPTOT

SUBROUTINE SORTIC(IC,DA,DE,DS,INDX,ICTEMP,DATEMP,DETEMP,DSTEMP, &
     NWIN2,NICP,NSURF,MAXICP,CALDER)
  !
  ! sort the thermodynamic data and ic's according to the first ic
  !
  use chm_kinds
  use number
  implicit none
  !
  INTEGER MAXICP
  real(chm_real) IC(MAXICP,*),ICTEMP(MAXICP,*),DA(*),DE(*),DS(*)
  real(chm_real) DATEMP(*),DETEMP(*),DSTEMP(*)
  INTEGER INDX(*)
  INTEGER NWIN2,NICP,NSURF
  LOGICAL CALDER
  !
  real(chm_real) Q
  INTEGER I,J,K,K1,L,N,IR,INDXT,I1
  !
  ! set up an index array by sorting the indices according to the first ic
  ! using the "heapsort" algorithm
  !
  N = NSURF/NWIN2
  DO I = 1,N
     INDX(I) = I
  enddo
  L = N/2 + 1
  IR = N
110 CONTINUE
  IF(L > 1) THEN
     L = L - 1
     INDXT = INDX(L)
     I1 = NWIN2*(INDXT - 1) + 1
     Q = IC(1,I1)
  ELSE
     INDXT = INDX(IR)
     I1 = NWIN2*(INDXT - 1) + 1
     Q = IC(1,I1)
     INDX(IR) = INDX(1)
     IR = IR - 1
     IF(IR == 1) THEN
        INDX(1) = INDXT
        GOTO 999
     END IF
  END IF
  I = L
  J = L + L
200 IF(J <= IR) THEN
     K = NWIN2*(INDX(J) - 1) + 1
     K1 = NWIN2*(INDX(J+1) - 1) + 1
     IF(J < IR) THEN
        IF(IC(1,K) < IC(1,K1)) THEN
           J = J + 1
           K = NWIN2*(INDX(J) - 1) + 1
        END IF
     END IF
     IF(Q < IC(1,K)) THEN
        INDX(I) = INDX(J)
        I = J
        J = J + J
     ELSE
        J = IR + 1
     END IF
     GOTO 200
  END IF
  INDX(I) = INDXT
  GOTO 110
999 CONTINUE
  !
  ! sort the arrays according to the ordering in the index array
  !
  DO I = 1,NSURF
     DATEMP(I) = DA(I)
     DO I1 = 1,NICP
        ICTEMP(I1,I) = IC(I1,I)
     enddo
     IF(CALDER) THEN
        DETEMP(I) = DE(I)
        DSTEMP(I) = DS(I)
     END IF
  enddo
  J = 0
  DO I = 1,N
     K = NWIN2*(INDX(I) - 1)
     DO I1 = 1,NWIN2
        J = J + 1
        K = K + 1
        DA(J) = DATEMP(K)
        DO L = 1,NICP
           IC(L,J) = ICTEMP(L,K)
        enddo
        IF(CALDER) THEN
           DE(J) = DETEMP(K)
           DS(J) = DSTEMP(K)
        END IF
     enddo
  enddo
  return
end SUBROUTINE SORTIC


#endif 
SUBROUTINE NULL_TP
  RETURN
END SUBROUTINE NULL_TP

