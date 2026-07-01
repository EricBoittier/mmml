#if KEY_RISM==0 /*rism_main*/
SUBROUTINE NULL_CYCL
  RETURN
END SUBROUTINE NULL_CYCL
#else /* (rism_main)*/
SUBROUTINE CYCLES(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This subroutine parses the control command for the iteration
  !     and calls the elementary cycle for the iteration procedure that
  !     solves the SSOZ equation with a chosen closure.
  !
  use dimens_fcm
  use memory
  use string
  use number
  use stream
  use rism
  use struc
  use distri
  use rism_control
  use fft
  use chm_kinds
  implicit none
  real(chm_real),allocatable,dimension(:,:) :: TSR
  real(chm_real),allocatable,dimension(:,:) :: TSK
  real(chm_real),allocatable,dimension(:,:) :: CS2R
  real(chm_real),allocatable,dimension(:,:) :: W
  real(chm_real),allocatable,dimension(:,:) :: WI
  real(chm_real),allocatable,dimension(:,:) :: USR2
  real(chm_real),allocatable,dimension(:,:) :: PMFR
  real(chm_real),allocatable,dimension(:,:) :: SHIFT
  real(chm_real),allocatable,dimension(:,:) :: WU
  real(chm_real),allocatable,dimension(:,:) :: XVV2K1
  real(chm_real),allocatable,dimension(:,:) :: XVV2K2
  real(chm_real),allocatable,dimension(:,:) :: WA
  real(chm_real),allocatable,dimension(:,:) :: WB
  real(chm_real),allocatable,dimension(:,:) :: WCXCW
  real(chm_real),allocatable,dimension(:) :: PHIABK
  real(chm_real),allocatable,dimension(:) :: PHIABR
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN
  !
  !
  !     Local variables
  !     ---------------

  !     Cycle iteration control variables
  real(chm_real) CDIE2,TOL,RMIX
  LOGICAL QINIT,LQGR,LQWR,LQUSR,LQTSR,LQBDR,LQCAV

  !     Control switch variables
  real(chm_real)  SWI(4),SWF(4),DSW(4)
  INTEGER NSW(4)

  !     Solute label
  INTEGER UA,UB,IUA,IUB,UUAB,IOFF

  !     extrapolation closure for VV only
  real(chm_real) AXTR(DPRVV),RXTR1(DPRVV),RXTR2(DPRVV)

  INTEGER NCYCLE,NPRINT,NDMAX
  INTEGER IP
  INTEGER IUNCSR,IUNGR
  !
  !     Local heap management
  !!      INTEGER IOFF1,IOFF2,IOFF3

  !     Parsing the command
  !     ===================

  !     Switching parameters
  CALL GTSWTCH(COMLYN,COMLEN,SW,DSW,SWI,SWF,NSW)
  !     Control parameters
  QINIT =CHECQUE(COMLYN,'INIT')
  LQUSR =CHECQUE(COMLYN,'US(R)')
  LQGR  =CHECQUE(COMLYN,'G(R)')
  LQWR  =CHECQUE(COMLYN,'W(R)')
  LQTSR =CHECQUE(COMLYN,'TS(R)')
  LQCAV =CHECQUE(COMLYN,'CAV(R)')
  LQBDR =CHECQUE(COMLYN,'BD(R)')
  NCYCLE=GTRMI(COMLYN,COMLEN,'NCYCLE',1)
  IUNCSR=GTRMI(COMLYN,COMLEN,'IUNCSR',0)
  IUNGR =GTRMI(COMLYN,COMLEN,'IUNGR',0)
  NPRINT=GTRMI(COMLYN,COMLEN,'NPRINT',NCYCLE)
  RMIX  =GTRMF(COMLYN,COMLEN,'RMIX',THIRD)
  NDMAX =GTRMI(COMLYN,COMLEN,'NDMAX',25)
  TOL   =GTRMF(COMLYN,COMLEN,'TOL',PT01)
  CDIE2 =GTRMF(COMLYN,COMLEN,'CDIE2',CDIE)

  !     Solution to solve:
  !     ------------------

  IF(CHECQUE(COMLYN,'VV'))THEN
     SOL  ='VV'
     call chmalloc('cycles.src','CYCLES','TSR',DVECT,NPVV,crl=TSR)
     call chmalloc('cycles.src','CYCLES','TSK',DVECT,NPVV,crl=TSK)
     call chmalloc('cycles.src','CYCLES','CS2R',DVECT,NPVV,crl=CS2R)
     call chmalloc('cycles.src','CYCLES','W',DVECT,NPV,crl=W)
     call chmalloc('cycles.src','CYCLES','WI',DVECT,NPV,crl=WI)
     call chmalloc('cycles.src','CYCLES','USR2',DVECT,NPVV,crl=USR2)
     call chmalloc('cycles.src','CYCLES','PMFR',DVECT,NPVV,crl=PMFR)
     call chmalloc('cycles.src','CYCLES','SHIFT',DVECT,NPVV,crl=SHIFT)

  ELSEIF(CHECQUE(COMLYN,'UV'))THEN
     SOL='UV'
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IOFF=PUV(IUA)
     UA=1+DSITV+DSITU*(IUA-1)
     call chmalloc('cycles.src','CYCLES','TSR',DVECT,NPUV(IUA),crl=TSR)
     call chmalloc('cycles.src','CYCLES','TSK',DVECT,NPUV(IUA),crl=TSK)
     call chmalloc('cycles.src','CYCLES','CS2R',DVECT,NPUV(IUA),crl=CS2R)
     call chmalloc('cycles.src','CYCLES','WU',DVECT,NPU(IUA),crl=WU)
     call chmalloc('cycles.src','CYCLES','XVV2K1',DVECT,NPV,crl=XVV2K1)
     call chmalloc('cycles.src','CYCLES','XVV2K2',DVECT,NPV,crl=XVV2K2)
     call chmalloc('cycles.src','CYCLES','USR2',DVECT,NPUV(IUA),crl=USR2)

  ELSEIF(CHECQUE(COMLYN,'UU'))THEN
     SOL='UU'
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IUB=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IF(IUA.LT.IUB)THEN
        IOFF=IUA
        IUA=IUB
        IUB=IOFF
     ENDIF
     !
     ! Set the pointers for arrays like x(i)...
     UA=1+DSITV+DSITU*(IUA-1)
     UB=1+DSITV+DSITU*(IUB-1)
     !
     !     pointer for pair npab
     UUAB=(IUA*(IUA-1))/2+IUB
     !
     !     pointer for ab correlation functions
     IOFF=PUU(UUAB)
     call chmalloc('cycles.src','CYCLES','TSR',DVECT,NPUU(UUAB),crl=TSR)
     call chmalloc('cycles.src','CYCLES','TSK',DVECT,NPUU(UUAB),crl=TSK)
     call chmalloc('cycles.src','CYCLES','CS2R',DVECT,NPUU(UUAB),crl=CS2R)
     call chmalloc('cycles.src','CYCLES','USR2',DVECT,NPUU(UUAB),crl=USR2)
     call chmalloc('cycles.src','CYCLES','WA',DVECT,NPU(IUA),crl=WA)
     call chmalloc('cycles.src','CYCLES','WB',DVECT,NPU(IUB),crl=WB)
     call chmalloc('cycles.src','CYCLES','WCXCW',DVECT,NPUU(UUAB),crl=WCXCW)
     call chmalloc('cycles.src','CYCLES','PHIABK',DVECT,crl=PHIABK)
     call chmalloc('cycles.src','CYCLES','PHIABR',DVECT,crl=PHIABR)
  ENDIF

  !     Closures:
  IF(CHECQUE(COMLYN,'HNC'))THEN
     CLOS='HNC'
  ELSEIF(CHECQUE(COMLYN,'PY'))THEN
     CLOS='PY '
  ELSEIF(CHECQUE(COMLYN,'PY2'))THEN
     CLOS='PY2'
  ELSEIF(CHECQUE(COMLYN,'XTR'))THEN
     CLOS='XTR'
     CALL GTSHFT(COMLYN,COMLEN,AXTR,RXTR1,RXTR2,NPVV)
  ENDIF


  !     Write out all the options in use (each entry is 15 characters)
  WRITE(OUTU,100) CLOS,NCYCLE,NPRINT, &
       RMIX,TOL,NDMAX, &
       IUNCSR,IUNGR

100 FORMAT(/,' ITERATION CYCLE',/,1X, &
       'closure:    ',A3,5X,'ncycle',I9,5X,'nprint',I9,/,1X, &
       'rmix',F11.2,5X,'tol',F12.5,5X,'ndmax',I10,/,1X, &
       'iuncsr',I9,/,1X,'iungr',I10)

  IF(CLOS.EQ.'XTR')THEN
     DO IP=1,NPVV
        WRITE(OUTU,101) AXTR(IP),RXTR1(IP),RXTR2(IP)
     ENDDO
  ENDIF
  WRITE(OUTU,*)
101 FORMAT(' axtr',F11.2,5X,'rxtr1',F10.2,5X,'rxtr2',F10.2)

  IF(SOL.EQ.'VV')THEN
     !     -------------------

     WRITE(OUTU,102) SOL
102  FORMAT(1X,'solution:    ',A2)

     CALL SOLVV(NSITV,NPVV,INTV,INTVV,X,Y,Z,SEGMID,A,B,C, &
          IPUSR,USR2,IPPHIK,IPGR, &
          IPXVVK,IPCSR,TSR,IPCSK, &
          TSK,CS2R,W,WI,PMFR, &
          CLOS,KBT,RHO,ADIE,AXTR,RXTR1,RXTR2,SHIFT, &
          SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR,NDMAX,NCYCLE,NPRINT, &
          TOL,RMIX,QINIT,LQUSR,LQGR,LQWR,LQTSR,LQBDR)

     call chmdealloc('cycles.src','CYCLES','TSR',DVECT,NPVV,crl=TSR)
     call chmdealloc('cycles.src','CYCLES','TSK',DVECT,NPVV,crl=TSK)
     call chmdealloc('cycles.src','CYCLES','CS2R',DVECT,NPVV,crl=CS2R)
     call chmdealloc('cycles.src','CYCLES','W',DVECT,NPV,crl=W)
     call chmdealloc('cycles.src','CYCLES','WI',DVECT,NPV,crl=WI)
     call chmdealloc('cycles.src','CYCLES','USR2',DVECT,NPVV,crl=USR2)
     call chmdealloc('cycles.src','CYCLES','PMFR',DVECT,NPVV,crl=PMFR)
     call chmdealloc('cycles.src','CYCLES','SHIFT',DVECT,NPVV,crl=SHIFT)

  ELSEIF(SOL.EQ.'UV')THEN
     !     -----------------------

     WRITE(OUTU,103) SOL,IUA
103  FORMAT(1X,'solution:    ',A2,5X,'solute   #',I5)

     CALL SOLUV(NSITV,INTV,IPXVVK,XVV2K1,XVV2K2,RHO, &
          NSITU(IUA),NPUV(IUA),X(UA),Y(UA),Z(UA),SEGMID(UA), &
          INTU(1,1,IUA),INTUV(1,1,IUA),A(IOFF),B(IOFF),C(IOFF), &
          IPUSR(1,IOFF),USR2, &
          IPPHIK(1,IOFF),IPGR(1,IOFF),IPCSR(1,IOFF), &
          TSR,IPCSK(1,IOFF),TSK, &
          CS2R,WU, &
          CLOS,KBT,SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR,NDMAX,NCYCLE, &
          NPRINT,TOL,RMIX,QINIT,LQUSR,LQWR,LQTSR,LQBDR)

     call chmdealloc('cycles.src','CYCLES','TSR',DVECT,NPUV(IUA),crl=TSR)
     call chmdealloc('cycles.src','CYCLES','TSK',DVECT,NPUV(IUA),crl=TSK)
     call chmdealloc('cycles.src','CYCLES','CS2R',DVECT,NPUV(IUA),crl=CS2R)
     call chmdealloc('cycles.src','CYCLES','WU',DVECT,NPU(IUA),crl=WU)
     call chmdealloc('cycles.src','CYCLES','XVV2K1',DVECT,NPV,crl=XVV2K1)
     call chmdealloc('cycles.src','CYCLES','XVV2K2',DVECT,NPV,crl=XVV2K2)
     call chmdealloc('cycles.src','CYCLES','USR2',DVECT,NPUV(IUA),crl=USR2)

  ELSEIF(SOL.EQ.'UU')THEN
     !     -----------------------

     WRITE(OUTU,104) SOL,IUA,IUB
104  FORMAT(1X,'solution:    ',A2,5X,'solute   #',I5, &
          5X,'solute   #',I5)

     CALL SOLUU(NSITV,INTV,IPXVVK,RHO,CDIE,CDIE2, &
          NSITU(IUA),NPUV(IUA),X(UA),Y(UA),Z(UA),SEGMID(UA), &
          INTU(1,1,IUA),INTUV(1,1,IUA),IPCSK(1,PUV(IUA)), &
          IPPHIK(1,PUV(IUA)),WA, &
          NSITU(IUB),NPUV(IUB),X(UB),Y(UB),Z(UB),SEGMID(UB), &
          INTU(1,1,IUB),INTUV(1,1,IUB),IPCSK(1,PUV(IUB)), &
          IPPHIK(1,PUV(IUB)),WB, &
          NPUU(UUAB),INTUU(1,1,UUAB),A(IOFF),B(IOFF),C(IOFF), &
          IPUSR(1,IOFF),USR2, &
          IPPHIK(1,IOFF),IPGR(1,IOFF), &
          IPCSR(1,IOFF),TSR,IPCSK(1,IOFF),TSK, &
          CS2R,WCXCW,PHIABR,PHIABK, &
          CLOS,KBT,SW,SWI,SWF,DSW,NSW,IUNCSR,IUNGR, &
          NDMAX,NCYCLE,NPRINT,TOL,RMIX,QINIT,LQUSR,LQWR, &
          LQTSR,LQBDR,LQCAV)

     call chmdealloc('cycles.src','CYCLES','TSR',DVECT,NPUU(UUAB),crl=TSR)
     call chmdealloc('cycles.src','CYCLES','TSK',DVECT,NPUU(UUAB),crl=TSK)
     call chmdealloc('cycles.src','CYCLES','CS2R',DVECT,NPUU(UUAB),crl=CS2R)
     call chmdealloc('cycles.src','CYCLES','USR2',DVECT,NPUU(UUAB),crl=USR2)
     call chmdealloc('cycles.src','CYCLES','WA',DVECT,NPU(IUA),crl=WA)
     call chmdealloc('cycles.src','CYCLES','WB',DVECT,NPU(IUB),crl=WB)
     call chmdealloc('cycles.src','CYCLES','WCXCW',DVECT,NPUU(UUAB),crl=WCXCW)
     call chmdealloc('cycles.src','CYCLES','PHIABK',DVECT,crl=PHIABK)
     call chmdealloc('cycles.src','CYCLES','PHIABR',DVECT,crl=PHIABR)
  ENDIF

  RETURN
END SUBROUTINE CYCLES

SUBROUTINE GTSWTCH(COMLYN,COMLEN,SW,DSW,SWI,SWF,NSW)
  !-----------------------------------------------------------------------
  !     This subroutine sets the various switches
  !     SW(1) == switch for the total potential
  !     SW(2) == switch for the sigma values of the van der Waals potential
  !     SW(3) == switch for the Coulomb charges
  !     SW(4) == switch for the bridge function (for XTR closure)
  use string
  use number
  use rism
  use chm_kinds
  implicit none
  !     Command parser
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !     Control switch variables
  real(chm_real)  SW(*),SWI(*),SWF(*),DSW(*)
  INTEGER NSW(4)

  INTEGER I

  !     Linear switch on the potential
  DSW(1)=GTRMF(COMLYN,COMLEN,'DSW(1)',ZERO)
  SW(1) =GTRMF(COMLYN,COMLEN,'SW(1)',ONE)
  SWI(1)=GTRMF(COMLYN,COMLEN,'SWI(1)',SW(1))
  SWF(1)=GTRMF(COMLYN,COMLEN,'SWF(1)',SW(1))

  !     Linear switch on the sigma
  DSW(2)=GTRMF(COMLYN,COMLEN,'DSW(2)',ZERO)
  SW(2) =GTRMF(COMLYN,COMLEN,'SW(2)',ONE)
  SWI(2)=GTRMF(COMLYN,COMLEN,'SWI(2)',SW(2))
  SWF(2)=GTRMF(COMLYN,COMLEN,'SWF(2)',SW(2))

  !     Linear switch on the charges
  DSW(3)=GTRMF(COMLYN,COMLEN,'DSW(3)',ZERO)
  SW(3) =GTRMF(COMLYN,COMLEN,'SW(3)',ONE)
  SWI(3)=GTRMF(COMLYN,COMLEN,'SWI(3)',SW(3))
  SWF(3)=GTRMF(COMLYN,COMLEN,'SWF(3)',SW(3))

  !     Switch on the bridge function
  DSW(4)=GTRMF(COMLYN,COMLEN,'DSW(4)',ZERO)
  SW(4) =GTRMF(COMLYN,COMLEN,'SW(4)',ONE)
  SWI(4)=GTRMF(COMLYN,COMLEN,'SWI(4)',SW(4))
  SWF(4)=GTRMF(COMLYN,COMLEN,'SWF(4)',SW(4))

  !     Switching factor loop
  DO I=1,4
     IF((SWI(I).EQ.SWF(I)).OR.(DSW(I).EQ.ZERO))THEN
        NSW(I)=1
     ELSE
        NSW(I)=INT( RSMALL+(SWF(I)-SWI(I))/DSW(I) ) + 1
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE GTSWTCH

SUBROUTINE GTSHFT(COMLYN,COMLEN,AXTR,RXTR1,RXTR2,NPAIR)
  !-----------------------------------------------------------------------
  !     This subroutine sets up the shift function used for the bridge
  !     calculation
  use string
  use number
  use rism
  use chm_kinds
  implicit none
  !     Command parser
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !     Control switch variables
  real(chm_real)  AXTR(*),RXTR1(*),RXTR2(*)
  INTEGER NPAIR

  CHARACTER(len=7) AX
  CHARACTER(len=8) RX1, RX2
  CHARACTER(len=1) ICH
  INTEGER IP

  !     first get the default values
  AXTR(1) =GTRMF(COMLYN,COMLEN,'AXTR',ONE)
  RXTR1(1)=GTRMF(COMLYN,COMLEN,'RXTR1',ZERO)
  RXTR2(1)=GTRMF(COMLYN,COMLEN,'RXTR2',ZERO)

  DO IP=2,NPAIR
     WRITE(ICH,'(I1)') IP
     AX='AXTR('//ICH//')'
     RX1='RXTR1('//ICH//')'
     RX2='RXTR2('//ICH//')'
     AXTR(IP) =GTRMF(COMLYN,COMLEN,AX,AXTR(1))
     RXTR1(IP)=GTRMF(COMLYN,COMLEN,RX1,RXTR1(1))
     RXTR2(IP)=GTRMF(COMLYN,COMLEN,RX2,RXTR2(1))
  ENDDO
  RETURN
END SUBROUTINE GTSHFT
#endif /* (rism_main)*/

